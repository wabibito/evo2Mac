from functools import partial
import huggingface_hub
from huggingface_hub import snapshot_download, constants, hf_hub_download
import os
import pkgutil
import torch
from typing import List, Tuple, Dict, Union
import warnings
import yaml

try:
    import transformer_engine
    HAS_TE = True
except ImportError:
    HAS_TE = False


from vortex.model.generation import generate as vortex_generate
from vortex.model.model import StripedHyena
from vortex.model.tokenizer import CharLevelTokenizer
from vortex.model.utils import dotdict, print_rank_0, load_checkpoint

from evo2.scoring import score_sequences, score_sequences_rc
from evo2.utils import MODEL_NAMES, HF_MODEL_NAME_MAP, CONFIG_MAP

# FP8-trained checkpoints where e4m3 emulation measurably recovers accuracy on
# Mac (no Transformer Engine), so it's applied by default. The 7B-8k checkpoints
# are bf16-robust (emulation is a ~±0.05pp no-op) and are deliberately excluded.
# 20B/40B are FP8-trained too — emulation applies — but they're memory-bound on
# Apple Silicon (see _ram_preflight): 20B ~40 GB weights, 40B ~80 GB.
FP8_EMULATION_DEFAULT_MODELS = {"evo2_1b_base", "evo2_20b", "evo2_40b", "evo2_40b_base"}

# Rough bf16 weight footprint (GB) per model, for a memory pre-flight warning.
_APPROX_WEIGHT_GB = {
    "evo2_1b_base": 4, "evo2_7b_base": 14, "evo2_7b": 14, "evo2_7b_262k": 14,
    "evo2_7b_microviridae": 14, "evo2_20b": 40, "evo2_40b": 80, "evo2_40b_base": 80,
}


def _get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class Evo2:
    def __init__(self, model_name: str = MODEL_NAMES[1], local_path: str = None):
        """
        Load an Evo 2 checkpoint.

        Uses local_path if specified, otherwise checks if in local HuggingFace ~cache.
        Automatically downloads checkpoint from HuggingFace if it does not exist locally.

        Vortex automatically handles device placement on CUDA, and splits model across
        multiple GPUs if available.
        For models split across multiple GPUs, you can specify which GPUs to use with
        CUDA_VISIBLE_DEVICES. If using multi-gpu, do not use .to(device) manually.

        Notes:
        Evo 2 40b is too large to fit on a single H100 GPU, so needs multiple GPUs.
        You can change where HuggingFace downloads to by setting the HF_HOME environment
        variable.
        """
        if model_name not in MODEL_NAMES:
            raise ValueError(
                f'Invalid model name {model_name}. Should be one of: '
                f'{", ".join(MODEL_NAMES)}.'
            )

        # Mac/MPS memory pre-flight: warn before a big download/load that the
        # weights may not fit. Loading proceeds (it may still work on a larger
        # Mac, or with truncation); we just don't pretend the limit isn't there.
        self._ram_preflight(model_name)

        config_path = CONFIG_MAP[model_name]

        if local_path is not None:
            self.model = self.load_evo2_model(None, config_path, local_path)
        else:
            self.model = self.load_evo2_model(model_name, config_path)

        self.tokenizer = CharLevelTokenizer(512)

        # Mac/MPS: Vortex handles CUDA placement itself. When CUDA is not
        # available, route the model and inputs to MPS (or CPU as a last resort).
        self.device = _get_default_device()
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
            # StripedHyena tracks per-block placement in a plain dict that
            # .to() doesn't migrate. The final forward does
            # `x = x.to(self.block_idx_to_device[0])` before the unembed,
            # which would yank x back to CPU. Rewrite the dict so every
            # entry points at our actual device.
            if hasattr(self.model, "block_idx_to_device"):
                for k in list(self.model.block_idx_to_device):
                    self.model.block_idx_to_device[k] = self.device

        # Mac/MPS: FP8 (e4m3) emulation for FP8-trained checkpoints.
        #
        # The 1B was trained with Transformer Engine FP8 input projections;
        # without TE (CUDA-only) it falls back to bf16 and goes near-random.
        # Emulating TE's per-tensor e4m3 projections recovers it (forward acc
        # 33%->75%, greedy-generation identity 33%->67%, ~matching the H100
        # reference). The 7B checkpoints are bf16-robust — emulation there is a
        # measured no-op (~±0.05pp), so it is NOT applied automatically.
        #
        # Default: ON for the 1B, OFF otherwise. EVO2MPS_FP8_EMULATION=0 forces
        # it off (1B runs degraded); =1 forces it on for any model.
        flag = os.environ.get("EVO2MPS_FP8_EMULATION")
        helps_by_default = (model_name or "") in FP8_EMULATION_DEFAULT_MODELS
        want_emulation = flag == "1" or (flag != "0" and helps_by_default)
        if (
            want_emulation
            and not torch.cuda.is_available()
            and not HAS_TE
        ):
            try:
                from evo2.fp8_emulation import apply_fp8_emulation
                ckpt = self._resolve_checkpoint_path(model_name, local_path)
                if ckpt is not None:
                    n = apply_fp8_emulation(self.model, ckpt)
                    if n:
                        warnings.warn(
                            f"Applied FP8 e4m3 emulation to {n} input projection(s) "
                            f"for '{model_name}'. This recovers accuracy lost to the "
                            f"bf16 fallback on Apple Silicon."
                        )
            except Exception as e:  # never block model load on emulation
                warnings.warn(f"FP8 emulation could not be applied: {e}")

    @staticmethod
    def _ram_preflight(model_name):
        """Warn (don't block) when a model's bf16 weights likely won't fit in
        unified memory on Mac. CUDA users are unaffected."""
        if torch.cuda.is_available():
            return
        weights_gb = _APPROX_WEIGHT_GB.get(model_name)
        if not weights_gb:
            return
        try:
            total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
        except (ValueError, OSError):
            return
        # The OS and other apps hold a chunk of unified memory; only ~80% is
        # realistically available to the model process.
        usable_gb = total_gb * 0.8
        # A forward pass needs activation memory on top of the weights; treat
        # ~1.3x weights as the practical floor for an 8K-context pass.
        needed = weights_gb * 1.3
        if weights_gb > usable_gb:
            warnings.warn(
                f"'{model_name}' weights are ~{weights_gb} GB but this machine has "
                f"~{total_gb:.0f} GB of memory — it will not fit and loading will "
                f"likely fail (MPS OOM). Use a 7B-8k or 1B checkpoint, or a larger Mac."
            )
        elif needed > usable_gb:
            warnings.warn(
                f"'{model_name}' weights are ~{weights_gb} GB vs ~{total_gb:.0f} GB "
                f"of memory — loading may succeed but an 8K-context forward pass can "
                f"OOM on MPS. Cap context (e.g. --max-len 2048) and expect it to be slow."
            )

    @staticmethod
    def _resolve_checkpoint_path(model_name, local_path):
        """Locate the merged .pt the model was loaded from (for FP8 scale
        recovery). Mirrors load_evo2_model's path logic."""
        if local_path is not None:
            return local_path
        if model_name is None:
            return None
        filename = f"{model_name}.pt"
        cached = os.path.join(os.path.dirname(constants.HF_HUB_CACHE), filename)
        if os.path.exists(cached):
            return cached
        import glob
        hits = glob.glob(
            os.path.expanduser(f"~/.cache/huggingface/**/{filename}"), recursive=True
        )
        return hits[0] if hits else None

    def forward(
        self,
        input_ids: torch.Tensor,
        return_embeddings: bool = False,
        layer_names=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with optional embedding extraction.
        
        Args:
            input_ids: Input token IDs
            return_embeddings: If True, returns embeddings from specified layers
            layer_names: List of layer names to extract embeddings from. Required if
                return_embeddings=True
            
        Returns:
            Tuple of (logits, embeddings_dict) if return_embeddings=True
            Tuple of (logits, None) otherwise
        """
        embeddings = {}
        handles = []
        
        if return_embeddings:
            if layer_names is None:
                raise ValueError(
                    "layer_names must be specified when return_embeddings=True. Look at "
                    "evo2_model.model.state_dict().keys() to see available layers."
                )
                
            def hook_fn(layer_name):
                def hook(_, __, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    embeddings[layer_name] = output.detach()
                return hook
                
            # Register hooks for requested layers
            for name in layer_names:
                layer = self.model.get_submodule(name)
                handles.append(layer.register_forward_hook(hook_fn(name)))
        
        try:
            # Mac/MPS: make sure inputs live on the same device as the model.
            if input_ids.device != torch.device(self.device):
                input_ids = input_ids.to(self.device)

            with torch.no_grad():
                logits = self.model.forward(input_ids)

            # On MPS/CPU, StripedHyena.forward returns
            # (logits, inference_params_dict). On CUDA the Vortex hooks
            # collapse this to just the tensor. Normalize.
            if isinstance(logits, tuple):
                logits = logits[0]

            if return_embeddings:
                return logits, embeddings
            return logits, None
            
        finally:
            for handle in handles:
                handle.remove()

    def __call__(self, input_ids, return_embeddings=False, layer_names=None):
        return self.forward(input_ids, return_embeddings, layer_names)

    def score_sequences(
        self,
        seqs: List[str],
        batch_size: int = 1,
        prepend_bos: bool = False,
        reduce_method: str = 'mean',
        average_reverse_complement: bool = False,
    ) -> List[float]:
        scoring_func = partial(
            score_sequences_rc if average_reverse_complement else score_sequences,
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            prepend_bos=prepend_bos,
            reduce_method=reduce_method,
            device=self.device,
        )

        with torch.no_grad():
            try:
                scores = scoring_func(seqs)
            except Exception as e:
                raise RuntimeError(f"Error during sequence scoring: {str(e)}") from e

        return scores
    
    def generate(
        self,
        prompt_seqs: List[str],
        n_tokens: int = 500,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        batched: bool = True,
        cached_generation: bool = True,
        verbose: int = 1,
        force_prompt_threshold: int = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Generate sequences from a list of prompts.

        force_prompt_threshold: If specified, avoids OOM errors through teacher forcing if the prompt is longer than this threshold.

        If force_prompt_threshold is none, sets default assuming 1xH100 (evo2_7b) and 2xH100 (evo2_40b) to help avoid OOM errors.
        """

        with torch.no_grad():
            output = vortex_generate(
                prompt_seqs=prompt_seqs,
                model=self.model,
                tokenizer=self.tokenizer,
                n_tokens=n_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                batched=batched,
                cached_generation=cached_generation,
                verbose=verbose,
                force_prompt_threshold=force_prompt_threshold,
                device=self.device,
            )
            return output


    def load_evo2_model(
            self,
            model_name: str = MODEL_NAMES[1],
            config_path: str = None,
            local_path: str = None,
            remove_shards: bool = True,
    ):
        """
        Load HuggingFace checkpoint using StripedHyena 2.

        If local_path is specified, loads from local_path.
        Otherwise, downloads from HuggingFace.
        If remove_shards is True, removes HF checkpoint shards after merging to .pt file.
        """
        if local_path is not None:
            print(f"Loading model from {local_path}...")
            print(f"Loading config from {config_path}...")
            config = yaml.safe_load(pkgutil.get_data(__name__, config_path))
            config = dotdict(config)

            if config.get("use_fp8_input_projections", False) and not HAS_TE:
                # Mac/MPS (no Transformer Engine): load with bf16 projections.
                # For FP8-trained checkpoints the e4m3 emulation re-applies the
                # FP8 numerics after load (see __init__). This works for every
                # model — the only remaining limit is memory, surfaced below.
                warnings.warn(
                    "Transformer Engine not installed. Loading bf16 projections; "
                    "FP8-trained checkpoints get e4m3 emulation applied after load."
                )
                config.use_fp8_input_projections = False

            model = StripedHyena(config)
            load_checkpoint(model, local_path)
            return model
        
        hf_model_name = HF_MODEL_NAME_MAP[model_name]
        filename = f"{model_name}.pt"
        
        final_weights_path = os.path.join(os.path.dirname(constants.HF_HUB_CACHE), filename)
        if os.path.exists(final_weights_path):
            print(f"Found existing merged file: {final_weights_path}")
            weights_path = final_weights_path
            
            hf_hub_download(
                repo_id=hf_model_name, 
                filename="config.json"
            )
        else:
            repo_dir = snapshot_download(
                repo_id=hf_model_name,
            )
            
            # Check if the complete file already exists in the repo
            repo_weights_path = os.path.join(repo_dir, filename)
            if os.path.exists(repo_weights_path):
                print(f"Found complete file in repo: {filename}")
                weights_path = repo_weights_path
            else:
                print(f"Looking for checkpoint shards for {filename}")
                parts = []
                part_num = 0

                while True:
                    part_path = os.path.join(repo_dir, f"{filename}.part{part_num}")
                    if os.path.exists(part_path):
                        parts.append(part_path)
                        part_num += 1
                    else:
                        break
                
                if parts:
                    print(f"Found {len(parts)} shards, merging them...")
                    with open(final_weights_path, 'wb') as outfile:
                        for part in parts:
                            print(f"Merging shard: {os.path.basename(part)}")
                            with open(part, 'rb') as infile:
                                while True:
                                    chunk = infile.read(8192*1024)
                                    if not chunk: 
                                        break
                                    outfile.write(chunk)
                    
                    print(f"Successfully merged all shards into {final_weights_path}")
                    weights_path = final_weights_path
                    if remove_shards and os.path.exists(final_weights_path):
                        for part in parts:
                            real_path = os.path.realpath(part)
                            if os.path.exists(real_path):
                                os.remove(real_path)
                            if os.path.exists(part):
                                os.remove(part)
                else:
                    raise FileNotFoundError(f"Could not find {filename} or any of its shards in {repo_dir}")
                
        config = yaml.safe_load(pkgutil.get_data(__name__, config_path))
        global_config = dotdict(config, Loader=yaml.FullLoader)

        if global_config.get("use_fp8_input_projections", False) and not HAS_TE:
            # Mac/MPS (no Transformer Engine): load bf16 projections for every
            # model; e4m3 emulation re-applies FP8 numerics after load.
            warnings.warn(
                "Transformer Engine not installed. Loading bf16 projections; "
                "FP8-trained checkpoints get e4m3 emulation applied after load."
            )
            global_config.use_fp8_input_projections = False

        model = StripedHyena(global_config)
        load_checkpoint(model, weights_path)

        return model
