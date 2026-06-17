#!/usr/bin/env python3
"""
evo2Mac web UI — Gradio app for running Evo 2 inference on Apple Silicon.

Launch:
    conda activate evo2Mac
    python webapp.py

Opens at http://localhost:7860.

Features:
    - Pick a Mac-feasible Evo 2 checkpoint (1B and 7B family).
    - Lazy-load on demand; keep loaded model cached for fast reuse.
    - Three actions:
        * Forward    — run a forward pass; show the predicted next-token
                       distribution at the last position.
        * Score      — mean log-likelihood of the input sequence
                       (optionally averaged with the reverse complement).
        * Generate   — autoregressive continuation from your prompt.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

# Quiet the noisy FFT warnings during forward; they're benign on MPS.
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")
warnings.filterwarnings("ignore", message="path is deprecated")

import gradio as gr
import torch


# Order matters: the dropdown shows these top-to-bottom, and the first entry is
# the default. Lead with the bf16-native 7B-8k checkpoints (the ones that
# reproduce upstream on Mac). evo2_1b_base is last and flagged FP8-degraded.
MAC_FEASIBLE = {
    "evo2_7b_base":         "~14 GB   bf16-native, recommended",
    "evo2_7b":              "~14 GB   bf16-native (1M ctx)",
    "evo2_7b_262k":         "~14 GB   bf16-native (262K ctx)",
    "evo2_7b_microviridae": "~14 GB   bf16-native",
    "evo2_1b_base":         "~4 GB    loads, but FP8-degraded — see note",
}

DEFAULT_MODEL = "evo2_7b_base"

# Models whose upstream config has use_fp8_input_projections: True. They require
# Transformer Engine (CUDA-only) for numerical accuracy; on Mac they load in
# bf16 with FP8 disabled and produce near-random predictions. Not a port bug.
FP8_DEGRADED = {"evo2_1b_base"}


# --- Lazy model cache ---------------------------------------------------------

_model_cache: dict[str, object] = {}


def _device_info() -> str:
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    if cuda:
        return f"CUDA available (will use cuda:0)"
    if mps:
        return f"MPS available (will use mps)"
    return "Falling back to CPU — this will be slow"


def _get_model(name: str):
    if name not in _model_cache:
        from evo2 import Evo2
        _model_cache[name] = Evo2(name)
    return _model_cache[name]


def _validate_dna(seq: str) -> str | None:
    seq = (seq or "").upper().strip()
    if not seq:
        return "Sequence is empty."
    bad = [c for c in seq if c not in "ACGT"]
    if bad:
        return f"Sequence contains non-ACGT characters: {sorted(set(bad))[:5]}"
    if len(seq) < 4:
        return "Sequence must be at least 4 nucleotides."
    return None


# --- Actions ------------------------------------------------------------------

def action_load(model_name: str, progress: gr.Progress = gr.Progress()):
    progress(0, desc=f"Loading {model_name} ...")
    t0 = time.time()
    try:
        m = _get_model(model_name)
    except Exception as e:
        return (
            f"Failed to load {model_name}: {type(e).__name__}: {e}\n\n"
            f"Tip: 7B models need ~16GB+ unified memory; on an 18GB Mac keep "
            f"sequences short (forward/generate over long prompts can OOM on MPS)."
        )
    warn = ""
    if model_name in FP8_DEGRADED:
        warn = (
            "WARNING: this is an FP8-trained checkpoint running de-quantized in "
            "bf16 (no Transformer Engine on Mac). Predictions are near-random — "
            "fine for testing the pipeline, NOT for real results. Use evo2_7b_base.\n\n"
        )
    return (
        f"{warn}"
        f"{model_name} loaded in {time.time() - t0:.1f}s on device {m.device}.\n"
        f"You can now run Forward / Score / Generate."
    )


def action_forward(model_name: str, sequence: str):
    err = _validate_dna(sequence)
    if err:
        return err, None
    sequence = sequence.upper().strip()
    try:
        m = _get_model(model_name)
    except Exception as e:
        return f"Model not loaded: {e}\nClick 'Load model' first.", None

    ids = torch.tensor(m.tokenizer.tokenize(sequence), dtype=torch.int).unsqueeze(0)
    t0 = time.time()
    logits, _ = m(ids)
    elapsed = time.time() - t0

    # Predicted next-token distribution after the last position.
    probs_last = torch.softmax(logits[0, -1].float(), dim=-1).cpu()
    bases = {"A": 65, "C": 67, "G": 71, "T": 84}
    base_probs = {b: float(probs_last[i]) for b, i in bases.items()}
    base_probs = dict(sorted(base_probs.items(), key=lambda kv: -kv[1]))
    mass_acgt = sum(base_probs.values())

    msg = (
        f"Forward pass: {elapsed:.2f}s on {m.device}\n"
        f"Logits shape: {tuple(logits.shape)}  dtype: {logits.dtype}\n"
        f"P(next base | prompt) — sums to {mass_acgt:.3f} on ACGT:\n"
        + "\n".join(f"    {b}: {p:.4f}" for b, p in base_probs.items())
    )
    return msg, base_probs


def action_score(model_name: str, sequence: str, rc_avg: bool):
    err = _validate_dna(sequence)
    if err:
        return err
    sequence = sequence.upper().strip()
    try:
        m = _get_model(model_name)
    except Exception as e:
        return f"Model not loaded: {e}\nClick 'Load model' first."

    t0 = time.time()
    scores = m.score_sequences(
        [sequence],
        batch_size=1,
        average_reverse_complement=rc_avg,
    )
    elapsed = time.time() - t0
    label = "mean logprob (RC-avg)" if rc_avg else "mean logprob"
    return (
        f"Scored in {elapsed:.2f}s on {m.device}\n"
        f"{label}: {scores[0]:.4f}\n"
        f"(closer to 0 = higher likelihood under the model)"
    )


def action_generate(
    model_name: str, sequence: str, n_tokens: int, temperature: float, top_k: int
):
    err = _validate_dna(sequence)
    if err:
        return err
    sequence = sequence.upper().strip()
    try:
        m = _get_model(model_name)
    except Exception as e:
        return f"Model not loaded: {e}\nClick 'Load model' first."

    n_tokens = max(1, min(int(n_tokens), 2048))
    temperature = max(0.05, min(float(temperature), 4.0))
    top_k = max(1, min(int(top_k), 4))

    t0 = time.time()
    out = m.generate(
        prompt_seqs=[sequence],
        n_tokens=n_tokens,
        temperature=temperature,
        top_k=top_k,
        cached_generation=True,
        verbose=0,
    )
    elapsed = time.time() - t0
    rate = n_tokens / elapsed if elapsed > 0 else float("inf")
    gen = out.sequences[0]
    return (
        f"Generated {n_tokens} tokens in {elapsed:.1f}s  ({rate:.1f} tok/s) on {m.device}\n\n"
        f"Prompt:\n{sequence}\n\n"
        f"Continuation:\n{gen}"
    )


# --- UI -----------------------------------------------------------------------

with gr.Blocks(title="evo2Mac") as demo:
    gr.Markdown(
        f"# evo2Mac\n"
        f"Apple Silicon port of [Evo 2](https://github.com/arcinstitute/evo2). "
        f"{_device_info()}.\n"
        f"\n> **Use a 7B-8k checkpoint** (`evo2_7b_base`, the default) for real "
        f"results — it runs in bf16 and matches upstream's H100 reference "
        f"(loss ≈0.39 vs 0.35). `evo2_1b_base` loads and is handy for quick "
        f"pipeline tests, but it is FP8-trained: without NVIDIA Transformer "
        f"Engine (CUDA-only) it runs de-quantized and its predictions are "
        f"**near-random** — not a port bug. See the repo README for details."
    )

    with gr.Row():
        model_dd = gr.Dropdown(
            label="Model",
            choices=[(f"{name}  —  {info}", name) for name, info in MAC_FEASIBLE.items()],
            value=DEFAULT_MODEL,
        )
        load_btn = gr.Button("Load model", variant="primary")
    load_status = gr.Textbox(label="Status", interactive=False, lines=2)
    load_btn.click(action_load, inputs=[model_dd], outputs=[load_status])

    with gr.Tab("Forward"):
        seq_fw = gr.Textbox(
            label="DNA sequence (ACGT)",
            value="ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG",
            lines=3,
        )
        fw_btn = gr.Button("Run forward pass", variant="primary")
        fw_out = gr.Textbox(label="Result", interactive=False, lines=10)
        fw_bar = gr.Label(label="P(next base)")
        fw_btn.click(action_forward, inputs=[model_dd, seq_fw], outputs=[fw_out, fw_bar])

    with gr.Tab("Score"):
        seq_sc = gr.Textbox(
            label="DNA sequence (ACGT)",
            value="ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG",
            lines=3,
        )
        rc_avg = gr.Checkbox(label="Average with reverse complement", value=False)
        sc_btn = gr.Button("Score sequence", variant="primary")
        sc_out = gr.Textbox(label="Result", interactive=False, lines=6)
        sc_btn.click(action_score, inputs=[model_dd, seq_sc, rc_avg], outputs=[sc_out])

    with gr.Tab("Generate"):
        seq_gen = gr.Textbox(
            label="DNA prompt (ACGT)",
            value="ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG",
            lines=3,
        )
        with gr.Row():
            n_tok = gr.Slider(1, 2048, value=64, step=1, label="n_tokens")
            temp = gr.Slider(0.05, 4.0, value=1.0, step=0.05, label="temperature")
            tk = gr.Slider(1, 4, value=4, step=1, label="top_k")
        gen_btn = gr.Button("Generate", variant="primary")
        gen_out = gr.Textbox(label="Result", interactive=False, lines=14)
        gen_btn.click(
            action_generate,
            inputs=[model_dd, seq_gen, n_tok, temp, tk],
            outputs=[gen_out],
        )

    gr.Markdown(
        "---\n"
        "**Tips**\n"
        "- Click *Load model* first — the first use loads the checkpoint "
        "(~14 GB for 7B, ~4 GB for 1B), which can take a minute or two.\n"
        "- `evo2_7b_base` is the default and the one to trust; the other 7B-8k "
        "checkpoints are also bf16-native. `evo2_1b_base` is FP8-degraded.\n"
        "- Generation is autoregressive; expect a few tok/s for the 7B on MPS.\n"
        "- On a 16–18 GB Mac, keep sequences short — long prompts can exhaust "
        "MPS memory during the forward pass."
    )


def main() -> int:
    host = os.environ.get("EVO2MAC_HOST", "127.0.0.1")
    port = int(os.environ.get("EVO2MAC_PORT", "7860"))
    share = os.environ.get("EVO2MAC_SHARE", "0") == "1"
    demo.launch(server_name=host, server_port=port, share=share, inbrowser=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
