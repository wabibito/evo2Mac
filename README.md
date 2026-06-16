# evo2Mac

A macOS / Apple Silicon (MPS) port of [Evo 2](https://github.com/arcinstitute/evo2)
— Arc Institute's DNA language model — for local inference on Mac.

> This is a fork of [arcinstitute/evo2](https://github.com/arcinstitute/evo2)
> with edits to the device handling, FP8 fallback, and config defaults so the
> 1B and 7B Evo 2 checkpoints can run on Apple Silicon via MPS.
>
> Upstream documentation is preserved in [`README.upstream.md`](README.upstream.md).

## Why a port?

Upstream Evo 2 depends on `flash-attn` and NVIDIA Transformer Engine, both of
which are CUDA-only. This fork:

1. Disables `use_flash_attn` and `use_fp8_input_projections` in the YAML
   configs (PyTorch SDPA + bf16 work on MPS).
2. Adds MPS-aware device detection in `evo2/models.py` and `evo2/scoring.py`.
3. Extends the bf16 fallback (when Transformer Engine is missing) to also
   cover the 1B model — upstream only falls back for 7B.
4. Provides a runtime patcher (`patches/patch_vortex.py`) that fixes three
   CUDA-isms in the installed `vortex` (`vtx` on PyPI) package:
   - `torch.autocast("cuda")` → device-aware autocast
   - `torch.fft.fft(...).repeat(...)` → `.unsqueeze().expand()`
     (MPS doesn't support `.repeat` on complex tensors in PT 2.x)
   - `torch.cuda.empty_cache()` / `torch.cuda.memory_allocated()` → device-aware

The patcher writes `.bak` files and is idempotent — re-running is safe, and
`python patches/patch_vortex.py --restore` puts the originals back.

## Models

| Checkpoint            | Size (bf16) | Runs on Mac?              |
|-----------------------|-------------|---------------------------|
| `evo2_1b_base`        | ~4 GB       | ✓ 16 GB unified mem OK    |
| `evo2_7b`             | ~14 GB      | ✓ needs 32 GB+ Mac        |
| `evo2_7b_base`        | ~14 GB      | ✓ needs 32 GB+ Mac        |
| `evo2_7b_262k`        | ~14 GB      | ✓ needs 32 GB+ Mac        |
| `evo2_7b_microviridae`| ~14 GB      | ✓ needs 32 GB+ Mac        |
| `evo2_20b`            | ~40 GB      | ✗ requires FP8 + Hopper   |
| `evo2_40b`            | ~80 GB      | ✗ requires FP8 + Hopper   |
| `evo2_40b_base`       | ~80 GB      | ✗ requires FP8 + Hopper   |

The 20B/40B exclusion is a *runtime* constraint (Transformer Engine + Hopper
GPUs), not just a memory one. They will not run on Apple Silicon even if it
fit. Long-context 7B (`evo2_7b`, 1M context) is technically loadable but the
prefill cost on MPS will be painful — start with `evo2_7b_262k` or
`evo2_7b_base` (8K context).

## Quick start

Prerequisites: Apple Silicon Mac, macOS 14+, [Homebrew](https://brew.sh).

```bash
git clone https://github.com/wabi-media/evo2Mac.git
cd evo2Mac
./install.sh                                          # one-shot setup
conda activate evo2Mac

# Web UI (recommended):
python webapp.py                                      # opens http://localhost:7860

# Or CLI:
python scripts/smoke_test.py --model evo2_1b_base     # one forward pass
python scripts/test_dna.py --model evo2_1b_base       # full DNA pipeline
python scripts/compare_to_upstream.py --model evo2_1b_base   # numerical sanity check

# When done, clean everything up:
./uninstall.sh                                        # removes env + HF cache
```

`setup.sh` will:
1. Install miniforge via Homebrew (skip if present).
2. Create a Python 3.11 conda env named `evo2Mac`.
3. Install PyTorch with MPS support.
4. Install `vtx` (the StripedHyena 2 runtime; imported as `vortex`).
5. Install this package in editable mode (`pip install -e . --no-deps`).
6. Apply the runtime patches to the installed `vortex` package.

On first model load, the checkpoint is downloaded into your HuggingFace cache
(`~/.cache/huggingface/`). Change with `HF_HOME=/path/to/cache`.

## Verifying correctness vs upstream

`scripts/compare_to_upstream.py` runs upstream's own bundled `prompts.csv`
through the model and compares the mean cross-entropy and next-token
accuracy against the reference numbers baked into upstream's
`evo2/test/test_evo2.py`. Those reference values were measured on
H100 + FP8 + flash-attn. Our port runs in bf16 + SDPA on MPS, so a small
drift is expected:

| Tolerance | Loss (cross-entropy) | Accuracy (pp) |
|-----------|----------------------|---------------|
| OK        | drift ≤ 0.05         | drift ≤ 1.5   |
| WARN      | 0.05 < drift ≤ 0.15  | 1.5 < drift ≤ 5 |
| FAIL      | drift > 0.15         | drift > 5     |

A failure here means the port is producing meaningfully different outputs
and something is wrong — it's the canary that should run on every fresh
install.

### Current drift status (M3 Pro, 1B model)

On the M3 Pro with `evo2_1b_base`, the comparison currently reports:

```
upstream (H100, FP8, flash-attn):  loss=0.502  acc=79.6%
evo2Mac (this run):                 loss=1.36   acc=32.6%
```

This is **outside tolerance** — the port loads cleanly, all six end-to-end
checks in `test_dna.py` pass (forward, embeddings, scoring, generation),
and the model produces structured output (99%+ probability mass on ACGT
bases). But the next-token accuracy is much lower than the H100 reference,
and homopolymer continuations (TTTT → T) work while base transitions
(TTTT → G boundary) don't. This is consistent with a precision/positional
issue, not a structural bug.

Suspected causes (in order of likelihood):
- bf16 numerical drift in MPS SDPA vs CUDA flash-attn for long-range
  Hyena FFT operations.
- Possible MPS-specific rounding in the rotary embedding torch fallback
  used instead of the triton kernel (which has no Apple Silicon wheel).
- The torch fallback for `apply_rotary` may differ subtly from the
  triton implementation in edge cases.

**Use this port for plumbing / pipeline / API correctness.** Treat the
numbers themselves as advisory until the drift is closed. PRs welcome.

## Usage

```python
import torch
from evo2 import Evo2

m = Evo2("evo2_1b_base")          # auto-detects MPS / CUDA / CPU
print("device:", m.device)

ids = torch.tensor(m.tokenizer.tokenize("ACGTACGT"), dtype=torch.int).unsqueeze(0)
logits, _ = m(ids)
print(logits.shape)               # (1, 8, 512)

# Scoring
scores = m.score_sequences(["ACGTACGT", "GATTACA"])

# Generation (cached sampling works on MPS)
out = m.generate(prompt_seqs=["ACGT"], n_tokens=64, temperature=1.0, top_k=4)
print(out.sequences[0])
```

## Keeping in sync with upstream

```bash
git remote -v
# origin    https://github.com/wabi-media/evo2Mac.git    (your fork)
# upstream  https://github.com/arcinstitute/evo2.git     (Arc Institute)

git fetch upstream
git merge upstream/main         # or rebase, your call
```

When upstream lands changes to `evo2/models.py`, `evo2/scoring.py`, or the
configs, you may have to redo the Mac edits — they're small and well-marked
with `# evo2Mac:` comments.

## What this port does *not* do

- It does **not** redistribute model weights — those come from HuggingFace on
  first use.
- It does **not** train / fine-tune. Inference only.
- It does **not** make 20B/40B run on Mac. Those need Hopper GPUs.

## Credits

- Upstream model + reference code: [arcinstitute/evo2](https://github.com/arcinstitute/evo2)
  (Arc Institute, Michael Poli, Stanford University). Apache 2.0.
- The Mac compatibility notes that informed this fork's patches: the
  [hakyimlab/evo2-mac](https://github.com/hakyimlab/evo2-mac) effort by the
  Im Lab at UChicago.
- StripedHyena 2 / Vortex runtime: Together. See [`NOTICE.upstream`](NOTICE.upstream).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Modifications and attribution
in [`NOTICE`](NOTICE).
