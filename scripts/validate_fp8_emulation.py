#!/usr/bin/env python3
"""
Validate the FP8 (e4m3) emulation for evo2_1b_base on Apple Silicon.

Loads the 1B checkpoint (FP8-trained), measures next-token loss/accuracy on
upstream's bundled prompts BEFORE and AFTER applying the TE-faithful e4m3
emulation to the input projections, and reports the delta against the H100
reference. The hypothesis: emulation moves accuracy from the ~30% bf16-fallback
floor toward the ~80% reference.

    conda activate evo2Mac
    python scripts/validate_fp8_emulation.py [--max-len 2048] [--max-seqs N]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import warnings
from importlib import resources

warnings.filterwarnings("ignore", message="An output with one or more elements was resized")
warnings.filterwarnings("ignore", message="path is deprecated")

import numpy as np
import torch
import torch.nn.functional as F

# Per-model H100 + FP8 + flash-attn references (from upstream test_evo2.py).
REFERENCE = {
    "evo2_1b_base": {"loss": 0.502, "acc": 79.6},
    "evo2_7b_base": {"loss": 0.352, "acc": 85.9},
    "evo2_7b":      {"loss": 0.348, "acc": 86.3},
    "evo2_7b_262k": {"loss": 0.352, "acc": 85.9},
}


def read_prompts() -> list[str]:
    with resources.path("evo2.test.data", "prompts.csv") as p:
        import csv
        seqs = []
        with open(p, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row:
                    seqs.append(row[0].strip())
    return seqs


def find_checkpoint(model_name: str) -> str:
    from huggingface_hub import constants
    cache_root = os.path.dirname(constants.HF_HUB_CACHE)
    direct = os.path.join(cache_root, f"{model_name}.pt")
    if os.path.exists(direct):
        return direct
    hits = glob.glob(os.path.expanduser(f"~/.cache/huggingface/**/{model_name}.pt"), recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"could not locate {model_name}.pt in the HF cache")


def evaluate(evo, seqs, max_len: int | None) -> tuple[float, float]:
    """Mean next-token cross-entropy and accuracy over the prompts.

    ``evo`` is the Evo2 wrapper (has .tokenizer/.device/__call__).
    """
    losses, accs = [], []
    for seq in seqs:
        if max_len:
            seq = seq[:max_len]
        ids = torch.tensor(evo.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to(evo.device)
        with torch.no_grad():
            logits, _ = evo(ids)
        logits = logits[0, :-1].float()
        targets = ids[0, 1:].long()
        loss = F.cross_entropy(logits, targets)
        pred = logits.argmax(-1)
        acc = (pred == targets).float().mean() * 100
        losses.append(loss.item())
        accs.append(acc.item())
    return float(np.mean(losses)), float(np.mean(accs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_1b_base")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--max-seqs", type=int, default=None)
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(1)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(1)

    # Load with emulation forced OFF so the baseline is the true bf16 fallback
    # (the Evo2 loader now applies emulation by default for the 1B). We then
    # apply it explicitly below to measure the before/after delta.
    os.environ["EVO2MAC_FP8_EMULATION"] = "0"
    from evo2 import Evo2
    from evo2.fp8_emulation import apply_fp8_emulation

    print(f"loading {args.model} (emulation off for baseline) ...")
    t0 = time.time()
    model = Evo2(args.model)
    print(f"  device: {model.device}  loaded in {time.time() - t0:.1f}s")

    ckpt = find_checkpoint(args.model)
    print(f"  checkpoint: {ckpt}")

    seqs = read_prompts()
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    print(f"  {len(seqs)} prompts" + (f", truncated to {args.max_len} bases" if args.max_len else ""))

    print("\n[baseline] bf16 fallback (FP8 disabled) ...")
    t0 = time.time()
    base_loss, base_acc = evaluate(model, seqs, args.max_len)
    print(f"  loss={base_loss:.4f}  acc={base_acc:.2f}%  ({time.time() - t0:.1f}s)")

    print("\napplying FP8 e4m3 emulation to input projections ...")
    n = apply_fp8_emulation(model.model, ckpt)
    print(f"  replaced {n} projection(s)")

    print("\n[emulated] TE-faithful e4m3 projections ...")
    t0 = time.time()
    emu_loss, emu_acc = evaluate(model, seqs, args.max_len)
    print(f"  loss={emu_loss:.4f}  acc={emu_acc:.2f}%  ({time.time() - t0:.1f}s)")

    ref = REFERENCE.get(args.model, {"loss": float("nan"), "acc": float("nan")})
    print("\n" + "=" * 62)
    print(f"  reference (H100, FP8):  loss={ref['loss']:.3f}  acc={ref['acc']:.1f}%")
    print(f"  bf16 fallback:          loss={base_loss:.4f}  acc={base_acc:.2f}%")
    print(f"  e4m3 emulated:          loss={emu_loss:.4f}  acc={emu_acc:.2f}%")
    print(f"  improvement:            Δloss={base_loss - emu_loss:+.4f}  Δacc={emu_acc - base_acc:+.2f}pp")
    print("=" * 62)
    if emu_acc > base_acc + 5:
        print("  emulation meaningfully improves accuracy.")
    else:
        print("  emulation did NOT meaningfully improve accuracy — investigate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
