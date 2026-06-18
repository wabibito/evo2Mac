#!/usr/bin/env python3
"""
One table comparing every Mac-feasible Evo 2 model — bf16 native and (for
FP8-trained checkpoints) e4m3 emulation — against upstream's H100 reference.

Forward-pass next-token loss/accuracy over upstream's bundled prompts. For
each model it always reports the bf16 (no-FP8) numbers; for FP8-trained
checkpoints it additionally applies the e4m3 emulation and reports that row.

    conda activate Evo2MPS
    python scripts/compare_all.py                       # all cached models
    python scripts/compare_all.py --models evo2_7b_base evo2_1b_base
    python scripts/compare_all.py --max-len 2048        # cap context (memory)

Only models already present in the HF cache are run unless --download is given.
"""

from __future__ import annotations

import argparse
import csv
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

# Verbatim full-precision H100 + FP8 references (upstream evo2/test/test_evo2.py).
REFERENCE = {
    "evo2_40b":      {"loss": 0.2159424, "acc": 91.673},
    "evo2_7b":       {"loss": 0.3476563, "acc": 86.346},
    "evo2_40b_base": {"loss": 0.2149658, "acc": 91.741},
    "evo2_7b_base":  {"loss": 0.3520508, "acc": 85.921},
    "evo2_1b_base":  {"loss": 0.501953125, "acc": 79.556},
    "evo2_20b":      {"loss": 0.2166748046875, "acc": 91.666},
}

# Models worth comparing on Mac, in display order. FP8-trained ones also get an
# emulated row; the 7B-8k checkpoints are bf16-native (no emulated row needed).
DEFAULT_MODELS = ["evo2_7b_base", "evo2_7b", "evo2_1b_base"]
FP8_TRAINED = {"evo2_1b_base"}  # the ones where emulation measurably helps on Mac


def read_prompts() -> list[str]:
    with resources.path("evo2.test.data", "prompts.csv") as p:
        seqs = []
        with open(p, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row:
                    seqs.append(row[0].strip())
    return seqs


def find_checkpoint(model_name: str) -> str | None:
    hits = glob.glob(os.path.expanduser(f"~/.cache/huggingface/**/{model_name}.pt"), recursive=True)
    return hits[0] if hits else None


def evaluate(evo, seqs, max_len: int | None) -> tuple[float, float]:
    losses, accs = [], []
    for seq in seqs:
        if max_len:
            seq = seq[:max_len]
        ids = torch.tensor(evo.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to(evo.device)
        with torch.no_grad():
            logits, _ = evo(ids)
        logits = logits[0, :-1].float()
        targets = ids[0, 1:].long()
        losses.append(F.cross_entropy(logits, targets).item())
        accs.append(((logits.argmax(-1) == targets).float().mean() * 100).item())
    return float(np.mean(losses)), float(np.mean(accs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--download", action="store_true",
                    help="run even models not yet in the HF cache (triggers download)")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(1)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(1)

    # Load each model with auto-emulation OFF so the bf16 row is the true
    # fallback; we apply emulation explicitly for the emulated row.
    os.environ["EVO2MPS_FP8_EMULATION"] = "0"
    from evo2 import Evo2
    from evo2.fp8_emulation import apply_fp8_emulation

    seqs = read_prompts()
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    ctx = f"{args.max_len}-trunc" if args.max_len else "full 8K"
    print(f"comparing {len(args.models)} model(s) over {len(seqs)} prompts ({ctx})\n")

    rows = []  # (model, mode, loss, acc, ref)
    for name in args.models:
        if not args.download and find_checkpoint(name) is None:
            print(f"  skip {name}: not in HF cache (use --download)")
            continue
        print(f"  {name}: loading ...", end="")
        t0 = time.time()
        model = Evo2(name)
        print(f" {time.time()-t0:.0f}s; bf16 ...", end="")
        bl, ba = evaluate(model, seqs, args.max_len)
        rows.append((name, "bf16 native", bl, ba, REFERENCE.get(name)))
        if name in FP8_TRAINED:
            ckpt = find_checkpoint(name)
            if ckpt:
                apply_fp8_emulation(model.model, ckpt)
                print(" emul ...", end="")
                el, ea = evaluate(model, seqs, args.max_len)
                rows.append((name, "e4m3 emul", el, ea, REFERENCE.get(name)))
        del model
        print(" done")

    # Table.
    print("\n" + "=" * 86)
    print(f"  {'model':<16} {'mode':<13} {'loss':>8} {'acc':>9} | "
          f"{'H100 loss':>9} {'H100 acc':>9} | {'Δacc':>8}")
    print("  " + "-" * 84)
    for name, mode, loss, acc, ref in rows:
        if ref:
            print(f"  {name:<16} {mode:<13} {loss:>8.4f} {acc:>8.2f}% | "
                  f"{ref['loss']:>9.4f} {ref['acc']:>8.2f}% | {acc - ref['acc']:>+7.2f}pp")
        else:
            print(f"  {name:<16} {mode:<13} {loss:>8.4f} {acc:>8.2f}% | "
                  f"{'—':>9} {'—':>9} | {'—':>8}")
    print("=" * 86)
    print("  bf16-native 7B checkpoints reproduce H100; the FP8-trained 1B needs")
    print("  e4m3 emulation (bf16 alone is near-random).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
