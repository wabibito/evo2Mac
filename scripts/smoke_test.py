#!/usr/bin/env python3
"""
Load an Evo 2 checkpoint on MPS (or CPU/CUDA) and run a tiny forward pass.

Usage:
    python scripts/smoke_test.py                       # defaults to evo2_1b_base
    python scripts/smoke_test.py --model evo2_7b_base  # needs 32GB+ unified memory
    python scripts/smoke_test.py --device cpu

Exit code is 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch


MAC_FEASIBLE = {
    "evo2_1b_base":         "~4 GB  (16GB Mac ok)",
    "evo2_7b":              "~14 GB (32GB+ Mac)",
    "evo2_7b_base":         "~14 GB (32GB+ Mac)",
    "evo2_7b_262k":         "~14 GB (32GB+ Mac)",
    "evo2_7b_microviridae": "~14 GB (32GB+ Mac)",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_1b_base", choices=list(MAC_FEASIBLE))
    ap.add_argument("--device", default=None, help="override device (mps/cpu/cuda:0)")
    ap.add_argument("--sequence", default="ACGT" * 8, help="DNA sequence to tokenize")
    args = ap.parse_args()

    print(f"torch: {torch.__version__}")
    print(f"mps available: {torch.backends.mps.is_available()}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"model: {args.model}  ({MAC_FEASIBLE[args.model]})")

    t0 = time.time()
    from evo2 import Evo2

    model = Evo2(args.model)
    if args.device is not None:
        model.device = args.device
        model.model = model.model.to(args.device)

    print(f"device: {model.device}")
    print(f"loaded in {time.time() - t0:.1f}s")

    ids = torch.tensor(
        model.tokenizer.tokenize(args.sequence), dtype=torch.int
    ).unsqueeze(0)
    print(f"input shape: {tuple(ids.shape)}")

    t1 = time.time()
    logits, _ = model(ids)
    print(f"forward in {time.time() - t1:.2f}s")
    print(f"logits shape: {tuple(logits.shape)}  dtype: {logits.dtype}  device: {logits.device}")

    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("FAIL: logits contain NaN/Inf")
        return 2
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
