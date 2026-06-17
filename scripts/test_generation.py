#!/usr/bin/env python3
"""
Greedy-generation identity test for evo2Mac, vs upstream's H100 reference.

This is the Mac/MPS port of upstream's `evo2/test/test_evo2_generation.py`
(which is hardcoded to `cuda:0`). It prompts each bundled sequence with its
first half, greedily generates the next `n_tokens`, and measures the percent
of nucleotides that match the real continuation — a stronger, end-to-end axis
than the teacher-forced forward pass in compare_to_upstream.py.

Upstream H100 references (direct comparison, no alignment):
    evo2_7b: 89.25%   evo2_40b: 91.15%   evo2_20b: 93.4%   evo2_1b_base: 68.0%

For evo2_1b_base, run with EVO2MAC_FP8_EMULATION=1 to test the FP8 e4m3
emulation against the 68.0% reference (a second, independent check beyond the
forward-pass accuracy in scripts/validate_fp8_emulation.py).

    conda activate evo2Mac
    python scripts/test_generation.py --model evo2_7b              # vs 89.25%
    EVO2MAC_FP8_EMULATION=1 python scripts/test_generation.py --model evo2_1b_base
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from importlib import resources

warnings.filterwarnings("ignore", message="An output with one or more elements was resized")
warnings.filterwarnings("ignore", message="path is deprecated")

import numpy as np
import torch

# Upstream H100 references (% matching nucleotides, greedy 500-token gen).
REFERENCE = {
    "evo2_40b": 91.15,
    "evo2_7b": 89.25,
    "evo2_20b": 93.4,
    "evo2_1b_base": 68.0,
}
# Tolerance: upstream uses eps=3 ("numeric differences by versions"); MPS bf16
# (and no flash-attn) drifts a bit more, so we report rather than hard-pass.
TOLERANCE = 3.0


def read_prompts() -> list[str]:
    with resources.path("evo2.test.data", "prompts.csv") as p:
        seqs = []
        with open(p, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row:
                    seqs.append(row[0])
    return seqs


def mid_point_split(seq: str, num_tokens: int) -> tuple[str, str]:
    mid = 2 * (len(seq) // 4)
    return seq[:mid], seq[mid : mid + num_tokens]


def identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return 100.0 * sum(x == y for x, y in zip(a[:n], b[:n])) / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_7b_base")
    ap.add_argument("--n-tokens", type=int, default=500)
    ap.add_argument("--max-seqs", type=int, default=None,
                    help="limit prompts (full set is slow on MPS)")
    ap.add_argument("--prompt-cap", type=int, default=None,
                    help="truncate each prompt to N bases to fit MPS memory")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(1)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(1)

    fp8 = os.environ.get("EVO2MAC_FP8_EMULATION") == "1"
    print(f"model: {args.model}   FP8 emulation: {'ON' if fp8 else 'off'}")

    from evo2 import Evo2
    t0 = time.time()
    model = Evo2(args.model)
    print(f"  device: {model.device}   loaded in {time.time() - t0:.1f}s")

    seqs = read_prompts()
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    print(f"  {len(seqs)} prompts, greedy gen of {args.n_tokens} tokens\n")

    scores = []
    for i, seq in enumerate(seqs):
        prompt, target = mid_point_split(seq, args.n_tokens)
        if args.prompt_cap:
            prompt = prompt[-args.prompt_cap:]
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                prompt_seqs=[prompt],
                n_tokens=args.n_tokens,
                temperature=1.0,
                top_k=1,          # greedy
                top_p=1.0,
                cached_generation=True,
                verbose=0,
            )
        gen = out.sequences[0]
        sc = identity(gen, target)
        scores.append(sc)
        print(f"  seq {i+1}/{len(seqs)}: identity={sc:.2f}%  ({time.time()-t0:.1f}s)")

    mean = float(np.mean(scores))
    ref = REFERENCE.get(args.model)

    print("\n" + "=" * 60)
    print(f"  {args.model} on {model.device}" + ("  (FP8 emulated)" if fp8 else ""))
    print(f"  mean matching nucleotides: {mean:.2f}%")
    if ref is not None:
        delta = mean - ref
        verdict = "OK" if abs(delta) <= TOLERANCE else "OUTSIDE tolerance"
        print(f"  upstream H100 reference:   {ref:.2f}%   (Δ {delta:+.2f}pp, {verdict})")
    else:
        print(f"  (no H100 reference recorded for {args.model})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
