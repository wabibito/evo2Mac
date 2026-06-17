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
# Verbatim from upstream evo2/test/test_evo2_generation.py `expected_scores`.
REFERENCE = {
    "evo2_40b": 91.15,
    "evo2_7b": 89.25,
    "evo2_1b_base": 68.0,
    "evo2_20b": 93.4,
}
# Upstream's own tolerance: eps=3 ("numeric differences by versions").
TOLERANCE = 3.0


def read_prompts() -> tuple[list[str], list[str]]:
    """Return (sequences, names) from the bundled prompts.csv."""
    with resources.path("evo2.test.data", "prompts.csv") as p:
        seqs, names = [], []
        with open(p, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row:
                    seqs.append(row[0])
                    names.append(row[1] if len(row) > 1 else f"seq{len(seqs)}")
    return seqs, names


def mid_point_split(seq: str, num_tokens: int) -> tuple[str, str]:
    mid = 2 * (len(seq) // 4)
    return seq[:mid], seq[mid : mid + num_tokens]


def identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return 100.0 * sum(x == y for x, y in zip(a[:n], b[:n])) / n


def gen_identities(model, seqs, n_tokens, prompt_cap) -> list[float]:
    """Per-prompt greedy-generation nucleotide identity vs the true continuation."""
    out_scores = []
    for i, seq in enumerate(seqs):
        prompt, target = mid_point_split(seq, n_tokens)
        if prompt_cap:
            prompt = prompt[-prompt_cap:]
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                prompt_seqs=[prompt],
                n_tokens=n_tokens,
                temperature=1.0,
                top_k=1,          # greedy
                top_p=1.0,
                cached_generation=True,
                verbose=0,
            )
        sc = identity(out.sequences[0], target)
        out_scores.append(sc)
        print(f"    seq {i+1}/{len(seqs)}: identity={sc:.2f}%  ({time.time()-t0:.1f}s)")
    return out_scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_7b_base")
    ap.add_argument("--n-tokens", type=int, default=500)
    ap.add_argument("--max-seqs", type=int, default=None,
                    help="limit prompts (full set is slow on MPS)")
    ap.add_argument("--prompt-cap", type=int, default=None,
                    help="truncate each prompt to N bases to fit MPS memory")
    ap.add_argument("--compare-fp8", action="store_true",
                    help="run each prompt both bf16 and FP8-emulated (per-prompt table)")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(1)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(1)

    from evo2 import Evo2
    seqs, names = read_prompts()
    if args.max_seqs:
        seqs, names = seqs[: args.max_seqs], names[: args.max_seqs]

    if args.compare_fp8:
        # Load with emulation OFF for the bf16 pass, then apply it for the second.
        os.environ["EVO2MAC_FP8_EMULATION"] = "0"
        from evo2.fp8_emulation import apply_fp8_emulation
        t0 = time.time()
        model = Evo2(args.model)
        print(f"model: {args.model}  device: {model.device}  loaded in {time.time()-t0:.1f}s")
        print(f"  {len(seqs)} prompts, greedy gen of {args.n_tokens} tokens\n")

        print("  [bf16] generating ...")
        bf16 = gen_identities(model, seqs, args.n_tokens, args.prompt_cap)
        ckpt = next(__import__("glob").iglob(
            os.path.expanduser(f"~/.cache/huggingface/**/{args.model}.pt"), recursive=True))
        n = apply_fp8_emulation(model.model, ckpt)
        print(f"\n  applied FP8 e4m3 emulation to {n} projection(s)")
        print("  [emul] generating ...")
        emu = gen_identities(model, seqs, args.n_tokens, args.prompt_cap)

        print("\nper-prompt generation identity (% matching nucleotides):")
        print(f"  {'prompt':<28} {'bf16':>8} {'e4m3 emul':>10}")
        print("  " + "-" * 50)
        for nm, b, e in zip(names, bf16, emu):
            print(f"  {nm[:28]:<28} {b:7.2f}% {e:9.2f}%")
        ref = REFERENCE.get(args.model)
        print("  " + "-" * 50)
        print(f"  {'MEAN':<28} {np.mean(bf16):7.2f}% {np.mean(emu):9.2f}%")
        if ref is not None:
            print(f"\n  upstream H100 reference: {ref:.2f}%")
        return 0

    fp8 = os.environ.get("EVO2MAC_FP8_EMULATION") == "1"
    print(f"model: {args.model}   FP8 emulation: {'ON' if fp8 else 'off'}")
    t0 = time.time()
    model = Evo2(args.model)
    print(f"  device: {model.device}   loaded in {time.time() - t0:.1f}s")
    print(f"  {len(seqs)} prompts, greedy gen of {args.n_tokens} tokens\n")

    scores = gen_identities(model, seqs, args.n_tokens, args.prompt_cap)
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
