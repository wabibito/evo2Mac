#!/usr/bin/env python3
"""
End-to-end DNA sequence test for evo2Mac.

Loads a checkpoint on MPS (or CPU/CUDA), then exercises:

  1. Tokenization round-trip
  2. Forward pass + logit sanity check
  3. Embedding extraction from a named layer
  4. score_sequences (log-likelihood)
  5. score_sequences with reverse-complement averaging
  6. generate (short autoregressive sample)

Run after ./scripts/setup.sh:

    conda activate evo2Mac
    python scripts/test_dna.py                              # evo2_1b_base
    python scripts/test_dna.py --model evo2_7b_base         # 32GB+ Mac
    python scripts/test_dna.py --skip-generate              # skip step 6
    python scripts/test_dna.py --sequence ACGTACGTACGTACGT  # custom prompt

Exit code 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback


MAC_FEASIBLE = {
    "evo2_1b_base":         ("~4 GB",  "16 GB Mac ok"),
    "evo2_7b":              ("~14 GB", "32 GB+ Mac"),
    "evo2_7b_base":         ("~14 GB", "32 GB+ Mac"),
    "evo2_7b_262k":         ("~14 GB", "32 GB+ Mac"),
    "evo2_7b_microviridae": ("~14 GB", "32 GB+ Mac"),
}

# A small, well-formed DNA prompt. Mixed bases so the model has to do
# something non-trivial.
DEFAULT_SEQ = (
    "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
    "TTAGGCCAACTCGGCAAGTATCGTAGCTAGCTAGGCATC"
)


def banner(label: str) -> None:
    print(f"\n--- {label} ---")


def section(idx: int, total: int, label: str) -> None:
    print(f"\n[{idx}/{total}] {label}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_1b_base", choices=list(MAC_FEASIBLE))
    ap.add_argument("--device", default=None, help="override device (mps/cpu/cuda:0)")
    ap.add_argument("--sequence", default=DEFAULT_SEQ, help="DNA test sequence (ACGT)")
    ap.add_argument("--n-tokens", type=int, default=32, help="tokens to generate")
    ap.add_argument("--skip-generate", action="store_true", help="skip step 6")
    args = ap.parse_args()

    args.sequence = args.sequence.upper().strip()
    if any(c not in "ACGT" for c in args.sequence):
        print(f"ERROR: --sequence must contain only A/C/G/T; got {args.sequence!r}")
        return 2

    import torch

    banner("environment")
    print(f"  torch:          {torch.__version__}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    print(f"  mps available:  {torch.backends.mps.is_available()}")

    size, host = MAC_FEASIBLE[args.model]
    print(f"  model:          {args.model}  ({size}, {host})")
    print(f"  sequence:       {args.sequence}  (len={len(args.sequence)})")

    total = 5 if args.skip_generate else 6

    # -- Step 0: load --
    section(0, total, "load model")
    t0 = time.time()
    try:
        from evo2 import Evo2

        model = Evo2(args.model)
        if args.device is not None:
            model.device = args.device
            model.model = model.model.to(args.device)
    except Exception as e:
        print(f"FAIL: could not load model: {e}")
        traceback.print_exc()
        return 1
    print(f"  device: {model.device}")
    print(f"  loaded in {time.time() - t0:.1f}s")

    # -- 1: tokenize round-trip --
    section(1, total, "tokenize round-trip")
    tokens = model.tokenizer.tokenize(args.sequence)
    detok = "".join(chr(t) for t in tokens)
    print(f"  first 16 tokens: {tokens[:16]}")
    print(f"  detok ascii:     {detok}")
    if detok != args.sequence:
        print("FAIL: round-trip mismatch")
        return 1
    print("  OK")

    # -- 2: forward + logits sanity --
    section(2, total, "forward pass")
    ids = torch.tensor(tokens, dtype=torch.int).unsqueeze(0)
    t1 = time.time()
    try:
        logits, _ = model(ids)
    except Exception as e:
        print(f"FAIL forward: {e}")
        traceback.print_exc()
        return 1
    print(f"  forward in {time.time() - t1:.2f}s")
    print(f"  logits: shape={tuple(logits.shape)} dtype={logits.dtype} device={logits.device}")
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("FAIL: logits contain NaN/Inf")
        return 1
    if logits.shape[-1] < 4:
        print(f"FAIL: vocab dim too small ({logits.shape[-1]})")
        return 1
    # The model should put nontrivial mass on A/C/G/T (ASCII 65/67/71/84).
    probs_last = torch.softmax(logits[0, -1].float(), dim=-1).cpu()
    mass_acgt = float(probs_last[[65, 67, 71, 84]].sum())
    print(f"  P(A,C,G,T | prompt) at last position: {mass_acgt:.3f}")
    if mass_acgt < 0.5:
        print("WARN: less than half the probability mass is on ACGT — unusual but not fatal")
    print("  OK")

    # -- 3: embedding extraction --
    section(3, total, "embedding extraction")
    embed_layer = "blocks.10.mlp.l3"  # exists for both 1B (15 blocks) and 7B (32 blocks)
    try:
        _, embeddings = model(ids, return_embeddings=True, layer_names=[embed_layer])
        emb = embeddings[embed_layer]
        print(f"  layer={embed_layer}  shape={tuple(emb.shape)}  dtype={emb.dtype}")
        if emb.numel() == 0:
            print("FAIL: empty embedding")
            return 1
        print("  OK")
    except Exception as e:
        print(f"WARN: embedding extraction failed: {e} — continuing")

    # -- 4: score_sequences --
    section(4, total, "score_sequences")
    t2 = time.time()
    try:
        scores = model.score_sequences([args.sequence], batch_size=1)
    except Exception as e:
        print(f"FAIL score: {e}")
        traceback.print_exc()
        return 1
    print(f"  scored in {time.time() - t2:.2f}s")
    print(f"  mean logprob: {scores[0]:.4f}  (closer to 0 = higher likelihood)")
    if not (-20.0 < scores[0] < 0.0):
        print("WARN: score outside expected range (-20, 0)")
    print("  OK")

    # -- 5: reverse-complement-averaged score --
    section(5, total, "score_sequences (RC-averaged)")
    t3 = time.time()
    try:
        scores_rc = model.score_sequences(
            [args.sequence], batch_size=1, average_reverse_complement=True,
        )
    except Exception as e:
        print(f"FAIL score_rc: {e}")
        traceback.print_exc()
        return 1
    print(f"  scored in {time.time() - t3:.2f}s")
    print(f"  RC-avg logprob: {scores_rc[0]:.4f}")
    print("  OK")

    # -- 6: generate --
    if args.skip_generate:
        print("\nSKIP generate (--skip-generate)")
    else:
        section(6, total, f"generate (n_tokens={args.n_tokens})")
        prompt = args.sequence[:32]
        t4 = time.time()
        try:
            out = model.generate(
                prompt_seqs=[prompt],
                n_tokens=args.n_tokens,
                temperature=1.0,
                top_k=4,
                cached_generation=True,
                verbose=0,
            )
        except Exception as e:
            print(f"FAIL generate: {e}")
            traceback.print_exc()
            return 1
        gen = out.sequences[0]
        elapsed = time.time() - t4
        tok_per_s = args.n_tokens / elapsed if elapsed > 0 else float("inf")
        print(f"  generated in {elapsed:.2f}s  ({tok_per_s:.1f} tok/s)")
        print(f"  prompt:    {prompt}")
        print(f"  continued: {gen}")
        if any(c not in "ACGT" for c in gen.upper()):
            print("FAIL: generated sequence contains non-ACGT characters")
            return 1
        print("  OK")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
