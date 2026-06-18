#!/usr/bin/env python3
"""
BRCA1 zero-shot variant-effect prediction on Apple Silicon (MPS).

A runnable, Mac-ready port of upstream's `notebooks/brca1/brca1_zero_shot_vep.ipynb`
(which hardcodes CUDA). It reproduces Evo 2's flagship application: score the
delta-likelihood of BRCA1 single-nucleotide variants and check that more-negative
deltas correspond to loss-of-function variants (AUROC vs the experimental
functional classification from Findlay et al. 2018).

Runs on the bf16-native 7B (recommended) or the FP8-emulated 1B. Because MPS
scoring is slow, default to a manageable window and a sample of variants; pass
--limit 0 to score them all.

    conda activate Evo2MPS
    python scripts/brca1_vep.py --model evo2_7b_base --limit 200 --window 2048
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="An output with one or more elements was resized")
warnings.filterwarnings("ignore", message="path is deprecated")

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "notebooks", "brca1")
XLSX = os.path.join(DATA, "41586_2018_461_MOESM3_ESM.xlsx")
CHR17 = os.path.join(DATA, "GRCh37.p13_chr17.fna.gz")


def load_variants(limit: int):
    import pandas as pd
    df = pd.read_excel(XLSX, header=2)
    df = df[["chromosome", "position (hg19)", "reference", "alt",
             "function.score.mean", "func.class"]].rename(columns={
        "chromosome": "chrom", "position (hg19)": "pos", "reference": "ref",
        "alt": "alt", "function.score.mean": "score", "func.class": "cls"})
    df["cls"] = df["cls"].replace(["FUNC", "INT"], "FUNC/INT")
    df = df.dropna(subset=["pos", "ref", "alt", "cls"]).reset_index(drop=True)
    if limit and limit > 0:
        # Stratified-ish sample: keep some of each class.
        lof = df[df["cls"] == "LOF"].head(limit // 2)
        fun = df[df["cls"] == "FUNC/INT"].head(limit - len(lof))
        df = pd.concat([lof, fun]).reset_index(drop=True)
    return df


def load_chr17():
    from Bio import SeqIO
    with gzip.open(CHR17, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            return str(rec.seq)
    raise RuntimeError("no sequence in chr17 fasta")


def windows(df, chr17, window):
    """Build (ref_seq, var_seq) windows centered on each variant."""
    refs, vars_, uniq, idx = [], [], {}, []
    for _, r in df.iterrows():
        p = int(r["pos"]) - 1
        a, b = max(0, p - window // 2), min(len(chr17), p + window // 2)
        ref_seq = chr17[a:b]
        rel = p - a
        if rel < 0 or rel >= len(ref_seq) or chr17[p].upper() != str(r["ref"]).upper():
            idx.append(None); vars_.append(None); continue
        var_seq = ref_seq[:rel] + str(r["alt"]).upper() + ref_seq[rel + 1:]
        if ref_seq not in uniq:
            uniq[ref_seq] = len(refs); refs.append(ref_seq)
        idx.append(uniq[ref_seq]); vars_.append(var_seq)
    return refs, vars_, idx


def auroc(deltas, is_lof):
    """AUROC that a lower delta predicts LOF (no sklearn dependency)."""
    pos = -np.asarray(deltas)[is_lof]      # flip: lower delta => more LOF
    neg = -np.asarray(deltas)[~is_lof]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
               for _ in [0])
    return float(wins) / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_7b_base")
    ap.add_argument("--limit", type=int, default=200, help="variants to score (0=all)")
    ap.add_argument("--window", type=int, default=2048, help="context window (bp)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if not os.path.exists(XLSX) or not os.path.exists(CHR17):
        print("BRCA1 data not found under notebooks/brca1/. Expected the .xlsx and chr17 .fna.gz.")
        return 1

    print(f"loading BRCA1 variants (limit={args.limit or 'all'}) ...")
    df = load_variants(args.limit)
    chr17 = load_chr17()
    refs, vars_, idx = windows(df, chr17, args.window)
    valid = [i for i, v in enumerate(vars_) if v is not None]
    print(f"  {len(df)} variants, {len(valid)} usable, {len(refs)} unique reference windows "
          f"(window={args.window} bp)")

    from evo2 import Evo2
    print(f"loading {args.model} ...")
    t0 = time.time()
    m = Evo2(args.model)
    print(f"  device {m.device}, loaded in {time.time()-t0:.0f}s")

    print("scoring reference windows ...")
    t0 = time.time()
    ref_scores = np.array(m.score_sequences(refs, batch_size=1))
    print(f"  {len(refs)} refs in {time.time()-t0:.0f}s")
    print("scoring variant windows ...")
    t0 = time.time()
    var_list = [vars_[i] for i in valid]
    var_scores = np.array(m.score_sequences(var_list, batch_size=1))
    print(f"  {len(var_list)} variants in {time.time()-t0:.0f}s")

    deltas = var_scores - ref_scores[[idx[i] for i in valid]]
    is_lof = np.array([df.iloc[i]["cls"] == "LOF" for i in valid])
    a = auroc(deltas, is_lof)

    print("\n" + "=" * 56)
    print(f"  BRCA1 zero-shot VEP — {args.model} on {m.device}")
    print(f"  scored {len(valid)} variants ({is_lof.sum()} LOF / {(~is_lof).sum()} FUNC-INT)")
    print(f"  mean delta  LOF: {deltas[is_lof].mean():+.4f}   FUNC/INT: {deltas[~is_lof].mean():+.4f}")
    print(f"  AUROC (lower delta predicts LOF): {a:.3f}")
    print("=" * 56)
    print("  Upstream reports AUROC ~0.9+ on the 7B with full 8K context; lower")
    print("  here is expected with a smaller window / sample / the FP8-emulated 1B.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
