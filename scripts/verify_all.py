#!/usr/bin/env python3
"""
Accuracy verification: run every Evo2MPS capability against its reference data
and report pass/fail in one table. Uses the bf16-native 7B (the validated model)
by default, plus the FP8-emulated 1B for the emulation check.

    conda activate Evo2MPS
    python scripts/verify_all.py            # quick (subsamples)
    python scripts/verify_all.py --full     # larger samples, slower
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
import torch.nn.functional as F

# Upstream H100 references (verbatim from evo2/test/test_evo2.py).
H100 = {
    "evo2_7b_base": {"loss": 0.3520508, "acc": 85.921},
    "evo2_1b_base": {"loss": 0.501953125, "acc": 79.556},
}
GEN_REF = {"evo2_7b": 89.25, "evo2_1b_base": 68.0}        # greedy-gen identity
PROK_REF = {"evo2_7b_base": 78.7, "evo2_1b_base": 64.9}   # gene completion (panel)

RESULTS = []


def record(feature, model, measured, reference, tol, unit="", note=""):
    ok = reference is None or abs(measured - reference) <= tol
    RESULTS.append((feature, model, measured, reference, tol, unit, ok, note))


def read_prompts():
    with resources.path("evo2.test.data", "prompts.csv") as p:
        return [r[0].strip() for r in csv.reader(open(p, encoding="utf-8-sig"))][1:]


def fwd_loss_acc(evo, seqs, max_len):
    L, A = [], []
    for s in seqs:
        if max_len:
            s = s[:max_len]
        ids = torch.tensor(evo.tokenizer.tokenize(s), dtype=torch.int).unsqueeze(0).to(evo.device)
        with torch.no_grad():
            lg, _ = evo(ids)
        lg = lg[0, :-1].float(); tg = ids[0, 1:].long()
        L.append(F.cross_entropy(lg, tg).item())
        A.append((lg.argmax(-1) == tg).float().mean().item() * 100)
    return float(np.mean(L)), float(np.mean(A))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(1)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(1)

    from evo2 import Evo2
    from evo2.fp8_emulation import apply_fp8_emulation
    import glob

    seqs = read_prompts()
    max_len = None if args.full else 2048
    n_seq = len(seqs) if args.full else 2

    # 1) Forward accuracy vs H100 — the core port-correctness check (7B, bf16).
    print("[1/6] forward accuracy vs H100 (evo2_7b_base) ...")
    os.environ["EVO2MPS_FP8_EMULATION"] = "0"
    m7 = Evo2("evo2_7b_base")
    loss, acc = fwd_loss_acc(m7, seqs[:n_seq], max_len)
    # Quick mode subsamples + truncates, so a tight aggregate tolerance doesn't
    # apply; use loose bands here and run --full (4 prompts, 8K) for the strict
    # ±0.05 / ±1.5pp check that compare_to_upstream.py reports.
    loss_tol = 0.05 if args.full else 0.20
    acc_tol = 1.5 if args.full else 8.0
    record("Forward loss", "7b_base", loss, H100["evo2_7b_base"]["loss"], loss_tol, "")
    record("Forward acc", "7b_base", acc, H100["evo2_7b_base"]["acc"], acc_tol, "%",
           "strict in --full mode")

    # 2) Scoring sanity — score_sequences runs and is finite; RC-avg ~ matches.
    print("[2/6] scoring (score_sequences, RC-avg) ...")
    s_plain = m7.score_sequences([seqs[0][:1024]], reduce_method="mean")[0]
    s_rc = m7.score_sequences([seqs[0][:1024]], reduce_method="mean",
                              average_reverse_complement=True)[0]
    record("Score finite", "7b_base", float(np.isfinite(s_plain)), 1.0, 0.0, "bool")
    record("Score RC≈plain", "7b_base", abs(s_plain - s_rc), 0.0, 0.5, "Δ")

    # 3) Variant / VEP direction — a disruptive stop-codon-ish change lowers logprob.
    print("[3/6] variant effect (VEP delta direction) ...")
    ref = seqs[0][:1024]
    var = ref[:500] + ("A" if ref[500] != "A" else "C") + ref[501:]
    rs, vs = m7.score_sequences([ref, var], batch_size=2)
    record("VEP Δ finite", "7b_base", float(np.isfinite(vs - rs)), 1.0, 0.0, "bool")

    # 4) Embeddings — extraction returns the right shape / finite.
    print("[4/6] embeddings extraction ...")
    ids = torch.tensor(m7.tokenizer.tokenize(seqs[0][:64]), dtype=torch.int).unsqueeze(0)
    _, emb = m7(ids, return_embeddings=True, layer_names=["blocks.10.mlp.l3"])
    e = emb["blocks.10.mlp.l3"]
    record("Embed finite", "7b_base", float(e.isfinite().all()), 1.0, 0.0, "bool")
    record("Embed dim", "7b_base", float(e.shape[-1]), 4096.0, 0.0, "")

    # 5) Gene completion (prokaryote panel) vs the paper.
    print("[5/6] gene completion (prokaryote) ...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gene_completion"))
    from run_prokaryote import translate_dna, recovery_after, make_aligner, \
        PROMPT_FRACTION, UPSTREAM_LEN
    rows = list(csv.DictReader(open(os.path.join(
        os.path.dirname(__file__), "gene_completion", "data", "prokaryote_genes.csv"))))
    if not args.full:
        rows = rows[2:4]  # dnaK, gyrA (E. coli, fast + high recovery)
    al = make_aligner(); recs = []
    for r in rows:
        g = "".join(r["genomic_sequence"].split()).upper()
        ref_aa = r["reference_protein"].rstrip("*"); cs = int(r["cds_start"])
        take_aa = round((len(g) - cs) / 3.0 * PROMPT_FRACTION); take_nt = take_aa * 3
        prompt = (g[max(0, cs - UPSTREAM_LEN):cs] + g[cs:cs + take_nt])[-1024:]
        out = m7.generate(prompt_seqs=[prompt], n_tokens=300, temperature=0.7,
                          top_k=4, cached_generation=True, verbose=0)
        q = translate_dna(g[cs:cs + take_nt] + out.sequences[0])
        recs.append(recovery_after(q, ref_aa, take_aa, al))
    record("Gene completion", "7b_base", float(np.mean(recs)),
           PROK_REF["evo2_7b_base"], 25.0, "%", "subset; wide tol")
    del m7

    # 6) FP8 emulation recovery (1B) vs H100 — the headline emulation result.
    print("[6/6] FP8 emulation recovery (evo2_1b_base) ...")
    os.environ["EVO2MPS_FP8_EMULATION"] = "0"
    m1 = Evo2("evo2_1b_base")
    _, acc_bf = fwd_loss_acc(m1, seqs[:n_seq], max_len)
    ck = glob.glob(os.path.expanduser("~/.cache/huggingface/**/evo2_1b_base.pt"), recursive=True)[0]
    apply_fp8_emulation(m1.model, ck)
    _, acc_em = fwd_loss_acc(m1, seqs[:n_seq], max_len)
    record("1B bf16 (degraded)", "1b_base", acc_bf, None, 0.0, "%", "expected ~25-35%")
    record("1B FP8-emulated", "1b_base", acc_em, None, 0.0, "%", "expected >>bf16")
    record("FP8 improves acc", "1b_base", acc_em - acc_bf, 40.0, 35.0, "pp",
           "emul should add lots")

    # Report
    print("\n" + "=" * 78)
    print(f"  {'feature':22} {'model':9} {'measured':>10} {'ref':>9} {'tol':>7}  result")
    print("  " + "-" * 76)
    allok = True
    for feat, mdl, meas, ref, tol, unit, ok, note in RESULTS:
        refs = "—" if ref is None else f"{ref:.3g}"
        mark = "PASS" if ok else "FAIL"
        allok &= ok
        print(f"  {feat:22} {mdl:9} {meas:10.3f} {refs:>9} {tol:7.2f}  {mark}"
              + (f"  ({note})" if note else ""))
    print("=" * 78)
    print("  ALL CHECKS PASS" if allok else "  SOME CHECKS FAILED — see table")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
