#!/usr/bin/env python3
"""
Gene Completion benchmark (prokaryote/archaea panel) on Apple Silicon (MPS).

Port of the Evo 2 paper's gene-completion sanity check (Brixi et al., Nature
2026), prokaryote panel only — it needs no external aligner (the eukaryote panel
requires `exonerate` for splice-aware scoring, omitted here). The model is
prompted with ~1 kb upstream + the first 30% of a gene's coding region and must
complete the CDS; the generated DNA is translated and globally aligned to the
reference protein, scoring percent amino-acid (AA) recovery over the completed
region.

Reference AA recovery (paper / maintainer's branch), prokaryote panel:
  Evo 2 1B 64.9 · 7B 78.7 · 20B 90.9 · 40B 92.0

    conda activate Evo2MPS
    python scripts/gene_completion/run_prokaryote.py --model evo2_7b_base
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="An output with one or more elements was resized")
warnings.filterwarnings("ignore", message="path is deprecated")

from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Seq import Seq

HERE = os.path.dirname(os.path.abspath(__file__))
GENES_CSV = os.path.join(HERE, "data", "prokaryote_genes.csv")

PROMPT_FRACTION = 0.30
UPSTREAM_LEN = 1000


def translate_dna(dna: str) -> str:
    usable = (len(dna) // 3) * 3
    return str(Seq(dna[:usable]).translate(to_stop=True))


def make_aligner() -> PairwiseAligner:
    a = PairwiseAligner()
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score = -11
    a.extend_gap_score = -1
    a.mode = "global"
    return a


def recovery_after(query_aa: str, ref_aa: str, ref_start: int, aligner) -> float:
    """Percent identity between generated and reference protein, counted only
    over reference residues at/after ref_start (the non-prompt region)."""
    if not query_aa:
        return 0.0
    aln = aligner.align(ref_aa, query_aa)[0]
    ref_idx = 0
    matches = counted = 0
    for (r0, r1), (q0, q1) in zip(aln.aligned[0], aln.aligned[1]):
        seg = r1 - r0
        for k in range(seg):
            rpos = r0 + k
            if rpos >= ref_start:
                counted += 1
                if ref_aa[rpos] == query_aa[q0 + k]:
                    matches += 1
        ref_idx = r1
    return 100.0 * matches / counted if counted else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_7b_base")
    ap.add_argument("--max-gen", type=int, default=900, help="tokens to generate per gene")
    ap.add_argument("--prompt-cap", type=int, default=None, help="truncate prompt to N bp (memory)")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    rows = list(csv.DictReader(open(GENES_CSV)))
    print(f"prokaryote gene-completion: {len(rows)} genes "
          f"({', '.join(r['gene'] for r in rows)})")

    from evo2 import Evo2
    print(f"loading {args.model} ...")
    t0 = time.time()
    m = Evo2(args.model)
    print(f"  device {m.device}, loaded in {time.time()-t0:.0f}s\n")

    aligner = make_aligner()
    recoveries = []
    for r in rows:
        genomic = "".join(r["genomic_sequence"].split()).upper()
        ref_aa = r["reference_protein"].rstrip("*")
        cds_start = int(r["cds_start"])
        coding_nt = len(genomic) - cds_start
        take_aa = round(coding_nt / 3.0 * PROMPT_FRACTION)
        take_nt = take_aa * 3
        start = max(0, cds_start - UPSTREAM_LEN)
        prompt = genomic[start:cds_start] + genomic[cds_start:cds_start + take_nt]
        if args.prompt_cap:
            prompt = prompt[-args.prompt_cap:]

        t0 = time.time()
        out = m.generate(prompt_seqs=[prompt], n_tokens=args.max_gen,
                         temperature=args.temperature, top_k=4, top_p=1.0,
                         cached_generation=True, verbose=0)
        gen = out.sequences[0]
        # The completed CDS = the 30% we gave (from cds_start) + the generation.
        completed_cds = genomic[cds_start:cds_start + take_nt] + gen
        query_aa = translate_dna(completed_cds)
        rec = recovery_after(query_aa, ref_aa, take_aa, aligner)
        recoveries.append(rec)
        print(f"  {r['gene']:6} ({r['organism'][:22]:22}) AA recovery {rec:5.1f}%  "
              f"({time.time()-t0:.0f}s)")

    mean = sum(recoveries) / len(recoveries)
    print("\n" + "=" * 56)
    print(f"  {args.model} on {m.device} — prokaryote panel")
    print(f"  mean AA recovery: {mean:.1f}%   (paper: 1B 64.9 · 7B 78.7 · 20B 90.9)")
    print("=" * 56)
    print("  Sampling variance is expected; a 7B should land in the 70s-80s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
