#!/usr/bin/env python3
"""
evo2Mac web UI — Gradio app for running Evo 2 inference on Apple Silicon.

Launch:
    conda activate evo2Mac
    python webapp.py

Opens at http://localhost:7860.

Features (one tab each):
    - Forward    — forward pass; predicted next-base distribution at the last
                   position, plus an optional per-position confidence track.
    - Score      — mean/sum (PLL) log-likelihood of a sequence, optional
                   reverse-complement averaging and BOS prepending.
    - Variant    — zero-shot variant effect: Δ delta-likelihood (variant − ref),
                   via a full pair or a single-nucleotide (ref+pos+alt) input.
    - Embeddings — extract hidden-state embeddings from a chosen layer;
                   download as .npy.
    - Batch      — paste many sequences or upload a FASTA file and score them
                   all into a sortable table (download as CSV).
    - Generate   — autoregressive continuation with temperature/top-k/top-p.
    - BRCA1 VEP  — Evo 2's flagship analysis: zero-shot BRCA1 variant-effect
                   prediction with AUROC vs the experimental classification.
    - Gene Completion — prokaryote gene-completion benchmark (% AA recovery).

Every tab shares one model selector and a lazy, cached model loader.
"""

from __future__ import annotations

import io
import os
import tempfile
import time
import warnings

# Quiet the noisy FFT warnings during forward; they're benign on MPS.
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")
warnings.filterwarnings("ignore", message="path is deprecated")

import gradio as gr
import numpy as np
import pandas as pd
import torch


# Order matters: the dropdown shows these top-to-bottom, and the first entry is
# the default. Lead with the bf16-native 7B-8k checkpoints (the ones that
# reproduce upstream on Mac). evo2_1b_base is last; it runs with FP8 emulation.
MAC_FEASIBLE = {
    "evo2_7b_base":         "14 GB · bf16-native · recommended",
    "evo2_7b":              "14 GB · bf16-native · 1M ctx",
    "evo2_7b_262k":         "14 GB · bf16-native · 262K ctx",
    "evo2_7b_microviridae": "14 GB · bf16-native",
    "evo2_1b_base":         "4 GB · FP8 emulated",
    "evo2_20b":             "40 GB · FP8 emulated · needs big RAM",
}

DEFAULT_MODEL = "evo2_7b_base"

# Models that are FP8-trained and so run with e4m3 emulation on Mac (the Evo2
# loader applies it automatically). Without it they'd be near-random; with it
# the 1B recovers to ~75% forward / ~74% generation identity. Confirmed: 20B
# also loads + runs (short contexts) on a 64 GB Mac. Still below the bf16-native
# 7B in accuracy, so we surface an informational banner.
FP8_EMULATED = {"evo2_1b_base", "evo2_20b", "evo2_40b", "evo2_40b_base"}

# blocks.10.mlp.l3 exists for both 1B (15 blocks) and 7B (32 blocks); it is the
# layer exercised by scripts/test_dna.py and a safe default for embeddings.
EMBED_LAYERS = [
    "blocks.10.mlp.l3",
    "blocks.20.mlp.l3",
    "blocks.28.mlp.l3",
    "norm",
    "unembed",
]
DEFAULT_EMBED_LAYER = "blocks.10.mlp.l3"

EXAMPLE_SEQ = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"


# --- Lazy model cache ---------------------------------------------------------

_model_cache: dict[str, object] = {}

# Approx download size per model (for the "needs download" hint).
_APPROX_DOWNLOAD = {
    "evo2_7b_base": "≈14 GB", "evo2_7b": "≈14 GB", "evo2_7b_262k": "≈14 GB",
    "evo2_7b_microviridae": "≈14 GB", "evo2_1b_base": "≈4 GB", "evo2_20b": "≈40 GB",
}


def _device_info() -> str:
    if torch.cuda.is_available():
        return "CUDA available (will use cuda:0)"
    if torch.backends.mps.is_available():
        return "MPS available (will use mps)"
    return "Falling back to CPU — this will be slow"


def _is_cached(name: str) -> bool:
    """True if the model's merged .pt is already in the HF cache."""
    import glob
    return bool(glob.glob(os.path.expanduser(f"~/.cache/huggingface/**/{name}.pt"), recursive=True))


def _hf_repo(name: str) -> str | None:
    try:
        from evo2.utils import HF_MODEL_NAME_MAP
        return HF_MODEL_NAME_MAP.get(name)
    except Exception:
        return None


def _ensure_downloaded(name: str, progress) -> None:
    """Download the model from HuggingFace if not cached, driving a Gradio
    progress bar from the underlying tqdm transfer bars."""
    if _is_cached(name):
        return
    repo = _hf_repo(name)
    size = _APPROX_DOWNLOAD.get(name, "several GB")
    progress(0, desc=f"Downloading {name} ({size}) from HuggingFace …")
    from huggingface_hub import snapshot_download
    # Gradio's Progress(track_tqdm=True) captures huggingface_hub's tqdm bars,
    # so the real byte-level transfer drives the on-screen bar.
    snapshot_download(repo_id=repo)


def _get_model(name: str, progress=None):
    if name not in _model_cache:
        if progress is not None:
            _ensure_downloaded(name, progress)
            progress(0.9, desc=f"Loading {name} into memory …")
        from evo2 import Evo2
        _model_cache[name] = Evo2(name)
    return _model_cache[name]


def _require_model(name: str):
    """Return (model, None) on success or (None, error_message)."""
    try:
        return _get_model(name), None
    except Exception as e:  # noqa: BLE001 - surface any load failure to the UI
        return None, (
            f"Model not loaded: {type(e).__name__}: {e}\n"
            f"Click 'Load model' first (7B needs ~16 GB+ unified memory)."
        )


def _validate_dna(seq: str) -> str | None:
    seq = (seq or "").upper().strip()
    if not seq:
        return "Sequence is empty."
    bad = [c for c in seq if c not in "ACGT"]
    if bad:
        return f"Sequence contains non-ACGT characters: {sorted(set(bad))[:5]}"
    if len(seq) < 4:
        return "Sequence must be at least 4 nucleotides."
    return None


def _clean(seq: str) -> str:
    return (seq or "").upper().strip()


def _fp8_banner(model_name: str) -> str:
    if model_name in FP8_EMULATED:
        return (
            "ℹ️ **FP8 e4m3 emulation (auto).** This checkpoint is FP8-trained; "
            "Transformer Engine is CUDA-only, so evo2Mac emulates its per-tensor "
            "e4m3 input projections (~75% forward accuracy / ~67% generation "
            "identity vs the H100 reference). Solid for exploration; the "
            "bf16-native `evo2_7b_base` is still the most accurate.\n"
        )
    return ""


# --- Actions: load ------------------------------------------------------------

def action_load(model_name: str, progress: gr.Progress = gr.Progress(track_tqdm=True)):
    """Download (if needed, with a live progress bar) and load the model."""
    t0 = time.time()
    cached = _is_cached(model_name)
    try:
        if not cached:
            size = _APPROX_DOWNLOAD.get(model_name, "several GB")
            progress(0, desc=f"Downloading {model_name} ({size}) …")
        m = _get_model(model_name, progress=progress)
        progress(1.0, desc="Ready")
    except Exception as e:  # noqa: BLE001
        return (
            f"❌ Failed to load {model_name}: {type(e).__name__}: {e}\n"
            f"Large models need lots of unified memory (20B ≈40 GB, 40B ≈80 GB)."
        )
    warn = ""
    if model_name in FP8_EMULATED:
        warn = (
            "ℹ️ FP8-trained checkpoint running with e4m3 emulation; recovers most "
            "of the FP8 accuracy. `evo2_7b_base` is still the most accurate.\n\n"
        )
    verb = "loaded" if cached else "downloaded & loaded"
    return (
        f"{warn}✅ {model_name} {verb} in {time.time() - t0:.1f}s on {m.device}.\n"
        f"Ready — use any tab below: Forward / Score / Variant / Embeddings / Batch / Generate."
    )


# --- Actions: forward ---------------------------------------------------------

def action_forward(model_name: str, sequence: str, show_track: bool):
    err = _validate_dna(sequence)
    if err:
        return err, None, None
    sequence = _clean(sequence)
    m, merr = _require_model(model_name)
    if merr:
        return merr, None, None

    ids = torch.tensor(m.tokenizer.tokenize(sequence), dtype=torch.int).unsqueeze(0)
    t0 = time.time()
    logits, _ = m(ids)
    elapsed = time.time() - t0

    bases = {"A": 65, "C": 67, "G": 71, "T": 84}
    probs_last = torch.softmax(logits[0, -1].float(), dim=-1).cpu()
    base_probs = {b: float(probs_last[i]) for b, i in bases.items()}
    base_probs = dict(sorted(base_probs.items(), key=lambda kv: -kv[1]))
    mass_acgt = sum(base_probs.values())

    msg = (
        f"Forward pass: {elapsed:.2f}s on {m.device}\n"
        f"Logits shape: {tuple(logits.shape)}  dtype: {logits.dtype}\n"
        f"P(next base | prompt) — {mass_acgt:.3f} of the mass on ACGT."
    )

    # Per-position confidence track: at each position, the model's probability
    # of the base that actually occurs next (a "how predictable is this seq"
    # readout). Length = len(seq) - 1.
    track_df = None
    if show_track:
        all_probs = torch.softmax(logits[0].float(), dim=-1).cpu()
        idx = {65: 0, 67: 1, 71: 2, 84: 3}
        tok = m.tokenizer.tokenize(sequence)
        rows = []
        for pos in range(len(tok) - 1):
            nxt = int(tok[pos + 1])
            if nxt in bases.values():
                p = float(all_probs[pos, nxt])
                rows.append({"position": pos, "p(next true base)": p,
                             "base": chr(nxt)})
        track_df = pd.DataFrame(rows)

    return msg, base_probs, track_df


# --- Actions: score -----------------------------------------------------------

def action_score(model_name: str, sequence: str, rc_avg: bool,
                 reduce_method: str, prepend_bos: bool):
    err = _validate_dna(sequence)
    if err:
        return err
    sequence = _clean(sequence)
    m, merr = _require_model(model_name)
    if merr:
        return merr

    t0 = time.time()
    scores = m.score_sequences(
        [sequence],
        batch_size=1,
        prepend_bos=prepend_bos,
        reduce_method=reduce_method,
        average_reverse_complement=rc_avg,
    )
    elapsed = time.time() - t0
    kind = "sum (PLL)" if reduce_method == "sum" else "mean"
    rc = " RC-avg" if rc_avg else ""
    return (
        f"Scored in {elapsed:.2f}s on {m.device}\n"
        f"{kind}{rc} logprob: {scores[0]:.4f}\n"
        f"(closer to 0 = higher likelihood under the model)"
    )


# --- Actions: variant ---------------------------------------------------------

def _apply_snv(seq: str, pos1: int, alt: str) -> tuple[str | None, str | None]:
    """Apply a 1-indexed single-nucleotide variant; return (mutant, error)."""
    p = int(pos1) - 1
    if p < 0 or p >= len(seq):
        return None, f"Position {pos1} is outside the sequence (length {len(seq)})."
    alt = (alt or "").upper().strip()
    if alt not in "ACGT" or len(alt) != 1:
        return None, f"ALT must be a single base A/C/G/T (got {alt!r})."
    return seq[:p] + alt + seq[p + 1:], None


def action_variant(model_name: str, mode: str, wt: str, mut: str,
                   snv_seq: str, snv_pos: int, snv_alt: str,
                   rc_avg: bool, reduce_method: str):
    """Zero-shot variant effect: delta log-likelihood (variant − reference).

    This is the BRCA1-notebook scoring: a more-negative delta means the variant
    lowers the model's likelihood, predicting greater functional disruption.
    """
    if mode == "SNV (ref + position + alt base)":
        ref = _clean(snv_seq)
        err = _validate_dna(ref)
        if err:
            return f"Reference: {err}", None
        ref_base = ref[int(snv_pos) - 1] if 0 < int(snv_pos) <= len(ref) else "?"
        var, err = _apply_snv(ref, snv_pos, snv_alt)
        if err:
            return err, None
        label = f"SNV {ref_base}{int(snv_pos)}{snv_alt.upper()}"
    else:
        ref, var = _clean(wt), _clean(mut)
        for nm, s in (("Reference", ref), ("Variant", var)):
            e = _validate_dna(s)
            if e:
                return f"{nm}: {e}", None
        label = "variant vs reference"

    m, merr = _require_model(model_name)
    if merr:
        return merr, None

    t0 = time.time()
    scores = m.score_sequences([ref, var], batch_size=2,
                               reduce_method=reduce_method,
                               average_reverse_complement=rc_avg)
    elapsed = time.time() - t0
    ref_s, var_s = float(scores[0]), float(scores[1])
    delta = var_s - ref_s  # evo2_delta_score: var − ref

    if delta <= -0.10:
        call = "likely DISRUPTIVE (strongly lowers likelihood)"
    elif delta < -0.02:
        call = "possibly disruptive (lowers likelihood)"
    elif delta <= 0.02:
        call = "near-neutral"
    else:
        call = "tolerated / favorable (raises likelihood)"

    note = ""
    if model_name in FP8_EMULATED:
        note = "\n(1B runs with FP8 emulation — good for exploration; the 7B is more accurate.)"
    msg = (
        f"{label} — scored in {elapsed:.2f}s on {m.device}\n"
        f"reference logprob: {ref_s:.4f}\n"
        f"variant   logprob: {var_s:.4f}\n"
        f"Δ delta-likelihood (var − ref): {delta:+.4f}  →  {call}\n\n"
        f"This is the zero-shot VEP score from Evo 2's BRCA1 analysis: more "
        f"negative = more likely to disrupt function. Calibrate thresholds against "
        f"a labeled panel for your locus.{note}"
    )
    df = pd.DataFrame({"sequence": ["reference", "variant"],
                       "logprob": [round(ref_s, 4), round(var_s, 4)]})
    return msg, df


# --- Actions: embeddings ------------------------------------------------------

def action_embed(model_name: str, sequence: str, layer_name: str):
    err = _validate_dna(sequence)
    if err:
        return err, None
    sequence = _clean(sequence)
    m, merr = _require_model(model_name)
    if merr:
        return merr, None

    ids = torch.tensor(m.tokenizer.tokenize(sequence), dtype=torch.int).unsqueeze(0)
    t0 = time.time()
    try:
        _, embeddings = m(ids, return_embeddings=True, layer_names=[layer_name])
    except Exception as e:  # noqa: BLE001 - bad layer name etc.
        return (
            f"Embedding extraction failed for layer '{layer_name}': "
            f"{type(e).__name__}: {e}\n"
            f"Tip: layer names look like 'blocks.10.mlp.l3'."
        ), None
    elapsed = time.time() - t0

    emb = embeddings[layer_name].float().cpu().numpy()  # (1, seq_len, dim)
    arr = emb[0]  # (seq_len, dim)

    # Mean-pooled vector is the usual per-sequence feature; save the full
    # (seq_len, dim) array for download.
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"evo2_emb_{layer_name.replace('.', '_')}_",
        suffix=".npy", delete=False,
    )
    np.save(tmp.name, arr)
    tmp.close()

    msg = (
        f"Embeddings from '{layer_name}' in {elapsed:.2f}s on {m.device}\n"
        f"Shape: {arr.shape}  (positions × hidden dim)  dtype: {arr.dtype}\n"
        f"Mean-pooled vector: dim={arr.shape[-1]}, "
        f"L2 norm={float(np.linalg.norm(arr.mean(0))):.3f}\n"
        f"min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}\n\n"
        f"Download is the full (positions × dim) array as .npy."
    )
    return msg, tmp.name


# --- Actions: batch -----------------------------------------------------------

def _parse_sequences(text: str, fasta_file) -> tuple[list[tuple[str, str]], str | None]:
    """Return ([(id, seq), ...], error). Accepts pasted lines or a FASTA upload."""
    records: list[tuple[str, str]] = []

    if fasta_file is not None:
        try:
            from Bio import SeqIO
            path = fasta_file.name if hasattr(fasta_file, "name") else fasta_file
            for rec in SeqIO.parse(path, "fasta"):
                records.append((rec.id, str(rec.seq)))
        except Exception as e:  # noqa: BLE001
            return [], f"Failed to parse FASTA: {type(e).__name__}: {e}"

    if text and text.strip():
        for i, line in enumerate(text.strip().splitlines()):
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            records.append((f"seq{i + 1}", line))

    if not records:
        return [], "No sequences provided. Paste lines or upload a FASTA file."
    if len(records) > 256:
        return [], f"Too many sequences ({len(records)}); cap is 256 per batch."
    return records, None


def action_batch(model_name: str, text: str, fasta_file, rc_avg: bool,
                 reduce_method: str):
    records, err = _parse_sequences(text, fasta_file)
    if err:
        return err, None, None

    cleaned, bad = [], []
    for rid, seq in records:
        s = _clean(seq)
        v = _validate_dna(s)
        if v:
            bad.append(f"{rid}: {v}")
        else:
            cleaned.append((rid, s))
    if not cleaned:
        return "All sequences invalid:\n" + "\n".join(bad[:10]), None, None

    m, merr = _require_model(model_name)
    if merr:
        return merr, None, None

    seqs = [s for _, s in cleaned]
    t0 = time.time()
    scores = m.score_sequences(
        seqs,
        batch_size=min(8, len(seqs)),
        reduce_method=reduce_method,
        average_reverse_complement=rc_avg,
    )
    elapsed = time.time() - t0

    df = pd.DataFrame({
        "id": [rid for rid, _ in cleaned],
        "length": [len(s) for _, s in cleaned],
        "logprob": [round(float(x), 4) for x in scores],
    }).sort_values("logprob", ascending=False, ignore_index=True)

    tmp = tempfile.NamedTemporaryFile(prefix="evo2_batch_", suffix=".csv", delete=False)
    df.to_csv(tmp.name, index=False)
    tmp.close()

    note = f"\n{len(bad)} skipped (invalid)." if bad else ""
    msg = (
        f"Scored {len(seqs)} sequences in {elapsed:.1f}s on {m.device} "
        f"({reduce_method}{', RC-avg' if rc_avg else ''}).{note}"
    )
    return msg, df, tmp.name


# --- Actions: generate --------------------------------------------------------

def action_generate(model_name: str, sequence: str, n_tokens: int,
                    temperature: float, top_k: int, top_p: float):
    err = _validate_dna(sequence)
    if err:
        return err
    sequence = _clean(sequence)
    m, merr = _require_model(model_name)
    if merr:
        return merr

    n_tokens = max(1, min(int(n_tokens), 2048))
    temperature = max(0.05, min(float(temperature), 4.0))
    top_k = max(1, min(int(top_k), 4))
    top_p = max(0.0, min(float(top_p), 1.0))

    t0 = time.time()
    out = m.generate(
        prompt_seqs=[sequence],
        n_tokens=n_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cached_generation=True,
        verbose=0,
    )
    elapsed = time.time() - t0
    rate = n_tokens / elapsed if elapsed > 0 else float("inf")
    gen = out.sequences[0]
    conf = ""
    if getattr(out, "logprobs_mean", None):
        conf = f"Mean logprob of generated tokens: {out.logprobs_mean[0]:.4f}\n"
    return (
        f"Generated {n_tokens} tokens in {elapsed:.1f}s ({rate:.1f} tok/s) on {m.device}\n"
        f"{conf}\n"
        f"Prompt:\n{sequence}\n\n"
        f"Continuation:\n{gen}"
    )


# --- Actions: BRCA1 variant-effect benchmark ---------------------------------

_BRCA1_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks", "brca1")


def action_brca1(model_name: str, limit: int, window: int,
                 progress: gr.Progress = gr.Progress(track_tqdm=True)):
    """Run Evo 2's flagship BRCA1 zero-shot variant-effect analysis."""
    xlsx = os.path.join(_BRCA1_DIR, "41586_2018_461_MOESM3_ESM.xlsx")
    chr17 = os.path.join(_BRCA1_DIR, "GRCh37.p13_chr17.fna.gz")
    if not (os.path.exists(xlsx) and os.path.exists(chr17)):
        return "BRCA1 data not found under notebooks/brca1/.", None, None
    m, merr = _require_model(model_name)
    if merr:
        return merr, None, None

    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
    import importlib
    bv = importlib.import_module("brca1_vep")

    progress(0.1, desc="Loading BRCA1 variants + chr17 ...")
    df = bv.load_variants(int(limit))
    chrom = bv.load_chr17()
    refs, vars_, idx = bv.windows(df, chrom, int(window))
    valid = [i for i, v in enumerate(vars_) if v is not None]
    if not valid:
        return "No usable variants (reference mismatch?).", None, None

    progress(0.4, desc=f"Scoring {len(refs)} ref windows ...")
    import numpy as _np
    ref_scores = _np.array(m.score_sequences(refs, batch_size=1))
    progress(0.7, desc=f"Scoring {len(valid)} variant windows ...")
    var_scores = _np.array(m.score_sequences([vars_[i] for i in valid], batch_size=1))
    deltas = var_scores - ref_scores[[idx[i] for i in valid]]
    is_lof = _np.array([df.iloc[i]["cls"] == "LOF" for i in valid])
    auroc = bv.auroc(deltas, is_lof)

    rows = []
    for k, i in enumerate(valid):
        r = df.iloc[i]
        rows.append({"variant": f"{r['ref']}{int(r['pos'])}{r['alt']}",
                     "class": r["cls"], "delta_score": round(float(deltas[k]), 4)})
    table = pd.DataFrame(rows).sort_values("delta_score", ignore_index=True)
    msg = (
        f"BRCA1 zero-shot VEP on {m.device} — {len(valid)} variants "
        f"({int(is_lof.sum())} LOF / {int((~is_lof).sum())} FUNC-INT)\n"
        f"mean Δ — LOF: {deltas[is_lof].mean():+.4f}   FUNC/INT: {deltas[~is_lof].mean():+.4f}\n"
        f"AUROC (lower Δ predicts loss-of-function): {auroc:.3f}\n\n"
        f"Lower delta-likelihood = more likely to disrupt BRCA1 function. Upstream "
        f"reports ~0.9+ AUROC on the 7B at full 8K; a smaller window/sample scores lower."
    )
    plot_df = pd.DataFrame({"class": ["LOF", "FUNC/INT"],
                            "mean delta": [float(deltas[is_lof].mean()),
                                           float(deltas[~is_lof].mean())]})
    return msg, table, plot_df


# --- Actions: gene completion benchmark --------------------------------------

def action_gene_completion(model_name: str, max_gen: int, temperature: float,
                           progress: gr.Progress = gr.Progress(track_tqdm=True)):
    """Run the prokaryote gene-completion benchmark (% amino-acid recovery)."""
    m, merr = _require_model(model_name)
    if merr:
        return merr, None

    import sys as _sys, csv as _csv, importlib
    gc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "scripts", "gene_completion")
    _sys.path.insert(0, gc_dir)
    rp = importlib.import_module("run_prokaryote")
    genes = os.path.join(gc_dir, "data", "prokaryote_genes.csv")
    rows = list(_csv.DictReader(open(genes)))
    aligner = rp.make_aligner()

    out_rows, recs = [], []
    for n, r in enumerate(rows):
        progress((n + 0.5) / len(rows), desc=f"Completing {r['gene']} ...")
        g = "".join(r["genomic_sequence"].split()).upper()
        ref_aa = r["reference_protein"].rstrip("*")
        cs = int(r["cds_start"])
        take_aa = round((len(g) - cs) / 3.0 * rp.PROMPT_FRACTION)
        take_nt = take_aa * 3
        prompt = (g[max(0, cs - rp.UPSTREAM_LEN):cs] + g[cs:cs + take_nt])[-1024:]
        gen = m.generate(prompt_seqs=[prompt], n_tokens=int(max_gen),
                         temperature=float(temperature), top_k=4,
                         cached_generation=True, verbose=0).sequences[0]
        q = rp.translate_dna(g[cs:cs + take_nt] + gen)
        rec = rp.recovery_after(q, ref_aa, take_aa, aligner)
        recs.append(rec)
        out_rows.append({"gene": r["gene"], "organism": r["organism"],
                         "AA recovery %": round(rec, 1)})
    mean = sum(recs) / len(recs)
    table = pd.DataFrame(out_rows)
    msg = (
        f"Gene completion (prokaryote panel) on {m.device}\n"
        f"mean AA recovery: {mean:.1f}%   (paper: 1B 64.9 · 7B 78.7 · 20B 90.9)\n\n"
        f"Each gene is prompted with ~1 kb upstream + 30% of its CDS; the "
        f"completion is translated and aligned to the reference protein."
    )
    return msg, table


# --- UI -----------------------------------------------------------------------

REDUCE_CHOICES = [("mean (per-base avg)", "mean"), ("sum (PLL)", "sum")]

# Teal/aqua theme. Soft base with the primary hue swapped to teal; custom CSS
# pushes the accent toward aqua on buttons, the active tab, and labels.
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

# Accent presets: (base, bright, text-on-accent). Teal/aqua is the default.
ACCENTS = {
    "Aqua / Teal": ("#00c9b1", "#1ee3cf", "#042f2a"),
    "Ocean Blue":  ("#2f7df6", "#5aa0ff", "#03122e"),
    "Violet":      ("#8b5cf6", "#a78bfa", "#1a0b33"),
    "Coral":       ("#fb6f5b", "#ff8f7e", "#330c06"),
    "Lime":        ("#84cc16", "#a3e635", "#13260a"),
}
DEFAULT_ACCENT = "Aqua / Teal"

CSS = """
:root {
  --accent: #00c9b1;
  --accent-bright: #1ee3cf;
  --accent-text: #042f2a;
}
#evo2-header {
  background: linear-gradient(100deg,
    color-mix(in srgb, var(--accent) 16%, transparent),
    color-mix(in srgb, var(--accent) 3%, transparent));
  border: 1px solid color-mix(in srgb, var(--accent) 32%, transparent);
  border-radius: 14px; padding: 16px 20px; margin-bottom: 6px;
}
#evo2-header h1 { margin: 0 0 4px 0; font-weight: 700; letter-spacing: -0.5px; }
button.primary, .primary > button {
  background: linear-gradient(90deg, var(--accent), var(--accent-bright)) !important;
  border: none !important; color: var(--accent-text) !important; font-weight: 600 !important;
}
button.primary:hover, .primary > button:hover { filter: brightness(1.08); }
.tab-nav button.selected {
  color: var(--accent-bright) !important;
  border-bottom: 2px solid var(--accent-bright) !important;
}
.gradio-container span[data-testid="block-info"], label > span { color: var(--accent) !important; }
.progress-bar, .meta-text + div .progress-bar { background: var(--accent-bright) !important; }
"""

# JS: apply an accent (by name) and toggle dark/light by flipping Gradio's
# .dark class on <body>. Runs in the browser, so it's live with no reload.
SET_ACCENT_JS = """
(name) => {
  const map = %s;
  const a = map[name] || map[%r];
  const r = document.documentElement.style;
  r.setProperty('--accent', a[0]);
  r.setProperty('--accent-bright', a[1]);
  r.setProperty('--accent-text', a[2]);
  return name;
}
""" % (
    "{" + ",".join(f'{k!r}:[{v[0]!r},{v[1]!r},{v[2]!r}]' for k, v in ACCENTS.items()) + "}",
    DEFAULT_ACCENT,
)

SET_MODE_JS = """
(mode) => {
  const dark = (mode === 'Dark');
  document.body.classList.toggle('dark', dark);
  return mode;
}
"""


def _model_choices():
    """Dropdown labels showing each model's size and cached/download state."""
    out = []
    for name, info in MAC_FEASIBLE.items():
        tag = "✓ cached" if _is_cached(name) else "⬇ will download"
        out.append((f"{name}  ·  {info}  ·  {tag}", name))
    return out


with gr.Blocks(title="evo2Mac", theme=THEME, css=CSS) as demo:
    with gr.Column(elem_id="evo2-header"):
        with gr.Row():
            gr.Markdown(
                f"# 🧬 evo2Mac\n"
                f"Run **Evo 2** genome models locally on Apple Silicon — forward "
                f"pass, scoring, variant effects, embeddings, batch, generation, "
                f"**BRCA1 VEP** & the **gene-completion** benchmark. "
                f"_{_device_info()}._"
            )
            with gr.Column(min_width=180, scale=0):
                mode_sel = gr.Radio(
                    ["Light", "Dark"], value="Light", label="Theme", scale=0,
                )
                accent_sel = gr.Dropdown(
                    list(ACCENTS.keys()), value=DEFAULT_ACCENT,
                    label="Highlight color", scale=0,
                )
        # Live appearance controls — run JS in the browser (no reload).
        mode_sel.change(None, inputs=[mode_sel], outputs=None, js=SET_MODE_JS)
        accent_sel.change(None, inputs=[accent_sel], outputs=None, js=SET_ACCENT_JS)

    with gr.Row():
        model_dd = gr.Dropdown(
            label="① Choose a model",
            info="Not downloaded yet? It auto-downloads from HuggingFace on load, with a progress bar.",
            choices=_model_choices(),
            value=DEFAULT_MODEL,
            scale=4,
        )
        load_btn = gr.Button("② Load / Download", variant="primary", scale=1)
    fp8_md = gr.Markdown(_fp8_banner(DEFAULT_MODEL))
    load_status = gr.Textbox(label="Status", interactive=False, lines=2,
                             value="Pick a model and click Load. First load of a new model downloads it.")
    load_btn.click(action_load, inputs=[model_dd], outputs=[load_status])
    model_dd.change(lambda n: _fp8_banner(n), inputs=[model_dd], outputs=[fp8_md])

    # --- Forward ---
    with gr.Tab("Forward"):
        gr.Markdown("Run a forward pass and inspect the predicted next-base distribution.")
        seq_fw = gr.Textbox(label="DNA sequence (ACGT)", value=EXAMPLE_SEQ, lines=3)
        track_cb = gr.Checkbox(label="Show per-position confidence track", value=True)
        fw_btn = gr.Button("Run forward pass", variant="primary")
        fw_out = gr.Textbox(label="Result", interactive=False, lines=4)
        fw_bar = gr.Label(label="P(next base) at last position")
        fw_track = gr.LinePlot(
            x="position", y="p(next true base)",
            title="Per-position probability of the true next base",
            height=240, visible=True,
        )
        fw_btn.click(action_forward, inputs=[model_dd, seq_fw, track_cb],
                     outputs=[fw_out, fw_bar, fw_track])

    # --- Score ---
    with gr.Tab("Score"):
        gr.Markdown("Log-likelihood of a sequence under the model.")
        seq_sc = gr.Textbox(label="DNA sequence (ACGT)", value=EXAMPLE_SEQ, lines=3)
        with gr.Row():
            sc_reduce = gr.Dropdown(label="Reduce", choices=REDUCE_CHOICES, value="mean")
            sc_rc = gr.Checkbox(label="Average with reverse complement", value=False)
            sc_bos = gr.Checkbox(label="Prepend BOS", value=False)
        sc_btn = gr.Button("Score sequence", variant="primary")
        sc_out = gr.Textbox(label="Result", interactive=False, lines=4)
        sc_btn.click(action_score, inputs=[model_dd, seq_sc, sc_rc, sc_reduce, sc_bos],
                     outputs=[sc_out])

    # --- Variant (zero-shot VEP) ---
    with gr.Tab("Variant"):
        gr.Markdown(
            "**Zero-shot variant effect prediction.** Reports the Δ delta-likelihood "
            "(variant − reference) — the same score Evo 2's BRCA1 analysis uses: "
            "more negative ⇒ more likely to disrupt function. "
            "Give either a full reference/variant pair, or a single-nucleotide "
            "variant as reference + position + alt base."
        )
        var_mode = gr.Radio(
            ["SNV (ref + position + alt base)", "Reference / variant pair"],
            value="SNV (ref + position + alt base)", label="Input mode",
        )
        with gr.Group() as snv_group:
            snv_seq = gr.Textbox(label="Reference sequence (ACGT)", value=EXAMPLE_SEQ, lines=3)
            with gr.Row():
                snv_pos = gr.Number(label="Variant position (1-indexed)", value=21, precision=0)
                snv_alt = gr.Textbox(label="Alt base (A/C/G/T)", value="A", max_lines=1)
        with gr.Group(visible=False) as pair_group:
            with gr.Row():
                seq_wt = gr.Textbox(label="Reference (ACGT)", value=EXAMPLE_SEQ, lines=3)
                seq_mut = gr.Textbox(label="Variant (ACGT)",
                                     value=EXAMPLE_SEQ[:20] + "A" + EXAMPLE_SEQ[21:], lines=3)
        with gr.Row():
            var_reduce = gr.Dropdown(label="Reduce", choices=REDUCE_CHOICES, value="mean")
            var_rc = gr.Checkbox(label="Average with reverse complement", value=True)
        var_btn = gr.Button("Predict variant effect", variant="primary")
        var_out = gr.Textbox(label="Result", interactive=False, lines=8)
        var_plot = gr.BarPlot(x="sequence", y="logprob",
                              title="reference vs variant logprob", height=220)

        var_mode.change(
            lambda mode: (gr.update(visible=mode.startswith("SNV")),
                          gr.update(visible=not mode.startswith("SNV"))),
            inputs=[var_mode], outputs=[snv_group, pair_group],
        )
        var_btn.click(
            action_variant,
            inputs=[model_dd, var_mode, seq_wt, seq_mut, snv_seq, snv_pos, snv_alt,
                    var_rc, var_reduce],
            outputs=[var_out, var_plot],
        )

    # --- Embeddings ---
    with gr.Tab("Embeddings"):
        gr.Markdown(
            "Extract hidden-state embeddings from a layer (for downstream ML, "
            "clustering, variant features). Download as `.npy`."
        )
        seq_emb = gr.Textbox(label="DNA sequence (ACGT)", value=EXAMPLE_SEQ, lines=3)
        emb_layer = gr.Dropdown(
            label="Layer", choices=EMBED_LAYERS, value=DEFAULT_EMBED_LAYER,
            allow_custom_value=True, info="e.g. blocks.10.mlp.l3",
        )
        emb_btn = gr.Button("Extract embeddings", variant="primary")
        emb_out = gr.Textbox(label="Result", interactive=False, lines=7)
        emb_file = gr.File(label="Download embeddings (.npy)")
        emb_btn.click(action_embed, inputs=[model_dd, seq_emb, emb_layer],
                      outputs=[emb_out, emb_file])

    # --- Batch ---
    with gr.Tab("Batch"):
        gr.Markdown(
            "Score many sequences at once. Paste one per line (or FASTA), or "
            "upload a `.fasta`/`.fa` file. Results are sortable; download as CSV."
        )
        batch_text = gr.Textbox(
            label="Sequences (one per line, or FASTA)",
            placeholder="ACGT...\nACGT...\n>id1\nACGT...",
            lines=6,
        )
        batch_file = gr.File(label="…or upload FASTA", file_types=[".fasta", ".fa", ".fna", ".txt"])
        with gr.Row():
            batch_reduce = gr.Dropdown(label="Reduce", choices=REDUCE_CHOICES, value="mean")
            batch_rc = gr.Checkbox(label="Average with reverse complement", value=False)
        batch_btn = gr.Button("Score batch", variant="primary")
        batch_out = gr.Textbox(label="Status", interactive=False, lines=2)
        batch_df = gr.Dataframe(label="Scores", interactive=False, wrap=True)
        batch_csv = gr.File(label="Download CSV")
        batch_btn.click(
            action_batch,
            inputs=[model_dd, batch_text, batch_file, batch_rc, batch_reduce],
            outputs=[batch_out, batch_df, batch_csv],
        )

    # --- Generate ---
    with gr.Tab("Generate"):
        gr.Markdown("Autoregressive continuation from a prompt.")
        seq_gen = gr.Textbox(label="DNA prompt (ACGT)", value=EXAMPLE_SEQ, lines=3)
        with gr.Row():
            n_tok = gr.Slider(1, 2048, value=64, step=1, label="n_tokens")
            temp = gr.Slider(0.05, 4.0, value=1.0, step=0.05, label="temperature")
            tk = gr.Slider(1, 4, value=4, step=1, label="top_k")
            tp = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="top_p")
        gen_btn = gr.Button("Generate", variant="primary")
        gen_out = gr.Textbox(label="Result", interactive=False, lines=14)
        gen_btn.click(action_generate, inputs=[model_dd, seq_gen, n_tok, temp, tk, tp],
                      outputs=[gen_out])

    # --- BRCA1 VEP (flagship application) ---
    with gr.Tab("BRCA1 VEP"):
        gr.Markdown(
            "**Evo 2's flagship application** — zero-shot prediction of *BRCA1* "
            "variant effects (Findlay et al. 2018 saturation-mutagenesis set). "
            "Scores each variant's Δ delta-likelihood and reports AUROC against "
            "the experimental loss-of-function classification. Uses the bundled "
            "chr17 reference + variant table; slow on MPS, so start small."
        )
        with gr.Row():
            brca_limit = gr.Slider(10, 500, value=40, step=10,
                                   label="variants to score (stratified sample)")
            brca_window = gr.Slider(512, 8192, value=1024, step=512,
                                    label="context window (bp)")
        brca_btn = gr.Button("Run BRCA1 variant-effect analysis", variant="primary")
        brca_out = gr.Textbox(label="Result", interactive=False, lines=6)
        brca_plot = gr.BarPlot(x="class", y="mean delta",
                               title="mean Δ-likelihood by class (lower = disruptive)",
                               height=220)
        brca_table = gr.Dataframe(label="Per-variant delta scores", interactive=False,
                                  wrap=True)
        brca_btn.click(action_brca1, inputs=[model_dd, brca_limit, brca_window],
                       outputs=[brca_out, brca_table, brca_plot])

    # --- Gene Completion benchmark ---
    with gr.Tab("Gene Completion"):
        gr.Markdown(
            "**Gene completion benchmark** (prokaryote/archaea panel from the Evo 2 "
            "paper). Each gene is prompted with ~1 kb upstream + 30% of its coding "
            "region; the model completes the CDS, which is translated and aligned to "
            "the reference protein for **% amino-acid recovery**. 4 genes "
            "(ftsZ, secY, dnaK, gyrA)."
        )
        with gr.Row():
            gc_maxgen = gr.Slider(100, 1200, value=400, step=50,
                                  label="tokens to generate per gene")
            gc_temp = gr.Slider(0.05, 1.5, value=0.7, step=0.05, label="temperature")
        gc_btn = gr.Button("Run gene-completion benchmark", variant="primary")
        gc_out = gr.Textbox(label="Result", interactive=False, lines=5)
        gc_table = gr.Dataframe(label="Per-gene AA recovery", interactive=False, wrap=True)
        gc_btn.click(action_gene_completion, inputs=[model_dd, gc_maxgen, gc_temp],
                     outputs=[gc_out, gc_table])

    gr.Markdown(
        "---\n"
        "**Tips**\n"
        "- Click *Load model* first — the first use downloads/loads the checkpoint "
        "(~14 GB for 7B, ~4 GB for 1B), which can take a minute or two.\n"
        "- `evo2_7b_base` is the default and the one to trust. `evo2_1b_base` is FP8-degraded.\n"
        "- Score/Variant/Batch: *mean* is per-base average log-likelihood; *sum* is the "
        "pseudo-log-likelihood (PLL).\n"
        "- On a 16–18 GB Mac, keep sequences short — long prompts can exhaust MPS memory."
    )


def main() -> int:
    host = os.environ.get("EVO2MAC_HOST", "127.0.0.1")
    port = int(os.environ.get("EVO2MAC_PORT", "7860"))
    share = os.environ.get("EVO2MAC_SHARE", "0") == "1"
    demo.launch(server_name=host, server_port=port, share=share, inbrowser=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
