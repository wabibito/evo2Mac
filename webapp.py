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
    - Variant    — score a wild-type vs a mutant sequence and report the
                   Δ log-likelihood (evo2's flagship effect-prediction use).
    - Embeddings — extract hidden-state embeddings from a chosen layer;
                   download as .npy.
    - Batch      — paste many sequences or upload a FASTA file and score them
                   all into a sortable table (download as CSV).

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
    "evo2_7b_base":         "~14 GB   bf16-native, recommended",
    "evo2_7b":              "~14 GB   bf16-native (1M ctx)",
    "evo2_7b_262k":         "~14 GB   bf16-native (262K ctx)",
    "evo2_7b_microviridae": "~14 GB   bf16-native",
    "evo2_1b_base":         "~4 GB    FP8 e4m3 emulation (auto)",
}

DEFAULT_MODEL = "evo2_7b_base"

# Models that are FP8-trained and so run with e4m3 emulation on Mac (the Evo2
# loader applies it automatically). Without it they'd be near-random; with it
# the 1B recovers to ~75% forward accuracy / ~67% generation identity. Still a
# notch below the bf16-native 7B, so we surface an informational banner.
FP8_EMULATED = {"evo2_1b_base"}

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


def _device_info() -> str:
    if torch.cuda.is_available():
        return "CUDA available (will use cuda:0)"
    if torch.backends.mps.is_available():
        return "MPS available (will use mps)"
    return "Falling back to CPU — this will be slow"


def _get_model(name: str):
    if name not in _model_cache:
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

def action_load(model_name: str, progress: gr.Progress = gr.Progress()):
    progress(0, desc=f"Loading {model_name} ...")
    t0 = time.time()
    m, err = _require_model(model_name)
    if err:
        return err
    warn = ""
    if model_name in FP8_EMULATED:
        warn = (
            "Note: FP8-trained checkpoint running with e4m3 emulation (no "
            "Transformer Engine on Mac). Recovers most of the FP8 accuracy; "
            "evo2_7b_base is still the most accurate.\n\n"
        )
    return (
        f"{warn}{model_name} loaded in {time.time() - t0:.1f}s on device {m.device}.\n"
        f"Ready: Forward / Score / Variant / Embeddings / Batch."
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

def action_variant(model_name: str, wt: str, mut: str, rc_avg: bool,
                   reduce_method: str):
    for label, s in (("Wild-type", wt), ("Mutant", mut)):
        err = _validate_dna(s)
        if err:
            return f"{label}: {err}", None
    wt, mut = _clean(wt), _clean(mut)
    m, merr = _require_model(model_name)
    if merr:
        return merr, None

    t0 = time.time()
    scores = m.score_sequences(
        [wt, mut],
        batch_size=2,
        reduce_method=reduce_method,
        average_reverse_complement=rc_avg,
    )
    elapsed = time.time() - t0
    wt_s, mut_s = float(scores[0]), float(scores[1])
    delta = mut_s - wt_s
    verdict = (
        "mutant is *more* likely (Δ > 0)" if delta > 0
        else "mutant is *less* likely (Δ < 0 — possible deleterious effect)"
        if delta < 0 else "no change"
    )
    msg = (
        f"Scored 2 sequences in {elapsed:.2f}s on {m.device}\n"
        f"WT  logprob: {wt_s:.4f}\n"
        f"Mut logprob: {mut_s:.4f}\n"
        f"Δ (mut − wt): {delta:+.4f}  →  {verdict}\n\n"
        f"More negative Δ = the variant lowers the model's likelihood, a proxy "
        f"for functional disruption. Compare against many variants for calibration."
    )
    df = pd.DataFrame(
        {"sequence": ["wild-type", "mutant"], "logprob": [wt_s, mut_s]}
    )
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


# --- UI -----------------------------------------------------------------------

REDUCE_CHOICES = [("mean (per-base avg)", "mean"), ("sum (PLL)", "sum")]

with gr.Blocks(title="evo2Mac", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"# 🧬 evo2Mac\n"
        f"Apple Silicon port of [Evo 2](https://github.com/arcinstitute/evo2). "
        f"{_device_info()}.\n\n"
        f"> **Use a 7B-8k checkpoint** (`evo2_7b_base`, the default) for real "
        f"results — bf16-native, matches upstream's H100 reference. "
        f"`evo2_1b_base` is FP8-degraded (near-random on Mac)."
    )

    with gr.Row():
        model_dd = gr.Dropdown(
            label="Model",
            choices=[(f"{name}  —  {info}", name) for name, info in MAC_FEASIBLE.items()],
            value=DEFAULT_MODEL,
            scale=4,
        )
        load_btn = gr.Button("Load model", variant="primary", scale=1)
    fp8_md = gr.Markdown(_fp8_banner(DEFAULT_MODEL))
    load_status = gr.Textbox(label="Status", interactive=False, lines=2)
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

    # --- Variant ---
    with gr.Tab("Variant"):
        gr.Markdown(
            "Score a **wild-type** vs a **mutant** sequence and report the "
            "Δ log-likelihood — evo2's flagship variant-effect use case."
        )
        with gr.Row():
            seq_wt = gr.Textbox(label="Wild-type (ACGT)", value=EXAMPLE_SEQ, lines=3)
            seq_mut = gr.Textbox(
                label="Mutant (ACGT)",
                value=EXAMPLE_SEQ[:20] + "A" + EXAMPLE_SEQ[21:], lines=3,
            )
        with gr.Row():
            var_reduce = gr.Dropdown(label="Reduce", choices=REDUCE_CHOICES, value="mean")
            var_rc = gr.Checkbox(label="Average with reverse complement", value=False)
        var_btn = gr.Button("Score variant", variant="primary")
        var_out = gr.Textbox(label="Result", interactive=False, lines=7)
        var_plot = gr.BarPlot(x="sequence", y="logprob", title="WT vs mutant logprob",
                              height=220)
        var_btn.click(action_variant, inputs=[model_dd, seq_wt, seq_mut, var_rc, var_reduce],
                      outputs=[var_out, var_plot])

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
