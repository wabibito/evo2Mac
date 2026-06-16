#!/usr/bin/env python3
"""
Compare evo2Mac (MPS) numerical output against upstream's published reference
values for arcinstitute/evo2 on CUDA.

Upstream `evo2/test/test_evo2.py` runs a forward pass over a bundled
prompts.csv (8K-context DNA prompts) and bakes in reference loss / accuracy
numbers measured on H100 / FP8 + flash-attn for each checkpoint:

    Evo 2 1B base:  Loss ~0.502, Accuracy ~79.56%
    Evo 2 7B base:  Loss ~0.352, Accuracy ~85.92%
    Evo 2 7B:       Loss ~0.348, Accuracy ~86.35%
    Evo 2 20B:      Loss ~0.217, Accuracy ~91.67%
    Evo 2 40B:      Loss ~0.216, Accuracy ~91.67%

Our Mac port runs in bf16 with SDPA (no FP8, no flash-attn) so a small
numerical drift is expected. The argmax-based accuracy should be nearly
identical; the cross-entropy loss may drift by O(1e-3) to O(1e-2) but a
correct port should NOT drift by more than that.

Usage:
    conda activate evo2Mac
    python scripts/compare_to_upstream.py --model evo2_1b_base
    python scripts/compare_to_upstream.py --model evo2_7b_base   # 32GB+ Mac
    python scripts/compare_to_upstream.py --device cpu           # force CPU
    python scripts/compare_to_upstream.py --compare-devices      # CPU vs MPS

CPU vs MPS (--compare-devices):
    Runs the *same* prompts on both CPU and MPS back to back and reports the
    per-sequence and mean drift between the two. CPU uses fp32/bf16 PyTorch
    reference kernels; MPS uses Metal kernels. If CPU matches upstream but MPS
    does not, the drift is MPS-specific (Metal SDPA / rotary / Hyena FFT
    rounding) rather than a structural port bug. This is the diagnostic for
    the open drift issue documented in README.md.

Exit code:
    0 — within tolerance (port matches upstream)
    1 — outside fail tolerance (port may be broken)
    2 — usage / load error
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from importlib import resources
from pathlib import Path

# Numbers from upstream evo2/test/test_evo2.py docstring + expected_metrics
# (commit 3a4d1d0). Measured on H100 with FP8 + flash-attn.
UPSTREAM_REFERENCE = {
    "evo2_1b_base": {"loss": 0.501953125,  "acc": 79.556},
    "evo2_7b_base": {"loss": 0.3520508,    "acc": 85.921},
    "evo2_7b":      {"loss": 0.3476563,    "acc": 86.346},
    "evo2_20b":     {"loss": 0.2166748046875, "acc": 91.666},
    "evo2_40b":     {"loss": 0.2159424,    "acc": 91.673},
    "evo2_40b_base":{"loss": 0.2149658,    "acc": 91.741},
}

# Tolerance bands. bf16+SDPA on MPS vs bf16+FP8+FlashAttn on H100 are not
# bit-equal; some drift is healthy. These thresholds are deliberately loose
# enough to absorb that but tight enough to catch a real bug.
LOSS_WARN  = 0.05    # 5e-2 absolute drift in cross-entropy nats
LOSS_FAIL  = 0.15    # 1.5e-1 — well outside numerical noise
ACC_WARN   = 1.5     # percentage points
ACC_FAIL   = 5.0     # 5 pp would mean the model is meaningfully degraded

MAC_FEASIBLE = {"evo2_1b_base", "evo2_7b", "evo2_7b_base", "evo2_7b_262k", "evo2_7b_microviridae"}

# Models whose upstream config sets use_fp8_input_projections: True. These were
# trained WITH FP8 input projections and require Transformer Engine (CUDA/Hopper)
# for numerical accuracy — upstream's own README says so. Forcing them to bf16
# (which is what the Mac port must do to load them at all, since TE is CUDA-only)
# strips those projections and degrades the model to near-random next-token
# prediction (loss ~ln(4)=1.386, acc ~25-35%). This is NOT a port bug and NOT
# closeable in bf16 — it's inherent to running an FP8 checkpoint without FP8.
#
# Only the 7B-8k checkpoints (evo2_7b, evo2_7b_base) ship with FP8 OFF upstream,
# so they are the models the Mac/MPS port can reproduce correctly. Validate the
# port against those; treat the 1B/20B/40B numbers as FP8-degraded.
FP8_REQUIRED = {"evo2_1b_base", "evo2_20b", "evo2_40b", "evo2_40b_base"}

# The 7B-1m config also has FP8 on upstream; 7B-8k (evo2_7b / evo2_7b_base) does
# not. Default the drift check to the model that actually runs correctly in bf16.
DEFAULT_MODEL = "evo2_7b_base"


def read_prompts() -> list[str]:
    """Load upstream's bundled prompts.csv (same one its own test uses)."""
    with resources.path("evo2.test.data", "prompts.csv") as p:
        path = Path(p)
    seqs: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if row and row[0].strip():
                seqs.append(row[0].strip())
    return seqs


def forward_pass(model, sequences, max_seqs: int | None) -> tuple[list[float], list[float]]:
    """Replicates upstream test_forward_pass but device-aware."""
    import torch
    import torch.nn.functional as F

    if max_seqs is not None:
        sequences = sequences[:max_seqs]

    losses: list[float] = []
    accuracies: list[float] = []

    for i, seq in enumerate(sequences, 1):
        ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).to(model.device)
        with torch.inference_mode():
            out = model.model.forward(ids.unsqueeze(0))
        logits = out[0] if isinstance(out, tuple) else out

        target_ids = ids[1:].long()
        pred_logits = logits[0, :-1, :]

        # bf16 -> fp32 for loss to avoid underflow in cross-entropy.
        loss = F.cross_entropy(pred_logits.float(), target_ids)
        pred_tokens = torch.argmax(pred_logits, dim=-1)
        acc = (target_ids == pred_tokens).float().mean().item()

        losses.append(loss.item())
        accuracies.append(acc)
        print(f"  seq {i:>2}/{len(sequences)}: loss={loss.item():.4f}  acc={acc*100:.2f}%",
              flush=True)

    return accuracies, losses


def force_device(model, device: str) -> None:
    """Move an already-loaded Evo2 wrapper to `device`, end to end.

    `Evo2.__init__` migrates both the module and StripedHyena's plain
    `block_idx_to_device` dict, but only for the auto-detected device. When we
    override the device after load we must repeat *both* steps, otherwise the
    final unembed does `x = x.to(block_idx_to_device[0])` and yanks activations
    back to the stale device — silently wrong on CPU, or an MPS/CPU mismatch
    crash. Keep this in lockstep with evo2/models.py.
    """
    import torch

    model.device = device
    model.model = model.model.to(device)
    if hasattr(model.model, "block_idx_to_device"):
        for k in list(model.model.block_idx_to_device):
            model.model.block_idx_to_device[k] = device

    # The Hyena filters lazily cache a time vector `self.t` (a plain attribute,
    # not a registered buffer) on whichever device the first forward ran on.
    # `.to()` does not migrate it, so a CPU->MPS switch leaves `self.t` on CPU
    # while `log_poles` moves to MPS -> "tensors on two devices" in
    # compute_filter. Invalidate the cache so it rebuilds on `device`.
    for module in model.model.modules():
        if hasattr(module, "t") and isinstance(getattr(module, "t"), torch.Tensor):
            module.t = None


def run_on_device(model, seqs, max_seqs, device: str):
    """Force `device`, run the forward pass, return (mean_loss, mean_acc_pct,
    per-seq losses, per-seq accs)."""
    import numpy as np
    import torch

    force_device(model, device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    print(f"\nrunning forward pass on {device} ...")
    t = time.time()
    accs, losses = forward_pass(model, seqs, max_seqs)
    elapsed = time.time() - t
    print(f"  {device} forward-pass time: {elapsed:.1f}s")
    return float(np.mean(losses)), float(np.mean(accs) * 100), losses, accs, elapsed


def compare_devices(model, seqs, max_seqs) -> int:
    """Run the same prompts on CPU and MPS and report the drift between them."""
    import torch

    if not torch.backends.mps.is_available():
        print("FAIL: --compare-devices needs MPS, which is not available here.")
        return 2

    cpu = run_on_device(model, seqs, max_seqs, "cpu")
    mps = run_on_device(model, seqs, max_seqs, "mps")

    cpu_loss, cpu_acc, cpu_losses, cpu_accs, cpu_t = cpu
    mps_loss, mps_acc, mps_losses, mps_accs, mps_t = mps

    print("\n" + "=" * 70)
    print("  per-sequence CPU vs MPS")
    print("=" * 70)
    print(f"  {'seq':>4}  {'cpu_loss':>9}  {'mps_loss':>9}  {'Δloss':>8}  "
          f"{'cpu_acc':>7}  {'mps_acc':>7}  {'Δacc':>7}")
    max_dloss = 0.0
    for i, (cl, ml, ca, ma) in enumerate(zip(cpu_losses, mps_losses, cpu_accs, mps_accs), 1):
        dloss = ml - cl
        dacc = (ma - ca) * 100
        max_dloss = max(max_dloss, abs(dloss))
        print(f"  {i:>4}  {cl:>9.4f}  {ml:>9.4f}  {dloss:>+8.4f}  "
              f"{ca*100:>6.2f}%  {ma*100:>6.2f}%  {dacc:>+6.2f}pp")

    print("=" * 70)
    print(f"  CPU : loss={cpu_loss:.4f}  acc={cpu_acc:.3f}%  ({cpu_t:.1f}s)")
    print(f"  MPS : loss={mps_loss:.4f}  acc={mps_acc:.3f}%  ({mps_t:.1f}s)")
    print(f"  Δ   : loss={mps_loss - cpu_loss:+.4f}  acc={mps_acc - cpu_acc:+.3f}pp  "
          f"max|Δloss/seq|={max_dloss:.4f}")
    if cpu_t > 0:
        print(f"  speed: MPS is {cpu_t / mps_t:.2f}x CPU" if mps_t > 0 else "")
    print("=" * 70)

    # Verdict: is the MPS drift away from CPU big enough to be the bug?
    if abs(mps_loss - cpu_loss) > LOSS_WARN or abs(mps_acc - cpu_acc) > ACC_WARN:
        print("  CPU and MPS diverge meaningfully — the drift is MPS-specific")
        print("  (Metal kernel rounding in SDPA / rotary / Hyena FFT).")
        print("  Compare each against the upstream reference to see which is correct.")
    else:
        print("  CPU and MPS agree — the drift (if any) is shared by both backends,")
        print("  i.e. it is NOT MPS-specific (likely bf16/SDPA vs FP8+flash-attn, or")
        print("  a structural port issue affecting both).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    choices=sorted(UPSTREAM_REFERENCE))
    ap.add_argument("--max-seqs", type=int, default=None,
                    help="cap number of prompts (default: all)")
    ap.add_argument("--device", default=None,
                    help="override device (mps/cpu/cuda:0)")
    ap.add_argument("--compare-devices", action="store_true",
                    help="run the same prompts on CPU and MPS and report the drift "
                         "between them (diagnostic for MPS-specific issues)")
    args = ap.parse_args()

    # CPU forward passes over 8K-context prompts take minutes; flush progress
    # so it streams through pipes/tee instead of buffering until exit.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    if args.model not in MAC_FEASIBLE:
        print(f"NOTE: {args.model} is not expected to run on Mac (FP8/Hopper required).")
        print(f"      Run anyway for diagnostics, but expect a load error.")

    if args.model in FP8_REQUIRED:
        print()
        print("=" * 70)
        print(f"  WARNING: {args.model} is trained with FP8 input projections")
        print(f"  (use_fp8_input_projections: True upstream) and requires")
        print(f"  Transformer Engine on a Hopper GPU for numerical accuracy.")
        print(f"  Transformer Engine is CUDA-only, so on Mac it loads in bf16")
        print(f"  with FP8 disabled — which degrades it to near-random output")
        print(f"  (expect loss ~1.3-1.4, acc ~25-35%). This is EXPECTED and is")
        print(f"  not a port bug; it cannot be fixed in bf16.")
        print(f"  To validate the Mac port, use a 7B-8k checkpoint instead:")
        print(f"      --model evo2_7b_base   (or evo2_7b)")
        print("=" * 70)

    import numpy as np
    import torch

    print(f"torch:         {torch.__version__}")
    print(f"cuda avail:    {torch.cuda.is_available()}")
    print(f"mps avail:     {torch.backends.mps.is_available()}")

    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(1)

    print(f"\nloading {args.model} ...")
    t0 = time.time()
    try:
        from evo2 import Evo2
        model = Evo2(args.model)
        if args.device is not None:
            force_device(model, args.device)
    except Exception as e:
        print(f"FAIL: could not load model: {e}")
        traceback.print_exc()
        return 2
    print(f"  device:    {model.device}")
    print(f"  loaded in: {time.time() - t0:.1f}s\n")

    print("loading upstream prompts.csv ...")
    seqs = read_prompts()
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    print(f"  {len(seqs)} prompts\n")

    if args.compare_devices:
        return compare_devices(model, seqs, args.max_seqs)

    print("running forward pass on each prompt ...")
    t1 = time.time()
    accs, losses = forward_pass(model, seqs, args.max_seqs)
    print(f"\ntotal forward-pass time: {time.time() - t1:.1f}s")

    mean_loss = float(np.mean(losses))
    mean_acc_pct = float(np.mean(accs) * 100)

    ref = UPSTREAM_REFERENCE[args.model]
    loss_delta = abs(mean_loss - ref["loss"])
    acc_delta = abs(mean_acc_pct - ref["acc"])

    print("\n" + "=" * 62)
    print(f"  results for {args.model} on {model.device}")
    print("=" * 62)
    print(f"  upstream (H100, FP8, flash-attn):")
    print(f"    loss  = {ref['loss']:.4f}")
    print(f"    acc   = {ref['acc']:.3f}%")
    print(f"  evo2Mac (this run):")
    print(f"    loss  = {mean_loss:.4f}    Δ = {loss_delta:.4f}")
    print(f"    acc   = {mean_acc_pct:.3f}%  Δ = {acc_delta:.3f} pp")
    print("=" * 62)

    failed = False
    if loss_delta > LOSS_FAIL:
        print(f"  FAIL: loss drift {loss_delta:.4f} exceeds {LOSS_FAIL} fail threshold")
        failed = True
    elif loss_delta > LOSS_WARN:
        print(f"  WARN: loss drift {loss_delta:.4f} exceeds {LOSS_WARN} warn threshold "
              "(bf16/SDPA can drift this much vs FP8+flash-attn — review)")
    else:
        print(f"  loss within ±{LOSS_WARN}: OK")

    if acc_delta > ACC_FAIL:
        print(f"  FAIL: accuracy drift {acc_delta:.3f}pp exceeds {ACC_FAIL}pp fail threshold")
        failed = True
    elif acc_delta > ACC_WARN:
        print(f"  WARN: accuracy drift {acc_delta:.3f}pp exceeds {ACC_WARN}pp warn threshold")
    else:
        print(f"  accuracy within ±{ACC_WARN}pp: OK")

    if failed:
        print("\nport appears to be producing wrong outputs.")
        return 1
    print("\nport matches upstream within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
