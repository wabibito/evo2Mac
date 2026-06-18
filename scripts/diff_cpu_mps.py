#!/usr/bin/env python3
"""
Localize where evo2_20b's forward diverges between CPU and MPS, layer by layer.
Same code/weights/input on both -> any growing divergence pinpoints an
MPS-specific numerical issue (leading theory for the 20B being near-random).

Each device runs in its OWN process (avoids holding two 40 GB copies and avoids
vortex caching device-specific filter state), dumping per-layer activations to
.pt files; a final pass diffs them.

    python scripts/diff_cpu_mps.py --model evo2_20b [--seq-len 64]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

import torch

CAPTURE = r"""
import os, sys, warnings, re
warnings.filterwarnings("ignore")
os.environ["EVO2MPS_FP8_EMULATION"] = "0"
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
import torch
from evo2 import Evo2

model_name, device, seq_len, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
torch.manual_seed(0)
m = Evo2(model_name)
if device == "cpu":
    m.model = m.model.to("cpu"); m.device = "cpu"
    if hasattr(m.model, "block_idx_to_device"):
        for k in list(m.model.block_idx_to_device): m.model.block_idx_to_device[k] = "cpu"

n_blocks = 1 + max(int(re.search(r"blocks\.(\d+)", n).group(1))
                   for n,_ in m.model.named_modules() if re.search(r"blocks\.\d+$", n))
names = ["embedding_layer"] + [f"blocks.0.{c}" for c in
          ("pre_norm","projections","filter","out_filter_dense","mlp")] \
        + [f"blocks.{i}" for i in range(n_blocks)] + ["norm"]
acts = {}
def mk(nm):
    def h(_m,_i,o):
        t = o[0] if isinstance(o, tuple) else o
        if torch.is_tensor(t): acts[nm] = t.detach().float().cpu()
    return h
hs = []
for nm in names:
    try: hs.append(m.model.get_submodule(nm).register_forward_hook(mk(nm)))
    except Exception: pass
seq = "ACGT" * (seq_len // 4)
ids = torch.tensor(m.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0)
with torch.no_grad(): m(ids)
for h in hs: h.remove()
torch.save(acts, out)
print(f"captured {len(acts)} layers on {device} -> {out}")
"""


def run_capture(model, device, seq_len, out_path):
    script = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    script.write(CAPTURE); script.close()
    r = subprocess.run([sys.executable, script.name, model, device, str(seq_len), out_path],
                       capture_output=True, text=True)
    os.unlink(script.name)
    ok = os.path.exists(out_path)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    print(f"  [{device}] {'ok' if ok else 'FAILED'}: " + " | ".join(tail[-1:]))
    if not ok:
        print("   stderr:", "\n   ".join((r.stderr or "").strip().splitlines()[-6:]))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_20b")
    ap.add_argument("--seq-len", type=int, default=64)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    tmp = tempfile.mkdtemp(prefix="evo2_diff_")
    cpu_f, mps_f = os.path.join(tmp, "cpu.pt"), os.path.join(tmp, "mps.pt")
    print(f"capturing {args.model} on each device (separate processes) ...")
    if not run_capture(args.model, "mps", args.seq_len, mps_f):
        return 1
    if not run_capture(args.model, "cpu", args.seq_len, cpu_f):
        return 1

    cpu = torch.load(cpu_f, weights_only=False)
    mps = torch.load(mps_f, weights_only=False)
    n_blocks = 1 + max(int(re.search(r"blocks\.(\d+)", k).group(1))
                       for k in cpu if re.fullmatch(r"blocks\.\d+", k))
    order = (["embedding_layer", "blocks.0.pre_norm", "blocks.0.projections",
              "blocks.0.filter", "blocks.0.out_filter_dense", "blocks.0.mlp"]
             + [f"blocks.{i}" for i in range(n_blocks)] + ["norm"])

    print("\nper-layer CPU-vs-MPS relative error:")
    print(f"  {'layer':26} {'rel-err':>11}")
    print("  " + "-" * 40)
    first_big = None
    for name in order:
        if name not in cpu or name not in mps:
            continue
        a, b = cpu[name], mps[name]
        r = float("nan") if a.shape != b.shape else \
            ((a - b).abs().mean() / a.abs().mean().clamp_min(1e-9)).item()
        flag = ""
        if r == r and r > 0.05 and first_big is None and re.fullmatch(r"blocks\.\d+", name):
            first_big = name; flag = "  <-- first large divergence"
        print(f"  {name:26} {r:>11.3e}{flag}")

    print()
    print(f"First large (>5%) block divergence: {first_big}" if first_big else
          "No large block divergence — CPU and MPS agree (20B issue is not CPU-vs-MPS).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
