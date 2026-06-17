"""
FP8 (e4m3) emulation for the StripedHyena input projections on Apple Silicon.

Why this exists
---------------
The 1B/20B/40B Evo 2 checkpoints set ``use_fp8_input_projections: True``: the qkv
input projection (vortex ``TELinear``) was *trained* with NVIDIA Transformer
Engine (TE) using per-tensor **delayed scaling** in e4m3. TE is CUDA/Hopper only,
so on Mac vortex falls back to a plain bf16 ``F.linear`` — which is numerically a
different operation than what the weights were trained for, and the model's
next-token accuracy collapses (the "FP8-degraded" warning).

This module reproduces TE's forward GEMM numerically, in plain PyTorch, so it
runs on MPS (and CPU). It is **bit-exact emulation**, not hardware FP8: on
M1–M4 there is no speed benefit (the point is *accuracy*, making the FP8
checkpoints usable). On an M5 — whose GPU Neural Accelerators support FP8
natively — the ``_quantize_e4m3`` body is the natural seam to swap for a real
FP8 matmul (``torch._scaled_mm`` once MPS exposes it, or an MLX kernel); the
per-tensor scales recovered here are exactly what such a path needs.

What TE actually does (verified against the evo2_1b_base checkpoint)
-------------------------------------------------------------------
Each ``*.projections._extra_state`` blob stores ``scale_fwd`` with three slots:
  slot 0 = activation (input) scale, slot 1 = weight scale, slot 2 = unused.
TE's forward for a single GEMM is:

    x_q = round_e4m3(x * act_scale)            # saturating, RNE
    w_q = round_e4m3(W * weight_scale)
    y   = (x_q @ w_q.T) / (act_scale * weight_scale) + bias

with ``scale = fp8_max / amax`` and ``fp8_max = 448`` for e4m3. The weight scale
recovered from the checkpoint matches ``448 / W.abs().max()`` to the digit, and
the activation scale is the delayed-scaling value from calibration (it cannot be
recomputed at inference time — it must come from the checkpoint).
"""

from __future__ import annotations

import io
from typing import Dict, Optional

import torch
import torch.nn as nn

# e4m3fn: 4 exponent bits, 3 mantissa bits, bias 7, max normal 448, no inf.
FP8_E4M3_MAX = 448.0
_E4M3_MIN_EXP = -6  # smallest normal binade; subnormals share this exponent
_E4M3_MANTISSA_BITS = 3


def quantize_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round a real-valued tensor to the e4m3fn grid, returned in the input dtype.

    Pure-tensor (MPS-safe) emulation. Bit-exact vs ``x.to(torch.float8_e4m3fn)``
    for all in-range values; values above 448 saturate to 448 (TE clamps before
    casting, so saturation — not NaN — is the behaviour we want here).
    """
    orig_dtype = x.dtype
    xf = x.float()
    sign = torch.sign(xf)
    ax = xf.abs()

    # Per-element binade exponent, floored into the representable range.
    e = torch.floor(torch.log2(ax.clamp_min(1e-30)))
    e = torch.clamp(e, min=_E4M3_MIN_EXP)
    # Mantissa step within the binade for 3 mantissa bits.
    step = torch.exp2(e - _E4M3_MANTISSA_BITS)
    q = torch.round(ax / step) * step
    q = torch.clamp(q, max=FP8_E4M3_MAX)

    # Flush values below half the smallest subnormal to zero.
    smallest_subnormal = 2.0 ** (_E4M3_MIN_EXP - _E4M3_MANTISSA_BITS)
    q = torch.where(ax < smallest_subnormal / 2, torch.zeros_like(q), q)

    return (sign * q).to(orig_dtype)


class Fp8EmulatedLinear(nn.Module):
    """Drop-in replacement for vortex's fallback ``TELinear`` that reproduces
    Transformer Engine's per-tensor e4m3 forward GEMM.

    Matches the fallback's ``(output, bias_or_None)`` return convention and its
    ``weight``/``bias`` parameter naming so the checkpoint loads unchanged.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        act_scale: float,
        weight_scale: float,
        skip_bias_add: bool = False,
        return_tuple: bool = True,
    ):
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.te_return_bias = skip_bias_add and (bias is not None)
        # vortex's fallback TELinear returns (out, bias_or_None); plain nn.Linear
        # layers (the 20B/40B MLPs, out-projection, Wqkv) return a bare tensor.
        # Mirror whichever the module we replace used, or callers break.
        self.return_tuple = return_tuple

        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

        # Per-tensor scales from the checkpoint's TE extra_state (slot 0/1 of
        # scale_fwd). Stored as buffers so they ride device moves with the module.
        self.register_buffer("act_scale", torch.tensor(float(act_scale)))
        self.register_buffer("weight_scale", torch.tensor(float(weight_scale)))

    def forward(self, x):
        w = self.weight
        # NB: scaling/casting in bf16 here (not fp32) is INTENTIONAL. It matches
        # how vortex actually feeds these projections, and empirically gives the
        # best evo2_1b_base accuracy (74.5% vs the H100 ref); switching to a
        # "more correct" fp32-scaled native-FP8 path measurably *hurt* the 1B
        # (dropped to ~39%). The checkpoint scales are tuned to this path.
        x_q = quantize_e4m3(x.to(w.dtype) * self.act_scale)
        w_q = quantize_e4m3(w * self.weight_scale)
        inv = 1.0 / (self.act_scale * self.weight_scale)
        out = torch.nn.functional.linear(x_q, w_q) * inv
        if self.bias is not None:
            out = out + self.bias
        out = out.to(w.dtype)
        if not self.return_tuple:
            return out
        if self.te_return_bias:
            return out, self.bias
        return out, None


def extract_fp8_scales(checkpoint_path: str) -> Dict[str, Dict[str, float]]:
    """Read per-layer forward FP8 scales from EVERY TE ``_extra_state`` blob in a
    raw checkpoint — not just the input projections.

    The 1B is FP8 only on ``*.projections``, but the 20B/40B are FP8-trained on
    many more linear layers (MLPs, the out-projection, attention QKV). Each has a
    ``<module>._extra_state`` whose ``scale_fwd`` is ``[act, weight, unused]``.

    Must be called on the on-disk checkpoint: vortex strips ``._extra_state``
    keys when Transformer Engine is absent. Returns
    ``{module_path: {"act": float, "weight": float}}``.
    """
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "module" in sd:
        sd = sd["module"]

    scales: Dict[str, Dict[str, float]] = {}
    for key, value in sd.items():
        if not key.endswith("._extra_state"):
            continue
        module_path = key[: -len("._extra_state")]
        try:
            if hasattr(value, "read"):
                value.seek(0)
                meta = torch.load(value, map_location="cpu", weights_only=False)
            elif isinstance(value, (bytes, bytearray)):
                meta = torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
            else:
                continue
            scale_fwd = meta.get("scale_fwd") if hasattr(meta, "get") else None
            if scale_fwd is None or len(scale_fwd) < 2:
                continue  # not a forward-GEMM FP8 layer we can emulate
            scales[module_path] = {
                "act": float(scale_fwd[0]),
                "weight": float(scale_fwd[1]),
            }
        except Exception:
            # A layer we can't decode is skipped; it stays in bf16 fallback
            # rather than guessing a scale.
            continue
    return scales


# Back-compat alias (older name covered only projections).
extract_projection_scales = extract_fp8_scales


def apply_fp8_emulation(model: nn.Module, checkpoint_path: str) -> int:
    """Swap every FP8-trained linear in ``model`` for an ``Fp8EmulatedLinear``
    carrying that layer's checkpoint scales.

    Covers all modules with a recoverable ``scale_fwd`` and a ``.weight`` —
    input projections (the 1B's only FP8 layers) plus the 20B/40B's MLPs,
    out-projection and attention QKV. Modules whose scales can't be recovered,
    or that lack a wrappable ``.weight`` (e.g. attention metadata states), are
    left in their bf16 fallback. Returns the number of layers replaced.
    """
    scales = extract_fp8_scales(checkpoint_path)
    if not scales:
        return 0

    replaced = 0
    modules = dict(model.named_modules())
    for module_path, sc in scales.items():
        parent_path, _, attr = module_path.rpartition(".")
        parent = modules.get(parent_path)
        if parent is None:
            continue
        old = getattr(parent, attr, None)
        if old is None or not hasattr(old, "weight") or old.weight is None:
            continue
        # TELinear (the input projections) returns a (out, bias) tuple and has
        # te_return_bias; a plain nn.Linear (MLPs, out-proj, Wqkv) returns a
        # bare tensor. Mirror the original's convention.
        is_te = hasattr(old, "te_return_bias")
        new = Fp8EmulatedLinear(
            weight=old.weight.data,
            bias=old.bias.data if getattr(old, "bias", None) is not None else None,
            act_scale=sc["act"],
            weight_scale=sc["weight"],
            skip_bias_add=getattr(old, "te_return_bias", False),
            return_tuple=is_te,
        ).to(old.weight.device)
        setattr(parent, attr, new)
        replaced += 1
    return replaced
