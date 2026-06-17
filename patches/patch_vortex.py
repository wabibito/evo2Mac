#!/usr/bin/env python3
"""
Patch the installed `vortex` package (PyPI distribution: `vtx`) for macOS/MPS.

These edits fix three CUDA-isms in the StripedHyena 2 reference implementation
that crash on Apple Silicon. They are device-aware: when CUDA is available the
patched code falls back to the original CUDA path.

The patcher is idempotent — re-running it on already-patched files is a no-op.
It writes a `.bak` next to each file the first time it edits that file.

Run after `pip install vtx`:

    python patches/patch_vortex.py

To undo:

    python patches/patch_vortex.py --restore
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util  # explicit: `import importlib` alone doesn't bind .util
import re
import shutil
import sys
from pathlib import Path


PATCH_MARK = "# evo2Mac patch"


def vortex_root() -> Path:
    """Resolve the installed `vortex` package directory."""
    try:
        spec = importlib.util.find_spec("vortex")
    except Exception as e:
        sys.exit(f"Could not import vortex: {e}\nDid you `pip install vtx` first?")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("`vortex` package not found. Run `pip install vtx` first.")
    return Path(next(iter(spec.submodule_search_locations)))


def back_up(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)


def restore(root: Path) -> None:
    n = 0
    for bak in root.rglob("*.bak"):
        target = bak.with_suffix("")
        shutil.copy2(bak, target)
        bak.unlink()
        print(f"  restored {target.relative_to(root)}")
        n += 1
    print(f"restored {n} file(s)")


def edit(path: Path, edits: list[tuple[str, str, str]]) -> int:
    """
    Apply a list of (pattern, replacement, label) edits to `path`.

    Each edit is skipped if the file already contains the patch mark for that
    label, so re-running is safe. Returns the number of edits actually applied.
    """
    if not path.exists():
        print(f"  ! missing: {path}")
        return 0
    text = path.read_text()
    applied = 0
    for pattern, replacement, label in edits:
        mark = f"{PATCH_MARK}: {label}"
        if mark in text:
            print(f"  - {path.name}: '{label}' already patched")
            continue
        new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if n == 0:
            print(f"  ! {path.name}: pattern for '{label}' not found")
            continue
        text = new_text
        applied += 1
        print(f"  + {path.name}: applied '{label}'")
    if applied:
        back_up(path)
        path.write_text(text)
    return applied


def patch_engine(root: Path) -> int:
    """
    vortex/model/engine.py
      1. torch.autocast("cuda")     -> device-aware autocast
      2. torch.fft.fft().repeat()   -> .unsqueeze().expand()
         (MPS doesn't support .repeat on complex tensors in PT 2.x)
    """
    p = root / "model" / "engine.py"
    edits = [
        (
            r'with torch\.autocast\(\s*"cuda"\s*\):',
            (
                'with torch.autocast(  # evo2Mac patch: autocast\n'
                '            "cuda" if torch.cuda.is_available()\n'
                '            else ("mps" if torch.backends.mps.is_available() else "cpu"),\n'
                '            dtype=torch.bfloat16,  # MPS defaults to fp16; Hyena FFT needs bf16 range\n'
                '        ):'
            ),
            "autocast",
        ),
        (
            r'state_S\s*=\s*torch\.fft\.fft\(\s*state_s\s*,\s*n=fft_size\s*\)\.repeat\(\s*bs\s*,\s*1\s*,\s*1\s*,\s*1\s*\)',
            (
                "state_S = torch.fft.fft(state_s, n=fft_size)  # evo2Mac patch: fft_repeat\n"
                "        state_S = state_S.unsqueeze(0).expand(bs, -1, -1, -1)"
            ),
            "fft_repeat",
        ),
    ]
    return edit(p, edits)


def patch_generation(root: Path) -> int:
    """
    vortex/model/generation.py
      torch.cuda.memory_allocated(device=x.device)  ->  torch.memory_allocated(x.device)

    There are two call sites (one logged after prefill, one at the end). Both
    must be patched; using replace-all on a regex anchored to the exact form.
    """
    p = root / "model" / "generation.py"
    if not p.exists():
        print(f"  ! missing: {p}")
        return 0
    text = p.read_text()
    label = "mem_allocated"
    mark = f"{PATCH_MARK}: {label}"
    if mark in text:
        print(f"  - generation.py: '{label}' already patched")
        return 0
    pattern = re.compile(r"torch\.cuda\.memory_allocated\(\s*device\s*=\s*x\.device\s*\)")
    new_text, n = pattern.subn(
        f"torch.memory_allocated(x.device)  {mark}", text
    )
    if n == 0:
        print(f"  ! generation.py: pattern for '{label}' not found")
        return 0
    back_up(p)
    p.write_text(new_text)
    print(f"  + generation.py: applied '{label}' x{n}")
    return n


def patch_engine_empty_cache(root: Path) -> int:
    """
    vortex/model/engine.py
      torch.cuda.empty_cache()  ->  device-aware no-op on non-CUDA

    In vtx 1.0.8 this lives in engine.py (older Vortex versions had it in
    model.py — adjust the file path if upstream moves it back).
    """
    p = root / "model" / "engine.py"
    edits = [
        (
            r"torch\.cuda\.empty_cache\(\)",
            (
                "(torch.cuda.empty_cache() if torch.cuda.is_available()  # evo2Mac patch: empty_cache\n"
                "                else (torch.mps.empty_cache() if torch.backends.mps.is_available() else None))"
            ),
            "empty_cache",
        ),
    ]
    return edit(p, edits)


_ROTARY_FALLBACK_SHIM = '''
# --- evo2Mac patch: rotary_torch_fallback ----------------------------------
# Replace the triton-backed apply_rotary with a torch fallback on non-CUDA.
# All call sites in this module (ApplyRotaryEmb, ApplyRotaryEmbQKV_, etc.)
# continue to use the same symbol name, so they pick up the fallback when
# the input tensor isn't on CUDA. This avoids editing each call site.
try:
    from vortex.ops.embedding.rotary import apply_rotary as _triton_apply_rotary
except (ImportError, ModuleNotFoundError):
    _triton_apply_rotary = None


def _apply_rotary_torch_fallback(
    x,
    cos,
    sin,
    seqlen_offsets=0,
    cu_seqlens=None,
    max_seqlen=None,
    interleaved=False,
    inplace=False,
    conjugate=False,
):
    """Inference-time torch fallback for vortex.ops.embedding.rotary.apply_rotary.

    Supports:
      - dense (non-varlen) inputs (cu_seqlens must be None)
      - int seqlen_offsets (per-batch tensor offsets not supported)
      - inplace=True (we copy_ back into x)

    Not supported (asserts):
      - cu_seqlens / varlen
      - conjugate (backward pass; we're inference-only on Mac)
      - tensor-valued seqlen_offsets
    """
    assert cu_seqlens is None, "varlen rotary not supported in torch fallback"
    assert not conjugate, "conjugate (backward) not supported in torch fallback"
    if isinstance(seqlen_offsets, torch.Tensor):
        raise NotImplementedError("tensor seqlen_offsets not supported in torch fallback")

    seqlen = x.shape[-3]  # (..., seqlen, nheads, headdim)
    cos_slice = cos[seqlen_offsets : seqlen_offsets + seqlen]
    sin_slice = sin[seqlen_offsets : seqlen_offsets + seqlen]
    out = apply_rotary_emb_torch(x, cos_slice, sin_slice, interleaved=interleaved)
    if inplace:
        x.copy_(out)
        return x
    return out


def _apply_rotary_dispatch(x, *args, **kwargs):
    if _triton_apply_rotary is not None and x.is_cuda:
        return _triton_apply_rotary(x, *args, **kwargs)
    return _apply_rotary_torch_fallback(x, *args, **kwargs)


# Rebind the module-level symbol so every call site in this file routes
# through the dispatcher.
apply_rotary = _apply_rotary_dispatch
# --- end evo2Mac patch -----------------------------------------------------
'''


def patch_rotary_qkv_force_view_path(root: Path) -> int:
    """
    vortex/model/rotary.py ApplyRotaryEmbQKV_.forward has two branches:

      1. fast path: reshape qk = qkv[:, :, :2].reshape(..., -1, D)
                    and apply rotary in-place to qk.
      2. slow path: q, k = qkv[:, :, 0], qkv[:, :, 1]
                    apply rotary in-place to q and k separately.

    On Mac the reshape in branch 1 can return a NEW tensor (not a view)
    because qkv[:, :, :2] is non-contiguous in memory. Our torch fallback's
    in-place copy_ then writes to that new tensor and the rotated values
    are never reflected back into qkv -> attention sees un-rotated Q/K ->
    near-uniform predictions.

    Force the slow (view) path whenever we're not on CUDA. q and k via
    integer indexing are genuine views into qkv, so in-place copy_ writes
    back correctly.
    """
    p = root / "model" / "rotary.py"
    edits = [
        (
            r"if cos_k is None and sin_k is None and qkv\.is_contiguous\(\):",
            (
                "if cos_k is None and sin_k is None and qkv.is_contiguous() "
                "and qkv.is_cuda:  # evo2Mac patch: qkv_view_path"
            ),
            "qkv_view_path",
        ),
    ]
    return edit(p, edits)


def patch_rotary_torch_fallback(root: Path) -> int:
    """
    vortex/model/rotary.py imports a triton-based `apply_rotary` from
    vortex/ops/embedding/rotary.py. Triton has no maintained macOS arm64
    wheel, so on Mac we need a torch fallback.

    Strategy: keep the triton import optional, and inject a shim *after*
    `apply_rotary_emb_torch` is defined that rebinds `apply_rotary` to a
    dispatcher (triton on CUDA, torch otherwise). Every call site in the
    file uses the bare name `apply_rotary(...)`, so no other edits needed.
    """
    p = root / "model" / "rotary.py"
    if not p.exists():
        print(f"  ! missing: {p}")
        return 0
    text = p.read_text()
    label = "rotary_torch_fallback"
    mark = f"{PATCH_MARK}: {label}"
    if mark in text:
        print(f"  - rotary.py: '{label}' already patched")
        return 0

    # 1. Make the original triton-backed import optional (so module load works
    #    on Mac even when vortex.ops.embedding.rotary blows up on `import triton`).
    text, n_import = re.subn(
        r"^from\s+vortex\.ops\.embedding\.rotary\s+import\s+apply_rotary\s*$",
        (
            "try:  # evo2Mac patch: rotary_torch_fallback\n"
            "    from vortex.ops.embedding.rotary import apply_rotary\n"
            "except (ImportError, ModuleNotFoundError):\n"
            "    apply_rotary = None"
        ),
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n_import == 0:
        print(f"  ! rotary.py: triton import line not found")
        return 0

    # 2. Insert the dispatcher shim right after `apply_rotary_emb_torch` is
    #    defined (so the shim can reference it).
    marker_end = "def rotate_half"  # apply_rotary_emb_torch comes before this and ends before rotate_half
    # Use the END of apply_rotary_emb_torch (matched by next blank line + class declaration)
    # Inject before the first `class ApplyRotaryEmb(`.
    pivot = "class ApplyRotaryEmb(torch.autograd.Function):"
    if pivot not in text:
        print(f"  ! rotary.py: anchor '{pivot}' not found")
        return 0
    text = text.replace(pivot, _ROTARY_FALLBACK_SHIM.lstrip() + "\n\n" + pivot, 1)

    back_up(p)
    p.write_text(text)
    print(f"  + rotary.py: applied '{label}'")
    return 1


def patch_attn_interface_optional(root: Path) -> int:
    """
    vortex/ops/attn_interface.py eagerly imports flash_attn_2_cuda (a CUDA-only
    C extension). On Mac that import fails, which makes the entire
    `vortex.ops` namespace fail to load — even though we never actually call
    the local_flash_attn_* helpers (we use SDPA via use_flash_attn=False).

    Patch: replace `import flash_attn_2_cuda as flash_attn_gpu` with a
    try/except that sets flash_attn_gpu = None when the module is missing.
    All call sites would then crash at call time — but they're never reached
    on Mac because the config disables flash attention.
    """
    p = root / "ops" / "attn_interface.py"
    edits = [
        (
            r"^import\s+flash_attn_2_cuda\s+as\s+flash_attn_gpu\s*$",
            (
                "try:  # evo2Mac patch: optional_flash_attn\n"
                "    import flash_attn_2_cuda as flash_attn_gpu\n"
                "except ImportError:\n"
                "    flash_attn_gpu = None"
            ),
            "optional_flash_attn",
        ),
    ]
    return edit(p, edits)


def patch_cuda_device_contexts(root: Path) -> int:
    """
    vortex uses `with torch.cuda.device(<device>):` in several places. On a
    non-CUDA system this raises (`ValueError: Expected a cuda device` or
    `RuntimeError: PyTorch was compiled without CUDA support`).

    Replace each `with torch.cuda.device(X):` with a guard that falls back
    to `contextlib.nullcontext()` when CUDA isn't available.

    Targets confirmed in vtx 1.0.8:
      vortex/model/model.py    (~3 sites)
      vortex/model/utils.py    (~2 sites)
    """
    targets = [
        root / "model" / "model.py",
        root / "model" / "utils.py",
    ]
    total = 0
    pattern = re.compile(r"with\s+torch\.cuda\.device\(\s*([^)]+?)\s*\)\s*:")
    label_mark = f"{PATCH_MARK}: cuda_device_ctx"

    for p in targets:
        if not p.exists():
            print(f"  ! missing: {p}")
            continue
        text = p.read_text()
        if label_mark in text:
            print(f"  - {p.name}: 'cuda_device_ctx' already patched")
            continue

        # Ensure `import contextlib` is present.
        if "import contextlib" not in text:
            # Insert after the first `import torch` (vortex files all import torch early).
            text, _ = re.subn(
                r"(import torch[^\n]*\n)",
                r"\1import contextlib  # evo2Mac patch: contextlib\n",
                text,
                count=1,
            )

        new_text, n = pattern.subn(
            (
                r"with (torch.cuda.device(\1) if torch.cuda.is_available() "
                f"else contextlib.nullcontext()):  {label_mark}"
            ),
            text,
        )
        if n == 0:
            print(f"  ! {p.name}: no `with torch.cuda.device(...)` matches")
            continue
        back_up(p)
        p.write_text(new_text)
        print(f"  + {p.name}: applied 'cuda_device_ctx' x{n}")
        total += n
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true", help="restore .bak files")
    args = ap.parse_args()

    root = vortex_root()
    print(f"vortex root: {root}")

    if args.restore:
        restore(root)
        return 0

    total = 0
    total += patch_engine(root)
    total += patch_generation(root)
    total += patch_engine_empty_cache(root)
    total += patch_cuda_device_contexts(root)
    total += patch_attn_interface_optional(root)
    total += patch_rotary_torch_fallback(root)
    total += patch_rotary_qkv_force_view_path(root)
    print(f"\napplied {total} edit(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
