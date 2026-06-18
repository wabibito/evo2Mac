# Running FP8-Trained Genome Models on Apple Silicon: A Faithful Port of Evo 2 to MPS, with Bit-Exact FP8 Emulation and a Diagnosed Limit

**Author:** wabibito
**Affiliation:** independent
**Date:** June 2026
**Code:** [Evo2MPS](https://github.com/wabibito/Evo2MPS) · [FP8-MPS](https://github.com/wabibito/FP8-MPS) (FP8-on-MPS library)

---

## Abstract

Evo 2 is a family of genome foundation models (StripedHyena 2 architecture)
released by the Arc Institute. The 1B, 20B, and 40B checkpoints are trained with
8-bit floating point (FP8, e4m3) via NVIDIA's Transformer Engine (TE) and are
documented to require an NVIDIA Hopper GPU for numerically correct inference;
only the 7B-8k checkpoints run in bf16 on commodity hardware. We present
**Evo2MPS**, a port of Evo 2 to Apple Silicon (the PyTorch Metal/MPS backend),
and study how far the FP8 checkpoints can be recovered on hardware with no FP8
support. We make four contributions. (1) A **bit-exact** emulation of e4m3
quantization in pure PyTorch tensor ops that runs on MPS, where PyTorch's native
`float8_e4m3fn` cast is unsupported; it matches the native cast on 100% of
100,000 in-range values. (2) A method to recover Transformer Engine's per-tensor
forward scales directly from a checkpoint's `_extra_state` blobs and replay TE's
forward GEMM, which restores `evo2_1b_base` from near-random (32.6% next-token
accuracy) to **74.5%** against an H100 reference of 79.6% — a +41.9 pp recovery,
closing the gap to ~5 pp. (3) A rigorous **negative result**: the same method
does *not* recover the 20B (it stays at ~25% vs a 91.7% reference), and we
diagnose why through three independent tests — a per-layer CPU-vs-MPS diff (the
20B computes identically on both, so it is not an MPS bug), a per-layer FP8-math
validation against native CPU FP8 (our emulation is correct to ~2×10⁻³), and a
direct measurement of error accumulation and activation outliers across depth.
The 20B's failure is the documented FP8 train/inference-mismatch failure mode of
large outlier-heavy models, not a port defect. (4) A general, framework-agnostic
**FP8-on-MPS library** extracted from this work, validated end-to-end on a real
post-training-quantized FP8 model (Qwen3-0.6B-FP8) running on MPS.

---

## 1. Introduction

Evo 2 [Brixi et al., Arc Institute, 2025] models DNA across all domains of life.
The released checkpoints span 1B–40B parameters. Per the official documentation,
the 1B/20B/40B models *"require FP8 via Transformer Engine for numerical accuracy
and an NVIDIA Hopper GPU,"* while the 7B-8k checkpoints run in bf16 anywhere.
Transformer Engine is CUDA-only and its FP8 GEMMs target Hopper/Ada tensor cores,
so on Apple Silicon — increasingly common for local ML — the FP8 checkpoints
either fail to load or, with the upstream bf16 fallback, produce near-random
output.

This paper asks a concrete question: **on hardware with no FP8 support, how much
of an FP8-trained model can be recovered, and where is the hard limit?** We
answer it for Evo 2 on Apple Silicon's MPS backend, and we are deliberate about
reporting both what worked (the 1B) and what did not (the 20B/40B), with enough
diagnosis to know *why*.

All numbers below were measured on an **Apple M2 Max, 64 GB unified memory,
macOS 14, PyTorch 2.12, Python 3.11**, against the reference loss/accuracy that
upstream bakes into `evo2/test/test_evo2.py` (measured on H100 + FP8 +
flash-attn). We use upstream's own four bundled prompts (`prompts.csv`).

---

## 2. Background

### 2.1 StripedHyena 2 and Evo 2

Evo 2 uses the StripedHyena 2 block: a Hyena (implicit long-convolution) mixer
interleaved with sparse attention, an input QKV projection, an output projection,
and a gated (SwiGLU) MLP. Inference runs through the `vortex` runtime.

### 2.2 FP8 and Transformer Engine

FP8 e4m3 has 4 exponent and 3 mantissa bits, a maximum normal of 448, and no
infinities. TE trains with **delayed per-tensor scaling**: for each GEMM it keeps
an amax history and computes a scale `s = 448 / amax`, so a tensor is cast as
`round_e4m3(x · s)` and dequantized by `1/s`. TE's `DelayedScaling` recipe
(amax history length 16, `Format.HYBRID`: e4m3 forward, e5m2 backward) is part of
the *trained function* — the weights adapt to this quantization.

### 2.3 The Mac/MPS gap

PyTorch's MPS backend has **no native `float8` dtype**: `x.to(torch.float8_e4m3fn)`
works on CPU and CUDA but raises `TypeError` on MPS (PyTorch issue #132624, open
as of 2026). There is therefore no native path to run FP8 weights on an Apple GPU.

---

## 3. Method

### 3.1 Bit-exact e4m3 emulation on MPS

We emulate e4m3 rounding with pure tensor ops that MPS supports (`log2`, `round`,
`exp2`, `clamp`, `where`):

```
e    = clamp(floor(log2(|x|)), min=-6)        # binade exponent
step = exp2(e - 3)                            # 3 mantissa bits
q    = clamp(round(|x| / step) * step, max=448)
q    = where(|x| < 2^(-9)/2, 0, q)            # flush sub-subnormals
return sign(x) * q
```

This saturates above 448 (rather than producing NaN, which is what an FP8 GEMM
path expects after its pre-scale clamp). **Validation:** across 100,000 values
`~ N(0, 3²)`, the emulation is *identical* to `torch.float8_e4m3fn` on every
in-range value (100% bitwise match). This is the foundational primitive: it lets
FP8 quantization run on the Apple GPU at all.

### 3.2 Recovering TE scales and replaying the forward GEMM

For TE-format checkpoints (bf16 weights + `_extra_state`), `vortex` *strips* the
`_extra_state` blobs when TE is absent, discarding the scales. We therefore read
them from the raw `.pt` before the model is built. Each blob decodes to a dict
whose `scale_fwd` is `[act_scale, weight_scale, unused]`; we verified
`448 / weight_scale == |W|.max()` to four decimals on every projection, confirming
the slot semantics. `Fp8EmulatedLinear` then replays TE's forward:

```
y = ( round_e4m3(x · act_scale) @ round_e4m3(W · weight_scale)ᵀ ) / (act_scale · weight_scale) + b
```

A model-walking pass swaps each FP8 linear for an `Fp8EmulatedLinear` carrying
that layer's scales. It is opt-in via `EVO2MPS_FP8_EMULATION` and default-on for
`evo2_1b_base`.

### 3.3 Reproducibility harness

We measure next-token cross-entropy and accuracy on upstream's four bundled
prompts and compare to the H100 references, reporting bf16-fallback vs emulated
vs reference at the aggregate (upstream publishes no per-layer or per-prompt
reference). A second axis — greedy-generation nucleotide identity — provides an
independent check. Both are runnable scripts (`validate_fp8_emulation.py`,
`test_generation.py`, `compare_all.py`).

---

## 4. Experiments and Results

### 4.1 The 7B checkpoints reproduce H100 in bf16 (sanity)

| Model | mode | acc | H100 ref | Δacc |
|---|---|---|---|---|
| evo2_7b_base | bf16 (full 8K) | 85.98% | 85.92% | **+0.06 pp** |
| evo2_7b | bf16 (full 8K) | 86.41% | 86.35% | **+0.06 pp** |

The bf16-native checkpoints match — and marginally beat — their H100 references,
confirming the port's device handling, rotary, Hyena FFT, and unembed are correct.

### 4.2 The 1B recovers with FP8 emulation (primary positive result)

`evo2_1b_base`, 4 prompts, full 8K context (forward pass):

| mode | loss | acc | vs H100 (79.556%) |
|---|---|---|---|
| bf16 native (no FP8) | 1.3643 | 32.63% | −46.93 pp |
| **e4m3 emulated** | 0.6105 | **74.51%** | **−5.05 pp** |

A second, independent axis, greedy-generation identity (H100 reference 68.0%),
corroborates the result: the bf16 fallback scores approximately 30% and the
emulated model approximately 70%. The residual of about 5 pp is consistent with
the parts of the H100 path we do not replicate (flash-attn and the full
Transformer Engine recipe).

### 4.3 The 20B does *not* recover (primary negative result)

`evo2_20b`, 4 prompts, 2048-truncated (memory):

| mode | acc | vs H100 (91.666%) |
|---|---|---|
| bf16 native | 24.22% | −67.45 pp |
| e4m3 emulated (117 layers) | ~25% | ~−66.5 pp |

Extending emulation from the 21 input projections to all 117 FP8 linears (MLPs,
out-projection, attention QKV) **did not help.** This motivated a root-cause
investigation rather than further tuning.

### 4.4 Diagnosis: three tests localize the cause

**Test A — per-layer CPU vs MPS (is it an MPS bug?).** We run the full 20B on CPU
and on MPS in separate processes (one 40 GB copy at a time, no OOM), capture each
block's output, and diff. Result: CPU and MPS **agree** — smooth bf16 drift from
0 to ~4% over 24 blocks, no single layer above 5%. **The 20B is not an MPS bug;
it computes the same on CPU.**

**Test B — per-layer FP8 math vs native CPU FP8 (is our emulation correct?).** We
diff `Fp8EmulatedLinear` against `torch.float8_e4m3fn` on CPU (the reference FP8
arithmetic, identical to an H100's) for sampled layers. Worst relative error
**2.2×10⁻³** (20B), **1.5×10⁻³** (1B). **Our FP8 emulation is numerically correct.**

**Test C — error accumulation and outliers.** We measure the FP8-vs-bf16
perturbation of each block's output, and the activation outlier ratio:

| block | FP8-vs-bf16 rel. change | outlier ratio (max/mean) |
|---|---|---|
| 0 | 6.3% | 14 |
| 3 | 7.6% | 6962 |
| 9 | 14.8% | 3922 |
| 12 | 15.8% | 3206 |
| 18 | 17.5% | 515 |

The perturbation **grows with depth** and the mid-stack activations carry
**extreme outliers (3000–7000× the mean)**, concentrated in the SwiGLU MLPs.

### 4.5 Interpretation

By elimination (Tests A, B) the 20B's failure is neither an MPS bug nor an
emulation error. Test C identifies the mechanism, which matches the published
FP8 train/inference-mismatch literature [LMSYS 2025; NVIDIA FP8 primer; TWEO,
CVPR 2026]: large transformers develop extreme activation outliers (especially in
SwiGLU), FP8 quantization error accumulates layer-by-layer, and *"as model size
increases the train–inference inconsistency becomes more severe."* The 20B was
trained with FP8 in the loop across ~120 layers; its learned function depends on
FP8's specific per-tensor treatment of those outliers. A subtlety from our data:
e4m3 ≈ bf16 for the 20B's *weight* magnitudes (so weight quantization barely
changes anything), but the model needs the true FP8 forward over its
*outlier-heavy activations* across a deep stack — which a higher-precision
emulation cannot reproduce once errors compound. The 1B escapes this: FP8 lives
on ~21 well-behaved projection layers, so projection emulation suffices.

**Conclusion:** the 20B/40B require true FP8 hardware, exactly as Arc states — now
shown by measurement, not assumed.

### 4.6 Generalization: a real PTQ FP8 model on MPS

The emulation generalizes beyond Evo 2. Standard post-training-quantized FP8
checkpoints (Nemotron, DeepSeek-V3, Qwen3-FP8) store *pre-quantized* e4m3 weights
plus a `weight_scale_inv` (per-tensor or per-block) — a *dequantize* path rather
than TE's *re-quantize* path. We implement both in the standalone **FP8-MPS**
library and run a real `Qwen3-0.6B-FP8` SwiGLU MLP (actual `F8_E4M3` weights) end
to end on MPS, matching the native-CPU-FP8 reference to **4.7×10⁻³** — compute
PyTorch/MPS cannot do natively.

---

## 5. Engineering Log: Methods Attempted

This section records each approach evaluated, including those that did not
succeed, for reproducibility and to document the full design space explored.

- **bf16 fallback (baseline).** Loads the FP8 models but yields near-random output
  for the 1B/20B/40B. Correct for the 7B. Outcome: effective for the 7B only.
- **Projection-only e4m3 emulation.** Recovered the 1B from 32.6% to 74.5%
  accuracy. Outcome: effective.
- **Defaulting emulation on for the 7B.** Measured as a no-op (within +/-0.05 pp),
  since the 7B is bf16-native; it is therefore deliberately excluded. Outcome:
  correctly scoped out.
- **Full-layer emulation for the 20B (117 linears).** Required generalizing the
  scale extractor to read all `_extra_state` blobs and adding a return-convention
  flag (the MLPs are plain `nn.Linear` modules returning a bare tensor rather than
  Transformer Engine's `(out, bias)` tuple, which initially raised a `TypeError`).
  The pass ran correctly but did not improve 20B accuracy. Outcome: ineffective;
  it produced the diagnosis in Section 4.5.
- **fp32 pre-scaling.** Computing `x * act_scale` in fp32 rather than bf16 improves
  agreement with generic native FP8 (and is correct for the FP8-MPS library), but
  it regressed the 1B from 74.5% to 39%: Evo 2's stored scales are tuned to the
  bf16-scaling path that `vortex` uses at inference. Outcome: reverted in Evo2MPS,
  retained in FP8-MPS. The discrepancy illustrates that a more numerically generic
  formulation is not necessarily better for a specific model.
- **Activation-clamp hypothesis.** We hypothesized that unclamped activation
  outliers were the cause and considered clamping them more aggressively.
  Measurement excluded this: the stored activation scales are well calibrated, so
  the maximum scaled activations land near 445 (just below the e4m3 maximum of
  448) and essentially no values are clamped. The emulation already handles
  outliers as Transformer Engine intends; the residual is intrinsic precision
  loss. Outcome: ineffective.
- **Current (just-in-time) scaling.** Transformer Engine supports both delayed
  scaling (a stored amax history) and current scaling (the scale computed from the
  actual tensor amax at runtime). Having used the stored delayed scales, we re-ran
  the 20B computing `act_scale = 448 / |x|.amax()` per forward pass. Accuracy moved
  from 21.65% to 21.51%, i.e. no meaningful change. This confirms that the scaling
  recipe is not the limiting factor: because e4m3 is numerically close to bf16 for
  the 20B's value ranges, no scaling variant has appreciable precision to recover.
  Outcome: ineffective.
- **Transformer Engine on CPU.** We evaluated whether Transformer Engine, which is
  numerically correct, could run off-GPU. It cannot: it requires CUDA 12.1+ and
  device compute capability 9.x, and its FP8 GEMMs fail or fall back to higher
  precision without Hopper, Ada, or Blackwell tensor cores. There is no CPU or MPS
  path to its FP8 arithmetic. Outcome: not available.
- **CPU-versus-MPS and FP8-math differential tests.** The two diagnostics that
  localized the cause (Section 4.4). Outcome: effective as diagnostics.
- **Survey of related ports.** An independent port (`hakyimlab/evo2-mac`) reaches
  the same limit, documenting the Hopper requirement and not recovering the 1B. A
  survey of all 505 forks plus a code search (Section 5b) found no fork that solves
  FP8 inference on non-Hopper hardware. Outcome: independent confirmation.

**Summary of the 20B attempts.** Every software avenue was exhausted: the bf16
fallback; projection-only and all-117-layer emulation; fp32, bf16, and current
scaling; activation clamping; and the device dimension (CPU and MPS agree). None
move the 20B off approximately 25% accuracy. The cause is none of these; rather,
the model's learned function depends on the precise FP8 forward computation it was
trained with, whose effect on the 20B's outlier-heavy activations compounds across
roughly 120 layers and cannot be reproduced in higher precision (Section 4.5).
This is a property of the checkpoint, not of the port.

---

## 5b. Fork Survey: Prior Art

To check whether any solution already existed, we crawled all **505 forks** of
`arcinstitute/evo2` (via the GitHub API), filtered to those that diverge from the
upstream baseline, and inspected every Mac/MPS/CPU/FP8-relevant candidate:

| Fork | Change | Solves FP8 on non-Hopper? |
|---|---|---|
| `gtaghon/evo2-mps` ("MPS-compatible") | vortex-level MPS patches; README still requires FP8 / Compute Capability 8.9+ for 1B/20B/40B; no FP8 handling in `models.py` | No |
| `YDXCPU/evo2` (ahead 7) | sets `use_fp8_input_projections: False` on the **7B** configs only + test tweaks | No |
| `hakyimlab/evo2-mac` | upstream README + bf16 fallback; does not rescue even the 1B | No |
| `hiixryo/evo2` (ahead 1) | a scoring script + build artifacts | No |
| ~490 others | identical push timestamp/size to baseline — untouched clones | No |

A GitHub-wide code search for `float8_e4m3 StripedHyena` and
`quantize_e4m3 vortex mps` returned **zero results**. To our knowledge, no prior
work implements FP8 emulation for this architecture. The positive result (1B
recovery) and the diagnosed negative result (20B) are, as far as the public fork
network shows, the first of their kind.

---

## 6. Limitations

- The H100 reference is an aggregate over four prompts; upstream publishes no
  per-prompt or per-layer reference, so per-layer correctness is established
  against CPU-native FP8, not against H100 directly.
- The 20B forward is memory-constrained on 64 GB (2048-context); the 40B
  (~80 GB) does not load on 64 GB at all.
- This is **emulation, not hardware FP8**: on M1–M4 there is no speedup; the
  contribution is correctness. On the M5 (native GPU FP8), the quantizer is the
  seam to swap for a real FP8 matmul.
- The 1B's residual ~5 pp to H100 is not closed; it likely requires replicating
  flash-attn and the full TE recipe.

---

## 6b. What Would Actually Run the 20B/40B

For completeness — having shown no software path on this Mac works — the
genuinely viable routes, each requiring different hardware or a changed model:

1. **An FP8-capable NVIDIA GPU** (compute capability ≥ 8.9 — Ada/Hopper/Blackwell,
   e.g. an RTX 4090, or a rented H100). This runs the *real* Transformer Engine
   FP8 path the checkpoints expect; it is the only way to get the published
   accuracy. NVIDIA also hosts Evo 2 40B via an API.
2. **An Apple M5 (or later) Mac.** The M5 GPU has native FP8 in its Neural
   Accelerators. Once MLX / PyTorch-MPS expose an FP8 GEMM, a true (not emulated)
   FP8 forward becomes possible on a Mac; the FP8-MPS library is structured so its
   quantizer is the seam to swap for that kernel.
3. **Produce a bf16-native 20B.** Fine-tuning / re-exporting the 20B without FP8
   input projections (via BioNemo or Savanna on GPU) would yield a checkpoint that
   runs accurately in bf16 anywhere — at the cost of a GPU training run.
4. **Use the 7B.** The bf16-native 7B-8k checkpoints already match H100 on this
   Mac (§4.1) and are the right local workhorse today; the 1B is usable with
   emulation. For most local-inference needs this is the pragmatic answer.

The takeaway: the 20B limit is a hardware/format boundary, not an engineering gap
to close in software on M1–M4 Apple Silicon.

---

## 7. Conclusion

We ported Evo 2 to Apple Silicon and showed that an FP8-trained model can be
substantially recovered on hardware without FP8 by faithfully emulating
Transformer Engine's per-tensor e4m3 GEMM — recovering the 1B to within ~5 pp of
its H100 reference. We also showed, by measurement rather than assumption, that
the 20B/40B cannot be recovered this way: their outlier-heavy activations and
deep FP8 stacks make them depend on true FP8 hardware, the documented
train/inference-mismatch limit. The reusable core is released as **FP8-MPS**, a
general FP8-on-Apple-Silicon library validated on a real PTQ FP8 model.

---

## References

- Brixi, G. et al. *Evo 2: genome modeling and design across all domains of life.*
  Arc Institute, 2025. https://github.com/ArcInstitute/evo2
- NVIDIA. *Using FP8 with Transformer Engine.* TE documentation.
- NVIDIA. *Floating-Point 8: An Introduction to Efficient, Lower-Precision AI
  Training.* developer.nvidia.com.
- NVIDIA. *Per-Tensor and Per-Block Scaling Strategies for Effective FP8 Training.*
- LMSYS. *Unified FP8: Moving Beyond Mixed Precision for Stable and Accelerated
  MoE RL.* 2025.
- Liang et al. *TWEO: Transformers Without Extreme Outliers Enables FP8 Training
  and Quantization.* CVPR 2026.
- NVIDIA. *Transformer Engine — Installation* (CUDA 12.1+ / compute capability 9.x
  requirement). docs.nvidia.com.
- PyTorch issue #132624: *Add float8 dtypes for the MPS backend.*

---

*Reproducibility:* all results regenerate via the scripts in `scripts/`
(`validate_fp8_emulation.py`, `test_generation.py`, `compare_all.py`,
`diff_cpu_mps.py`) on an Apple-Silicon Mac with the `Evo2MPS` conda environment.
See the [FP8-MPS methods note](https://github.com/wabibito/FP8-MPS/blob/main/docs/METHODS.md)
for the general FP8-on-MPS quantization details.
