# gpt-oss TPU LayerNorm Fusion — Implementation Plan

## Goal

Apply **RMSNorm gamma absorption** ("layernorm fusion") to OpenAI's **gpt-oss** models,
run them on **TPU** via HuggingFace, and **benchmark the inference speedup** — replicating
the technique we previously validated on Qwen3 / GPU.

## Decisions

| Decision | Choice |
|---|---|
| Inference framework | **PyTorch/XLA + HuggingFace Transformers** (reuse existing HF flow + monkey-patch pattern) |
| Hardware | **TPU v6e (Trillium)** |
| First target | **gpt-oss-20b** (fast iteration), then scale the same pipeline to **gpt-oss-120b** |

---

## Background: what we did on GPU (Qwen3)

1. Fuse weights to absorb the RMSNorm scale (γ) into the following linear layer.
2. Quantize the fused weights to fit on fewer GPUs.
3. Monkey-patch HF inference to call custom **CUDA** RMSNorm kernels (BF16).

---

## Two concepts that must not be confused

Almost every design decision below follows from this distinction:

- **Operator fusion** — the compiler glues adjacent ops (normalize → scale → …) into a single
  kernel to avoid memory round-trips. **XLA does this automatically on TPU.**
- **Gamma absorption** — we *mathematically* fold the learned γ parameter into a different weight
  matrix (`W' = W · diag(γ)`), so γ no longer exists as a separate op. This rewrites the model's
  weights. **XLA does NOT do this for us** — it remains our job, exactly as on GPU.

### What this means for expected gains

- **The norm is a small slice of total work.** Forward-pass time is dominated by the big matmuls
  (attention projections and the MoE expert FFNs). RMSNorm is ~1–3% of FLOPs, so by Amdahl's law
  the *ceiling* on a norm-only optimization is small. Expect **single-digit %** end-to-end.
- **Decode benefits most.** At batch=1 autoregressive decode (memory-bandwidth-bound), removing the
  γ multiply also removes an activation round-trip to HBM, so the gain is proportionally larger there.
- **A custom TPU kernel likely will NOT beat XLA.** On GPU, much of our measured win came from
  replacing slow PyTorch-*eager* norm with a hand-written CUDA kernel. On TPU that lever disappears
  because XLA's baseline norm is *already* fused. So the remaining, honestly-measurable benefit on
  TPU is the **pure gamma-absorption effect**, and the custom-kernel step becomes an *investigation*
  ("can a Pallas kernel beat XLA?") rather than an assumed speedup.

| GPU (eager baseline) | TPU (XLA baseline) |
|---|---|
| Win = combine 3 Q/K/V matmuls → 1 **+** eliminate the norm HBM round-trip (custom kernel); γ absorption is the *enabler*, not the source | XLA already fuses elementwise ops and avoids round-trips in the baseline, so the only manual lever left is the combined-QKV matmul + removing the (already-cheap) γ op → **expected small** |

> The fused weights by themselves speed up nothing (the GPU repo says so explicitly). The speedup
> comes from the kernel/graph restructuring they enable — and XLA already performs most of that
> restructuring automatically. Quantifying the residual is the experiment.

---

## How the fusion maps onto gpt-oss

gpt-oss layer: `input_layernorm → attention (GQA + sinks) → post_attention_layernorm → MoE (router + experts)`

**Only the attention QKV site is fused** — identical to the proven Qwen3 implementation:

- **input_layernorm γ** → absorb into `q_proj`, `k_proj`, `v_proj`, **concatenated into one combined matmul**.
- **post_attention_layernorm → MoE is NOT fused.** The router sits between the norm and the experts,
  so there is no single linear layer to absorb γ into. (Same reasoning as Qwen3.)
- After absorption, the fused weights satisfy `W_combined = concat(q,k,v) · diag(γ)` and the
  `input_layernorm` op is skipped at runtime (it lives inside the combined matmul).

The win being replicated: the non-fused path writes `norm(x)` to HBM and reads it back for three
separate Q/K/V matmuls; the fused path is one combined matmul + one normalize, eliminating the
round-trip. **On TPU, XLA already does much of this automatically — see "expected gains" above.**

**MXFP4 note:** gpt-oss ships natively in MXFP4, but MXFP4 compute is Triton/GPU-only — there is no
TPU path. We dequantize the **q/k/v weights** to **BF16** for fusion (mirroring the repo's existing
NVFP4-dequant helper), and run the MoE experts in BF16 on TPU. This simplifies (or removes) the
separate quantization stage from the GPU pipeline.

> gpt-oss specifics to confirm during implementation: it uses **attention sinks** and
> **alternating full / sliding-window** attention, and (unlike Qwen3) likely has **no per-head
> q_norm/k_norm** — so the patched attention forward must reproduce gpt-oss's pipeline, not Qwen3's.

---

## Implementation & benchmarking — 4 steps

Each step is a checkpoint with its own benchmark, so we can attribute every change to a measurable delta.

### Step 1 — Baseline: download, load on TPU, benchmark
- Download stock gpt-oss-20b; load onto TPU v6e via PyTorch/XLA + HuggingFace (MXFP4 → BF16).
- Run a fixed benchmark workload; capture **latency, throughput, tokens/sec** (prefill and decode separately).
- *Purpose:* validate the environment and establish the reference numbers. **γ is still present.**

### Step 2 — Custom algorithm (QKV fusion + monkey patch), benchmark
- Apply the offline weight transform: absorb `input_layernorm` γ into a **combined Q/K/V matmul** (attention only).
- Monkey-patch the gpt-oss decoder layer to **skip `input_layernorm`** and call the combined matmul;
  let XLA fuse the rest (no custom kernel — plain `F.linear` on the combined weight).
- Verify **numerical parity**: fused-model logits match the Step-1 baseline within tolerance.
- Re-run the same benchmark; the delta vs. Step 1 is the residual gain over XLA's automatic fusion.
- *Optional sub-investigation:* a hand-written **Pallas** kernel — only if profiling shows the norm
  is a bottleneck, framed as "can we beat XLA's auto-fusion?" (often no).

### Step 3 — Inference engine, benchmark
- Load the **stock** model under an optimized serving engine (vLLM-TPU) with continuous batching / paged attention.
- Re-run the benchmark; this measures the **serving-throughput axis**, which is orthogonal to our fusion.
- *Purpose:* establish how much the engine alone buys us, independent of the algorithm.

### Step 4 — Custom algorithm + inference engine, benchmark
- Load the **gamma-absorbed** model under the same inference engine.
- Re-run the benchmark; this tests whether our fusion **still helps inside an already-optimized server**.
- *Purpose:* the realistic production configuration — fusion benefit on top of engine optimizations.

### After Step 4 — Scale to gpt-oss-120b
- Apply the identical pipeline to 120b with **sharding** (PyTorch/XLA GSPMD) across a larger v6e slice.

---

## Repo extension plan

Grow the existing Qwen3/GPU repo rather than forking it:

- **Model-agnostic core** — fusion transforms in a per-architecture registry (`qwen3`, `gpt_oss`).
- **Backend adapters** — `gpu` (existing CUDA path) and `tpu` (new PyTorch/XLA path).
- **Quantization** — make optional / pluggable; the TPU path targets BF16 (MXFP4 dequant).
- **Benchmark harness** — keep the runner & metrics; add a TPU backend and the gpt-oss model entry.

---

## Risks / open questions

- **Sharding for 120b** — confirm v6e slice size and GSPMD mesh; not a blocker for 20b (fits a v6e-8 host).
- **MoE performance on HF + XLA** — the MoE block, not the norm, is the likely throughput bottleneck.
- **Numerical parity** — confirm fused logits match baseline, especially after MXFP4 → BF16 dequant + re-fuse.
- **Modest speedup is the expected outcome** — gamma absorption removes one elementwise op + its memory
  traffic; the scientific contribution is *quantifying* this on TPU given XLA already fuses, not a large headline number.

---

## Success criteria

- gpt-oss-20b runs fused on TPU with logits matching baseline within tolerance.
- A reproducible 4-step benchmark isolating: baseline → fusion → engine → fusion+engine.
- The pipeline generalizes to gpt-oss-120b.
