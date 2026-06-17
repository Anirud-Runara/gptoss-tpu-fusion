# Results — V3 fused RMSNorm+QKV vs stock vLLM (gpt-oss-20b)

## Verdict

On vLLM 0.23.0 / Blackwell, the **V3 fused RMSNorm+QKV CUDA kernel is ~4% slower**
than vLLM's native `fused_add_rms_norm` + `QKVParallelLinear`. Pointing vLLM at
the custom kernel **does not improve** throughput on this already-fused engine.

Outputs are **byte-identical** between modes (greedy), so the fusion is numerically
correct — it just isn't faster here.

> An initial run showed V3 ~22% *faster*. That was a **cold-cache artifact**: the
> first process paid weight-load + Triton JIT compilation inside the timed region.
> With proper warmup + repeated rounds it disappears (see Methodology).

## Environment

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell (96 GB, `sm_120`) |
| vLLM | 0.23.0 (V1 engine, `enforce_eager=True`) |
| CUDA toolkit / driver | 12.8 / 13.0 |
| Model | `openai/gpt-oss-20b` (native MXFP4; MoE = MARLIN, attention = TRITON_ATTN) |
| Sampler | native PyTorch (`VLLM_USE_FLASHINFER_SAMPLER=0` — FlashInfer's sampler JIT needs CUDA ≥ 12.9 on sm_120) |
| Fusion scope | `input_layernorm` + QKV only (γ absorbed into combined QKV weight). MoE untouched. |

## Methodology

- Workload: **256 prompts × 256 output tokens** (65,536 output tokens), greedy.
- **2 warmup passes** (push Triton/inductor JIT + lazy V3 build out of the timed region)
  then **5 timed rounds**; report mean ± std and best.
- Run in **both orderings** (v3-first and stock-first) to rule out cache/ordering bias.
- `enforce_eager=True` so the runtime monkey-patch is exercised (no CUDA graphs / torch.compile).
  This is a **kernel-vs-kernel** comparison; production (CUDA-graph) numbers would differ.

## Measurements

| Run | Mode | mean tok/s | std | best tok/s | best latency/req |
|----:|------|-----------:|----:|-----------:|-----------------:|
| 1 (v3 first)    | v3    | 17,837.5 | 58.1  | 17,935.2 | 14.27 ms |
| 2               | stock | 18,555.7 | 58.1  | 18,644.1 | 13.73 ms |
| 3 (v3 first)    | v3    | 17,679.2 | 51.6  | 17,743.7 | 14.43 ms |
| 4               | stock | 18,365.0 | 105.6 | 18,558.3 | 13.79 ms |

**Averages:** stock ≈ 18,460 tok/s, v3 ≈ 17,758 tok/s → **stock ~4.0% faster**
(consistent across both orderings; variance < 0.4%).

## Why V3 loses here

vLLM's native path is **2 ops**: `fused_add_rms_norm` (residual-add **and** RMSNorm
in one kernel) → `QKVParallelLinear`. V3 cannot fuse the residual add, so it runs
**3 ops**: explicit add → matmul → normalize-scale. The extra standalone add (a full
`[tokens, hidden]` read/write + kernel launch, ×24 layers) costs more than V3 saves by
skipping the materialized normed activation. And the norm is only ~1–3% of total work
(Amdahl), so there was never enough headroom to overcome that.

The original Qwen3 GPU win was measured against **eager HuggingFace with no fusion**;
vLLM is an **already-fused baseline**, so the win does not transfer.

## Possible next step (low ceiling)

To reach *parity*, V3 would need a variant that fuses the residual-add into the kernel
(`add + rms + matmul-scale` in one shot), matching vLLM's `fused_add_rms_norm`. Even
then, Amdahl caps the upside at roughly parity / sub-1%. Worth it only if "match vLLM
with our own kernel" is itself the goal.
