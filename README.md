# gptoss-fusion

**RMSNorm + QKV fusion for OpenAI's gpt-oss — integrating a custom CUDA kernel into vLLM and measuring the speedup.**

>  **Work in progress.**

## Goal

Absorb the attention `input_layernorm` scale (γ) into a combined Q/K/V matmul for
**gpt-oss**, plug a hand-tuned **V3 CUDA kernel** (fused RMSNorm + combined-QKV)
into **vLLM** in place of its native `RMSNorm` + `QKVParallelLinear`, and
**benchmark V3-fused vLLM vs. stock vLLM** on GPU. (Only the attention QKV site is
fused — the MoE block is left untouched, since the router sits between the norm
and the experts.)

> Earlier exploration targeted gpt-oss on TPU (HuggingFace + PyTorch/XLA); that
> code still lives under `backends/tpu/`. The current focus is the GPU + vLLM
> integration under `backends/cuda/`.

## Results

See [`RESULTS.md`](RESULTS.md). Headline: on vLLM 0.23.0 / Blackwell, the V3 fused
kernel is **~4% slower** than vLLM's native `fused_add_rms_norm` + `QKVParallelLinear`
— vLLM already fuses the residual-add that V3 must do separately, and the norm is too
small a fraction to overcome it. Outputs are numerically identical. Raw numbers in
[`benchmarks/results/vllm_bench.csv`](benchmarks/results/vllm_bench.csv).

## What This Repo Does

This repo benchmarks a custom V3 CUDA kernel against vLLM's native gpt-oss
layernorm/QKV path.

The benchmark compares:

- `stock`: vLLM's native `input_layernorm` + `QKVParallelLinear`
- `v3`: a monkey-patched vLLM path that routes `input_layernorm + QKV` through
  the custom V3 fused RMSNorm + combined-QKV CUDA kernel
- `v1`: an older fused-kernel variant kept for comparison

Only the attention input layernorm and QKV projection are changed. The attention
backend, output projection, post-attention layernorm, and MoE block remain
vLLM-native.

## Layout

```
benchmarks/bench_vllm_v3.py          # stock vs V3 benchmark entry point
backends/vllm/patch_vllm_gpt_oss.py  # vLLM monkey-patch for gpt-oss blocks
backends/cuda/fused_forward.py       # V1/V3 fused RMSNorm + combined-QKV modules
csrc/denominator_kernel.cu           # custom CUDA RMSNorm denominator kernels
csrc/denominator.cpp                 # PyTorch CUDA extension bindings
benchmarks/results/vllm_bench.csv    # accumulated benchmark results
RESULTS.md                           # measured results and interpretation
```

Older TPU/HuggingFace exploration still exists under `backends/tpu/` and related
scripts, but it is not the current benchmark path.

## Running the Benchmark

Install dependencies, then run each mode in a separate process because the vLLM
patch is global:

```bash
python benchmarks/bench_vllm_v3.py --mode stock --model openai/gpt-oss-20b
python benchmarks/bench_vllm_v3.py --mode v3 --model openai/gpt-oss-20b
```

For an explicit CUDA extension build, you can set the target architecture:

```bash
GPTOSS_CUDA_ARCH=sm_80 pip install -e .
```

The benchmark script also JIT-builds the extension on first import if needed.

## Current Result

See [`RESULTS.md`](RESULTS.md). On vLLM 0.23.0 / Blackwell, the custom V3 kernel is
about 4% slower than vLLM's native path. The outputs match, but vLLM already fuses
residual-add + RMSNorm, while the V3 path has to perform the residual add separately.
