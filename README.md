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

## Plan — 4 steps

1. **Baseline** — load stock gpt-oss-20b on TPU, benchmark latency / throughput.
2. **Fusion** — apply the offline fusion + runtime patch (skip `input_layernorm`,
   combined QKV matmul), verify numerical parity, benchmark the delta vs. Step 1.
3. **Inference engine** — load the stock model under vLLM-TPU, benchmark.
4. **Fusion + engine** — fused model under vLLM-TPU, benchmark.

Current focus: **Steps 1–2** (the HuggingFace path). vLLM integration comes later.

## Layout

```
core/weight_transform.py        # fusion math: transform_gpt_oss_layer()
backends/tpu/patch_gpt_oss.py   # runtime XLA monkey-patch (Step 2 algorithm)
scripts/fuse_gpt_oss.py         # offline fuse + save + push-to-hub
scripts/install_deps.sh         # TPU VM dependency install
PLAN.md                         # detailed plan & rationale
```

## Quick start (on a TPU v6e VM)

```bash
bash scripts/install_deps.sh
huggingface-cli login
python scripts/fuse_gpt_oss.py --model-id openai/gpt-oss-20b \
    --output-dir gpt-oss-20b-fused-bf16
```

See [`PLAN.md`](PLAN.md) for the full rationale (including why the TPU speedup is
expected to be modest — XLA already fuses much of what a custom kernel would on GPU).
