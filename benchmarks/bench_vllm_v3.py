#!/usr/bin/env python3
"""
Benchmark gpt-oss under vLLM: stock kernels vs the V3 fused RMSNorm+QKV kernel.

Runs vLLM's offline LLM API (single process, enforce_eager) so the class-level
monkey-patch takes effect and we get a fair kernel-vs-kernel comparison.

    # stock vLLM
    python benchmarks/bench_vllm_v3.py --mode stock --model openai/gpt-oss-20b
    # V3-fused vLLM
    GPTOSS_CUDA_ARCH=sm_80 python benchmarks/bench_vllm_v3.py --mode v3 --model openai/gpt-oss-20b

Run each mode in its own process (the patch is global). Compare the printed
throughput / latency. Use --check to also print a sample completion so you can
eyeball that V3 output is coherent (a correctness smoke test).
"""

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Keep the engine in-process so the class-level patch applies (TP=1 single GPU).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def parse_args():
    p = argparse.ArgumentParser(description="vLLM gpt-oss: stock vs V3 fused kernel")
    p.add_argument("--mode", choices=["stock", "v3", "v1"], default="stock")
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--check", action="store_true", help="print a sample completion")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode in ("v3", "v1"):
        from backends.vllm.patch_vllm_gpt_oss import patch_vllm_gpt_oss
        patch_vllm_gpt_oss(variant=args.mode.upper())  # MUST run before LLM()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        enforce_eager=True,            # fair kernel-vs-kernel; required by the patch
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )

    base = ("Explain, in detail, how a transformer language model processes a "
            "sequence of tokens from input embeddings to output logits. ")
    prompts = [base + f"(variant {i})" for i in range(args.num_prompts)]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Warmup (also triggers the lazy V3 build + any compilation).
    _ = llm.generate(prompts[: min(8, len(prompts))], sp, use_tqdm=False)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    elapsed = time.perf_counter() - t0

    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    print("\n==== RESULT ====")
    print(f"mode            : {args.mode}")
    print(f"prompts         : {len(prompts)}")
    print(f"output tokens   : {n_out}")
    print(f"elapsed         : {elapsed:.3f}s")
    print(f"throughput      : {n_out / elapsed:.1f} tok/s")
    print(f"latency/req     : {elapsed / len(prompts) * 1e3:.1f} ms (batched)")
    if args.check:
        print(f"\nsample output   : {outs[0].outputs[0].text[:300]!r}")


if __name__ == "__main__":
    main()
