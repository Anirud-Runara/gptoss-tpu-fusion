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

# FlashInfer's sampler JIT needs CUDA >= 12.9 to build for Blackwell (sm_120);
# on CUDA 12.8 it raises "SM 12.x requires CUDA >= 12.9". Fall back to vLLM's
# native PyTorch sampler. (Our V3 kernel builds fine on 12.8.)
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def parse_args():
    p = argparse.ArgumentParser(description="vLLM gpt-oss: stock vs V3 fused kernel")
    p.add_argument("--mode", choices=["stock", "v3", "v1"], default="stock")
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--num-prompts", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--repeat", type=int, default=5, help="timed rounds (after warmup)")
    p.add_argument("--warmup-rounds", type=int, default=2, help="full untimed passes")
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--check", action="store_true", help="print a sample completion")
    p.add_argument("--csv", default=None,
                   help="append results here (default: benchmarks/results/vllm_bench.csv)")
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

    import statistics

    # Warmup: full passes so Triton/inductor JIT + the lazy V3 build land OUTSIDE
    # the timed region (the first run otherwise eats JIT-compilation spikes).
    for _ in range(args.warmup_rounds):
        _ = llm.generate(prompts, sp, use_tqdm=False)

    times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        times.append(time.perf_counter() - t0)

    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    thr = [n_out / t for t in times]
    mean, best = statistics.mean(thr), max(thr)
    std = statistics.pstdev(thr) if len(thr) > 1 else 0.0
    print("\n==== RESULT ====")
    print(f"mode            : {args.mode}")
    print(f"prompts x toks  : {len(prompts)} x {args.max_tokens}  ({n_out} out tokens)")
    print(f"rounds          : {args.repeat}  per-round s = {['%.3f' % t for t in times]}")
    print(f"throughput      : mean {mean:.1f} ± {std:.1f} tok/s   |   best {best:.1f} tok/s")
    print(f"latency/req     : best {min(times) / len(prompts) * 1e3:.2f} ms (batched)")
    if args.check:
        print(f"\nsample output   : {outs[0].outputs[0].text[:300]!r}")

    # Persist a row so runs accumulate (the result is otherwise only on stdout).
    import csv
    import datetime
    csv_path = args.csv or os.path.join(_REPO_ROOT, "benchmarks", "results", "vllm_bench.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "mode", "model", "num_prompts", "max_tokens", "repeat",
                        "mean_tok_s", "std_tok_s", "best_tok_s", "best_latency_ms", "per_round_s"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), args.mode, args.model,
                    len(prompts), args.max_tokens, args.repeat,
                    f"{mean:.1f}", f"{std:.1f}", f"{best:.1f}",
                    f"{min(times) / len(prompts) * 1e3:.2f}",
                    ";".join(f"{t:.3f}" for t in times)])
    print(f"appended results -> {csv_path}")


if __name__ == "__main__":
    main()
