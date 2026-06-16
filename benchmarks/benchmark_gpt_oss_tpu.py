#!/usr/bin/env python3
"""
Benchmark gpt-oss on TPU (PyTorch/XLA): Step 1 (baseline) vs Step 2 (fused).

Measures, for each mode:
  * prefill latency  — one forward over [batch, prompt_len]
  * decode throughput — end-to-end generate() of gen_len new tokens (tokens/sec)

The only difference between the two modes is the runtime RMSNorm+QKV patch
(backends/tpu/patch_gpt_oss.py); everything else (weights, attention backend,
dtype, shapes) is identical, so the measured delta is attributable to the fusion.

Because XLA compiles per shape, the first call of every shape pays a one-time
compilation cost. We run --warmup iterations (untimed) before timing.

Usage (on a TPU v6e VM):
    # baseline only
    python benchmarks/benchmark_gpt_oss_tpu.py --model-id openai/gpt-oss-20b --mode baseline
    # both, plus numerical parity check, write CSV
    python benchmarks/benchmark_gpt_oss_tpu.py \
        --model-id gpt-oss-20b-fused-bf16 --mode both --check-parity \
        --prompt-len 128 --gen-len 128 --iters 5 --warmup 2
"""

import argparse
import csv
import os
import sys
import time

# Repo root on path so `core` / `backends` import.
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BENCH_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("PJRT_DEVICE", "TPU")


def parse_args():
    p = argparse.ArgumentParser(description="gpt-oss TPU fusion benchmark (Steps 1-2)")
    p.add_argument("--model-id", default="openai/gpt-oss-20b",
                   help="HF model id or local path (stock OR offline-fused checkpoint)")
    p.add_argument("--mode", choices=["baseline", "fused", "both"], default="both")
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--gen-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--iters", type=int, default=5, help="timed iterations")
    p.add_argument("--warmup", type=int, default=2, help="untimed warmup iterations")
    p.add_argument("--check-parity", action="store_true",
                   help="compare baseline vs fused logits on one forward")
    p.add_argument("--parity-tol", type=float, default=2e-2,
                   help="max abs logit diff considered a pass (BF16 is noisy)")
    p.add_argument("--csv", default=None, help="output CSV path (default: auto under results/)")
    return p.parse_args()


def _load_model(model_id, device, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    load_kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    try:
        from transformers import Mxfp4Config
        load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).eval().to(device)
    tok = AutoTokenizer.from_pretrained(model_id)
    return model, tok


def _sync(xm):
    xm.mark_step()
    xm.wait_device_ops()


def _bench_mode(model, input_ids, args, xm, torch):
    """Return dict of prefill_ms, gen_tok_per_s for the given (already-patched-or-not) model."""
    # ---- prefill: one forward over the prompt ----
    def prefill():
        with torch.no_grad():
            return model(input_ids=input_ids, use_cache=False).logits

    for _ in range(args.warmup):
        prefill(); _sync(xm)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        prefill(); _sync(xm)
    prefill_ms = (time.perf_counter() - t0) / args.iters * 1e3

    # ---- decode: end-to-end greedy generate ----
    gen_kwargs = dict(max_new_tokens=args.gen_len, do_sample=False, use_cache=True)

    def generate():
        with torch.no_grad():
            return model.generate(input_ids, **gen_kwargs)

    for _ in range(args.warmup):
        generate(); _sync(xm)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        generate(); _sync(xm)
    gen_s = (time.perf_counter() - t0) / args.iters
    new_tokens = args.batch_size * args.gen_len
    return {"prefill_ms": prefill_ms, "gen_tok_per_s": new_tokens / gen_s,
            "gen_s": gen_s}


def main():
    args = parse_args()
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    print(f"XLA device: {device}")
    print(f"Loading {args.model_id} ...")
    model, tok = _load_model(args.model_id, device, torch)

    # Fixed-shape input (random token ids in-vocab) for stable, recompile-free timing.
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (args.batch_size, args.prompt_len), device=device)

    results = {}
    parity = None
    baseline_logits = None

    if args.mode in ("baseline", "both"):
        print("\n=== Step 1: baseline ===")
        if args.check_parity:
            with torch.no_grad():
                baseline_logits = model(input_ids=input_ids, use_cache=False).logits.float().cpu()
        results["baseline"] = _bench_mode(model, input_ids, args, xm, torch)
        print(results["baseline"])

    if args.mode in ("fused", "both"):
        print("\n=== Step 2: fused (RMSNorm+QKV patch) ===")
        from backends.tpu.patch_gpt_oss import patch_gpt_oss_model
        patch_gpt_oss_model(model, device=device)
        if args.check_parity:
            with torch.no_grad():
                fused_logits = model(input_ids=input_ids, use_cache=False).logits.float().cpu()
            if baseline_logits is not None:
                max_abs = (fused_logits - baseline_logits).abs().max().item()
                parity = {"max_abs_logit_diff": max_abs,
                          "pass": max_abs <= args.parity_tol}
                print(f"parity: {parity}")
        results["fused"] = _bench_mode(model, input_ids, args, xm, torch)
        print(results["fused"])

    if "baseline" in results and "fused" in results:
        sp = results["baseline"]["gen_s"] / results["fused"]["gen_s"]
        print(f"\nspeedup (generate, baseline/fused): {sp:.4f}x")

    # ---- write CSV ----
    csv_path = args.csv or os.path.join(
        _BENCH_DIR, "results",
        f"benchmark_{os.path.basename(args.model_id.rstrip('/'))}"
        f"_p{args.prompt_len}_g{args.gen_len}_b{args.batch_size}.csv",
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "prefill_ms", "gen_tok_per_s", "gen_s"])
        for mode, r in results.items():
            w.writerow([mode, f"{r['prefill_ms']:.3f}", f"{r['gen_tok_per_s']:.3f}",
                        f"{r['gen_s']:.4f}"])
        if parity is not None:
            w.writerow([])
            w.writerow(["parity_max_abs_logit_diff", parity["max_abs_logit_diff"]])
            w.writerow(["parity_pass", parity["pass"]])
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
