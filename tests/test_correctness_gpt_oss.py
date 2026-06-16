"""
Correctness tests for the gpt-oss RMSNorm+QKV fusion (TPU/XLA path).

Adapted from the Qwen3 repo's test_correctness_qwen3.py, with the CUDA kernel
units replaced by the pure-PyTorch FusedRMSNormCombinedLinearXLA module (so the
unit tests run anywhere — CPU, GPU, or TPU).

Levels:
  1. Unit  — FusedRMSNormCombinedLinearXLA vs plain RMSNorm->Linear (FP32 & BF16,
             2D & 3D). Validates the fusion MATH. Runs on CPU, no model download.
  2. Small integration — build a TINY random gpt-oss, patch a copy, compare logits
             and greedy tokens. Validates the PATCH PLUMBING (sinks, cache, kwargs,
             layer/attention forward rewrite) on CPU in seconds.
  3. Real integration (opt-in: --real) — load a full gpt-oss checkpoint, patch a
             copy, compare. Heavy; run on the TPU/host box.

Run:
    python3 -m tests.test_correctness_gpt_oss            # unit + small integration
    python3 -m tests.test_correctness_gpt_oss --real --model-id openai/gpt-oss-20b
"""

import argparse
import copy
import os
import sys

import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.weight_transform import compute_fused_weights_rmsnorm_combined
from backends.tpu.patch_gpt_oss import (
    FusedRMSNormCombinedLinearXLA,
    patch_gpt_oss_model,
)

# Representative gpt-oss-like attention dims (head_dim=64, GQA). Exact values do
# not affect correctness — the fusion math is dimension-agnostic — they just make
# the shapes realistic. (label, h, q_dim, k_dim, v_dim)
GPT_OSS_DIMS = [
    ("repr h=2880 (64q/8kv x64)", 2880, 4096, 512, 512),
    ("small h=512",               512,  512,  128, 128),
]


def _build_fused(rms_norm, linears):
    W, b, splits, _h, eps = compute_fused_weights_rmsnorm_combined(rms_norm, linears)
    return FusedRMSNormCombinedLinearXLA(W, b, splits, eps)


# ───────────────────────── 1. Unit tests ──────────────────────────

def test_unit_fp32():
    print("=" * 64)
    print("UNIT: FusedRMSNormCombinedLinearXLA vs RMSNorm->Linear (FP32)")
    print("=" * 64)
    torch.manual_seed(0)
    tol = 1e-4
    for label, h, q, k, v in GPT_OSS_DIMS:
        rms = nn.RMSNorm(h, eps=1e-6)
        nn.init.normal_(rms.weight, mean=1.0, std=0.1)
        linears = [nn.Linear(h, od, bias=False) for od in (q, k, v)]
        for lin in linears:
            nn.init.normal_(lin.weight, std=0.02)
        fused = _build_fused(rms, linears)
        for shape in [(1, h), (32, h), (2, 16, h)]:
            x = torch.randn(*shape)
            with torch.no_grad():
                ref = [lin(rms(x)) for lin in linears]
                got = fused(x)
            md = max((r - g).abs().max().item() for r, g in zip(ref, got))
            status = "PASS" if md < tol else "FAIL"
            print(f"  [{status}] {label} {tuple(shape)}: max_diff={md:.2e}")
            assert md < tol, f"FP32 unit failed {label} {shape}: {md}"
    print("  FP32 unit tests passed.\n")


def test_unit_bf16():
    print("=" * 64)
    print("UNIT: FusedRMSNormCombinedLinearXLA vs RMSNorm->Linear (BF16)")
    print("=" * 64)
    torch.manual_seed(0)
    tol = 0.5  # BF16 noise floor, matches the reference repo's threshold
    for label, h, q, k, v in GPT_OSS_DIMS:
        rms = nn.RMSNorm(h, eps=1e-6).bfloat16()
        nn.init.normal_(rms.weight, mean=1.0, std=0.1)
        linears = [nn.Linear(h, od, bias=False).bfloat16() for od in (q, k, v)]
        for lin in linears:
            nn.init.normal_(lin.weight, std=0.02)
        # Compute fused weights in FP32 then cast to BF16 (numerical stability).
        rms_f = copy.deepcopy(rms).float()
        lin_f = [copy.deepcopy(l).float() for l in linears]
        W, b, splits, _h, eps = compute_fused_weights_rmsnorm_combined(rms_f, lin_f)
        fused = FusedRMSNormCombinedLinearXLA(W.bfloat16(), b.bfloat16(), splits, eps)
        for shape in [(1, h), (32, h)]:
            x = torch.randn(*shape, dtype=torch.bfloat16)
            with torch.no_grad():
                ref = [lin(rms(x)) for lin in linears]
                got = fused(x)
            md = max((r.float() - g.float()).abs().max().item() for r, g in zip(ref, got))
            status = "PASS" if md < tol else "FAIL"
            print(f"  [{status}] {label} {tuple(shape)}: max_diff={md:.2e}")
            assert md < tol, f"BF16 unit failed {label} {shape}: {md}"
    print("  BF16 unit tests passed.\n")


# ──────────────────── 2. Small integration (CPU) ───────────────────

def _tiny_gpt_oss():
    """Build a tiny random gpt-oss model on CPU, or return None if unavailable."""
    try:
        from transformers import GptOssConfig, GptOssForCausalLM
    except Exception as e:
        print(f"  SKIP: transformers/gpt-oss unavailable ({e})")
        return None
    try:
        cfg = GptOssConfig(
            vocab_size=512,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=32,
            num_local_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=128,
            sliding_window=32,
        )
        torch.manual_seed(0)
        model = GptOssForCausalLM(cfg).eval().to(torch.float32)
        return model
    except Exception as e:
        print(f"  SKIP: could not construct tiny GptOssConfig/Model ({e})")
        return None


def test_small_integration():
    print("=" * 64)
    print("INTEGRATION (tiny random gpt-oss, CPU): logits + greedy tokens")
    print("=" * 64)
    model = _tiny_gpt_oss()
    if model is None:
        return
    fused = copy.deepcopy(model)
    patch_gpt_oss_model(fused, device=torch.device("cpu"))

    ids = torch.randint(0, model.config.vocab_size, (1, 16))

    with torch.no_grad():
        lo = model(input_ids=ids).logits
        lf = fused(input_ids=ids).logits
    max_diff = (lo.float() - lf.float()).abs().max().item()
    mean_diff = (lo.float() - lf.float()).abs().mean().item()
    # FP32 model: matmul reassociation only -> very tight.
    status = "PASS" if max_diff < 1e-3 else "FAIL"
    print(f"  [{status}] logits: max={max_diff:.2e} mean={mean_diff:.2e}")
    assert max_diff < 1e-3, f"tiny integration logits diverged: {max_diff}"

    with torch.no_grad():
        go = model.generate(ids, max_new_tokens=12, do_sample=False, use_cache=True)
        gf = fused.generate(ids, max_new_tokens=12, do_sample=False, use_cache=True)
    match = torch.equal(go, gf)
    print(f"  [{'PASS' if match else 'FAIL'}] greedy tokens identical: {match}")
    assert match, "tiny integration greedy tokens differ"
    print("  Small integration passed.\n")


# ───────────────────── 3. Real integration (opt-in) ─────────────────

def test_real_integration(model_id):
    print("=" * 64)
    print(f"INTEGRATION (real: {model_id})")
    print("=" * 64)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  SKIP: transformers not installed")
        return
    load_kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    try:
        from transformers import Mxfp4Config
        load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    except Exception:
        pass
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).eval()
    except Exception as e:
        print(f"  SKIP: cannot load {model_id} ({e})")
        return
    device = next(model.parameters()).device
    fused = copy.deepcopy(model)
    patch_gpt_oss_model(fused, device=device)

    for text in ["def fibonacci(n):", "The capital of France is"]:
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            lo = model(**ids).logits
            lf = fused(**ids).logits
        md = (lo.float() - lf.float()).abs().max().item()
        status = "PASS" if md < 2.0 else "FAIL"  # BF16 large-model tolerance
        print(f"  [{status}] \"{text[:40]}\": max_logit_diff={md:.2e}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="also run the heavy real-model test")
    ap.add_argument("--model-id", default="openai/gpt-oss-20b")
    args = ap.parse_args()

    test_unit_fp32()
    test_unit_bf16()
    test_small_integration()
    if args.real:
        test_real_integration(args.model_id)

    print("=" * 64)
    print("DONE")
    print("=" * 64)
