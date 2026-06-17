#!/usr/bin/env python3
"""
Single-layer microbenchmark for the gpt-oss RMSNorm+QKV fusion on TPU/XLA.

Mirrors the methodology of the Qwen3 GPU repo's single-layer sweep: build ONE
real gpt-oss decoder layer (actual 20B dims pulled from the HF config; random
weights — values don't affect timing) on a single chip, and time it through XLA
in two modes:

  * baseline : input_layernorm(x) -> self_attn(...)              (γ present)
  * fused    : patched self_attn(x)  (RMSNorm+QKV folded, input_layernorm skipped)

Two measurements per shape:
  * attention-only — the purest fusion signal (norm + QKV + attn + o_proj)
  * full-layer     — realistic (adds the MoE block, identical in both modes)

One layer (~1.6 GB) fits comfortably on one v6e chip, so NO sharding is needed.
XLA compiles per shape, so we run --warmup untimed iters before timing.

Usage:
    python3 benchmarks/benchmark_gpt_oss_layer.py \
        --model-id openai/gpt-oss-20b --seqs 128,512,1024,2048 --iters 20 --warmup 5
"""

import argparse
import csv
import os
import sys
import time

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BENCH_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("PJRT_DEVICE", "TPU")


def parse_args():
    p = argparse.ArgumentParser(description="gpt-oss single-layer fusion microbenchmark")
    p.add_argument("--model-id", default="openai/gpt-oss-20b",
                   help="HF model id — only the CONFIG is downloaded (dims), not weights")
    p.add_argument("--seqs", default="128,512,1024,2048",
                   help="comma-separated prefill sequence lengths to sweep")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--iters", type=int, default=20, help="timed iterations")
    p.add_argument("--warmup", type=int, default=5, help="untimed warmup iterations")
    p.add_argument("--parity-tol", type=float, default=5e-2,
                   help="max abs diff (BF16) for the baseline-vs-fused parity check")
    p.add_argument("--csv", default=None)
    return p.parse_args()


def _sync(xm):
    xm.mark_step()
    xm.wait_device_ops()


def _time(fn, xm, warmup, iters):
    for _ in range(warmup):
        fn(); _sync(xm)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(); _sync(xm)
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def main():
    args = parse_args()
    import torch
    import torch_xla.core.xla_model as xm
    try:
        from transformers import GptOssConfig, GptOssModel
    except ImportError:
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssConfig, GptOssModel
    from backends.tpu.patch_gpt_oss import _patch_decoder_layer

    device = xm.xla_device()
    print(f"XLA device: {device}")

    cfg = GptOssConfig.from_pretrained(args.model_id)
    cfg.num_hidden_layers = 1
    cfg._attn_implementation = "eager"  # eager handles sinks (s_aux); identical in both modes
    print(f"dims: hidden={cfg.hidden_size} heads={cfg.num_attention_heads} "
          f"kv={cfg.num_key_value_heads} head_dim={getattr(cfg,'head_dim','?')} "
          f"experts={getattr(cfg,'num_local_experts','?')}")

    torch.manual_seed(0)
    model = GptOssModel(cfg).eval().to(torch.bfloat16).to(device)
    layer = model.layers[0]
    rotary = model.rotary_emb
    h = cfg.hidden_size

    seqs = [int(s) for s in args.seqs.split(",") if s.strip()]
    rows = []

    for L in seqs:
        B = args.batch_size
        x = torch.randn(B, L, h, dtype=torch.bfloat16, device=device)
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pe = rotary(x, position_ids)  # (cos, sin)

        # ---- baseline closures (captured BEFORE patching) ----
        def base_attn():
            n = layer.input_layernorm(x)
            return layer.self_attn(hidden_states=n, position_embeddings=pe,
                                   attention_mask=None)[0]

        def base_layer():
            return layer(hidden_states=x, position_embeddings=pe, attention_mask=None)

        # capture reference outputs for the parity check
        with torch.no_grad():
            ref_attn = base_attn()
            ref_layer = base_layer()
        _sync(xm)

        attn_base_ms = _time(lambda: base_attn(), xm, args.warmup, args.iters)
        layer_base_ms = _time(lambda: base_layer(), xm, args.warmup, args.iters)

        # ---- patch this layer, then re-measure ----
        _patch_decoder_layer(layer, device)

        def fused_attn():
            return layer.self_attn(hidden_states=x, position_embeddings=pe,
                                   attention_mask=None)[0]

        def fused_layer():
            return layer(hidden_states=x, position_embeddings=pe, attention_mask=None)

        with torch.no_grad():
            got_attn = fused_attn()
            got_layer = fused_layer()
        _sync(xm)

        attn_diff = (ref_attn.float() - got_attn.float()).abs().max().item()
        layer_diff = (ref_layer.float() - got_layer.float()).abs().max().item()

        attn_fused_ms = _time(lambda: fused_attn(), xm, args.warmup, args.iters)
        layer_fused_ms = _time(lambda: fused_layer(), xm, args.warmup, args.iters)

        # undo the patch so the next shape starts from a clean baseline layer
        _restore_layer(layer)

        attn_sp = attn_base_ms / attn_fused_ms
        layer_sp = layer_base_ms / layer_fused_ms
        ok = "PASS" if max(attn_diff, layer_diff) <= args.parity_tol else "FAIL"
        print(f"\nseq={L} batch={B}  [parity {ok}: attn={attn_diff:.2e} layer={layer_diff:.2e}]")
        print(f"  attention-only: base={attn_base_ms:.3f}ms  fused={attn_fused_ms:.3f}ms  "
              f"speedup={attn_sp:.4f}x")
        print(f"  full-layer    : base={layer_base_ms:.3f}ms  fused={layer_fused_ms:.3f}ms  "
              f"speedup={layer_sp:.4f}x")

        rows.append(dict(seq=L, batch=B,
                         attn_base_ms=attn_base_ms, attn_fused_ms=attn_fused_ms, attn_speedup=attn_sp,
                         layer_base_ms=layer_base_ms, layer_fused_ms=layer_fused_ms, layer_speedup=layer_sp,
                         attn_parity=attn_diff, layer_parity=layer_diff))

    csv_path = args.csv or os.path.join(
        _BENCH_DIR, "results", f"layerbench_{os.path.basename(args.model_id.rstrip('/'))}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nWrote {csv_path}")


def _restore_layer(layer):
    """Remove instance-level patched forwards so the layer reverts to its class methods."""
    for obj in (layer, layer.self_attn):
        if "forward" in obj.__dict__:
            del obj.__dict__["forward"]
    if hasattr(layer.self_attn, "fused_qkv"):
        del layer.self_attn.fused_qkv


if __name__ == "__main__":
    main()
