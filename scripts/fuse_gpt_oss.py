#!/usr/bin/env python3
"""
Offline in-place RMSNorm->QKV fusion for GPT-OSS, then save / push to HF.

What this script does:
  1. Loads GPT-OSS (default: openai/gpt-oss-20b) in BF16. GPT-OSS ships in MXFP4
     for the MoE experts; MXFP4 has no TPU compute path, so we DEQUANTIZE to BF16
     on load (the q/k/v projections we fuse are already dense BF16).
  2. For every GptOssDecoderLayer, absorbs input_layernorm.weight (γ) into
     q_proj / k_proj / v_proj (W *= γ) and resets the norm to ones.
  3. Saves the fused BF16 checkpoint, optionally pushing it to the HF Hub.

This offline transform is the *artifact*. On its own it does NOT speed anything
up (the stock forward still runs the now-identity norm) — the measurable speedup
comes from the runtime patch in backends/tpu/patch_gpt_oss.py. We upload it so
the benchmark stages (and any inference engine) can load fused weights directly.

Usage:
    python3 scripts/fuse_gpt_oss.py \\
        --model-id openai/gpt-oss-20b \\
        --output-dir gpt-oss-20b-fused-bf16 \\
        --push-to-hub <your-hf-username>/gpt-oss-20b-rmsnorm-fused

Requires a box with enough RAM/VRAM to hold gpt-oss-20b in BF16 (~40 GB).
"""

import argparse
import gc
import os
import time


def parse_args():
    p = argparse.ArgumentParser(description="Offline GPT-OSS RMSNorm->QKV fusion")
    p.add_argument("--model-id", default="openai/gpt-oss-20b",
                   help="HuggingFace model ID or local path")
    p.add_argument("--local-model-dir", default=None,
                   help="If set, load from this local directory instead of HF Hub")
    p.add_argument("--output-dir", default="gpt-oss-20b-fused-bf16",
                   help="Directory to save the fused model")
    p.add_argument("--max-shard-size", default="5GB",
                   help="Shard size passed to save_pretrained (default: 5GB)")
    p.add_argument("--push-to-hub", default=None,
                   help="If set, repo id to push the fused checkpoint to "
                        "(e.g. user/gpt-oss-20b-rmsnorm-fused)")
    p.add_argument("--private", action="store_true",
                   help="Create the pushed Hub repo as private")
    p.add_argument("--sanity-check", action="store_true",
                   help="After saving, reload and run a short greedy generation")
    return p.parse_args()


def fuse_layer(layer) -> None:
    """In-place: absorb input_layernorm γ into q/k/v_proj, then reset norm to ones.

    RMSNorm(x) @ W = ((x / rms(x)) * γ) @ W = (x / rms(x)) @ (W * γ)
    so W_new[i, :] = W[i, :] * γ  (each row of W scaled by γ).
    """
    attn = layer.self_attn
    gamma = layer.input_layernorm.weight.data  # [h]
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
        proj.weight.data.mul_(gamma)
    layer.input_layernorm.weight.data.fill_(1.0)


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.local_model_dir or args.model_id
    print(f"Loading {model_path} in BF16 (MXFP4 experts dequantized to BF16) ...")

    # GPT-OSS MXFP4 -> BF16. Newer transformers dequantize when no MXFP4 kernels
    # are present; Mxfp4Config(dequantize=True) makes that explicit when available.
    load_kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    try:
        from transformers import Mxfp4Config
        load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    except Exception:
        print("  (Mxfp4Config unavailable — relying on automatic dequantization)")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    n_layers = len(model.model.layers)
    print(f"Fusing input_layernorm -> QKV for {n_layers} layers ...")
    t1 = time.time()
    for i, layer in enumerate(model.model.layers):
        fuse_layer(layer)
        if (i + 1) % 8 == 0 or i == n_layers - 1:
            print(f"  layer {i+1:4d}/{n_layers}  ({time.time() - t1:.0f}s)")
    print(f"Fusion complete in {time.time() - t1:.1f}s")
    gc.collect()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving fused model to {args.output_dir} (shard {args.max_shard_size}) ...")
    t2 = time.time()
    model.save_pretrained(args.output_dir, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved in {time.time() - t2:.1f}s")

    if args.sanity_check:
        print("Sanity check: short greedy generation from the saved checkpoint ...")
        prompt = "def quicksort(arr):"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        print("  ", tokenizer.decode(out[0], skip_special_tokens=True))

    if args.push_to_hub:
        print(f"Pushing to HF Hub: {args.push_to_hub} (private={args.private}) ...")
        model.push_to_hub(args.push_to_hub, private=args.private)
        tokenizer.push_to_hub(args.push_to_hub, private=args.private)
        print("Push complete.")

    print(f"Done. Fused BF16 model at: {args.output_dir}")


if __name__ == "__main__":
    main()
