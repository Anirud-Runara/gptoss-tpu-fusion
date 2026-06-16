#!/usr/bin/env python3
"""
Memory-light, layer-by-layer offline RMSNorm->QKV fusion for gpt-oss.

Streams safetensors shards one at a time and folds input_layernorm.weight (γ)
into q/k/v_proj.weight with pure tensor ops, so peak RAM ≈ one shard instead of
the whole model. Much quicker than instantiating the full HF model (this is the
approach used for the Qwen3-480B checkpoint).

    python3 scripts/fuse_gpt_oss_sharded.py \
        --model-dir /path/to/gpt-oss-20b \
        --output-dir gpt-oss-20b-fused-bf16

Notes
  * Pure BF16 fusion: γ scales the *input* features of q/k/v (the columns of W),
    so W_new = W * γ. Biases (if any) are unchanged — γ only affects the matmul
    input, and bias is added after. input_layernorm.weight is reset to ones.
  * The MoE expert tensors (MXFP4 blocks + scales) are copied through VERBATIM —
    the fusion never touches them, and they dequantize to BF16 at load exactly as
    in the stock model. Only the attention q/k/v (already BF16) are modified.
  * Everything non-.safetensors (config, tokenizer, index) is copied across so the
    output directory is a complete, loadable checkpoint.
"""

import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

QKV_RE = re.compile(r"\.layers\.(\d+)\.self_attn\.(q|k|v)_proj\.weight$")
GAMMA_RE = re.compile(r"\.layers\.(\d+)\.input_layernorm\.weight$")


def main():
    ap = argparse.ArgumentParser(description="Streaming gpt-oss RMSNorm->QKV fusion")
    ap.add_argument("--model-dir", required=True, help="local dir of the source checkpoint")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    idx = os.path.join(args.model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    # Pass 1: collect all gammas (tiny — one [h] vector per layer).
    gammas = {}
    for s in shards:
        with safe_open(os.path.join(args.model_dir, s), framework="pt") as f:
            for name in f.keys():
                m = GAMMA_RE.search(name)
                if m:
                    gammas[int(m.group(1))] = f.get_tensor(name)
    print(f"Found {len(gammas)} input_layernorm gammas")

    # Pass 2: rewrite each shard, folding γ into q/k/v and resetting γ to ones.
    folded = 0
    for s in shards:
        out = {}
        with safe_open(os.path.join(args.model_dir, s), framework="pt") as f:
            meta = f.metadata() or {}
            for name in f.keys():
                t = f.get_tensor(name)
                qkv = QKV_RE.search(name)
                if qkv:
                    layer = int(qkv.group(1))
                    if layer not in gammas:
                        raise KeyError(f"No gamma for layer {layer} ({name})")
                    t = (t * gammas[layer].to(t.dtype)).contiguous()  # fold γ into columns
                    folded += 1
                elif GAMMA_RE.search(name):
                    t = torch.ones_like(t)                            # reset γ = 1
                out[name] = t
        meta.setdefault("format", "pt")
        save_file(out, os.path.join(args.output_dir, s), metadata=meta)
        print(f"  wrote {s}  ({len(out)} tensors)")

    # Copy everything else (config, tokenizer, index json, ...).
    for fn in os.listdir(args.model_dir):
        src = os.path.join(args.model_dir, fn)
        if os.path.isfile(src) and not fn.endswith(".safetensors"):
            shutil.copy2(src, os.path.join(args.output_dir, fn))

    print(f"Done. Folded {folded} q/k/v projections -> {args.output_dir}")


if __name__ == "__main__":
    main()
