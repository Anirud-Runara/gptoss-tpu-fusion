"""
Pre-compute fused RMSNorm -> QKV weights for GPT-OSS.

RMSNorm followed by a Linear can be folded into a single matmul:

    Linear(RMSNorm(x)) = ((x / rms(x)) * gamma) @ W.T + b
                       =  (x / rms(x))          @ (W * gamma).T + b

so absorbing gamma into W gives a weight matrix W_new = W * gamma, and the only
runtime work left is the (gamma-free) normalization. For GPT-OSS we additionally
concatenate q/k/v into one combined matmul.

Scope: only the attention QKV site is fused. The MoE block is NOT fused — the
router sits between post_attention_layernorm and the experts, so there is no
single linear layer to absorb that norm into.
"""

import torch
import torch.nn as nn


def compute_fused_weights_rmsnorm_combined(
    rms_norm,
    linears: list[nn.Linear],
) -> tuple[torch.Tensor, torch.Tensor, list[int], int, float]:
    """
    Fuse an RMSNorm + several Linear layers that share it into one combined matmul.

    The weight matrices are concatenated along dim=0 so a single matmul replaces
    the separate projections (e.g. Q/K/V).

    Args:
        rms_norm: the shared RMSNorm module (exposes `weight` = gamma, and either
                  `eps` or `variance_epsilon`).
        linears:  list of nn.Linear sharing this norm, in order.

    Returns:
        W_combined:  [sum(out_dims), h] fused weight (gamma absorbed)
        b_combined:  [sum(out_dims)]    fused bias (zeros where a linear had none)
        split_sizes: per-linear output dims, for torch.split at runtime
        h:           hidden dimension
        eps:         RMSNorm epsilon
    """
    gamma = rms_norm.weight.data  # [h]
    h = gamma.shape[0]
    eps = rms_norm.eps if hasattr(rms_norm, "eps") else rms_norm.variance_epsilon

    W_parts, b_parts, split_sizes = [], [], []
    for linear in linears:
        W = linear.weight.data  # [out_i, h], dense BF16 for GPT-OSS attention
        b = (
            linear.bias.data
            if linear.bias is not None
            else torch.zeros(W.size(0), device=W.device, dtype=W.dtype)
        )
        W_parts.append(W * gamma.to(W.dtype))  # element-wise, no centering
        b_parts.append(b)
        split_sizes.append(W.size(0))

    W_combined = torch.cat(W_parts, dim=0)  # [sum(out_dims), h]
    b_combined = torch.cat(b_parts, dim=0)  # [sum(out_dims)]
    return W_combined, b_combined, split_sizes, h, eps


def transform_gpt_oss_layer(decoder_layer) -> dict:
    """
    Compute fused combined QKV weights for one GptOssDecoderLayer.

    Fuses:
        input_layernorm -> [self_attn.q_proj, self_attn.k_proj, self_attn.v_proj]

    Notes specific to GPT-OSS:
      - No per-head q_norm/k_norm (nothing extra to preserve after projection).
      - Attention sinks (self_attn.sinks) and alternating full / sliding-window
        attention are handled by the runtime patch, not here.
      - q/k/v are dense BF16 (only the MoE experts ship in MXFP4, and those are
        not fused), so no dequantization is needed here.

    Returns:
        {"attn_qkv": (W_combined, b_combined, split_sizes, h, eps)}
    """
    attn = decoder_layer.self_attn
    norm = decoder_layer.input_layernorm
    qkv_linears = [attn.q_proj, attn.k_proj, attn.v_proj]
    return {"attn_qkv": compute_fused_weights_rmsnorm_combined(norm, qkv_linears)}
