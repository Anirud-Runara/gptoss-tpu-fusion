"""
Fused RMSNorm + combined-QKV forward modules (CUDA), ported from the Qwen3 repo.

These wrap the V3 (512-thread) and V1 (256-thread) CUDA kernels. The combined
variant replaces `RMSNorm -> [q_proj, k_proj, v_proj]` with:

    raw = F.linear(x, W_combined)          # one cuBLAS matmul on RAW x; W_combined = concat(q,k,v)·diag(γ)
    rmsnorm_normalize_*(x, raw, b, h, eps) # one kernel: computes rms(x) and scales raw in-place

This skips materializing the normalized activation (the saving vs. a separate
RMSNorm kernel + matmul), and γ is pre-absorbed into W_combined. Returns the
Q/K/V slices.

V3 is the kernel we benchmark against vLLM's native RMSNorm + QKVParallelLinear.
"""

import torch
import torch.nn.functional as F

from .load_cuda import denominator_cuda


class FusedRMSNormCombinedLinearV3(torch.nn.Module):
    """V3: fused RMSNorm + combined QKV (512-thread Welford-based normalize)."""

    def __init__(self, W_combined: torch.Tensor, b_combined: torch.Tensor,
                 split_sizes: list[int], h: int, eps: float):
        super().__init__()
        self.register_buffer("W_combined", W_combined)
        self.register_buffer("b_combined", b_combined)
        self.split_sizes = split_sizes
        self.h = h
        self.eps = eps

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.size(-1))

        raw_output = F.linear(x_2d, self.W_combined)
        denominator_cuda.rmsnorm_normalize_512(
            x_2d, raw_output, self.b_combined, self.h, self.eps
        )

        parts = torch.split(raw_output, self.split_sizes, dim=-1)
        batch_shape = orig_shape[:-1]
        return [p.contiguous().reshape(*batch_shape, p.size(-1)) for p in parts]


class FusedRMSNormCombinedLinearV1(torch.nn.Module):
    """V1: fused RMSNorm + combined QKV (256-thread normalize) — for comparison."""

    def __init__(self, W_combined: torch.Tensor, b_combined: torch.Tensor,
                 split_sizes: list[int], h: int, eps: float):
        super().__init__()
        self.register_buffer("W_combined", W_combined)
        self.register_buffer("b_combined", b_combined)
        self.split_sizes = split_sizes
        self.h = h
        self.eps = eps

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.size(-1))

        raw_output = F.linear(x_2d, self.W_combined)
        denominator_cuda.rmsnorm_normalize(
            x_2d, raw_output, self.b_combined, self.h, self.eps
        )

        parts = torch.split(raw_output, self.split_sizes, dim=-1)
        batch_shape = orig_shape[:-1]
        return [p.contiguous().reshape(*batch_shape, p.size(-1)) for p in parts]


_VARIANTS = {
    "V1": FusedRMSNormCombinedLinearV1,
    "V3": FusedRMSNormCombinedLinearV3,
}


def build_fused_qkv(rms_norm, q_proj, k_proj, v_proj, variant: str = "V3", device=None):
    """Build a fused QKV module from a layer's RMSNorm + q/k/v Linears.

    Uses core.weight_transform to absorb γ into the concatenated weight, then
    wraps it in the chosen CUDA variant. `device` defaults to the weights' device.
    """
    from core.weight_transform import compute_fused_weights_rmsnorm_combined

    W, b, splits, h, eps = compute_fused_weights_rmsnorm_combined(
        rms_norm, [q_proj, k_proj, v_proj]
    )
    if device is None:
        device = W.device
    cls = _VARIANTS[variant]
    return cls(W.to(device), b.to(device), splits, h, eps)
