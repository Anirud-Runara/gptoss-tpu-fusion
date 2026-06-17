"""
Monkey-patch vLLM (0.23.0) gpt-oss to route input_layernorm + QKV through the V3
fused CUDA kernel, in place of native RMSNorm + QKVParallelLinear.

Target classes (vllm.model_executor.models.gpt_oss):
  * OAIAttention.forward(hidden_states, positions)
        qkv,_ = self.qkv_proj(h); q,k,v = qkv.split(...); q,k = rotary(...);
        attn_output = self.attn(q,k,v); output,_ = self.o_proj(attn_output)
        (attention SINKS are handled inside self.attn — we do NOT touch that.)
  * TransformerBlock.forward(hidden_states, positions, residual)
        input_layernorm is a FUSED add+RMSNorm:
            hidden, residual = self.input_layernorm(hidden, residual)
        i.e. residual_out = hidden + residual, normed = rmsnorm(residual_out)*γ.

What we change
--------------
We replace `input_layernorm -> qkv_proj` with the V3 fused module:
  residual_out = hidden + residual          # explicit add (vLLM fuses this into its norm)
  q,k,v        = V3(residual_out)            # γ-free rmsnorm + combined QKV (γ folded into W)
then run the rest of attention (rotary, self.attn, o_proj) unchanged, and the
unchanged post_attention_layernorm + MoE.

The fused weight is built LAZILY on first forward from the layer's own loaded
weights: W_combined = qkv_proj.weight * input_layernorm.γ, b = qkv_proj.bias.
No pre-fused checkpoint required (mirrors the Qwen3 runtime patch).

Requirements / caveats (verify on the A100)
-------------------------------------------
  * Run vLLM with enforce_eager=True — this is a Python kernel swap, not yet
    CUDA-graph / torch.compile compatible.
  * The patch is applied at CLASS level, so it must run in the SAME process that
    builds the model. For TP=1 single-GPU, set VLLM_ENABLE_V1_MULTIPROCESSING=0
    so the model is in-process (the benchmark script does this). If the model
    runs in a separate worker, use the vLLM plugin entry point instead (see
    register() at the bottom).
  * Scope: attention QKV only. post_attention_layernorm + MoE untouched.
"""

import os
import torch

_PATCHED = False
_VERBOSE = os.environ.get("GPTOSS_FUSION_VERBOSE", "1") == "1"


def patch_vllm_gpt_oss(variant: str = "V3") -> None:
    """Patch vLLM's gpt-oss classes to use the V3 fused QKV. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    from vllm.model_executor.models import gpt_oss as M

    Attn = M.OAIAttention
    Block = M.TransformerBlock

    # ---- attention: run the post-QKV pipeline from precomputed q,k,v ----
    def forward_from_qkv(self, q, k, v, positions):
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    Attn.forward_from_qkv = forward_from_qkv

    # ---- decoder block: fold the residual add, skip γ-norm, use V3 ----
    def block_forward(self, hidden_states, positions, residual):
        # Replicate the add that vLLM's fused add+rmsnorm performs internally.
        if residual is None:
            residual = hidden_states
        else:
            residual = hidden_states + residual

        fused = getattr(self, "_v3_fused_qkv", None)
        if fused is None:
            fused = _build_fused_qkv(self, variant)
            self._v3_fused_qkv = fused

        q, k, v = fused(residual)
        hidden_states = self.attn.forward_from_qkv(q, k, v, positions)

        # Unchanged: post_attention_layernorm (fused add+norm) + MoE.
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        output = self.mlp(hidden_states)
        return output, residual

    Block.forward = block_forward
    _PATCHED = True
    if _VERBOSE:
        print(f"[gptoss-fusion] patched vLLM gpt-oss (OAIAttention/TransformerBlock) "
              f"to use {variant} fused QKV.")


def _build_fused_qkv(block, variant: str):
    """Build the V3 (or V1) fused module from a TransformerBlock's loaded weights."""
    from backends.cuda.fused_forward import _VARIANTS

    attn = block.attn
    qkv = attn.qkv_proj
    gamma = block.input_layernorm.weight.data                  # [hidden]
    W = qkv.weight.data                                        # [q+2kv, hidden] at TP=1
    if W.dim() != 2:
        raise RuntimeError(
            f"Expected a dense 2D qkv_proj.weight; got shape {tuple(W.shape)}. "
            "If qkv_proj is quantized/packed, the V3 path needs a dequantized "
            "attention projection (gpt-oss attention is normally BF16)."
        )
    b = qkv.bias.data if getattr(qkv, "bias", None) is not None else \
        torch.zeros(W.size(0), dtype=W.dtype, device=W.device)

    W_comb = (W * gamma.to(W.dtype)).contiguous()              # absorb γ into columns
    split_sizes = [attn.q_size, attn.kv_size, attn.kv_size]
    h = gamma.shape[0]
    eps = block.input_layernorm.variance_epsilon

    cls = _VARIANTS[variant]
    module = cls(W_comb.to(W.device), b.to(W.device), split_sizes, h, eps)
    if _VERBOSE:
        idx = getattr(block, "layer_idx", "?")
        print(f"[gptoss-fusion] built {variant} fused_qkv for layer {idx} "
              f"(W={tuple(W_comb.shape)}, split={split_sizes}, eps={eps})")
    return module


# ---- optional vLLM plugin entry point (robust across worker processes) ----
# Register in pyproject/setup as:
#   [project.entry-points."vllm.general_plugins"]
#   gptoss_fusion = "backends.vllm.patch_vllm_gpt_oss:register"
def register():
    """vLLM general-plugin hook — runs inside each worker process."""
    patch_vllm_gpt_oss(os.environ.get("GPTOSS_FUSION_VARIANT", "V3"))
