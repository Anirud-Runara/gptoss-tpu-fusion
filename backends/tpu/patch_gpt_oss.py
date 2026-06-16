"""
Runtime monkey-patch for GPT-OSS on TPU (PyTorch/XLA) — RMSNorm + QKV fusion.


* GPT-OSS attention specifics. There is NO per-head q_norm/k_norm.
GPT-OSS uses attention sinks (self_attn.sinks, passed as `s_aux`) and
alternating full / sliding-window attention (self_attn.sliding_window).
The patched attention forward reproduces the upstream GptOssAttention
pipeline verbatim except that input_layernorm + Q/K/V become one fused call.

What is fused
-------------
  input_layernorm -> [q_proj, k_proj, v_proj]   <- FUSED (γ absorbed into a
                                                   single combined matmul)
  post_attention_layernorm -> MoE               <- NOT fused (router in between)

Two-phase model 
--------------------------------------
  1. Offline artifact (scripts/fuse_gpt_oss.py): bakes γ into q/k/v and sets the
     norm to ones. Runs through the stock forward; speeds up nothing on its own.
  2. Runtime patch (THIS file): swaps in the fused combined-QKV module so the
     benchmark measures the fusion. Requires no compiled extension.

NOTE: attribute/forward details were ported from transformers `modeling_gpt_oss`.
The cache-update call and exact kwargs should be confirmed against the installed
transformers version with tests/test_correctness_gpt_oss.py before trusting
benchmark numbers.
"""

import sys
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from core.weight_transform import transform_gpt_oss_layer


# ------------------------------------------------------------------ #
# Fused module (pure PyTorch — XLA fuses it)                         #
# ------------------------------------------------------------------ #

class FusedRMSNormCombinedLinearXLA(torch.nn.Module):
    """
    Normalize-only RMSNorm (γ already absorbed into W_combined) followed by a
    single combined Q/K/V matmul. Replaces:

        input_layernorm(x) -> q_proj / k_proj / v_proj  (norm + 3 matmuls)

    with:

        x_norm = x / rms(x)                  # NO γ here (γ lives in W_combined)
        out    = x_norm @ W_combined.T + b   # one matmul; W_combined = concat(q,k,v)·diag(γ)
        q, k, v = split(out)

    rms() is computed in float32 then cast back, matching GptOssRMSNorm so logits
    stay within BF16 tolerance of the unpatched model.
    """

    def __init__(self, W_combined: torch.Tensor, b_combined: torch.Tensor,
                 split_sizes: list[int], eps: float):
        super().__init__()
        self.register_buffer("W_combined", W_combined)
        self.register_buffer("b_combined", b_combined)
        self.split_sizes = split_sizes
        self.eps = eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_dtype = x.dtype
        xf = x.to(torch.float32)
        variance = xf.pow(2).mean(-1, keepdim=True)
        x_norm = (xf * torch.rsqrt(variance + self.eps)).to(input_dtype)

        out = F.linear(x_norm, self.W_combined, self.b_combined)
        q, k, v = torch.split(out, self.split_sizes, dim=-1)
        return q, k, v



def offline_fuse_gpt_oss_layer(layer) -> None:
    """In-place: W *= γ for q/k/v, then set input_layernorm to ones.

    Produces the storable/uploadable checkpoint. 
    """
    attn = layer.self_attn
    gamma = layer.input_layernorm.weight.data.clone()
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
        proj.weight.data.mul_(gamma)
    layer.input_layernorm.weight.data.fill_(1.0)


# ------------------------------------------------------------------ #
# Runtime patch                                                      #
# ------------------------------------------------------------------ #

def patch_gpt_oss_model(model, device=None):
    """Patch every GptOssDecoderLayer to use the fused RMSNorm+QKV path."""
    if device is None:
        device = next(model.parameters()).device
    for layer in model.model.layers:
        _patch_decoder_layer(layer, device)
    return model


def _patch_decoder_layer(layer, device):
    fused = transform_gpt_oss_layer(layer)
    W_comb, b_comb, split_sizes, _h, eps = fused["attn_qkv"]
    layer.self_attn.fused_qkv = FusedRMSNormCombinedLinearXLA(
        W_comb.to(device), b_comb.to(device), split_sizes, eps
    )
    _patch_attention_forward(layer.self_attn)
    _patch_layer_forward(layer)


def _patch_attention_forward(attn):
    """
    Replace GptOssAttention.forward to use the fused QKV (skipping the standalone
    input_layernorm), then reproduce the upstream pipeline: reshape -> RoPE ->
    attention (with sinks via s_aux + sliding_window) -> o_proj.

    Helper symbols are pulled from the attention class's own module so we track
    the installed transformers version.
    """
    _mod = sys.modules[type(attn).__module__]
    apply_rotary_pos_emb = _mod.apply_rotary_pos_emb
    eager_attention_forward = _mod.eager_attention_forward
    ALL_ATTENTION_FUNCTIONS = _mod.ALL_ATTENTION_FUNCTIONS

    def patched_forward(
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        # Fused RMSNorm + Q/K/V (input_layernorm folded into fused_qkv weights).
        q_raw, k_raw, v_raw = attn.fused_qkv(hidden_states)
        query_states = q_raw.view(hidden_shape).transpose(1, 2)
        key_states   = k_raw.view(hidden_shape).transpose(1, 2)
        value_states = v_raw.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": kwargs.get("cache_position")}
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn.layer_idx, cache_kwargs
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
            sliding_window=attn.sliding_window,
            s_aux=attn.sinks,            # attention sinks — GPT-OSS specific
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)
        return attn_output, attn_weights

    attn.forward = patched_forward


def _patch_layer_forward(layer):
    """
    Replace GptOssDecoderLayer.forward to SKIP input_layernorm (folded into
    fused_qkv) and return a bare hidden_states tensor. The MoE path
    (post_attention_layernorm + mlp) is unchanged; mlp returns (hidden, router_logits).
    """

    def patched_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        # input_layernorm SKIPPED — fused into fused_qkv weights.
        hidden_states, _ = layer.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MoE path — post_attention_layernorm runs as normal.
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        mlp_out = layer.mlp(hidden_states)
        if isinstance(mlp_out, (tuple, list)):
            mlp_out = mlp_out[0]
        hidden_states = residual + mlp_out

        return hidden_states

    layer.forward = patched_forward
