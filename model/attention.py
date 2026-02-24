# model/attention.py
# Multi-Head Latent Attention (MLA) with Rotary Position Embeddings (RoPE)
# Based on DeepSeek-V2/V3 architecture

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DeepSeekConfig


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return (
            self.cos_cached[:, :, :seq_len].to(x.dtype),
            self.sin_cached[:, :, :seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class MultiHeadLatentAttention(nn.Module):
    """
    DeepSeek MLA: low-rank KV compression + decoupled RoPE.
    Only the qk_rope_head_dim slice is rotated; the nope slice is passed as-is.
    """

    def __init__(self, config: DeepSeekConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.qk_rope_dim = config.qk_rope_head_dim
        self.qk_nope_dim = config.qk_nope_head_dim
        self.head_dim = self.qk_nope_dim + self.qk_rope_dim
        self.v_head_dim = config.v_head_dim
        self.hidden_size = config.hidden_size
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.attn_drop = config.attention_dropout
        self.softmax_scale = self.head_dim ** -0.5

        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.num_kv_heads * self.qk_rope_dim,
            bias=False,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_kv_heads * (self.qk_nope_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            dim=self.qk_rope_dim, max_seq_len=config.max_seq_len, theta=config.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        use_cache: bool = False,
    ):
        B, T, _ = hidden_states.shape

        # Q
        if self.q_lora_rank is not None:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q = self.q_proj(hidden_states)
        q = q.view(B, T, self.num_heads, self.head_dim)
        q_nope, q_rope = q[..., :self.qk_nope_dim], q[..., self.qk_nope_dim:]

        # KV
        kv_raw = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent = kv_raw[..., :self.kv_lora_rank]
        k_rope = kv_raw[..., self.kv_lora_rank:].view(B, T, self.num_kv_heads, self.qk_rope_dim)
        kv_latent = self.kv_a_layernorm(kv_latent)
        kv = self.kv_b_proj(kv_latent).view(
            B, T, self.num_kv_heads, self.qk_nope_dim + self.v_head_dim
        )
        k_nope = kv[..., :self.qk_nope_dim]
        value_states = kv[..., self.qk_nope_dim:]

        # RoPE
        past_len = past_key_value[0].shape[1] if past_key_value is not None else 0
        cos, sin = self.rotary_emb(q_rope, seq_len=T + past_len)
        cos = cos[0, 0, past_len: past_len + T].unsqueeze(0).unsqueeze(2)
        sin = sin[0, 0, past_len: past_len + T].unsqueeze(0).unsqueeze(2)
        q_rope, k_rope = apply_rotary_emb(q_rope, k_rope, cos, sin)

        # Merge nope + rope
        query_states = torch.cat([q_nope, q_rope], dim=-1).permute(0, 2, 1, 3)
        key_states = torch.cat([k_nope, k_rope], dim=-1)
        value_states = value_states

        # KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        present = (key_states, value_states) if use_cache else None

        # GQA expand
        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=2)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=2)

        key_states = key_states.permute(0, 2, 1, 3)
        value_states = value_states.permute(0, 2, 1, 3)

        attn_out = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
            scale=self.softmax_scale,
        )
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, self.num_heads * self.v_head_dim)
        attn_out = self.o_proj(attn_out)
        return attn_out, None, present
