# model/transformer.py
# Full DeepSeek-style transformer: decoder layers + LM head
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import DeepSeekConfig
from .attention import MultiHeadLatentAttention
from .moe import MoELayer, DenseMLP


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------
class DeepSeekDecoderLayer(nn.Module):
    """
    Single transformer decoder block:
      x -> LN -> MLA -> residual
      x -> LN -> MoE (or Dense MLP) -> residual
    """

    def __init__(self, config: DeepSeekConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MultiHeadLatentAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Use dense FFN for the first layer and every non-MoE layer
        use_moe = config.use_moe and layer_idx > 0
        if use_moe:
            self.mlp = MoELayer(config)
        else:
            self.mlp = DenseMLP(config)
        self.is_moe = use_moe

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        use_cache: bool = False,
        # FIX: propagate is_causal flag down to MultiHeadLatentAttention
        is_causal: bool = True,
    ):
        # ----- Self-Attention -----
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, _, present_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            is_causal=is_causal,  # FIX: was not passed before
        )
        hidden_states = residual + attn_out

        # ----- FFN / MoE -----
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            mlp_out, aux_loss = self.mlp(hidden_states)
        else:
            mlp_out = self.mlp(hidden_states)
            aux_loss = torch.tensor(0.0, device=hidden_states.device)
        hidden_states = residual + mlp_out
        return hidden_states, aux_loss, present_kv


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------
class DeepSeekModel(nn.Module):
    """Decoder-only transformer body (without LM head)."""

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList([
            DeepSeekDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
    ):
        B, T = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)  # (B, T, D)

        # FIX: is_causal=True is now threaded through all layers.
        # When decoding with a KV cache (T==1) each layer sets sdpa_is_causal=False
        # because past_key_value is provided -- that's correct behaviour.
        past_key_values = past_key_values or [None] * len(self.layers)
        new_kv_cache: List = []
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device)

        for i, layer in enumerate(self.layers):
            hidden_states, aux_loss, present_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[i],
                use_cache=use_cache,
                is_causal=True,  # FIX: always pass is_causal=True
            )
            total_aux_loss = total_aux_loss + aux_loss
            new_kv_cache.append(present_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, new_kv_cache, total_aux_loss


class DeepSeekForCausalLM(nn.Module):
    """DeepSeek causal language model with LM head."""

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.model = DeepSeekModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)

            # FIX #6: Move weight tying AFTER _init_weights to prevent re-initialization
        # Weight tying must happen after all parameters are initialized
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ):
        hidden_states, new_kv_cache, aux_loss = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)  # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # Shift so next-token prediction: predict token t+1 from context t
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            ce_loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
            loss = ce_loss + aux_loss

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": new_kv_cache,
            "aux_loss": aux_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Simple autoregressive generation with top-p/top-k sampling."""
        eos_token_id = eos_token_id or self.config.eos_token_id
        generated = input_ids.clone()
        past_kv = None

        for _ in range(max_new_tokens):
            out = self(
                generated if past_kv is None else generated[:, -1:],
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = out["past_key_values"]
            next_logits = out["logits"][:, -1, :] / max(temperature, 1e-5)

            # Top-K filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < topk_vals[:, -1:]] = float("-inf")

            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # FIX #2: Standard nucleus sampling - keep tokens until cumsum > top_p
            remove = cumulative_probs > top_p
            remove[..., 0] = False  # Always keep at least the top-1 token                sorted_logits[remove] = float("-inf")
                next_logits.scatter_(1, sorted_idx, sorted_logits)

            probs = next_logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            if (next_token == eos_token_id).all():
                break

        return generated

    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (not only_trainable) or p.requires_grad
        )
