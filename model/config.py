# model/config.py
# DeepSeek-style model configuration

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeepSeekConfig:
    """Configuration for the DeepSeek-Replica language model."""

    # --- Vocabulary & Sequence ---
    vocab_size: int = 102400          # BPE vocabulary size
    max_seq_len: int = 4096           # Maximum sequence length
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # --- Model Dimensions ---
    hidden_size: int = 2048           # d_model
    intermediate_size: int = 10944    # FFN intermediate dim (MLP fallback)
    num_hidden_layers: int = 28       # Number of transformer blocks
    num_attention_heads: int = 16     # Number of query heads
    num_key_value_heads: int = 16     # GQA: set < num_attention_heads for grouped-query

    # --- Multi-Head Latent Attention (MLA) ---
    kv_lora_rank: int = 512           # Low-rank dim for KV compression
    q_lora_rank: int = 1536           # Low-rank dim for Q compression (0 = disabled)
    qk_rope_head_dim: int = 64        # Head dim used for RoPE on Q/K
    qk_nope_head_dim: int = 128       # Head dim NOT using RoPE
    v_head_dim: int = 128             # Value head dimension

    # --- RoPE ---
    rope_theta: float = 10000.0       # Base for rotary embeddings
    rope_scaling: Optional[dict] = None  # Extended context scaling config

    # --- Mixture of Experts (MoE) ---
    use_moe: bool = True
    num_experts: int = 64             # Total number of experts
    num_experts_per_tok: int = 6      # Top-K experts activated per token
    num_shared_experts: int = 2       # Always-on shared experts
    moe_intermediate_size: int = 1408 # Expert FFN hidden size
    expert_load_balance_coef: float = 0.01  # Auxiliary load-balancing loss weight

    # --- Norms & Init ---
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    # --- Dropout ---
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    # --- Training ---
    tie_word_embeddings: bool = False
    use_cache: bool = True

    def head_dim(self) -> int:
        """Total head dimension (nope + rope)."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0, (
            "hidden_size must be divisible by num_attention_heads"
        )
        if self.q_lora_rank == 0:
            self.q_lora_rank = None


# ---------------------------------------------------------------------------
# Pre-defined size variants (mirrors DeepSeek model family sizes)
# ---------------------------------------------------------------------------

def get_config(size: str = "small") -> DeepSeekConfig:
    """Return a pre-defined config by size name."""
    configs = {
        "tiny": DeepSeekConfig(
            hidden_size=512, num_hidden_layers=8,
            num_attention_heads=8, num_key_value_heads=8,
            intermediate_size=2048, num_experts=8,
            num_experts_per_tok=2, moe_intermediate_size=512,
            kv_lora_rank=128, q_lora_rank=256,
            qk_rope_head_dim=32, qk_nope_head_dim=32, v_head_dim=32,
        ),
        "small": DeepSeekConfig(),   # 2B-class defaults above
        "medium": DeepSeekConfig(
            hidden_size=4096, num_hidden_layers=30,
            num_attention_heads=32, num_key_value_heads=32,
            intermediate_size=14336, num_experts=64,
            num_experts_per_tok=6, moe_intermediate_size=1408,
            kv_lora_rank=512, q_lora_rank=1536,
            qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128,
        ),
    }
    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from {list(configs.keys())}")
    return configs[size]
