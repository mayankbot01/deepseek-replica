# model/__init__.py
"""DeepSeek-Replica model package."""

from .config import DeepSeekConfig, get_config
from .attention import MultiHeadLatentAttention, RotaryEmbedding
from .moe import MoELayer, DenseMLP, MoEGate, ExpertMLP
from .transformer import DeepSeekForCausalLM, DeepSeekModel, DeepSeekDecoderLayer, RMSNorm

__all__ = [
    "DeepSeekConfig",
    "get_config",
    "MultiHeadLatentAttention",
    "RotaryEmbedding",
    "MoELayer",
    "DenseMLP",
    "MoEGate",
    "ExpertMLP",
    "DeepSeekForCausalLM",
    "DeepSeekModel",
    "DeepSeekDecoderLayer",
    "RMSNorm",
]
