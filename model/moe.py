# model/moe.py
# Mixture of Experts (MoE) layer
# Based on DeepSeek-V2/V3 MoE design with auxiliary load-balancing loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .config import DeepSeekConfig


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network (single expert)
# ---------------------------------------------------------------------------

class ExpertMLP(nn.Module):
    """A single SwiGLU expert used inside MoE."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Shared Expert (always active, no routing)
# ---------------------------------------------------------------------------

class SharedExpertMLP(ExpertMLP):
    """Shared expert that processes every token unconditionally."""
    pass


# ---------------------------------------------------------------------------
# Top-K Gating Router
# ---------------------------------------------------------------------------

class MoEGate(nn.Module):
    """
    Softmax router with top-K selection.
    Returns expert indices and softmax-normalized weights.
    Computes auxiliary load-balancing loss to prevent expert collapse.
    """

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.alpha = config.expert_load_balance_coef
        self.weight = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B*T, D)
        Returns:
            topk_weights: (B*T, top_k)  -- softmax-normalised expert weights
            topk_indices: (B*T, top_k)  -- selected expert indices
            aux_loss:     scalar         -- load-balancing auxiliary loss
        """
        logits = self.weight(hidden_states)         # (B*T, E)
        scores = F.softmax(logits, dim=-1)          # (B*T, E)
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1, sorted=False)

        # Re-normalize top-k weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Auxiliary load-balancing loss (DeepSeek formulation)
        # Minimizes variance in expert load across the batch
        if self.training:
            # fraction of tokens routed to each expert
            num_tokens = hidden_states.shape[0]
            expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()  # (B*T, K, E)
            tokens_per_expert = expert_mask.sum(dim=[0, 1]) / (num_tokens * self.top_k)  # (E,)
            # mean router probability per expert
            mean_prob = scores.mean(dim=0)                                                # (E,)
            aux_loss = self.alpha * self.num_experts * (tokens_per_expert * mean_prob).sum()
        else:
            aux_loss = torch.tensor(0.0, device=hidden_states.device)

        return topk_weights, topk_indices, aux_loss


# ---------------------------------------------------------------------------
# MoE Layer
# ---------------------------------------------------------------------------

class MoELayer(nn.Module):
    """
    DeepSeek-style Mixture of Experts FFN layer.

    Architecture:
        output = sum_over_selected_experts(w_i * expert_i(x))
               + sum_over_shared_experts(shared_j(x))

    Uses a token-level dispatch (scatter/gather) for efficient batched expert calls.
    """

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.num_shared = config.num_shared_experts
        hidden = config.hidden_size
        inter = config.moe_intermediate_size

        # Routed experts
        self.experts = nn.ModuleList([
            ExpertMLP(hidden, inter) for _ in range(self.num_experts)
        ])

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            SharedExpertMLP(hidden, inter) for _ in range(self.num_shared)
        ])

        self.gate = MoEGate(config)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, T, D)
        Returns:
            output: (B, T, D)
            aux_loss: scalar
        """
        B, T, D = hidden_states.shape
        flat = hidden_states.view(B * T, D)  # (N, D) where N = B*T

        # ----- Router -----
        topk_weights, topk_indices, aux_loss = self.gate(flat)  # (N, K)

        # ----- Dispatch to routed experts -----
        # We use a simple loop for clarity; production implementations
        # use grouped GEMM / triton kernels.
        expert_output = torch.zeros_like(flat)  # (N, D)

        # Transpose for expert-centric iteration: (K, N)
        for k in range(self.top_k):
            idx = topk_indices[:, k]            # (N,) expert assignment for slot k
            weights = topk_weights[:, k]        # (N,)
            for e in range(self.num_experts):
                token_mask = (idx == e)         # bool mask (N,)
                if not token_mask.any():
                    continue
                tokens = flat[token_mask]       # (n_e, D)
                out_e = self.experts[e](tokens) # (n_e, D)
                expert_output[token_mask] += weights[token_mask].unsqueeze(-1) * out_e

        # ----- Shared experts -----
        shared_output = torch.zeros_like(flat)
        for shared_expert in self.shared_experts:
            shared_output = shared_output + shared_expert(flat)

        output = (expert_output + shared_output).view(B, T, D)
        return output, aux_loss


# ---------------------------------------------------------------------------
# Dense FFN fallback (used in first / last layers in some DeepSeek configs)
# ---------------------------------------------------------------------------

class DenseMLP(nn.Module):
    """Standard SwiGLU FFN (no routing) for non-MoE layers."""

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
