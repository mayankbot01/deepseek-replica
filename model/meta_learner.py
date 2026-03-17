# model/meta_learner.py
# MAML-style meta-learning + adaptive expert routing evolution
# Enables the model to rapidly adapt to new distributions with few gradient steps

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Dict, Optional, Tuple
from .config import DeepSeekConfig


# ---------------------------------------------------------------------------
# MAML Inner-Loop Optimizer
# ---------------------------------------------------------------------------
class MAMLInnerLoop:
    """
    Model-Agnostic Meta-Learning (MAML) inner loop.
    Computes fast-adapted parameters with K gradient steps on a support set.
    The outer (meta) loop then optimizes the initialization across tasks.

    Usage:
        maml = MAMLInnerLoop(model, inner_lr=1e-3, num_inner_steps=5)
        adapted_params = maml.adapt(support_loss_fn)
        query_loss = maml.query_loss(adapted_params, query_loss_fn)
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 1e-3,
        num_inner_steps: int = 5,
        first_order: bool = False,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        # first_order=True uses FOMAML (cheaper, slightly less accurate)
        self.first_order = first_order

    def adapt(self, support_loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run K inner-gradient steps and return adapted parameter dict.
        Args:
            support_loss: scalar loss on the support (task-specific) batch
        Returns:
            adapted_params: dict mapping param name -> adapted tensor
        """
        # Clone current parameters as fast weights
        fast_weights = {n: p.clone() for n, p in self.model.named_parameters()}

        # Iterative inner updates
        for _ in range(self.num_inner_steps):
            grads = torch.autograd.grad(
                support_loss,
                fast_weights.values(),
                create_graph=not self.first_order,
                allow_unused=True,
            )
            fast_weights = {
                n: w - self.inner_lr * (g if g is not None else torch.zeros_like(w))
                for (n, w), g in zip(fast_weights.items(), grads)
            }
        return fast_weights

    def query_loss(
        self,
        adapted_params: Dict[str, torch.Tensor],
        query_batch: Tuple[torch.Tensor, torch.Tensor],
        loss_fn,
    ) -> torch.Tensor:
        """
        Evaluate loss on the query set using adapted (fast) weights.
        This loss is used to update the meta-initialization via the outer loop.
        """
        input_ids, labels = query_batch
        # Temporarily inject fast weights into model
        original_params = {n: p.data.clone() for n, p in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            param.data = adapted_params[name].data

        output = self.model(input_ids=input_ids, labels=labels)
        loss = output["loss"]

        # Restore original weights
        for name, param in self.model.named_parameters():
            param.data = original_params[name]

        return loss


# ---------------------------------------------------------------------------
# Expert Routing Evolution (AdaptiveRouter)
# ---------------------------------------------------------------------------
class AdaptiveRouter(nn.Module):
    """
    Self-adaptive MoE router that tracks per-expert performance statistics
    and adjusts routing temperature dynamically.

    Mechanism:
      - Tracks exponential moving average (EMA) of each expert's output norm
        as a proxy for expert "activity" and quality.
      - Adjusts a per-expert temperature bias: experts that consistently
        produce higher-quality representations get a routing bonus.
      - This lets underused experts recover and overloaded experts share load
        BEYOND the static auxiliary loss penalty.
    """

    def __init__(self, config: DeepSeekConfig, ema_decay: float = 0.99):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.ema_decay = ema_decay

        # Learned router weights (replaces/supplements MoEGate)
        self.router_weight = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # Per-expert adaptive bias (learnable, updated via EMA of quality)
        self.register_buffer(
            "expert_quality_ema", torch.ones(config.num_experts)
        )
        self.adaptive_bias = nn.Parameter(torch.zeros(config.num_experts))

    @torch.no_grad()
    def update_quality_ema(self, expert_outputs: torch.Tensor, expert_indices: torch.Tensor):
        """
        Update EMA quality estimate for each expert based on the L2 norm
        of its output tokens (higher norm = more confident/expressive output).

        Args:
            expert_outputs: (N, D) tensor of expert outputs
            expert_indices: (N,) tensor of which expert produced each output
        """
        for e in range(self.num_experts):
            mask = expert_indices == e
            if mask.any():
                quality = expert_outputs[mask].norm(dim=-1).mean()
                self.expert_quality_ema[e] = (
                    self.ema_decay * self.expert_quality_ema[e]
                    + (1 - self.ema_decay) * quality
                )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adaptive routing with quality-adjusted logits.
        Returns (topk_weights, topk_indices, aux_loss)
        """
        logits = self.router_weight(hidden_states)  # (N, E)
        # Add adaptive bias: experts with higher quality get routing bonus
        quality_bias = F.normalize(self.expert_quality_ema.unsqueeze(0), dim=-1)
        logits = logits + self.adaptive_bias.unsqueeze(0) + quality_bias

        scores = F.softmax(logits, dim=-1)  # (N, E)
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1, sorted=False)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Load balancing auxiliary loss
        N = hidden_states.shape[0]
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=[0, 1]) / (N * self.top_k)
        mean_prob = scores.mean(dim=0)
        aux_loss = self.num_experts * (tokens_per_expert * mean_prob).sum()

        return topk_weights, topk_indices, aux_loss


# ---------------------------------------------------------------------------
# Curriculum Learning Controller
# ---------------------------------------------------------------------------
class CurriculumController:
    """
    Dynamically adjusts training difficulty based on model performance.

    Strategy:
      - Measures recent average loss over a sliding window.
      - If loss stops decreasing (plateau), advances curriculum level.
      - Curriculum levels map to increasingly complex data subsets or
        longer sequence lengths.

    Example curriculum levels:
      Level 0: short sequences (128 tokens), simple sentences
      Level 1: medium sequences (256 tokens)
      Level 2: full sequences (512+ tokens), complex documents
    """

    def __init__(
        self,
        num_levels: int = 3,
        plateau_patience: int = 500,
        improvement_threshold: float = 0.005,
    ):
        self.num_levels = num_levels
        self.plateau_patience = plateau_patience
        self.improvement_threshold = improvement_threshold
        self.current_level = 0
        self.loss_history: List[float] = []
        self.steps_without_improvement = 0
        self.best_loss = float("inf")

    def update(self, loss: float) -> int:
        """
        Record a loss value and potentially advance the curriculum level.
        Returns the current curriculum level (0-indexed).
        """
        self.loss_history.append(loss)

        if loss < self.best_loss - self.improvement_threshold:
            self.best_loss = loss
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        if (
            self.steps_without_improvement >= self.plateau_patience
            and self.current_level < self.num_levels - 1
        ):
            self.current_level += 1
            self.steps_without_improvement = 0
            self.best_loss = loss  # reset baseline for new level
            print(
                f"[Curriculum] Advanced to level {self.current_level} "
                f"after {len(self.loss_history)} steps. Loss: {loss:.4f}"
            )

        return self.current_level

    @property
    def seq_len_for_level(self) -> Dict[int, int]:
        """Map curriculum level to recommended sequence length."""
        return {0: 128, 1: 256, 2: 512, 3: 1024}

    def get_seq_len(self) -> int:
        return self.seq_len_for_level.get(self.current_level, 512)


# ---------------------------------------------------------------------------
# Self-Distillation Engine
# ---------------------------------------------------------------------------
class SelfDistillationEngine:
    """
    Knowledge distillation from the model's own past (frozen) checkpoint.

    The idea:
      - Periodically freeze a snapshot of the current model as the "teacher".
      - The live model ("student") is trained with a combined loss:
          L = alpha * CE(student, labels) + (1-alpha) * KL(student || teacher)
      - This prevents catastrophic forgetting during online/continual learning.
      - The teacher is refreshed every `refresh_every` steps.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.7,
        temperature: float = 4.0,
        refresh_every: int = 1000,
    ):
        self.student = model
        self.alpha = alpha          # weight of CE loss
        self.temperature = temperature
        self.refresh_every = refresh_every
        self.teacher: Optional[nn.Module] = None
        self._steps = 0
        self._refresh_teacher()

    def _refresh_teacher(self):
        """Snapshot the current model as teacher (frozen)."""
        self.teacher = deepcopy(self.student)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        student_output: dict,
    ) -> torch.Tensor:
        """
        Compute blended distillation + CE loss.

        Args:
            input_ids: token ids (B, T)
            labels: next-token labels (B, T)
            student_output: dict with 'loss' and 'logits' from student forward
        Returns:
            total_loss: scalar
        """
        self._steps += 1
        if self._steps % self.refresh_every == 0:
            self._refresh_teacher()

        ce_loss = student_output["loss"]
        student_logits = student_output["logits"]  # (B, T, V)

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_output = self.teacher(input_ids=input_ids)
            teacher_logits = teacher_output["logits"]  # (B, T, V)

        # KL divergence (soft labels) - shift for next-token prediction
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits[:, :-1] / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits[:, :-1] / T, dim=-1)

        kl_loss = F.kl_div(
            student_log_probs.reshape(-1, student_log_probs.size(-1)),
            teacher_probs.reshape(-1, teacher_probs.size(-1)),
            reduction="batchmean",
        ) * (T ** 2)

        return self.alpha * ce_loss + (1 - self.alpha) * kl_loss
