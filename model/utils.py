# model/utils.py
# Training utilities: checkpointing, logging, metrics, profiling

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str = "deepseek", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout with timestamp."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = get_logger()


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return number of (trainable) parameters."""
    return sum(
        p.numel() for p in model.parameters()
        if (not trainable_only) or p.requires_grad
    )


def format_param_count(n: int) -> str:
    """Format param count as human-readable string (e.g. 1.23B)."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def print_model_summary(model: nn.Module):
    """Print a concise model summary including per-module param counts."""
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    print(f"{'Model Summary':=^60}")
    print(f"  Total parameters:     {format_param_count(total):>10}")
    print(f"  Trainable parameters: {format_param_count(trainable):>10}")
    print(f"  Frozen parameters:    {format_param_count(total - trainable):>10}")
    print("=" * 60)
    for name, module in model.named_children():
        n = count_parameters(module, trainable_only=False)
        print(f"  {name:<30} {format_param_count(n):>10}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config,
    output_dir: str,
    loss: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a full training checkpoint to {output_dir}/step_{step:07d}.pt
    Returns the path of the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"step_{step:07d}.pt"

    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "loss": loss,
    }
    if extra:
        payload.update(extra)

    torch.save(payload, ckpt_path)
    log.info(f"Checkpoint saved: {ckpt_path}  (step {step}, loss {loss:.4f if loss else 'N/A'})")
    return ckpt_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a checkpoint into model (and optionally optimizer).
    Returns the full checkpoint dict.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if missing:
        log.warning(f"Missing keys in state_dict: {missing}")
    if unexpected:
        log.warning(f"Unexpected keys in state_dict: {unexpected}")
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    log.info(
        f"Loaded checkpoint from {checkpoint_path}  "
        f"(step {ckpt.get('step', '?')}, loss {ckpt.get('loss', 'N/A')})"
    )
    return ckpt


def find_latest_checkpoint(output_dir: str) -> Optional[Path]:
    """Return the path to the highest-step .pt checkpoint in output_dir."""
    ckpts = sorted(Path(output_dir).glob("step_*.pt"))
    return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------
# Metrics tracker (rolling average)
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Accumulates metrics and computes rolling averages."""

    def __init__(self, window: int = 100):
        self.window = window
        self._values: Dict[str, list] = {}
        self._total: Dict[str, float] = {}
        self._count: Dict[str, int] = {}

    def update(self, **kwargs: float):
        for k, v in kwargs.items():
            if k not in self._values:
                self._values[k] = []
                self._total[k] = 0.0
                self._count[k] = 0
            self._values[k].append(v)
            self._total[k] += v
            self._count[k] += 1
            if len(self._values[k]) > self.window:
                removed = self._values[k].pop(0)
                self._total[k] -= removed

    def mean(self, key: str) -> float:
        if key not in self._values or not self._values[key]:
            return 0.0
        return sum(self._values[key]) / len(self._values[key])

    def report(self) -> str:
        parts = []
        for k in self._values:
            parts.append(f"{k}={self.mean(k):.4f}")
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Throughput / speed measurement
# ---------------------------------------------------------------------------

class Throughput:
    """Measures tokens-per-second throughput."""

    def __init__(self):
        self._t0 = time.perf_counter()
        self._tokens = 0

    def update(self, num_tokens: int):
        self._tokens += num_tokens

    def reset(self):
        self._t0 = time.perf_counter()
        self._tokens = 0

    @property
    def tokens_per_sec(self) -> float:
        elapsed = time.perf_counter() - self._t0
        return self._tokens / max(elapsed, 1e-6)

    def __repr__(self) -> str:
        tps = self.tokens_per_sec
        if tps >= 1000:
            return f"{tps / 1000:.1f}K tok/s"
        return f"{tps:.0f} tok/s"


# ---------------------------------------------------------------------------
# Learning rate schedule helpers
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Compute cosine-decay LR with linear warmup (stateless)."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Gradient norm
# ---------------------------------------------------------------------------

def get_grad_norm(model: nn.Module) -> float:
    """Compute the global L2 norm of all gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().norm(2).item() ** 2
    return math.sqrt(total_norm)


# ---------------------------------------------------------------------------
# Seed everything
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info(f"Random seed set to {seed}")
