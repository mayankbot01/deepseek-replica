#!/usr/bin/env python3
# train.py
# Training script for DeepSeek-Replica
import argparse
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import DeepSeekForCausalLM, get_config
from tokenizer.tokenizer import DeepSeekTokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TextDataset(Dataset):
    """Simple text dataset that reads a plain-text file and windows it."""

    def __init__(self, tokenizer: DeepSeekTokenizer, file_path: str, seq_len: int = 512):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        all_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        # chunk into fixed-length windows
        self.samples = [
            all_ids[i : i + seq_len + 1]
            for i in range(0, len(all_ids) - seq_len, seq_len)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, labels


def collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return input_ids, labels


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
class CosineScheduler:
    """Linear warmup then cosine decay learning-rate scheduler."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float = 1e-5,
    ):
        self.opt = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup_steps:
            lr = self.max_lr * self.step_num / max(self.warmup_steps, 1)
        elif self.step_num >= self.max_steps:
            lr = self.min_lr
        else:
            progress = (self.step_num - self.warmup_steps) / (
                self.max_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


def count_parameters(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M"


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(
    data_path: str,
    output_dir: str = "checkpoints",
    model_size: str = "tiny",
    batch_size: int = 4,
    grad_accum: int = 8,
    max_steps: int = 10_000,
    warmup_steps: int = 500,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    seq_len: int = 512,
    save_every: int = 1000,
    log_every: int = 50,
    device: Optional[str] = None,
    tokenizer_dir: Optional[str] = None,
    fp16: bool = False,
    bf16: bool = True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    print(f"Training on {device} with dtype {dtype}")

    # FIX: torch.autocast requires device_type = "cuda" or "cpu", NOT "cuda:0"
    # Extract the base device type from the full device string
    device_type = device.split(":")[0]  # "cuda:0" -> "cuda", "cpu" -> "cpu"

    # ----- Tokenizer -----
    if tokenizer_dir and Path(tokenizer_dir).exists():
        tokenizer = DeepSeekTokenizer.load(tokenizer_dir)
    else:
        tokenizer = DeepSeekTokenizer()
        # Train tokenizer from data file
        with open(data_path, "r", encoding="utf-8") as f:
            texts = f.read().split("\n")
        tokenizer.train(texts, vocab_size=8000)
        if tokenizer_dir:
            tokenizer.save(tokenizer_dir)
    print(f"Tokenizer: {tokenizer}")

    # ----- Model -----
    config = get_config(model_size)
    config.vocab_size = len(tokenizer)
    config.max_seq_len = seq_len
    model = DeepSeekForCausalLM(config)
    # FIX: move to device BEFORE casting to dtype to avoid double copy
    model = model.to(device)
    model = model.to(dtype)
    print(f"Model size: {count_parameters(model)}")

    # ----- Data -----
    dataset = TextDataset(tokenizer, data_path, seq_len=seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device_type == "cuda"),
    )
    print(f"Dataset: {len(dataset)} samples | {len(loader)} batches per epoch")

    # ----- Optimizer & Scheduler -----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = CosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        max_lr=learning_rate,
        min_lr=learning_rate / 10,
    )

    # ----- Gradient scaler for fp16 -----
    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device_type == "cuda"))

    # ----- Training loop -----
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    micro_step = 0  # FIX: track micro-steps separately from optimizer steps
    running_loss = 0.0
    t0 = time.time()
    optimizer.zero_grad()

    while global_step < max_steps:
        for input_ids, labels in loader:
            if global_step >= max_steps:
                break

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # FIX: use device_type ("cuda"/"cpu") not full device string ("cuda:0")
            with torch.autocast(device_type=device_type, dtype=dtype, enabled=(fp16 or bf16)):
                out = model(input_ids=input_ids, labels=labels)
                loss = out["loss"] / grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum
            micro_step += 1

            # FIX: use micro_step to check gradient accumulation boundary
            if micro_step % grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr = scheduler.step()
                global_step += 1

                if global_step % log_every == 0:
                    elapsed = time.time() - t0
                    avg_loss = running_loss / log_every
                    perplexity = math.exp(min(avg_loss, 20))
                    print(
                        f"Step {global_step:>6d}/{max_steps} | "
                        f"loss {avg_loss:.4f} | ppl {perplexity:.2f} | "
                        f"lr {lr:.2e} | {elapsed:.0f}s elapsed"
                    )
                    running_loss = 0.0
                    t0 = time.time()

                if global_step % save_every == 0 or global_step == max_steps:
                    ckpt_path = output_dir / f"step_{global_step:07d}.pt"
                    torch.save(
                        {
                            "step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": config,
                        },
                        ckpt_path,
                    )
                    print(f"Checkpoint saved: {ckpt_path}")

    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepSeek-Replica LLM")
    parser.add_argument("--data", required=True, help="Path to training text file")
    parser.add_argument(
        "--output", default="checkpoints", help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--size",
        default="tiny",
        choices=["tiny", "small", "medium"],
        help="Model size variant",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tokenizer_dir", default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    args = parser.parse_args()
    train(
        data_path=args.data,
        output_dir=args.output,
        model_size=args.size,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        seq_len=args.seq_len,
        save_every=args.save_every,
        log_every=args.log_every,
        device=args.device,
        tokenizer_dir=args.tokenizer_dir,
        bf16=args.bf16,
        fp16=args.fp16,
    )
