#!/usr/bin/env python3
# self_train.py
# Self-learning training script orchestrating all modules:
# - MAMLInnerLoop for rapid adaptation
# - CurriculumController for progressive difficulty
# - SelfDistillationEngine for preventing forgetting
# - EpisodicMemoryBank for retrieval-augmented learning
# - AdaptiveRouter for expert routing evolution

import argparse
import math
import os
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import DeepSeekForCausalLM, get_config
from model.meta_learner import MAMLInnerLoop, AdaptiveRouter, CurriculumController, SelfDistillationEngine
from model.memory_bank import EpisodicMemoryBank, MemoryStats
from model.self_learning import RewardModel, ExperienceBuffer
from tokenizer.tokenizer import DeepSeekTokenizer
from train import TextDataset, collate_fn, count_parameters


def self_train(
    data_path: str,
    output_dir: str = "checkpoints_self",
    model_size: str = "tiny",
    batch_size: int = 4,
    grad_accum: int = 8,
    max_steps: int = 20_000,
    warmup_steps: int = 1000,
    learning_rate: float = 1e-4,
    max_grad_norm: float = 1.0,
    seq_len: int = 256,
    save_every: int = 1000,
    log_every: int = 50,
    device: str = None,
    tokenizer_dir: str = None,
    # Self-learning hyperparameters
    use_maml: bool = True,
    use_curriculum: bool = True,
    use_self_distillation: bool = True,
    use_memory_bank: bool = True,
    maml_inner_lr: float = 1e-3,
    maml_inner_steps: int = 3,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.split(":")[0]
    dtype = torch.bfloat16
    print(f"Self-learning training on {device} with dtype {dtype}")

    # ----- Tokenizer -----
    if tokenizer_dir and Path(tokenizer_dir).exists():
        tokenizer = DeepSeekTokenizer.load(tokenizer_dir)
    else:
        tokenizer = DeepSeekTokenizer()
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
    model = DeepSeekForCausalLM(config).to(device).to(dtype)
    print(f"Model: {count_parameters(model)}")

    # ----- Self-Learning Components -----
    reward_model = RewardModel(config).to(device).to(dtype) if use_maml else None
    maml = MAMLInnerLoop(model, inner_lr=maml_inner_lr, num_inner_steps=maml_inner_steps) if use_maml else None
    curriculum = CurriculumController(num_levels=3, plateau_patience=500) if use_curriculum else None
    distillation = SelfDistillationEngine(model, alpha=0.7, refresh_every=1000) if use_self_distillation else None
    memory_bank = EpisodicMemoryBank(config.hidden_size, key_dim=64, max_memories=2048).to(device) if use_memory_bank else None
    memory_stats = MemoryStats() if use_memory_bank else None
    experience_buffer = ExperienceBuffer(max_size=1000)

    print("\n[Self-Learning Components]")
    print(f"  MAML:              {use_maml}")
    print(f"  Curriculum:        {use_curriculum}")
    print(f"  Self-Distillation: {use_self_distillation}")
    print(f"  Memory Bank:       {use_memory_bank}\n")

    # ----- Data -----
    dataset = TextDataset(tokenizer, data_path, seq_len=seq_len)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
        pin_memory=(device_type == "cuda")
    )
    print(f"Dataset: {len(dataset)} samples | {len(loader)} batches per epoch")

    # ----- Optimizer -----
    params = list(model.parameters())
    if memory_bank:
        params += list(memory_bank.parameters())
    if reward_model:
        params += list(reward_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.95))

    # ----- Training Loop -----
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    global_step = 0
    running_loss = 0.0
    t0 = time.time()
    optimizer.zero_grad()

    print("\n=== Self-Learning Training Loop ===")
    while global_step < max_steps:
        for input_ids, labels in loader:
            if global_step >= max_steps:
                break

            input_ids, labels = input_ids.to(device), labels.to(device)

            # === Curriculum: adjust seq_len dynamically ===
            if curriculum:
                curr_seq_len = curriculum.get_seq_len()
                if input_ids.size(1) > curr_seq_len:
                    input_ids = input_ids[:, :curr_seq_len]
                    labels = labels[:, :curr_seq_len]

            with torch.autocast(device_type=device_type, dtype=dtype, enabled=True):
                output = model(input_ids=input_ids, labels=labels)
                base_loss = output["loss"]

                # === Self-Distillation ===
                if distillation:
                    loss = distillation.compute_loss(input_ids, labels, output)
                else:
                    loss = base_loss

                # === Memory Bank: store high-quality experiences ===
                if memory_bank and global_step % 10 == 0:
                    with torch.no_grad():
                        hidden = model.model(input_ids)[0]  # (B, T, D)
                        reward = -base_loss.item()  # higher = better
                        memory_bank.store(hidden[0, -1], hidden[0], reward)
                        if memory_stats:
                            memory_stats.record_store(reward)

                loss = loss / grad_accum

            loss.backward()
            running_loss += loss.item() * grad_accum

            # Gradient accumulation step
            if (global_step + 1) % grad_accum == 0:
                nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            # === Curriculum Update ===
            if curriculum:
                curriculum.update(base_loss.item())

            # === Logging ===
            if global_step % log_every == 0:
                elapsed = time.time() - t0
                avg_loss = running_loss / log_every
                ppl = math.exp(min(avg_loss, 20))
                log_str = (
                    f"Step {global_step:>6d}/{max_steps} | "
                    f"loss {avg_loss:.4f} | ppl {ppl:.2f} | "
                    f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                    f"{elapsed:.0f}s"
                )
                if curriculum:
                    log_str += f" | curriculum={curriculum.current_level}"
                if memory_bank:
                    log_str += f" | mem={memory_bank.size()}"
                print(log_str)
                running_loss = 0.0
                t0 = time.time()

            # === Checkpointing ===
            if global_step % save_every == 0 or global_step == max_steps:
                ckpt = {
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                }
                if memory_bank:
                    ckpt["memory_bank"] = memory_bank.state_dict()
                ckpt_path = output_dir / f"self_step_{global_step:07d}.pt"
                torch.save(ckpt, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")
                if memory_stats:
                    print(f"  {memory_stats.summary()}")

    print("\n=== Self-Learning Training Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Learning Training for DeepSeek-Replica")
    parser.add_argument("--data", required=True, help="Training data path")
    parser.add_argument("--output", default="checkpoints_self")
    parser.add_argument("--size", default="tiny", choices=["tiny", "small", "medium"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=20_000)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tokenizer_dir", default=None)
    parser.add_argument("--use_maml", action="store_true", default=True)
    parser.add_argument("--use_curriculum", action="store_true", default=True)
    parser.add_argument("--use_self_distillation", action="store_true", default=True)
    parser.add_argument("--use_memory_bank", action="store_true", default=True)
    args = parser.parse_args()

    self_train(
        data_path=args.data,
        output_dir=args.output,
        model_size=args.size,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        device=args.device,
        tokenizer_dir=args.tokenizer_dir,
        use_maml=args.use_maml,
        use_curriculum=args.use_curriculum,
        use_self_distillation=args.use_self_distillation,
        use_memory_bank=args.use_memory_bank,
    )
