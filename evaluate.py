#!/usr/bin/env python3
# evaluate.py  -  Perplexity & downstream evaluation for DeepSeek-Replica
"""
Usage:
    python evaluate.py \\
        --checkpoint checkpoints/step_005000.pt \\
        --tokenizer_dir tokenizer_saved/ \\
        --data path/to/test.txt \\
        --seq_len 512 \\
        --batch_size 4
"""

import argparse
import math
import time
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

from model import DeepSeekForCausalLM, DeepSeekConfig
from tokenizer.tokenizer import DeepSeekTokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Sliding-window token dataset for evaluation."""

    def __init__(self, token_ids: List[int], seq_len: int):
        self.ids = token_ids
        self.seq_len = seq_len
        # non-overlapping windows
        self.n = len(token_ids) // (seq_len + 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        start = idx * (self.seq_len + 1)
        chunk = self.ids[start : start + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: DeepSeekForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    device_type: str,
    max_batches: int = None,
) -> float:
    """Compute token-level perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for i, (x, y) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        # FIX: use device_type ("cuda"/"cpu") not full device string
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            out = model(input_ids=x, labels=y)

        loss = out["loss"].item()
        n_tokens = (y != -100).sum().item()
        total_loss += loss * n_tokens
        total_tokens += n_tokens

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            ppl_so_far = math.exp(total_loss / total_tokens)
            print(
                f"  [{i+1:>5d}] ppl={ppl_so_far:.2f}  tokens={total_tokens:,}"
                f"  elapsed={elapsed:.1f}s"
            )

    perplexity = math.exp(total_loss / total_tokens)
    return perplexity


@torch.no_grad()
def compute_bits_per_byte(
    model: DeepSeekForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    device_type: str,
) -> float:
    """Compute bits-per-byte (BPB)."""
    model.eval()
    total_nll = 0.0
    total_bytes = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        # FIX: use device_type ("cuda"/"cpu")
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            out = model(input_ids=x, labels=y)
        nll = out["loss"].item()
        n_tokens = (y != -100).sum().item()
        total_nll += nll * n_tokens
        # Rough: assume ~1 token ~ 4 bytes (UTF-8 average for BPE)
        total_bytes += n_tokens * 4

    bpb = (total_nll / math.log(2)) / total_bytes
    return bpb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a DeepSeek-Replica checkpoint")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .pt checkpoint saved by train.py")
    p.add_argument("--tokenizer_dir", type=str, required=True,
                   help="Directory with saved BPE tokenizer files")
    p.add_argument("--data", type=str, required=True,
                   help="Plain-text evaluation file")
    p.add_argument("--seq_len", type=int, default=512,
                   help="Sequence length for evaluation windows (default: 512)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=None,
                   help="Cap evaluation at this many batches (for quick checks)")
    p.add_argument("--device", type=str, default=None,
                   help="Device string, e.g. 'cuda:0'. Auto-detects if not set.")
    p.add_argument("--bits_per_byte", action="store_true",
                   help="Also report bits-per-byte metric")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Device ----
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FIX: extract base device type for torch.autocast
    device_type = device.type  # torch.device.type gives "cuda" or "cpu"
    print(f"Device: {device}")

    # ---- Load tokenizer ----
    print("Loading tokenizer ...")
    # FIX: DeepSeekTokenizer uses .load() not .from_pretrained()
    tokenizer = DeepSeekTokenizer.load(args.tokenizer_dir)

    # ---- Tokenize data ----
    print(f"Tokenizing {args.data} ...")
    text = Path(args.data).read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text)
    print(f"Total tokens: {len(token_ids):,}")

    dataset = TextDataset(token_ids, seq_len=args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device_type == "cuda"),
    )
    print(f"Evaluation windows: {len(dataset):,}")

    # ---- Load model ----
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config: DeepSeekConfig = ckpt["config"]
    model = DeepSeekForCausalLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    # ---- Evaluate ----
    print("\nComputing perplexity ...")
    ppl = compute_perplexity(
        model, dataloader, device, device_type, max_batches=args.max_batches
    )
    print(f"\n{'='*40}")
    print(f"Perplexity : {ppl:.4f}")
    print(f"Bits/token : {math.log2(ppl):.4f}")

    if args.bits_per_byte:
        bpb = compute_bits_per_byte(model, dataloader, device, device_type)
        print(f"Bits/byte  : {bpb:.4f}")

    print(f"{'='*40}")


if __name__ == "__main__":
    main()
