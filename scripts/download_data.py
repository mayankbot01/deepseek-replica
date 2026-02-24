#!/usr/bin/env python3
# scripts/download_data.py
"""
Download and preprocess public text datasets for training DeepSeek-Replica.

Supported datasets (via Hugging Face `datasets`):
  - openwebtext   : ~8 GB decompressed, English web text
  - wikitext-103  : ~500 MB, clean Wikipedia text
  - tinystories   : ~475 MB, synthetic short stories (great for tiny models)
  - bookcorpusopen: ~5 GB, open-source book text

Usage:
    python scripts/download_data.py \\
        --dataset openwebtext \\
        --output_dir data/openwebtext/ \\
        --tokenizer_dir tokenizer_saved/ \\
        --train_split 0.995 \\
        --max_tokens 2_000_000_000
"""

import argparse
import os
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download & preprocess text datasets")
    p.add_argument(
        "--dataset",
        type=str,
        default="openwebtext",
        choices=["openwebtext", "wikitext-103", "tinystories", "bookcorpusopen"],
        help="Dataset name (default: openwebtext)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/",
        help="Directory to write train.txt and val.txt",
    )
    p.add_argument(
        "--tokenizer_dir",
        type=str,
        default=None,
        help="If set, also pre-tokenize and save binary .bin files for fast loading",
    )
    p.add_argument(
        "--train_split",
        type=float,
        default=0.995,
        help="Fraction of data to use as training set (rest goes to val)",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Cap total training tokens (useful for quick experiments)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return p.parse_args()


def get_text_column(dataset_name: str) -> str:
    mapping = {
        "openwebtext": "text",
        "wikitext-103": "text",
        "tinystories": "text",
        "bookcorpusopen": "text",
    }
    return mapping[dataset_name]


def get_hf_path(dataset_name: str):
    mapping = {
        "openwebtext": ("openwebtext", None),
        "wikitext-103": ("wikitext", "wikitext-103-raw-v1"),
        "tinystories": ("roneneldan/TinyStories", None),
        "bookcorpusopen": ("bookcorpusopen", None),
    }
    return mapping[dataset_name]


def main():
    args = parse_args()
    random.seed(args.seed)

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` package is required. Install with:\n"
            "  pip install datasets"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_path, hf_config = get_hf_path(args.dataset)
    text_col = get_text_column(args.dataset)

    print(f"Downloading '{args.dataset}' from Hugging Face Hub ...")
    if hf_config:
        ds = load_dataset(hf_path, hf_config, split="train", streaming=False)
    else:
        ds = load_dataset(hf_path, split="train", streaming=False)

    print(f"Dataset size: {len(ds):,} documents")

    # Shuffle and split
    indices = list(range(len(ds)))
    random.shuffle(indices)
    split_idx = int(len(indices) * args.train_split)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    def write_split(idx_list, filename, max_tokens=None):
        path = output_dir / filename
        total_tokens = 0
        print(f"Writing {path} ({len(idx_list):,} documents) ...")
        with open(path, "w", encoding="utf-8") as f:
            for i, idx in enumerate(idx_list):
                text = ds[idx][text_col].strip()
                if not text:
                    continue
                f.write(text + "\n\n")
                # Rough token count: ~4 chars/token
                total_tokens += len(text) // 4
                if max_tokens and total_tokens >= max_tokens:
                    print(f"  Reached max_tokens cap ({max_tokens:,}) after {i+1:,} docs.")
                    break
                if (i + 1) % 50_000 == 0:
                    print(f"  Progress: {i+1:,} / {len(idx_list):,} docs")
        print(f"  Done. Est. tokens: {total_tokens:,}")
        return path

    train_file = write_split(train_idx, "train.txt", max_tokens=args.max_tokens)
    val_file = write_split(val_idx, "val.txt")

    print(f"\nData saved:")
    print(f"  Train : {train_file}")
    print(f"  Val   : {val_file}")

    # Optionally pre-tokenize
    if args.tokenizer_dir:
        print("\nPre-tokenizing data ...")
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from tokenizer.tokenizer import DeepSeekTokenizer
            import struct

            tokenizer = DeepSeekTokenizer.from_pretrained(args.tokenizer_dir)

            for split_file, bin_name in [(train_file, "train.bin"), (val_file, "val.bin")]:
                bin_path = output_dir / bin_name
                print(f"  Tokenizing {split_file} -> {bin_path}")
                text = split_file.read_text(encoding="utf-8")
                ids = tokenizer.encode(text)
                print(f"    Tokens: {len(ids):,}")
                with open(bin_path, "wb") as f:
                    f.write(struct.pack(f"{len(ids)}I", *ids))
                print(f"    Saved to {bin_path}")

        except Exception as e:
            print(f"  Warning: pre-tokenization failed: {e}")
            print("  You can tokenize manually via train.py.")

    print("\nDone!")


if __name__ == "__main__":
    main()
