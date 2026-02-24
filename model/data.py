# model/data.py
# Dataset utilities for DeepSeek-Replica
# Supports: plain text files, JSONL, and HuggingFace datasets

import json
import random
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import Dataset, IterableDataset


# ---------------------------------------------------------------------------
# Plain-text windowed dataset (same as in train.py, but more configurable)
# ---------------------------------------------------------------------------

class TextFileDataset(Dataset):
    """
    Reads a plain UTF-8 text file and returns fixed-length token windows.
    Overlapping stride is supported for denser supervision.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer,
        seq_len: int = 512,
        stride: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.seq_len = seq_len
        stride = stride or seq_len  # non-overlapping by default

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        all_ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
        self.samples: List[List[int]] = []
        for i in range(0, len(all_ids) - seq_len, stride):
            self.samples.append(all_ids[i : i + seq_len + 1])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.samples[idx]
        return {
            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
            "labels":    torch.tensor(chunk[1:],  dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# JSONL dataset  (each line: {"text": "..."} or {"input": ..., "output": ...})
# ---------------------------------------------------------------------------

class JsonlDataset(Dataset):
    """
    Reads a JSONL file where each line is a JSON object.
    Supports two formats:
      - Pretraining: {"text": "..."}  -> tokenize full text
      - Instruction tuning: {"instruction": "...", "output": "..."}
        -> "<bos>INST: {instruction}\n\nOUT: {output}<eos>"
    """

    INST_TEMPLATE = "<bos>INST: {instruction}\n\nOUT: {output}<eos>"

    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer,
        seq_len: int = 512,
        mode: str = "pretrain",  # "pretrain" | "instruct"
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mode = mode
        self.records: List[str] = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if mode == "instruct":
                    text = self.INST_TEMPLATE.format(
                        instruction=obj.get("instruction", obj.get("input", "")),
                        output=obj.get("output", ""),
                    )
                else:
                    text = obj.get("text", "")
                if text:
                    self.records.append(text)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.tokenizer.encode(
            self.records[idx],
            add_bos=(self.mode == "pretrain"),
            add_eos=(self.mode == "pretrain"),
            max_length=self.seq_len + 1,
        )
        # Pad or truncate to seq_len + 1
        if len(ids) < self.seq_len + 1:
            ids = ids + [self.tokenizer.pad_token_id] * (self.seq_len + 1 - len(ids))
        return {
            "input_ids": torch.tensor(ids[:-1], dtype=torch.long),
            "labels":    torch.tensor(ids[1:],  dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Streaming dataset for very large corpora (memory-efficient)
# ---------------------------------------------------------------------------

class StreamingTextDataset(IterableDataset):
    """
    Streams tokens from one or more text files without loading all into RAM.
    Suitable for pre-training on hundreds of GBs of text.
    """

    def __init__(
        self,
        file_paths: List[Union[str, Path]],
        tokenizer,
        seq_len: int = 512,
        shuffle_files: bool = True,
        seed: int = 42,
    ):
        self.file_paths = [Path(p) for p in file_paths]
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle_files = shuffle_files
        self.seed = seed

    def _token_stream(self) -> Iterator[int]:
        paths = list(self.file_paths)
        if self.shuffle_files:
            random.Random(self.seed).shuffle(paths)
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ids = self.tokenizer.encode(line, add_bos=False, add_eos=False)
                    yield from ids

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer: List[int] = []
        for token_id in self._token_stream():
            buffer.append(token_id)
            if len(buffer) == self.seq_len + 1:
                yield {
                    "input_ids": torch.tensor(buffer[:-1], dtype=torch.long),
                    "labels":    torch.tensor(buffer[1:],  dtype=torch.long),
                }
                buffer = []


# ---------------------------------------------------------------------------
# HuggingFace datasets wrapper (optional)
# ---------------------------------------------------------------------------

def make_hf_dataset(
    dataset_name: str,
    tokenizer,
    seq_len: int = 512,
    split: str = "train",
    text_column: str = "text",
    num_proc: int = 4,
    streaming: bool = False,
):
    """
    Wrap a HuggingFace dataset for language modelling.
    Requires: pip install datasets

    Example:
        ds = make_hf_dataset("wikitext", tokenizer, split="train")
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    raw = load_dataset(dataset_name, split=split, streaming=streaming)

    def tokenize(examples):
        out = tokenizer.batch_encode(
            examples[text_column],
            padding=False,
            max_length=seq_len + 1,
            add_bos=True,
            add_eos=True,
        )
        return out

    if streaming:
        return raw.map(tokenize, batched=True, remove_columns=[text_column])
    return raw.map(
        tokenize,
        batched=True,
        remove_columns=raw.column_names,
        num_proc=num_proc,
    )


# ---------------------------------------------------------------------------
# Collate function (handles variable-length batches with padding)
# ---------------------------------------------------------------------------

def collate_pad(batch: List[Dict[str, torch.Tensor]], pad_id: int = 0):
    """
    Collate a list of {input_ids, labels} dicts into a padded batch tensor.
    Also produces attention_mask marking real vs padded positions.
    """
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids_list, labels_list, mask_list = [], [], []
    for b in batch:
        L = b["input_ids"].size(0)
        pad = max_len - L
        input_ids_list.append(
            torch.cat([b["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)])
        )
        mask_list.append(
            torch.cat([torch.ones(L, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)])
        )
    return {
        "input_ids":      torch.stack(input_ids_list),
        "labels":         torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }
