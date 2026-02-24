# tokenizer/tokenizer.py
# BPE Tokenizer for DeepSeek-Replica
# Wraps HuggingFace tokenizers library or provides a simple custom BPE

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class DeepSeekTokenizer:
    """
    Byte-Pair Encoding (BPE) tokenizer compatible with DeepSeek's vocabulary.

    Supports:
    - Training BPE from scratch on a text corpus
    - Save / load vocabulary
    - Encode / decode text
    - Special token handling (BOS, EOS, PAD, UNK)
    """

    SPECIAL_TOKENS = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
    }

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
    ):
        self.vocab: Dict[str, int] = vocab or dict(self.SPECIAL_TOKENS)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.merges: List[Tuple[str, str]] = merges or []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {m: i for i, m in enumerate(self.merges)}

        self.pad_token_id = self.SPECIAL_TOKENS["<pad>"]
        self.bos_token_id = self.SPECIAL_TOKENS["<bos>"]
        self.eos_token_id = self.SPECIAL_TOKENS["<eos>"]
        self.unk_token_id = self.SPECIAL_TOKENS["<unk>"]

    # ------------------------------------------------------------------
    # BPE core
    # ------------------------------------------------------------------

    @staticmethod
    def _get_pairs(word: Tuple[str, ...]) -> set:
        """Return set of consecutive symbol pairs."""
        return {(word[i], word[i + 1]) for i in range(len(word) - 1)}

    def _bpe(self, token: str) -> List[str]:
        """Apply BPE merges to a single pre-tokenized word."""
        word = tuple(token)  # individual bytes/chars
        pairs = self._get_pairs(word)
        if not pairs:
            return list(word)

        while True:
            # Find the highest-priority merge
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        return list(word)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self, texts: List[str], vocab_size: int = 32000, min_frequency: int = 2):
        """Train BPE on a list of strings."""
        from collections import Counter

        print(f"Training BPE tokenizer to vocab size {vocab_size} ...")

        # Build initial character-level vocabulary from corpus
        word_freqs: Counter = Counter()
        for text in texts:
            for word in text.split():
                word_freqs[" ".join(list(word)) + " </w>"] += 1

        # Convert to split-word format
        splits: Dict[str, List[str]] = {
            word: word.split() for word in word_freqs
        }

        # Seed vocab with all unique characters
        char_vocab: set = set()
        for word in splits.values():
            char_vocab.update(word)
        for ch in sorted(char_vocab):
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)
                self.id_to_token[self.vocab[ch]] = ch

        # BPE merge loop
        while len(self.vocab) < vocab_size:
            pair_freqs: Counter = Counter()
            for word, freq in word_freqs.items():
                syms = splits[word]
                for i in range(len(syms) - 1):
                    pair_freqs[(syms[i], syms[i + 1])] += freq

            if not pair_freqs:
                break
            best_pair = pair_freqs.most_common(1)[0]
            if best_pair[1] < min_frequency:
                break

            pair = best_pair[0]
            self.merges.append(pair)
            self.bpe_ranks[pair] = len(self.bpe_ranks)
            new_token = pair[0] + pair[1]
            self.vocab[new_token] = len(self.vocab)
            self.id_to_token[self.vocab[new_token]] = new_token

            # Apply new merge to all words
            for word in list(splits.keys()):
                syms = splits[word]
                new_syms: List[str] = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == pair[0] and syms[i + 1] == pair[1]:
                        new_syms.append(new_token)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                splits[word] = new_syms

        print(f"Final vocab size: {len(self.vocab)}")

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Tokenize text to a list of integer ids."""
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.bos_token_id)

        for word in text.split():
            bpe_tokens = self._bpe(word)
            for tok in bpe_tokens:
                tokens.append(self.vocab.get(tok, self.unk_token_id))

        if add_eos:
            tokens.append(self.eos_token_id)
        if max_length is not None:
            tokens = tokens[:max_length]
        return tokens

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of integer ids back to text."""
        special_ids = set(self.SPECIAL_TOKENS.values())
        parts: List[str] = []
        for i in ids:
            tok = self.id_to_token.get(i, "<unk>")
            if skip_special_tokens and i in special_ids:
                continue
            parts.append(tok)
        text = " ".join(parts).replace(" </w>", "").replace("</w>", "")
        return text

    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
        max_length: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> Dict[str, List[List[int]]]:
        """Encode a batch of texts with optional padding."""
        encoded = [
            self.encode(t, add_bos=add_bos, add_eos=add_eos, max_length=max_length)
            for t in texts
        ]
        if padding:
            max_len = max(len(e) for e in encoded)
            attention_masks = []
            padded = []
            for e in encoded:
                mask = [1] * len(e) + [0] * (max_len - len(e))
                e_padded = e + [self.pad_token_id] * (max_len - len(e))
                padded.append(e_padded)
                attention_masks.append(mask)
            return {"input_ids": padded, "attention_mask": attention_masks}
        return {"input_ids": encoded, "attention_mask": [[1] * len(e) for e in encoded]}

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, directory: Union[str, Path]):
        """Save vocabulary and merges to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        with open(directory / "merges.txt", "w", encoding="utf-8") as f:
            f.write("#version: 0.1\n")
            for a, b in self.merges:
                f.write(f"{a} {b}\n")
        print(f"Tokenizer saved to {directory}")

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "DeepSeekTokenizer":
        """Load a tokenizer from a saved directory."""
        directory = Path(directory)
        with open(directory / "vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
        merges: List[Tuple[str, str]] = []
        with open(directory / "merges.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))
        return cls(vocab=vocab, merges=merges)

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return f"DeepSeekTokenizer(vocab_size={len(self.vocab)}, merges={len(self.merges)})"


# ---------------------------------------------------------------------------
# HuggingFace wrapper (preferred for production use)
# ---------------------------------------------------------------------------

def load_hf_tokenizer(model_name_or_path: str = "deepseek-ai/DeepSeek-R1"):
    """
    Load the official DeepSeek tokenizer via HuggingFace transformers.
    Requires: pip install transformers
    """
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    except ImportError:
        raise ImportError(
            "transformers is required for load_hf_tokenizer. "
            "Install with: pip install transformers"
        )
