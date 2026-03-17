# model/memory_bank.py
# Episodic Memory Bank for Retrieval-Augmented Self-Improvement
# Stores (hidden_state, token_sequence) pairs and retrieves semantically similar
# past contexts to augment current training — enabling the model to consolidate
# knowledge over time without forgetting.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import heapq


# ---------------------------------------------------------------------------
# MemoryEntry: single stored experience
# ---------------------------------------------------------------------------
class MemoryEntry:
    """
    Represents a single stored memory:
      - key:  compressed hidden-state vector (used for similarity retrieval)
      - value: original token sequence or hidden states
      - reward: scalar quality signal (higher = more valuable memory)
      - step: training step at which this was stored
    """
    __slots__ = ["key", "value", "reward", "step"]

    def __init__(self, key: torch.Tensor, value: torch.Tensor, reward: float, step: int):
        self.key = key.detach().cpu()    # (D_key,)
        self.value = value.detach().cpu()  # (T, D) or (T,)
        self.reward = reward
        self.step = step

    def __lt__(self, other):
        # For heap: lower reward = lower priority (min-heap used as bounded buffer)
        return self.reward < other.reward


# ---------------------------------------------------------------------------
# EpisodicMemoryBank
# ---------------------------------------------------------------------------
class EpisodicMemoryBank(nn.Module):
    """
    Fixed-capacity episodic memory for retrieval-augmented self-improvement.

    Architecture:
      - A key projection: maps hidden states -> low-dim keys for fast lookup.
      - Stores top-N experiences by reward (bounded priority queue).
      - Retrieval: cosine similarity over keys returns top-K relevant memories.
      - The retrieved value vectors can then be prepended to the context
        or concatenated with the query as retrieval-augmented context.

    This enables the model to:
      1. Remember high-quality generations and reuse their representations.
      2. Avoid catastrophic forgetting of rare-but-important patterns.
      3. Act as an external working memory that grows with experience.
    """

    def __init__(
        self,
        hidden_size: int,
        key_dim: int = 64,
        max_memories: int = 4096,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.key_dim = key_dim
        self.max_memories = max_memories

        # Learnable projection: hidden_state -> compact retrieval key
        self.key_proj = nn.Linear(hidden_size, key_dim, bias=False)

        # Internal memory storage (min-heap bounded by max_memories)
        self._memories: List[MemoryEntry] = []
        self._step = 0

    def encode_key(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Project a hidden state (last token representation) into key space.
        Args:
            hidden_state: (D,) or (1, D)
        Returns:
            key: (D_key,) L2-normalized
        """
        h = hidden_state.view(-1, self.hidden_size)
        key = self.key_proj(h)  # (1, key_dim)
        return F.normalize(key, dim=-1).squeeze(0)  # (key_dim,)

    def store(
        self,
        hidden_state: torch.Tensor,
        value: torch.Tensor,
        reward: float,
    ):
        """
        Store a new memory entry, evicting lowest-reward entry if at capacity.

        Args:
            hidden_state: representative hidden vector for this memory (D,)
            value: the content to recall (e.g. hidden states or token ids) (T, D)
            reward: quality score; higher memories are retained longer
        """
        self._step += 1
        with torch.no_grad():
            key = self.encode_key(hidden_state)

        entry = MemoryEntry(key, value, reward, self._step)

        if len(self._memories) < self.max_memories:
            heapq.heappush(self._memories, entry)
        else:
            # Replace lowest-reward memory if new one is better
            if reward > self._memories[0].reward:
                heapq.heapreplace(self._memories, entry)

    def retrieve(
        self,
        query_hidden: torch.Tensor,
        top_k: int = 4,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve top-K most relevant memory values using cosine similarity.

        Args:
            query_hidden: (D,) current query hidden state
            top_k: number of memories to retrieve
        Returns:
            retrieved: (K, T, D) tensor of retrieved value sequences, or None
        """
        if len(self._memories) == 0:
            return None

        with torch.no_grad():
            query_key = self.encode_key(query_hidden)  # (key_dim,)

            # Stack all memory keys
            keys = torch.stack([m.key for m in self._memories], dim=0)  # (M, key_dim)
            keys = keys.to(query_key.device)

            # Cosine similarity
            sims = torch.matmul(keys, query_key)  # (M,)
            k = min(top_k, len(self._memories))
            _, top_indices = torch.topk(sims, k)

            # Retrieve values
            retrieved = []
            for idx in top_indices.tolist():
                v = self._memories[idx].value.to(query_key.device)
                retrieved.append(v)

        return retrieved  # list of (T, D) tensors

    def size(self) -> int:
        return len(self._memories)

    def clear(self):
        self._memories.clear()


# ---------------------------------------------------------------------------
# MemoryAugmentedLayer: wraps a decoder layer with memory-augmented context
# ---------------------------------------------------------------------------
class MemoryAugmentedLayer(nn.Module):
    """
    Thin wrapper around a DeepSeekDecoderLayer that prepends retrieved
    memory keys to the attention context via cross-attention.

    Instead of full cross-attention (expensive), we use a lightweight
    gated residual: the retrieved memory hidden states are projected to
    the same hidden dimension and added to the input with a learned gate.

    Gate mechanism prevents retrieved noise from corrupting activations:
      output = hidden + sigmoid(gate) * memory_contribution
    """

    def __init__(self, hidden_size: int, key_dim: int = 64):
        super().__init__()
        self.memory_proj = nn.Linear(key_dim, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_bank: Optional[EpisodicMemoryBank],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, D) from previous decoder layer
            memory_bank: optional EpisodicMemoryBank instance
        Returns:
            augmented hidden states: (B, T, D)
        """
        if memory_bank is None or memory_bank.size() == 0:
            return hidden_states

        # Use last-token representation as retrieval query
        query = hidden_states[:, -1, :].mean(0)  # (D,)
        retrieved = memory_bank.retrieve(query, top_k=4)

        if retrieved is None:
            return hidden_states

        # Aggregate retrieved memories: average pool
        # Each item is (T, D) or (D,), we use mean over T
        memory_vecs = []
        for r in retrieved:
            r = r.to(hidden_states.device)
            if r.dim() == 2:
                memory_vecs.append(r.mean(0))  # (D,)
            elif r.dim() == 1:
                memory_vecs.append(r)           # (D,)

        if not memory_vecs:
            return hidden_states

        mem_agg = torch.stack(memory_vecs, dim=0).mean(0)  # (D,)
        mem_agg = mem_agg.unsqueeze(0).unsqueeze(0)         # (1, 1, D)
        mem_agg = mem_agg.expand(hidden_states.shape[0], hidden_states.shape[1], -1)

        # Gated residual
        gate_input = torch.cat([hidden_states, mem_agg], dim=-1)  # (B, T, 2D)
        gate_val = torch.sigmoid(self.gate(gate_input))            # (B, T, D)
        return hidden_states + gate_val * mem_agg


# ---------------------------------------------------------------------------
# MemoryStats: tracking and diagnostics
# ---------------------------------------------------------------------------
class MemoryStats:
    """
    Lightweight diagnostics tracker for the memory bank.
    Tracks:
      - total stores, total retrievals
      - average reward of stored memories
      - retrieval hit rate (how often retrieved memories exceed min quality)
    """

    def __init__(self):
        self.num_stores = 0
        self.num_retrievals = 0
        self.total_reward = 0.0
        self.hit_quality_threshold = 0.5
        self.hit_count = 0

    def record_store(self, reward: float):
        self.num_stores += 1
        self.total_reward += reward

    def record_retrieval(self, rewards: List[float]):
        self.num_retrievals += 1
        for r in rewards:
            if r >= self.hit_quality_threshold:
                self.hit_count += 1

    def summary(self) -> str:
        avg_r = self.total_reward / max(self.num_stores, 1)
        hit_rate = self.hit_count / max(self.num_retrievals, 1)
        return (
            f"MemoryBank | stores={self.num_stores} | "
            f"avg_reward={avg_r:.3f} | "
            f"retrievals={self.num_retrievals} | "
            f"hit_rate={hit_rate:.2%}"
        )
