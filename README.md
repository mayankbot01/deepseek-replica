# DeepSeek-Replica

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


A **from-scratch** Python replica of the DeepSeek large language model architecture, built purely with PyTorch — no Hugging Face model code. This repo implements every major architectural innovation from the DeepSeek-V2/V3 technical reports.

## Architecture Highlights

| Component | Description |
|---|---|
| **MLA** | Multi-Head Latent Attention with decoupled RoPE and low-rank KV compression |
| **MoE** | Mixture of Experts with top-K routing, shared experts, and aux load-balancing loss |
| **RoPE** | Rotary Position Embeddings with optional extended-context scaling |
| **GQA** | Grouped-Query Attention for efficient inference |
| **SwiGLU** | Gated linear unit activation for all FFN layers |
| **RMSNorm** | Root Mean Square Layer Normalization (no mean subtraction) |
| **BPE** | Custom Byte-Pair Encoding tokenizer with save/load support |

## Repository Structure

```
deepseek-replica/
├── model/
│   ├── __init__.py          # Package exports
│   ├── config.py            # DeepSeekConfig dataclass + size variants
│   ├── attention.py         # MultiHeadLatentAttention (MLA) + RotaryEmbedding
│   ├── moe.py               # MoELayer, MoEGate, ExpertMLP, DenseMLP
│   └── transformer.py       # DeepSeekForCausalLM, DeepSeekModel, DeepSeekDecoderLayer
├── tokenizer/
│   └── tokenizer.py         # BPE tokenizer with train/encode/decode/save/load
├── train.py                 # Full training loop with cosine LR schedule
├── inference.py             # Text generation + interactive REPL
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/mayankbot01/deepseek-replica.git
cd deepseek-replica
pip install -r requirements.txt
```

### 2. Train a tiny model on your own text

```bash
python train.py \
  --data path/to/your_corpus.txt \
  --size tiny \
  --batch_size 4 \
  --grad_accum 8 \
  --max_steps 5000 \
  --seq_len 256 \
  --tokenizer_dir tokenizer_saved/ \
  --output checkpoints/
```

### 3. Run inference

```bash
# Single prompt
python inference.py \
  --checkpoint checkpoints/step_0005000.pt \
  --tokenizer_dir tokenizer_saved/ \
  --prompt "The future of artificial intelligence is"

# Interactive REPL
python inference.py \
  --checkpoint checkpoints/step_0005000.pt \
  --tokenizer_dir tokenizer_saved/
```

### 4. Use in Python

```python
from model import DeepSeekForCausalLM, get_config
from tokenizer.tokenizer import DeepSeekTokenizer
import torch

# Build tiny model
config = get_config("tiny")
model  = DeepSeekForCausalLM(config)
print(f"Parameters: {model.num_parameters()/1e6:.1f}M")

# Forward pass
batch  = torch.randint(0, config.vocab_size, (2, 128))
output = model(input_ids=batch, labels=batch)
print(f"Loss: {output['loss'].item():.4f}")
```

## Model Size Variants

| Size | Layers | Hidden | Experts | Params (approx) |
|------|--------|--------|---------|------------------|
| `tiny` | 8 | 512 | 8 | ~30M |
| `small` | 28 | 2048 | 64 | ~2B |
| `medium` | 30 | 4096 | 64 | ~7B |

## Key Design Decisions

### Multi-Head Latent Attention (MLA)
Instead of caching the full K/V tensors, MLA projects them into a low-rank latent space (`kv_lora_rank = 512`). This cuts the KV cache memory by ~13x compared to standard MHA while maintaining quality.

### Decoupled RoPE
Rotary embeddings are applied only to a subset of head dimensions (`qk_rope_head_dim = 64`). The remaining dimensions (`qk_nope_head_dim = 128`) carry absolute position-independent content. This makes extending context length cheaper.

### Mixture of Experts
Each MoE layer routes each token to the top-K (default: 6 of 64) experts. Two shared experts always run on every token, providing a stable residual path. An auxiliary load-balancing loss prevents expert collapse.

### Training
- AdamW optimizer with `(beta1=0.9, beta2=0.95)`
- Cosine learning rate decay with linear warmup
- Gradient clipping at 1.0
- BFloat16 mixed precision
- Gradient accumulation for large effective batch sizes

## License

MIT — see [LICENSE](LICENSE) for details.

## References

- [DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
