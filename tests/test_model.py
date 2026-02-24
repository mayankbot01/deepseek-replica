# tests/test_model.py
"""Unit tests for the DeepSeek-Replica model."""
import pytest
import torch


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_tiny_config_imports(self):
        from model.config import DeepSeekConfig, get_config
        cfg = get_config("tiny")
        assert isinstance(cfg, DeepSeekConfig)

    def test_config_fields(self):
        from model.config import get_config
        cfg = get_config("tiny")
        assert cfg.num_hidden_layers > 0
        assert cfg.hidden_size > 0
        assert cfg.vocab_size > 0
        assert cfg.num_attention_heads > 0

    def test_all_size_variants(self):
        from model.config import get_config
        for size in ["tiny", "small", "medium"]:
            cfg = get_config(size)
            assert cfg is not None


# ---------------------------------------------------------------------------
# Attention tests
# ---------------------------------------------------------------------------

class TestAttention:
    @pytest.fixture
    def tiny_cfg(self):
        from model.config import get_config
        return get_config("tiny")

    def test_mla_forward_shape(self, tiny_cfg):
        from model.attention import MultiHeadLatentAttention
        attn = MultiHeadLatentAttention(tiny_cfg)
        attn.eval()
        B, T, H = 2, 16, tiny_cfg.hidden_size
        x = torch.randn(B, T, H)
        out, _ = attn(x)
        assert out.shape == (B, T, H), f"Expected ({B},{T},{H}), got {out.shape}"

    def test_mla_causal_mask(self, tiny_cfg):
        """Output at position i must not depend on positions > i."""
        from model.attention import MultiHeadLatentAttention
        attn = MultiHeadLatentAttention(tiny_cfg)
        attn.eval()
        B, T, H = 1, 8, tiny_cfg.hidden_size
        x = torch.randn(B, T, H)
        x_mod = x.clone()
        x_mod[0, -1] = x_mod[0, -1] * 0  # zero out last token
        out1, _ = attn(x)
        out2, _ = attn(x_mod)
        # First T-1 positions should be identical (causal)
        assert torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-5)


# ---------------------------------------------------------------------------
# MoE tests
# ---------------------------------------------------------------------------

class TestMoE:
    @pytest.fixture
    def tiny_cfg(self):
        from model.config import get_config
        return get_config("tiny")

    def test_moe_forward_shape(self, tiny_cfg):
        from model.moe import MoELayer
        moe = MoELayer(tiny_cfg)
        moe.eval()
        B, T, H = 2, 16, tiny_cfg.hidden_size
        x = torch.randn(B, T, H)
        out, aux_loss = moe(x)
        assert out.shape == (B, T, H)
        assert aux_loss.ndim == 0  # scalar

    def test_moe_aux_loss_positive(self, tiny_cfg):
        from model.moe import MoELayer
        moe = MoELayer(tiny_cfg)
        B, T, H = 2, 16, tiny_cfg.hidden_size
        x = torch.randn(B, T, H)
        _, aux_loss = moe(x)
        assert aux_loss.item() >= 0.0


# ---------------------------------------------------------------------------
# End-to-end model tests
# ---------------------------------------------------------------------------

class TestForCausalLM:
    @pytest.fixture
    def model_and_config(self):
        from model.config import get_config
        from model.transformer import DeepSeekForCausalLM
        cfg = get_config("tiny")
        model = DeepSeekForCausalLM(cfg)
        model.eval()
        return model, cfg

    def test_forward_logits_shape(self, model_and_config):
        model, cfg = model_and_config
        B, T = 2, 32
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        out = model(input_ids=ids)
        logits = out["logits"]
        assert logits.shape == (B, T, cfg.vocab_size)

    def test_forward_with_labels(self, model_and_config):
        model, cfg = model_and_config
        B, T = 2, 32
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        out = model(input_ids=ids, labels=ids)
        assert "loss" in out
        assert out["loss"].item() > 0.0

    def test_num_parameters(self, model_and_config):
        model, cfg = model_and_config
        n = model.num_parameters()
        assert n > 0
        print(f"\nTiny model params: {n/1e6:.1f}M")

    def test_generate(self, model_and_config):
        model, cfg = model_and_config
        prompt = torch.randint(0, cfg.vocab_size, (1, 8))
        gen = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=50)
        assert gen.shape[0] == 1
        assert gen.shape[1] == 8 + 10

    def test_gradient_flow(self, model_and_config):
        """All parameters should receive gradients on a backward pass."""
        model, cfg = model_and_config
        model.train()
        B, T = 1, 16
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        out = model(input_ids=ids, labels=ids)
        out["loss"].backward()
        no_grad = [
            name for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert len(no_grad) == 0, f"No gradient for: {no_grad[:5]}"
