"""
tests/test_integration_extended.py

Extended integration tests — combines attention with different norm variants,
verifies end-to-end pipelines, and tests realistic usage patterns.
"""

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from lorentz_transformer import (
    LorentzMultiHeadAttention,
    MinkowskiLayerNorm,
    MinkowskiLayerNormImproved,
    MinkowskiLayerNormOptimized,
    MinkowskiLayerNormStable,
    compute_dt2_info,
    hutchinson_diag_hessian,
)


@dataclass
class Cfg:
    d_model: int = 32
    n_heads: int = 4
    lorentz_alpha: float = 0.25
    dropout: float = 0.0


# ======================================================================
# Attention + Norm variant combinations
# ======================================================================


class TestAttentionWithNormVariants:
    """Test LorentzMultiHeadAttention combined with each norm variant."""

    @pytest.fixture
    def cfg(self):
        return Cfg()

    @pytest.fixture
    def timelike_mask(self, cfg):
        mask = torch.zeros(cfg.d_model, dtype=torch.bool)
        mask[: cfg.d_model // 4] = True
        return mask

    @pytest.mark.parametrize(
        "norm_cls",
        [
            MinkowskiLayerNormOptimized,
            MinkowskiLayerNormStable,
            MinkowskiLayerNormImproved,
        ],
    )
    def test_attention_then_norm(self, cfg, timelike_mask, norm_cls):
        """Attention → Norm pipeline should produce valid output."""
        attn = LorentzMultiHeadAttention(cfg)
        norm = norm_cls(cfg.d_model)

        attn.set_timelike_mask(timelike_mask)
        norm.set_timelike_mask(timelike_mask)

        x = torch.randn(2, 8, cfg.d_model)
        attn_out, weights = attn(x)
        normed = norm(attn_out)

        assert normed.shape == x.shape
        assert torch.isfinite(normed).all()
        assert weights.shape == (2, cfg.n_heads, 8, 8)

    @pytest.mark.parametrize(
        "norm_cls",
        [
            MinkowskiLayerNormOptimized,
            MinkowskiLayerNormStable,
            MinkowskiLayerNormImproved,
        ],
    )
    def test_backward_through_attn_and_norm(self, cfg, timelike_mask, norm_cls):
        """Backward pass through the full pipeline should work."""
        attn = LorentzMultiHeadAttention(cfg)
        norm = norm_cls(cfg.d_model)

        attn.set_timelike_mask(timelike_mask)
        norm.set_timelike_mask(timelike_mask)

        x = torch.randn(2, 8, cfg.d_model, requires_grad=True)
        attn_out, weights = attn(x)
        normed = norm(attn_out)
        loss = normed.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ======================================================================
# Residual connection pattern
# ======================================================================


class TestResidualPattern:
    """Test the common residual + norm pattern."""

    def test_residual_connection(self):
        """x + attention(x) → norm should produce valid output."""
        cfg = Cfg()
        attn = LorentzMultiHeadAttention(cfg)
        norm = MinkowskiLayerNorm(cfg.d_model)

        mask = torch.randint(0, 2, (cfg.d_model,)).bool()
        attn.set_timelike_mask(mask)
        norm.set_timelike_mask(mask)

        x = torch.randn(2, 8, cfg.d_model, requires_grad=True)
        attn_out, _ = attn(x)
        residual = x + attn_out
        normed = norm(residual)

        assert normed.shape == x.shape
        assert torch.isfinite(normed).all()

        normed.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ======================================================================
# compute_dt2_info with attention output
# ======================================================================


class TestDt2InfoIntegration:
    """Test compute_dt2_info with actual attention weights."""

    def test_dt2_from_standard_attention(self):
        """compute_dt2_info on standard attention output."""
        cfg = Cfg(lorentz_alpha=0.0)
        attn = LorentzMultiHeadAttention(cfg)
        x = torch.randn(2, 8, cfg.d_model)
        _, weights = attn(x)
        dt2 = compute_dt2_info(weights)

        assert dt2.shape == torch.Size([])
        assert dt2.item() >= 0
        assert torch.isfinite(dt2)

    def test_dt2_from_lorentz_attention(self):
        """compute_dt2_info on Lorentz-corrected attention output."""
        cfg = Cfg(lorentz_alpha=0.25)
        attn = LorentzMultiHeadAttention(cfg)
        mask = torch.randint(0, 2, (cfg.d_model,)).bool()
        attn.set_timelike_mask(mask)

        x = torch.randn(2, 8, cfg.d_model)
        _, weights = attn(x)
        dt2 = compute_dt2_info(weights)

        assert dt2.shape == torch.Size([])
        assert dt2.item() >= 0
        assert torch.isfinite(dt2)

    def test_dt2_with_causal_mask(self):
        """compute_dt2_info after causal masking."""
        cfg = Cfg()
        attn = LorentzMultiHeadAttention(cfg)
        L = 8
        x = torch.randn(1, L, cfg.d_model)
        causal = torch.triu(
            torch.full((L, L), float("-inf")), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        _, weights = attn(x, attention_mask=causal)
        dt2 = compute_dt2_info(weights)

        assert torch.isfinite(dt2)
        assert dt2.item() >= 0


# ======================================================================
# hutchinson_diag_hessian integration with attention
# ======================================================================


class TestHutchinsonIntegration:
    """Test hutchinson_diag_hessian with LorentzMultiHeadAttention."""

    def test_hessian_of_attention_qproj(self):
        """Hutchinson on q_proj weight should produce finite estimates."""
        cfg = Cfg()
        attn = LorentzMultiHeadAttention(cfg)
        x = torch.randn(1, 4, cfg.d_model)

        def loss_fn():
            out, _ = attn(x)
            return out.sum()

        G = hutchinson_diag_hessian(loss_fn, attn.q_proj.weight, n_samples=5)
        assert G.shape == attn.q_proj.weight.shape
        assert torch.isfinite(G).all()

    def test_hessian_identifies_nonzero_curvature(self):
        """Hessian diagonal should be non-zero for a meaningful loss."""
        cfg = Cfg()
        attn = LorentzMultiHeadAttention(cfg)
        x = torch.randn(1, 4, cfg.d_model)

        def loss_fn():
            out, _ = attn(x)
            return (out ** 2).sum()

        G = hutchinson_diag_hessian(
            loss_fn, attn.q_proj.weight, n_samples=10
        )
        assert (G != 0).any()


# ======================================================================
# Multi-layer pipeline
# ======================================================================


class TestMultiLayerPipeline:
    """Test stacking multiple attention + norm layers."""

    def test_two_layer_stack(self):
        """Two stacked Lorentz layers should produce valid output."""
        cfg = Cfg()
        mask = torch.randint(0, 2, (cfg.d_model,)).bool()

        attn1 = LorentzMultiHeadAttention(cfg)
        norm1 = MinkowskiLayerNorm(cfg.d_model)
        attn2 = LorentzMultiHeadAttention(cfg)
        norm2 = MinkowskiLayerNorm(cfg.d_model)

        for m in [attn1, norm1, attn2, norm2]:
            m.set_timelike_mask(mask)

        x = torch.randn(2, 8, cfg.d_model, requires_grad=True)

        # Layer 1
        h, _ = attn1(x)
        h = norm1(x + h)

        # Layer 2
        h2, _ = attn2(h)
        out = norm2(h + h2)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_three_layer_no_nan(self):
        """Three stacked layers should not produce NaN or Inf."""
        cfg = Cfg()

        layers = nn.ModuleList()
        norms = nn.ModuleList()
        for _ in range(3):
            layers.append(LorentzMultiHeadAttention(cfg))
            norms.append(MinkowskiLayerNormImproved(cfg.d_model))

        mask = torch.randint(0, 2, (cfg.d_model,)).bool()
        for layer in layers:
            layer.set_timelike_mask(mask)
        for norm in norms:
            norm.set_timelike_mask(mask)

        x = torch.randn(1, 6, cfg.d_model)
        h = x
        for layer, norm in zip(layers, norms):
            attn_out, _ = layer(h)
            h = norm(h + attn_out)

        assert torch.isfinite(h).all()


# ======================================================================
# Import verification
# ======================================================================


class TestImports:
    """Verify that all public exports are accessible."""

    def test_top_level_exports(self):
        """All expected symbols should be importable from the top level."""
        import lorentz_transformer

        assert hasattr(lorentz_transformer, "LorentzMultiHeadAttention")
        assert hasattr(lorentz_transformer, "compute_dt2_info")
        assert hasattr(lorentz_transformer, "hutchinson_diag_hessian")
        assert hasattr(lorentz_transformer, "MinkowskiLayerNorm")
        assert hasattr(lorentz_transformer, "MinkowskiLayerNormImproved")
        assert hasattr(lorentz_transformer, "MinkowskiLayerNormOptimized")
        assert hasattr(lorentz_transformer, "MinkowskiLayerNormStable")

    def test_core_exports(self):
        """All expected symbols should be importable from core."""
        import lorentz_transformer.core

        assert hasattr(lorentz_transformer.core, "LorentzMultiHeadAttention")
        assert hasattr(lorentz_transformer.core, "compute_dt2_info")
        assert hasattr(lorentz_transformer.core, "hutchinson_diag_hessian")
        assert hasattr(lorentz_transformer.core, "MinkowskiLayerNorm")
        assert hasattr(lorentz_transformer.core, "MinkowskiLayerNormImproved")
        assert hasattr(lorentz_transformer.core, "MinkowskiLayerNormOptimized")
        assert hasattr(lorentz_transformer.core, "MinkowskiLayerNormStable")
