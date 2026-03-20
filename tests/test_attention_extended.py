"""
tests/test_attention_extended.py

Extended test coverage for attention.py — covers guard clauses, edge cases,
and behavioral paths not exercised by the base test suite.
"""

import math
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from lorentz_transformer.core.attention import (
    LorentzMultiHeadAttention,
    compute_dt2_info,
    hutchinson_diag_hessian,
)


@dataclass
class Cfg:
    d_model: int = 64
    n_heads: int = 4
    lorentz_alpha: float = 0.25
    dropout: float = 0.0


# ======================================================================
# hutchinson_diag_hessian – guard-clause and edge-case coverage
# ======================================================================


class TestHutchinsonEdgeCases:
    """Cover the None-gradient continue paths and other edge cases."""

    def test_basic_gradient_produces_finite_result(self):
        """Basic loss with a contributing param should produce finite output."""
        dummy = nn.Parameter(torch.randn(4))

        G = hutchinson_diag_hessian(
            lambda: (dummy ** 2).sum(), dummy, n_samples=1
        )
        assert G.shape == dummy.shape
        assert torch.isfinite(G).all()

    def test_single_sample(self):
        """n_samples=1 should still produce a finite result."""
        param = nn.Parameter(torch.randn(8))
        G = hutchinson_diag_hessian(
            lambda: (param ** 2).sum(), param, n_samples=1
        )
        assert G.shape == param.shape
        assert torch.isfinite(G).all()
        assert (G != 0).any()

    def test_large_n_samples_improves_estimate(self):
        """More samples should converge closer to the true diagonal."""
        param = nn.Parameter(torch.randn(16))

        def loss_fn():
            return (param ** 2).sum()

        G_few = hutchinson_diag_hessian(loss_fn, param, n_samples=5)
        G_many = hutchinson_diag_hessian(loss_fn, param, n_samples=200)

        # True Hessian diagonal of ||x||^2 is 2.0 everywhere.
        err_few = (G_few - 2.0).abs().mean().item()
        err_many = (G_many - 2.0).abs().mean().item()
        assert err_many < err_few or err_many < 0.5

    def test_with_1d_parameter(self):
        """Works with a 1-d parameter (scalar-like)."""
        param = nn.Parameter(torch.tensor([3.0]))
        G = hutchinson_diag_hessian(
            lambda: (param ** 2).sum(), param, n_samples=50
        )
        assert G.shape == torch.Size([1])
        assert abs(G.item() - 2.0) < 1.0

    def test_cubic_loss(self):
        """For f(x)=x^3, Hessian diag = 6x. Should match on average."""
        param = nn.Parameter(torch.tensor([2.0]))
        G = hutchinson_diag_hessian(
            lambda: (param ** 3).sum(), param, n_samples=100
        )
        # True diag = 6 * 2.0 = 12.0
        assert abs(G.item() - 12.0) < 4.0

    def test_output_is_detached(self):
        """Returned tensor should not require grad."""
        param = nn.Parameter(torch.randn(8))
        G = hutchinson_diag_hessian(
            lambda: (param ** 2).sum(), param, n_samples=5
        )
        assert not G.requires_grad

    def test_nan_in_hessian_is_cleaned(self):
        """NaN values produced during Hv should be cleaned to 0."""
        param = nn.Parameter(torch.randn(4))

        def loss_fn():
            # sqrt can produce nan gradients near zero
            return (param.abs() + 1e-30).sqrt().sum()

        G = hutchinson_diag_hessian(loss_fn, param, n_samples=10)
        assert torch.isfinite(G).all()


# ======================================================================
# compute_dt2_info – additional edge cases
# ======================================================================


class TestComputeDt2InfoExtended:
    """Extended tests for the information-time metric."""

    def test_single_head_single_batch(self):
        """Works with minimal dimensions B=1, H=1."""
        attn_w = torch.rand(1, 1, 8, 8)
        result = compute_dt2_info(attn_w)
        assert result.shape == torch.Size([])
        assert torch.isfinite(result)

    def test_single_position(self):
        """Works with sequence length L=1."""
        attn_w = torch.ones(1, 2, 1, 1)
        result = compute_dt2_info(attn_w)
        assert torch.isfinite(result)

    def test_all_zeros(self):
        """All-zero attention weights should not produce NaN."""
        attn_w = torch.zeros(1, 2, 4, 4)
        result = compute_dt2_info(attn_w)
        assert torch.isfinite(result)

    def test_positive_random_weights(self):
        """Positive random weights (realistic case) produce finite output."""
        attn_w = torch.rand(1, 2, 4, 4).abs() + 1e-6
        result = compute_dt2_info(attn_w)
        assert torch.isfinite(result)
        assert result.item() >= 0

    def test_large_sequence_length(self):
        """Should work with larger sequence lengths."""
        attn_w = torch.rand(1, 4, 512, 512)
        result = compute_dt2_info(attn_w)
        assert result.shape == torch.Size([])
        assert torch.isfinite(result)

    def test_multiple_heads_averaged(self):
        """Result should be independent of head permutation."""
        attn_w = torch.rand(1, 4, 8, 8)
        # Permuting heads shouldn't change result since we average over heads
        attn_w_perm = attn_w[:, [2, 0, 3, 1], :, :]
        r1 = compute_dt2_info(attn_w)
        r2 = compute_dt2_info(attn_w_perm)
        assert torch.allclose(r1, r2, atol=1e-6)


# ======================================================================
# LorentzMultiHeadAttention – additional behavioral tests
# ======================================================================


class TestLorentzAttentionExtended:
    """Extended tests for LorentzMultiHeadAttention."""

    @pytest.fixture
    def cfg(self):
        return Cfg()

    @pytest.fixture
    def module(self, cfg):
        return LorentzMultiHeadAttention(cfg)

    # --- set_timelike_mask ---

    def test_set_timelike_mask_from_float_tensor(self, cfg):
        """Float tensor mask should be converted to bool."""
        attn = LorentzMultiHeadAttention(cfg)
        float_mask = torch.zeros(cfg.d_model)
        float_mask[:16] = 1.0
        attn.set_timelike_mask(float_mask)

        assert attn._has_mask
        assert attn.timelike_mask.dtype == torch.bool
        assert attn.timelike_mask[:16].all()
        assert not attn.timelike_mask[16:].any()

    def test_set_timelike_mask_twice_overrides(self, cfg):
        """Calling set_timelike_mask a second time should replace the mask."""
        attn = LorentzMultiHeadAttention(cfg)
        mask_a = torch.ones(cfg.d_model, dtype=torch.bool)
        mask_b = torch.zeros(cfg.d_model, dtype=torch.bool)
        attn.set_timelike_mask(mask_a)
        assert attn._has_mask
        attn.set_timelike_mask(mask_b)
        assert not attn._has_mask

    # --- Minkowski correction details ---

    def test_minkowski_correction_with_partial_mask(self, cfg):
        """Partial mask (some True, some False) should produce valid output."""
        attn = LorentzMultiHeadAttention(cfg)
        mask = torch.zeros(cfg.d_model, dtype=torch.bool)
        mask[:cfg.d_model // 4] = True  # 25% timelike
        attn.set_timelike_mask(mask)

        x = torch.randn(2, 8, cfg.d_model)
        out, w = attn(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        assert torch.isfinite(w).all()

    def test_minkowski_correction_with_causal_mask_combined(self, cfg):
        """Minkowski correction should work together with a causal mask."""
        attn = LorentzMultiHeadAttention(cfg)
        mask = torch.randint(0, 2, (cfg.d_model,)).bool()
        attn.set_timelike_mask(mask)

        L = 12
        x = torch.randn(1, L, cfg.d_model)
        causal = torch.triu(
            torch.full((L, L), float("-inf")), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        out, w = attn(x, attention_mask=causal)

        # Future positions should still have zero weight
        for i in range(L):
            for j in range(i + 1, L):
                assert w[0, 0, i, j].item() < 1e-6

        assert torch.isfinite(out).all()

    def test_last_intervals_raw_differs_from_masked(self, cfg):
        """Raw intervals (pre-mask) should differ from post-mask intervals."""
        attn = LorentzMultiHeadAttention(cfg)
        L = 8
        x = torch.randn(1, L, cfg.d_model)
        causal = torch.triu(
            torch.full((L, L), float("-inf")), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        attn(x, attention_mask=causal)

        assert attn.last_intervals_raw is not None
        assert attn.last_intervals is not None
        # The raw scores shouldn't contain -inf, but masked ones should
        assert not torch.isinf(attn.last_intervals_raw).any()
        assert torch.isinf(attn.last_intervals).any()

    def test_diagnostic_intervals_saved_with_minkowski(self, cfg):
        """Diagnostic tensors should be saved even with Minkowski correction."""
        attn = LorentzMultiHeadAttention(cfg)
        mask = torch.ones(cfg.d_model, dtype=torch.bool)
        attn.set_timelike_mask(mask)

        x = torch.randn(2, 6, cfg.d_model)
        attn(x)

        assert attn.last_intervals is not None
        assert attn.last_intervals_raw is not None
        assert attn.last_intervals.shape == (2, cfg.n_heads, 6, 6)

    # --- Varying alpha ---

    def test_higher_alpha_produces_larger_deviation(self):
        """Higher alpha should produce a bigger deviation from standard attn."""
        x = torch.randn(1, 8, 64)
        mask = torch.ones(64, dtype=torch.bool)

        cfg_low = Cfg(lorentz_alpha=0.1)
        cfg_high = Cfg(lorentz_alpha=0.9)

        attn_low = LorentzMultiHeadAttention(cfg_low)
        attn_high = LorentzMultiHeadAttention(cfg_high)
        # Share weights
        attn_high.load_state_dict(attn_low.state_dict())
        attn_low.set_timelike_mask(mask)
        attn_high.set_timelike_mask(mask)

        # Standard baseline (alpha=0)
        cfg_std = Cfg(lorentz_alpha=0.0)
        attn_std = LorentzMultiHeadAttention(cfg_std)
        attn_std.load_state_dict(attn_low.state_dict())
        attn_std.eval()
        attn_low.eval()
        attn_high.eval()

        out_std, _ = attn_std(x)
        out_low, _ = attn_low(x)
        out_high, _ = attn_high(x)

        diff_low = (out_low - out_std).abs().mean().item()
        diff_high = (out_high - out_std).abs().mean().item()
        assert diff_high > diff_low

    # --- Dropout ---

    def test_dropout_active_in_train_mode(self):
        """With dropout > 0, train mode should introduce variance."""
        cfg = Cfg(dropout=0.5)
        attn = LorentzMultiHeadAttention(cfg)
        attn.train()
        x = torch.randn(2, 8, 64)

        torch.manual_seed(0)
        out1, _ = attn(x)
        torch.manual_seed(1)
        out2, _ = attn(x)

        # Outputs should differ due to dropout
        assert not torch.allclose(out1, out2, atol=1e-5)

    def test_dropout_inactive_in_eval_mode(self):
        """With dropout > 0, eval mode should be deterministic."""
        cfg = Cfg(dropout=0.5)
        attn = LorentzMultiHeadAttention(cfg)
        attn.eval()
        x = torch.randn(2, 8, 64)

        out1, _ = attn(x)
        out2, _ = attn(x)

        assert torch.allclose(out1, out2, atol=1e-6)

    # --- Config edge cases ---

    def test_single_head(self):
        """Should work with a single attention head."""
        cfg = Cfg(d_model=16, n_heads=1)
        attn = LorentzMultiHeadAttention(cfg)
        x = torch.randn(1, 4, 16)
        out, w = attn(x)
        assert out.shape == (1, 4, 16)
        assert w.shape == (1, 1, 4, 4)

    def test_many_heads(self):
        """Should work when each head has dimension 1."""
        cfg = Cfg(d_model=64, n_heads=64)
        attn = LorentzMultiHeadAttention(cfg)
        x = torch.randn(1, 4, 64)
        out, w = attn(x)
        assert out.shape == (1, 4, 64)
        assert w.shape == (1, 64, 4, 4)

    def test_default_alpha_and_dropout(self):
        """Default config values should be used when not provided."""

        @dataclass
        class MinimalCfg:
            d_model: int = 32
            n_heads: int = 4

        attn = LorentzMultiHeadAttention(MinimalCfg())
        assert attn.alpha == 0.25
        assert attn.drop.p == 0.0

    # --- Gradient with Minkowski correction ---

    def test_gradient_flow_with_minkowski_correction(self, cfg):
        """Gradients should flow through the Minkowski correction path."""
        attn = LorentzMultiHeadAttention(cfg)
        mask = torch.randint(0, 2, (cfg.d_model,)).bool()
        attn.set_timelike_mask(mask)

        x = torch.randn(1, 6, cfg.d_model, requires_grad=True)
        out, _ = attn(x)
        out.sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_flow_through_all_projections(self, cfg):
        """All projection weights should receive gradients."""
        attn = LorentzMultiHeadAttention(cfg)
        mask = torch.randint(0, 2, (cfg.d_model,)).bool()
        attn.set_timelike_mask(mask)

        x = torch.randn(1, 6, cfg.d_model)
        out, _ = attn(x)
        out.sum().backward()

        for name, param in attn.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite grad"

    # --- Numerical stability ---

    def test_stability_with_very_small_values(self, cfg):
        """Very small inputs should not cause NaN."""
        attn = LorentzMultiHeadAttention(cfg)
        mask = torch.randint(0, 2, (cfg.d_model,)).bool()
        attn.set_timelike_mask(mask)
        x = torch.randn(1, 4, cfg.d_model) * 1e-6
        out, w = attn(x)
        assert torch.isfinite(out).all()
        assert torch.isfinite(w).all()

    def test_stability_with_large_values_and_mask(self, cfg):
        """Large values with Minkowski correction should remain stable."""
        attn = LorentzMultiHeadAttention(cfg)
        mask = torch.randint(0, 2, (cfg.d_model,)).bool()
        attn.set_timelike_mask(mask)
        x = torch.randn(1, 4, cfg.d_model) * 100
        out, w = attn(x)
        assert torch.isfinite(out).all()
        assert torch.isfinite(w).all()

    # --- Batch size 1 ---

    def test_batch_size_one(self, cfg):
        """Batch size 1 should work correctly."""
        attn = LorentzMultiHeadAttention(cfg)
        x = torch.randn(1, 4, cfg.d_model)
        out, w = attn(x)
        assert out.shape == (1, 4, cfg.d_model)

    # --- Sequence length 1 ---

    def test_sequence_length_one(self, cfg):
        """Sequence length 1 should work correctly."""
        attn = LorentzMultiHeadAttention(cfg)
        x = torch.randn(2, 1, cfg.d_model)
        out, w = attn(x)
        assert out.shape == (2, 1, cfg.d_model)
        assert w.shape == (2, cfg.n_heads, 1, 1)
