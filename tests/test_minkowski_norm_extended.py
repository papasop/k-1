"""
tests/test_minkowski_norm_extended.py

Extended test coverage for minkowski_norm.py — covers all norm variants,
base class helper methods, affine/mean-shift combinations, and edge cases.
"""

import math

import pytest
import torch
import torch.nn as nn

from lorentz_transformer.core.minkowski_norm import (
    MinkowskiLayerNorm,
    MinkowskiLayerNormImproved,
    MinkowskiLayerNormOptimized,
    MinkowskiLayerNormStable,
    _BaseMinkowskiLayerNorm,
)


# ======================================================================
# _BaseMinkowskiLayerNorm — internal helper coverage
# ======================================================================


class TestBaseHelpers:
    """Directly exercise the base-class helpers."""

    def test_l2_norm_sq(self):
        """_l2_norm_sq should return sum of squares per vector."""
        norm = MinkowskiLayerNormOptimized(d_model=3, elementwise_affine=False)
        x = torch.tensor([[3.0, 4.0, 0.0]])
        result = norm._l2_norm_sq(x)
        assert torch.allclose(result, torch.tensor([[25.0]]))

    def test_minkowski_norm_sq_no_mask(self):
        """Without a timelike mask all dimensions are spacelike → equals L2."""
        norm = MinkowskiLayerNormOptimized(d_model=3, elementwise_affine=False)
        x = torch.tensor([[3.0, 4.0, 0.0]])
        result = norm._minkowski_norm_sq(x)
        # No timelike → spacelike = 25, timelike = 0 → |25 - 0| = 25
        assert torch.allclose(result, torch.tensor([[25.0]]))

    def test_minkowski_norm_sq_with_mask(self):
        """With mask, timelike components are subtracted."""
        norm = MinkowskiLayerNormOptimized(d_model=3, elementwise_affine=False)
        norm.set_timelike_mask([True, False, False])
        x = torch.tensor([[3.0, 4.0, 0.0]])
        result = norm._minkowski_norm_sq(x)
        # spacelike = 16 + 0 = 16, timelike = 9 → |16 - 9| = 7
        assert torch.allclose(result, torch.tensor([[7.0]]))

    def test_apply_affine_identity(self):
        """With default weight=1 and bias=0, affine is identity."""
        norm = MinkowskiLayerNormOptimized(d_model=3)
        x = torch.tensor([[1.0, 2.0, 3.0]])
        result = norm._apply_affine(x)
        assert torch.allclose(result, x)

    def test_apply_affine_custom(self):
        """Custom weight and bias should be applied correctly."""
        norm = MinkowskiLayerNormOptimized(d_model=2)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([2.0, 0.5]))
            norm.bias.copy_(torch.tensor([1.0, -1.0]))
        x = torch.tensor([[3.0, 4.0]])
        result = norm._apply_affine(x)
        expected = torch.tensor([[7.0, 1.0]])  # 3*2+1, 4*0.5-1
        assert torch.allclose(result, expected)

    def test_validate_input_wrong_dim_raises(self):
        """Input with wrong last dimension should raise ValueError."""
        norm = MinkowskiLayerNormOptimized(d_model=4)
        with pytest.raises(ValueError, match="Expected last dimension 4"):
            norm._validate_input(torch.randn(2, 3))

    def test_validate_input_flattens(self):
        """Multi-dimensional input should be flattened to 2D."""
        norm = MinkowskiLayerNormOptimized(d_model=4, use_mean_shift=False)
        x = torch.randn(2, 3, 4)
        x_flat, original_shape = norm._validate_input(x)
        assert x_flat.shape == (6, 4)
        assert original_shape == (2, 3, 4)

    def test_validate_input_with_mean_shift(self):
        """Mean shift should center each vector."""
        norm = MinkowskiLayerNormOptimized(d_model=2, use_mean_shift=True)
        x = torch.tensor([[2.0, 4.0]])
        x_flat, _ = norm._validate_input(x)
        expected = torch.tensor([[-1.0, 1.0]])
        assert torch.allclose(x_flat, expected)

    def test_validate_mask_wrong_shape_raises(self):
        """Mask with wrong shape should raise ValueError."""
        norm = MinkowskiLayerNormOptimized(d_model=4)
        with pytest.raises(ValueError, match="Expected timelike mask shape"):
            norm._validate_mask([True, False])

    def test_validate_mask_2d_raises(self):
        """2D mask should raise ValueError."""
        norm = MinkowskiLayerNormOptimized(d_model=4)
        with pytest.raises(ValueError):
            norm._validate_mask(torch.zeros(2, 4, dtype=torch.bool))

    def test_validate_mask_correct(self):
        """Correctly shaped mask should pass validation."""
        norm = MinkowskiLayerNormOptimized(d_model=4)
        result = norm._validate_mask([True, False, True, False])
        assert result.shape == (4,)
        assert result.dtype == torch.bool


# ======================================================================
# elementwise_affine=False — all variants
# ======================================================================


class TestElementwiseAffineOff:
    """Verify elementwise_affine=False across all norm variants."""

    @pytest.mark.parametrize(
        "cls",
        [MinkowskiLayerNormOptimized, MinkowskiLayerNormImproved],
    )
    def test_no_learnable_params(self, cls):
        """Weight and bias should be buffers, not parameters."""
        norm = cls(d_model=4, elementwise_affine=False)
        param_names = [n for n, _ in norm.named_parameters()]
        assert "weight" not in param_names
        assert "bias" not in param_names

    @pytest.mark.parametrize(
        "cls",
        [MinkowskiLayerNormOptimized, MinkowskiLayerNormImproved],
    )
    def test_forward_produces_valid_output(self, cls):
        """Forward pass should work without learnable affine parameters."""
        norm = cls(d_model=4, elementwise_affine=False)
        x = torch.randn(2, 3, 4)
        out = norm(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_stable_no_learnable_params(self):
        norm = MinkowskiLayerNormStable(d_model=4, elementwise_affine=False)
        param_names = [n for n, _ in norm.named_parameters()]
        assert "weight" not in param_names
        assert "bias" not in param_names


# ======================================================================
# use_mean_shift=True — all variants
# ======================================================================


class TestMeanShiftAllVariants:
    """Test use_mean_shift=True across all norm variants."""

    def test_optimized_mean_shift(self):
        """Optimized variant with mean shift should center then normalize."""
        eps = 1e-5
        norm = MinkowskiLayerNormOptimized(
            d_model=2, eps=eps, elementwise_affine=False, use_mean_shift=True
        )
        x = torch.tensor([[2.0, 4.0]])
        out = norm(x)
        centered = torch.tensor([[-1.0, 1.0]])
        expected = centered / math.sqrt(2.0 + eps)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_stable_mean_shift(self):
        """Stable variant with mean shift should center then normalize."""
        eps = 1e-5
        norm = MinkowskiLayerNormStable(
            d_model=2, eps=eps, elementwise_affine=False, use_mean_shift=True
        )
        x = torch.tensor([[2.0, 4.0]])
        out = norm(x)
        centered = torch.tensor([[-1.0, 1.0]])
        expected = centered / math.sqrt(2.0 + eps)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_improved_mean_shift(self):
        """Improved variant with mean shift should center then normalize."""
        eps = 1e-5
        norm = MinkowskiLayerNormImproved(
            d_model=2, eps=eps, elementwise_affine=False, use_mean_shift=True
        )
        x = torch.tensor([[2.0, 4.0]])
        out = norm(x)
        centered = torch.tensor([[-1.0, 1.0]])
        expected = centered / math.sqrt(2.0 + eps)
        assert torch.allclose(out, expected, atol=1e-6)


# ======================================================================
# MinkowskiLayerNormOptimized — extended
# ======================================================================


class TestOptimizedExtended:
    """Additional tests for the Optimized variant."""

    def test_zero_input(self):
        """Zero input should produce zero output (L2 norm → eps only)."""
        norm = MinkowskiLayerNormOptimized(
            d_model=4, elementwise_affine=False
        )
        x = torch.zeros(1, 4)
        out = norm(x)
        assert torch.allclose(out, torch.zeros(1, 4))

    def test_large_input(self):
        """Large input should be normalized to unit-ish range."""
        norm = MinkowskiLayerNormOptimized(
            d_model=4, elementwise_affine=False
        )
        x = torch.randn(2, 3, 4) * 1000
        out = norm(x)
        assert torch.isfinite(out).all()
        # Each vector should be roughly unit length
        norms = (out ** 2).sum(dim=-1).sqrt()
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

    def test_gradient_flow(self):
        """Gradients should flow through the optimized path."""
        norm = MinkowskiLayerNormOptimized(d_model=4)
        x = torch.randn(2, 3, 4, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ======================================================================
# MinkowskiLayerNormStable — extended
# ======================================================================


class TestStableExtended:
    """Additional tests for the Stable variant."""

    def test_valid_minkowski_path_used(self):
        """When intervals are valid, Minkowski norm should be used."""
        eps = 1e-5
        # Create input where Minkowski interval is clearly positive
        norm = MinkowskiLayerNormStable(
            d_model=2,
            eps=eps,
            elementwise_affine=False,
            minkowski_fallback_threshold=0.5,
        )
        norm.set_timelike_mask([False, True])
        # x = [1.0, 10.0] → spacelike=1, timelike=100 → |1-100|=99
        x = torch.tensor([[1.0, 10.0]])
        out = norm(x)
        expected = x / math.sqrt(99.0 + eps)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_fallback_threshold_zero_always_falls_back(self):
        """With threshold=0.0 any zero-interval vector triggers fallback."""
        eps = 1e-5
        norm = MinkowskiLayerNormStable(
            d_model=2,
            eps=eps,
            elementwise_affine=False,
            minkowski_fallback_threshold=0.0,
        )
        norm.set_timelike_mask([True, False])
        # [3, 3] → |9-9|=0 ≤ eps → invalid. Ratio 1.0 > threshold 0.0 → L2
        x = torch.tensor([[3.0, 3.0]])
        out = norm(x)
        expected = x / math.sqrt(18.0 + eps)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_fallback_threshold_one_never_falls_back(self):
        """With threshold=1.0, even all-invalid should use Minkowski."""
        eps = 1e-5
        norm = MinkowskiLayerNormStable(
            d_model=2,
            eps=eps,
            elementwise_affine=False,
            minkowski_fallback_threshold=1.0,
        )
        norm.set_timelike_mask([True, False])
        # [3, 3] → |9-9|=0 → interval ≈ eps. 100% invalid but threshold 1.0
        x = torch.tensor([[3.0, 3.0]])
        out = norm(x)
        # Should use sqrt(clamp(0, min=eps)) = sqrt(eps)
        expected = x / math.sqrt(eps)
        assert torch.allclose(out, expected, atol=1e-3)

    def test_gradient_flow_minkowski_path(self):
        """Gradients through the Minkowski path should be finite."""
        norm = MinkowskiLayerNormStable(
            d_model=4, minkowski_fallback_threshold=1.0
        )
        norm.set_timelike_mask([False, False, True, True])
        x = torch.randn(2, 3, 4, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_no_mask_uses_l2(self):
        """Without a mask, Stable should always use L2."""
        eps = 1e-5
        norm = MinkowskiLayerNormStable(
            d_model=2, eps=eps, elementwise_affine=False
        )
        x = torch.tensor([[3.0, 4.0]])
        out = norm(x)
        expected = x / math.sqrt(25.0 + eps)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_batch_consistency(self):
        """Different batch elements should be normalized independently."""
        norm = MinkowskiLayerNormStable(
            d_model=4, elementwise_affine=False
        )
        norm.set_timelike_mask([True, False, False, False])
        x = torch.randn(4, 4)
        out = norm(x)
        for i in range(4):
            single = norm(x[i:i + 1])
            assert torch.allclose(out[i:i + 1], single, atol=1e-6)


# ======================================================================
# MinkowskiLayerNormImproved — extended
# ======================================================================


class TestImprovedExtended:
    """Additional tests for the Improved (recommended) variant."""

    def test_valid_minkowski_interval_used(self):
        """When interval > eps, Minkowski norm should be used."""
        eps = 1e-5
        norm = MinkowskiLayerNormImproved(
            d_model=2, eps=eps, elementwise_affine=False
        )
        norm.set_timelike_mask([False, True])
        # x = [1.0, 10.0] → spacelike=1, timelike=100 → |1-100|=99 > eps
        x = torch.tensor([[1.0, 10.0]])
        out = norm(x)
        expected = x / math.sqrt(99.0 + eps)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_mixed_valid_and_invalid_intervals(self):
        """Per-element fallback: some use Minkowski, some use L2."""
        eps = 1e-5
        norm = MinkowskiLayerNormImproved(
            d_model=2, eps=eps, elementwise_affine=False
        )
        norm.set_timelike_mask([True, False])
        # Row 0: [3,3] → |9-9|=0 ≤ eps → falls back to L2 = 18
        # Row 1: [1,10] → |1-100|=99 > eps → uses Minkowski
        x = torch.tensor([[3.0, 3.0], [1.0, 10.0]])
        out = norm(x)

        expected_0 = x[0] / math.sqrt(18.0 + eps)
        expected_1 = x[1] / math.sqrt(99.0 + eps)

        assert torch.allclose(out[0], expected_0, atol=1e-5)
        assert torch.allclose(out[1], expected_1, atol=1e-5)

    def test_all_timelike_mask(self):
        """All-True mask: interval = |0 - sum_sq| = sum_sq → same as L2."""
        eps = 1e-5
        norm = MinkowskiLayerNormImproved(
            d_model=2, eps=eps, elementwise_affine=False
        )
        norm.set_timelike_mask([True, True])
        x = torch.tensor([[3.0, 4.0]])
        # spacelike=0, timelike=25, |0-25|=25 = L2
        out = norm(x)
        expected = x / math.sqrt(25.0 + eps)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_all_spacelike_mask(self):
        """All-False mask: interval = |sum_sq - 0| = sum_sq → same as L2."""
        eps = 1e-5
        norm = MinkowskiLayerNormImproved(
            d_model=2, eps=eps, elementwise_affine=False
        )
        norm.set_timelike_mask([False, False])
        x = torch.tensor([[3.0, 4.0]])
        out = norm(x)
        # All-False mask → _has_mask is False → uses L2
        expected = x / math.sqrt(25.0 + eps)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_recommended_variant_flag(self):
        """recommended_variant should be True for Improved."""
        assert MinkowskiLayerNormImproved.recommended_variant is True
        assert MinkowskiLayerNormOptimized.recommended_variant is False
        assert MinkowskiLayerNormStable.recommended_variant is False

    def test_backward_compat_alias(self):
        """MinkowskiLayerNorm should be MinkowskiLayerNormImproved."""
        assert MinkowskiLayerNorm is MinkowskiLayerNormImproved

    def test_gradient_flow_minkowski_path(self):
        """Gradients through the Minkowski path should be finite."""
        norm = MinkowskiLayerNormImproved(d_model=4)
        norm.set_timelike_mask([False, False, True, True])
        x = torch.randn(2, 3, 4, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_4d_input_shape(self):
        """Should handle 4D input (e.g., image-like)."""
        norm = MinkowskiLayerNormImproved(d_model=8, elementwise_affine=False)
        x = torch.randn(2, 3, 4, 8)
        out = norm(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_weight_grad_updates(self):
        """Affine weight and bias should have gradients after backward."""
        norm = MinkowskiLayerNormImproved(d_model=4)
        norm.set_timelike_mask([True, False, True, False])
        x = torch.randn(2, 3, 4)
        out = norm(x)
        out.sum().backward()
        assert norm.weight.grad is not None
        assert norm.bias.grad is not None
        assert torch.isfinite(norm.weight.grad).all()
        assert torch.isfinite(norm.bias.grad).all()


# ======================================================================
# Cross-variant consistency
# ======================================================================


class TestCrossVariantConsistency:
    """Verify relationships between different norm variants."""

    def test_all_variants_match_on_l2_without_mask(self):
        """Without a mask, all three should produce the same output."""
        d = 8
        x = torch.randn(2, 3, d)
        out_opt = MinkowskiLayerNormOptimized(
            d, elementwise_affine=False
        )(x)
        out_stb = MinkowskiLayerNormStable(
            d, elementwise_affine=False
        )(x)
        out_imp = MinkowskiLayerNormImproved(
            d, elementwise_affine=False
        )(x)
        assert torch.allclose(out_opt, out_stb, atol=1e-6)
        assert torch.allclose(out_opt, out_imp, atol=1e-6)

    def test_optimized_ignores_mask(self):
        """Optimized should produce the same output with or without mask."""
        d = 8
        x = torch.randn(2, 3, d)
        norm = MinkowskiLayerNormOptimized(d, elementwise_affine=False)
        out_no_mask = norm(x)
        norm.set_timelike_mask([True] * 4 + [False] * 4)
        out_with_mask = norm(x)
        assert torch.allclose(out_no_mask, out_with_mask, atol=1e-6)

    def test_improved_and_stable_differ_on_boundary(self):
        """On a boundary case, Improved and Stable may differ due to
        per-element vs. batch-level fallback strategies."""
        d = 2
        # Mix of degenerate and valid rows
        x = torch.tensor([[3.0, 3.0], [1.0, 5.0]])
        mask_list = [True, False]

        norm_stb = MinkowskiLayerNormStable(
            d,
            elementwise_affine=False,
            minkowski_fallback_threshold=0.4,
        )
        norm_imp = MinkowskiLayerNormImproved(d, elementwise_affine=False)

        norm_stb.set_timelike_mask(mask_list)
        norm_imp.set_timelike_mask(mask_list)

        out_stb = norm_stb(x)
        out_imp = norm_imp(x)

        # They may differ because Stable falls back batch-wide, while
        # Improved falls back per-element.
        # At least one row should differ.
        assert not torch.allclose(out_stb, out_imp, atol=1e-6)


# ======================================================================
# Edge cases
# ======================================================================


class TestNormEdgeCases:
    """Edge cases that apply broadly."""

    def test_single_element(self):
        """Input with shape (d_model,) should work."""
        norm = MinkowskiLayerNormImproved(d_model=4, elementwise_affine=False)
        x = torch.randn(4)
        out = norm(x)
        assert out.shape == (4,)
        assert torch.isfinite(out).all()

    def test_d_model_one(self):
        """d_model=1 should work."""
        for cls in [
            MinkowskiLayerNormOptimized,
            MinkowskiLayerNormStable,
            MinkowskiLayerNormImproved,
        ]:
            norm = cls(d_model=1, elementwise_affine=False)
            x = torch.tensor([5.0])
            out = norm(x)
            assert out.shape == (1,)
            assert torch.isfinite(out).all()

    def test_large_d_model(self):
        """Large d_model should work without issues."""
        norm = MinkowskiLayerNormImproved(
            d_model=2048, elementwise_affine=False
        )
        x = torch.randn(1, 2048)
        out = norm(x)
        assert out.shape == (1, 2048)
        assert torch.isfinite(out).all()

    def test_inf_input_handled(self):
        """Inf input should not crash (output may be NaN but not error)."""
        norm = MinkowskiLayerNormImproved(
            d_model=4, elementwise_affine=False
        )
        x = torch.tensor([[float("inf"), 1.0, 2.0, 3.0]])
        # Should not raise
        out = norm(x)
        assert out.shape == (1, 4)

    def test_set_timelike_mask_from_list(self):
        """set_timelike_mask should accept a Python list."""
        norm = MinkowskiLayerNormImproved(d_model=4)
        norm.set_timelike_mask([True, False, True, False])
        assert norm._has_mask
        assert norm.timelike_mask[0].item() is True
        assert norm.timelike_mask[1].item() is False

    def test_set_timelike_mask_from_tuple(self):
        """set_timelike_mask should accept a Python tuple."""
        norm = MinkowskiLayerNormImproved(d_model=3)
        norm.set_timelike_mask((False, True, False))
        assert norm._has_mask
        assert norm.timelike_mask[1].item() is True
