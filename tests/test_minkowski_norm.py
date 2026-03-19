"""
tests/test_minkowski_norm.py

MinkowskiLayerNorm 的单元测试。
"""

import math

import pytest
import torch

from lorentz_transformer import (
    MinkowskiLayerNorm,
    MinkowskiLayerNormImproved,
    MinkowskiLayerNormOptimized,
    MinkowskiLayerNormStable,
)
from lorentz_transformer.core import (
    MinkowskiLayerNorm as CoreMinkowskiLayerNorm,
    MinkowskiLayerNormImproved as CoreMinkowskiLayerNormImproved,
    MinkowskiLayerNormOptimized as CoreMinkowskiLayerNormOptimized,
    MinkowskiLayerNormStable as CoreMinkowskiLayerNormStable,
)


class TestMinkowskiLayerNorm:
    """MinkowskiLayerNorm 的测试。"""

    @pytest.fixture
    def norm(self):
        """创建默认归一化模块。"""
        return MinkowskiLayerNorm(d_model=4)

    @pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 4)])
    def test_output_shape(self, shape):
        """输出形状应该与输入相同。"""
        x = torch.randn(*shape)
        norm = MinkowskiLayerNorm(d_model=4)
        output = norm(x)
        assert output.shape == x.shape

    def test_exported_from_core_and_package(self):
        """模块应该从 core 和顶层包导出。"""
        assert CoreMinkowskiLayerNorm == MinkowskiLayerNorm
        assert CoreMinkowskiLayerNormImproved == MinkowskiLayerNormImproved
        assert CoreMinkowskiLayerNormOptimized == MinkowskiLayerNormOptimized
        assert CoreMinkowskiLayerNormStable == MinkowskiLayerNormStable

    def test_euclidean_norm_without_timelike_mask(self):
        """无类时mask时应退化为欧氏范数归一化。"""
        norm = MinkowskiLayerNorm(
            d_model=2,
            eps=1e-5,
            elementwise_affine=False,
        )
        x = torch.tensor([[3.0, 4.0]])

        output = norm(x)
        expected = x / math.sqrt(25.0 + 1e-5)

        assert torch.allclose(output, expected, atol=1e-6)

    def test_timelike_mask_uses_minkowski_norm(self):
        """类时mask应触发闵可夫斯基范数。"""
        norm = MinkowskiLayerNorm(
            d_model=2,
            eps=1e-5,
            elementwise_affine=False,
        )
        norm.set_timelike_mask([False, True])
        x = torch.tensor([[3.0, 4.0]])

        output = norm(x)
        expected = x / math.sqrt(abs(9.0 - 16.0) + 1e-5)

        assert torch.allclose(output, expected, atol=1e-6)
        assert norm._has_mask

    def test_mean_shift_applies_before_normalization(self):
        """启用 mean shift 时应先对最后一维去均值。"""
        norm = MinkowskiLayerNorm(
            d_model=2,
            eps=1e-5,
            elementwise_affine=False,
            use_mean_shift=True,
        )
        x = torch.tensor([[2.0, 4.0]])

        output = norm(x)
        centered = torch.tensor([[-1.0, 1.0]])
        expected = centered / math.sqrt(2.0 + 1e-5)

        assert torch.allclose(output, expected, atol=1e-6)

    def test_affine_parameters_are_applied(self):
        """可学习 weight 和 bias 应正确应用。"""
        norm = MinkowskiLayerNorm(d_model=2, eps=1e-5)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([2.0, -1.0]))
            norm.bias.copy_(torch.tensor([0.5, 1.5]))

        x = torch.tensor([[3.0, 4.0]])
        base = x / math.sqrt(25.0 + 1e-5)
        expected = (
            base * torch.tensor([[2.0, -1.0]])
            + torch.tensor([[0.5, 1.5]])
        )

        output = norm(x)
        assert torch.allclose(output, expected, atol=1e-6)

    def test_gradient_flow(self, norm):
        """梯度应该可以正常反向传播。"""
        x = torch.randn(2, 3, 4, requires_grad=True)
        output = norm(x)

        output.sum().backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_numerical_stability_with_large_values(self, norm):
        """大值输入不应产生 NaN 或 Inf。"""
        x = torch.randn(2, 3, 4) * 1_000
        norm.set_timelike_mask((True, False, True, False))

        output = norm(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_set_timelike_mask_validates_shape(self, norm):
        """mask 维度不匹配时应报错。"""
        with pytest.raises(ValueError):
            norm.set_timelike_mask([True, False])

    def test_forward_validates_last_dimension(self, norm):
        """输入末维不匹配时应报错。"""
        with pytest.raises(ValueError):
            norm(torch.randn(2, 3, 5))

    def test_optimized_variant_ignores_timelike_mask(self):
        """Optimized 版本应始终使用 L2 范数。"""
        eps = 1e-5
        norm = MinkowskiLayerNormOptimized(
            d_model=2,
            eps=eps,
            elementwise_affine=False,
        )
        x = torch.tensor([[3.0, 4.0]])

        norm.set_timelike_mask([True, True])
        output = norm(x)
        expected = x / math.sqrt(25.0 + eps)

        assert torch.allclose(output, expected, atol=1e-6)
        assert norm._has_mask

    def test_stable_variant_falls_back_to_l2_when_interval_degenerate(self):
        """Stable 版本在退化区间上应回退到 L2。"""
        eps = 1e-5
        norm = MinkowskiLayerNormStable(
            d_model=2,
            eps=eps,
            elementwise_affine=False,
            minkowski_fallback_threshold=0.0,
        )
        norm.set_timelike_mask([True, False])
        x = torch.tensor([[3.0, 3.0]])

        output = norm(x)
        expected = x / math.sqrt(18.0 + eps)

        assert torch.allclose(output, expected, atol=1e-6)

    def test_stable_variant_respects_fallback_threshold(self):
        """Stable 版本应根据阈值决定是否整体回退。"""
        eps = 1e-5
        x = torch.tensor([[3.0, 3.0], [3.0, 0.0]])
        low_threshold = MinkowskiLayerNormStable(
            d_model=2,
            eps=eps,
            elementwise_affine=False,
            minkowski_fallback_threshold=0.4,
        )
        high_threshold = MinkowskiLayerNormStable(
            d_model=2,
            eps=eps,
            elementwise_affine=False,
            minkowski_fallback_threshold=0.6,
        )

        low_threshold.set_timelike_mask([True, False])
        high_threshold.set_timelike_mask([True, False])

        low_output = low_threshold(x)
        high_output = high_threshold(x)

        low_expected_first = torch.tensor(
            [3.0 / math.sqrt(18.0 + eps), 3.0 / math.sqrt(18.0 + eps)]
        )
        high_expected_first = torch.tensor(
            [3.0 / math.sqrt(eps), 3.0 / math.sqrt(eps)]
        )

        assert torch.allclose(low_output[0], low_expected_first, atol=1e-6)
        assert torch.allclose(high_output[0], high_expected_first, atol=1e-3)
        assert not torch.allclose(low_output[0], high_output[0], atol=1e-3)

    def test_stable_variant_can_disable_minkowski_path(self):
        """Stable 版本可显式禁用 Minkowski 计算。"""
        eps = 1e-5
        norm = MinkowskiLayerNormStable(
            d_model=2,
            eps=eps,
            elementwise_affine=False,
            use_minkowski=False,
        )
        norm.set_timelike_mask([True, False])
        x = torch.tensor([[3.0, 4.0]])

        output = norm(x)
        expected = x / math.sqrt(25.0 + eps)

        assert torch.allclose(output, expected, atol=1e-6)

    def test_improved_variant_uses_l2_without_mask(self):
        """Improved 版本在没有 mask 时应使用 L2。"""
        eps = 1e-5
        norm = MinkowskiLayerNormImproved(
            d_model=2,
            eps=eps,
            elementwise_affine=False,
        )
        x = torch.tensor([[5.0, 12.0]])

        output = norm(x)
        expected = x / math.sqrt(169.0 + eps)

        assert torch.allclose(output, expected, atol=1e-6)

    def test_improved_variant_falls_back_to_l2_for_small_intervals(self):
        """Improved 版本在闵氏区间过小时应自动回退。"""
        eps = 1e-5
        norm = MinkowskiLayerNormImproved(
            d_model=2,
            eps=eps,
            elementwise_affine=False,
        )
        norm.set_timelike_mask([True, False])
        x = torch.tensor([[3.0, 3.0]])

        output = norm(x)
        expected = x / math.sqrt(18.0 + eps)

        assert torch.allclose(output, expected, atol=1e-6)
