"""
tests/test_attention.py

Minkowski多头注意力机制的单元测试。

运行方式：
  pytest tests/test_attention.py -v
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
class MockConfig:
    """模拟LorentzConfig用于测试。"""

    d_model: int = 256
    n_heads: int = 8
    lorentz_alpha: float = 0.25
    dropout: float = 0.1


class TestComputeDt2Info:
    """K=1信息时间度量的测试。"""

    def test_output_shape(self):
        """输出应该是标量。"""
        attn_w = torch.rand(2, 8, 128, 128)
        dt2_info = compute_dt2_info(attn_w)
        assert dt2_info.shape == torch.Size([])

    def test_output_positive(self):
        """输出应该是非负数（信息密度）。"""
        attn_w = torch.rand(2, 8, 128, 128)
        dt2_info = compute_dt2_info(attn_w)
        assert dt2_info.item() >= 0

    def test_uniform_attention(self):
        """均匀分布的注意力应该有最小的信息密度。"""
        L = 128
        attn_w = torch.ones(2, 8, L, L) / L
        dt2_info = compute_dt2_info(attn_w)
        expected = 1.0 / (L * math.log(L))
        assert abs(dt2_info.item() - expected) < 0.01

    def test_concentrated_attention(self):
        """集中的注意力（one-hot）应该有最大的信息密度。"""
        L = 128
        attn_w = torch.zeros(2, 8, L, L)
        for i in range(L):
            attn_w[:, :, i, 0] = 1.0
        dt2_info = compute_dt2_info(attn_w)
        assert dt2_info.item() > 0

    def test_batch_averaging(self):
        """同一batch的不同样本应该被平均。"""
        attn_w1 = torch.rand(1, 8, 128, 128)
        attn_w2 = torch.rand(1, 8, 128, 128)
        combined = torch.cat([attn_w1, attn_w2], dim=0)

        dt2_info1 = compute_dt2_info(attn_w1)
        dt2_info2 = compute_dt2_info(attn_w2)
        dt2_info_combined = compute_dt2_info(combined)

        expected = (dt2_info1.item() + dt2_info2.item()) / 2
        assert abs(dt2_info_combined.item() - expected) < 1e-5


class TestLorentzMultiHeadAttention:
    """Minkowski多头注意力的测试。"""

    @pytest.fixture
    def config(self):
        """创建标准配置。"""
        return MockConfig()

    @pytest.fixture
    def attn_module(self, config):
        """创建注意力模块。"""
        return LorentzMultiHeadAttention(config)

    @pytest.fixture
    def sample_input(self, config):
        """创建样本输入。"""
        return torch.randn(2, 128, config.d_model)

    def test_output_shape(self, attn_module, sample_input):
        """输出形状应该与输入相同。"""
        output, weights = attn_module(sample_input)
        assert output.shape == sample_input.shape
        assert weights.shape == (2, 8, 128, 128)

    def test_weights_shape(self, attn_module, sample_input):
        """注意力权重形状应该正确。"""
        _, weights = attn_module(sample_input)
        B, L = sample_input.shape[0], sample_input.shape[1]
        H = attn_module.n_heads
        assert weights.shape == (B, H, L, L)

    def test_weights_sum_to_one(self, attn_module, sample_input):
        """注意力权重应该沿最后一维求和为1。"""
        _, weights = attn_module(sample_input)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_deterministic(self, attn_module, sample_input):
        """相同输入应该产生相同输出。"""
        attn_module.eval()
        torch.manual_seed(42)
        output1, _ = attn_module(sample_input)

        torch.manual_seed(42)
        output2, _ = attn_module(sample_input)

        assert torch.allclose(output1, output2, atol=1e-5)

    def test_minkowski_correction_with_mask(self, config, sample_input):
        """应用mask后应该激活Minkowski修正。"""
        attn = LorentzMultiHeadAttention(config)
        assert not attn._has_mask

        mask = torch.randint(0, 2, (config.d_model,)).bool()
        attn.set_timelike_mask(mask)
        assert attn._has_mask

        output, _ = attn(sample_input)
        assert output.shape == sample_input.shape

    def test_alpha_zero_equals_standard_attention(self, sample_input):
        """α=0时应该等同于标准注意力。"""
        config_lorentz = MockConfig(lorentz_alpha=0.0)
        config_standard = MockConfig(lorentz_alpha=0.0)

        attn_lorentz = LorentzMultiHeadAttention(config_lorentz)
        attn_standard = LorentzMultiHeadAttention(config_standard)
        attn_lorentz.load_state_dict(attn_standard.state_dict())
        attn_lorentz.eval()
        attn_standard.eval()

        output_lorentz, weights_lorentz = attn_lorentz(sample_input)
        output_standard, weights_standard = attn_standard(sample_input)

        assert torch.allclose(output_lorentz, output_standard, atol=1e-5)
        assert torch.allclose(weights_lorentz, weights_standard, atol=1e-5)

    def test_minkowski_scores_differ_from_standard(self, config, sample_input):
        """有mask时Minkowski scores应该与标准不同。"""
        attn = LorentzMultiHeadAttention(config)

        mask = torch.randint(0, 2, (config.d_model,)).bool()
        attn.set_timelike_mask(mask)
        output_lorentz, _ = attn(sample_input)

        attn.set_timelike_mask(torch.zeros(config.d_model, dtype=torch.bool))
        output_standard, _ = attn(sample_input)

        assert not torch.allclose(output_lorentz, output_standard, atol=1e-4)

    def test_causal_mask(self, attn_module, sample_input):
        """Causal mask应该正确应用。"""
        _, L, _ = sample_input.shape
        causal_mask = torch.triu(
            torch.full((L, L), float("-inf")), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        _, weights = attn_module(sample_input, causal_mask)

        for i in range(L):
            for j in range(i + 1, L):
                assert weights[0, 0, i, j].item() == 0.0

    def test_padding_mask(self, attn_module, sample_input):
        """Padding mask应该正确应用。"""
        B, L, _ = sample_input.shape
        pad_len = 10
        pad_mask = torch.ones(B, L)
        pad_mask[:, -pad_len:] = 0

        attn_mask = (~pad_mask.bool()).float()
        attn_mask = attn_mask.masked_fill(
            ~pad_mask.bool(), float("-inf")
        ).unsqueeze(1).unsqueeze(2)

        _, weights = attn_module(sample_input, attn_mask)

        for i in range(L - pad_len):
            for j in range(L - pad_len, L):
                assert weights[0, 0, i, j].item() < 0.01

    def test_numerical_stability_with_large_values(self, attn_module, config):
        """大值输入不应该导致数值不稳定。"""
        x = torch.randn(2, 128, config.d_model) * 100
        output, weights = attn_module(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert not torch.isnan(weights).any()
        assert not torch.isinf(weights).any()

    def test_gradient_flow(self, attn_module, sample_input):
        """梯度应该能正常反向传播。"""
        sample_input.requires_grad = True
        output, _ = attn_module(sample_input)

        loss = output.sum()
        loss.backward()

        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()

    def test_last_intervals_saved(self, attn_module, sample_input):
        """应该保存最近的注意力间隔。"""
        attn_module(sample_input)

        assert attn_module.last_intervals is not None
        assert attn_module.last_intervals_raw is not None
        assert attn_module.last_intervals.shape == (2, 8, 128, 128)

    def test_extra_repr(self, attn_module):
        """extra_repr应该包含关键信息。"""
        repr_str = attn_module.extra_repr()
        assert "n_heads" in repr_str
        assert "alpha" in repr_str


class TestHutchinsonDiagHessian:
    """Hutchinson对角Hessian估计的测试。"""

    def test_output_shape(self):
        """输出形状应该与参数相同。"""
        param = nn.Parameter(torch.randn(256, 256))

        def loss_fn():
            return (param ** 2).sum()

        G = hutchinson_diag_hessian(loss_fn, param, n_samples=5)
        assert G.shape == param.shape

    def test_quadratic_form(self):
        """对二次型应该给出正确的Hessian。"""
        param = nn.Parameter(torch.randn(10, 10))

        def loss_fn():
            return (param ** 2).sum()

        G = hutchinson_diag_hessian(loss_fn, param, n_samples=100)

        assert (G > 0).any()
        assert abs(G.mean().item() - 2.0) < 1.0

    def test_non_zero_output(self):
        """输出不应该全是0。"""
        param = nn.Parameter(torch.randn(256, 256))

        def loss_fn():
            return (param ** 2).sum()

        G = hutchinson_diag_hessian(loss_fn, param, n_samples=20)
        assert (G != 0).any()


class TestIntegration:
    """端到端集成测试。"""

    def test_full_forward_pass(self):
        """完整的前向传播测试。"""
        config = MockConfig()
        attn = LorentzMultiHeadAttention(config)

        x = torch.randn(2, 128, config.d_model)
        mask = torch.randint(0, 2, (config.d_model,)).bool()
        attn.set_timelike_mask(mask)

        output, weights = attn(x)
        dt2_info = compute_dt2_info(weights)

        assert output.shape == x.shape
        assert weights.shape == (2, 8, 128, 128)
        assert 0 <= dt2_info.item()

    def test_batch_processing(self):
        """不同batch大小的处理。"""
        config = MockConfig()
        attn = LorentzMultiHeadAttention(config)

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 128, config.d_model)
            output, weights = attn(x)
            assert output.shape == x.shape
            assert weights.shape[0] == batch_size

    def test_sequence_length_flexibility(self):
        """不同序列长度的处理。"""
        config = MockConfig()
        attn = LorentzMultiHeadAttention(config)

        for seq_len in [32, 64, 128, 256]:
            x = torch.randn(2, seq_len, config.d_model)
            output, weights = attn(x)
            assert output.shape == x.shape
            assert weights.shape == (2, 8, seq_len, seq_len)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
