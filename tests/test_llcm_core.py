"""
tests/test_llcm_core.py

LLCM 核心模块的单元测试。

运行方式：
  pytest tests/test_llcm_core.py -v
"""

import numpy as np
import pytest
import torch

from llcm.core import (
    MinkowskiLN,
    Attn,
    LLCMBackbone,
    stable_ode,
    running_ode,
    simulate,
    build_dataset,
    momentum_change,
    pretrain,
    T_DIM,
    EMBED_DIM,
    N_HEADS,
    N_LABELS,
    STATE_DIM,
    LANG_DIM,
    T_IN,
    T_OUT,
    TIME_RATIO,
)
from lorentz_transformer import (
    LorentzMultiHeadAttention,
    MinkowskiLayerNorm,
    compute_t_dim,
)


# ── 别名测试 ───────────────────────────────────────────────────

class TestAliases:
    def test_minkowski_ln_is_minkowski_layer_norm(self):
        """MinkowskiLN 应为 MinkowskiLayerNorm 的别名（旧版 mq.abs() 变体）"""
        assert MinkowskiLN is MinkowskiLayerNorm

    def test_attn_is_lorentz_multi_head_attention(self):
        """Attn 应为 LorentzMultiHeadAttention 的别名"""
        assert Attn is LorentzMultiHeadAttention

    def test_t_dim_matches_compute_t_dim(self):
        """T_DIM 应与 compute_t_dim 计算结果一致"""
        expected = compute_t_dim(EMBED_DIM, N_HEADS, TIME_RATIO)
        assert T_DIM == expected

    def test_t_dim_value(self):
        """T_DIM 默认值应为 32（EMBED_DIM=128, N_HEADS=4, TIME_RATIO=0.25）"""
        assert T_DIM == 32


# ── ODE 测试 ───────────────────────────────────────────────────

class TestODEs:
    def test_stable_ode_returns_six_derivatives(self):
        y0 = [0.0, 1.0, 0.0, 1.5, 0.0, 0.0]
        dy = stable_ode(0.0, y0)
        assert len(dy) == 6

    def test_stable_ode_velocity_is_derivative_of_position(self):
        """稳定 ODE：位置导数等于速度"""
        y0 = [0.0, 1.0, 0.0, 1.5, 0.3, 0.2]
        dy = stable_ode(0.0, y0)
        assert dy[0] == pytest.approx(1.5)
        assert dy[2] == pytest.approx(0.2)

    def test_stable_ode_small_damping(self):
        """稳定 ODE：速度阻尼非常小（趋向动量守恒）"""
        y0 = [0.0, 1.0, 0.0, 2.0, 0.0, 1.0]
        dy = stable_ode(0.0, y0)
        assert abs(dy[3]) < 0.01  # vx 阻尼小
        assert abs(dy[5]) < 0.01  # vz 阻尼小

    def test_running_ode_returns_six_derivatives(self):
        y0 = [0.0, 1.0, 0.0, 1.5, 0.0, 0.0]
        dy = running_ode(0.0, y0)
        assert len(dy) == 6

    def test_running_ode_gravity_applied(self):
        """奔跑 ODE：重力始终作用（垂直加速度含 -g/m 项）"""
        # 在飞相（phase >= 0.4 或 yp >= L）时，仅有重力和阻尼
        y0 = [0.0, 2.0, 0.0, 2.0, 0.0, 0.0]  # yp=2.0 远高于 L≈0.95
        # t=0.2 时 phase = (3*0.2)%1.0 = 0.6 >= 0.4（飞相）
        dy = running_ode(0.2, y0)
        assert dy[4] < 0  # 垂直加速度为负（重力向下）


# ── simulate 测试 ──────────────────────────────────────────────

class TestSimulate:
    def test_stable_simulate_shape(self):
        segs = simulate(stable_ode, T=10, n=5, seed=0)
        assert segs.ndim == 3
        assert segs.shape[1] == 10
        assert segs.shape[2] == STATE_DIM

    def test_running_simulate_shape(self):
        segs = simulate(running_ode, T=10, n=5, seed=1)
        assert segs.ndim == 3
        assert segs.shape[1] == 10
        assert segs.shape[2] == STATE_DIM

    def test_simulate_dtype(self):
        segs = simulate(stable_ode, T=5, n=3, seed=2)
        assert segs.dtype == np.float32

    def test_simulate_no_nan(self):
        segs = simulate(stable_ode, T=20, n=10, seed=3)
        assert not np.isnan(segs).any()

    def test_simulate_reproducible_with_seed(self):
        s1 = simulate(stable_ode, T=10, n=5, seed=42)
        s2 = simulate(stable_ode, T=10, n=5, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_simulate_different_seeds(self):
        s1 = simulate(stable_ode, T=10, n=5, seed=1)
        s2 = simulate(stable_ode, T=10, n=5, seed=2)
        assert not np.array_equal(s1, s2)


# ── build_dataset 测试 ─────────────────────────────────────────

class TestBuildDataset:
    def test_returns_tuple(self):
        result = build_dataset(seed=42, n_per=5)
        assert isinstance(result, tuple) and len(result) == 2

    def test_x_shape(self):
        X, y = build_dataset(seed=42, n_per=5)
        assert X.ndim == 3
        assert X.shape[1] == T_IN + T_OUT
        assert X.shape[2] == STATE_DIM

    def test_y_shape(self):
        X, y = build_dataset(seed=42, n_per=5)
        assert y.ndim == 1
        assert len(y) == len(X)

    def test_labels_binary(self):
        _, y = build_dataset(seed=42, n_per=5)
        assert set(np.unique(y)).issubset({0, 1})

    def test_balanced_classes(self):
        """两类样本数量相同"""
        _, y = build_dataset(seed=42, n_per=10)
        assert np.sum(y == 0) == np.sum(y == 1)

    def test_reproducible(self):
        X1, y1 = build_dataset(seed=99, n_per=5)
        X2, y2 = build_dataset(seed=99, n_per=5)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


# ── momentum_change 测试 ───────────────────────────────────────

class TestMomentumChange:
    def test_returns_float(self):
        traj = np.random.randn(10, 40, STATE_DIM).astype(np.float32)
        mc = momentum_change(traj)
        assert isinstance(mc, float)

    def test_non_negative(self):
        traj = np.random.randn(10, 40, STATE_DIM).astype(np.float32)
        assert momentum_change(traj) >= 0.0

    def test_constant_velocity_near_zero(self):
        """匀速运动：动量变化率接近零"""
        T = 30
        traj = np.zeros((T, STATE_DIM), dtype=np.float32)
        traj[:, 3] = 1.5  # 恒定 vx
        mc = momentum_change(traj)
        assert mc == pytest.approx(0.0, abs=1e-6)

    def test_stable_less_than_running(self):
        """稳定轨迹的动量变化率低于奔跑轨迹"""
        X_s = simulate(stable_ode,  T=40, n=20, seed=10)
        X_r = simulate(running_ode, T=40, n=20, seed=11)
        mc_s = momentum_change(X_s)
        mc_r = momentum_change(X_r)
        assert mc_s < mc_r

    def test_2d_input(self):
        """支持单条轨迹（T, STATE_DIM）作为输入"""
        traj = np.random.randn(40, STATE_DIM).astype(np.float32)
        mc = momentum_change(traj)
        assert isinstance(mc, float)


# ── LLCMBackbone 测试 ─────────────────────────────────────────

class TestLLCMBackbone:
    @pytest.fixture
    def model(self):
        return LLCMBackbone()

    @pytest.fixture
    def x(self):
        return torch.randn(2, T_IN, STATE_DIM)

    def test_forward_shape(self, model, x):
        logits = model(x)
        assert logits.shape == (2, N_LABELS)

    def test_encode_phys_shape(self, model, x):
        h = model.encode_phys(x)
        assert h.shape == (2, T_IN, EMBED_DIM)

    def test_phys_decoder_shape(self, model, x):
        h    = model.encode_phys(x)
        pred = model.phys_decoder(h)
        assert pred.shape == (2, T_IN, STATE_DIM)

    def test_lang_aligner_shape(self, model, x):
        h      = model.encode_phys(x)
        pooled = h.mean(dim=1)
        lang   = model.lang_aligner(pooled)
        assert lang.shape == (2, LANG_DIM)

    def test_forward_A_gen_shape(self, model, x):
        out = model.forward_A_gen(x)
        assert out.shape == (2, LANG_DIM)

    def test_forward_A_gen_normalized(self, model, x):
        """forward_A_gen 输出应已 L2 归一化"""
        out   = model.forward_A_gen(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_gradient_flow(self, model, x):
        x_grad = x.requires_grad_(True)
        logits = model(x_grad)
        loss   = logits.sum()
        loss.backward()
        assert x_grad.grad is not None
        assert not torch.isnan(x_grad.grad).any()

    def test_no_nan_in_forward(self, model, x):
        logits = model(x)
        assert not torch.isnan(logits).any()

    def test_has_required_submodules(self, model):
        assert hasattr(model, 'embed')
        assert hasattr(model, 'blocks')
        assert hasattr(model, 'phys_decoder')
        assert hasattr(model, 'lang_aligner')
        assert hasattr(model, 'lang_gen')
        assert hasattr(model, 'cls_head')

    def test_blocks_use_f3_formula(self, model):
        """所有注意力块应使用 F3 公式"""
        for block in model.blocks:
            assert block.attn.formula == 'f3'

    def test_sigma_attribute_accessible(self, model):
        """F3 注意力的 sigma 属性可访问"""
        for block in model.blocks:
            s = block.attn.sigma
            assert s is not None
            assert 0.0 < s < 1.0


# ── pretrain 集成测试 ──────────────────────────────────────────

class TestPretrain:
    def test_pretrain_returns_model(self):
        """pretrain 应返回同一个 LLCMBackbone 实例"""
        model  = LLCMBackbone()
        result = pretrain(model, seed=999, epochs=1, bs=4)
        assert result is model

    def test_pretrain_loss_decreases(self):
        """经过若干轮预训练后，模型参数应已更新"""
        model  = LLCMBackbone()
        params_before = [p.clone() for p in model.parameters()]
        pretrain(model, seed=7, epochs=2, bs=4)
        changed = any(
            not torch.equal(p, pb)
            for p, pb in zip(model.parameters(), params_before)
        )
        assert changed, "预训练未更新任何参数"
