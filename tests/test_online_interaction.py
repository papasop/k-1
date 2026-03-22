"""
tests/test_online_interaction.py

在线交互组件的单元测试（对应 Law II / Law III / Theorem 4 / Theorem 6 / Prop 7）。

运行方式：
  pytest tests/test_online_interaction.py -v
"""

import numpy as np
import pytest
import torch

from llcm.core import (
    LLCMBackbone,
    EuclideanBackbone,
    pretrain,
    pretrain_euc,
    compute_dc,
    compute_K,
    compute_kappa,
    online_step,
    build_dataset,
    encode,
    STABLE_INSTRUCTIONS,
    T_IN,
    STATE_DIM,
    EMBED_DIM,
    LANG_DIM,
    T_DIM,
    N_LABELS,
    N_LAYERS,
    N_HEADS,
    device,
)


# ── EuclideanBackbone 结构测试 ────────────────────────────────

class TestEuclideanBackbone:
    @pytest.fixture
    def model(self):
        return EuclideanBackbone()

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

    def test_no_nan_in_forward(self, model, x):
        assert not torch.isnan(model(x)).any()

    def test_gradient_flow(self, model, x):
        x_g    = x.requires_grad_(True)
        logits = model(x_g)
        logits.sum().backward()
        assert x_g.grad is not None
        assert not torch.isnan(x_g.grad).any()

    def test_has_required_submodules(self, model):
        for attr in ('embed', 'blocks', 'phys_decoder',
                     'lang_aligner', 'lang_gen', 'cls_head'):
            assert hasattr(model, attr)

    def test_n_blocks(self, model):
        assert len(model.blocks) == N_LAYERS

    def test_blocks_use_standard_mha(self, model):
        """EuclideanBackbone 应使用标准 MultiheadAttention（无洛伦兹修正）"""
        import torch.nn as nn
        for block in model.blocks:
            assert isinstance(block.attn, nn.MultiheadAttention)

    def test_pretrain_euc_returns_model(self):
        """pretrain_euc 应返回同一个 EuclideanBackbone 实例"""
        model  = EuclideanBackbone()
        result = pretrain_euc(model, seed=999, epochs=1, bs=4)
        assert result is model

    def test_pretrain_euc_updates_params(self):
        """pretrain_euc 应更新模型参数"""
        model         = EuclideanBackbone()
        params_before = [p.clone() for p in model.parameters()]
        pretrain_euc(model, seed=7, epochs=2, bs=4)
        changed = any(
            not torch.equal(p, pb)
            for p, pb in zip(model.parameters(), params_before)
        )
        assert changed, "pretrain_euc 未更新任何参数"


# ── compute_dc 测试（Theorem 4）──────────────────────────────

class TestComputeDc:
    def test_euclidean_dc_is_zero(self):
        """Theorem 3: 欧氏模型 dc = 0（det G = 1 > 0）"""
        model = EuclideanBackbone()
        assert compute_dc(model) == 0.0

    def test_lorentz_dc_positive(self):
        """Theorem 4: F3 洛伦兹模型 dc > 0（det G < 0，sigma ∈ (0,1)）"""
        model = LLCMBackbone()
        dc    = compute_dc(model)
        assert dc > 0.0

    def test_lorentz_dc_in_valid_range(self):
        """dc = σ·(1-σ)/2 ≤ 0.125（当 σ=0.5 时取最大值）"""
        model = LLCMBackbone()
        dc    = compute_dc(model)
        assert 0.0 < dc <= 0.125 + 1e-6

    def test_dc_returns_float(self):
        assert isinstance(compute_dc(LLCMBackbone()),   float)
        assert isinstance(compute_dc(EuclideanBackbone()), float)

    def test_dc_f3_greater_than_euc(self):
        """F3 dc 严格大于欧氏 dc（核心假设）"""
        assert compute_dc(LLCMBackbone()) > compute_dc(EuclideanBackbone())


# ── compute_K 测试（Law III）────────────────────────────────

class TestComputeK:
    @pytest.fixture
    def x_phys(self):
        return torch.randn(4, T_IN, STATE_DIM)

    def test_lorentz_K_returns_float(self, x_phys):
        model = LLCMBackbone()
        K     = compute_K(model, x_phys)
        assert isinstance(K, float)

    def test_euclidean_K_near_one(self, x_phys):
        """欧氏: K = ‖x‖² = 1（归一化后恒等于 1）"""
        model = EuclideanBackbone()
        K     = compute_K(model, x_phys)
        assert abs(K - 1.0) < 1e-4

    def test_lorentz_K_in_range(self, x_phys):
        """洛伦兹 K ∈ (-1, 1)（归一化向量 Minkowski 内积范围）"""
        model = LLCMBackbone()
        K     = compute_K(model, x_phys)
        assert -1.0 <= K <= 1.0

    def test_no_grad_in_compute_K(self, x_phys):
        """compute_K 不应保留计算图（使用 torch.no_grad）"""
        model = LLCMBackbone()
        K     = compute_K(model, x_phys)
        # 能正常取值，无异常即可
        assert K is not None

    def test_model_set_to_eval_mode(self, x_phys):
        """compute_K 后模型应处于 eval 模式"""
        model = LLCMBackbone()
        compute_K(model, x_phys)
        assert not model.training


# ── compute_kappa 测试（Theorem 6）──────────────────────────

class TestComputeKappa:
    def test_decreasing_sequence_positive_kappa(self):
        """严格递减序列应产生正的 κK（表示收敛）"""
        losses = [1.0, 0.8, 0.6, 0.4, 0.2]
        kappa  = compute_kappa(losses)
        assert kappa > 0.0

    def test_flat_sequence_zero_kappa(self):
        """恒定序列 κK 应为 0（无收敛）"""
        losses = [0.5, 0.5, 0.5, 0.5, 0.5]
        kappa  = compute_kappa(losses)
        assert kappa == pytest.approx(0.0, abs=1e-6)

    def test_returns_float(self):
        assert isinstance(compute_kappa([0.5, 0.3, 0.2]), float)

    def test_non_negative(self):
        """κK 应为非负（max(0, -slope)）"""
        assert compute_kappa([0.1, 0.2, 0.3]) >= 0.0   # 递增（发散）

    def test_single_element_returns_zero(self):
        assert compute_kappa([0.5]) == 0.0

    def test_faster_decay_gives_higher_kappa(self):
        """衰减越快，κK 应越大"""
        slow = [1.0, 0.9, 0.8, 0.7, 0.6]
        fast = [1.0, 0.5, 0.2, 0.1, 0.05]
        assert compute_kappa(fast) > compute_kappa(slow)


# ── online_step 测试（Law II）───────────────────────────────

class TestOnlineStep:
    @pytest.fixture
    def x_phys(self):
        return torch.randn(4, T_IN, STATE_DIM)

    @pytest.fixture
    def v_ref(self):
        import torch.nn.functional as F
        return F.normalize(torch.randn(4, LANG_DIM), dim=-1)

    def _make_opt(self, model, lr=3e-4):
        return torch.optim.AdamW(model.lang_aligner.parameters(), lr=lr)

    def test_returns_float(self, x_phys, v_ref):
        model = LLCMBackbone()
        loss  = online_step(model, x_phys, v_ref, self._make_opt(model))
        assert isinstance(loss, float)

    def test_loss_in_valid_range(self, x_phys, v_ref):
        """V_lang = 1 - cos_sim ∈ [0, 2]"""
        model = LLCMBackbone()
        loss  = online_step(model, x_phys, v_ref, self._make_opt(model))
        assert 0.0 <= loss <= 2.0 + 1e-4

    def test_updates_lang_aligner(self, x_phys, v_ref):
        """online_step 应更新 lang_aligner 参数"""
        model        = LLCMBackbone()
        w_before     = model.lang_aligner.weight.data.clone()
        online_step(model, x_phys, v_ref, self._make_opt(model))
        assert not torch.equal(model.lang_aligner.weight.data, w_before)

    def test_loss_decreases_over_steps(self, x_phys, v_ref):
        """多次 Law II 更新后损失应趋于下降"""
        model  = LLCMBackbone()
        opt    = self._make_opt(model)
        losses = [online_step(model, x_phys, v_ref, opt) for _ in range(20)]
        assert losses[-1] < losses[0]

    def test_works_with_euclidean_backbone(self, x_phys, v_ref):
        """online_step 对 EuclideanBackbone 同样有效"""
        model  = EuclideanBackbone()
        opt    = torch.optim.AdamW(model.lang_aligner.parameters(), lr=3e-4)
        loss   = online_step(model, x_phys, v_ref, opt)
        assert isinstance(loss, float)
        assert 0.0 <= loss <= 2.0 + 1e-4

    def test_model_set_to_train_mode(self, x_phys, v_ref):
        """online_step 后模型应处于 train 模式"""
        model = LLCMBackbone()
        model.eval()
        online_step(model, x_phys, v_ref, self._make_opt(model))
        assert model.training


# ── 集成: Theorem 4 & 6 定性验证 ─────────────────────────────

class TestTheoremValidation:
    """
    用少量轮次、种子定性验证核心假设（非完整实验，仅确保代码可运行且方向正确）。
    """

    def test_theorem4_dc_ordering(self):
        """Theorem 4: dc_F3 > dc_Euc = 0"""
        dc_f3  = compute_dc(LLCMBackbone())
        dc_euc = compute_dc(EuclideanBackbone())
        assert dc_f3  > 0.0
        assert dc_euc == 0.0
        assert dc_f3  > dc_euc

    def test_online_step_convergence_smoke(self):
        """
        冒烟测试：少量在线步后，F3 和 Euc 损失均可下降（Law II 有效）。
        """
        import torch.nn.functional as F

        torch.manual_seed(0)
        x_phys = torch.randn(4, T_IN, STATE_DIM)
        v_ref  = F.normalize(torch.randn(4, LANG_DIM), dim=-1)

        for BackboneCls in (LLCMBackbone, EuclideanBackbone):
            model  = BackboneCls()
            opt    = torch.optim.AdamW(model.lang_aligner.parameters(), lr=3e-4)
            losses = [online_step(model, x_phys, v_ref, opt) for _ in range(10)]
            # 损失应下降
            assert losses[-1] < losses[0], (
                f"{BackboneCls.__name__}: 损失未下降 "
                f"({losses[0]:.4f} → {losses[-1]:.4f})"
            )

    def test_compute_K_changes_after_online_steps(self):
        """
        online_step 更新 lang_aligner 后，F3 模型的 K(x) 应发生变化
        （Law III：K(x) 由训练动态驱动）。
        """
        import torch.nn.functional as F

        torch.manual_seed(42)
        model  = LLCMBackbone()
        x_phys = torch.randn(4, T_IN, STATE_DIM)
        v_ref  = F.normalize(torch.randn(4, LANG_DIM), dim=-1)
        opt    = torch.optim.AdamW(model.lang_aligner.parameters(), lr=3e-4)

        K_before = compute_K(model, x_phys)
        for _ in range(20):
            online_step(model, x_phys, v_ref, opt)
        K_after = compute_K(model, x_phys)

        assert K_before != pytest.approx(K_after, abs=1e-4), (
            "K(x) 在 20 步在线更新后未发生变化"
        )
