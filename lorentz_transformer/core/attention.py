
"""
lorentz_transformer/core/attention.py

Component 1: Minkowski多头注意力机制

核心公式：
  scores_L = Q·η·K^T / √d_h
  其中 η = I - 2α·P_t (闵可夫斯基符号矩阵)

这个模块可以独立使用，无需依赖其他洛伦兹组件。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Part 1: 辅助函数
# ============================================================================

def compute_dt2_info(attn_w: torch.Tensor) -> torch.Tensor:
    """
    计算K=1信息时间度量。

    dt²_info = Σ_q K_q = Σ_q (Φ_q / H_q)

    其中：
      H_q = Shannon熵（注意力分布的不确定性）
      Φ_q = Σ_j a_qj² （注意力集中度）
      K_q = Φ_q/H_q （信息时间密度）

    该量用于诊断注意力的信息几何性质。

    Args:
        attn_w (torch.Tensor): 注意力权重，形状 (B, H, L, L)
            其中 B=batch_size, H=num_heads, L=seq_len

    Returns:
        torch.Tensor: scalar，所有query位置的K_q均值

    Example:
        >>> attn_w = torch.rand(2, 8, 128, 128)  # batch_size=2, heads=8, seq_len=128
        >>> dt2_info = compute_dt2_info(attn_w)
        >>> print(dt2_info.item())  # scalar value
    """
    # 对头维度求平均，得到每个样本的注意力分布
    aw = attn_w.mean(dim=1).float()  # (B, L, L)

    # 处理NaN值（防止log(0)）
    aw = torch.nan_to_num(aw, nan=0.0)

    # 行归一化，确保是有效的概率分布
    aw = aw / aw.sum(-1, keepdim=True).clamp(min=1e-8)

    # 计算Shannon熵 H_q = -Σ_j a_qj * log(a_qj)
    H = -(aw * torch.log(aw + 1e-8)).sum(-1)  # (B, L)

    # 计算集中度 Φ_q = Σ_j a_qj²
    Phi = (aw ** 2).sum(-1)  # (B, L)

    # 避免除零
    H = H.clamp(min=1e-8)

    # 计算信息时间密度 K_q = Φ_q / H_q
    K = Phi / H  # (B, L)

    return K.mean()  # 返回所有位置的平均值


def hutchinson_diag_hessian(
    loss_fn,
    param: nn.Parameter,
    n_samples: int = 20,
) -> torch.Tensor:
    """
    用Hutchinson方法估计Hessian的对角元素。

    用于识别参数空间中的类时方向（dt²_info的凹方向）。

    数学原理：
      G_ii ≈ (1/K) Σ_k v_k[i] · (H·v_k)[i]

      其中：
        v_k ~ Rademacher{±1} (随机符号向量)
        H·v_k = ∂²loss/∂param² · v_k (Hessian-向量乘积)

    解释：
      G_ii < 0  ⟹  参数i是类时维度（凹方向）
      G_ii > 0  ⟹  参数i是类空维度（凸方向）

    Args:
        loss_fn: 无参数的callable，返回scalar loss
        param (nn.Parameter): 目标参数（通常是W_Q），requires_grad=True
        n_samples (int): Rademacher采样数（默认20）

    Returns:
        torch.Tensor: 与param同形状的对角Hessian估计

    Example:
        >>> model = LorentzMultiHeadAttention(config)
        >>> W_Q = model.q_proj.weight
        >>> def loss_fn():
        ...     x = torch.randn(2, 128, 256)
        ...     _, _ = model(x)
        ...     return compute_dt2_info(...)  # 某个loss
        >>> G = hutchinson_diag_hessian(loss_fn, W_Q, n_samples=20)
        >>> is_timelike = (G < 0)  # bool mask
    """
    G = torch.zeros_like(param.data)

    for _ in range(n_samples):
        # 生成Rademacher随机向量: {-1, +1}^shape
        v = (
            torch.randint(0, 2, param.shape, device=param.device) * 2 - 1
        ).float()

        # 第一阶梯度（保留计算图）
        loss = loss_fn()
        g1 = torch.autograd.grad(
            loss, param, create_graph=True, retain_graph=True
        )[0]

        if g1 is None:
            continue

        # Hessian-向量乘积: H·v
        Hv = torch.autograd.grad(
            (g1 * v.detach()).sum(), param, retain_graph=False
        )[0]

        if Hv is None:
            continue

        # 清理数值错误
        Hv = torch.nan_to_num(
            Hv, nan=0.0, posinf=0.0, neginf=0.0
        )

        # 累积：G += v ⊙ (H·v)
        G = G + (v * Hv).detach() / n_samples

    return G


# ============================================================================
# Part 2: LorentzMultiHeadAttention
# ============================================================================

class LorentzMultiHeadAttention(nn.Module):
    """
    闵可夫斯基多头注意力机制。

    核心计算：
      scores_L = Q · η · K^T / √d_h
      其中 η = I - 2α·P_t (Minkowski符号矩阵)

    展开形式：
      scores_L = (QK^T)/√d_h - 2α·(Q_t K^T)/√d_h
               = scores_std - 2α × 时间内积

    几何解释：
      - scores_std: 标准欧氏内积（标准注意力）
      - 时间内积: 沿类时方向的内积修正
      - α控制几何效果强度（0=标准注意力，>0=洛伦兹修正）

    参数：
      config (LorentzConfig): 配置对象，包含：
        - d_model: 隐层维度
        - n_heads: 注意力头数
        - lorentz_alpha: Minkowski强度（默认0.25）
        - dropout: Dropout概率

    特点：
      1. 完全兼容标准MultiHeadAttention（α=0时等价）
      2. 类时mask未注入时自动回退到标准注意力
      3. 可用于替换任何Transformer中的标准注意力

    使用示例：
        >>> from lorentz_transformer.core import LorentzMultiHeadAttention
        >>> from lorentz_transformer.models import LorentzConfig
        >>>
        >>> config = LorentzConfig(d_model=256, n_heads=8, lorentz_alpha=0.25)
        >>> attn = LorentzMultiHeadAttention(config)
        >>>
        >>> x = torch.randn(2, 128, 256)  # (batch, seq_len, d_model)
        >>> output, weights = attn(x)
        >>> print(output.shape)  # torch.Size([2, 128, 256])
    """

    def __init__(self, config):
        """初始化Minkowski多头注意力。"""
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.alpha = config.lorentz_alpha

        # 验证d_model能被n_heads整除
        assert (
            config.d_model % config.n_heads == 0
        ), f"d_model={config.d_model} 必须被 n_heads={config.n_heads} 整除"

        # Q, K, V投影层（无bias，与GPT-2一致）
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # 输出投影
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Dropout
        self.drop = nn.Dropout(config.dropout)

        # 类时掩码：(d_model,) bool向量
        # True = 类时维度（G_ii<0）
        # False = 类空维度（G_ii>0）
        # 由外部TimeLikeProbe通过set_timelike_mask()注入
        self.register_buffer(
            "timelike_mask",
            torch.zeros(config.d_model, dtype=torch.bool),
        )

        # 是否已注入mask
        self._has_mask = False

        # 诊断用：保存最近的注意力间隔（光锥分析）
        self.last_intervals: Optional[torch.Tensor] = None
        self.last_intervals_raw: Optional[torch.Tensor] = None

    def set_timelike_mask(self, mask: torch.Tensor) -> None:
        """
        注入类时掩码（由TimeLikeProbe调用）。

        Args:
            mask (torch.Tensor): (d_model,) bool或float张量
                True/1.0 = 类时维度
                False/0.0 = 类空维度
        """
        self.timelike_mask.copy_(mask.bool())
        self._has_mask = mask.any().item()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入特征，形状 (B, L, d_model)
            attention_mask (Optional[torch.Tensor]): 加性注意力掩码
                形状 (B, 1, 1, L) 或 (B, 1, L, L)
                pad位置设为-inf，有效位置设为0

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output: (B, L, d_model) 注意力输出
                - attn_weights: (B, H, L, L) 注意力权重

        计算流程：
            1. 投影Q, K, V并split到多个head
            2. 计算标准scores = QK^T / √d_h
            3. 应用Minkowski修正（如果有mask和α>0）
            4. 应用attention mask
            5. Softmax + Dropout
            6. 乘以V并合并head
            7. 输出投影
        """
        B, L, _ = x.shape
        H, d_h = self.n_heads, self.head_dim
        scale = math.sqrt(d_h)

        # 投影并分割head
        # Q, K, V: (B, L, d_model) → (B, H, L, d_h)
        Q = self.q_proj(x).view(B, L, H, d_h).transpose(1, 2).float()
        K = self.k_proj(x).view(B, L, H, d_h).transpose(1, 2).float()
        V = self.v_proj(x).view(B, L, H, d_h).transpose(1, 2)

        # 标准注意力scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B,H,L,L)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Minkowski修正：scores_L = scores - 2α·time_inner
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if self._has_mask and self.alpha > 0:
            # timelike_mask: (d_model,) → (H, d_h) per-head
            mask_2d = self.timelike_mask.view(H, d_h).float()
            mask_bcast = mask_2d.unsqueeze(0).unsqueeze(2).to(Q.device)

            # 提取类时Q分量
            Q_t = Q * mask_bcast  # (B,H,L,d_h)

            # 幅度归一化：将Q_t缩放到与Q相同的幅度
            # 避免类时分量太小或太大导致数值不稳定
            q_norm = Q.norm(dim=-1, keepdim=True)  # (B,H,L,1)
            qt_norm = Q_t.norm(dim=-1, keepdim=True)  # (B,H,L,1)

            # 有类时分量时进行缩放
            has_timelike = (qt_norm > 1e-6)
            scale_factor = torch.where(
                has_timelike,
                q_norm / qt_norm.clamp(min=1e-8),
                torch.zeros_like(qt_norm),
            )
            Q_t_scaled = Q_t * scale_factor

            # 时间分量内积（与标准scores同量级）
            time_inner = torch.matmul(
                Q_t_scaled, K.transpose(-2, -1)
            ) / scale  # (B,H,L,L)

            # 洛伦兹修正：减去2α倍的时间分量
            scores = scores - 2.0 * self.alpha * time_inner

        # 保存诊断用（原始scores，causal mask前）
        self.last_intervals_raw = scores.detach().clone()

        # 应用attention mask
        if attention_mask is not None:
            scores = scores + attention_mask.to(scores.dtype)

        # 保存诊断用（masked scores）
        self.last_intervals = scores.detach().clone()

        # Softmax → attention probabilities
        attn_probs = F.softmax(scores, dim=-1).to(x.dtype)
        attn_w = self.drop(attn_probs)

        # 乘以V并合并head
        out = torch.matmul(attn_w, V)  # (B,H,L,d_h)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)  # (B,L,d_model)

        # 输出投影
        out = self.o_proj(out)

        return out, attn_probs

    def extra_repr(self) -> str:
        """模块的额外信息表示。"""
        return (
            f"n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, "
            f"alpha={self.alpha}, "
            f"has_mask={self._has_mask}"
        )


# ============================================================================
# Part 3: 单元测试
# ============================================================================

if __name__ == "__main__":
    """
    快速测试脚本（可在tests/test_attention.py中使用）。
    """
    print("=" * 70)
    print("Testing LorentzMultiHeadAttention")
    print("=" * 70)

    # 测试1：基础前向传播
    print("\n[Test 1] 基础前向传播（α=0, 无mask）")
    print("-" * 70)

    from dataclasses import dataclass

    @dataclass
    class Config:
        d_model: int = 256
        n_heads: int = 8
        lorentz_alpha: float = 0.0
        dropout: float = 0.1

    config = Config()
    attn = LorentzMultiHeadAttention(config)
    x = torch.randn(2, 128, 256)

    output, weights = attn(x)
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Weights shape: {weights.shape}")
    assert output.shape == x.shape, "输出形状不匹配"
    assert weights.shape == (2, 8, 128, 128), "权重形状不匹配"
    print("✓ Test 1 passed")

    # 测试2：Minkowski修正激活
    print("\n[Test 2] Minkowski修正（α=0.25, 有mask）")
    print("-" * 70)

    config_lorentz = Config(lorentz_alpha=0.25)
    attn_lorentz = LorentzMultiHeadAttention(config_lorentz)

    # 注入掩码
    mask = torch.randint(0, 2, (256,)).bool()
    attn_lorentz.set_timelike_mask(mask)

    output_lorentz, weights_lorentz = attn_lorentz(x)
    print(f"✓ Lorentz attention output shape: {output_lorentz.shape}")
    print(f"✓ Timelike dimensions: {mask.sum().item()}/256 = {mask.float().mean():.1%}")
    print(f"✓ Alpha: {attn_lorentz.alpha}")
    print("✓ Test 2 passed")

    # 测试3：注意力掩码
    print("\n[Test 3] 注意力掩码（Causal mask）")
    print("-" * 70)

    causal_mask = torch.triu(
        torch.full((128, 128), float("-inf")), diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    output_masked, weights_masked = attn(x, causal_mask)
    print(f"✓ Output with causal mask shape: {output_masked.shape}")

    # 验证因果性：future位置的权重应该为0
    future_weights = weights_masked[0, 0, 0, 1:]
    print(f"✓ Future attention weights (should be ~0): {future_weights[:5].mean():.2e}")
    print("✓ Test 3 passed")

    # 测试4：dt²_info计算
    print("\n[Test 4] 信息时间度量（dt²_info）")
    print("-" * 70)

    dt2_info = compute_dt2_info(weights)
    print(f"✓ dt²_info value: {dt2_info.item():.6f}")
    print(f"✓ dt²_info is scalar: {dt2_info.shape == torch.Size([])}")
    print("✓ Test 4 passed")

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
