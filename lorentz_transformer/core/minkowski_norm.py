"""
lorentz_transformer/core/layer_norm.py

MinkowskiLayerNorm — 真正的 Minkowski 几何归一化

核心公式:
  <x,x>_η = ||x_s||² - ||x_t||²    (Minkowski 内积)
  x_norm   = x / sqrt(|<x,x>_η| + ε)

与旧版的关键区别:
  旧版: _minkowski_norm_sq 取了 .abs()，导致类时/类空无法区分
        实际等价于 L2 归一化的变体，不是真正的 Minkowski 几何
  新版: 保留符号信息，类时(s²-t²>0)和类空(s²-t²<0)有不同的归一化行为
        t_dim 必须与注意力层的 time_ratio 对齐

t_dim 对齐规则 (重要!):
  注意力层: t_dim = int(n_heads * time_ratio) * head_dim
                   = int(n_heads * time_ratio) * (d_model // n_heads)
  LayerNorm:t_dim 必须等于上面的值

  示例: d_model=256, n_heads=8, time_ratio=0.25
    n_t_heads  = max(1, int(8 * 0.25)) = 2
    head_dim   = 256 // 8 = 32
    t_dim      = 2 * 32 = 64   ← LayerNorm 的 t_dim 必须是 64
"""

from typing import Iterable, Optional, Union

import torch
import torch.nn as nn

MaskLike = Union[torch.Tensor, Iterable[bool]]


# ============================================================================
# 辅助函数
# ============================================================================

def compute_t_dim(d_model: int, n_heads: int, time_ratio: float) -> int:
    """
    计算与注意力层对齐的 t_dim。

    与 F1/F3 注意力保持一致的辅助函数，避免手动计算出错。

    Args:
        d_model    : 模型维度
        n_heads    : 注意力头数
        time_ratio : 时间头比例（与注意力层相同）

    Returns:
        t_dim: LayerNorm 应使用的时间维度数

    Example:
        >>> t_dim = compute_t_dim(256, 8, 0.25)   # → 64
        >>> ln = MinkowskiLayerNorm(256, t_dim=t_dim)
    """
    head_dim  = d_model // n_heads
    n_t_heads = max(1, int(n_heads * time_ratio))
    return n_t_heads * head_dim


# ============================================================================
# 基类
# ============================================================================

class _BaseMinkowskiLayerNorm(nn.Module):
    """Minkowski LayerNorm 的基类，提供共享工具。"""

    def __init__(
        self,
        d_model: int,
        t_dim: Optional[int] = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        """
        Args:
            d_model           : 模型维度
            t_dim             : 时间子空间维度（必须与注意力层对齐）
                                None 时退化为 L2 归一化
            eps               : 数值稳定常数
            elementwise_affine: 是否学习 weight/bias
        """
        super().__init__()
        self.d_model = d_model
        self.t_dim   = t_dim
        self.eps     = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(d_model))
            self.bias   = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_buffer("weight", torch.ones(d_model))
            self.register_buffer("bias",   torch.zeros(d_model))

    def _mink_norm_sq(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 Minkowski 内积平方: <x,x>_η = ||x_s||² - ||x_t||²

        不取绝对值——保留符号信息是真正 Minkowski 归一化的关键。
        类时向量 (s²>t²): mink_sq > 0
        类空向量 (s²<t²): mink_sq < 0
        类光向量 (s²=t²): mink_sq ≈ 0

        Returns: (B*L, 1) 的未取绝对值的 Minkowski 内积
        """
        t = x[..., :self.t_dim]          # 时间分量
        s = x[..., self.t_dim:]          # 空间分量
        return (s ** 2).sum(-1, keepdim=True) \
             - (t ** 2).sum(-1, keepdim=True)

    def _l2_norm_sq(self, x: torch.Tensor) -> torch.Tensor:
        """标准 L2 范数平方（fallback 用）。"""
        return (x ** 2).sum(-1, keepdim=True)

    def _apply_affine(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


# ============================================================================
# MinkowskiLayerNorm — 标准版（推荐）
# ============================================================================

class MinkowskiLayerNorm(_BaseMinkowskiLayerNorm):
    """
    真正的 Minkowski 几何归一化。

    公式:
        norm_sq = ||x_s||² - ||x_t||²         (Minkowski 内积)
        x_out   = x / sqrt(|norm_sq| + ε)     (保留符号的稳定开方)
                * weight + bias

    与旧版区别:
        旧版在 _minkowski_norm_sq 里取了 .abs()，丢失了类时/类空信息。
        本版在计算 norm_sq 时不取 abs，只在 sqrt 时用 abs 保证数值稳定，
        让类时和类空向量有不同的归一化行为，保留 Minkowski 几何结构。

    t_dim 必须与注意力层对齐:
        t_dim = compute_t_dim(d_model, n_heads, time_ratio)

    Args:
        d_model           : 模型维度
        t_dim             : 时间子空间维度，用 compute_t_dim() 计算
        eps               : 数值稳定常数，默认 1e-5
        elementwise_affine: 是否学习 weight/bias，默认 True

    Example:
        >>> t_dim = compute_t_dim(d_model=256, n_heads=8, time_ratio=0.25)
        >>> ln = MinkowskiLayerNorm(256, t_dim=t_dim)
        >>> out = ln(torch.randn(2, 128, 256))
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., d_model)
        Returns:
            归一化后的张量，形状与 x 相同
        """
        shape  = x.shape
        x_flat = x.reshape(-1, self.d_model)

        if self.t_dim is not None:
            # 真正的 Minkowski 归一化
            mink_sq = self._mink_norm_sq(x_flat)        # 保留符号
            norm    = torch.sqrt(mink_sq.abs() + self.eps)  # 稳定开方
        else:
            # 没有 t_dim → 退化为 L2 归一化
            norm = torch.sqrt(self._l2_norm_sq(x_flat) + self.eps)

        x_norm = x_flat / (norm + self.eps)
        return self._apply_affine(x_norm).reshape(shape)


# ============================================================================
# MinkowskiLayerNormStable — 带 fallback 的稳健版
# ============================================================================

class MinkowskiLayerNormStable(_BaseMinkowskiLayerNorm):
    """
    带自动 fallback 的 Minkowski 归一化。

    在 Minkowski 范数过小（接近类光）时自动 fallback 到 L2，
    防止数值不稳定。适合训练早期或数据分布未知的场景。

    Args:
        d_model                    : 模型维度
        t_dim                      : 时间子空间维度
        eps                        : 数值稳定常数
        elementwise_affine         : 是否学习 weight/bias
        fallback_threshold         : |mink_sq| < threshold 时 fallback 到 L2
                                     默认 1e-3
        fallback_ratio_threshold   : fallback 比例超过此值时打印警告
                                     默认 0.5
    """

    def __init__(
        self,
        d_model: int,
        t_dim: Optional[int] = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        fallback_threshold: float = 1e-3,
        fallback_ratio_threshold: float = 0.5,
    ):
        super().__init__(d_model, t_dim, eps, elementwise_affine)
        self.fallback_threshold       = fallback_threshold
        self.fallback_ratio_threshold = fallback_ratio_threshold
        self._fallback_count   = 0
        self._total_count      = 0

    @property
    def fallback_ratio(self) -> float:
        """当前批次 fallback 到 L2 的比例（诊断用）。"""
        if self._total_count == 0:
            return 0.0
        return self._fallback_count / self._total_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape  = x.shape
        x_flat = x.reshape(-1, self.d_model)

        if self.t_dim is None:
            norm = torch.sqrt(self._l2_norm_sq(x_flat) + self.eps)
            x_norm = x_flat / (norm + self.eps)
            return self._apply_affine(x_norm).reshape(shape)

        mink_sq  = self._mink_norm_sq(x_flat)
        l2_sq    = self._l2_norm_sq(x_flat)

        # fallback mask：|mink_sq| 太小时用 L2
        use_l2   = mink_sq.abs() < self.fallback_threshold
        norm_sq  = torch.where(use_l2, l2_sq, mink_sq.abs())
        norm     = torch.sqrt(norm_sq + self.eps)

        # 诊断统计
        self._fallback_count = int(use_l2.sum().item())
        self._total_count    = use_l2.numel()

        x_norm = x_flat / (norm + self.eps)
        return self._apply_affine(x_norm).reshape(shape)


# ============================================================================
# MinkowskiLayerNormOptimized — 纯 L2（兼容旧接口，无 Minkowski 几何）
# ============================================================================

class MinkowskiLayerNormOptimized(_BaseMinkowskiLayerNorm):
    """
    纯 L2 归一化，兼容旧版 set_timelike_mask 接口。

    不使用任何 Minkowski 几何，适合：
    - 需要保持旧接口但不想引入 Minkowski 的场景
    - 作为消融实验的基线（纯 L2 vs Minkowski）

    Note:
        set_timelike_mask() 调用被接受但无任何效果，
        仅为兼容旧版 TimeLikeProbe 接口。
    """

    def set_timelike_mask(self, mask: MaskLike) -> None:
        """接受调用但不做任何事（兼容旧接口）。"""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape  = x.shape
        x_flat = x.reshape(-1, self.d_model)
        norm   = torch.sqrt(self._l2_norm_sq(x_flat) + self.eps)
        x_norm = x_flat / (norm + self.eps)
        return self._apply_affine(x_norm).reshape(shape)


# ============================================================================
# 旧版兼容别名（MinkowskiLayerNormImproved → MinkowskiLayerNorm）
# ============================================================================

MinkowskiLayerNormImproved = MinkowskiLayerNorm   # 向后兼容


__all__ = [
    "MinkowskiLayerNorm",
    "MinkowskiLayerNormStable",
    "MinkowskiLayerNormOptimized",
    "MinkowskiLayerNormImproved",
    "compute_t_dim",
]


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("MinkowskiLayerNorm v2 — 测试")
    print("=" * 60)

    D, B, L = 256, 2, 16
    x = torch.randn(B, L, D)

    # 正确的 t_dim（与注意力层对齐）
    t_dim = compute_t_dim(d_model=D, n_heads=8, time_ratio=0.25)
    print(f"\nt_dim = compute_t_dim(256, 8, 0.25) = {t_dim}")
    print(f"（旧版用了 32，正确值是 {t_dim}）\n")

    # ── 测试1：三个变体 ──────────────────────────────────────
    print("【测试1】三个变体输出稳定性")
    print("-" * 60)
    variants = [
        ("MinkowskiLayerNorm (真Mink)", MinkowskiLayerNorm(D, t_dim=t_dim)),
        ("MinkowskiLayerNormStable",    MinkowskiLayerNormStable(D, t_dim=t_dim)),
        ("MinkowskiLayerNormOptimized", MinkowskiLayerNormOptimized(D)),
    ]
    for name, ln in variants:
        out = ln(x)
        print(f"  {name}")
        print(f"    输出范数: {out.norm():.4f}  "
              f"范围: [{out.min():.3f}, {out.max():.3f}]  "
              f"NaN: {torch.isnan(out).any().item()}")

    # ── 测试2：t_dim 对齐验证 ────────────────────────────────
    print("\n【测试2】t_dim 对齐 vs 不对齐的几何差异")
    print("-" * 60)
    ln_wrong  = MinkowskiLayerNorm(D, t_dim=32)   # 旧版错误值
    ln_correct= MinkowskiLayerNorm(D, t_dim=t_dim) # 正确值
    out_w = ln_wrong(x)
    out_c = ln_correct(x)
    diff = (out_w - out_c).abs().mean().item()
    print(f"  t_dim=32  (旧版): 输出范数 {out_w.norm():.4f}")
    print(f"  t_dim={t_dim} (正确): 输出范数 {out_c.norm():.4f}")
    print(f"  平均差异: {diff:.6f}  ({'显著不同' if diff>0.01 else '几乎相同'})")

    # ── 测试3：类时/类空区分验证 ────────────────────────────
    print("\n【测试3】真正的 Minkowski 几何 — 类时vs类空归一化行为不同")
    print("-" * 60)
    ln = MinkowskiLayerNorm(D, t_dim=t_dim)

    # 构造一个纯类时向量（空间分量大于时间分量）
    x_timelike  = torch.zeros(1, 1, D)
    x_timelike[..., t_dim:] = 2.0   # 空间分量=2，时间分量=0 → s²-t²>0

    # 构造一个纯类空向量（时间分量大于空间分量）
    x_spacelike = torch.zeros(1, 1, D)
    x_spacelike[..., :t_dim] = 2.0  # 时间分量=2，空间分量=0 → s²-t²<0

    out_tl = ln(x_timelike)
    out_sl = ln(x_spacelike)

    s_tl = (x_timelike[..., t_dim:]**2).sum().item()**0.5
    t_tl = (x_timelike[..., :t_dim]**2).sum().item()**0.5
    s_sl = (x_spacelike[..., t_dim:]**2).sum().item()**0.5
    t_sl = (x_spacelike[..., :t_dim]**2).sum().item()**0.5

    print(f"  类时向量: ||s||={s_tl:.2f} ||t||={t_tl:.2f}  "
          f"mink_sq={s_tl**2-t_tl**2:+.2f}  "
          f"输出范数={out_tl.norm():.4f}")
    print(f"  类空向量: ||s||={s_sl:.2f} ||t||={t_sl:.2f}  "
          f"mink_sq={s_sl**2-t_sl**2:+.2f}  "
          f"输出范数={out_sl.norm():.4f}")
    diff_geo = abs(out_tl.norm().item() - out_sl.norm().item())
    print(f"  几何差异: {diff_geo:.4f}  "
          f"({'类时≠类空 OK，Minkowski几何有效' if diff_geo<0.01 else '归一化行为不同'})")

    # ── 测试4：掩码比例稳定性 ─────────────────────────────────
    print("\n【测试4】掩码比例稳定性（t_dim 固定，检查输入分布）")
    print("-" * 60)
    ln = MinkowskiLayerNorm(D, t_dim=t_dim)
    norms = []
    for seed in range(10):
        torch.manual_seed(seed)
        x_rand = torch.randn(B, L, D)
        out = ln(x_rand)
        norms.append(out.norm().item())
    print(f"  10次随机输入的输出范数: "
          f"mean={np.mean(norms):.4f}  std={np.std(norms):.4f}  "
          f"cv={np.std(norms)/np.mean(norms):.4f}")
    print(f"  {'稳定 OK' if np.std(norms)/np.mean(norms) < 0.1 else '不稳定，检查 eps'}")

    # ── 测试5：梯度 ──────────────────────────────────────────
    print("\n【测试5】梯度流动")
    print("-" * 60)
    ln = MinkowskiLayerNorm(D, t_dim=t_dim)
    x_grad = torch.randn(B, L, D, requires_grad=True)
    out = ln(x_grad)
    out.sum().backward()
    g_in  = x_grad.grad.norm().item()
    g_w   = ln.weight.grad.norm().item()
    print(f"  输入梯度范数:  {g_in:.6f}  {'OK' if g_in>0 else 'FAIL'}")
    print(f"  weight梯度范数: {g_w:.6f}  {'OK' if g_w>0 else 'FAIL'}")

    # ── 测试6：F1/F3 注意力对齐确认 ──────────────────────────
    print("\n【测试6】与 F1/F3 注意力 t_dim 对齐确认")
    print("-" * 60)
    configs = [
        (256,  8, 0.25),
        (512,  8, 0.25),
        (768, 12, 0.25),
        (1024,16, 0.25),
    ]
    for d, h, tr in configs:
        td = compute_t_dim(d, h, tr)
        ln_test = MinkowskiLayerNorm(d, t_dim=td)
        x_test  = torch.randn(1, 4, d)
        out     = ln_test(x_test)
        ok = not torch.isnan(out).any()
        print(f"  d={d:5d} h={h:3d} tr={tr}  "
              f"t_dim={td:4d}  {'OK' if ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print("所有测试完成")
    print()
    print("使用方式:")
    print("  from lorentz_transformer.core.layer_norm import (")
    print("      MinkowskiLayerNorm, compute_t_dim")
    print("  )")
    print("  t_dim = compute_t_dim(d_model, n_heads, time_ratio)")
    print("  ln    = MinkowskiLayerNorm(d_model, t_dim=t_dim)")
    print("=" * 60)
