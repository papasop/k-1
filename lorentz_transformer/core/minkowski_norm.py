"""
lorentz_transformer/core/minkowski_norm.py

MinkowskiLayerNorm — Minkowski 几何归一化

核心公式:
  <x,x>_η = ||x_s||² - ||x_t||²    (Minkowski 内积)
  x_norm   = x / sqrt(|<x,x>_η| + ε)

支持两种类时维度指定方式:
  1. set_timelike_mask(mask): 布尔 mask，True=类时维度
  2. t_dim 参数: 前 t_dim 维为类时（与注意力层对齐）

t_dim 对齐规则 (与注意力层配合使用):
  t_dim = int(n_heads * time_ratio) * head_dim
        = int(n_heads * time_ratio) * (d_model // n_heads)

  示例: d_model=256, n_heads=8, time_ratio=0.25
    n_t_heads  = max(1, int(8 * 0.25)) = 2
    head_dim   = 256 // 8 = 32
    t_dim      = 2 * 32 = 64
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
        use_mean_shift: bool = False,
    ):
        """
        Args:
            d_model           : 模型维度
            t_dim             : 时间子空间维度（与注意力层对齐，前 t_dim 维为类时）
                                None 时退化为 L2 归一化（除非调用了 set_timelike_mask）
            eps               : 数值稳定常数
            elementwise_affine: 是否学习 weight/bias
            use_mean_shift    : 归一化前是否对最后一维去均值
        """
        super().__init__()
        self.d_model       = d_model
        self.t_dim         = t_dim
        self.eps           = eps
        self.use_mean_shift = use_mean_shift

        # 类时 mask（通过 set_timelike_mask 设置）
        self._timelike_mask: Optional[torch.Tensor] = None
        self._has_mask: bool = False

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(d_model))
            self.bias   = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_buffer("weight", torch.ones(d_model))
            self.register_buffer("bias",   torch.zeros(d_model))

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def set_timelike_mask(self, mask: MaskLike) -> None:
        """
        设置类时维度 mask。

        Args:
            mask: 长度为 d_model 的布尔序列，True 表示该维度为类时维度。
                  接受 list、tuple 或 torch.Tensor。

        Raises:
            ValueError: mask 形状与 d_model 不匹配。
        """
        validated = self._validate_mask(mask)
        self._timelike_mask = validated
        self._has_mask = bool(validated.any().item())

    @property
    def timelike_mask(self) -> Optional[torch.Tensor]:
        """当前类时掩码（只读属性）。"""
        return self._timelike_mask

    # ------------------------------------------------------------------
    # 验证工具
    # ------------------------------------------------------------------

    def _validate_input(
        self, x: torch.Tensor
    ):
        """
        验证输入并展平为 (N, d_model)。

        Returns:
            x_flat      : (N, d_model) 张量
            original_shape: 原始形状

        Raises:
            ValueError: 最后一维 != d_model
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected last dimension {self.d_model}, "
                f"got {x.shape[-1]}"
            )
        original_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)
        if self.use_mean_shift:
            x_flat = x_flat - x_flat.mean(dim=-1, keepdim=True)
        return x_flat, original_shape

    def _validate_mask(self, mask: MaskLike) -> torch.Tensor:
        """
        验证并转换 mask 为 bool 张量。

        Returns:
            torch.Tensor of shape (d_model,) and dtype=torch.bool

        Raises:
            ValueError: mask 形状不是 (d_model,)
        """
        if isinstance(mask, torch.Tensor):
            mask_t = mask.bool()
        else:
            mask_t = torch.tensor(list(mask), dtype=torch.bool)

        if mask_t.ndim != 1 or mask_t.shape[0] != self.d_model:
            raise ValueError(
                f"Expected timelike mask shape ({self.d_model},), "
                f"got {tuple(mask_t.shape)}"
            )
        return mask_t

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _minkowski_norm_sq(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 |<x,x>_η| = |s² - t²|（Minkowski 内积的绝对值）。

        优先级: 1) _timelike_mask (若已设置), 2) t_dim (若已设置), 3) L2_sq

        Args:
            x: (N, d_model)

        Returns:
            (N, 1) 张量，值为非负实数
        """
        if self._has_mask and self._timelike_mask is not None:
            mask = self._timelike_mask.to(x.device)   # (d_model,)
            t_sq = (x * mask.float()).pow(2).sum(-1, keepdim=True)
            s_sq = (x * (~mask).float()).pow(2).sum(-1, keepdim=True)
            return (s_sq - t_sq).abs()
        elif self.t_dim is not None:
            t_sq = x[..., :self.t_dim].pow(2).sum(-1, keepdim=True)
            s_sq = x[..., self.t_dim:].pow(2).sum(-1, keepdim=True)
            return (s_sq - t_sq).abs()
        else:
            # 没有类时信息 → 全部视为类空，等价于 L2²
            return x.pow(2).sum(-1, keepdim=True)

    def _l2_norm_sq(self, x: torch.Tensor) -> torch.Tensor:
        """L2 范数平方，形状 (N, 1)。"""
        return x.pow(2).sum(-1, keepdim=True)

    def _apply_affine(self, x: torch.Tensor) -> torch.Tensor:
        """应用仿射变换 weight * x + bias。"""
        return x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


# ============================================================================
# MinkowskiLayerNormImproved（推荐默认）— 逐样本回退
# ============================================================================

class MinkowskiLayerNormImproved(_BaseMinkowskiLayerNorm):
    """
    Minkowski 归一化（推荐版本）。

    公式:
        mink_sq = |s² - t²|
        norm_sq = mink_sq  if mink_sq > eps  (per-sample)
                = L2_sq    otherwise (fallback to L2)
        x_out   = x / sqrt(norm_sq + eps) * weight + bias

    特点:
        - 逐样本回退: 近光锥样本退化到 L2，其余用 Minkowski
        - 支持 set_timelike_mask（mask 优先）和 t_dim（构造参数）两种模式

    recommended_variant = True
    """

    recommended_variant: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., d_model)
        Returns:
            归一化张量，形状与 x 相同
        """
        x_flat, shape = self._validate_input(x)

        if self._has_mask or self.t_dim is not None:
            mink_sq = self._minkowski_norm_sq(x_flat)   # (N,1), ≥ 0
            l2_sq   = self._l2_norm_sq(x_flat)
            # 逐样本: 接近光锥时退化到 L2
            use_l2  = mink_sq <= self.eps
            norm_sq = torch.where(use_l2, l2_sq, mink_sq)
        else:
            norm_sq = self._l2_norm_sq(x_flat)

        norm   = torch.sqrt(norm_sq + self.eps)
        x_norm = x_flat / norm
        return self._apply_affine(x_norm).reshape(shape)


# ============================================================================
# MinkowskiLayerNorm — 向后兼容别名
# ============================================================================

# MinkowskiLayerNorm 指向推荐变体
MinkowskiLayerNorm = MinkowskiLayerNormImproved


# ============================================================================
# MinkowskiLayerNormStable — 批次级回退
# ============================================================================

class MinkowskiLayerNormStable(_BaseMinkowskiLayerNorm):
    """
    稳健的 Minkowski 归一化，支持批次级回退。

    与 Improved 的区别:
        Improved: 逐样本决定是否回退到 L2
        Stable:   若批次中近光锥样本比例超过阈值，整个批次回退到 L2

    Args:
        d_model                   : 模型维度
        t_dim                     : 时间维度（可选）
        eps                       : 数值稳定常数
        elementwise_affine        : 是否学习 weight/bias
        use_mean_shift            : 是否去均值
        minkowski_fallback_threshold: 触发批次级回退的近光锥样本比例阈值（0~1）
                                      若 ratio > threshold → 整批回退到 L2
        use_minkowski             : 为 False 时始终使用 L2

    recommended_variant = False
    """

    recommended_variant: bool = False

    def __init__(
        self,
        d_model: int,
        t_dim: Optional[int] = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        use_mean_shift: bool = False,
        minkowski_fallback_threshold: float = 0.5,
        use_minkowski: bool = True,
    ):
        super().__init__(d_model, t_dim, eps, elementwise_affine, use_mean_shift)
        self.minkowski_fallback_threshold = minkowski_fallback_threshold
        self.use_minkowski = use_minkowski

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat, shape = self._validate_input(x)

        if not self.use_minkowski or (not self._has_mask and self.t_dim is None):
            # 不使用 Minkowski 或没有类时信息 → L2
            norm = torch.sqrt(self._l2_norm_sq(x_flat) + self.eps)
            x_norm = x_flat / norm
            return self._apply_affine(x_norm).reshape(shape)

        mink_sq = self._minkowski_norm_sq(x_flat)   # (N,1), ≥ 0
        l2_sq   = self._l2_norm_sq(x_flat)

        # 批次级回退判断
        near_lightlike = (mink_sq <= self.eps)       # (N,1) bool
        ratio = near_lightlike.float().mean()
        if ratio > self.minkowski_fallback_threshold:
            # 整批使用 L2
            norm_sq = l2_sq
        else:
            # 整批使用 Minkowski
            norm_sq = mink_sq

        norm   = torch.sqrt(norm_sq + self.eps)
        x_norm = x_flat / norm
        return self._apply_affine(x_norm).reshape(shape)


# ============================================================================
# MinkowskiLayerNormOptimized — 纯 L2（兼容旧接口）
# ============================================================================

class MinkowskiLayerNormOptimized(_BaseMinkowskiLayerNorm):
    """
    纯 L2 归一化，兼容 set_timelike_mask 接口。

    始终使用 L2 范数，不应用 Minkowski 几何。
    适合消融实验基线（对比 Minkowski 变体）。

    recommended_variant = False
    """

    recommended_variant: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat, shape = self._validate_input(x)
        norm   = torch.sqrt(self._l2_norm_sq(x_flat) + self.eps)
        x_norm = x_flat / norm
        return self._apply_affine(x_norm).reshape(shape)


# ============================================================================
# 公共导出
# ============================================================================

__all__ = [
    "MinkowskiLayerNorm",
    "MinkowskiLayerNormImproved",
    "MinkowskiLayerNormStable",
    "MinkowskiLayerNormOptimized",
    "_BaseMinkowskiLayerNorm",
    "MaskLike",
    "compute_t_dim",
]
