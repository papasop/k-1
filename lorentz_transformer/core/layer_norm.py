"""
lorentz_transformer/core/layer_norm.py

MinkowskiLayerNorm — Minkowski 几何层归一化

三个变体:
  MinkowskiLayerNormOptimized  : 纯 L2，消融基线（tracks mask but ignores it）
  MinkowskiLayerNormImproved   : per-element fallback，推荐变体
  MinkowskiLayerNormStable     : batch-wide fallback，可配置阈值

MinkowskiLayerNorm = MinkowskiLayerNormImproved （向后兼容别名）

核心公式（有 mask 时）:
  mink_sq = |||x_s||² - ||x_t||²|    (绝对 Minkowski 内积)
  x_out   = x / sqrt(mink_sq + ε)

辅助函数 compute_t_dim 用于与注意力层对齐 t_dim（F1/F3 接口兼容）。
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
    计算与 F1/F3 注意力层对齐的 t_dim。

    Args:
        d_model    : 模型维度
        n_heads    : 注意力头数
        time_ratio : 时间头比例（与注意力层相同）

    Returns:
        t_dim: LayerNorm 应使用的时间维度数

    Example:
        >>> t_dim = compute_t_dim(256, 8, 0.25)   # → 64
    """
    head_dim  = d_model // n_heads
    n_t_heads = max(1, int(n_heads * time_ratio))
    return n_t_heads * head_dim


# ============================================================================
# 基类
# ============================================================================

class _BaseMinkowskiLayerNorm(nn.Module):
    """Minkowski LayerNorm 基类。提供 mask 管理和共享工具方法。"""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        use_mean_shift: bool = False,
        t_dim: Optional[int] = None,
    ):
        """
        Args:
            d_model           : 模型维度
            eps               : 数值稳定常数
            elementwise_affine: 是否学习 weight/bias
            use_mean_shift    : 是否在归一化前先对最后一维去均值
            t_dim             : 类时维度数（前 t_dim 个维度视为类时）。
                                等价于调用 set_timelike_mask，方便与
                                compute_t_dim / F1/F3 注意力层对齐。
        """
        super().__init__()
        self.d_model        = d_model
        self.eps            = eps
        self.use_mean_shift = use_mean_shift
        self._has_mask      = False

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(d_model))
            self.bias   = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_buffer("weight", torch.ones(d_model))
            self.register_buffer("bias",   torch.zeros(d_model))

        # 类时 mask：哪些维度是时间维度
        self.register_buffer(
            "timelike_mask",
            torch.zeros(d_model, dtype=torch.bool),
        )

        if t_dim is not None:
            if not (0 < t_dim < d_model):
                raise ValueError(
                    f"t_dim must satisfy 0 < t_dim < d_model={d_model}, got {t_dim}"
                )
            mask = torch.zeros(d_model, dtype=torch.bool)
            mask[:t_dim] = True
            self.set_timelike_mask(mask)

    # ------------------------------------------------------------------
    # Mask 管理
    # ------------------------------------------------------------------

    def _validate_mask(self, mask: MaskLike) -> torch.Tensor:
        """验证并转换 mask 为 bool 张量。"""
        if isinstance(mask, torch.Tensor):
            if mask.ndim != 1:
                raise ValueError(
                    f"Expected 1D timelike mask, got shape {tuple(mask.shape)}"
                )
            mask_t = mask.bool()
        else:
            mask_t = torch.tensor(list(mask), dtype=torch.bool)

        if mask_t.shape != (self.d_model,):
            raise ValueError(
                f"Expected timelike mask shape ({self.d_model},), "
                f"got {tuple(mask_t.shape)}"
            )
        return mask_t

    def set_timelike_mask(self, mask: MaskLike) -> None:
        """设置类时维度 mask。"""
        mask_t = self._validate_mask(mask)
        self.timelike_mask.copy_(mask_t)
        self._has_mask = bool(mask_t.any().item())

    # ------------------------------------------------------------------
    # 输入验证
    # ------------------------------------------------------------------

    def _validate_input(
        self, x: torch.Tensor
    ):
        """验证输入形状，可选去均值，返回 (x_flat, original_shape)。"""
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected last dimension {self.d_model}, got {x.shape[-1]}"
            )
        original_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)
        if self.use_mean_shift:
            x_flat = x_flat - x_flat.mean(dim=-1, keepdim=True)
        return x_flat, original_shape

    # ------------------------------------------------------------------
    # 范数计算
    # ------------------------------------------------------------------

    def _minkowski_norm_sq(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 |spacelike² - timelike²|（绝对 Minkowski 内积平方）。

        无 mask 时：全维度为类空，等同于 L2²。
        有 mask 时：mask[i]=True → 第 i 维为类时。

        Returns: (N, 1) 张量
        """
        mask = self.timelike_mask
        t_sq = (x[..., mask] ** 2).sum(-1, keepdim=True)
        s_sq = (x[..., ~mask] ** 2).sum(-1, keepdim=True)
        return (s_sq - t_sq).abs()

    def _l2_norm_sq(self, x: torch.Tensor) -> torch.Tensor:
        """标准 L2 范数平方。"""
        return (x ** 2).sum(-1, keepdim=True)

    def _apply_affine(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


# ============================================================================
# MinkowskiLayerNormOptimized — 纯 L2（消融基线）
# ============================================================================

class MinkowskiLayerNormOptimized(_BaseMinkowskiLayerNorm):
    """
    纯 L2 归一化。接受 set_timelike_mask 调用（跟踪 _has_mask）但始终使用 L2。

    适合消融实验（纯 L2 vs Minkowski 对比）。
    """

    recommended_variant: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat, original_shape = self._validate_input(x)
        norm_sq    = self._l2_norm_sq(x_flat)
        normalized = x_flat / torch.sqrt(norm_sq + self.eps)
        return self._apply_affine(normalized).reshape(original_shape)


# ============================================================================
# MinkowskiLayerNormImproved — per-element fallback（推荐）
# ============================================================================

class MinkowskiLayerNormImproved(_BaseMinkowskiLayerNorm):
    """
    带 per-element fallback 的 Minkowski 归一化（推荐默认变体）。

    - 无 mask：L2 归一化
    - 有 mask：mink_sq ≥ ε → Minkowski；mink_sq < ε → 回退到 L2（per-element）
    """

    recommended_variant: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat, original_shape = self._validate_input(x)

        if not self._has_mask:
            norm_sq = self._l2_norm_sq(x_flat)
        else:
            mink_sq = self._minkowski_norm_sq(x_flat)
            l2_sq   = self._l2_norm_sq(x_flat)
            is_valid = mink_sq >= self.eps
            norm_sq  = torch.where(is_valid, mink_sq, l2_sq)

        normalized = x_flat / torch.sqrt(norm_sq + self.eps)
        return self._apply_affine(normalized).reshape(original_shape)


# ============================================================================
# MinkowskiLayerNormStable — batch-wide fallback
# ============================================================================

class MinkowskiLayerNormStable(_BaseMinkowskiLayerNorm):
    """
    带 batch-wide fallback 的 Minkowski 归一化。

    若批次内退化向量（|mink_sq| < ε）比例 > minkowski_fallback_threshold，
    则整个批次回退到 L2 归一化。

    Args:
        minkowski_fallback_threshold : 退化比例阈值，超过则 batch-wide 回退（默认 0.5）
        use_minkowski                : False 时强制使用 L2（默认 True）
    """

    recommended_variant: bool = False

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        use_mean_shift: bool = False,
        minkowski_fallback_threshold: float = 0.5,
        use_minkowski: bool = True,
        t_dim: Optional[int] = None,
    ):
        super().__init__(d_model, eps, elementwise_affine, use_mean_shift, t_dim)
        self.minkowski_fallback_threshold = minkowski_fallback_threshold
        self.use_minkowski                = use_minkowski
        self._fallback_count = 0
        self._total_count    = 0

    @property
    def fallback_ratio(self) -> float:
        """上次 forward 中回退到 L2 的向量比例（诊断用）。"""
        if self._total_count == 0:
            return 0.0
        return self._fallback_count / self._total_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat, original_shape = self._validate_input(x)

        if not self._has_mask or not self.use_minkowski:
            norm_sq = self._l2_norm_sq(x_flat)
        else:
            mink_sq = self._minkowski_norm_sq(x_flat)
            is_degenerate    = mink_sq.squeeze(-1) < self.eps
            degenerate_ratio = is_degenerate.float().mean()

            self._fallback_count = int(is_degenerate.sum().item())
            self._total_count    = is_degenerate.numel()

            if degenerate_ratio > self.minkowski_fallback_threshold:
                norm_sq = self._l2_norm_sq(x_flat)
            else:
                norm_sq = mink_sq

        normalized = x_flat / torch.sqrt(norm_sq + self.eps)
        return self._apply_affine(normalized).reshape(original_shape)


# ============================================================================
# 向后兼容别名
# ============================================================================

MinkowskiLayerNorm = MinkowskiLayerNormImproved


__all__ = [
    "MinkowskiLayerNorm",
    "MinkowskiLayerNormImproved",
    "MinkowskiLayerNormStable",
    "MinkowskiLayerNormOptimized",
    "_BaseMinkowskiLayerNorm",
    "MaskLike",
    "compute_t_dim",
]
