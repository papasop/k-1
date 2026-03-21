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

# Re-export everything from minkowski_norm for backwards compatibility.
# minkowski_norm.py is the canonical implementation; layer_norm.py is
# the public-facing module name documented in the README.
from .minkowski_norm import (  # noqa: F401
    MinkowskiLayerNorm,
    MinkowskiLayerNormImproved,
    MinkowskiLayerNormOptimized,
    MinkowskiLayerNormStable,
    _BaseMinkowskiLayerNorm,
    MaskLike,
    compute_t_dim,
)

__all__ = [
    "MinkowskiLayerNorm",
    "MinkowskiLayerNormStable",
    "MinkowskiLayerNormOptimized",
    "MinkowskiLayerNormImproved",
    "compute_t_dim",
]
