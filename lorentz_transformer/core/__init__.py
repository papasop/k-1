"""
lorentz_transformer.core
Core modules for the Lorentz Transformer package.
"""
from .attention import (
    LorentzMultiHeadAttention,
    compute_dt2_info,
    hutchinson_diag_hessian,
)
from .layer_norm import (          # ← 从 minkowski_norm 改为 layer_norm
    MinkowskiLayerNorm,
    MinkowskiLayerNormImproved,
    MinkowskiLayerNormOptimized,
    MinkowskiLayerNormStable,
    compute_t_dim,                 # ← 新增
)

__all__ = [
    # 注意力
    "LorentzMultiHeadAttention",
    "compute_dt2_info",
    "hutchinson_diag_hessian",
    # LayerNorm
    "MinkowskiLayerNorm",
    "MinkowskiLayerNormImproved",
    "MinkowskiLayerNormOptimized",
    "MinkowskiLayerNormStable",
    "compute_t_dim",               # ← 新增
]
