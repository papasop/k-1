"""
lorentz_transformer/core/__init__.py

核心注意力组件导出。
"""

from .attention import (
    LorentzMultiHeadAttention,
    compute_dt2_info,
    hutchinson_diag_hessian,
)

__all__ = [
    "LorentzMultiHeadAttention",
    "compute_dt2_info",
    "hutchinson_diag_hessian",
]
