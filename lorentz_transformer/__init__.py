"""
lorentz_transformer/__init__.py

主包的初始化文件
"""

__version__ = "1.0.0"
__author__ = "papasop"

from . import core
from .core import (
    LorentzMultiHeadAttention,
    MinkowskiLayerNorm,
    compute_dt2_info,
    hutchinson_diag_hessian,
)

__all__ = [
    "core",
    "LorentzMultiHeadAttention",
    "MinkowskiLayerNorm",
    "compute_dt2_info",
    "hutchinson_diag_hessian",
]
