"""Minimal Transformer-style block built from packaged Lorentz components."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from lorentz_transformer import (
    LorentzMultiHeadAttention,
    MinkowskiLayerNorm,
    compute_t_dim,
)


@dataclass
class Config:
    d_model: int = 96
    n_heads: int = 6
    formula: str = 'f3'
    time_ratio: float = 0.25
    dropout: float = 0.1


class LorentzBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention = LorentzMultiHeadAttention(config)
        t_dim = compute_t_dim(config.d_model, config.n_heads, config.time_ratio)
        self.norm = MinkowskiLayerNorm(config.d_model, t_dim=t_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x)
        return self.norm(x + attn_out)


config = Config()
block = LorentzBlock(config)

x = torch.randn(2, 8, config.d_model)
output = block(x)
print("block output:", output.shape)
