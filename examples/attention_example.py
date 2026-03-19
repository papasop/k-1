"""Standalone example for LorentzMultiHeadAttention."""

from dataclasses import dataclass

import torch

from lorentz_transformer import LorentzMultiHeadAttention


@dataclass
class Config:
    d_model: int = 64
    num_heads: int = 4
    lorentz_alpha: float = 0.2
    dropout: float = 0.0


config = Config()
attention = LorentzMultiHeadAttention(config)
timelike_mask = torch.tensor(
    [idx < config.d_model // 4 for idx in range(config.d_model)]
)
attention.set_timelike_mask(timelike_mask)

x = torch.randn(1, 6, config.d_model)
output, weights = attention(x)

print("attention output:", output.shape)
print("attention weights:", weights.shape)
