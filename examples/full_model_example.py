"""Minimal Transformer-style block built from packaged Lorentz components."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from lorentz_transformer import LorentzMultiHeadAttention, MinkowskiLayerNorm


@dataclass
class Config:
    d_model: int = 96
    n_heads: int = 6
    lorentz_alpha: float = 0.25
    dropout: float = 0.1


class LorentzBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention = LorentzMultiHeadAttention(config)
        self.norm = MinkowskiLayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x)
        return self.norm(x + attn_out)


config = Config()
block = LorentzBlock(config)
mask = torch.tensor([idx % 2 == 0 for idx in range(config.d_model)])
block.attention.set_timelike_mask(mask)
block.norm.set_timelike_mask(mask)

x = torch.randn(2, 8, config.d_model)
output = block(x)
print("block output:", output.shape)
