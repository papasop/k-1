"""Quick example of Lorentz attention followed by Minkowski layer norm."""

from dataclasses import dataclass

import torch

from lorentz_transformer import LorentzMultiHeadAttention, MinkowskiLayerNorm


@dataclass
class Config:
    d_model: int = 128
    n_heads: int = 8
    lorentz_alpha: float = 0.25
    dropout: float = 0.1


config = Config()
attention = LorentzMultiHeadAttention(config)
normalization = MinkowskiLayerNorm(config.d_model)

x = torch.randn(2, 12, config.d_model)
attn_out, attn_weights = attention(x)
output = normalization(attn_out)

print("output:", output.shape)
print("weights:", attn_weights.shape)
