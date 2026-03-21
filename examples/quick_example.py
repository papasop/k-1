"""Quick example of Lorentz attention followed by Minkowski layer norm."""

from dataclasses import dataclass

import torch

from lorentz_transformer import (
    LorentzMultiHeadAttention,
    MinkowskiLayerNorm,
    compute_t_dim,
)


@dataclass
class Config:
    d_model: int = 256
    n_heads: int = 8
    formula: str = 'f3'
    time_ratio: float = 0.25
    dropout: float = 0.1


config = Config()
attn  = LorentzMultiHeadAttention(config)
t_dim = compute_t_dim(config.d_model, config.n_heads, config.time_ratio)
norm  = MinkowskiLayerNorm(config.d_model, t_dim=t_dim)

x = torch.randn(2, 16, config.d_model)
attn_out, attn_weights = attn(x)
output = norm(attn_out)

print(output.shape)            # torch.Size([2, 16, 256])
print(f"σ = {attn.sigma:.3f}") # 光锥强度，训练中自适应收敛
