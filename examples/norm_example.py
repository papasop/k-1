"""Standalone example for Minkowski layer normalization variants."""

import torch

from lorentz_transformer import (
    MinkowskiLayerNorm,
    MinkowskiLayerNormOptimized,
    MinkowskiLayerNormStable,
)


x = torch.randn(2, 5, 32)
mask = torch.tensor([idx % 3 == 0 for idx in range(32)])

layers = {
    "default": MinkowskiLayerNorm(32),
    "optimized": MinkowskiLayerNormOptimized(32),
    "stable": MinkowskiLayerNormStable(32),
}

for name, layer in layers.items():
    layer.set_timelike_mask(mask)
    output = layer(x)
    print(name, output.shape, torch.isnan(output).any().item())
