from dataclasses import dataclass

import torch

from lorentz_transformer import (
    LorentzMultiHeadAttention,
    MinkowskiLayerNorm,
    MinkowskiLayerNormImproved,
)


@dataclass
class MockConfig:
    d_model: int = 64
    n_heads: int = 4
    lorentz_alpha: float = 0.25
    dropout: float = 0.0


def test_attention_and_minkowski_norm_work_together():
    config = MockConfig()
    attn = LorentzMultiHeadAttention(config)
    norm = MinkowskiLayerNorm(config.d_model)

    timelike_mask = torch.tensor(
        [idx % 2 == 0 for idx in range(config.d_model)]
    )
    attn.set_timelike_mask(timelike_mask)
    norm.set_timelike_mask(timelike_mask)

    x = torch.randn(2, 10, config.d_model, requires_grad=True)
    attn_out, attn_weights = attn(x)
    output = norm(attn_out)
    loss = output.square().mean() + attn_weights.square().mean()
    loss.backward()

    assert output.shape == (2, 10, config.d_model)
    assert attn_weights.shape == (2, config.n_heads, 10, 10)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_default_minkowski_export_points_to_recommended_variant():
    assert MinkowskiLayerNorm is MinkowskiLayerNormImproved
