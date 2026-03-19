import unittest
from dataclasses import dataclass

import torch

from lorentz_transformer import (
    LorentzMultiHeadAttention,
    compute_dt2_info,
    hutchinson_diag_hessian,
)


@dataclass
class Config:
    d_model: int = 256
    n_heads: int = 8
    lorentz_alpha: float = 0.0
    dropout: float = 0.0


class LorentzMultiHeadAttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(2, 16, 256)

    def test_forward_matches_expected_shapes(self) -> None:
        attn = LorentzMultiHeadAttention(Config())

        output, weights = attn(self.x)

        self.assertEqual(output.shape, self.x.shape)
        self.assertEqual(weights.shape, (2, 8, 16, 16))

    def test_timelike_mask_enables_lorentz_correction(self) -> None:
        attn_std = LorentzMultiHeadAttention(Config())
        attn_lorentz = LorentzMultiHeadAttention(Config(lorentz_alpha=0.25))
        attn_lorentz.load_state_dict(attn_std.state_dict())

        mask = torch.zeros(256, dtype=torch.bool)
        mask[:32] = True
        attn_lorentz.set_timelike_mask(mask)

        output_std, _ = attn_std(self.x)
        output_lorentz, _ = attn_lorentz(self.x)

        self.assertFalse(torch.allclose(output_std, output_lorentz))
        self.assertTrue(attn_lorentz._has_mask)

    def test_attention_mask_blocks_future_positions(self) -> None:
        attn = LorentzMultiHeadAttention(Config())
        causal_mask = torch.triu(
            torch.full((16, 16), float("-inf")), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        _, weights = attn(self.x, causal_mask)

        self.assertTrue(torch.allclose(weights[0, 0, 0, 1:], torch.zeros(15)))

    def test_compute_dt2_info_returns_scalar(self) -> None:
        attn = LorentzMultiHeadAttention(Config())
        _, weights = attn(self.x)

        dt2_info = compute_dt2_info(weights)

        self.assertEqual(dt2_info.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(dt2_info))

    def test_hutchinson_diag_hessian_matches_parameter_shape(self) -> None:
        param = torch.nn.Parameter(torch.randn(8, 8))

        def loss_fn():
            return (param ** 2).sum()

        diag = hutchinson_diag_hessian(loss_fn, param, n_samples=4)

        self.assertEqual(diag.shape, param.shape)
        self.assertTrue(torch.isfinite(diag).all())


if __name__ == "__main__":
    unittest.main()
