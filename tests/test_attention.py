import math
import unittest
from dataclasses import dataclass

import torch

from lorentz_transformer.core import (
    LorentzMultiHeadAttention,
    compute_dt2_info,
    hutchinson_diag_hessian,
)


@dataclass
class Config:
    d_model: int = 16
    n_heads: int = 4
    lorentz_alpha: float = 0.0
    dropout: float = 0.0


class LorentzAttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(2, 5, 16)

    def test_forward_shape_matches_standard_attention_path(self) -> None:
        attn = LorentzMultiHeadAttention(Config())
        out, weights = attn(self.x)

        self.assertEqual(out.shape, self.x.shape)
        self.assertEqual(weights.shape, (2, 4, 5, 5))

        q = attn.q_proj(self.x).view(2, 5, 4, 4).transpose(1, 2).float()
        k = attn.k_proj(self.x).view(2, 5, 4, 4).transpose(1, 2).float()
        v = attn.v_proj(self.x).view(2, 5, 4, 4).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(4)
        expected_weights = torch.softmax(scores, dim=-1).to(self.x.dtype)
        expected_out = torch.matmul(expected_weights, v)
        expected_out = expected_out.transpose(1, 2).contiguous().view(2, 5, 16)
        expected_out = attn.o_proj(expected_out)

        self.assertTrue(torch.allclose(weights, expected_weights, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(out, expected_out, atol=1e-6, rtol=1e-5))

    def test_timelike_mask_changes_scores_when_enabled(self) -> None:
        attn = LorentzMultiHeadAttention(Config(lorentz_alpha=0.5))
        mask = torch.zeros(16, dtype=torch.bool)
        mask[:4] = True
        attn.set_timelike_mask(mask)

        _, weights_lorentz = attn(self.x)

        baseline = LorentzMultiHeadAttention(Config(lorentz_alpha=0.0))
        baseline.load_state_dict(attn.state_dict(), strict=False)
        _, weights_std = baseline(self.x)

        self.assertFalse(torch.allclose(weights_lorentz, weights_std))
        self.assertTrue(attn._has_mask)

    def test_additive_attention_mask_blocks_future_positions(self) -> None:
        attn = LorentzMultiHeadAttention(Config())
        causal_mask = torch.triu(
            torch.full((5, 5), float("-inf")),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        _, weights = attn(self.x, causal_mask)

        blocked = torch.triu(weights[0, 0], diagonal=1)
        self.assertTrue(torch.allclose(blocked, torch.zeros_like(blocked), atol=1e-6))

    def test_compute_dt2_info_returns_finite_scalar(self) -> None:
        weights = torch.full((2, 4, 5, 5), 0.2)
        dt2_info = compute_dt2_info(weights)

        self.assertEqual(dt2_info.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(dt2_info))

    def test_hutchinson_diag_hessian_returns_parameter_shaped_tensor(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0, -2.0], requires_grad=True))

        def loss_fn() -> torch.Tensor:
            return (param[0] ** 2) + (3.0 * param[1] ** 2)

        diag = hutchinson_diag_hessian(loss_fn, param, n_samples=8)

        self.assertEqual(diag.shape, param.shape)
        self.assertTrue(torch.isfinite(diag).all())


if __name__ == "__main__":
    unittest.main()
