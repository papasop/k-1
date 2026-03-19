import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_dt2_info(attn_w: torch.Tensor) -> torch.Tensor:
    """
    计算K=1信息时间度量。

    dt²_info = Σ_q K_q = Σ_q (Φ_q / H_q)

    其中：
      H_q = Shannon熵（注意力分布的不确定性）
      Φ_q = Σ_j a_qj² （注意力集中度）
      K_q = Φ_q/H_q （信息时间密度）

    该量用于诊断注意力的信息几何性质。

    Args:
        attn_w (torch.Tensor): 注意力权重，形状 (B, H, L, L)
            其中 B=batch_size, H=num_heads, L=seq_len

    Returns:
        torch.Tensor: scalar，所有query位置的K_q均值
    """
    aw = attn_w.mean(dim=1).float()
    aw = torch.nan_to_num(aw, nan=0.0)
    aw = aw / aw.sum(-1, keepdim=True).clamp(min=1e-8)

    H = -(aw * torch.log(aw + 1e-8)).sum(-1)
    Phi = (aw ** 2).sum(-1)
    H = H.clamp(min=1e-8)

    K = Phi / H
    return K.mean()


def hutchinson_diag_hessian(
    loss_fn,
    param: nn.Parameter,
    n_samples: int = 20,
) -> torch.Tensor:
    """
    用Hutchinson方法估计Hessian的对角元素。

    用于识别参数空间中的类时方向（dt²_info的凹方向）。

    数学原理：
      G_ii ≈ (1/K) Σ_k v_k[i] · (H·v_k)[i]

      其中：
        v_k ~ Rademacher{±1} (随机符号向量)
        H·v_k = ∂²loss/∂param² · v_k (Hessian-向量乘积)
    """
    G = torch.zeros_like(param.data)

    for _ in range(n_samples):
        v = (
            torch.randint(0, 2, param.shape, device=param.device) * 2 - 1
        ).float()

        loss = loss_fn()
        g1 = torch.autograd.grad(
            loss, param, create_graph=True, retain_graph=True
        )[0]

        if g1 is None:
            continue

        Hv = torch.autograd.grad(
            (g1 * v.detach()).sum(), param, retain_graph=False
        )[0]

        if Hv is None:
            continue

        Hv = torch.nan_to_num(Hv, nan=0.0, posinf=0.0, neginf=0.0)
        G = G + (v * Hv).detach() / n_samples

    return G


class LorentzMultiHeadAttention(nn.Module):
    """
    闵可夫斯基多头注意力机制。

    核心计算：
      scores_L = Q · η · K^T / √d_h
      其中 η = I - 2α·P_t (Minkowski符号矩阵)

    展开形式：
      scores_L = (QK^T)/√d_h - 2α·(Q_t K^T)/√d_h
               = scores_std - 2α × 时间内积
    """

    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.alpha = config.lorentz_alpha

        assert (
            config.d_model % config.n_heads == 0
        ), f"d_model={config.d_model} 必须被 n_heads={config.n_heads} 整除"

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.drop = nn.Dropout(config.dropout)

        self.register_buffer(
            "timelike_mask",
            torch.zeros(config.d_model, dtype=torch.bool),
        )

        self._has_mask = False
        self.last_intervals: Optional[torch.Tensor] = None
        self.last_intervals_raw: Optional[torch.Tensor] = None

    def set_timelike_mask(self, mask: torch.Tensor) -> None:
        """注入类时掩码（由TimeLikeProbe调用）。"""
        self.timelike_mask.copy_(mask.bool())
        self._has_mask = mask.any().item()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入特征，形状 (B, L, d_model)
            attention_mask (Optional[torch.Tensor]): 加性注意力掩码
                形状 (B, 1, 1, L) 或 (B, 1, L, L)
                pad位置设为-inf，有效位置设为0
        """
        B, L, _ = x.shape
        H, d_h = self.n_heads, self.head_dim
        scale = math.sqrt(d_h)

        Q = self.q_proj(x).view(B, L, H, d_h).transpose(1, 2).float()
        K = self.k_proj(x).view(B, L, H, d_h).transpose(1, 2).float()
        V = self.v_proj(x).view(B, L, H, d_h).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if self._has_mask and self.alpha > 0:
            mask_2d = self.timelike_mask.view(H, d_h).float()
            mask_bcast = mask_2d.unsqueeze(0).unsqueeze(2).to(Q.device)

            Q_t = Q * mask_bcast
            q_norm = Q.norm(dim=-1, keepdim=True)
            qt_norm = Q_t.norm(dim=-1, keepdim=True)

            has_timelike = qt_norm > 1e-6
            scale_factor = torch.where(
                has_timelike,
                q_norm / qt_norm.clamp(min=1e-8),
                torch.zeros_like(qt_norm),
            )
            Q_t_scaled = Q_t * scale_factor

            time_inner = torch.matmul(
                Q_t_scaled, K.transpose(-2, -1)
            ) / scale
            scores = scores - 2.0 * self.alpha * time_inner

        self.last_intervals_raw = scores.detach().clone()

        if attention_mask is not None:
            scores = scores + attention_mask.to(scores.dtype)

        self.last_intervals = scores.detach().clone()

        attn_w = F.softmax(scores, dim=-1).to(x.dtype)
        attn_w = self.drop(attn_w)

        out = torch.matmul(attn_w, V)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.o_proj(out)

        return out, attn_w

    def extra_repr(self) -> str:
        """模块的额外信息表示。"""
        return (
            f"n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, "
            f"alpha={self.alpha}, "
            f"has_mask={self._has_mask}"
        )
