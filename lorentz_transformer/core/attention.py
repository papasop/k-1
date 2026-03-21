"""
lorentz_transformer/core/attention.py

Minkowski 多头注意力 — 三种公式统一接口

formula 选择指南:
  'f1' 显式分离  : 推理 / 数学 / 物理仿真 / 机器人轨迹
  'f2' 洛伦兹修正: 兼容旧版（有退化风险，不推荐新项目）
  'f3' 结合版    : 大语言模型 / 视频 / 科学文本（推荐默认）

核心公式:
  F1: score =  -Q_t Kt^T  +  Q_s Ks^T
  F2: score =  QK^T/sqrt(d) - 2a*Q_t_scaled*K^T/sqrt(d)
  F3: score =  -sigma*Q_t Kt^T  +  Q_s Ks^T  (sigma 可学习)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 辅助函数（保持与原版兼容）
# ============================================================================

def compute_dt2_info(attn_w: torch.Tensor) -> torch.Tensor:
    """
    计算 K=1 信息时间度量。
    dt2_info = mean(Phi_q / H_q)
    """
    aw = attn_w.mean(dim=1).float()
    aw = torch.nan_to_num(aw, nan=0.0)
    aw = aw / aw.sum(-1, keepdim=True).clamp(min=1e-8)
    H   = -(aw * torch.log(aw + 1e-8)).sum(-1)
    Phi = (aw ** 2).sum(-1)
    H   = H.clamp(min=1e-8)
    return (Phi / H).mean()


def hutchinson_diag_hessian(
    loss_fn,
    param: nn.Parameter,
    n_samples: int = 20,
) -> torch.Tensor:
    """Hutchinson 方法估计 Hessian 对角元素。"""
    G = torch.zeros_like(param.data)
    for _ in range(n_samples):
        v  = (torch.randint(0, 2, param.shape,
               device=param.device) * 2 - 1).float()
        loss = loss_fn()
        g1  = torch.autograd.grad(
            loss, param, create_graph=True, retain_graph=True)[0]
        if g1 is None:
            continue
        Hv = torch.autograd.grad(
            (g1 * v.detach()).sum(), param, retain_graph=False)[0]
        if Hv is None:
            continue
        Hv = torch.nan_to_num(Hv, nan=0.0, posinf=0.0, neginf=0.0)
        G  = G + (v * Hv).detach() / n_samples
    return G


# ============================================================================
# LorentzMultiHeadAttention
# ============================================================================

VALID_FORMULAS = ("f1", "f2", "f3")


class LorentzMultiHeadAttention(nn.Module):
    """
    Minkowski 多头注意力，支持三种公式。

    formula='f1'  显式时空分离，硬编码负号，不可退化
                  适用: 数学推理 / 代码生成 / 物理仿真 / 机器人轨迹

    formula='f2'  洛伦兹修正（兼容旧版）
                  适用: 加载已有 F2 权重时使用
                  注意: alpha 可能学不动，退化为欧氏

    formula='f3'  结合版，sigma 可学习（推荐默认）
                  适用: 大语言模型 / 视频理解 / 科学文本
                  特点: 不退化，sigma 自适应任务因果强度

    Args:
        config: 配置对象，需包含:
            d_model       (int)   : 模型维度
            n_heads       (int)   : 注意力头数（或 num_heads）
            formula       (str)   : 'f1'|'f2'|'f3'，默认 'f3'
            time_ratio    (float) : 时间头比例，默认 0.25（F1/F3）
            lorentz_alpha (float) : 修正强度，默认 0.25（F2）
            dropout       (float) : dropout，默认 0.0

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Cfg:
        ...     d_model: int = 256
        ...     n_heads: int = 8
        ...     formula: str = 'f3'
        ...     time_ratio: float = 0.25
        >>> attn = LorentzMultiHeadAttention(Cfg())
        >>> out, w = attn(torch.randn(2, 128, 256))
    """

    def __init__(self, config):
        super().__init__()

        if not hasattr(config, "d_model"):
            raise AttributeError("config must define d_model")

        n_heads = getattr(config, "n_heads",
                  getattr(config, "num_heads", None))
        if n_heads is None:
            raise AttributeError("config must define n_heads or num_heads")

        self.d_model  = int(config.d_model)
        self.n_heads  = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        dropout       = float(getattr(config, "dropout", 0.0))

        assert self.d_model % self.n_heads == 0, (
            f"d_model={self.d_model} 必须能被 n_heads={self.n_heads} 整除"
        )

        # 公式选择（默认 f2，保持向后兼容）
        self.formula = str(getattr(config, "formula", "f2")).lower()
        if self.formula not in VALID_FORMULAS:
            raise ValueError(
                f"formula must be one of {VALID_FORMULAS}, got '{self.formula}'"
            )

        # 时间/空间头数（F1 / F3）
        time_ratio = float(getattr(config, "time_ratio", 0.25))
        self.n_t   = max(1, int(self.n_heads * time_ratio))
        self.n_s   = self.n_heads - self.n_t
        self.t_dim = self.n_t * self.head_dim
        self.s_dim = self.n_s * self.head_dim

        # F2 超参数
        self.alpha = float(getattr(config, "lorentz_alpha", 0.25))

        # 投影层
        if self.formula == "f1":
            self.q_t = nn.Linear(self.d_model, self.t_dim, bias=False)
            self.k_t = nn.Linear(self.d_model, self.t_dim, bias=False)
            self.q_s = nn.Linear(self.d_model, self.s_dim, bias=False)
            self.k_s = nn.Linear(self.d_model, self.s_dim, bias=False)

        elif self.formula == "f2":
            self.q_proj   = nn.Linear(self.d_model, self.d_model, bias=False)
            self.k_proj   = nn.Linear(self.d_model, self.d_model, bias=False)
            self.q_t_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.register_buffer(
                "timelike_mask",
                torch.zeros(self.d_model, dtype=torch.bool),
            )

        else:  # f3
            self.q_t     = nn.Linear(self.d_model, self.t_dim, bias=False)
            self.k_t     = nn.Linear(self.d_model, self.t_dim, bias=False)
            self.q_s     = nn.Linear(self.d_model, self.s_dim, bias=False)
            self.k_s     = nn.Linear(self.d_model, self.s_dim, bias=False)
            # sigma=sigmoid(w_sigma) in (0,1), init 0 -> sigma=0.5
            self.w_sigma = nn.Parameter(torch.zeros(1))

        # _has_mask 对所有公式均有效
        self._has_mask = False

        # 共享：V + 输出 + Dropout
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.drop   = nn.Dropout(dropout)

        # 诊断
        self.last_intervals:     Optional[torch.Tensor] = None
        self.last_intervals_raw: Optional[torch.Tensor] = None

    def set_timelike_mask(self, mask: torch.Tensor) -> None:
        """
        注入类时掩码。

        F2 公式将 mask 用于 Lorentz 修正计算；
        F1/F3 公式记录 _has_mask 状态（mask 不影响内部计算）。
        接受 float/bool tensor 或布尔序列，自动转换为 bool。
        """
        mask_bool = mask.bool() if isinstance(mask, torch.Tensor) else torch.tensor(list(mask), dtype=torch.bool)
        self._has_mask = bool(mask_bool.any().item())
        if self.formula == "f2":
            self.timelike_mask.copy_(mask_bool)

    @property
    def sigma(self) -> Optional[float]:
        """F3 当前的 Minkowski 强度 sigma in (0,1)。"""
        if self.formula == "f3":
            return float(torch.sigmoid(self.w_sigma).item())
        return None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x              : (B, L, d_model)
            attention_mask : (B,1,1,L) 或 (B,1,L,L)，pad=-inf，有效=0

        Returns:
            output      : (B, L, d_model)
            attn_weights: (B, H, L, L)
        """
        B, L, _ = x.shape
        scale   = math.sqrt(self.head_dim)

        if self.formula == "f1":
            score = self._forward_f1(x, B, L, scale)
        elif self.formula == "f2":
            score = self._forward_f2(x, B, L, scale)
        else:
            score = self._forward_f3(x, B, L, scale)

        self.last_intervals_raw = score.detach().clone()

        if attention_mask is not None:
            score = score + attention_mask.to(score.dtype)

        self.last_intervals = score.detach().clone()

        attn_probs = F.softmax(score, dim=-1).to(x.dtype)
        attn_w     = self.drop(attn_probs)

        H, d_h = self.n_heads, self.head_dim
        V   = self.v_proj(x).view(B, L, H, d_h).transpose(1, 2)
        out = torch.matmul(attn_w, V)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out), attn_probs

    def _forward_f1(self, x, B, L, scale):
        """F1: -Q_t Kt^T + Q_s Ks^T, output shape (B, H, L, L)"""
        d_h = self.head_dim
        # (B, L, n_t, d_h)
        Qt = self.q_t(x).view(B, L, self.n_t, d_h)
        Kt = self.k_t(x).view(B, L, self.n_t, d_h)
        Qs = self.q_s(x).view(B, L, self.n_s, d_h)
        Ks = self.k_s(x).view(B, L, self.n_s, d_h)
        # Transpose to (B, H, L, d_h) for batched matmul
        Qt = Qt.permute(0, 2, 1, 3)   # (B, n_t, L, d_h)
        Kt = Kt.permute(0, 2, 1, 3)
        Qs = Qs.permute(0, 2, 1, 3)   # (B, n_s, L, d_h)
        Ks = Ks.permute(0, 2, 1, 3)
        st = torch.matmul(Qt, Kt.transpose(-2, -1))  # (B, n_t, L, L)
        ss = torch.matmul(Qs, Ks.transpose(-2, -1))  # (B, n_s, L, L)
        return torch.cat([-st, ss], dim=1) / scale   # (B, H, L, L)

    def _forward_f2(self, x, B, L, scale):
        """F2: QK^T/sqrt(d) - 2a*Q_t*K^T/sqrt(d)"""
        H, d_h = self.n_heads, self.head_dim
        Q = self.q_proj(x).view(B,L,H,d_h).transpose(1,2).float()
        K = self.k_proj(x).view(B,L,H,d_h).transpose(1,2).float()
        scores = torch.matmul(Q, K.transpose(-2,-1)) / scale

        if self._has_mask and self.alpha > 0:
            mask_2d = self.timelike_mask.view(H, d_h).float()
            mask_bc = mask_2d.unsqueeze(0).unsqueeze(2).to(Q.device)
            Q_t     = self.q_t_proj(x).view(B,L,H,d_h).transpose(1,2).float()
            Q_t     = Q_t * mask_bc
            t_inner = torch.matmul(Q_t, K.transpose(-2,-1)) / scale
            scores  = scores - 2.0 * self.alpha * t_inner

        return scores

    def _forward_f3(self, x, B, L, scale):
        """F3: -sigma*Q_t Kt^T + Q_s Ks^T, output shape (B, H, L, L)"""
        d_h = self.head_dim
        Qt = self.q_t(x).view(B, L, self.n_t, d_h).permute(0, 2, 1, 3)  # (B, n_t, L, d_h)
        Kt = self.k_t(x).view(B, L, self.n_t, d_h).permute(0, 2, 1, 3)
        Qs = self.q_s(x).view(B, L, self.n_s, d_h).permute(0, 2, 1, 3)  # (B, n_s, L, d_h)
        Ks = self.k_s(x).view(B, L, self.n_s, d_h).permute(0, 2, 1, 3)
        st    = torch.matmul(Qt, Kt.transpose(-2, -1))  # (B, n_t, L, L)
        ss    = torch.matmul(Qs, Ks.transpose(-2, -1))  # (B, n_s, L, L)
        sigma = torch.sigmoid(self.w_sigma)
        return torch.cat([-sigma * st, ss], dim=1) / scale  # (B, H, L, L)

    def extra_repr(self) -> str:
        base = (f"formula={self.formula}, "
                f"n_heads={self.n_heads}, head_dim={self.head_dim}, "
                f"alpha={self.alpha}")
        if self.formula == "f1":
            return base + f", n_t={self.n_t}, n_s={self.n_s}"
        elif self.formula == "f2":
            return base + f", has_mask={self._has_mask}"
        else:
            s = self.sigma
            return base + f", sigma={s:.3f}, n_t={self.n_t}, n_s={self.n_s}"


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class Cfg:
        d_model:       int   = 256
        n_heads:       int   = 8
        formula:       str   = "f3"
        time_ratio:    float = 0.25
        lorentz_alpha: float = 0.25
        dropout:       float = 0.0

    x = torch.randn(2, 128, 256)
    causal = torch.triu(
        torch.full((128, 128), float("-inf")), diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    print("=" * 55)
    print("LorentzMultiHeadAttention -- 三公式测试")
    print("=" * 55)

    results = {}
    for formula in ["f1", "f2", "f3"]:
        cfg  = Cfg(formula=formula)
        attn = LorentzMultiHeadAttention(cfg)
        if formula == "f2":
            attn.set_timelike_mask(torch.randint(0, 2, (256,)).bool())
        params = sum(p.numel() for p in attn.parameters())
        out, w = attn(x, causal)
        assert out.shape == (2, 128, 256)
        assert w.shape   == (2, 8, 128, 128)
        extra = ""
        if formula == "f3":
            extra = f"  sigma={attn.sigma:.3f}"
        elif formula == "f2":
            extra = f"  alpha={attn.alpha}"
        print(f"  {formula}: params={params:,}{extra}  OK")
        results[formula] = attn

    dt2 = compute_dt2_info(w)
    print(f"  dt2_info={dt2.item():.6f}  OK")
    print()
    print("选择指南:")
    guide = [
        ("f1", "推理/数学/物理/机器人", "硬约束，不退化"),
        ("f2", "加载旧版权重", "有退化风险，不推荐新项目"),
        ("f3", "LLM/视频/科学文本", "推荐默认，sigma自适应"),
    ]
    for f, use, note in guide:
        print(f"  [{f}] {use:20s}  -- {note}")
    print()
    print("All tests passed.")
