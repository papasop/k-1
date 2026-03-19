# ============================================================
# lorentz_transformer.py
# 洛伦兹Transformer 单文件完整版
#
# Colab 运行:
#   log = quick_train()    # 快速测试 (2000 steps)
#   log = full_train()     # 完整训练 (10000 steps)
#
# 终端运行:
#   python lorentz_transformer.py --mode quick --n_hops 2
#   python lorentz_transformer.py --mode full  --n_hops 2
#   python lorentz_transformer.py --mode test
#
# 包含内容:
#   Part 1: LorentzConfig, LorentzMultiHeadAttention,
#           TimeLikeProbe, LorentzTransformer
#   Part 2: GeodesicAdam, LorentzCosineScheduler
#   Part 3: TrainConfig, 数据生成, train()
#   Part 4: quick_train(), full_train() 便捷函数
# ============================================================

import os, sys, math, time, json, argparse, shutil
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==========================================================
# Part 1: 模型
# ==========================================================
# ============================================================
# model.py
# 洛伦兹Transformer核心架构
#
# 基于K=1信息几何场方程，参数空间为伪黎曼流形。
# 三个组件全部集成在模型内部：
#
#   Component 1: Minkowski注意力
#     scores_L = Q η K^T / √d
#     η = I - 2α P_t  （闵可夫斯基符号矩阵）
#
#   Component 2: 类时投影矩阵P_t（动态更新）
#     G_diag = Hutchinson(∂²dt²_info/∂W_Q²)
#     P_t = diag(G_diag < 0)
#     每N步更新，EMA平滑
#
#   Component 3: 类时子流形正则化
#     R(θ) = λ_s ||(I-P_t)W_Q||²
#     作为附加损失项，在train.py里调用
#
# 接口设计：
#   model = LorentzTransformer(config)
#   logits = model(input_ids)
#   model.update_timelike(batch, step)     # 更新P_t
#   reg_loss = model.regularization_loss() # 类时正则化
#   model.save_lorentz_state(path)         # 保存含P_t的完整状态
#   model.load_lorentz_state(path)         # 加载完整状态
# ============================================================


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
@dataclass
class LorentzConfig:
    """
    洛伦兹Transformer的完整配置

    模型结构：
      vocab_size:  词表大小
      d_model:     隐层维度
      n_heads:     注意力头数
      n_layers:    Transformer层数
      d_ff:        前馈网络中间维度（默认4×d_model）
      max_seq_len: 最大序列长度
      dropout:     Dropout概率

    洛伦兹参数（Component 1）：
      lorentz_alpha:   Minkowski注意力强度（0=标准，1=完全洛伦兹）
                       推荐从0.25开始，弱baseline任务可增大

    P_t更新参数（Component 2）：
      hess_update_freq:  每N步更新一次P_t
      hess_warmup_steps: 初期跳过（Adam二阶矩未稳定）
      hutchinson_k:      Hutchinson估计采样数
      ema_decay:         P_t的EMA平滑系数（保留旧值的比例）

    正则化参数（Component 3）：
      lambda_spacelike:  类空参数正则化强度（惩罚类空参数，保护知识）
      lambda_timelike:   类时参数奖励强度（0=不奖励，通常保持0）
    """
    # 模型结构
    vocab_size:  int   = 50257    # GPT-2词表大小
    d_model:     int   = 768
    n_heads:     int   = 12
    n_layers:    int   = 12
    d_ff:        int   = 0        # 0表示自动设为4×d_model
    max_seq_len: int   = 1024
    dropout:     float = 0.1

    # Component 1: Minkowski注意力
    lorentz_alpha:      float = 0.25
    use_minkowski_norm: bool  = True   # 闵可夫斯基LayerNorm
    lambda_cone:        float = 0.0    # 光锥损失权重（0=关闭，0.01=开启）

    # Component 2: P_t更新
    hess_update_freq:  int   = 40
    hess_warmup_steps: int   = 100
    hutchinson_k:      int   = 20
    ema_decay:         float = 0.7   # 保留旧值70%，新值30%

    # Component 3: 类时正则化
    lambda_spacelike: float = 0.0   # 参数正则化已禁用（幅度小≠不重要）
    lambda_timelike:  float = 0.0

    def __post_init__(self):
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model
        assert self.d_model % self.n_heads == 0, \
            f"d_model={self.d_model} 必须能被 n_heads={self.n_heads} 整除"

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


# ─────────────────────────────────────────────
# dt²_info 计算（K=1场方程的核心量）
# ─────────────────────────────────────────────
def compute_dt2_info(attn_w: torch.Tensor) -> torch.Tensor:
    """
    计算K=1信息时间度量

    dt²_info = Σ_q K_q = Σ_q Φ_q/H_q

    其中：
      H_q = Shannon熵（注意力分布的不确定性）
      Φ_q = Σ_j a_qj²（注意力集中度）
      K_q = Φ_q/H_q（信息密度）

    attn_w: (B, H, L, L) 注意力权重
    返回: scalar，所有query位置的K_q均值
    """
    aw = attn_w.mean(dim=1).float()                    # (B, L, L) 头平均
    aw = torch.nan_to_num(aw, nan=0.0)
    aw = aw / aw.sum(-1, keepdim=True).clamp(min=1e-8)

    H   = -(aw * torch.log(aw + 1e-8)).sum(-1)        # (B, L) 熵
    Phi = (aw ** 2).sum(-1)                            # (B, L) 集中度
    H   = H.clamp(min=1e-8)

    return (Phi / H).mean()


# ─────────────────────────────────────────────
# Hutchinson对角Hessian估计
# ─────────────────────────────────────────────
def hutchinson_diag_hessian(loss_fn, param: nn.Parameter,
                             n_samples: int = 20) -> torch.Tensor:
    """
    用Hutchinson估计量计算 ∂²L/∂param² 的对角元素

    G_ii ≈ (1/K) Σ_k v_k[i] * (H v_k)[i]
    其中 v_k ~ Rademacher{±1}

    loss_fn: callable() → scalar，必须通过param可微
    param:   目标参数（W_Q），requires_grad=True
    返回:    与param同形状的对角Hessian估计
             G_ii < 0 → 类时（dt²_info在此方向凹）
             G_ii > 0 → 类空（凸）
    """
    G = torch.zeros_like(param.data)

    for _ in range(n_samples):
        # Rademacher随机向量
        v = (torch.randint(0, 2, param.shape,
                           device=param.device) * 2 - 1).float()

        # 一阶梯度（保留计算图）
        loss = loss_fn()
        g1 = torch.autograd.grad(
            loss, param,
            create_graph=True,
            retain_graph=True
        )[0]
        if g1 is None:
            continue

        # Hessian-vector product
        Hv = torch.autograd.grad(
            (g1 * v.detach()).sum(),
            param,
            retain_graph=False
        )[0]
        if Hv is None:
            continue

        Hv_clean = torch.nan_to_num(Hv, nan=0.0, posinf=0.0, neginf=0.0)
        G = G + (v * Hv_clean).detach() / n_samples

    return G


# ─────────────────────────────────────────────
# Component 1: Minkowski多头注意力
# ─────────────────────────────────────────────
class LorentzMultiHeadAttention(nn.Module):
    """
    闵可夫斯基内积注意力

    核心公式：
      scores_L = Q η K^T / √d_h
      η = I - 2α P_t

    展开：
      scores_L = QK^T/√d_h - 2α (QP_t)K^T/√d_h
               = scores_std - 2α × 时间分量内积

    P_t 由外部TimeLikeProbe计算并通过 set_timelike_mask() 注入。
    α=0 完全退化为标准注意力（安全fallback）。

    对每个head独立处理：
      P_t_head[h] = diag(mask[h*d_h:(h+1)*d_h])
      时间分量 = Q_h @ P_t_head[h] @ K_h^T / √d_h
    """
    def __init__(self, config: LorentzConfig):
        super().__init__()
        self.n_heads  = config.n_heads
        self.head_dim = config.head_dim
        self.d_model  = config.d_model
        self.alpha    = config.lorentz_alpha

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.drop   = nn.Dropout(config.dropout)

        # 类时掩码：(d_model,) 布尔向量
        # True=类时（G_ii<0），False=类空（G_ii>0）
        # 由TimeLikeProbe通过set_timelike_mask()注入
        self.register_buffer('timelike_mask',
                              torch.zeros(config.d_model, dtype=torch.bool))
        self._has_mask = False

        # 诊断：最近一次的洛伦兹间隔（用于光锥分析）
        self.last_intervals:     Optional[torch.Tensor] = None
        self.last_intervals_raw: Optional[torch.Tensor] = None

    def set_timelike_mask(self, mask: torch.Tensor):
        """
        注入类时掩码
        mask: (d_model,) bool张量，True=类时
        """
        self.timelike_mask.copy_(mask.bool())
        self._has_mask = mask.any().item()

    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None):
        """
        x:              (B, L, d_model)
        attention_mask: (B, 1, 1, L) 或 (B, 1, L, L)，加性掩码（-inf为mask位置）
        返回: (output, attn_weights)
        """
        B, L, _ = x.shape
        H, d_h  = self.n_heads, self.head_dim
        scale   = math.sqrt(d_h)

        Q = self.q_proj(x).view(B, L, H, d_h).transpose(1, 2).float()
        K = self.k_proj(x).view(B, L, H, d_h).transpose(1, 2).float()
        V = self.v_proj(x).view(B, L, H, d_h).transpose(1, 2)

        # 标准scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale   # (B,H,L,L)

        # ── Minkowski修正（Component 1）──
        if self._has_mask and self.alpha > 0:
            # 类时掩码 → per-head对角权重
            # timelike_mask: (d_model,) → (H, d_h)
            mask_2d   = self.timelike_mask.view(H, d_h).float()    # (H, d_h)
            mask_bcast = mask_2d.unsqueeze(0).unsqueeze(2)         # (1,H,1,d_h)
            mask_dev   = mask_bcast.to(Q.device)

            # 类时Q分量
            Q_t = Q * mask_dev                                     # (B,H,L,d_h)

            # 归一化：将Q_t缩放到与Q相同的幅度
            # 只对Q_t非零的query做归一化，零向量直接跳过
            q_norm       = Q.norm(dim=-1, keepdim=True)             # (B,H,L,1)
            qt_norm      = Q_t.norm(dim=-1, keepdim=True)           # (B,H,L,1)
            has_timelike = (qt_norm > 1e-6)                         # 有类时分量
            # 安全缩放：Q_t非零时对齐幅度，零时保持零
            scale_factor = torch.where(
                has_timelike,
                q_norm / qt_norm.clamp(min=1e-8),
                torch.zeros_like(qt_norm)
            )
            Q_t_scaled = Q_t * scale_factor

            # 时间分量内积（幅度与标准scores同量级）
            time_inner = torch.matmul(Q_t_scaled,
                         K.transpose(-2, -1)) / scale              # (B,H,L,L)

            # 洛伦兹修正：减去2α倍时间分量
            scores = scores - 2.0 * self.alpha * time_inner

        # 광추 통계용: causal mask 전 순수 scores 저장 (lightcone_loss용)
        self.last_intervals_raw = scores.detach().clone()

        # attention mask
        if attention_mask is not None:
            scores = scores + attention_mask.to(scores.dtype)

        # 기존 광추 통계용: causal mask 후 저장 (역사적 비교 유지)
        self.last_intervals = scores.detach().clone()

        attn_w = F.softmax(scores, dim=-1).to(x.dtype)
        attn_w = self.drop(attn_w)

        out = torch.matmul(attn_w, V)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out), attn_w


# ─────────────────────────────────────────────
# Minkowski LayerNorm（洛伦兹归一化）
# ─────────────────────────────────────────────
class MinkowskiLayerNorm(nn.Module):
    """
    闵可夫斯基归一化。

    标准LayerNorm: x / ||x||_2  （欧氏）
    MinkowskiLayerNorm: x / sqrt(|<x,x>_η|)  （闵可夫斯基）

    η-内积: <x,x>_η = ||x_s||² - ||x_t||²
      类空分量(x_s)贡献正，类时分量(x_t)贡献负。
    mask未注入时退化为标准LayerNorm。
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps     = eps
        self.weight  = nn.Parameter(torch.ones(d_model))
        self.bias    = nn.Parameter(torch.zeros(d_model))
        self.register_buffer('timelike_mask',
                              torch.zeros(d_model, dtype=torch.bool))
        self._has_mask = False
        self._blend    = 0.0   # 0=标准norm，1=完全η-norm（用于warmup）

    def set_timelike_mask(self, mask: torch.Tensor):
        self.timelike_mask.copy_(mask.bool())
        self._has_mask = mask.any().item()

    def set_blend(self, alpha: float):
        self._blend = max(0.0, min(1.0, alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean  = x.mean(dim=-1, keepdim=True)
        x_c   = x - mean
        # 标准norm（始终计算）
        var   = x_c.pow(2).mean(dim=-1, keepdim=True)
        x_std = x_c / (var + self.eps).sqrt()
        if not self._has_mask or self._blend == 0.0:
            x_n = x_std
        else:
            # η-norm
            m      = self.timelike_mask.float()
            x_t    = x_c * m
            x_s    = x_c * (1.0 - m)
            eta_sq = (x_s.pow(2).sum(dim=-1, keepdim=True) -
                      x_t.pow(2).sum(dim=-1, keepdim=True))
            x_eta  = x_c / (eta_sq.abs() + self.eps).sqrt()
            # 线性blend，避免mask注入时的激活冲击
            x_n    = (1.0 - self._blend) * x_std + self._blend * x_eta
        return self.weight * x_n + self.bias


# ─────────────────────────────────────────────
# 洛伦兹前馈网络
# ─────────────────────────────────────────────
class FeedForward(nn.Module):
    """
    洛伦兹前馈网络。

    标准FFN: x → Linear → GELU → Linear（各向同性）

    洛伦兹FFN：
      1. 将输入分解为类时分量(x_t)和类空分量(x_s)
      2. 类时分量: Linear_t → SiLU（平滑，保留负值，适合时间方向）
      3. 类空分量: Linear_s → GELU（标准，适合空间方向）
      4. 合并后投影回d_model

    设计原理：
      类时方向对应因果信息流，SiLU（x*sigmoid(x））保留负激活，
      允许类时方向传递"抑制"信号（时间上的因果否定）。
      类空方向对应语义定位，GELU保持标准行为。

    mask未注入时退化为标准FFN（安全fallback）。
    """
    def __init__(self, config: LorentzConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_ff    = config.d_ff
        self.drop    = nn.Dropout(config.dropout)

        # 표준 FFN (fallback 및 Phase 1)
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

        # 类时专用 FFN
        self.w1_t = nn.Linear(config.d_model, config.d_ff // 2, bias=False)
        self.w2_t = nn.Linear(config.d_ff // 2, config.d_model, bias=False)

        # 类空专用 FFN
        self.w1_s = nn.Linear(config.d_model, config.d_ff // 2, bias=False)
        self.w2_s = nn.Linear(config.d_ff // 2, config.d_model, bias=False)

        self.register_buffer('timelike_mask',
                              torch.zeros(config.d_model, dtype=torch.bool))
        self._has_mask = False
        self._blend    = 0.0   # 线性混合比例（0=标准，1=完全洛伦兹）

    def set_timelike_mask(self, mask: torch.Tensor):
        self.timelike_mask.copy_(mask.bool())
        self._has_mask = mask.any().item()

    def set_blend(self, alpha: float):
        """设置洛伦兹FFN的混合比例（0=标准，1=完全洛伦兹）"""
        self._blend = max(0.0, min(1.0, alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std_out = self.drop(self.w2(F.gelu(self.w1(x))))

        if not self._has_mask or self._blend == 0.0:
            return std_out

        m   = self.timelike_mask.float()
        x_t = x * m
        x_s = x * (1.0 - m)
        h_t = self.drop(self.w2_t(F.silu(self.w1_t(x_t))))
        h_s = self.drop(self.w2_s(F.gelu(self.w1_s(x_s))))
        lor_out = h_t + h_s

        # 线性混合：blend=0→标准，blend=1→洛伦兹
        return (1.0 - self._blend) * std_out + self._blend * lor_out


# ─────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────
class LorentzBlock(nn.Module):
    def __init__(self, config: LorentzConfig):
        super().__init__()
        self.attn  = LorentzMultiHeadAttention(config)
        self.ff    = FeedForward(config)
        # use_minkowski_norm=False时使用标准LayerNorm
        NormClass  = MinkowskiLayerNorm if config.use_minkowski_norm else nn.LayerNorm
        self.norm1 = NormClass(config.d_model)
        self.norm2 = NormClass(config.d_model)

    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None):
        ao, aw = self.attn(self.norm1(x), attention_mask)
        x = x + ao
        x = x + self.ff(self.norm2(x))
        return x, aw


# ─────────────────────────────────────────────
# Component 2: 类时投影矩阵管理器
# ─────────────────────────────────────────────
class TimeLikeProbe:
    """
    类时投影矩阵管理器（在注意力权重空间计算Hessian）

    核心改进：
      不在W_Q参数空间计算Hessian（梯度链条太长，数值~1e-30）。
      直接对注意力权重矩阵 a_{ij} 计算 dt²_info 的对角Hessian。
      此方式与k1_lorentz.py实验完全一致，λ_min在1e-3到1e-5量级。

    计算流程：
      1. 前向传播得到注意力权重 a (B,H,L,L)，设为叶子节点
      2. 对 a 计算 dt²_info = Σ_q Φ_q/H_q
      3. Hutchinson估计 ∂²(dt²_info)/∂a² 的对角元素
      4. G_ii < 0 → 类时（该注意力维度使dt²_info变小）
      5. 将注意力空间的类时信息映射到W_Q的mask

    注意力空间 → W_Q mask的映射：
      attention是对称的(L×L)，
      对每个query位置q，如果∂²dt²_info/∂a_{qq} < 0，
      则该query对应的W_Q输出维度标记为类时。
    """
    def __init__(self, model: 'LorentzTransformer',
                 config: LorentzConfig):
        self.model  = model
        self.config = config

        self._mask_ema = [
            torch.zeros(config.d_model, device='cpu')
            for _ in range(config.n_layers)
        ]
        self._threshold = 0.3
        self.timelike_fracs  = [0.0] * config.n_layers
        self.lambda_mins     = [0.0] * config.n_layers
        self.n_updates       = 0

    def step(self, x: torch.Tensor, global_step: int):
        """
        主更新接口。与k1_lorentz.py完全相同的实现：
          - 直接对真实W_Q（requires_grad=True）求Hessian
          - loss_fn: 前向到目标层，用真实W_Q计算dt²_info
          - EMA平滑：decay=0.7保留旧值
          - 阈值：G_ii < 0 → 类时
        """
        cfg = self.config
        if global_step < cfg.hess_warmup_steps:
            return
        if global_step % cfg.hess_update_freq != 0:
            return

        xb     = x[:min(32, len(x))].detach()
        device = next(self.model.parameters()).device
        xb     = xb.to(device)

        self.model.eval()
        new_G_diags = []

        for li, block in enumerate(self.model.blocks):
            W_Q = block.attn.q_proj.weight  # 真实参数，直接求Hessian
            if not W_Q.requires_grad:
                new_G_diags.append(None)
                continue

            # loss_fn: 与k1_lorentz.py完全一致
            # 前向传播到第li层，提取注意力权重，计算dt²_info
            def make_loss_fn(layer_idx):
                def loss_fn():
                    h = (self.model.embed(xb) +
                         self.model.pos_enc(xb.shape[1], device))
                    for i, blk in enumerate(self.model.blocks):
                        ao, aw = blk.attn(blk.norm1(h))
                        h = h + ao
                        h = h + blk.ff(blk.norm2(h))
                        if i == layer_idx:
                            return -compute_dt2_info(aw)
                    return torch.tensor(0.0, device=device)
                return loss_fn

            loss_fn = make_loss_fn(li)

            try:
                G = hutchinson_diag_hessian(
                    loss_fn, W_Q,
                    n_samples=cfg.hutchinson_k)

                lmin = G.min().item()
                self.lambda_mins[li] = lmin

                # nan/inf 처리 후 EMA 업데이트
                G_safe = torch.nan_to_num(
                    G.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                G_vec  = G_safe.mean(dim=1).cpu()   # (d_model,)

                if self._mask_ema[li].abs().sum() == 0:
                    self._mask_ema[li] = G_vec
                else:
                    self._mask_ema[li] = (
                        cfg.ema_decay * self._mask_ema[li] +
                        (1 - cfg.ema_decay) * G_vec)

                # nan이 EMA에 들어가지 않도록 보호
                self._mask_ema[li] = torch.nan_to_num(
                    self._mask_ema[li], nan=0.0)

                # G_ema < 0 → 类时
                binary_mask = (self._mask_ema[li] < 0)
                block.attn.set_timelike_mask(binary_mask.to(device))
                # hasattr保护：baseline使用nn.LayerNorm，没有set_timelike_mask
                if hasattr(block.norm1, 'set_timelike_mask'):
                    block.norm1.set_timelike_mask(binary_mask.to(device))
                if hasattr(block.norm2, 'set_timelike_mask'):
                    block.norm2.set_timelike_mask(binary_mask.to(device))
                if hasattr(block.ff, 'set_timelike_mask'):
                    block.ff.set_timelike_mask(binary_mask.to(device))
                self.timelike_fracs[li] = binary_mask.float().mean().item()

                lmin_safe = G_safe.min().item()
                self.lambda_mins[li] = lmin_safe
                new_G_diags.append(G_safe)

            except Exception:
                new_G_diags.append(None)

        # 位置编码和norm_f使用第一层的mask
        if self.model.blocks:
            first_mask = self.model.blocks[0].attn.timelike_mask
            self.model.pos_enc.set_timelike_mask(first_mask)
            # norm_f也注入mask（输出归一化几何一致性）
            if hasattr(self.model.norm_f, 'set_timelike_mask'):
                self.model.norm_f.set_timelike_mask(first_mask)

        self.model.train()
        self.n_updates += 1

    def get_param_mask_pairs(self, device):
        """返回 (W_Q参数, mask) 列表，供GeodesicAdam使用"""
        pairs = []
        for li, block in enumerate(self.model.blocks):
            mask = block.attn.timelike_mask
            pairs.append((block.attn.q_proj.weight, mask.to(device)))
        return pairs

    def report(self) -> str:
        lines = ['TimeLikeProbe诊断:']
        for li in range(self.config.n_layers):
            frac = self.timelike_fracs[li]
            lmin = self.lambda_mins[li]
            bar  = '█' * int(frac * 20)
            lines.append(
                f'  layer {li:2d}: frac={frac:.3f} λ_min={lmin:.2e} {bar}')
        lines.append(f'  总更新次数: {self.n_updates}')
        return '\n'.join(lines)


# ─────────────────────────────────────────────
# 洛伦兹位置编码
# ─────────────────────────────────────────────
class LorentzPositionalEncoding(nn.Module):
    """
    洛伦兹位置编码：时间坐标和空间坐标分离。

    标准位置编码：pos_emb(i) ∈ R^d_model（各向同性）
    所有d_model维度平等对待位置信息。

    洛伦兹位置编码：
      时间分量（类时维度）：编码因果顺序（时间箭头方向）
        使用单调递增函数：t_enc[i] = sin(i / L) 归一化到[-1, 1]
        因果位置的时间坐标应该是严格有序的

      空间分量（类空维度）：编码语义位置（标准正弦编码）
        使用标准Transformer正弦编码：sin/cos(i / 10000^(2k/d))

    两者拼接后投影到d_model维度（不增加参数量，只重新分配）。

    mask未注入时退化为标准位置编码。
    """
    def __init__(self, config: LorentzConfig):
        super().__init__()
        self.d_model  = config.d_model
        self.max_len  = config.max_seq_len

        # 标准可学习位置嵌入（fallback）
        self.pos_emb  = nn.Embedding(config.max_seq_len, config.d_model)

        # 时间分量投影（类时维度 → 可学习缩放）
        self.time_scale = nn.Parameter(torch.ones(1))

        self.register_buffer('timelike_mask',
                              torch.zeros(config.d_model, dtype=torch.bool))
        self._has_mask = False
        self._blend    = 0.0   # 线性混合比例（0=标准，1=完全洛伦兹）

        # 预计算正弦位置编码
        self._build_sinusoidal(config.max_seq_len, config.d_model)

    def _build_sinusoidal(self, max_len: int, d_model: int):
        """预计算标准正弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model//2])
        self.register_buffer('sinusoidal', pe)

    def set_timelike_mask(self, mask: torch.Tensor):
        self.timelike_mask.copy_(mask.bool())
        self._has_mask = mask.any().item()

    def set_blend(self, alpha: float):
        """设置洛伦兹编码的混合比例（0=标准，1=完全洛伦兹）"""
        self._blend = max(0.0, min(1.0, alpha))

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        返回位置编码 (1, L, d_model)
        支持线性混合：blend=0时纯标准，blend=1时纯洛伦兹
        """
        pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        std_enc  = self.pos_emb(pos_idx)

        if not self._has_mask or self._blend == 0.0:
            return std_enc

        m        = self.timelike_mask.float()
        sin_enc  = self.sinusoidal[:seq_len].unsqueeze(0)
        t = torch.arange(seq_len, device=device).float()
        if seq_len > 1:
            t = t / (seq_len - 1)
        t = t.unsqueeze(0).unsqueeze(-1)

        time_enc  = t * m * self.time_scale
        space_enc = sin_enc * (1.0 - m)
        lor_enc   = time_enc + space_enc

        # 线性混合：blend=0→标准，blend=1→洛伦兹
        return (1.0 - self._blend) * std_enc + self._blend * lor_enc


# ─────────────────────────────────────────────
# 主模型：洛伦兹Transformer
# ─────────────────────────────────────────────
class LorentzTransformer(nn.Module):
    """
    洛伦兹Transformer

    基于K=1信息几何场方程的Transformer架构。
    三个组件全部集成：
      - LorentzMultiHeadAttention（Component 1）
      - TimeLikeProbe（Component 2，作为属性）
      - regularization_loss()（Component 3）

    标准用法（train.py里）：
      model  = LorentzTransformer(config)
      probe  = model.probe                    # 访问TimeLikeProbe
      optim  = GeodesicAdam(model.parameters())

      # 训练步
      logits = model(input_ids, attention_mask)
      loss   = ce_loss + model.regularization_loss()
      loss.backward()
      optim.update_timelike_masks(
          probe.get_param_mask_pairs(device))  # 更新梯度分解
      optim.step()
      probe.step(input_ids, global_step)       # 更新P_t
    """
    def __init__(self, config: LorentzConfig):
        super().__init__()
        self.config = config
        # 실험 설정 alpha를 별도 저장 - set_lorentz_alpha()가 config를 오염시켜도 유지됨
        self._experiment_alpha = config.lorentz_alpha

        # 嵌入层
        self.embed   = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_enc = LorentzPositionalEncoding(config)   # 洛伦兹位置编码
        self.drop    = nn.Dropout(config.dropout)

        # Transformer块
        self.blocks = nn.ModuleList([
            LorentzBlock(config) for _ in range(config.n_layers)
        ])

        # 输出层
        NormClass   = MinkowskiLayerNorm if config.use_minkowski_norm else nn.LayerNorm
        self.norm_f = NormClass(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size,
                                  bias=False)

        # 权重共享（标准GPT做法）
        self.lm_head.weight = self.embed.weight

        # Component 2：类时投影矩阵管理器
        self.probe = TimeLikeProbe(self, config)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """标准GPT-2风格初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0,
                                std=0.02 / math.sqrt(2 * self.config.n_layers))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None):
        """
        input_ids:      (B, L) long
        attention_mask: (B, L) bool，True=有效位置（自动转为加性掩码）
        返回: logits (B, L, vocab_size)
        """
        B, L = input_ids.shape
        assert L <= self.config.max_seq_len, \
            f"序列长度{L}超过最大{self.config.max_seq_len}"

        device = input_ids.device

        # 洛伦兹位置编码
        pos_enc = self.pos_enc(L, device)
        h       = self.drop(self.embed(input_ids) + pos_enc)

        # 构造加性attention mask
        attn_bias = None
        if attention_mask is not None:
            # (B, L) → (B, 1, 1, L)，pad位置设为-inf
            attn_bias = (~attention_mask).float()
            attn_bias = attn_bias.masked_fill(
                ~attention_mask, float('-inf')).unsqueeze(1).unsqueeze(2)

        # 因果掩码（自回归）
        causal = torch.triu(
            torch.full((L, L), float('-inf'), device=device),
            diagonal=1).unsqueeze(0).unsqueeze(0)   # (1,1,L,L)

        if attn_bias is not None:
            combined_mask = attn_bias + causal
        else:
            combined_mask = causal

        # 前向传播
        for block in self.blocks:
            h, _ = block(h, combined_mask)

        h = self.norm_f(h)
        return self.lm_head(h)

    def lightcone_loss(self, lambda_cone: float = 0.01) -> torch.Tensor:
        """
        光锥损失：惩罚注意力间隔违反因果几何。
        因果方向(j>i)应为类时(间隔<0)，非因果方向(j≤i)应为类空(间隔>0)。
        """
        if lambda_cone <= 0:
            return torch.tensor(0.0)
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        n = 0
        for block in self.blocks:
            iv = block.attn.last_intervals_raw
            if iv is None:
                continue
            B, H, L, _ = iv.shape
            # 2D causal mask → 4D broadcast
            causal = torch.triu(
                torch.ones(L, L, device=iv.device, dtype=torch.bool),
                diagonal=1).unsqueeze(0).unsqueeze(0)   # (1,1,L,L)

            # 因果方向(j>i)：间隔应<0，惩罚>0部分
            loss_c  = torch.clamp(iv * causal.float(),  min=0).sum() / causal.sum().clamp(min=1)
            # 非因果方向(j≤i)：间隔应>0，惩罚<0部分
            noncausal = ~causal
            loss_nc = torch.clamp(-iv * noncausal.float(), min=0).sum() / noncausal.sum().clamp(min=1)
            total = total + loss_c + loss_nc
            n += 1
        return lambda_cone * (total / max(n, 1))

    def regularization_loss(self) -> torch.Tensor:
        """
        Component 3：类时子流形正则化损失

        R(θ) = λ_s · Σ_l ||(I-P_t) W_Q_l||²

        惩罚每层W_Q的类空分量，保护类空方向不被过度更新。
        这约束参数更新在类时子流形上，是EWC的几何版本。

        返回: scalar tensor，加到任务损失上
        """
        cfg  = self.config
        loss = torch.tensor(0.0,
                            device=next(self.parameters()).device)

        if cfg.lambda_spacelike == 0 and cfg.lambda_timelike == 0:
            return loss

        for block in self.blocks:
            W_Q  = block.attn.q_proj.weight   # (d_model, d_model)
            mask = block.attn.timelike_mask.float()  # (d_model,) 0/1

            if not block.attn._has_mask:
                continue

            # 类空分量：(I - P_t) W_Q
            # P_t作用于行（output维度）
            mask_row = mask.unsqueeze(1)         # (d_model, 1) 广播
            W_spacelike = W_Q * (1 - mask_row)  # 类空行
            W_timelike  = W_Q * mask_row         # 类时行

            if cfg.lambda_spacelike > 0:
                loss = loss + cfg.lambda_spacelike * W_spacelike.pow(2).sum()

            if cfg.lambda_timelike > 0:
                loss = loss - cfg.lambda_timelike * W_timelike.pow(2).sum()

        return loss

    def get_num_params(self, non_embedding=True):
        """返回参数量"""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_enc.pos_emb.weight.numel()
        return n

    def lightcone_stats(self) -> dict:
        """
        返回各层的光锥统计（诊断用）
        interval < 0 → 类时，> 0 → 类空
        """
        stats = {}
        for li, block in enumerate(self.blocks):
            iv = block.attn.last_intervals
            if iv is not None:
                stats[f'layer_{li}'] = {
                    'timelike_frac': (iv < 0).float().mean().item(),
                    'mean_interval': iv[iv != float('-inf')].mean().item() if (iv != float('-inf')).any() else 0.0,
                    'std_interval':  iv.std().item(),
                }
        return stats

    def set_lorentz_alpha(self, alpha: float):
        """动态调整洛伦兹强度（用于训练阶段切换）。
        仅修改运行时attn.alpha，不污染config.lorentz_alpha实验配置。
        """
        # 不写回config，保留实验超参数的完整性
        for block in self.blocks:
            block.attn.alpha = alpha

    def save_lorentz_state(self, path: str):
        """
        保存完整状态，包括P_t的EMA历史

        标准 torch.save(model.state_dict()) 不包含probe的状态。
        这个方法额外保存probe，确保训练可以从任意checkpoint恢复。
        """
        # config를 저장하기 전에 실험 초기 alpha로 복원한 복사본 사용
        # (set_lorentz_alpha()가 runtime 중에 config.lorentz_alpha를 변경하지 않지만
        #  방어적으로 _experiment_alpha로 덮어쓴다)
        import copy as _copy
        save_config = _copy.copy(self.config)
        save_config.lorentz_alpha = self._experiment_alpha
        state = {
            'model_state': self.state_dict(),
            'config':      save_config,
            'experiment_alpha': self._experiment_alpha,  # 명시적 보존
            # 훈련 재개를 위한 상태 (train()에서 채워짐, 없으면 None)
            'train_step':       getattr(self, '_saved_train_step', None),
            'best_val_loss':    getattr(self, '_saved_best_val_loss', None),
            'phase2_entered':   getattr(self, '_saved_phase2_entered', None),
            'probe_mask_ema': self.probe._mask_ema,
            'probe_n_updates': self.probe.n_updates,
            'probe_timelike_fracs': self.probe.timelike_fracs,
            # 运行态：直接保存，不在加载时推算
            'runtime': {
                'pos_enc_blend': self.pos_enc._blend,
                'pos_enc_has_mask': self.pos_enc._has_mask,
                'norm_f_has_mask': getattr(self.norm_f, '_has_mask', False),
                'blocks': [{
                    'attn_has_mask': blk.attn._has_mask,
                    'norm1_has_mask': getattr(blk.norm1, '_has_mask', False),
                    'norm2_has_mask': getattr(blk.norm2, '_has_mask', False),
                    'ff_has_mask':   getattr(blk.ff,   '_has_mask', False),
                    'ff_blend':      getattr(blk.ff,   '_blend',    0.0),
                } for blk in self.blocks],
            },
        }
        torch.save(state, path)

    @classmethod
    def load_lorentz_state(cls, path: str,
                            device: str = 'cpu') -> 'LorentzTransformer':
        """
        从checkpoint加载完整状态（包括probe）
        """
        import torch.serialization as _ts
        _ts.add_safe_globals([LorentzConfig])
        state  = torch.load(path, map_location=device, weights_only=False)
        config = state['config']
        model  = cls(config)
        model.load_state_dict(state['model_state'])
        # _experiment_alpha 복원 (이전 checkpoint 호환: experiment_alpha 없으면 config에서)
        model._experiment_alpha = state.get('experiment_alpha', config.lorentz_alpha)
        model.probe._mask_ema       = state['probe_mask_ema']
        model.probe.n_updates       = state['probe_n_updates']
        model.probe.timelike_fracs  = state['probe_timelike_fracs']
        # 恢复各层的timelike_mask到所有组件
        for li, block in enumerate(model.blocks):
            ema = state['probe_mask_ema'][li]
            mask = (ema < 0)  # G_ema < 0 → 类时
            m = mask.to(device)
            block.attn.set_timelike_mask(m)
            if hasattr(block.norm1, 'set_timelike_mask'):
                block.norm1.set_timelike_mask(m)
            if hasattr(block.norm2, 'set_timelike_mask'):
                block.norm2.set_timelike_mask(m)
            if hasattr(block.ff, 'set_timelike_mask'):
                block.ff.set_timelike_mask(m)
        # 位置编码和norm_f使用第一层的mask
        if model.blocks:
            first_m = model.blocks[0].attn.timelike_mask
            model.pos_enc.set_timelike_mask(first_m)
            if hasattr(model.norm_f, 'set_timelike_mask'):
                model.norm_f.set_timelike_mask(first_m)
        # 恢复运行态（优先用保存的runtime，兼容旧checkpoint）
        rt = state.get('runtime', None)
        if rt is not None:
            model.pos_enc.set_blend(rt['pos_enc_blend'])
            model.pos_enc._has_mask = rt['pos_enc_has_mask']
            if hasattr(model.norm_f, '_has_mask'):
                model.norm_f._has_mask = rt['norm_f_has_mask']
            for i, blk in enumerate(model.blocks):
                if i < len(rt['blocks']):
                    bs = rt['blocks'][i]
                    blk.attn._has_mask = bs['attn_has_mask']
                    if hasattr(blk.norm1, '_has_mask'):
                        blk.norm1._has_mask = bs['norm1_has_mask']
                    if hasattr(blk.norm2, '_has_mask'):
                        blk.norm2._has_mask = bs['norm2_has_mask']
                    if hasattr(blk.ff, '_has_mask'):
                        blk.ff._has_mask = bs['ff_has_mask']
                    if hasattr(blk.ff, 'set_blend'):
                        blk.ff.set_blend(bs['ff_blend'])
        else:
            # 旧checkpoint兼容：无runtime字段
            import warnings
            warnings.warn(
                "Checkpoint without runtime state detected. "
                "pos_enc._blend and ff._blend reset to 0.0. "
                "Weight restoration is accurate but training state may differ.",
                RuntimeWarning, stacklevel=2)
            model.pos_enc.set_blend(0.0)
            for blk in model.blocks:
                if hasattr(blk.ff, 'set_blend'):
                    blk.ff.set_blend(0.0)
        # 训练续训状态（train()读取这些属性）
        model._saved_train_step     = state.get('train_step',     None)
        model._saved_best_val_loss  = state.get('best_val_loss',  None)
        model._saved_phase2_entered = state.get('phase2_entered', None)
        return model.to(device)


# ─────────────────────────────────────────────
# 快速验证
# ─────────────────────────────────────────────

# ==========================================================
# Part 2: 优化器
# ==========================================================
# ============================================================
# optimizer.py
# 洛伦兹Transformer第二组件：测地线Adam优化器
#
# GeodesicAdam：在洛伦兹流形上沿类时测地线更新参数
#
# 核心思想：
#   标准Adam：所有梯度方向平等（欧氏空间假设）
#   GeodesicAdam：梯度分解为类时（G_ii<0）和类空（G_ii>0）分量
#                 类时方向步长更大（信息几何上更有效的方向）
#                 类空方向步长更小（保护已有知识）
#
# 数学：
#   g_t = P_t ⊙ g          (类时梯度，mask=1的位置)
#   g_s = (I-P_t) ⊙ g      (类空梯度，mask=0的位置)
#   g_geo = scale_t×g_t + scale_s×g_s
#   θ_{t+1} = Adam_update(g_geo)
#
# 实验结果（geodesic_adam.py）：
#   最优配置：scale_t=2.0, scale_s=0.5（ratio=4）
#   r规律在GeodesicAdam下同样成立（r=-0.90到-0.997）
#   seed 1（弱baseline）：geo_200_050 delta=+0.025
#
# 用法：
#   optimizer = GeodesicAdam(model.parameters(),
#                             lr=3e-4, scale_t=2.0, scale_s=0.5)
#   # 每步更新P_t后：
#   optimizer.update_masks(model.probe.get_param_mask_pairs(device))
#   optimizer.step()
# ============================================================


class GeodesicAdam(torch.optim.Adam):
    """
    测地线Adam优化器

    继承标准Adam，在调用父类step之前对梯度做类时/类空分解：

      g_t    = mask ⊙ g           (类时分量，沿负Hessian方向)
      g_s    = (1-mask) ⊙ g       (类空分量)
      g_geo  = scale_t×g_t + scale_s×g_s
      → Adam(g_geo)

    其中mask来自TimeLikeProbe（G_ii<0的位置为1）。

    参数：
      params:    模型参数（同torch.optim.Adam）
      lr:        基础学习率
      betas:     Adam的β1, β2
      eps:       Adam的ε
      weight_decay: 权重衰减
      scale_t:   类时方向学习率乘数（推荐2.0，实验验证）
      scale_s:   类空方向学习率乘数（推荐0.5，实验验证）
      adaptive_scale: 是否根据类时比例自适应调整缩放
                      True时：比例越高，类时放大越保守
                      False时：固定使用scale_t和scale_s

    自适应缩放（adaptive_scale=True）：
      如果类时比例frac很高（如0.8），说明大部分方向都是类时，
      放大所有类时方向会导致有效学习率显著上升。
      自适应模式下：effective_scale_t = 1 + (scale_t-1) × (1-frac)
      确保类时比例高时自动降低放大幅度。

    注意：
      只对通过update_masks()注册了mask的参数做分解。
      其他参数（位置编码、LayerNorm等）使用标准Adam。
      这保证了只有W_Q受到洛伦兹几何约束。
    """

    def __init__(self,
                 params,
                 lr: float = 3e-4,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 scale_t: float = 2.0,
                 scale_s: float = 0.5,
                 adaptive_scale: bool = False):

        super().__init__(params, lr=lr, betas=betas,
                         eps=eps, weight_decay=weight_decay)

        self.scale_t        = scale_t
        self.scale_s        = scale_s
        self.adaptive_scale = adaptive_scale

        # param_id → timelike_mask (与param同形状的0/1 float张量)
        self._masks: Dict[int, torch.Tensor] = {}

        # 诊断统计
        self.n_masked_params   = 0     # 有mask的参数数量
        self.last_timelike_frac = 0.0  # 最近一次的类时比例均值

    def update_masks(self,
                     param_mask_pairs: List[Tuple[nn.Parameter,
                                                   torch.Tensor]]):
        """
        注入类时掩码

        param_mask_pairs: list of (param, mask)
          param: nn.Parameter（通常是W_Q）
          mask:  与param同形状的bool或float张量
                 True/1.0 = 类时，False/0.0 = 类空

        调用时机：在每次probe.step()之后调用，确保mask是最新的。

        train.py里的调用方式：
          probe.step(x, global_step)
          optimizer.update_masks(probe.get_param_mask_pairs(device))
        """
        self._masks.clear()
        fracs = []
        for param, mask in param_mask_pairs:
            m = mask.float().to(param.device)
            self._masks[id(param)] = m
            fracs.append(m.mean().item())

        self.n_masked_params   = len(param_mask_pairs)
        self.last_timelike_frac = sum(fracs) / len(fracs) if fracs else 0.0

    def _get_effective_scales(self, frac: float) -> Tuple[float, float]:
        """
        计算有效的类时/类空缩放系数

        固定模式：直接返回scale_t和scale_s
        自适应模式：根据类时比例调整，防止有效学习率过大
        """
        if not self.adaptive_scale:
            return self.scale_t, self.scale_s

        # 自适应：类时比例越高，放大越保守
        # effective_scale_t ∈ [1.0, scale_t]
        # frac=0 → scale_t（最大放大）
        # frac=1 → 1.0（不放大，所有方向都是类时时无法区分）
        eff_t = 1.0 + (self.scale_t - 1.0) * (1.0 - frac)
        eff_s = self.scale_s
        return eff_t, eff_s

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        执行一步参数更新

        流程：
          1. 对有mask的参数，把grad分解为类时/类空分量并重新缩放
          2. 调用标准Adam的step（使用修改后的grad）
          3. 恢复grad（避免影响后续调用）
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 存储原始grad（用于恢复）
        original_grads: Dict[int, Optional[torch.Tensor]] = {}

        # ── 梯度分解（仅对有mask的参数）──
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                mask = self._masks.get(id(p))
                if mask is None:
                    continue   # 无mask：使用标准Adam，不修改

                # 保存原始grad
                original_grads[id(p)] = p.grad.data.clone()

                g    = p.grad.data                          # (d_out, d_in)
                mask = mask.to(g.device)

                # 类时/类空分解（P_t作用于行/output维度）
                # mask: (d_model,)=(d_out,), g: (d_out, d_in)
                # reshape为(d_out,1)才能沿行广播，不污染列方向
                if g.dim() == 2:
                    m = mask.view(-1, 1)   # (d_out, 1)
                else:
                    m = mask               # 1D参数直接用
                g_t  = m * g               # 类时分量
                g_s  = (1.0 - m) * g       # 类空分量

                # 有效缩放系数
                eff_t, eff_s = self._get_effective_scales(
                    self.last_timelike_frac)

                # 合并：类时放大，类空缩小
                g_geo = eff_t * g_t + eff_s * g_s
                p.grad.data = g_geo

        # ── 标准Adam更新（使用修改后的grad）──
        super().step(closure=None)

        # ── 恢复原始grad（保持调用者的grad状态）──
        for group in self.param_groups:
            for p in group['params']:
                if id(p) in original_grads:
                    p.grad.data = original_grads[id(p)]

        return loss

    def state_dict_lorentz(self) -> dict:
        """
        保存包含洛伦兹状态的完整state_dict

        标准optimizer.state_dict()不包含mask信息。
        这个方法额外保存mask，用于从checkpoint完整恢复。
        """
        base = super().state_dict()
        base['lorentz'] = {
            'scale_t':         self.scale_t,
            'scale_s':         self.scale_s,
            'adaptive_scale':  self.adaptive_scale,
            'n_masked_params': self.n_masked_params,
            # mask不保存（会从probe重新计算）
        }
        return base

    @classmethod
    def load_lorentz(cls, state_dict: dict,
                     params, **kwargs) -> 'GeodesicAdam':
        """
        从state_dict恢复优化器（包含洛伦兹配置）
        """
        lorentz_cfg = state_dict.pop('lorentz', {})
        kwargs.setdefault('scale_t', lorentz_cfg.get('scale_t', 2.0))
        kwargs.setdefault('scale_s', lorentz_cfg.get('scale_s', 0.5))
        kwargs.setdefault('adaptive_scale',
                          lorentz_cfg.get('adaptive_scale', False))
        opt = cls(params, **kwargs)
        opt.load_state_dict(state_dict)
        return opt

    def report(self) -> str:
        """打印优化器状态摘要"""
        lines = [
            f'GeodesicAdam:',
            f'  scale_t={self.scale_t}  scale_s={self.scale_s}  '
            f'ratio={self.scale_t/self.scale_s:.1f}',
            f'  adaptive_scale={self.adaptive_scale}',
            f'  masked_params={self.n_masked_params}',
            f'  timelike_frac={self.last_timelike_frac:.3f}',
        ]
        if self.adaptive_scale and self.last_timelike_frac > 0:
            eff_t, eff_s = self._get_effective_scales(
                self.last_timelike_frac)
            lines.append(
                f'  effective: scale_t={eff_t:.3f}  scale_s={eff_s:.3f}')
        return '\n'.join(lines)


# ─────────────────────────────────────────────
# 学习率调度器（配合GeodesicAdam使用）
# ─────────────────────────────────────────────
class LorentzCosineScheduler:
    """
    配合洛伦兹四阶段训练的余弦学习率调度器

    四阶段（对应train.py的训练协议）：
      Phase 0 [0, warmup):          线性warmup，α=0（标准注意力）
      Phase 1 [warmup, 10%):        余弦衰减，仅正则化，α=0
      Phase 2 [10%, 90%):           余弦衰减，全组件，α=lorentz_alpha
      Phase 3 [90%, 100%]:          余弦衰减+α退火到0，逐步关闭洛伦兹

    用法：
      scheduler = LorentzCosineScheduler(optimizer, total_steps,
                                          warmup_steps, base_lr)
      # 每步调用：
      scheduler.step(global_step, model)
    """

    def __init__(self,
                 optimizer: GeodesicAdam,
                 total_steps: int,
                 warmup_steps: int,
                 base_lr: float,
                 min_lr: float = 1e-5,
                 lorentz_alpha: float = 0.25,
                 lorentz_start: float = 0.5):

        self.optimizer     = optimizer
        self.total_steps   = total_steps
        self.warmup_steps  = warmup_steps
        self.base_lr       = base_lr
        self.min_lr        = min_lr
        self.lorentz_alpha = lorentz_alpha

        # 阶段边界（与训练循环的lorentz_start_step保持一致）
        self.phase1_end = int(total_steps * lorentz_start)
        self.phase2_end = int(total_steps * 0.90)

    def get_lr(self, step: int) -> float:
        """计算当前步的学习率"""
        if step < self.warmup_steps:
            # 线性warmup
            return self.base_lr * step / max(1, self.warmup_steps)

        # warmup结束后：余弦衰减
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def get_alpha(self, step: int) -> float:
        """计算当前步的洛伦兹强度α
        
        两阶段激活（避免norm blend和attn同时冲击）：
          Phase 2 进入后 0~1000步: blend warmup, α=0 (norm先适应)
          Phase 2 进入后 1000~2000步: α线性增加到lorentz_alpha (attn后激活)
        """
        if step < self.phase1_end:
            return 0.0   # Phase 1：纯标准注意力

        if step < self.phase2_end:
            phase2_steps = step - self.phase1_end
            if phase2_steps < 1000:
                # Stage A: norm blend warmup 단계, α=0 유지
                return 0.0
            else:
                # Stage B: α warmup (blend는 이미 1.0)
                alpha_warmup = min(1.0, (phase2_steps - 1000) / 1000.0)
                return self.lorentz_alpha * alpha_warmup

        # Phase 3：α余弦退火到0
        progress = (step - self.phase2_end) / max(
            1, self.total_steps - self.phase2_end)
        return self.lorentz_alpha * 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self, global_step: int, model=None):
        """
        更新学习率和α

        global_step: 当前全局训练步
        model:       LorentzTransformer实例（用于更新α）
        """
        lr    = self.get_lr(global_step)
        alpha = self.get_alpha(global_step)

        # 更新学习率
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

        # 更新洛伦兹强度
        if model is not None:
            model.set_lorentz_alpha(alpha)

        return lr, alpha

    def report(self, step: int) -> str:
        lr    = self.get_lr(step)
        alpha = self.get_alpha(step)
        phase = (0 if step < self.warmup_steps else
                 1 if step < self.phase1_end else
                 2 if step < self.phase2_end else 3)
        return (f'step={step}  phase={phase}  '
                f'lr={lr:.2e}  α={alpha:.4f}')

    def set_lorentz_start(self, start_ratio: float):
        """两阶段训练：设置洛伦兹激活起点"""
        self.phase1_end = int(self.total_steps * start_ratio)
        self.phase2_end = int(self.total_steps * (
            start_ratio + (1.0 - start_ratio) * 0.9))


# ─────────────────────────────────────────────
# 快速验证
# ─────────────────────────────────────────────

# ==========================================================
# Part 3: 训练
# ==========================================================
# ============================================================
# train.py
# 洛伦兹Transformer完整训练循环
#
# 用法：
#   python train.py                          # 默认配置
#   python train.py --d_model 256 --n_layers 6
#   python train.py --dataset wikitext       # 真实数据
#   python train.py --resume checkpoint.pt   # 从断点继续
#
# 四阶段训练协议（LorentzCosineScheduler）：
#   Phase 0 [0, warmup):    warmup，α=0（标准注意力）
#   Phase 1 [warmup, 10%):  余弦衰减，仅正则化，α=0
#   Phase 2 [10%, 90%):     全组件，α=lorentz_alpha
#   Phase 3 [90%, 100%]:    α余弦退火到0，稳定收敛
#
# 诊断输出（每eval_interval步）：
#   - train/val loss
#   - 各层类时比例（timelike_frac）
#   - 光锥统计（真实链vs噪声的类时分布）
#   - r规律探针（delta = with_lorentz - without_lorentz）
# ============================================================



# ─────────────────────────────────────────────
# 训练配置
# ─────────────────────────────────────────────
@dataclass
class TrainConfig:
    # 数据
    dataset:      str   = 'synthetic'   # 'synthetic' | 'wikitext' | 'openwebtext'
    data_dir:     str   = './data'
    n_hops:       int   = 2             # 合成任务的跳数（synthetic模式）
    n_entities:   int   = 32
    n_train:      int   = 32000
    n_val:        int   = 4000

    # 训练
    total_steps:  int   = 10000
    batch_size:   int   = 64
    grad_clip:    float = 1.0
    base_lr:      float = 3e-4
    min_lr:       float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int   = 200

    # GeodesicAdam
    scale_t:        float = 2.0
    scale_s:        float = 0.5
    adaptive_scale: bool  = False

    # 日志和保存
    eval_interval:  int  = 200
    save_interval:  int  = 1000
    log_interval:   int  = 50
    out_dir:        str  = './checkpoints'

    # 两阶段训练
    lorentz_start:  float = 0.5   # 前N%标准训练，后(1-N)%洛伦兹激活

    # 恢复
    resume:         str  = ''    # checkpoint路径

    # 设备
    device:         str  = 'auto'


# ─────────────────────────────────────────────
# 合成数据生成（无需外部数据集）
# ─────────────────────────────────────────────
def generate_synthetic(cfg: TrainConfig, model_cfg: LorentzConfig,
                       seed: int = 0):
    """
    生成多跳推理合成数据集

    格式：[pair1_e0, pair1_e1, ..., SEP, query_e0, MASK] → answer
    每个样本包含真实链 + 噪声对，模型需要找到正确的多跳答案。
    """
    rng   = np.random.RandomState(seed)
    n_ent = cfg.n_entities
    SEP   = n_ent; PAD = n_ent+1; MASK = n_ent+2
    vocab = n_ent + 3

    # 更新model_cfg的vocab_size
    model_cfg.vocab_size = vocab

    total = cfg.n_train + cfg.n_val
    X, Y  = [], []

    for _ in range(total):
        entities   = rng.choice(n_ent, cfg.n_hops+1, replace=False)
        chain      = entities.tolist()
        real_pairs = [[chain[i], chain[i+1]] for i in range(cfg.n_hops)]
        used       = set(chain)

        # 噪声对
        noise_pairs = []
        for _ in range(2):
            rem = [e for e in range(n_ent) if e not in used]
            if len(rem) < 2: break
            rng.shuffle(rem)
            noise_pairs.append([rem[0], rem[1]])
            used.update(rem[:2])

        all_pairs = real_pairs + noise_pairs
        perm      = rng.permutation(len(all_pairs)).tolist()
        all_pairs = [all_pairs[i] for i in perm]

        tokens = []
        for p in all_pairs: tokens += p
        tokens += [SEP, chain[0], MASK]

        seq_len = model_cfg.max_seq_len
        while len(tokens) < seq_len: tokens.append(PAD)
        X.append(tokens[:seq_len])
        Y.append(chain[-1])

    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    return (X[:cfg.n_train], Y[:cfg.n_train],
            X[cfg.n_train:], Y[cfg.n_train:], vocab)


# ─────────────────────────────────────────────
# 真实数据加载（wikitext / openwebtext）
# ─────────────────────────────────────────────
def load_real_dataset(cfg: TrainConfig, model_cfg: LorentzConfig):
    """
    加载真实语言数据集

    需要：pip install datasets transformers
    返回：(train_tokens, val_tokens) — 已tokenize的长序列
    """
    try:
        from datasets import load_dataset
        from transformers import GPT2Tokenizer
    except ImportError:
        raise ImportError(
            "真实数据集需要：pip install datasets transformers")

    print(f"加载数据集: {cfg.dataset}")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model_cfg.vocab_size = tokenizer.vocab_size   # 50257

    if cfg.dataset == 'wikitext':
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1')   # 快速版（2M tokens）
    elif cfg.dataset == 'openwebtext':
        pct = getattr(cfg, 'owt_subset_pct', 10)
        print(f"  OpenWebText subset: {pct}%")
        ds_train = load_dataset('openwebtext', split=f'train[:{pct}%]')
        ds = {'train': ds_train}
    else:
        raise ValueError(f"未知数据集: {cfg.dataset}")

    train_ids = []
    val_ids   = []
    print("  Tokenizing...")
    # openwebtext는 validation split이 없으므로 train의 마지막 1%를 사용
    if cfg.dataset == 'openwebtext':
        all_ids = []
        count = 0
        for text in ds['train']['text']:
            if not text.strip(): continue
            ids = tokenizer(text, add_special_tokens=False)['input_ids']
            all_ids.extend(ids)
            count += 1
            if count % 5000 == 0:
                print(f"    train: {count} docs, {len(all_ids):,} tokens")
        split_idx = int(len(all_ids) * 0.99)
        train_ids = all_ids[:split_idx]
        val_ids   = all_ids[split_idx:]
        print(f"  train={len(train_ids):,}  val={len(val_ids):,} tokens  OK")
        return (torch.tensor(train_ids, dtype=torch.long),
                torch.tensor(val_ids,   dtype=torch.long))

    for split, target in [('train', train_ids), ('validation', val_ids)]:
        if split not in ds: continue
        count = 0
        for text in ds[split]['text']:
            if not text.strip(): continue
            # 긴 문서를 max_seq_len 단위로 분할 (truncation 대신)
            ids = tokenizer(text, add_special_tokens=False)['input_ids']
            target.extend(ids)
            count += 1
            if count % 1000 == 0 and count > 0:
                print(f"    {split}: {count} docs, {len(target):,} tokens")
    print(f"  train={len(train_ids):,}  val={len(val_ids):,} tokens  OK")

    return (torch.tensor(train_ids, dtype=torch.long),
            torch.tensor(val_ids,   dtype=torch.long))


# ─────────────────────────────────────────────
# 批次生成
# ─────────────────────────────────────────────
class SyntheticLoader:
    """合成数据批次加载器"""
    def __init__(self, X, Y, batch_size, device):
        self.X = X; self.Y = Y
        self.batch_size = batch_size
        self.device     = device
        self.idx        = 0

    def next_batch(self):
        n = len(self.X)
        if self.idx + self.batch_size > n:
            perm     = torch.randperm(n)
            self.X   = self.X[perm]
            self.Y   = self.Y[perm]
            self.idx = 0
        xb = self.X[self.idx:self.idx+self.batch_size].to(self.device)
        yb = self.Y[self.idx:self.idx+self.batch_size].to(self.device)
        self.idx += self.batch_size
        return xb, yb

    def eval_loss(self, model, mask_token_id, n_batches=10):
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(n_batches):
                xb, yb = self.next_batch()
                mp = (xb == mask_token_id).nonzero(as_tuple=False)[:, 1]
                logits = model(xb)
                pred   = logits[torch.arange(len(xb)), mp]
                losses.append(F.cross_entropy(pred, yb).item())
        model.train()
        return np.mean(losses)


class TokenLoader:
    """真实语言数据批次加载器"""
    def __init__(self, tokens, seq_len, batch_size, device):
        self.tokens     = tokens
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.device     = device

    def next_batch(self):
        n    = len(self.tokens) - self.seq_len - 1
        if n <= 0:
            raise ValueError(f"Token count({len(self.tokens)}) < seq_len({self.seq_len})")
        idxs = torch.randint(0, n, (self.batch_size,))
        x    = torch.stack([self.tokens[i:i+self.seq_len] for i in idxs])
        y    = torch.stack([self.tokens[i+1:i+self.seq_len+1] for i in idxs])
        return x.to(self.device), y.to(self.device)

    def eval_loss(self, model, n_batches=10):
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(n_batches):
                xb, yb = self.next_batch()
                logits = model(xb)
                loss   = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), yb.view(-1))
                losses.append(loss.item())
        model.train()
        return np.mean(losses)


# ─────────────────────────────────────────────
# 诊断: r规律探针
# ─────────────────────────────────────────────
@torch.no_grad()
def r_law_probe(model: LorentzTransformer,
                xb: torch.Tensor, yb: torch.Tensor,
                mask_token_id: int) -> dict:
    """
    r规律探针：比较当前模型 vs 关闭洛伦兹的模型的准确率差值。
    进入前完整保存所有运行态，退出前完整恢复，不污染训练状态。
    """
    was_training = model.training
    model.eval()

    # ── 保存完整运行态快照（包括lightcone诊断缓存）──
    # 실제 운행 시 alpha는 attn.alpha에 저장 (set_lorentz_alpha()가 config를 쓰지 않으므로)
    snap_alpha = model.blocks[0].attn.alpha if model.blocks else model.config.lorentz_alpha
    snap_pos_blend = model.pos_enc._blend
    snap_pos_has_mask = model.pos_enc._has_mask
    snap_norm_f_has_mask = getattr(model.norm_f, '_has_mask', None)
    # 快照lightcone诊断缓存，防止probe污染后续eval读取
    snap_intervals = {}
    for li, blk in enumerate(model.blocks):
        snap_intervals[li] = {
            'last_intervals':     blk.attn.last_intervals,
            'last_intervals_raw': blk.attn.last_intervals_raw,
        }

    snap_blocks = []
    for blk in model.blocks:
        snap_blocks.append({
            'attn_has_mask': blk.attn._has_mask,
            'norm1_blend':   getattr(blk.norm1, '_blend', 0.0),
            'norm2_blend':   getattr(blk.norm2, '_blend', 0.0),
            'norm1_has_mask': getattr(blk.norm1, '_has_mask', None),
            'norm2_has_mask': getattr(blk.norm2, '_has_mask', None),
            'ff_has_mask':   getattr(blk.ff,   '_has_mask', None),
            'ff_blend':      getattr(blk.ff,   '_blend',    0.0),
        })

    mp = (xb == mask_token_id).nonzero(as_tuple=False)[:, 1]

    # ── acc_with：使用当前完整洛伦兹状态 ──
    logits = model(xb)
    pred   = logits[torch.arange(len(xb)), mp].argmax(-1)
    acc_with = (pred == yb).float().mean().item()

    # ── 完全关闭所有洛伦兹组件 ──
    model.set_lorentz_alpha(0.0)
    model.pos_enc.set_blend(0.0)
    model.pos_enc._has_mask = False
    if snap_norm_f_has_mask is not None:
        model.norm_f._has_mask = False
    if hasattr(model.norm_f, 'set_blend'):
        model.norm_f.set_blend(0.0)
    for blk in model.blocks:
        blk.attn._has_mask = False
        if hasattr(blk.norm1, '_has_mask'):
            blk.norm1._has_mask = False
        if hasattr(blk.norm1, 'set_blend'):
            blk.norm1.set_blend(0.0)
        if hasattr(blk.norm2, '_has_mask'):
            blk.norm2._has_mask = False
        if hasattr(blk.norm2, 'set_blend'):
            blk.norm2.set_blend(0.0)
        if hasattr(blk.ff, '_has_mask'):
            blk.ff._has_mask = False
        if hasattr(blk.ff, 'set_blend'):
            blk.ff.set_blend(0.0)

    logits0 = model(xb)
    pred0   = logits0[torch.arange(len(xb)), mp].argmax(-1)
    acc_without = (pred0 == yb).float().mean().item()

    # ── 完整恢复快照 ──
    model.set_lorentz_alpha(snap_alpha)
    model.pos_enc.set_blend(snap_pos_blend)
    model.pos_enc._has_mask = snap_pos_has_mask
    if snap_norm_f_has_mask is not None:
        model.norm_f._has_mask = snap_norm_f_has_mask
    if hasattr(model.norm_f, 'set_blend'):
        model.norm_f.set_blend(getattr(model.norm_f, '_blend', 0.0))
    for blk, snap in zip(model.blocks, snap_blocks):
        blk.attn._has_mask = snap['attn_has_mask']
        if snap['norm1_has_mask'] is not None:
            blk.norm1._has_mask = snap['norm1_has_mask']
        if hasattr(blk.norm1, 'set_blend'):
            blk.norm1.set_blend(snap.get('norm1_blend', 0.0))
        if snap['norm2_has_mask'] is not None:
            blk.norm2._has_mask = snap['norm2_has_mask']
        if hasattr(blk.norm2, 'set_blend'):
            blk.norm2.set_blend(snap.get('norm2_blend', 0.0))
        if snap['ff_has_mask'] is not None:
            blk.ff._has_mask = snap['ff_has_mask']
        if hasattr(blk.ff, 'set_blend'):
            blk.ff.set_blend(snap['ff_blend'])

    # 恢复lightcone诊断缓存（防止probe两次前向污染eval读取）
    for li, blk in enumerate(model.blocks):
        if li in snap_intervals:
            blk.attn.last_intervals     = snap_intervals[li]['last_intervals']
            blk.attn.last_intervals_raw = snap_intervals[li]['last_intervals_raw']

    # 恢复进入前的training状态（不无条件设为train）
    model.train(was_training)

    return {
        'acc_with':    acc_with,
        'acc_without': acc_without,
        'delta':       acc_with - acc_without,
    }


# ─────────────────────────────────────────────
# 主训练循环
# ─────────────────────────────────────────────
def train(train_cfg: TrainConfig, model_cfg: LorentzConfig):
    # ── 设备设置 ──
    if train_cfg.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(train_cfg.device)
    print(f"Device: {device}")

    # ── 输出目录 ──
    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 数据加载 ──
    print(f"\n数据集: {train_cfg.dataset}")
    if train_cfg.dataset == 'synthetic':
        (X_tr, Y_tr, X_vl, Y_vl, vocab) = generate_synthetic(
            train_cfg, model_cfg)
        mask_token = vocab - 1
        train_loader = SyntheticLoader(X_tr, Y_tr,
                                        train_cfg.batch_size, device)
        val_loader   = SyntheticLoader(X_vl, Y_vl,
                                        train_cfg.batch_size, device)
        is_synthetic = True
        print(f"  合成数据: train={len(X_tr)}  val={len(X_vl)}  "
              f"vocab={vocab}  {train_cfg.n_hops}-hop")
    else:
        train_tok, val_tok = load_real_dataset(train_cfg, model_cfg)
        train_loader = TokenLoader(train_tok, model_cfg.max_seq_len,
                                   train_cfg.batch_size, device)
        val_loader   = TokenLoader(val_tok,   model_cfg.max_seq_len,
                                   train_cfg.batch_size, device)
        is_synthetic = False
        mask_token   = -1
        print(f"  真实数据: train={len(train_tok):,}  val={len(val_tok):,}")

    # ── 模型初始化 ──
    _resume_step        = 0
    _resume_phase2      = False
    _resume_best_loss   = float('inf')
    if train_cfg.resume:
        print(f"\n从Checkpoint恢复: {train_cfg.resume}")
        model = LorentzTransformer.load_lorentz_state(
            train_cfg.resume, device=str(device))
        model_cfg = model.config
        # 훈련 재개 상태 읽기
        _resume_step      = model._saved_train_step     or 0
        _resume_phase2    = model._saved_phase2_entered or False
        _resume_best_loss = model._saved_best_val_loss  or float('inf')
        print(f"  재개 step={_resume_step}  phase2={_resume_phase2}  "
              f"best={_resume_best_loss:.4f}")
    else:
        model = LorentzTransformer(model_cfg).to(device)

    n_params = model.get_num_params() / 1e6
    print(f"\n模型: {n_params:.1f}M params  "
          f"d={model_cfg.d_model}  L={model_cfg.n_layers}  "
          f"H={model_cfg.n_heads}  α={model_cfg.lorentz_alpha}")

    # ── 优化器 ──
    optimizer = GeodesicAdam(
        model.parameters(),
        lr=train_cfg.base_lr,
        weight_decay=train_cfg.weight_decay,
        scale_t=train_cfg.scale_t,
        scale_s=train_cfg.scale_s,
        adaptive_scale=train_cfg.adaptive_scale,
    )
    # 재개 시 옵티마이저 상태 복원
    if train_cfg.resume and _resume_step > 0:
        # 모델 checkpoint와 쌍을 이루는 optimizer 파일 찾기
        # 규칙: best.pt → best_opt.pt, step_N.pt → step_N_opt.pt
        _ckpt_stem = Path(train_cfg.resume).stem   # e.g. "best", "step_5000"
        opt_path = str(Path(train_cfg.resume).parent / f'{_ckpt_stem}_opt.pt')
        import os as _os
        if _os.path.exists(opt_path):
            opt_state = torch.load(opt_path, map_location=device,
                                   weights_only=False)
            # GeodesicAdam.load_lorentz()는 classmethod로 새 인스턴스 반환
            # 기존 optimizer에 state를 직접 로드하려면 load_state_dict 사용
            lorentz_extra = opt_state.pop('lorentz', {})
            optimizer.load_state_dict(opt_state)
            # scale_t/s/adaptive_scale 복원
            optimizer.scale_t        = lorentz_extra.get('scale_t',        train_cfg.scale_t)
            optimizer.scale_s        = lorentz_extra.get('scale_s',        train_cfg.scale_s)
            optimizer.adaptive_scale = lorentz_extra.get('adaptive_scale', train_cfg.adaptive_scale)
            print(f"  옵티마이저 상태 복원: {opt_path}")
        else:
            print(f"  경고: 옵티마이저 상태 파일 없음 ({opt_path}), 새로 시작")

    # ── 调度器 ──
    scheduler = LorentzCosineScheduler(
        optimizer,
        total_steps   = train_cfg.total_steps,
        warmup_steps  = train_cfg.warmup_steps,
        base_lr       = train_cfg.base_lr,
        min_lr        = train_cfg.min_lr,
        lorentz_alpha = model_cfg.lorentz_alpha,
        lorentz_start = train_cfg.lorentz_start,
    )

    # ──   ──
    log = {
        'train_loss': [], 'val_loss': [],
        'timelike_fracs': [], 'r_law_delta': [],
        'lr': [], 'alpha': [], 'step': [],
    }
    best_val_loss = _resume_best_loss
    _phase2_entered_init = _resume_phase2
    t0 = time.time()

    print(f"\n{'='*55}")
    print(f"训练开始  total_steps={train_cfg.total_steps}")
    print(f"{'='*55}\n")

    # ── 两阶段边界 ──
    lorentz_start_step = int(train_cfg.total_steps * train_cfg.lorentz_start)
    scheduler.set_lorentz_start(train_cfg.lorentz_start)
    _phase2_entered = _phase2_entered_init  # resume 시 Phase 2 상태 복원

    print(f"两阶段训练:")
    print(f"  Phase 1 [1~{lorentz_start_step}步]: 标准Adam，充分收敛")
    print(f"  Phase 2 [{lorentz_start_step+1}~{train_cfg.total_steps}步]: "
          f"洛伦兹激活 (α={model_cfg.lorentz_alpha})")
    print()

    # 训练开始前确定洛伦兹路径（用初始配置值，不用被调度器修改的运行值）
    _initial_lorentz_alpha = model_cfg.lorentz_alpha
    is_lorentz_active = (_initial_lorentz_alpha > 0)
    use_norm_ablation  = (model_cfg.use_minkowski_norm and not is_lorentz_active)

    # 实验路径标签（日志可读性）
    if use_norm_ablation:
        print("⚠ 注意: use_minkowski_norm=True 且 lorentz_alpha=0")
        print("  MinkowskiLayerNorm在mask未注入时退化为标准LayerNorm。")
        print("  此路径等同于baseline，不是独立的LayerNorm消融实验。")
        print("  如需真正的MinkowskiLayerNorm效果，请设置lorentz_alpha>0。")

    # ──   ──
    for step in range(_resume_step + 1, train_cfg.total_steps + 1):

        # Phase 2 进入处理（仅洛伦兹模式）
        if is_lorentz_active and step == lorentz_start_step + 1 and not _phase2_entered:
            _phase2_entered = True
            print(f"\n{'='*55}")
            print(f"Phase 2 开始 (step={step}): 洛伦兹激活")
            print("  在收敛的模型上初始化P_t (3次)...")
            # 초기화: warmup과 freq 체크 모두 우회
            # hess_update_freq의 배수이면서 hess_warmup_steps보다 큰 값 사용
            _freq = model_cfg.hess_update_freq
            _warm = model_cfg.hess_warmup_steps
            _init_step = (_warm // _freq + 1) * _freq  # 첫 번째 유효한 스텝
            for _ in range(3):
                xb_init, _ = train_loader.next_batch()
                model.probe.step(xb_init, _init_step)
            fracs = model.probe.timelike_fracs
            lmins = model.probe.lambda_mins
            print(f"  类时比例: {[round(f,3) for f in fracs]}")
            print(f"  λ_min:    {[f'{l:.2e}' for l in lmins]}")
            print(f"  MinkowskiLayerNorm: blend=0固定（标准norm），Minkowski效果仅通过attn的α实现")
            print(f"{'='*55}\n")

        # blend 전체 비활성화:
        # pos_enc blend, norm blend 모두 0.0 고정
        # Minkowski 기하 효과는 오직 attention α로만 구현
        # (pos_enc 전환도 수렴 후 충격 발생 확인)
        model.pos_enc.set_blend(0.0)
        for blk in model.blocks:
            if hasattr(blk.norm1, 'set_blend'):
                blk.norm1.set_blend(0.0)
            if hasattr(blk.norm2, 'set_blend'):
                blk.norm2.set_blend(0.0)
            blk.ff.set_blend(0.0)
        if hasattr(model.norm_f, 'set_blend'):
            model.norm_f.set_blend(0.0)

        # 调度器更新
        lr, alpha = scheduler.step(step, model)

        # P_t更新：仅洛伦兹模式的Phase 2
        if is_lorentz_active and step > lorentz_start_step:
            xb_probe, _ = train_loader.next_batch()
            model.probe.step(xb_probe, step)
            pairs = model.probe.get_param_mask_pairs(device)
            optimizer.update_masks(pairs)
        elif use_norm_ablation:
            # MinkowskiLayerNorm消融注意：mask未注入时MinkowskiLayerNorm
            # 退化为标准LayerNorm行为（见forward()），因此此路径
            # 实际上等同于baseline，不能单独支撑"LayerNorm组件有效"的结论。
            # 真正的MinkowskiLayerNorm效果需要Phase 2注入mask后才体现。
            xb_probe = None
            model.set_lorentz_alpha(0.0)
        else:
            xb_probe = None   # 明确标记：baseline不使用xb_probe
            model.set_lorentz_alpha(0.0)

        # 加载批次
        if is_synthetic:
            xb, yb = train_loader.next_batch()
        else:
            xb, yb = train_loader.next_batch()

        # 前向传播
        model.train()
        logits = model(xb)

        # 计算损失
        if is_synthetic:
            mp     = (xb == mask_token).nonzero(as_tuple=False)[:, 1]
            pred   = logits[torch.arange(len(xb)), mp]
            loss   = F.cross_entropy(pred, yb)
        else:
            # TokenLoader已经返回x[t]->y[t]=x[t+1]，直接对齐
            loss = F.cross_entropy(
                logits.reshape(-1, model_cfg.vocab_size),
                yb.reshape(-1))

        # 正则化损失 (Component 3)
        reg_loss  = model.regularization_loss()
        # 光锥损失（Phase 2에서만 활성화）
        cone_loss = (model.lightcone_loss(model_cfg.lambda_cone)
                     if step > lorentz_start_step and model_cfg.lambda_cone > 0
                     else torch.tensor(0.0, device=device))
        total_loss = loss + reg_loss + cone_loss

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        train_cfg.grad_clip)
        optimizer.step()

        # ──   ──
        if step % train_cfg.log_interval == 0:
            fracs = model.probe.timelike_fracs
            fstr  = ' '.join(f'{f:.2f}' for f in fracs)
            elapsed = time.time() - t0
            print(f"step={step:5d}  loss={loss.item():.4f}  "
                  f"reg={reg_loss.item():.2e}  "
                  f"cone={cone_loss.item():.2e}  "
                  f"lr={lr:.2e}  α={alpha:.3f}  "
                  f"timelike=[{fstr}]  "
                  f"t={elapsed:.0f}s")

        # ──  ──
        if step % train_cfg.eval_interval == 0:
            if is_synthetic:
                val_loss = val_loader.eval_loss(
                    model, mask_token, n_batches=20)
            else:
                val_loss = val_loader.eval_loss(model, n_batches=20)

            # r规律  ( )
            r_delta = None
            if is_synthetic:
                xb_r, yb_r = val_loader.next_batch()
                probe_result = r_law_probe(
                    model, xb_r, yb_r, mask_token)
                r_delta = probe_result['delta']
                r_str   = (f"  r_law: acc_with={probe_result['acc_with']:.3f}  "
                           f"acc_without={probe_result['acc_without']:.3f}  "
                           f"delta={r_delta:+.4f}")
            else:
                r_str = ""

            # 光锥统计前向（仅洛伦兹模式，baseline跳过以避免xb_probe未定义）
            if is_lorentz_active and step > lorentz_start_step:
                _ = model(xb_probe)
            else:
                _ = model(xb)
            lc    = model.lightcone_stats()
            lc_str = '  '.join(
                f"L{li}:{s['timelike_frac']:.2f}"
                for li, s in enumerate(lc.values()))

            print(f"\n{'─'*55}")
            print(f"[EVAL step={step}]  val_loss={val_loss:.4f}  "
                  f"(best={best_val_loss:.4f})")
            print(f"  光锥: [{lc_str}]")
            if r_str: print(r_str)
            print(f"  {model.probe.report()}")
            print(f"{'─'*55}\n")

            # 保存日志
            log['step'].append(step)
            log['val_loss'].append(val_loss)
            log['train_loss'].append(loss.item())
            log['timelike_fracs'].append(model.probe.timelike_fracs[:])
            log['r_law_delta'].append(r_delta)
            log['lr'].append(lr)
            log['alpha'].append(alpha)

            # 保存最优检查点
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = str(out_dir / 'best.pt')
                # 훈련 상태를 모델에 임시 첨부 후 저장
                model._saved_train_step     = step
                model._saved_best_val_loss  = best_val_loss
                model._saved_phase2_entered = _phase2_entered
                model.save_lorentz_state(ckpt_path)
                # optimizer 파일명 = 모델 파일명 + _opt (항상 쌍으로 매칭)
                opt_ckpt = str(out_dir / 'best_opt.pt')
                torch.save(optimizer.state_dict_lorentz(), opt_ckpt)
                print(f"  ✓ best checkpoint : {ckpt_path}")

        # ──   ──
        if step % train_cfg.save_interval == 0:
            ckpt_path = str(out_dir / f'step_{step}.pt')
            model._saved_train_step     = step
            model._saved_best_val_loss  = best_val_loss
            model._saved_phase2_entered = _phase2_entered
            model.save_lorentz_state(ckpt_path)
            # optimizer 파일명 = step_N_opt.pt (모델과 항상 쌍)
            torch.save(optimizer.state_dict_lorentz(),
                       str(out_dir / f'step_{step}_opt.pt'))
            with open(str(out_dir / 'log.json'), 'w') as f:
                json.dump(log, f, indent=2)
            print(f"  checkpoint: {ckpt_path}")

    # ──   ──
    # final.pt에도 resume metadata 첨부
    model._saved_train_step     = train_cfg.total_steps
    model._saved_best_val_loss  = best_val_loss
    model._saved_phase2_entered = _phase2_entered
    model.save_lorentz_state(str(out_dir / 'final.pt'))
    torch.save(optimizer.state_dict_lorentz(),
               str(out_dir / 'final_opt.pt'))
    with open(str(out_dir / 'log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    # r_law delta 최종 통계
    deltas = [d for d in log['r_law_delta'] if d is not None]
    if deltas:
        import numpy as np
        d_arr   = np.array(deltas)
        d_mean  = d_arr.mean()
        d_pos   = (d_arr > 0).sum()
        d_neg   = (d_arr < 0).sum()
        d_zero  = (d_arr == 0).sum()
        verdict = ('✓ 洛伦兹有益' if d_mean > 0.005 else
                   '✗ 洛伦兹有害' if d_mean < -0.005 else
                   '△ 接近零（临界区）')
    else:
        d_mean = 0; d_pos = d_neg = d_zero = 0
        verdict = '无数据'

    print(f"\n{'='*55}")
    print(f"训练完成  best_val_loss={best_val_loss:.4f}")
    print(f"{'='*55}")
    print(f"r_law delta 统计:")
    print(f"  均值={d_mean:+.4f}  正={d_pos}次  负={d_neg}次  零={d_zero}次")
    print(f"  判断: {verdict}")
    print(f"Checkpoints: {out_dir}")
    print(f"{'='*55}")
    return log


# ─────────────────────────────────────────────
#  
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
#  
# ─────────────────────────────────────────────

def scale_train(d_model=256, n_layers=6, n_hops=2,
                total_steps=5000, lorentz_alpha=0.25):
    """
    规模验证训练
    默认: 256维/6层（比quick_train大4倍，比GPT-2小3倍）
    也可用: d_model=512, n_layers=8
    """
    model_cfg = LorentzConfig(
        d_model=d_model,
        n_heads=max(4, d_model//64),   # 每head 64维
        n_layers=n_layers,
        max_seq_len=64,
        lorentz_alpha=lorentz_alpha,
        hess_warmup_steps=200,
        hess_update_freq=40,
        hutchinson_k=20,
        lambda_spacelike=0.0,
    )
    n_params = (sum([
        d_model * d_model * 4,          # QKV + O proj per layer
        d_model * d_model * 4 * 2,      # FFN
        d_model,                         # LayerNorm
    ]) * n_layers + d_model * 35) / 1e6
    print(f"规模: d={d_model}  L={n_layers}  "
          f"H={model_cfg.n_heads}  ~{n_params:.1f}M params")

    # 学习率根据模型大小自动调整
    lr = 3e-4 if d_model <= 256 else 2e-4   # 512维用2e-4，比256维大2倍
    train_cfg = TrainConfig(
        dataset='synthetic', n_hops=n_hops,
        n_train=32000, n_val=4000,
        total_steps=total_steps, batch_size=32,
        base_lr=lr,
        scale_t=2.0, scale_s=0.5,
        lorentz_start=0.5,
        eval_interval=1000, log_interval=500,   # 15000步调整输出频率
        out_dir=f'./checkpoints_scale_{d_model}',
    )
    return train(train_cfg, model_cfg)


def openwebtext_train(d_model=256, n_layers=6, total_steps=50000,
                     lorentz_alpha=0.25, subset_pct=10):
    """
    OpenWebText 대규모 데이터 훈련。
    wikitext-2의 ~1600배 데이터로 모델을 충분히 수렴시킨다。

    subset_pct: 전체 OpenWebText의 몇 %를 사용할지（기본 10%=~400M tokens）
      - subset_pct=1  → ~40M tokens  (빠른 테스트)
      - subset_pct=10 → ~400M tokens (추천)
      - subset_pct=100→ ~4B tokens   (전체, 매우 느림)

    필요: pip install datasets transformers
    Colab Pro+ 또는 A100 권장。
    """
    model_cfg = LorentzConfig(
        vocab_size=50257,
        d_model=d_model,
        n_heads=max(4, d_model//64),
        n_layers=n_layers,
        max_seq_len=512,
        lorentz_alpha=lorentz_alpha,
        hess_warmup_steps=5000,         # 더 큰 데이터에 맞게 warmup 연장
        hess_update_freq=200,
        hutchinson_k=10,
        lambda_spacelike=0.0,
        lambda_cone=0.0,
        use_minkowski_norm=True,
    )
    lr = 3e-4 if d_model <= 256 else 1e-4
    train_cfg = TrainConfig(
        dataset=f'openwebtext',
        total_steps=total_steps, batch_size=32,  # 더 큰 배치
        base_lr=lr,
        scale_t=2.0, scale_s=0.5,
        lorentz_start=0.5,
        eval_interval=2000, log_interval=500,
        out_dir=f'./checkpoints_owt_{d_model}',
    )
    # openwebtext subset 설정
    train_cfg.owt_subset_pct = subset_pct
    return train(train_cfg, model_cfg)


def openwebtext_baseline(d_model=256, n_layers=6, total_steps=50000,
                         subset_pct=10):
    """OpenWebText 標準Transformer 对照组"""
    model_cfg = LorentzConfig(
        vocab_size=50257,
        d_model=d_model,
        n_heads=max(4, d_model//64),
        n_layers=n_layers,
        max_seq_len=512,
        lorentz_alpha=0.0,
        hess_warmup_steps=5000,
        hess_update_freq=200,
        hutchinson_k=10,
        lambda_spacelike=0.0,
        lambda_cone=0.0,
        use_minkowski_norm=False,
    )
    lr = 3e-4 if d_model <= 256 else 1e-4
    train_cfg = TrainConfig(
        dataset='openwebtext',
        total_steps=total_steps, batch_size=32,
        base_lr=lr,
        scale_t=1.0, scale_s=1.0,
        lorentz_start=1.0,
        eval_interval=2000, log_interval=500,
        out_dir=f'./checkpoints_owt_baseline_{d_model}',
    )
    train_cfg.owt_subset_pct = subset_pct
    return train(train_cfg, model_cfg)


def baseline_train(d_model=256, n_layers=6, total_steps=10000):
    """
    标准Transformer对比实验（版本2的精确对照组）
    与版本2完全相同的参数量和训练设置，仅关闭所有洛伦兹组件。
    用于得到干净的洛伦兹 vs 标准 对比数字。
    """
    model_cfg = LorentzConfig(
        vocab_size=50257,
        d_model=d_model,
        n_heads=max(4, d_model//64),
        n_layers=n_layers,
        max_seq_len=512,
        lorentz_alpha=0.0,              # 关闭Minkowski注意力
        hess_warmup_steps=2000,
        hess_update_freq=100,
        hutchinson_k=10,
        lambda_spacelike=0.0,
        lambda_cone=0.0,
        use_minkowski_norm=False,       # 关闭MinkowskiLayerNorm，用标准LayerNorm
    )
    lr = 3e-4 if d_model <= 256 else 1e-4
    train_cfg = TrainConfig(
        dataset='wikitext',
        total_steps=total_steps, batch_size=16,
        base_lr=lr,
        scale_t=1.0, scale_s=1.0,      # Geodesic Adam无差异化（标准Adam等效）
        lorentz_start=0.5,              # 与wikitext_train相同协议，Phase 2仅α=0不激活洛伦兹
        eval_interval=1000, log_interval=200,
        out_dir=f'./checkpoints_baseline_{d_model}',
    )
    return train(train_cfg, model_cfg)


def wikitext_train(d_model=256, n_layers=6, total_steps=10000,
                   lorentz_alpha=0.25):
    """
    真实语言数据训练（wikitext-2，~2M tokens）
    需要: pip install datasets transformers
    """
    model_cfg = LorentzConfig(
        vocab_size=50257,
        d_model=d_model,
        n_heads=max(4, d_model//64),
        n_layers=n_layers,
        max_seq_len=512,
        lorentz_alpha=lorentz_alpha,
        hess_warmup_steps=2000,
        hess_update_freq=100,
        hutchinson_k=10,
        lambda_spacelike=0.0,
        lambda_cone=0.0,                # 版本2：关闭光锥损失
        use_minkowski_norm=True,        # 版本2：MinkowskiLayerNorm开启
    )
    lr = 3e-4 if d_model <= 256 else 1e-4
    train_cfg = TrainConfig(
        dataset='wikitext',
        total_steps=total_steps, batch_size=16,
        base_lr=lr,
        scale_t=2.0, scale_s=0.5,
        lorentz_start=0.5,
        eval_interval=1000, log_interval=200,
        out_dir=f'./checkpoints_wikitext_{d_model}',
    )
    return train(train_cfg, model_cfg)


def quick_train(n_hops=2, total_steps=2000, d_model=128,
                n_layers=4, lorentz_alpha=0.25):
    """   ( )"""
    model_cfg = LorentzConfig(
        d_model=d_model, n_heads=4, n_layers=n_layers,
        max_seq_len=32, lorentz_alpha=lorentz_alpha,
        hess_warmup_steps=100, hess_update_freq=40,
        hutchinson_k=20,
        lambda_cone=0.01,
    )
    train_cfg = TrainConfig(
        dataset='synthetic', n_hops=n_hops,
        n_train=6400, n_val=1600,
        total_steps=total_steps, batch_size=64,
        base_lr=3e-4, scale_t=2.0, scale_s=0.5,
        lorentz_start=0.5,
        eval_interval=200, log_interval=100,
        out_dir='./checkpoints_quick',
    )
    return train(train_cfg, model_cfg)


def full_train(n_hops=2, total_steps=10000, d_model=128,
               n_layers=4, lorentz_alpha=0.25):
    """本格训练（合成数据，10000步）"""
    model_cfg = LorentzConfig(
        d_model=d_model, n_heads=4, n_layers=n_layers,
        max_seq_len=32, lorentz_alpha=lorentz_alpha,
        hess_warmup_steps=100, hess_update_freq=40,
        hutchinson_k=20, lambda_spacelike=0.0,
    )
    train_cfg = TrainConfig(
        dataset='synthetic', n_hops=n_hops,
        n_train=32000, n_val=4000,
        total_steps=total_steps, batch_size=64,
        base_lr=3e-4, scale_t=2.0, scale_s=0.5,
        lorentz_start=0.5,
        eval_interval=500, log_interval=100,
        out_dir='./checkpoints_full',
    )
    return train(train_cfg, model_cfg)


if __name__ == '__main__':
    # ══════════════════════════════════════════════
    # 当前：洛伦兹版本2（正式对比实验）
    # 对照组已完成：baseline best_val_loss = 6.6794
    # ══════════════════════════════════════════════

    # ── 洛伦兹版本2 ──
    log = wikitext_train(
        d_model      = 256,
        n_layers     = 6,
        total_steps  = 10000,
        lorentz_alpha= 0.25,
    )

    # ── 对照组（已完成，best=6.6794）──
    # log = baseline_train(
    #     d_model     = 256,
    #     n_layers    = 6,
    #     total_steps = 10000,
    # )
