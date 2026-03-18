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
    lorentz_alpha: float = 0.25

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
        self.last_intervals: Optional[torch.Tensor] = None

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

        # attention mask
        if attention_mask is not None:
            scores = scores + attention_mask.to(scores.dtype)

        # 存储洛伦兹间隔（诊断用）
        self.last_intervals = (scores * scale).detach()

        attn_w = F.softmax(scores, dim=-1).to(x.dtype)
        attn_w = self.drop(attn_w)

        out = torch.matmul(attn_w, V)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out), attn_w


# ─────────────────────────────────────────────
# 前馈网络
# ─────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, config: LorentzConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────
class LorentzBlock(nn.Module):
    def __init__(self, config: LorentzConfig):
        super().__init__()
        self.attn  = LorentzMultiHeadAttention(config)
        self.ff    = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

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
                         self.model.pos_emb(
                             torch.arange(xb.shape[1],
                                          device=device).unsqueeze(0)))
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
                self.timelike_fracs[li] = binary_mask.float().mean().item()

                lmin_safe = G_safe.min().item()
                self.lambda_mins[li] = lmin_safe
                new_G_diags.append(G_safe)

            except Exception:
                new_G_diags.append(None)

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

        # 嵌入层
        self.embed   = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop    = nn.Dropout(config.dropout)

        # Transformer块
        self.blocks = nn.ModuleList([
            LorentzBlock(config) for _ in range(config.n_layers)
        ])

        # 输出层
        self.norm_f = nn.LayerNorm(config.d_model)
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

        # 位置编码
        pos = torch.arange(L, device=device).unsqueeze(0)
        h   = self.drop(self.embed(input_ids) + self.pos_emb(pos))

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
            n -= self.pos_emb.weight.numel()
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
        """动态调整洛伦兹强度（用于训练阶段切换）"""
        self.config.lorentz_alpha = alpha
        for block in self.blocks:
            block.attn.alpha = alpha

    def save_lorentz_state(self, path: str):
        """
        保存完整状态，包括P_t的EMA历史

        标准 torch.save(model.state_dict()) 不包含probe的状态。
        这个方法额外保存probe，确保训练可以从任意checkpoint恢复。
        """
        state = {
            'model_state': self.state_dict(),
            'config':      self.config,
            'probe_mask_ema': self.probe._mask_ema,
            'probe_n_updates': self.probe.n_updates,
            'probe_timelike_fracs': self.probe.timelike_fracs,
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
        model.probe._mask_ema       = state['probe_mask_ema']
        model.probe.n_updates       = state['probe_n_updates']
        model.probe.timelike_fracs  = state['probe_timelike_fracs']
        # 恢复各层的timelike_mask
        for li, block in enumerate(model.blocks):
            ema = state['probe_mask_ema'][li]
            mask = (ema > model.probe._threshold).to(device)
            block.attn.set_timelike_mask(mask)
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

                # 类时/类空分解（逐元素乘，对角P_t作用于每行）
                g_t  = mask * g                             # 类时分量
                g_s  = (1.0 - mask) * g                    # 类空分量

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
            f'  timelike_frac={self.last_类时比例:.3f}',
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
                 lorentz_alpha: float = 0.25):

        self.optimizer     = optimizer
        self.total_steps   = total_steps
        self.warmup_steps  = warmup_steps
        self.base_lr       = base_lr
        self.min_lr        = min_lr
        self.lorentz_alpha = lorentz_alpha

        # 阶段边界
        self.phase1_end = int(total_steps * 0.10)
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
        """计算当前步的洛伦兹强度α"""
        if step < self.phase1_end:
            return 0.0   # Phase 0/1：纯标准注意力

        if step < self.phase2_end:
            return self.lorentz_alpha   # Phase 2：完整洛伦兹

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
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1')
    elif cfg.dataset == 'openwebtext':
        ds = load_dataset('openwebtext', split='train[:1%]')
    else:
        raise ValueError(f"未知数据集: {cfg.dataset}")

    def tokenize(examples):
        return tokenizer(examples['text'], truncation=False)

    train_ids = []
    val_ids   = []
    for split, target in [('train', train_ids), ('validation', val_ids)]:
        if split not in ds: continue
        for text in ds[split]['text']:
            ids = tokenizer(text)['input_ids']
            target.extend(ids)

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
        n    = len(self.tokens) - self.seq_len
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
    r规律探针：比较当前模型 vs 关闭洛伦兹的模型的准确率差值

    delta > 0 → 模型还在弱baseline区间，洛伦兹有益
    delta < 0 → 模型已收敛，洛伦兹可能有害
    delta ≈ 0 → 临界点

    用于动态判断是否继续洛伦兹训练
    """
    model.eval()

    # 当前α（有洛伦兹）
    alpha_orig = model.config.lorentz_alpha
    mp = (xb == mask_token_id).nonzero(as_tuple=False)[:, 1]

    logits = model(xb)
    pred   = logits[torch.arange(len(xb)), mp].argmax(-1)
    acc_with = (pred == yb).float().mean().item()

    # 关闭洛伦兹（α=0）
    # alpha=0 + mask 비활성화로 완전한 표준 주의력 구현
    model.set_lorentz_alpha(0.0)
    for block in model.blocks:
        block.attn._has_mask = False
    logits0 = model(xb)
    pred0   = logits0[torch.arange(len(xb)), mp].argmax(-1)
    acc_without = (pred0 == yb).float().mean().item()

    # 恢复
    model.set_lorentz_alpha(alpha_orig)
    for block in model.blocks:
        block.attn._has_mask = block.attn.timelike_mask.any().item()
    model.train()

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
    if train_cfg.resume:
        print(f"\n从Checkpoint恢复: {train_cfg.resume}")
        model = LorentzTransformer.load_lorentz_state(
            train_cfg.resume, device=str(device))
        model_cfg = model.config
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

    # ── 调度器 ──
    scheduler = LorentzCosineScheduler(
        optimizer,
        total_steps  = train_cfg.total_steps,
        warmup_steps = train_cfg.warmup_steps,
        base_lr      = train_cfg.base_lr,
        min_lr       = train_cfg.min_lr,
        lorentz_alpha= model_cfg.lorentz_alpha,
    )

    # ──   ──
    log = {
        'train_loss': [], 'val_loss': [],
        'timelike_fracs': [], 'r_law_delta': [],
        'lr': [], 'alpha': [], 'step': [],
    }
    best_val_loss = float('inf')
    t0 = time.time()

    print(f"\n{'='*55}")
    print(f"训练开始  total_steps={train_cfg.total_steps}")
    print(f"{'='*55}\n")

    # ── 两阶段边界 ──
    lorentz_start_step = int(train_cfg.total_steps * train_cfg.lorentz_start)
    scheduler.set_lorentz_start(train_cfg.lorentz_start)
    _phase2_entered = False

    print(f"两阶段训练:")
    print(f"  Phase 1 [1~{lorentz_start_step}步]: 标准Adam，充分收敛")
    print(f"  Phase 2 [{lorentz_start_step+1}~{train_cfg.total_steps}步]: "
          f"洛伦兹激活 (α={model_cfg.lorentz_alpha})")
    print()

    # ──   ──
    for step in range(1, train_cfg.total_steps + 1):

        # Phase 2 进入处理
        if step == lorentz_start_step + 1 and not _phase2_entered:
            _phase2_entered = True
            print(f"\n{'='*55}")
            print(f"Phase 2 开始 (step={step}): 洛伦兹激活")
            print("  在收敛的模型上初始化P_t (3次)...")
            for _ in range(3):
                xb_init, _ = train_loader.next_batch()
                # warmup   : step=999999
                model.probe.step(xb_init, 999999)
            fracs = model.probe.timelike_fracs
            lmins = model.probe.lambda_mins
            print(f"  类时比例: {[round(f,3) for f in fracs]}")
            print(f"  λ_min:    {[f'{l:.2e}' for l in lmins]}")
            print(f"{'='*55}\n")

        # 调度器更新
        lr, alpha = scheduler.step(step, model)

        # P_t更新：仅Phase 2
        if step > lorentz_start_step:
            xb_probe, _ = train_loader.next_batch()
            model.probe.step(xb_probe, step)
            pairs = model.probe.get_param_mask_pairs(device)
            optimizer.update_masks(pairs)
        else:
            model.set_lorentz_alpha(0.0)  # Phase 1: 洛伦兹 

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
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, model_cfg.vocab_size),
                yb[:, 1:].reshape(-1))

        # 正则化损失 (Component 3)
        reg_loss = model.regularization_loss()
        total_loss = loss + reg_loss

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

            #   (xb_probe  xb )
            _probe_x = xb_probe if step > lorentz_start_step else xb
            _ = model(_probe_x)
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
                model.save_lorentz_state(ckpt_path)
                torch.save(optimizer.state_dict_lorentz(),
                           str(out_dir / 'best_optimizer.pt'))
                print(f"  ✓ best checkpoint : {ckpt_path}")

        # ──   ──
        if step % train_cfg.save_interval == 0:
            ckpt_path = str(out_dir / f'step_{step}.pt')
            model.save_lorentz_state(ckpt_path)
            with open(str(out_dir / 'log.json'), 'w') as f:
                json.dump(log, f, indent=2)
            print(f"  checkpoint: {ckpt_path}")

    # ──   ──
    model.save_lorentz_state(str(out_dir / 'final.pt'))
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

    train_cfg = TrainConfig(
        dataset='synthetic', n_hops=n_hops,
        n_train=32000, n_val=4000,
        total_steps=total_steps, batch_size=32,   # 更大模型用小batch
        base_lr=1e-4,                              # 大模型用小lr
        scale_t=2.0, scale_s=0.5,
        lorentz_start=0.5,
        eval_interval=500, log_interval=200,
        out_dir=f'./checkpoints_scale_{d_model}',
    )
    return train(train_cfg, model_cfg)


def wikitext_train(d_model=256, n_layers=6, total_steps=10000,
                   lorentz_alpha=0.25):
    """
    真实语言数据训练（wikitext-103）
    需要: pip install datasets transformers
    """
    model_cfg = LorentzConfig(
        vocab_size=50257,               # GPT-2词表
        d_model=d_model,
        n_heads=max(4, d_model//64),
        n_layers=n_layers,
        max_seq_len=256,
        lorentz_alpha=lorentz_alpha,
        hess_warmup_steps=500,          # 真实数据需要更长warmup
        hess_update_freq=100,
        hutchinson_k=10,               # 真实数据Hutchinson适当减少
        lambda_spacelike=0.0,
    )
    train_cfg = TrainConfig(
        dataset='wikitext',
        total_steps=total_steps, batch_size=16,
        base_lr=1e-4,
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
    # Colab/Jupyter argparse    → 
    p = argparse.ArgumentParser(description='Lorentz Transformer')
    p.add_argument('--mode', default='quick',
                   choices=['quick', 'full', 'test', 'scale', 'wikitext'])
    p.add_argument('--n_hops',        type=int,   default=2)
    p.add_argument('--total_steps',   type=int,   default=0)
    p.add_argument('--d_model',       type=int,   default=128)
    p.add_argument('--n_layers',      type=int,   default=4)
    p.add_argument('--lorentz_alpha', type=float, default=0.25)
    args, _ = p.parse_known_args()   #     

    if args.mode == 'test':
        print("Smoke test (10 steps)...")
        log = quick_train(n_hops=1, total_steps=10, d_model=64, n_layers=2)
        print(f"✓   val_loss={log['val_loss'][-1]:.4f}")
    elif args.mode == 'quick':
        steps = args.total_steps or 2000
        quick_train(args.n_hops, steps, args.d_model,
                    args.n_layers, args.lorentz_alpha)
    elif args.mode == 'scale':
        steps = args.total_steps or 5000
        scale_train(args.d_model or 256, args.n_layers or 6,
                    args.n_hops, steps, args.lorentz_alpha)
    elif args.mode == 'wikitext':
        steps = args.total_steps or 10000
        wikitext_train(args.d_model or 256, args.n_layers or 6,
                       steps, args.lorentz_alpha)
    else:
        steps = args.total_steps or 10000
        full_train(args.n_hops, steps, args.d_model,
                   args.n_layers, args.lorentz_alpha)
