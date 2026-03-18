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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


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
    lambda_spacelike: float = 1e-4
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

        G = G + (v * Hv).detach() / n_samples

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
            mask_2d = self.timelike_mask.view(H, d_h).float()   # (H, d_h)

            # 对每个head：Q_t = Q_h * mask_h（类时分量）
            # Q: (B, H, L, d_h)
            # mask_2d[h]: (d_h,) → 广播到 (B, 1, L, d_h)
            mask_bcast = mask_2d.unsqueeze(0).unsqueeze(2)      # (1,H,1,d_h)
            Q_t = Q * mask_bcast.to(Q.device)                   # 类时Q分量

            # 时间分量内积
            time_inner = torch.matmul(Q_t,
                         K.transpose(-2, -1)) / scale           # (B,H,L,L)

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
    管理各层W_Q的类时方向检测和P_t更新

    核心流程：
      每 hess_update_freq 步（warmup后）：
        1. 对每层W_Q计算 dt²_info 的对角Hessian（Hutchinson）
        2. G_ii < 0 的位置标记为类时
        3. EMA平滑：mask_ema = decay×mask_old + (1-decay)×mask_new
        4. 注入到各层的LorentzMultiHeadAttention

    设计原则：
      - 与模型和优化器解耦：外部调用 probe.update()
      - EMA平滑防止mask剧烈跳变
      - warmup期间不计算（Adam二阶矩未稳定）
      - 失败时保留上一次的mask（安全fallback）

    用法：
      probe = TimeLikeProbe(model, config)
      # 在train.py的训练循环里：
      probe.step(batch_x, step_count)
    """
    def __init__(self, model: 'LorentzTransformer',
                 config: LorentzConfig):
        self.model  = model
        self.config = config

        # 每层的类时掩码EMA（float，[0,1]之间，代表"类时概率"）
        # 初始化为0.5（中性）
        self._mask_ema = [
            torch.zeros(config.d_model, device='cpu')
            for _ in range(config.n_layers)
        ]
        # threshold=0.3：需要EMA持续 > 0.3才标记为类时
        # 配合可信度检查（|λ_min| > 1e-5），防止噪声信号
        self._threshold = 0.3

        # 诊断统计
        self.timelike_fracs  = [0.0] * config.n_layers
        self.lambda_mins     = [0.0] * config.n_layers
        self.n_updates       = 0

    def _make_loss_fn(self, layer_idx: int,
                      W_Q_param: nn.Parameter,
                      x: torch.Tensor):
        """
        构造第layer_idx层的dt²_info损失函数

        只前向到目标层，提取注意力权重，计算dt²_info。
        W_Q_param是可微的替换参数，其他权重detach。
        """
        model  = self.model
        config = self.config

        def loss_fn():
            B, L = x.shape
            h = (model.embed(x) +
                 model.pos_emb(
                     torch.arange(L, device=x.device).unsqueeze(0)))

            for i, block in enumerate(model.blocks):
                if i < layer_idx:
                    h, _ = block(h)
                elif i == layer_idx:
                    xn = block.norm1(h)
                    # 用替换参数W_Q_param计算Q
                    Q  = F.linear(xn.float(), W_Q_param)
                    K  = block.attn.k_proj(xn).float()
                    d_h = config.head_dim
                    H_n = config.n_heads
                    Q = Q.view(B,L,H_n,d_h).transpose(1,2)
                    K = K.view(B,L,H_n,d_h).transpose(1,2)
                    sc = (torch.matmul(Q, K.transpose(-2,-1)) /
                          math.sqrt(d_h))
                    aw = F.softmax(sc, dim=-1)
                    return -compute_dt2_info(aw)
                else:
                    break
            return torch.tensor(0.0, device=x.device,
                                requires_grad=True)
        return loss_fn

    def step(self, x: torch.Tensor, global_step: int):
        """
        主更新接口，在train.py的每个训练步调用

        x:           当前batch输入 (B, L)
        global_step: 全局训练步数
        """
        cfg = self.config

        if global_step < cfg.hess_warmup_steps:
            return
        if global_step % cfg.hess_update_freq != 0:
            return

        # 用小batch计算（节省显存）
        xb = x[:16].detach()
        device = next(self.model.parameters()).device
        xb = xb.to(device)

        for li, block in enumerate(self.model.blocks):
            W_Q = block.attn.q_proj.weight   # (d_model, d_model)

            # 创建可微副本
            W_Q_param = nn.Parameter(W_Q.detach().clone())

            loss_fn = self._make_loss_fn(li, W_Q_param, xb)

            try:
                G = hutchinson_diag_hessian(
                    loss_fn, W_Q_param,
                    n_samples=cfg.hutchinson_k)

                # 记录最小值（用于诊断）
                lmin = G.min().item()
                self.lambda_mins[li] = lmin

                # 可信度检查：Hessian值太小 → 梯度接近零 → 符号是噪声
                # 发生在训练初期或梯度消失时，跳过更新保留旧mask
                if abs(lmin) < 1e-5:
                    continue

                # G_ii < 0 → 类时（二值信号）
                timelike_signal = (G < 0).float()          # (d_model, d_model)
                timelike_vec    = timelike_signal.mean(dim=1).cpu()  # (d_model,)

                # EMA平滑（decay=0.7：保留70%旧值）
                decay = cfg.ema_decay
                self._mask_ema[li] = (decay * self._mask_ema[li] +
                                      (1 - decay) * timelike_vec)

                # 二值化：EMA > 0.3 → 类时
                # 需要EMA持续在0.3以上，防止噪声触发
                binary_mask = (self._mask_ema[li] > self._threshold)

                block.attn.set_timelike_mask(binary_mask.to(device))
                self.timelike_fracs[li] = binary_mask.float().mean().item()

            except Exception as e:
                # 失败：保留上一次的mask，不中断训练
                pass

        self.n_updates += 1

    def get_param_mask_pairs(self, device):
        """
        返回 (W_Q_param, mask) 列表，供GeodesicAdam使用
        """
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
                f'  layer {li:2d}: frac={frac:.3f} λ_min={lmin:.6f} {bar}')
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
if __name__ == '__main__':
    print('='*55)
    print('LorentzTransformer 快速验证')
    print('='*55)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 小规格测试（快速）
    cfg = LorentzConfig(
        vocab_size  = 1000,
        d_model     = 128,
        n_heads     = 4,
        n_layers    = 4,
        max_seq_len = 64,
        lorentz_alpha      = 0.25,
        hess_update_freq   = 10,
        hess_warmup_steps  = 5,
        hutchinson_k       = 5,
    )

    model = LorentzTransformer(cfg).to(device)
    print(f'参数量: {model.get_num_params()/1e6:.2f}M')
    print(f'config: d={cfg.d_model} h={cfg.n_heads} L={cfg.n_layers} '
          f'α={cfg.lorentz_alpha}')

    # 前向测试
    x      = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
    mask   = torch.ones(2, 32, dtype=torch.bool, device=device)
    logits = model(x, mask)
    print(f'\n前向通过: input={list(x.shape)} → logits={list(logits.shape)}')

    # 损失计算
    loss_ce  = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        x[:, 1:].reshape(-1))
    loss_reg = model.regularization_loss()
    loss_tot = loss_ce + loss_reg
    print(f'CE loss={loss_ce.item():.4f}  '
          f'Reg loss={loss_reg.item():.6f}  '
          f'Total={loss_tot.item():.4f}')

    # 反向测试
    loss_tot.backward()
    print('反向通过 ✓')

    # P_t更新测试
    print('\nP_t更新测试...')
    # warmup=5, update_freq=10，需要step>=15才能触发第2次更新
    # 多跑35步确保EMA有3次更新（step=5,15,25,35）
    for step in range(40):
        model.probe.step(x, step)
    print(model.probe.report())

    # 光锥统计（需要再次前向）
    _ = model(x, mask)
    lc = model.lightcone_stats()
    print('\n光锥统计:')
    for layer, s in lc.items():
        print(f'  {layer}: timelike={s["timelike_frac"]:.3f}  '
              f'mean_interval={s["mean_interval"]:.3f}')

    # Checkpoint保存/加载测试
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp = f.name
    model.save_lorentz_state(tmp)
    model2 = LorentzTransformer.load_lorentz_state(tmp, device=str(device))
    # eval模式关闭dropout，确保输出确定性
    model.eval(); model2.eval()
    with torch.no_grad():
        out1 = model(x, mask)
        out2 = model2(x, mask)
    assert torch.allclose(out1, out2, atol=1e-5), 'Checkpoint不一致！'
    print('\nCheckpoint保存/加载 ✓')
    model.train()  # 恢复训练模式
    os.unlink(tmp)

    print('\n所有测试通过 ✓')
    print('下一步: optimizer.py (GeodesicAdam)')
