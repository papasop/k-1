"""
llcm/core.py
============
LLCM（Lorentz Light-Cone Model）核心模块

理论基础：
  Realizability.pdf（Li 2026）— Theorem 5：洛伦兹签名从代价函数唯一涌现
  K=1 Chronogeometrodynamics（Li 2026）— Theorem 4：det G < 0 ⟺ dc > 0

所有实验从这里 import，避免重复定义：
  from llcm.core import (
      MinkowskiLN, Attn, LLCMBackbone,
      stable_ode, running_ode, simulate, build_dataset,
      momentum_change, encode, pretrain
  )

模块对应关系：
  模块1（物理预训练）：LLCMBackbone + pretrain()
  模块2（语言编码器）：encode()，基于 sentence-transformers
  模块3（方向A）：    LLCMBackbone.forward_A_gen() + lang_gen 头
  模块4（lang_aligner）：LLCMBackbone.lang_aligner
  模块5（phys_decoder）：LLCMBackbone.phys_decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

from lorentz_transformer import (
    LorentzMultiHeadAttention,
    MinkowskiLayerNorm,
    compute_t_dim,
)

# ── 默认超参数（实验文件可覆盖） ───────────────────────────────
EMBED_DIM  = 128
N_HEADS    = 4
N_LAYERS   = 3
TIME_RATIO = 0.25
T_IN       = 20
T_OUT      = 20
STATE_DIM  = 6
LANG_DIM   = 384
N_LABELS   = 2
N_PER      = 50
EP_PRE     = 80
LR_PRE     = 3e-4
BS         = 16
T_DIM      = compute_t_dim(EMBED_DIM, N_HEADS, TIME_RATIO)

LABELS = {0: 'momentum_stable', 1: 'momentum_changing'}
DESCRIPTIONS = {
    0: ["平稳匀速运动，动量保持守恒",
        "smooth constant velocity motion with conserved momentum",
        "steady movement without any acceleration"],
    1: ["动量持续变化，存在外力作用",
        "changing momentum with continuous force application",
        "accelerating or decelerating movement"],
}
STABLE_INSTRUCTIONS = [
    "平稳匀速运动，动量保持守恒",
    "机器人以恒定速度移动，没有加速或减速",
    "smooth constant velocity motion with conserved momentum",
    "steady movement without acceleration changes",
]

# ── 设备 ──────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 类型别名（供实验文件直接 import） ─────────────────────────
# MinkowskiLN：使用旧版 mq.abs() 变体（训练稳定，实验已验证）
# 符号修复版（sign * sqrt(...)）在 MSE 轨迹预测上训练不稳定，是开放研究问题
MinkowskiLN = MinkowskiLayerNorm

# Attn：F3 公式（sigma 自适应，推荐默认）
Attn = LorentzMultiHeadAttention


# ── 语言编码器（模块2） ────────────────────────────────────────
_lang_enc = None


def get_lang_enc():
    global _lang_enc
    if _lang_enc is None:
        from sentence_transformers import SentenceTransformer
        _lang_enc = SentenceTransformer('all-MiniLM-L6-v2')
    return _lang_enc


def encode(texts, dev=None):
    """语言嵌入，懒加载 sentence-transformers"""
    if dev is None:
        dev = device
    return get_lang_enc().encode(
        texts, convert_to_tensor=True,
        show_progress_bar=False).to(dev)


# ── 物理常数 ───────────────────────────────────────────────────
_GRAVITY_M_S2        = 9.81    # 重力加速度 (m/s²)
_MASS_KG             = 70.0    # 人体质量 (kg)
_DAMPING_COEFF       = 0.001   # 速度阻尼系数（稳定 ODE）
_GAIT_FREQ_HZ        = 3.0     # 步态频率 (Hz)
_LEG_REST_LEN_M      = 0.95    # 腿静止长度 (m)
_LEG_STRIDE_AMP_M    = 0.12    # 腿长振幅 (m)
_STANCE_PHASE_FRAC   = 0.4     # 支撑相占步态周期比例
_LEG_SPRING_N_M      = 2000.0  # 腿弹簧刚度 (N/m)
_LEG_DAMPER_NS_M     = 30.0    # 腿阻尼系数 (N·s/m)
_HORIZ_DRIVE_N       = 200.0   # 水平驱动力系数 (N/(m/s))
_TARGET_SPEED_M_S    = 2.0     # 目标奔跑速度 (m/s)


# ── 物理数据 ───────────────────────────────────────────────────

def stable_ode(t, y):
    """
    稳定匀速运动 ODE（动量守恒）
    Assumption R 满足：沿速度方向的位移代价趋向零
    """
    x, yp, z, vx, vy, vz = y
    return [vx, 0, vz,
            -_DAMPING_COEFF * vx, 0, -_DAMPING_COEFF * vz]


def running_ode(t, y):
    """
    奔跑运动 ODE（动量不守恒）
    外力驱动，速度持续变化（弹簧质量腿模型，SLIP）
    """
    x, yp, z, vx, vy, vz = y
    phase = (_GAIT_FREQ_HZ * t) % 1.0
    L = _LEG_REST_LEN_M + _LEG_STRIDE_AMP_M * abs(
        np.sin(_GAIT_FREQ_HZ * np.pi * t)
    )
    pen = max(0, L - yp)
    Fv = (_LEG_SPRING_N_M * pen - _LEG_DAMPER_NS_M * vy) \
        if (phase < _STANCE_PHASE_FRAC and yp < L) else 0.0
    Fh = _HORIZ_DRIVE_N * (_TARGET_SPEED_M_S - vx)
    return [vx, vy, vz,
            Fh / _MASS_KG,
            (Fv - _MASS_KG * _GRAVITY_M_S2) / _MASS_KG,
            -_DAMPING_COEFF * vz]


def simulate(ode_fn, T=T_IN + T_OUT, n=N_PER, seed=None):
    """
    运行 ODE 仿真，生成 n 条轨迹。

    Args:
        ode_fn : ODE 右侧函数（stable_ode 或 running_ode）
        T      : 每条轨迹的时间步数（默认 T_IN + T_OUT = 40）
        n      : 轨迹数量
        seed   : 随机种子（用于初始条件采样）

    Returns:
        np.ndarray, shape (n_success, T, STATE_DIM), dtype float32
    """
    rng = np.random.default_rng(seed)
    t_end = T * 0.05
    t_eval = np.linspace(0, t_end, T)
    segs = []
    for _ in range(n):
        y0 = [
            rng.uniform(-1.0, 1.0),   # x
            rng.uniform(0.8, 1.2),    # y（高度）
            rng.uniform(-0.5, 0.5),   # z
            rng.uniform(0.5, 2.5),    # vx
            rng.uniform(-0.1, 0.1),   # vy
            rng.uniform(-0.2, 0.2),   # vz
        ]
        sol = solve_ivp(
            ode_fn, [0, t_end], y0,
            t_eval=t_eval, max_step=0.01, dense_output=False,
        )
        if sol.success and sol.y.shape[1] >= T:
            segs.append(sol.y.T[:T].astype(np.float32))
    if len(segs) == 0:
        segs = [np.zeros((T, STATE_DIM), dtype=np.float32)]
    return np.stack(segs)


def build_dataset(seed=42, n_per=N_PER):
    """
    构建训练/测试数据集（标签0=稳定，标签1=奔跑）。

    Args:
        seed  : 随机种子（stable_ode 使用 seed，running_ode 使用 seed+1）
        n_per : 每类轨迹数量

    Returns:
        X : np.ndarray, shape (2*n_success, T_IN+T_OUT, STATE_DIM), float32
        y : np.ndarray, shape (2*n_success,), int64
    """
    X_s = simulate(stable_ode,  T=T_IN + T_OUT, n=n_per, seed=seed)
    X_r = simulate(running_ode, T=T_IN + T_OUT, n=n_per, seed=seed + 1)
    X = np.concatenate([X_s, X_r], axis=0)
    y = np.array([0] * len(X_s) + [1] * len(X_r), dtype=np.int64)
    return X, y


def momentum_change(traj):
    """
    计算轨迹动量变化率（越小越守恒）。

    Args:
        traj : np.ndarray, shape (..., T, STATE_DIM)
               最后两维分别为时间步和状态维度，速度分量为 traj[..., 3:]

    Returns:
        float: 平均动量变化率（L2 范数均值）
    """
    vel = traj[..., 3:]
    dp = np.diff(vel, axis=-2)
    return float(np.linalg.norm(dp, axis=-1).mean())


# ── 内部配置 ───────────────────────────────────────────────────

class _LLCMCfg:
    """传递给 LorentzMultiHeadAttention 的最小配置对象"""
    d_model    = EMBED_DIM
    n_heads    = N_HEADS
    formula    = 'f3'
    time_ratio = TIME_RATIO
    dropout    = 0.0


# ── LLCM 变换块 ────────────────────────────────────────────────

class _LLCMBlock(nn.Module):
    """单个 LLCM 变换块：光锥注意力 + Minkowski LayerNorm + FFN"""

    def __init__(self):
        super().__init__()
        self.attn  = Attn(_LLCMCfg())
        self.norm1 = MinkowskiLN(EMBED_DIM, t_dim=T_DIM)
        self.ffn   = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * 4),
            nn.GELU(),
            nn.Linear(EMBED_DIM * 4, EMBED_DIM),
        )
        self.norm2 = MinkowskiLN(EMBED_DIM, t_dim=T_DIM)

    def forward(self, x):
        a, _ = self.attn(x)
        x    = self.norm1(x + a)
        x    = self.norm2(x + self.ffn(x))
        return x


# ── LLCMBackbone ──────────────────────────────────────────────

class LLCMBackbone(nn.Module):
    """
    LLCM 骨干网络（物理预训练 + 语言对齐）。

    子模块：
      embed        : STATE_DIM → EMBED_DIM 输入嵌入
      blocks       : N_LAYERS 个 _LLCMBlock（光锥注意力 + MinkowskiLN）
      phys_decoder : EMBED_DIM → STATE_DIM 轨迹解码（模块5）
      lang_aligner : EMBED_DIM → LANG_DIM  语言对齐投影（模块4）
      lang_gen     : EMBED_DIM → LANG_DIM  生成方向A语言头（模块3）
      cls_head     : EMBED_DIM → N_LABELS  分类头
    """

    def __init__(self):
        super().__init__()
        self.embed        = nn.Linear(STATE_DIM, EMBED_DIM)
        self.blocks       = nn.ModuleList(
            [_LLCMBlock() for _ in range(N_LAYERS)]
        )
        self.phys_decoder = nn.Linear(EMBED_DIM, STATE_DIM)
        self.lang_aligner = nn.Linear(EMBED_DIM, LANG_DIM)
        self.lang_gen     = nn.Linear(EMBED_DIM, LANG_DIM)
        self.cls_head     = nn.Linear(EMBED_DIM, N_LABELS)

    def encode_phys(self, x):
        """
        物理序列编码。

        Args:
            x : (B, T, STATE_DIM)

        Returns:
            h : (B, T, EMBED_DIM)
        """
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x):
        """
        分类前向（返回 logits）。

        Args:
            x : (B, T, STATE_DIM)

        Returns:
            logits : (B, N_LABELS)
        """
        h      = self.encode_phys(x)
        pooled = h.mean(dim=1)
        return self.cls_head(pooled)

    def forward_A_gen(self, x):
        """
        方向A生成：物理序列 → 语言嵌入（模块3）。

        Args:
            x : (B, T, STATE_DIM)

        Returns:
            lang_out : (B, LANG_DIM)，已 L2 归一化
        """
        h      = self.encode_phys(x)
        pooled = h.mean(dim=1)
        return F.normalize(self.lang_gen(pooled), dim=-1)


# ── 物理预训练（模块1） ────────────────────────────────────────

def pretrain(model, seed=0, epochs=EP_PRE, lr=LR_PRE, bs=BS):
    """
    物理预训练：用 ODE 轨迹以 MSE 损失训练轨迹预测任务。

    模型以前 T_IN 帧为输入，预测后 T_OUT 帧（T_IN == T_OUT == 20）。
    预训练数据通过 build_dataset 生成，与微调数据完全隔离。

    Args:
        model  : LLCMBackbone 实例
        seed   : 预训练数据随机种子（与微调数据种子不同）
        epochs : 训练轮数（默认 EP_PRE=80）
        lr     : 学习率（默认 LR_PRE=3e-4，AdamW）
        bs     : 批大小（默认 BS=16）

    Returns:
        model : 训练后的 LLCMBackbone（已移至 device）
    """
    model.to(device).train()
    X, _ = build_dataset(seed=seed)
    X_t  = torch.from_numpy(X).to(device)
    ds   = TensorDataset(X_t[:, :T_IN], X_t[:, T_IN:])
    dl   = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)
    opt  = torch.optim.AdamW(model.parameters(), lr=lr)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        for x_in, x_out in dl:
            opt.zero_grad()
            h    = model.encode_phys(x_in)      # (B, T_IN, EMBED_DIM)
            pred = model.phys_decoder(h)         # (B, T_IN, STATE_DIM)
            # T_IN == T_OUT so pred and x_out have identical shape (B, 20, 6)
            loss = F.mse_loss(pred, x_out)
            loss.backward()
            opt.step()
        sch.step()
    return model
