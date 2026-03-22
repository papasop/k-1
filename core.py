"""
core.py — LLCM 研究原型共用模块

所有实验脚本共享此模块的模型定义、数据生成、预训练、评估工具。

从 core.py import:
    from core import (
        LLCMBackbone, pretrain,          # 模型和预训练
        build_dataset, simulate,          # 数据生成
        momentum_change, encode,          # 评估工具
        stable_ode, running_ode,          # 物理 ODE
        real_physics_baseline,            # 真实物理基准
        device, EMBED_DIM, T_DIM,         # 超参数
        LABELS, DESCRIPTIONS,             # 标签定义
    )

使用示例:
    model = LLCMBackbone(mode='f3').to(device)
    pretrain(model, seed=0)
    X, L = build_dataset(seed=42)
    lorentz = model.embed_seq(X.to(device))
    geo = model.measure_lorentz(lorentz)
    print(f"类时比例: {geo['tl_ratio']:.1%}  mq均值: {geo['mq_mean']:+.3f}")
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from lorentz_transformer import (
    LorentzMultiHeadAttention,
    MinkowskiLayerNorm,
    compute_t_dim,
)

# ============================================================================
# 设备 / 超参数
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_DIM  = 128      # 嵌入维度
N_HEADS    = 4        # 注意力头数
N_LAYERS   = 3        # Transformer 层数
TIME_RATIO = 0.25     # 时间头比例（理论必须，不要改）
STATE_DIM  = 6        # 物理状态维度 [x, y, z, vx, vy, vz]

T_IN  = 20            # 输入帧数
T_OUT = 20            # 预测帧数
T_DIM = compute_t_dim(EMBED_DIM, N_HEADS, TIME_RATIO)   # 时间维度数

# ============================================================================
# 标签定义
# ============================================================================

LABELS = ["stable", "running", "walking", "jumping"]

DESCRIPTIONS = {
    "stable":  "平稳守恒运动，匀速直线，动量完全守恒",
    "running": "跑步运动，速度较快，有轻微波动",
    "walking": "行走运动，速度适中，周期性模式",
    "jumping": "跳跃运动，有明显的垂直速度变化",
}

# ============================================================================
# 物理 ODE（数据生成）
# ============================================================================


def stable_ode(n_steps: int = 60, dt: float = 0.1, seed: int = 0) -> np.ndarray:
    """
    匀速直线运动 ODE（动量完全守恒）。

    Returns:
        traj: (n_steps, 6) — [x, y, z, vx, vy, vz]
    """
    rng = np.random.RandomState(seed)
    v0  = rng.randn(3) * 0.5
    x0  = rng.randn(3)
    traj = np.zeros((n_steps, 6))
    traj[0] = np.concatenate([x0, v0])
    for i in range(1, n_steps):
        traj[i, :3] = traj[i - 1, :3] + v0 * dt
        traj[i, 3:] = v0
    return traj.astype(np.float32)


def running_ode(n_steps: int = 60, dt: float = 0.1, seed: int = 0) -> np.ndarray:
    """
    跑步运动 ODE（速度有随机扰动）。

    Returns:
        traj: (n_steps, 6) — [x, y, z, vx, vy, vz]
    """
    rng = np.random.RandomState(seed)
    v   = rng.randn(3) * 1.0
    x   = rng.randn(3)
    traj = np.zeros((n_steps, 6))
    traj[0] = np.concatenate([x, v])
    for i in range(1, n_steps):
        v = v + rng.randn(3) * 0.05
        x = x + v * dt
        traj[i, :3] = x
        traj[i, 3:] = v
    return traj.astype(np.float32)


def walking_ode(n_steps: int = 60, dt: float = 0.1, seed: int = 0) -> np.ndarray:
    """
    行走运动 ODE（周期性速度模式）。

    Returns:
        traj: (n_steps, 6) — [x, y, z, vx, vy, vz]
    """
    rng  = np.random.RandomState(seed)
    v0   = rng.randn(3) * 0.5
    x    = rng.randn(3)
    traj = np.zeros((n_steps, 6))
    traj[0] = np.concatenate([x, v0])
    for i in range(1, n_steps):
        phase = i * dt * 2 * np.pi / 0.8
        v     = v0 + np.array([0.0, 0.0, 0.1 * np.sin(phase)])
        x     = x + v * dt
        traj[i, :3] = x
        traj[i, 3:] = v
    return traj.astype(np.float32)


def jumping_ode(n_steps: int = 60, dt: float = 0.1, seed: int = 0) -> np.ndarray:
    """
    跳跃运动 ODE（有明显垂直速度变化，含弹性碰地）。

    Returns:
        traj: (n_steps, 6) — [x, y, z, vx, vy, vz]
    """
    rng  = np.random.RandomState(seed)
    v    = rng.randn(3) * 0.3
    v[2] = 2.0   # 初始垂直速度
    x    = rng.randn(3)
    x[2] = abs(x[2])   # 起点在地面以上
    g    = np.array([0.0, 0.0, -9.8])
    traj = np.zeros((n_steps, 6))
    traj[0] = np.concatenate([x, v])
    for i in range(1, n_steps):
        v = v + g * dt
        x = x + v * dt
        if x[2] < 0:
            x[2] = 0.0
            v[2] = abs(v[2]) * 0.7   # 弹性碰撞
        traj[i, :3] = x
        traj[i, 3:] = v
    return traj.astype(np.float32)


def simulate(
    label: str,
    n_steps: int = 60,
    dt: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """
    按标签生成物理轨迹。

    Args:
        label  : 'stable' | 'running' | 'walking' | 'jumping'
        n_steps: 帧数
        dt     : 时间步长
        seed   : 随机种子

    Returns:
        traj: (n_steps, STATE_DIM)
    """
    _ode_map = {
        "stable":  stable_ode,
        "running": running_ode,
        "walking": walking_ode,
        "jumping": jumping_ode,
    }
    if label not in _ode_map:
        raise ValueError(f"Unknown label: '{label}'. Must be one of {LABELS}")
    return _ode_map[label](n_steps, dt, seed)


# ============================================================================
# 数据集构建
# ============================================================================


def build_dataset(
    n_per_label: int = 20,
    n_steps: int = T_IN + T_OUT,
    seed: int = 0,
):
    """
    构建物理运动数据集。

    Args:
        n_per_label: 每个标签的轨迹数量
        n_steps    : 每条轨迹的总帧数
        seed       : 随机种子

    Returns:
        X: (N, T, STATE_DIM) 轨迹张量（float32）
        L: (N,) 标签张量（int64）
    """
    rng    = np.random.RandomState(seed)
    trajs  = []
    labels = []

    for label_idx, label in enumerate(LABELS):
        for _ in range(n_per_label):
            s    = int(rng.randint(0, 10000))
            traj = simulate(label, n_steps=n_steps, seed=s)
            trajs.append(traj)
            labels.append(label_idx)

    X = torch.tensor(np.stack(trajs), dtype=torch.float32)   # (N, T, 6)
    L = torch.tensor(labels, dtype=torch.long)                # (N,)
    return X, L


# ============================================================================
# 评估工具
# ============================================================================


def momentum_change(traj: np.ndarray) -> float:
    """
    计算轨迹的动量变化量（越低越守恒）。

    Args:
        traj: (T, STATE_DIM) 轨迹，后 3 维为速度

    Returns:
        mean momentum change rate
    """
    vel = traj[:, 3:]
    dp  = np.diff(vel, axis=0)
    return float(np.linalg.norm(dp, axis=-1).mean())


def real_physics_baseline(n_trajs: int = 50, seed: int = 0) -> float:
    """
    真实物理基准：匀速运动的动量变化率（理想值接近 0）。

    Args:
        n_trajs: 基准轨迹数
        seed   : 随机种子

    Returns:
        mean momentum change rate
    """
    rng     = np.random.RandomState(seed)
    changes = []
    for _ in range(n_trajs):
        s    = int(rng.randint(0, 10000))
        traj = stable_ode(n_steps=T_IN + T_OUT, seed=s)
        changes.append(momentum_change(traj))
    return float(np.mean(changes))


# ============================================================================
# LLCMBackbone（内部构建块）
# ============================================================================


@dataclass
class _BackboneConfig:
    d_model:    int   = EMBED_DIM
    n_heads:    int   = N_HEADS
    formula:    str   = "f3"
    time_ratio: float = TIME_RATIO
    dropout:    float = 0.1


class _LorentzBlock(nn.Module):
    """单个 Lorentz Transformer 块（注意力 + LayerNorm + FFN）。"""

    def __init__(self, config: _BackboneConfig):
        super().__init__()
        t_dim      = compute_t_dim(config.d_model, config.n_heads, config.time_ratio)
        self.attn  = LorentzMultiHeadAttention(config)
        self.norm1 = MinkowskiLayerNorm(config.d_model, t_dim=t_dim)
        self.norm2 = MinkowskiLayerNorm(config.d_model, t_dim=t_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x)
        x    = self.norm1(x + a)
        x    = self.norm2(x + self.ffn(x))
        return x


# ============================================================================
# LLCMBackbone
# ============================================================================


class LLCMBackbone(nn.Module):
    """
    LLCM 物理感知流形主干网络。

    Architecture:
        input_proj  : STATE_DIM → d_model
        blocks      : N_LAYERS × LorentzBlock (attn + norm + ffn)
        output_proj : d_model → STATE_DIM （轨迹预测头）

    Args:
        mode       : 注意力公式 'f1' | 'f2' | 'f3'（默认 'f3'）
        d_model    : 嵌入维度（默认 EMBED_DIM=128）
        n_heads    : 注意力头数（默认 N_HEADS=4）
        n_layers   : Transformer 层数（默认 N_LAYERS=3）
        time_ratio : 类时头比例（默认 TIME_RATIO=0.25；层3用 0.5）

    Example:
        >>> model = LLCMBackbone(mode='f3').to(device)
        >>> pretrain(model, seed=0)
        >>> X, L = build_dataset(seed=42)
        >>> lorentz = model.embed_seq(X.to(device))
        >>> geo = model.measure_lorentz(lorentz)
        >>> print(f"类时比例: {geo['tl_ratio']:.1%}  mq均值: {geo['mq_mean']:+.3f}")
    """

    def __init__(
        self,
        mode:       str   = "f3",
        d_model:    int   = EMBED_DIM,
        n_heads:    int   = N_HEADS,
        n_layers:   int   = N_LAYERS,
        time_ratio: float = TIME_RATIO,
    ):
        super().__init__()
        config = _BackboneConfig(
            d_model=d_model,
            n_heads=n_heads,
            formula=mode,
            time_ratio=time_ratio,
        )
        self.t_dim       = compute_t_dim(d_model, n_heads, time_ratio)
        self.input_proj  = nn.Linear(STATE_DIM, d_model)
        self.blocks      = nn.ModuleList([_LorentzBlock(config) for _ in range(n_layers)])
        self.output_proj = nn.Linear(d_model, STATE_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        轨迹预测前向传播。

        Args:
            x: (B, T, STATE_DIM)

        Returns:
            pred: (B, T, STATE_DIM)
        """
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)

    def embed_seq(self, x: torch.Tensor) -> torch.Tensor:
        """
        将状态序列嵌入到洛伦兹潜在空间。

        Args:
            x: (B, T, STATE_DIM)

        Returns:
            lorentz: (B, T, d_model) 洛伦兹嵌入
        """
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return h

    def measure_lorentz(self, lorentz: torch.Tensor) -> dict:
        """
        测量洛伦兹嵌入的几何统计量。

        Args:
            lorentz: (B, T, d_model)

        Returns:
            dict with keys:
                'tl_ratio' : 类时向量比例（mq > 0 的比例）
                'mq_mean'  : Minkowski 内积均值
                'tl_count' : 类时向量数量
                'total'    : 总向量数量
        """
        flat   = lorentz.reshape(-1, lorentz.shape[-1]).detach()
        t_comp = flat[:, :self.t_dim]
        s_comp = flat[:, self.t_dim:]
        # LLCM 约定（与 README 5分钟检测一致）：
        #   mq = ||s||² - ||t||²
        #   mq > 0 → 类时向量（spacelike维平方和大于timelike维平方和）
        mq      = s_comp.pow(2).sum(-1) - t_comp.pow(2).sum(-1)
        tl_mask = mq > 0
        return {
            "tl_ratio": float(tl_mask.float().mean().item()),
            "mq_mean":  float(mq.mean().item()),
            "tl_count": int(tl_mask.sum().item()),
            "total":    int(tl_mask.numel()),
        }


# ============================================================================
# encode — 将原始轨迹编码为归一化嵌入
# ============================================================================


def encode(model: LLCMBackbone, X: torch.Tensor) -> torch.Tensor:
    """
    将轨迹数据集编码为洛伦兹嵌入（序列均值池化后 L2 归一化）。

    Args:
        model: LLCMBackbone，已预训练
        X    : (N, T, STATE_DIM) 轨迹数据

    Returns:
        emb: (N, d_model) 归一化嵌入（每条轨迹一个向量）
    """
    model.eval()
    with torch.no_grad():
        lorentz = model.embed_seq(X.to(device))   # (N, T, d_model)
        emb     = lorentz.mean(dim=1)              # (N, d_model)
        emb     = F.normalize(emb, dim=-1)
    return emb


# ============================================================================
# pretrain — 轨迹预测预训练（建立感知流形）
# ============================================================================


def pretrain(
    model: LLCMBackbone,
    seed:        int   = 0,
    n_epochs:    int   = 60,
    lr:          float = 3e-4,
    n_per_label: int   = 30,
    verbose:     bool  = False,
    mom_weight:  float = 0.0,
) -> float:
    """
    在合成物理运动数据上预训练 LLCMBackbone（轨迹预测任务）。

    Args:
        model      : LLCMBackbone
        seed       : 随机种子（决定训练数据，与微调数据完全分开）
        n_epochs   : 训练 epoch 数
        lr         : AdamW 学习率
        n_per_label: 每标签轨迹数
        verbose    : 是否打印训练进度
        mom_weight : 动量守恒损失权重（0=仅MSE，>0时加入速度差分L²惩罚）

    Returns:
        final_loss: 最终训练损失
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, _ = build_dataset(n_per_label=n_per_label, seed=seed + 100)
    X_train    = X_train.to(device)             # (N, T_IN+T_OUT, 6)
    x_in       = X_train[:, :T_IN, :]           # 前 T_IN 帧输入
    x_out      = X_train[:, T_IN:T_IN + T_OUT, :]  # 后 T_OUT 帧标签

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    model.train()
    final_loss = float("inf")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred       = model(x_in)[:, :T_OUT, :]   # (N, T_OUT, 6)
        mse        = F.mse_loss(pred, x_out)
        if mom_weight > 0.0:
            vel      = pred[:, :, 3:]                    # (N, T_OUT, 3) 速度分量
            dp       = vel[:, 1:, :] - vel[:, :-1, :]   # (N, T_OUT-1, 3) 速度差分
            mom_loss = (dp ** 2).mean()
            loss     = mse + mom_weight * mom_loss
        else:
            loss = mse
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        final_loss = float(loss.item())
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch + 1:3d}/{n_epochs}  loss={final_loss:.4f}")

    model.eval()
    return final_loss


# ============================================================================
# 公开接口
# ============================================================================

__all__ = [
    # 模型和预训练
    "LLCMBackbone",
    "pretrain",
    # 数据生成
    "build_dataset",
    "simulate",
    # 评估工具
    "momentum_change",
    "encode",
    # 物理 ODE
    "stable_ode",
    "running_ode",
    "walking_ode",
    "jumping_ode",
    # 真实物理基准
    "real_physics_baseline",
    # 超参数
    "device",
    "EMBED_DIM",
    "T_DIM",
    "T_IN",
    "T_OUT",
    # 标签定义
    "LABELS",
    "DESCRIPTIONS",
]
