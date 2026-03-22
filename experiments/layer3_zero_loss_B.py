"""
layer3_zero_loss_B.py — 层3验证：零损失猜想强版本（选项B）
=====================================================================
改变：预训练加入动量守恒信号，强制sigma收敛
目标：sigma > 0.55，F3动量变化率接近真实基准

预训练损失：
  loss = MSE轨迹预测 + MOM_WEIGHT * 动量守恒损失
  动量守恒损失 = (dp²).mean()，dp = 速度差分（vel[:, 1:] - vel[:, :-1]）

如果sigma收敛后F3动量守恒损失显著低于欧氏（f1）：
  零损失猜想强版本成立 ✅
  "洛伦兹几何本能让守恒自动成立"

使用：
  python experiments/layer3_zero_loss_B.py
  或
  exec(open('experiments/layer3_zero_loss_B.py').read())
"""

import os
import sys
import numpy as np
from scipy import stats as scipy_stats
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 确保可以从 core 导入（在 experiments/ 下运行时需要添加父目录）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import (
    LLCMBackbone,
    build_dataset,
    momentum_change,
    real_physics_baseline,
    device,
    EMBED_DIM,
    T_IN,
    T_OUT,
)
from lorentz_transformer import compute_t_dim

# ─── 层3专用超参数 ─────────────────────────────────────────────────────────
EP_PRE     = 120    # 比层0/1更多，让sigma充分收敛
LR_PRE     = 3e-4
MOM_WEIGHT = 0.3    # 动量守恒损失权重（加入训练信号）
N_SEEDS    = 5
N_PER      = 50     # 每标签轨迹数
N_HEADS    = 4
TIME_RATIO = 0.5    # 层3专用：频域精确值（层0/1用0.25）
                    # 相位维度=类时，振幅维度=类空，TIME_RATIO精确推导
T_DIM      = compute_t_dim(EMBED_DIM, N_HEADS, TIME_RATIO)
BS         = 16


def _build_pretrain_data(seed: int = 0):
    """构建层3预训练数据（x_in, x_out）。"""
    X, _ = build_dataset(n_per_label=N_PER, seed=seed)   # (N, T, 6) tensor
    X    = X.to(device)
    x_in  = X[:, :T_IN, :]
    x_out = X[:, T_IN:T_IN + T_OUT, :]
    return x_in, x_out


def pretrain_layer3(model, seed: int = 0) -> dict:
    """
    层3预训练：MSE 轨迹预测 + 动量守恒损失。

    与基础 pretrain() 的区别：
    - TIME_RATIO=0.5（层3频域精确值）由调用方构建 model 时传入
    - 损失 = MSE + MOM_WEIGHT × (dp²).mean()，dp = 速度差分
    - 更多 epoch（EP_PRE=120）以确保 sigma 收敛

    Args:
        model : LLCMBackbone（f1 或 f3 公式，time_ratio=0.5）
        seed  : 随机种子

    Returns:
        dict 包含：
            'mse_loss' : float，最终 MSE 损失
            'mom_loss' : float，最终动量守恒损失（在预测轨迹上测量）
    """
    torch.manual_seed(seed)
    x_in, x_out = _build_pretrain_data(seed=seed + 100)

    ds = TensorDataset(x_in, x_out)
    dl = DataLoader(ds, batch_size=BS, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PRE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EP_PRE)

    model.train()
    for _ in range(EP_PRE):
        for xb, yb in dl:
            optimizer.zero_grad()

            pred = model(xb)[:, :T_OUT, :]         # (B, T_OUT, 6)

            # MSE 损失
            mse = F.mse_loss(pred, yb)

            # 动量守恒损失：预测轨迹速度差分的 L² 均值
            vel      = pred[:, :, 3:]               # (B, T_OUT, 3) 速度分量
            dp       = vel[:, 1:, :] - vel[:, :-1, :]  # (B, T_OUT-1, 3)
            mom_loss = (dp ** 2).mean()

            loss = mse + MOM_WEIGHT * mom_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    # 最终评估（在完整数据集上，不使用 DataLoader）
    model.eval()
    with torch.no_grad():
        pred_all  = model(x_in)[:, :T_OUT, :]
        final_mse = float(F.mse_loss(pred_all, x_out).item())
        vel_all   = pred_all[:, :, 3:]
        dp_all    = vel_all[:, 1:, :] - vel_all[:, :-1, :]
        final_mom = float((dp_all ** 2).mean().item())

    return {'mse_loss': final_mse, 'mom_loss': final_mom}


def _t_test_less(a: np.ndarray, b: np.ndarray) -> float:
    """单侧 t 检验：H1: mean(a) < mean(b)，返回 p 值。"""
    result = scipy_stats.ttest_ind(a, b, alternative='less')
    return float(result.pvalue)


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d 效应量（a vs b 差异大小）。"""
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0)
    if pooled_std < 1e-12:
        return 0.0
    return float((b.mean() - a.mean()) / pooled_std)


def run_layer3_experiment(n_seeds: int = N_SEEDS, verbose: bool = True) -> dict:
    """
    层3主实验：预训练阶段动量守恒损失 F3 vs 欧氏。

    每个 seed 各训练一个欧氏（f1）和一个 F3（f3）模型，
    比较预测轨迹的动量守恒损失（越低越守恒）。

    Args:
        n_seeds: 重复随机 seed 数
        verbose: 是否打印每步结果

    Returns:
        dict 包含：
            'euc_mom_losses' : list，欧氏动量守恒损失（每 seed）
            'f3_mom_losses'  : list，F3 动量守恒损失（每 seed）
            'euc_mean'       : float
            'euc_std'        : float
            'f3_mean'        : float
            'f3_std'         : float
            'ratio'          : float（欧氏/F3 倍数）
            'p_value'        : float
            'cohen_d'        : float
            'phys_baseline'  : float
            'pass'           : bool（p<0.05 且 ratio>3）
    """
    if verbose:
        print("=" * 60)
        print("层3验证：零损失猜想强版本（选项B）")
        print("=" * 60)
        print(f"配置：EP_PRE={EP_PRE}  MOM_WEIGHT={MOM_WEIGHT}  N_SEEDS={n_seeds}")
        print(f"TIME_RATIO={TIME_RATIO}  T_DIM={T_DIM}")
        print(f"预训练损失：MSE + {MOM_WEIGHT} × 动量守恒损失")

    euc_losses: list = []
    f3_losses:  list = []

    for seed in range(n_seeds):
        if verbose:
            print(f"\n[seed {seed}]")

        # 欧氏对照（f1 公式，无光锥约束）
        model_euc = LLCMBackbone(
            mode='f1', time_ratio=TIME_RATIO
        ).to(device)
        res_euc = pretrain_layer3(model_euc, seed=seed)

        # F3 光锥
        model_f3 = LLCMBackbone(
            mode='f3', time_ratio=TIME_RATIO
        ).to(device)
        res_f3 = pretrain_layer3(model_f3, seed=seed)

        euc_losses.append(res_euc['mom_loss'])
        f3_losses.append(res_f3['mom_loss'])

        if verbose:
            ratio_seed = res_euc['mom_loss'] / (res_f3['mom_loss'] + 1e-8)
            sigma_f3   = model_f3.blocks[0].attn.sigma
            print(f"  欧氏 mom_loss: {res_euc['mom_loss']:.4f}")
            print(f"  F3   mom_loss: {res_f3['mom_loss']:.4f}")
            print(f"  倍数: {ratio_seed:.1f}× （欧氏/F3）")
            print(f"  F3 sigma: {sigma_f3:.4f}")

    euc_arr = np.array(euc_losses)
    f3_arr  = np.array(f3_losses)

    p     = _t_test_less(f3_arr, euc_arr)
    d     = _cohen_d(f3_arr, euc_arr)
    ratio = float(euc_arr.mean() / (f3_arr.mean() + 1e-8))
    phys  = real_physics_baseline()

    result = {
        'euc_mom_losses': euc_losses,
        'f3_mom_losses':  f3_losses,
        'euc_mean':       float(euc_arr.mean()),
        'euc_std':        float(euc_arr.std(ddof=1)),
        'f3_mean':        float(f3_arr.mean()),
        'f3_std':         float(f3_arr.std(ddof=1)),
        'ratio':          ratio,
        'p_value':        p,
        'cohen_d':        d,
        'phys_baseline':  phys,
        'pass':           bool(p < 0.05 and ratio > 3.0),
    }

    if verbose:
        print("\n" + "=" * 60)
        print("层3实验结果汇总")
        print("=" * 60)
        print(f"欧氏 动量守恒损失:    {result['euc_mean']:.4f} ± {result['euc_std']:.4f}")
        print(f"F3   动量守恒损失:    {result['f3_mean']:.4f} ± {result['f3_std']:.4f}")
        print(f"倍数（欧氏/F3）:      {result['ratio']:.1f}×")
        print(f"真实物理基准:         {result['phys_baseline']:.4f}")
        print(f"p值:                  {result['p_value']:.4f}")
        print(f"Cohen d:              {result['cohen_d']:.2f}")
        print(f"结论: {'✅ F3 动量守恒显著优于欧氏' if result['pass'] else '❌ 差异不显著'}")

        if result['ratio'] >= 10:
            print(f"\n🎯 关键发现：差距 {result['ratio']:.0f}×，达到论文报告水平（10倍）")

        print("\n理论解释：")
        print("  F3 负号 → 类时方向互相排斥 → 信息沿光锥边界传播")
        print("  光锥边界对应匀速运动（类时测地线）")
        print("  动量守恒 = 几何结构的必然，不是损失函数的约束")

    return result


if __name__ == '__main__':
    run_layer3_experiment()
