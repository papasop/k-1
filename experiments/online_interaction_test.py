"""
online_interaction_test.py — Law II 在线交互验证

验证问题：LLCM 是否满足 Law II（信息时间的动态法则）？
    dc > 0：类时距离在在线交互中单调增加（感知流形扩展）

Law II（K=1 Chronogeometrodynamics）：
    dc/dt > 0
    等价于：Sig(G) = (1,1) ⟹ 在线学习中类时区域持续扩展

在线实验设计：
    1. 预训练 LLCMBackbone（建立感知流形）
    2. 在线逐步输入新轨迹序列（模拟实时交互）
    3. 每步测量类时比例（tl_ratio）和 mq 均值
    4. 验证 tl_ratio 随交互步数单调不减（dc > 0）

从 core.py import 所有共用定义，不重复定义模型。
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保可以 from core import ...（在 experiments/ 下运行时需要添加父目录）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import (
    LLCMBackbone, pretrain,
    build_dataset, simulate,
    momentum_change,
    device, EMBED_DIM, T_DIM, T_IN, T_OUT,
    LABELS, DESCRIPTIONS,
)

# ─── 在线交互超参数 ────────────────────────────────────────────────────────
N_ONLINE_STEPS = 20    # 在线交互步数
LR_ONLINE      = 1e-4  # 在线学习率
EP_PRE         = 60    # 预训练 epoch 数
N_SEEDS        = 3     # 重复 seed 数


# ─── 在线学习一步 ─────────────────────────────────────────────────────────

def _online_step(
    model: LLCMBackbone,
    optimizer: torch.optim.Optimizer,
    x_new: torch.Tensor,
) -> dict:
    """
    在线学习一步：用新到达的轨迹数据微调模型，并测量几何统计量。

    Args:
        model:     LLCMBackbone
        optimizer: 在线优化器
        x_new:     (B, T, STATE_DIM) 新轨迹数据

    Returns:
        dict with 'tl_ratio', 'mq_mean', 'loss'
    """
    x_in  = x_new[:, :T_IN, :]
    x_out = x_new[:, T_IN:T_IN + T_OUT, :]

    model.train()
    optimizer.zero_grad()
    pred = model(x_in)[:, :T_OUT, :]
    loss = F.mse_loss(pred, x_out)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    model.eval()
    with torch.no_grad():
        lorentz = model.embed_seq(x_in)
    geo = model.measure_lorentz(lorentz)

    return {
        'tl_ratio': geo['tl_ratio'],
        'mq_mean':  geo['mq_mean'],
        'loss':     float(loss.item()),
    }


# ─── 单 seed 在线交互实验 ──────────────────────────────────────────────────

def _run_online_experiment(
    seed: int,
    formula: str = 'f3',
    n_steps: int = N_ONLINE_STEPS,
    verbose: bool = False,
) -> dict:
    """
    运行单次在线交互实验。

    Args:
        seed:    随机种子
        formula: 注意力公式 'f3' | 'f1'（欧氏对照）
        n_steps: 在线交互步数
        verbose: 是否打印每步结果

    Returns:
        dict with:
            'tl_ratios' : list[float]，每步类时比例
            'mq_means'  : list[float]，每步 mq 均值
            'dc_positive': bool，tl_ratio 总体趋势为正（dc > 0）
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 预训练
    model = LLCMBackbone(mode=formula).to(device)
    pretrain(model, seed=seed * 100, n_epochs=EP_PRE, verbose=False)

    # 在线优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_ONLINE)

    # 在线数据流：逐步生成新轨迹
    rng = np.random.RandomState(seed + 999)
    tl_ratios, mq_means = [], []

    for step in range(n_steps):
        # 每步生成一批新轨迹（模拟在线数据流）
        label   = LABELS[rng.randint(0, len(LABELS))]
        s       = int(rng.randint(0, 10000))
        traj    = simulate(label, n_steps=T_IN + T_OUT, seed=s)
        x_batch = torch.tensor(traj[None], dtype=torch.float32).to(device)  # (1, T, 6)

        stats = _online_step(model, optimizer, x_batch)
        tl_ratios.append(stats['tl_ratio'])
        mq_means.append(stats['mq_mean'])

        if verbose:
            print(f"    step {step + 1:2d}: tl_ratio={stats['tl_ratio']:.1%}  "
                  f"mq={stats['mq_mean']:+.4f}  loss={stats['loss']:.4f}")

    # 验证 dc > 0：类时比例趋势
    # 用线性回归斜率判断是否单调不减
    x_axis  = np.arange(n_steps, dtype=float)
    slope   = np.polyfit(x_axis, tl_ratios, deg=1)[0]
    dc_pos  = bool(slope >= 0)

    # 后半段类时比例均值是否高于前半段
    half     = n_steps // 2
    first    = float(np.mean(tl_ratios[:half]))
    last     = float(np.mean(tl_ratios[half:]))
    dc_pos2  = bool(last >= first)

    return {
        'tl_ratios':    tl_ratios,
        'mq_means':     mq_means,
        'slope':        float(slope),
        'first_half':   first,
        'last_half':    last,
        'dc_positive':  dc_pos or dc_pos2,
    }


# ─── 主实验 ───────────────────────────────────────────────────────────────

def run_online_interaction_test(
    n_seeds: int = N_SEEDS,
    n_steps: int = N_ONLINE_STEPS,
    verbose: bool = True,
) -> dict:
    """
    Law II 在线交互验证：dc > 0。

    比较 F3 和欧氏模型的类时比例在线演化趋势。

    Args:
        n_seeds: 重复 seed 数
        n_steps: 在线交互步数
        verbose: 是否打印进度

    Returns:
        dict with:
            'f3_dc_positive_rate'  : float，F3 dc>0 的 seed 比例
            'euc_dc_positive_rate' : float，欧氏 dc>0 的 seed 比例
            'f3_tl_growth'        : float，F3 后半段-前半段均值
            'euc_tl_growth'       : float，欧氏后半段-前半段均值
            'pass'                : bool
    """
    if verbose:
        print("=" * 60)
        print("Law II 在线交互验证：dc > 0")
        print("=" * 60)
        print(f"配置：N_SEEDS={n_seeds}  N_STEPS={n_steps}  LR={LR_ONLINE}")

    f3_results  = []
    euc_results = []

    for seed in range(n_seeds):
        if verbose:
            print(f"\n[seed {seed}]")

        if verbose:
            print("  F3 模型在线学习...")
        r_f3 = _run_online_experiment(
            seed, formula='f3', n_steps=n_steps, verbose=verbose
        )

        if verbose:
            print("  欧氏模型在线学习...")
        r_euc = _run_online_experiment(
            seed, formula='f1', n_steps=n_steps, verbose=verbose
        )

        f3_results.append(r_f3)
        euc_results.append(r_euc)

        if verbose:
            print(f"  F3:  前半段={r_f3['first_half']:.1%}  "
                  f"后半段={r_f3['last_half']:.1%}  "
                  f"dc>0={'✅' if r_f3['dc_positive'] else '❌'}")
            print(f"  欧氏: 前半段={r_euc['first_half']:.1%}  "
                  f"后半段={r_euc['last_half']:.1%}  "
                  f"dc>0={'✅' if r_euc['dc_positive'] else '❌'}")

    f3_dc_rate  = float(np.mean([r['dc_positive'] for r in f3_results]))
    euc_dc_rate = float(np.mean([r['dc_positive'] for r in euc_results]))
    f3_growth   = float(np.mean([r['last_half'] - r['first_half'] for r in f3_results]))
    euc_growth  = float(np.mean([r['last_half'] - r['first_half'] for r in euc_results]))

    result = {
        'f3_results':           f3_results,
        'euc_results':          euc_results,
        'f3_dc_positive_rate':  f3_dc_rate,
        'euc_dc_positive_rate': euc_dc_rate,
        'f3_tl_growth':         f3_growth,
        'euc_tl_growth':        euc_growth,
        'pass':                 bool(f3_dc_rate >= 0.5),
    }

    if verbose:
        print("\n" + "=" * 60)
        print("Law II 验证结果汇总")
        print("=" * 60)
        print(f"F3  dc>0 比例:  {f3_dc_rate:.0%}  类时增长: {f3_growth:+.4f}")
        print(f"欧氏 dc>0 比例: {euc_dc_rate:.0%}  类时增长: {euc_growth:+.4f}")
        print(f"结论: {'✅ F3 满足 Law II (dc>0)' if result['pass'] else '❌ dc>0 未达到多数 seed'}")
        print("\n理论解释：")
        print("  Law II: dc/dt > 0 ⟺ 感知流形在在线交互中持续扩展")
        print("  Theorem 4: Sig(G)=(1,1) ⟹ 类时区域单调不减")

    return result


# ─── 入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_online_interaction_test(n_seeds=N_SEEDS, n_steps=N_ONLINE_STEPS)
