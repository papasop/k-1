"""
online_interaction_test.py
==========================
模块4b：Law II 在线收敛速度验证

验证问题：F3 光锥几何是否让系统在线收敛更快（dc > 0 非平凡稳定边界）？
当前状态：验证进行中 🔄

理论背景（K=1 Chronogeometrodynamics Theorem 4）：
    (a) 系统存在非平凡稳定边界 dc > 0
          ⟺
    (b) Sig(G) = (1,1)，等价于 det G < 0

    F3 光锥几何 → 洛伦兹签名 → det G < 0 → dc > 0
    dc > 0 意味着系统有一个非平凡的稳定边界：
    在这个边界内，系统快速收敛；在边界外，系统发散。

在线收敛测试：
    给模型一系列新轨迹，测量预测误差随步骤减少的速度。
    F3 应该比欧氏更快收敛到稳定误差（更小的 dc）。

复现方法：
    exec(open('online_interaction_test.py').read())
    # 或直接运行
    # python online_interaction_test.py
"""

import numpy as np
import torch
import torch.nn.functional as F

from core import (
    LLCMBackbone,
    pretrain,
    build_dataset,
    device,
    T_IN,
    T_OUT,
)

# ── 配置 ────────────────────────────────────────────────────────
N_SEEDS        = 5
EP_PRE         = 60     # 预训练 epochs（同层0/1）
N_ONLINE_STEPS = 20     # 在线更新步数
LR_ONLINE      = 1e-3   # 在线学习率（比预训练大）


# ── 在线适应测试 ───────────────────────────────────────────────

def _online_adapt(model: LLCMBackbone, X_new: torch.Tensor, n_steps: int):
    """
    在线适应：给定新轨迹，测量每步更新后的预测误差。

    Args:
        model  : LLCMBackbone（已预训练）
        X_new  : (N, T_IN+T_OUT, STATE_DIM) 新轨迹
        n_steps: 在线更新步数

    Returns:
        errors: (n_steps,) 每步预测 MSE
    """
    model.train()
    X_in  = X_new[:, :T_IN, :].to(device)
    X_out = X_new[:, T_IN:T_IN + T_OUT, :].to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR_ONLINE)
    errors    = []

    for step in range(n_steps):
        optimizer.zero_grad()
        pred = model(X_in)[:, :T_OUT, :]
        loss = F.mse_loss(pred, X_out)
        loss.backward()
        optimizer.step()
        errors.append(float(loss.item()))

    return errors


def _compute_dc(errors):
    """
    估计收敛速度 dc（误差减少量 / 步数）。
    dc > 0 对应非平凡稳定边界。

    Args:
        errors: (n_steps,) 每步预测误差

    Returns:
        dc: 平均收敛速度（误差每步减少量）
    """
    if len(errors) < 2:
        return 0.0
    diffs = [errors[i] - errors[i + 1] for i in range(len(errors) - 1)]
    return float(np.mean(diffs))


# ── 主实验 ─────────────────────────────────────────────────────

def run_online_interaction_test():
    """
    运行在线交互测试：F3 vs 欧氏在线收敛速度对比。

    Returns:
        dict with keys: f3_dc_mean, euc_dc_mean, n_f3_faster
    """
    print("模块4b：Law II 在线收敛速度验证")
    print("=" * 55)
    print(f"  N_SEEDS={N_SEEDS}, EP_PRE={EP_PRE}, N_ONLINE_STEPS={N_ONLINE_STEPS}")
    print()
    print("  理论预测（Theorem 4）：")
    print("  F3 洛伦兹签名 → dc > 0 → 在线收敛更快")
    print()

    f3_dcs  = []
    euc_dcs = []

    # 固定测试数据
    X_test, _ = build_dataset(n_per_label=10, seed=42)

    for i in range(N_SEEDS):
        seed = i
        dcs_per_mode = {}

        for mode in ("f3", "euclid"):
            # 欧氏对照：'f2' 无 timelike mask → 退化为标准注意力（欧氏）
            model = LLCMBackbone(mode=mode if mode == "f3" else "f2").to(device)
            pretrain(model, seed=seed * 1000, n_epochs=EP_PRE)
            errors = _online_adapt(model, X_test, n_steps=N_ONLINE_STEPS)
            dc     = _compute_dc(errors)
            dcs_per_mode[mode] = dc

        f3_dcs.append(dcs_per_mode["f3"])
        euc_dcs.append(dcs_per_mode["euclid"])

        direction = "✅" if dcs_per_mode["f3"] > dcs_per_mode["euclid"] else "❌"
        print(f"  Seed {i + 1}/{N_SEEDS}: "
              f"F3 dc={dcs_per_mode['f3']:.4f}  "
              f"Euclidean dc={dcs_per_mode['euclid']:.4f}  {direction}")

    f3_dc_mean  = float(np.mean(f3_dcs))
    euc_dc_mean = float(np.mean(euc_dcs))
    n_f3_faster = sum(1 for f, e in zip(f3_dcs, euc_dcs) if f > e)

    print()
    print("─" * 55)
    print(f"  F3 dc 均值       : {f3_dc_mean:.4f}")
    print(f"  Euclidean dc 均值: {euc_dc_mean:.4f}")
    print(f"  {n_f3_faster}/{N_SEEDS} seeds: F3 dc > Euclidean dc")
    print()

    if f3_dc_mean > 0:
        print("  ✅ dc > 0：F3 系统存在非平凡稳定边界（Law II 支持）")
    else:
        print("  ⚠️  dc ≤ 0：可能需要更多预训练 epoch 或更小的学习率")

    if n_f3_faster >= (N_SEEDS + 1) // 2:
        print(f"  F3 在线收敛速度优于欧氏（{n_f3_faster}/{N_SEEDS} seeds）")
    else:
        print("  ⚠️  结果不一致，Law II 验证进行中")

    print()
    print("  注：此实验为验证进行中（🔄），结果可能不稳定。")
    print("     dc > 0 是 Theorem 4 对应的非平凡稳定边界存在条件。")

    return {
        "f3_dc_mean":  f3_dc_mean,
        "euc_dc_mean": euc_dc_mean,
        "n_f3_faster": n_f3_faster,
    }


if __name__ == "__main__":
    run_online_interaction_test()
