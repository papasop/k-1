"""
在线交互实验（online_interaction_test.py）
==========================================
从 K=1 Chronogeometrodynamics 提取的在线学习公式：

  Law II:   dx/dt = (JG − D)∇V_total
  Law III:  K(x) = x⊤Gx → 1（统计吸引子）
  Theorem 4: dc > 0 ⟺ det G < 0（洛伦兹几何必要条件）
  Theorem 6: κK = 4dc（收敛速度由几何决定）
  Prop 7:   σ² = 2dc·T_tol（噪声-温度匹配）

实验设计：
  第一阶段：离线预训练（建立洛伦兹几何）
  第二阶段：在线交互（Law II 驱动更新）
    - 机器人执行运动 → 物理状态嵌入 x
    - 人类语言反馈 → V_lang（代价信号）
    - Law II 更新 lang_aligner 权重
    - 测量收敛速度是否符合 κK = 4dc

验证的核心假设：
  F3（洛伦兹）的 dc > 0，在线收敛速度 κK_F3 > κK_euc
  欧氏的 dc = 0（Theorem 3），在线学习需要更多交互轮次

使用：
  python examples/online_interaction_test.py
  # 或：
  exec(open('examples/online_interaction_test.py').read())
"""

import numpy as np
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')

from llcm.core import (
    LLCMBackbone,
    EuclideanBackbone,
    pretrain,
    pretrain_euc,
    compute_dc,
    compute_K,
    compute_kappa,
    online_step,
    build_dataset,
    encode,
    STABLE_INSTRUCTIONS,
    T_IN,
    STATE_DIM,
    EMBED_DIM,
    LANG_DIM,
    T_DIM,
    EP_PRE,
    device,
)

# ── 在线交互超参数（对应 Law II）──────────────────────────────
LR_ONLINE  = 3e-4   # D（耗散项，学习率）
N_INTERACT = 20     # 每轮交互步数
N_ROUNDS   = 10     # 总交互轮数
N_SEEDS    = 5      # 随机种子数

print(f'Device: {device}')
print(f'T_DIM={T_DIM}, EMBED_DIM={EMBED_DIM}, LANG_DIM={LANG_DIM}')


# ── 单次种子完整实验 ────────────────────────────────────────────

def run_seed(seed):
    """
    完整在线交互实验（单个随机种子）。

    第一阶段：离线预训练（LLCMBackbone F3 + EuclideanBackbone）
    第二阶段：在线交互（N_ROUNDS 轮 × N_INTERACT 步，Law II 驱动）

    Args:
        seed : 随机种子

    Returns:
        dict with keys:
            dc_f3, dc_euc          : 几何修正项（Theorem 4）
            kappa_f3, kappa_euc    : 在线收敛速度（Theorem 6）
            K_hist_f3, K_hist_euc  : K(x) 历史（Law III）
            loss_hist_f3, loss_hist_euc : 每轮平均损失
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── 第一阶段：离线预训练（建立洛伦兹 / 欧氏几何）────────
    print(f'  Seed {seed}: pre-training F3 ...', end=' ', flush=True)
    model_f3 = LLCMBackbone()
    pretrain(model_f3, seed=seed * 100, epochs=EP_PRE)
    print('done.   pre-training Euc ...', end=' ', flush=True)
    model_euc = EuclideanBackbone()
    pretrain_euc(model_euc, seed=seed * 100, epochs=EP_PRE)
    print('done.')

    # 语言反馈参考向量（人类语言信号 → V_lang 代价信号）
    v_ref = F.normalize(
        encode(STABLE_INSTRUCTIONS).mean(dim=0, keepdim=True), dim=-1
    ).to(device)

    # ── 第二阶段：在线交互（Law II 驱动）────────────────────
    # 仅更新 lang_aligner 权重（模块4，对应 Law II 的 D 耗散项）
    opt_f3  = torch.optim.AdamW(
        model_f3.lang_aligner.parameters(),  lr=LR_ONLINE)
    opt_euc = torch.optim.AdamW(
        model_euc.lang_aligner.parameters(), lr=LR_ONLINE)

    K_hist_f3,     K_hist_euc     = [], []
    loss_hist_f3,  loss_hist_euc  = [], []

    for rnd in range(N_ROUNDS):
        # 机器人执行运动 → 物理状态嵌入 x
        X, _ = build_dataset(seed=seed * 10000 + rnd, n_per=4)
        x_phys  = torch.from_numpy(X[:4, :T_IN]).to(device)
        v_batch = v_ref.expand(4, -1)

        # Law II 交互步（D 耗散项驱动权重更新）
        round_losses_f3,  round_losses_euc  = [], []
        for _ in range(N_INTERACT):
            l_f3  = online_step(model_f3,  x_phys, v_batch, opt_f3)
            l_euc = online_step(model_euc, x_phys, v_batch, opt_euc)
            round_losses_f3.append(l_f3)
            round_losses_euc.append(l_euc)

        loss_hist_f3.append(float(np.mean(round_losses_f3)))
        loss_hist_euc.append(float(np.mean(round_losses_euc)))

        # Law III: 测量 K(x) = x⊤Gx（统计吸引子）
        K_hist_f3.append(compute_K(model_f3,  x_phys))
        K_hist_euc.append(compute_K(model_euc, x_phys))

    dc_f3     = compute_dc(model_f3)
    dc_euc    = compute_dc(model_euc)
    kappa_f3  = compute_kappa(loss_hist_f3)
    kappa_euc = compute_kappa(loss_hist_euc)

    return {
        'dc_f3':         dc_f3,
        'dc_euc':        dc_euc,
        'kappa_f3':      kappa_f3,
        'kappa_euc':     kappa_euc,
        'K_hist_f3':     K_hist_f3,
        'K_hist_euc':    K_hist_euc,
        'loss_hist_f3':  loss_hist_f3,
        'loss_hist_euc': loss_hist_euc,
    }


# ── 主实验 ─────────────────────────────────────────────────────

print('=' * 60)
print('Online Interaction Test — K=1 Chronogeometrodynamics')
print('=' * 60)

results = [run_seed(s) for s in range(N_SEEDS)]

dc_f3_arr  = np.array([r['dc_f3']    for r in results])
dc_euc_arr = np.array([r['dc_euc']   for r in results])
kk_f3_arr  = np.array([r['kappa_f3'] for r in results])
kk_euc_arr = np.array([r['kappa_euc']for r in results])

T_tol = N_ROUNDS * N_INTERACT   # Prop 7 temperature tolerance

# ── Theorem 4: dc > 0 ⟺ det G < 0 ─────────────────────────────
print(f'\n── Theorem 4: dc > 0 ⟺ det G < 0 ──')
print(f'  F3 (洛伦兹):  dc = {dc_f3_arr.mean():.4f} ± {dc_f3_arr.std():.4f}')
print(f'  欧氏:         dc = {dc_euc_arr.mean():.4f} ± {dc_euc_arr.std():.4f}')
T4_ok = bool(dc_f3_arr.mean() > 1e-4 and dc_euc_arr.mean() == 0.0)
print(f'  dc_F3 > 0，dc_Euc = 0: {"✓ PASS" if T4_ok else "✗ FAIL"}')

# ── Theorem 6: κK = 4dc ─────────────────────────────────────────
print(f'\n── Theorem 6: κK = 4dc（收敛速度由几何决定）──')
print(f'  F3  κK = {kk_f3_arr.mean():.4f} ± {kk_f3_arr.std():.4f}')
print(f'  Euc κK = {kk_euc_arr.mean():.4f} ± {kk_euc_arr.std():.4f}')
T6_ok = bool(kk_f3_arr.mean() >= kk_euc_arr.mean())
print(f'  κK_F3 >= κK_Euc: {"✓ PASS" if T6_ok else "✗ FAIL"}')
pred_kappa_f3 = 4.0 * dc_f3_arr.mean()
print(f'  4·dc_F3 = {pred_kappa_f3:.4f}（预测值）  '
      f'κK_F3 = {kk_f3_arr.mean():.4f}（实测值）')

# ── Prop 7: σ² = 2dc·T_tol ──────────────────────────────────────
print(f'\n── Prop 7: σ² = 2dc·T_tol（噪声-温度匹配）──')
sigma2_f3  = 2.0 * dc_f3_arr.mean()  * T_tol
sigma2_euc = 2.0 * dc_euc_arr.mean() * T_tol
print(f'  T_tol = {T_tol}')
print(f'  F3  σ² = 2·dc·T_tol = {sigma2_f3:.4f}')
print(f'  Euc σ² = 2·dc·T_tol = {sigma2_euc:.4f}')

# ── 逐种子: 4dc vs κK (F3) ──────────────────────────────────────
print(f'\n── 逐种子: 4dc vs κK（F3）──')
for i, r in enumerate(results):
    pred  = 4.0 * r['dc_f3']
    meas  = r['kappa_f3']
    ratio = meas / pred if pred > 1e-8 else float('nan')
    K_lo  = min(r['K_hist_f3'])
    K_hi  = max(r['K_hist_f3'])
    print(f'  Seed {i}: 4dc={pred:.4f}  κK={meas:.4f}  '
          f'ratio={ratio:.2f}  K_F3∈[{K_lo:.3f},{K_hi:.3f}]')

# ── 汇总 ────────────────────────────────────────────────────────
print('\n' + '=' * 60)
all_pass = T4_ok and T6_ok
print(f'Theorem 4 (dc_F3 > 0, dc_Euc = 0):  {"PASS" if T4_ok else "FAIL"}')
print(f'Theorem 6 (κK_F3 >= κK_Euc):         {"PASS" if T6_ok else "FAIL"}')
print(f'Overall: {"ALL PASS" if all_pass else "SOME FAIL"}')
print('=' * 60)
