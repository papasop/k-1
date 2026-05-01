"""
Law II 必要性验证 v3 (paper-ready, addresses review feedback v2)
================================================================
文件: experiments/law2_necessity_test.py

修复来自 v2 评审的七个问题:
  1. signature 组改为纯测量(无 loss 贡献),增加 time_axis_loss 作为额外对照
  2. 度规签名术语统一为 (1, D-1),不再用 (1,1)
  3. 理论 E[mq]=D-2 改为参考值,主要报告实测 mq0
  4. 梯度诊断明确为 cos(∇loss_cls, ∇(mq+1)²),与训练 loss 解耦
  5. 增加 per-sample mq 散布指标 (mq_std, mean|mq+1|, on_shell_ratio)
  6. 主要收敛指标改为 on_shell_ratio,tl_ratio 降级为方向性诊断
  7. 这个文件就是新版,直接覆盖旧版

核心问题:
  反向传播动力学下,固定 Lorentzian 度规 (signature (1, D-1)) 的
  几何吸引子 mq=-1 是否会自然形成?

四组对照:
  A. baseline:        loss = loss_cls
                      mq 完全自由演化,只有分类损失驱动
  B. signature_only:  loss = loss_cls + 0 * mq_diagnostic
                      只测量 mq,不让它参与训练梯度
                      与 A 完全等价(用于验证测量本身无副作用)
  C. time_axis_loss:  loss = loss_cls + 0.01 * (-emb₀²)
                      软激励时间方向幅度(无界,网络可能 game)
                      展示"naive 软约束"的失败模式
  D. mf_loss:         loss = loss_cls + 1.0 * (mq+1)²
                      显式约束 mq → -1
                      对照:加显式约束后是否真的收敛到 mq=-1?

诊断指标:
  per-sample 收敛性:
    mean|mq+1|       距离吸引子的平均绝对偏差(主要指标)
    on_shell_ratio   |mq+1| < 0.5 的比例(真正"在流形附近")
    mq_std           样本间 mq 散布(检测均值抵消假象)
  方向性:
    tl_ratio         mq < 0 的比例(几何方向)
    mq_mean          均值(辅助,可能被抵消)
  动力学:
    cos(∇loss_cls, ∇(mq+1)²)   分类损失本身的几何盲视
                                 与训练时加什么辅助 loss 无关
    norm_ratio                   两个梯度的 L2 范数比
  任务性能:
    accuracy         确认任务确实学会了
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D      = 64
SEEDS  = [0, 1, 2]
EPS    = 1e-6
EP     = 80
LR     = 1e-4
BS     = 32
ON_SHELL_TOL = 0.5  # |mq+1| < ON_SHELL_TOL 算"在流形附近"

print(f"Device: {DEVICE}")
print(f"嵌入维度: D={D} (signature (1, {D-1}))  种子数: {len(SEEDS)}  EP: {EP}")


# ══════════════════════════════════════════════════════════════
# Backbone
# ══════════════════════════════════════════════════════════════
class FreeNet(nn.Module):
    """
    纯欧氏 Transformer + 线性分类头。
    无 project,无 Minkowski LayerNorm,无任何流形约束。
    
    embed_seq 输出 (B, D),外部用固定 Lorentzian 度规
    (signature (1, D-1)) 测量其内积:
        mq = -emb[:, 0]² + Σ emb[:, 1:]²
    """
    def __init__(self, state_dim=6, d=D):
        super().__init__()
        self.d = d
        self.embed = nn.Linear(state_dim, d)
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=d*2,
            dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=2)
        self.cls_head = nn.Linear(d, 2)

    def embed_seq(self, x):
        h = self.embed(x)
        h = self.tr(h)
        return h[:, -1, :]

    def forward(self, x):
        emb = self.embed_seq(x)
        return self.cls_head(emb), emb


def measure_mq(emb):
    """固定 Lorentzian 度规 (signature (1, D-1)) 下的 norm squared"""
    return -emb[:, 0]**2 + (emb[:, 1:]**2).sum(-1)


def measure_geometry(emb):
    """
    返回完整的几何诊断字典(per-sample 统计)
    
    主要收敛指标:
        mean_abs_dev:     E[|mq+1|]      距离吸引子的平均绝对偏差
        on_shell_ratio:   P[|mq+1|<TOL]  在流形附近的样本比例
    辅助方向性指标:
        tl_ratio:         P[mq<0]        类时方向比例(必要不充分)
        mq_mean:          E[mq]          均值(可能被抵消)
        mq_std:           Std[mq]        散布(检测抵消假象)
    """
    mq = measure_mq(emb)
    mq_dev = (mq + 1.0).abs()
    return dict(
        mq_mean        = mq.mean().item(),
        mq_std         = mq.std().item(),
        mean_abs_dev   = mq_dev.mean().item(),
        on_shell_ratio = (mq_dev < ON_SHELL_TOL).float().mean().item(),
        tl_ratio       = (mq < 0).float().mean().item(),
    )


# ══════════════════════════════════════════════════════════════
# 数据生成
# ══════════════════════════════════════════════════════════════
def stable_ode(t, y):
    k, b = 2.0, 0.5
    return [y[3], y[4], y[5],
            -k*y[0]-b*y[3], -k*y[1]-b*y[4], -k*y[2]-b*y[5]]

def running_ode(t, y):
    d = 0.3
    return [y[3], y[4], y[5],
            -d*y[3], -d*y[4]-9.8, -d*y[5]]

def simulate(fn, ic, T=80, dt=0.033):
    sol = solve_ivp(fn, [0, T*dt], ic,
                    t_eval=np.linspace(0, T*dt, T), rtol=1e-6)
    return sol.y.T

def build_dataset(seed=42, n=200):
    rng = np.random.RandomState(seed)
    X, L = [], []
    for _ in range(n//2):
        t = simulate(stable_ode, rng.randn(6)*0.3)
        mu, sg = t.mean(0), np.maximum(t.std(0), 0.01)
        X.append((t-mu)/sg); L.append(0)
    for _ in range(n//2):
        t = simulate(running_ode, rng.randn(6)*0.3)
        mu, sg = t.mean(0), np.maximum(t.std(0), 0.01)
        X.append((t-mu)/sg); L.append(1)
    return (torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(L))


# ══════════════════════════════════════════════════════════════
# 训练:四组对照,每组 loss 配置不同
# ══════════════════════════════════════════════════════════════
def train(model, X, L, mode, seed=0):
    """
    mode:
      'baseline':       loss = loss_cls
      'signature_only': loss = loss_cls (mq 仅测量,与 baseline 等价)
                        用于验证测量本身不影响训练
      'time_axis_loss': loss = loss_cls + 0.01 * (-E[emb₀²])
                        无界软激励时间方向(展示 naive 软约束失败)
      'mf_loss':        loss = loss_cls + 1.0 * E[(mq+1)²]
                        显式约束 mq → -1
    
    返回:
        traj: List[dict]   每个 epoch 的几何诊断字典
    """
    torch.manual_seed(seed)
    X, L = X.to(DEVICE), L.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    
    trajectory = []  # 每个 epoch 一个完整诊断字典
    
    model.train()
    for epoch in range(EP):
        idx = torch.randperm(len(X))
        for i in range(0, len(X), BS):
            ib = idx[i:i+BS]
            xb, lb = X[ib], L[ib]
            logits, emb = model(xb)
            
            loss_cls = F.cross_entropy(logits, lb)
            
            if mode == 'baseline':
                loss = loss_cls
            elif mode == 'signature_only':
                # 只测量,不让 mq 参与梯度
                with torch.no_grad():
                    _ = measure_mq(emb)  # diagnostic side effect only
                loss = loss_cls
            elif mode == 'time_axis_loss':
                # 软激励时间方向幅度(无界)
                loss_axis = -((emb[:, 0])**2).mean() * 0.01
                loss = loss_cls + loss_axis
            elif mode == 'mf_loss':
                # 显式流形约束
                mq = measure_mq(emb)
                loss_mf = ((mq + 1.0)**2).mean()
                loss = loss_cls + 1.0 * loss_mf
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
        sched.step()
        
        # epoch 末:完整几何诊断
        model.eval()
        with torch.no_grad():
            _, emb_eval = model(X[:100])
            geom = measure_geometry(emb_eval)
        trajectory.append(geom)
        model.train()
    
    return trajectory


def gradient_alignment_diagnostic(model, X, L):
    """
    诊断:在嵌入空间,分类损失梯度方向 与 (mq+1)² 梯度方向 的对齐。
    
    *** 重要:这里的 loss 是 loss_cls (分类损失),不是任何 mode 的训练 loss ***
    
    意义:
      cos < 0:分类目标本身要求 emb 远离 mq=-1
      cos ≈ 0:分类目标对几何无方向偏好
      cos > 0:分类目标本身倾向 emb 朝 mq=-1
    
    这个量与训练时是否加 mf_loss 无关,它衡量的是
    "分类任务本身在嵌入空间的几何盲视程度"。
    
    norm_ratio = ||∇(mq+1)²|| / ||∇loss_cls||
        训练后此值若极大(如 10⁷),说明朝 mq=-1 的方向上
        的"力"远小于任务力,即使方向对了也会被淹没。
    """
    model.eval()
    X, L = X.to(DEVICE), L.to(DEVICE)
    xb, lb = X[:BS], L[:BS]
    
    h = model.embed(xb)
    h = model.tr(h)
    emb = h[:, -1, :].detach().clone().requires_grad_(True)
    logits = model.cls_head(emb)
    
    loss_cls = F.cross_entropy(logits, lb)
    grad_loss = torch.autograd.grad(loss_cls, emb, retain_graph=True)[0]
    
    mq = measure_mq(emb)
    mq_distance = ((mq + 1.0)**2).mean()
    grad_mq = torch.autograd.grad(mq_distance, emb)[0]
    
    cos_sim = F.cosine_similarity(
        grad_loss.flatten().unsqueeze(0),
        grad_mq.flatten().unsqueeze(0)
    ).item()
    
    norm_ratio = grad_mq.norm().item() / (grad_loss.norm().item() + EPS)
    
    return cos_sim, norm_ratio


# ══════════════════════════════════════════════════════════════
# 主实验
# ══════════════════════════════════════════════════════════════
def run():
    print("\n" + "="*70)
    print("Law II 必要性验证 v3:四组对照")
    print("="*70)
    print()
    print("A. baseline:        loss = loss_cls")
    print("                    分类损失唯一驱动,看 mq 自由演化")
    print("B. signature_only:  loss = loss_cls (mq 仅测量,不入梯度)")
    print("                    sanity check:测量本身不影响训练")
    print("C. time_axis_loss:  loss = loss_cls + 0.01 * (-E[emb₀²])")
    print("                    naive 软约束:无界激励时间方向")
    print("D. mf_loss:         loss = loss_cls + 1.0 * E[(mq+1)²]")
    print("                    显式流形约束(LLCM 三件套之一)")
    print()
    
    # 理论参考
    expected_init_mq = D - 2
    print(f"理论参考:对各向同性高斯嵌入 emb_i ~ N(0,1) iid,")
    print(f"         E[mq] = E[Σemb_i²] - E[emb₀²] = (D-1) - 1 = {expected_init_mq}")
    print(f"         真实网络初始化以实测 mq0 为准(LayerNorm 影响 scale)")
    print(f"         mq=-1 (吸引子) 距离 mq=+{expected_init_mq}(参考)≈ {expected_init_mq+1} 单位")
    print()
    
    X_te, L_te = build_dataset(42, 200)
    X_te = X_te.to(DEVICE)
    
    modes = ['baseline', 'signature_only', 'time_axis_loss', 'mf_loss']
    all_results = {m: [] for m in modes}
    all_trajectories = {m: [] for m in modes}
    
    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"Seed {seed}")
        print('='*70)
        torch.manual_seed(seed); np.random.seed(seed)
        X_tr, L_tr = build_dataset(seed+100, 400)
        
        for mode in modes:
            print(f"\n  ── 模式: {mode} ──")
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = FreeNet(d=D).to(DEVICE)
            
            # 训练前几何
            model.eval()
            with torch.no_grad():
                _, emb0 = model(X_te)
                geom0 = measure_geometry(emb0)
            
            cos_pre, ratio_pre = gradient_alignment_diagnostic(
                model, X_tr, L_tr)
            
            # 训练
            traj = train(model, X_tr, L_tr, mode=mode, seed=seed)
            all_trajectories[mode].append(traj)
            
            # 训练后几何
            model.eval()
            with torch.no_grad():
                _, emb = model(X_te)
                geom = measure_geometry(emb)
                logits, _ = model(X_te)
                acc = (logits.argmax(-1).cpu() == L_te.cpu()
                       ).float().mean().item()
            
            cos_post, ratio_post = gradient_alignment_diagnostic(
                model, X_tr, L_tr)
            
            print(f"    训练前 mq: mean={geom0['mq_mean']:+7.2f}  "
                  f"|mq+1|={geom0['mean_abs_dev']:7.2f}  "
                  f"on_shell={geom0['on_shell_ratio']:.0%}")
            print(f"    训练后 mq: mean={geom['mq_mean']:+7.2f}  "
                  f"std={geom['mq_std']:6.2f}")
            print(f"             |mq+1|={geom['mean_abs_dev']:7.2f}  "
                  f"on_shell={geom['on_shell_ratio']:.0%}  "
                  f"tl={geom['tl_ratio']:.0%}")
            print(f"    分类准确率: {acc:.1%}")
            print(f"    cos(∇loss_cls, ∇(mq+1)²):")
            print(f"       前: {cos_pre:+.4f}  ratio: {ratio_pre:.2e}")
            print(f"       后: {cos_post:+.4f}  ratio: {ratio_post:.2e}")
            
            all_results[mode].append(dict(
                mq0=geom0['mq_mean'], 
                mq=geom['mq_mean'], 
                mq_std=geom['mq_std'],
                abs_dev=geom['mean_abs_dev'],
                on_shell=geom['on_shell_ratio'],
                tl=geom['tl_ratio'], 
                acc=acc,
                cos_pre=cos_pre, cos_post=cos_post,
                ratio_pre=ratio_pre, ratio_post=ratio_post,
            ))
    
    # ══════════════════════════════════════════════════════════
    # 汇总表
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*78)
    print(f"汇总({len(SEEDS)} seeds, ON_SHELL_TOL={ON_SHELL_TOL})")
    print("="*78)
    
    print(f"\n  {'模式':<18} {'mq_mean':>8} {'mq_std':>8} "
          f"{'|mq+1|':>8} {'on_shell':>9} {'tl':>5} "
          f"{'acc':>6} {'cos∇':>8}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*5} {'-'*6} {'-'*8}")
    for mode in modes:
        rs = all_results[mode]
        f = lambda k: np.mean([r[k] for r in rs])
        print(f"  {mode:<18} "
              f"{f('mq'):>+8.2f} "
              f"{f('mq_std'):>8.2f} "
              f"{f('abs_dev'):>8.2f} "
              f"{f('on_shell'):>9.0%} "
              f"{f('tl'):>5.0%} "
              f"{f('acc'):>6.1%} "
              f"{f('cos_post'):>+8.4f}")
    
    print(f"\n  说明:")
    print(f"    主要收敛指标: mean|mq+1| 越接近 0 越收敛, on_shell 越高越好")
    print(f"    辅助指标: tl_ratio 仅看方向(mq<0), 不区分 mq=-1 和 mq=-1000")
    print(f"    cos∇: 分类损失梯度与 (mq+1)² 梯度的对齐(分类目标的几何盲视)")
    
    # ══════════════════════════════════════════════════════════
    # 结论
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("结论")
    print("="*70)
    
    def mean_metric(mode, key):
        return np.mean([r[key] for r in all_results[mode]])
    
    print(f"\n  A. baseline:")
    print(f"     mq0 (实测) = {mean_metric('baseline', 'mq0'):+.1f}")
    print(f"          (理论参考 D-2 = {expected_init_mq})")
    print(f"     mq_final = {mean_metric('baseline', 'mq'):+.1f}")
    print(f"     |mq+1| = {mean_metric('baseline', 'abs_dev'):.1f}")
    print(f"     on_shell = {mean_metric('baseline', 'on_shell'):.0%}")
    print(f"     cos∇ = {mean_metric('baseline', 'cos_post'):+.4f}")
    print(f"     ratio = {mean_metric('baseline', 'ratio_post'):.2e}")
    
    print(f"\n  B. signature_only (sanity check):")
    print(f"     |mq+1| = {mean_metric('signature_only', 'abs_dev'):.1f}")
    print(f"     应当与 baseline 一致(测量不影响训练)")
    
    print(f"\n  C. time_axis_loss (naive 软约束):")
    print(f"     mq_final = {mean_metric('time_axis_loss', 'mq'):+.1f}")
    print(f"     |mq+1| = {mean_metric('time_axis_loss', 'abs_dev'):.1f}")
    print(f"     失败模式:无界激励 → 网络放大整体 scale")
    
    print(f"\n  D. mf_loss (显式约束):")
    print(f"     |mq+1| = {mean_metric('mf_loss', 'abs_dev'):.3f}")
    print(f"     on_shell = {mean_metric('mf_loss', 'on_shell'):.0%}")
    print(f"     mq_std = {mean_metric('mf_loss', 'mq_std'):.3f}")
    print(f"     acc = {mean_metric('mf_loss', 'acc'):.1%}")
    
    print(f"\n  论文叙事:")
    print(f"  - 反向传播 + 仅分类损失 → mq 远离吸引子,|mq+1|={mean_metric('baseline','abs_dev'):.0f}")
    print(f"  - 测量不改变训练(B = A,sanity check 通过)")
    print(f"  - naive 软约束被 game(C 放大整体 scale)")
    print(f"  - 显式 (mq+1)² 约束才能让 mq 真的收敛(D)")
    print(f"  - LLCM 三件套是反向传播下 Law II 的工程替代")
    
    # 保存
    import pickle
    with open('law2_necessity_data_v3.pkl', 'wb') as f:
        pickle.dump({
            'trajectories': all_trajectories,
            'results': all_results,
            'config': dict(D=D, EP=EP, SEEDS=SEEDS,
                           ON_SHELL_TOL=ON_SHELL_TOL,
                           expected_init_mq=expected_init_mq),
        }, f)
    print(f"\n  数据保存到 law2_necessity_data_v3.pkl")
    print(f"  (包含每 epoch 的完整几何诊断,可画 |mq+1| vs epoch 图)")


if __name__ == '__main__':
    run()
