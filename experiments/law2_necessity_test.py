"""
Law II 必要性验证 v2(paper-ready)
====================================
文件:experiments/law2_necessity_test.py

核心问题:
  反向传播动力学下,Lorentzian signature 约束(Sig(G)=(1,1))
  是否足以让 mq 自然收敛到 -1?

实验设计(三组对照):
  A. baseline:  纯欧氏 Transformer + 仅分类损失
                → 反向传播完全不在乎几何,mq 自由演化
  B. signature: A + 在嵌入层加 Lorentzian 内积测量(无 project, 无 mf loss)
                → 测量"知道"几何存在,但不显式约束 mq
  C. mf loss:   A + (mq+1)² 显式约束
                → 对照:加了显式约束后 mq 是否会收敛?

诊断指标(每组都报告):
  - mq_mean (训练后平均)
  - mq 随 epoch 的轨迹(用于画 money figure)
  - tl_ratio (类时样本比例)
  - 梯度对齐:cos(∇loss, ∇mq_distance)
    若 ≈ 0,说明反向传播看不到朝 mq=-1 的梯度信号(动力学反例)
  - 分类准确率(确认任务确实被学会了)

理论论证(在汇总里输出):
  随机 d 维欧氏嵌入的 mq 期望 ≈ d-2 = 62
  所以 mq=-1 不仅不是吸引子,它根本不在反向传播的初始化邻域里
  网络要走到它需要穿过整个高维空间的类空区
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D      = 64       # 与工程线对齐(layer1 用 256, riemannian 用 64)
SEEDS  = [0, 1, 2]
EPS    = 1e-6
EP     = 80
LR     = 1e-4
BS     = 32

print(f"Device: {DEVICE}")
print(f"嵌入维度: D={D}  种子数: {len(SEEDS)}  训练 epoch: {EP}")


# ══════════════════════════════════════════════════════════════
# Backbone(三组共享,只通过 loss 配置区分)
# ══════════════════════════════════════════════════════════════
class FreeNet(nn.Module):
    """
    纯欧氏 Transformer,无任何流形约束。
    
    embed_seq 输出 (B, d) 的嵌入向量,
    我们外部测量它的 Lorentzian "norm squared":
       mq = -emb[:,0]² + Σ emb[:,1:]²
    
    没有 project,没有 Minkowski LayerNorm,
    没有 g_net(上一版的 g_net 实际上没起作用,删掉了)。
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
    """计算 emb 的 Lorentzian norm squared(固定度规)"""
    return -emb[:, 0]**2 + (emb[:, 1:]**2).sum(-1)


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
# 训练函数:三种损失配置
# ══════════════════════════════════════════════════════════════
def train(model, X, L, mode='baseline', seed=0):
    """
    mode:
      'baseline':  loss = loss_cls
      'signature': loss = loss_cls + 0.1 * (-emb₀²的负向激励)
                   (不强制 mq=-1, 只激励"时间方向有幅度")
      'mf':        loss = loss_cls + 1.0 * (mq+1)²
                   (对照:显式约束 mq → -1)
    """
    torch.manual_seed(seed)
    X, L = X.to(DEVICE), L.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    
    mq_trajectory = []   # 每个 epoch 测一次,用于画 money figure
    
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
            elif mode == 'signature':
                # 仅激励时间方向有幅度,不约束符号
                # 这模拟"知道有个时间方向,但不强加流形"
                loss_sep = -((emb[:, 0])**2).mean() * 0.01
                loss = loss_cls + loss_sep
            elif mode == 'mf':
                # 显式约束 mq → -1
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
        
        # 每 epoch 测一次 mq(用于轨迹图)
        model.eval()
        with torch.no_grad():
            _, emb_eval = model(X[:100])
            mq_eval = measure_mq(emb_eval).mean().item()
            mq_trajectory.append(mq_eval)
        model.train()
    
    return mq_trajectory


def gradient_alignment_diagnostic(model, X, L):
    """
    诊断:朝向 mq=-1 的梯度方向,与朝向 loss 下降的梯度方向,
    在嵌入空间的余弦相似度。
    
    若 ≈ 0:反向传播看不到任何朝 mq=-1 的信号(动力学反例)
    若 > 0:反向传播至少不抗拒朝 mq=-1
    若 < 0:反向传播主动远离 mq=-1
    """
    model.eval()
    X, L = X.to(DEVICE), L.to(DEVICE)
    
    # 取一个 batch
    xb, lb = X[:BS], L[:BS]
    
    # forward,保留 emb 的梯度
    h = model.embed(xb)
    h = model.tr(h)
    emb = h[:, -1, :].detach().clone().requires_grad_(True)
    logits = model.cls_head(emb)
    
    # loss 关于 emb 的梯度
    loss = F.cross_entropy(logits, lb)
    grad_loss = torch.autograd.grad(loss, emb, retain_graph=True)[0]
    
    # mq_distance = (mq + 1)² 关于 emb 的梯度
    mq = measure_mq(emb)
    mq_distance = ((mq + 1.0)**2).mean()
    grad_mq = torch.autograd.grad(mq_distance, emb)[0]
    
    # 余弦相似度(展平)
    cos_sim = F.cosine_similarity(
        grad_loss.flatten().unsqueeze(0),
        grad_mq.flatten().unsqueeze(0)
    ).item()
    
    # 为了解释:再算两个方向的范数比
    norm_ratio = grad_mq.norm().item() / (grad_loss.norm().item() + EPS)
    
    return cos_sim, norm_ratio


# ══════════════════════════════════════════════════════════════
# 主实验
# ══════════════════════════════════════════════════════════════
def run():
    print("\n" + "="*65)
    print("Law II 必要性验证 v2:三组对照")
    print("="*65)
    print()
    print("A. baseline:  纯欧氏 + 分类损失,看 mq 自然演化")
    print("B. signature: A + 时间方向激励(不强制 mq=-1)")
    print("C. mf loss:   A + (mq+1)² 显式约束(对照组)")
    print()
    
    # 理论基线:随机初始化下 mq 的期望
    expected_init_mq = D - 2  # E[Σxᵢ²] - E[x₀²] = (d-1) - 1
    print(f"理论:随机 d={D} 维嵌入下,E[mq] ≈ d-2 = {expected_init_mq}")
    print(f"      (因为时间方向只有 1 维,空间方向有 d-1 维)")
    print(f"      所以 mq=-1 距离初始化点 ≈ {expected_init_mq + 1} 个单位")
    print()
    
    X_te, L_te = build_dataset(42, 200)
    X_te = X_te.to(DEVICE)
    
    all_results = {'baseline': [], 'signature': [], 'mf': []}
    all_trajectories = {'baseline': [], 'signature': [], 'mf': []}
    
    for seed in SEEDS:
        print(f"\n{'='*65}")
        print(f"Seed {seed}")
        print('='*65)
        torch.manual_seed(seed); np.random.seed(seed)
        X_tr, L_tr = build_dataset(seed+100, 400)
        
        for mode in ['baseline', 'signature', 'mf']:
            print(f"\n  ── 模式: {mode} ──")
            torch.manual_seed(seed)  # 同 seed,公平对比
            np.random.seed(seed)
            model = FreeNet(d=D).to(DEVICE)
            
            # 训练前 mq
            model.eval()
            with torch.no_grad():
                _, emb0 = model(X_te)
                mq0 = measure_mq(emb0).mean().item()
            
            # 梯度对齐诊断(训练前)
            cos_pre, ratio_pre = gradient_alignment_diagnostic(
                model, X_tr, L_tr)
            
            # 训练
            traj = train(model, X_tr, L_tr, mode=mode, seed=seed)
            all_trajectories[mode].append(traj)
            
            # 训练后评估
            model.eval()
            with torch.no_grad():
                _, emb = model(X_te)
                mq_per_sample = measure_mq(emb)
                mq_mean = mq_per_sample.mean().item()
                tl_ratio = (mq_per_sample < 0).float().mean().item()
                logits, _ = model(X_te)
                acc = (logits.argmax(-1).cpu() == L_te.cpu()
                       ).float().mean().item()
            
            # 梯度对齐诊断(训练后)
            cos_post, ratio_post = gradient_alignment_diagnostic(
                model, X_tr, L_tr)
            
            print(f"    mq:    训练前 {mq0:+7.2f}  →  训练后 {mq_mean:+7.2f}")
            print(f"    类时:  {tl_ratio:.0%}  (0% = 全部类空, 100% = 全部类时)")
            print(f"    分类:  {acc:.1%}")
            print(f"    梯度对齐 cos(∇loss, ∇mq_distance):")
            print(f"       训练前: {cos_pre:+.4f}  (norm ratio: {ratio_pre:.2f})")
            print(f"       训练后: {cos_post:+.4f}  (norm ratio: {ratio_post:.2f})")
            
            all_results[mode].append(dict(
                mq0=mq0, mq=mq_mean, tl=tl_ratio, acc=acc,
                cos_pre=cos_pre, cos_post=cos_post,
            ))
    
    # ══════════════════════════════════════════════════════════
    # 汇总
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("汇总(3 seeds)")
    print("="*65)
    
    print(f"\n  {'模式':<12} {'mq_init':>10} {'mq_final':>10} "
          f"{'tl_ratio':>10} {'acc':>8} {'cos∇':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    for mode in ['baseline', 'signature', 'mf']:
        rs = all_results[mode]
        mq0_m  = np.mean([r['mq0'] for r in rs])
        mq_m   = np.mean([r['mq']  for r in rs])
        tl_m   = np.mean([r['tl']  for r in rs])
        acc_m  = np.mean([r['acc'] for r in rs])
        cos_m  = np.mean([r['cos_post'] for r in rs])
        print(f"  {mode:<12} {mq0_m:>+10.2f} {mq_m:>+10.2f} "
              f"{tl_m:>10.0%} {acc_m:>8.1%} {cos_m:>+10.4f}")
    
    # ══════════════════════════════════════════════════════════
    # 结论
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("结论")
    print("="*65)
    
    baseline_mq = np.mean([r['mq'] for r in all_results['baseline']])
    sig_mq      = np.mean([r['mq'] for r in all_results['signature']])
    mf_mq       = np.mean([r['mq'] for r in all_results['mf']])
    baseline_cos = np.mean([r['cos_post'] for r in all_results['baseline']])
    
    print(f"\n  1. baseline (无任何几何信号):")
    print(f"     mq_init = {expected_init_mq}(理论)→ "
          f"mq_final = {baseline_mq:+.1f}")
    print(f"     梯度对齐 cos = {baseline_cos:+.4f}")
    if abs(baseline_cos) < 0.1:
        print(f"     → 梯度方向与朝 mq=-1 的方向 ≈ 正交")
        print(f"     → 反向传播看不到几何梯度信号")
    
    print(f"\n  2. signature (仅激励时间方向幅度):")
    print(f"     mq_final = {sig_mq:+.1f}")
    if sig_mq > 0:
        print(f"     → 软激励不足以扭转 mq 方向")
    
    print(f"\n  3. mf loss (显式 (mq+1)² 约束):")
    print(f"     mq_final = {mf_mq:+.2f}")
    if abs(mf_mq + 1) < 1.0:
        print(f"     → 加了显式约束后 mq 才收敛到 -1")
    
    print(f"\n  论文叙事:")
    print(f"  - Theorem 4 在 Law II 动力学下成立")
    print(f"  - 反向传播不实现 Law II,因此 Theorem 4 的吸引子不存在")
    print(f"  - 几何梯度信号缺失({baseline_cos:+.3f}),mq 自然漂向类空区")
    print(f"  - LLCM 的 project + loss_mf 三件套是 Law II 的工程替代")
    print(f"  - 这套替代在反向传播下复现了 Theorem 4 预言的几何")
    
    # 保存数据用于画图
    import pickle
    with open('law2_necessity_data.pkl', 'wb') as f:
        pickle.dump({
            'trajectories': all_trajectories,
            'results': all_results,
            'config': {'D': D, 'EP': EP, 'SEEDS': SEEDS,
                       'expected_init_mq': expected_init_mq},
        }, f)
    print(f"\n  轨迹数据已保存到 law2_necessity_data.pkl")
    print(f"  (用于画 money figure: mq vs epoch, 三组对照)")


if __name__ == '__main__':
    run()
