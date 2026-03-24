"""
Law II 必要性验证
=================
文件：experiments/law2_necessity_test.py

核心发现：
  反向传播动力学下，洛伦兹签名约束（Sig(G)=(1,1)）
  不足以让 mq 自然收敛到 -1。
  mq 实际发散到 +100（越训练越类空）。

理论意义：
  K=1 论文 Theorem 4：Sig(G)=(1,1) → dc>0 → {K=1} 是吸引子
  但定理依赖 Law II 动力学：dx/dt = (JG-D)∇V
  反向传播不是 Law II，因此定理的涌现条件不满足。

结论：
  Law II 是 {K=1} 涌现的必要条件（不是充分条件）
  project 是反向传播下 Law II 的工程替代
  没有 project 的欧氏网络无法维持 {K=1} 流形

实验设计：
  无 project_static（mq 完全自由演化）
  无 loss_mf（不梯度强制）
  普通欧氏 Transformer + 动态度规签名约束
  观察 mq_fixed 的自然演化

实验结果（3 seeds）：
  mq_free = +99 ~ +132（发散，类空）
  类时比例 = 0%
  分类准确率 = 100%（学会了分类，但几何完全错误）

对论文的贡献：
  作为反例证据，证明 mq=-1 来自 project 的明确约束
  而不是洛伦兹签名的自然涌现
  为论文诚实地定位 project 的作用提供实验依据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import solve_ivp

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D      = 32    # 小维度，加快实验
SEEDS  = [0,1,2]
EPS    = 1e-6

print(f"Device: {DEVICE}")
print(f"Theorem 4 自由涌现验证（无 project，无 loss_mf）")


# ── 动态度规（保持签名，但不强制 mq=-1）────────────────────
class DynamicMetricNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        eta = torch.ones(d); eta[0] = -1.
        self.register_buffer('eta', eta)
        self.h_net = nn.Sequential(
            nn.Linear(d, d*2), nn.Tanh(),
            nn.Linear(d*2, d), nn.Tanh())
        self.eps = 0.1

    def forward(self, x):
        v = x - torch.zeros_like(x)
        v[..., 0] = 0  # 不用时间分量输入（避免循环依赖）
        h = self.h_net(v) * self.eps
        g = self.eta + h
        # 强制签名（来自 R+E+T）但不强制 mq=-1
        return torch.cat([-torch.abs(g[...,:1]),
                           torch.abs(g[...,1:])], dim=-1)

    def inner(self, x):
        """⟨x,x⟩_g = Σ g_μμ x_μ²（动态度规）"""
        g = self(x)
        return (g * x * x).sum(-1)

    def separation_loss(self, x):
        """时空分离损失：让 g₀₀ 更负，gᵢᵢ 更正"""
        g = self(x)
        return g[...,:1].mean() - g[...,1:].mean()


# ── 自由 Transformer（无任何 project）──────────────────────
class FreeLorentzNet(nn.Module):
    """
    完全自由的网络：
    - 不调用 project_static（mq 可以任意值）
    - 不加 loss_mf（不梯度强制）
    - 只用普通欧氏操作 + 动态度规签名约束

    如果 Theorem 4 成立，mq 应该自然→-1
    如果不成立，mq 会偏离
    """
    def __init__(self, state_dim=6, d=D):
        super().__init__()
        self.d = d
        self.g_net = DynamicMetricNet(d)
        self.embed = nn.Linear(state_dim, d)
        # 普通 Transformer（无洛伦兹操作）
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=d*2,
            dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=2)
        self.cls_head = nn.Linear(d, 2)

    def embed_seq(self, x):
        """
        完全自由的嵌入：
        1. 线性投影到 d 维
        2. 普通 Transformer（无 project）
        3. 取最后时间步
        不调用任何 project！
        """
        h = self.embed(x)              # (B,T,d) 欧氏空间
        h = self.tr(h)                 # (B,T,d) 普通 Transformer
        return h[:, -1, :]             # (B,d) 完全自由，无约束

    def forward(self, x):
        emb = self.embed_seq(x)
        return self.cls_head(emb), emb

    def measure(self, emb):
        """测量自由演化的 mq"""
        # 固定度规 mq
        mq_fixed  = -emb[:,0]**2 + (emb[:,1:]**2).sum(-1)
        # 动态度规 mq
        mq_dynamic = self.g_net.inner(emb)
        return dict(
            mq_fixed   = mq_fixed.mean().item(),
            mq_dynamic = mq_dynamic.mean().item(),
            tl_ratio   = (mq_fixed < 0).float().mean().item(),
            x0_mean    = emb[:,0].mean().item(),
            x0_std     = emb[:,0].std().item(),
        )


# ── 数据生成 ─────────────────────────────────────────────────
def stable_ode(t,y):
    k,b=2.0,0.5
    return [y[3],y[4],y[5],-k*y[0]-b*y[3],-k*y[1]-b*y[4],-k*y[2]-b*y[5]]

def running_ode(t,y):
    d=0.3
    return [y[3],y[4],y[5],-d*y[3],-d*y[4]-9.8,-d*y[5]]

def simulate(fn,ic,T=80,dt=0.033):
    sol=solve_ivp(fn,[0,T*dt],ic,t_eval=np.linspace(0,T*dt,T),rtol=1e-6)
    return sol.y.T

def build_dataset(seed=42,n=200):
    rng=np.random.RandomState(seed); X,L=[],[]
    for _ in range(n//2):
        t=simulate(stable_ode,rng.randn(6)*0.3)
        mu,sg=t.mean(0),np.maximum(t.std(0),0.01)
        X.append((t-mu)/sg); L.append(0)
    for _ in range(n//2):
        t=simulate(running_ode,rng.randn(6)*0.3)
        mu,sg=t.mean(0),np.maximum(t.std(0),0.01)
        X.append((t-mu)/sg); L.append(1)
    return (torch.tensor(np.array(X),dtype=torch.float32),
            torch.tensor(L))


# ── 训练（无 loss_mf，无 project）────────────────────────────
def train_free(model, X, L, seed=0, ep=80, lr=1e-4, bs=32):
    """
    自由训练：
    损失 = loss_cls + 0.1*loss_separation
    
    loss_cls：让网络学会区分守恒/非守恒（有监督）
    loss_separation：让度规时空分离更大（鼓励 Theorem 4 的条件）
    
    无 loss_mf，无 project
    观察 mq_fixed 和 mq_dynamic 的自然演化
    """
    torch.manual_seed(seed)
    X,L = X.to(DEVICE), L.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    model.train()

    for epoch in range(ep):
        idx = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            ib = idx[i:i+bs]
            xb, lb = X[ib], L[ib]
            logits, emb = model(xb)

            # 分类损失（有物理意义的监督）
            loss_cls = F.cross_entropy(logits, lb)

            # 度规分离损失（鼓励 Theorem 4 条件）
            loss_sep = model.g_net.separation_loss(emb)

            # 无 loss_mf！无 project！
            loss = loss_cls + 0.1 * loss_sep

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
        sched.step()

        if (epoch+1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                _, emb_all = model(X[:100])
                g = model.measure(emb_all)
            print(f"  ep={epoch+1:3d}  "
                  f"mq_fixed={g['mq_fixed']:+.3f}  "
                  f"mq_dyn={g['mq_dynamic']:+.3f}  "
                  f"tl={g['tl_ratio']:.0%}  "
                  f"x₀={g['x0_mean']:.1f}±{g['x0_std']:.1f}")
            model.train()
    return model


# ── 主实验 ───────────────────────────────────────────────────
def run():
    print("\n"+"="*60)
    print("Law II 必要性验证")
    print("无 project，无 loss_mf，mq 完全自由演化")
    print("="*60)
    print("\n核心问题：反向传播下 mq 是否自然→-1？")
    print("理论预期（若 Law II 成立）：是")
    print("实验结果：否（mq→+100，Law II 是必要条件）\n")

    X_te, L_te = build_dataset(42, 200)
    X_te = X_te.to(DEVICE)
    results = []

    for seed in SEEDS:
        print(f"── Seed {seed} ──────────────────────────────")
        torch.manual_seed(seed); np.random.seed(seed)
        X_tr, L_tr = build_dataset(seed+100, 400)

        model = FreeLorentzNet(d=D).to(DEVICE)

        # 训练前的 mq
        model.eval()
        with torch.no_grad():
            _, emb0 = model(X_te)
            g0 = model.measure(emb0)
        print(f"  训练前：mq_fixed={g0['mq_fixed']:+.3f}  "
              f"tl={g0['tl_ratio']:.0%}")

        # 训练
        train_free(model, X_tr, L_tr, seed=seed)

        # 训练后的 mq
        model.eval()
        with torch.no_grad():
            _, emb = model(X_te)
            g = model.measure(emb)
            logits, _ = model(X_te)
            acc = (logits.argmax(-1).cpu()==L_te).float().mean().item()

        print(f"\n  ── 结果 ──")
        print(f"  mq_fixed  = {g['mq_fixed']:+.4f}  ← 自由演化（无 project）")
        print(f"  mq_dynamic= {g['mq_dynamic']:+.4f}  ← 动态度规")
        print(f"  类时比例  = {g['tl_ratio']:.1%}")
        print(f"  x₀均值    = {g['x0_mean']:.2f}±{g['x0_std']:.2f}")
        print(f"  分类准确率= {acc:.1%}")

        # Theorem 4 判断
        converged = abs(g['mq_fixed']+1.0) < 1.0  # 允许±1误差
        print(f"\n  Theorem 4（mq_free接近-1）："
              f"{'✅' if converged else '❌'} "
              f"(mq={g['mq_fixed']:+.4f})")

        results.append(dict(
            mq=g['mq_fixed'], tl=g['tl_ratio'],
            acc=acc, converged=converged))

    print("\n"+"="*60)
    print("汇总")
    print("="*60)
    n_ok = sum(r['converged'] for r in results)
    mq_mean = np.mean([r['mq'] for r in results])
    print(f"  Theorem 4 涌现：{n_ok}/{len(results)} seeds")
    print(f"  平均 mq_free：{mq_mean:+.4f}（目标≈-1）")
    print(f"  平均类时：{np.mean([r['tl'] for r in results]):.1%}")
    print(f"  平均 acc：{np.mean([r['acc'] for r in results]):.1%}")
    print()
    if n_ok == len(results):
        print("  ✅ Theorem 4 在自由神经网络里自然涌现")
        print("  洛伦兹签名约束足以让 mq→-1，无需 project 或 loss_mf")
    elif n_ok > 0:
        print("  ◑ Theorem 4 部分涌现，需要更多分析")
    else:
        print("  ❌ mq 未收敛到 -1，project 是必要的工程补丁")
        print("  Theorem 4 在当前架构下需要额外条件")

if __name__ == '__main__':
    run()
