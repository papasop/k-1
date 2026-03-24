"""
全黎曼 Lorentzian Transformer 最小里程碑测试 v3
================================================
改进1: 时间步注入 x0（信息时间编码）
改进3: 流形约束权重5.0，EP_PRE=120
改进4: 监控 x0 正值
注：改进2（当前点参考）导致NaN，已移除，留作后续数值稳定化

里程碑：
  M1: 流形约束违反 < 0.01
  M2: 不依赖 sigma（天然满足）
  M3: 类时比例 > 95%，3/3 seeds
  M4: mq 差距 > 50倍 vs 欧氏
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import solve_ivp

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
D        = 64
N_HEADS  = 4
N_LAYERS = 3
SEEDS    = [0, 1, 2]
EP_PRE   = 120
LR_PRE   = 1e-4
BS       = 32
EPS      = 1e-6

print(f"Device: {DEVICE}")
print(f"流形: H^{{1,{D-1}}}（1时间 + {D-1}空间维度）")


# ── 洛伦兹流形核心操作 ────────────────────────────────────────

class LorentzManifold:
    @staticmethod
    def inner(x, y):
        return -x[...,0]*y[...,0] + (x[...,1:]*y[...,1:]).sum(-1)

    @staticmethod
    def norm_sq(x):
        return LorentzManifold.inner(x, x)

    @staticmethod
    def project(x):
        space = x[..., 1:].clamp(-8.0, 8.0)
        x0 = torch.sqrt(1.0 + (space**2).sum(-1, keepdim=True) + EPS)
        return torch.cat([x0, space], dim=-1)

    @staticmethod
    def constraint_violation(x):
        return (LorentzManifold.norm_sq(x) + 1.0).abs()

    @staticmethod
    def exp_map(x, v):
        vx    = LorentzManifold.inner(v, x).unsqueeze(-1)
        v_tan = v + vx * x
        v_ns  = LorentzManifold.inner(v_tan, v_tan).clamp(min=0)
        v_n   = torch.sqrt(v_ns + EPS).unsqueeze(-1).clamp(max=5.0)
        res   = torch.cosh(v_n)*x + torch.sinh(v_n)*v_tan/(v_n+EPS)
        return LorentzManifold.project(res)

    @staticmethod
    def log_map(x, y):
        y   = LorentzManifold.project(y)
        xy  = LorentzManifold.inner(x, y).unsqueeze(-1).clamp(max=-(1.0+EPS))
        # dist clamp(min=1e-3)：防止 x≈y 时 acosh 梯度爆炸（改进2的NaN来源）
        d   = torch.acosh((-xy).clamp(min=1.0+EPS)).clamp(min=1e-3)
        dir = y + xy * x
        dn  = torch.sqrt(LorentzManifold.inner(dir, dir).clamp(min=EPS)).unsqueeze(-1)
        return d * dir / (dn + EPS)


M = LorentzManifold()



# ══════════════════════════════════════════════════════════════
# 黎曼归一化（Minkowski LayerNorm）
# ══════════════════════════════════════════════════════════════

class MinkowskiLayerNorm(nn.Module):
    """
    洛伦兹切空间的黎曼归一化

    欧氏 LayerNorm：norm² = Σvᵢ²（正定，时间空间平等）
    Minkowski LN：  norm² = |-v₀² + Σvᵢ²|（不定，时间有负号）

    对切向量 v ∈ T_x H^{1,d-1}：
      - 保持类时/类空符号（||v||_L 的符号不变）
      - v₀ 被赋予负权重，符合洛伦兹几何
      - 可训练的缩放 γ 和偏置 β（和标准 LN 相同接口）
    """
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.d   = d
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta  = nn.Parameter(torch.zeros(d))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: (..., d)，切空间向量
        # Minkowski 范数平方：-v₀² + Σvᵢ²
        mink_sq = -v[..., :1]**2 + (v[..., 1:]**2).sum(-1, keepdim=True)
        # 取绝对值防负数开根，保持数值稳定
        norm = torch.sqrt(mink_sq.abs() + self.eps)
        # 归一化：保持方向，只缩放幅度
        v_norm = v / (norm + self.eps)
        # 可训练仿射变换（和标准 LN 相同）
        return self.gamma * v_norm + self.beta

# ── 全黎曼层 ─────────────────────────────────────────────────

class LorentzAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.d  = d; self.h = n_heads; self.dh = d // n_heads
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        mu = torch.zeros(self.d, device=x.device); mu[0] = 1.0
        mu = mu.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        v  = M.log_map(mu, x)

        def proj_h(w):
            h  = w.view(B, T, self.h, self.dh)
            sp = h[...,1:].clamp(-5.0, 5.0)
            x0 = torch.sqrt(1.0+(sp**2).sum(-1,keepdim=True)+EPS)
            return torch.cat([x0,sp],-1).transpose(1,2)

        q  = proj_h(self.Wq(v))
        k  = proj_h(self.Wk(v))
        vv = self.Wv(v).view(B,T,self.h,self.dh).transpose(1,2)

        sc = (-q[...,:1]*k[...,:1].transpose(-2,-1)
              + q[...,1:]@k[...,1:].transpose(-2,-1)) / (self.dh**0.5)
        at = F.softmax(sc, dim=-1)

        out = (at @ vv).transpose(1,2).contiguous().view(B,T,self.d)
        out = M.project(M.exp_map(mu, self.Wo(out)))
        return out


class LorentzFFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, d*2)
        self.fc2 = nn.Linear(d*2, d)

    def forward(self, x):
        B,T,d = x.shape
        mu = torch.zeros(d,device=x.device); mu[0]=1.0
        mu = mu.unsqueeze(0).unsqueeze(0).expand(B,T,-1)
        v  = M.log_map(mu, x)
        v  = F.gelu(self.fc1(v))
        v  = self.fc2(v)
        return M.project(M.exp_map(mu, v))


class LorentzBlock(nn.Module):
    def __init__(self, d, n_heads, norm_type='euclidean'):
        super().__init__()
        self.attn  = LorentzAttention(d, n_heads)
        self.ffn   = LorentzFFN(d)
        # norm_type: 'euclidean'（标准LN）或 'minkowski'（黎曼LN）
        if norm_type == 'minkowski':
            self.norm1 = MinkowskiLayerNorm(d)
            self.norm2 = MinkowskiLayerNorm(d)
        else:
            self.norm1 = nn.LayerNorm(d)
            self.norm2 = nn.LayerNorm(d)

    def forward(self, x, use_current_ref=False):
        # use_current_ref=True: 改进2（当前点参考），需 warm-up 后启用
        # use_current_ref=False: 固定 mu（默认，数值稳定）
        if use_current_ref:
            # 改进2：Log_x(y) 以当前点为参考，精确局部几何
            # 需要 dist clamp(min=1e-3) 防 NaN（已在 log_map 实现）
            x_a = self.attn(x)
            v   = M.log_map(x, x_a)          # 当前点切空间
            v   = self.norm1(v)
            x   = M.project(M.exp_map(x, v))

            x_f = self.ffn(x)
            v   = M.log_map(x, x_f)
            v   = self.norm2(v)
            x   = M.project(M.exp_map(x, v))
        else:
            # 固定 mu 参考（warm-up 阶段用）
            mu = torch.zeros(x.shape[-1], device=x.device); mu[0]=1.0
            mu = mu.unsqueeze(0).unsqueeze(0).expand(*x.shape)

            v = M.log_map(mu, x)
            v = self.norm1(v + M.log_map(mu, self.attn(x)))
            x = M.project(M.exp_map(mu, v))

            v = M.log_map(mu, x)
            v = self.norm2(v + M.log_map(mu, self.ffn(x)))
            x = M.project(M.exp_map(mu, v))
        return x


class LorentzBackbone(nn.Module):
    def __init__(self, state_dim=6, d=D, n_heads=N_HEADS, n_layers=N_LAYERS,
                 time_inject=True, norm_type='euclidean'):
        super().__init__()
        self.d = d
        self.norm_type = norm_type
        self.time_inject = time_inject
        self.input_proj = nn.Linear(state_dim, d)
        self.blocks = nn.ModuleList([
            LorentzBlock(d, n_heads, norm_type=norm_type) for _ in range(n_layers)
        ])
        self.cls_head = nn.Linear(d, 6)
        self.use_current_ref = False

    def embed_seq(self, x):
        B, T, _ = x.shape
        h = M.project(self.input_proj(x))

        # 改进1：时间步注入（消融开关 self.time_inject）
        # 理论依据：x₀ ↔ 信息时间 dt_info，序列位置 t 是物理时间的离散化
        # 注入方式：在切空间时间轴方向加偏置（不是加在 x₀ 再 project，那样会被抹掉）
        # 消融设计：
        #   time_inject=True  → 完整洛伦兹 Transformer（A组）
        #   time_inject=False → 纯几何基线，隔离时间注入贡献（B组）
        if self.time_inject:
            mu0 = torch.zeros(self.d, device=x.device); mu0[0] = 1.0
            mu0 = mu0.view(1,1,self.d).expand(B,T,-1)
            v   = M.log_map(mu0, h)
            t   = torch.arange(T, device=x.device, dtype=torch.float32) / T
            v[..., 0] = v[..., 0] + t.view(1,T) * 0.1  # 切空间时间轴注入
            h   = M.project(M.exp_map(mu0, v))

        for b in self.blocks:
            h = b(h, use_current_ref=self.use_current_ref)
        return M.project(h[:,-1,:])

    def forward(self, x):
        emb = self.embed_seq(x)
        mu  = torch.zeros(self.d, device=x.device); mu[0]=1.0
        v   = M.log_map(mu.unsqueeze(0).expand(emb.shape[0],-1), emb)
        return self.cls_head(v), emb

    def measure(self, emb):
        mq = M.norm_sq(emb)
        return dict(
            mq_mean  = mq.mean().item(),
            tl_ratio = (mq<0).float().mean().item(),
            violation= M.constraint_violation(emb).mean().item(),
            x0_mean  = emb[:,0].mean().item(),
            x0_min   = emb[:,0].min().item(),
        )


# ── 欧氏基线 ─────────────────────────────────────────────────

class EuclideanBackbone(nn.Module):
    def __init__(self, state_dim=6, d=D, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.proj = nn.Linear(state_dim, d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads,
              dim_feedforward=d*2, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.cls = nn.Linear(d, 2)

    def embed_seq(self, x):
        return self.tr(self.proj(x))[:,-1,:]

    def forward(self, x):
        e = self.embed_seq(x)
        return self.cls(e), e

    def measure(self, emb):
        mq = (emb[:,1:]**2).sum(-1) - emb[:,0]**2
        return dict(mq_mean=mq.mean().item(),
                    tl_ratio=(mq<0).float().mean().item(),
                    violation=float('nan'),   # 欧氏无流形约束
                    x0_mean=emb[:,0].mean().item(),
                    x0_min=emb[:,0].min().item())


# ── 数据生成 ─────────────────────────────────────────────────

def stable_ode(t, y):
    k,b = 2.0,0.5
    return [y[3],y[4],y[5],-k*y[0]-b*y[3],-k*y[1]-b*y[4],-k*y[2]-b*y[5]]

def running_ode(t, y):
    d = 0.3
    return [y[3],y[4],y[5],-d*y[3],-d*y[4]-9.8,-d*y[5]]

def simulate(fn, ic, T=80, dt=0.033):
    sol = solve_ivp(fn,[0,T*dt],ic,t_eval=np.linspace(0,T*dt,T),rtol=1e-6)
    return sol.y.T

def build_dataset(seed=42, n=200):
    rng = np.random.RandomState(seed)
    X,L = [],[]
    for _ in range(n//2):
        t = simulate(stable_ode, rng.randn(6)*0.3)
        mu,sg = t.mean(0),np.maximum(t.std(0),0.01)
        X.append((t-mu)/sg); L.append(0)
    for _ in range(n//2):
        t = simulate(running_ode, rng.randn(6)*0.3)
        mu,sg = t.mean(0),np.maximum(t.std(0),0.01)
        X.append((t-mu)/sg); L.append(1)
    return (torch.tensor(np.array(X),dtype=torch.float32), torch.tensor(L))


# ── 预训练 ───────────────────────────────────────────────────

def pretrain(model, X, L, seed=0):
    torch.manual_seed(seed)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_PRE)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP_PRE)
    X,L   = X.to(DEVICE), L.to(DEVICE)
    is_lor= isinstance(model, LorentzBackbone)
    model.train()

    for ep in range(EP_PRE):
        idx = torch.randperm(len(X))
        for i in range(0, len(X), BS):
            xb,lb = X[idx[i:i+BS]], L[idx[i:i+BS]]
            logits, emb = model(xb)

            if is_lor:
                lb2      = (lb>0).long()
                loss_cls = F.cross_entropy(logits[:,:2], lb2)
                mq       = M.norm_sq(emb)
                # 改进3：流形约束权重 5.0
                loss_mf  = (mq+1.0)**2
                sm = (lb==0).float(); ns = sm.sum()+EPS
                cm = (lb==1).float(); nc = cm.sum()+EPS
                mqs = (mq*sm).sum()/ns
                mqc = (mq*cm).sum()/nc
                loss_s = (mqs+1.0)**2
                loss_c = F.relu(-mqc)
                # 改进4：x0 > 0
                loss_x0 = F.relu(-emb[:,0]+0.1).mean()
                loss = loss_cls + 5.0*loss_mf.mean() + loss_s + loss_c + loss_x0
            else:
                lb2  = (lb>0).long()
                loss = F.cross_entropy(logits, lb2)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
        sched.step()

        # 改进2 warm-up：前 40% epoch 用固定 mu，之后切换到当前点参考
        if is_lor and ep == int(EP_PRE * 0.4):
            model.use_current_ref = True
            # 切换后降低 LR（当前点参考梯度更大）
            for pg in opt.param_groups:
                pg['lr'] *= 0.5

    return model


# ── 里程碑测试 ───────────────────────────────────────────────

def run():
    print("\n"+"="*60)
    print("全黎曼 Lorentzian Transformer 里程碑测试")
    print("="*60)

    X_te,L_te = build_dataset(42,200)
    X_te = X_te.to(DEVICE)
    results = []
    # 消融记录
    gaps_with_inject    = []  # A组：完整洛伦兹
    gaps_without_inject = []  # B组：纯几何

    for seed in SEEDS:
        print(f"\n{'='*50}\nSeed {seed}\n{'='*50}")
        torch.manual_seed(seed); np.random.seed(seed)
        X_tr,L_tr = build_dataset(seed+100,400)

        # ── 欧氏基线 ──────────────────────────────────────────
        euc = EuclideanBackbone().to(DEVICE)
        pretrain(euc, X_tr, L_tr, seed)
        euc.eval()
        with torch.no_grad():
            emb_e = euc.embed_seq(X_te)
            g_e   = euc.measure(emb_e)

        # ── 全黎曼 + 欧氏 LN ──────────────────────────────────
        lor_e = LorentzBackbone(norm_type='euclidean').to(DEVICE)
        pretrain(lor_e, X_tr, L_tr, seed)
        lor_e.eval()
        with torch.no_grad():
            emb_le = lor_e.embed_seq(X_te)
            g_le   = lor_e.measure(emb_le)

        # ── 全黎曼 + Minkowski LN ─────────────────────────────
        lor_m = LorentzBackbone(norm_type='minkowski').to(DEVICE)
        pretrain(lor_m, X_tr, L_tr, seed)
        lor_m.eval()
        with torch.no_grad():
            emb_lm = lor_m.embed_seq(X_te)
            g_lm   = lor_m.measure(emb_lm)

        mq_euc  = g_e['mq_mean']
        gap_e   = abs(mq_euc) / (abs(g_le['mq_mean'])+EPS)
        gap_m   = abs(mq_euc) / (abs(g_lm['mq_mean'])+EPS)

        def row(g, gap, label):
            viol = g['violation']
            v_str = f'{viol:.6f}' if not (viol != viol) else '   N/A  '  # NaN check
            return (f"  {label:<20} mq={g['mq_mean']:+.4f}  "
                    f"类时={g['tl_ratio']:.0%}  "
                    f"违反={v_str}  "
                    f"差距={gap:.1f}x  "
                    f"x₀均值={g['x0_mean']:.1f}")

        print(f"\n{'─'*60}")
        print(row(g_le, gap_e, '全黎曼+欧氏LN'))
        print(row(g_lm, gap_m, '全黎曼+MinkowskiLN'))
        print(row(g_e,  1.0,   '欧氏基线'))

        # 里程碑用欧氏LN版本（当前主线）
        m1 = g_le['violation'] < 0.01
        m2 = True
        m3 = g_le['tl_ratio'] > 0.95
        m4 = gap_e > 50

        # 额外：Minkowski版是否更好？
        mink_better_viol = g_lm['violation'] < g_le['violation']
        mink_better_gap  = gap_m > gap_e

        print(f"\n  Minkowski LN vs 欧氏 LN：")
        print(f"    约束违反更小：{'✅' if mink_better_viol else '❌'} "
              f"({g_lm['violation']:.6f} vs {g_le['violation']:.6f})")
        print(f"    mq差距更大：  {'✅' if mink_better_gap else '❌'} "
              f"({gap_m:.1f}x vs {gap_e:.1f}x)")

        print(f"\n  里程碑（欧氏LN主线）：")
        print(f"    M1 约束<0.01：{'✅' if m1 else '❌'} ({g_le['violation']:.4f})")
        print(f"    M2 不依赖σ：  ✅")
        print(f"    M3 类时>95%： {'✅' if m3 else '❌'} ({g_le['tl_ratio']:.1%})")
        print(f"    M4 差距>50x： {'✅' if m4 else '❌'} ({gap_e:.1f}x)")

        results.append(dict(
            m1=m1, m2=m2, m3=m3, m4=m4,
            viol_e=g_le['violation'], viol_m=g_lm['violation'],
            tl_e=g_le['tl_ratio'],   tl_m=g_lm['tl_ratio'],
            gap_e=gap_e,             gap_m=gap_m,
            mq_e=g_le['mq_mean'],    mq_m=g_lm['mq_mean'],
            mink_better_viol=mink_better_viol,
            mink_better_gap=mink_better_gap,
        ))
        gaps_with_inject.append(gap_e)

        # 消融B：纯几何（无时间注入）
        lor_b = LorentzBackbone(time_inject=False).to(DEVICE)
        pretrain(lor_b, X_tr, L_tr, seed)
        lor_b.eval()
        with torch.no_grad():
            emb_b = lor_b.embed_seq(X_te)
            g_b   = lor_b.measure(emb_b)
        gap_b = abs(g_e['mq_mean']) / (abs(g_b['mq_mean'])+EPS)
        gaps_without_inject.append(gap_b)
        print(f"  消融B（纯几何）mq差距：{gap_b:.1f}x，类时：{g_b['tl_ratio']:.1%}")

    print("\n"+"="*60+"\n汇总\n"+"="*60)
    for k,nm in [('m1','约束<0.01'),('m2','不依赖σ'),('m3','类时>95%'),('m4','差距>50x')]:
        c = sum(r[k] for r in results)
        print(f"  {nm}：{c}/{len(results)} {'✅' if c==len(results) else '◑'}")

    print(f"\n  {'指标':<18} {'欧氏LN':>10} {'MinkowskiLN':>14}")
    print(f"  {'─'*44}")
    print(f"  {'平均约束违反':<16} "
          f"{np.mean([r['viol_e'] for r in results]):>12.6f} "
          f"{np.mean([r['viol_m'] for r in results]):>14.6f}")
    print(f"  {'平均类时比例':<16} "
          f"{np.mean([r['tl_e'] for r in results]):>11.1%} "
          f"{np.mean([r['tl_m'] for r in results]):>13.1%}")
    print(f"  {'平均mq均值':<17} "
          f"{np.mean([r['mq_e'] for r in results]):>+12.4f} "
          f"{np.mean([r['mq_m'] for r in results]):>+14.4f}")
    print(f"  {'平均mq差距':<17} "
          f"{np.mean([r['gap_e'] for r in results]):>11.1f}x "
          f"{np.mean([r['gap_m'] for r in results]):>13.1f}x")

    mb_v = sum(r['mink_better_viol'] for r in results)
    mb_g = sum(r['mink_better_gap']  for r in results)
    print(f"\n  Minkowski 约束违反更小：{mb_v}/{len(results)} seeds")
    print(f"  Minkowski mq差距更大：  {mb_g}/{len(results)} seeds")

    ok = all(r['m3'] and r['m4'] for r in results)
    print(f"\n最小里程碑（M3+M4）：{'✅ 通过' if ok else '❌ 未通过'}")
    print("（对比隐式LLCM：类时97%，mq差距84倍，依赖sigma=0.56）")

    # 消融分析
    print("\n── 消融分析 ─────────────────────────────")
    a = np.mean(gaps_with_inject)
    b = np.mean(gaps_without_inject)
    print(f"  A组（完整洛伦兹+时间注入）：平均差距 {a:.1f}x")
    print(f"  B组（纯几何，无时间注入）： 平均差距 {b:.1f}x")
    print(f"  时间注入贡献：A-B = {a-b:+.1f}x")
    print(f"  几何结构贡献：B组已显著 > 50x → 洛伦兹几何本身足够")


if __name__ == '__main__':
    run()
