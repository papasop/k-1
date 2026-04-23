"""
层1最小验证：语言能否索引洛伦兹物理空间
==========================================
最小设计：
  - 50个样本/类，2类（stable vs changing）
  - 预训练 60 epoch
  - 微调 150 epoch
  - 1个seed，快速出结果

验证的问题：
  方向A: 物理轨迹 → 洛伦兹空间 → 语言描述嵌入对齐得分
         F3对齐得分 > 欧氏 → 洛伦兹空间更容易被语言索引 → 层1成立

使用：
    exec(open('layer1_minimal_test.py').read())
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader, TensorDataset
import warnings; warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── 超参数（最小版） ───────────────────────────────────────────
EMBED_DIM  = 128
N_HEADS    = 4
N_LAYERS   = 3
TIME_RATIO = 0.25
T_IN       = 20
T_OUT      = 20
STATE_DIM  = 6
LANG_DIM   = 384
N_LABELS   = 2       # 只用2类：stable vs changing
N_PER      = 50      # 每类50个样本
EP_PRE     = 60      # 预训练60 epoch
EP_FT      = 150     # 微调150 epoch（方向B需要更多收敛时间）
LR_PRE     = 3e-4
LR_FT      = 1e-4
BS         = 16


def _t_dim():
    return max(1, int(N_HEADS * TIME_RATIO)) * (EMBED_DIM // N_HEADS)


T_DIM = _t_dim()

# 标签和语言描述
LABELS = {0: 'momentum_stable', 1: 'momentum_changing'}
DESCRIPTIONS = {
    0: ["平稳匀速运动，动量保持守恒",
        "smooth constant velocity motion with conserved momentum"],
    1: ["动量持续变化，存在外力作用",
        "changing momentum with continuous force application"],
}

# ── 语言编码器 ─────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
lang_enc = SentenceTransformer('all-MiniLM-L6-v2')
print(f'语言编码器加载完成  dim={LANG_DIM}')


def encode_text(texts, dev=device):
    return lang_enc.encode(texts, convert_to_tensor=True,
                           show_progress_bar=False).to(dev)


# ── 物理数据 ───────────────────────────────────────────────────
def stable_ode(t, y):
    x, yp, z, vx, vy, vz = y
    return [vx, 0, vz, -0.001 * vx, 0, -0.001 * vz]


def running_ode(t, y):
    x, yp, z, vx, vy, vz = y
    g = 9.81; m = 70.0
    phase = (3.0 * t) % 1.0
    L = 0.95 + 0.12 * abs(np.sin(3.0 * np.pi * t))
    pen = max(0, L - yp)
    Fv = (2000 * pen - 30 * vy) if (phase < 0.4 and yp < L) else 0.0
    Fh = 200 * (2.0 - vx)
    return [vx, vy, vz, Fh / m, (Fv - m * g) / m, -0.001 * vz]


def simulate(ode_fn, n=N_PER, seed=0):
    rng = np.random.default_rng(seed)
    T = T_IN + T_OUT
    t_end = T * 0.05
    t_eval = np.linspace(0, t_end, T)
    segs = []
    for _ in range(n):
        y0 = [
            rng.uniform(-1.0, 1.0),
            rng.uniform(0.8, 1.2),
            rng.uniform(-0.5, 0.5),
            rng.uniform(0.5, 2.5),
            rng.uniform(-0.1, 0.1),
            rng.uniform(-0.2, 0.2),
        ]
        sol = solve_ivp(ode_fn, [0, t_end], y0,
                        t_eval=t_eval, max_step=0.01, dense_output=False)
        if sol.success and sol.y.shape[1] >= T:
            segs.append(sol.y.T[:T].astype(np.float32))
    if len(segs) == 0:
        print(f'  警告：所有仿真失败，以零轨迹代替（ODE 可能在此配置下不稳定）')
        segs = [np.zeros((T, STATE_DIM), dtype=np.float32)]
    return np.stack(segs)


def build_dataset(seed=42, n_per=N_PER):
    Xs = simulate(stable_ode,  n=n_per, seed=seed)
    Xr = simulate(running_ode, n=n_per, seed=seed + 1)
    X = np.concatenate([Xs, Xr], axis=0)
    y = np.array([0] * len(Xs) + [1] * len(Xr), dtype=np.int64)
    return X, y


# ── 模型定义 ───────────────────────────────────────────────────
from lorentz_transformer import (
    LorentzMultiHeadAttention,
    MinkowskiLayerNorm,
)


class _F3Cfg:
    d_model    = EMBED_DIM
    n_heads    = N_HEADS
    formula    = 'f3'
    time_ratio = TIME_RATIO
    dropout    = 0.0


class _LorentzBlock(nn.Module):
    """F3 光锥注意力块 + Minkowski LayerNorm"""

    def __init__(self):
        super().__init__()
        self.attn  = LorentzMultiHeadAttention(_F3Cfg())
        self.norm1 = MinkowskiLayerNorm(EMBED_DIM, t_dim=T_DIM)
        self.ffn   = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * 4),
            nn.GELU(),
            nn.Linear(EMBED_DIM * 4, EMBED_DIM),
        )
        self.norm2 = MinkowskiLayerNorm(EMBED_DIM, t_dim=T_DIM)

    def forward(self, x):
        a, _ = self.attn(x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x


class _EuclidBlock(nn.Module):
    """标准欧氏点积注意力块（对照组）"""

    def __init__(self):
        super().__init__()
        self.attn  = nn.MultiheadAttention(EMBED_DIM, N_HEADS,
                                           batch_first=True, bias=False)
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.ffn   = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * 4),
            nn.GELU(),
            nn.Linear(EMBED_DIM * 4, EMBED_DIM),
        )
        self.norm2 = nn.LayerNorm(EMBED_DIM)

    def forward(self, x):
        # self-attention: query, key, value all from x
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x


class Backbone(nn.Module):
    """
    物理预训练骨干网络。

    子模块:
      embed    : STATE_DIM → EMBED_DIM
      blocks   : N_LAYERS 个变换块（Lorentz 或 Euclid）
      phys_dec : EMBED_DIM → STATE_DIM（轨迹预测头）
      lang_aln : EMBED_DIM → LANG_DIM（语言对齐头）
    """

    def __init__(self, block_cls):
        super().__init__()
        self.embed    = nn.Linear(STATE_DIM, EMBED_DIM)
        self.blocks   = nn.ModuleList([block_cls() for _ in range(N_LAYERS)])
        self.phys_dec = nn.Linear(EMBED_DIM, STATE_DIM)
        self.lang_aln = nn.Linear(EMBED_DIM, LANG_DIM)

    def encode_phys(self, x):
        """物理序列编码 → (B, T, EMBED_DIM)"""
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return h

    def forward(self, x):
        """轨迹预测 → (B, T, STATE_DIM)"""
        return self.phys_dec(self.encode_phys(x))

    def lang_align(self, x):
        """物理序列 → 归一化语言嵌入 → (B, LANG_DIM)"""
        h = self.encode_phys(x)
        p = h.mean(dim=1)
        return F.normalize(self.lang_aln(p), dim=-1)


# ── 预训练 ─────────────────────────────────────────────────────

def _pretrain(model, X_pre):
    """物理轨迹预训练（MSE 损失）"""
    model.to(device).train()
    Xt = torch.from_numpy(X_pre).to(device)
    ds = TensorDataset(Xt[:, :T_IN], Xt[:, T_IN:])
    dl = DataLoader(ds, batch_size=BS, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_PRE)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP_PRE)
    last_loss = float('inf')
    for _ in range(EP_PRE):
        for x_in, x_out in dl:
            opt.zero_grad()
            pred = model(x_in)[:, :T_OUT]
            loss = F.mse_loss(pred, x_out)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_loss = loss.item()
        sch.step()
    print(f'  预训练完成，最终 loss={last_loss:.4f}')
    return model


# ── 语言对齐微调 ────────────────────────────────────────────────

def _build_lang_mat():
    """构建语言嵌入矩阵 (N_LABELS, LANG_DIM)"""
    rows = []
    for lbl in range(N_LABELS):
        embs = encode_text(DESCRIPTIONS[lbl])
        rows.append(F.normalize(embs.mean(0), dim=-1))
    return torch.stack(rows).to(device)


def _finetune_lang(model, X_ft, y_ft, lang_mat):
    """语言对齐微调（CLIP 式余弦相似度损失）"""
    model.to(device).train()
    Xt = torch.from_numpy(X_ft).to(device)
    yt = torch.from_numpy(y_ft).to(device)
    ds = TensorDataset(Xt[:, :T_IN], yt)
    dl = DataLoader(ds, batch_size=BS, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.lang_aln.parameters(), lr=LR_FT)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP_FT)
    last_loss = float('inf')
    for _ in range(EP_FT):
        for x_in, y_in in dl:
            opt.zero_grad()
            phys_emb = model.lang_align(x_in)      # (B, LANG_DIM)
            target   = lang_mat[y_in]               # (B, LANG_DIM)
            loss     = -(phys_emb * target).sum(-1).mean()  # 最大化余弦相似度
            loss.backward()
            opt.step()
            last_loss = loss.item()
        sch.step()
    print(f'  微调完成，最终 loss={last_loss:.4f}')
    return model


# ── 评估：方向A对齐得分 ─────────────────────────────────────────

def _eval_alignment(model, X_ft, y_ft, lang_mat):
    """计算每条轨迹的物理嵌入与对应语言描述的余弦相似度"""
    model.eval()
    Xt = torch.from_numpy(X_ft).to(device)
    scores = []
    with torch.no_grad():
        for i in range(len(Xt)):
            phys   = model.lang_align(Xt[i:i + 1, :T_IN])[0]
            target = lang_mat[y_ft[i]]
            scores.append((phys * target).sum().item())
    return np.array(scores)


# ── 测量类时比例（层2） ────────────────────────────────────────

def _measure_timelike(model, X_ft):
    """测量 F3 骨干的类时向量比例"""
    model.eval()
    Xt = torch.from_numpy(X_ft).to(device)
    with torch.no_grad():
        h = model.encode_phys(Xt[:, :T_IN])   # (N, T_IN, EMBED_DIM)
    flat   = h.reshape(-1, EMBED_DIM)
    t_comp = flat[:, :T_DIM]
    s_comp = flat[:, T_DIM:]
    mq     = s_comp.pow(2).sum(-1) - t_comp.pow(2).sum(-1)
    tl_ratio = mq.gt(0).float().mean().item()
    mq_mean  = mq.mean().item()
    return tl_ratio, mq_mean


# ================================================================
# 主程序
# ================================================================
print('\n' + '=' * 60)
print('层1最小验证：语言能否索引洛伦兹物理空间')
print('=' * 60)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# [1] 构建数据集
print('\n[1] 构建数据集...')
X_pre, _      = build_dataset(seed=SEED,         n_per=N_PER)
X_ft,  y_ft   = build_dataset(seed=SEED + 100,   n_per=N_PER)
print(f'  预训练数据: {X_pre.shape}')
print(f'  微调数据:   {X_ft.shape}, 标签分布: {np.bincount(y_ft)}')

# 语言矩阵（只构建一次）
lang_mat = _build_lang_mat()

# [2] 构建模型
print('\n[2] 构建模型...')
model_f3  = Backbone(_LorentzBlock).to(device)
model_euc = Backbone(_EuclidBlock).to(device)

# [3] 物理预训练
print('\n[3] 物理预训练（60 epoch）...')
print('  F3 模型:')
_pretrain(model_f3,  X_pre)
print('  欧氏模型:')
_pretrain(model_euc, X_pre)

# [4] 层2测量（预训练后）
print('\n[4] 层2测量：类时比例（F3 模型预训练后）...')
tl_ratio, mq_mean = _measure_timelike(model_f3, X_ft)
print(f'  F3：类时比例={tl_ratio:.1%}，mq 均值={mq_mean:+.3f}')

# [5] 语言对齐微调
print('\n[5] 语言对齐微调（150 epoch）...')
print('  F3 模型:')
_finetune_lang(model_f3,  X_ft, y_ft, lang_mat)
print('  欧氏模型:')
_finetune_lang(model_euc, X_ft, y_ft, lang_mat)

# [6] 方向A评估
print('\n[6] 方向A评估：物理→语言对齐得分...')
sc_f3  = _eval_alignment(model_f3,  X_ft, y_ft, lang_mat)
sc_euc = _eval_alignment(model_euc, X_ft, y_ft, lang_mat)
print(f'  F3  对齐得分: {sc_f3.mean():.3f} ± {sc_f3.std():.3f}')
print(f'  欧氏对齐得分: {sc_euc.mean():.3f} ± {sc_euc.std():.3f}')
print(f'  差异 (F3 − 欧氏): {sc_f3.mean() - sc_euc.mean():+.3f}')

# [7] 统计检验
from scipy.stats import ttest_rel

t_stat, p_val = ttest_rel(sc_f3, sc_euc)
diff  = sc_f3 - sc_euc
d_eff = diff.mean() / (diff.std() + 1e-8)
print(f'\n  paired t-test: t={t_stat:.3f}, p={p_val:.4f}')
print(f"  Cohen's d: {d_eff:.2f}")

# [8] 逐标签分析
print('\n[8] 逐标签对齐得分...')
for lbl_id, lbl_name in LABELS.items():
    mask = y_ft == lbl_id
    f3_lbl  = sc_f3[mask].mean()
    euc_lbl = sc_euc[mask].mean()
    print(f'  [{lbl_name}] F3={f3_lbl:.3f}  欧氏={euc_lbl:.3f}  '
          f'差异={f3_lbl - euc_lbl:+.3f}')

# [9] 结论
passed = sc_f3.mean() > sc_euc.mean()
print('\n' + '=' * 60)
print('层1验证结论：')
print(f"  方向A: F3 对齐得分 ({sc_f3.mean():.3f}) "
      f"{'>' if passed else '<'} 欧氏 ({sc_euc.mean():.3f})")
print(f'  p={p_val:.4f}  d={d_eff:.2f}')
print(f'  层1{"✓ 成立" if passed else "✗ 未成立"}')
print('=' * 60)
