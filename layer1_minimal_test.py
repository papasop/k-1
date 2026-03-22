"""
layer1_minimal_test.py — 层1最小验证：语言能否索引洛伦兹物理空间

最小设计：
  - 50个样本/类，2类（stable vs changing）
  - 预训练 60 epoch
  - 微调 150 epoch
  - 1个seed，快速出结果

验证问题：
  方向A: 物理轨迹 → 洛伦兹空间 → 语言描述嵌入对齐得分
         F3对齐得分 > 欧氏 → 洛伦兹空间更容易被语言索引 → 层1成立

使用：
    python layer1_minimal_test.py
    # 或
    exec(open('layer1_minimal_test.py').read())
"""

import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# ── 路径设置（从任意目录运行时均能找到 llcm）──────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from llcm.core import (
    LLCMBackbone,
    pretrain,
    build_dataset,
    encode,
    device,
    EMBED_DIM,
    T_IN,
    LANG_DIM,
    N_PER,
    EP_PRE,
    LABELS,
    DESCRIPTIONS,
)

# ── 层1专用超参数 ─────────────────────────────────────────────────────
EP_FT   = 150     # 微调 epoch 数（方向A对齐头）
LR_FT   = 1e-4    # 微调学习率
N_SEEDS = 1       # 最小版：1个seed，快速出结果


# ── 语言编码辅助函数 ──────────────────────────────────────────────────

def _encode_texts(text_lists):
    """
    对每个标签的多个描述进行编码并平均。

    Args:
        text_lists : list[list[str]]，每个元素是一个标签的描述列表

    Returns:
        embs : (n_labels, LANG_DIM) 语言嵌入张量，已 L2 归一化
    """
    result = []
    for texts in text_lists:
        e = encode(texts)          # (n_texts, LANG_DIM)
        result.append(e.mean(0))   # 多描述平均
    embs = torch.stack(result)     # (n_labels, LANG_DIM)
    return F.normalize(embs, dim=-1)


# ── 对齐头 ────────────────────────────────────────────────────────────

class _AlignHead(nn.Module):
    """将物理嵌入（EMBED_DIM）映射到语言空间（LANG_DIM）并 L2 归一化。"""

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.GELU(),
            nn.Linear(EMBED_DIM, LANG_DIM),
        )

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


# ── 单 seed 对齐得分计算 ──────────────────────────────────────────────

def _run_one_seed(seed: int, formula: str) -> float:
    """
    用指定 formula 和 seed 运行层1实验，返回语言对齐得分。

    训练流程：
        1. 用 build_dataset(seed + 100) 预训练 LLCMBackbone（轨迹预测）
        2. 用 build_dataset(seed=42) 构建固定测试集
        3. 冻结 backbone，训练 AlignHead（CLIP 损失）
        4. 计算测试集物理嵌入与语言嵌入的余弦相似度

    Args:
        seed   : 随机种子
        formula: 注意力公式 'f1'（欧氏对照）或 'f3'（洛伦兹 LLCM）

    Returns:
        align_score : 平均语言对齐余弦相似度（越高越好）
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. 物理预训练（用不同种子隔离预训练数据）
    model = LLCMBackbone(formula=formula).to(device)
    pretrain(model, seed=seed + 100, epochs=EP_PRE)

    # 2. 固定测试集（种子固定，保证两种公式在同一测试集上对比）
    X_np, y_np = build_dataset(seed=42, n_per=N_PER)
    X_test = torch.from_numpy(X_np[:, :T_IN]).to(device)
    y_test = torch.from_numpy(y_np).to(device)

    # 3. 语言嵌入（每个标签多个描述取均值后归一化）
    label_keys = sorted(LABELS.keys())
    lang_texts = [DESCRIPTIONS[k] for k in label_keys]
    lang_embs  = _encode_texts(lang_texts)              # (N_LABELS, LANG_DIM)

    # 4. 训练数据集（种子与测试集不同，避免数据泄露）
    X_tr_np, y_tr_np = build_dataset(seed=seed + 200, n_per=N_PER)
    X_train = torch.from_numpy(X_tr_np[:, :T_IN]).to(device)
    y_train = torch.from_numpy(y_tr_np).to(device)

    # 5. 训练对齐头（冻结 backbone）
    model.eval()
    head = _AlignHead().to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=LR_FT)

    for _ in range(EP_FT):
        opt.zero_grad()
        with torch.no_grad():
            phys_emb = model.encode_phys(X_train).mean(dim=1)  # (N, EMBED_DIM)
        proj = head(phys_emb)                                   # (N, LANG_DIM)
        # CLIP 损失：最大化物理嵌入与正确类别语言嵌入的余弦相似度
        sim  = proj @ lang_embs.T                               # (N, N_LABELS)
        loss = F.cross_entropy(sim * 20.0, y_train)
        loss.backward()
        opt.step()

    # 6. 评估对齐得分（测试集）
    head.eval()
    with torch.no_grad():
        phys_test = model.encode_phys(X_test).mean(dim=1)      # (N, EMBED_DIM)
        proj_test = head(phys_test)                             # (N, LANG_DIM)
        tgt_test  = lang_embs[y_test]                          # (N, LANG_DIM)
        # 已归一化，点积即余弦相似度
        scores    = (proj_test * tgt_test).sum(dim=-1)         # (N,)

    return float(scores.mean().item())


# ── 统计检验 ─────────────────────────────────────────────────────────

def _cohen_d(a, b):
    """Cohen's d 效应量（正值表示 a > b），使用样本方差（ddof=1）。"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2 + 1e-9)
    return float((a.mean() - b.mean()) / pooled_std)


def _t_test(a, b):
    """单尾 Welch t 检验，H1: mean(a) > mean(b)。"""
    from scipy import stats
    t, p_two = stats.ttest_ind(np.asarray(a), np.asarray(b), equal_var=False)
    p_one = p_two / 2 if t > 0 else 1.0 - p_two / 2
    return float(t), float(p_one)


# ── 主实验 ───────────────────────────────────────────────────────────

def run_layer1_experiment(n_seeds: int = N_SEEDS, verbose: bool = True):
    """
    层1验证：洛伦兹空间语言对齐实验。

    对每个 seed 分别运行欧氏（f1）和洛伦兹（f3）两种公式，
    对比语言对齐得分，检验 F3 > 欧氏 是否成立。

    Args:
        n_seeds : 运行的 seed 数量（默认 1，最小验证）
        verbose : 是否打印详细结果

    Returns:
        dict with keys:
            'euc_scores' : list[float] — 欧氏对照得分（formula='f1'）
            'f3_scores'  : list[float] — F3 洛伦兹得分（formula='f3'）
            'euc_mean'   : float
            'f3_mean'    : float
            'advantage'  : float — F3 均值 - 欧氏均值
            'pass'       : bool  — F3 > 欧氏 是否成立
    """
    if verbose:
        print("=" * 60)
        print("层1验证：洛伦兹空间语言对齐实验")
        print("=" * 60)

    euc_scores, f3_scores = [], []

    for seed in range(n_seeds):
        if verbose:
            print(f"\n[seed {seed}]")

        s_euc = _run_one_seed(seed, formula='f1')   # 欧氏对照（无光锥约束）
        s_f3  = _run_one_seed(seed, formula='f3')   # LLCM F3（洛伦兹几何）

        euc_scores.append(s_euc)
        f3_scores.append(s_f3)

        if verbose:
            print(f"  欧氏对齐得分 (f1): {s_euc:.4f}")
            print(f"  F3  对齐得分 (f3): {s_f3:.4f}")
            print(f"  差值 (f3-euc):     {s_f3 - s_euc:+.4f}")

    euc_mean = float(np.mean(euc_scores))
    f3_mean  = float(np.mean(f3_scores))

    result = {
        'euc_scores': euc_scores,
        'f3_scores' : f3_scores,
        'euc_mean'  : euc_mean,
        'f3_mean'   : f3_mean,
        'advantage' : f3_mean - euc_mean,
        'pass'      : f3_mean > euc_mean,
    }

    if verbose:
        print("\n" + "-" * 60)
        print(f"欧氏均值 : {euc_mean:.4f}")
        print(f"F3  均值 : {f3_mean:.4f}")
        print(f"F3 优势  : {result['advantage']:+.4f}")
        if n_seeds >= 2:
            t, p = _t_test(f3_scores, euc_scores)
            d    = _cohen_d(f3_scores, euc_scores)
            print(f"t={t:.3f}  p={p:.4f}  d={d:.2f}")
        verdict = ("✓ 层1成立：洛伦兹空间更容易被语言索引"
                   if result['pass'] else "✗ 层1未通过")
        print(verdict)
        print("=" * 60)

    return result


# ── 消融验证 ─────────────────────────────────────────────────────────

def run_ablation_tests(verbose: bool = True):
    """
    4 项消融验证，排除混淆因素。

    Test1: 随机基线（随机对齐头权重，不训练）
    Test2: 无 CLIP 损失（用分类损失替代）
    Test3: 打乱语言嵌入（验证语义内容是必要条件）
    Test4: 逐标签对比（stable 和 changing 分别的优势）

    Returns:
        dict with keys:
            'test1_random_baseline' : float
            'test2_no_clip_loss'    : float
            'test3_shuffled_lang'   : float
            'test4_per_label'       : dict[str, float]
    """
    if verbose:
        print("\n" + "=" * 60)
        print("消融验证")
        print("=" * 60)

    results = {}

    # ── 共享：预训练 F3 backbone + 固定测试集 ───────────────────────
    torch.manual_seed(0)
    np.random.seed(0)

    model_f3 = LLCMBackbone(formula='f3').to(device)
    pretrain(model_f3, seed=100, epochs=EP_PRE)
    model_f3.eval()

    X_np, y_np = build_dataset(seed=42, n_per=N_PER)
    X_test = torch.from_numpy(X_np[:, :T_IN]).to(device)
    y_test = torch.from_numpy(y_np).to(device)

    label_keys = sorted(LABELS.keys())
    lang_texts = [DESCRIPTIONS[k] for k in label_keys]
    lang_embs  = _encode_texts(lang_texts)              # (N_LABELS, LANG_DIM)

    with torch.no_grad():
        phys_test = model_f3.encode_phys(X_test).mean(dim=1)  # (N, EMBED_DIM)

    # ── 训练数据（消融共用）──────────────────────────────────────────
    X_tr_np, y_tr_np = build_dataset(seed=200, n_per=N_PER)
    X_train = torch.from_numpy(X_tr_np[:, :T_IN]).to(device)
    y_train = torch.from_numpy(y_tr_np).to(device)

    # ── Test1: 随机基线 ──────────────────────────────────────────────
    if verbose:
        print("\nTest1: 随机基线（不训练对齐头）")

    head_rand = _AlignHead().to(device)   # 随机初始化，不训练
    head_rand.eval()
    with torch.no_grad():
        proj_rand  = head_rand(phys_test)
        tgt_rand   = lang_embs[y_test]
        score_rand = float((proj_rand * tgt_rand).sum(dim=-1).mean().item())

    results['test1_random_baseline'] = score_rand
    if verbose:
        print(f"  随机基线得分: {score_rand:.4f}  (期望接近 0)")

    # ── Test2: 无 CLIP 损失（用分类损失替代）────────────────────────
    if verbose:
        print("\nTest2: 无 CLIP 损失（用分类损失替代）")

    head_cls     = _AlignHead().to(device)
    cls_layer    = nn.Linear(LANG_DIM, len(LABELS)).to(device)
    opt_cls      = torch.optim.Adam(
        list(head_cls.parameters()) + list(cls_layer.parameters()), lr=LR_FT
    )

    head_cls.train()
    for _ in range(EP_FT):
        opt_cls.zero_grad()
        with torch.no_grad():
            phys_emb = model_f3.encode_phys(X_train).mean(dim=1)
        proj   = head_cls(phys_emb)
        logits = cls_layer(proj)
        loss   = F.cross_entropy(logits, y_train)
        loss.backward()
        opt_cls.step()

    head_cls.eval()
    with torch.no_grad():
        proj_cls  = head_cls(phys_test)
        tgt_cls   = lang_embs[y_test]
        score_cls = float((proj_cls * tgt_cls).sum(dim=-1).mean().item())

    results['test2_no_clip_loss'] = score_cls
    if verbose:
        print(f"  无CLIP损失得分: {score_cls:.4f}  (期望低于有CLIP损失)")

    # ── Test3: 打乱语言嵌入 ─────────────────────────────────────────
    if verbose:
        print("\nTest3: 打乱语言嵌入（排除架构固有偏差）")

    shuffled_lang = lang_embs[torch.randperm(len(lang_embs))]
    head_shuf     = _AlignHead().to(device)
    opt_shuf      = torch.optim.Adam(head_shuf.parameters(), lr=LR_FT)

    head_shuf.train()
    for _ in range(EP_FT):
        opt_shuf.zero_grad()
        with torch.no_grad():
            phys_emb = model_f3.encode_phys(X_train).mean(dim=1)
        proj  = head_shuf(phys_emb)
        sim   = proj @ shuffled_lang.T
        loss  = F.cross_entropy(sim * 20.0, y_train)
        loss.backward()
        opt_shuf.step()

    head_shuf.eval()
    with torch.no_grad():
        proj_shuf  = head_shuf(phys_test)
        # 评估时也用打乱的语言嵌入（与训练目标一致）
        tgt_shuf   = shuffled_lang[y_test]
        score_shuf = float((proj_shuf * tgt_shuf).sum(dim=-1).mean().item())

    results['test3_shuffled_lang'] = score_shuf
    if verbose:
        print(f"  打乱语言嵌入得分: {score_shuf:.4f}  (期望低于正确语言嵌入)")

    # ── Test4: 逐标签对比 ───────────────────────────────────────────
    if verbose:
        print("\nTest4: 逐标签对比（stable vs changing）")

    head_per = _AlignHead().to(device)
    opt_per  = torch.optim.Adam(head_per.parameters(), lr=LR_FT)

    head_per.train()
    for _ in range(EP_FT):
        opt_per.zero_grad()
        with torch.no_grad():
            phys_emb = model_f3.encode_phys(X_train).mean(dim=1)
        proj  = head_per(phys_emb)
        sim   = proj @ lang_embs.T
        loss  = F.cross_entropy(sim * 20.0, y_train)
        loss.backward()
        opt_per.step()

    head_per.eval()
    per_label_scores = {}
    with torch.no_grad():
        proj_per = head_per(phys_test)

    for lk in label_keys:
        lk_idx = label_keys.index(lk)       # position in lang_embs (0-based)
        mask = (y_test == lk)
        if mask.sum() == 0:
            continue
        tgt_lk = lang_embs[lk_idx].unsqueeze(0).expand(mask.sum(), -1)
        s_lk   = float((proj_per[mask] * tgt_lk).sum(dim=-1).mean().item())
        per_label_scores[LABELS[lk]] = s_lk
        if verbose:
            print(f"  {LABELS[lk]}: {s_lk:.4f}")

    results['test4_per_label'] = per_label_scores

    if verbose:
        print("\n" + "-" * 60)
        print("消融验证完成")
        print("=" * 60)

    return results


# ── 主入口 ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    result = run_layer1_experiment(n_seeds=N_SEEDS)

    if result['pass']:
        print("\n结论：层1成立 — 洛伦兹空间（F3）比欧氏空间（F1）更容易被语言索引。")
    else:
        print("\n结论：层1未通过。")

    run_ablation_tests()
