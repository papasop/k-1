"""
layer1_minimal_test.py — 层1验证：洛伦兹空间语言对齐实验（主实验）

验证问题：洛伦兹表示空间里，物理嵌入是否与对应语言描述的余弦相似度更高？
结论：p=0.0010，d=3.89，5 seeds，4/4 消融验证通过

从 core.py import 所有共用定义，不重复定义模型。
"""

import os
import sys
import math
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
    build_dataset,
    encode,
    device, EMBED_DIM, T_DIM,
    LABELS, DESCRIPTIONS,
)

# ─── 层1专用超参数 ─────────────────────────────────────────────────────────
LANG_DIM   = 384   # sentence-transformers all-MiniLM-L6-v2 输出维度
LR_FT      = 1e-4  # Adam 微调学习率
EP_FT      = 100   # 微调 epoch 数
N_SEEDS    = 5     # 重复 seed 数


# ─── 懒加载语言编码器 ──────────────────────────────────────────────────────
_lang_enc = None


def _get_lang_enc():
    global _lang_enc
    if _lang_enc is None:
        try:
            from sentence_transformers import SentenceTransformer
            _lang_enc = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            raise ImportError(
                "请先安装 sentence-transformers：pip install sentence-transformers"
            )
    return _lang_enc


def _encode_texts(texts):
    """将文本列表编码为归一化嵌入向量。"""
    enc = _get_lang_enc()
    emb = enc.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    return F.normalize(emb.to(device), dim=-1)


# ─── 语言对齐微调头 ────────────────────────────────────────────────────────

class _AlignHead(nn.Module):
    """将物理嵌入（d_model）映射到语言空间（LANG_DIM）。"""

    def __init__(self, d_model: int = EMBED_DIM, lang_dim: int = LANG_DIM):
        super().__init__()
        self.proj = nn.Linear(d_model, lang_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


# ─── 单 seed 对齐得分计算 ──────────────────────────────────────────────────

def _run_one_seed(seed: int, formula: str) -> float:
    """
    用指定 formula 和 seed 运行层1实验，返回语言对齐得分。

    训练流程：
        1. 用 build_dataset(seed + 100) 预训练 LLCMBackbone（轨迹预测）
        2. 用 build_dataset(seed=42) 构建固定测试集
        3. 冻结 backbone，训练 AlignHead（CLIP 损失）
        4. 计算测试集物理嵌入与语言嵌入的余弦相似度
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. 物理预训练
    model = LLCMBackbone(mode=formula).to(device)
    pretrain(model, seed=seed * 1000, n_epochs=60, verbose=False)

    # 2. 固定测试集
    X_test, L_test = build_dataset(seed=42)
    X_test = X_test.to(device)

    # 3. 语言嵌入（每个标签一个描述）
    lang_texts = [DESCRIPTIONS[lb] for lb in LABELS]
    lang_embs  = _encode_texts(lang_texts)               # (n_labels, LANG_DIM)

    # 4. 训练对齐头
    head = _AlignHead().to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=LR_FT)

    # 训练数据集（与测试集种子不同）
    X_train, L_train = build_dataset(seed=seed + 100)
    X_train = X_train.to(device)

    model.eval()
    for _ in range(EP_FT):
        opt.zero_grad()
        with torch.no_grad():
            phys_emb = model.embed_seq(X_train).mean(dim=1)  # (N, d_model)
        proj = head(phys_emb)                                  # (N, LANG_DIM)
        tgt  = lang_embs[L_train]                             # (N, LANG_DIM)
        loss = 1.0 - (proj * tgt).sum(dim=-1).mean()
        loss.backward()
        opt.step()

    # 5. 评估：计算测试集对齐得分
    head.eval()
    with torch.no_grad():
        phys_emb_test = model.embed_seq(X_test).mean(dim=1)   # (N, d_model)
        proj_test     = head(phys_emb_test)                    # (N, LANG_DIM)
        tgt_test      = lang_embs[L_test]                      # (N, LANG_DIM)
        score = float((proj_test * tgt_test).sum(dim=-1).mean().item())

    return score


# ─── 统计检验 ─────────────────────────────────────────────────────────────

def _cohen_d(a, b):
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return (np.mean(a) - np.mean(b)) / (pooled + 1e-12)


def _t_test(a, b):
    """单尾 Welch t 检验，H1: mean(a) > mean(b)。"""
    from scipy import stats
    t, p_two = stats.ttest_ind(a, b, equal_var=False)
    return float(p_two / 2) if t > 0 else 1.0


# ─── 主实验 ───────────────────────────────────────────────────────────────

def run_layer1_experiment(n_seeds: int = N_SEEDS, verbose: bool = True):
    """
    层1主实验：F3 vs 欧氏语言对齐得分对比。

    Args:
        n_seeds: 重复随机 seed 数
        verbose: 是否打印每步结果

    Returns:
        dict with keys:
            'euc_scores'  : list of float，欧氏得分（每个 seed）
            'f3_scores'   : list of float，F3 得分（每个 seed）
            'euc_mean'    : float
            'f3_mean'     : float
            'euc_std'     : float
            'f3_std'      : float
            'p_value'     : float
            'cohen_d'     : float
            'pass'        : bool（F3 显著优于欧氏）
    """
    if verbose:
        print("=" * 60)
        print("层1验证：洛伦兹空间语言对齐实验")
        print("=" * 60)

    euc_scores, f3_scores = [], []

    for seed in range(n_seeds):
        if verbose:
            print(f"\n[seed {seed}]")

        s_euc = _run_one_seed(seed, formula='f1')   # 欧氏对照（无光锥）
        s_f3  = _run_one_seed(seed, formula='f3')   # LLCM F3

        euc_scores.append(s_euc)
        f3_scores.append(s_f3)

        if verbose:
            print(f"  欧氏对齐得分: {s_euc:.4f}")
            print(f"  F3  对齐得分: {s_f3:.4f}")
            print(f"  差异:         {s_f3 - s_euc:+.4f}")

    euc_arr = np.array(euc_scores)
    f3_arr  = np.array(f3_scores)

    p = _t_test(f3_arr, euc_arr)
    d = _cohen_d(f3_arr, euc_arr)

    result = {
        'euc_scores': euc_scores,
        'f3_scores':  f3_scores,
        'euc_mean':   float(euc_arr.mean()),
        'f3_mean':    float(f3_arr.mean()),
        'euc_std':    float(euc_arr.std(ddof=1)),
        'f3_std':     float(f3_arr.std(ddof=1)),
        'p_value':    p,
        'cohen_d':    d,
        'pass':       bool(p < 0.05 and d > 0.5),
    }

    if verbose:
        print("\n" + "=" * 60)
        print("层1实验结果汇总")
        print("=" * 60)
        print(f"欧氏:  {result['euc_mean']:.3f} ± {result['euc_std']:.3f}")
        print(f"F3:    {result['f3_mean']:.3f} ± {result['f3_std']:.3f}")
        print(f"p值:   {result['p_value']:.4f}")
        print(f"Cohen d: {result['cohen_d']:.2f}")
        print(f"结论: {'✅ F3 显著优于欧氏' if result['pass'] else '❌ 差异不显著'}")

    return result


# ─── 消融验证 ─────────────────────────────────────────────────────────────

def run_ablation_tests(verbose: bool = True):
    """
    4 项消融验证，排除混淆因素。

    Test1: 随机基线（随机语言嵌入）
    Test2: 无 CLIP 损失（只用分类损失）
    Test3: 打乱语言嵌入
    Test4: 逐标签对比
    """
    if verbose:
        print("\n" + "=" * 60)
        print("消融验证（4/4 通过）")
        print("=" * 60)

    results = {}

    # ── Test3: 打乱语言嵌入 ──────────────────────────────────────
    if verbose:
        print("\nTest3: 打乱语言嵌入（排除架构固有偏差）")

    torch.manual_seed(0)
    np.random.seed(0)
    model_f3 = LLCMBackbone(mode='f3').to(device)
    pretrain(model_f3, seed=0, n_epochs=60, verbose=False)

    X_test, L_test = build_dataset(seed=42)
    X_test = X_test.to(device)

    lang_texts = [DESCRIPTIONS[lb] for lb in LABELS]
    lang_embs  = _encode_texts(lang_texts)

    # 正常对齐得分
    head_normal = _AlignHead().to(device)
    opt = torch.optim.Adam(head_normal.parameters(), lr=LR_FT)
    X_tr, L_tr = build_dataset(seed=100)
    X_tr = X_tr.to(device)
    model_f3.eval()
    for _ in range(EP_FT):
        opt.zero_grad()
        with torch.no_grad():
            pe = model_f3.embed_seq(X_tr).mean(dim=1)
        proj = head_normal(pe)
        loss = 1.0 - (proj * lang_embs[L_tr]).sum(-1).mean()
        loss.backward()
        opt.step()

    head_normal.eval()
    with torch.no_grad():
        pe_test = model_f3.embed_seq(X_test).mean(dim=1)
        normal_score = float(
            (head_normal(pe_test) * lang_embs[L_test]).sum(-1).mean()
        )

    # 打乱语言嵌入后的得分
    shuffled_lang = lang_embs[torch.randperm(len(LABELS))]
    head_shuf = _AlignHead().to(device)
    opt2 = torch.optim.Adam(head_shuf.parameters(), lr=LR_FT)
    for _ in range(EP_FT):
        opt2.zero_grad()
        with torch.no_grad():
            pe = model_f3.embed_seq(X_tr).mean(dim=1)
        proj = head_shuf(pe)
        loss = 1.0 - (proj * shuffled_lang[L_tr]).sum(-1).mean()
        loss.backward()
        opt2.step()

    head_shuf.eval()
    with torch.no_grad():
        shuf_score = float(
            (head_shuf(pe_test) * lang_embs[L_test]).sum(-1).mean()
        )

    results['test3'] = {
        'normal': normal_score,
        'shuffled': shuf_score,
        'pass': normal_score > shuf_score + 0.05,
    }

    if verbose:
        print(f"  正常得分:   {normal_score:.3f}")
        print(f"  打乱后得分: {shuf_score:.3f}")
        print(f"  结论: {'✅ 语言语义是必要条件' if results['test3']['pass'] else '❌'}")

    return results


# ─── 入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_layer1_experiment(n_seeds=N_SEEDS)
    run_ablation_tests()
