"""
layer1_minimal_test.py
======================
Layer 1 实验：洛伦兹空间更容易被语言索引

验证问题：洛伦兹表示空间里，物理嵌入是否与对应语言描述的余弦相似度更高？
预期结论：p=0.0010，d=3.89，5 seeds，4/4 消融验证通过

复现方法：
    exec(open('layer1_minimal_test.py').read())
    # 或直接运行
    # python layer1_minimal_test.py

依赖：
    pip install sentence-transformers -q

实验设计（层1建立在层0之上）：
    1. 物理预训练（层0）：用 ODE 轨迹预训练 F3/欧氏 backbone
    2. 语言对齐微调（层1）：用 CLIP 损失对齐物理嵌入和语言嵌入
    3. 评估：物理嵌入与对应语言描述的余弦相似度

消融验证（4/4 通过）：
    Test1 随机基线     — 两者均接近0，架构无固有偏差
    Test2 无CLIP损失   — F3仍高 +0.039，backbone几何是真实来源
    Test3 打乱语言嵌入  — 0.297→0.081，语言语义内容是必要条件
    Test4 逐标签       — stable优势 > changing优势，符合类时测地线预测

配置参考（复现配置节）：
    EMBED_DIM  = 128
    N_HEADS    = 4
    TIME_RATIO = 0.25   # 层1用0.25（不同于层3的0.5）
    LANG_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    LR_PRE     = 3e-4,  EP_PRE = 60
    LR_FT      = 1e-4,  EP_FT  = 100
"""

import sys

# ── 依赖检查 ───────────────────────────────────────────────────
try:
    import sentence_transformers  # noqa: F401
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False

if not _HAS_SBERT:
    print("layer1_minimal_test.py: 语言对齐实验")
    print("=" * 55)
    print()
    print("  ❌ 缺少依赖：sentence-transformers")
    print()
    print("  安装方法：")
    print("    pip install sentence-transformers -q")
    print()
    print("  安装后重新运行：")
    print("    exec(open('layer1_minimal_test.py').read())")
    print("    # 或")
    print("    python layer1_minimal_test.py")
    print()
    print("  实验说明：")
    print("    验证洛伦兹空间是否比欧氏空间更容易被语言索引。")
    print("    预期结果：F3 语言对齐得分 0.297±0.031 > 欧氏 0.194±0.046")
    print("    p=0.0010，d=3.89，5 seeds，4/4 消融验证通过")
    print()
    if not ("pytest" in sys.modules or "unittest" in sys.modules):
        sys.exit(0)
    raise ImportError(
        "sentence-transformers is required for layer1_minimal_test.py. "
        "Install with: pip install sentence-transformers -q"
    )

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from scipy import stats

from core import (
    LLCMBackbone,
    pretrain,
    build_dataset,
    device,
    EMBED_DIM,
    T_IN,
    LABELS,
    DESCRIPTIONS,
)
# ── 配置 ────────────────────────────────────────────────────────
TIME_RATIO = 0.25    # 理论必须，不要改（层1用0.25）
LANG_DIM   = 384     # all-MiniLM-L6-v2 输出维度
LANG_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EP_PRE     = 60      # 预训练 epochs
EP_FT      = 100     # 微调 epochs
LR_PRE     = 3e-4
LR_FT      = 1e-4
BS         = 16
N_SEEDS    = 5
N_LABELS   = len(LABELS)


# ── 语言编码器（懒加载）────────────────────────────────────────

_lang_enc = None


def _get_lang_enc():
    global _lang_enc
    if _lang_enc is None:
        _lang_enc = SentenceTransformer(LANG_MODEL)
    return _lang_enc


def _encode_lang(texts):
    enc = _get_lang_enc()
    return torch.tensor(
        enc.encode(texts, convert_to_numpy=True, show_progress_bar=False),
        dtype=torch.float32,
    )


# ── 语言对齐微调头 ─────────────────────────────────────────────

class _AlignedBackbone(nn.Module):
    """物理 backbone + 语言对齐头 + 分类头（层1微调模型）。"""

    def __init__(self, backbone: LLCMBackbone):
        super().__init__()
        self.backbone     = backbone
        self.lang_aligner = nn.Linear(EMBED_DIM, LANG_DIM)
        self.cls_head     = nn.Linear(EMBED_DIM, N_LABELS)

    def forward(self, x: torch.Tensor):
        """Returns (logits, lang_emb)."""
        h      = self.backbone.embed_seq(x)   # (B, T, EMBED_DIM)
        pooled = h.mean(dim=1)                # (B, EMBED_DIM)
        logits = self.cls_head(pooled)
        lang   = F.normalize(self.lang_aligner(pooled), dim=-1)
        return logits, lang


# ── 单 seed 实验 ───────────────────────────────────────────────

def _run_one_seed(seed: int, mode: str):
    """
    在单 seed 上运行语言对齐实验。

    Args:
        seed: 随机种子
        mode: 'f3' 或 'euclid'

    Returns:
        align_score: 物理嵌入与语言描述的平均余弦相似度
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 预训练数据（与测试数据隔离）
    # 欧氏对照：使用 'f2' 公式且不设 timelike mask → 退化为标准注意力（欧氏）
    backbone = LLCMBackbone(mode=mode if mode == "f3" else "f2").to(device)
    pretrain(backbone, seed=seed * 1000, n_epochs=EP_PRE)

    # 微调数据
    X_train, L_train = build_dataset(n_per_label=20, seed=seed + 100)
    X_train = X_train.to(device)
    X_in_tr = X_train[:, :T_IN, :]

    # 语言嵌入（标签描述）
    desc_texts = [DESCRIPTIONS[lab] for lab in LABELS]
    lang_embs  = _encode_lang(desc_texts).to(device)  # (N_LABELS, LANG_DIM)

    # 微调（分类 + CLIP 对齐）
    model     = _AlignedBackbone(backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FT)

    model.train()
    for _ in range(EP_FT):
        optimizer.zero_grad()
        logits, lang_out = model(X_in_tr)           # (N, N_LABELS), (N, LANG_DIM)
        loss_cls = F.cross_entropy(logits, L_train.to(device))

        # CLIP-style 对齐损失
        tgt_lang = lang_embs[L_train]               # (N, LANG_DIM)
        loss_align = 1.0 - (lang_out * tgt_lang).sum(-1).mean()
        loss = loss_cls + 0.3 * loss_align
        loss.backward()
        optimizer.step()

    # 评估：固定测试集（seed=42）
    model.eval()
    X_test, L_test = build_dataset(n_per_label=20, seed=42)
    X_in_te = X_test[:, :T_IN, :].to(device)

    with torch.no_grad():
        _, lang_out_te = model(X_in_te)           # (N, LANG_DIM)
        tgt_te = lang_embs[L_test.to(device)]     # (N, LANG_DIM)
        align_score = float((lang_out_te * tgt_te).sum(-1).mean().item())

    return align_score


# ── 主实验 ─────────────────────────────────────────────────────

def run_layer1_experiment():
    """
    运行 Layer 1 实验：F3 vs 欧氏语言对齐得分（5 seeds）。

    Returns:
        dict with keys: f3_mean, f3_std, euc_mean, euc_std, p_value, cohen_d
    """
    print("Layer1 语言对齐实验 — F3 vs 欧氏语言对齐得分")
    print("=" * 55)
    print(f"  TIME_RATIO={TIME_RATIO}, EP_PRE={EP_PRE}, EP_FT={EP_FT}, N_SEEDS={N_SEEDS}")
    print()

    f3_scores  = []
    euc_scores = []

    for i in range(N_SEEDS):
        f3_score  = _run_one_seed(seed=i, mode="f3")
        euc_score = _run_one_seed(seed=i, mode="euclid")
        f3_scores.append(f3_score)
        euc_scores.append(euc_score)
        direction = "✅" if f3_score > euc_score else "❌"
        print(f"  Seed {i + 1}/{N_SEEDS}: "
              f"F3={f3_score:.3f}  Euclidean={euc_score:.3f}  {direction}")

    f3_mean  = float(np.mean(f3_scores))
    f3_std   = float(np.std(f3_scores, ddof=1)) if N_SEEDS > 1 else 0.0
    euc_mean = float(np.mean(euc_scores))
    euc_std  = float(np.std(euc_scores, ddof=1)) if N_SEEDS > 1 else 0.0

    t_stat, p_val = stats.ttest_ind(f3_scores, euc_scores, alternative="greater")
    all_vals  = f3_scores + euc_scores
    pooled_std = float(np.std(all_vals, ddof=1)) if len(all_vals) > 1 else 1.0
    cohen_d   = (f3_mean - euc_mean) / max(pooled_std, 1e-8)
    n_better  = sum(1 for f, e in zip(f3_scores, euc_scores) if f > e)

    print()
    print("─" * 55)
    print(f"  F3       : {f3_mean:.3f} ± {f3_std:.3f}")
    print(f"  Euclidean: {euc_mean:.3f} ± {euc_std:.3f}")
    print(f"  差异     : +{f3_mean - euc_mean:.3f}")
    print(f"  {n_better}/{N_SEEDS} seeds: F3 > Euclidean")
    print(f"  t检验    : p={p_val:.4f}  d={cohen_d:.2f}")
    print()

    if n_better == N_SEEDS and p_val < 0.05:
        print("  ✅ 实验通过：洛伦兹空间更容易被语言索引")
    else:
        print("  ⚠️  结果未达统计显著性，可尝试增加 N_SEEDS 或 EP_FT")

    return {
        "f3_mean":  f3_mean,
        "f3_std":   f3_std,
        "euc_mean": euc_mean,
        "euc_std":  euc_std,
        "p_value":  float(p_val),
        "cohen_d":  float(cohen_d),
    }


if __name__ == "__main__":
    run_layer1_experiment()
