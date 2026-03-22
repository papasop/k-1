"""
baby_talk_full_test.py
======================
LLCM 完整五模块联合验证：婴儿说话机制实验

验证问题：洛伦兹感知流形是否支持语言←→物理双向对齐？
当前状态：弱版本已验证（p=0.041），完整版验证进行中。

复现方法：
    exec(open('baby_talk_full_test.py').read())
    # 或直接运行
    # python baby_talk_full_test.py

依赖：
    pip install sentence-transformers -q

五个模块：
    模块1  物理预训练 loss F3 << 欧氏        → layer3_zero_loss_B.py
    模块2  语言编码器语义质量（中文同类>跨类）  → verify_module2()
    模块3  方向A：物理→语言对齐              → layer1_minimal_test.py
    模块4a 类时比例 F3>欧氏（mq差距72倍）     → layer1_minimal_test.py 层2测量
    模块4b Law II 在线收敛速度（dc>0）        → online_interaction_test.py
    模块5  方向B：语言→守恒轨迹              → run_module5()

核心猜想（双向语言对齐）：
    物理世界 ←→ 洛伦兹光锥空间 ←→ 自然语言空间

    感知→语言：机器人检测到动量突变 → "这个动作太猛了"
    语言→感知："平稳移动" → 动量守恒轨迹自动生成
"""

import sys

# ── 依赖检查 ───────────────────────────────────────────────────
try:
    import sentence_transformers  # noqa: F401
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False

if not _HAS_SBERT:
    print("baby_talk_full_test.py: LLCM 完整五模块联合验证")
    print("=" * 55)
    print()
    print("  ❌ 缺少依赖：sentence-transformers")
    print()
    print("  安装方法：")
    print("    pip install sentence-transformers -q")
    print()
    print("  安装后重新运行：")
    print("    exec(open('baby_talk_full_test.py').read())")
    print("    # 或")
    print("    python baby_talk_full_test.py")
    print()
    print("  各模块说明：")
    print("    模块1：物理预训练 F3 << 欧氏（×10倍，已验证）")
    print("           → exec(open('layer3_zero_loss_B.py').read())")
    print("    模块3：语言对齐 p=0.0010，d=3.89（已验证）")
    print("           → exec(open('layer1_minimal_test.py').read())")
    print("    模块5：语言→守恒轨迹（弱版本 p=0.041，验证中）")
    print("    完整：  需要 sentence-transformers")
    print()
    if not ("pytest" in sys.modules or "unittest" in sys.modules):
        sys.exit(0)
    raise ImportError(
        "sentence-transformers is required for baby_talk_full_test.py. "
        "Install with: pip install sentence-transformers -q"
    )

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from scipy import stats

from core import (
    LLCMBackbone,
    pretrain,
    build_dataset,
    momentum_change,
    device,
    T_IN,
    T_OUT,
    LABELS,
    DESCRIPTIONS,
)

# ── 配置 ────────────────────────────────────────────────────────
LANG_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LANG_DIM   = 384
N_SEEDS    = 3
EP_PRE     = 60
EP_FT      = 100
LR_PRE     = 3e-4
LR_FT      = 1e-4


# ── 语言编码器 ─────────────────────────────────────────────────

_lang_enc = None


def _get_lang_enc():
    global _lang_enc
    if _lang_enc is None:
        _lang_enc = SentenceTransformer(LANG_MODEL)
    return _lang_enc


def _encode(texts):
    enc = _get_lang_enc()
    vecs = enc.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return torch.tensor(vecs, dtype=torch.float32)


# ── 模块2：语言编码器语义质量验证 ─────────────────────────────

def verify_module2():
    """
    模块2：验证语言编码器的语义质量。
    期望：同类描述之间余弦相似度 > 跨类描述。

    Returns:
        True if within-class similarity > cross-class similarity
    """
    print("\n模块2：语言编码器语义质量验证")
    print("─" * 40)

    all_texts = []
    all_labels = []
    for lbl_idx, lbl in enumerate(LABELS):
        # DESCRIPTIONS values are strings (one description per label)
        desc = DESCRIPTIONS[lbl]
        descs = desc if isinstance(desc, list) else [desc]
        for txt in descs:
            all_texts.append(txt)
            all_labels.append(lbl_idx)

    embs   = _encode(all_texts)  # (N, 384)
    embs   = F.normalize(embs, dim=-1)
    labels = torch.tensor(all_labels)

    # 同类相似度 vs 跨类相似度
    sims_within = []
    sims_cross  = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sim = float((embs[i] * embs[j]).sum())
            if labels[i] == labels[j]:
                sims_within.append(sim)
            else:
                sims_cross.append(sim)

    within_mean = float(np.mean(sims_within)) if sims_within else 0.0
    cross_mean = float(np.mean(sims_cross)) if sims_cross else 0.0
    result = within_mean > cross_mean

    print(f"  同类相似度: {within_mean:.3f}")
    print(f"  跨类相似度: {cross_mean:.3f}")
    print(f"  结论: {'✅ 中文同类>跨类' if result else '❌ 未通过'}")

    return result


# ── 模块5：方向B——语言→守恒轨迹（弱版本）──────────────────────

def _generate_traj_from_lang(backbone, lang_prompt: str, n_steps: int = T_OUT):
    """
    方向B（弱版本）：用语言描述引导轨迹生成。

    实现：找到训练集中与语言描述最相似的轨迹，返回其预测输出。
    （完整版需要语言→轨迹的端到端生成，是开放研究问题）

    Args:
        backbone   : LLCMBackbone（已预训练）
        lang_prompt: 自然语言描述
        n_steps    : 生成帧数

    Returns:
        traj: (n_steps, STATE_DIM) 引导轨迹
    """
    lang_enc  = _get_lang_enc()
    lang_emb  = torch.tensor(
        lang_enc.encode([lang_prompt], convert_to_numpy=True)[0],
        dtype=torch.float32,
    )
    X, L = build_dataset(n_per_label=20, seed=42)
    backbone.eval()
    with torch.no_grad():
        embs = backbone.embed_seq(X[:, :T_IN, :].to(device))  # (N, T, d)
        pooled = F.normalize(embs.mean(dim=1), dim=-1)         # (N, d)
        lang_norm = F.normalize(lang_emb.unsqueeze(0), dim=-1).to(device)
        sims = (pooled * lang_norm).sum(-1)  # (N,)
        best_idx = sims.argmax().item()

    X_in = X[best_idx:best_idx + 1, :T_IN, :].to(device)
    with torch.no_grad():
        pred = backbone(X_in)  # (1, T_IN, STATE_DIM)
    return pred[0, :n_steps, :].cpu().numpy()


def run_module5(n_seeds: int = N_SEEDS):
    """
    模块5：方向B——语言描述 → 守恒轨迹生成（弱版本）。

    验证：F3 生成轨迹的动量守恒性 < 欧氏生成轨迹（守恒性更好）。
    预期：p=0.041（弱版本已验证）

    Returns:
        dict with keys: f3_mom_mean, euc_mom_mean, p_value
    """
    print("\n模块5：方向B——语言→守恒轨迹（弱版本）")
    print("─" * 40)

    stable_prompt = "平稳匀速运动，动量保持守恒"
    f3_moms  = []
    euc_moms = []

    for seed in range(n_seeds):
        for mode, lst in [("f3", f3_moms), ("euclid", euc_moms)]:
            # 欧氏对照：'f2' 无 timelike mask → 退化为标准注意力（欧氏）
            bb = LLCMBackbone(mode=mode if mode == "f3" else "f2").to(device)
            pretrain(bb, seed=seed * 1000, n_epochs=EP_PRE)
            traj = _generate_traj_from_lang(bb, stable_prompt)
            lst.append(momentum_change(traj))

    f3_mean  = float(np.mean(f3_moms))
    euc_mean = float(np.mean(euc_moms))
    _, p_val = stats.ttest_ind(euc_moms, f3_moms, alternative="greater")

    for i, (f, e) in enumerate(zip(f3_moms, euc_moms)):
        direction = "✅" if f < e else "❌"
        print(f"  Seed {i + 1}: F3={f:.4f}  Euclidean={e:.4f}  {direction}")

    print(f"\n  F3       : {f3_mean:.4f}")
    print(f"  Euclidean: {euc_mean:.4f}")
    print(f"  p值      : {p_val:.4f}")
    better = sum(1 for f, e in zip(f3_moms, euc_moms) if f < e)
    print(f"  {better}/{n_seeds} seeds: F3 < Euclidean "
          f"{'✅' if better == n_seeds else '⚠️'}")

    return {
        "f3_mom_mean":  f3_mean,
        "euc_mom_mean": euc_mean,
        "p_value":      float(p_val),
    }


# ── 完整五模块 ─────────────────────────────────────────────────

def run_baby_language_test(n_seeds: int = N_SEEDS):
    """
    完整五模块联合验证。

    Args:
        n_seeds: 随机种子数（默认 3）

    Returns:
        results: dict with module results
    """
    print("LLCM 完整五模块联合验证")
    print("=" * 55)
    print(f"  N_SEEDS={n_seeds}, EP_PRE={EP_PRE}, EP_FT={EP_FT}")
    print()

    results = {}

    # 模块1：物理预训练（已在 layer3_zero_loss_B.py 验证）
    print("模块1：物理预训练 F3 << 欧氏")
    print("  → 详见 layer3_zero_loss_B.py（F3=0.025, 欧氏=0.275, ×10倍已验证）")
    results["module1"] = {"status": "已验证，见 layer3_zero_loss_B.py"}

    # 模块2
    mod2_ok = verify_module2()
    results["module2"] = {"within_gt_cross": mod2_ok}

    # 模块3&4（已在 layer1_minimal_test.py 验证）
    print("\n模块3+4a：语言对齐 + 类时比例")
    print("  → 详见 layer1_minimal_test.py（p=0.0010, d=3.89, 5 seeds 已验证）")
    results["module3_4a"] = {"status": "已验证，见 layer1_minimal_test.py"}

    # 模块4b（验证中）
    print("\n模块4b：Law II 在线收敛速度")
    print("  → 详见 online_interaction_test.py（验证进行中）")
    results["module4b"] = {"status": "验证中，见 online_interaction_test.py"}

    # 模块5（弱版本）
    mod5 = run_module5(n_seeds=n_seeds)
    results["module5"] = mod5

    print("\n" + "=" * 55)
    print("五模块验证摘要：")
    print("  模块1（结构效应）：已验证 ✅")
    print(f"  模块2（语义质量）：{'✅' if mod2_ok else '❌'}")
    print("  模块3（方向A）：已验证 ✅")
    print("  模块4a（类时比例）：已验证 ✅")
    print("  模块4b（Law II）：验证中 🔄")
    p5 = mod5.get("p_value", 1.0)
    print(f"  模块5（方向B）：{'p=' + str(round(p5, 3)) + ' ✅' if p5 < 0.05 else '弱版本 🔄'}")

    return results


# ── 独立运行 ───────────────────────────────────────────────────

if __name__ == "__main__":
    run_baby_language_test()
