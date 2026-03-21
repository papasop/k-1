"""
lorentz_transformer/core/minkowski_norm.py

Backward-compatibility shim — all symbols are re-exported from layer_norm.py.
"""

from .layer_norm import (  # noqa: F401
    MinkowskiLayerNorm,
    MinkowskiLayerNormImproved,
    MinkowskiLayerNormOptimized,
    MinkowskiLayerNormStable,
    _BaseMinkowskiLayerNorm,
    compute_t_dim,
    MaskLike,
)

__all__ = [
    "MinkowskiLayerNorm",
    "MinkowskiLayerNormStable",
    "MinkowskiLayerNormOptimized",
    "MinkowskiLayerNormImproved",
    "_BaseMinkowskiLayerNorm",
    "compute_t_dim",
    "MaskLike",
]


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("MinkowskiLayerNorm v2 — 测试")
    print("=" * 60)

    D, B, L = 256, 2, 16
    x = torch.randn(B, L, D)

    # 正确的 t_dim（与注意力层对齐）
    t_dim = compute_t_dim(d_model=D, n_heads=8, time_ratio=0.25)
    print(f"\nt_dim = compute_t_dim(256, 8, 0.25) = {t_dim}")
    print(f"（旧版用了 32，正确值是 {t_dim}）\n")

    # ── 测试1：三个变体 ──────────────────────────────────────
    print("【测试1】三个变体输出稳定性")
    print("-" * 60)
    variants = [
        ("MinkowskiLayerNorm (真Mink)", MinkowskiLayerNorm(D, t_dim=t_dim)),
        ("MinkowskiLayerNormStable",    MinkowskiLayerNormStable(D, t_dim=t_dim)),
        ("MinkowskiLayerNormOptimized", MinkowskiLayerNormOptimized(D)),
    ]
    for name, ln in variants:
        out = ln(x)
        print(f"  {name}")
        print(f"    输出范数: {out.norm():.4f}  "
              f"范围: [{out.min():.3f}, {out.max():.3f}]  "
              f"NaN: {torch.isnan(out).any().item()}")

    # ── 测试2：t_dim 对齐验证 ────────────────────────────────
    print("\n【测试2】t_dim 对齐 vs 不对齐的几何差异")
    print("-" * 60)
    ln_wrong  = MinkowskiLayerNorm(D, t_dim=32)   # 旧版错误值
    ln_correct= MinkowskiLayerNorm(D, t_dim=t_dim) # 正确值
    out_w = ln_wrong(x)
    out_c = ln_correct(x)
    diff = (out_w - out_c).abs().mean().item()
    print(f"  t_dim=32  (旧版): 输出范数 {out_w.norm():.4f}")
    print(f"  t_dim={t_dim} (正确): 输出范数 {out_c.norm():.4f}")
    print(f"  平均差异: {diff:.6f}  ({'显著不同' if diff>0.01 else '几乎相同'})")

    # ── 测试3：类时/类空区分验证 ────────────────────────────
    print("\n【测试3】真正的 Minkowski 几何 — 类时vs类空归一化行为不同")
    print("-" * 60)
    ln = MinkowskiLayerNorm(D, t_dim=t_dim)

    # 构造一个纯类时向量（空间分量大于时间分量）
    x_timelike  = torch.zeros(1, 1, D)
    x_timelike[..., t_dim:] = 2.0   # 空间分量=2，时间分量=0 → s²-t²>0

    # 构造一个纯类空向量（时间分量大于空间分量）
    x_spacelike = torch.zeros(1, 1, D)
    x_spacelike[..., :t_dim] = 2.0  # 时间分量=2，空间分量=0 → s²-t²<0

    out_tl = ln(x_timelike)
    out_sl = ln(x_spacelike)

    s_tl = (x_timelike[..., t_dim:]**2).sum().item()**0.5
    t_tl = (x_timelike[..., :t_dim]**2).sum().item()**0.5
    s_sl = (x_spacelike[..., t_dim:]**2).sum().item()**0.5
    t_sl = (x_spacelike[..., :t_dim]**2).sum().item()**0.5

    print(f"  类时向量: ||s||={s_tl:.2f} ||t||={t_tl:.2f}  "
          f"mink_sq={s_tl**2-t_tl**2:+.2f}  "
          f"输出范数={out_tl.norm():.4f}")
    print(f"  类空向量: ||s||={s_sl:.2f} ||t||={t_sl:.2f}  "
          f"mink_sq={s_sl**2-t_sl**2:+.2f}  "
          f"输出范数={out_sl.norm():.4f}")
    diff_geo = abs(out_tl.norm().item() - out_sl.norm().item())
    print(f"  几何差异: {diff_geo:.4f}  "
          f"({'Minkowski几何有效' if diff_geo > 0.01 else '差异过小，检查t_dim'})")

    # ── 测试4：掩码比例稳定性 ─────────────────────────────────
    print("\n【测试4】掩码比例稳定性（t_dim 固定，检查输入分布）")
    print("-" * 60)
    ln = MinkowskiLayerNorm(D, t_dim=t_dim)
    norms = []
    for seed in range(10):
        torch.manual_seed(seed)
        x_rand = torch.randn(B, L, D)
        out = ln(x_rand)
        norms.append(out.norm().item())
    print(f"  10次随机输入的输出范数: "
          f"mean={np.mean(norms):.4f}  std={np.std(norms):.4f}  "
          f"cv={np.std(norms)/np.mean(norms):.4f}")
    print(f"  {'稳定 OK' if np.std(norms)/np.mean(norms) < 0.1 else '不稳定，检查 eps'}")

    # ── 测试5：梯度 ──────────────────────────────────────────
    print("\n【测试5】梯度流动")
    print("-" * 60)
    ln = MinkowskiLayerNorm(D, t_dim=t_dim)
    x_grad = torch.randn(B, L, D, requires_grad=True)
    out = ln(x_grad)
    out.sum().backward()
    g_in  = x_grad.grad.norm().item()
    g_w   = ln.weight.grad.norm().item()
    print(f"  输入梯度范数:  {g_in:.6f}  {'OK' if g_in>0 else 'FAIL'}")
    print(f"  weight梯度范数: {g_w:.6f}  {'OK' if g_w>0 else 'FAIL'}")

    # ── 测试6：F1/F3 注意力对齐确认 ──────────────────────────
    print("\n【测试6】与 F1/F3 注意力 t_dim 对齐确认")
    print("-" * 60)
    configs = [
        (256,  8, 0.25),
        (512,  8, 0.25),
        (768, 12, 0.25),
        (1024,16, 0.25),
    ]
    for d, h, tr in configs:
        td = compute_t_dim(d, h, tr)
        ln_test = MinkowskiLayerNorm(d, t_dim=td)
        x_test  = torch.randn(1, 4, d)
        out     = ln_test(x_test)
        ok = not torch.isnan(out).any()
        print(f"  d={d:5d} h={h:3d} tr={tr}  "
              f"t_dim={td:4d}  {'OK' if ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print("所有测试完成")
    print()
    print("使用方式:")
    print("  from lorentz_transformer.core.layer_norm import (")
    print("      MinkowskiLayerNorm, compute_t_dim")
    print("  )")
    print("  t_dim = compute_t_dim(d_model, n_heads, time_ratio)")
    print("  ln    = MinkowskiLayerNorm(d_model, t_dim=t_dim)")
    print("=" * 60)
