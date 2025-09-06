# == v2.5 one-click dynamic feedback ==
# MODE = mul_centered / add_centered  |  NPTS=4096  |  mp.dps=15
# Upgrades vs v2.4:
#   [A] ρ 可选 box / gauss（默认 gauss）+ 可调带宽
#   [B] F(ρ) 采用“分位数→tanh”软映射（量纲自洽+不易平台）
#   [C] c 的逐层细化搜索（在 q_cond 约束内最大化 K_max）

import time, math, numpy as np
from tqdm import tqdm
import mpmath as mp

# ----------------------------- config ---------------------------------
MP_DPS      = 15
MODE        = "mul_centered"        # "mul_centered" or "add_centered"
NPTS        = 4096
T0          = 939.024301
SIGMA       = 10.0
ALPHA       = 1e-3
BETA        = 1e-3

# ρ 估计
RHO_METHOD  = "gauss"               # "gauss" or "box"
RHO_BW_FAC  = 1.20                  # 有效带宽系数：Δ_eff = Δ * RHO_BW_FAC
GAUSS_BW_FAC= 0.80                  # 高斯核 h = GAUSS_BW_FAC * Δ_eff
PAD_SPACINGS= 6.0                   # 额外拉取零点的窗口（单位：平均间距）

# F(ρ) 软映射（mul_centered 模式）
F_MIN       = 0.55
F_MAX       = 1.70
QUANT_LO    = 0.12                  # 分位数下界（建议 0.10~0.18）
QUANT_HI    = 0.88                  # 分位数上界（建议 0.82~0.90）
EDGE_EPS    = 0.98                  # 边界“接近度” tanh 校准
UNIT_MEAN_F = True                  # 映射后再做 unit-mean 归一（稳定 Iq）
Q_FLOOR_FRAC= 0.10                  # q(t) 正性下限（相对 W0）

# c 搜索
C0          = -0.35
QCOND_MAX   = 3.0
COARSE_LO   = -1.00
COARSE_HI   = +1.00
COARSE_STEP = 0.10
REFINE_LVLS = 3                     # 细化层数
REFINE_PTS  = 21                    # 每层采样点数（奇数）
REFINE_SHRINK=0.40                  # 每层对区间收缩比例

# ----------------------------- utils ----------------------------------
mp.mp.dps = MP_DPS

def trapezoid_complex_safe(y, x):
    y = np.asarray(y); x = np.asarray(x, float)
    return np.trapezoid(y, x)

def N_approx(T):
    if T <= 0: return 0.0
    return (T/(2*math.pi))*math.log(T/(2*math.pi)) - T/(2*math.pi) + 7.0/8.0

def fetch_local_zeros(T_left, T_right, pad=0.0):
    a = max(T_left - pad, 1.0); b = T_right + pad
    n_a = max(1, int(math.floor(N_approx(a) - 5)))
    n_b = max(n_a+1, int(math.ceil (N_approx(b) + 5)))
    zs  = []
    pbar = tqdm(range(n_a, n_b+1), desc="Fetching local zeros")
    for n in pbar:
        try:
            t = float(mp.zetazero(n).imag)
            if a <= t <= b: zs.append(t)
        except: pass
    return np.array(sorted(zs), float)

def build_grid(T0, sigma, npts):
    tb = np.linspace(T0 - sigma, T0 + sigma, npts)
    t_rel = tb - T0
    return tb, t_rel

def eval_zeta_on_grid(tb):
    pbar = tqdm(tb, desc="Evaluating zeta")
    return np.array([mp.zeta(0.5 + 1j*mp.mpf(t)) for t in pbar], complex)

def derivative_log_zeta_along_t(zeta_arr, tb):
    amp = np.log(np.abs(zeta_arr) + 0.0)
    ang = np.unwrap(np.angle(zeta_arr))
    d_amp = np.gradient(amp, tb)
    d_ang = np.gradient(ang, tb)
    return d_amp + 1j*d_ang

def mean_spacing(T):
    return 2*math.pi / max(1e-12, math.log(T/(2*math.pi)))

# ---- rho estimators ----
def rho_box(tb, zeros, Delta_eff):
    if zeros.size == 0: return np.zeros_like(tb)
    T = tb[:, None]; Z = zeros[None, :]
    mask = (np.abs(Z - T) <= (Delta_eff/2.0))
    return mask.sum(axis=1).astype(float) / float(Delta_eff)

def rho_gauss(tb, zeros, h):
    if zeros.size == 0: return np.zeros_like(tb)
    T = tb[:, None]; Z = zeros[None, :]
    K = np.exp(-0.5 * ((T - Z)/h)**2) / (np.sqrt(2*np.pi)*h)
    return np.sum(K, axis=1)

def robust_z(x):
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    z = (x - med) / (1.4826*mad)      # ~N(0,1) 对应标度
    return z, med, mad

def corr(x, y):
    xv = np.asarray(x, float); yv = np.asarray(y, float)
    sx, sy = np.std(xv), np.std(yv)
    if sx == 0 or sy == 0: return 0.0
    return float(np.corrcoef(xv, yv)[0,1])

# ------------------------ q(t) construction ----------------------------
def quantile_tanh_map_from_z(z, fmin, fmax, q_lo, q_hi, edge_eps=0.98):
    """在 z 的分位区间 [q_lo, q_hi] 线性居中，外侧由 tanh 平滑饱和到 [fmin, fmax]。"""
    z = np.asarray(z, float)
    zl, zh = np.quantile(z, [q_lo, q_hi])
    zmid   = 0.5*(zl + zh)
    zhalf  = 0.5*(zh - zl) + 1e-12
    center = 0.5*(fmin + fmax)
    half   = 0.5*(fmax - fmin)
    K = np.arctanh(edge_eps)         # 让两端接近 fmin/fmax
    u = (z - zmid) / zhalf
    F = center + half * np.tanh(K * u)
    return F, (zl, zh)

def build_q(tb, t_rel, rho, mode, c, alpha, beta,
            fmin=F_MIN, fmax=F_MAX, q_floor_frac=Q_FLOOR_FRAC,
            unit_mean=UNIT_MEAN_F):
    W0 = np.ones_like(tb)
    if mode == "mul_centered":
        z, med, mad = robust_z(rho)             # 稳健标准化
        F_raw = 1.0 + c * z                     # 仅用于诊断“原始越界率”
        F, (zl, zh) = quantile_tanh_map_from_z(c*z, fmin, fmax, QUANT_LO, QUANT_HI, EDGE_EPS)
        if unit_mean:
            F = F / (np.mean(F) + 1e-15)
        q = W0 * F + alpha*(t_rel**2) + beta
        sat_lo_raw = float(np.mean(F_raw <= fmin))
        sat_hi_raw = float(np.mean(F_raw >= fmax))
        print(f"[QuantileMap] z-quantiles @({QUANT_LO:.2f},{QUANT_HI:.2f}) -> ({zl:.3g},{zh:.3g}); "
              f"raw<=min:{sat_lo_raw:.2%} raw>=max:{sat_hi_raw:.2%}")
    elif mode == "add_centered":
        w = 1.0 + np.log(2.0 + np.abs(t_rel))
        z, med, mad = robust_z(rho)
        q = W0 + c*z*w + alpha*(t_rel**2) + beta
    else:
        raise ValueError("MODE must be 'mul_centered' or 'add_centered'.")
    # positivity floor
    q_floor = q_floor_frac * float(np.mean(W0)) + 1e-12
    q = np.maximum(q, q_floor)
    return q

# ---------------------------- K metrics --------------------------------
def compute_metrics(tb, t_rel, zeta_arr, D_arr, q_arr):
    absD2 = np.abs(D_arr)**2
    I1 = float(np.trapezoid(absD2 / q_arr, tb))                  # ∫ |D|^2 / q
    Iq = float(np.trapezoid(q_arr, tb))
    It2 = float(np.trapezoid(t_rel**2, tb))

    K_max = I1
    num_base = trapezoid_complex_safe(D_arr, tb)                  # ∫ D
    K_base   = (abs(num_base)**2) / Iq
    num_opt  = trapezoid_complex_safe((np.conj(D_arr)/q_arr) * D_arr, tb)  # ≈ I1
    den_opt  = I1
    K_opt    = (abs(num_opt)**2) / den_opt if den_opt > 0 else 0.0
    rel_err  = 0.0 if K_max == 0 else abs(K_opt - K_max)/K_max

    q_min = float(np.min(q_arr))
    q_max = float(np.max(q_arr))
    q_cond = q_max / max(q_min, 1e-18)
    min_abs_zeta =  float(np.min(np.abs(zeta_arr)))

    return dict(K_max=K_max, K_opt=K_opt, K_base=K_base, Gain=(K_opt/K_base if K_base>0 else np.inf),
                rel_err=rel_err, Iq=Iq, It2=It2, q_min=q_min, q_max=q_max, q_cond=q_cond,
                min_abs_zeta=min_abs_zeta)

# --------------------------- c searchers -------------------------------
def eval_at_c(tb, t_rel, zeta_arr, D_arr, rho, c):
    q_arr = build_q(tb, t_rel, rho, MODE, c, ALPHA, BETA,
                    fmin=F_MIN, fmax=F_MAX, q_floor_frac=Q_FLOOR_FRAC,
                    unit_mean=UNIT_MEAN_F)
    return compute_metrics(tb, t_rel, zeta_arr, D_arr, q_arr)

def coarse_sweep(tb, t_rel, zeta_arr, D_arr, rho):
    print("\n[c-sweep coarse]")
    best = None
    for c in np.arange(COARSE_LO, COARSE_HI + 1e-12, COARSE_STEP):
        met = eval_at_c(tb, t_rel, zeta_arr, D_arr, rho, float(c))
        print(f"  c={c:+.2f} | K_max={met['K_max']:>10.3g}  K_opt={met['K_opt']:>10.3g}  "
              f"K_base={met['K_base']:.6g}  Gain={met['Gain']:.3g}  "
              f"q_cond={met['q_cond']:.4g}  rel_err={met['rel_err']:.1e}")
        if met['q_cond'] <= QCOND_MAX and (best is None or met['K_max'] > best[1]['K_max']):
            best = (c, met)
    return best

def refine_around(tb, t_rel, zeta_arr, D_arr, rho, c_center, width, levels=2, pts=15):
    best = None
    lo = c_center - width; hi = c_center + width
    for lv in range(levels):
        cs = np.linspace(lo, hi, pts)
        for c in cs:
            met = eval_at_c(tb, t_rel, zeta_arr, D_arr, rho, float(c))
            if met['q_cond'] <= QCOND_MAX and (best is None or met['K_max'] > best[1]['K_max']):
                best = (float(c), met)
        width *= REFINE_SHRINK
        if best is None:
            mid = 0.5*(lo+hi)
        else:
            mid = best[0]
        lo, hi = mid - width, mid + width
    return best

# ------------------------------ main -----------------------------------
def main():
    t0 = time.time()
    print(f"== v2.5 one-click dynamic feedback ==\nMODE = {MODE}  |  NPTS={NPTS}  |  mp.dps={MP_DPS}")

    tb, t_rel = build_grid(T0, SIGMA, NPTS)
    Delta     = mean_spacing(T0)
    Delta_eff = Delta * RHO_BW_FAC
    pad       = PAD_SPACINGS * Delta

    zeros = fetch_local_zeros(tb[0], tb[-1], pad=pad)
    print("[Feedback] building ρ ... ", end="", flush=True)

    if RHO_METHOD == "gauss":
        h = GAUSS_BW_FAC * Delta_eff
        rho = rho_gauss(tb, zeros, h)
        print(f"gauss (h≈{h:.3f}, Δ≈{Delta_eff:.3f})")
    else:
        rho = rho_box(tb, zeros, Delta_eff)
        print(f"box   (Δ≈{Delta_eff:.3f})")

    zeta_arr = eval_zeta_on_grid(tb)
    D_arr    = derivative_log_zeta_along_t(zeta_arr, tb)

    print(f"[Diagnostic] corr(|D|^2, rho) ≈ {corr(np.abs(D_arr)**2, rho):.3f}")

    # ---------- single run ----------
    met0 = eval_at_c(tb, t_rel, zeta_arr, D_arr, rho, C0)
    print("\n[Single Run @ windowed T0]")
    print(f"  T0 ≈ {T0}, sigma={SIGMA}, alpha={ALPHA}, beta={BETA}, mode={MODE}, c={C0}")
    print(f"  K_max         : {met0['K_max']:.3g}")
    print(f"  K_opt         : {met0['K_opt']:.3g}   (should ≈ K_max)")
    print(f"  K_baseline    : {met0['K_base']:.6g}")
    print(f"  Gain(opt/base): {met0['Gain']:.3g}")
    print(f"  rel_err       : {met0['rel_err']:.3e}")
    print(f"  Iq=∫q: {met0['Iq']:.4g}, It2=∫t^2: {met0['It2']:.4g}")
    print(f"  q_min={met0['q_min']:.6g}, q_max={met0['q_max']:.6g}, q_cond={met0['q_cond']:.6g}, min|ζ|={met0['min_abs_zeta']:.3g}")

    # ---------- coarse sweep ----------
    best = coarse_sweep(tb, t_rel, zeta_arr, D_arr, rho)

    # ---------- refine ----------
    if best is not None:
        c_star, met_star = best
        width = max(0.15, 0.25*abs(COARSE_HI - COARSE_LO))
        best_ref = refine_around(tb, t_rel, zeta_arr, D_arr, rho,
                                 c_star, width, levels=REFINE_LVLS, pts=REFINE_PTS)
        if best_ref is not None:
            c_star, met_star = best_ref
        print(f"\n[Pick] q_cond≤{QCOND_MAX:.1f}: best c={c_star:+.4f} | "
              f"K_max={met_star['K_max']:.3g}  Gain={met_star['Gain']:.3g}  q_cond={met_star['q_cond']:.3g}")
    else:
        print(f"\n[Pick] No c met q_cond≤{QCOND_MAX:.1f}.")

    print(f"\nDone. Elapsed ≈ {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
