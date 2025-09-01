# ===========================================
#  Explicit-Formula Bridge (Weak) — Tests
#  (Gaussian/Fejér test sums, Heat trace, Moments)
#  SciPy-free; depends only on numpy/mpmath/matplotlib
# ===========================================
import numpy as np, math, mpmath as mp
import matplotlib.pyplot as plt
from math import log

# ----------------- Config -----------------
X1, X2   = 2, 200_000     # 可改到 1_000_000（会更慢）
M_SAMPLE = 2500
M_GRID   = 2400
NBASIS   = 64
ALPHA    = 1e-3
WEIGHT_W = "1/x"
NEIG     = 40             # 取前 NEIG 个本征
N_ZEROS  = 60             # 取前 N_ZEROS 个零点用于对齐
mp.mp.dps = 50
np.random.seed(0)

# ----------------- Utilities -----------------
def sieve_pi(N):
    N = int(N)
    is_prime = np.ones(N+1, dtype=bool); is_prime[:2] = False
    for p in range(2, int(N**0.5)+1):
        if is_prime[p]: is_prime[p*p:N+1:p] = False
    return np.cumsum(is_prime.astype(np.int64))

def Li_array(xs):
    return np.array([0.0 if x<2 else float(mp.li(x)) for x in xs])

def weight_w(x): return 1.0 if WEIGHT_W=="1" else 1.0/max(x,1e-12)
def rho0_of_x(x): t = max(log(x), 1e-12); return 1.0/t

def build_t_grid_and_basis(X1, X2, NBASIS):
    t1, t2 = math.log(X1), math.log(X2)
    knots = np.linspace(t1, t2, NBASIS)
    def hat_j(j, t):
        if j == 0: left, center, right = knots[0], knots[0], knots[1]
        elif j == NBASIS-1: left, center, right = knots[-2], knots[-1], knots[-1]
        else: left, center, right = knots[j-1], knots[j], knots[j+1]
        if t <= left or t >= right: return 0.0
        if t <= center: return (t-left)/max(center-left,1e-12)
        return (right-t)/max(right-center,1e-12)
    return knots, hat_j

def build_design_matrix(xs, knots, hat_fn):
    Tf = np.linspace(math.log(X1), math.log(xs.max()), 4000)
    dT = Tf[1]-Tf[0]
    NB = len(knots)
    Kj = np.zeros((NB, Tf.size))
    for j in range(NB):
        Kj[j,:] = np.array([(math.exp(tt)/max(tt,1e-12))*hat_fn(j,tt) for tt in Tf])
    Cj = np.cumsum(Kj, axis=1) * dT
    A = np.zeros((xs.size, NB))
    for i, x in enumerate(xs):
        ti = math.log(max(x, X1))
        k = min(np.searchsorted(Tf, ti), Tf.size-1)
        A[i,:] = Cj[:,k]
    return A

def fit_u_linearized(xs, pi_true, Li_vals, knots, hat_fn, ridge=1e-2, smooth=2e-1):
    y = (pi_true - Li_vals)
    A = build_design_matrix(xs, knots, hat_fn)
    W = np.ones_like(xs, dtype=float)   # t-均匀
    NB = len(knots)
    D = np.zeros((NB-1, NB))
    D[np.arange(NB-1), np.arange(NB-1)] = -1.0
    D[np.arange(NB-1), np.arange(1,NB)] =  1.0
    ATA = (A*W[:,None]).T @ (A*W[:,None]) + ridge*np.eye(NB) + smooth*(D.T@D)
    ATy = (A*W[:,None]).T @ (y*W)
    coeffs = np.linalg.solve(ATA, ATy)
    def u_of_x(x):
        t = math.log(max(x, X1))
        return float(np.dot(coeffs, [hat_fn(j,t) for j in range(NB)]))
    return coeffs, np.vectorize(u_of_x)

def pi_var_integral_I(xs, u_vec):
    Tf = np.linspace(np.log(2), np.log(xs.max()), 4000)
    dT = Tf[1]-Tf[0]
    g = np.array([ (math.exp(tt)/max(tt,1e-12))*math.exp(u_vec(math.exp(tt))) for tt in Tf ])
    G = np.cumsum(g)*dT
    out = np.zeros_like(xs, dtype=float)
    for i, x in enumerate(xs):
        ti = math.log(max(x, 2))
        k = min(np.searchsorted(Tf, ti), Tf.size-1)
        out[i] = G[k]
    return out

def ls_calibration(Ixs, pi_true):
    num = float(np.dot(Ixs, pi_true)); den = float(np.dot(Ixs, Ixs)) if np.dot(Ixs, Ixs)>1e-30 else 1.0
    return math.log(num/den)

# Tridiagonal H = D A D
def build_tridiag_H(u_vec):
    x = np.linspace(X1, X2, M_GRID)
    h = x[1]-x[0]
    w = np.array([weight_w(xi) for xi in x])
    w_mid = 0.5*(w[:-1] + w[1:])
    main_A = np.zeros(M_GRID); off_A  = np.zeros(M_GRID-1)
    main_A[1:-1] = (w_mid[1:] + w_mid[:-1])/(h*h)
    off_A[:]     = - w_mid/(h*h)
    main_A[0] = w_mid[0]/(h*h); main_A[-1] = w_mid[-1]/(h*h)
    main_A = main_A + ALPHA*w
    u_vals = u_vec(x); u_clip = np.maximum(u_vals, -0.99)
    rho0 = np.array([rho0_of_x(xi) for xi in x])
    rho_star = 0.5 * rho0 * np.exp(u_clip) * (u_clip + 1.0)
    rho_star = np.maximum(rho_star, 1e-16)
    D = 1.0/np.sqrt(rho_star)
    main_H = (D**2) * main_A
    off_H  = (D[:-1]*D[1:]) * off_A
    return main_H, off_H

# Sturm count + bisection (small k)
def sturm_count(lmbd, d, e):
    n = len(d); cnt = 0
    p = d[0] - lmbd
    if p <= 0: cnt += 1
    for i in range(1, n):
        denom = p if abs(p) > 1e-300 else (-1e-300 if p<0 else 1e-300)
        p = d[i] - lmbd - (e[i-1]**2)/denom
        if p <= 0: cnt += 1
    return cnt

def gersh_bounds(d, e):
    if len(d) == 1: return float(d[0]-1), float(d[0]+1)
    r = np.zeros_like(d)
    r[0] = abs(e[0]); r[-1] = abs(e[-1])
    if len(d) > 2: r[1:-1] = np.abs(e[:-1]) + np.abs(e[1:])
    lo = float(np.min(d - r)) - 1e-8
    hi = float(np.max(d + r)) + 1e-8
    return lo, hi

def tridiag_eigvals_smallk(d, e, k, tol=1e-10):
    lo, hi = gersh_bounds(d, e)
    eigs = []; left = lo
    for i in range(1, k+1):
        L, R = left, hi
        while sturm_count(R, d, e) < i: R = (R + hi)*0.5
        for _ in range(80):
            mid = 0.5*(L+R)
            c = sturm_count(mid, d, e)
            if c >= i: R = mid
            else: L = mid
            if R - L < tol: break
        eigs.append(0.5*(L+R)); left = R + 1e-12
    return np.array(eigs)

def riemann_gammas(N):
    return np.array([ float(mp.zetazero(n).imag) for n in range(1, N+1) ])

def align_kstar(sqrt_lams, gammas):
    K = min(len(sqrt_lams), len(gammas))
    x = gammas[:K]; y = sqrt_lams[:K]
    k = float(np.dot(x,y) / max(np.dot(x,x), 1e-18))
    resid = y - k*x
    eps = 1e-12
    MRE = float(np.mean(np.abs(resid)/(np.abs(x)+eps)))
    p95 = float(np.quantile(np.abs(resid), 0.95))
    return k, resid, MRE, p95

# ----------------- Model build -----------------
print(">> Generating pi(x) truth & sampling ...")
pi_cum = sieve_pi(X2)
xs = np.unique((np.exp(np.linspace(np.log(X1), np.log(X2), M_SAMPLE))).astype(int))
pi_true = pi_cum[xs]
Li_vals = Li_array(xs)

print(">> Fitting u(t) & LS calibration ...")
knots, hat_fn = build_t_grid_and_basis(X1, X2, NBASIS)
coeffs, u_vec0 = fit_u_linearized(xs, pi_true, Li_vals, knots, hat_fn)
Ixs = pi_var_integral_I(xs, u_vec0)
c_ls = ls_calibration(Ixs, pi_true)

def u_vec(x): return u_vec0(x) + c_ls

rmse = float(np.sqrt(np.mean((np.exp(c_ls)*Ixs - pi_true)**2)))
print(f"RMSE(pi_var,true)={rmse:.3f}")

print(">> Operator spectrum & alignment ...")
main_H, off_H = build_tridiag_H(u_vec)
evals = tridiag_eigvals_smallk(main_H, off_H, NEIG, tol=1e-10)
evals = np.maximum(evals, 0.0)
sqrt_lams = np.sqrt(evals)
gammas = riemann_gammas(N_ZEROS)
kstar, resid, MRE, p95 = align_kstar(sqrt_lams, gammas)
print(f"Alignment: k*={kstar:.9e}, MRE={MRE:.3e}, p95={p95:.3e}")

# ----------------- Bridge Test 1: Gaussian/Fejér sums -----------------
print("\n>> Bridge-1: Test-function sums (Gaussian & Fejér)")
def phi_gauss(t, B): return np.exp(-(t/B)**2)
def phi_fejer(t, B):
    y = (np.sinc(t/(np.pi*B)))**2  # numpy sinc = sin(pi x)/(pi x)
    return y

Bs = np.array([20, 30, 40, 60, 80])  # 频宽
rows = []
for B in Bs:
    SHg = float(np.sum(phi_gauss(sqrt_lams, B)))
    SZg = float(np.sum(phi_gauss(kstar*gammas[:NEIG], B)))
    err_g = abs(SHg - SZg) / max(abs(SZg), 1e-12)

    SHf = float(np.sum(phi_fejer(sqrt_lams, B)))
    SZf = float(np.sum(phi_fejer(kstar*gammas[:NEIG], B)))
    err_f = abs(SHf - SZf) / max(abs(SZf), 1e-12)

    rows.append((B, SHg, SZg, err_g, SHf, SZf, err_f))
    print(f"B={B:>3} | Gaussian rel.err={err_g:.3e} | Fejér rel.err={err_f:.3e}")

import csv
with open("bridge_test_functions.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["B","SH_gauss","SZ_gauss","relerr_gauss","SH_fejer","SZ_fejer","relerr_fejer"])
    for r in rows: w.writerow(r)

# ----------------- Bridge Test 2: Heat trace -----------------
print("\n>> Bridge-2: Heat trace Tr e^{-tau H}")
taus = np.geomspace(1e-6, 5e-3, 30)  # 小到中等时间
heat_rows = []
for tau in taus:
    TH = float(np.sum(np.exp(-tau*evals)))
    TZ = float(np.sum(np.exp(-tau*(kstar*gammas[:NEIG])**2)))
    rel = abs(TH - TZ)/max(abs(TZ), 1e-12)
    heat_rows.append((tau, TH, TZ, rel))
print(f"Heat trace median rel.err = {np.median([r[3] for r in heat_rows]):.3e}")
with open("bridge_heat_trace.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["tau","TrH","TrZ","relerr"])
    for r in heat_rows: w.writerow(r)

# Plot heat trace comparison
plt.figure(figsize=(7.2,5.2))
plt.loglog(taus, [r[1] for r in heat_rows], 'o-', label="Tr e^{-τH}")
plt.loglog(taus, [r[2] for r in heat_rows], 's--', label="∑ e^{-τ(k*γ)^2}")
plt.xlabel("τ"); plt.ylabel("trace"); plt.grid(True, which='both', alpha=0.3); plt.legend()
plt.title("Heat-trace bridge")
plt.tight_layout(); plt.savefig("bridge_heat_trace.png", dpi=150)

# ----------------- Bridge Test 3: Low even moments -----------------
print("\n>> Bridge-3: Even moments (cutoff at NEIG)")
moments = []
for m in [1,2,3,4]:
    MH = float(np.sum((sqrt_lams)**(2*m)))
    MZ = float(np.sum((kstar*gammas[:NEIG])**(2*m)))
    rel = abs(MH - MZ)/max(abs(MZ),1e-12)
    moments.append((m, MH, MZ, rel))
    print(f"moment 2m, m={m}: rel.err={rel:.3e}")
with open("bridge_moments.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["m","MH","MZ","relerr"])
    for r in moments: w.writerow(r)

print("\n=== Files saved ===")
print("- bridge_test_functions.csv")
print("- bridge_heat_trace.csv, bridge_heat_trace.png")
print("- bridge_moments.csv")
# ---- Moment-calibrated k (one-parameter), compare errors ----
import numpy as np

# 现有数据：sqrt_lams, gammas, NEIG
y = sqrt_lams[:NEIG]                  # sqrt(lambda_n)
x = gammas[:NEIG]                     # gamma_n
lam = y**2

# 旧：LS on sqrt (你已有)
k_LS = float(np.dot(x, y) / max(np.dot(x, x), 1e-18))

# 新：偶矩标定
k_M1 = float(np.sqrt( np.sum(lam) / max(np.sum(x**2), 1e-18) ))               # match sum lambda
k_M2 = float( ( np.sum(lam**2) / max(np.sum(x**4), 1e-18) ) ** 0.25 )         # match sum lambda^2

def moment_rel_errors(k):
    Z = k * x
    rel1 = abs(np.sum(lam)      - np.sum(Z**2))  / max(abs(np.sum(Z**2)), 1e-18)
    rel2 = abs(np.sum(lam**2)   - np.sum(Z**4))  / max(abs(np.sum(Z**4)), 1e-18)
    rel3 = abs(np.sum(lam**3)   - np.sum(Z**6))  / max(abs(np.sum(Z**6)), 1e-18)
    rel4 = abs(np.sum(lam**4)   - np.sum(Z**8))  / max(abs(np.sum(Z**8)), 1e-18)
    return rel1, rel2, rel3, rel4

for name, k in [("k_LS", k_LS), ("k_M1", k_M1), ("k_M2", k_M2)]:
    r1, r2, r3, r4 = moment_rel_errors(k)
    print(f"{name}: rel(m1)={r1:.3e}, rel(m2)={r2:.3e}, rel(m3)={r3:.3e}, rel(m4)={r4:.3e}")
# ---- Heat-trace calibrated k (one-parameter) ----
import numpy as np

taus = np.geomspace(1e-6, 5e-3, 50)
lamN = (sqrt_lams[:NEIG])**2
gamN = gammas[:NEIG]

def heat_loss(k):
    TH = np.exp(-np.outer(taus, lamN)).sum(axis=1)
    TZ = np.exp(-np.outer(taus, (k*gamN)**2)).sum(axis=1)
    r  = TH - TZ
    return float(np.dot(r, r))

# 1D 线搜索（对 log k）
k0 = k_LS
logk = np.log(k0 if k0>0 else 1e-12)
for _ in range(60):
    # 三点抛物线微调
    h = 0.05
    f0 = heat_loss(np.exp(logk))
    f1 = heat_loss(np.exp(logk+h))
    f_1= heat_loss(np.exp(logk-h))
    # 近似牛顿步
    g  = (f1 - f_1)/(2*h)
    H  = (f1 - 2*f0 + f_1)/(h*h)
    step = - g / (H if H!=0 else 1.0)
    step = np.clip(step, -0.2, 0.2)
    logk += step
k_heat = float(np.exp(logk))
print(f"k_heat = {k_heat:.9e}  (from k_LS={k_LS:.9e})")

# 复核偶矩与热核误差
def heat_rel_median(k):
    TH = np.exp(-np.outer(taus, lamN)).sum(axis=1)
    TZ = np.exp(-np.outer(taus, (k*gamN)**2)).sum(axis=1)
    rel = np.abs(TH-TZ)/np.maximum(np.abs(TZ), 1e-18)
    return float(np.median(rel))
print("median rel.err (heat):", f"{heat_rel_median(k_heat):.3e}")

r1,r2,r3,r4 = moment_rel_errors(k_heat)
print(f"k_heat moments rel: m1={r1:.3e}, m2={r2:.3e}, m3={r3:.3e}, m4={r4:.3e}")
# ==== Patch 2: affine fit on lambda  lambda ≈ A + B * gamma^2 ====
K = min(len(sqrt_lams), len(gammas), NEIG)
lam = (sqrt_lams[:K])**2
x2  = (gammas[:K])**2
X = np.vstack([np.ones(K), x2]).T
A,B = np.linalg.lstsq(X, lam, rcond=None)[0]
A, B = float(A), float(B)

# moments & heat checks
lamZ = A + B*x2
def rel(a,b): return abs(a-b)/max(abs(b),1e-18)
r1 = rel(np.sum(lam),   np.sum(lamZ))
r2 = rel(np.sum(lam**2),np.sum(lamZ**2))

taus = np.geomspace(1e-6, 5e-3, 30)
TH = np.exp(-np.outer(taus, lam)).sum(axis=1)
TZ = np.exp(-np.outer(taus, lamZ)).sum(axis=1)
heat_med = float(np.median(np.abs(TH-TZ)/np.maximum(np.abs(TZ),1e-18)))

print("\n[Affine lambda fit  A + B γ^2 ]")
print(f"  A={A:.3e}, B={B:.9e}")
print(f"  rel m1={r1:.3e}, rel m2={r2:.3e},  heat median rel.err={heat_med:.2e}")
# ===== Unified Bridge Summary & Diagnostics =====
import numpy as np, csv

# 需要已有变量：sqrt_lams, gammas, NEIG
K = min(len(sqrt_lams), len(gammas), NEIG)
y  = sqrt_lams[:K]
x  = gammas[:K]
lam = y**2
x2  = x**2

# ---- helper ----
def rel(a,b): return float(abs(a-b)/max(abs(b),1e-18))
def heat_median_rel(lamH, lamZ, taus=np.geomspace(1e-6,5e-3,50)):
    TH = np.exp(-np.outer(taus, lamH)).sum(axis=1)
    TZ = np.exp(-np.outer(taus, lamZ)).sum(axis=1)
    return float(np.median(np.abs(TH-TZ)/np.maximum(np.abs(TZ),1e-18)))

def moments_rel(lamH, lamZ):
    r1 = rel(np.sum(lamH),   np.sum(lamZ))
    r2 = rel(np.sum(lamH**2),np.sum(lamZ**2))
    r3 = rel(np.sum(lamH**3),np.sum(lamZ**3))
    r4 = rel(np.sum(lamH**4),np.sum(lamZ**4))
    return r1,r2,r3,r4

# ---- 1) k-based 标定 ----
k_LS = float(np.dot(x,y)/max(np.dot(x,x),1e-18))
k_M1 = float(np.sqrt( np.sum(lam)  / max(np.sum(x2),1e-18)))
k_M2 = float((       np.sum(lam**2)/ max(np.sum(x2**2),1e-18))**0.25)

# 热核最小二乘标定
taus = np.geomspace(1e-6,5e-3,50)
def heat_loss(k):
    TH = np.exp(-np.outer(taus, lam)).sum(axis=1)
    TZ = np.exp(-np.outer(taus, (k*x)**2)).sum(axis=1)
    r = TH - TZ
    return float(np.dot(r,r))
logk = np.log(max(k_LS,1e-18))
for _ in range(60):
    h=0.05; f0=heat_loss(np.exp(logk)); f1=heat_loss(np.exp(logk+h)); f_1=heat_loss(np.exp(logk-h))
    g=(f1-f_1)/(2*h); H=(f1-2*f0+f_1)/(h*h); step = -g/(H if H!=0 else 1.0)
    logk += np.clip(step, -0.2, 0.2)
k_heat = float(np.exp(logk))

# ---- 2) sqrt 仿射 y ≈ a + k (x + δ) ----
X2 = np.vstack([np.ones(K), x]).T
a2,k2 = np.linalg.lstsq(X2, y, rcond=None)[0]
X3 = np.vstack([np.ones(K), x, np.ones(K)]).T
a3,k3,kd = np.linalg.lstsq(X3, y, rcond=None)[0]
delta3 = float(kd/k3) if abs(k3)>1e-18 else 0.0

# ---- 3) lambda 仿射 lam ≈ A + B x^2 ----
Xλ = np.vstack([np.ones(K), x2]).T
A,B = np.linalg.lstsq(Xλ, lam, rcond=None)[0]

# ---- assemble models ----
models = []
def add_model(name, lamZ, extra):
    m1,m2,m3,m4 = moments_rel(lam, lamZ)
    hm = heat_median_rel(lam, lamZ, taus)
    models.append([name, m1, m2, m3, m4, hm] + extra)

# k family
for nm,k in [("k_LS",k_LS),("k_M1",k_M1),("k_M2",k_M2),("k_heat",k_heat)]:
    lamZ = (k*x)**2
    add_model(nm, lamZ, [k, np.nan, np.nan, np.nan])

# sqrt affine
lamZ2 = (a2 + k2*x)**2
add_model("sqrt_affine_ak", lamZ2, [k2, a2, 0.0, np.nan])
lamZ3 = (a3 + k3*(x+delta3))**2
add_model("sqrt_affine_akδ", lamZ3, [k3, a3, delta3, np.nan])

# lambda affine
lamZλ = (A + B*x2)
add_model("lambda_affine_AB", lamZλ, [np.sqrt(max(B,0.0)), A, 0.0, B])

# ---- diagnostics ----
ratio_B_over_kLS2 = float(B / max(k_LS**2,1e-30))
A_over_mean_lambda = float(A / max(np.mean(lam),1e-30))

print("\n=== Bridge Summary (moments rel.err & heat median rel.err) ===")
hdr = ["model","rel_m1","rel_m2","rel_m3","rel_m4","heat_med_rel","k/equiv","a/A","delta","B"]
for row in models:
    name = row[0]
    print(f"{name:>18} | m1={row[1]:.3e}, m2={row[2]:.3e}, m3={row[3]:.3e}, m4={row[4]:.3e}, heat={row[5]:.3e}")

print(f"\nDiagnostics:  B/(k_LS^2) = {ratio_B_over_kLS2:.3e},   A/mean(λ) = {A_over_mean_lambda:.3e}")

with open("bridge_summary.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(hdr)
    for row in models: w.writerow(row)
print("Saved bridge_summary.csv")

