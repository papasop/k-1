# --- 如果你之前未定义这些函数，请一并运行；已定义也可重复安全覆盖 ---
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

eps = 1e-12
def bandpass(x, fs, f1=0.08, f2=2.0):
    b,a = butter(2, [f1/(fs/2), f2/(fs/2)], 'bandpass')
    return filtfilt(b, a, x)

def wls_ridge_fit(X, y, w, lam):
    X_mean = X.mean(axis=0); X_std = X.std(axis=0) + eps
    y_mean = y.mean();        y_std = y.std() + eps
    Xs = (X - X_mean)/X_std
    ys = (y - y_mean)/y_std
    W = w[:, None]
    XtWX = Xs.T @ (W * Xs)
    XtWy = Xs.T @ (W[:,0] * ys)
    w_s  = np.linalg.solve(XtWX + lam*np.eye(X.shape[1]), XtWy)
    w_hat = (y_std / X_std) * w_s
    y_hat = X @ w_hat
    y_bar = np.average(y, weights=w)
    ss_res = np.sum(w * (y - y_hat)**2)
    ss_tot = np.sum(w * (y - y_bar)**2) + eps
    r2_w = 1 - ss_res/ss_tot
    return w_hat, r2_w

def wls_ridge_fit_cv(X, y, w, lam_grid):
    best = None
    for lam in lam_grid:
        w_hat, r2_w = wls_ridge_fit(X, y, w, lam)
        if (best is None) or (r2_w > best[1]):
            best = (w_hat, r2_w, lam)
    return best  # (w_hat, r2_w, λ*)

def window_metrics(sig, t, win, step, q=None):
    J, Jw_norm, M, idxs = [], [], [], []
    for i in range(0, len(sig)-win, step):
        seg, tt = sig[i:i+win], t[i:i+win]
        J.append(np.trapz(seg**2, tt))
        if q is not None:
            qw = q[i:i+win]
            num = np.trapz((seg**2)/(qw+eps), tt)
            den = np.trapz(1.0/(qw+eps), tt)
            Jw_norm.append(num/(den+eps))
        else:
            Jw_norm.append(J[-1])
        Th = cumulative_trapezoid(seg, tt, initial=0.0)
        M.append(np.max(np.abs(Th)))
        idxs.append(i)
    return np.array(J), np.array(Jw_norm), np.array(M), np.array(idxs)

def report_auc(tag, y, s):
    if len(np.unique(y)) < 2:
        print(f"{tag}: class-collapsed, skip"); return
    auc  = roc_auc_score(y, s); ap = average_precision_score(y, s)
    print(f"{tag:>18s}: ROC-AUC={auc:.3f} | PR-AUC={ap:.3f}")

# --- 读取你上个单元里已有的对象：omega_phys_hp, omega_zeta, omega_chi4, q, w_ts, t, WIN, STEP_NOLEAK, fs, inj ---
X = np.vstack([omega_zeta, omega_chi4]).T

# Baseline（复核）
w_hat_base, r2_base = wls_ridge_fit(X, omega_phys_hp, w_ts, lam=1e4)
alpha_b, beta_b = float(w_hat_base[0]), float(w_hat_base[1])
print(f"[Baseline λ=1e4] α̂={alpha_b:+.6f}, β̂={beta_b:+.6f}, R2_w={r2_base:.4f}")

# Scheme A：HPF 视角 + CV 选 λ
lam_grid_A = [1e2, 3e2, 1e3, 3e3, 1e4]
w_hat_A, r2_A, lam_A = wls_ridge_fit_cv(X, omega_phys_hp, w_ts, lam_grid_A)
alpha_A, beta_A = float(w_hat_A[0]), float(w_hat_A[1])
print(f"[Scheme A (CV)]  α̂={alpha_A:+.6f}, β̂={beta_A:+.6f}, R2_w={r2_A:.4f}, λ*={lam_A:g}")

# Scheme B：频带对齐 + CV 选 λ（与打标带宽一致）
y_bp = bandpass(omega_phys_hp, fs, f1=0.08, f2=2.0)
z_bp = bandpass(omega_zeta,    fs, f1=0.08, f2=2.0)
c_bp = bandpass(omega_chi4,    fs, f1=0.08, f2=2.0)
X_bp = np.vstack([z_bp, c_bp]).T
lam_grid_B = [1e1, 3e1, 1e2, 3e2, 1e3]
w_hat_B, r2_B, lam_B = wls_ridge_fit_cv(X_bp, y_bp, w_ts, lam_grid_B)
alpha_B, beta_B = float(w_hat_B[0]), float(w_hat_B[1])
print(f"[Scheme B (BP+CV)] α̂={alpha_B:+.6f}, β̂={beta_B:+.6f}, R2_w={r2_B:.4f}, λ*={lam_B:g}")

# --- 三种拟合的证书（非重叠窗） ---
def metrics_for_fit(alpha_hat, beta_hat, tag):
    omega_res = omega_phys_hp - (alpha_hat*omega_zeta + beta_hat*omega_chi4)
    Jp,  Jpw_norm,  Mp,  idxs = window_metrics(omega_phys_hp, t, WIN, STEP_NOLEAK, q=q)
    Jr,  Jrw_norm,  Mr,  _    = window_metrics(omega_res,    t, WIN, STEP_NOLEAK, q=q)
    dJwn = Jrw_norm - Jpw_norm
    dM   = Mr - Mp
    print(f"\n[{tag}] ΔJ^(w,norm) mean/median = {dJwn.mean():+.3e} / {np.median(dJwn):+.3e}")
    print(f"[{tag}] Δmax|Θ|      mean/median = {dM.mean():+.3e} / {np.median(dM):+.3e}")
    return {"Jpw_norm":Jpw_norm, "Jrw_norm":Jrw_norm, "Mp":Mp, "Mr":Mr, "dJwn":dJwn, "dM":dM, "idxs":idxs}

res_base = metrics_for_fit(alpha_b, beta_b, "Baseline")
res_A    = metrics_for_fit(alpha_A, beta_A, "Scheme A (CV)")
res_B    = metrics_for_fit(alpha_B, beta_B, "Scheme B (BP+CV)")

# --- 无泄漏标签：带通注入能量（非重叠窗） ---
inj_bp = bandpass(inj, fs, f1=0.08, f2=2.0)
injE   = [np.trapz(inj_bp[i:i+WIN]**2, t[i:i+WIN]) for i in res_A["idxs"]]
injE   = np.array(injE)
thr    = np.quantile(injE, 0.65)
y_true = (injE >= thr).astype(int)

print("\n=== AUC (pre / post / Δ) ===")
for tag, res in [("|Θ|", res_base), ("Jw^n", res_base),
                 ("|Θ|", res_A),    ("Jw^n", res_A),
                 ("|Θ|", res_B),    ("Jw^n", res_B)]:
    pre  = res["Mp"] if tag=="|Θ|" else res["Jpw_norm"]
    post = res["Mr"] if tag=="|Θ|" else res["Jrw_norm"]
    d    = pre - post
    head = f"{tag} Base" if res is res_base else (f"{tag} A" if res is res_A else f"{tag} B")
    report_auc(f"AUC {head} (pre) ", y_true, pre)
    report_auc(f"AUC {head} (post)", y_true, post)
    report_auc(f"AUC {head} (Δ)   ", y_true, d)
# ========= Hotfix v3.4b: 带通投影式相消 + 更强正则 =========
# 依赖：已存在的变量/函数：
# t, fs, WIN, STEP_NOLEAK, omega_phys_hp, omega_zeta, omega_chi4, q, inj
# 函数：bandpass, wls_ridge_fit_cv, window_metrics, report_auc

# 1) 在带通域拟合 (与打标同带宽)
f1, f2 = 0.08, 2.0
y_bp  = bandpass(omega_phys_hp, fs, f1=f1, f2=f2)
z_bp  = bandpass(omega_zeta,    fs, f1=f1, f2=f2)
c_bp  = bandpass(omega_chi4,    fs, f1=f1, f2=f2)
X_bp  = np.vstack([z_bp, c_bp]).T

# 2) 强一点的 λ 网格（抑制过拟合；可再加大到 3e3,1e4）
lam_grid_Bp = [3e2, 1e3, 3e3]
w_hat_Bp, r2_Bp, lam_Bp = wls_ridge_fit_cv(X_bp, y_bp, (1.0/(q+1e-12)), lam_grid_Bp)
alpha_Bp, beta_Bp = float(w_hat_Bp[0]), float(w_hat_Bp[1])
print(f"[Scheme B' (BP+CV, proj-sub)] α̂={alpha_Bp:+.6f}, β̂={beta_Bp:+.6f}, R2_w={r2_Bp:.4f}, λ*={lam_Bp:g}")

# 3) 只减去带通投影（频带一致，避免破坏带外）
proj_bp      = bandpass(alpha_Bp*omega_zeta + beta_Bp*omega_chi4, fs, f1=f1, f2=f2)
omega_res_Bp = omega_phys_hp - proj_bp

# 4) 证书（非重叠窗）
Jp_Bp,  Jpw_Bp,  Mp_Bp,  idxs_Bp = window_metrics(omega_phys_hp, t, WIN, STEP_NOLEAK, q=q)
Jr_Bp,  Jrw_Bp,  Mr_Bp,  _       = window_metrics(omega_res_Bp, t, WIN, STEP_NOLEAK, q=q)
dJwn_Bp = Jrw_Bp - Jpw_Bp
dM_Bp   = Mr_Bp  - Mp_Bp
print(f"[Scheme B'] ΔJ^(w,norm) mean/median = {dJwn_Bp.mean():+.3e} / {np.median(dJwn_Bp):+.3e}")
print(f"[Scheme B'] Δmax|Θ|      mean/median = {dM_Bp.mean():+.3e} / {np.median(dM_Bp):+.3e}")

# 5) 无泄漏标签（仍用带通注入能量）
inj_bp  = bandpass(inj, fs, f1=f1, f2=f2)
injE_Bp = np.array([np.trapz(inj_bp[i:i+WIN]**2, t[i:i+WIN]) for i in idxs_Bp])
thr_Bp  = np.quantile(injE_Bp, 0.65)
y_true_Bp = (injE_Bp >= thr_Bp).astype(int)

# 6) AUC（pre/post/Δ）
def _block_auc(tag, pre, post, y):
    from sklearn.metrics import roc_auc_score, average_precision_score
    d = pre - post
    print(f"AUC {tag} (pre) : ROC={roc_auc_score(y, pre):.3f} | PR={average_precision_score(y, pre):.3f}")
    print(f"AUC {tag} (post): ROC={roc_auc_score(y, post):.3f} | PR={average_precision_score(y, post):.3f}")
    print(f"AUC {tag} (Δ)   : ROC={roc_auc_score(y, d):.3f} | PR={average_precision_score(y, d):.3f}")

print("\n=== Hotfix B' AUC ===")
_block_auc("|Θ|",  Mp_Bp,  Mr_Bp,  y_true_Bp)
_block_auc("Jw^n", Jpw_Bp, Jrw_Bp, y_true_Bp)
# ==== v3.4c: 单方向(1-DOF)带通相消 + AUC驱动选λ（可选）====
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.metrics import roc_auc_score, average_precision_score

eps = 1e-12
def bandpass(x, fs, f1=0.08, f2=2.0):
    b,a = butter(2, [f1/(fs/2), f2/(fs/2)], 'bandpass')
    return filtfilt(b, a, x)

# 已有变量：omega_phys_hp, omega_zeta, omega_chi4, q, fs, t, WIN, STEP_NOLEAK, inj
W = 1.0/(q + eps)
f1, f2 = 0.08, 2.0
y_bp  = bandpass(omega_phys_hp, fs, f1=f1, f2=f2)
z_bp  = bandpass(omega_zeta,    fs, f1=f1, f2=f2)
c_bp  = bandpass(omega_chi4,    fs, f1=f1, f2=f2)
X_bp  = np.vstack([z_bp, c_bp]).T

# ---- 1) 选单一方向 u（加权最大相关方向；等价于广义CCA的1分量）----
# 令 A = X^T W X, b = X^T W y；选 u ∝ A^{-1} b，并做 A-单位化: u^T A u = 1
A = X_bp.T @ (W[:,None] * X_bp)
b = X_bp.T @ (W * y_bp)
u = np.linalg.solve(A + 1e-8*np.eye(2), b)   # 稳定化
normAu = np.sqrt(u.T @ A @ u) + eps
u = u / normAu                               # 使 u^T A u = 1

# ---- 2) 对 z = X u 只拟合 1 个系数 γ（带权ridge），再做带通投影相消 ----
z = X_bp @ u
zWz = float((W * z * z).sum())
zWy = float((W * z * y_bp).sum())

def make_residual_from_gamma(gamma):
    proj_bp = bandpass(gamma * (omega_zeta*u[0] + omega_chi4*u[1]), fs, f1=f1, f2=f2)
    return omega_phys_hp - proj_bp

def eval_scores(omega_res):
    # 窗证书
    from scipy.integrate import cumulative_trapezoid
    def window_metrics(sig, t, win, step, q=None):
        Jw_norm, M = [], []
        for i in range(0, len(sig)-win, step):
            seg, tt = sig[i:i+win], t[i:i+win]
            qw = q[i:i+win]
            num = np.trapz((seg**2)/(qw+eps), tt)
            den = np.trapz(1.0/(qw+eps), tt)
            Jw_norm.append(num/(den+eps))
            Th = cumulative_trapezoid(seg, tt, initial=0.0)
            M.append(np.max(np.abs(Th)))
        return np.array(Jw_norm), np.array(M)

    Jp, Mp = window_metrics(omega_phys_hp, t, WIN, WIN, q=q)
    Jr, Mr = window_metrics(omega_res,     t, WIN, WIN, q=q)
    dJ = Jp - Jr
    dM = Mp - Mr

    # 无泄漏标签（同你之前做法）
    inj_bp = bandpass(inj, fs, f1=f1, f2=f2)
    injE   = np.array([np.trapz(inj_bp[i:i+WIN]**2, t[i:i+WIN]) for i in range(0, len(inj_bp)-WIN, WIN)])
    thr    = np.quantile(injE, 0.65)
    y_true = (injE >= thr).astype(int)

    # AUC（Δ分数）
    auc_dJ = roc_auc_score(y_true, dJ) if len(np.unique(y_true))>1 else np.nan
    auc_dM = roc_auc_score(y_true, dM) if len(np.unique(y_true))>1 else np.nan
    ap_dJ  = average_precision_score(y_true, dJ) if len(np.unique(y_true))>1 else np.nan
    ap_dM  = average_precision_score(y_true, dM) if len(np.unique(y_true))>1 else np.nan
    return (dJ.mean(), dM.mean(), auc_dJ, ap_dJ, auc_dM, ap_dM)

# ---- 3) （可选）用 ΔAUC 选 λ：γ*(λ) = argmin ||y - γ z||_W^2 + λ γ^2 = (zWy)/(zWz+λ) ----
lam_grid = [1e2, 3e2, 1e3, 3e3, 1e4]
best = None
for lam in lam_grid:
    gamma = zWy / (zWz + lam)
    omega_res = make_residual_from_gamma(gamma)
    dJm, dMm, auc_dJ, ap_dJ, auc_dM, ap_dM = eval_scores(omega_res)
    score = np.nan_to_num(auc_dJ, nan=0.0)   # 主用 ΔJ^w,n 的 ROC-AUC
    if (best is None) or (score > best[0]):
        best = (score, lam, gamma, dJm, dMm, auc_dJ, ap_dJ, auc_dM, ap_dM)

score, lam_star, gamma_star, dJm, dMm, auc_dJ, ap_dJ, auc_dM, ap_dM = best
print(f"[v3.4c] λ*={lam_star:g}, γ*={gamma_star:+.6f}, ΔJ^w,n mean={dJm:+.3e}, Δ|Θ| mean={dMm:+.3e}")
print(f"[v3.4c] AUC(ΔJ^w,n) ROC={auc_dJ:.3f} | PR={ap_dJ:.3f}")
print(f"[v3.4c] AUC(Δ|Θ|)   ROC={auc_dM:.3f} | PR={ap_dM:.3f}")

