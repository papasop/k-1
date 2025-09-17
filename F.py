# ==================== ONE-CELL: Real Zeros + REALFAST + LUT + UTH + Ridge ====================
# 1) 生成 Riemann 非平凡零点（Hardy Z 扫描 + 二分精化）
# 2) 生成/复用 ζ 相位速率 (REALFAST + LUT 缓存)
# 3) 载入真实/正控/冒烟 物理速率，跑 UTH + ridge
# 4) 导出窗口级 ΔJ, Δ|maxΘ|, VR(ΔJ-only) 以及摘要与质检图
# ======================================================================

import os, math, numpy as np, pandas as pd, mpmath as mp, matplotlib.pyplot as plt
mp.mp.dps = 60
RNG = np.random.default_rng(20250911)

# ----------------------------- 可调参数 -----------------------------
# A. 零点扫描（越大越慢；200~300 通常几分钟内可完成）
Z_T_MAX        = 200.0      # 扫描上限 T（~200 约 80 个零点量级）
Z_DT_SCAN      = 0.25       # 扫描步长（越小越密，越慢）
Z_REFINE_ITERS = 60         # 二分最大迭代
Z_SAVE_PATH    = "/content/riemann_zeros.csv"

# B. ζ 相位速率（REALFAST）
REALFAST_H      = 3e-3      # 复合步长（2e-3~5e-3）
REALFAST_STRIDE = 12        # 稀疏插值步长（8/12/16）
DUR_SEC         = 3600.0    # 建议 1 小时以稳 CI/VR
DT_SEC          = 0.04
T0              = 0.0
KAPPA           = 1.0
AUTO_KAPPA      = False     # 有真实 omega_phys.csv 时可开启一次粗搜再固定

LUT_NPZ         = "/content/zeta_lut.npz"
LUT_CSV         = "/content/zeta_phase_field_realfast.csv"

# C. 物理速率来源（优先级：真实文件 > 正控 > 冒烟）
PHYS_CSV        = "/content/omega_phys.csv"
POSITIVE_CONTROL= True      # 没有真数据时，是否生成“正控”（掺入 ζ，马上能看到 VR>1）
POS_Z_WEIGHT    = 0.12      # 正控中 ζ 的权重 0.08~0.15

# D. UTH + Ridge
U_WIN, U_STEP   = 64.0, 32.0
CJ              = 0.02       # VR ≳ exp(-CJ * ΔJ)，若有 LGT 回归 cJ 可替换
BOOT_N          = 2000
LOWPASS_FR      = 0.10
SMOOTH_N        = 1          # 1=不平滑；>1 移动均值（奇数）

# ----------------------------- 工具函数 -----------------------------
def riemann_theta(t):
    # θ(t) = Im(log Γ(1/4 + i t/2)) - (t/2) log π
    z = mp.mpf('0.25') + 0.5j * mp.mpf(str(t))
    return mp.im(mp.log(mp.gamma(z))) - (mp.mpf(str(t))/2) * mp.log(mp.pi)

def hardy_Z(t):
    # Z(t) = e^{i θ(t)} ζ(1/2 + i t)，理论上实值；取 Re 以抑制数值噪声
    s = mp.mpf('0.5') + 1j*mp.mpf(str(t))
    th = riemann_theta(t)
    return mp.re(mp.exp(1j*th) * mp.zeta(s))

def bracket_root_Z(a, b, iters=60):
    fa = hardy_Z(a); fb = hardy_Z(b)
    if mp.sign(fa) == 0: return float(a)
    if mp.sign(fb) == 0: return float(b)
    if mp.sign(fa) == mp.sign(fb): return None
    lo, hi = mp.mpf(str(a)), mp.mpf(str(b))
    for _ in range(iters):
        mid = (lo+hi)/2
        fm  = hardy_Z(mid)
        if mp.sign(fm) == 0: return float(mid)
        if mp.sign(fm) == mp.sign(fa):
            lo, fa = mid, fm
        else:
            hi, fb = mid, fm
    return float((lo+hi)/2)

def scan_zeros(T_max, dt):
    ts = np.arange(0.0, T_max+dt, dt)
    zs = []
    z_prev = hardy_Z(ts[0])
    for k in range(1, len(ts)):
        z_cur = hardy_Z(ts[k])
        if mp.sign(z_prev) == 0:
            zs.append(ts[k-1])
        elif mp.sign(z_prev) != mp.sign(z_cur):
            r = bracket_root_Z(ts[k-1], ts[k], iters=Z_REFINE_ITERS)
            if r is not None: zs.append(r)
        z_prev = z_cur
    return np.array(zs, float)

def moving_mean(x, win=1):
    x = np.asarray(x, float)
    if win<=1: return x
    win=int(win) if int(win)%2==1 else int(win)+1
    pad=win//2; xpad=np.pad(x,(pad,pad),'edge'); ker=np.ones(win)/win
    return np.convolve(xpad, ker, 'valid')

def tau_star_1e(x, dt):
    x = np.asarray(x, float) - np.mean(x)
    n = 1<<(len(x)-1).bit_length()
    X = np.fft.rfft(x, n=n*2)
    ac = np.fft.irfft(X*np.conj(X))[:len(x)]
    ac = ac/ac[0] if ac[0]!=0 else ac
    thr = np.exp(-1.0)
    idx = np.argmax(ac<=thr) if np.any(ac<=thr) else 0
    if idx==0:
        area = np.trapz(ac[ac>0], dx=dt)
        return max(dt, area)
    return max(dt, idx*dt)

def sliding_indices(n, win, step):
    out=[]; i=0
    while i+win<=n: out.append((i,i+win)); i+=step
    if not out and n>0: out=[(0,n)]
    return out

def window_metrics(omega, dt):
    J = float(np.sum(omega*omega)*dt)
    theta = np.cumsum(omega)*dt
    M = float(np.max(np.abs(theta)))
    return J, M

def bootstrap_ci(x, nboot=2000, alpha=0.05):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x)==0: return (np.nan, np.nan, np.nan)
    n=len(x); rng=np.random.default_rng(20250911)
    boots=np.empty(nboot)
    for b in range(nboot):
        boots[b] = np.mean(x[rng.integers(0,n,size=n)])
    mean=float(np.mean(boots))
    lo  =float(np.percentile(boots, 100*alpha/2))
    hi  =float(np.percentile(boots, 100*(1-alpha/2)))
    if not (lo <= mean <= hi): mean=float(np.mean(x))  # 防呆
    return mean, lo, hi

def phase_scramble(x):
    x=np.asarray(x,float); n=len(x)
    X=np.fft.rfft(x); mag=np.abs(X); ang=np.angle(X)
    k=len(X); rand=RNG.uniform(-np.pi,np.pi,size=k); rand[0]=ang[0]
    if k>1 and (n%2==0): rand[-1]=ang[-1]
    Y=mag*np.exp(1j*rand); return np.fft.irfft(Y, n=n)

def low(x, frac=0.1):
    X=np.fft.rfft(x - np.mean(x)); kmax=max(1,int(len(X)*frac))
    F=np.zeros_like(X); F[:kmax]=X[:kmax]
    return np.fft.irfft(F, n=len(x))

# ----------------------------- 1) 生成零点 -----------------------------
if not os.path.exists(Z_SAVE_PATH):
    print(f"[ZEROS] scanning 0..{Z_T_MAX} with dt={Z_DT_SCAN} ...")
    gammas = scan_zeros(Z_T_MAX, Z_DT_SCAN)
    pd.DataFrame({"gamma":gammas}).to_csv(Z_SAVE_PATH, index=False)
    print(f"[ZEROS] saved {Z_SAVE_PATH}  zeros={len(gammas)}")
else:
    print(f"[ZEROS] found existing {Z_SAVE_PATH}")
    gammas = pd.read_csv(Z_SAVE_PATH)["gamma"].values

# ----------------------------- 2) ζ 相位速率（REALFAST + LUT） -----------------------------
t_sec  = np.arange(0.0, DUR_SEC, DT_SEC)
T_line = T0 + KAPPA*t_sec

def build_realfast(T_line, h=REALFAST_H, stride=REALFAST_STRIDE):
    T_sparse = T_line[::stride]
    omega_sparse=[]
    for T in T_sparse:
        s0 = mp.mpf('0.5') + 1j*mp.mpf(str(T))
        s1 = (mp.mpf('0.5') - mp.mpf(str(h))) + 1j*mp.mpf(str(T))
        F0 = mp.log(mp.zeta(s0))
        F1 = mp.log(mp.zeta(s1))
        val = (mp.re(F0) - mp.re(F1)) / mp.mpf(str(h))  # 近似 Re(ζ'/ζ)
        omega_sparse.append(float(val))
    omega_dense = np.interp(T_line, T_sparse, np.array(omega_sparse))
    return omega_dense

# 可选：有真实物理速率时，先粗搜 KAPPA 对齐低频
if AUTO_KAPPA and os.path.exists(PHYS_CSV):
    phys = pd.read_csv(PHYS_CSV)
    tcol = next((c for c in ["t_sec","time","t","sec"] if c in phys.columns), None)
    assert tcol and "omega_phys" in phys.columns, "omega_phys.csv 需含 t_sec/time/t/sec 与 omega_phys"
    t_max = min(t_sec.max(), phys[tcol].max())
    t_sec = t_sec[t_sec<=t_max]
    omega_phys_ref = np.interp(t_sec, phys[tcol].values, phys["omega_phys"].values)
    best=(None,None)
    for k in np.linspace(0.4, 2.5, 9):
        T_test = T0 + k*t_sec
        oz_tmp = build_realfast(T_test, h=REALFAST_H, stride=min(REALFAST_STRIDE, 16))
        a=low(omega_phys_ref, 0.1); b=low(oz_tmp, 0.1)
        num=np.dot(a-a.mean(), b-b.mean()); den=np.linalg.norm(a-a.mean())*np.linalg.norm(b-b.mean())
        corr = num/den if den>0 else 0.0
        if best[0] is None or corr>best[0]: best=(corr,k)
    KAPPA=float(best[1]); print(f"[KAPPA] auto ≈ {KAPPA:.4f} (low-freq corr≈{best[0]:.3f})")
    T_line = T0 + KAPPA*t_sec

if os.path.exists(LUT_NPZ):
    lut = np.load(LUT_NPZ)
    T_lut = lut["T_line"]; OZ_lut = lut["omega_zeta"]
    omega_zeta = np.interp(T_line, T_lut, OZ_lut)
    print(f"[LUT] loaded {LUT_NPZ} rows={len(T_lut)} -> interp to {len(T_line)}")
else:
    omega_zeta = build_realfast(T_line)
    np.savez(LUT_NPZ, T_line=T_line, omega_zeta=omega_zeta)
    pd.DataFrame(dict(t_sec=t_sec, T_line=T_line, omega_zeta_deg=omega_zeta)).to_csv(LUT_CSV, index=False)
    print(f"[LUT] saved {LUT_NPZ} and {LUT_CSV} rows={len(t_sec)}  (h={REALFAST_H}, stride={REALFAST_STRIDE})")

# ----------------------------- 3) 物理速率（真实/正控/冒烟） -----------------------------
if os.path.exists(PHYS_CSV):
    phys = pd.read_csv(PHYS_CSV)
    tcol = next((c for c in ["t_sec","time","t","sec"] if c in phys.columns), None)
    assert tcol and "omega_phys" in phys.columns, "omega_phys.csv 需含 t_sec/time/t/sec 与 omega_phys"
    t_max = min(t_sec.max(), phys[tcol].max())
    m = t_sec<=t_max
    t_sec = t_sec[m]; T_line=T_line[m]; omega_zeta=omega_zeta[m]
    omega_phys = np.interp(t_sec, phys[tcol].values, phys["omega_phys"].values)
    print("[PHYS] loaded real omega_phys.csv")
else:
    if POSITIVE_CONTROL:
        base = 0.8*np.sin(2*np.pi*0.03*t_sec) + 0.4*np.sin(2*np.pi*0.007*t_sec)
        omega_phys = POS_Z_WEIGHT*omega_zeta + base + 0.3*RNG.standard_normal(len(t_sec))
        pd.DataFrame({"t_sec":t_sec,"omega_phys":omega_phys}).to_csv(PHYS_CSV, index=False)
        print(f"[PHYS] generated POSITIVE CONTROL -> {PHYS_CSV} (ζ weight={POS_Z_WEIGHT})")
    else:
        base = 0.8*np.sin(2*np.pi*0.03*t_sec) + 0.4*np.sin(2*np.pi*0.007*t_sec)
        omega_phys = base + 0.3*RNG.standard_normal(len(t_sec))
        print("[PHYS] WARNING: fallback to SMOKE synthetic phys (no ζ), VR≈1 是预期")

# 可选平滑（两侧 padding）
if SMOOTH_N>1:
    omega_zeta = moving_mean(omega_zeta, SMOOTH_N)
    omega_phys = moving_mean(omega_phys, SMOOTH_N)
    n = min(len(omega_zeta), len(omega_phys))
    omega_zeta, omega_phys = omega_zeta[:n], omega_phys[:n]
    t_sec = t_sec[:n]; T_line=T_line[:n]

# ----------------------------- 4) UTH + ridge + 报告 -----------------------------
dt = float(np.median(np.diff(t_sec)))
tau_star = tau_star_1e(omega_phys, dt)
Omega_star = 1.0/tau_star
U_total = (t_sec[-1]-t_sec[0])/tau_star
Twin = U_WIN * tau_star
Tstep= U_STEP* tau_star
print(f"[UTH] tau*={tau_star:.6f}s, Omega*={Omega_star:.6f}rad/s | U_total={U_total:.1f}, U_win={U_WIN}, U_step={U_STEP}")
print(f"[UTH] Twin={Twin:.2f}s, Tstep={Tstep:.2f}s, dt={dt:.4f}s")

# Ridge（无截距）
X = omega_zeta.astype(float)
y = omega_phys.astype(float)
lam = 1e-3 * float(np.dot(X,X))
alpha_hat = float(np.dot(X,y)/(np.dot(X,X)+lam))
res = y - alpha_hat*X
scr = phase_scramble(X)
alpha_scr = float(np.dot(scr,y)/(np.dot(scr,scr)+lam))
res_scr = y - alpha_scr*scr

# 滑窗
win  = int(round(Twin/dt))
step = int(round(Tstep/dt))
wins = sliding_indices(len(t_sec), win, step)

rows=[]
for i0,i1 in wins:
    sl=slice(i0,i1); tb,te=t_sec[i0], t_sec[i1-1]
    Jb,Mb = window_metrics(y[sl], dt)
    Jr,Mr = window_metrics(res[sl], dt)
    Js,Ms = window_metrics(res_scr[sl], dt)
    rows.append(dict(
        t_center_sec=0.5*(tb+te), left=tb, right=te,
        J_baseline=Jb, M_baseline=Mb,
        J_ridge=Jr,   M_ridge=Mr,
        J_scram=Js,   M_scram=Ms,
        dJ_ridge=Jr-Jb, dM_ridge=Mr-Mb,
        dJ_scram=Js-Jb, dM_scram=Ms-Mb
    ))
wm = pd.DataFrame(rows)

m_dJ, lo_dJ, hi_dJ = bootstrap_ci(wm["dJ_ridge"].values, BOOT_N)
m_dM, lo_dM, hi_dM = bootstrap_ci(wm["dM_ridge"].values, BOOT_N)
wm["VR_gain_from_dJ_ridge"] = np.exp(-CJ*wm["dJ_ridge"].values)
wm["VR_gain_from_dJ_scram"] = np.exp(-CJ*wm["dJ_scram"].values)
med_gain = float(np.median(wm["VR_gain_from_dJ_ridge"]))
p90_gain = float(np.percentile(wm["VR_gain_from_dJ_ridge"], 90))
p10_gain = float(np.percentile(wm["VR_gain_from_dJ_ridge"], 10))

# 保存
wm_path  = "/content/uth_window_metrics.csv"
sum_path = "/content/uth_ablation_summary.csv"
par_path = "/content/uth_params.txt"
wm.to_csv(wm_path, index=False)
pd.DataFrame([
    dict(arm="ridge_only", alpha_hat=alpha_hat,
         mean_dJ=m_dJ, lo_dJ=lo_dJ, hi_dJ=hi_dJ,
         mean_dM=m_dM, lo_dM=lo_dM, hi_dM=hi_dM,
         median_vr_factor=med_gain),
    dict(arm="scrambled", alpha_hat=alpha_scr,
         mean_dJ=float(np.mean(wm["dJ_scram"])), lo_dJ=np.nan, hi_dJ=np.nan,
         mean_dM=float(np.mean(wm["dM_scram"])), lo_dM=np.nan, hi_dM=np.nan,
         median_vr_factor=float(np.median(wm["VR_gain_from_dJ_scram"])))
]).to_csv(sum_path, index=False)
with open(par_path,"w") as f:
    f.write(f"[ZEROS] T_max={Z_T_MAX}, dt={Z_DT_SCAN}, zeros={len(gammas)}\n")
    f.write(f"[REALFAST] h={REALFAST_H}, stride={REALFAST_STRIDE}, KAPPA={KAPPA}\n")
    f.write(f"[UTH] tau*={tau_star:.6f}, Omega*={Omega_star:.6f}, U_total={U_total:.2f}, "
            f"U_win={U_WIN}, U_step={U_STEP}, Twin={Twin:.2f}, Tstep={Tstep:.2f}, dt={dt:.4f}, windows={len(wins)}\n")
    f.write(f"[Ridge] alpha_hat={alpha_hat:.6f}, alpha_scr={alpha_scr:.6f}, lambda={lam:.3e}\n")
    f.write(f"[VR] cJ={CJ:.4f}; median={med_gain:.4f}; p90={p90_gain:.4f}; p10={p10_gain:.4f}\n")

print("\n=== SUMMARY (Real Zeros + REALFAST + LUT + UTH) ===")
print(f"[Zeros] count={len(gammas)}  saved={Z_SAVE_PATH}")
print(f"[LUT]    path={LUT_NPZ} ({'present' if os.path.exists(LUT_NPZ) else 'new'})")
print(f"[UTH]    windows={len(wins)}")
print(f"[Ridge]  alpha_hat={alpha_hat:.6f}  alpha_scr={alpha_scr:.6f}  lambda={lam:.3e}")
print(f"[ΔJ]     mean={m_dJ:.4f}  95% CI [{lo_dJ:.4f}, {hi_dJ:.4f}]")
print(f"[Δ|maxΘ|] mean={m_dM:.4f}  95% CI [{lo_dM:.4f}, {hi_dM:.4f}]")
print(f"[VR ΔJ-only] median={med_gain:.4f}  P90={p90_gain:.4f}  P10={p10_gain:.4f}")
print("Saved:", wm_path, sum_path, par_path)

# 低频质检
plt.figure(figsize=(7,3))
plt.plot(t_sec, low(omega_zeta, LOWPASS_FR), label="zeta_low")
plt.plot(t_sec, low(omega_phys, LOWPASS_FR), label="phys_low")
plt.legend(); plt.grid(True); plt.title("Low-freq components"); plt.xlabel("t_sec"); plt.tight_layout(); plt.show()
# =======================================================================================
# === VR 稳健后处理（基于你刚才的输出）===
import re, numpy as np, pandas as pd, matplotlib.pyplot as plt

wm = pd.read_csv("/content/uth_window_metrics.csv")
with open("/content/uth_params.txt") as f:
    txt = f.read()

# 解析 Twin 与 cJ（脚本已在 params 里写了）
Twin = float(re.search(r" Twin=([0-9.]+)", txt).group(1))
cJ   = float(re.search(r"cJ=([0-9.]+)", txt).group(1))

dJ = wm["dJ_ridge"].to_numpy(float)

# 1) 原始下界
VR_raw  = np.exp(-cJ * dJ)

# 2) 按窗长归一（把 ΔJ 变“每窗尺度”）
VR_rate = np.exp(-(cJ / Twin) * dJ)

# 3) 截尾稳健（抑制极端窗）
lo, hi = np.percentile(dJ, [1, 99])
dJ_win = np.clip(dJ, lo, hi)
VR_raw_win = np.exp(-cJ * dJ_win)

def stats(v):
    return (float(np.median(v)),
            float(np.percentile(v,10)),
            float(np.percentile(v,90)))

print(f"[INFO] Twin={Twin:.2f}s  cJ={cJ:.4f}")
print("VR_raw     (median, P10, P90):", stats(VR_raw))
print("VR_rate    (median, P10, P90):", stats(VR_rate))
print("VR_raw_win (median, P10, P90):", stats(VR_raw_win))

out = pd.DataFrame({
    "VR_raw": VR_raw,
    "VR_rate": VR_rate,
    "VR_raw_win": VR_raw_win,
})
out.to_csv("/content/uth_vr_postprocessed.csv", index=False)
print("Saved /content/uth_vr_postprocessed.csv")

# 小图看一眼尾部被抑制效果（对比 log VR）
plt.figure(figsize=(6,3))
plt.hist(np.log(VR_raw+1e-12), bins=60, alpha=0.5, label="log VR_raw")
plt.hist(np.log(VR_raw_win+1e-12), bins=60, alpha=0.5, label="log VR_raw_win")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
