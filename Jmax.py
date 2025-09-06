import numpy as np
import matplotlib.pyplot as plt

# ========= 参数（更“证书友好”） =========
SEED = 7
T     = 20.0
FS    = 400.0
SPLIT = 0.5               # 50% 锁定 + 50% 解锁
R0        = 1.0
amp_mod   = 0.05          # 幅度调制更小
amp_noise = 0.003         # 噪声更小 -> 锁定段 J≈0
unlock_omega0 = 1.0       # 解锁平均角速度
unlock_wobble = 0.25      # 解锁抖动
q0, lam   = 0.5, 5.0      # q(t)=q0+lam*sigma^2
SMOOTH_WIN = 201          # 平滑窗口大一些
N_SURR, ALPHA = 400, 0.95
TAU_OMEGA = 0.02          # 稳健几何锁门限 |omega|<tau

np.random.seed(SEED)

def smooth_ma(x, win=31):
    win = int(win)
    if win < 3: return x.copy()
    if win % 2 == 0: win += 1
    k = np.ones(win) / win
    return np.convolve(x, k, mode='same')

def central_grad(x, dt):
    g = np.empty_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2*dt)
    g[0]    = (x[1]  - x[0])   / dt
    g[-1]   = (x[-1] - x[-2])  / dt
    return g

def Jmax_integral(omega_hat, q, dt):
    integrand = (omega_hat**2) / q
    return np.trapezoid(integrand, dx=dt), integrand

def circular_shift(arr, k):
    k = int(k) % len(arr)
    if k == 0: return arr.copy()
    return np.concatenate([arr[-k:], arr[:-k]])

# ========= 合成数据（锁定段真·相位常数） =========
N  = int(T*FS)
t  = np.linspace(0, T, N, endpoint=False)
dt = t[1] - t[0]
split_idx = int(SPLIT*N)

theta = np.zeros(N)
R     = np.ones(N)*R0*(1.0 + amp_mod*np.sin(2*np.pi*0.1*t))
# Lock: 真·常相位
theta[:split_idx] = 0.35
# Unlock: 线性 + 抖动
theta[split_idx:] = theta[split_idx-1] + unlock_omega0*(t[split_idx:]-t[split_idx]) \
                    + unlock_wobble*np.sin(2*np.pi*0.5*(t[split_idx:]-t[split_idx]))

# 加观测噪声
x = R*np.cos(theta) + amp_noise*np.random.randn(N)
y = R*np.sin(theta) + amp_noise*np.random.randn(N)

# ========= 估计 R, theta, sigma, omega =========
R_hat   = np.sqrt(x**2 + y**2)
theta_h = np.unwrap(np.arctan2(y, x))

logR_s  = smooth_ma(np.log(np.clip(R_hat, 1e-8, None)), SMOOTH_WIN)
theta_s = smooth_ma(theta_h, SMOOTH_WIN)

sigma_hat = central_grad(logR_s, dt)   # (log R).dot
omega_hat = central_grad(theta_s, dt)  # theta.dot

# 锁定段均值去偏（抵消估计的小偏置/DC）
mu_lock  = np.mean(omega_hat[:split_idx])
omega_L0 = omega_hat[:split_idx] - mu_lock
omega_U0 = omega_hat[split_idx:] - mu_lock

# ========= q(t) 与 J_max =========
q = q0 + lam*(sigma_hat**2)
q = np.maximum(q, 1e-8)
q_L, q_U = q[:split_idx], q[split_idx:]

J_L, integ_L = Jmax_integral(omega_L0, q_L, dt)
J_U, integ_U = Jmax_integral(omega_U0, q_U, dt)

# 锁定段零假设阈值（循环移位）
J_surr = []
for _ in range(N_SURR):
    k = np.random.randint(0, len(omega_L0))
    Jtmp, _ = Jmax_integral(circular_shift(omega_L0, k), circular_shift(q_L, k), dt)
    J_surr.append(Jtmp)
J_surr = np.array(J_surr)
thr = np.quantile(J_surr, ALPHA)

# Holonomy（用新 API）
Theta_L = np.trapezoid(omega_L0, dx=dt)
Theta_U = np.trapezoid(omega_U0, dx=dt)

# 稳健几何锁占比（避免 K 的奇点）
frac_lock_geo = np.mean(np.abs(omega_L0) < TAU_OMEGA)
frac_unlk_geo = np.mean(np.abs(omega_U0) < TAU_OMEGA)

# ========= 打印 =========
print("==== SCI 证书（Rayleigh）与几何指标（v2） ====")
print(f"N={N}, dt={dt:.4f}s, 锁定/解锁 = {SPLIT*100:.1f}%/{(1-SPLIT)*100:.1f}%")
print(f"[锁定] J_max(L) = {J_L:.6e}   (零假设 {int(ALPHA*100)}% 阈值: {thr:.6e})")
print(f"[解锁] J_max(U) = {J_U:.6e}   (> 阈值 ? {'YES' if J_U>thr else 'NO'})")
print(f"[锁定] Holonomy Θ_L = {Theta_L:.6e} rad")
print(f"[解锁] Holonomy Θ_U = {Theta_U:.6e} rad")
print(f"[锁定] 稳健几何锁占比  Pr(|ω|<{TAU_OMEGA}) ≈ {100*frac_lock_geo:.2f}%")
print(f"[解锁] 稳健几何锁占比  Pr(|ω|<{TAU_OMEGA}) ≈ {100*frac_unlk_geo:.2f}%")
print("等价链（实验感知）：K=1  ↔  ω=0  ↔  J_max=0  （锁定段应同时满足，且 U 段显著大于阈值）")

# ========= 画图 =========
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0])

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, x, lw=1.0, label='x(t)')
ax1.plot(t, y, lw=1.0, label='y(t)', alpha=0.85)
ax1.axvline(t[split_idx], color='k', ls='--', alpha=0.6)
ax1.set_title('Observed x,y'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t, R_hat, lw=1.0, label='R_hat')
ax2_t = ax2.twinx()
ax2_t.plot(t, theta_h, lw=1.0, color='tab:orange', label='theta_hat')
ax2.axvline(t[split_idx], color='k', ls='--', alpha=0.6)
ax2.set_title('R (amp) & θ (phase)'); ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t[:split_idx], omega_L0, lw=1.0, label='omega (lock, debiased)')
ax3.plot(t[split_idx:], omega_U0, lw=1.0, label='omega (unlock, debiased)')
ax3.axhline(0, color='k', lw=0.7, alpha=0.6)
ax3.axvline(t[split_idx], color='k', ls='--', alpha=0.6)
ax3.set_title('ω (debiased)'); ax3.legend(); ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1, 1])
integrand = np.zeros_like(t)
integrand[:split_idx] = (omega_L0**2)/q_L
integrand[split_idx:] = (omega_U0**2)/q_U
ax4.plot(t, integrand, lw=1.0, label=r'$\omega^2/q$')
ax4.axvline(t[split_idx], color='k', ls='--', alpha=0.6)
ax4.set_title(r'Integrand of $\mathcal{J}_{\max}$'); ax4.legend(); ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[2, 0])
cum = np.cumsum(integrand)*dt
ax5.plot(t, cum, lw=1.2, color='tab:green', label='Cumulative J')
ax5.axvline(t[split_idx], color='k', ls='--', alpha=0.6)
ax5.set_title('Cumulative J (lock→unlock)'); ax5.legend(); ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(J_surr, bins=28, alpha=0.75, label='Null (lock, circular shifts)')
ax6.axvline(thr, color='r', lw=2.0, label=f'{int(ALPHA*100)}% threshold')
ax6.axvline(J_U, color='k', lw=2.0, ls='--', label='J_max (unlock)')
ax6.set_title('Null distribution & unlock J'); ax6.legend(); ax6.grid(True, alpha=0.3)

fig.suptitle('SCI: Geometry–Statistics Alignment  (K=1 ⇔ ω=0 ⇔ J=0 on lock)', fontsize=13)
plt.tight_layout()
plt.show()
