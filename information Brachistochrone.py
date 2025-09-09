# Colab-ready, single-cell notebook for §5.1 verification (+ new time-domain formulas)
# - No file I/O
# - Prints + matplotlib figures only
# - Verifies: Theorem 5 (continuous), Gamma-coordinate derivatives/curvature,
#             finite-window (truncated) strong-converse, discrete (geometric) strong-converse,
#             Corollary 6 empirical equivalence (LOCK/UNLOCK),
#             NEW: time-domain brachistochrone certificate, phase uniformization,
#                  semi-sine/cosine laws, acceleration-circle identity, curvature law,
#                  geometry recovery from time-domain data.

import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

mp.mp.dps = 50  # precision

# ---------------------------
# Helper: continuous densities
# ---------------------------
def p_exp(t, mu):
    return (1.0/mu) * mp.e**(-t/mu) if t>=0 else mp.mpf('0')

def p_gamma(t, K, beta):  # shape K, rate beta
    if t<0: 
        return mp.mpf('0')
    return (beta**K / mp.gamma(K)) * (t**(K-1)) * mp.e**(-beta*t)

# Entropy and KL for continuous on [0, ∞) via mpmath.quad
def H_cont(p, args=()):
    f = lambda tt: p(tt, *args) * mp.log(max(p(tt, *args), mp.mpf('1e-100')))
    val = - mp.quad(f, [0, mp.inf])
    return float(val)

def KL_cont(p, q, par_p=(), par_q=()):
    f = lambda tt: p(tt, *par_p) * ( mp.log(max(p(tt, *par_p), mp.mpf('1e-100'))) -
                                     mp.log(max(q(tt, *par_q), mp.mpf('1e-100'))) )
    val = mp.quad(f, [0, mp.inf])
    return float(val)

# ---------------------------
# Gamma-coordinate analytics
# ---------------------------
psi  = mp.digamma
psi1 = mp.polygamma  # polygamma(n, x), so trigamma = polygamma(1, x)

def H_mu_gamma(K, mu):
    return float(K - mp.log(K) + mp.log(mu) + mp.log(mp.gamma(K)) + (1-K)*psi(K))

def H_mu_prime(K):
    # H'_mu(K) = 1 - 1/K + (1-K) psi'(K); psi'(K) = polygamma(1, K)
    return float(1 - 1.0/K + (1-K)*psi1(1, K))

def H_mu_second(K):
    # H''_mu(K) = 1/K^2 - psi'(K) + (1-K) psi''(K); psi''(K) = polygamma(2, K)
    return float(1.0/(K*K) - psi1(1, K) + (1-K)*psi1(2, K))

# ---------------------------
# Finite window (truncated) helpers on [0, gamma]
# ---------------------------
def p_trunc_exp(t, lam, gamma):
    if t<0 or t>gamma: return mp.mpf('0')
    Z = 1 - mp.e**(-lam*gamma)
    return (lam * mp.e**(-lam*t)) / Z

def mean_trunc_exp(lam, gamma):
    f = lambda tt: tt * p_trunc_exp(tt, lam, gamma)
    return mp.quad(f, [0, gamma])

def find_lambda_for_mu(mu, gamma):
    f = lambda lam: mean_trunc_exp(lam, gamma) - mu
    lam0 = 1.0/mu
    sol = mp.findroot(f, lam0)
    return float(sol)

def H_cont_trunc(p, args=(), a=0.0, b=1.0):
    f = lambda tt: p(tt, *args) * mp.log(max(p(tt, *args), mp.mpf('1e-100')))
    val = - mp.quad(f, [a, b])
    return float(val)

def KL_cont_trunc(p, q, par_p=(), par_q=(), a=0.0, b=1.0):
    f = lambda tt: p(tt, *par_p) * ( mp.log(max(p(tt, *par_p), mp.mpf('1e-100'))) -
                                     mp.log(max(q(tt, *par_q), mp.mpf('1e-100'))) )
    val = mp.quad(f, [a, b])
    return float(val)

def p_trunc_gamma(t, K, beta, gamma):
    if t<0 or t>gamma: return mp.mpf('0')
    un = (beta**K / mp.gamma(K)) * (t**(K-1)) * mp.e**(-beta*t)
    Z = mp.gammainc(K, 0, beta*gamma) / mp.gamma(K)
    return un / Z

def mean_trunc_gamma(K, beta, gamma):
    f = lambda tt: tt * p_trunc_gamma(tt, K, beta, gamma)
    return mp.quad(f, [0, gamma])

def find_beta_for_trunc_gamma_mean(K, mu, gamma):
    f = lambda b: mean_trunc_gamma(K, b, gamma) - mu
    b0 = K/mu
    sol = mp.findroot(f, b0)
    return float(sol)

# ---------------------------
# Discrete (geometric) helpers
# ---------------------------
def H_discrete(P):
    P = np.asarray(P, dtype=float)
    P = P[P>0]
    return float(-np.sum(P * np.log(P)))

def KL_discrete(P, Q):
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    mask = (P>0)
    return float(np.sum(P[mask] * (np.log(P[mask]) - np.log(Q[mask]))))

def geometric_pmf(mu, n_max=10000):
    p = 1.0/mu
    n = np.arange(1, n_max+1)
    P = p * (1-p)**(n-1)
    P /= P.sum()
    return P

def two_point_same_mean(mu, m):
    alpha = (m - mu)/(m - 1.0)
    P = np.zeros(m+1, dtype=float)
    P[1] = alpha
    P[m] = 1.0 - alpha
    return P[1:]

# ---------------------------
# Numerical diff helpers
# ---------------------------
def finite_diff_np(x, t):
    dt = np.diff(t)
    dx = np.diff(x)
    v = np.empty_like(x)
    v[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
    v[0]    = dx[0]/dt[0]
    v[-1]   = dx[-1]/dt[-1]
    return v

# ---------------------------
# PART I: Theorem 5 (continuous, untruncated) & derivatives
# ---------------------------
print("=== §5.1 Theorem 5: Max-entropy-rate selects exponential (continuous) ===")
mu = 1.0
Kc = 1.5
beta_c = Kc/mu
H_exp = H_cont(p_exp, (mu,))
H_comp = H_cont(p_gamma, (Kc, beta_c))
KL_comp_exp = KL_cont(p_gamma, p_exp, (Kc, beta_c), (mu,))

print(f"H[Exp(mu)]: {H_exp:.12f}   (should be 1 + ln mu = 1.0)")
print(f"H[Gamma(K={Kc})]: {H_comp:.12f}")
print(f"H[Exp] - H[Gamma]: {H_exp - H_comp:.12f}   vs   KL(Gamma||Exp): {KL_comp_exp:.12f}")

R_exp = H_exp/mu
R_comp = H_comp/mu
print(f"Entropy rate R[Exp]: {R_exp:.12f}   R[Gamma]: {R_comp:.12f}   (Exp should be larger)")

print("\n=== Gamma-coordinate derivatives/curvature at K=1 (mu=1) ===")
H1  = H_mu_gamma(1.0, mu)
Hp1 = H_mu_prime(1.0)
Hpp1= H_mu_second(1.0)
print(f"H(1)   = {H1:.12f}")
print(f"H'(1)  = {Hp1:.12e}  (~ 0)")
print(f"H''(1) = {Hpp1:.15f}  (should be 1 - pi^2/6 = {-np.pi**2/6 + 1:.15f})")

Ks = np.linspace(0.3, 3.0, 200)
Hp_vals = np.array([H_mu_prime(k) for k in Ks])
sign_change_idx = [int(i) for i in np.where(np.sign(Hp_vals[:-1]) != np.sign(Hp_vals[1:]))[0]]
print(f"\nSign changes in H'(K) across [0.3,3.0]: indices = {sign_change_idx} (expect exactly one near K=1)")

H_vals = np.array([H_mu_gamma(k, mu) for k in Ks])
plt.figure()
plt.plot(Ks, H_vals)
plt.axvline(1.0, ls='--')
plt.title("H_mu(K) with mu=1 (unique maximum at K=1)")
plt.xlabel("K"); plt.ylabel("H_mu(K)")
plt.show()

plt.figure()
plt.plot(Ks, Hp_vals)
plt.axhline(0.0, ls='--')
plt.axvline(1.0, ls='--')
plt.title("H'_mu(K) sign flip at K=1")
plt.xlabel("K"); plt.ylabel("H'_mu(K)")
plt.show()

# ---------------------------
# PART II: Finite window (truncated) strong-converse
# ---------------------------
print("\n=== Finite window [0, gamma]: truncated exponential is max-entropy (same mean) ===")
mu = 0.7
gamma = 2.0
lam = find_lambda_for_mu(mu, gamma)

H_q = H_cont_trunc(p_trunc_exp, (lam, gamma), 0.0, gamma)

K_alt = 1.5
beta_alt = find_beta_for_trunc_gamma_mean(K_alt, mu, gamma)
H_p = H_cont_trunc(p_trunc_gamma, (K_alt, beta_alt, gamma), 0.0, gamma)
KL_pq = KL_cont_trunc(p_trunc_gamma, p_trunc_exp, (K_alt, beta_alt, gamma), (lam, gamma), 0.0, gamma)

print(f"lambda (trunc-exp mean=mu): {lam:.6f},   H[q_mu,gamma]={H_q:.12f}")
print(f"Trunc-Gamma(K={K_alt}, beta={beta_alt:.6f})  H[p]={H_p:.12f}")
print(f"H[q] - H[p] = {H_q - H_p:.12f}   vs   KL(p||q) = {KL_pq:.12f}   (should match)")

# ---------------------------
# PART III: Discrete (geometric) strong-converse
# ---------------------------
print("\n=== Discrete (N>=1): geometric is max-entropy (same mean) ===")
mu_d = 5.0
G_full = geometric_pmf(mu_d, n_max=200000)
H_G_exact = H_discrete(G_full)

m = 10
alpha = (m - mu_d)/(m - 1.0)
H_P = -(alpha*np.log(alpha) + (1-alpha)*np.log(1-alpha))

p_geo = 1.0/mu_d
Q1 = p_geo
Qm = p_geo * (1-p_geo)**(m-1)
KL_exact = alpha*np.log(alpha/Q1) + (1-alpha)*np.log((1-alpha)/Qm)

print(f"H[Geom(mu={mu_d})] ~= {H_G_exact:.12f},   H[Two-point]={H_P:.12f}")
print(f"H[Geom]-H[Two] = {H_G_exact - H_P:.12f}   vs   KL(P||Geom) = {KL_exact:.12f}   (should match)")

# ---------------------------
# PART IV: Corollary 6 empirical (LOCK/UNLOCK)
# ---------------------------
print("\n=== Corollary 6 empirical LOCK/UNLOCK ===")

def synth_signal_np(T=20.0, N=6000, sigma=0.5, omega_fun=None, theta0=np.pi/4):
    t = np.linspace(0, T, N)
    if omega_fun is None:
        omega = np.zeros_like(t)
    else:
        omega = omega_fun(t)
    theta = theta0 + np.cumsum(omega) * (t[1]-t[0])
    R = np.exp(sigma * t)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    Rrate = finite_diff_np(np.log(R), t)
    thetadot = finite_diff_np(theta, t)
    den = (Rrate - thetadot * np.tan(theta))
    mask = (np.abs(np.sin(theta)*np.cos(theta)) > 1e-2) & (np.abs(den) > 1e-4)
    K = np.empty_like(t); K[:] = np.nan
    K[mask] = (Rrate[mask] + thetadot[mask] * 1/np.tan(theta[mask])) / den[mask]
    return t, x, y, theta, R, Rrate, thetadot, K, mask

def Jmax_from_omega_hat_np(t, omega_hat, q=None):
    if q is None:
        q = np.ones_like(omega_hat)
    return float(np.trapezoid((omega_hat**2)/q, t))

def hazard_estimate_np(Tsamples, bins=60):
    T = np.asarray(Tsamples)
    T = T[T>0]
    t_edges = np.linspace(0, np.percentile(T, 99.0), bins+1)
    widths = np.diff(t_edges)
    centers = 0.5*(t_edges[1:]+t_edges[:-1])
    S = np.array([(T >= t_edges[i]).mean() for i in range(bins)])
    counts, _ = np.histogram(T, bins=t_edges)
    n_at_risk = S * len(T)
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = np.where(n_at_risk>0, (counts / n_at_risk) / widths, np.nan)
    return centers, lam

def logsurvival_np(Tsamples, grid=200):
    T = np.sort(Tsamples)
    tmax = np.percentile(T, 99.0)
    tt = np.linspace(0, tmax, grid)
    S = np.array([(T >= u).mean() for u in tt])
    S = np.clip(S, 1e-12, 1.0)
    return tt, np.log(S)

def linfit_np(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    theta, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = theta*x + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-12
    R2 = 1 - ss_res/ss_tot
    return theta, b, R2

# LOCK
mu = 1.0; sigma = 0.5
tL, xL, yL, thetaL, RL, RrateL, omegaL, KL, maskL = synth_signal_np(
    T=20.0, N=6000, sigma=sigma, omega_fun=lambda t: 0*t, theta0=np.pi/4)
Jmax_L = Jmax_from_omega_hat_np(tL, omegaL)
rng = np.random.default_rng(7)
N_T = 200000
T_lock = rng.exponential(scale=mu, size=N_T)
centers_L, haz_L = hazard_estimate_np(T_lock, bins=60)
tt_LS, lnS_L = logsurvival_np(T_lock, grid=200)
_, _, R2_L = linfit_np(tt_LS, lnS_L)

print("=== LOCK ===")
print(f"mean(|K-1|) on valid sector: {np.nanmean(np.abs(KL[maskL]-1)):.3e}")
print(f"J_max: {Jmax_L:.3e}")
print(f"log-survival R^2 (Exp): {R2_L:.6f}")

plt.figure(); plt.plot(tL[maskL][::10], KL[maskL][::10])
plt.title("LOCK: K(t) (subsampled)"); plt.xlabel("t"); plt.ylabel("K(t)"); plt.show()
plt.figure(); plt.plot(tL, omegaL); plt.title("LOCK: omega(t) == 0"); plt.xlabel("t"); plt.ylabel("omega(t)"); plt.show()
plt.figure(); plt.plot(centers_L, haz_L); plt.title("LOCK: Hazard ~ const"); plt.xlabel("t"); plt.ylabel("lambda_hat(t)"); plt.show()
plt.figure(); plt.plot(tt_LS, lnS_L); plt.title("LOCK: log S(t) linear"); plt.xlabel("t"); plt.ylabel("log S(t)"); plt.show()

# UNLOCK
a, b = 0.4, 0.8
omega_fun_U = lambda t: a*np.sin(b*t)
tU, xU, yU, thetaU, RU, RrateU, omegaU, KU, maskU = synth_signal_np(
    T=20.0, N=6000, sigma=sigma, omega_fun=omega_fun_U, theta0=np.pi/4)
Jmax_U = Jmax_from_omega_hat_np(tU, omegaU)
Kg = 1.5; theta_scale = mu / Kg
T_unlock = rng.gamma(shape=Kg, scale=theta_scale, size=N_T)
centers_U, haz_U = hazard_estimate_np(T_unlock, bins=60)
tt_US, lnS_U = logsurvival_np(T_unlock, grid=200)
_, _, R2_U = linfit_np(tt_US, lnS_U)

print("\n=== UNLOCK ===")
print(f"mean(|K-1|) on valid sector: {np.nanmean(np.abs(KU[maskU]-1)):.3e}")
print(f"J_max: {Jmax_U:.3e}")
print(f"log-survival R^2 (Gamma K={Kg}): {R2_U:.6f}")

plt.figure(); plt.plot(tU[maskU][::10], KU[maskU][::10])
plt.title("UNLOCK: K(t) (subsampled)"); plt.xlabel("t"); plt.ylabel("K(t)"); plt.show()
plt.figure(); plt.plot(tU, omegaU); plt.title("UNLOCK: omega(t)=a sin(bt)"); plt.xlabel("t"); plt.ylabel("omega(t)"); plt.show()
plt.figure(); plt.plot(centers_U, haz_U); plt.title(f"UNLOCK: Hazard (Gamma K={Kg})"); plt.xlabel("t"); plt.ylabel("lambda_hat(t)"); plt.show()
plt.figure(); plt.plot(tt_US, lnS_U); plt.title("UNLOCK: log S deviates"); plt.xlabel("t"); plt.ylabel("log S(t)"); plt.show()

tol_K = 5e-3
is_K1_L = np.nanmean(np.abs(KL[maskL]-1)) < tol_K
is_w0_L = np.allclose(omegaL, 0.0, atol=1e-12)
is_J0_L = Jmax_L < 1e-8
is_exp_L = R2_L > 0.999

is_K1_U = np.nanmean(np.abs(KU[maskU]-1)) < tol_K
is_w0_U = np.allclose(omegaU, 0.0, atol=1e-12)
is_J0_U = Jmax_U < 1e-8
is_exp_U = R2_U > 0.999

print("\n=== Corollary 6 empirical check ===")
print(f"LOCK:   [K=1? {is_K1_L}] [omega==0? {is_w0_L}] [J_max==0? {is_J0_L}] [Exp(mu)? {is_exp_L}]")
print(f"UNLOCK: [K=1? {is_K1_U}] [omega==0? {is_w0_U}] [J_max==0? {is_J0_U}] [Exp(mu)? {is_exp_U}]")

print("\nAll sections up to Corollary 6 executed.")

# ---------------------------
# PART V (NEW): Time-domain brachistochrone verification
# ---------------------------
print("\n=== NEW: Time-domain brachistochrone certificate & geometry recovery ===")

g = 9.81
r_true = 1.25         # cycloid generator radius
Delta_y_true = 2.0 * r_true
Omega_true = 0.5 * np.sqrt(g / r_true)
t_end = np.pi * np.sqrt(r_true / g)   # bottom time for phi from 0 to pi

# time grid
N = 4000
t = np.linspace(0.0, t_end, N)

# analytic signals on optimal motion
v = 2.0 * np.sqrt(g * r_true) * np.sin(Omega_true * t)           # semi-sine
a_t = g * np.cos(Omega_true * t)                                  # semi-cosine
j = - g * Omega_true * np.sin(Omega_true * t)                     # jerk
res = j + (Omega_true**2) * v                                     # certificate residual

# residual metrics
res_L2 = np.trapezoid(res**2, t)
res_Linf = np.max(np.abs(res))

# acceleration-circle identity: (a_t/g)^2 + (j/(Omega g))^2 == 1
X = a_t / g
Y = j / (Omega_true * g)
acc_circle_err = np.max(np.abs(X**2 + Y**2 - 1.0))

# curvature law: kappa(t) = 1/(4 r sin(Omega t)); check via a_n = v^2 kappa
eps = 1e-6
mask = (np.sin(Omega_true * t) > eps)
kappa = np.zeros_like(t)
kappa[mask] = 1.0 / (4.0 * r_true * np.sin(Omega_true * t[mask]))
a_n_from_kappa = np.zeros_like(t)
a_n_from_kappa[mask] = (v[mask]**2) * kappa[mask]
a_n_true = g * np.sin(Omega_true * t)
curvature_max_rel_err = np.max(np.abs(a_n_from_kappa[mask] - a_n_true[mask]) / np.maximum(1e-12, np.abs(a_n_true[mask])))

# phase uniformization: dt/dphi = sqrt(r/g) const
# Using phi/2 = Omega t  => dphi/dt = 2 Omega = sqrt(g/r)
phi = 2.0 * Omega_true * t
dphi_dt = np.gradient(phi, t)
phase_uniform_const = np.sqrt(g / r_true)
phase_uniform_Linf = np.max(np.abs(dphi_dt - phase_uniform_const))

# geometry recovery from time-domain
# (1) from linear regression on j ~ - (Omega^2) v  ==> Omega^2_hat = - <j, v> / <v, v>
Omega2_hat = - np.trapz(j * v, t) / np.trapz(v * v, t)
Omega_hat = np.sqrt(max(Omega2_hat, 0.0))
r_hat_from_Omega = g / (4.0 * Omega2_hat)
# (2) from vmax: r = vmax^2/(4g)
vmax = np.max(v)
r_hat_from_vmax = (vmax**2) / (4.0 * g)
# (3) from total time t_end: r = g (t_end/pi)^2
r_hat_from_tend = g * (t_end / np.pi)**2

print(f"g = {g:.4f}, r_true = {r_true:.6f}, Delta_y_true = {Delta_y_true:.6f}")
print(f"Omega_true^2 = {Omega_true**2:.9f}  (should equal g/(4r))")
print(f"Certificate residual: L2 = {res_L2:.3e}, Linf = {res_Linf:.3e}")
print(f"Acceleration-circle max error: {acc_circle_err:.3e}")
print(f"Curvature law max relative error: {curvature_max_rel_err:.3e}")
print(f"Phase uniformization Linf error: {phase_uniform_Linf:.3e}")
print(f"Recovered r from Omega: {r_hat_from_Omega:.9f}  (rel.err = {(r_hat_from_Omega/r_true - 1):+.3e})")
print(f"Recovered r from vmax:  {r_hat_from_vmax:.9f}  (rel.err = {(r_hat_from_vmax/r_true - 1):+.3e})")
print(f"Recovered r from t_end: {r_hat_from_tend:.9f}  (rel.err = {(r_hat_from_tend/r_true - 1):+.3e})")

# Plots for the new formulas
plt.figure(); plt.plot(t, v); plt.title("v(t) = 2 sqrt(g r) sin(Omega t)"); plt.xlabel("t"); plt.ylabel("v(t)"); plt.show()
plt.figure(); plt.plot(t, a_t); plt.title("a_t(t) = g cos(Omega t)"); plt.xlabel("t"); plt.ylabel("a_t(t)"); plt.show()
plt.figure(); plt.plot(t, res); plt.title("Certificate residual: j + Omega^2 v"); plt.xlabel("t"); plt.ylabel("residual"); plt.show()

# curvature plot avoiding singularity near t=0
plt.figure(); 
idx = np.where(mask)[0]
plt.plot(t[idx], a_n_from_kappa[idx] - a_n_true[idx])
plt.title("a_n from kappa minus true a_n (should be ~ 0)")
plt.xlabel("t"); plt.ylabel("difference"); plt.show()

# acceleration-circle identity plot
plt.figure(); plt.plot(t, X**2 + Y**2); plt.title("(a_t/g)^2 + (j/(Omega g))^2"); plt.xlabel("t"); plt.ylabel("sum"); plt.show()

print("\nNEW section done. This verifies:")
print("  • Phase uniformization: dphi/dt = sqrt(g/r) (constant)")
print("  • Semi-sine velocity and semi-cosine tangential acceleration")
print("  • One-line certificate: j + Omega^2 v = 0 (residual ~ 0)")
print("  • Acceleration-circle identity: (a_t/g)^2 + (j/(Omega g))^2 = 1")
print("  • Curvature law: kappa(t) = 1 / (4 r sin(Omega t))")
print("  • Geometry recovery: r from Omega, vmax, and total time t_end")
# Colab-ready, single-cell notebook for §5.1 verification (+ new time-domain formulas)
# - No file I/O
# - Prints + matplotlib figures only
# - Verifies: Theorem 5 (continuous), Gamma-coordinate derivatives/curvature,
#             finite-window (truncated) strong-converse, discrete (geometric) strong-converse,
#             Corollary 6 empirical equivalence (LOCK/UNLOCK),
#             NEW: time-domain brachistochrone certificate, phase uniformization,
#                  semi-sine/cosine laws, acceleration-circle identity, curvature law,
#                  geometry recovery from time-domain data.

import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

mp.mp.dps = 50  # precision

# ---------------------------
# Helper: continuous densities
# ---------------------------
def p_exp(t, mu):
    return (1.0/mu) * mp.e**(-t/mu) if t>=0 else mp.mpf('0')

def p_gamma(t, K, beta):  # shape K, rate beta
    if t<0: 
        return mp.mpf('0')
    return (beta**K / mp.gamma(K)) * (t**(K-1)) * mp.e**(-beta*t)

# Entropy and KL for continuous on [0, ∞) via mpmath.quad
def H_cont(p, args=()):
    f = lambda tt: p(tt, *args) * mp.log(max(p(tt, *args), mp.mpf('1e-100')))
    val = - mp.quad(f, [0, mp.inf])
    return float(val)

def KL_cont(p, q, par_p=(), par_q=()):
    f = lambda tt: p(tt, *par_p) * ( mp.log(max(p(tt, *par_p), mp.mpf('1e-100'))) -
                                     mp.log(max(q(tt, *par_q), mp.mpf('1e-100'))) )
    val = mp.quad(f, [0, mp.inf])
    return float(val)

# ---------------------------
# Gamma-coordinate analytics
# ---------------------------
psi  = mp.digamma
psi1 = mp.polygamma  # polygamma(n, x), so trigamma = polygamma(1, x)

def H_mu_gamma(K, mu):
    return float(K - mp.log(K) + mp.log(mu) + mp.log(mp.gamma(K)) + (1-K)*psi(K))

def H_mu_prime(K):
    # H'_mu(K) = 1 - 1/K + (1-K) psi'(K); psi'(K) = polygamma(1, K)
    return float(1 - 1.0/K + (1-K)*psi1(1, K))

def H_mu_second(K):
    # H''_mu(K) = 1/K^2 - psi'(K) + (1-K) psi''(K); psi''(K) = polygamma(2, K)
    return float(1.0/(K*K) - psi1(1, K) + (1-K)*psi1(2, K))

# ---------------------------
# Finite window (truncated) helpers on [0, gamma]
# ---------------------------
def p_trunc_exp(t, lam, gamma):
    if t<0 or t>gamma: return mp.mpf('0')
    Z = 1 - mp.e**(-lam*gamma)
    return (lam * mp.e**(-lam*t)) / Z

def mean_trunc_exp(lam, gamma):
    f = lambda tt: tt * p_trunc_exp(tt, lam, gamma)
    return mp.quad(f, [0, gamma])

def find_lambda_for_mu(mu, gamma):
    f = lambda lam: mean_trunc_exp(lam, gamma) - mu
    lam0 = 1.0/mu
    sol = mp.findroot(f, lam0)
    return float(sol)

def H_cont_trunc(p, args=(), a=0.0, b=1.0):
    f = lambda tt: p(tt, *args) * mp.log(max(p(tt, *args), mp.mpf('1e-100')))
    val = - mp.quad(f, [a, b])
    return float(val)

def KL_cont_trunc(p, q, par_p=(), par_q=(), a=0.0, b=1.0):
    f = lambda tt: p(tt, *par_p) * ( mp.log(max(p(tt, *par_p), mp.mpf('1e-100'))) -
                                     mp.log(max(q(tt, *par_q), mp.mpf('1e-100'))) )
    val = mp.quad(f, [a, b])
    return float(val)

def p_trunc_gamma(t, K, beta, gamma):
    if t<0 or t>gamma: return mp.mpf('0')
    un = (beta**K / mp.gamma(K)) * (t**(K-1)) * mp.e**(-beta*t)
    Z = mp.gammainc(K, 0, beta*gamma) / mp.gamma(K)
    return un / Z

def mean_trunc_gamma(K, beta, gamma):
    f = lambda tt: tt * p_trunc_gamma(tt, K, beta, gamma)
    return mp.quad(f, [0, gamma])

def find_beta_for_trunc_gamma_mean(K, mu, gamma):
    f = lambda b: mean_trunc_gamma(K, b, gamma) - mu
    b0 = K/mu
    sol = mp.findroot(f, b0)
    return float(sol)

# ---------------------------
# Discrete (geometric) helpers
# ---------------------------
def H_discrete(P):
    P = np.asarray(P, dtype=float)
    P = P[P>0]
    return float(-np.sum(P * np.log(P)))

def KL_discrete(P, Q):
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    mask = (P>0)
    return float(np.sum(P[mask] * (np.log(P[mask]) - np.log(Q[mask]))))

def geometric_pmf(mu, n_max=10000):
    p = 1.0/mu
    n = np.arange(1, n_max+1)
    P = p * (1-p)**(n-1)
    P /= P.sum()
    return P

def two_point_same_mean(mu, m):
    alpha = (m - mu)/(m - 1.0)
    P = np.zeros(m+1, dtype=float)
    P[1] = alpha
    P[m] = 1.0 - alpha
    return P[1:]

# ---------------------------
# Numerical diff helpers
# ---------------------------
def finite_diff_np(x, t):
    dt = np.diff(t)
    dx = np.diff(x)
    v = np.empty_like(x)
    v[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
    v[0]    = dx[0]/dt[0]
    v[-1]   = dx[-1]/dt[-1]
    return v

# ---------------------------
# PART I: Theorem 5 (continuous, untruncated) & derivatives
# ---------------------------
print("=== §5.1 Theorem 5: Max-entropy-rate selects exponential (continuous) ===")
mu = 1.0
Kc = 1.5
beta_c = Kc/mu
H_exp = H_cont(p_exp, (mu,))
H_comp = H_cont(p_gamma, (Kc, beta_c))
KL_comp_exp = KL_cont(p_gamma, p_exp, (Kc, beta_c), (mu,))

print(f"H[Exp(mu)]: {H_exp:.12f}   (should be 1 + ln mu = 1.0)")
print(f"H[Gamma(K={Kc})]: {H_comp:.12f}")
print(f"H[Exp] - H[Gamma]: {H_exp - H_comp:.12f}   vs   KL(Gamma||Exp): {KL_comp_exp:.12f}")

R_exp = H_exp/mu
R_comp = H_comp/mu
print(f"Entropy rate R[Exp]: {R_exp:.12f}   R[Gamma]: {R_comp:.12f}   (Exp should be larger)")

print("\n=== Gamma-coordinate derivatives/curvature at K=1 (mu=1) ===")
H1  = H_mu_gamma(1.0, mu)
Hp1 = H_mu_prime(1.0)
Hpp1= H_mu_second(1.0)
print(f"H(1)   = {H1:.12f}")
print(f"H'(1)  = {Hp1:.12e}  (~ 0)")
print(f"H''(1) = {Hpp1:.15f}  (should be 1 - pi^2/6 = {-np.pi**2/6 + 1:.15f})")

Ks = np.linspace(0.3, 3.0, 200)
Hp_vals = np.array([H_mu_prime(k) for k in Ks])
sign_change_idx = [int(i) for i in np.where(np.sign(Hp_vals[:-1]) != np.sign(Hp_vals[1:]))[0]]
print(f"\nSign changes in H'(K) across [0.3,3.0]: indices = {sign_change_idx} (expect exactly one near K=1)")

H_vals = np.array([H_mu_gamma(k, mu) for k in Ks])
plt.figure()
plt.plot(Ks, H_vals)
plt.axvline(1.0, ls='--')
plt.title("H_mu(K) with mu=1 (unique maximum at K=1)")
plt.xlabel("K"); plt.ylabel("H_mu(K)")
plt.show()

plt.figure()
plt.plot(Ks, Hp_vals)
plt.axhline(0.0, ls='--')
plt.axvline(1.0, ls='--')
plt.title("H'_mu(K) sign flip at K=1")
plt.xlabel("K"); plt.ylabel("H'_mu(K)")
plt.show()

# ---------------------------
# PART II: Finite window (truncated) strong-converse
# ---------------------------
print("\n=== Finite window [0, gamma]: truncated exponential is max-entropy (same mean) ===")
mu = 0.7
gamma = 2.0
lam = find_lambda_for_mu(mu, gamma)

H_q = H_cont_trunc(p_trunc_exp, (lam, gamma), 0.0, gamma)

K_alt = 1.5
beta_alt = find_beta_for_trunc_gamma_mean(K_alt, mu, gamma)
H_p = H_cont_trunc(p_trunc_gamma, (K_alt, beta_alt, gamma), 0.0, gamma)
KL_pq = KL_cont_trunc(p_trunc_gamma, p_trunc_exp, (K_alt, beta_alt, gamma), (lam, gamma), 0.0, gamma)

print(f"lambda (trunc-exp mean=mu): {lam:.6f},   H[q_mu,gamma]={H_q:.12f}")
print(f"Trunc-Gamma(K={K_alt}, beta={beta_alt:.6f})  H[p]={H_p:.12f}")
print(f"H[q] - H[p] = {H_q - H_p:.12f}   vs   KL(p||q) = {KL_pq:.12f}   (should match)")

# ---------------------------
# PART III: Discrete (geometric) strong-converse
# ---------------------------
print("\n=== Discrete (N>=1): geometric is max-entropy (same mean) ===")
mu_d = 5.0
G_full = geometric_pmf(mu_d, n_max=200000)
H_G_exact = H_discrete(G_full)

m = 10
alpha = (m - mu_d)/(m - 1.0)
H_P = -(alpha*np.log(alpha) + (1-alpha)*np.log(1-alpha))

p_geo = 1.0/mu_d
Q1 = p_geo
Qm = p_geo * (1-p_geo)**(m-1)
KL_exact = alpha*np.log(alpha/Q1) + (1-alpha)*np.log((1-alpha)/Qm)

print(f"H[Geom(mu={mu_d})] ~= {H_G_exact:.12f},   H[Two-point]={H_P:.12f}")
print(f"H[Geom]-H[Two] = {H_G_exact - H_P:.12f}   vs   KL(P||Geom) = {KL_exact:.12f}   (should match)")

# ---------------------------
# PART IV: Corollary 6 empirical (LOCK/UNLOCK)
# ---------------------------
print("\n=== Corollary 6 empirical LOCK/UNLOCK ===")

def synth_signal_np(T=20.0, N=6000, sigma=0.5, omega_fun=None, theta0=np.pi/4):
    t = np.linspace(0, T, N)
    if omega_fun is None:
        omega = np.zeros_like(t)
    else:
        omega = omega_fun(t)
    theta = theta0 + np.cumsum(omega) * (t[1]-t[0])
    R = np.exp(sigma * t)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    Rrate = finite_diff_np(np.log(R), t)
    thetadot = finite_diff_np(theta, t)
    den = (Rrate - thetadot * np.tan(theta))
    mask = (np.abs(np.sin(theta)*np.cos(theta)) > 1e-2) & (np.abs(den) > 1e-4)
    K = np.empty_like(t); K[:] = np.nan
    K[mask] = (Rrate[mask] + thetadot[mask] * 1/np.tan(theta[mask])) / den[mask]
    return t, x, y, theta, R, Rrate, thetadot, K, mask

def Jmax_from_omega_hat_np(t, omega_hat, q=None):
    if q is None:
        q = np.ones_like(omega_hat)
    return float(np.trapezoid((omega_hat**2)/q, t))

def hazard_estimate_np(Tsamples, bins=60):
    T = np.asarray(Tsamples)
    T = T[T>0]
    t_edges = np.linspace(0, np.percentile(T, 99.0), bins+1)
    widths = np.diff(t_edges)
    centers = 0.5*(t_edges[1:]+t_edges[:-1])
    S = np.array([(T >= t_edges[i]).mean() for i in range(bins)])
    counts, _ = np.histogram(T, bins=t_edges)
    n_at_risk = S * len(T)
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = np.where(n_at_risk>0, (counts / n_at_risk) / widths, np.nan)
    return centers, lam

def logsurvival_np(Tsamples, grid=200):
    T = np.sort(Tsamples)
    tmax = np.percentile(T, 99.0)
    tt = np.linspace(0, tmax, grid)
    S = np.array([(T >= u).mean() for u in tt])
    S = np.clip(S, 1e-12, 1.0)
    return tt, np.log(S)

def linfit_np(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    theta, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = theta*x + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-12
    R2 = 1 - ss_res/ss_tot
    return theta, b, R2

# LOCK
mu = 1.0; sigma = 0.5
tL, xL, yL, thetaL, RL, RrateL, omegaL, KL, maskL = synth_signal_np(
    T=20.0, N=6000, sigma=sigma, omega_fun=lambda t: 0*t, theta0=np.pi/4)
Jmax_L = Jmax_from_omega_hat_np(tL, omegaL)
rng = np.random.default_rng(7)
N_T = 200000
T_lock = rng.exponential(scale=mu, size=N_T)
centers_L, haz_L = hazard_estimate_np(T_lock, bins=60)
tt_LS, lnS_L = logsurvival_np(T_lock, grid=200)
_, _, R2_L = linfit_np(tt_LS, lnS_L)

print("=== LOCK ===")
print(f"mean(|K-1|) on valid sector: {np.nanmean(np.abs(KL[maskL]-1)):.3e}")
print(f"J_max: {Jmax_L:.3e}")
print(f"log-survival R^2 (Exp): {R2_L:.6f}")

plt.figure(); plt.plot(tL[maskL][::10], KL[maskL][::10])
plt.title("LOCK: K(t) (subsampled)"); plt.xlabel("t"); plt.ylabel("K(t)"); plt.show()
plt.figure(); plt.plot(tL, omegaL); plt.title("LOCK: omega(t) == 0"); plt.xlabel("t"); plt.ylabel("omega(t)"); plt.show()
plt.figure(); plt.plot(centers_L, haz_L); plt.title("LOCK: Hazard ~ const"); plt.xlabel("t"); plt.ylabel("lambda_hat(t)"); plt.show()
plt.figure(); plt.plot(tt_LS, lnS_L); plt.title("LOCK: log S(t) linear"); plt.xlabel("t"); plt.ylabel("log S(t)"); plt.show()

# UNLOCK
a, b = 0.4, 0.8
omega_fun_U = lambda t: a*np.sin(b*t)
tU, xU, yU, thetaU, RU, RrateU, omegaU, KU, maskU = synth_signal_np(
    T=20.0, N=6000, sigma=sigma, omega_fun=omega_fun_U, theta0=np.pi/4)
Jmax_U = Jmax_from_omega_hat_np(tU, omegaU)
Kg = 1.5; theta_scale = mu / Kg
T_unlock = rng.gamma(shape=Kg, scale=theta_scale, size=N_T)
centers_U, haz_U = hazard_estimate_np(T_unlock, bins=60)
tt_US, lnS_U = logsurvival_np(T_unlock, grid=200)
_, _, R2_U = linfit_np(tt_US, lnS_U)

print("\n=== UNLOCK ===")
print(f"mean(|K-1|) on valid sector: {np.nanmean(np.abs(KU[maskU]-1)):.3e}")
print(f"J_max: {Jmax_U:.3e}")
print(f"log-survival R^2 (Gamma K={Kg}): {R2_U:.6f}")

plt.figure(); plt.plot(tU[maskU][::10], KU[maskU][::10])
plt.title("UNLOCK: K(t) (subsampled)"); plt.xlabel("t"); plt.ylabel("K(t)"); plt.show()
plt.figure(); plt.plot(tU, omegaU); plt.title("UNLOCK: omega(t)=a sin(bt)"); plt.xlabel("t"); plt.ylabel("omega(t)"); plt.show()
plt.figure(); plt.plot(centers_U, haz_U); plt.title(f"UNLOCK: Hazard (Gamma K={Kg})"); plt.xlabel("t"); plt.ylabel("lambda_hat(t)"); plt.show()
plt.figure(); plt.plot(tt_US, lnS_U); plt.title("UNLOCK: log S deviates"); plt.xlabel("t"); plt.ylabel("log S(t)"); plt.show()

tol_K = 5e-3
is_K1_L = np.nanmean(np.abs(KL[maskL]-1)) < tol_K
is_w0_L = np.allclose(omegaL, 0.0, atol=1e-12)
is_J0_L = Jmax_L < 1e-8
is_exp_L = R2_L > 0.999

is_K1_U = np.nanmean(np.abs(KU[maskU]-1)) < tol_K
is_w0_U = np.allclose(omegaU, 0.0, atol=1e-12)
is_J0_U = Jmax_U < 1e-8
is_exp_U = R2_U > 0.999

print("\n=== Corollary 6 empirical check ===")
print(f"LOCK:   [K=1? {is_K1_L}] [omega==0? {is_w0_L}] [J_max==0? {is_J0_L}] [Exp(mu)? {is_exp_L}]")
print(f"UNLOCK: [K=1? {is_K1_U}] [omega==0? {is_w0_U}] [J_max==0? {is_J0_U}] [Exp(mu)? {is_exp_U}]")

print("\nAll sections up to Corollary 6 executed.")

# ---------------------------
# PART V (NEW): Time-domain brachistochrone verification
# ---------------------------
print("\n=== NEW: Time-domain brachistochrone certificate & geometry recovery ===")

g = 9.81
r_true = 1.25         # cycloid generator radius
Delta_y_true = 2.0 * r_true
Omega_true = 0.5 * np.sqrt(g / r_true)
t_end = np.pi * np.sqrt(r_true / g)   # bottom time for phi from 0 to pi

# time grid
N = 4000
t = np.linspace(0.0, t_end, N)

# analytic signals on optimal motion
v = 2.0 * np.sqrt(g * r_true) * np.sin(Omega_true * t)           # semi-sine
a_t = g * np.cos(Omega_true * t)                                  # semi-cosine
j = - g * Omega_true * np.sin(Omega_true * t)                     # jerk
res = j + (Omega_true**2) * v                                     # certificate residual

# residual metrics
res_L2 = np.trapezoid(res**2, t)
res_Linf = np.max(np.abs(res))

# acceleration-circle identity: (a_t/g)^2 + (j/(Omega g))^2 == 1
X = a_t / g
Y = j / (Omega_true * g)
acc_circle_err = np.max(np.abs(X**2 + Y**2 - 1.0))

# curvature law: kappa(t) = 1/(4 r sin(Omega t)); check via a_n = v^2 kappa
eps = 1e-6
mask = (np.sin(Omega_true * t) > eps)
kappa = np.zeros_like(t)
kappa[mask] = 1.0 / (4.0 * r_true * np.sin(Omega_true * t[mask]))
a_n_from_kappa = np.zeros_like(t)
a_n_from_kappa[mask] = (v[mask]**2) * kappa[mask]
a_n_true = g * np.sin(Omega_true * t)
curvature_max_rel_err = np.max(np.abs(a_n_from_kappa[mask] - a_n_true[mask]) / np.maximum(1e-12, np.abs(a_n_true[mask])))

# phase uniformization: dt/dphi = sqrt(r/g) const
# Using phi/2 = Omega t  => dphi/dt = 2 Omega = sqrt(g/r)
phi = 2.0 * Omega_true * t
dphi_dt = np.gradient(phi, t)
phase_uniform_const = np.sqrt(g / r_true)
phase_uniform_Linf = np.max(np.abs(dphi_dt - phase_uniform_const))

# geometry recovery from time-domain
# (1) from linear regression on j ~ - (Omega^2) v  ==> Omega^2_hat = - <j, v> / <v, v>
Omega2_hat = - np.trapz(j * v, t) / np.trapz(v * v, t)
Omega_hat = np.sqrt(max(Omega2_hat, 0.0))
r_hat_from_Omega = g / (4.0 * Omega2_hat)
# (2) from vmax: r = vmax^2/(4g)
vmax = np.max(v)
r_hat_from_vmax = (vmax**2) / (4.0 * g)
# (3) from total time t_end: r = g (t_end/pi)^2
r_hat_from_tend = g * (t_end / np.pi)**2

print(f"g = {g:.4f}, r_true = {r_true:.6f}, Delta_y_true = {Delta_y_true:.6f}")
print(f"Omega_true^2 = {Omega_true**2:.9f}  (should equal g/(4r))")
print(f"Certificate residual: L2 = {res_L2:.3e}, Linf = {res_Linf:.3e}")
print(f"Acceleration-circle max error: {acc_circle_err:.3e}")
print(f"Curvature law max relative error: {curvature_max_rel_err:.3e}")
print(f"Phase uniformization Linf error: {phase_uniform_Linf:.3e}")
print(f"Recovered r from Omega: {r_hat_from_Omega:.9f}  (rel.err = {(r_hat_from_Omega/r_true - 1):+.3e})")
print(f"Recovered r from vmax:  {r_hat_from_vmax:.9f}  (rel.err = {(r_hat_from_vmax/r_true - 1):+.3e})")
print(f"Recovered r from t_end: {r_hat_from_tend:.9f}  (rel.err = {(r_hat_from_tend/r_true - 1):+.3e})")

# Plots for the new formulas
plt.figure(); plt.plot(t, v); plt.title("v(t) = 2 sqrt(g r) sin(Omega t)"); plt.xlabel("t"); plt.ylabel("v(t)"); plt.show()
plt.figure(); plt.plot(t, a_t); plt.title("a_t(t) = g cos(Omega t)"); plt.xlabel("t"); plt.ylabel("a_t(t)"); plt.show()
plt.figure(); plt.plot(t, res); plt.title("Certificate residual: j + Omega^2 v"); plt.xlabel("t"); plt.ylabel("residual"); plt.show()

# curvature plot avoiding singularity near t=0
plt.figure(); 
idx = np.where(mask)[0]
plt.plot(t[idx], a_n_from_kappa[idx] - a_n_true[idx])
plt.title("a_n from kappa minus true a_n (should be ~ 0)")
plt.xlabel("t"); plt.ylabel("difference"); plt.show()

# acceleration-circle identity plot
plt.figure(); plt.plot(t, X**2 + Y**2); plt.title("(a_t/g)^2 + (j/(Omega g))^2"); plt.xlabel("t"); plt.ylabel("sum"); plt.show()

print("\nNEW section done. This verifies:")
print("  • Phase uniformization: dphi/dt = sqrt(g/r) (constant)")
print("  • Semi-sine velocity and semi-cosine tangential acceleration")
print("  • One-line certificate: j + Omega^2 v = 0 (residual ~ 0)")
print("  • Acceleration-circle identity: (a_t/g)^2 + (j/(Omega g))^2 = 1")
print("  • Curvature law: kappa(t) = 1 / (4 r sin(Omega t))")
print("  • Geometry recovery: r from Omega, vmax, and total time t_end")
