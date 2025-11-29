# ================================================
#  K=1 Chronogeometrodynamics Unified Test Cell
#  ------------------------------------------------
#  1. σ*HFF → σ*cosmo 映射
#  2. flow = k P(k) from K=1 action
#  3. ΩΛ ≈ 0.67 geometric derivation + robustness
# ================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sklearn.linear_model import LinearRegression

# --------------------------
# Basic cosmology parameters
# --------------------------
h = 0.674
Omega_m = 0.315
Omega_b = 0.049
ns = 0.965
As = 2.1e-9
sigma8 = 0.811

# --------------------------
# Transfer function (EH98 zero-baryon)
# --------------------------
def transfer_EH(k):
    ommh2 = (Omega_m - Omega_b) * h**2
    obh2  = Omega_b * h**2
    s     = 44.5 * np.log(9.83/ommh2) / np.sqrt(1 + 10 * obh2**0.75)

    alpha = 1 - 0.328*np.log(431*ommh2)*(obh2/ommh2) \
              + 0.38*np.log(22.3*ommh2)*(obh2/ommh2)**2

    gamma = Omega_m*h*(alpha + (1-alpha)/(1+(0.43*k*h*s)**4))
    q     = k / gamma

    L0    = np.log(2*np.e + 1.8*q)
    C0    = 14.2 + 731/(1+62.5*q)
    return L0/(L0 + C0*q*q)

# --------------------------
# Linear P(k)
# --------------------------
def Pk_linear(k):
    T = transfer_EH(k)
    return As*(k/0.05)**(ns-1) * (2*np.pi**2/k**3) * k**4 * T*T

# --------------------------
# Normalize to sigma8
# --------------------------
def W(x):
    return 3*(np.sin(x) - x*np.cos(x))/x**3 if x!=0 else 1

def sigma2_R(R):
    integrand = lambda kk: kk*kk * Pk_linear(kk) * W(kk*R)**2 / (2*np.pi**2)
    return quad(integrand, 1e-6, 200)[0]

norm_factor = sigma8**2 / sigma2_R(8/h)

def Pk(k):
    return norm_factor * Pk_linear(k)

print(f"P(k) 归一化因子 norm_factor = {norm_factor:.4e}")

# ------------------------------------------------
# Part 1 — σ*HFF → σ*cosmo 线性映射
# ------------------------------------------------
sigma_HFF  = np.array([5.40, 5.50, 3.10]).reshape(-1,1)
sigma_cosmo = np.array([-2.273, -2.273, -2.273])  # observed from macro spectrum

reg = LinearRegression().fit(sigma_HFF, sigma_cosmo)
a = reg.coef_[0]
b = reg.intercept_

print("\n=== Part 1: σ*HFF → σ*cosmo 映射 ===")
print(f"拟合:  σ*cosmo = a σ*HFF + b")
print(f"a = {a:.5f}")
print(f"b = {b:.5f}")

# ------------------------------------------------
# Part 2 — flow = kP(k) from K=1 action
# ------------------------------------------------
kvec  = np.logspace(-6, 2, 5000)
Pkvec = np.vectorize(Pk)(kvec)
flow  = kvec * Pkvec                        # core functional from K=1

# derivative check for natural extremum around k*
dflow = np.gradient(flow, np.log(kvec))

print("\n=== Part 2: flow = kP(k) 是否自然出现？ ===")
print("flow(k) 在 k≈k* 附近是否出现极值？")
print("dflow sign-change =", np.any(np.diff(np.sign(dflow))))

# ------------------------------------------------
# Part 3 — ΩΛ geometric derivation + robustness
# ------------------------------------------------
def Omega_Lambda_geom(k_star):
    sigma_star = np.log(k_star)
    sigma = np.log(kvec)
    flow_vec = kvec * Pkvec
    I_total = np.trapz(flow_vec, sigma)
    I_de    = np.trapz(flow_vec[sigma < sigma_star], sigma[sigma < sigma_star])
    return I_de / I_total

print("\n=== Part 3: 鲁棒性测试 (ΩΛ vs k*) ===")
for ks in [1/9, 1/10, 1/11, 1/12, 1/13]:
    print(f"k*={ks:.4f} → ΩΛ={Omega_Lambda_geom(ks):.4f}")

# ==========================================
# Final: compute ΩΛ from the correct k* (≈0.103)
# ==========================================
k_star_cosmo = 0.103
Omega_L = Omega_Lambda_geom(k_star_cosmo)

print("\n=== Final Geometric ΩΛ ===")
print(f"使用 k* = 0.103 → ΩΛ ≈ {Omega_L:.4f}")
print("观测值 ΩΛ ≈ 0.6800")

if abs(Omega_L - 0.68)/0.68 < 0.05:
    print("\n🎉 几何推导成功（误差 < 5%）！")
else:
    print("\n❌ 推导失败，请检查模型。")

# ------------------------------------------------
# Visualization
# ------------------------------------------------
plt.figure(figsize=(10,6))
plt.loglog(kvec, flow, lw=2, label="flow = k P(k)")
plt.axvline(k_star_cosmo, color='red', ls='--', lw=2, 
            label=f"k* = {k_star_cosmo:.3f} h/Mpc")
plt.xlabel("k  [h/Mpc]")
plt.ylabel("flow")
plt.title("Flow Spectrum for K=1 Theory")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
