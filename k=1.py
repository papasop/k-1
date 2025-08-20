import numpy as np
from scipy.special import polygamma, gamma, digamma
import sympy as sp
import mpmath as mp
import matplotlib.pyplot as plt

# 设置高精度
mp.mp.dps = 20  # 20位精度

# 公式1: Gamma kernel p(T;K)
def gamma_kernel(T, K):
    return (T**(K-1) * np.exp(-T)) / gamma(K)

# 测试K=1时p(T;1)=e^{-T}
T_test = np.linspace(0.1, 5, 10)
p_K1 = gamma_kernel(T_test, 1)
exp_T = np.exp(-T_test)
print("p(K=1) vs e^{-T}:", np.allclose(p_K1, exp_T))  # True

# 公式3: H(K)闭式
def H_K_closed(K):
    return K + np.log(gamma(K)) + (1 - K) * digamma(K)

# 公式4: H'(K)
def H_prime(K):
    return 1 + (1 - K) * polygamma(1, K)

# 公式5: H''(K)
def H_double_prime(K):
    return -polygamma(1, K) + (1 - K) * polygamma(2, K)

# 特殊值验证
psi1_1 = polygamma(1, 1)  # π²/6 ≈1.6449340668482264
zeta3 = mp.zeta(3)  # 1.2020569031595942
psi2_1 = polygamma(2, 1)  # -2*zeta(3) ≈-2.4041138063191885

print("ψ^{(1)}(1):", psi1_1)
print("ζ(3):", float(zeta3))
print("ψ^{(2)}(1):", psi2_1)

# 公式6: H''(1) = -ψ^{(1)}(1) = -π²/6 <0
H_dd_1 = H_double_prime(1)
print("H''(1):", H_dd_1, "<0:", H_dd_1 < 0)  # -1.6449340668482264 <0 True

# 符号验证（使用sympy）
K_sym = sp.symbols('K')
psi_sym = sp.digamma(K_sym)
H_sym = K_sym + sp.ln(sp.gamma(K_sym)) + (1 - K_sym) * psi_sym
H_prime_sym = sp.diff(H_sym, K_sym)
H_dd_sym = sp.diff(H_prime_sym, K_sym)
print("H'(K) symbols:", H_prime_sym)
print("H''(K) symbols:", H_dd_sym)

# K_max(ε)示例 (Eq.kmax): 假设a=1, b=-3, c=2
a, b, c = 1, -3, 2
disc = b**2 - 4*a*c  # 1 >0
root1 = (-b + np.sqrt(disc)) / (2*c)  # 1.0
root2 = (-b - np.sqrt(disc)) / (2*c)  # 0.5
print("K_max roots example:", root1, root2)

# W_δ plateau: d²H/dK² < -δ (δ=0.2)
K_range = np.linspace(0.5, 1.5, 100)
H_dd_range = H_double_prime(K_range)
W_delta = np.sum(H_dd_range < -0.2) / len(K_range) * (1.5-0.5)  # ≈宽度
print("Approximate |W_δ| for δ=0.2:", W_delta)  # ≈1.0 (全负曲率范围宽)

# 时间涌现条件 (Eq.perturbation_trigger): 符号，无需计算

# 绘图: H''(K)曲率平台
plt.plot(K_range, H_dd_range)
plt.axhline(-0.2, color='r', linestyle='--')
plt.xlabel('K')
plt.ylabel("H''(K)")
plt.title('Negative Curvature Plateau at K=1')
plt.show()