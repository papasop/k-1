# ============================================
#  σ*HFF  →  σ*cosmo 映射：自动解析形式搜索器
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. 你的 HFF σ* 数据（来自日志）
# -----------------------------

sigma_HFF = np.array([
    5.40,   # MACS0717
    5.50,   # MACS0416
    3.10    # Abell2744
])

# 宇宙学 σ* (cosmo) = -2.273 (来自 ΩΛ ≈ 0.67 的推导)
sigma_cosmo_target = -2.273
sigma_cosmo = np.array([sigma_cosmo_target] * len(sigma_HFF))

print("HFF σ* values:", sigma_HFF)
print("Cosmic σ*:    ", sigma_cosmo_target)

# =====================================================
# 2. 自动搜索可能的映射函数族
# =====================================================

# 候选函数形态 (可扩展)
def mapping_funcs():
    return {
        "linear": lambda x, a, b: a*x + b,
        "log":    lambda x, a, b: a*np.log(x) + b,
        "inv":    lambda x, a, b: a*(1/x) + b,
        "sqrt":   lambda x, a, b: a*np.sqrt(x) + b,
        "power":  lambda x, a, b: a*(x**0.3) + b,
        "exp":    lambda x, a, b: a*np.exp(-x/10) + b,
        "mixed1": lambda x, a, b: a*np.log(x) + b*np.sqrt(x),
        "mixed2": lambda x, a, b: a/x + b*np.log(x),
    }

# 参数搜索范围
a_range = np.linspace(-5, 5, 121)
b_range = np.linspace(-10, 10, 81)

best_model = None
best_mse = 1e99
best_name = None

# =====================================================
# 3. 遍历模型族 + 参数搜索
# =====================================================

for name, func in mapping_funcs().items():
    for a, b in product(a_range, b_range):
        pred = func(sigma_HFF, a, b)
        mse = mean_squared_error(pred, sigma_cosmo)
        if mse < best_mse:
            best_mse = mse
            best_model = (name, a, b)
            best_pred = pred

# ========================================
# 4. 输出最佳解
# ========================================

best_name, best_a, best_b = best_model
print("\n=== Best Mapping Found ===")
print(f"Model family: {best_name}")
print(f"a = {best_a:.6f}")
print(f"b = {best_b:.6f}")
print(f"MSE = {best_mse:.6e}")

print("\nPredicted σ*cosmo values:")
print(best_pred)

# ========================================
# 5. 可视化结果
# ========================================

plt.figure(figsize=(7,5))
plt.scatter(sigma_HFF, sigma_cosmo, color="red", label="Target σ*cosmo")
plt.scatter(sigma_HFF, best_pred, color="blue", label="Best-fit prediction")
plt.xlabel("σ* (HFF)")
plt.ylabel("σ* (Cosmo)")
plt.title(f"Best Mapping: {best_name}\na={best_a:.4f}, b={best_b:.4f}")
plt.legend()
plt.grid(True)
plt.show()
