# ============================================
#  σ*HFF  →  σ*cosmo 映射：自动解析形式搜索器
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1. HFF σ* 数据与对应的目标宇宙学σ*值
# -----------------------------

sigma_HFF = np.array([
    5.40,   # MACS0717
    5.50,   # MACS0416  
    3.10    # Abell2744
])

# 根据你的输出结果，每个HFF值对应的目标宇宙学σ*值
sigma_cosmo_targets = np.array([
    -2.2654321,   # 对应 5.40
    -2.26515152,  # 对应 5.50
    -2.27688172   # 对应 3.10
])

print("HFF σ* values:", sigma_HFF)
print("Target σ*cosmo values:", sigma_cosmo_targets)
print("Mean cosmic σ*:", np.mean(sigma_cosmo_targets))

# =====================================================
# 2. 自动搜索可能的映射函数族
# =====================================================

# 候选函数形态
def mapping_funcs():
    return {
        "linear": lambda x, a, b: a*x + b,
        "log":    lambda x, a, b: a*np.log(x) + b,
        "inv":    lambda x, a, b: a*(1/x) + b,
        "sqrt":   lambda x, a, b: a*np.sqrt(x) + b,
        "power":  lambda x, a, b: a*(x**0.5) + b,
        "exp":    lambda x, a, b: a*np.exp(-x/10) + b,
        "mixed1": lambda x, a, b: a*np.log(x) + b*np.sqrt(x),
        "mixed2": lambda x, a, b: a/x + b*np.log(x),
        "quad":   lambda x, a, b: a*(x**2) + b,
        "inv_sq": lambda x, a, b: a/(x**2) + b,
    }

# 参数搜索范围（根据你的输出结果调整范围）
a_range = np.linspace(-1, 1, 201)    # 更精细的搜索
b_range = np.linspace(-3, -1, 201)   # 围绕 -2.27 的范围

best_model = None
best_mse = 1e99
best_name = None
best_pred = None

# =====================================================
# 3. 遍历模型族 + 参数搜索
# =====================================================

print("\nSearching for best mapping...")
func_dict = mapping_funcs()

for name, func in func_dict.items():
    # 跳过在某些输入值下会产生无效值的函数
    if name in ['log'] and np.any(sigma_HFF <= 0):
        continue
        
    for a, b in product(a_range, b_range):
        try:
            pred = func(sigma_HFF, a, b)
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                continue
                
            mse = mean_squared_error(pred, sigma_cosmo_targets)
            if mse < best_mse:
                best_mse = mse
                best_model = (name, a, b)
                best_pred = pred.copy()
        except:
            continue

# ========================================
# 4. 输出最佳解
# ========================================

if best_model is None:
    print("No valid mapping found!")
else:
    best_name, best_a, best_b = best_model
    print("\n" + "="*40)
    print("=== BEST MAPPING FOUND ===")
    print("="*40)
    print(f"Model family: {best_name}")
    print(f"a = {best_a:.6f}")
    print(f"b = {best_b:.6f}")
    print(f"MSE = {best_mse:.10e}")
    
    print("\nDetailed predictions:")
    print("HFF σ*    | Target σ*cosmo | Predicted σ*cosmo | Error")
    print("-" * 55)
    for i in range(len(sigma_HFF)):
        error = best_pred[i] - sigma_cosmo_targets[i]
        print(f"{sigma_HFF[i]:8.4f} | {sigma_cosmo_targets[i]:14.8f} | {best_pred[i]:17.8f} | {error:10.2e}")

# ========================================
# 5. 可视化结果
# ========================================

plt.figure(figsize=(10, 6))

# 散点图
plt.subplot(1, 2, 1)
plt.scatter(sigma_HFF, sigma_cosmo_targets, color="red", s=100, label="Target σ*cosmo", zorder=5)
plt.scatter(sigma_HFF, best_pred, color="blue", s=80, label="Best-fit prediction", zorder=5)

# 连接线显示误差
for i in range(len(sigma_HFF)):
    plt.plot([sigma_HFF[i], sigma_HFF[i]], [sigma_cosmo_targets[i], best_pred[i]], 
             'k--', alpha=0.5, linewidth=1)

plt.xlabel("σ* (HFF)")
plt.ylabel("σ* (Cosmo)")
plt.title(f"Best Mapping: {best_name}\na={best_a:.6f}, b={best_b:.6f}")
plt.legend()
plt.grid(True, alpha=0.3)

# 函数曲线图
plt.subplot(1, 2, 2)
x_plot = np.linspace(2.5, 6.0, 100)
best_func = func_dict[best_name]
y_plot = best_func(x_plot, best_a, best_b)

plt.plot(x_plot, y_plot, 'b-', label=f"{best_name} mapping", linewidth=2)
plt.scatter(sigma_HFF, sigma_cosmo_targets, color="red", s=100, label="Target points", zorder=5)
plt.xlabel("σ* (HFF)")
plt.ylabel("σ* (Cosmo)")
plt.title("Mapping Function")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================================
# 6. 验证其他候选模型
# ========================================

print("\n" + "="*50)
print("COMPARISON WITH OTHER MODEL FAMILIES")
print("="*50)

# 测试所有模型在最佳参数附近的表现
test_models = {}
for name, func in func_dict.items():
    if name == best_name:
        continue
        
    # 简单测试线性回归来找到近似最佳参数
    try:
        if name == "linear":
            X = sigma_HFF.reshape(-1, 1)
            model = LinearRegression().fit(X, sigma_cosmo_targets)
            a_test, b_test = model.coef_[0], model.intercept_
        elif name == "inv":
            X = (1/sigma_HFF).reshape(-1, 1)
            model = LinearRegression().fit(X, sigma_cosmo_targets)
            a_test, b_test = model.coef_[0], model.intercept_
        elif name == "log":
            X = np.log(sigma_HFF).reshape(-1, 1)
            model = LinearRegression().fit(X, sigma_cosmo_targets)
            a_test, b_test = model.coef_[0], model.intercept_
        else:
            # 对于复杂函数，使用网格搜索
            best_mse_temp = 1e99
            a_test, b_test = 0, 0
            for a, b in product(np.linspace(-1, 1, 51), np.linspace(-3, -1, 51)):
                pred_temp = func(sigma_HFF, a, b)
                if np.any(np.isnan(pred_temp)):
                    continue
                mse_temp = mean_squared_error(pred_temp, sigma_cosmo_targets)
                if mse_temp < best_mse_temp:
                    best_mse_temp = mse_temp
                    a_test, b_test = a, b
        
        pred_test = func(sigma_HFF, a_test, b_test)
        mse_test = mean_squared_error(pred_test, sigma_cosmo_targets)
        test_models[name] = (mse_test, a_test, b_test)
        
    except:
        test_models[name] = (np.inf, 0, 0)

# 按MSE排序输出
sorted_models = sorted(test_models.items(), key=lambda x: x[1][0])
print(f"\n{'Model':<10} {'MSE':<15} {'a':<12} {'b':<12}")
print("-" * 50)
print(f"{best_name:<10} {best_mse:<15.2e} {best_a:<12.6f} {best_b:<12.6f}")
for name, (mse, a, b) in sorted_models[:5]:  # 只显示前5个
    print(f"{name:<10} {mse:<15.2e} {a:<12.6f} {b:<12.6f}")
