# ===============================================================
# 🌀 SCI ↔ DAG-UTH v3 — Time Emergence with Bundle-Covariant Matched Filtering
# ===============================================================

# 安装必要的库
!pip install astropy matplotlib tqdm requests numpy mpmath scipy scikit-image -q

import numpy as np
import matplotlib.pyplot as plt
import requests
import os
from astropy.io import fits
from tqdm import tqdm
import mpmath as mp
from scipy import signal
from scipy.stats import norm
from skimage.transform import resize
mp.mp.dps = 50

# -------------------- 数据集 URLs --------------------
DATASETS = [
    ("Abell2744_CATSv4", "https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"),
    ("MACS0416_CATSv4",  "https://archive.stsci.edu/pub/hlsp/frontier/macs0416/models/cats/v4/hlsp_frontier_model_macs0416_cats_v4_kappa.fits"),
    ("Abell370_CATSv4",  "https://archive.stsci.edu/pub/hlsp/frontier/abell370/models/cats/v4/hlsp_frontier_model_abell370_cats_v4_kappa.fits")
]

# -------------------- 工具函数 --------------------
def download(url, fname):
    """下载数据集"""
    if not os.path.exists(fname):
        print(f"↓ Downloading {fname} ...")
        r = requests.get(url)
        if r.status_code != 200:
            raise Exception(f"Failed to download {url}")
        with open(fname, 'wb') as f:
            f.write(r.content)
    return fname

def zscore(a):
    """标准化数据"""
    m, s = np.mean(a), np.std(a) + 1e-12
    return (a - m) / s

def lap2(f):
    """二维离散拉普拉斯算子"""
    if f.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {f.shape}")
    return (
        -4 * f +
        np.roll(f, 1, 0) + np.roll(f, -1, 0) +
        np.roll(f, 1, 1) + np.roll(f, -1, 1)
    )

def dag_propagate(f, h, mu, gamma, alpha, step):
    """DAG-UTH 传播：扩散 + 非线性反馈，包含 Φ 和 H 的演化"""
    g = f.copy()
    h_new = float(h)  # 确保 h 是标量
    for _ in range(step):
        try:
            lap_g = lap2(g)
            lap_h = 0  # H 是标量，假设没有空间结构
            g_new = g + mu * lap_g - gamma * g * h_new
            h_new = h_new + alpha * np.mean(g**2)  # 简化为均值，忽略 lap_h
            g = g_new
        except Exception as e:
            print(f"Error in dag_propagate: {e}")
            return g, h_new
    return g, h_new

# -------------------- SCI 计算 --------------------
def sci_K_field(data, mu, gamma, alpha=0.1, steps=10):
    """计算 Φ(t), H(t), K(t) 时间演化"""
    if data.size == 0:
        raise ValueError("Input data is empty")
    f = zscore(data.copy())
    h = np.var(f)  # 初始 H(t) 为场的方差
    K_vals, Phi_vals, H_vals = [], [], []
    for _ in range(steps):
        try:
            f_next, h_next = dag_propagate(f, h, mu, gamma, alpha, 1)
            Phi = np.mean(f_next * f)  # 结构密度
            H = h_next  # 直接使用传播的 H(t)
            K_vals.append(Phi / (H + 1e-12))
            Phi_vals.append(Phi)
            H_vals.append(H)
            f, h = f_next, h_next
        except Exception as e:
            print(f"Error in sci_K_field: {e}")
            return np.array(K_vals), np.array(Phi_vals), np.array(H_vals)
    return np.array(K_vals), np.array(Phi_vals), np.array(H_vals)

# -------------------- 束协变匹配滤波 --------------------
def bundle_covariant_matched_filter(K_vals, t, q=None, bandlimit=None):
    """实现束协变匹配滤波，计算 Jmax 和最优滤波器"""
    if len(K_vals) == 0 or len(t) == 0 or len(K_vals) != len(t):
        raise ValueError(f"Invalid K_vals or t: {len(K_vals)}, {len(t)}")
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    R = np.abs(K_vals) + 1e-12
    sigma = np.gradient(np.log(R), dt)
    K = K_vals
    omega_hat = sigma * (K - 1) / (K + 1 + 1e-12)

    if q is None:
        q = np.var(K_vals) * np.ones_like(t) + 1e-12
    else:
        q = np.array(q) + 1e-12

    u = omega_hat / q
    if bandlimit:
        freq = np.fft.fftfreq(len(t), dt)
        fft_u = np.fft.fft(u)
        fft_u[np.abs(freq) > bandlimit] = 0
        phi_star = np.fft.ifft(fft_u).real
    else:
        phi_star = u

    norm_q = np.sqrt(np.sum(np.abs(phi_star)**2 * q * dt))
    phi_star = phi_star / (norm_q + 1e-12)
    Jmax = np.sum(np.abs(omega_hat)**2 / q * dt)
    return Jmax, phi_star, omega_hat

def glrt_power(Jmax, alpha=0.05):
    """计算 GLRT 检测功率"""
    z = norm.ppf(1 - alpha)
    P_D = 1 - norm.cdf(z - np.sqrt(Jmax))
    return z, P_D

# -------------------- 主实验 --------------------
def run_experiment(name, url, mu=0.08, gamma=0.6, alpha=0.1, steps=10, bandlimit=0.5):
    print(f"\n=== {name} ===")
    try:
        path = download(url, f"{name}.fits")
        data = fits.getdata(path)
        data = np.nan_to_num(data)
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError(f"Invalid data shape: {data.shape}")
        data = data[:min(data.shape[0], 2000), :min(data.shape[1], 2000)]
        data = zscore(data)
        print(f"✅ Loaded κ-field: {data.shape}")
        data = resize(data, (500, 500))
        print(f"↓ Resized: {data.shape}")

        K, Φ, H = sci_K_field(data, mu, gamma, alpha, steps)
        if len(K) == 0:
            raise ValueError("K_vals is empty")

        t = np.arange(len(K))
        Jmax, phi_star, omega_hat = bundle_covariant_matched_filter(K, t, bandlimit=bandlimit)
        z, P_D = glrt_power(Jmax)

        dK = np.gradient(K, t[1] - t[0]) if len(t) > 1 else np.zeros_like(K)
        idx = np.where((np.abs(K - 1) > 0.05) & (np.abs(dK) > 1e-3))[0]
        t_emerge = idx[0] if len(idx) > 0 else None
        if t_emerge is not None:
            print(f"🕒 Time Emergence detected at t = {t_emerge}, K = {K[t_emerge]:.3f}, Jmax = {Jmax:.3f}, P_D = {P_D:.3f}")
        else:
            print("❄️ No emergence detected (still in structural equilibrium).")

        plt.figure(figsize=(15, 8))
        plt.subplot(2, 2, 1)
        plt.plot(t, Φ, 'r', label='Φ(t)')
        plt.plot(t, H, 'b', label='H(t)')
        plt.legend()
        plt.title('Φ(t), H(t)')
        plt.xlabel("t")

        plt.subplot(2, 2, 2)
        plt.plot(t, K, 'k', lw=2)
        plt.axhline(1, ls='--', color='gray')
        plt.title('K(t) — SCI Ratio')
        plt.xlabel("t")

        plt.subplot(2, 2, 3)
        plt.plot(t, dK, 'm', lw=2)
        plt.axhline(0, ls='--', color='gray')
        plt.title('dK/dt (Time Genesis Indicator)')

        plt.subplot(2, 2, 4)
        plt.plot(t, omega_hat, 'g', label='ω̂(t)')
        plt.plot(t, phi_star, 'b--', label='φ*(t)')
        plt.legend()
        plt.title(f'Estimated ω(t) and φ*(t), Jmax = {Jmax:.3f}')
        plt.xlabel("t")

        plt.tight_layout()
        plt.show()

        return K, Jmax, P_D
    except Exception as e:
        print(f"Error in run_experiment for {name}: {e}")
        return np.array([]), 0, 0

# -------------------- 参数扫描 --------------------
def parameter_scan(data, name, mu_range=[0.02, 0.12], gamma_range=[0.5, 0.9], steps=10, alpha=0.1):
    """扫描 μ 和 γ 参数以寻找平衡点"""
    mus = np.linspace(mu_range[0], mu_range[1], 5)
    gammas = np.linspace(gamma_range[0], gamma_range[1], 5)
    results = []
    for mu in mus:
        for gamma in gammas:
            try:
                K, _, _ = sci_K_field(data, mu, gamma, alpha, steps)
                mean_K = np.mean(K) if len(K) > 0 else np.nan
                state = check_equilibrium(K) if len(K) > 0 else "Invalid"
                results.append((mu, gamma, mean_K, state))
                print(f"{name}: μ={mu:.2f}, γ={gamma:.2f}, mean(K)={mean_K:.3f}, {state}")
            except Exception as e:
                print(f"Error in parameter_scan for μ={mu:.2f}, γ={gamma:.2f}: {e}")
    return results

def check_equilibrium(K_vals):
    """检查平衡状态"""
    if len(K_vals) == 0:
        return "Invalid"
    if np.all(np.abs(K_vals - 1) < 0.05):
        return "Equilibrium (pre-time)"
    elif np.mean(K_vals) > 1.05:
        return "Post-time structural phase"
    else:
        return "Transition regime"

# -------------------- 运行所有数据集 --------------------
results = {}
for name, url in DATASETS:
    print(f"\n=== Running {name} ===")
    try:
        path = download(url, f"{name}.fits")
        data = fits.getdata(path)
        data = np.nan_to_num(data)
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError(f"Invalid data shape: {data.shape}")
        data = data[:min(data.shape[0], 2000), :min(data.shape[1], 2000)]
        data = zscore(data)
        data = resize(data, (500, 500))

        K, Jmax, P_D = run_experiment(name, url, mu=0.08, gamma=0.6, alpha=0.1, steps=10, bandlimit=0.5)
        scan_results = parameter_scan(data, name, steps=10)

        state = check_equilibrium(K)
        results[name] = {'mean_K': np.mean(K) if len(K) > 0 else np.nan, 'state': state, 'Jmax': Jmax, 'P_D': P_D, 'scan': scan_results}
        print(f"{name}: {state}, mean(K)={np.mean(K):.3f if len(K) > 0 else 'N/A'}, Jmax={Jmax:.3f}, P_D={P_D:.3f}")
    except Exception as e:
        print(f"Error processing {name}: {e}")

# -------------------- 汇总结果 --------------------
print("\n=== 汇总结果 ===")
for name, res in results.items():
    mean_K = res['mean_K']
    mean_K_str = f"{mean_K:.3f}" if not np.isnan(mean_K) else "N/A"
    print(f"{name}: {res['state']}, mean(K)={mean_K_str}, Jmax={res['Jmax']:.3f}, P_D={res['P_D']:.3f}")