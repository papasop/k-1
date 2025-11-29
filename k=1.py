# ============================================================
# 三星系团 + 三定律 + ζ-flow vs Noise 对照
# Part 1: 结构尺度扫描
# Part 2: 流动尺度扫描
# Laws I & II: 信息时间度规 + 结构作用
# σ* 自动检测 + 分布图
# Clusters: MACS0717, Abell2744, MACS0416
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_laplace
from astropy.io import fits
import urllib.request, os, time, warnings
from mpmath import mp

warnings.filterwarnings("ignore")
mp.dps = 15

# -----------------------------
# 0. 全局参数
# -----------------------------
CROP_SIZE = 96
SIGMA_STRUCT_RANGE = np.round(np.arange(2.5, 6.6, 0.1), 2).tolist()
SIGMA_FLOW_RANGE   = np.round(np.arange(0.5, 4.6, 0.1), 2).tolist()
SIGMA_FLOW_FIXED   = 2.0   # Part1: 流动尺度固定
SIGMA_STRUCT_FIXED = 5.0   # Part2: 结构尺度固定

N_BOOT = 200       # bootstrap 次数
N_SHUFFLE = 2000   # Law II 乱序路径数

# 星系团 κ-map来源
CLUSTER_FITS = {
    "MACS0717":  "https://archive.stsci.edu/pub/hlsp/frontier/macs0717/models/cats/v4/hlsp_frontier_model_macs0717_cats_v4_kappa.fits",
    "Abell2744": "https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits",
    "MACS0416":  "https://archive.stsci.edu/pub/hlsp/frontier/macs0416/models/cats/v4/hlsp_frontier_model_macs0416_cats_v4_kappa.fits"
}

# ζ-zero 相关
T0 = 2000.0
DT = 0.4
MZEROS = 150

# 随机数
rng_global = np.random.default_rng(2025)

# -----------------------------
# 1. 工具函数
# -----------------------------

def load_cluster(name):
    """下载并裁剪星系团 κ-map，返回标准化 log κ 作为 φ_real"""
    url = CLUSTER_FITS[name]
    filename = f"{name}.fits"
    if not os.path.exists(filename):
        print(f"📡 下载 {name} 数据...")
        urllib.request.urlretrieve(url, filename)
    else:
        print(f"✅ 检测到本地数据: {filename}")

    with fits.open(filename) as h:
        data = h[0].data

    cy, cx = 1000, 1000
    y1, y2 = cy - CROP_SIZE//2, cy + CROP_SIZE//2
    x1, x2 = cx - CROP_SIZE//2, cx + CROP_SIZE//2
    patch = data[y1:y2, x1:x2]

    valid_min = np.nanmin(patch[patch > 0])
    patch = np.maximum(patch, valid_min)
    phi = np.log(patch)
    phi = (phi - np.mean(phi)) / np.std(phi)
    return phi

def part1by1(n):
    n &= 0x0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n

def z_order_index(H, W):
    y, x = np.indices((H, W))
    morton = (part1by1(y.flatten()) | (part1by1(x.flatten()) << 1))
    return morton.argsort()

def embed_1d_to_2d(series_1d, shape, z_indices):
    h, w = shape
    n_pix = h * w
    s = series_1d
    if len(s) < n_pix:
        s = np.pad(s, (0, n_pix - len(s)), "edge")
    flat = np.zeros(n_pix)
    flat[z_indices] = s[:n_pix]
    return flat.reshape(h, w)

def bootstrap_mean(arr, n_boot=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    flat = arr.flatten()
    n = len(flat)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(flat, size=n, replace=True)
        means.append(np.mean(sample))
    means = np.array(means)
    return np.mean(means), np.percentile(means, 2.5), np.percentile(means, 97.5)

def grad_alignment(phi_field, psi_field, smooth_sigma):
    pf = gaussian_filter(phi_field, sigma=smooth_sigma)
    sf = gaussian_filter(psi_field, sigma=smooth_sigma)
    gpy, gpx = np.gradient(pf)
    gsy, gsx = np.gradient(sf)
    dot = gpx*gsx + gpy*gsy
    norm = np.sqrt(gpx**2+gpy**2)*np.sqrt(gsx**2+gsy**2)
    norm[norm == 0] = 1e-10
    align = dot / norm
    return np.nan_to_num(align, nan=0.0, posinf=0.0, neginf=0.0)

# -----------------------------
# 2. ζ 零点与 θ(t)
# -----------------------------

# t 轴 & θ_base
N_pix_total = CROP_SIZE*CROP_SIZE
t_vals = np.linspace(T0 - 0.5*DT*N_pix_total,
                     T0 + 0.5*DT*N_pix_total,
                     N_pix_total)
t_vals = np.maximum(t_vals, 1.0)
theta_base = 0.5 * np.log(t_vals / (2*np.pi))

print("⏳ 计算 ζ 零点...")
start_zeros = time.time()
n_start = int((T0/(2*np.pi))*np.log(T0/(2*np.pi)))
k_min = max(1, n_start - MZEROS)
k_max = n_start + MZEROS
zeros_real = np.array([float(mp.zetazero(k).imag) for k in range(k_min, k_max)])
print(f"✅ ζ 零点计算完成，用时 {time.time()-start_zeros:.1f}s")

def theta_from_zeros(zeros, t_vals, theta_base):
    θp = theta_base.copy()
    for g in zeros:
        d = t_vals - g
        θp += d / (d*d + 0.25)
    θ = np.cumsum(θp) * (t_vals[1] - t_vals[0])
    θ -= np.mean(θ)
    return θ

def theta_noise(t_vals, theta_base, rng):
    """噪声流：用随机高斯过程替代 ζ-零点"""
    θp = theta_base + rng.normal(loc=0.0, scale=1.0, size=t_vals.shape)
    θ = np.cumsum(θp) * (t_vals[1] - t_vals[0])
    θ -= np.mean(θ)
    return θ

# -----------------------------
# 3. 结构尺度扫描 & 流动尺度扫描
# -----------------------------

def struct_scan(phi, theta_2d, sigma_flow_fixed, rng_boot):
    """Part1：结构尺度扫描（固定流动尺度）"""
    H, W = phi.shape
    results = []
    for σ in SIGMA_STRUCT_RANGE:
        ψ = gaussian_laplace(theta_2d, sigma=σ)
        ψ = (ψ - np.mean(ψ)) / np.std(ψ)
        align_map = grad_alignment(phi, ψ, smooth_sigma=sigma_flow_fixed)
        mean_align, ci_low, ci_high = bootstrap_mean(align_map, n_boot=N_BOOT, rng=rng_boot)
        results.append((σ, mean_align, ci_low, ci_high))
    return np.array(results)

def flow_scan(phi, theta_2d, sigma_struct_fixed, rng_boot):
    """Part2：流动尺度扫描（固定结构尺度）"""
    H, W = phi.shape
    results = []
    ψ = gaussian_laplace(theta_2d, sigma=sigma_struct_fixed)
    ψ = (ψ - np.mean(ψ)) / np.std(ψ)

    for σ_flow in SIGMA_FLOW_RANGE:
        align_map = grad_alignment(phi, ψ, smooth_sigma=σ_flow)
        mean_align, ci_low, ci_high = bootstrap_mean(align_map, n_boot=N_BOOT, rng=rng_boot)
        results.append((σ_flow, mean_align, ci_low, ci_high))
    return np.array(results)

# -----------------------------
# 4. Law I & II: 从结构尺度扫描构造 Φ, H, dt_info, A
# -----------------------------

def law_I_II_from_struct_scan(res_struct, rng_shuffle, n_shuffle=N_SHUFFLE):
    """
    输入：结构尺度扫描结果 res_struct: (σ, mean, ci_low, ci_high)
    输出：t_info(σ), A_real, A_shuffle 分布, p-value
    """
    σ = res_struct[:,0]
    μ = res_struct[:,1]
    lo = res_struct[:,2]
    hi = res_struct[:,3]

    # 结构势 Φ: 越对齐势越低 → Φ = -μ
    Phi = -μ
    # 阻力/熵 H: 采用 CI 宽度
    H = hi - lo
    eps = 1e-6
    H = np.maximum(H, eps)

    # dt_info = dΦ/H
    dPhi = np.diff(Phi)
    dt_info = dPhi / H[:-1]
    t_info = np.concatenate([[0.0], np.cumsum(dt_info)])
    A_real = np.sum(np.abs(dt_info))

    # Law II: 与乱序路径对比
    A_shuffle = np.zeros(n_shuffle)
    idx = np.arange(len(σ))
    for i in range(n_shuffle):
        rng_shuffle.shuffle(idx)
        Phi_sh = Phi[idx]
        H_sh = H[idx]
        dPhi_sh = np.diff(Phi_sh)
        dt_sh = dPhi_sh / H_sh[:-1]
        A_shuffle[i] = np.sum(np.abs(dt_sh))

    p_value = np.mean(A_shuffle <= A_real)
    return {
        "sigma": σ,
        "Phi": Phi,
        "H": H,
        "t_info": t_info,
        "A_real": A_real,
        "A_shuffle": A_shuffle,
        "p_value": p_value
    }

# -----------------------------
# 5. 自动寻找 σ*（过零点临界区）
# -----------------------------

def detect_sigma_star(res_struct):
    """
    根据 mean_align 与 CI:
    - 找 CI 跨 0 的区间
    - σ* 定义为该区间内 |mean| 最小的 σ
    若无 CI 跨 0，则 σ* 定义为全局 |mean| 最小
    """
    σ = res_struct[:,0]
    μ = res_struct[:,1]
    lo = res_struct[:,2]
    hi = res_struct[:,3]

    mask_cross = (lo <= 0) & (hi >= 0)
    if np.any(mask_cross):
        idx = np.where(mask_cross)[0]
        sub = idx[np.argmin(np.abs(μ[mask_cross]))]
        sigma_star = σ[sub]
        return {
            "sigma_star": float(sigma_star),
            "has_cross": True,
            "cross_range": (float(σ[idx[0]]), float(σ[idx[-1]]))
        }
    else:
        j = np.argmin(np.abs(μ))
        return {
            "sigma_star": float(σ[j]),
            "has_cross": False,
            "cross_range": (float(σ[j]), float(σ[j]))
        }

# -----------------------------
# 6. 主循环：三集群 ζ-flow + noise-flow
# -----------------------------

clusters = ["MACS0717", "Abell2744", "MACS0416"]

results = {}  # 存放每个团的所有结果

for cname in clusters:
    print(f"\n==================== {cname} ====================")
    # 固定随机种子以便可重复
    rng_boot = np.random.default_rng(1000)
    rng_noise = np.random.default_rng(hash(cname) & 0xffffffff)
    rng_shuffle = np.random.default_rng(2000)

    # 6.1 加载 κ 场
    phi = load_cluster(cname)
    Hc, Wc = phi.shape
    zmap = z_order_index(Hc, Wc)

    # 6.2 构造 ζ-flow θ_2d
    theta_zeta_1d = theta_from_zeros(zeros_real, t_vals, theta_base)
    theta_zeta_2d = embed_1d_to_2d(theta_zeta_1d, phi.shape, zmap)

    # 6.3 结构尺度扫描 & 流动尺度扫描 (ζ-flow)
    res_struct_zeta = struct_scan(phi, theta_zeta_2d, SIGMA_FLOW_FIXED, rng_boot)
    res_flow_zeta   = flow_scan(phi, theta_zeta_2d, SIGMA_STRUCT_FIXED, rng_boot)

    # 6.4 Law I & II (ζ-flow)
    law_zeta = law_I_II_from_struct_scan(res_struct_zeta, rng_shuffle, n_shuffle=N_SHUFFLE)

    # 6.5 噪声流 θ_2d
    theta_noise_1d = theta_noise(t_vals, theta_base, rng_noise)
    theta_noise_2d = embed_1d_to_2d(theta_noise_1d, phi.shape, zmap)

    # 6.6 结构尺度扫描 (noise-flow)
    res_struct_noise = struct_scan(phi, theta_noise_2d, SIGMA_FLOW_FIXED, rng_boot)
    law_noise = law_I_II_from_struct_scan(res_struct_noise, rng_shuffle, n_shuffle=N_SHUFFLE)

    # 6.7 σ* 自动检测 (ζ-flow / noise-flow)
    sigma_star_zeta  = detect_sigma_star(res_struct_zeta)
    sigma_star_noise = detect_sigma_star(res_struct_noise)

    results[cname] = {
        "phi": phi,
        "res_struct_zeta": res_struct_zeta,
        "res_flow_zeta": res_flow_zeta,
        "law_zeta": law_zeta,
        "res_struct_noise": res_struct_noise,
        "law_noise": law_noise,
        "sigma_star_zeta": sigma_star_zeta,
        "sigma_star_noise": sigma_star_noise
    }

    # 简要打印 σ* & 作用
    print(f"\n[ {cname} · ζ-flow ]")
    print(f"σ* ≈ {sigma_star_zeta['sigma_star']:.2f}, cross={sigma_star_zeta['has_cross']}, range={sigma_star_zeta['cross_range']}")
    print(f"A_real(ζ) = {law_zeta['A_real']:.4f},  p(ζ) ≈ {100*law_zeta['p_value']:.2f}%")
    print(f"[ {cname} · noise-flow ]")
    print(f"σ*_noise ≈ {sigma_star_noise['sigma_star']:.2f}, cross={sigma_star_noise['has_cross']}, range={sigma_star_noise['cross_range']}")
    print(f"A_real(noise) = {law_noise['A_real']:.4f},  p(noise) ≈ {100*law_noise['p_value']:.2f}%")

print("\n✅ 三星系团 ζ-flow + noise-flow 扫描 & I+II 检验完成。")

# -----------------------------
# 7. 可视化：三集群结构 & 流动尺度扫描
# -----------------------------

def plot_struct_scan_all(results):
    plt.figure(figsize=(12,7))
    for cname in clusters:
        res = results[cname]["res_struct_zeta"]
        σ = res[:,0]; μ = res[:,1]; lo = res[:,2]; hi = res[:,3]
        plt.plot(σ, μ, 'o-', label=f"{cname} (ζ-flow)")
        plt.fill_between(σ, lo, hi, alpha=0.2)
    plt.axhline(0, linestyle='--')
    plt.xlabel(r'$\sigma_{\rm struct}$')
    plt.ylabel('Mean Alignment')
    plt.title("Structural Scale Scan (ζ-flow, 3 clusters)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_flow_scan_all(results):
    plt.figure(figsize=(12,7))
    for cname in clusters:
        res = results[cname]["res_flow_zeta"]
        σ = res[:,0]; μ = res[:,1]; lo = res[:,2]; hi = res[:,3]
        plt.plot(σ, μ, 'o-', label=f"{cname} (ζ-flow)")
        plt.fill_between(σ, lo, hi, alpha=0.2)
    plt.axhline(0, linestyle='--')
    plt.xlabel(r'$\sigma_{\rm flow}$')
    plt.ylabel('Mean Alignment')
    plt.title(f"Flow Scale Scan (ζ-flow, σ_struct={SIGMA_STRUCT_FIXED})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_struct_scan_all(results)
plot_flow_scan_all(results)

# -----------------------------
# 8. σ* 分布图 & 表格输出（论文用）
# -----------------------------

print("\n📊 σ* 结果表 (ζ-flow):")
print("Cluster      σ*      CI-cross    cross_range")
sigma_star_list = []

for cname in clusters:
    info = results[cname]["sigma_star_zeta"]
    sigma_star_list.append(info["sigma_star"])
    print(f"{cname:10s} {info['sigma_star']:5.2f}    {str(info['has_cross']):>5s}    {info['cross_range']}")

plt.figure(figsize=(8,5))
x = np.arange(len(clusters))
plt.bar(x, sigma_star_list)
plt.xticks(x, clusters)
plt.ylabel(r'$\sigma^{*}_{\rm struct}$')
plt.title(r'$\sigma^{*}$ Distribution Across Clusters (ζ-flow)')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# -----------------------------
# 9. Law II：ζ-flow vs Noise-flow 作用分布对比
# -----------------------------

plt.figure(figsize=(12,8))

for i, cname in enumerate(clusters):
    law_zeta  = results[cname]["law_zeta"]
    law_noise = results[cname]["law_noise"]
    A_real_z  = law_zeta["A_real"]
    A_real_n  = law_noise["A_real"]
    A_sh_z    = law_zeta["A_shuffle"]
    A_sh_n    = law_noise["A_shuffle"]

    plt.subplot(3,2,2*i+1)
    plt.hist(A_sh_z, bins=40, alpha=0.7, label='ζ-flow random')
    plt.axvline(A_real_z, linestyle='--', linewidth=2, label=f'A_real ζ = {A_real_z:.2f}')
    plt.title(f"{cname} · Law II (ζ-flow)")
    plt.xlabel('A')
    plt.ylabel('count')
    plt.grid(True)
    plt.legend()

    plt.subplot(3,2,2*i+2)
    plt.hist(A_sh_n, bins=40, alpha=0.7, label='noise-flow random')
    plt.axvline(A_real_n, linestyle='--', linewidth=2, label=f'A_real noise = {A_real_n:.2f}')
    plt.title(f"{cname} · Law II (noise-flow)")
    plt.xlabel('A')
    plt.ylabel('count')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

print("✅ σ* 分布图 & Law II ζ vs noise 对比完成。")
