# ===============================================================
# 🌀 SCI ↔ DAG-UTH v3 — Time Emergence Colab
# ===============================================================

!pip install astropy matplotlib tqdm requests numpy mpmath -q

import numpy as np, matplotlib.pyplot as plt, requests, os
from astropy.io import fits
from tqdm import tqdm
import mpmath as mp
mp.mp.dps = 50

# -------------------- Dataset URLs --------------------
DATASETS = [
    ("Abell2744_CATSv4", "https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"),
    ("MACS0416_CATSv4",  "https://archive.stsci.edu/pub/hlsp/frontier/macs0416/models/cats/v4/hlsp_frontier_model_macs0416_cats_v4_kappa.fits"),
    ("Abell370_CATSv4",  "https://archive.stsci.edu/pub/hlsp/frontier/abell370/models/cats/v4/hlsp_frontier_model_abell370_cats_v4_kappa.fits")
]

# -------------------- Utils --------------------
def download(url, fname):
    if not os.path.exists(fname):
        print(f"↓ Downloading {fname} ...")
        r = requests.get(url)
        with open(fname, 'wb') as f: f.write(r.content)
    return fname

def zscore(a):
    m, s = np.mean(a), np.std(a) + 1e-12
    return (a - m) / s

def lap2(f):
    """Discrete Laplacian for 2D field"""
    return (
        -4*f +
        np.roll(f,1,0) + np.roll(f,-1,0) +
        np.roll(f,1,1) + np.roll(f,-1,1)
    )

def dag_propagate(f, mu, gamma, step):
    """DAG propagation: diffusion + nonlinear feedback"""
    g = f.copy()
    for _ in range(step):
        lap = lap2(g)
        g = g + mu * lap - gamma * (g - np.tanh(g))
    return g

# -------------------- SCI Computation --------------------
def sci_K_field(data, mu, gamma, steps=25):
    """Compute Φ(t), H(t), K(t) time evolution"""
    f = zscore(data.copy())
    K_vals, Phi_vals, H_vals = [], [], []
    for _ in range(steps):
        pred = dag_propagate(f, mu, gamma, 1)
        Φ = np.mean(pred * f)
        H = np.var(pred - f)
        K_vals.append(Φ / (H + 1e-12))
        Phi_vals.append(Φ)
        H_vals.append(H)
        f = pred
    return np.array(K_vals), np.array(Phi_vals), np.array(H_vals)

# -------------------- Main Experiment --------------------
def run_experiment(name, url):
    print(f"\n=== {name} ===")
    path = download(url, f"{name}.fits")
    data = fits.getdata(path)
    data = np.nan_to_num(data)
    data = data[:2000, :2000]  # crop for stability
    data = zscore(data)
    print(f"✅ Loaded κ-field: {data.shape}")
    # downscale for faster testing
    from skimage.transform import resize
    data = resize(data, (500,500))
    print(f"↓ Resized: {data.shape}")

    μ, γ = 0.20, 0.35
    K, Φ, H = sci_K_field(data, μ, γ, steps=25)

    # Plot time evolution
    t = np.arange(len(K))
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(t, Φ, 'r', label='Φ(t)')
    plt.plot(t, H, 'b', label='H(t)')
    plt.legend(); plt.title('Φ(t), H(t)')
    plt.xlabel("t")

    plt.subplot(1,3,2)
    plt.plot(t, K, 'k', lw=2)
    plt.axhline(1, ls='--', color='gray')
    plt.title('K(t) — SCI Ratio'); plt.xlabel("t")

    plt.subplot(1,3,3)
    dK = np.gradient(K)
    plt.plot(t, dK, 'm', lw=2)
    plt.axhline(0, ls='--', color='gray')
    plt.title('dK/dt (Time Genesis Indicator)')
    plt.tight_layout()
    plt.show()

    # Detect Time Emergence Point
    idx = np.where(np.abs(K-1) > 0.05)[0]
    if len(idx)>0:
        t_emerge = idx[0]
        print(f"🕒 Time Emergence detected at t = {t_emerge}, K = {K[t_emerge]:.3f}")
    else:
        print("❄️ No emergence detected (still in structural equilibrium).")

# -------------------- Run for all datasets --------------------
for name, url in DATASETS:
    run_experiment(name, url)
def check_equilibrium(K_vals):
    if np.all(np.abs(K_vals - 1) < 0.05):
        return "Equilibrium (pre-time)"
    elif np.mean(K_vals) > 1.05:
        return "Post-time structural phase"
    else:
        return "Transition regime"

for name, url in DATASETS:
    path = f"{name}.fits"
    data = fits.getdata(path)
    data = np.nan_to_num(data)
    data = data[:2000, :2000]
    data = zscore(data)
    K, Φ, H = sci_K_field(data, 0.20, 0.35, 10)
    print(f"{name}: {check_equilibrium(K)}, mean(K)={np.mean(K):.3f}")