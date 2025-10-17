# ===== Fast version: Abell 2744 × ζ phase structure (Hadamard / explicit zeros) =====
# !pip -q install astropy mpmath scipy numpy matplotlib

import numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_laplace
from scipy.stats import skew, kurtosis
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import urllib.request, os, warnings
from mpmath import mp
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------- Parameters ----------------
mp.dps   = 30          # precision ↓ for speed
MZEROS   = 300         # zeros per side
DT       = 1.20        # t-step
WZ       = 120.0       # zero-window
SIGMA_LAP = 8.0        # Laplacian scale
URL  = "https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"
CROP = (slice(975,1025), slice(975,1025))

# ---------------- Load κ map ----------------
def dl(url,p="kappa_abell2744.fits"):
    if not os.path.exists(p): urllib.request.urlretrieve(url,p)
    return p
def load(p,crop=None):
    with fits.open(p) as h: d,hdr=h[0].data,h[0].header
    if crop is not None: d=d[crop[0],crop[1]]
    try:
        w=WCS(hdr); sc=np.sqrt(np.abs(np.prod(proj_plane_pixel_scales(w))))*3600
    except: sc=np.nan
    return np.array(d,float), sc
kappa,arc=load(dl(URL),CROP)
H,W=kappa.shape; N=H*W
print("kappa",kappa.shape,"scale",arc)
phi=np.log(np.where(kappa>0,kappa,np.nanmedian(kappa)))
print("skew/kurt:",skew(phi.ravel()),kurtosis(phi.ravel()))

# ---------------- Z-order embed ----------------
def z_idx(H,W):
    y,x=np.indices((H,W))
    def part(n):
        n&=0xFFFF; n=(n|(n<<8))&0x00FF00FF
        n=(n|(n<<4))&0x0F0F0F0F; n=(n|(n<<2))&0x33333333
        n=(n|(n<<1))&0x55555555; return n
    k=(part(x.ravel().astype(np.uint32))<<1)|part(y.ravel().astype(np.uint32))
    return np.argsort(k)
def embed(v,H,W):
    o=z_idx(H,W); out=np.zeros(H*W); out[o]=v[:H*W]; return out.reshape(H,W)

# ---------------- ζ zeros + phase ----------------
def N_vM(T): return (T/(2*np.pi))*(np.log(T/(2*np.pi))-1)+7/8
def t_zero(n):
    z=mp.zetazero(n); return float(z.imag if hasattr(z,"imag") else z)
T0=2000.0; n0=int(round(N_vM(T0))); tn=t_zero(n0)
if tn<T0:
    while t_zero(n0)<T0: n0+=1
else:
    while n0>1 and t_zero(n0)>T0: n0-=1
n_lo,n_hi=max(1,n0-MZEROS),n0+MZEROS
zeros=[]
for k in range(n_lo,n_hi+1):
    zeros.append(t_zero(k))
    if (k-n_lo)%100==0: print(f"  zeros {k-n_lo}/{n_hi-n_lo}")
zeros=np.array(zeros,float)

t=np.linspace(T0-0.5*DT*(N-1),T0+0.5*DT*(N-1),N)
t=np.maximum(t,10.0)
th_p=0.5*np.log(t/(2*np.pi))
for tn in zeros:
    if abs(tn-T0)>0.5*DT*(N-1)+WZ: continue
    d=t-tn; th_p+=d/(d*d+0.25)
theta=np.cumsum((th_p[:-1]+th_p[1:])*0.5*(t[1]-t[0]))
theta=np.concatenate([[0],theta])
theta=(theta-theta.mean())/(theta.std()+1e-12)

# ---------------- Laplacian + plots ----------------
lap=gaussian_laplace(embed(theta,H,W),sigma=SIGMA_LAP)
print("∇²θ skew/kurt:",skew(lap.ravel()),kurtosis(lap.ravel()))

plt.figure(figsize=(12,4))
plt.subplot(1,3,1);plt.imshow(phi,origin='lower');plt.title("phi=ln κ")
plt.subplot(1,3,2);plt.imshow(embed(theta,H,W),origin='lower');plt.title("θ(t) from zeros")
plt.subplot(1,3,3);plt.imshow(lap,origin='lower');plt.title("∇²θ");plt.tight_layout();plt.show()
print("Done (fast mode).")
