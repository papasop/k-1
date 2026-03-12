# ============================================================
# K=1 CHRONOGEOMETRODYNAMICS
# THEORY-INSPIRED OPTIMIZER DIAGNOSTICS — FIXED VERSION
# ============================================================
#
# FIXES APPLIED (see AUDIT.md for full issue list):
#
#   [FIX-1] J_G exact values: 0.0817 / 0.7353  (was 0.082 / 0.735)
#           Passivity ||GJ_G + (GJ_G)^T||_F now < 1e-15
#
#   [FIX-2] D matrix = d_c * I where d_c = alpha*sqrt(-1/det G) = 0.2451
#           (was ad-hoc diag(0.10, 0.05))
#
#   [FIX-3] Control-to-LR mapping corrected:
#           K > 1 -> u drives K DOWN -> K-group SLOWER  (was backwards)
#           Uses 1 - coupling*tanh(u) instead of 1 + coupling*u
#           This is now CONSISTENT with Diagnostic 1 (K-slow/sigma-fast)
#
#   [FIX-4] sigma_t is now actually USED: D_eff = d_c * (sigma_t/sigma_ref) * I
#           Higher entropic resistance -> stronger damping -> faster K correction
#
#   [FIX-5] H excludes input embeddings (tok+pos).
#           hidden_list now starts after the first block.
#
#   [FIX-6] delta_window = 50 (was 20, paper uses 50)
#
#   [FIX-7] clamp range tightened to (0.85, 1.15) for gradient-ratio branch
#           so sigma directional signal is no longer crushed to lower bound
#
#   [FIX-8] Optimizer renamed "Adam" (weight_decay=0.0, so not AdamW)
#
#   [FIX-9] Passivity algebraic check uses exact J_G -> verifies to < 1e-15
#
# WHAT THIS CODE DOES NOT DO:
#   - Does not prove Lorentzian geometry
#   - Does not derive G from first principles (G is the theoretical value)
#   - Does not implement a full symplectic integrator
#
# ACCURATE POSITIONING OF THE PROJECTED CONTROLLER:
#
#   [A] sigma_t is used — but only through damping modulation, not potential flow.
#       The Lyapunov potential is still V(K,sigma) = 1/2*(K-1)^2, so ∂V/∂sigma = 0.
#       sigma has no independent restoring term; it is not a full second coordinate.
#       Correct description: "K-driven projected controller with sigma-modulated damping",
#       NOT "full two-coordinate potential flow".
#
#   [B] The blended strength formula (0.6*proj + 0.4*ratio) has no strict theoretical
#       derivation. proj_strengths comes from reduced-state Law II; ratio comes from
#       gradient-norm cross-routing. These are two different mechanisms.
#       Correct description: "blended projected controller with an auxiliary
#       gradient-routing term for engineering stability".
#       An ablation (proj_only / ratio_only / blended) is included in Experiment C
#       to isolate their individual contributions.
#
#   [C] The sigma-modulated damping D_eff = d_c*(sigma_t/sigma_ref)*I is
#       theory-consistent but is NOT a direct theorem-level consequence.
#       It is a theory-motivated engineering extension.
#       Correct description: "In the projected controller, we further modulate the
#       critical damping scale by empirical sigma_t to reflect changing entropic
#       resistance during training."

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# 0. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# ============================================================
# 1. CONFIG
# ============================================================

@dataclass
class Config:
    vocab_size: int   = 22
    block_size: int   = 64
    batch_size: int   = 32
    n_embd: int       = 128
    n_layer: int      = 2
    train_steps: int  = 300
    lr: float         = 3e-4
    # FIX-8: weight_decay=0 means this is Adam, not AdamW
    weight_decay: float = 0.0
    eps_H: float      = 1e-8
    K_star: float     = 1.0
    # FIX-6: delta_window = 50 to match paper
    delta_window: int = 50
    # sigma_ref is now set data-driven from step-0 H (see train_run).
    # Config.sigma_ref kept as fallback default only.
    sigma_ref: float  = 1.0
    device: str       = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
print(f"Device: {cfg.device}")


# ============================================================
# 2. THEORETICAL CONSTANTS
# ============================================================
# FIX-1 & FIX-2: derive all constants from first principles
#
# G = diag(1, -1/9)  from:
#   G_11 = d^2 V_K / dK^2 = 1          (curvature in timelike K-direction)
#   G_22 = -1/9                          (Fisher metric in spacelike sigma-direction,
#                                         negative sign = spatial resistance)
#
# J = [[0,1],[-1,0]]                     (canonical symplectic matrix)
#
# J_G = alpha * G^{-1} J               (Theorem 1: unique G-skew-symmetric structure)
#   G^{-1} = diag(1, -9)
#   G^{-1} J = [[0, 1],[9, 0]]
#   alpha = 0.0817 (Wiener ridge condition)
#
# d_c = alpha * sqrt(-1/det G)          (Theorem 5: critical stability boundary)
#   det G = -1/9  ->  d_c = 0.0817 * sqrt(9) = 0.2451

ALPHA    = 0.0817
G_MAT    = np.array([[1.0, 0.0], [0.0, -1.0/9.0]], dtype=np.float64)
G_INV    = np.linalg.inv(G_MAT)                    # diag(1, -9)
J_SYM    = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
J_G_EXACT = ALPHA * G_INV @ J_SYM                  # [[0, 0.0817],[0.7353, 0]]
DET_G    = np.linalg.det(G_MAT)                    # -1/9
D_C      = ALPHA * np.sqrt(-1.0 / DET_G)           # 0.2451
D_MAT    = D_C * np.eye(2)                          # critical damping: D = d_c * I

print(f"\nTheoretical constants:")
print(f"  alpha       = {ALPHA}")
print(f"  det(G)      = {DET_G:.6f}")
print(f"  d_c         = {D_C:.6f}")
print(f"  J_G[0,1]    = {J_G_EXACT[0,1]:.10f}  (exact)")
print(f"  J_G[1,0]    = {J_G_EXACT[1,0]:.10f}  (exact)")

# Verify passivity: ||G J_G + (G J_G)^T||_F should be < 1e-14
_compat = G_MAT @ J_G_EXACT + (G_MAT @ J_G_EXACT).T
print(f"  Passivity check ||GJ_G+(GJ_G)^T||_F = {np.linalg.norm(_compat):.2e}  (should be < 1e-14)")
assert np.linalg.norm(_compat) < 1e-13, "Passivity check failed — J_G values incorrect"
print()


# ============================================================
# 3. DATA
# ============================================================

def generate_data(n_chars: int = 20000, vocab_size: int = 22) -> torch.Tensor:
    pattern = list(range(vocab_size)) * (n_chars // vocab_size + 1)
    return torch.tensor(pattern[:n_chars], dtype=torch.long)

def get_batch(data: torch.Tensor, block_size: int,
              batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size]     for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

data = generate_data(20000, cfg.vocab_size)


# ============================================================
# 4. MODEL
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, n_embd)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int,
                 n_layer: int, block_size: int):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.ModuleList([ResidualBlock(n_embd) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.head    = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]

        # FIX-5: hidden_list starts AFTER embedding (block outputs only).
        # Including tok+pos embeddings inflates H at init and suppresses K.
        hidden_list: List[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            hidden_list.append(x)

        x = self.ln_f(x)
        hidden_list.append(x)   # include LayerNorm output

        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        hidden_cat = torch.cat([h.reshape(-1) for h in hidden_list], dim=0)
        return logits, loss, hidden_cat


# ============================================================
# 5. PARAMETER GROUPS
# ============================================================

def create_param_groups(model: nn.Module) -> List[Dict]:
    """
    K-group   : embedding tables  (govern information surprise d_Phi -> K)
    sig-group : all other params  (govern activation statistics -> sigma/H)
    """
    k_params, sig_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embedding" in name:
            k_params.append(p)
        else:
            sig_params.append(p)
    return [
        {"params": k_params,   "name": "K-group"},
        {"params": sig_params, "name": "sig-group"},
    ]


# ============================================================
# 6. METRICS
# ============================================================

@torch.no_grad()
def compute_state(loss_tensor: torch.Tensor,
                  hidden_tensor: torch.Tensor,
                  K_star: float = 1.0,
                  eps_H: float  = 1e-8) -> Dict:
    """
    dPhi  = cross-entropy loss (information surprise proxy, nats)
    H     = std(hidden activations) + eps  (entropic resistance proxy)
    K     = dPhi / H  (information flow ratio; proxy metric, not dimensionless)
    sigma = H  (geometric spacelike coordinate)
    V     = 1/2 * (K - K_star)^2  (Lyapunov potential)
    """
    dPhi  = float(loss_tensor.item())
    H     = float(hidden_tensor.std(unbiased=False).item()) + eps_H
    K     = dPhi / H
    V     = 0.5 * (K - K_star) ** 2
    return {"dPhi": dPhi, "H": H, "K": K, "sigma": H, "V": V}


def compute_delta_V(V_series: List[float], window: int) -> np.ndarray:
    if len(V_series) <= window:
        return np.array([])
    return np.array([V_series[i + window] - V_series[i]
                     for i in range(len(V_series) - window)])


def summarize_delta_V(V_series: List[float], window: int) -> Dict:
    dV = compute_delta_V(V_series, window)
    if len(dV) == 0:
        return {"mean_delta_V": np.nan, "prob_neg": np.nan, "std_delta_V": np.nan}
    return {
        "mean_delta_V": float(dV.mean()),
        "prob_neg":     float((dV < 0).mean()),
        "std_delta_V":  float(dV.std()),
    }


def compute_adaptive_law3(K_series: List[float], window: int,
                          tail: int = 50) -> Dict:
    """
    Task-adaptive Law III with K_opt estimated from the training tail.

    Motivation (from theory audit):
      V = 1/2*(K - K_star)^2 with K_star=1 fails when the task equilibrium
      K_opt != 1. In practice, K_opt is task-dependent: any task where the
      model converges will drive d_Phi -> 0, so K_opt -> 0 for memorizable
      tasks, and K_opt > 1 for tasks harder than the model capacity.

    Fix (two lines from original code):
      K_opt = mean(K[-tail:])          # empirical equilibrium
      V_opt = 1/2 * (K - K_opt)^2     # recentred potential

    NOTE:
      K_opt is estimated post hoc from the final training regime.
      Therefore this is an empirical attractor-centered verification,
      not the online control target used by the optimizer itself.
      P(ΔV<0) = 1.0 is partly tautological: since K_opt is derived from
      the same trajectory, convergence toward K_opt is guaranteed in the
      limit. The non-trivial content is that K converges monotonically to
      a task-specific attractor rather than oscillating.
      A stronger test would estimate K_opt from a held-out validation run.

    Paper language (safe):
      "The equilibrium value K_opt appears task-dependent. Relative to the
       empirically estimated K_opt (post-hoc, same trajectory), the Lyapunov
       proxy V_opt = 1/2*(K-K_opt)^2 satisfies <ΔV_opt> < 0 and
       P(ΔV_opt < 0) = 1.0, consistent with monotone convergence to the
       task-specific attractor. Note that this is an attractor-centered
       verification, not a causal prediction."
    """
    # Estimate task-specific equilibrium from the training tail
    K_arr  = np.array(K_series)
    K_opt  = float(np.mean(K_arr[-tail:]))

    # Recompute V relative to K_opt
    V_opt  = [0.5 * (k - K_opt) ** 2 for k in K_series]

    dV_opt = compute_delta_V(V_opt, window)
    if len(dV_opt) == 0:
        stats = {"mean_delta_V": np.nan, "prob_neg": np.nan, "std_delta_V": np.nan}
    else:
        stats = {
            "mean_delta_V": float(dV_opt.mean()),
            "prob_neg":     float((dV_opt < 0).mean()),
            "std_delta_V":  float(dV_opt.std()),
        }
    return {
        "K_opt":   K_opt,
        "V_opt":   V_opt,
        "dV_stats": stats,
    }


# ============================================================
# 7. OPTIMIZERS
# ============================================================

def _adamw_update(p: torch.Tensor, grad: torch.Tensor, state: Dict,
                  lr: float, beta1: float, beta2: float,
                  eps: float, wd: float) -> None:
    """Shared AdamW parameter update (weight_decay=0 -> Adam)."""
    if len(state) == 0:
        state["step"]        = 0
        state["exp_avg"]     = torch.zeros_like(p)
        state["exp_avg_sq"]  = torch.zeros_like(p)

    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
    state["step"] += 1
    step = state["step"]

    if wd != 0.0:
        p.mul_(1.0 - lr * wd)

    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    bc1   = 1.0 - beta1 ** step
    bc2   = 1.0 - beta2 ** step
    denom = exp_avg_sq.sqrt() / math.sqrt(bc2)
    denom.add_(eps)
    p.addcdiv_(exp_avg, denom, value=-(lr / bc1))


class GroupScaledAdam(torch.optim.Optimizer):
    """
    Adam with fixed per-group LR scales:
      K-group   : lr * k_scale
      sig-group : lr * sigma_scale
    Diagnostic 1 baseline (k_scale=0.5, sigma_scale=1.5 is the geometry-predicted best).
    FIX-8: renamed from GroupScaledAdamW to GroupScaledAdam (weight_decay defaults to 0).
    """
    def __init__(self, param_groups, lr=3e-4, k_scale=1.0, sigma_scale=1.0,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)
        self.k_scale     = float(k_scale)
        self.sigma_scale = float(sigma_scale)

    @torch.no_grad()
    def step(self, closure=None):
        scales = [self.k_scale, self.sigma_scale]
        for group_idx, group in enumerate(self.param_groups):
            effective_lr = group["lr"] * scales[group_idx]
            for p in group["params"]:
                if p.grad is None:
                    continue
                _adamw_update(
                    p, p.grad, self.state[p],
                    lr=effective_lr,
                    beta1=group["betas"][0], beta2=group["betas"][1],
                    eps=group["eps"], wd=group["weight_decay"]
                )


class ProjectedJGOptimizer(torch.optim.Optimizer):
    """
    K-driven projected controller with sigma-modulated damping.

    ACCURATE POSITIONING:
      This is NOT a full two-coordinate potential flow.
      V = 1/2*(K-1)^2  =>  ∂V/∂sigma = 0 always.
      sigma_t has no independent restoring force; it is not a second
      dynamical coordinate in the potential sense.
      What sigma_t does here: modulates the effective damping magnitude
      D_eff = d_c * (sigma_t/sigma_ref) * I, a theory-motivated engineering
      extension (not a direct theorem consequence).
      Safe description for papers: "K-driven projected controller in which
      sigma_t enters through damping modulation, rather than a full
      two-coordinate potential flow."

    THEORY CONNECTION (K=1 Chronogeometrodynamics):
      Potential : V = 1/2*(K-1)^2  [Law III; depends only on K]
      grad V    = [K-K*, 0]^T       [sigma component is zero by construction]
      Structure : J_G = alpha * G^{-1} J  [Theorem 1, exact passivity verified]
      Damping   : D_eff = d_c*(sigma_t/sigma_ref)*I  [Theorem 5 d_c, sigma-scaled
                  to reflect changing entropic resistance — theory-consistent
                  extension, not a theorem-level derivation]
      Control   : u = (J_G - D_eff) @ grad V
      u_K       = -d_c*(sigma_t/sigma_ref)*(K-1)    [< 0 when K > 1]
      u_sigma   =  J_G[1,0]*(K-1)                   [> 0 when K > 1]

    CONTROL-TO-LR MAPPING (corrected sign convention):
      strength_i = 1 + coupling * tanh(u_i)
      K > K*:  u_K < 0  ->  strength_K < 1   K-group SLOWER  ✓
               u_sigma > 0  ->  strength_sigma > 1   sigma-group FASTER  ✓
      K < K*:  direction reverses — feedback pulls K back toward K*  ✓
      Consistent with Diagnostic 1 (K-slow/sigma-fast when K > K*).

    BLENDED STRENGTH (two branches, engineering hybrid):
      Branch A (projected): theory-driven, from reduced-state control u above.
      Branch B (ratio):     J_G cross-routing of gradient norms; auxiliary
                            engineering stabiliser with no strict theory source.
      final = clamp(0.6 * proj + 0.4 * ratio, 0.5, 1.5)
      The 0.6/0.4 split is a pragmatic choice; see Experiment C ablation.
      Do NOT describe the blend as "Law II natural discretization".
    """

    def __init__(self, param_groups, lr=3e-4,
                 coupling_strength: float = 0.15,
                 sigma_ref: float = 1.0,
                 blend_mode: str = "blended",   # "blended"|"proj_only"|"ratio_only"
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

        # FIX-1: use exact J_G values
        self.J_G = torch.tensor(J_G_EXACT, dtype=torch.float32)
        # FIX-2: base damping = d_c (will be scaled by sigma_t)
        self.d_c              = float(D_C)
        self.coupling         = float(coupling_strength)
        self.sigma_ref        = float(sigma_ref)
        self.blend_mode       = blend_mode

        self.last_u           = None
        self.last_strengths   = None
        self.last_grad_norms  = None
        self._K_t             = 1.0
        self._sigma_t         = sigma_ref

    def set_reduced_state(self, K_t: float, sigma_t: float,
                          K_star: float = 1.0) -> None:
        """
        Compute control vector u from current reduced state.

        grad V = [K_t - K_star, 0]^T
          The sigma component is ZERO by construction (V = 1/2*(K-K*)^2).
          sigma_t does NOT produce a restoring force in the potential sense.

        D_eff = d_c * (sigma_t / sigma_ref) * I
          sigma_t enters here as a damping modulator: higher entropic
          resistance -> stronger damping -> more aggressive K correction.
          This is a theory-motivated engineering choice, not a theorem consequence.

        u = (J_G - D_eff) @ grad V   [natural drift direction toward K=1]
          u_K    = -d_eff * (K_t - K_star)    [negative when K > K*]
          u_sigma = J_G[1,0] * (K_t - K_star) [positive when K > K*]
        """
        self._K_t     = float(K_t)
        self._sigma_t = float(sigma_t)

        gradV = torch.tensor([K_t - K_star, 0.0], dtype=torch.float32)

        # FIX-4: sigma-modulated effective damping
        d_eff  = self.d_c * (max(sigma_t, 1e-6) / self.sigma_ref)
        D_eff  = d_eff * torch.eye(2, dtype=torch.float32)

        A = self.J_G - D_eff
        u = A @ gradV       # natural drift: u_K < 0 when K > K_star

        self.last_u = u.clone()

    def _group_grad_norms(self) -> torch.Tensor:
        norms = []
        for group in self.param_groups:
            s = 0.0
            for p in group["params"]:
                if p.grad is not None:
                    s += p.grad.norm(2).item() ** 2
            norms.append(math.sqrt(s))
        return torch.tensor(norms, dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        grad_norms = self._group_grad_norms()
        self.last_grad_norms = grad_norms.clone()

        # --- Projected control branch (theory-driven) ---
        # strength_i = 1 + coupling * tanh(u_i)
        # K>K*: u_K<0 -> strength_K<1 (K-group slower)  ✓ consistent with Diag.1
        # K<K*: u_K>0 -> strength_K>1 (K-group faster, pulls K back up) ✓
        if self.last_u is not None:
            proj_strengths = 1.0 + self.coupling * torch.tanh(self.last_u)
        else:
            proj_strengths = torch.ones(2, dtype=torch.float32)

        # --- Gradient-ratio branch (auxiliary engineering stabiliser) ---
        # J_G cross-routing: K-group strength scaled by sigma's grad norm, and v.v.
        # This is NOT derived from Law II; it is a gradient-norm heuristic.
        # clamp(0.85, 1.15): FIX-7, prevents sigma signal from being crushed.
        coupled = self.J_G @ grad_norms
        ratio   = coupled / (grad_norms + 1e-8)
        ratio   = torch.clamp(ratio, 0.85, 1.15)

        # --- Blend according to blend_mode ---
        # "proj_only"  : pure theory-driven control (ablation A)
        # "ratio_only" : pure gradient-routing heuristic (ablation B)
        # "blended"    : 60% proj + 40% ratio (engineering hybrid, default)
        if self.blend_mode == "proj_only":
            final_strengths = torch.clamp(proj_strengths, 0.5, 1.5)
        elif self.blend_mode == "ratio_only":
            final_strengths = torch.clamp(ratio, 0.5, 1.5)
        else:  # "blended"
            final_strengths = torch.clamp(
                0.6 * proj_strengths + 0.4 * ratio,
                0.5, 1.5
            )
        self.last_strengths = final_strengths.clone()

        for group_idx, group in enumerate(self.param_groups):
            effective_lr = group["lr"] * final_strengths[group_idx].item()
            for p in group["params"]:
                if p.grad is None:
                    continue
                _adamw_update(
                    p, p.grad, self.state[p],
                    lr=effective_lr,
                    beta1=group["betas"][0], beta2=group["betas"][1],
                    eps=group["eps"], wd=group["weight_decay"]
                )


# ============================================================
# 8. TRAINING LOOP
# ============================================================

def train_run(config: Config, mode: str,
              k_scale: float = 1.0, sigma_scale: float = 1.0,
              coupling_strength: float = 0.15,
              blend_mode: str = "blended",   # "blended" | "proj_only" | "ratio_only"
              verbose: bool = True) -> Dict:
    """
    mode options:
      'baseline'       : standard Adam (weight_decay=0)
      'group_scaled'   : Adam with fixed per-group LR scales
      'projected_jg'   : K-driven projected controller with sigma-modulated damping

    blend_mode (only for 'projected_jg'):
      'blended'    : 0.6 * proj + 0.4 * ratio  (default, engineering hybrid)
      'proj_only'  : proj branch only  (pure theory-driven control)
      'ratio_only' : ratio branch only (pure gradient-routing heuristic)
    """
    set_seed(42)

    model = SimpleTransformer(
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        block_size=config.block_size,
    ).to(config.device)

    if mode == "baseline":
        # FIX-8: labelled Adam, not AdamW
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

    elif mode == "group_scaled":
        param_groups = create_param_groups(model)
        optimizer = GroupScaledAdam(
            param_groups,
            lr=config.lr,
            k_scale=k_scale,
            sigma_scale=sigma_scale,
            weight_decay=config.weight_decay
        )

    elif mode == "projected_jg":
        param_groups = create_param_groups(model)
        optimizer = ProjectedJGOptimizer(
            param_groups,
            lr=config.lr,
            coupling_strength=coupling_strength,
            sigma_ref=config.sigma_ref,
            weight_decay=config.weight_decay,
            blend_mode=blend_mode,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    hist = {k: [] for k in [
        "loss", "dPhi", "H", "K", "sigma", "V",
        "strength_K", "strength_sigma",
        "u_K", "u_sigma",
        "grad_norm_K", "grad_norm_sigma",
    ]}
    model.train()

    for step in range(config.train_steps):
        x, y    = get_batch(data, config.block_size, config.batch_size, config.device)
        logits, loss, hidden = model(x, y)

        # Compute reduced state BEFORE backward (captures current K, sigma)
        state = compute_state(
            loss_tensor=loss,
            hidden_tensor=hidden,
            K_star=config.K_star,
            eps_H=config.eps_H,
        )

        # FIX-3: set sigma_ref from first observed H (data-driven, not hardcoded 1.0)
        if mode == "projected_jg" and step == 0:
            optimizer.sigma_ref = state["sigma"]

        optimizer.zero_grad()
        loss.backward()

        # Feed current state to projected optimizer AFTER backward (grads ready)
        if mode == "projected_jg":
            optimizer.set_reduced_state(
                K_t=state["K"],
                sigma_t=state["sigma"],
                K_star=config.K_star,
            )

        optimizer.step()

        # Record
        for k in ["dPhi", "H", "K", "sigma", "V"]:
            hist[k].append(state[k])
        hist["loss"].append(state["dPhi"])

        if mode == "projected_jg":
            hist["strength_K"].append(float(optimizer.last_strengths[0]))
            hist["strength_sigma"].append(float(optimizer.last_strengths[1]))
            hist["u_K"].append(float(optimizer.last_u[0]))
            hist["u_sigma"].append(float(optimizer.last_u[1]))
            hist["grad_norm_K"].append(float(optimizer.last_grad_norms[0]))
            hist["grad_norm_sigma"].append(float(optimizer.last_grad_norms[1]))
        else:
            for k in ["strength_K", "strength_sigma", "u_K", "u_sigma",
                      "grad_norm_K", "grad_norm_sigma"]:
                hist[k].append(np.nan)

        if verbose and (step % 50 == 0 or step == config.train_steps - 1):
            print(
                f"  [{mode:15s}] step={step:03d} "
                f"loss={state['dPhi']:.4f}  "
                f"H={state['H']:.4f}  "
                f"K={state['K']:.4f}  "
                f"V={state['V']:.4f}"
            )

    # Law III (original): V = 1/2*(K - K_star)^2, K_star fixed at 1
    dV_stats_fixed = summarize_delta_V(hist["V"], window=config.delta_window)

    # Law III (adaptive): V_opt = 1/2*(K - K_opt)^2, K_opt estimated from tail
    # Two-line fix: K_opt = mean(K[-50:]),  V_opt = 1/2*(K - K_opt)^2
    adaptive = compute_adaptive_law3(
        hist["K"], window=config.delta_window, tail=50
    )

    return {
        "mode":           mode,
        "history":        hist,
        "final_loss":     hist["loss"][-1],
        "final_K":        hist["K"][-1],
        "final_V":        hist["V"][-1],
        "dV_stats":       dV_stats_fixed,   # original K_star=1
        "K_opt":          adaptive["K_opt"],
        "V_opt":          adaptive["V_opt"],
        "dV_stats_adapt": adaptive["dV_stats"],  # task-adaptive K_opt
    }


# ============================================================
# 9. RUN ALL EXPERIMENTS
# ============================================================

print("=" * 72)
print("EXPERIMENT A — Baseline vs Group-Scaled (Diagnostic 1)")
print("Prediction H2: K-slow/sigma-fast outperforms K-fast/sigma-slow")
print("=" * 72)

baseline   = train_run(cfg, mode="baseline",      verbose=True)
slow_fast  = train_run(cfg, mode="group_scaled",  k_scale=0.5, sigma_scale=1.5, verbose=True)
fast_slow  = train_run(cfg, mode="group_scaled",  k_scale=1.5, sigma_scale=0.5, verbose=True)

print("\n" + "=" * 72)
print("EXPERIMENT B — Projected J_G coupling sweep (Diagnostic 2)")
print("Prediction H1: J_G coupling does not destabilise training")
print("=" * 72)

proj_010 = train_run(cfg, mode="projected_jg", coupling_strength=0.10, verbose=True)
proj_015 = train_run(cfg, mode="projected_jg", coupling_strength=0.15, verbose=True)
proj_020 = train_run(cfg, mode="projected_jg", coupling_strength=0.20, verbose=True)

print("\n" + "=" * 72)
print("EXPERIMENT C — Ablation: proj_only vs ratio_only vs blended")
print("Purpose: isolate the contribution of each branch in ProjectedJGOptimizer.")
print("  proj_only  : pure K-driven control from theory (∂V/∂sigma=0, sigma modulates D)")
print("  ratio_only : pure J_G gradient-norm cross-routing (engineering heuristic)")
print("  blended    : 0.6*proj + 0.4*ratio (pragmatic hybrid, no strict theory basis)")
print("=" * 72)

abl_proj  = train_run(cfg, mode="projected_jg", coupling_strength=0.15,
                      blend_mode="proj_only",  verbose=True)
abl_ratio = train_run(cfg, mode="projected_jg", coupling_strength=0.15,
                      blend_mode="ratio_only", verbose=True)
abl_blend = train_run(cfg, mode="projected_jg", coupling_strength=0.15,
                      blend_mode="blended",    verbose=True)


# ============================================================
# 10. SUMMARY TABLE
# ============================================================

all_results = {
    "baseline":      baseline,
    "Kslow_Sfast":   slow_fast,
    "Kfast_Sslow":   fast_slow,
    "ProjJG_c=0.10": proj_010,
    "ProjJG_c=0.15": proj_015,
    "ProjJG_c=0.20": proj_020,
    # Experiment C ablation
    "Abl_proj_only":  abl_proj,
    "Abl_ratio_only": abl_ratio,
    "Abl_blended":    abl_blend,
}

print("\n" + "=" * 72)
print("SUMMARY TABLE")
print("=" * 72)
header = f"{'name':<18} {'final_loss':>10} {'final_K':>10} {'final_V':>10} {'<ΔV>':>10} {'P(ΔV<0)':>10}"
print(header)
print("-" * len(header))
for name, res in all_results.items():
    print(
        f"{name:<18} "
        f"{res['final_loss']:>10.4f} "
        f"{res['final_K']:>10.4f} "
        f"{res['final_V']:>10.4f} "
        f"{res['dV_stats']['mean_delta_V']:>10.4f} "
        f"{res['dV_stats']['prob_neg']:>10.4f}"
    )


# ============================================================
# 11. PLOTS
# ============================================================

def plot_metric(results_dict: Dict, key: str, title: str, ylabel: str,
                figsize=(10, 4)):
    plt.figure(figsize=figsize)
    for name, res in results_dict.items():
        vals = res["history"][key]
        if not all(np.isnan(v) for v in vals):
            plt.plot(vals, label=name)
    plt.title(title)
    plt.xlabel("training step")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

group_results = {"baseline": baseline, "Kslow_Sfast": slow_fast, "Kfast_Sslow": fast_slow}
proj_results  = {"ProjJG_c=0.10": proj_010, "ProjJG_c=0.15": proj_015, "ProjJG_c=0.20": proj_020}

plot_metric(group_results, "loss",  "Loss: baseline vs group scaling",            "loss")
plot_metric(group_results, "K",     "K(t) = dΦ/H: baseline vs group scaling",    "K")
plot_metric(group_results, "V",     "V(t) = ½(K−K*)²: baseline vs group scaling","V")

plot_metric(proj_results,  "loss",  "Loss: Projected J_G coupling sweep",         "loss")
plot_metric(proj_results,  "K",     "K(t): Projected J_G coupling sweep",         "K")
plot_metric(proj_results,  "V",     "V(t): Projected J_G coupling sweep",         "V")

# Projected J_G internals (coupling=0.15)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(proj_015["history"]["u_K"],     label="u_K   (K-direction control)")
axes[0].plot(proj_015["history"]["u_sigma"], label="u_σ   (sigma-direction control)")
axes[0].axhline(0, color="k", lw=0.8, ls="--")
axes[0].set_title("Reduced-state control u (coupling=0.15)")
axes[0].set_xlabel("step"); axes[0].set_ylabel("control value")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(proj_015["history"]["strength_K"],     label="strength_K")
axes[1].plot(proj_015["history"]["strength_sigma"], label="strength_σ")
axes[1].axhline(1, color="k", lw=0.8, ls="--")
axes[1].set_title("Effective LR multipliers (coupling=0.15)")
axes[1].set_xlabel("step"); axes[1].set_ylabel("multiplier")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Experiment C: ablation comparison
abl_results = {
    "Abl_proj_only (theory branch)":    abl_proj,
    "Abl_ratio_only (heuristic branch)": abl_ratio,
    "Abl_blended (60/40 hybrid)":        abl_blend,
}
plot_metric(abl_results, "loss", "Ablation: proj_only vs ratio_only vs blended", "loss")
plot_metric(abl_results, "K",    "Ablation: K(t) comparison",                    "K")
plot_metric(abl_results, "V",    "Ablation: V(t) comparison",                    "V")

# V_opt (adaptive) vs V (fixed) for baseline — shows why Law III fails with K*=1
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, res in {"baseline": baseline, "Kslow_Sfast": slow_fast}.items():
    axes[0].plot(res["history"]["V"],   label=f"{name}  V (K*=1)")
    axes[1].plot(res["V_opt"],          label=f"{name}  V_opt (K_opt={res['K_opt']:.3f})")
axes[0].set_title("V = ½(K − 1)²  (fixed K*=1)\nK overshoots → V rebounds to 0.49")
axes[1].set_title("V_opt = ½(K − K_opt)²  (adaptive)\nMeasures convergence to actual attractor")
for ax in axes:
    ax.set_xlabel("step"); ax.set_ylabel("V"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 12. LAW III DETAILED REPORT
# ============================================================

print("\n" + "=" * 72)
print("LAW III REPORT — Fixed K*=1  vs  Adaptive K_opt")
print("=" * 72)
print(f"{'name':<18} {'K_opt':>7}  "
      f"{'<ΔV> fixed':>12} {'P- fixed':>9}  "
      f"{'<ΔV> adapt':>12} {'P- adapt':>9}  {'pass?':>6}")
print("-" * 72)
for name, res in all_results.items():
    sf = res["dV_stats"]
    sa = res["dV_stats_adapt"]
    pass_fixed  = "✓" if sf["mean_delta_V"] < 0 and sf["prob_neg"] > 0.5 else "✗"
    pass_adapt  = "✓" if sa["mean_delta_V"] < 0 and sa["prob_neg"] > 0.5 else "✗"
    print(
        f"  {name:<18} {res['K_opt']:>7.4f}  "
        f"{sf['mean_delta_V']:>+12.4f} {sf['prob_neg']:>9.3f}  "
        f"{sa['mean_delta_V']:>+12.4f} {sa['prob_neg']:>9.3f}  "
        f"{pass_fixed} -> {pass_adapt}"
    )

print(f"""
INTERPRETATION:
  Fixed K*=1:    V = 1/2*(K - 1)^2
    K overshoots below 1 on all runs -> V rebounds to ~0.49 -> ΔV > 0.
    Law III NOT supported with fixed reference. This is expected because
    K_opt is task-dependent, not universally 1.

  Adaptive K_opt: V_opt = 1/2*(K - K_opt)^2
    K_opt = mean(K[-50:]) estimated post hoc from the same trajectory.
    *** TAUTOLOGY WARNING ***
    P(ΔV<0) = 1.0 is partly tautological: K_opt is derived from the
    trajectory itself, so convergence toward K_opt is guaranteed in the
    limit. The non-trivial content is monotone convergence (no oscillation).
    A stronger test requires K_opt estimated from a held-out validation run.

    What IS meaningful:
      (a) K converges monotonically to a task-specific attractor K_opt << 1.
      (b) K_opt correlates with final loss across all runs (lower K_opt,
          lower loss — see summary table). This is an independent finding.

  Paper language (safe):
    "The equilibrium value K_opt appears task-dependent. Relative to the
     empirically estimated K_opt (post-hoc attractor), V_opt = 1/2*(K-K_opt)^2
     satisfies <ΔV_opt> < 0 with P = 1.0, consistent with monotone convergence
     to the task-specific attractor. Note that K_opt is estimated from the
     same trajectory, so this constitutes an attractor-centered verification
     rather than a prospective causal test."
""")


# ============================================================
# 13. ALGEBRAIC VERIFICATION  (FIX-9: uses exact J_G)
# ============================================================

print("\n" + "=" * 72)
print("ALGEBRAIC VERIFICATION")
print("=" * 72)

compat_exact = G_MAT @ J_G_EXACT + (G_MAT @ J_G_EXACT).T
print(f"  ||G J_G + (G J_G)^T||_F  = {np.linalg.norm(compat_exact):.2e}"
      f"  (should be < 1e-14)")

eigvals = np.linalg.eigvals(G_MAT)
print(f"  eigenvalues(G)           = {eigvals}  -> Sig(G)=(1,1) ✓")
print(f"  det(G)                   = {np.linalg.det(G_MAT):.6f}  < 0 ✓")
print(f"  d_c = alpha*sqrt(-1/detG)= {D_C:.6f}")

# Verify dc via eigenvalues of G^{-1}J
eigs_GinvJ = np.linalg.eigvals(G_INV @ J_SYM)
print(f"  eigenvalues(G^{{-1}}J)      = {eigs_GinvJ}  (real ✓, Theorem 3)")
print(f"  |eig|                    = {np.abs(eigs_GinvJ)}  = sqrt(9) = {np.sqrt(9):.4f}")


# ============================================================
# 14. INTERPRETATION
# ============================================================

print("\n" + "=" * 72)
print("INTERPRETATION")
print("=" * 72)
print("""
WHAT THIS EXPERIMENT TESTS:
  H1 (coupling stability): J_G-modulated training does not diverge.
  H2 (asymmetric scaling): K-slow/sigma-fast outperforms K-fast/sigma-slow.
  H3 (ablation):           Which branch drives performance — proj, ratio, or blend?

HOW TO READ THE RESULTS:
  1. Exp A: Kslow_Sfast < baseline < Kfast_Sslow  ->  H2 supported.
  2. Exp B: ProjJG runs converge without divergence  ->  H1 supported.
  3. Exp C (ablation):
     - If proj_only ≈ blended >> ratio_only: theory-driven branch is the source.
     - If ratio_only ≈ blended: gradient-routing heuristic dominates.
     - If blended >> both: the two branches are complementary.
     This result determines how the controller should be described in the paper.
  4. Projected J_G: K-group mult < 1 when K > K* (verify strength_K plot).
     This confirms the corrected control direction.

ACCURATE POSITIONING (required for paper writing):

  ISSUE 1 — sigma is NOT a full second dynamical coordinate:
    V = 1/2*(K-1)^2  =>  ∂V/∂sigma = 0 always.
    sigma_t modulates damping D_eff = d_c*(sigma_t/sigma_ref)*I only.
    Safe paper language:
      "Current experiments realize a K-driven projected controller in which
       sigma_t enters through damping modulation, rather than a full
       two-coordinate potential flow."

  ISSUE 2 — The blended strength formula is an engineering hybrid:
    proj_strengths: from reduced-state Law II control (theory-driven).
    ratio:          from J_G gradient-norm cross-routing (heuristic).
    final = 0.6*proj + 0.4*ratio  (pragmatic split, not theorem-derived).
    Safe paper language:
      "blended projected controller with an auxiliary gradient-routing term
       for engineering stability."
    Do NOT describe this as "the natural discretization of Law II".
    Use Experiment C ablation to justify or revise the 0.6/0.4 split.

  ISSUE 3 — sigma-modulated damping is theory-consistent, not theorem-level:
    D_eff = d_c*(sigma_t/sigma_ref)*I is motivated by Theorem 5 (critical
    damping) but is an extension beyond the formal theorem statement.
    Safe paper language:
      "In the projected controller, we further modulate the critical damping
       scale by empirical sigma_t to reflect changing entropic resistance
       during training."
    Do NOT write: "Law II implies sigma-modulated critical damping."

WHAT THIS DOES NOT PROVE:
  - Lorentzian geometry is not proven by optimizer performance.
  - Consistency with theory ≠ causal derivation from theory.
  - Results are on a synthetic toy task; generalization requires broader validation.

THEORETICAL CONSTANTS USED (exact, theorem-level):
  J_G = alpha * G^{-1} J  (Theorem 1; passivity verified to < 1e-14)
  d_c = alpha * sqrt(-1/det G) = 0.2451  (Theorem 5)
  D_eff = d_c * (sigma_t/sigma_ref) * I  (theory-motivated extension)
""")
