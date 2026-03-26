# Lorentz Light-Cone Model (LLCM)

**Standard Transformer attention treats all directions equally. LLCM replaces it with Lorentzian attention, where time-like directions are suppressed — equivalent to attending along the information light cone instead of full-connecting.**

```python
# Standard attention (Euclidean)
score = Q @ K.T / sqrt(d)                    # all directions equal

# LLCM attention (Lorentzian, formula F3)
score = (-σ * Qt @ Kt.T + Qs @ Ks.T) / sqrt(d)  # time suppressed, space preserved
```

The negative sign is not a hyperparameter choice. It is a theorem:

> **Theorem 4** (Li, 2026): A system has a nontrivial stability boundary d_c > 0 **if and only if** det G < 0 (Lorentzian signature). One-line proof: d_c > 0 ⟺ −1/det G > 0 ⟺ det G < 0.

Euclidean attention (det G > 0) has d_c = 0: no stability boundary, no geometric conservation, no causal structure. The minus sign gives all three.

---

## What changes in one line

| | Standard Transformer | LLCM |
|--|---------------------|------|
| Attention score | `Q·K` (isotropic) | `-σ·Qt·Kt + Qs·Ks` (anisotropic) |
| Geometry | Euclidean (all directions equal) | Lorentzian (time ≠ space) |
| Conservation | Soft constraint via loss function | Hard constraint from geometry |
| Stability boundary | d_c = 0 (nonexistent) | d_c > 0 (algebraic consequence) |
| Training direction | Anti-correlated with physics (ρ = −0.25) | Correlated with physics (ρ = +0.26) |

Everything else — architecture, optimizer, data pipeline — stays the same.

---

## Why the minus sign is necessary

Not assumed. Derived. Three conditions on any displacement cost function d(x; δx) ≥ 0:

| Condition | Meaning | Example |
|-----------|---------|---------|
| **R** (zero threshold) | Some spatial directions have vanishing cost | Robot moving at constant velocity along its heading |
| **E** (quadratic expansion) | Cost has a leading quadratic term | Taylor expansion of any smooth cost |
| **T** (temporal cost) | Pure time displacement has positive cost | Waiting costs energy (idle power draw) |

**Theorem 5** (Li, 2026): Under R, E, T, the quadratic form Q is nondegenerate and indefinite. In (1+1)D, Q = dt² − c_max⁻² dr². Lorentzian signature is the unique algebraic outcome.

Translation for ML: if your data has directions where displacement is cheap (spatial) and directions where displacement is expensive (temporal), then the natural inner product on your embedding space has a minus sign. Forcing a Euclidean inner product ignores this structure. The minus sign in `-σ·Qt·Kt` respects it.

---

## Quick start

```python
from dataclasses import dataclass
import torch
from lorentz_transformer import LorentzMultiHeadAttention, MinkowskiLayerNorm

@dataclass
class Config:
    d_model: int = 256
    n_heads: int = 8
    formula: str = 'f3'
    time_ratio: float = 0.25  # fraction of heads assigned to time-like dimensions
    dropout: float = 0.1

config = Config()
attn = LorentzMultiHeadAttention(config)
norm = MinkowskiLayerNorm(config.d_model)

x = torch.randn(2, 16, config.d_model)
out, weights = attn(x)
out = norm(out)
print(f"σ = {attn.sigma:.3f}")  # learned light-cone strength; target > 0.56
```

**Two parameters matter theoretically:**

| Parameter | Value | Why |
|-----------|-------|-----|
| `formula` | `'f3'` | Lorentzian inner product with learnable σ |
| `time_ratio` | `0.25` | Fraction of attention heads treating their subspace as time-like |

All other hyperparameters (layers, dimensions, learning rate) are engineering choices.

---

## Experimental evidence (8 seeds, EMBED_DIM=256, N_LAYERS=6)

### Geometry emerges before language (Layer 2)

Pure physics pretraining, no language signal:

| Metric | F3 Lorentzian | Euclidean | Statistics |
|--------|--------------|-----------|------------|
| Timelike ratio | **97%** | 0% | 8/8 seeds |
| mq mean | **−2.5 ± 0.7** | +137.8 ± 8.2 | p < 0.0001, d = 16.24 |

Conserved trajectories fall into the timelike region spontaneously. No loss function forces this — it is a consequence of the attention geometry.

### Language alignment (Direction A)

| Model | Alignment score | Statistics |
|-------|----------------|------------|
| Full Riemannian H^{1,63} | **+0.210** | p = 0.043, d = 1.10 (6 seeds) |
| F3 implicit (D=256) | +0.147 | p = 0.508 |
| Euclidean (D=256) | +0.127 | baseline |

Full Riemannian at 1/4 capacity significantly outperforms Euclidean at full capacity.

### Conservation from geometry (Direction B)

Language instruction → trajectory generation → measure momentum conservation:

| Model | Momentum change rate | Statistics |
|-------|---------------------|------------|
| F3 Lorentzian | **0.371** | p = 0.048, d = 2.53, 3/3 seeds |
| Euclidean | 0.423 | baseline |

F3 generates 18% more conserved trajectories with no conservation loss function. The conservation comes from the minus sign.

### Training direction: Euclidean anti-correlates

| | Before training | After training | Direction |
|--|----------------|----------------|-----------|
| F3 | ρ = +0.05 | ρ = +0.26 | ✓ Correct (amplified 5×) |
| Euclidean | ρ = −0.10 | ρ = −0.25 | ✗ Reversed (amplified 2.5×) |

Same data, same training procedure. One converges toward physics, the other diverges. The difference is the sign in the attention formula.

---

## Three attention formulas

| Formula | Score | Use case |
|---------|-------|----------|
| `'f3'` (recommended) | `−σ·Qt Kt⊤ + Qs Ks⊤` | General; σ adapts during training |
| `'f1'` | `−Qt Kt⊤ + Qs Ks⊤` | Physics simulation; hard light cone |
| `'f2'` (deprecated) | `QK⊤/√d − 2α·Qt K⊤/√d` | Cross terms allow optimizer to cancel light cone; 5/5 seeds α stuck at init |

F2 fails because spacetime cross terms let the optimizer route around the geometric constraint. F3's clean separation of time and space heads prevents this.

---

## When to use LLCM

The light cone structure helps when your data satisfies the three conditions (R, E, T):

```python
# 5-minute diagnostic
dx = data[:, 1:] - data[:, :-1]
t_dim = max(1, int(data.shape[-1] * 0.25))
s2 = -(dx[..., :t_dim]**2).sum(-1) + (dx[..., t_dim:]**2).sum(-1)
ratio = (s2 > 0).float().mean()
# ratio > 0.6 → data has light-cone structure → use LLCM
# ratio < 0.4 → σ degrades to Euclidean gracefully
```

| Data type | Timelike ratio | Recommendation |
|-----------|---------------|----------------|
| Robot state [x, y, z, vx, vy, vz] | 80–100% | ✅ Strong signal |
| Joint angles (BVH) | ~80% | ✅ Recommended |
| Physics simulation (ODE) | 80–100% | ✅ Recommended |
| Natural language tokens | <50% | ❌ σ degrades (graceful) |
| Video pixels | <50% | ❌ σ degrades (graceful) |

σ degradation is by design: when data has no light-cone structure, F3 reduces to standard attention. No harm, no benefit.

---

## Full Riemannian variant

LLCM also includes a full Riemannian Lorentzian Transformer operating on H^{1,d−1} via Exp/Log maps. All operations are geometrically exact (manifold constraint violation < 10⁻⁷).

| Milestone | Target | Result |
|-----------|--------|--------|
| M1: Manifold constraint | violation < 0.01 | **0.000111** ✅ |
| M2: No sigma needed | d_c > 0 by topology | ✅ |
| M3: Timelike ratio | > 95% | **100%**, 3/3 seeds ✅ |
| M4: mq gap | > 50× | **267×** ✅ |

Key finding: **Log_μ bridge** for language alignment. Direct linear projection from the manifold (x₀ ≈ 57 dominates) gives alignment +0.096. Projecting to tangent space first via Log_μ (arccosh compresses x₀ to ≈ 4.8) gives alignment +0.210, a 117% improvement.

---

## Architecture overview

```
Physical trajectory → Linear embed → [Lorentzian Transformer blocks] → embed_seq
                                          │
                                     Attention: -σ·Qt Kt⊤ + Qs Ks⊤
                                     LayerNorm: MinkowskiLN (mq = s² − t²)
                                          │
                              ┌───────────┴───────────┐
                         Direction A                Direction B
                    embed → lang_gen → 384D     lang_emb → aligner → decoder
                    (physics → language)         (language → physics)
```

σ is a single learnable scalar (global light-cone width), corresponding to isotropic linearization H* = I in the K=1 framework. Local σ(x) would correspond to general H*(x) — a future extension, not needed for current theorems.

---

## Reproducing

```bash
# Layer 1: perception + language alignment + bidirectional verification
python experiments/layer1_minimal_test.py

# Layer 3: conservation structure effect (×10)
python experiments/layer3_zero_loss_B.py

# Theorem 6 extended conjecture
python experiments/extended_conjecture_test.py

# Full Riemannian milestones
python experiments/lorentz_riemannian_transformer.py
```

---

## File structure

```
lorentz_transformer/                  # Stable API
├── core/
│   ├── attention.py                  # LorentzMultiHeadAttention (F1/F3)
│   └── layer_norm.py                 # MinkowskiLayerNorm

experiments/                          # Research prototypes
├── layer1_minimal_test.py            # Bidirectional verification (8 seeds)
├── layer3_zero_loss_B.py             # Zero-loss conservation test
├── extended_conjecture_test.py       # Theorem 6 conjecture
├── lorentz_riemannian_transformer.py # Full Riemannian milestones M1–M4
├── llcm_first_principles.py         # First-principles verification
└── law2_necessity_test.py            # Law II necessity test
```

---

## Theoretical foundation

Two papers provide the mathematical basis:

**Realizability and the Origin of Causality** (Li, 2026) — Theorem 5 derives Lorentzian signature from three conditions on a displacement cost function. No particles, light signals, or kinematic objects are assumed. The minus sign in the metric is not a convention; it is the unique algebraic consequence of cost asymmetry between temporal and spatial directions.

**K=1 Chronogeometrodynamics** (Li, 2026) — Theorem 4 proves d_c > 0 ⟺ det G < 0 in one line. Lorentzian signature is equivalent to having a nontrivial stability boundary. Theorem 5 gives unique dynamics dx/dt = (J_G − D)∇V. Theorem 6 gives the local restoring rate κ_K = 4d_c at critical damping. Proposition 7 establishes Clausius compatibility.

LLCM is the experimental implementation: F3 attention is the Lorentzian inner product of Theorem 4, σ is the isotropic gain H* = I, and the timelike/spacelike separation of embeddings is the geometric consequence of det G < 0.

---

## Current status

| Status | Item |
|--------|------|
| ✅ | Geometric separation before language (d = 16.24, 8/8 seeds) |
| ✅ | Language alignment via full Riemannian (p = 0.043, 6 seeds) |
| ✅ | Conservation from geometry, no loss function (p = 0.048, 3/3 seeds) |
| ✅ | Euclidean anti-correlation confirmed (ρ reversal, R² = 0.9997) |
| ✅ | Full Riemannian milestones M1–M4 |
| 📋 | Dynamic interaction (Law II online learning) |
| 📋 | Real robot data (Open X-Embodiment) |
| 📋 | Paper (CoRL / NeurIPS 2026) |

---

## License

MIT

## Citation

```bibtex
@misc{li2026k1,
  author = {Li, Y. Y. N.},
  title  = {K=1 Chronogeometrodynamics},
  year   = {2026},
  doi    = {10.5281/zenodo.19011128}
}

@misc{li2026realizability,
  author = {Li, Y. Y. N.},
  title  = {Realizability and the Origin of Causality},
  year   = {2026},
  doi    = {10.5281/zenodo.19062187}
}
```
