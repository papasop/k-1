# Lorentz Light-Cone Model (LLCM)

**LLCM is the engineering substitute for Law II dynamics under backpropagation.**
Theorem 4 (Li, 2026) proves that Lorentzian signature is the unique condition for
a stability attractor at d_c > 0, but the proof assumes Law II dynamics
`dx/dt = (J_G − D)∇V`. Backpropagation does not implement Law II — and we show
experimentally that under backprop alone, the Theorem 4 attractor does not form
(mq diverges from −1, gradient signal is anti-aligned). LLCM replaces the missing
dynamics with three explicit components: Lorentzian attention (F3), Minkowski
LayerNorm, and a manifold constraint loss. Together, they reproduce the geometry
Theorem 4 predicts and yield measurable downstream gains on physics-structured data.

```python
# Standard attention (Euclidean) — directionally isotropic
score = Q @ K.T / sqrt(d)

# F3 attention (Lorentzian, anisotropic)
score = (-σ * Qt @ Kt.T + Qs @ Ks.T) / sqrt(d)
```

The minus sign is not a hyperparameter. Theorem 5 (Li, 2026) derives it from
three asymmetry conditions on any displacement cost: cost vanishes in some
directions (R), expands quadratically (E), and is strictly positive in time (T).
Under R+E+T, the metric signature is uniquely Lorentzian — det G < 0.

The minus sign in F3 is the discrete-attention realization of this signature
on token embeddings. Whether that realization actually produces the Theorem 4
attractor is an empirical question about backprop, not a corollary of the theorem.
We answer that question below.

---

## What is shown (and what is not)

LLCM does **not** show that Lorentzian geometry "emerges" from training. Earlier
versions of this README claimed that; the claim does not survive a clean
ablation. What LLCM does show:

| Claim | Evidence | Status |
|---|---|---|
| Lorentzian geometry **does not emerge** under pure backprop | Section 4 | ✅ Demonstrated |
| Three explicit components are sufficient to maintain the geometry | Section 5 | ✅ Demonstrated |
| Maintained geometry yields measurable downstream gains | Section 6 | ✅ Demonstrated |

The original "geometry emerges before language" framing was wrong — what was
observed in earlier experiments was geometry being maintained by an asymmetric
loss recipe, not emerging from F3 attention alone. Section 4 explains why.

---

## Section 4 — Backpropagation does not implement Law II

This section is the core of the paper. It establishes that the geometric
machinery in LLCM is necessary, not decorative.

**Setup.** A plain Euclidean Transformer (D=64, 2 layers) is trained on a
binary classification task over physics trajectories (stable vs. dissipative),
with three loss configurations:

- **baseline**: classification loss only
- **signature**: classification + soft incentive `-0.01·E[emb₀²]` on the time component
- **mf**: classification + explicit constraint `(mq+1)²` where `mq = -emb₀² + Σemb_i²`

For each configuration we measure the final value of mq, the fraction of samples
in the timelike region (mq < 0), and the gradient alignment between ∇loss and
∇(mq+1)² in the embedding space.

**Results (3 seeds, D=64, 80 epochs):**

| Mode | mq_init | mq_final | timelike % | accuracy | cos(∇loss, ∇mq) |
|---|---|---|---|---|---|
| baseline | +68.7 | **+164.5** | 0% | 100.0% | **−0.25** |
| signature | +68.7 | **+2000.6** | 0% | 95.3% | −0.06 |
| mf | +68.7 | **−0.89** | 89% | 96.8% | −0.02 |

The expected initialization mq for random d=64 embeddings is `d − 2 = 62`,
matching observation. So the attractor at mq = −1 sits 63 units away from
the natural starting region.

**Reading the table:**

1. **Backprop actively repels the attractor.** The cos = −0.25 in baseline is
   not "no signal" — it is anti-alignment. The gradient that minimizes
   classification loss systematically points away from mq = −1.

2. **Soft incentives backfire.** The `signature` configuration adds a small pull
   on the time component. The optimizer routes around it by inflating all
   embedding components — mq blows up to +2000.

3. **Explicit constraint works, and is nearly free.** Adding `(mq+1)²`
   collapses mq to −0.89 (89% timelike) at a cost of 3 percentage points of
   classification accuracy.

4. **The signal-to-noise problem is structural.** In the trained baseline
   embedding, the gradient norm in the mq=−1 direction is ~10⁷ times smaller
   than the loss gradient norm. Even if the direction were correct, the signal
   would sit far below the optimizer's effective floor.

**Conclusion.** Theorem 4's attractor exists in Law II dynamics but not in
backpropagation. The geometric structure must be installed, not inferred.

```bash
# Reproduce
python experiments/law2_necessity_test.py
```

---

## Section 5 — LLCM as the minimum engineering substitute

LLCM provides three components that together restore the missing dynamics:

| Component | Role | What it substitutes for |
|---|---|---|
| **F3 attention** | `score = -σ·Qt·Kt + Qs·Ks` | Lorentzian inner product on tokens |
| **Minkowski LayerNorm** | Normalizes by `\|s²−t²\|` instead of `\|x\|²` | Preserves signature through layer operations |
| **Manifold loss / project** | `L_mf = (mq+1)²` or hard projection | Replaces the Law II flow toward {K=1} |

The three components are not interchangeable conveniences. F3 alone (without
the loss) does not maintain geometry under backprop, as Section 4 establishes.
The loss alone (without F3) maintains mq = −1 but loses the directional
asymmetry needed for the downstream effects in Section 6.

### Riemannian variant (full geodesic operations on H^{1,d−1})

The Riemannian version operates on the manifold via Exp/Log maps, with all
operations geometrically exact:

| Milestone | Target | Result | Status |
|---|---|---|---|
| M1: Manifold constraint violation | < 0.01 | 0.000111 | ✅ |
| M2: σ-independence | d_c > 0 by topology | ✅ |
| M3: Timelike ratio | > 95% | 100% (3/3 seeds) | ✅ |
| M4: mq separation vs Euclidean | > 50× | 267× | ✅ |

The Riemannian version is the upper bound. F3 is the lightweight realization
that captures most of the geometric structure with standard attention machinery.

---

## Section 6 — Downstream gains on physics-structured data

When the geometry is maintained (via the three-part substitute), F3 outperforms
Euclidean baselines on physics trajectory tasks. These are not evidence of
geometric emergence; they are evidence that the maintained geometry has
downstream value.

**Direction A — physics trajectory → language alignment:**

| Model | Alignment score | Statistics |
|---|---|---|
| Full Riemannian H^{1,63} | **+0.210** | p = 0.043, d = 1.10 (6 seeds) |
| F3 (D=256) | +0.147 | p = 0.508 |
| Euclidean (D=256) | +0.127 | baseline |

Note that all three models receive the same geometric separation losses
(`loss_push_s`, `loss_push_c`, `loss_sigma`) — the only architectural
difference is the attention formula. This is the fair-comparison protocol.

**Direction B — language instruction → conserved trajectory:**

| Model | Momentum change rate | Statistics |
|---|---|---|
| F3 Lorentzian | **0.371** | p = 0.048, d = 2.53, 3/3 seeds |
| Euclidean | 0.423 | baseline |

F3-generated trajectories from a "stable conservation" instruction exhibit 18%
lower momentum change rate, with no conservation loss term — the conservation
is mediated by the maintained geometry.

**Important caveat.** With n=3 to n=6 seeds, p-values near 0.05 are fragile.
The current results indicate a directional effect; confirmatory statistics
require expansion to n ≥ 10.

---

## When LLCM applies

The three-part substitute is worth installing when the data has light-cone
structure under the R+E+T conditions: directions of vanishing displacement
cost (spatial-like) coexist with directions of strictly positive cost
(temporal-like). A 5-minute diagnostic:

```python
# Compute timelike ratio under candidate (t, s) split
dx = data[:, 1:] - data[:, :-1]
t_dim = max(1, int(data.shape[-1] * 0.25))
s2 = -(dx[..., :t_dim]**2).sum(-1) + (dx[..., t_dim:]**2).sum(-1)
ratio = (s2 > 0).float().mean()
# ratio > 0.6 → data has light-cone structure → LLCM helps
# ratio < 0.4 → σ degrades to Euclidean gracefully
```

| Data type | Timelike ratio | Recommendation |
|---|---|---|
| Robot state [x, y, z, vx, vy, vz] | 80–100% | Strong signal |
| Joint angles (BVH) | ~80% | Recommended |
| Physics simulation (ODE) | 80–100% | Recommended |
| Natural language tokens | <50% | σ degrades (graceful fallback) |
| Video pixels | <50% | σ degrades (graceful fallback) |

When data lacks light-cone structure, F3 reduces to standard attention via
σ → 0. There is no penalty for trying it.

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
    time_ratio: float = 0.25
    dropout: float = 0.1

config = Config()
attn = LorentzMultiHeadAttention(config)
norm = MinkowskiLayerNorm(config.d_model)

x = torch.randn(2, 16, config.d_model)
out, weights = attn(x)
out = norm(out)
print(f"σ = {attn.sigma:.3f}")  # learned light-cone strength
```

**Two parameters matter theoretically:**

| Parameter | Value | Meaning |
|---|---|---|
| `formula` | `'f3'` | Lorentzian inner product with learnable σ |
| `time_ratio` | `0.25` | Fraction of attention heads designated time-like |

All other hyperparameters are engineering choices.

**Critical companion:** the manifold loss (`(mq+1)²` or `project` equivalent).
F3 attention without this loss does not produce the geometric structure that
gives F3 its downstream advantage. Section 4 establishes why.

---

## Three attention formulas

| Formula | Score | Notes |
|---|---|---|
| `'f3'` (recommended) | `−σ·Qt Kt⊤ + Qs Ks⊤` | Learnable σ adapts to data |
| `'f1'` | `−Qt Kt⊤ + Qs Ks⊤` | Hard light cone, physics simulation |
| `'f2'` (deprecated) | `QK⊤/√d − 2α·Qt K⊤/√d` | Cross-terms allow optimizer to cancel light cone |

F2 fails because spacetime cross-terms let the optimizer route around the
geometric constraint (5/5 seeds α stuck at initialization). F3's clean
separation of time and space heads prevents this — analogous to why the
`signature` configuration in Section 4 fails while `mf` succeeds.

---

## Architecture overview

```
Physical trajectory
        │
        ▼
  Linear embed
        │
        ▼
[F3 Lorentzian attention]  ← -σ·Qt·Kt + Qs·Ks
[Minkowski LayerNorm]      ← normalize by |s²−t²|
[Manifold constraint]      ← project or (mq+1)²
        │
        ▼
   embed_seq
        │
   ┌────┴────┐
   ▼         ▼
Direction A  Direction B
embed →     lang_emb →
lang_gen    aligner →
            decoder
```

σ is a single learnable scalar (global light-cone width), corresponding to
isotropic linearization H* = I in the K=1 framework. Local σ(x) would
correspond to general H*(x) — a future extension.

---

## File structure

```
lorentz_transformer/                  # Stable API
├── core/
│   ├── attention.py                  # LorentzMultiHeadAttention (F1/F3)
│   └── layer_norm.py                 # MinkowskiLayerNorm

experiments/
├── law2_necessity_test.py            # Section 4: baseline vs signature vs mf
├── lorentz_riemannian_transformer.py # Section 5: M1–M4 milestones
├── layer1_minimal_test.py            # Section 6: Direction A bidirectional
├── baby_talk_full_test.py            # Section 6: full five-module pipeline
├── layer3_zero_loss_B.py             # Section 6: Direction B conservation
└── llcm_first_principles.py         # WIP: dt_info + dynamic metric (in progress)
```

---

## Reproducing the paper

```bash
# Section 4 — Law II necessity (the core ablation)
python experiments/law2_necessity_test.py

# Section 5 — Riemannian milestones M1–M4
python experiments/lorentz_riemannian_transformer.py

# Section 6 — Downstream gains
python experiments/layer1_minimal_test.py     # Direction A
python experiments/baby_talk_full_test.py     # Five-module pipeline
python experiments/layer3_zero_loss_B.py      # Direction B (10×)
```

Each script writes a pickle with raw per-seed results. The Section 4 script
additionally writes mq trajectories per epoch for the Figure 4 plot.

---

## Theoretical foundation

Two papers provide the mathematical basis:

**Realizability and the Origin of Causality** (Li, 2026) — Theorem 5 derives
Lorentzian signature from three conditions on a displacement cost function.
No particles, light signals, or kinematic objects are assumed. The minus sign
in the metric is the unique algebraic consequence of cost asymmetry between
temporal and spatial directions.

**K=1 Chronogeometrodynamics** (Li, 2026) — Theorem 4 proves d_c > 0 ⟺
det G < 0. Lorentzian signature is equivalent to having a nontrivial stability
boundary. Theorem 5 gives unique dynamics dx/dt = (J_G − D)∇V (Law II).
Theorem 6 gives the local restoring rate κ_K = 4d_c at critical damping.

LLCM's relationship to these theorems is now precise:

- Theorem 5 justifies the **sign** in F3 attention (the minus is derived, not chosen).
- Theorem 4 specifies what Law II would produce **if** Law II were the dynamics.
- Backprop is not Law II (Section 4).
- LLCM is the engineering substitute that makes the Theorem 4 geometry available
  under backprop dynamics (Section 5).
- The geometry has downstream value on physics-structured data (Section 6).

---

## Current status

| Status | Item |
|---|---|
| ✅ | Law II necessity demonstrated (cos = −0.25, 3 seeds, D=64) |
| ✅ | mf loss restores attractor (mq = −0.89, 89% timelike, acc ≥ 96.5%) |
| ✅ | Riemannian milestones M1–M4 (violation < 10⁻⁴, 267× separation) |
| ✅ | Direction A: Riemannian alignment +0.210 (p = 0.043, n = 6) |
| ✅ | Direction B: F3 conservation gain 18% (p = 0.048, n = 3) |
| 🔄 | Expand all downstream experiments to n ≥ 10 seeds |
| 📋 | Real robot data (Open X-Embodiment) |
| 📋 | First-principles dt_info implementation (in progress) |
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
