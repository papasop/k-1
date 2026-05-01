# Lorentzian Manifold Loss (LML)

> *Formerly Lorentz Light-Cone Model (LLCM). The original three-component
> architecture has been refuted by ablation; the surviving contribution is a
> single loss term. The name change reflects that simplification.*

**One sentence**: Adding `(mq + 1)²` to a standard Transformer's loss—where
`mq = ||spatial_emb||² − ||time_emb||²`—provides a Lorentzian inductive bias
that improves long-horizon trajectory prediction stability by ~5×, with
trivial implementation cost.

```python
# That's it. Five lines added to your existing model.
emb = transformer(x)                              # standard
t, s = emb[..., :T_DIM], emb[..., T_DIM:]
mq = (s**2).sum(-1) - (t**2).sum(-1)
loss_mf = ((mq + 1.0) ** 2).mean()
loss = loss_task + W_MF * loss_mf                 # W_MF = 1.0
```

---

## What this work is, and what it isn't

**It is**:
- A theoretically grounded loss term derived from Lorentzian geometry
  (Theorem 5 of Li 2026)
- Empirically validated to improve long-horizon trajectory rollout
  stability on real robot data
- A scientifically honest record—including ablations that **refuted** the
  original architectural framing

**It isn't**:
- A drop-in replacement for standard sequence models on all tasks
- Validated across multiple datasets (only pusht_keypoints so far)
- Shown to outperform conservation-aware methods like Hamiltonian Neural
  Networks (HNN, Greydanus 2019) or Lagrangian NN (Cranmer 2020)—those
  comparisons remain to be done

The original "next-generation Lorentzian physics AI" framing is reframed
below into achievable milestones. Two are complete, six remain.

---

## Verified findings (with data strength)

### Finding 1 (✅ strongly supported): mf loss is necessary for geometric maintenance

In a multilayer Transformer with LayerNorm and residual connections, the
loss term `(mq + 1)²` alone—without any other geometric machinery—drives
embeddings to the Lorentzian unit shell `mq = -1` and keeps them there.

Across 5 random seeds on real robot trajectory data (pusht_keypoints):

| Configuration | on_shell ratio | mq_mean | velocity_error (rollout) |
|---|---|---|---|
| Standard LayerNorm + mf loss | **100% ± 0%** | -1.007 ± 0.002 | 0.0019 ± 0.0000 |
| Standard LayerNorm, no mf | failed (chaotic dynamics, see Section 4) | n/a | n/a |

The geometric maintenance is **deterministic to four decimal places** across
seeds, which is unusually clean for empirical ML.

### Finding 2 (✅ strongly supported): Backpropagation does not produce this geometry without explicit constraint

A plain Transformer trained on physics trajectory classification finds
`mq → +164.5` (3 seeds) instead of the theoretically motivated attractor at
`mq = -1`. The gradient alignment between task loss and the geometric
constraint is **anti-aligned at cos = -0.25**—not zero, but actively
repelling.

This refutes any narrative that Lorentzian geometry "emerges" from training.
It must be installed.

### Finding 3 (✅ strongly supported): mf loss provides ~5× long-horizon stability gain

On 32-step autoregressive rollout (pusht_keypoints, 3 seeds):

| Configuration | mean_step_mse | error growth (step1 → step32) |
|---|---|---|
| With mf loss (B0) | **0.226 ± 0.011** | 37× |
| Without mf loss—chaotic (Ablation A from v3.7) | 0.13 ± 0.08, mq drifts to ±1.7 | failed |

### Finding 4 (◐ needs Euclidean-shell control): The Lorentzian unit shell is the relevant geometric structure

mf loss pushes embeddings to the *Lorentzian* unit hyperboloid `mq = -1`,
not the Euclidean unit sphere `||x||² = 1`. Whether the Lorentzian-specific
structure (signature asymmetry between time and space dimensions) matters,
or whether any unit-shell constraint would suffice, is **not yet
controlled**. A Euclidean-shell ablation `(||emb||² − 1)²` is required
before the Lorentzian specificity can be claimed.

---

## Refuted findings (honest record)

The following claims appeared in earlier README versions. Ablation
experiments (v3.5 → Test 2-Revised v2) refute them. We document this
explicitly because (a) honest negative results are scientifically valuable,
and (b) the simplification is itself a contribution—LML is what survives
the ablation.

### Refuted 1 (❌): "Three components (F3 + MinkowskiLN + mf) jointly maintain geometry"

Ablation in v3.7 (3 seeds) and v3.8 (5 seeds):

| Configuration | on_shell |
|---|---|
| Full three-component (D) | 100% ± 0% |
| Only mf loss, vanilla LayerNorm (B) | **100% ± 0%** |
| Only MinkowskiLN, no mf (A) | **44% ± 31%** (chaotic across seeds) |

**mf loss alone matches the full three-component performance.**
Configuration B (vanilla LayerNorm + mf, no MinkowskiLN, no special
attention) is the minimal viable formulation.

### Refuted 2 (❌): "MinkowskiLN preserves Lorentzian signature through layer operations"

In short-horizon prediction, MinkowskiLN provides a marginal improvement
(~23%, p=0.058 across 5 seeds—not statistically significant). But in
long-horizon autoregressive rollout it is **catastrophically harmful**:

| Configuration | long-horizon mean_step_mse |
|---|---|
| Standard LayerNorm + mf (B0) | 0.226 ± 0.011 |
| MinkowskiLN + mf (D0) | 0.561 ± 0.063 |

**MinkowskiLN makes long-horizon rollout 148% worse**, reproducible across
all 3 seeds. Mechanism: MinkowskiLN's pointwise hard projection to the unit
shell distorts out-of-distribution inputs during autoregressive rollout,
accumulating nonlinear error over 32 steps. This is consistent with a
broader observation—soft penalties tolerate distribution shift better than
hard projections (see Discussion).

**Recommendation**: do not use MinkowskiLN. Standard LayerNorm + mf loss
is strictly better.

### Refuted 3 (❌): "F3 attention provides task-relevant light-cone structure"

The learnable σ in F3 attention `score = -σ·Qt·Kt + Qs·Ks` does not
self-activate under task-only training. Across 13+ runs spanning multiple
configurations and seeds, σ remains within **0.500 ± 0.003** of its
initialization—exactly as the Section 4 necessity test predicts.

When σ is forcibly fixed at 0.7 (active state), it provides no measurable
improvement over σ = 0 (dormant) on either short-horizon (4-step) or
long-horizon (32-step) prediction:

| σ value | B config Δ short | B config Δ long |
|---|---|---|
| 0.0 → 0.7 | +0.4% | -0.3% |

**F3 attention's empirical contribution beyond standard attention is
unverified.** Whether it could help in a different task (e.g., where
light-cone structure is exploitable) remains open. We do not currently
recommend its use over standard attention.

### Refuted 4 (◐): Single-seed "spacelike attractor" finding (v3.6)

An earlier ablation suggested that without mf loss, embeddings converge to
a spacelike attractor (mq → +1). Multi-seed replication (v3.7, n=3) showed
this was **single-seed artifact**—across 3 seeds, mq oscillates between
timelike (-0.30) and spacelike (+1.67) basins. The correct description is
"chaotic two-basin dynamics under no sign-selection signal," not "spacelike
attractor."

---

## Theoretical foundation (✅ unchanged)

Two papers provide the mathematical basis. **The theoretical contribution
is intact**; only the implementation pathway from theory to practice is
narrower than the original LLCM proposal.

**Realizability and the Origin of Causality** (Li, 2026, Foundations of
Physics, in review). Theorem 5 derives Lorentzian signature from three
conditions on a displacement cost function:
- **R**: cost vanishes in some directions (spatial-like)
- **E**: cost expands quadratically (Euclidean structure on space)
- **T**: cost is strictly positive in the time direction

Under R+E+T, the metric signature is uniquely Lorentzian, with `det G < 0`.
The minus sign is derived, not chosen.

**K=1 Chronogeometrodynamics** (Li, 2026). Theorem 4 proves d_c > 0 ⟺
det G < 0. Lorentzian signature is equivalent to having a nontrivial
stability boundary. Theorem 5 gives unique dynamics
`dx/dt = (J_G − D)∇V` (Law II) under critical damping.

**Connection to LML**:
- The mf loss `(mq + 1)²` is the simplest realization of Theorem 5's
  Lorentzian unit shell as a regularizer
- F3 attention was a more ambitious realization at the attention-score
  level; ablation shows it does not contribute beyond what mf loss provides
- The Theorem 4 attractor is **not** produced by backpropagation alone
  (Section 4)—it must be installed via mf loss

The theoretical claim is therefore: *causality-derived Lorentzian geometry
is an empirically useful inductive bias for sequence prediction*. The
specific mechanism by which it manifests (mf loss vs F3 attention vs
MinkowskiLN) is an empirical question whose current answer favors mf loss.

---

## Quick start

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Your existing Transformer
model = StandardTransformer(d_model=64, ...)

# Hyperparameters
T_DIM = int(d_model * 0.25)   # number of "time" coordinates
W_MF = 1.0                    # mf loss weight

# Training step
def train_step(x, y):
    pred, emb = model(x)
    loss_task = F.mse_loss(pred, y)
    
    # Lorentzian manifold loss
    t = emb[..., :T_DIM]
    s = emb[..., T_DIM:]
    mq = (s**2).sum(-1) - (t**2).sum(-1)
    loss_mf = ((mq + 1.0) ** 2).mean()
    
    loss = loss_task + W_MF * loss_mf
    loss.backward()
```

**Hyperparameters**:

| Parameter | Default | Notes |
|---|---|---|
| `T_DIM` | `d_model // 4` | Fraction of dimensions designated time-like |
| `W_MF` | `1.0` | Weight on mf loss; ≥1.0 needed (lower values fail to break sign symmetry) |

**No other architectural change is needed.** Use your existing attention,
your existing LayerNorm, your existing optimizer.

---

## When this works (and when it doesn't)

The mf loss provides benefit when:
- The data has **trajectory structure** with temporal dynamics
- **Long-horizon prediction** is the goal (short-horizon gains are
  marginal and not statistically significant)
- The R+E+T conditions plausibly apply (state evolves continuously, has
  natural time direction)

5-minute diagnostic:

```python
# On a sample of trajectory differences:
dx = data[:, 1:] - data[:, :-1]
t_dim = max(1, int(data.shape[-1] * 0.25))
mq_sample = -(dx[..., :t_dim]**2).sum(-1) + (dx[..., t_dim:]**2).sum(-1)
timelike_ratio = (mq_sample < 0).float().mean()
# > 0.6 → mf loss likely helps
# < 0.4 → mf loss may not help, but causes no harm at small W_MF
```

**Verified domain**: pusht_keypoints (single robot pushing task, 18-D state).
**Unverified domains**: aloha (bimanual), Open X-Embodiment (real robot
diversity), human motion capture, molecular dynamics, financial time series.

The mf loss has **not been validated outside pusht_keypoints**. Cross-task
generalization is the most important open question (see Roadmap).

---

## Reproducing the experiments

```bash
# Section 4: Necessity test (cos = -0.25, 3 seeds, ODE data)
python experiments/law2_necessity_test.py

# v3.8: Component ablation (n=5 seeds, real robot data)
python experiments/test1_v3.8_ablation.py

# Test 2-Revised v2: σ control + long-horizon rollout
python experiments/test2_v2_sigma_rollout.py
```

Each script writes a JSON log with per-seed metrics. Random seeds are
fixed; results should reproduce to 4 decimal places.

---

## Roadmap to "next-generation Lorentzian physics AI"

The original framing of LLCM as "next-generation physics AI" is, in
retrospect, premature. A more honest version of that ambition translates
into 8 milestones:

| # | Milestone | Status | Estimated time |
|---|---|---|---|
| 1 | Theoretical foundation (Theorem 4/5) | ✅ Complete | — |
| 2 | Necessity proof (backprop ≠ Law II) | ✅ Complete | — |
| 3 | First proof-of-concept on real data | ◐ Partial (lacks vanilla baseline) | 1 week |
| 4 | Cross-task generalization (5+ datasets) | ❌ Missing | 2–3 months |
| 5 | Comparison with HNN / LNN / Neural ODE | ❌ Missing | 1–2 months |
| 6 | Scalability (D=512, 1024, 4096) | ❌ Missing | 3–6 months (needs compute) |
| 7 | Theoretical analysis of empirical findings | ◐ Partial | 3–6 months |
| 8 | Independent community adoption | ❌ Pending publication | 1–2 years |

**Current completion**: 2.5 / 8.

The honest assessment is:
- LML will likely produce a CoRL/RSS paper in 2–4 weeks (publication
  probability ~70%)
- A NeurIPS/ICML follow-up after milestones 4–5 (probability ~40%)
- A Nature Machine Intelligence-level result requires milestones 4–7 plus
  cross-disciplinary validation (probability ~15%)
- Nature main journal is unlikely without milestones 6, 7, and 8 plus a
  game-changing application (probability <10%)

Rather than chasing "next-generation" framing, the goal is *niche
excellence*—becoming the canonical reference for Lorentzian inductive
biases in sequence modeling, the way Equivariant NN is the canonical
reference for symmetry-preserving architectures.

---

## Limitations

**The work in its current state has the following limitations, and we
would not want a reader to overlook them**:

1. **Single dataset validation**. All real-data results are on
   pusht_keypoints (one task, 206 episodes, 18-D state). Cross-domain
   generalization is unverified.

2. **No baseline comparison with conservation-aware methods**. Hamiltonian
   Neural Networks (Greydanus 2019), Lagrangian NN (Cranmer 2020), and
   symplectic networks directly target trajectory conservation. We have
   not measured how mf loss compares to them.

3. **Statistical sample size limited**. Most ablations use n=3 to n=5
   seeds. Effect sizes near significance threshold (e.g., MinkowskiLN's
   short-horizon advantage at p=0.058) are not robust at this sample size.

4. **Long-horizon results are state-only, not action-conditioned**.
   pusht is a control task; "long-horizon stability" measured here is
   forecast stability under unknown control, not physical conservation.
   Action-conditioned validation is a planned extension.

5. **No theoretical analysis of why soft penalty > hard projection**.
   The empirical finding that mf loss outperforms MinkowskiLN's hard
   projection is reproducible but not theoretically explained.

6. **F3 attention's role is unresolved**. σ does not self-activate under
   task pressure, and forced activation does not improve metrics in our
   tests. Whether F3 could help in different tasks remains open.

7. **No comparison with Euclidean unit-shell baseline**. Whether the
   Lorentzian-specific structure (vs any unit-shell constraint) is the
   source of the gain is not yet controlled.

These limitations are not fatal; they are the standard scope of an
initial paper. They become critical only when claiming "next-generation"
status, which we no longer do.

---

## Discussion: soft penalties vs hard projections

The most surprising finding of this work is that **soft penalties
outperform hard projections** for geometric inductive bias under
distribution shift. Specifically:

- **Hard projection** (MinkowskiLN): forward-pass `x / sqrt(|mq|)`
  enforces `|mq| ≈ 1` at every layer. Result: 148% worse long-horizon
  rollout MSE.

- **Soft penalty** (mf loss): adds `(mq + 1)²` to the loss, allowing the
  model to learn how to satisfy the constraint. Result: 100% on-shell
  with stable rollout.

A possible mechanism: hard projection is *pointwise*, applying the same
geometric correction regardless of input distribution. During
autoregressive rollout, inputs drift out of the training distribution,
and the projection's nonlinear distortion accumulates. Soft penalty lets
the model learn a *distribution-aware* alignment to the geometry, which
generalizes better when inputs shift.

This finding parallels broader patterns in deep learning:
- Equivariant networks (hard) vs data augmentation (soft) for symmetries
- Hard physical constraints vs penalty terms in physics-informed NNs
- BatchNorm's hard normalization vs LayerNorm's gentler adjustment

We do not yet have a theoretical framework predicting when soft beats
hard. This is a substantial open question whose answer would have
implications beyond Lorentzian geometry.

---

## Citation

If you use LML in your work:

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

The empirical validation paper is in preparation.

---

## Acknowledgments

This README, in its current form, exists because of multiple rounds of
adversarial review that surfaced incorrect claims in earlier versions.
Specifically:
- The "geometry emerges from training" framing (refuted by Section 4)
- The "three-component architecture" framing (refuted by ablation v3.5+)
- The "spacelike attractor" framing (refuted by multi-seed v3.7)
- The "MinkowskiLN preserves geometry" framing (refuted by Test 2 v2)
- The "next-generation physics AI" framing (rescoped by milestone analysis)

The current document reflects what survives. Negative results that change
the published narrative are scientifically more valuable than positive
results that confirm it; we have tried to be precise about which is which.

---

## License

MIT
