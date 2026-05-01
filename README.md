# Lorentzian Manifold Loss (LML)

> *Formerly Lorentz Light-Cone Model (LLCM). The original three-component
> architecture has been refuted by ablation. The surviving contribution
> is a single soft penalty `(mq + 1)²` whose Lorentzian-specific
> signature (time/space asymmetry) is empirically necessary—Euclidean
> unit-shell alternatives fail to reproduce the long-horizon stability
> gain.*

 Adding `(mq + 1)²` to a Transformer's loss—where
`mq = ||spatial_emb||² − ||time_emb||²`—reliably constrains embeddings
to the Lorentzian unit hyperboloid (n=20 seeds) and induces a stability
trade-off: long-horizon endpoint MSE improves 16.9% over no constraint
(p < 0.001) and 18.3% over a scale-matched Euclidean unit-shell
baseline (p < 0.0001, |d_z| = 1.29). Mid-rollout MSE is unchanged;
per-step velocity error worsens. Multiple comparison correction
applied.

```python
# Five lines added to your existing model.
emb = transformer(x)
t, s = emb[..., :T_DIM], emb[..., T_DIM:]
mq = (s**2).sum(-1) - (t**2).sum(-1)
loss_mf = ((mq + 1.0) ** 2).mean()
loss = loss_task + W_MF * loss_mf
```

---


## Verified findings (n=20 paired, with multiple comparison correction)

### Finding 1 (✅ strongly supported, n=20): mf loss reliably constrains the Lorentzian invariant

Across 20 random seeds on pusht_keypoints, mf loss drives `mq` to its
target value with high consistency:

| Configuration | mq mean | mq std | on-shell ratio |
|---|---|---|---|
| With mf loss (B0) | **-1.006** | 0.004 | mean 0.9991 (min 0.9977) |
| Without mf loss (B0_no_mf) | **+33.05** | 8.3 | far off-shell |

**Geometric displacement effect: 30+ units of mq separation between
configurations**, robust across all 20 seeds in each group.

### Finding 2 (✅ strongly supported, n=20): Backpropagation does not produce Lorentzian geometry without explicit constraint

The control condition (B0_no_mf, identical architecture, no mf loss
term) demonstrates:

- 20/20 seeds drifted to spacelike configurations (mq mean = +33.05)
- mq range across seeds: +20.5 to +47.1
- Zero seeds converged anywhere near the theoretical attractor mq = -1

This empirically validates the necessity test (gradient anti-alignment
cos(∇loss, ∇mq) = -0.25 on synthetic ODE data with n=3 seeds). The
n=20 control on real robot data provides a much stronger version of
the same conclusion.

### Finding 3 (✅ primary, p < 0.001): mf loss reduces long-horizon endpoint MSE by 16.9% vs no-constraint baseline

Pre-registered primary finding. On 32-step autoregressive rollout
(pusht_keypoints, 20 paired seeds):

| Metric | B0 (with mf) | B0_no_mf (control) | Paired t-test |
|---|---|---|---|
| **final_step_mse** | 0.395 ± 0.030 | 0.476 ± 0.058 | t(19) = -5.32, p < 0.0001 |

- Mean paired difference: **-0.0806** (95% CI [-0.112, -0.049])
- Effect size: **|d_z| = 1.19** (large)
- Direction: **18/20 seeds B0 better** (sign test p = 0.0004)
- Robustness: Wilcoxon p < 0.001 — agrees with primary
- Pre-registered analysis: paired t-test as primary, Wilcoxon
  secondary

### Finding 4 (✅ secondary, BH-corrected): mf loss reduces velocity variance drift by 3.8%

Within the secondary metric family (m=3) tested with Benjamini-Hochberg
correction:

| Metric | B0 (with mf) | B0_no_mf (control) | Paired t-test |
|---|---|---|---|
| **velocity_variance_error** | 0.000982 ± 0.000017 | 0.001021 ± 0.000047 | t(19) = -3.77, p = 0.0013 |

- Mean paired difference: **-0.000039** (95% CI [-0.000061, -0.000017])
- Effect size: **|d_z| = 0.84** (large)
- BH-corrected significant (raw p = 0.0013 < BH threshold)
- Bonferroni also passes

### Finding 5 (✅ NEW, n=20 paired): Lorentzian-specific signature is necessary; Euclidean unit-shell alternatives fail

To test whether the trade-off is Lorentzian-specific or whether any
unit-shell soft penalty produces it, we ran a 4-condition control
experiment (n=20 paired seeds each, total 80 runs):

| Condition | Penalty | init constraint | final_step_mse |
|---|---|---|---|
| **L** (Lorentzian) | `(mq + 1)²` | ~1281 | **0.395 ± 0.030** |
| **N** (no constraint) | none | 0 | 0.476 ± 0.058 |
| **E_dim** (D-normalized Euclidean) | `(\|\|emb\|\|²/D − 1)²` | ~0 | 0.483 ± 0.059 |
| **E_unit** (unit-shell Euclidean) | `(\|\|emb\|\|² − 1)²` | ~3969 | 0.591 ± 0.0001 |

**Key paired comparisons** (all 20 seeds, identical init/data):

| Comparison | mean diff | 95% CI | p-value | |d_z| |
|---|---|---|---|---|
| **L vs E_dim** (PRIMARY) | -0.088 | [-0.120, -0.056] | < 0.0001 | 1.29 |
| L vs E_unit (sanity) | -0.196 | [-0.210, -0.181] | < 0.0001 | 6.42 |
| L vs N (reproduces v2) | -0.081 | [-0.112, -0.049] | < 0.0001 | 1.19 |
| **E_dim vs N** | **+0.008** | [+0.004, +0.012] | 0.0006 | 0.92 |
| E_unit vs N | +0.115 | [+0.088, +0.142] | < 0.0001 | 1.98 |

**What this rules out**:

- ❌ **"Any unit-shell soft penalty produces the trade-off"** —
  refuted. E_dim is slightly *worse* than no constraint
  (p = 0.0006, +1.6%). E_unit catastrophically breaks training
  (+24% worse).
- ❌ **"Euclidean shell penalty is equivalent to Lorentzian"** —
  refuted. L is significantly better than both Euclidean variants
  on 19/20 and 20/20 seeds respectively.

**What this supports**:

- ✅ Lorentzian-signature asymmetry (time-like vs space-like
  dimensions) is empirically necessary for the trade-off pattern,
  not just shell-confinement.
- ✅ Theorem 5's causality-derived sign structure is consistent
  with the empirical specificity.

**Honest decomposition of the L vs E_dim effect**:

The total L − E_dim difference is -0.088 on final_step_mse. Decomposed:

- L − N = -0.081 (mf loss benefit, **already known from v2**)
- N − E_dim = -0.008 (Euclidean penalty slightly **hurts** vs no constraint)
- Sum: -0.089 (≈ -0.088, within rounding)

So roughly **91% of the L vs E_dim gap comes from "L is better than no
constraint" (v2's finding), and ~9% comes from "Euclidean shell is
slightly worse than no constraint"**. The Lorentzian-specific advantage
is real and statistically robust, but the dominant contribution to the
L > E_dim gap is the mf loss benefit itself, not Euclidean shell
explicitly hurting.

**Important scope of the claim**:

- We tested two Euclidean variants (`(||emb||²-1)²`, `(||emb||²/D-1)²`)
  and Lorentzian (`(mq+1)²`). Of these tested conditions, only
  Lorentzian provides the trade-off.
- We did NOT test all possible geometric shell penalties. There may
  exist other geometric inductive biases (hyperbolic, spherical with
  different scaling, etc.) that we did not evaluate.
- The defensible claim is "**of the geometric shell penalties tested,
  only the Lorentzian variant produces the trade-off**", not
  "Lorentzian is uniquely optimal among all possible inductive biases".

### Honest negative finding (BH-significant): mf loss **increases** per-step velocity error by 7.8%

| Metric | B0 (with mf) | B0_no_mf | Paired t-test |
|---|---|---|---|
| **velocity_error** | 0.001855 ± 0.000124 | 0.001721 ± 0.000120 | t(19) = +3.40, p = 0.0030 |

- Mean paired difference: **+0.000134** (95% CI [+0.000051, +0.000216])
- 95% CI entirely above 0 — **mf loss reliably hurts** this metric
- BH-corrected significant
- Direction: 5/20 seeds B0 better — consistently worse with mf

This is documented honestly. It is not noise; it is a real
trade-off cost.

### Mean step MSE: no significant difference

| Metric | B0 (with mf) | B0_no_mf | Paired t-test |
|---|---|---|---|
| mean_step_mse | 0.221 ± 0.014 | 0.213 ± 0.018 | t(19) = +1.37, p = 0.187 |

95% CI crosses zero. Not a difference.

---

## The trade-off pattern: paper's central claim

Combining the 4-metric findings yields a coherent picture:

| Metric | mf vs no-mf | Status |
|---|---|---|
| final_step_mse (long-horizon endpoint) | **−16.9%** | ✅ verified, primary |
| velocity_variance_error (rollout variance) | **−3.8%** | ✅ verified, BH-corrected |
| velocity_error (per-step prediction) | **+7.8%** | ❌ verified hurt, BH-corrected |
| mean_step_mse (averaged over rollout) | +3.5% | ◯ no significant difference |

Adding the Lorentzian-specificity dimension:

| Comparison | final_step_mse advantage |
|---|---|
| Lorentzian vs no constraint | **-16.9%** (-0.081) |
| Lorentzian vs scale-matched Euclidean (E_dim) | **-18.3%** (-0.088) |
| Lorentzian vs unit-shell Euclidean (E_unit) | **-33.1%** (-0.196) |

**Interpretation**: mf loss does **not** improve "predict the next
frame accurately" (per-step velocity is worse). It **does** improve
"stay stable across long rollout" (final endpoint and variance drift
are better). And the Lorentzian-signature asymmetry (mq with negative
sign on time dimensions) is empirically necessary—Euclidean unit-shell
alternatives, even when scale-matched, do not reproduce the gain.

**Mechanism interpretation (post-hoc, not directly proven)**: the
trade-off pattern is consistent with Theorem 4's characterization of
mq = -1 as a stability attractor. The geometric constraint restricts
how far embeddings can drift in autoregressive rollout, dampening
cumulative error growth at the cost of local prediction fidelity. The
Lorentzian-specific asymmetry between time and space dimensions
appears to be the relevant structural feature, distinct from generic
shell confinement.

**Practical implication**:
- **Use mf loss** when long-horizon endpoint stability matters more
  than per-step accuracy
- **Do not use mf loss** when per-step accuracy is the primary metric
- **Lorentzian, not just any shell penalty**: Euclidean alternatives
  do not provide the same benefit on this task

---

## Refuted findings (honest record)

### Refuted 1 (❌): "Three components (F3 + MinkowskiLN + mf) jointly maintain geometry"

mf loss alone matches three-component performance (n=5 ablation).
F3 attention and MinkowskiLN are not necessary.

### Refuted 2 (❌): "MinkowskiLN preserves Lorentzian signature through layer operations"

| Configuration | mean_step_mse |
|---|---|
| Standard LayerNorm + mf (B0) | 0.226 ± 0.011 |
| MinkowskiLN + mf (D0) | 0.561 ± 0.063 |

MinkowskiLN makes long-horizon rollout 148% worse. We do not
recommend MinkowskiLN.

### Refuted 3 (❌): "F3 attention provides task-relevant light-cone structure"

Learnable σ does not self-activate (stays at ≈0.500 ± 0.003 across 13+
runs). Forced σ activation provides no measurable benefit.

### Refuted 4 (◐): Single-seed "spacelike attractor" finding

Multi-seed replication showed mq drifts to +33 ± 8 — much more
strongly spacelike than the original single-seed finding suggested.

### Refuted 5 (◐ partially refined): "5× long-horizon stability gain"

The original "5×" claim was based on comparing against a chaotic
baseline lacking both LayerNorm and mf loss. The proper control
(B0_no_mf with LayerNorm, n=20) shows the actual benefit is
endpoint-specific (16.9% on final_step_mse), not "5× across rollout".

### Refuted 6 (❌): "Deterministic attractor" / "4-decimal reproducibility"

Print rounding artifact. Raw float CV is ~6.7%, normal ML range.

### Refuted 7 (❌, **NEW from Euclidean control**): "Any unit-shell soft penalty would produce the trade-off"

This was a plausible reviewer-style hypothesis: maybe the trade-off
comes from generic shell confinement, not Lorentzian signature
specifically. The Euclidean control experiment (n=20 paired) refutes
this:

- E_unit (`||emb||² = 1`): catastrophically breaks training
  (final_step_mse 0.591 vs Lorentzian's 0.395)
- E_dim (`||emb||²/D = 1`, scale-matched to Lorentzian's natural
  range): slightly **hurts** vs no constraint (+1.6%, p = 0.0006)
- Only Lorentzian among the tested shell penalties produces the
  trade-off

The hypothesis was tested fairly (E_dim has init constraint matched
to Lorentzian's at LayerNorm output) and rejected. The Lorentzian
signature (time-space asymmetry) appears to be the empirically
necessary structural feature.

---

## Theoretical foundation (✅ unchanged)

Two papers provide the mathematical basis. **The theoretical
contribution is intact**.

**Realizability and the Origin of Causality** (Li, 2026, Foundations
of Physics, in review). Theorem 5 derives Lorentzian signature from
three conditions on a displacement cost function:
- **R**: cost vanishes in some directions (spatial-like)
- **E**: cost expands quadratically (Euclidean structure on space)
- **T**: cost is strictly positive in the time direction

Under R+E+T, the metric signature is uniquely Lorentzian, with
`det G < 0`. The minus sign is derived from causality, not chosen.

**K=1 Chronogeometrodynamics** (Li, 2026). Theorem 4 establishes
d_c > 0 ⟺ det G < 0. Lorentzian signature is equivalent to having a
nontrivial stability boundary, with mq = -1 acting as the stability
attractor.

**Connection to LML, calibrated to current data**:
- mf loss `(mq+1)²` is a soft penalty implementing the Lorentzian
  unit shell as a regularizer ✅
- This soft penalty maintains the geometry empirically ✅
- The trade-off pattern is consistent with stability-attractor theory
- **Lorentzian signature is empirically necessary**: scale-matched
  Euclidean unit-shell alternatives do not reproduce the benefit
- Whether the same trade-off appears on other tasks/architectures is
  open

---

## Quick start

```python
import torch
import torch.nn.functional as F

model = StandardTransformer(d_model=64, ...)

T_DIM = int(d_model * 0.25)
W_MF = 1.0

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

**What this gives you (verified on pusht_keypoints, n=20 paired)**:
- Embeddings constrained to Lorentzian unit shell (mq → -1)
- 16.9% reduction in long-horizon final-step MSE vs no constraint
- 18.3% reduction vs scale-matched Euclidean shell baseline
- 3.8% reduction in velocity variance drift
- Trivial implementation cost (5 lines)

**What this costs you (also verified)**:
- 7.8% INCREASE in per-step velocity error
- The trade-off is real and statistically robust

**Why Lorentzian, not Euclidean shell**: the Euclidean control
experiment (n=20 paired) shows that scale-matched Euclidean unit-shell
penalties either break training (E_unit) or slightly hurt task
performance (E_dim). The Lorentzian signature is necessary.

**When to use mf loss**: applications where long-horizon endpoint
stability matters more than per-step accuracy.

---

## Reproducing the experiments

```bash
# Section 4: Necessity test (cos = -0.25 on ODE data, n=3)
python experiments/law2_necessity_test.py

# v3.8: Component ablation on real data (n=5)
python experiments/test1_v3.8_ablation.py

# Test 2-Revised v2: σ control + long-horizon rollout (n=3)
python experiments/test2_v2_sigma_rollout.py

# Reproducibility Test v2: n=20 paired comparison (Findings 1-4)
python experiments/reproducibility_test_v2.py

# Paired t-test on primary metric
python experiments/paired_test_final_mse.py

# Multi-metric paired tests with BH correction
python experiments/multi_metric_paired_test.py

# Euclidean shell control: n=20 × 4 conditions (Finding 5)
python experiments/euclidean_shell_control_v2.py
```

All experiments use deterministic CUDA flags, fixed seeds, explicit
DataLoader generator binding. Statistical tests follow pre-registered
analysis plans. Logs saved as JSON.

---

## Roadmap

| # | Milestone | Status | Estimated time |
|---|---|---|---|
| 1 | Theoretical foundation (Theorem 4/5) | ✅ Complete (Li 2026) | — |
| 2 | Necessity proof (backprop ≠ Law II) | ✅ Strengthened (n=20) | — |
| 3 | First proof-of-concept on real data | ✅ Complete with trade-off characterization | — |
| 3a | Lorentzian-specificity (Euclidean control) | ✅ Complete (n=20 × 4 conditions) | — |
| 4 | Cross-task generalization | ❌ Not started | 2-3 months |
| 5 | Comparison with HNN / LNN / Neural ODE | ❌ Not started | 1-2 months |
| 6 | Scalability (D=512+) | ❌ Not started | 3-6 months |
| 7 | Theoretical analysis of trade-off mechanism | ◐ Hypothesized | 3-6 months |
| 8 | Independent community adoption | ❌ Pending publication | 1-2 years |

**Current completion**: 3.5/8.



## Limitations

1. **Single dataset**. All real-data results on pusht_keypoints (one
   task, 206 episodes, 18-D state).

2. **No comparison with conservation-aware methods**. HNN, LNN,
   symplectic networks, Neural ODE not benchmarked.

3. **The improvement is metric-specific, with a real cost**. mf loss
   improves final-step MSE and velocity variance drift, but harms
   per-step velocity error.

4. **Limited Euclidean alternatives tested**. Two Euclidean unit-shell
   variants (E_unit, E_dim). Other geometric inductive biases
   (hyperbolic Poincaré ball, spherical with different scaling,
   non-shell penalties) not compared. The claim "of tested shell
   penalties, only Lorentzian works" does not generalize to
   "Lorentzian is the unique optimal inductive bias".

5. **L vs E_dim effect decomposition**: ~91% of the gap comes from
   "L > N" (v2's primary finding), ~9% from "E_dim is slightly worse
   than N". The Lorentzian-specific contribution is real but should
   not be conflated with the dominant L vs N effect.

6. **State-only rollout, not action-conditioned**. pusht is a control
   task; real conservation evaluation requires action conditioning.

7. **F3 attention's role unresolved**. Mechanism unclear; perhaps
   relevant in tasks with explicit light-cone structure (untested).

8. **No LLM experiments**. All claims about LLM applicability are
   hypothetical.

9. **Mechanism interpretation is post-hoc**. The "stability attractor
   trade-off" interpretation is consistent with Theorem 4 but not
   directly proven.

---

## Discussion: the stability-vs-accuracy trade-off, with Lorentzian specificity

Based on n=20 paired data, with multiple comparison correction, and
n=20 paired Euclidean control:

**mf loss is a Lorentzian-specific geometric regularizer that produces
a coherent trade-off pattern: per-step accuracy decreases, long-horizon
stability increases, geometric structure is maintained. The
Lorentzian-signature asymmetry between time and space dimensions is
empirically necessary—scale-matched Euclidean unit-shell alternatives
do not reproduce the benefit. The pattern aligns with the theoretical
role of mq = -1 as a stability attractor (Theorem 4), though direct
mechanistic proof remains future work.**

The trade-off framing makes the contribution **falsifiable on other
tasks**: cross-task testing should reproduce the trade-off pattern
(worse per-step, better long-horizon) if the geometric mechanism is
the cause; and Euclidean control on those tasks should also fail to
reproduce, if Lorentzian-specificity holds beyond pusht.

What we did **not** show:
- Lorentzian is the optimal geometric inductive bias overall
  (only proven against tested Euclidean alternatives)
- The trade-off generalizes across datasets (untested)
- mf loss helps at LLM scale (untested)
- The trade-off is uniformly desirable (depends on application
  metrics)

What we **did** show:
- A reproducible, theoretically motivated geometric regularizer
- Statistically robust trade-off across 4 metrics, with BH correction
- Lorentzian-specific value via fair Euclidean control (n=20 × 4)
- An honest negative finding (per-step velocity worse) that supports
  rather than undermines the mechanism interpretation
- Clear practical guidance: when to use, when not to
- Refutation of the "any shell penalty works" alternative explanation

Future work should test cross-task generalization, compare against
conservation-aware methods (HNN, LNN), explore the mechanism more
deeply (perturbation analysis, embedding drift trajectories), and
determine whether the trade-off appears at scale.

---

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

The empirical paper is in preparation, framed as a calibrated
trade-off characterization with Lorentzian-specificity: a soft
geometric regularizer with verified endpoint stability gain, real
per-step cost, and empirical specificity to the Lorentzian signature
versus tested Euclidean alternatives.

---

## Acknowledgments

This README has gone through approximately ten major revisions, each
triggered by ablation experiments or statistical analyses that refined
or refuted claims in earlier versions:

- "Geometry emerges from training" — refuted by necessity test
- "Three-component architecture" — refuted by v3.5+ ablations
- "Spacelike attractor" — refined by multi-seed (v3.7 n=3, then n=20)
- "MinkowskiLN preserves geometry" — refuted by Test 2 v2
- "Next-generation physics AI" — rescoped by milestone analysis
- "5× long-horizon stability gain" — refined to endpoint-specific
  16.9% gain via proper baseline (n=20)
- "Deterministic attractor" — refuted as print rounding artifact
- "mf loss provides no task improvement" (a temporary over-correction
  during v2 control analysis) — corrected by paired t-test on
  multiple metrics revealing the trade-off pattern
- "Any unit-shell soft penalty produces the trade-off" — refuted by
  Euclidean control (n=20 × 4 conditions): Lorentzian-specificity
  empirically supported
- Current version: **trade-off characterization with primary +
  BH-corrected secondary findings + Lorentzian-specificity evidence
  via Euclidean control + honest decomposition of L vs E_dim effect**

Each revision has narrowed the claim and increased the evidence. The
current document reflects what survives sustained adversarial review,
proper baseline control, paired statistical testing, multiple
comparison correction, and a 4-condition geometric specificity
experiment.

---

## License

MIT
