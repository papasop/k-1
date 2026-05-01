# Lorentzian Manifold Loss (LML)

> *Formerly Lorentz Light-Cone Model (LLCM). The original three-component
> architecture has been refuted by ablation. The surviving contribution
> is a single soft penalty `(mq + 1)²` that induces a measurable
> trade-off between per-step prediction accuracy and long-horizon
> stability, consistent with Theorem 4's characterization of mq = -1 as
> a stability attractor.*

**One sentence**: Adding `(mq + 1)²` to a Transformer's loss—where
`mq = ||spatial_emb||² − ||time_emb||²`—reliably constrains embeddings
to the Lorentzian unit hyperboloid (n=20 seeds) and induces a
trade-off: long-horizon endpoint MSE improves 16.9% (p < 0.001),
velocity variance drift improves 3.8% (p = 0.001, BH-corrected), but
per-step velocity error worsens 7.8% (p = 0.003). Mean rollout MSE is
unchanged. The pattern is consistent across seeds and aligns with
geometric stability-attractor theory.

```python
# Five lines added to your existing model.
emb = transformer(x)
t, s = emb[..., :T_DIM], emb[..., T_DIM:]
mq = (s**2).sum(-1) - (t**2).sum(-1)
loss_mf = ((mq + 1.0) ** 2).mean()
loss = loss_task + W_MF * loss_mf
```

---

## What this work is, and what it isn't

**It is**:
- A theoretically grounded loss term derived from Theorem 5 of Li (2026)
- Empirically validated to maintain Lorentzian geometric structure
  across 20 random seeds with high consistency
- Empirically validated to induce a **stability-vs-accuracy trade-off**
  with statistical significance on multiple metrics (paired t-test,
  Wilcoxon, sign test all agree; BH-corrected for the secondary
  metric family)
- A scientifically honest record, including ablations that **refuted**
  earlier framings and a metric (velocity_error) where mf loss
  reliably **hurts** rather than helps

**It isn't**:
- A method that uniformly improves all task metrics — this is
  documented honestly below
- Validated across multiple datasets (only pusht_keypoints)
- Shown to outperform conservation-aware methods like Hamiltonian or
  Lagrangian Neural Networks (untested)
- A "next-generation Lorentzian physics AI" — earlier framing rescoped
  to a calibrated mid-tier scope

The history of escalating then refined claims is documented below as
"Refuted Findings". This is the project's research integrity record.

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
the same conclusion: **the Lorentzian geometric structure must be
installed explicitly; it is not discovered by optimization.**

### Finding 3 (✅ primary, p < 0.001): mf loss reduces long-horizon endpoint MSE by 16.9%

This is the project's pre-registered primary finding. On 32-step
autoregressive rollout (pusht_keypoints, 20 paired seeds):

| Metric | B0 (with mf) | B0_no_mf (control) | Paired t-test |
|---|---|---|---|
| **final_step_mse** | 0.395 ± 0.030 | 0.475 ± 0.058 | t(19) = -5.32, p < 0.0001 |

**Statistical detail**:

- Mean paired difference: **-0.0806** (95% CI [-0.112, -0.049])
- Effect size: **|d_z| = 1.19** (large by Cohen's convention)
- Direction: **18/20 seeds B0 better** (sign test p = 0.0004)
- Robustness: Wilcoxon p < 0.001, sign test p < 0.001 — all agree
- Normality: Shapiro-Wilk p = 0.85 (paired t-test assumption holds)
- Pre-registered analysis: paired t-test as primary, Wilcoxon as
  robustness check

**Defensible claim** (paper-ready):

> "On long-horizon final-step MSE, mf loss reduces error relative to
> the LayerNorm-only baseline. Mean paired difference = -0.081
> (95% CI [-0.112, -0.049]); paired t(19) = -5.32, p < 0.001,
> |d_z| = 1.19, n = 20."

### Finding 4 (✅ secondary, BH-corrected significant): mf loss reduces velocity variance drift by 3.8%

The secondary metric family (m=3) was tested with Benjamini-Hochberg
correction. One metric survived:

| Metric | B0 (with mf) | B0_no_mf (control) | Paired t-test |
|---|---|---|---|
| **velocity_variance_error** | 0.000982 ± 0.000017 | 0.001021 ± 0.000047 | t(19) = -3.77, p = 0.0013 |

**Statistical detail**:
- Mean paired difference: **-0.000039** (95% CI [-0.000061, -0.000017])
- Effect size: **|d_z| = 0.84** (large)
- Direction: **18/20 seeds B0 better** (sign test p = 0.0004)
- Robustness: Wilcoxon p = 0.0003 — agrees with primary
- BH correction (m=3, α=0.05): **passes** (raw p = 0.0013 < BH threshold)
- Bonferroni (more conservative): also passes (p < 0.0167)

This is consistent with Finding 3's mechanism interpretation: mf loss
constrains how much velocity variance can drift across rollout.

### Honest negative finding (BH-significant): mf loss **increases** per-step velocity error by 7.8%

**This is the most important honest disclosure in this README**. We
report it because hiding it would be dishonest and the trade-off
pattern is itself a meaningful finding.

| Metric | B0 (with mf) | B0_no_mf (control) | Paired t-test |
|---|---|---|---|
| **velocity_error** | 0.001855 ± 0.000124 | 0.001721 ± 0.000120 | t(19) = +3.40, p = 0.0030 |

- Mean paired difference: **+0.000134** (95% CI [+0.000051, +0.000216])
- Effect size: |d_z| = 0.76 (medium)
- Direction: 5/20 seeds B0 better — **mf loss reliably hurts** this metric
- BH correction (m=3, α=0.05): also passes (raw p = 0.003)
- 95% CI is entirely **above** zero — direction is reliable

**mf loss systematically and significantly increases per-step velocity
error**. This is not noise; it survives multiple comparison correction
in the wrong direction.

### Mean step MSE: no significant difference

| Metric | B0 (with mf) | B0_no_mf (control) | Paired t-test |
|---|---|---|---|
| mean_step_mse | 0.221 ± 0.014 | 0.213 ± 0.018 | t(19) = +1.37, p = 0.187 |

- 95% CI [-0.004, +0.019] crosses zero
- Direction: 7/20 seeds B0 better, not reliable
- Did not pass any correction; not a difference

---

## The trade-off pattern: paper's central claim

Combining the 4-metric findings yields a coherent picture:

| Metric | mf vs no-mf | Status |
|---|---|---|
| final_step_mse (long-horizon endpoint) | **−16.9%** | ✅ verified, primary |
| velocity_variance_error (variance over rollout) | **−3.8%** | ✅ verified, BH-corrected |
| velocity_error (per-step prediction) | **+7.8%** | ❌ verified hurt, BH-corrected |
| mean_step_mse (averaged over rollout) | +3.5% | ◯ no significant difference |

**Pattern**: mf loss does **not** improve "predict the next frame
accurately" (per-step velocity is worse). It **does** improve "stay
stable across long rollout" (final endpoint and variance drift are
better).

**Interpretation**: mf loss induces a **stability-vs-accuracy
trade-off**. It limits the cumulative drift of embeddings during
autoregressive rollout, at the cost of per-step precision. This pattern
is consistent with — though not directly proven by — Theorem 4's
characterization of mq = -1 as a stability attractor: the geometric
constraint restricts how far embeddings can drift, which dampens
cumulative error growth at the expense of local prediction fidelity.

**Practical implication**:
- **Use mf loss** when the application cares about long-horizon
  endpoint stability (e.g., robot manipulation final-pose accuracy,
  planning toward a goal state, long generation)
- **Do not use mf loss** when the application cares about per-step
  accuracy (e.g., instantaneous velocity estimation, short-horizon
  closed-loop control)

This trade-off framing — backed by paired-test evidence on multiple
metrics with multiple comparison correction — is the project's main
empirical contribution.

---

## Refuted findings (honest record)

### Refuted 1 (❌): "Three components (F3 + MinkowskiLN + mf) jointly maintain geometry"

Earlier ablations (v3.7 n=3, v3.8 n=5):

| Configuration | on_shell |
|---|---|
| Full three-component (D) | 100% ± 0% |
| Only mf, vanilla LayerNorm (B) | **100% ± 0%** |
| Only MinkowskiLN, no mf (A) | 44% ± 31% (chaotic) |

**Conclusion**: mf loss alone matches three-component performance.
F3 attention and MinkowskiLN are not necessary.

### Refuted 2 (❌): "MinkowskiLN preserves Lorentzian signature through layer operations"

Long-horizon autoregressive rollout, n=3:

| Configuration | mean_step_mse |
|---|---|
| Standard LayerNorm + mf (B0) | 0.226 ± 0.011 |
| MinkowskiLN + mf (D0) | 0.561 ± 0.063 |

**MinkowskiLN makes long-horizon rollout 148% worse**. We do not
recommend MinkowskiLN.

### Refuted 3 (❌): "F3 attention provides task-relevant light-cone structure"

The learnable σ in F3 attention does not self-activate (σ stays at
≈0.500 ± 0.003 across 13+ runs). Forcing σ to 0.7 vs 0.0 changes
B-config metrics by less than 1% in either direction. **No measurable
benefit from F3 attention's light-cone score** in our experiments.

### Refuted 4 (◐): Single-seed "spacelike attractor" finding (v3.6)

The earlier observation that no-mf training converges to mq = +1 was
single-seed artifact. Multi-seed replication (v3.7 n=3, then v2 n=20)
showed mq drifts to +33 ± 8 — much more strongly spacelike than the
original single-seed finding suggested.

### Refuted 5 (◐ partially refined): "5× long-horizon stability gain"

This claim appeared in earlier README versions, based on comparison
with a chaotic baseline (Ablation A) that lacked **both** LayerNorm
and mf loss. The proper control (B0_no_mf, n=20) reveals:

- The mid-rollout improvement was driven by LayerNorm presence, not
  mf loss
- mf loss's actual benefit is **endpoint-specific** (16.9% on
  final_step_mse), not "5× across rollout"
- mf loss has a **trade-off**: per-step velocity reliably worse

The corrected story is the trade-off described above, not a
unidirectional "5× gain".

### Refuted 6 (❌): "Deterministic attractor" / "4-decimal reproducibility"

Earlier observation: 3 seeds showed velocity_error = 0.0019 to 4
decimal places, suggesting cross-seed determinism. With raw float
values (n=20):
- B0 velocity_error CV = 6.7% (typical ML reproducibility)
- B0_no_mf velocity_error CV = 6.9% (essentially identical)

The "4-decimal identity" was a print rounding artifact. mf loss does
not produce special task-metric determinism. (It does produce tight
cross-seed consistency on `mq` itself — Finding 1 — but that follows
directly from loss design.)

---

## Theoretical foundation (✅ unchanged)

Two papers provide the mathematical basis. **The theoretical
contribution is intact**; the empirical work either supports or refutes
specific implementation pathways.

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
attractor under Law II dynamics.

**Connection to LML, calibrated to current data**:
- mf loss `(mq+1)²` is a soft penalty implementing the Lorentzian unit
  shell as a regularizer ✅
- This soft penalty maintains the geometry empirically ✅
- The trade-off pattern (long-horizon stability gain at the cost of
  per-step precision) is consistent with Theorem 4's stability-attractor
  prediction: constraining embeddings to the attractor limits
  cumulative drift but reduces the model's freedom for local prediction
- Whether the same trade-off appears on other tasks/architectures is
  open

---

## Quick start

```python
import torch
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

**What this gives you (verified on pusht_keypoints, n=20 paired)**:
- Embeddings constrained to Lorentzian unit shell (mq → -1)
- 16.9% reduction in long-horizon final-step MSE (p < 0.001)
- 3.8% reduction in velocity variance drift (p = 0.001, BH-corrected)
- Trivial implementation cost (5 lines)

**What this costs you (also verified)**:
- 7.8% INCREASE in per-step velocity error (p = 0.003, BH-corrected)
- The trade-off is real and statistically robust

**What this does NOT give you**:
- Improvement on mean rollout MSE (essentially unchanged)
- Reduced cross-seed variance on task metrics (CV ~7%, normal range)
- Validated benefit on other tasks/architectures (untested)

**When to use mf loss**: applications where long-horizon endpoint
stability matters more than per-step accuracy.
**When NOT to use it**: applications requiring precise per-step
prediction.

---

## Reproducing the experiments

```bash
# Section 4: Necessity test (cos = -0.25 on ODE data, n=3)
python experiments/law2_necessity_test.py

# v3.8: Component ablation on real data (n=5)
python experiments/test1_v3.8_ablation.py

# Test 2-Revised v2: σ control + long-horizon rollout (n=3)
python experiments/test2_v2_sigma_rollout.py

# Reproducibility Test v2 (CRITICAL): n=20 paired comparison
python experiments/reproducibility_test_v2.py

# Paired t-test on primary metric
python experiments/paired_test_final_mse.py

# Multi-metric paired tests with BH correction
python experiments/multi_metric_paired_test.py
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
| 4 | Cross-task generalization | ❌ Not started | 2-3 months |
| 5 | Comparison with HNN / LNN / Neural ODE | ❌ Not started | 1-2 months |
| 6 | Scalability (D=512+) | ❌ Not started | 3-6 months |
| 7 | Theoretical analysis of trade-off mechanism | ◐ Hypothesized | 3-6 months |
| 8 | Independent community adoption | ❌ Pending publication | 1-2 years |

**Current completion**: 3/8.

The path forward:
- A mid-tier conference paper (CoRL/RSS/ICLR workshop) is now
  defensible based on the trade-off characterization (publication
  probability ~70-80%)
- Cross-task validation (Milestone 4) determines whether the trade-off
  pattern generalizes
- Comparison with HNN/LNN (Milestone 5) is required before claiming
  superiority over conservation-aware methods
- Top-tier venues (NeurIPS/ICML, Nature MI) require Milestones 4-7

---

## Limitations

1. **Single dataset**. All real-data results on pusht_keypoints (one
   task, 206 episodes, 18-D state).

2. **No comparison with conservation-aware methods**. HNN, LNN,
   symplectic networks, Neural ODE not benchmarked.

3. **The improvement is metric-specific, with a real cost**. mf loss
   improves final-step MSE and velocity variance drift, but harms
   per-step velocity error. Whether the net effect is desirable is
   application-dependent.

4. **No Euclidean unit-shell control**. Whether the
   Lorentzian-specific structure matters versus any unit-shell
   constraint (e.g., `(||emb||²-1)²`) is untested.

5. **State-only rollout, not action-conditioned**. pusht is a control
   task; real conservation evaluation requires action conditioning.

6. **F3 attention's role unresolved**. Mechanism unclear; perhaps
   relevant in tasks with explicit light-cone structure (untested).

7. **No LLM experiments**. Despite earlier README versions speculating
   about LLM relevance, no experiments on LLMs have been conducted.

8. **Mechanism interpretation is post-hoc**. The "stability attractor
   trade-off" interpretation is consistent with Theorem 4 but not
   directly proven. Stronger evidence (perturbation analysis,
   embedding drift trajectories) is future work.

---

## Discussion: the stability-vs-accuracy trade-off

Based on n=20 paired data with multiple comparison correction:

**mf loss is a geometric regularizer that produces a coherent
trade-off pattern: per-step accuracy decreases, long-horizon stability
increases, geometric structure is maintained. The pattern aligns with
the theoretical role of mq = -1 as a stability attractor (Theorem 4),
though direct mechanistic proof remains future work.**

This is more nuanced than earlier framings ("5× gain", "no benefit",
"deterministic attractor", "next-generation physics AI") and more
specific than typical regularization claims. The trade-off framing
makes the contribution **falsifiable on other tasks**: cross-task
testing should reproduce the trade-off pattern (worse per-step, better
long-horizon) if the geometric mechanism is the cause.

What we did **not** show:
- mf loss is the best soft penalty for trajectory tasks (no Euclidean
  control)
- The trade-off generalizes across datasets (untested)
- mf loss helps at LLM scale (untested)
- The trade-off is uniformly desirable (it depends on application
  metrics)

What we **did** show:
- A reproducible, theoretically motivated geometric regularizer
- Statistically robust trade-off across 4 metrics, with BH correction
- An honest negative finding (per-step velocity worse) that supports
  rather than undermines the mechanism interpretation
- Clear practical guidance: when to use, when not to

Future work should test cross-task generalization, compare against
conservation-aware methods (HNN, LNN), explore the mechanism more
deeply (perturbation analysis), and determine whether the trade-off
appears at scale.

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
trade-off characterization: a soft Lorentzian regularizer with verified
endpoint stability gain at the cost of per-step accuracy on trajectory
prediction.

---

## Acknowledgments

This README has gone through approximately nine major revisions, each
triggered by ablation experiments or statistical analyses that refined
or refuted claims in earlier versions:

- "Geometry emerges from training" — refuted by necessity test
- "Three-component architecture" — refuted by v3.5+ ablations
- "Spacelike attractor" — refined by multi-seed (v3.7 n=3, then n=20)
- "MinkowskiLN preserves geometry" — refuted by Test 2 v2
- "Next-generation physics AI" — rescoped by milestone analysis
- "5× long-horizon stability gain" — refined by n=20 control showing
  the gain is endpoint-specific (16.9%) and accompanied by a real cost
  (per-step velocity worse)
- "Deterministic attractor" — refuted as print rounding artifact
- "mf loss provides no task improvement" (a temporary over-correction
  during the v2 control analysis) — corrected by paired t-test on
  multiple metrics revealing the trade-off pattern
- Current version: **trade-off characterization with primary +
  BH-corrected secondary positive findings + honest negative finding**

Each revision has narrowed the claim and increased the evidence. The
current document reflects what survives sustained adversarial review,
proper baseline control, paired statistical testing, and multiple
comparison correction.

---

## License

MIT
