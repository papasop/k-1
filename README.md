# Lorentzian Manifold Loss (LML)

> *Formerly Lorentz Light-Cone Model (LLCM). The original three-component
> architecture has been refuted by ablation; subsequent control experiments
> further refuted the task-level improvement claim. The surviving
> contribution is geometric maintenance, not task improvement.*

**One sentence**: Adding `(mq + 1)²` to a Transformer's loss—where
`mq = ||spatial_emb||² − ||time_emb||²`—reliably constrains embeddings to
the Lorentzian unit hyperboloid `mq = -1`, but does **not** improve task
metrics over a standard LayerNorm baseline on pusht_keypoints. This is a
honest record of what the soft penalty does and does not do.

```python
# Five lines. Maintains Lorentzian geometry. Does not improve task MSE.
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
- Empirically validated to maintain Lorentzian geometric structure across
  20 random seeds with high reproducibility
- A scientifically honest record of multiple ablation rounds, including
  the **most recent control experiment that refuted the task-improvement
  claim from earlier README versions**

**It isn't**:
- A method that improves trajectory prediction task metrics (the v2
  control experiment showed mf loss does not significantly improve mean
  step MSE or velocity error over a LayerNorm-only baseline)
- Validated across multiple datasets (only pusht_keypoints)
- Shown to outperform conservation-aware methods like Hamiltonian or
  Lagrangian Neural Networks
- A "next-generation Lorentzian physics AI" — earlier framing rescoped
  to honest mid-tier scope

The history of escalating then refuted claims is documented below as
"Refuted Findings". This is the project's research integrity record.

---

## Verified findings (with v2 control data)

### Finding 1 (✅ strongly supported, n=20): mf loss reliably constrains the Lorentzian invariant

Across 20 random seeds on pusht_keypoints, mf loss drives `mq` to its
target value with high consistency:

| Configuration | mq mean | mq std | on-shell ratio |
|---|---|---|---|
| With mf loss (B0) | **-1.006** | 0.004 | mean 0.9991 (min 0.9977) |
| Without mf loss (B0_no_mf) | **+33.05** | 8.3 | not measured (clearly off-shell) |

**Geometric displacement effect: 30+ units of mq separation between
configurations**, robust across all 20 seeds in each group. No seed in the
no-mf condition spontaneously approached the Lorentzian unit shell.

This finding is qualitatively confirmed by both groups behaving as
expected by the loss design—mf loss enforces what it is designed to
enforce. It is a sanity check for the geometric mechanism, not a
surprising emergent phenomenon.

### Finding 2 (✅ strongly supported, n=20): Backpropagation does not produce Lorentzian geometry without explicit constraint

The control condition (B0_no_mf, identical architecture, no mf loss term)
provides direct evidence that standard training does not find the
Lorentzian unit shell:

- 20/20 seeds drifted to spacelike configurations (mq mean +33.05)
- mq range across seeds: +20.5 to +47.1
- Zero seeds converged anywhere near the theoretical attractor at mq = -1

This empirically validates the earlier necessity test (Section 4 of
prior versions), where the gradient anti-alignment cos(∇loss, ∇mq) =
-0.25 was measured on synthetic ODE data with n=3 seeds. The new n=20
control on real robot data provides a much stronger version of the
same conclusion: **the Lorentzian geometric structure must be installed
explicitly; it is not discovered by optimization.**

---

## Refuted findings (honest record)

### Refuted 1 (❌): "Three components (F3 + MinkowskiLN + mf) jointly maintain geometry"

Earlier ablations (v3.7 n=3, v3.8 n=5):

| Configuration | on_shell |
|---|---|
| Full three-component (D) | 100% ± 0% |
| Only mf, vanilla LayerNorm (B) | **100% ± 0%** |
| Only MinkowskiLN, no mf (A) | 44% ± 31% (chaotic) |

**Conclusion**: mf loss alone matches three-component performance for
geometric maintenance. F3 attention and MinkowskiLN are not necessary.

### Refuted 2 (❌): "MinkowskiLN preserves Lorentzian signature through layer operations"

Long-horizon autoregressive rollout, n=3:

| Configuration | mean_step_mse |
|---|---|
| Standard LayerNorm + mf (B0) | 0.226 ± 0.011 |
| MinkowskiLN + mf (D0) | 0.561 ± 0.063 |

**MinkowskiLN makes long-horizon rollout 148% worse**. Hard pointwise
projection accumulates nonlinear distortion under distribution shift
during rollout. We do not recommend MinkowskiLN.

### Refuted 3 (❌): "F3 attention provides task-relevant light-cone structure"

The learnable σ in F3 attention does not self-activate (σ stays at
≈0.500 ± 0.003 across 13+ runs). Forcing σ to 0.7 vs 0.0:

| σ change | B-config Δ short | B-config Δ long |
|---|---|---|
| 0.0 → 0.7 | +0.4% | -0.3% |

**No measurable benefit from F3 attention's light-cone score**.

### Refuted 4 (◐): Single-seed "spacelike attractor" finding (v3.6)

Earlier observation that no-mf training converges to mq = +1 was
single-seed artifact. Multi-seed replication showed chaotic two-basin
dynamics, not a stable spacelike attractor. The current v2 data with
n=20 (mq = +33 ± 8) further refines the picture: **without mf, mq
drifts to highly spacelike values, not just slightly positive**.

### Refuted 5 (❌, **NEW from v2 control**): "mf loss provides ~5× long-horizon stability gain"

This claim appeared in earlier README versions, based on comparison with
a chaotic baseline (Ablation A) that lacked both LayerNorm and mf loss.
The proper control—**LayerNorm without mf loss (B0_no_mf)**—was added
in v2 reproducibility test (n=20):

| Metric | B0 (with mf) | B0_no_mf (control) | Last-frame baseline |
|---|---|---|---|
| velocity_error | 0.001855 ± 0.000124 | **0.001721** ± 0.000120 | 0.001749 |
| velocity_variance_error | 0.000982 ± 0.000017 | 0.001021 ± 0.000047 | 0.001126 |
| mean_step_mse | 0.221 ± 0.014 | **0.213** ± 0.018 | 0.317 |
| final_step_mse | **0.395** ± 0.030 | 0.476 ± 0.058 | 0.674 |

**Key conclusions from this control experiment**:

1. **mf loss does NOT improve velocity error or mean MSE over the
   LayerNorm-only baseline**. B0_no_mf is slightly better on these
   metrics (3-7%, within noise).

2. **mf loss DOES show modest improvement on final_step_mse** (~17%,
   B0 = 0.395 vs B0_no_mf = 0.476), with B0 also showing lower
   cross-seed std on this metric. Whether this 17% endpoint
   improvement is statistically significant requires a paired t-test
   (CV is ~7-12% on both groups, so significance is not guaranteed).

3. **The "5× long-horizon stability" claim was a baseline error**: the
   earlier Ablation A baseline lacked LayerNorm AND mf loss, so the
   gain attributed to mf was actually the gain from having LayerNorm at
   all. With LayerNorm present, mf loss adds geometric constraint but
   minimal task improvement.

4. **The reproducibility claim ("deterministic to four decimal places")
   was a print rounding artifact**. With raw float values shown:

   ```
   B0 velocity_error CV = 6.7%   (typical ML reproducibility)
   B0_no_mf velocity_error CV = 6.9% (essentially identical)
   ```

   No special determinism; mf loss does not reduce cross-seed variance
   on task metrics.

This refuted claim is the most significant correction in this README
version. **The contribution of mf loss is geometric, not task-level.**

---

## Theoretical foundation (✅ unchanged)

Two papers provide the mathematical basis. The theoretical contribution
is intact and is independent of any empirical claims about task
performance.

**Realizability and the Origin of Causality** (Li, 2026, Foundations of
Physics, in review). Theorem 5 derives Lorentzian signature from three
conditions on a displacement cost function:
- **R**: cost vanishes in some directions (spatial-like)
- **E**: cost expands quadratically (Euclidean structure on space)
- **T**: cost is strictly positive in the time direction

Under R+E+T, the metric signature is uniquely Lorentzian, with
`det G < 0`. The minus sign is derived from causality, not chosen.

**K=1 Chronogeometrodynamics** (Li, 2026). Theorem 4 establishes
d_c > 0 ⟺ det G < 0. Lorentzian signature is equivalent to having a
nontrivial stability boundary.

**Connection to LML, with v2 calibration**:
- mf loss `(mq+1)²` is a soft penalty implementing Theorem 5's
  Lorentzian unit shell as a regularizer ✅
- This soft penalty does maintain the geometry empirically ✅
- The geometry maintenance does **not** translate to task-metric
  improvement on pusht_keypoints ❌
- The theoretical framework remains valid; whether geometric inductive
  bias helps on **other tasks** (where R+E+T conditions apply more
  strongly) is open

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

**What this gives you**:
- Embeddings constrained to Lorentzian unit shell (mq → -1)
- 20-seed reproducibility on geometric maintenance
- Trivial implementation cost (5 lines)

**What this does NOT give you**:
- Improved task metrics over standard LayerNorm baseline (on
  pusht_keypoints; other domains untested)
- Reduced cross-seed variance on task metrics (CV ~7%, normal ML range)
- Generalization to LLM scale or other architectures (untested)

**When to use mf loss**: when downstream applications benefit from
embeddings living on a Lorentzian manifold (e.g., for geometric
reasoning, future research). When you only care about task MSE on
trajectory prediction, mf loss provides no measurable advantage.

---

## Reproducing the experiments

```bash
# Section 4: Necessity test (cos = -0.25 on ODE data, n=3 seeds)
python experiments/law2_necessity_test.py

# v3.8: Component ablation on real data (n=5 seeds)
python experiments/test1_v3.8_ablation.py

# Test 2-Revised v2: σ control + long-horizon rollout
python experiments/test2_v2_sigma_rollout.py

# Reproducibility Test v2 (CRITICAL): 20 seeds × B0 + 20 seeds × B0_no_mf
# This is the experiment that refuted the "5× stability gain" claim.
python experiments/reproducibility_test_v2.py
```

All experiments use deterministic CUDA flags, fixed seeds, and explicit
DataLoader generator binding. Logs are saved as JSON.

---

## Roadmap (revised after v2 control)

The original 8-milestone roadmap remains, but the assessment of progress
is now more conservative:

| # | Milestone | Status (revised) | Estimated time |
|---|---|---|---|
| 1 | Theoretical foundation (Theorem 4/5) | ✅ Complete (Li 2026) | — |
| 2 | Necessity proof (backprop ≠ Law II) | ✅ Strengthened by v2 (n=20) | — |
| 3 | First proof-of-concept on real data | **◐ Partially refuted**: geometric maintenance works, task improvement does not | — |
| 4 | Cross-task generalization | ❌ Not started | 2-3 months |
| 5 | Comparison with HNN / LNN / Neural ODE | ❌ Not started | 1-2 months |
| 6 | Scalability (D=512+) | ❌ Not started | 3-6 months |
| 7 | Theoretical analysis of empirical findings | ◐ Partial | 3-6 months |
| 8 | Independent community adoption | ❌ Pending publication | 1-2 years |

**Honest current completion**: 1.5 / 8 (down from 2.5 due to v2 refuting
Milestone 3's task-improvement aspect).

The path forward is narrower than originally framed:
- A paper documenting the **negative findings on geometric inductive bias**
  is publishable in a workshop or negative-results venue (probability
  ~50-60%)
- A traditional positive-result paper requires finding a task where mf
  loss does provide measurable improvement (cross-task validation, M4)
- Nature/Nature MI level claims are not supported by current data

---

## Limitations

**The work has the following limitations, all of which should be noted**:

1. **Single dataset**. All real-data results on pusht_keypoints (one
   task, 206 episodes, 18-D state). Cross-domain generalization untested.

2. **No comparison with conservation-aware methods**. HNN, LNN,
   symplectic networks, Neural ODE not benchmarked.

3. **Task improvement claim refuted**. The v2 control (n=20 with proper
   LayerNorm-only baseline) shows mf loss does not improve mean step MSE
   or velocity error. The earlier "5× stability gain" was due to an
   incorrect baseline lacking LayerNorm.

4. **Possible modest endpoint effect**. mf loss may improve final_step_mse
   by ~17% (B0=0.395 vs B0_no_mf=0.476) and reduce variance on this
   metric. Statistical significance requires paired t-test (not yet run).

5. **No Euclidean unit-shell control**. Whether the Lorentzian-specific
   structure matters versus any unit-shell constraint is untested.
   `(||emb||²−1)²` ablation is needed.

6. **State-only rollout, not action-conditioned**. pusht is a control
   task; real conservation evaluation requires action conditioning.

7. **F3 attention's role unresolved**. Mechanism unclear; perhaps
   relevant in tasks with explicit light-cone structure (untested).

8. **No LLM experiments**. Despite earlier escalation in framing, no
   experiments on LLMs have been conducted. All claims about LLM
   relevance are hypothetical.

---

## Discussion: what does mf loss actually do?

Based on v2 control data, the honest interpretation is:

**mf loss is a geometric regularizer that maintains a specific manifold
structure (Lorentzian unit shell at mq=-1) without significantly
affecting task performance, on the pusht_keypoints trajectory
prediction task.**

This is more modest than earlier framings ("solves long-horizon stability",
"deterministic attractor", "next-generation physics AI"). But it is
defensible by the data.

Possible value of geometric maintenance without task gain:
- Downstream applications that reason on the embedding manifold
  (clustering, retrieval, geometric loss in subsequent tasks)
- Composition with other losses where Lorentzian structure interacts
  productively (e.g., pretraining for tasks with explicit time-space
  asymmetry)
- Theoretical interest as an empirical instance of Theorem 5

What we did **not** show, and what we earlier claimed in error:
- That this geometric maintenance provides direct task benefit on
  trajectory prediction
- That mf loss reduces cross-seed variance ("deterministic attractor")
- That it constitutes a major innovation in trajectory forecasting

Future work should test whether the geometric maintenance has value on
tasks where time-space asymmetry is more strongly required (causality
benchmarks, physical simulation with explicit conservation laws,
hierarchical embeddings).

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

The empirical paper, if completed, will be a "negative findings" /
"calibrated mid-tier" contribution rather than a "next-generation"
claim, reflecting what the data actually supports.

---

## Acknowledgments

This README has gone through approximately seven major revisions, each
triggered by ablation experiments that refuted claims in earlier
versions. Specifically:

- "Geometry emerges from training" — refuted by Section 4 necessity test
- "Three-component architecture" — refuted by v3.5+ ablations showing
  mf loss alone is sufficient for geometric maintenance
- "Spacelike attractor" — refuted by v3.7 multi-seed showing chaotic
  dynamics rather than stable attractor
- "MinkowskiLN preserves geometry" — refuted by Test 2 v2 showing 148%
  worse long-horizon rollout
- "Next-generation physics AI" — rescoped by milestone analysis
- "5× long-horizon stability gain" — **refuted by v2 reproducibility
  control (n=20)** showing mf loss does not improve task metrics over
  LayerNorm-only baseline; the earlier comparison used an incorrect
  baseline that lacked LayerNorm
- "Deterministic attractor" / "4-decimal reproducibility" — refuted as
  print rounding artifact; raw float CV is ~6.7%, normal ML range

Each correction is a step toward calibrated claims. The current
document reflects what survives sustained adversarial review.

The author thanks the various adversarial reviewers (human and AI) who
identified each over-claim, and acknowledges that the painful process
of refuting one's own results in real time is rarer in ML than it
should be.

---

## License

MIT
