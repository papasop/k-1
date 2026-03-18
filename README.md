# Lorentz Transformer

> **Transformer parameter space is a pseudo-Riemannian (Lorentzian) manifold.**
> 
> 50–80% of W_Q parameters have negative diagonal Hessian of dt²_info — confirmed across 3 seeds, 2 tasks, 4 α values, 80 epochs.


---

## What is this

Standard Transformers assume their parameter space is flat Euclidean space. This repository shows it is not.

The K=1 information geometry field equation defines a metric on the attention weight manifold:

```
dt²_info = Σ_q K_q      K_q = Φ_q / H_q
```

The Hessian of this metric with respect to W_Q determines curvature. Experimental measurement via Hutchinson estimator (K=20) reveals the curvature is **indefinite** — roughly half the parameter directions are concave (timelike) and half convex (spacelike). This is the defining property of a **Lorentzian manifold**.

From this single geometric fact, three architectural components follow directly.

---

## Three Findings

### Finding 1: Pseudo-Riemannian Structure (Primary)

`G_ii = ∂²(dt²_info) / ∂W_Q[i]²`

| Task | α | Seeds | G_ii < 0 fraction |
|------|---|-------|-------------------|
| 1-hop | 0.25 | 3/3 | 51–58% |
| 1-hop | 0.5  | 3/3 | 59–70% |
| 1-hop | 1.0  | 3/3 | 60–80% |
| 2-hop | 0.25 | 2/2 | 54–73% |
| 2-hop | 1.0  | 1/1 | 58–80% |

`G_ii < 0` means dt²_info is concave in that parameter direction → timelike.  
`G_ii > 0` means convex → spacelike.  
Result is stable across all conditions.

### Finding 2: Lightcone Flip (New)

With Minkowski attention at α=1.0, the lightcone correctly identifies causal token pairs:

| Task | α | Real chain (timelike) | Noise (timelike) | Gap |
|------|---|-----------------------|------------------|-----|
| 1-hop | 0.0 | 0.289 | 0.631 | −0.342 ✗ |
| 1-hop | 0.5 | 0.430 | 0.551 | −0.122 ✗ |
| 1-hop | 1.0 | 0.688 | 0.416 | **+0.271 ✓** |
| 2-hop | 0.0 | 0.535 | 0.600 | −0.065 ✗ |
| 2-hop | 1.0 | 0.488 | 0.415 | **+0.073 ✓** |

Both tasks flip at α=1.0. The Minkowski inner product separates causal from non-causal connections.

### Finding 3: R-Law (Unified Training Dynamics)

Across every experiment — K-field, CGD, K=1 Lorentz — the correlation between baseline accuracy and injection delta is r ≈ −1.0. This is not specific to any injection mechanism. It is a fundamental property of Transformer training dynamics.

| Experiment | Condition | r(baseline, Δ) |
|------------|-----------|----------------|
| K-field (1-hop) | d=−1, α=0.5 | −0.997 |
| K=1 Lorentz (1-hop) | α=0.25 | −0.966 |
| K=1 Lorentz (1-hop) | α=0.5  | −0.999 |
| K=1 Lorentz (2-hop) | α=0.25 | −1.000 |

**Interpretation:** Lorentz Transformer benefits models in the learning regime (low baseline) and is neutral or harmful on saturated baselines. Optimal use case: pre-training from scratch or fine-tuning on genuinely novel tasks.

---

## Architecture

All three components share one object: the **timelike projection matrix P_t**.

```
G_diag  = Hutchinson estimate of ∂²(dt²_info)/∂W_Q²
P_t     = diag(G_diag < 0)        # 1 where timelike, 0 where spacelike
η       = I − 2α P_t              # Minkowski signature matrix
```

### Component 1: Minkowski Attention

```python
scores_L = Q η K^T / √d
         = QK^T/√d  −  2α · (Q P_t) K^T / √d
```

Standard Euclidean inner product → Minkowski inner product.  
α=0 recovers standard attention exactly.  
α=1.0 flips the lightcone (causal pairs become timelike).

### Component 2: Geodesic Adam

```python
g_t = P_t @ grad          # timelike gradient component
g_s = grad - g_t          # spacelike gradient component
grad_geodesic = scale_t * g_t + scale_s * g_s   # scale_t > 1 > scale_s
optimizer.step(grad_geodesic)
```

Timelike directions receive a larger step (information-rich, geodesic path).  
Spacelike directions receive a smaller step (knowledge-preserving).

### Component 3: Timelike Regularization

```python
R(θ) = λ_s * ‖(I − P_t) θ‖²   # penalize spacelike params
loss  = loss_task + R(θ)
```

Constrains updates to the timelike submanifold.  
Geometric alternative to EWC: no task boundary, no parameter snapshots.

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `α` | 0.25 | Lorentz strength (0 = standard attention) |
| `scale_t` | 1.5 | Timelike LR multiplier |
| `scale_s` | 0.5 | Spacelike LR multiplier |
| `λ_s` | 1e-4 | Spacelike regularization |
| `N` | 40 steps | P_t update frequency |
| `K` | 20 | Hutchinson samples |
| `warmup` | 50–100 | Steps before first P_t update |
| `EMA` | 0.3 | Fraction of new P_t per update |

---

## Computing P_t

| Method | HVP Cost | Detects G_ii < 0 | LLM Scale (d=4096) |
|--------|----------|------------------|---------------------|
| Hutchinson (K=20) | 20 HVPs | ✓ | ~336ms/layer — slow |
| Lanczos (k=10) | 10 HVPs | ✓ | ~1–5ms/layer ✓ |
| Random Lanczos (Nyström) | 2k HVPs | ✓ | ~5–20ms/layer ✓ |
| K-FAC | 0 extra | ✗ (S is PSD) | **Cannot detect timelike directions** |

> **Warning:** K-FAC approximates H ≈ A ⊗ S where A and S are covariance matrices (positive semi-definite). Their Kronecker product is also PSD — it can never have negative elements. K-FAC cannot detect timelike directions. Use Lanczos instead.

HVP implementation:
```python
g1 = autograd.grad(loss_fn(), W_Q, create_graph=True)[0]
Hv = autograd.grad((g1 * v).sum(), W_Q)[0]   # Hessian-vector product
```

---

## Files

| File | Description |
|------|-------------|
| `experiments/k1_lorentz.py` | Main experiment: Hutchinson Hessian + Minkowski attention (3 steps) |
| `experiments/geodesic_adam.py` | Geodesic Adam optimizer + TimeLikeManager |
| `experiments/lanczos_lorentz.py` | Lanczos vs Hutchinson; LLM scale feasibility |
| `experiments/gpt2_lorentz_test.py` | GPT-2 validation: λ_min, stability, depth monotonicity |
| `experiments/lorentz_inner.py` | Minkowski inner product attention with lightcone diagnostics |
| `experiments/lorentz_scores.py` | Position-B: Mahalanobis distance penalty on scores |
| `experiments/cgd_experiment.py` | CGD three-method experiment (Γ^info, PLLR, NULL subspace) |
| `experiments/dtfv2_experiment.py` | K-field routing experiment (v8, 3 seeds, val-set selection) |
| `experiments/kfac_feasibility.py` | K-FAC vs Lanczos feasibility (shows K-FAC fails) |
| `docs/lorentz_spec.docx` | Full architecture specification |

---

## Roadmap

- [x] Pseudo-Riemannian structure confirmed (50–80% timelike, r=−1.0, lightcone flip)
- [x] Three architecture components specified
- [x] Minkowski attention implemented and tested (`k1_lorentz.py`)
- [x] Geodesic Adam implemented (`geodesic_adam.py`)
- [ ] Geodesic Adam experimental validation (1-hop / 2-hop)
- [ ] Three components joint training (Lorentz Full)
- [ ] Scale to 256d / 6L
- [ ] GPT-2 (768d / 12L) — Random Lanczos, real language data
- [ ] Lorentz LLM from scratch at 1B+ scale
- [ ] Evaluate on MuSiQue / HotpotQA (multi-hop reasoning)

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/lorentz-transformer
cd lorentz-transformer
pip install torch numpy scipy
```

```bash
# Main experiment: confirms pseudo-Riemannian structure + lightcone flip
python experiments/k1_lorentz.py

# Geodesic Adam: timelike/spacelike gradient decomposition
python experiments/geodesic_adam.py

# Lanczos feasibility at LLM scale
python experiments/lanczos_lorentz.py
```

Expected output from `k1_lorentz.py`:
```
layer 0: 负=9489(57.92%) 正=6895(42.08%)  类时参数=0.579
layer 1: 负=9194(56.12%) 正=7190(43.88%)  类时参数=0.561
...
光锥 [K1-Lorentz(α=1.0)]: 真实链类时=0.688  噪声类时=0.416  差=+0.271  ✓因果
```

---

## Theory

The K=1 field equation defines information spacetime geometry for attention mechanisms.
Key objects:

```
H_q   = −Σ_j a_qj log(a_qj)          Shannon entropy at query q
Φ_q   = Σ_j a_qj²                     Attention concentration
K_q   = Φ_q / H_q                     K-field (information density)

dt²_info = Σ_q K_q                    Information time metric

G_ij  = ∂²(dt²_info) / ∂θ_i ∂θ_j    Curvature tensor (Hessian)
G_ii < 0  →  timelike  (concave)
G_ii > 0  →  spacelike (convex)

η = I − 2α P_t                        Minkowski metric matrix
scores_L = Q η K^T / √d              Lorentzian attention
```

Derivation of the Minkowski attention formula:
```
scores_L = Q(I − 2αP_t)K^T / √d
         = QK^T/√d − 2α(QP_t)K^T/√d
         = scores_std − 2α × (timelike inner product)
```

---

## Citation

```bibtex
@misc{lorentz-transformer,
  title  = {Lorentz Transformer: Pseudo-Riemannian Architecture from K=1 Field Equations},
  author = {},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/lorentz-transformer}
}
```

---

## License

MIT

## Citation

```bibtex
@article{li2026k1,
  author  = {Li, Y. Y. N.},
  title   = {K=1 Chronogeometrodynamics: Lorentzian Geometry from Information Time},
  year    = {2026},
  doi     = {10.5281/zenodo.18949565}
}
```

## License

[MIT](LICENSE)
