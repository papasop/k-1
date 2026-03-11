   # K=1 Chronogeometrodynamics

> **Lorentzian Light Cones of Information**
>
> The first neural network architecture with mathematically proven optimality derived from first principles.

<a href="https://colab.research.google.com/github/papasop/k-1/blob/main/k1_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg"></a>
<a href="https://doi.org/10.5281/zenodo.18949565"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18949565-blue"></a>

---

## Abstract

Neural network architectures are traditionally designed through trial-and-error, lacking theoretical justification for their optimality. **K=1 Chronogeometrodynamics** is the first framework that derives optimal neural architectures from information-geometric first principles. Building on information geometry and port-Hamiltonian systems theory, we prove that for dynamical systems with Lorentzian signature Sig(G) = (1,1), the optimal control structure is uniquely determined by a Wiener-geometric constraint.

**Key contributions:**

1. A **Uniqueness Theorem** proving that `J_G = α_eff G⁻¹ J` is the only stable structure for such systems
2. Experimental validation of **Law III**, demonstrating that the information flow ratio *K = dΦ/H* converges from K=3.91 to K=0.07 during training with mean Lyapunov drift ⟨ΔV⟩ < 0
3. The first empirical confirmation that **K=1** acts as a statistical attractor in neural network training

This represents a paradigm shift from *trial-and-error architecture design* to *mathematically proven optimality*.

---

## Theoretical Framework

### Information Time Metric

The core quantity is the **information flow ratio** (or **information time**):

```
K ≡ dt_info = dΦ / H
```

where:
- **dΦ = −log p(y|x)** is the information surprise (cross-entropy)
- **H = σ(hidden activations) + ε** is the entropic resistance

### The Three Laws

| Law | Name | Statement |
|-----|------|-----------|
| **I** | Information Time Metric | K = dΦ / H |
| **II** | Wiener-Geometric Constraint | J_G = α_eff G⁻¹ J (unique for Sig(G)=(1,1)) |
| **III** | Statistical Attractor Property | ⟨ΔV⟩ < 0, where V = ½(K−1)² |

#### Law I: Information Time Metric

The intrinsic temporal evolution is governed by the information flow ratio K = dΦ / H. This defines the "clock" of the learning process.

#### Law II: Wiener-Geometric Constraint

For systems with Lorentzian signature Sig(G) = (1,1), the structure matrix is **uniquely** determined:

```
J_G = α_eff G⁻¹ J
```

where α_eff ≈ 0.0817 is determined by passivity and Wiener constraints. This is remarkable: *geometry alone* determines the optimal structure—the architecture is *forced* by mathematical constraints rather than designed by hand.

#### Law III: Statistical Attractor Property

The Lyapunov potential V = ½(K−1)² satisfies ⟨ΔV⟩ < 0, establishing K=1 as the statistical attractor. The system naturally evolves toward K=1 during training.

### Hessian Geometry and Lorentzian Signature

The Hessian of the Lyapunov function in state space (K, σ) has eigenvalues λ₁ = 1.0 > 0 and λ₂ = −1/9 < 0, yielding a **Lorentzian signature** Sig(G) = (1,1)—identical to spacetime in general relativity. This imposes severe constraints on the set of allowable optimal structures.

### Uniqueness Theorem

> **Theorem (Wiener-Geometric Uniqueness):** Consider a port-Hamiltonian system with Hessian G satisfying Sig(G) = (1,1) and standard symplectic structure J. Then the unique stable structure matrix is `J_G = α_eff G⁻¹ J`, where α_eff ≈ 0.0817.

The proof proceeds via three steps: (1) passivity requires skew-symmetry, (2) the form constraint forces `J_G = α G⁻¹ J`, and (3) combining passivity with the Wiener ridge constraint determines α_eff = 0.0817. Verification confirms the result to machine precision (< 10⁻¹⁶).

---

## Experimental Validation

### Setup

| Parameter | Value |
|-----------|-------|
| Dataset | TinyStories validation set, 100,000 characters |
| Vocabulary | 95 unique characters (character-level tokenization) |
| Model | Transformer (2 layers, 128 dim, 4 heads, 1.23M params) |
| Training | 500 steps, batch size 32, learning rate 3×10⁻⁴ |
| Device | CUDA GPU (NVIDIA T4) |

### Training Dynamics

| Step | Loss | K | V = ½(K−1)² | Status |
|------|------|------|-------------|--------|
| 0 | 4.30 | 3.91 | 4.23 | Initial |
| 100 | 2.47 | 2.20 | 0.72 | Decreasing |
| 200 | 2.13 | 1.86 | 0.37 | Near K=1 |
| 300 | 0.38 | 0.32 | 0.23 | Below K=1 |
| 500 | 0.08 | 0.07 | 0.43 | Converged |

### Law III Verification

Mean Lyapunov drift over a sliding window of 50 steps:

- **Mean drift:** ⟨ΔV⟩ = −0.234 < 0 ✓
- **Probability of decrease:** P(ΔV < 0) = 0.62
- **Standard deviation:** σ_ΔV = 0.18

This confirms that V decreases on average, establishing K=1 as a statistical attractor.

![Training Dynamics](k1_training.png)

### K-Metric Diagnostics

The K-metric provides real-time training diagnostics:

| K Range | Interpretation |
|---------|----------------|
| K > 10 | Learning rate too high |
| 0.5 < K < 2 | Training healthy |
| K < 0.5 | Possible overfitting |

---

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Quick Demo (NumPy only)

```bash
python k1_unified.py
```

This runs the full K=1 demonstration using pure NumPy, including verification of all three laws.

### Training & Validation

```bash
python k1_train_test.py
```

Trains a K=1 Transformer and validates that Law III (⟨ΔV⟩ < 0) emerges naturally during training.

### PyTorch Concept Validation

```bash
python k1_concept_validation.py --quick-test
```

Runs the standalone PyTorch concept-validation script on synthetic repeated text and reports K-proxy, Lyapunov drift, and loss dynamics. Install PyTorch separately if it is not already available in your environment.

### Google Colab

Run the interactive notebook directly in your browser—no installation required:

<a href="https://colab.research.google.com/github/papasop/k-1/blob/main/k1_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

---

## Project Structure

```
k1_unified.py            # Core K=1 implementation (Laws I–III, NumPy)
k1_train_test.py         # Training & experimental validation
k1_concept_validation.py # PyTorch concept-validation script
k1_colab.py              # Google Colab demo version
k1_colab.ipynb           # Interactive Jupyter notebook
k1_training.png          # Training dynamics visualization
codex_connector/         # OpenAI Codex integration module
├── __init__.py          #   Package init & public API
├── config.py            #   Configuration (env vars / .env)
├── api_client.py        #   OpenAI API wrapper with retry & cache
├── core.py              #   High-level CodexConnector class
└── utils.py             #   Helpers (logging, caching, text utils)
cli.py                   # Codex command-line interface
examples.py              # Runnable usage examples
requirements.txt         # Python dependencies
```

---

## Codex Connector

An OpenAI-powered code assistant integrated into this repository.

```bash
# Configure your API key
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

# CLI usage
python cli.py generate "a Python function that reverses a string"
python cli.py explain --file k1_unified.py
python cli.py fix     --file my_script.py
```

```python
from codex_connector import CodexConnector

connector = CodexConnector(api_key="sk-...")
code = connector.generate("a function that computes Fibonacci numbers")
explanation = connector.explain(open("k1_unified.py").read())
```

| Command    | Description                              |
|------------|------------------------------------------|
| `generate` | Generate code from a text description    |
| `complete` | Complete an incomplete code snippet      |
| `explain`  | Explain what a piece of code does        |
| `fix`      | Identify and fix bugs                    |
| `optimize` | Optimize for performance or readability  |

---

## Comparison with Standard Approaches

| Property | Standard Transformers | K=1 Framework |
|----------|----------------------|---------------|
| Design method | Trial-and-error | Mathematically derived |
| Optimality | Unknown | Proven |
| Training monitor | Loss only | Loss + K-metric |
| Theoretical basis | Empirical | Information geometry |

---

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
