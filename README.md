# SCI Verification Framework

This repository contains numerical verifications and experimental protocols for the paper:

**“K = 1 as the Exponential Law of Structural Conservation (SCI):  
A Group-Theoretic, Bundle-Covariant Formulation with Information-Theoretic Certificates”
*Y.Y.N. Li,https://doi.org/10.5281/zenodo.17065624

---

## 🔍 Overview

The Structural Conservation Index (SCI) framework defines the **arrow of time** as a **geometric byproduct of symmetry breaking** in a \(\U(1)/\SO(2)\) bundle.  
This repository provides:

- **Colab verification scripts** of all core equations.  
- **Numerical experiments** linking SCI, holonomy, curvature, and metriplectic entropy production.  
- **Non-Abelian extensions** (\(\SU(2)/\SO(3)\)) with Wilson lines.  

---

## 📑 Verified Core Equations

The Colab notebook verifies the following:

1. **Dynamics** (eq:lin):  
   \(\dot v = (\sigma H + \omega J) v\)

2. **SCI definition** (eq:Kdef):  
   \(K(t) = \frac{\sigma + \omega \cot \theta}{\sigma - \omega \tan \theta}\)

3. **Phase-lock theorem**:  
   \(K=1 \iff \omega=0\)

4. **Near-conservation bridge** (eq:omegalin):  
   \(\omega \approx \frac{(K-1)\sigma}{\tan\theta+\cot\theta}\)

5. **Holonomy** (eq:holonomy):  
   \(\Theta = \int \omega\, dt\)

6. **Threshold criterion** (eq:threshold):  
   \(|\Theta| > \Omega_c \implies\) irreversible jump

7. **Entropy curvature certificate** (eq:curvcert):  
   \(H''(1) = -\pi^2/6\)

8. **Drift law** (eq:drift):  
   \(\frac{dK_{\max}}{d\varepsilon} = -P_{K\varepsilon}/P_{KK}\)

9. **Non-Abelian extension**:  
   - Directional SCI (\(\SU(2)\))  
   - Wilson line ordering

---


