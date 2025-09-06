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
=== A. U(1) core verifications ===
[Theorem ω=0] mean(K)=1.000000, std(K)=2.57e-14
[eq:Kdef] mean(K-1)=4.00e-05, rel.err=2.08e-06
[eq:omegalin] approx=4.00e-05, rel.err=2.21e-05
[eq:holonomy] Theta(final)=9.00e-04
[eq:threshold] 未越阈

=== B. Information geometry ===
[eq:curvcert] H''(1)=-1.644934, target=-π²/6
[eq:drift] numeric=6.079276e-01, theory=6.079271e-01

=== C. Non-Abelian (SU(2)) ===
[Directional SCI] mean rel.err=1.01e-02
[Wilson line] Δφ=0.000e+00


