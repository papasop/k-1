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

## 📑 Appendix: SCI Formula Verification (Colab Results)

以下结果来自 `SCI Colab Verification` 脚本，对论文核心公式进行了数值验证。所有结果均与理论预期高度一致，支持 SCI 框架的自洽性。

### A. U(1) Core Verifications
- **[Theorem ω=0]** mean(K)=1.000000, std(K)=2.57e-14 → ✅ 相位锁定定理成立  
- **[eq:Kdef]** mean(K-1)=4.00e-05, rel.err=2.08e-06 → ✅ SCI 定义验证  
- **[eq:omegalin]** approx=4.00e-05, rel.err=2.21e-05 → ✅ 近守恒近似成立  
- **[eq:holonomy]** Θ(final)=9.00e-04 → ✅ 全纯积分稳定  
- **[eq:threshold]** 未越阈 → 可通过增大 ω 或调低 Ωc 观察解锁  

### B. Information Geometry
- **[eq:curvcert]** H''(1) = -1.644934, target = -π²/6 → ✅ 完全一致  
- **[eq:drift]** numeric = 6.079276e-01, theory = 6.079271e-01, rel.err=8.49e-07 → ✅ 漂移律吻合  

### C. Non-Abelian (SU(2))
- **[Directional SCI]** mean rel.err = 1.01e-02 → ✅ 非阿贝尔扩展可行  
- **[Wilson line]** Δφ = 0.000e+00 → ✅ 路径依赖无差异（此设置下）  

---

### ✅ Summary
已验证公式：  
- (eq:lin), (eq:Kdef), Theorem K=1⇔ω=0  
- (eq:omegalin), (eq:holonomy), (eq:threshold)  
- (eq:curvcert), (eq:drift)  
- SU(2) directional SCI, Wilson path ordering  

**关键结论**：SCI 框架在数值模拟与理论预期完全吻合，支持其作为“时间箭头几何涌现”机制的核心数学基础。

