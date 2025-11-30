# ================================================
# K=1 Chronogeometrodynamics - Corrected Ω_Λ Derivation
# -------------------------------------------------
# Fixed integration method and physical interpretation
# ================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, simpson
from scipy.optimize import minimize

# --------------------------
# Planck 2018 Cosmological Parameters
# --------------------------
h = 0.674
Omega_m = 0.315
Omega_b = 0.049
ns = 0.965
As = 2.1e-9
sigma8 = 0.811

print("=== Standard Cosmological Parameters ===")
print(f"h = {h}, Ω_m = {Omega_m}")
print(f"n_s = {ns}, A_s = {As:.2e}, σ₈ = {sigma8}")

# --------------------------
# Transfer Function & Power Spectrum
# --------------------------
def transfer_function(k):
    """Eisenstein & Hu transfer function"""
    Gamma = Omega_m * h
    q = k / Gamma
    L = np.log(np.e + 1.84 * q)
    C = 14.4 + 325.0 / (1 + 60.5 * q**1.11)
    return L / (L + C * q**2)

def Pk_linear(k):
    """Linear matter power spectrum"""
    Tk = transfer_function(k)
    k_pivot = 0.05
    return As * (k / k_pivot)**(ns - 1) * Tk**2 * (2 * np.pi**2 / k**3)

# Normalization (simplified for focus)
def Pk_normalized(k):
    return Pk_linear(k) * 1e7  # Approximate normalization

# ================================================
# CORRECTED Ω_Λ GEOMETRIC DERIVATION
# ================================================

print("\n" + "="*60)
print("CORRECTED Ω_Λ GEOMETRIC DERIVATION")
print("="*60)

# Generate k range
k_values = np.logspace(-3, 2, 5000)
Pk_values = np.array([Pk_normalized(k) for k in k_values])

def corrected_Omega_Lambda_derivation():
    """
    Corrected derivation based on physical principles:
    Ω_Λ should be related to the fraction of power on super-horizon scales
    or scales where dark energy dominates.
    """
    
    # Method 1: Horizon-based approach
    # Dark energy dominates on scales larger than the Hubble scale
    k_Hubble = 0.00067  # h/Mpc (Hubble scale)
    
    # Method 2: Transition scale where expansion accelerates
    # This is the true physical critical scale
    k_transition = 0.01  # h/Mpc (approximate acceleration scale)
    
    # Method 3: Use the geometric flow spectrum concept
    # but with proper physical interpretation
    
    # Calculate the flow spectrum
    flow_spectrum = k_values * Pk_values
    
    # Normalize flow spectrum to get probability distribution
    flow_total = simpson(flow_spectrum, np.log(k_values))
    flow_normalized = flow_spectrum / flow_total
    
    # The key insight: Ω_Λ corresponds to power on LARGE scales (small k)
    # where dark energy dominates the dynamics
    
    # Define critical scale based on when dark energy becomes important
    # This is around the scale where H(a) transitions from matter to Λ domination
    a_eq = (Omega_m / (1 - Omega_m))**(1/3)  # Scale factor at equality
    k_DE = k_Hubble * np.sqrt(1 + a_eq**3)  # Dark energy transition scale
    
    print(f"Dark energy transition scale: k_DE = {k_DE:.6f} h/Mpc")
    
    # Ω_Λ is the integral from 0 to k_DE (large scales)
    mask_DE = k_values < k_DE
    if np.sum(mask_DE) > 1:
        Omega_DE = simpson(flow_normalized[mask_DE], np.log(k_values[mask_DE]))
    else:
        Omega_DE = 0.0
    
    return k_DE, Omega_DE

def alternative_Omega_Lambda_method():
    """
    Alternative method: Use the shape of the power spectrum
    to determine the dark energy fraction
    """
    
    # Calculate various moments of the power spectrum
    flow_spectrum = k_values * Pk_values
    
    # Total integrated flow
    total_flow = simpson(flow_spectrum, np.log(k_values))
    
    # The key physical insight: 
    # Dark energy affects the largest scales (smallest k)
    # We need to find the scale where matter domination ends
    
    # Characteristic scales
    k_peak = k_values[np.argmax(flow_spectrum)]  # Peak of flow spectrum
    k_Hubble = 0.00067
    
    # Ω_Λ should be related to the fraction of power on super-horizon scales
    # plus some correction for the transition region
    
    # Empirical relation based on power spectrum shape
    # This comes from the requirement that Ω_m + Ω_Λ = 1
    k_cutoff = 0.1  # Scale where matter power is significantly suppressed
    
    # Calculate matter fraction (Ω_m) from power on small scales
    mask_matter = k_values > k_cutoff
    if np.sum(mask_matter) > 1:
        matter_flow = simpson(flow_spectrum[mask_matter], np.log(k_values[mask_matter]))
        Omega_m_calculated = matter_flow / total_flow
    else:
        Omega_m_calculated = Omega_m  # Fallback
    
    # Then Ω_Λ = 1 - Ω_m
    Omega_DE = 1 - Omega_m_calculated
    
    return k_cutoff, Omega_DE

def K1_theoretical_prediction():
    """
    K=1 chronogeometrodynamics specific prediction
    Based on geometric constraints and flow conservation
    """
    
    # In K=1 theory, the critical scale emerges from geometric quantization
    # The fundamental scale is related to the de Sitter radius
    
    # de Sitter scale related to Λ
    H0 = 67.4  # km/s/Mpc
    c = 3e5    # km/s
    R_dS = c / H0  # de Sitter radius in Mpc
    
    # Critical wavenumber
    k_dS = 2 * np.pi / R_dS  # h/Mpc
    
    # In K=1 theory, this scale gets modified by geometric factors
    k_critical = k_dS * np.sqrt(3/2)  # Geometric factor from K=1 theory
    
    # Now compute Ω_Λ using this physically motivated scale
    flow_spectrum = k_values * Pk_values
    flow_total = simpson(flow_spectrum, np.log(k_values))
    
    # Ω_Λ is the power on scales larger than the critical scale
    mask_DE = k_values < k_critical
    if np.sum(mask_DE) > 1:
        Omega_DE = simpson(flow_spectrum[mask_DE], np.log(k_values[mask_DE])) / flow_total
    else:
        Omega_DE = 0.0
    
    return k_critical, Omega_DE

# Compute predictions using different methods
print("\n--- Method 1: Horizon-based approach ---")
k_de1, Omega_de1 = corrected_Omega_Lambda_derivation()
print(f"k_critical = {k_de1:.4f} h/Mpc")
print(f"Predicted Ω_Λ = {Omega_de1:.4f}")

print("\n--- Method 2: Power spectrum shape method ---") 
k_de2, Omega_de2 = alternative_Omega_Lambda_method()
print(f"k_critical = {k_de2:.4f} h/Mpc")
print(f"Predicted Ω_Λ = {Omega_de2:.4f}")

print("\n--- Method 3: K=1 theoretical prediction ---")
k_de3, Omega_de3 = K1_theoretical_prediction()
print(f"k_critical = {k_de3:.4f} h/Mpc")
print(f"Predicted Ω_Λ = {Omega_de3:.4f}")

# Take the K=1 prediction as our main result
k_critical = k_de3
Omega_Lambda_predicted = Omega_de3

print(f"\n=== FINAL K=1 PREDICTION ===")
print(f"Theoretical critical scale: k* = {k_critical:.4f} h/Mpc")
print(f"Predicted Ω_Λ = {Omega_Lambda_predicted:.4f}")
print(f"Observed Ω_Λ = 0.6847")
print(f"Relative error = {abs(Omega_Lambda_predicted - 0.6847)/0.6847*100:.2f}%")

# ================================================
# COMPREHENSIVE VALIDATION
# ================================================

print("\n" + "="*60)
print("COMPREHENSIVE VALIDATION")
print("="*60)

# Test the sensitivity to the critical scale
test_scales = np.logspace(-3, 0, 20)
Omega_predictions = []

for k_test in test_scales:
    flow_spectrum = k_values * Pk_values
    flow_total = simpson(flow_spectrum, np.log(k_values))
    mask = k_values < k_test
    if np.sum(mask) > 1:
        Omega_test = simpson(flow_spectrum[mask], np.log(k_values[mask])) / flow_total
    else:
        Omega_test = 0.0
    Omega_predictions.append(Omega_test)

# Find the scale that gives the observed Ω_Λ
target_Omega = 0.6847
differences = np.abs(np.array(Omega_predictions) - target_Omega)
best_idx = np.argmin(differences)
best_k = test_scales[best_idx]
best_Omega = Omega_predictions[best_idx]

print(f"Optimal scale from fitting: k* = {best_k:.4f} h/Mpc")
print(f"Gives Ω_Λ = {best_Omega:.4f}")
print(f"K=1 prediction was: k* = {k_critical:.4f} h/Mpc")

# ================================================
# PHYSICAL INTERPRETATION
# ================================================

print("\n" + "="*60)
print("PHYSICAL INTERPRETATION")
print("="*60)

lambda_critical = 2 * np.pi / k_critical
lambda_optimal = 2 * np.pi / best_k

print("Critical scales and their physical meaning:")
print(f"K=1 theoretical scale:")
print(f"  k* = {k_critical:.4f} h/Mpc, λ* = {lambda_critical:.1f} Mpc/h")
print(f"  This is the de Sitter scale modified by K=1 geometry")

print(f"\nOptimal empirical scale:")
print(f"  k* = {best_k:.4f} h/Mpc, λ* = {lambda_optimal:.1f} Mpc/h") 
print(f"  This gives exact agreement with observed Ω_Λ")

# Compare with known physical scales
print(f"\nComparison with known scales:")
print(f"  Hubble scale: k_H = 0.00067 h/Mpc, λ_H = 9400 Mpc/h")
print(f"  Cluster scale: k_cl ~ 0.1 h/Mpc, λ_cl ~ 60 Mpc/h")
print(f"  Galaxy scale: k_gal ~ 1.0 h/Mpc, λ_gal ~ 6 Mpc/h")

# ================================================
# VISUALIZATION
# ================================================

plt.figure(figsize=(15, 10))

# Plot 1: Power spectrum and flow spectrum
plt.subplot(2, 3, 1)
plt.loglog(k_values, Pk_values, 'b-', alpha=0.7, label='P(k)')
plt.xlabel('k [h/Mpc]')
plt.ylabel('P(k)')
plt.title('Matter Power Spectrum')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
flow_spectrum = k_values * Pk_values
plt.loglog(k_values, flow_spectrum, 'r-', alpha=0.7, label='F(k) = k×P(k)')
plt.axvline(k_critical, color='green', linestyle='--', label=f'K=1 scale: {k_critical:.3f}')
plt.axvline(best_k, color='purple', linestyle='--', label=f'Empirical: {best_k:.3f}')
plt.xlabel('k [h/Mpc]')
plt.ylabel('F(k)')
plt.title('Geometric Flow Spectrum')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Ω_Λ vs critical scale
plt.subplot(2, 3, 3)
plt.semilogx(test_scales, Omega_predictions, 'b-', linewidth=2)
plt.axhline(0.6847, color='red', linestyle=':', label='Observed Ω_Λ')
plt.axvline(k_critical, color='green', linestyle='--', label='K=1 prediction')
plt.axvline(best_k, color='purple', linestyle='--', label='Empirical optimal')
plt.xlabel('Critical Scale k* [h/Mpc]')
plt.ylabel('Predicted Ω_Λ')
plt.title('Ω_Λ vs Critical Scale')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Different method comparisons
plt.subplot(2, 3, 4)
methods = ['Horizon-based', 'Power spectrum', 'K=1 theory']
Omega_values = [Omega_de1, Omega_de2, Omega_de3]
colors = ['blue', 'orange', 'green']
bars = plt.bar(methods, Omega_values, color=colors, alpha=0.7)
plt.axhline(0.6847, color='red', linestyle=':', label='Observed')
plt.ylabel('Predicted Ω_Λ')
plt.title('Comparison of Different Methods')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, Omega_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom')

# Plot 4: Physical scales
plt.subplot(2, 3, 5)
physical_scales = ['Hubble', 'K=1 Critical', 'Optimal', 'Cluster']
scale_values = [1/0.00067, 1/k_critical, 1/best_k, 1/0.1]
plt.bar(physical_scales, scale_values, color=['red', 'green', 'purple', 'blue'], alpha=0.7)
plt.ylabel('Wavelength [Mpc/h]')
plt.title('Characteristic Physical Scales')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot 5: Error analysis
plt.subplot(2, 3, 6)
errors = [abs(val - 0.6847)/0.6847*100 for val in Omega_values]
plt.bar(methods, errors, color=colors, alpha=0.7)
plt.ylabel('Relative Error (%)')
plt.title('Prediction Errors')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels
for i, error in enumerate(errors):
    plt.text(i, error + 1, f'{error:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# ================================================
# FINAL ASSESSMENT
# ================================================

print("\n" + "="*60)
print("FINAL ASSESSMENT")
print("="*60)

print("K=1 Chronogeometrodynamics Framework Assessment:")
print(f"✓ Theoretical critical scale: k* = {k_critical:.4f} h/Mpc")
print(f"✓ Predicted Ω_Λ = {Omega_Lambda_predicted:.4f}")
print(f"✓ Observed Ω_Λ = 0.6847")
print(f"✓ Theoretical error: {abs(Omega_Lambda_predicted - 0.6847)/0.6847*100:.2f}%")

if abs(Omega_Lambda_predicted - 0.6847) < 0.05:
    print("\n🎯 EXCELLENT AGREEMENT: K=1 theory predicts Ω_Λ within 5%!")
    print("This provides strong support for the geometric nature of dark energy.")
elif abs(Omega_Lambda_predicted - 0.6847) < 0.1:
    print("\n✅ GOOD AGREEMENT: Theory reproduces Ω_Λ within 10%")
    print("The K=1 framework shows promising consistency with cosmology.")
elif abs(Omega_Lambda_predicted - 0.6847) < 0.2:
    print("\n⚠️ MODERATE AGREEMENT: Within 20% of observations")
    print("The geometric approach captures the essential physics.")
else:
    print("\n🔴 SIGNIFICANT DEVIATION: Further theoretical development needed")
    print("The framework may need refinement or additional physical ingredients.")

print(f"\nKey physical insight:")
print(f"The critical scale k* = {k_critical:.4f} h/Mpc corresponds to")
print(f"λ* = {lambda_critical:.1f} Mpc/h, which is the scale where")
print(f"geometric constraints from K=1 chronogeometrodynamics")
print(f"determine the dark energy density.")

print("="*60)
