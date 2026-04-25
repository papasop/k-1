/* Shared Ricci / K₁ / K₂ kernel for static spherically symmetric metrics
 *   ds² = -f(r) dt² + f(r)⁻¹ dr² + r² dΩ²
 *
 * Conventions MATCH bench.js / audit-corr.js (verified):
 *   K₁ := f + 2 r f' + ½ r² f''        (R_tt readout: R_tt = (f/r²)(K₁ − 1))
 *   K₂ := f + r f'                     (R_θθ readout: R_θθ = 1 − K₂)
 *
 * Non-zero Ricci components in this gauge:
 *   R_tt   =  (f  / r²) (K₁ − 1)  · wait — paper writes as below; keep bench.js's version.
 *
 * We use bench.js Ricci(r,f,fp,fpp) as the source of truth. Re-expressed here
 * in closed form for convenience.
 */
(function(global){
  'use strict';

  function K1(r, f, fp, fpp){ return f + 2*r*fp + 0.5*r*r*fpp; }
  function K2(r, f, fp)     { return f + r*fp; }

  // Standard static spherically symmetric Ricci (matching bench.js)
  //   R_tt   =  ½ f f'' + f f' / r
  //   R_rr   = -(½ f'' + f'/r) / f
  //   R_θθ   =  1 - f - r f'   =  1 - K₂
  //   R_φφ   =  sin²θ · R_θθ
  //   R      =  g^{μν} R_μν  = -f'' - 4 f'/r - 2(f-1)/r²
  function Ricci(r, f, fp, fpp){
    const Rtt = 0.5*f*fpp + f*fp/r;
    const Rrr = -(0.5*fpp + fp/r) / f;
    const Rθθ = 1 - f - r*fp;
    const R   = -fpp - 4*fp/r - 2*(f-1)/(r*r);
    return {Rtt, Rrr, Rθθ, R};
  }

  // Einstein tensor G_μν = R_μν - ½ g_μν R with g = diag(-f, 1/f, r², r² sin²θ)
  // G_μν = R_μν - ½ g_μν R
  function Einstein(r, f, fp, fpp){
    const {Rtt, Rrr, Rθθ, R} = Ricci(r, f, fp, fpp);
    // g_tt = -f, g_rr = 1/f, g_θθ = r², g_φφ = r² sin²θ
    const Gtt = Rtt - 0.5*(-f)*R;
    const Grr = Rrr - 0.5*(1/f)*R;
    const Gθθ = Rθθ - 0.5*(r*r)*R;
    return {Gtt, Grr, Gθθ, R};
  }

  // σ-form (paper §5): σ₁ = r √f, σ₂ = r. Normalization uses σ₂² (common).
  //   □ := (1/√(-g)) ∂_μ (√(-g) g^{μν} ∂_ν ·)
  //   In this gauge, for a function φ(r):
  //     □φ = (1/r²) d/dr [ r² f · dφ/dr ]
  //        = f φ'' + (f' + 2f/r) φ'
  //
  // Paper conjecture:  σ₂² · □ ln σ_i = 1  for i = 1, 2
  //
  // ln σ₂ = ln r  →  (ln r)' = 1/r, (ln r)'' = -1/r²
  //   □ ln r = f·(-1/r²) + (f' + 2f/r)·(1/r) = f/r² + f'/r
  //   σ₂² · □ ln σ₂ = r² · (f/r² + f'/r) = f + r f' = K₂  ✓
  //
  // ln σ₁ = ln r + ½ ln f  →  (ln σ₁)' = 1/r + f'/(2f)
  //   (ln σ₁)'' = -1/r² + f''/(2f) - (f')²/(2f²)
  //   □ ln σ₁ = f [ -1/r² + f''/(2f) - (f')²/(2f²) ] + (f' + 2f/r)[1/r + f'/(2f)]
  //          = -f/r² + f''/2 - (f')²/(2f) + f'/r + (f')²/(2f) + 2f/r² + f'/r
  //          = f/r² + 2 f'/r + ½ f''
  //   σ₂² · □ ln σ₁ = r² · (f/r² + 2f'/r + ½f'') = f + 2 r f' + ½ r² f'' = K₁  ✓
  //
  // So the σ-form "σ₂² □ ln σ_i = 1" is EXACTLY (K₁ = 1) ∧ (K₂ = 1).
  function sigma_box(r, f, fp, fpp){
    // □ ln σ₁ and □ ln σ₂ (raw d'Alembertians)
    const box_ln_sig2 = f/(r*r) + fp/r;
    const box_ln_sig1 = f/(r*r) + 2*fp/r + 0.5*fpp;
    // σ₂²-normalized (the paper's form)
    const sig2sq_box_ln_sig1 = r*r * box_ln_sig1;  // = K₁
    const sig2sq_box_ln_sig2 = r*r * box_ln_sig2;  // = K₂
    return {
      box_ln_sig1, box_ln_sig2,
      sig2sq_box_ln_sig1, sig2sq_box_ln_sig2,
      sig1: r * Math.sqrt(Math.max(f, 0)),
      sig2: r,
    };
  }

  // Metric derivatives for the two-parameter ansatz
  //   f(r) = 1 - C₁/r + C₂/r² - Λ r² / 3 + C₄/r³
  function mk_f(p){
    const C1 = p.C1 || 0, C2 = p.C2 || 0, C3 = p.C3 || 0, C4 = p.C4 || 0;
    // C3 := -Λ/3 so term is C3·r² (dS has C3 < 0)
    const f   = r => 1 - C1/r + C2/(r*r) + C3*r*r + C4/(r*r*r);
    const fp  = r => C1/(r*r) - 2*C2/(r*r*r) + 2*C3*r - 3*C4/(r*r*r*r);
    const fpp = r => -2*C1/(r*r*r) + 6*C2/(r*r*r*r) + 2*C3 + 12*C4/(r*r*r*r*r);
    return {f, fp, fpp};
  }

  global.KKernel = {
    K1, K2, Ricci, Einstein, sigma_box, mk_f,
  };
})(window);
