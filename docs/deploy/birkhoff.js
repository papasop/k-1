/* Birkhoff uniqueness panel — explicit ODE-level derivation
 *
 * Claim chain (paper §5, terminal steps):
 *   (i)  In static spherically symmetric gauge with f(r) > 0,
 *           K₁ ≡ f + 2rf' + r²f''/2,  K₂ ≡ f + rf' = d(rf)/dr
 *        with R = 2(1 − K₁)/r²,  R_θθ = 1 − K₂,
 *        and R_tt = (f/r²)·(K₁ − K₂).
 *   (ii) K₁ = 1  ⇔  R = 0              (one 2nd-order Euler ODE, 2 constants:
 *                                        f = 1 + A/r + B/r²  — admits RN, non-vacuum)
 *   (iii) K₂ = 1 ⇔  R_θθ = 0           (one 1st-order ODE, 1 constant:
 *                                        f = 1 + C/r, which forces K₁ ≡ 1 as well)
 *
 *   Rigidity (Lemma 5.3 in the paper):  K₂ ≡ 1  ⇒  K₁ ≡ 1,  but K₁ ≡ 1  ⇏  K₂ ≡ 1.
 *   The two legs are NOT symmetric in this gauge.
 *
 * What this panel shows, numerically:
 *
 *   Stage 1 · One leg alone (K₁ = 1) has a 2-parameter solution family:
 *       f(r) = 1 - C₁/r - C₂/r²     (Corollary; verified numerically by
 *                                    integrating (d/dr)(r²f') from (r₀, f₀, f'₀))
 *       So K₁ = 1 is NOT enough — it admits RN (non-vacuum, R_μν ≠ 0).
 *
 *   Stage 2 · The second leg (K₂ = 1) pins C₂ = 0 uniquely:
 *       K₂ = f + rf' = 1   →   (rf)' = 1   →   f = 1 - C₁/r
 *       So the two-parameter family collapses to Schwarzschild.
 *
 *   Stage 3 · Rename C₁ = 2M (physical mass). Unique Schwarzschild f = 1 - 2M/r.
 *
 * The interactive: user sets (f₀, f'₀) at a reference radius r₀.
 *   • Stage 1 numerically integrates the K₁=1 ODE outward and shows f(r) matches
 *     1 - C₁/r - C₂/r² with (C₁, C₂) recovered from (f₀, f'₀).
 *   • The panel reports "residual from the Corollary" (should be ~0: the ODE
 *     HAS this 2-parameter family as its general solution — algebraic identity).
 *   • Stage 2 sweeps C₂ and measures ||K₂ - 1||∞ on the window. The minimum
 *     is attained at C₂ = 0 exactly; we plot that curve and mark the zero.
 *   • Stage 3 reads off C₁ → M.
 *
 * This does NOT re-derive GR. It makes the DOF-counting in Row 5 executable:
 * 2 constants from R_tt=0, minus 1 from R_θθ=0 = 1 parameter left = M.
 */
(function(){
  'use strict';
  const $ = id => document.getElementById(id);
  const TOL = 1e-8;

  function fmt(v, dec=4){
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a < 1e-3 || a >= 1e5) return v.toExponential(2);
    return v.toFixed(dec);
  }

  // ---- formulas (matching bench.js / audit-corr.js conventions) ----
  // K1 = f + 2 r f' + ½ r² f''      ⇔   R_tt = 0
  // K2 = f + r f'                   ⇔   R_θθ = 0
  function K1_from(r, f, fp, fpp){ return f + 2*r*fp + 0.5*r*r*fpp; }
  function K2_from(r, f, fp){      return f + r*fp; }

  // K1 = 1  ⇔  ½ r² f'' + 2 r f' + (f - 1) = 0
  //      ⇔  r² f'' + 4 r f' + 2(f - 1) = 0
  // General solution: f = 1 - C₁/r - C₂/r²
  // Substitute u = r² f  →  general sol f = 1 + A/r + B/r²
  // We parametrize with C₁ = -A, C₂ = -B so f = 1 - C₁/r - C₂/r².
  function solveK1(r0, f0, fp0){
    // Given (f, f') at r = r₀, recover (C₁, C₂):
    //   f  = 1 - C₁/r - C₂/r²
    //   f' = C₁/r² + 2 C₂/r³
    // Two linear equations:
    //   -(f₀ - 1)·r₀  = C₁ + C₂/r₀      (×r₀ ·)
    //   fp₀·r₀²·r₀    = C₁·r₀ + 2 C₂    (×r₀² ·)
    // Solve 2×2 system:
    //   [ 1   1/r₀ ] [C₁]   [ (1-f₀)·r₀ ]
    //   [ r₀  2    ] [C₂] = [ fp₀·r₀³   ]
    const a11 = 1,   a12 = 1/r0;
    const a21 = r0,  a22 = 2;
    const b1 = (1 - f0)*r0;
    const b2 = fp0 * r0*r0*r0;
    const det = a11*a22 - a12*a21;   // = 2 - 1 = 1
    const C1 = (b1*a22 - b2*a12) / det;
    const C2 = (a11*b2 - a21*b1) / det;
    return { C1, C2 };
  }

  // Reconstruct f, f', f'' from the recovered (C₁, C₂)
  function recon(r, C1, C2){
    return {
      f:   1 - C1/r - C2/(r*r),
      fp:  C1/(r*r) + 2*C2/(r*r*r),
      fpp: -2*C1/(r*r*r) - 6*C2/(r*r*r*r),
    };
  }

  // Independent ODE check: with (f₀, f'₀) as ICs, integrate K₁ = 1 as an ODE
  //   f'' = -[4 r f' + 2(f-1)] / r²
  // and compare to closed-form at endpoint.
  function integrateK1(r0, f0, fp0, rEnd, N=2000){
    const h = (rEnd - r0) / N;
    let r = r0, f = f0, fp = fp0;
    for (let i=0; i<N; i++){
      const rhs = (r_, f_, fp_) => -(4*r_*fp_ + 2*(f_-1)) / (r_*r_);
      // RK4
      const k1f = fp,                   k1p = rhs(r, f, fp);
      const k2f = fp + 0.5*h*k1p,       k2p = rhs(r + 0.5*h, f + 0.5*h*k1f, fp + 0.5*h*k1p);
      const k3f = fp + 0.5*h*k2p,       k3p = rhs(r + 0.5*h, f + 0.5*h*k2f, fp + 0.5*h*k2p);
      const k4f = fp + h*k3p,           k4p = rhs(r + h,     f + h*k3f,     fp + h*k3p);
      f  += h * (k1f + 2*k2f + 2*k3f + k4f) / 6;
      fp += h * (k1p + 2*k2p + 2*k3p + k4p) / 6;
      r  += h;
    }
    return { r, f, fp };
  }

  // ---------- Stage 1: K₁=1 → 2-parameter family ----------
  function mountStage1(){
    const card = $('bk-s1');
    if (!card) return;
    const r0S  = $('bk-r0'),  r0V  = $('bk-r0v');
    const f0S  = $('bk-f0'),  f0V  = $('bk-f0v');
    const fp0S = $('bk-fp0'), fp0V = $('bk-fp0v');
    const C1O  = $('bk-C1'),  C2O  = $('bk-C2');
    const odeO = $('bk-ode');
    const resid = $('bk-resid');
    const vOut = $('bk-verdict');
    const cv = $('bk-canvas'); const ctx = cv?.getContext('2d');

    function draw(){
      const r0  = +r0S.value;  r0V.textContent  = r0.toFixed(2);
      const f0  = +f0S.value;  f0V.textContent  = f0.toFixed(3);
      const fp0 = +fp0S.value; fp0V.textContent = fp0.toFixed(3);

      const { C1, C2 } = solveK1(r0, f0, fp0);
      C1O.textContent = fmt(C1);
      C2O.textContent = fmt(C2);

      // Cross-check: RK4-integrate the K₁=1 ODE from r₀ out to r₀+4, then
      // compare f(rEnd) to closed-form 1 - C₁/r - C₂/r².
      const rEnd = r0 + 4;
      const num  = integrateK1(r0, f0, fp0, rEnd);
      const cls  = recon(rEnd, C1, C2);
      const errF  = Math.abs(num.f  - cls.f);
      const errFp = Math.abs(num.fp - cls.fp);
      odeO.textContent =
        `RK4(r₀ → ${rEnd.toFixed(2)}) : f=${fmt(num.f)} · f'=${fmt(num.fp)}   ` +
        `vs   closed-form : f=${fmt(cls.f)} · f'=${fmt(cls.fp)}`;

      const err = Math.max(errF, errFp);
      resid.textContent = `‖f_ODE − (1 − C₁/r − C₂/r²)‖ = ${err.toExponential(2)}   ·   tol = 1e-6`;

      const pass = err < 1e-6;
      vOut.className = 'ac-verdict ' + (pass ? 'verd-pass' : 'verd-fail');
      if (pass){
        const c2Tol = 2e-2;  // slider granularity is ~1e-3 in f,f' → propagates to ~1e-2 in C₂
        const c2Big = Math.abs(C2) > c2Tol;
        vOut.innerHTML =
          `<b>GENERAL SOLUTION CONFIRMED.</b> The K₁=1 ODE admits a 2-parameter family ` +
          `<code>f = 1 − C₁/r − C₂/r²</code>. For your ICs (f₀, f'₀) at r₀=${r0.toFixed(2)}: ` +
          `C₁ = ${fmt(C1)}, C₂ = ${fmt(C2)}. ` +
          (c2Big
            ? `<br>Since C₂ ≠ 0, this is <b>not yet vacuum</b> — R_θθ = −C₂/r² ≠ 0. Stage 2 below will cut this dimension.`
            : `<br>Your ICs sit on the C₂ ≈ 0 slice — this is already the Stage-2 vacuum locus (Schwarzschild with M = C₁/2 ≈ ${(C1/2).toFixed(2)}). Move f'₀ away to see the family fan out.`);
      } else {
        vOut.textContent = `✗ RK4 and closed-form diverged — numerical integration drift (not a theorem failure).`;
      }

      // plot f(r) on [r₀, r₀+5], compare integrated vs closed-form
      if (ctx){
        const W = cv.width = cv.clientWidth * devicePixelRatio;
        const H = cv.height = cv.clientHeight * devicePixelRatio;
        ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
        const w = cv.clientWidth, h = cv.clientHeight;
        ctx.clearRect(0,0,w,h);
        const pad = {l:46, r:10, t:10, b:22};
        const r1 = r0, r2 = r0+5;
        // sample closed-form f
        const N = 200;
        const xs = [], ys = [];
        for (let i=0;i<=N;i++){
          const r = r1 + (r2-r1)*i/N;
          xs.push(r); ys.push(1 - C1/r - C2/(r*r));
        }
        const ymin = Math.min(...ys, 0.3), ymax = Math.max(...ys, 1.1);
        const X = r => pad.l + (r - r1) / (r2-r1) * (w - pad.l - pad.r);
        const Y = y => pad.t + (1 - (y - ymin)/(ymax - ymin)) * (h - pad.t - pad.b);

        // axes
        ctx.strokeStyle = 'oklch(0.88 0.01 268)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
        // y = 1 reference
        ctx.strokeStyle = 'oklch(0.9 0.02 55)'; ctx.setLineDash([3,3]);
        ctx.beginPath(); ctx.moveTo(pad.l, Y(1)); ctx.lineTo(w - pad.r, Y(1)); ctx.stroke();
        ctx.setLineDash([]);
        // closed-form curve
        ctx.strokeStyle = 'oklch(0.52 0.18 268)'; ctx.lineWidth = 2.0;
        ctx.beginPath(); for(let i=0;i<=N;i++){ const x = X(xs[i]), y = Y(ys[i]); if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); } ctx.stroke();

        // labels
        ctx.fillStyle = 'oklch(0.45 0.02 268)'; ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillText('f(r) = 1 − C₁/r − C₂/r²', pad.l + 6, pad.t + 14);
        ctx.fillText('y=1', w - pad.r - 26, Y(1) - 4);
        ctx.fillText(r1.toFixed(1), pad.l - 4, h - 6);
        ctx.fillText(r2.toFixed(1), w - pad.r - 16, h - 6);
        ctx.fillText(ymax.toFixed(2), 6, pad.t + 10);
        ctx.fillText(ymin.toFixed(2), 6, h - pad.b + 3);
      }
    }
    [r0S, f0S, fp0S].forEach(el => el.oninput = draw);
    draw();
  }

  // ---------- Stage 2: K₂=1 sweep picks out C₂=0 ----------
  function mountStage2(){
    const card = $('bk-s2');
    if (!card) return;
    const C1S = $('bk2-C1'), C1V = $('bk2-C1v');
    const C2S = $('bk2-C2'), C2V = $('bk2-C2v');
    const K2norm = $('bk2-norm');
    const vOut = $('bk2-verdict');
    const cv = $('bk2-canvas'); const ctx = cv?.getContext('2d');

    function draw(){
      const C1 = +C1S.value; C1V.textContent = C1.toFixed(2);
      const C2 = +C2S.value; C2V.textContent = C2.toFixed(3);

      // For f = 1 - C₁/r - C₂/r²:  K₂ = f + r f' = 1 + C₂/r²
      // So ‖K₂ - 1‖∞ on [r₁, r₂] = |C₂| / r₁²  (max at inner edge)
      const r1 = 2, r2 = 8;
      const sup = Math.abs(C2) / (r1*r1);
      K2norm.textContent = `‖K₂ − 1‖∞ on [${r1}, ${r2}] = ${fmt(sup)}`;

      const vacuum = Math.abs(C2) < 1e-6;
      vOut.className = 'ac-verdict ' + (vacuum ? 'verd-pass' : 'verd-info');
      if (vacuum){
        vOut.innerHTML =
          `<b>VACUUM LOCUS REACHED.</b> With C₂ = 0, the second leg K₂ = 1 holds identically — ` +
          `the 2-parameter family has collapsed to the 1-parameter Schwarzschild family ` +
          `<code>f = 1 − C₁/r</code>. Stage 3 identifies C₁ = 2M.`;
      } else {
        vOut.innerHTML =
          `C₂ ≠ 0, so K₂ − 1 = −C₂/r² is a non-zero function on the entire window. ` +
          `Only the single point C₂ = 0 in this 1-D family satisfies K₂ = 1 everywhere — ` +
          `that is the uniqueness of the second integration constant.`;
      }

      // plot K₂-1 vs r for the current (C₁, C₂) and for the C₂=0 slice
      if (ctx){
        const W = cv.width = cv.clientWidth * devicePixelRatio;
        const H = cv.height = cv.clientHeight * devicePixelRatio;
        ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
        const w = cv.clientWidth, h = cv.clientHeight;
        ctx.clearRect(0,0,w,h);
        const pad = {l:46, r:10, t:10, b:22};
        const N = 200;
        const ys = [];
        for (let i=0;i<=N;i++){
          const r = r1 + (r2-r1)*i/N;
          ys.push(-C2/(r*r));  // K₂ - 1
        }
        const ymag = Math.max(Math.abs(C2)/(r1*r1), 0.05);
        const ymin = -ymag*1.2, ymax = ymag*1.2;
        const X = r => pad.l + (r - r1)/(r2-r1)*(w - pad.l - pad.r);
        const Y = y => pad.t + (1 - (y - ymin)/(ymax-ymin))*(h - pad.t - pad.b);

        ctx.strokeStyle = 'oklch(0.88 0.01 268)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
        // zero reference
        ctx.strokeStyle = 'oklch(0.66 0.14 140)'; ctx.lineWidth = 1.6; ctx.setLineDash([5,3]);
        ctx.beginPath(); ctx.moveTo(pad.l, Y(0)); ctx.lineTo(w-pad.r, Y(0)); ctx.stroke();
        ctx.setLineDash([]);
        // K₂-1 curve for current C₂
        ctx.strokeStyle = vacuum ? 'oklch(0.55 0.16 140)' : 'oklch(0.55 0.2 30)';
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        for (let i=0;i<=N;i++){
          const r = r1 + (r2-r1)*i/N;
          const x = X(r), y = Y(ys[i]);
          if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        }
        ctx.stroke();

        ctx.fillStyle = 'oklch(0.45 0.02 268)'; ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillText('K₂ − 1 = −C₂/r²', pad.l + 6, pad.t + 14);
        ctx.fillText('0', w - pad.r - 12, Y(0) - 4);
        ctx.fillText('r = '+r1, pad.l - 4, h - 6);
        ctx.fillText('r = '+r2, w - pad.r - 28, h - 6);
      }
    }
    [C1S, C2S].forEach(el => el.oninput = draw);
    draw();
  }

  // ---------- Stage 3: Unique f = 1 - 2M/r ----------
  function mountStage3(){
    const card = $('bk-s3');
    if (!card) return;
    const MS = $('bk3-M'), MV = $('bk3-Mv');
    const rhOut = $('bk3-rh');
    const kappaOut = $('bk3-kappa');
    const T_H = $('bk3-TH');
    const cv = $('bk3-canvas'); const ctx = cv?.getContext('2d');

    function draw(){
      const M = +MS.value; MV.textContent = M.toFixed(3);
      // f = 1 - 2M/r
      const rh = 2*M;
      const kappa = 1/(4*M);      // surface gravity
      const Th = kappa/(2*Math.PI); // Hawking temperature (units κ/2π)
      rhOut.textContent = `r_s = 2M = ${fmt(rh)}`;
      kappaOut.textContent = `κ = 1/(4M) = ${fmt(kappa)}`;
      T_H.textContent = `T_H = κ/(2π) = ${fmt(Th)}`;

      if (ctx){
        const W = cv.width = cv.clientWidth * devicePixelRatio;
        const H = cv.height = cv.clientHeight * devicePixelRatio;
        ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
        const w = cv.clientWidth, h = cv.clientHeight;
        ctx.clearRect(0,0,w,h);
        const pad = {l:46, r:10, t:10, b:22};
        const r1 = Math.max(0.1, rh - 1.5), r2 = rh + 6;
        const N = 240;
        const xs = [], ys = [];
        for (let i=0;i<=N;i++){
          const r = r1 + (r2-r1)*i/N;
          xs.push(r); ys.push(r > 0 ? 1 - 2*M/r : NaN);
        }
        const ymin = -1, ymax = 1.2;
        const X = r => pad.l + (r-r1)/(r2-r1)*(w - pad.l - pad.r);
        const Y = y => pad.t + (1 - (y-ymin)/(ymax-ymin))*(h - pad.t - pad.b);

        // axes
        ctx.strokeStyle = 'oklch(0.88 0.01 268)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h-pad.b); ctx.lineTo(w-pad.r, h-pad.b); ctx.stroke();
        // y=0 line
        ctx.strokeStyle = 'oklch(0.9 0.02 55)'; ctx.setLineDash([3,3]);
        ctx.beginPath(); ctx.moveTo(pad.l, Y(0)); ctx.lineTo(w-pad.r, Y(0)); ctx.stroke();
        ctx.setLineDash([]);
        // horizon line
        ctx.strokeStyle = 'oklch(0.65 0.15 30)'; ctx.setLineDash([5,4]);
        ctx.beginPath(); ctx.moveTo(X(rh), pad.t); ctx.lineTo(X(rh), h-pad.b); ctx.stroke();
        ctx.setLineDash([]);
        // f(r) curve
        ctx.strokeStyle = 'oklch(0.52 0.18 268)'; ctx.lineWidth = 2.2;
        ctx.beginPath();
        let drew = false;
        for (let i=0;i<=N;i++){
          if (!isFinite(ys[i])) { drew = false; continue; }
          const x = X(xs[i]), y = Y(ys[i]);
          if (!drew) { ctx.moveTo(x,y); drew = true; } else ctx.lineTo(x,y);
        }
        ctx.stroke();
        // labels
        ctx.fillStyle = 'oklch(0.45 0.02 268)'; ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillText('f(r) = 1 − 2M/r', pad.l + 6, pad.t + 14);
        ctx.fillStyle = 'oklch(0.45 0.15 30)';
        ctx.fillText('r_s = 2M', X(rh) + 4, pad.t + 28);
        ctx.fillStyle = 'oklch(0.45 0.02 268)';
        ctx.fillText(r1.toFixed(1), pad.l - 4, h - 6);
        ctx.fillText(r2.toFixed(1), w - pad.r - 18, h - 6);
      }
    }
    MS.oninput = draw;
    draw();
  }

  function boot(){
    mountStage1();
    mountStage2();
    mountStage3();
  }
  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', boot);
  else
    boot();
})();
