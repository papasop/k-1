/* ============================================================
   Table 1 · Point ↔ Field correspondence — interactive tests
   5 rows × 2 columns (point-level 2D | field-level 4D)
   Each row exposes a numerical test with adjustable inputs,
   a pass condition, residuals vs tolerance, and a verdict.
   Mounted into <section id="corr">.
   ============================================================ */

(function(){
  'use strict';
  const $ = (id)=>document.getElementById(id);

  const TOL = 1e-9;

  function fmt(x, d){
    d = d==null?4:d;
    if (!Number.isFinite(x)) return '—';
    const a = Math.abs(x);
    if (a !== 0 && (a < 1e-3 || a >= 1e4)) return x.toExponential(Math.max(1,d-2));
    return x.toFixed(d);
  }
  function setResid(el, pairs, tol){
    const parts = pairs.map(([k,v])=> `${k} = ${fmt(v, 3)}`).join('   ');
    el.textContent = parts + `   ·   tol = ${tol.toExponential(0)}`;
  }
  function verdict(el, pass, text){
    el.textContent = text;
    if (pass === null){
      el.style.background = 'var(--paper-2)';
      el.style.color = 'var(--ink-2)';
      el.style.borderColor = 'var(--line)';
      return;
    }
    el.style.background = pass ? 'color-mix(in oklch, var(--accent-soft) 85%, transparent)'
                               : 'oklch(0.94 0.06 25)';
    el.style.color = pass ? 'var(--accent-ink)' : 'oklch(0.35 0.15 25)';
    el.style.borderColor = pass ? 'var(--accent)' : 'oklch(0.65 0.15 25)';
  }

  function mk_f(C1, C2){
    const f   = (r)=> 1 - C1/r - C2/(r*r);
    const fp  = (r)=> C1/(r*r) + 2*C2/(r*r*r);
    const fpp = (r)=> -2*C1/(r*r*r) - 6*C2/(r*r*r*r);
    return { f, fp, fpp };
  }
  function K1_of(r, f, fp, fpp){ return f(r) + 2*r*fp(r) + 0.5*r*r*fpp(r); }
  function K2_of(r, f, fp){      return f(r) + r*fp(r); }
  function Ricci(r, f, fp, fpp){
    const fv = f(r), fpv = fp(r), fppv = fpp(r);
    const Rtt = 0.5*fv*fppv + fv*fpv/r;
    const Rrr = -Rtt / (fv*fv);
    const Rθθ = 1 - fv - r*fpv;
    return { Rtt, Rrr, Rθθ };
  }

  function mountRow1(){
    const card = $('row-principle');
    if(!card) return;

    const kel = $('r1p-kel'), kelv = $('r1p-kelv');
    const x1s = $('r1p-x1'),  x1v  = $('r1p-x1v');
    const Kout = $('r1p-K'), residP = $('r1p-resid'), vP = $('r1p-verdict');

    function drawP(){
      const kl = +kel.value;  kelv.textContent = kl.toFixed(2);
      const x1 = +x1s.value;  x1v.textContent = x1.toFixed(2);
      const x2 = Math.sqrt(1 + kl*kl * x1*x1);
      const K  = -(kl*kl)*x1*x1 + x2*x2;
      Kout.textContent = fmt(K, 6);
      const resid = Math.abs(K-1);
      setResid(residP, [['|K − 1|', resid]], TOL);
      const ok = resid < TOL;
      verdict(vP, ok, ok
        ? `✓ K = 1 exactly at x = (${x1.toFixed(2)}, ${x2.toFixed(3)})`
        : `✗ K = ${fmt(K,6)} ≠ 1`);
    }
    [kel, x1s].forEach(el=>el.oninput=drawP);

    const rs = $('r1f-r'), rv = $('r1f-rv');
    const Ms = $('r1f-M'), Mv = $('r1f-Mv');
    const k1o = $('r1f-K1'), k2o = $('r1f-K2');
    const residF = $('r1f-resid'), vF = $('r1f-verdict');

    function drawF(){
      const r = +rs.value;  rv.textContent = r.toFixed(2);
      const M = +Ms.value;  Mv.textContent = M.toFixed(2);
      const { f, fp, fpp } = mk_f(2*M, 0);
      if (r <= 2*M + 1e-6){
        k1o.textContent = '—'; k2o.textContent = '—';
        residF.textContent = 'r ≤ 2M: σ₁ = r√f undefined (real branch)';
        verdict(vF, false, `✗ inside horizon — framework undefined here`);
        return;
      }
      const K1 = K1_of(r, f, fp, fpp);
      const K2 = K2_of(r, f, fp);
      k1o.textContent = fmt(K1,6);
      k2o.textContent = fmt(K2,6);
      const rK1 = Math.abs(K1-1), rK2 = Math.abs(K2-1);
      setResid(residF, [['|K₁−1|', rK1], ['|K₂−1|', rK2]], TOL);
      const ok = rK1 < TOL && rK2 < TOL;
      verdict(vF, ok, ok
        ? `✓ both eigenvalue conditions hold exactly`
        : `✗ violation on Schwarzschild — something is wrong`);
    }
    [rs, Ms].forEach(el=>el.oninput=drawF);

    drawP(); drawF();
  }

  function mountRow2(){
    const card = $('row-balance');
    if(!card) return;

    const kel = $('r2p-kel'), kelv = $('r2p-kelv');
    const Kout = $('r2p-K'), residP = $('r2p-resid'), vP = $('r2p-verdict');
    function drawP(){
      const kl = +kel.value; kelv.textContent = kl.toFixed(2);
      const x1 = 0, x2 = 1;
      const K  = -(kl*kl)*x1*x1 + x2*x2;
      const xGGx = (kl*kl)*(kl*kl)*x1*x1 + x2*x2;
      const μ_other = -(kl*kl);
      Kout.textContent = fmt(K, 6);
      const r = Math.abs(K-1);
      setResid(residP, [['|K−1|', r], ['x_*ᵀG²x_*', xGGx], ['μ₁ (off-base)', μ_other]], TOL);
      const ok = r < TOL;
      verdict(vP, ok,
        ok ? `✓ unit cost K = 1 at x_* = (0, 1) — base-point identity, independent of κℓ; κℓ sets the off-base eigenvalue μ₁ = −(κℓ)² = ${fmt(μ_other,3)}`
           : `✗`);
    }
    kel.oninput = drawP;

    const rs = $('r2f-r'), rv = $('r2f-rv');
    const Ms = $('r2f-M'), Mv = $('r2f-Mv');
    const iSel = $('r2f-i');
    const lhsEl = $('r2f-lhs'), rhsEl = $('r2f-rhs');
    const residF = $('r2f-resid'), vF = $('r2f-verdict');

    function drawF(){
      const r = +rs.value; rv.textContent = r.toFixed(2);
      const M = +Ms.value; Mv.textContent = M.toFixed(2);
      const i = +iSel.value;
      const { f, fp, fpp } = mk_f(2*M, 0);
      if (r <= 2*M + 1e-6){
        lhsEl.textContent = '—'; rhsEl.textContent = '—';
        residF.textContent = 'inside horizon';
        verdict(vF, false, `✗ inside horizon — σᵢ not real`);
        return;
      }
      const sigma2 = r;
      const Ki = (i===1) ? K1_of(r, f, fp, fpp) : K2_of(r, f, fp);
      const lhs = Ki / (sigma2*sigma2);
      const rhs = 1 / (sigma2*sigma2);
      lhsEl.textContent = fmt(lhs);
      rhsEl.textContent = fmt(rhs);
      const resid = Math.abs(lhs - rhs);
      setResid(residF, [['|LHS − RHS|', resid]], TOL);
      const ok = resid < TOL;
      verdict(vF, ok, ok
        ? `✓ □ln σ${i==1?'₁':'₂'} = 1/σ₂² on Schwarzschild`
        : `✗ mismatch`);
    }
    [rs, Ms, iSel].forEach(el=>el.oninput = drawF);

    drawP(); drawF();
  }

  function mountRow3(){
    const card = $('row-vacuum');
    if(!card) return;

    const xs = $('r3p-x'), xv = $('r3p-xv');
    const Kout = $('r3p-K'), residP = $('r3p-resid'), vP = $('r3p-verdict');
    function drawP(){
      const x1 = +xs.value; xv.textContent = x1.toFixed(2);
      const K = -x1*x1 + 1;
      const V = 0.5*(K-1)*(K-1);
      Kout.textContent = fmt(K, 4);
      const r = Math.abs(K-1);
      setResid(residP, [['|K − 1|', r], ['V', V]], TOL);
      const ok = r < 1e-6;
      verdict(vP, ok, ok
        ? `✓ on vacuum locus K = 1, V = 0`
        : `✗ off locus — penalty V = ½(K−1)² = ${fmt(V,5)} > 0`);
    }
    xs.oninput = drawP;

    const C1s = $('r3f-C1'), C1v = $('r3f-C1v');
    const C2s = $('r3f-C2'), C2v = $('r3f-C2v');
    const rs = $('r3f-r'), rv = $('r3f-rv');
    const k1o = $('r3f-K1'), k2o = $('r3f-K2');
    const rttO = $('r3f-Rtt'), rrrO = $('r3f-Rrr'), rθθO = $('r3f-Rθθ');
    const residF = $('r3f-resid'), vF = $('r3f-verdict');

    function drawF(){
      const C1 = +C1s.value; C1v.textContent = C1.toFixed(2);
      const C2 = +C2s.value; C2v.textContent = C2.toFixed(2);
      const r = +rs.value;   rv.textContent = r.toFixed(2);
      const { f, fp, fpp } = mk_f(C1, C2);
      const fv = f(r);
      if (fv <= 0){
        k1o.textContent='—'; k2o.textContent='—';
        rttO.textContent='—'; rrrO.textContent='—'; rθθO.textContent='—';
        residF.textContent = 'outside real branch (f ≤ 0)';
        verdict(vF, false, `✗ f(r) ≤ 0 — outside domain`);
        return;
      }
      const K1 = K1_of(r, f, fp, fpp);
      const K2 = K2_of(r, f, fp);
      const { Rtt, Rrr, Rθθ } = Ricci(r, f, fp, fpp);
      k1o.textContent = fmt(K1);
      k2o.textContent = fmt(K2);
      rttO.textContent = fmt(Rtt);
      rrrO.textContent = fmt(Rrr);
      rθθO.textContent = fmt(Rθθ);
      const rK1 = Math.abs(K1-1), rK2 = Math.abs(K2-1);
      const rR = Math.max(Math.abs(Rtt), Math.abs(Rrr), Math.abs(Rθθ));
      setResid(residF, [['|K₁−1|', rK1], ['|K₂−1|', rK2], ['max|Rμν|', rR]], TOL);
      const condK = rK1 < TOL && rK2 < TOL;
      const condR = rR  < TOL;
      if (condK && condR){
        verdict(vF, true,  `✓ no contradiction at (C₁=${fmt(C1,2)}, C₂=${fmt(C2,2)}, r=${fmt(r,2)}) · both K = 1 and Rμν = 0 hold here (vacuum slice of the ansatz)`);
      } else if (!condK && !condR){
        verdict(vF, null,  `vacuous at this sample: K ≠ 1 and Rμν ≠ 0 — both sides of the iff fail simultaneously, so no test was performed (still consistent with the theorem; scan §P for an interval-level check)`);
      } else if (!condK && condR){
        verdict(vF, false, `✗ would-be counter-example: Rμν = 0 yet K ≠ 1 at this sample — falsifies reverse direction`);
      } else {
        verdict(vF, false, `✗ would-be counter-example: K = 1 yet Rμν ≠ 0 at this sample — falsifies forward direction`);
      }
    }
    [C1s, C2s, rs].forEach(el=>el.oninput = drawF);

    drawP(); drawF();
  }

  function mountRow4(){
    const card = $('row-matter');
    if(!card) return;

    const ks = $('r4p-K'), kv = $('r4p-Kv');
    const vout = $('r4p-V'), residP = $('r4p-resid'), vP = $('r4p-verdict');
    function drawP(){
      const K = +ks.value; kv.textContent = K.toFixed(3);
      const V = 0.5*(K-1)*(K-1);
      vout.textContent = fmt(V, 5);
      const r = Math.abs(K-1);
      setResid(residP, [['|K−1|', r], ['V', V]], TOL);
      if (r < 1e-6)
        verdict(vP, null, `display only · K = 1 sits on the vacuum locus, so V = 0 (no penalty) — no claim being tested`);
      else
        verdict(vP, null, `display only · off the locus, V = ½(K−1)² = ${fmt(V,5)} > 0 — schematic penalty, no claim being tested`);
    }
    ks.oninput = drawP;

    const rho = $('r4f-rho'), rhoV = $('r4f-rhoV');
    const rs  = $('r4f-r'),  rv   = $('r4f-rv');
    const k2o = $('r4f-K2'), rhoO = $('r4f-rhoOut');
    const residF = $('r4f-resid'), vF = $('r4f-verdict');

    function drawF(){
      const ρ0 = +rho.value; rhoV.textContent = ρ0.toFixed(3);
      const r  = +rs.value;  rv.textContent  = r.toFixed(2);
      const K2 = 1 - 8*Math.PI * ρ0 * r*r;
      const ρ_rec = (1 - K2) / (8*Math.PI * r*r);
      k2o.textContent  = fmt(K2, 6);
      rhoO.textContent = fmt(ρ_rec, 5);
      const resid = Math.abs(ρ_rec - ρ0);
      setResid(residF, [['|ρ_rec − ρ₀|', resid]], TOL);
      const ok = resid < TOL;
      verdict(vF, ok, ok
        ? `✓ definitional round-trip exact · ρ → K₂ → ρ_rec = ${fmt(ρ_rec,4)} (this is an algebraic inversion, not a theorem test — that lives in §3)`
        : `✗ round-trip drifted — numerical regression`);
    }
    [rho, rs].forEach(el=>el.oninput = drawF);

    drawP(); drawF();
  }

  function mountRow5(){
    const card = $('row-count');
    if(!card) return;

    const p1 = $('r5p-K1');
    const vP = $('r5p-verdict');
    function drawP(){
      const on = p1.checked ? 1 : 0;
      verdict(vP, on===1,
        on===1 ? `✓ 1 condition active · 1 DOF in 2D → fully pinned`
               : `✗ 0 conditions · 1 DOF → under-determined`);
    }
    p1.onchange = drawP;

    const f1 = $('r5f-K1'), f2 = $('r5f-K2');
    const soln = $('r5f-sol');
    const vF = $('r5f-verdict');
    function drawF(){
      const a = f1.checked, b = f2.checked;
      let txt;
      if (!a && !b){
        txt = 'f free — 2-param family';
        verdict(vF, false, `0 conditions · 2 DOF → generic static SS metric, R_μν unconstrained`);
      } else if (a && !b){
        txt = 'f = 1 − C₁/r − C₂/r²   (R_scalar = 0, RN-like)';
        verdict(vF, null, `1 condition · 2 DOF → C₂ free · R_scalar = 0 holds (since R = 2(1−K₁)/r²) but R_μν ≠ 0 unless C₂ = 0 — partially pinned, not yet vacuum`);
      } else if (!a && b){
        txt = 'f = 1 − 2m/r  (m = const., mass-shell only)';
        verdict(vF, null, `1 condition · 2 DOF → R_θθ = 0 holds (since R_θθ = 1−K₂) but R_tt ≠ 0 unless K₁ = 1 too — partially pinned, not yet vacuum`);
      } else {
        txt = 'f = 1 − 2M/r  (Schwarzschild, unique)';
        verdict(vF, true, `✓ 2 conditions · 2 DOF → R_μν = 0 uniquely recovered (Schwarzschild)`);
      }
      soln.textContent = txt;
    }
    f1.onchange = drawF;
    f2.onchange = drawF;

    drawP(); drawF();
  }

  function mountPaperToggles(){
    document.querySelectorAll('.corr-paper-toggle').forEach(btn=>{
      btn.addEventListener('click', ()=>{
        const tgt = document.getElementById(btn.dataset.target);
        if (!tgt) return;
        const open = tgt.hasAttribute('data-open');
        if (open){ tgt.removeAttribute('data-open'); tgt.style.maxHeight='0'; btn.textContent = btn.dataset.labelClosed; }
        else     { tgt.setAttribute('data-open',''); tgt.style.maxHeight = tgt.scrollHeight + 'px'; btn.textContent = btn.dataset.labelOpen; }
      });
    });
  }

  function boot(){
    mountRow1();
    mountRow2();
    mountRow3();
    mountRow4();
    mountRow5();
    mountPaperToggles();
  }
  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', boot);
  else
    boot();
})();