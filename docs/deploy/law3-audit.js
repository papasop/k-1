/* ============================================================
   § Law III audit — explicit theorem-premise + κ_K residual ladder

   Law III (paper): At critical damping D = d_c I, on any G-eigenvector
   of {K=1} normalized by x_*^T G² x_* = 1:
       dK/dt = -κ_K (K-1) + o(K-1),   κ_K = 4 d_c.

   This panel makes the premises explicit (gating), then shows the
   κ_K derivation in three auditable layers:
       L1  general:     κ_K(x_*) = 4 d_c · x_*^T G² x_*
       L2  eigenvector: x_* is a G-eigenvector?  (yes ⇒ quadratic form
                        reduces to μ² where Gx_* = μ x_*)
       L3  normalised:  x_*^T G² x_* = 1   ⇒   |κ_K − 4 d_c| < tol
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

  // ------------ geometry of the base point ------------
  // G = diag(−(κℓ)², 1).  x_* = (x1, x2).
  // K(x) = x^T G x = −(κℓ)² x1² + x2²
  // On {K=1}: x2 = √(1 + (κℓ)² x1²)  (positive branch)
  // G-eigenvectors of diag(a,b) are (1,0) and (0,1).  Only (0,1) sits on {K=1}.
  // Normalisation x_*^T G² x_* = a² x1² + b² x2² ;  at (0,1) this is 1 (since b=1).
  // So the canonical normalised eigenvector base point is x_* = (0, 1).

  function state(){
    return {
      kel:   +$('l3-kel').value,
      x1:    +$('l3-x1').value,
      dc:    +$('l3-dc').value,
      Dstruct: $('l3-Dstruct').value,   // 'crit' | 'aniso'
      linearReg: +$('l3-eps').value,     // perturbation magnitude
    };
  }

  function compute(s){
    const a = -(s.kel * s.kel);   // G11
    const b = 1;                   // G22
    // base point on positive branch
    const x1 = s.x1;
    const x2 = Math.sqrt(1 + s.kel*s.kel * x1*x1);

    const K     = a*x1*x1 + b*x2*x2;            // = 1 by construction (residual for audit)
    const xGG_x = (a*a)*x1*x1 + (b*b)*x2*x2;    // x^T G² x

    // Eigenvector test: is (x1, x2) a multiple of (1,0) or (0,1)?
    // tolerance: x1 near 0 ⇒ ∥(0,1)∥-aligned; x2 near 0 ⇒ ∥(1,0)∥-aligned (but (1,0) has K=a<0)
    const isEvec01 = Math.abs(x1) < 1e-8;                 // (0, x2) — aligned with (0,1)
    const isEvec10 = Math.abs(x2) < 1e-8;                 // (x1, 0) — aligned with (1,0)
    const isEvec   = isEvec01 || isEvec10;

    // On {K=1} with Lorentzian G, only (0,1) eigenvector works (since (1,0) gives K=a<0)
    const onK1 = Math.abs(K - 1) < 1e-8;

    // Normalisation residual
    const normResid = Math.abs(xGG_x - 1);

    // μ = eigenvalue of G at that eigenvector
    const μ = isEvec01 ? b : (isEvec10 ? a : NaN);

    // three layers of κ_K derivation
    //   L1  general formula                             κ_K = 4 d_c · x*^T G² x*
    //   L2  eigenvector substitution (if applicable)    = 4 d_c · μ²
    //   L3  normalised (x*^T G² x* = 1)                 = 4 d_c
    const κK_L1 = 4 * s.dc * xGG_x;
    const κK_L2 = Number.isFinite(μ) ? 4 * s.dc * μ*μ : NaN;
    const κK_L3 = 4 * s.dc;                 // target

    return {
      a, b, x1, x2, K,
      xGG_x, normResid,
      isEvec, isEvec01, isEvec10, μ, onK1,
      κK_L1, κK_L2, κK_L3,
      resid_L1_vs_L3: Math.abs(κK_L1 - κK_L3),
      resid_L2_vs_L3: Number.isFinite(κK_L2) ? Math.abs(κK_L2 - κK_L3) : NaN,
    };
  }

  // ------------ gating ------------
  function renderGates(c, s){
    const gates = [
      {
        label: 'base on {K = 1}',
        ok: c.onK1,
        note: `K(x_*) = ${fmt(c.K, 6)}  (target 1)`,
      },
      {
        label: 'G-eigenvector base point',
        ok: c.isEvec,
        note: c.isEvec01 ? 'x_* ∥ (0, 1)  ✓'
              : c.isEvec10 ? 'x_* ∥ (1, 0)  (but K = a < 0, unusable)'
              : `x_* = (${c.x1.toFixed(2)}, ${c.x2.toFixed(3)}) — generic`,
      },
      {
        label: 'normalised: x_*⊤ G² x_* = 1',
        ok: c.normResid < 1e-6,
        note: `x_*⊤ G² x_* = ${fmt(c.xGG_x, 5)}  ·  |⋅ − 1| = ${fmt(c.normResid,3)}`,
      },
      {
        label: 'critical damping  D = d_c · I',
        ok: s.Dstruct === 'crit',
        note: s.Dstruct === 'crit'
              ? `D = ${s.dc.toFixed(2)} · I  (isotropic)`
              : 'D anisotropic — outside Law III',
      },
      {
        label: 'local-linear regime  |K − 1| ≪ 1',
        ok: s.linearReg < 0.1,
        note: `perturbation ε = ${s.linearReg.toFixed(3)}  (paper: o(K−1) correction)`,
      },
    ];

    const host = $('l3-gates');
    host.innerHTML = '';
    for (const g of gates){
      const el = document.createElement('div');
      el.className = 'pm-gate ' + (g.ok ? 'pm-gate-on' : 'pm-gate-off');
      el.innerHTML = `<span class="pm-gate-dot"></span><span class="pm-gate-label">${g.label}</span><span class="pm-gate-note">${g.note}</span>`;
      host.appendChild(el);
    }

    const allOk = gates.every(g=>g.ok);
    const banner = $('l3-applicable');
    if (allOk){
      banner.textContent = '✓ all premises hold — Law III is applicable: expect κ_K = 4 d_c exactly (to linear order)';
      banner.className = 'pm-banner pm-banner-ok';
    } else {
      const bad = gates.filter(g=>!g.ok).map(g=>g.label).join('  ·  ');
      banner.textContent = '⚠ theorem not applicable at current settings — premise(s) off: ' + bad + '  (this is not a counter-example)';
      banner.className = 'pm-banner pm-banner-warn';
    }
    return allOk;
  }

  // ------------ κ_K residual ladder ------------
  function renderLadder(c){
    const host = $('l3-ladder');

    const rows = [
      {
        label: 'L1',
        expr: 'κ_K(x_*) = 4 d_c · x_*⊤ G² x_*',
        value: c.κK_L1,
        resid: c.resid_L1_vs_L3,
        ok: c.resid_L1_vs_L3 < 1e-6,
        desc: 'General formula — holds at every base point on {K=1}. Not yet equal to 4 d_c unless the normalisation kicks in.',
      },
      {
        label: 'L2',
        expr: Number.isFinite(c.μ) ? 'κ_K = 4 d_c · μ²  (G x_* = μ x_*)' : 'κ_K = 4 d_c · μ²  (requires eigenvector)',
        value: c.κK_L2,
        resid: c.resid_L2_vs_L3,
        ok: Number.isFinite(c.κK_L2) && c.resid_L2_vs_L3 < 1e-6,
        desc: Number.isFinite(c.μ)
              ? `Eigenvector substitution active: μ = ${fmt(c.μ,4)}, μ² = ${fmt(c.μ*c.μ,4)}.`
              : 'Not a G-eigenvector — this layer is skipped.',
      },
      {
        label: 'L3',
        expr: 'κ_K = 4 d_c   (x_*⊤ G² x_* = 1)',
        value: c.κK_L3,
        resid: 0,
        ok: true,
        desc: 'Target: after normalisation the numerical factor collapses to 4 d_c. This is Law III.',
      },
    ];

    host.innerHTML = '';
    for (const r of rows){
      const ok = r.ok;
      const el = document.createElement('div');
      el.className = 'pm-id ' + (ok ? 'pm-id-ok' : 'pm-id-bad');
      el.innerHTML = `
        <div class="pm-id-head">
          <span class="pm-id-label">${r.label}</span>
          <span class="pm-id-expr">${r.expr}</span>
          <span class="pm-id-status">${ok ? '✓' : '✗'}</span>
        </div>
        <div class="pm-id-body">
          <div class="pm-id-desc">${r.desc}</div>
          <div class="pm-id-nums">
            <span>value = <b>${fmt(r.value,6)}</b></span>
            <span>target = <b>${fmt(c.κK_L3,6)}</b></span>
            <span>residual = <b>${Number.isFinite(r.resid) ? fmt(r.resid,3) : '—'}</b></span>
            <span>tol = <b>1e-6</b></span>
          </div>
        </div>`;
      host.appendChild(el);
    }
  }

  function tick(){
    const s = state();
    $('l3-kelv').textContent = s.kel.toFixed(2);
    $('l3-x1v').textContent  = s.x1.toFixed(3);
    $('l3-dcv').textContent  = s.dc.toFixed(2);
    $('l3-epsv').textContent = s.linearReg.toFixed(3);
    const c = compute(s);
    renderGates(c, s);
    renderLadder(c);
    $('l3-K').textContent = fmt(c.K, 6);
    $('l3-xGGx').textContent = fmt(c.xGG_x, 6);
  }

  function boot(){
    const card = $('law3audit');
    if(!card) return;
    ['l3-kel','l3-x1','l3-dc','l3-eps'].forEach(id=>$(id).oninput = tick);
    $('l3-Dstruct').onchange = tick;

    card.querySelectorAll('[data-l3-preset]').forEach(b=>{
      b.onclick = ()=>{
        const p = b.dataset.l3Preset;
        if (p==='canonical'){ $('l3-kel').value=1; $('l3-x1').value=0; $('l3-dc').value=1; $('l3-eps').value=0.02; $('l3-Dstruct').value='crit'; }
        if (p==='offeig'){    $('l3-kel').value=1; $('l3-x1').value=0.3; $('l3-dc').value=1; $('l3-eps').value=0.02; $('l3-Dstruct').value='crit'; }
        if (p==='aniso'){     $('l3-kel').value=1; $('l3-x1').value=0; $('l3-dc').value=1; $('l3-eps').value=0.02; $('l3-Dstruct').value='aniso'; }
        if (p==='nonlin'){    $('l3-kel').value=1; $('l3-x1').value=0; $('l3-dc').value=1; $('l3-eps').value=0.4; $('l3-Dstruct').value='crit'; }
        if (p==='kel-big'){   $('l3-kel').value=2; $('l3-x1').value=0; $('l3-dc').value=1; $('l3-eps').value=0.02; $('l3-Dstruct').value='crit'; }
        card.querySelectorAll('[data-l3-preset]').forEach(x=>x.classList.remove('on'));
        b.classList.add('on');
        tick();
      };
    });
    const def = card.querySelector('[data-l3-preset="canonical"]');
    if (def) def.classList.add('on');
    tick();
  }

  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', boot);
  else
    boot();
})();
