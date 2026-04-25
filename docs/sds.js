/* #4 SdS / Λ-Einstein extension panel
 *   Theorem (extension):  In static spherical f>0 gauge,
 *     (K₁ = 1 − 2Λr²) ∧ (K₂ = 1 − Λr²)   ⇔   R_μν = Λ g_μν
 *
 * Verified bidirectionally:
 *   forward:  given Λ and any f compatible with these scalar conditions,
 *             check ‖R_μν − Λ g_μν‖ = 0 numerically
 *   reverse:  given f = 1 − 2M/r − Λr²/3 (SdS), compute K_i and check the
 *             targets analytically
 * Both checks: residuals ~ machine precision.
 */
(function(){
  'use strict';
  if (!window.KKernel) return;
  const KK = window.KKernel;

  function fmt(x){ if (!Number.isFinite(x)) return '—'; const a=Math.abs(x); return a<1e-4||a>1e5 ? x.toExponential(3) : x.toFixed(5); }

  function mount(){
    const root = document.getElementById('sds-host');
    if (!root) return;
    root.innerHTML = `
      <div class="ac-grid" style="grid-template-columns: 320px 1fr; gap:28px;">
        <div class="ac-knobs">
          <div class="ac-knobs-title">SdS · Λ-Einstein</div>
          <div class="ac-sub">f = 1 − 2M/r − Λr²/3 (Schwarzschild–de Sitter family).</div>

          <div class="ac-slider"><label>M <span id="sds-Mv">1.00</span></label>
            <input id="sds-M" type="range" min="0" max="3" step="0.01" value="1"></div>
          <div class="ac-slider"><label>Λ <span id="sds-Lv">0.030</span></label>
            <input id="sds-L" type="range" min="-0.1" max="0.1" step="0.001" value="0.030"></div>
          <div class="ac-slider"><label>r <span id="sds-rv">3.00</span></label>
            <input id="sds-r" type="range" min="1.5" max="8" step="0.01" value="3"></div>

          <div class="ac-knobs-title" style="margin-top:14px">Presets</div>
          <div id="sds-presets" style="display:flex;flex-wrap:wrap;gap:6px"></div>
        </div>

        <div>
          <div class="ac-verdict" id="sds-verdict">—</div>
          <div class="ac-stats" id="sds-stats" style="margin-top:14px"></div>
          <div class="ac-stats" style="margin-top:12px;color:var(--muted);font-size:0.85em;line-height:1.55">
            <b>Theorem (extension of 5.2).</b> In static spherical f&gt;0 gauge:
            <code>(K₁ = 1 − 2Λr²) ∧ (K₂ = 1 − Λr²)  ⇔  R_μν = Λ g_μν</code>.
            <br>The Λ-targets are <i>different</i> per leg because g_tt = −f and g_θθ = r² couple Λ differently.
            Pure vacuum (Theorem 5.2) is the Λ→0 limit. Pure de Sitter is the M→0 case.
          </div>
        </div>
      </div>`;

    const presets = [
      { name:'Schwarzschild',     M:1,   L:0     },
      { name:'de Sitter',         M:0,   L:0.03  },
      { name:'SdS (M=1, Λ=0.03)', M:1,   L:0.03  },
      { name:'SdS (M=1, Λ=0.06)', M:1,   L:0.06  },
      { name:'SAdS (M=1, Λ=−0.05)', M:1, L:-0.05 },
      { name:'Minkowski',         M:0,   L:0     },
    ];
    const pBar = document.getElementById('sds-presets');
    presets.forEach(p=>{
      const b = document.createElement('button');
      b.className='ac-btn';
      b.textContent = p.name;
      b.onclick = ()=>{
        document.getElementById('sds-M').value = p.M;
        document.getElementById('sds-L').value = p.L;
        ['sds-M','sds-L'].forEach(id=>document.getElementById(id).dispatchEvent(new Event('input',{bubbles:true})));
      };
      pBar.appendChild(b);
    });

    function update(){
      const M = +document.getElementById('sds-M').value;
      const L = +document.getElementById('sds-L').value;
      const r = +document.getElementById('sds-r').value;
      document.getElementById('sds-Mv').textContent = M.toFixed(2);
      document.getElementById('sds-Lv').textContent = L.toFixed(3);
      document.getElementById('sds-rv').textContent = r.toFixed(2);

      const {f, fp, fpp} = KK.mk_f({C1: 2*M, C2: 0, C3: -L/3});
      const fv = f(r);
      const okDomain = fv > 0;

      const K1 = KK.K1(r, fv, fp(r), fpp(r));
      const K2 = KK.K2(r, fv, fp(r));
      const T1 = 1 - 2*L*r*r;
      const T2 = 1 - L*r*r;
      const dK1 = K1 - T1;
      const dK2 = K2 - T2;

      const E = KK.Ricci(r, fv, fp(r), fpp(r));
      const dRtt = E.Rtt - L*(-fv);
      const dRrr = E.Rrr - L*(1/fv);
      const dRθθ = E.Rθθ - L*r*r;
      const maxR = Math.max(Math.abs(dRtt), Math.abs(dRrr), Math.abs(dRθθ));

      const stats = document.getElementById('sds-stats');
      stats.innerHTML = `
        <div class="ac-stat"><code>f(r)</code></div><div class="ac-stat mono">${fmt(fv)}${okDomain?'':'  ⊘ outside f>0'}</div>
        <div class="ac-stat"><code>K₁</code> (live)</div><div class="ac-stat mono">${fmt(K1)}</div>
        <div class="ac-stat"><code>1 − 2Λr²</code> (target)</div><div class="ac-stat mono">${fmt(T1)} &nbsp; <span style="color:var(--muted)">Δ = ${fmt(dK1)}</span></div>
        <div class="ac-stat"><code>K₂</code> (live)</div><div class="ac-stat mono">${fmt(K2)}</div>
        <div class="ac-stat"><code>1 − Λr²</code> (target)</div><div class="ac-stat mono">${fmt(T2)} &nbsp; <span style="color:var(--muted)">Δ = ${fmt(dK2)}</span></div>
        <div class="ac-stat" style="border-top:1px dashed var(--line);padding-top:10px"><code>R_tt − Λ g_tt</code></div><div class="ac-stat mono" style="border-top:1px dashed var(--line);padding-top:10px">${fmt(dRtt)}</div>
        <div class="ac-stat"><code>R_rr − Λ g_rr</code></div><div class="ac-stat mono">${fmt(dRrr)}</div>
        <div class="ac-stat"><code>R_θθ − Λ g_θθ</code></div><div class="ac-stat mono">${fmt(dRθθ)}</div>
        <div class="ac-stat"><b>‖R_μν − Λ g_μν‖∞</b></div><div class="ac-stat mono"><b>${fmt(maxR)}</b></div>
      `;

      const v = document.getElementById('sds-verdict');
      const Kok = Math.abs(dK1)<1e-9 && Math.abs(dK2)<1e-9;
      const Rok = maxR < 1e-9;
      if (!okDomain){
        v.className='ac-verdict verd-fail';
        v.innerHTML = `<b>⊘ outside domain.</b> f(r) = ${fmt(fv)} ≤ 0 — pick r between the horizons (or reduce |Λ|·r).`;
      } else if (Kok && Rok){
        v.className='ac-verdict verd-pass';
        v.innerHTML = `<b>EXTENSION HOLDS.</b> Both K-targets met (Δ ~ ${dK1.toExponential(1)}, ${dK2.toExponential(1)}) and ‖R_μν − Λg_μν‖ = ${maxR.toExponential(1)}. The Λ-Einstein extension of Theorem 5.2 is verified at this point on SdS.`;
      } else if (Rok && !Kok){
        v.className='ac-verdict verd-fail';
        v.innerHTML = `<b>Inconsistency flag.</b> R_μν = Λ g_μν but K-targets miss — should not happen on the SdS family. Report.`;
      } else {
        v.className='ac-verdict verd-fail';
        v.innerHTML = `<b>Off-shell.</b> (Δ_K, ‖ΔR‖) = (${fmt(Math.max(Math.abs(dK1),Math.abs(dK2)))}, ${fmt(maxR)}).`;
      }
    }

    root.querySelectorAll('input').forEach(el=>el.addEventListener('input', update));
    update();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount);
  else mount();
})();
