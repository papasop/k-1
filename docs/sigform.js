/* #2 σ-form explicit check panel
 *   σ₂² · □ ln σ_i = 1  (paper §5, Route B step 1)
 *   Verify directly, without routing through R_μν.
 *
 * Panel: slider over (C₁=2M, C₂=Q², C₃=-Λ/3), pick r, show
 *   • σ₂² □ ln σ₁  (= K₁)
 *   • σ₂² □ ln σ₂  (= K₂)
 * with verdict per preset.
 */
(function(){
  'use strict';
  if (!window.KKernel) return;
  const KK = window.KKernel;

  function fmt(x){ if (!Number.isFinite(x)) return '—'; const a = Math.abs(x); return a<1e-4 || a>1e5 ? x.toExponential(3) : x.toFixed(5); }

  function init(){
    const root = document.getElementById('sigform-host');
    if (!root) return;

    root.innerHTML = `
      <div class="ac-grid" style="grid-template-columns: 320px 1fr; gap:28px;">
        <div class="ac-knobs">
          <div class="ac-knobs-title">σ-form direct check</div>
          <div class="ac-sub">σ₂ = r.  Check σ₂²·□ln σ_i vs 1.</div>

          <div class="ac-slider">
            <label>preset</label>
            <div id="sf-presets" class="ac-row" style="display:flex;flex-wrap:wrap;gap:6px"></div>
          </div>

          <div class="ac-slider"><label>C₁ = 2M <span id="sf-C1v">2.00</span></label>
            <input id="sf-C1" type="range" min="0" max="4" step="0.01" value="2"></div>
          <div class="ac-slider"><label>C₂ = Q² <span id="sf-C2v">0.00</span></label>
            <input id="sf-C2" type="range" min="0" max="2" step="0.01" value="0"></div>
          <div class="ac-slider"><label>C₃ = −Λ/3 <span id="sf-C3v">0.000</span></label>
            <input id="sf-C3" type="range" min="-0.1" max="0.1" step="0.001" value="0"></div>
          <div class="ac-slider"><label>r <span id="sf-rv">3.00</span></label>
            <input id="sf-r" type="range" min="1.2" max="10" step="0.01" value="3"></div>
        </div>

        <div>
          <div class="ac-verdict" id="sf-verdict">—</div>
          <div class="ac-stats" id="sf-stats" style="margin-top:14px"></div>
          <div class="ac-stats" id="sf-notes" style="margin-top:12px;color:var(--muted);font-size:0.85em">
            Paper conjecture (Route B, step 1): <code>σ₂² · □ ln σ_i = 1, i=1,2</code>, with σ₁=r√f, σ₂=r.<br>
            Direct calculation gives <code>σ₂² □ ln σ₁ = K₁</code> and <code>σ₂² □ ln σ₂ = K₂</code>, so the σ-form conjecture IS (K₁,K₂)=(1,1).<br>
            This panel tests the σ-form identity <b>without</b> routing through R_μν.
          </div>
        </div>
      </div>`;

    const presets = [
      { name:'Schwarzschild (M=1)', C1:2, C2:0, C3:0 },
      { name:'RN (M=1,Q²=0.5)',      C1:2, C2:0.5, C3:0 },
      { name:'dS (Λ=0.03)',           C1:0, C2:0, C3:-0.01 },
      { name:'SdS (M=1,Λ=0.03)',     C1:2, C2:0, C3:-0.01 },
      { name:'Minkowski',             C1:0, C2:0, C3:0 },
    ];

    const pBar = document.getElementById('sf-presets');
    presets.forEach((p,i)=>{
      const b = document.createElement('button');
      b.className='ac-btn';
      b.textContent = p.name;
      b.onclick = ()=>{
        document.getElementById('sf-C1').value = p.C1;
        document.getElementById('sf-C2').value = p.C2;
        document.getElementById('sf-C3').value = p.C3;
        ['sf-C1','sf-C2','sf-C3'].forEach(id=>document.getElementById(id).dispatchEvent(new Event('input',{bubbles:true})));
      };
      pBar.appendChild(b);
    });

    function update(){
      const C1 = +document.getElementById('sf-C1').value;
      const C2 = +document.getElementById('sf-C2').value;
      const C3 = +document.getElementById('sf-C3').value;
      const r  = +document.getElementById('sf-r').value;
      document.getElementById('sf-C1v').textContent = C1.toFixed(2);
      document.getElementById('sf-C2v').textContent = C2.toFixed(2);
      document.getElementById('sf-C3v').textContent = C3.toFixed(3);
      document.getElementById('sf-rv').textContent  = r.toFixed(2);

      const {f, fp, fpp} = KK.mk_f({C1, C2, C3});
      const fv = f(r), fpv = fp(r), fppv = fpp(r);
      const s = KK.sigma_box(r, fv, fpv, fppv);

      const d1 = s.sig2sq_box_ln_sig1 - 1;
      const d2 = s.sig2sq_box_ln_sig2 - 1;
      const ok1 = Math.abs(d1) < 1e-9;
      const ok2 = Math.abs(d2) < 1e-9;

      const Lam = -3*C3;
      const targ1 = 1 - 2*Lam*r*r;
      const targ2 = 1 - Lam*r*r;
      const d1L = s.sig2sq_box_ln_sig1 - targ1;
      const d2L = s.sig2sq_box_ln_sig2 - targ2;

      const stats = document.getElementById('sf-stats');
      stats.innerHTML = `
        <div class="ac-stat"><code>σ₁ = r√f</code></div>
        <div class="ac-stat mono">${fmt(s.sig1)}</div>
        <div class="ac-stat"><code>σ₂ = r</code></div>
        <div class="ac-stat mono">${fmt(s.sig2)}</div>
        <div class="ac-stat"><code>σ₂² □ ln σ₁</code></div>
        <div class="ac-stat mono">${fmt(s.sig2sq_box_ln_sig1)} &nbsp; <span style="color:var(--muted)">(=K₁)</span></div>
        <div class="ac-stat"><code>σ₂² □ ln σ₂</code></div>
        <div class="ac-stat mono">${fmt(s.sig2sq_box_ln_sig2)} &nbsp; <span style="color:var(--muted)">(=K₂)</span></div>
        <div class="ac-stat">vs 1 (pure vacuum)</div>
        <div class="ac-stat mono">Δ₁ = ${fmt(d1)} &nbsp;·&nbsp; Δ₂ = ${fmt(d2)}</div>
        <div class="ac-stat">vs Λ-Einstein targets</div>
        <div class="ac-stat mono">K₁ vs (1−2Λr²): ${fmt(d1L)} &nbsp;·&nbsp; K₂ vs (1−Λr²): ${fmt(d2L)}</div>
      `;

      const v = document.getElementById('sf-verdict');
      if (ok1 && ok2){
        v.className = 'ac-verdict verd-pass';
        v.innerHTML = `<b>σ-FORM HOLDS.</b> Both <code>σ₂² □ ln σ_i = 1</code> to machine precision. ` +
                      `This is the direct field-level K=1 (Schwarzschild-type vacuum).`;
      } else if (Math.abs(d1L)<1e-9 && Math.abs(d2L)<1e-9 && Math.abs(Lam) > 1e-6){
        v.className = 'ac-verdict verd-pass';
        v.innerHTML = `<b>σ-FORM (Λ-modified) HOLDS.</b> ` +
                      `<code>σ₂²□ln σ₁ = 1−2Λr²</code> and <code>σ₂²□ln σ₂ = 1−Λr²</code> ` +
                      `with Λ=${Lam.toFixed(3)} — the de Sitter / SdS generalization. ` +
                      `Different targets per leg because g_tt = −f and g_θθ = r² couple Λ differently.`;
      } else if (ok1 && !ok2){
        v.className = 'ac-verdict verd-warn';
        v.innerHTML = `<b>σ₁ leg holds, σ₂ leg breaks.</b> <code>K₁ = 1</code> ` +
                      `but <code>K₂ = ${fmt(s.sig2sq_box_ln_sig2)} ≠ 1</code>. ` +
                      `Typical of RN-type matter (C₂ ≠ 0): R_θθ ≠ 0, R_tt ≠ 0 (since K₁−K₂ ≠ 0).`;
      } else if (ok2 && !ok1){
        v.className = 'ac-verdict verd-warn';
        v.innerHTML = `<b>σ₂ leg holds, σ₁ leg breaks.</b> <code>K₂ = 1</code> ` +
                      `but <code>K₁ = ${fmt(s.sig2sq_box_ln_sig1)} ≠ 1</code>. ` +
                      `Note: by the rigidity lemma (paper §5.2), K₂=1 <i>pointwise</i> forces K₁=1. ` +
                      `A single-r K₂=1 hit doesn't invoke this — need K₂≡1 on an interval.`;
      } else {
        v.className = 'ac-verdict verd-fail';
        v.innerHTML = `<b>σ-FORM FAILS both legs.</b> (Δ₁, Δ₂) = (${fmt(d1)}, ${fmt(d2)}). Not vacuum in this gauge.`;
      }
    }

    root.querySelectorAll('input').forEach(el=>el.addEventListener('input', update));
    update();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();