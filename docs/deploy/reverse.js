/* #3 Reverse-leg counter-example panel
 *
 * Theorem 5.2:  R_μν = 0   ⇔   K₁ = K₂ = 1
 * (in static spherical f>0 gauge)
 *
 * Question: in the same gauge, can we satisfy *one* leg without the other?
 *
 *   Forward direction (R_μν=0 ⇒ K₁=K₂=1):   no counter-example exists in
 *     this gauge — Birkhoff. The forward implication is sharp.
 *
 *   Reverse legs:
 *     (a) K₁ = 1 alone   →  Euler ODE ½r²f'' + 2rf' + (f−1) = 0
 *                         general solution f = 1 + A/r + B/r² (RN family).
 *                         Counter-example: A=2M, B=Q²>0 (RN).
 *     (b) K₂ = 1 alone   →  (rf)' = 1, f = 1 + C/r (Schwarzschild family,
 *                         possibly with negative mass C=-2M' for C>0).
 *                         CLAIM (rigidity): K₂ ≡ 1 on an open interval
 *                         ⇒ K₁ ≡ 1 on the same interval.
 *                         No counter-example.  (verified by direct sub.)
 *
 * Result:  the two legs are *not* symmetric.  K₂=1 alone ⇒ K₁=1; but
 * K₁=1 alone does NOT ⇒ K₂=1.  RN is the explicit family that breaks
 * the K₁ leg open.  Theorem 5.2 needs *both* legs as hypothesis only
 * because of the K₁=1 direction.
 */
(function(){
  'use strict';
  if (!window.KKernel) return;
  const KK = window.KKernel;
  const fmt = (x)=>{ if(!Number.isFinite(x)) return '—'; const a=Math.abs(x); return a<1e-4||a>1e5 ? x.toExponential(3) : x.toFixed(5); };

  function mount(){
    const root = document.getElementById('reverse-host');
    if (!root) return;
    root.innerHTML = `
      <div class="ac-grid" style="grid-template-columns: 320px 1fr; gap:28px;">
        <div class="ac-knobs">
          <div class="ac-knobs-title">Test the legs separately</div>
          <div class="ac-sub">Pick which leg you force, then dial parameters and watch the <i>other</i> leg.</div>

          <div class="ac-slider"><label>Leg forced</label>
            <select id="rv-leg" style="width:100%;padding:6px;font-family:'JetBrains Mono',monospace;background:var(--paper);border:1px solid var(--line);color:var(--ink-1);border-radius:4px">
              <option value="K1">K₁ = 1 (RN family: f = 1 + A/r + B/r²)</option>
              <option value="K2">K₂ = 1 (Schwarzschild-like: f = 1 + C/r)</option>
              <option value="free">neither — generic two-parameter ansatz</option>
            </select>
          </div>

          <div class="ac-slider" id="rv-A-row"><label>A &nbsp;<span style="color:var(--muted);text-transform:none">(= −2M for Schw. sign)</span> <span id="rv-Av">−2.00</span></label>
            <input id="rv-A" type="range" min="-3" max="3" step="0.01" value="-2"></div>
          <div class="ac-slider" id="rv-B-row"><label>B &nbsp;<span style="color:var(--muted);text-transform:none">(= +Q² for RN)</span> <span id="rv-Bv">0.50</span></label>
            <input id="rv-B" type="range" min="-1" max="2" step="0.01" value="0.5"></div>
          <div class="ac-slider"><label>r <span id="rv-rv">3.00</span></label>
            <input id="rv-r" type="range" min="1.5" max="8" step="0.01" value="3"></div>
        </div>

        <div>
          <div class="ac-verdict" id="rv-verdict">—</div>
          <div class="ac-stats" id="rv-stats" style="margin-top:14px"></div>
          <div class="ac-stats" style="margin-top:12px;color:var(--muted);font-size:0.85em;line-height:1.55">
            <b>Asymmetry between the legs.</b> Theorem 5.2 takes <code>K₁ = K₂ = 1</code> as <i>conjunction</i>. The interactive evidence:
            <ul style="margin:6px 0 0 18px;padding:0">
              <li><b>K₂ = 1 leg is rigid.</b> Forcing K₂ ≡ 1 on an open r-interval forces f = 1 + C/r, and direct substitution gives K₁ ≡ 1. So <code>(K₂≡1) ⇒ K₁≡1</code> — the K₂ leg alone determines vacuum.</li>
              <li><b>K₁ = 1 leg is NOT rigid.</b> Forcing K₁ ≡ 1 gives f = 1 + A/r + B/r², a 2-parameter family. K₂ = 1 + A/r + B/r² + r·(−A/r² − 2B/r³) = 1 − B/r². So K₂ = 1 only at B = 0. <b>RN (B = Q² &gt; 0) is the live counter-example.</b></li>
            </ul>
            The forward direction R_μν = 0 ⇒ K₁ = K₂ = 1 is closed by Birkhoff (Schwarzschild is the unique vacuum SS static metric).
          </div>
        </div>
      </div>`;

    const Aslider = document.getElementById('rv-A');
    const Bslider = document.getElementById('rv-B');
    const Arow = document.getElementById('rv-A-row');
    const Brow = document.getElementById('rv-B-row');
    const legSel = document.getElementById('rv-leg');

    function update(){
      const leg = legSel.value;
      // For K2=1 leg: f = 1 + C/r,  i.e. B is forced to 0; A stays free (with A = C).
      // For K1=1 leg: f = 1 + A/r + B/r²,  both A,B free.
      // For 'free': same general 2-param.
      let A = +Aslider.value;
      let B = +Bslider.value;
      if (leg === 'K2'){
        B = 0; Bslider.value = 0;  // hard-pin
        Brow.style.opacity = 0.4; Bslider.disabled = true;
      } else {
        Brow.style.opacity = 1; Bslider.disabled = false;
      }
      const r = +document.getElementById('rv-r').value;
      document.getElementById('rv-Av').textContent = A.toFixed(2);
      document.getElementById('rv-Bv').textContent = B.toFixed(2);
      document.getElementById('rv-rv').textContent = r.toFixed(2);

      // KKernel mk_f signature: f = 1 - C1/r - C2/r²  → here we want f = 1 + A/r + B/r²
      // so C1 = -A, C2 = -B
      const {f, fp, fpp} = KK.mk_f({C1: -A, C2: -B, C3: 0});
      const fv = f(r), fpv = fp(r), fppv = fpp(r);
      const ok = fv > 0;
      const K1 = KK.K1(r, fv, fpv, fppv);
      const K2 = KK.K2(r, fv, fpv);
      const E  = KK.Ricci(r, fv, fpv, fppv);
      const ricciNorm = Math.max(Math.abs(E.Rtt), Math.abs(E.Rrr), Math.abs(E.Rθθ));

      // Sanity: on the RN family K₁ = 1 algebraically; on the K₂=1 family, K₂ = 1 algebraically.
      const K1_dev = K1 - 1;
      const K2_dev = K2 - 1;

      const stats = document.getElementById('rv-stats');
      stats.innerHTML = `
        <div class="ac-stat"><code>f(r)</code></div><div class="ac-stat mono">${fmt(fv)}${ok?'':'  ⊘'}</div>
        <div class="ac-stat"><code>K₁</code></div><div class="ac-stat mono">${fmt(K1)} <span style="color:var(--muted)">(Δ = ${fmt(K1_dev)})</span></div>
        <div class="ac-stat"><code>K₂</code></div><div class="ac-stat mono">${fmt(K2)} <span style="color:var(--muted)">(Δ = ${fmt(K2_dev)})</span></div>
        <div class="ac-stat" style="border-top:1px dashed var(--line);padding-top:10px"><code>R_tt</code></div><div class="ac-stat mono" style="border-top:1px dashed var(--line);padding-top:10px">${fmt(E.Rtt)}</div>
        <div class="ac-stat"><code>R_θθ</code> = 1−K₂</div><div class="ac-stat mono">${fmt(E.Rθθ)}</div>
        <div class="ac-stat"><code>‖R_μν‖∞</code></div><div class="ac-stat mono">${fmt(ricciNorm)}</div>
      `;

      const v = document.getElementById('rv-verdict');
      const K1ok = Math.abs(K1_dev) < 1e-9;
      const K2ok = Math.abs(K2_dev) < 1e-9;
      const Rok  = ricciNorm < 1e-9;

      if (!ok){
        v.className='ac-verdict verd-fail';
        v.innerHTML = `<b>Outside domain.</b> f(r) ≤ 0 — pick larger r or smaller |A|, |B|.`;
        return;
      }

      if (leg === 'K1'){
        // forced K₁=1; check K₂
        if (K1ok && K2ok && Rok){
          v.className='ac-verdict verd-pass';
          v.innerHTML = `<b>K₁ = 1 holds.</b> At B = 0 the K₁=1 family <i>also</i> satisfies K₂ = 1 — the Schwarzschild point of the RN family. R_μν = 0. <i>Slide B away from 0 to see the K₁=1 leg fail to rigidify the geometry.</i>`;
        } else if (K1ok && !K2ok){
          v.className='ac-verdict verd-warn';
          v.innerHTML = `<b>K₁ = 1, K₂ ≠ 1 — leg-asymmetry verified.</b> Algebraic identity K₁ = 1 holds (RN family), but K₂ = 1 − B/r² = ${fmt(K2)} ≠ 1, so R_θθ = ${fmt(E.Rθθ)} ≠ 0. RN is a genuine 2-parameter family in the K₁=1 leg — Theorem 5.2 needs the second leg to pin it down.`;
        } else {
          v.className='ac-verdict verd-fail';
          v.innerHTML = `<b>Numerical anomaly.</b> K₁ should be 1 algebraically. Δ = ${fmt(K1_dev)}.`;
        }
      } else if (leg === 'K2'){
        // forced K₂=1 (B pinned to 0); check K₁
        if (K2ok && K1ok && Rok){
          v.className='ac-verdict verd-pass';
          v.innerHTML = `<b>K₂ = 1 leg is rigid.</b> Forcing K₂ ≡ 1 (i.e. f = 1 + A/r) <i>automatically</i> gives K₁ = 1 and R_μν = 0. The K₂=1 ODE has a 1-parameter family, but rigidity collapses it onto Schwarzschild (with mass M = −A/2). <b>The K₂ leg alone determines vacuum.</b>`;
        } else if (K2ok && !K1ok){
          v.className='ac-verdict verd-fail';
          v.innerHTML = `<b>Rigidity broken — bug.</b> K₂ = 1 should imply K₁ = 1 by direct substitution. Δ_K₁ = ${fmt(K1_dev)}.`;
        } else {
          v.className='ac-verdict verd-fail';
          v.innerHTML = `<b>Numerical drift.</b> K₂ should be 1 here; got Δ = ${fmt(K2_dev)}.`;
        }
      } else {
        // free
        if (Rok){
          v.className='ac-verdict verd-pass';
          v.innerHTML = `<b>Vacuum point.</b> R_μν = 0; (K₁, K₂) = (${fmt(K1)}, ${fmt(K2)}). Theorem 5.2 says this requires (K₁, K₂) = (1, 1) — confirmed.`;
        } else {
          v.className='ac-verdict verd-warn';
          v.innerHTML = `<b>Generic point.</b> (K₁, K₂) = (${fmt(K1)}, ${fmt(K2)}); ‖R_μν‖ = ${fmt(ricciNorm)}.`;
        }
      }
    }

    legSel.addEventListener('change', update);
    [Aslider, Bslider, document.getElementById('rv-r')].forEach(el => el.addEventListener('input', update));
    update();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount);
  else mount();
})();
