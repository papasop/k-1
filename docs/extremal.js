/* #5 Extremal / near-horizon panel
 *
 * RN: f = 1 − 2M/r + Q²/r²,  horizons at r_± = M ± √(M²−Q²).
 * Extremal: Q = M ⇒ r_± = M (double root).
 *
 * Three windows:
 *   (a) K₁(r), K₂(r) profile across r ∈ [r_h , r_h+δ]   — log-log near horizon
 *   (b) Near-horizon AdS₂×S² check:  set ρ = r − M; for extremal RN,
 *        f ≈ ρ²/M², so K₁ → 0  and  K₂ → 1 − 2M/r_h + Q²/r_h² + ... ≠ 1
 *   (c) "Distance to (1,1)":  ‖(K₁−1, K₂−1)‖ as a function of M−Q
 *        — non-extremal limits to 0 outside; extremal pins K₁→0 at the horizon.
 */
(function(){
  'use strict';
  if (!window.KKernel) return;
  const KK = window.KKernel;
  const fmt = (x)=> { if (!Number.isFinite(x)) return '—'; const a=Math.abs(x); return a<1e-4||a>1e5?x.toExponential(3):x.toFixed(5); };

  function rPlus(M,Q){ const d = M*M - Q*Q; return d>=0 ? M + Math.sqrt(d) : NaN; }

  function mount(){
    const root = document.getElementById('extremal-host');
    if (!root) return;
    root.innerHTML = `
      <div class="ac-grid" style="grid-template-columns: 320px 1fr; gap:28px;">
        <div class="ac-knobs">
          <div class="ac-knobs-title">RN family</div>
          <div class="ac-sub">f = 1 − 2M/r + Q²/r². Slide Q→M to drive the two horizons together.</div>

          <div class="ac-slider"><label>M <span id="ex-Mv">1.00</span></label>
            <input id="ex-M" type="range" min="0.5" max="2" step="0.01" value="1"></div>
          <div class="ac-slider"><label>Q/M <span id="ex-qv">0.95</span></label>
            <input id="ex-q" type="range" min="0" max="1" step="0.005" value="0.95"></div>
          <div class="ac-slider"><label>ε = (r − r₊)/r₊ <span id="ex-ev">0.050</span></label>
            <input id="ex-e" type="range" min="0.001" max="0.5" step="0.001" value="0.05"></div>

          <div class="ac-knobs-title" style="margin-top:14px">Presets</div>
          <div id="ex-presets" style="display:flex;flex-wrap:wrap;gap:6px"></div>
        </div>

        <div>
          <div class="ac-verdict" id="ex-verdict">—</div>
          <div class="ac-stats" id="ex-stats" style="margin-top:14px"></div>
          <canvas id="ex-canvas" width="900" height="320" style="margin-top:16px;width:100%;border:1px solid var(--line);border-radius:6px;background:#fff"></canvas>
          <div class="ac-stats" style="margin-top:12px;color:var(--muted);font-size:0.85em;line-height:1.55">
            <b>Reading.</b> RN is <i>not</i> vacuum: it is sourced by the EM stress-energy, so R_μν ≠ 0 outside the horizon.
            On this family <code>f = 1 − 2M/r + Q²/r²</code>, the K-pair is K₁ ≡ 1 (algebraic identity, Lemma 5.3(b)) but K₂ = 1 − Q²/r² ≠ 1.
            So RN sits <i>off</i> the K₁=K₂=1 vertex of Theorem 5.2: the K₁=1 leg holds, the K₂=1 leg fails — that is exactly why R_tt = (f/r²)(K₁−K₂) ≠ 0.
            <br>Driving Q → M pinches the two horizons together at r₊ = M; there K₂(r₊) = 1 − Q²/r₊² → 0 (while K₁ stays at 1), and the near-horizon geometry approaches AdS₂ × S².
          </div>
        </div>
      </div>`;

    const presets = [
      { name:'Schwarzschild',     M:1, q:0    },
      { name:'Mild RN (q=0.5)',   M:1, q:0.5  },
      { name:'Strong RN (q=0.9)', M:1, q:0.9  },
      { name:'Near-extremal',     M:1, q:0.99 },
      { name:'Extremal',          M:1, q:1.00 },
    ];
    const pBar = document.getElementById('ex-presets');
    presets.forEach(p=>{
      const b = document.createElement('button');
      b.className='ac-btn';
      b.textContent = p.name;
      b.onclick = ()=>{
        document.getElementById('ex-M').value = p.M;
        document.getElementById('ex-q').value = p.q;
        ['ex-M','ex-q'].forEach(id=>document.getElementById(id).dispatchEvent(new Event('input',{bubbles:true})));
      };
      pBar.appendChild(b);
    });

    function update(){
      const M = +document.getElementById('ex-M').value;
      const q = +document.getElementById('ex-q').value;
      const eps = +document.getElementById('ex-e').value;
      const Q = q*M;
      document.getElementById('ex-Mv').textContent = M.toFixed(2);
      document.getElementById('ex-qv').textContent = q.toFixed(3);
      document.getElementById('ex-ev').textContent = eps.toFixed(3);

      const rh = rPlus(M,Q);
      const isExtremal = Math.abs(q-1) < 1e-6;

      const r = rh*(1+eps);
      const {f, fp, fpp} = KK.mk_f({C1: 2*M, C2: Q*Q, C3: 0});
      const fv = f(r), fpv = fp(r), fppv = fpp(r);
      const K1 = KK.K1(r, fv, fpv, fppv);
      const K2 = KK.K2(r, fv, fpv);

      const E = KK.Ricci(r, fv, fpv, fppv);
      const ricciNorm = Math.max(Math.abs(E.Rtt), Math.abs(E.Rrr), Math.abs(E.Rθθ));

      const dist11 = Math.hypot(K1-1, K2-1);
      const productKK = K1*K2;

      const stats = document.getElementById('ex-stats');
      stats.innerHTML = `
        <div class="ac-stat"><code>r₊</code></div><div class="ac-stat mono">${fmt(rh)} ${isExtremal?'<span style="color:var(--accent)">·· double root ··</span>':''}</div>
        <div class="ac-stat"><code>r = r₊(1+ε)</code></div><div class="ac-stat mono">${fmt(r)}</div>
        <div class="ac-stat"><code>f(r)</code></div><div class="ac-stat mono">${fmt(fv)}</div>
        <div class="ac-stat"><code>K₁(r)</code></div><div class="ac-stat mono">${fmt(K1)} <span style="color:var(--muted)">(≡ 1 on RN family — identity, not equation of motion)</span></div>
        <div class="ac-stat"><code>K₂(r)</code></div><div class="ac-stat mono">${fmt(K2)} <span style="color:var(--muted)">(= 1 − Q²/r²; → 0 as Q → M, r → r₊)</span></div>
        <div class="ac-stat"><code>K₁ − K₂</code> (controls R_tt)</div><div class="ac-stat mono">${fmt(K1-K2)}</div>
        <div class="ac-stat"><code>R_tt</code> = (f/r²)(K₁−K₂)</div><div class="ac-stat mono">${fmt(E.Rtt)}</div>
        <div class="ac-stat"><code>R_θθ</code> = 1 − K₂</div><div class="ac-stat mono">${fmt(E.Rθθ)}</div>
        <div class="ac-stat"><code>‖R_μν‖∞</code></div><div class="ac-stat mono">${fmt(ricciNorm)}</div>
      `;

      const v = document.getElementById('ex-verdict');
      const ricciOK = ricciNorm < 1e-9;
      const K1eq1 = Math.abs(K1 - 1) < 1e-9;
      if (!Number.isFinite(rh)){
        v.className='ac-verdict verd-fail';
        v.innerHTML = `<b>Naked singularity (Q&gt;M).</b> No horizon — the K-flow has no near-horizon regime to probe.`;
      } else if (Math.abs(Q) < 1e-9){
        v.className='ac-verdict verd-pass';
        v.innerHTML = `<b>Schwarzschild (Q=0).</b> Vacuum: ‖R_μν‖ = ${fmt(ricciNorm)}, K₁ = ${fmt(K1)}, K₂ = ${fmt(K2)} — Theorem 5.2 vertex (K₁=K₂=1).`;
      } else if (K1eq1){
        const K2_target = 1 - (Q*Q)/(r*r);
        v.className='ac-verdict verd-warn';
        if (isExtremal && eps < 0.02){
          v.innerHTML = `<b>Near-extremal RN throat.</b> K₁ = 1 holds (algebraic identity for f = 1 − 2M/r + Q²/r²), but K₂ = ${fmt(K2)} ≈ 1 − Q²/r² = ${fmt(K2_target)}. ` +
                        `R_μν is <i>not</i> zero — RN is sourced by the EM stress-energy. As ε→0 the horizon pinches K₂ → 1 − Q²/r₊² = 0; geometry approaches AdS₂ × S².`;
        } else {
          v.innerHTML = `<b>RN exterior (charged, sourced).</b> K₁ = ${fmt(K1)} (identically 1 on RN), K₂ = ${fmt(K2)} = 1 − Q²/r². ` +
                        `Since K₁ ≠ K₂, R_tt = (f/r²)(K₁−K₂) = ${fmt(E.Rtt)} ≠ 0 — RN does <i>not</i> satisfy the vacuum equations.  ` +
                        `Theorem 5.2 (R_μν = 0 ⇔ K₁ = K₂ = 1) excludes RN: the K=1 leg holds, the K₂=1 leg fails.`;
        }
      } else {
        v.className='ac-verdict verd-fail';
        v.innerHTML = `<b>Off-RN family.</b> Numerical check failed: K₁ = ${fmt(K1)} should be 1 on this ansatz. (Likely a derivative or evaluation artefact at small ε.)`;
      }

      const c = document.getElementById('ex-canvas'), ctx = c.getContext('2d');
      ctx.clearRect(0,0,c.width,c.height);
      const margin = {l:60, r:30, t:18, b:38};
      const W = c.width - margin.l - margin.r, H = c.height - margin.t - margin.b;
      const rMin = rh*(1+1e-3), rMax = rh*3;
      const N = 300;
      const samples = [];
      for (let i=0;i<N;i++){
        const t = i/(N-1);
        const ri = rMin*Math.pow(rMax/rMin, t);
        const fi = f(ri), fpi = fp(ri), fppi = fpp(ri);
        if (fi<=0) continue;
        samples.push({r:ri, K1:KK.K1(ri,fi,fpi,fppi), K2:KK.K2(ri,fi,fpi)});
      }
      let yMin = -0.2, yMax = 1.4;
      samples.forEach(s=>{ yMin = Math.min(yMin, s.K1, s.K2); yMax = Math.max(yMax, s.K1, s.K2); });
      yMin = Math.max(yMin, -1); yMax = Math.min(yMax, 2.5);

      const xOf = (r)=> margin.l + W * Math.log(r/rMin)/Math.log(rMax/rMin);
      const yOf = (y)=> margin.t + H*(1 - (y-yMin)/(yMax-yMin));

      ctx.strokeStyle='#cbd5e1'; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(margin.l, margin.t); ctx.lineTo(margin.l, margin.t+H); ctx.lineTo(margin.l+W, margin.t+H); ctx.stroke();
      ctx.strokeStyle='#94a3b8'; ctx.setLineDash([4,4]);
      ctx.beginPath(); ctx.moveTo(margin.l, yOf(1)); ctx.lineTo(margin.l+W, yOf(1)); ctx.stroke();
      ctx.strokeStyle='#e2e8f0';
      ctx.beginPath(); ctx.moveTo(margin.l, yOf(0)); ctx.lineTo(margin.l+W, yOf(0)); ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle='#475569'; ctx.font='12px ui-sans-serif,system-ui';
      ctx.fillText('K = 1', margin.l+W-46, yOf(1)-4);
      ctx.fillText('K₁(r), K₂(r) vs log r/r₊', margin.l+8, margin.t+12);
      ctx.fillText('r/r₊ →', margin.l+W-50, margin.t+H+22);
      [1,1.5,2,2.5,3].forEach(t=>{
        const xx = xOf(rh*t);
        ctx.strokeStyle='#cbd5e1';
        ctx.beginPath(); ctx.moveTo(xx, margin.t+H); ctx.lineTo(xx, margin.t+H+4); ctx.stroke();
        ctx.fillStyle='#64748b'; ctx.fillText(t.toString(), xx-5, margin.t+H+18);
      });
      [yMin, 0, 1, yMax].forEach(yv=>{
        if (yv===null) return;
        const yy = yOf(yv);
        ctx.strokeStyle='#cbd5e1';
        ctx.beginPath(); ctx.moveTo(margin.l-4, yy); ctx.lineTo(margin.l, yy); ctx.stroke();
        ctx.fillStyle='#64748b'; ctx.fillText(yv.toFixed(1), margin.l-32, yy+4);
      });

      ctx.strokeStyle='#dc2626'; ctx.lineWidth=2; ctx.beginPath();
      samples.forEach((s,i)=>{ const x=xOf(s.r), y=yOf(s.K1); if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y); });
      ctx.stroke();
      ctx.strokeStyle='#2563eb'; ctx.lineWidth=2; ctx.beginPath();
      samples.forEach((s,i)=>{ const x=xOf(s.r), y=yOf(s.K2); if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y); });
      ctx.stroke();

      const xc = xOf(r);
      ctx.strokeStyle='#0f172a'; ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.moveTo(xc, margin.t); ctx.lineTo(xc, margin.t+H); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle='#dc2626'; ctx.beginPath(); ctx.arc(xc, yOf(K1), 4, 0, 2*Math.PI); ctx.fill();
      ctx.fillStyle='#2563eb'; ctx.beginPath(); ctx.arc(xc, yOf(K2), 4, 0, 2*Math.PI); ctx.fill();

      ctx.fillStyle='#dc2626'; ctx.fillRect(margin.l+W-130, margin.t+24, 12, 3);
      ctx.fillStyle='#0f172a'; ctx.fillText('K₁(r)', margin.l+W-112, margin.t+28);
      ctx.fillStyle='#2563eb'; ctx.fillRect(margin.l+W-130, margin.t+42, 12, 3);
      ctx.fillStyle='#0f172a'; ctx.fillText('K₂(r)', margin.l+W-112, margin.t+46);
    }

    root.querySelectorAll('input').forEach(el=>el.addEventListener('input', update));
    update();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount);
  else mount();
})();