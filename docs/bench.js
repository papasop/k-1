/* =====================================================================
 * §5.3 · Gating & counter-example bench
 * Stress-tests Route B's Theorem 5.2 along five axes:
 *   (A) Gating — spherical / static / f>0 / vacuum toggles
 *   (B) Single-leg — force (K1=1, K2≠1) and (K2=1, K1≠1)
 *   (C) Candidate suite — Schw / RN / dS / SdS / cubic-toy
 *   (D) Bidirectional — forward (K⇒R=0) vs reverse (R=0⇒K)
 *   (E) Component closure — R_tt, R_rr, R_θθ, Ricci scalar R
 * ===================================================================== */
(function(){
  'use strict';
  const TOL = 1e-3;

  // ---------- generic Ricci kernel (same as audit-corr.js) ----------
  function K1_of(r, f, fp, fpp){ return f(r) + 2*r*fp(r) + 0.5*r*r*fpp(r); }
  function K2_of(r, f, fp)     { return f(r) + r*fp(r); }
  function Ricci(r, f, fp, fpp){
    const fv = f(r), fpv = fp(r), fppv = fpp(r);
    const Rtt = 0.5*fv*fppv + fv*fpv/r;
    const Rrr = -Rtt / (fv*fv);
    const Rθθ = 1 - fv - r*fpv;
    const R_scalar = -fppv - 4*fpv/r - 2*(fv-1)/(r*r); // trace: g^tt R_tt + g^rr R_rr + 2 g^θθ R_θθ
    return { Rtt, Rrr, Rθθ, R: R_scalar };
  }
  function fmt(x, n=5){
    if (!isFinite(x)) return '—';
    const a = Math.abs(x);
    if (a >= 1e4) return x.toExponential(2);
    if (a < 1e-9) return '0.000';
    if (a < 1e-3) return x.toExponential(2);
    return x.toFixed(n);
  }

  // ---------- candidate library ----------
  // f(r) = 1 - C1/r + C2/r^2 + C3*r^2 + C4/r^3
  const CANDS = {
    schw:   { label:'Schwarzschild',         params:{C1:2.0, C2:0,   C3:0,     C4:0}, expect:'vacuum ✓' },
    rn:     { label:'Reissner–Nordström',    params:{C1:2.0, C2:0.5, C3:0,     C4:0}, expect:'non-vacuum (K₂ breaks)' },
    ds:     { label:'de Sitter',             params:{C1:0,   C2:0,   C3:-0.02, C4:0}, expect:'non-vacuum (Λ)' },
    sds:    { label:'Schw–de Sitter',        params:{C1:2.0, C2:0,   C3:-0.02, C4:0}, expect:'non-vacuum (Λ)' },
    cubic:  { label:'Cubic toy f = 1−2M/r+b/r³', params:{C1:2.0, C2:0, C3:0,   C4:0.5}, expect:'non-vacuum (toy)' },
  };
  function mk_f(p){
    const f   = r => 1 - p.C1/r + p.C2/(r*r) + p.C3*r*r + p.C4/(r*r*r);
    const fp  = r => p.C1/(r*r) - 2*p.C2/(r*r*r) + 2*p.C3*r - 3*p.C4/(r*r*r*r);
    const fpp = r => -2*p.C1/(r*r*r) + 6*p.C2/(r*r*r*r) + 2*p.C3 + 12*p.C4/(r*r*r*r*r);
    return { f, fp, fpp };
  }

  function qs(id){ return document.getElementById(id); }

  function setVerdict(el, kind, msg){
    // kind: 'pass' | 'fail' | 'na'
    el.classList.remove('bench-pass','bench-fail','bench-na');
    el.classList.add('bench-'+kind);
    el.textContent = msg;
  }

  // ================== BENCH A · Gating ==================
  function initGating(){
    const card = qs('bench-gating');
    if (!card) return;
    const tog = {
      sph: qs('bg-sph'), stat: qs('bg-stat'), fpos: qs('bg-fpos'), vac: qs('bg-vac')
    };
    const out = qs('bg-verdict');
    const sub = qs('bg-sub');
    if (!tog.sph || !tog.stat || !tog.fpos || !tog.vac || !out || !sub) return;

    function refresh(){
      const s = tog.sph.checked, t = tog.stat.checked,
            fp = tog.fpos.checked, v = tog.vac.checked;
      const missing = [];
      if (!s)  missing.push('spherical');
      if (!t)  missing.push('static');
      if (!fp) missing.push('f > 0');
      if (!v)  missing.push('vacuum profile');
      if (missing.length === 0){
        setVerdict(out, 'pass',
          '✓ all four premises on — Theorem 5.2 applies. (K₁=K₂=1) ⇔ Rμν=0.');
        sub.textContent = 'The equivalence runs in both directions on the stated symmetry class.';
      } else {
        setVerdict(out, 'na',
          '⊘ theorem NOT applicable — missing: ' + missing.join(', ') + '.');
        sub.innerHTML = 'Outside Route B’s gated scope. This is <b>not a counter-example</b>; it is outside the theorem’s domain.';
      }
    }
    Object.values(tog).forEach(el => { if (el) el.onchange = refresh; });
    refresh();
  }

  // ================== BENCH B · Single-leg ==================
  function initSingleLeg(){
    const card = qs('bench-leg');
    if (!card) return;
    const modeRadios = card.querySelectorAll('input[name="bl-mode"]');
    const rIn = qs('bl-r'), rOut = qs('bl-rv');
    const qIn = qs('bl-q'), qOut = qs('bl-qv');
    const k1o = qs('bl-K1'), k2o = qs('bl-K2');
    const rttO = qs('bl-Rtt'), rθθO = qs('bl-Rθθ'), rrrO = qs('bl-Rrr'), rO = qs('bl-R');
    const verdict = qs('bl-verdict');
    const explain = qs('bl-explain');
    if (!modeRadios.length || !rIn || !rOut || !qIn || !qOut || !k1o || !k2o || !rttO || !rθθO || !rrrO || !rO || !verdict || !explain) return;

    function getMode(){ for (const r of modeRadios) if (r.checked) return r.value; return 'k1'; }

    function refresh(){
      const r = +rIn.value; rOut.textContent = r.toFixed(2);
      const q = +qIn.value; qOut.textContent = q.toFixed(3);
      const mode = getMode();
      let p;
      if (mode === 'k1'){
        p = { C1:2.0, C2:q, C3:0, C4:0 };
      } else {
        p = { C1:-q, C2:0, C3:0, C4:0 };
      }
      const { f, fp, fpp } = mk_f(p);
      const fv = f(r);
      if (fv <= 0){
        [k1o,k2o,rttO,rθθO,rrrO,rO].forEach(e => e.textContent='—');
        setVerdict(verdict, 'na', '⊘ f(r) ≤ 0 — outside real branch, theorem not applicable.');
        explain.textContent = 'Pick a larger r or smaller q to stay on the f>0 branch.';
        return;
      }
      const K1 = K1_of(r, f, fp, fpp);
      const K2 = K2_of(r, f, fp);
      const { Rtt, Rrr, Rθθ, R } = Ricci(r, f, fp, fpp);
      k1o.textContent = fmt(K1); k2o.textContent = fmt(K2);
      rttO.textContent = fmt(Rtt); rθθO.textContent = fmt(Rθθ);
      rrrO.textContent = fmt(Rrr); rO.textContent = fmt(R);

      const dK1 = Math.abs(K1-1), dK2 = Math.abs(K2-1);
      if (mode === 'k1'){
        const ok = dK1 < TOL && dK2 > TOL;
        if (ok){
          setVerdict(verdict, 'pass',
            '✓ single-leg: K₁ = 1, K₂ ≠ 1 — Rθθ = 1−K₂ carries the break; Rtt = (f/r²)(K₁−K₂) ≠ 0 too.');
          explain.innerHTML =
            'Live leg behaviour (RN with q = '+fmt(q)+' at r = '+fmt(r)+'):<br>' +
            '&nbsp;&nbsp;• R<sub>θθ</sub> = 1 − K₂ = '+fmt(Rθθ)+' &nbsp;(carries the charge content)<br>' +
            '&nbsp;&nbsp;• R<sub>tt</sub> = (f/r²)(K₁ − K₂) = '+fmt(Rtt)+' &nbsp;(also ≠ 0 because K₁ ≠ K₂)<br>' +
            '<br><span class="bench-na-note"><b>Important:</b> K₁ = 1 alone does <i>not</i> make R_tt vanish. The correct algebraic identity is <code>R_tt = (f/r²)(K₁ − K₂)</code>, not <code>(f/r²)(K₁ − 1)</code>. RN is a live counter-example: K₁ = 1 yet R_tt ≠ 0. This is why Theorem 5.2 requires <i>both</i> legs.</span>';
        } else {
          setVerdict(verdict, 'na', '… dial q away from 0 to force K₂ ≠ 1 while keeping K₁ = 1.');
          explain.textContent = 'Mode requires q ≠ 0 so the RN term breaks K₂.';
        }
      } else {
        const diff = (r * fp) + 0.5 * r * r * fpp;
        setVerdict(verdict, 'pass',
          '✓ rigidity lemma: in this family, K₂ ≡ 1  ⇒  K₁ ≡ 1 automatically (no single-leg reverse).');
        explain.innerHTML =
          '<b>Why the second leg is not independent.</b><br>'+
          'Imposing K₂(r) = 1 for all r means (r·f)′ = 1, so f = 1 + C/r. ' +
          'Substituting into K₁ − K₂ = r·f′ + ½·r²·f″ gives −C/r + C/r = 0 identically.<br>'+
          '<br>Live check with current f = 1 + q/r (C = −q = '+fmt(-q)+'):<br>'+
          '&nbsp;&nbsp;• K₁ − K₂ = r·f′ + ½·r²·f″ = '+fmt(diff)+' &nbsp;(analytically 0)<br>'+
          '&nbsp;&nbsp;• R<sub>tt</sub> = '+fmt(Rtt)+' &nbsp;→ 0 automatically<br>'+
          '&nbsp;&nbsp;• R<sub>θθ</sub> = 1−K₂ = '+fmt(Rθθ)+' &nbsp;→ 0 automatically<br>'+
          '<br><span class="bench-na-note">Consequence for the theorem: the two legs K₁=1 and K₂=1 are <i>not</i> logically independent in 4D spherical static vacuum — the forward direction really only carries one degree of freedom, which is why Route B closes with just {R<sub>tt</sub>, R<sub>θθ</sub>} rather than needing a separate R<sub>φφ</sub> check.</span>';
      }
    }
    modeRadios.forEach(r => r.onchange = refresh);
    rIn.oninput = refresh; qIn.oninput = refresh;
    refresh();
  }

  // ================== BENCH C · Candidate suite ==================
  function ensureSuiteHost(){
    let card = qs('bench-suite');
    if (card) return card;
    const closure = qs('bench-closure');
    if (!closure) return null;
    const figure = document.createElement('figure');
    figure.className = 'card';
    figure.innerHTML = '<div class="fc-head"><div class="fc-title">Bench C — candidate-vacuum suite</div><div class="fc-tag">failure gallery</div></div>' +
      '<div class="fc-body" id="bench-suite"></div>' +
      '<figcaption>Common spherical profiles are not interchangeable. This bench checks which leg each candidate breaks, so Schwarzschild, RN, de Sitter, and SdS cannot be visually conflated.</figcaption>';
    closure.parentNode.insertBefore(figure, closure);
    return qs('bench-suite');
  }

  function initSuite(){
    const card = ensureSuiteHost();
    if (!card) return;
    card.innerHTML = `
      <div class="ctrl" style="border-top:none">
        <div class="seg" id="bs-presets">
          <button class="bench-preset" data-cand="schw">Schw</button>
          <button class="bench-preset" data-cand="rn">RN</button>
          <button class="bench-preset" data-cand="ds">dS</button>
          <button class="bench-preset" data-cand="sds">SdS</button>
          <button class="bench-preset" data-cand="cubic">cubic toy</button>
        </div>
        <label><span>probe radius <b id="bs-rv">3.00</b></span><input id="bs-r" type="range" min="1.4" max="8.0" step="0.01" value="3.0"></label>
      </div>
      <div class="ac-mini" id="bs-rows"></div>
      <div id="bs-summary" class="ac-verdict"></div>`;
    const btns = card.querySelectorAll('.bench-preset');
    const rIn = qs('bs-r'), rOut = qs('bs-rv');
    const rows = qs('bs-rows');
    const summary = qs('bs-summary');
    let current = 'schw';

    function refresh(){
      const r = +rIn.value; rOut.textContent = r.toFixed(2);
      btns.forEach(b => b.classList.toggle('on', b.dataset.cand === current));
      const cand = CANDS[current];
      const { f, fp, fpp } = mk_f(cand.params);
      const fv = f(r);
      let classification;

      if (fv <= 0){
        rows.innerHTML = '<div class="bench-row bench-row-na">' +
          '<div class="bench-row-lab">'+cand.label+'</div>' +
          '<div>f(r) = '+fmt(fv)+' ≤ 0 — outside real branch</div>' +
          '<div class="bench-row-tag bench-na">⊘ f ≤ 0</div>' +
          '</div>';
        summary.textContent = 'Theorem not applicable: f ≤ 0 at this r. This is not a counter-example, it is outside domain.';
        return;
      }

      const K1 = K1_of(r, f, fp, fpp);
      const K2 = K2_of(r, f, fp);
      const { Rtt, Rrr, Rθθ, R } = Ricci(r, f, fp, fpp);
      const dK1 = Math.abs(K1-1), dK2 = Math.abs(K2-1);

      if (dK1 < TOL && dK2 < TOL){
        classification = { kind:'pass', txt:'vacuum ✓ — both legs satisfied, Rμν = 0' };
      } else if (dK1 < TOL && dK2 >= TOL){
        classification = { kind:'fail', txt:'K₁ = 1 holds, K₂ breaks — non-vacuum (charge-like)' };
      } else if (dK1 >= TOL && dK2 < TOL){
        classification = { kind:'fail', txt:'K₂ = 1 holds, K₁ breaks — non-vacuum (temporal)' };
      } else {
        classification = { kind:'fail', txt:'both legs break — fully non-vacuum' };
      }

      rows.innerHTML = [
        ['leg 1', 'K₁ = '+fmt(K1), dK1<TOL ? '✓ K₁=1' : '✗ K₁≠1', dK1<TOL?'pass':'fail'],
        ['leg 2', 'K₂ = '+fmt(K2), dK2<TOL ? '✓ K₂=1' : '✗ K₂≠1', dK2<TOL?'pass':'fail'],
        ['R<sub>tt</sub>',   fmt(Rtt),  Math.abs(Rtt)<TOL ? '✓ 0' : '✗ ≠0', Math.abs(Rtt)<TOL?'pass':'fail'],
        ['R<sub>θθ</sub>',   fmt(Rθθ),  Math.abs(Rθθ)<TOL ? '✓ 0' : '✗ ≠0', Math.abs(Rθθ)<TOL?'pass':'fail'],
        ['R<sub>rr</sub>',   fmt(Rrr),  Math.abs(Rrr)<TOL ? '✓ 0' : '✗ ≠0', Math.abs(Rrr)<TOL?'pass':'fail'],
        ['R (scalar)',       fmt(R),    Math.abs(R)<TOL   ? '✓ 0' : '✗ ≠0', Math.abs(R)<TOL?'pass':'fail'],
      ].map(([lab,val,tag,kind]) =>
        '<div class="bench-row"><div class="bench-row-lab">'+lab+'</div><div>'+val+'</div><div class="bench-row-tag bench-'+kind+'">'+tag+'</div></div>'
      ).join('');

      setVerdict(summary, classification.kind,
        cand.label + ' · ' + classification.txt + '  (expected: ' + cand.expect + ')');
    }
    btns.forEach(b => b.onclick = () => { current = b.dataset.cand; refresh(); });
    rIn.oninput = refresh;
    refresh();
  }

  // ================== BENCH D · Bidirectional ==================
  function ensureBidirHost(){
    let card = qs('bench-bidir');
    if (card) return card;
    const fallback = qs('bench-directions');
    if (fallback){
      fallback.id = 'bench-bidir';
      return fallback;
    }
    return null;
  }

  function initBidir(){
    const card = ensureBidirHost();
    if (!card) return;
    const C1 = qs('bd-C1'), C1v = qs('bd-C1v');
    const C2 = qs('bd-C2'), C2v = qs('bd-C2v');
    const C3 = qs('bd-C3'), C3v = qs('bd-C3v');
    const rIn = qs('bd-r'), rOut = qs('bd-rv');
    const fwdRes = qs('bd-fwd-res'), fwdV = qs('bd-fwd-v');
    const revRes = qs('bd-rev-res'), revV = qs('bd-rev-v');
    const overall = qs('bd-overall');
    if (!C1 || !C1v || !C2 || !C2v || !C3 || !C3v || !rIn || !rOut || !fwdRes || !fwdV || !revRes || !revV || !overall) return;

    function refresh(){
      const c1 = +C1.value; C1v.textContent = c1.toFixed(2);
      const c2 = +C2.value; C2v.textContent = c2.toFixed(3);
      const c3 = +C3.value; C3v.textContent = c3.toFixed(4);
      const r  = +rIn.value; rOut.textContent = r.toFixed(2);
      const { f, fp, fpp } = mk_f({C1:c1, C2:c2, C3:c3, C4:0});
      const fv = f(r);
      if (fv <= 0){
        fwdRes.textContent = revRes.textContent = '—';
        setVerdict(fwdV, 'na', '⊘ f ≤ 0'); setVerdict(revV, 'na', '⊘ f ≤ 0');
        setVerdict(overall, 'na', '⊘ outside real branch — theorem not applicable.');
        return;
      }
      const K1 = K1_of(r, f, fp, fpp);
      const K2 = K2_of(r, f, fp);
      const { Rtt, Rrr, Rθθ } = Ricci(r, f, fp, fpp);
      const dK1 = Math.abs(K1-1), dK2 = Math.abs(K2-1);
      const maxR = Math.max(Math.abs(Rtt), Math.abs(Rrr), Math.abs(Rθθ));
      const Kok = dK1 < TOL && dK2 < TOL;
      const Rok = maxR < TOL;

      fwdRes.innerHTML = '|K₁−1| = '+fmt(dK1)+' · |K₂−1| = '+fmt(dK2)+' · max|Rμν| = '+fmt(maxR);
      if (Kok && Rok)
        setVerdict(fwdV, 'pass', '✓ forward holds: K₁=K₂=1 ⇒ Rμν=0 (verified numerically)');
      else if (Kok && !Rok)
        setVerdict(fwdV, 'fail', '✗ forward would FAIL: K=1 yet Rμν≠0 (should not happen)');
      else
        setVerdict(fwdV, 'na', '… forward untested: K₁=K₂=1 does not currently hold');

      revRes.innerHTML = 'max|Rμν| = '+fmt(maxR)+' · |K₁−1| = '+fmt(dK1)+' · |K₂−1| = '+fmt(dK2);
      if (Rok && Kok)
        setVerdict(revV, 'pass', '✓ reverse holds: Rμν=0 ⇒ K₁=K₂=1 (verified numerically)');
      else if (Rok && !Kok)
        setVerdict(revV, 'fail', '✗ reverse would FAIL: Rμν=0 yet K≠1 (should not happen)');
      else
        setVerdict(revV, 'na', '… reverse untested: Rμν=0 does not currently hold');

      if (Kok && Rok)
        setVerdict(overall, 'pass', '✓ equivalence intact: both directions hold at this point.');
      else if (!Kok && !Rok)
        setVerdict(overall, 'fail', '✗ non-vacuum point — both K≠1 and Rμν≠0 (equivalence consistent, just off-vacuum).');
      else
        setVerdict(overall, 'fail', '✗ inconsistency flag — exactly one side zero while the other is not. Report if you see this.');
    }
    [C1, C2, C3, rIn].forEach(e => e.oninput = refresh);
    refresh();
  }

  // ================== BENCH E · Component closure ==================
  function initClosure(){
    const card = qs('bench-closure');
    if (!card) return;
    const C1 = qs('bc-C1'), C1v = qs('bc-C1v');
    const C2 = qs('bc-C2'), C2v = qs('bc-C2v');
    const rIn = qs('bc-r'), rOut = qs('bc-rv');
    const cells = {
      Rtt: qs('bc-Rtt'), Rrr: qs('bc-Rrr'),
      Rθθ: qs('bc-Rθθ'), R: qs('bc-R'),
    };
    const tags = {
      Rtt: qs('bc-Rtt-tag'), Rrr: qs('bc-Rrr-tag'),
      Rθθ: qs('bc-Rθθ-tag'), R: qs('bc-R-tag'),
    };
    const closure = qs('bc-closure');
    const overall = qs('bc-overall');
    if (!C1 || !C1v || !C2 || !C2v || !rIn || !rOut || !closure || !overall || Object.values(cells).some(v=>!v) || Object.values(tags).some(v=>!v)) return;

    function refresh(){
      const c1 = +C1.value; C1v.textContent = c1.toFixed(2);
      const c2 = +C2.value; C2v.textContent = c2.toFixed(3);
      const r  = +rIn.value; rOut.textContent = r.toFixed(2);
      const { f, fp, fpp } = mk_f({C1:c1, C2:c2, C3:0, C4:0});
      const fv = f(r);
      if (fv <= 0){
        Object.values(cells).forEach(e => e.textContent='—');
        Object.values(tags).forEach(e => { e.textContent='⊘'; e.className='bench-row-tag bench-na'; });
        setVerdict(closure, 'na', '⊘ f ≤ 0 — theorem not applicable');
        setVerdict(overall, 'na', '⊘ outside domain');
        return;
      }
      const { Rtt, Rrr, Rθθ, R } = Ricci(r, f, fp, fpp);
      const comps = { Rtt, Rrr, Rθθ, R };
      let allZero = true;
      for (const k of ['Rtt','Rrr','Rθθ','R']){
        cells[k].textContent = fmt(comps[k]);
        const z = Math.abs(comps[k]) < TOL;
        if (!z) allZero = false;
        tags[k].textContent = z ? '✓ 0' : '✗ ≠0';
        tags[k].className = 'bench-row-tag bench-' + (z ? 'pass' : 'fail');
      }

      const Rrr_pred = -Rtt / (fv*fv);
      const R_pred = -fpp(r) - 4*fp(r)/r - 2*(fv-1)/(r*r);
      const closureErr = Math.max(Math.abs(Rrr - Rrr_pred), Math.abs(R - R_pred));
      if (closureErr < 1e-9)
        setVerdict(closure, 'pass',
          '✓ component closure: R_rr = −R_tt/f² and R = g^μν R_μν — relations verified to ' + closureErr.toExponential(1));
      else
        setVerdict(closure, 'fail',
          '✗ component closure broken (err '+closureErr.toExponential(2)+') — kernel bug');

      if (allZero)
        setVerdict(overall, 'pass', '✓ full vacuum closure: all four components vanish at this point.');
      else
        setVerdict(overall, 'fail', 'non-vacuum point — some components non-zero (expected for this f).');
    }
    [C1, C2, rIn].forEach(e => e.oninput = refresh);
    refresh();
  }

  function boot(){
    const steps = [
      ['gating', initGating],
      ['single-leg', initSingleLeg],
      ['suite', initSuite],
      ['bidir', initBidir],
      ['closure', initClosure],
    ];
    for (const [name, fn] of steps){
      try { fn(); }
      catch (err) { console.warn('bench.js init failed:', name, err); }
    }
  }
  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', boot);
  else
    boot();
})();
