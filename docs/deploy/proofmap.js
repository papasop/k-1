/* ============================================================
   § Proof map — Theorem 5.2 (Vacuum equivalence)
   K₁ = K₂ = 1   ⇔   R_μν = 0  (static spherical, Schwarzschild gauge)

   Live numerical audit of the proof's individual steps:
     • Gating by assumptions (f ∈ C², f > 0, spherical, static, r > 0)
     • 5 identity residuals (analytic vs alternative computation)
     • Forward (K=1 ⇒ R=0) and Reverse (R=0 ⇒ K=1) directions
     • Failure gallery (assumption-violating presets)
   ============================================================ */

(function(){
  'use strict';
  const $ = (id)=>document.getElementById(id);

  const TOL_ID  = 1e-6;   // identity tolerance (FD introduces ~1e-7..1e-6)
  const TOL_EQ  = 1e-9;   // exact equality tolerance

  // ---------- fmt ----------
  function fmt(x, d){
    d = d==null?4:d;
    if (!Number.isFinite(x)) return '—';
    const a = Math.abs(x);
    if (a !== 0 && (a < 1e-3 || a >= 1e4)) return x.toExponential(Math.max(1,d-2));
    return x.toFixed(d);
  }

  // ---------- finite-difference derivatives ----------
  function d1(f, r, h){
    h = h || Math.max(1e-5, Math.abs(r)*1e-5);
    return (-f(r+2*h) + 8*f(r+h) - 8*f(r-h) + f(r-2*h)) / (12*h);
  }
  function d2(f, r, h){
    h = h || Math.max(1e-4, Math.abs(r)*1e-4);
    return (-f(r+2*h) + 16*f(r+h) - 30*f(r) + 16*f(r-h) - f(r-2*h)) / (12*h*h);
  }

  // ---------- model: ansatz-driven f(r) ----------
  // Ansatz menu (each is a one-parameter family on top of Schwarzschild's −C1/r):
  //   schw    : f = 1 − C1/r                     (vacuum, K=1, R=0)
  //   rn      : f = 1 − C1/r − C2/r²             (RN-like; non-vacuum unless C2=0)
  //   linear  : f = 1 − C1/r + C3·r              (linear source term; non-vacuum)
  //   dS      : f = 1 − C1/r − (Λ/3)·r²          (Schwarzschild–de Sitter; non-vacuum unless Λ=0)
  // The Schwarzschild ansatz gives a 1-param family inside the R=0 set, so the
  // reverse direction (R=0 ⇒ K=1) actually has content. The other three are
  // designed counter-examples — they parametrise R≠0 directions in metric space.
  function mk_f(state){
    const a = state.ansatz || 'rn';
    const C1 = +state.C1;
    if (a === 'schw'){
      const f   = (r)=> 1 - C1/r;
      const fp  = (r)=> C1/(r*r);
      const fpp = (r)=> -2*C1/(r*r*r);
      return { f, fp, fpp };
    }
    if (a === 'linear'){
      const C3 = +state.C3;
      const f   = (r)=> 1 - C1/r + C3*r;
      const fp  = (r)=> C1/(r*r) + C3;
      const fpp = (r)=> -2*C1/(r*r*r);
      return { f, fp, fpp };
    }
    if (a === 'dS'){
      const Λ = +state.Lambda;
      const f   = (r)=> 1 - C1/r - (Λ/3)*r*r;
      const fp  = (r)=> C1/(r*r) - (2*Λ/3)*r;
      const fpp = (r)=> -2*C1/(r*r*r) - 2*Λ/3;
      return { f, fp, fpp };
    }
    // rn (default): includes C2=0 ⇒ Schwarzschild as a special case
    const C2 = +state.C2;
    const f   = (r)=> 1 - C1/r - C2/(r*r);
    const fp  = (r)=> C1/(r*r) + 2*C2/(r*r*r);
    const fpp = (r)=> -2*C1/(r*r*r) - 6*C2/(r*r*r*r);
    return { f, fp, fpp };
  }

  // Pretty label for the current ansatz (used in verdict copy)
  function ansatzLabel(state){
    const a = state.ansatz || 'rn';
    if (a === 'schw')   return 'f = 1 − C₁/r';
    if (a === 'linear') return 'f = 1 − C₁/r + C₃·r';
    if (a === 'dS')     return 'f = 1 − C₁/r − (Λ/3)·r²';
    return 'f = 1 − C₁/r − C₂/r²';
  }

  // Analytic K's
  function K1_an(r, f, fp, fpp){ return f(r) + 2*r*fp(r) + 0.5*r*r*fpp(r); }
  function K2_an(r, f, fp){      return f(r) + r*fp(r); }

  // Analytic Ricci (for ds² = -f dt² + f⁻¹ dr² + r² dΩ²)
  //   R_tt    = (f/2) f'' + (f f')/r
  //   R_rr    = -R_tt / f²
  //   R_θθ    = 1 - f - r f'
  //   R (sc.) = -f'' - 4 f'/r - 2(f-1)/r²
  function Ricci_an(r, f, fp, fpp){
    const fv = f(r), fpv = fp(r), fppv = fpp(r);
    const Rtt = 0.5*fv*fppv + fv*fpv/r;
    const Rrr = -Rtt / (fv*fv);
    const Rθθ = 1 - fv - r*fpv;
    const R_scalar = -fppv - 4*fpv/r - 2*(fv-1)/(r*r);
    return { Rtt, Rrr, Rθθ, R_scalar };
  }

  // ---------- gating ----------
  function checkAssumptions(state){
    const { r } = state;
    const { f } = mk_f(state);
    const checks = [
      { id:'C2',     label:'f ∈ C²',                  ok:true,            note:'analytic on r > 0' },
      { id:'fpos',   label:'f(r) > 0',                ok:f(r) > 0,        note:`f(${r.toFixed(2)}) = ${fmt(f(r),4)}` },
      { id:'sph',    label:'spherical symmetry',      ok:state.spherical!==false, note:'ds² = -f dt² + f⁻¹ dr² + r² dΩ²' },
      { id:'stat',   label:'static (∂_t = 0)',        ok:true,            note:'f depends on r only' },
      { id:'rpos',   label:'r > 0 (no central sing.)',ok:r > 0,           note:`r = ${r.toFixed(2)}` },
    ];
    const allOk = checks.every(c=>c.ok);
    return { checks, allOk };
  }

  function renderGates(host, checks, allOk){
    host.innerHTML = '';
    for (const c of checks){
      const el = document.createElement('div');
      el.className = 'pm-gate ' + (c.ok ? 'pm-gate-on' : 'pm-gate-off');
      el.innerHTML = `<span class="pm-gate-dot"></span><span class="pm-gate-label">${c.label}</span><span class="pm-gate-note">${c.note}</span>`;
      host.appendChild(el);
    }
    const banner = document.getElementById('pm-applicable');
    if (banner){
      if (allOk){
        banner.textContent = '✓ all assumptions hold — Theorem 5.2 is applicable here';
        banner.className = 'pm-banner pm-banner-ok';
      } else {
        banner.textContent = '⚠ theorem not applicable at current settings — one or more assumptions violated (this is not a counter-example)';
        banner.className = 'pm-banner pm-banner-warn';
      }
    }
    return allOk;
  }

  // ---------- identities ----------
  function evalIdentities(state, r){
    const { f, fp, fpp } = mk_f(state);

    // I1: K₁ (analytic) vs K₁ from FD of f
    const K1_a = K1_an(r, f, fp, fpp);
    const K1_fd = f(r) + 2*r*d1(f,r) + 0.5*r*r*d2(f,r);

    // I2: K₂ analytic vs FD
    const K2_a = K2_an(r, f, fp);
    const K2_fd = f(r) + r*d1(f,r);

    // I3: R_tt vs (f/r²)(K₁-K₂)   — this is the key identity
    const { Rtt, Rrr, Rθθ, R_scalar } = Ricci_an(r, f, fp, fpp);
    const Rtt_K = (f(r) / (r*r)) * (K1_a - K2_a);

    // I4: R_θθ vs 1 - K₂
    const Rθθ_K = 1 - K2_a;

    // I5: R (scalar) vs 2(1-K₁)/r²    — true for these metrics? Let's check.
    //   R = -f'' - 4f'/r - 2(f-1)/r²
    //   2(1-K₁)/r² = 2/r² · (1 - f - 2rf' - r²f''/2)
    //              = 2/r² - 2f/r² - 4f'/r - f''
    //              = -f'' - 4f'/r + 2(1-f)/r²     ✓ identical
    const R_K = 2*(1 - K1_a) / (r*r);

    return {
      I1: { lhs:K1_a, rhs:K1_fd, resid:Math.abs(K1_a-K1_fd), tol:TOL_ID,
            label:'I₁', expr:'K₁ = f + 2r·f′ + (r²/2)·f″',
            desc:'Analytic K₁ matches K₁ rebuilt from finite-difference derivatives of f.' },
      I2: { lhs:K2_a, rhs:K2_fd, resid:Math.abs(K2_a-K2_fd), tol:TOL_ID,
            label:'I₂', expr:'K₂ = f + r·f′',
            desc:'Analytic K₂ matches K₂ from FD.' },
      I3: { lhs:Rtt, rhs:Rtt_K, resid:Math.abs(Rtt-Rtt_K), tol:TOL_EQ,
            label:'I₃', expr:'R_tt = (f / r²) · (K₁ − K₂)',
            desc:'Direct R_tt equals the K-form — the bridge identity.' },
      I4: { lhs:Rθθ, rhs:Rθθ_K, resid:Math.abs(Rθθ-Rθθ_K), tol:TOL_EQ,
            label:'I₄', expr:'R_θθ = 1 − K₂',
            desc:'Angular Ricci component is exactly 1 − K₂.' },
      I5: { lhs:R_scalar, rhs:R_K, resid:Math.abs(R_scalar-R_K), tol:TOL_EQ,
            label:'I₅', expr:'R = 2 · (1 − K₁) / r²',
            desc:'Ricci scalar equals 2(1−K₁)/r² — vanishes iff K₁ = 1.' },
      Ricci: { Rtt, Rrr, Rθθ, R_scalar },
      Ks:    { K1:K1_a, K2:K2_a },
    };
  }

  function renderIdentities(host, ids){
    host.innerHTML = '';
    const order = ['I1','I2','I3','I4','I5'];
    for (const k of order){
      const e = ids[k];
      const ok = e.resid < e.tol;
      const row = document.createElement('div');
      row.className = 'pm-id ' + (ok ? 'pm-id-ok' : 'pm-id-bad');
      row.innerHTML = `
        <div class="pm-id-head">
          <span class="pm-id-label">${e.label}</span>
          <span class="pm-id-expr">${e.expr}</span>
          <span class="pm-id-status">${ok ? '✓' : '✗'}</span>
        </div>
        <div class="pm-id-body">
          <div class="pm-id-desc">${e.desc}</div>
          <div class="pm-id-nums">
            <span>LHS = <b>${fmt(e.lhs,6)}</b></span>
            <span>RHS = <b>${fmt(e.rhs,6)}</b></span>
            <span>residual = <b>${fmt(e.resid,3)}</b></span>
            <span>tol = <b>${e.tol.toExponential(0)}</b></span>
          </div>
        </div>`;
      host.appendChild(row);
    }
  }

  // ---------- r-interval scan ----------
  // Sweeps r ∈ [r1, r2] (N samples) and returns max |K1−1|, max |K2−1|,
  // max |Rtt|, max |Rθθ|, max |R_scalar|. f>0 cells are skipped (so the
  // scan stays well-defined past horizons / coordinate singularities).
  function scanFR(state, r1, r2, N){
    N = N || 64;
    const { f, fp, fpp } = mk_f(state);
    let mxK1 = 0, mxK2 = 0, mxRtt = 0, mxRθθ = 0, mxR = 0;
    let kept = 0;
    for (let i=0; i<N; i++){
      const r = r1 + (r2 - r1) * (i / (N - 1));
      if (!(r > 0)) continue;
      const fv = f(r);
      if (!(fv > 0)) continue;            // skip past horizons / out-of-domain
      const fpv = fp(r), fppv = fpp(r);
      const K1 = K1_an(r, f, fp, fpp);
      const K2 = K2_an(r, f, fp);
      const Rtt = 0.5*fv*fppv + fv*fpv/r;
      const Rθθ = 1 - fv - r*fpv;
      const Rsc = -fppv - 4*fpv/r - 2*(fv-1)/(r*r);
      mxK1 = Math.max(mxK1, Math.abs(K1 - 1));
      mxK2 = Math.max(mxK2, Math.abs(K2 - 1));
      mxRtt = Math.max(mxRtt, Math.abs(Rtt));
      mxRθθ = Math.max(mxRθθ, Math.abs(Rθθ));
      mxR  = Math.max(mxR,  Math.abs(Rsc));
      kept++;
    }
    return { mxK1, mxK2, mxRtt, mxRθθ, mxR, kept, N, r1, r2 };
  }

  // ---------- forward / reverse equivalence (interval-aware) ----------
  function renderEquivalence(fwdHost, revHost, state){
    const r1 = state.rscan1, r2 = state.rscan2;
    const sc = scanFR(state, r1, r2, 96);
    const Kflat = Math.max(sc.mxK1, sc.mxK2);
    const Rflat = Math.max(sc.mxRtt, sc.mxRθθ, sc.mxR);
    const Kclose = Kflat < TOL_ID;
    const Rzero  = Rflat < TOL_ID;
    const ans    = ansatzLabel(state);
    const range  = `r ∈ [${r1.toFixed(2)}, ${r2.toFixed(2)}], ${sc.kept}/${sc.N} samples in domain`;

    // Forward: (K=1 on interval) ⇒ (R=0 on interval)
    //   Verdict copy is *honest*: we only claim "consistent" on the swept interval
    //   under the chosen ansatz. The theorem itself is much stronger; the panel
    //   is a sanity check, not a proof.
    fwdHost.innerHTML = `
      <div class="pm-dir-head">Forward &nbsp;<span>K₁ = K₂ = 1 &nbsp;⇒&nbsp; R_μν = 0</span></div>
      <div class="pm-dir-sub">ansatz · ${ans} &nbsp;·&nbsp; ${range}</div>
      <div class="pm-dir-check">premise &nbsp;K₁ = K₂ = 1 on interval?  <b>${Kclose?'✓':'✗'}</b>
        &nbsp;·&nbsp; max<sub>r</sub>|K₁−1| = ${fmt(sc.mxK1,3)},
        max<sub>r</sub>|K₂−1| = ${fmt(sc.mxK2,3)}</div>
      <div class="pm-dir-check">conclusion &nbsp;R_μν = 0 on interval?  <b>${Rzero?'✓':'✗'}</b>
        &nbsp;·&nbsp; max<sub>r</sub>|R<sub>tt</sub>| = ${fmt(sc.mxRtt,3)},
        max<sub>r</sub>|R<sub>θθ</sub>| = ${fmt(sc.mxRθθ,3)},
        max<sub>r</sub>|R| = ${fmt(sc.mxR,3)}</div>
      <div class="pm-dir-verdict ${Kclose===Rzero?'ok':'bad'}">
        ${Kclose && Rzero  ? `✓ no contradiction on the sampled interval — premise and conclusion both hold under this ansatz`
        : !Kclose && !Rzero ? `premise fails on the sampled interval (K ≢ 1 here); forward implication is vacuous on this ansatz · this is consistent with the theorem, not a test of it`
        : `✗ premise holds but conclusion fails (or vice versa) on the sampled interval — would falsify the theorem`}
      </div>`;

    // Reverse: (R=0 on interval) ⇒ (K=1 on interval)
    //   The interesting case is the Schwarzschild ansatz, which is a 1-param
    //   family inside the R=0 locus — the reverse direction has actual content
    //   there. The non-vacuum ansatze (rn with C2≠0, linear, dS with Λ≠0)
    //   parametrise R≠0 directions and so make the reverse premise *false*;
    //   the test then becomes a counter-example check on K=1.
    const isCex = !Rzero && Kclose;   // R≠0 but K=1 — would break the theorem
    revHost.innerHTML = `
      <div class="pm-dir-head">Reverse &nbsp;<span>R_μν = 0 &nbsp;⇒&nbsp; K₁ = K₂ = 1</span></div>
      <div class="pm-dir-sub">ansatz · ${ans} &nbsp;·&nbsp; ${range}</div>
      <div class="pm-dir-check">premise &nbsp;R_μν = 0 on interval?  <b>${Rzero?'✓':'✗'}</b>
        &nbsp;·&nbsp; max<sub>r</sub>|R_μν| ≈ ${fmt(Rflat,3)}</div>
      <div class="pm-dir-check">conclusion &nbsp;K₁ = K₂ = 1 on interval?  <b>${Kclose?'✓':'✗'}</b>
        &nbsp;·&nbsp; max<sub>r</sub>|K−1| ≈ ${fmt(Kflat,3)}</div>
      <div class="pm-dir-verdict ${isCex?'bad':'ok'}">
        ${Rzero && Kclose  ? `✓ no contradiction on the sampled interval — both R = 0 and K = 1 under this ansatz (Schwarzschild branch)`
        : !Rzero && !Kclose ? `premise fails on this ansatz (R ≢ 0); reverse implication is vacuous · the panel is checking that K ≠ 1 here too — i.e. no reverse counter-example was hit`
        : isCex ? `✗ would-be counter-example: R ≠ 0 yet K = 1 on the interval — falsifies the reverse direction`
        : `premise holds but conclusion fails — theorem would be broken`}
      </div>`;
  }

  // ---------- main render ----------
  function render(state){
    const gatesHost = $('pm-gates');
    const { checks, allOk } = checkAssumptions(state);
    renderGates(gatesHost, checks, allOk);

    const body = $('pm-body');
    if (!allOk){
      body.classList.add('pm-disabled');
    } else {
      body.classList.remove('pm-disabled');
    }

    // identity + equivalence panels compute regardless (for transparency)
    const ids = evalIdentities(state, state.r);
    renderIdentities($('pm-ids'), ids);
    renderEquivalence($('pm-fwd'), $('pm-rev'), state);

    // summary figures in header
    $('pm-K1').textContent = fmt(ids.Ks.K1, 6);
    $('pm-K2').textContent = fmt(ids.Ks.K2, 6);
    $('pm-Rtt').textContent = fmt(ids.Ricci.Rtt, 4);
    $('pm-Rrr').textContent = fmt(ids.Ricci.Rrr, 4);
    $('pm-Rθθ').textContent = fmt(ids.Ricci.Rθθ, 4);

    // conclusion endpoint — is R_μν = 0 (to tol)?
    const maxR = Math.max(Math.abs(ids.Ricci.Rtt),
                          Math.abs(ids.Ricci.Rrr),
                          Math.abs(ids.Ricci.Rθθ));
    const concMax = $('pm-concl-maxR');
    const concState = $('pm-concl-state');
    if (concMax) concMax.textContent = `max |R_μν| = ${fmt(maxR, 4)}`;
    if (concState){
      if (!allOk){
        concState.innerHTML = '<span style="color:oklch(0.45 0.15 65)">theorem not applicable at current settings</span>';
      } else if (maxR < TOL_EQ){
        concState.innerHTML = '<span style="color:var(--accent)">✓ endpoint reached · R<sub>μν</sub> = 0</span>';
      } else {
        concState.innerHTML = '<span style="color:oklch(0.5 0.2 25)">✗ endpoint not reached · R<sub>μν</sub> ≠ 0 (non-vacuum branch)</span>';
      }
    }
  }

  // ---------- boot ----------
  function boot(){
    const card = $('proofmap');
    if(!card) return;

    const C1s = $('pm-C1'), C2s = $('pm-C2'), rs = $('pm-r'), sph = $('pm-sph');
    const C1v = $('pm-C1v'), C2v = $('pm-C2v'), rv = $('pm-rv');

    // optional new controls — present iff the HTML side has been wired up;
    // panel still works with just C1/C2/r if any of these are missing.
    const C3s = $('pm-C3'), C3v = $('pm-C3v');         // linear-ansatz coeff
    const Λs  = $('pm-Lam'), Λv = $('pm-Lamv');        // dS Λ
    const r1s = $('pm-rscan1'), r1v = $('pm-rscan1v'); // scan range
    const r2s = $('pm-rscan2'), r2v = $('pm-rscan2v');
    const ansatzBtns = card.querySelectorAll('[data-pm-ansatz]');

    let ansatz = 'rn';

    function readState(){
      return {
        ansatz,
        C1: +C1s.value, C2: +C2s.value, r: +rs.value,
        C3:     C3s ? +C3s.value : 0,
        Lambda: Λs  ? +Λs.value  : 0,
        rscan1: r1s ? +r1s.value : Math.max(0.5, +rs.value - 2.0),
        rscan2: r2s ? +r2s.value : (+rs.value + 4.0),
        spherical: sph.checked,
      };
    }
    function syncAnsatzVisibility(){
      // show only the param controls relevant to the active ansatz; C1 is always live
      const visible = {
        rn:     ['pm-row-C2'],
        schw:   [],
        linear: ['pm-row-C3'],
        dS:     ['pm-row-Lam'],
      }[ansatz] || [];
      ['pm-row-C2','pm-row-C3','pm-row-Lam'].forEach(id=>{
        const el = document.getElementById(id);
        if (el) el.style.display = visible.includes(id) ? '' : 'none';
      });
      ansatzBtns.forEach(b=>b.classList.toggle('on', b.dataset.pmAnsatz === ansatz));
    }
    function tick(){
      const st = readState();
      C1v.textContent = st.C1.toFixed(2);
      C2v.textContent = st.C2.toFixed(2);
      rv.textContent  = st.r.toFixed(2);
      if (C3v) C3v.textContent = st.C3.toFixed(3);
      if (Λv)  Λv.textContent  = st.Lambda.toFixed(3);
      if (r1v) r1v.textContent = st.rscan1.toFixed(2);
      if (r2v) r2v.textContent = st.rscan2.toFixed(2);
      // keep r2 > r1 by a small margin
      if (r1s && r2s && +r2s.value <= +r1s.value + 0.1){
        r2s.value = (+r1s.value + 0.1).toFixed(2);
        if (r2v) r2v.textContent = (+r2s.value).toFixed(2);
      }
      render(readState());
    }
    [C1s, C2s, rs, C3s, Λs, r1s, r2s].forEach(e=>{ if (e) e.oninput = tick; });
    sph.onchange = tick;

    ansatzBtns.forEach(b=>{
      b.onclick = ()=>{
        ansatz = b.dataset.pmAnsatz;
        syncAnsatzVisibility();
        tick();
      };
    });

    // Failure presets — also pin the ansatz so the panel state is unambiguous
    card.querySelectorAll('[data-pm-preset]').forEach(b=>{
      b.onclick = ()=>{
        const p = b.dataset.pmPreset;
        if (p==='vacuum')    { ansatz='rn'; C1s.value=2.0; C2s.value=0.0; rs.value=4.0; sph.checked=true; }
        if (p==='rn')        { ansatz='rn'; C1s.value=2.0; C2s.value=0.6; rs.value=4.0; sph.checked=true; }
        if (p==='horizon')   { ansatz='rn'; C1s.value=2.0; C2s.value=0.0; rs.value=1.5; sph.checked=true; }
        if (p==='nonsph')    { ansatz='rn'; C1s.value=2.0; C2s.value=0.0; rs.value=4.0; sph.checked=false; }
        if (p==='nearsing')  { ansatz='rn'; C1s.value=2.0; C2s.value=0.0; rs.value=0.5; sph.checked=true; }
        if (p==='linear' && C3s){ ansatz='linear'; C1s.value=2.0; C3s.value=0.05; rs.value=4.0; sph.checked=true; }
        if (p==='dS' && Λs)     { ansatz='dS';     C1s.value=2.0; Λs.value=0.05;  rs.value=4.0; sph.checked=true; }
        card.querySelectorAll('[data-pm-preset]').forEach(x=>x.classList.remove('on'));
        b.classList.add('on');
        syncAnsatzVisibility();
        tick();
      };
    });
    // default preset highlight
    const def = card.querySelector('[data-pm-preset="vacuum"]');
    if (def) def.classList.add('on');

    syncAnsatzVisibility();
    tick();
  }
  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', boot);
  else
    boot();
})();
