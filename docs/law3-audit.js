(function(){
  const C = document.getElementById('law3audit');
  if(!C || C.dataset.bound) return;
  C.dataset.bound = '1';

  const ctx = C.getContext('2d');
  const det = document.getElementById('a1-det');
  const damp = document.getElementById('a1-d');
  const detv = document.getElementById('a1-detv');
  const dv = document.getElementById('a1-dv');
  const basev = document.getElementById('a1-basev');
  const linv = document.getElementById('a1-linv');
  const kappa = document.getElementById('a1-kappa');
  const usable = document.getElementById('a1-usable');
  const verdict = document.getElementById('a1-verdict');
  if(!ctx || !det || !damp || !detv || !dv || !basev || !linv || !kappa || !usable || !verdict) return;

  const root = C.closest('.card') || document;
  const baseBtns = [...root.querySelectorAll('[data-b]')];
  const linBtns = [...root.querySelectorAll('[data-l]')];

  let base = 'eig';
  let lin = 'loc';

  function draw(detNum, dRatio, coeff, canUse){
    const dpr = window.devicePixelRatio || 1;
    const w = C.clientWidth || 900;
    const h = C.clientHeight || 160;
    C.width = w * dpr;
    C.height = h * dpr;
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,w,h);

    const pad = {l:30,r:18,t:18,b:22};
    const innerW = w - pad.l - pad.r;
    const innerH = h - pad.t - pad.b;

    ctx.fillStyle = '#f8f6ef';
    ctx.fillRect(0,0,w,h);
    ctx.strokeStyle = 'rgba(0,0,0,.12)';
    ctx.strokeRect(pad.l,pad.t,innerW,innerH);

    const detNorm = Math.min(1, Math.abs(detNum)/3);
    const ddNorm = Math.max(0, Math.min(1.6, dRatio))/1.6;

    const bars = [
      {label:'det<0', v:detNorm, ok:detNum < 0, color:'oklch(0.52 0.15 268)'},
      {label:'d/dc', v:ddNorm, ok:Math.abs(dRatio - 1) < 0.06, color:'oklch(0.66 0.14 55)'},
      {label:'base', v:base === 'eig' ? 1 : 0.55, ok:base === 'eig', color:'oklch(0.58 0.12 210)'},
      {label:'local', v:lin === 'loc' ? 1 : 0.55, ok:lin === 'loc', color:'oklch(0.58 0.12 150)'}
    ];

    const gap = 22;
    const bw = (innerW - gap * (bars.length - 1)) / bars.length;
    bars.forEach((b,i)=>{
      const x = pad.l + i * (bw + gap);
      const bh = innerH * b.v;
      const y = pad.t + innerH - bh;
      ctx.fillStyle = b.ok ? b.color : 'rgba(120,120,130,.35)';
      ctx.fillRect(x, y, bw, bh);
      ctx.strokeStyle = 'rgba(0,0,0,.18)';
      ctx.strokeRect(x, pad.t, bw, innerH);
      ctx.fillStyle = '#4a5058';
      ctx.font = '11px JetBrains Mono';
      ctx.fillText(b.label, x + 4, h - 7);
    });

    const meterX = pad.l + innerW - 74;
    ctx.strokeStyle = 'rgba(0,0,0,.18)';
    ctx.strokeRect(meterX, pad.t, 56, innerH);
    const markerY = pad.t + innerH - Math.max(0, Math.min(1.2, coeff/6))/1.2 * innerH;
    ctx.strokeStyle = canUse ? 'oklch(0.58 0.12 150)' : 'oklch(0.66 0.14 55)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(meterX - 6, markerY);
    ctx.lineTo(meterX + 62, markerY);
    ctx.stroke();
    ctx.fillStyle = '#4a5058';
    ctx.fillText('4dc', meterX + 14, markerY - 6);
  }

  function update(){
    const detNum = +det.value;
    const dRatio = +damp.value;
    const dc = detNum < 0 ? 1/Math.sqrt(Math.abs(detNum)) : 0;
    const coeff = 4 * dc;
    const canUse = detNum < 0 && Math.abs(dRatio - 1) < 0.06 && base === 'eig' && lin === 'loc';

    detv.textContent = detNum.toFixed(2);
    dv.textContent = dRatio.toFixed(2);
    basev.textContent = base === 'eig' ? 'eigenvector' : 'generic point';
    linv.textContent = lin === 'loc' ? 'local' : 'far / nonlinear';
    kappa.textContent = coeff.toFixed(3);
    usable.textContent = canUse ? 'yes' : 'premise off';

    baseBtns.forEach(btn => btn.classList.toggle('on', btn.dataset.b === base));
    linBtns.forEach(btn => btn.classList.toggle('on', btn.dataset.l === lin));

    if(detNum >= 0){
      verdict.className = 'ac-verdict ac-fail';
      verdict.innerHTML = '<b>Outside theorem domain.</b> The restoring-law coefficient is only interpreted from the Lorentzian side <span class="mono">det G &lt; 0</span>.';
    } else if(Math.abs(dRatio - 1) >= 0.06){
      verdict.className = 'ac-verdict';
      verdict.innerHTML = '<b>Critical damping not met.</b> The expression <span class="mono">4 d_c</span> is still displayed, but the paper only licenses it as the local restoring coefficient at <span class="mono">d = d_c</span>.';
    } else if(base !== 'eig'){
      verdict.className = 'ac-verdict';
      verdict.innerHTML = '<b>Wrong base point.</b> Away from a <span class="mono">G</span>-eigenvector on <span class="mono">{K=1}</span>, the theorem-level coefficient is not the canonical local restoring rate.';
    } else if(lin !== 'loc'){
      verdict.className = 'ac-verdict';
      verdict.innerHTML = '<b>Beyond local linearization.</b> The page is now in heuristic territory: the theorem gives a local coefficient, not a global attractor law.';
    } else {
      verdict.className = 'ac-verdict ac-pass';
      verdict.innerHTML = 'All theorem premises are aligned. The local restoring coefficient shown in the paper, <span class="mono">κ_K = 4 d_c</span>, is the correct leading-order reading here.';
    }

    draw(detNum, dRatio, coeff, canUse);

    if(window.MathJax && window.MathJax.typesetPromise){
      window.MathJax.typesetPromise([verdict]).catch(()=>{});
    }
  }

  baseBtns.forEach(btn => btn.addEventListener('click', ()=>{ base = btn.dataset.b || 'eig'; update(); }));
  linBtns.forEach(btn => btn.addEventListener('click', ()=>{ lin = btn.dataset.l || 'loc'; update(); }));
  det.addEventListener('input', update);
  damp.addEventListener('input', update);
  window.addEventListener('resize', update);
  update();
})();
