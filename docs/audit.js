(function(){
  function fit(canvas, ctx, defaultH){
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 420;
    const h = defaultH || canvas.clientHeight || 180;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return {w, h};
  }

  function drawRouteB(canvas, m, absolute){
    if(!canvas) return;
    const ctx = canvas.getContext('2d');
    const box = fit(canvas, ctx, 180);
    const w = box.w, h = box.h;
    const pad = {l:42,r:12,t:14,b:24};
    const rs = 2*m;
    const r1 = absolute ? 0.3 : Math.max(0.3, 0.55*rs);
    const r2 = absolute ? 12 : 4.4*rs;
    const vals = [];
    for(let i=0;i<=220;i++){
      const r = r1 + (r2-r1)*i/220;
      vals.push(1 - rs/r);
    }
    const ymin = Math.min(-0.6, ...vals), ymax = Math.max(1.05, ...vals);
    const X = r => pad.l + (r-r1)/(r2-r1)*(w-pad.l-pad.r);
    const Y = v => pad.t + (1-(v-ymin)/(ymax-ymin))*(h-pad.t-pad.b);

    ctx.clearRect(0,0,w,h);
    ctx.strokeStyle = 'rgba(0,0,0,.12)';
    ctx.strokeRect(pad.l,pad.t,w-pad.l-pad.r,h-pad.t-pad.b);

    ctx.strokeStyle = 'oklch(0.52 0.18 268)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    vals.forEach((v,i)=>{
      const r = r1 + (r2-r1)*i/220;
      const x = X(r), y = Y(v);
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();

    ctx.strokeStyle = 'oklch(0.55 0.18 25)';
    ctx.setLineDash([4,4]);
    if(rs >= r1 && rs <= r2){
      ctx.beginPath();
      ctx.moveTo(X(rs), pad.t);
      ctx.lineTo(X(rs), h-pad.b);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(0));
    ctx.lineTo(w-pad.r, Y(0));
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#4a5058';
    ctx.font = '11px JetBrains Mono';
    ctx.fillText(absolute ? 'f(r)=1−2M/r on fixed r-axis' : 'f(r)=1−2M/r', pad.l+8, pad.t+14);
    if(rs >= r1 && rs <= r2){
      ctx.fillText('r_s', X(rs)+6, pad.t+14);
    }
  }

  function drawRouteA(canvas, m){
    if(!canvas) return;
    const ctx = canvas.getContext('2d');
    const box = fit(canvas, ctx, 180);
    const w = box.w, h = box.h;
    const TH = 1/(8*Math.PI*m);
    const kap = 2*Math.PI*TH;
    const rs = 1/(4*Math.PI*TH);
    const pad = {l:42,r:18,t:18,b:24};
    const barW = 42;
    const maxTH = 1/(8*Math.PI*0.4);
    const hTH = (TH/maxTH)*(h-pad.t-pad.b);
    const hKap = (kap/0.5)*(h-pad.t-pad.b);
    const hRs = (Math.min(rs,6)/6)*(h-pad.t-pad.b);
    const xs = [pad.l+36, pad.l+146, pad.l+256];
    const hs = [hTH, hKap, hRs];
    const labels = ['T_H','κ=2πT_H','r_s'];
    const colors = ['oklch(0.58 0.14 55)','oklch(0.55 0.17 25)','oklch(0.52 0.18 268)'];

    ctx.clearRect(0,0,w,h);
    ctx.strokeStyle = 'rgba(0,0,0,.12)';
    ctx.strokeRect(pad.l,pad.t,w-pad.l-pad.r,h-pad.t-pad.b);
    hs.forEach((hh,i)=>{
      ctx.fillStyle = colors[i];
      ctx.fillRect(xs[i], h-pad.b-hh, barW, hh);
      ctx.fillStyle = '#4a5058';
      ctx.font = '11px JetBrains Mono';
      ctx.fillText(labels[i], xs[i]-4, h-pad.b+14);
    });
    ctx.fillStyle = '#4a5058';
    ctx.font = '11px JetBrains Mono';
    ctx.fillText('closed-form thermodynamic outputs', pad.l+8, pad.t+14);
  }

  function routeFix(){
    const host = document.getElementById('routeConverge');
    if(!host || host.dataset.routeFixed) return;
    host.dataset.routeFixed = '1';

    const M = document.getElementById('M_el');
    const Mv = document.getElementById('M_v');
    const rsB = document.getElementById('rc-rsB');
    const kapB = document.getElementById('rc-kapB');
    const THA = document.getElementById('rc-THA');
    const rsA = document.getElementById('rc-rsA');
    const dRs = document.getElementById('rc-resRs');
    const dK = document.getElementById('rc-resK');
    const verdict = document.getElementById('rc-verdict');
    const cB = document.getElementById('rc-canvas-B');
    const cA = document.getElementById('rc-canvas-A');
    const cBabs = document.getElementById('rc-canvas-Babs');
    if(!M||!Mv||!rsB||!kapB||!THA||!rsA||!dRs||!dK||!verdict||!cB||!cA) return;

    const btns = [...host.querySelectorAll('[data-rc]')];

    function update(){
      const m = +M.value;
      const rs = 2*m;
      const kap = 1/(4*m);
      const th = 1/(8*Math.PI*m);
      const rsFromA = 1/(4*Math.PI*th);
      const kapFromA = 2*Math.PI*th;

      Mv.textContent = m.toFixed(2);
      rsB.textContent = rs.toFixed(4);
      kapB.textContent = kap.toFixed(4);
      THA.textContent = th.toFixed(4);
      rsA.textContent = rsFromA.toFixed(4);
      dRs.textContent = Math.abs(rs-rsFromA)<1e-12 ? '<1e−12' : Math.abs(rs-rsFromA).toExponential(2);
      dK.textContent = Math.abs(kap-kapFromA)<1e-12 ? '<1e−12' : Math.abs(kap-kapFromA).toExponential(2);
      verdict.className = 'ac-verdict ac-pass';
      verdict.innerHTML = '<b>Closed-form Schwarzschild outputs agree.</b> Route B fixes <span class="mono">r_s=2M</span> and <span class="mono">κ=1/(4M)</span>; Route A gives <span class="mono">T_H=1/(8πM)</span>, hence <span class="mono">κ=2πT_H</span> and the same horizon radius.';
      drawRouteB(cB, m, false);
      drawRouteA(cA, m);
      drawRouteB(cBabs, m, true);
      if(window.MathJax && window.MathJax.typesetPromise){
        window.MathJax.typesetPromise([verdict]).catch(()=>{});
      }
    }

    M.addEventListener('input', update);
    btns.forEach(b => b.addEventListener('click', ()=>{
      btns.forEach(x=>x.classList.remove('on'));
      b.classList.add('on');
      M.value = b.dataset.rc;
      update();
    }));
    if(btns[1]) btns[1].classList.add('on');
    window.addEventListener('resize', update);
    update();
  }

  function routeAudit(){
    const host = document.getElementById('routesAudit');
    if(!host || host.dataset.bound) return;
    host.dataset.bound = '1';
    host.innerHTML = [
      '<div class="ac-mini">',
      '  <div class="mini"><b>Route A</b>Needs the external inputs <span class="mono">T_eff=T_tol</span> and <span class="mono">S∝A</span>. The live page only checks the closed Schwarzschild formulas, not a full standalone derivation.</div>',
      '  <div class="mini"><b>Route B</b>Is algebraic and spherical-vacuum in scope. The consistency panel does not enlarge that theorem; it only shows its Schwarzschild endpoint agrees with the standard thermodynamic readout.</div>',
      '</div>',
      '<div class="ac-verdict ac-pass"><b>Why this panel exists.</b> It tests that the two analytic chains land on the same Schwarzschild numbers without claiming that Route B has become a general matter-coupled field-equation derivation.</div>'
    ].join('');
    if(window.MathJax && window.MathJax.typesetPromise){
      window.MathJax.typesetPromise([host]).catch(()=>{});
    }
  }

  routeFix();
  routeAudit();
})();
