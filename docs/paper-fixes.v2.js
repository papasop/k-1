(function(){
  function routeFix(){
    const host=document.getElementById('routeConverge');
    if(!host || host.dataset.routeFixed) return;
    host.dataset.routeFixed='1';
    const M=document.getElementById('M_el');
    const Mv=document.getElementById('M_v');
    const rsB=document.getElementById('rc-rsB');
    const kapB=document.getElementById('rc-kapB');
    const THA=document.getElementById('rc-THA');
    const rsA=document.getElementById('rc-rsA');
    const dRs=document.getElementById('rc-resRs');
    const dK=document.getElementById('rc-resK');
    const verdict=document.getElementById('rc-verdict');
    const cB=document.getElementById('rc-canvas-B');
    const cA=document.getElementById('rc-canvas-A');
    if(!M||!Mv||!rsB||!kapB||!THA||!rsA||!dRs||!dK||!verdict||!cB||!cA) return;
    const ctxB=cB.getContext('2d'), ctxA=cA.getContext('2d');
    const btns=[...host.querySelectorAll('[data-rc]')];
    function fit(canvas, ctx){
      const dpr=window.devicePixelRatio||1;
      const w=canvas.clientWidth||420;
      const h=200;
      canvas.width=w*dpr; canvas.height=h*dpr;
      ctx.setTransform(dpr,0,0,dpr,0,0);
      return {w,h};
    }
    function drawRouteB(m){
      const {w,h}=fit(cB,ctxB);
      ctxB.clearRect(0,0,w,h);
      const rs=2*m;
      const r1=Math.max(0.3,0.55*rs), r2=4.4*rs;
      const pad={l:42,r:12,t:14,b:24};
      const vals=[];
      for(let i=0;i<=220;i++){
        const r=r1+(r2-r1)*i/220;
        vals.push(1-rs/r);
      }
      const ymin=Math.min(-0.35,...vals), ymax=Math.max(1.05,...vals);
      const X=r=>pad.l+(r-r1)/(r2-r1)*(w-pad.l-pad.r);
      const Y=v=>pad.t+(1-(v-ymin)/(ymax-ymin))*(h-pad.t-pad.b);
      ctxB.strokeStyle='rgba(0,0,0,.12)';
      ctxB.strokeRect(pad.l,pad.t,w-pad.l-pad.r,h-pad.t-pad.b);
      ctxB.strokeStyle='oklch(0.52 0.18 268)';
      ctxB.lineWidth=2;
      ctxB.beginPath();
      vals.forEach((v,i)=>{
        const r=r1+(r2-r1)*i/220;
        const x=X(r), y=Y(v);
        if(i===0) ctxB.moveTo(x,y); else ctxB.lineTo(x,y);
      });
      ctxB.stroke();
      ctxB.strokeStyle='oklch(0.55 0.18 25)';
      ctxB.setLineDash([4,4]);
      ctxB.beginPath(); ctxB.moveTo(X(rs), pad.t); ctxB.lineTo(X(rs), h-pad.b); ctxB.stroke();
      ctxB.beginPath(); ctxB.moveTo(pad.l, Y(0)); ctxB.lineTo(w-pad.r, Y(0)); ctxB.stroke();
      ctxB.setLineDash([]);
      ctxB.fillStyle='var(--ink-3)'; ctxB.font='11px JetBrains Mono';
      ctxB.fillText('f(r)=1−2M/r', pad.l+8, pad.t+14);
      ctxB.fillText('r_s', X(rs)+6, pad.t+14);
    }
    function drawRouteA(m){
      const {w,h}=fit(cA,ctxA);
      ctxA.clearRect(0,0,w,h);
      const TH=1/(8*Math.PI*m), kap=2*Math.PI*TH, rs=1/(4*Math.PI*TH);
      const pad={l:42,r:18,t:18,b:24}, barW=42, maxTH=1/(8*Math.PI*0.4);
      const hTH=(TH/maxTH)*(h-pad.t-pad.b);
      const hKap=(kap/(0.5))*(h-pad.t-pad.b);
      const hRs=(Math.min(rs,6)/6)*(h-pad.t-pad.b);
      const xs=[pad.l+36,pad.l+146,pad.l+256], hs=[hTH,hKap,hRs];
      const labels=['T_H','κ=2πT_H','r_s'];
      const colors=['oklch(0.58 0.14 55)','oklch(0.55 0.17 25)','oklch(0.52 0.18 268)'];
      ctxA.strokeStyle='rgba(0,0,0,.12)'; ctxA.strokeRect(pad.l,pad.t,w-pad.l-pad.r,h-pad.t-pad.b);
      hs.forEach((hh,i)=>{
        ctxA.fillStyle=colors[i]; ctxA.fillRect(xs[i], h-pad.b-hh, barW, hh);
        ctxA.fillStyle='var(--ink-3)'; ctxA.font='11px JetBrains Mono'; ctxA.fillText(labels[i], xs[i]-4, h-pad.b+14);
      });
      ctxA.fillStyle='var(--ink-3)'; ctxA.font='11px JetBrains Mono';
      ctxA.fillText('closed-form thermodynamic outputs', pad.l+8, pad.t+14);
    }
    function update(){
      const m=+M.value, rs=2*m, kap=1/(4*m), th=1/(8*Math.PI*m), rsFromA=1/(4*Math.PI*th), kapFromA=2*Math.PI*th;
      Mv.textContent=m.toFixed(2); rsB.textContent=rs.toFixed(4); kapB.textContent=kap.toFixed(4); THA.textContent=th.toFixed(4); rsA.textContent=rsFromA.toFixed(4);
      dRs.textContent=Math.abs(rs-rsFromA)<1e-12?'<1e−12':Math.abs(rs-rsFromA).toExponential(2);
      dK.textContent=Math.abs(kap-kapFromA)<1e-12?'<1e−12':Math.abs(kap-kapFromA).toExponential(2);
      verdict.className='ac-verdict ac-pass';
      verdict.innerHTML='<b>Closed-form Schwarzschild outputs agree.</b> Route B fixes <span class="mono">r_s=2M</span> and <span class="mono">κ=1/(4M)</span>; Route A gives <span class="mono">T_H=1/(8πM)</span>, hence <span class="mono">κ=2πT_H</span> and the same horizon radius.';
      drawRouteB(m); drawRouteA(m);
      if(window.MathJax&&window.MathJax.typesetPromise) window.MathJax.typesetPromise([verdict]);
    }
    M.addEventListener('input', update);
    btns.forEach(b=>b.onclick=()=>{ btns.forEach(x=>x.classList.remove('on')); b.classList.add('on'); M.value=b.dataset.rc; update();});
    if(btns[1]) btns[1].classList.add('on');
    window.addEventListener('resize', update);
    update();
  }

  function birkhoffStage3Fix(){
    const M=document.getElementById('bk3-M');
    const cv=document.getElementById('bk3-canvas');
    if(!M || !cv || M.dataset.bk3Fixed) return;
    M.dataset.bk3Fixed='1';
    const ctx=cv.getContext('2d');
    function update(){
      const m=+M.value;
      const mv=document.getElementById('bk3-Mv'), rh=document.getElementById('bk3-rh'), kap=document.getElementById('bk3-kappa'), th=document.getElementById('bk3-TH');
      if(!mv||!rh||!kap||!th) return;
      mv.textContent=m.toFixed(3);
      rh.textContent=(2*m).toFixed(4);
      kap.textContent=(1/(4*m)).toFixed(4);
      th.textContent=(1/(8*Math.PI*m)).toFixed(4);
      const dpr=window.devicePixelRatio||1, w=cv.clientWidth||900, h=220;
      cv.width=w*dpr; cv.height=h*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.clearRect(0,0,w,h);
      const pad={l:44,r:12,t:14,b:24}, rs=2*m, r1=Math.max(0.15,0.4*rs), r2=4.2*rs;
      const X=r=>pad.l+(r-r1)/(r2-r1)*(w-pad.l-pad.r), Y=v=>pad.t+(1-v)*(h-pad.t-pad.b);
      ctx.strokeStyle='rgba(0,0,0,.12)'; ctx.strokeRect(pad.l,pad.t,w-pad.l-pad.r,h-pad.t-pad.b);
      ctx.strokeStyle='oklch(0.52 0.18 268)'; ctx.lineWidth=2; ctx.beginPath();
      for(let i=0;i<=220;i++){
        const r=r1+(r2-r1)*i/220, f=1-rs/r, x=X(r), y=Y(Math.max(-0.25,Math.min(1.05,f)));
        if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }
      ctx.stroke();
      ctx.setLineDash([4,4]); ctx.strokeStyle='oklch(0.55 0.18 25)';
      ctx.beginPath(); ctx.moveTo(X(rs),pad.t); ctx.lineTo(X(rs),h-pad.b); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pad.l,Y(0)); ctx.lineTo(w-pad.r,Y(0)); ctx.stroke();
      ctx.setLineDash([]); ctx.fillStyle='var(--ink-3)'; ctx.font='11px JetBrains Mono';
      ctx.fillText('f(r)=1−2M/r', pad.l+8, pad.t+14); ctx.fillText('r_s', X(rs)+6, pad.t+14);
    }
    M.addEventListener('input', update);
    window.addEventListener('resize', update);
    update();
  }

  function apply(){
    routeFix();
    birkhoffStage3Fix();
    if(location.hash){
      const el=document.querySelector(location.hash);
      if(el) setTimeout(()=>el.scrollIntoView(), 60);
    }
  }

  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', apply, {once:true});
  else apply();
})();
