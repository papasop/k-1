




<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lorentz Transformer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

  :root {
    --bg:       #0d1117;
    --surface:  #161b22;
    --surface2: #1c2128;
    --border:   #30363d;
    --border2:  #21262d;
    --text:     #e6edf3;
    --muted:    #8b949e;
    --dim:      #484f58;
    --blue:     #58a6ff;
    --blue2:    #1f6feb;
    --cyan:     #79c0ff;
    --green:    #3fb950;
    --green2:   #238636;
    --yellow:   #d29922;
    --orange:   #f0883e;
    --red:      #f85149;
    --purple:   #bc8cff;
    --pink:     #ff7b72;
    --accent:   #388bfd;
    --glow:     rgba(88,166,255,0.15);
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.7;
    min-height: 100vh;
  }

  /* ── Layout ── */
  .page { max-width: 980px; margin: 0 auto; padding: 0 16px; }

  /* ── Header Bar ── */
  .gh-header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 0;
    position: sticky; top: 0; z-index: 100;
    backdrop-filter: blur(8px);
  }
  .gh-header .page { display: flex; align-items: center; gap: 12px; }
  .gh-logo { color: var(--text); font-size: 22px; opacity: 0.9; }
  .repo-path { font-family: 'JetBrains Mono', monospace; font-size: 14px; color: var(--blue); }
  .repo-path span { color: var(--muted); }
  .badges { margin-left: auto; display: flex; gap: 8px; }
  .badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 500;
    padding: 3px 10px; border-radius: 20px;
    border: 1px solid currentColor;
  }
  .badge-blue  { color: var(--blue);   border-color: var(--blue2); background: rgba(31,111,235,0.1); }
  .badge-green { color: var(--green);  border-color: var(--green2); background: rgba(63,185,80,0.1); }
  .badge-purple{ color: var(--purple); border-color: rgba(188,140,255,0.4); background: rgba(188,140,255,0.1); }

  /* ── Hero ── */
  .hero {
    padding: 64px 0 48px;
    border-bottom: 1px solid var(--border);
    position: relative; overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 800px 400px at 50% 0%, rgba(56,139,253,0.08), transparent);
    pointer-events: none;
  }
  .hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: var(--cyan);
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 20px;
    display: flex; align-items: center; gap: 8px;
  }
  .hero-tag::before {
    content: ''; display: block;
    width: 24px; height: 1px; background: var(--cyan);
  }
  h1.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(36px, 6vw, 64px);
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 20px;
    background: linear-gradient(135deg, #e6edf3 0%, #58a6ff 50%, #bc8cff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .hero-sub {
    font-size: 18px; color: var(--muted); max-width: 600px;
    margin-bottom: 32px; font-weight: 300;
  }
  .hero-stats {
    display: flex; gap: 32px; flex-wrap: wrap;
    margin-bottom: 32px;
  }
  .stat { display: flex; flex-direction: column; }
  .stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px; font-weight: 700; color: var(--blue);
    line-height: 1;
  }
  .stat-val.green { color: var(--green); }
  .stat-val.purple { color: var(--purple); }
  .stat-label { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .cta-row { display: flex; gap: 12px; flex-wrap: wrap; }
  .btn {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px; font-weight: 500;
    padding: 10px 20px; border-radius: 6px;
    cursor: pointer; text-decoration: none;
    transition: all 0.15s ease;
    display: inline-flex; align-items: center; gap: 8px;
  }
  .btn-primary {
    background: var(--accent); color: white;
    border: 1px solid rgba(255,255,255,0.1);
  }
  .btn-primary:hover { background: #4493f8; }
  .btn-outline {
    background: transparent; color: var(--text);
    border: 1px solid var(--border);
  }
  .btn-outline:hover { background: var(--surface2); border-color: var(--border2); }

  /* ── Content layout ── */
  .content { display: grid; grid-template-columns: 1fr 260px; gap: 24px; padding: 32px 0 64px; }
  .main { min-width: 0; }
  .sidebar { position: sticky; top: 60px; align-self: start; }

  /* ── Alert boxes ── */
  .alert {
    padding: 16px 20px; border-radius: 8px;
    border-left: 3px solid; margin: 20px 0;
    font-size: 14px;
  }
  .alert-blue  { background: rgba(56,139,253,0.1); border-color: var(--accent); }
  .alert-green { background: rgba(63,185,80,0.1);  border-color: var(--green); }
  .alert-yellow{ background: rgba(210,153,34,0.1); border-color: var(--yellow); }
  .alert-purple{ background: rgba(188,140,255,0.1);border-color: var(--purple); }
  .alert-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; font-weight: 700;
    letter-spacing: 0.05em; text-transform: uppercase;
    margin-bottom: 6px;
  }
  .alert-blue   .alert-title { color: var(--blue); }
  .alert-green  .alert-title { color: var(--green); }
  .alert-yellow .alert-title { color: var(--yellow); }
  .alert-purple .alert-title { color: var(--purple); }

  /* ── Section headings ── */
  h2 {
    font-family: 'Syne', sans-serif;
    font-size: 22px; font-weight: 700;
    padding-bottom: 10px;
    margin: 40px 0 16px;
    border-bottom: 1px solid var(--border);
    scroll-margin-top: 70px;
  }
  h3 {
    font-family: 'Syne', sans-serif;
    font-size: 16px; font-weight: 600;
    margin: 24px 0 10px; color: var(--cyan);
  }
  p { margin: 10px 0; font-size: 15px; }

  /* ── Math display ── */
  .math-block {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--blue2);
    padding: 16px 20px;
    border-radius: 6px;
    margin: 16px 0;
    overflow-x: auto;
    white-space: pre;
    color: var(--cyan);
    line-height: 1.8;
  }

  /* ── Code blocks ── */
  pre {
    background: #010409;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    overflow-x: auto;
    margin: 16px 0;
    position: relative;
  }
  pre .lang-tag {
    position: absolute; top: 10px; right: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--dim);
    text-transform: uppercase; letter-spacing: 0.1em;
  }
  code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px; line-height: 1.7;
  }
  p code, li code {
    background: rgba(110,118,129,0.15);
    padding: 2px 6px; border-radius: 4px;
    font-size: 12px; color: var(--pink);
  }

  /* Syntax colors */
  .kw  { color: #ff7b72; }
  .fn  { color: #d2a8ff; }
  .str { color: #a5d6ff; }
  .cm  { color: #8b949e; font-style: italic; }
  .num { color: #79c0ff; }
  .op  { color: #ff7b72; }

  /* ── Tables ── */
  .table-wrap { overflow-x: auto; margin: 16px 0; border-radius: 8px; border: 1px solid var(--border); }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  thead tr { background: var(--surface2); }
  th {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; font-weight: 500;
    padding: 10px 14px; text-align: left;
    color: var(--muted); letter-spacing: 0.03em;
    border-bottom: 1px solid var(--border);
  }
  td { padding: 10px 14px; border-bottom: 1px solid var(--border2); }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255,255,255,0.02); }
  .tag-pass  { color: var(--green);  font-weight: 600; }
  .tag-warn  { color: var(--yellow); font-weight: 600; }
  .tag-fail  { color: var(--red);    font-weight: 600; }
  .tag-note  { color: var(--muted);  font-size: 12px; }
  .mono { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--cyan); }

  /* ── Component cards ── */
  .components { display: grid; gap: 16px; margin: 20px 0; }
  .comp-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px;
    position: relative; overflow: hidden;
    transition: border-color 0.2s;
  }
  .comp-card:hover { border-color: var(--accent); }
  .comp-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  }
  .comp-card.c1::before { background: linear-gradient(90deg, var(--blue), var(--cyan)); }
  .comp-card.c2::before { background: linear-gradient(90deg, var(--green), var(--cyan)); }
  .comp-card.c3::before { background: linear-gradient(90deg, var(--purple), var(--pink)); }
  .comp-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 6px;
  }
  .comp-title {
    font-family: 'Syne', sans-serif;
    font-size: 18px; font-weight: 700; margin-bottom: 10px;
  }
  .c1 .comp-title { color: var(--blue); }
  .c2 .comp-title { color: var(--green); }
  .c3 .comp-title { color: var(--purple); }
  .comp-formula {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px; color: var(--cyan);
    background: rgba(0,0,0,0.3);
    padding: 10px 14px; border-radius: 6px;
    margin: 12px 0; overflow-x: auto; white-space: nowrap;
  }
  .comp-desc { font-size: 14px; color: var(--muted); }

  /* ── Sidebar ── */
  .sidebar-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
    margin-bottom: 16px;
  }
  .sidebar-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; font-weight: 600;
    color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 12px;
  }
  .toc-item {
    display: block; padding: 5px 8px;
    font-size: 13px; color: var(--muted);
    text-decoration: none; border-radius: 4px;
    transition: all 0.15s;
  }
  .toc-item:hover { color: var(--text); background: var(--surface2); }
  .toc-item.sub { padding-left: 20px; font-size: 12px; }
  .meta-row { display: flex; justify-content: space-between; align-items: center; margin: 6px 0; }
  .meta-key { font-size: 13px; color: var(--muted); }
  .meta-val { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text); }

  /* ── Result pills ── */
  .result-row { display: flex; align-items: center; gap: 10px; margin: 8px 0; }
  .pill {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    padding: 2px 8px; border-radius: 20px; font-weight: 600;
  }
  .pill-green { background: rgba(63,185,80,0.15); color: var(--green); border: 1px solid rgba(63,185,80,0.3); }
  .pill-red   { background: rgba(248,81,73,0.15);  color: var(--red);   border: 1px solid rgba(248,81,73,0.3); }
  .pill-blue  { background: rgba(88,166,255,0.15); color: var(--blue);  border: 1px solid rgba(88,166,255,0.3); }

  /* ── Timeline ── */
  .timeline { position: relative; padding-left: 24px; margin: 16px 0; }
  .timeline::before { content:''; position:absolute; left:7px; top:6px; bottom:6px; width:1px; background:var(--border); }
  .tl-item { position: relative; margin-bottom: 20px; }
  .tl-dot {
    position: absolute; left: -20px; top: 5px;
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--surface); border: 2px solid var(--border);
  }
  .tl-item.done .tl-dot { background: var(--green); border-color: var(--green); }
  .tl-item.next .tl-dot { background: var(--blue); border-color: var(--blue); box-shadow: 0 0 8px var(--blue); }
  .tl-step { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--muted); }
  .tl-title { font-weight: 500; font-size: 14px; margin: 2px 0; }
  .tl-desc { font-size: 13px; color: var(--muted); }

  /* ── Divider ── */
  hr { border: none; border-top: 1px solid var(--border); margin: 32px 0; }

  /* ── Lists ── */
  ul, ol { padding-left: 20px; margin: 10px 0; }
  li { margin: 6px 0; font-size: 15px; }
  li::marker { color: var(--accent); }

  /* ── Responsive ── */
  @media (max-width: 768px) {
    .content { grid-template-columns: 1fr; }
    .sidebar { position: static; }
    .hero-stats { gap: 20px; }
    .badges { display: none; }
  }

  /* ── Animations ── */
  @keyframes fadeUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:none; } }
  .hero > * { animation: fadeUp 0.5s ease both; }
  .hero > *:nth-child(1) { animation-delay: 0.1s; }
  .hero > *:nth-child(2) { animation-delay: 0.2s; }
  .hero > *:nth-child(3) { animation-delay: 0.3s; }
  .hero > *:nth-child(4) { animation-delay: 0.4s; }
  .hero > *:nth-child(5) { animation-delay: 0.5s; }
</style>
</head>
<body>

<!-- GitHub-style header -->
<header class="gh-header">
  <div class="page">
    <span class="gh-logo">⬡</span>
    <span class="repo-path"><span>research /</span> lorentz-transformer</span>
    <div class="badges">
      <span class="badge badge-green">experiments: passing</span>
      <span class="badge badge-blue">seeds: 3</span>
      <span class="badge badge-purple">pseudo-Riemannian: confirmed</span>
    </div>
  </div>
</header>

<!-- Hero -->
<section class="hero">
  <div class="page">
    <div class="hero-tag">K=1 Field Equations · Information Geometry · Lorentzian Manifold</div>
    <h1 class="hero-title">Lorentz Transformer</h1>
    <p class="hero-sub">
      A Transformer architecture grounded in the geometry of its own parameter space —
      where the metric tensor is indefinite, timelike directions carry causal information,
      and attention is computed with a Minkowski inner product.
    </p>
    <div class="hero-stats">
      <div class="stat">
        <span class="stat-val">50–80%</span>
        <span class="stat-label">W_Q parameters timelike (G_ii &lt; 0)</span>
      </div>
      <div class="stat">
        <span class="stat-val green">+0.271</span>
        <span class="stat-label">lightcone gap at α=1.0 (1-hop)</span>
      </div>
      <div class="stat">
        <span class="stat-val purple">−1.000</span>
        <span class="stat-label">r-law correlation across all experiments</span>
      </div>
      <div class="stat">
        <span class="stat-val">3</span>
        <span class="stat-label">seeds · 2 tasks · 4 α values</span>
      </div>
    </div>
    <div class="cta-row">
      <a href="#architecture" class="btn btn-primary">↓ Architecture</a>
      <a href="#results" class="btn btn-outline">Experimental Results</a>
      <a href="#roadmap" class="btn btn-outline">Roadmap</a>
    </div>
  </div>
</section>

<!-- Main content -->
<div class="page">
<div class="content">

<!-- ── MAIN ── -->
<main class="main">

  <!-- Foundation -->
  <h2 id="foundation">📐 Theoretical Foundation</h2>

  <p>
    The K=1 paper defines an <strong>information time metric</strong> on the attention weight manifold.
    Its Hessian with respect to W_Q determines the curvature of parameter space.
    Experimental measurement (Hutchinson estimator, K=20) reveals this curvature is <em>indefinite</em> —
    roughly half the parameter directions are concave (timelike) and half convex (spacelike).
    This is the signature of a <strong>pseudo-Riemannian (Lorentzian) manifold</strong>.
  </p>

  <div class="math-block">dt²_info = Σ_q K_q       K_q = Φ_q / H_q

G_ij = ∂²(dt²_info) / ∂W_Q[i] ∂W_Q[j]

G_ii < 0  →  TIMELIKE   (dt²_info concave in direction i)
G_ii > 0  →  SPACELIKE  (dt²_info convex  in direction i)</div>

  <div class="alert alert-blue">
    <div class="alert-title">🔬 Experimental Confirmation</div>
    Confirmed across <strong>3 seeds × 2 tasks × 4 α values × 80 epochs</strong>.
    Timelike fraction ranges from 50% to 80% depending on task complexity and α.
    2-hop tasks show consistently higher timelike fraction than 1-hop,
    because harder tasks produce more dispersed attention (higher H → lower K → more concave dt²_info).
  </div>

  <!-- Architecture -->
  <h2 id="architecture">🏗️ Architecture</h2>
  <p>
    All three components share one core object: the <strong>timelike projection matrix P_t</strong>.
    It is computed from the negative diagonal Hessian directions and updated every N=40 steps.
  </p>

  <div class="math-block">P_t = diag(mask)    mask[i] = 1 if G_ii < 0 else 0

η = I − 2α P_t       (Minkowski signature matrix)</div>

  <div class="components">

    <!-- C1 -->
    <div class="comp-card c1">
      <div class="comp-num">Component 01</div>
      <div class="comp-title">Minkowski Attention</div>
      <div class="comp-formula">scores_L = Q η K^T / √d  =  QK^T/√d  −  2α (QP_t)K^T/√d</div>
      <div class="comp-desc">
        Replace the Euclidean inner product with a Minkowski inner product.
        Timelike token pairs (causally connected) are suppressed by the negative term;
        spacelike pairs are unaffected. At α=0 this is exactly standard attention.
        At α=1.0, the lightcone flips: real causal chain pairs become more timelike
        than noise pairs (<strong>+0.271 gap</strong> confirmed experimentally).
      </div>
    </div>

    <!-- C2 -->
    <div class="comp-card c2">
      <div class="comp-num">Component 02</div>
      <div class="comp-title">Geodesic Adam Optimizer</div>
      <div class="comp-formula">Δθ = − lr_t · P_t ∇L  −  lr_s · (I−P_t) ∇L      lr_t > lr_s</div>
      <div class="comp-desc">
        Decompose each gradient into timelike and spacelike components.
        Apply a larger learning rate to timelike directions (information-rich, geodesic path)
        and a smaller rate to spacelike directions (knowledge-preserving).
        This is natural gradient descent adapted for an indefinite metric.
      </div>
    </div>

    <!-- C3 -->
    <div class="comp-card c3">
      <div class="comp-num">Component 03</div>
      <div class="comp-title">Timelike Submanifold Regularization</div>
      <div class="comp-formula">R(θ) = λ_s ‖(I−P_t)θ‖²  −  λ_t ‖P_t θ‖²</div>
      <div class="comp-desc">
        Penalize spacelike parameter components (protect existing knowledge)
        while optionally rewarding timelike components (encourage learning).
        Geometrically, this confines updates to the timelike submanifold —
        a principled alternative to EWC that requires no task boundary and no parameter snapshots.
      </div>
    </div>

  </div>

  <!-- P_t computation -->
  <h2 id="pt">🔧 Computing P_t</h2>

  <div class="table-wrap">
    <table>
      <thead>
        <tr><th>Method</th><th>HVP Cost</th><th>Detects G_ii &lt; 0?</th><th>LLM Scale (d=4096)</th></tr>
      </thead>
      <tbody>
        <tr>
          <td class="mono">Hutchinson (K=20)</td>
          <td>20 HVPs</td>
          <td class="tag-pass">✓ Yes</td>
          <td class="tag-warn">~336ms/layer (slow)</td>
        </tr>
        <tr>
          <td class="mono">Lanczos (k=10)</td>
          <td>10 HVPs</td>
          <td class="tag-pass">✓ Yes (min eigenvalue)</td>
          <td class="tag-pass">~1–5ms/layer ✓</td>
        </tr>
        <tr>
          <td class="mono">Random Lanczos (Nyström)</td>
          <td>2k HVPs</td>
          <td class="tag-pass">✓ Yes (most stable)</td>
          <td class="tag-pass">~5–20ms/layer ✓</td>
        </tr>
        <tr>
          <td class="mono">K-FAC</td>
          <td>0 extra</td>
          <td class="tag-fail">✗ No (S is PSD)</td>
          <td class="tag-note">Cannot detect timelike directions</td>
        </tr>
      </tbody>
    </table>
  </div>

  <p>HVP (Hessian-vector product) = two autograd passes with <code>create_graph=True</code>.
  Lanczos is recommended: 4× faster than Hutchinson, retains sign information.</p>

  <div class="alert alert-yellow">
    <div class="alert-title">⚠️ K-FAC Warning</div>
    K-FAC approximates <code>H ≈ A ⊗ S</code> where both A and S are covariance matrices (positive semi-definite).
    The Kronecker product of two PSD matrices is PSD — it can never have negative diagonal elements.
    <strong>K-FAC cannot detect timelike directions.</strong> Use Lanczos instead.
  </div>

  <!-- Hyperparameters -->
  <h2 id="hyperparams">⚙️ Hyperparameters</h2>

  <div class="table-wrap">
    <table>
      <thead><tr><th>Parameter</th><th>Default</th><th>Description</th></tr></thead>
      <tbody>
        <tr><td class="mono">α</td><td class="mono">0.25</td><td>Lorentz strength. Start at 0.25; 1.0 flips lightcone but hurts accuracy on strong baselines.</td></tr>
        <tr><td class="mono">lr_t</td><td class="mono">1.5 × lr</td><td>Timelike learning rate multiplier.</td></tr>
        <tr><td class="mono">lr_s</td><td class="mono">0.5 × lr</td><td>Spacelike learning rate multiplier.</td></tr>
        <tr><td class="mono">λ_s</td><td class="mono">1e-4</td><td>Spacelike regularization strength.</td></tr>
        <tr><td class="mono">λ_t</td><td class="mono">0</td><td>Timelike reward (optional; set 0 for stability).</td></tr>
        <tr><td class="mono">N</td><td class="mono">40 steps</td><td>P_t update frequency.</td></tr>
        <tr><td class="mono">K / k</td><td class="mono">20 / 10</td><td>Hutchinson samples / Lanczos iterations.</td></tr>
        <tr><td class="mono">EMA α</td><td class="mono">0.3</td><td>Fraction of new P_t per update (prevents oscillation).</td></tr>
        <tr><td class="mono">warmup</td><td class="mono">50–100</td><td>Steps before first P_t update (Adam moments must stabilize).</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Results -->
  <h2 id="results">📊 Experimental Results</h2>

  <h3>Pseudo-Riemannian Structure (Primary Finding)</h3>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Task</th><th>α</th><th>Seeds</th><th>G_ii &lt; 0 Fraction</th></tr></thead>
      <tbody>
        <tr><td>1-hop</td><td class="mono">0.25</td><td>3/3</td><td class="tag-pass">51–58%</td></tr>
        <tr><td>1-hop</td><td class="mono">0.5</td> <td>3/3</td><td class="tag-pass">59–70%</td></tr>
        <tr><td>1-hop</td><td class="mono">1.0</td> <td>3/3</td><td class="tag-pass">60–80%</td></tr>
        <tr><td>2-hop</td><td class="mono">0.25</td><td>2/2</td><td class="tag-pass">54–73%</td></tr>
        <tr><td>2-hop</td><td class="mono">0.5</td> <td>2/2</td><td class="tag-pass">62–76%</td></tr>
        <tr><td>2-hop</td><td class="mono">1.0</td> <td>1/1</td><td class="tag-pass">58–80%</td></tr>
      </tbody>
    </table>
  </div>

  <h3>Lightcone Flip (Key New Finding)</h3>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Task</th><th>α</th><th>Real chain (timelike)</th><th>Noise (timelike)</th><th>Gap</th></tr></thead>
      <tbody>
        <tr><td>1-hop</td><td class="mono">0.0</td><td>0.289</td><td>0.631</td><td class="tag-fail">−0.342</td></tr>
        <tr><td>1-hop</td><td class="mono">0.5</td><td>0.430</td><td>0.551</td><td class="tag-warn">−0.122</td></tr>
        <tr><td>1-hop</td><td class="mono">1.0</td><td>0.688</td><td>0.416</td><td class="tag-pass">+0.271 ✓</td></tr>
        <tr><td>2-hop</td><td class="mono">0.0</td><td>0.535</td><td>0.600</td><td class="tag-warn">−0.065</td></tr>
        <tr><td>2-hop</td><td class="mono">1.0</td><td>0.488</td><td>0.415</td><td class="tag-pass">+0.073 ✓</td></tr>
      </tbody>
    </table>
  </div>

  <h3>R-Law (Unified Training Dynamics)</h3>
  <p>
    Across <strong>every</strong> experiment (K-field, CGD, K=1 Lorentz), the correlation between
    baseline accuracy and injection delta is r ≈ −1.0. This is not a property of any specific
    injection mechanism — it is a fundamental law of Transformer training dynamics.
  </p>

  <div class="table-wrap">
    <table>
      <thead><tr><th>Experiment</th><th>α / condition</th><th>r(baseline, Δ)</th></tr></thead>
      <tbody>
        <tr><td>K-field (1-hop)</td><td class="mono">d=−1, α=0.5</td><td class="tag-pass">−0.997</td></tr>
        <tr><td>K=1 Lorentz (1-hop)</td><td class="mono">α=0.25</td><td class="tag-pass">−0.966</td></tr>
        <tr><td>K=1 Lorentz (1-hop)</td><td class="mono">α=0.5</td><td class="tag-pass">−0.999</td></tr>
        <tr><td>K=1 Lorentz (2-hop)</td><td class="mono">α=0.25</td><td class="tag-pass">−1.000</td></tr>
      </tbody>
    </table>
  </div>

  <div class="alert alert-purple">
    <div class="alert-title">💡 R-Law Interpretation</div>
    The r-law predicts: Lorentz Transformer benefits models in the learning regime (low baseline accuracy)
    and is neutral-to-harmful on saturated baselines.
    <strong>Optimal use case: pre-training from scratch or fine-tuning on genuinely novel tasks.</strong>
    At α=0.25, 2-hop seed 0 shows +0.016 accuracy improvement — the first positive performance result.
  </div>

  <!-- Code -->
  <h2 id="code">💻 Implementation</h2>

  <pre><code><span class="cm"># Core: compute timelike projection matrix P_t</span>
<span class="kw">def</span> <span class="fn">hutchinson_diag_hessian</span>(loss_fn, W_Q, K=<span class="num">20</span>):
    G = torch.zeros_like(W_Q)
    <span class="kw">for</span> _ <span class="kw">in</span> range(K):
        v  = (torch.randint(<span class="num">0</span>, <span class="num">2</span>, W_Q.shape) * <span class="num">2</span> - <span class="num">1</span>).float()
        g1 = autograd.grad(loss_fn(), W_Q, create_graph=<span class="kw">True</span>)[<span class="num">0</span>]
        Hv = autograd.grad((g1 * v).sum(), W_Q)[<span class="num">0</span>]
        G += (v * Hv).detach() / K
    <span class="kw">return</span> G  <span class="cm"># G < 0 → timelike, G > 0 → spacelike</span>

<span class="cm"># Minkowski attention scores</span>
P_t    = (G_diag < <span class="num">0</span>).float().diag()   <span class="cm"># timelike projection</span>
eta    = torch.eye(d) - <span class="num">2</span> * alpha * P_t <span class="cm"># Minkowski metric</span>
scores = (Q @ eta @ K.transpose(-<span class="num">2</span>, -<span class="num">1</span>)) / math.sqrt(d_h)

<span class="cm"># Geodesic Adam gradient decomposition</span>
g_t = P_t @ grad.flatten()       <span class="cm"># timelike component</span>
g_s = grad.flatten() - g_t       <span class="cm"># spacelike component</span>
grad_lorentz = (lr_t_scale * g_t + lr_s_scale * g_s).reshape(grad.shape)

<span class="cm"># Timelike regularization</span>
R = lambda_s * ((I - P_t) @ theta).pow(<span class="num">2</span>).sum()
loss_total = loss_task + R
<span class="lang-tag">python</span></code></pre>

  <!-- Files -->
  <h2 id="files">📁 Repository Files</h2>

  <div class="table-wrap">
    <table>
      <thead><tr><th>File</th><th>Description</th></tr></thead>
      <tbody>
        <tr><td class="mono">k1_lorentz.py</td><td>Main experiment: Hutchinson Hessian + Minkowski attention (3 steps)</td></tr>
        <tr><td class="mono">lanczos_lorentz.py</td><td>Lanczos vs Hutchinson comparison; LLM scale feasibility</td></tr>
        <tr><td class="mono">gpt2_lorentz_test.py</td><td>GPT-2 validation: experiments A/B/C (λ_min, stability, depth)</td></tr>
        <tr><td class="mono">lorentz_inner.py</td><td>Minkowski inner product attention with lightcone diagnostics</td></tr>
        <tr><td class="mono">lorentz_scores.py</td><td>Position-B: Mahalanobis distance penalty on scores</td></tr>
        <tr><td class="mono">cgd_experiment.py</td><td>CGD three-method experiment (Γ^info, PLLR, NULL subspace)</td></tr>
        <tr><td class="mono">dtfv2_experiment.py</td><td>K-field routing experiment (v8, 3 seeds, val-set selection)</td></tr>
        <tr><td class="mono">kfac_feasibility.py</td><td>K-FAC vs Lanczos feasibility analysis (shows K-FAC fails)</td></tr>
        <tr><td class="mono">lorentz_spec.docx</td><td>Full architecture specification document</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Roadmap -->
  <h2 id="roadmap">🗺️ Roadmap</h2>
  <div class="timeline">
    <div class="tl-item done">
      <div class="tl-dot"></div>
      <div class="tl-step">COMPLETED</div>
      <div class="tl-title">Pseudo-Riemannian structure confirmed</div>
      <div class="tl-desc">50-80% timelike fraction, r=-1.0, lightcone flip at α=1.0</div>
    </div>
    <div class="tl-item done">
      <div class="tl-dot"></div>
      <div class="tl-step">COMPLETED</div>
      <div class="tl-title">Three architecture components specified</div>
      <div class="tl-desc">Minkowski attention, Geodesic Adam, Timelike regularization</div>
    </div>
    <div class="tl-item next">
      <div class="tl-dot"></div>
      <div class="tl-step">NEXT</div>
      <div class="tl-title">Geodesic Adam implementation + validation</div>
      <div class="tl-desc">Test on 128d synthetic tasks; verify r-law holds with geodesic optimizer</div>
    </div>
    <div class="tl-item">
      <div class="tl-dot"></div>
      <div class="tl-step">PLANNED</div>
      <div class="tl-title">Full Lorentz Transformer at 256d / 6L</div>
      <div class="tl-desc">All three components integrated; train from scratch</div>
    </div>
    <div class="tl-item">
      <div class="tl-dot"></div>
      <div class="tl-step">PLANNED</div>
      <div class="tl-title">GPT-2 scale validation (768d / 12L)</div>
      <div class="tl-desc">Random Lanczos; real language data; lightcone on natural text</div>
    </div>
    <div class="tl-item">
      <div class="tl-dot"></div>
      <div class="tl-step">PLANNED</div>
      <div class="tl-title">Lorentz LLM at 1B+ scale</div>
      <div class="tl-desc">Train from scratch; evaluate on MuSiQue, HotpotQA multi-hop benchmarks</div>
    </div>
  </div>

</main>

<!-- ── SIDEBAR ── -->
<aside class="sidebar">

  <div class="sidebar-card">
    <div class="sidebar-title">Contents</div>
    <a href="#foundation" class="toc-item">📐 Foundation</a>
    <a href="#architecture" class="toc-item">🏗️ Architecture</a>
    <a href="#pt" class="toc-item sub">↳ Computing P_t</a>
    <a href="#hyperparams" class="toc-item">⚙️ Hyperparameters</a>
    <a href="#results" class="toc-item">📊 Results</a>
    <a href="#code" class="toc-item">💻 Code</a>
    <a href="#files" class="toc-item">📁 Files</a>
    <a href="#roadmap" class="toc-item">🗺️ Roadmap</a>
  </div>

  <div class="sidebar-card">
    <div class="sidebar-title">Key Numbers</div>
    <div class="meta-row"><span class="meta-key">Seeds</span><span class="meta-val">3</span></div>
    <div class="meta-row"><span class="meta-key">Tasks</span><span class="meta-val">1-hop, 2-hop</span></div>
    <div class="meta-row"><span class="meta-key">α values</span><span class="meta-val">0, 0.25, 0.5, 1.0</span></div>
    <div class="meta-row"><span class="meta-key">Epochs</span><span class="meta-val">80</span></div>
    <div class="meta-row"><span class="meta-key">Hutchinson K</span><span class="meta-val">20</span></div>
    <div class="meta-row"><span class="meta-key">d_model</span><span class="meta-val">128</span></div>
    <div class="meta-row"><span class="meta-key">n_layers</span><span class="meta-val">4</span></div>
  </div>

  <div class="sidebar-card">
    <div class="sidebar-title">Experiment Status</div>
    <div class="result-row"><span class="pill pill-green">✓ PASS</span><span style="font-size:13px">Pseudo-Riemannian confirmed</span></div>
    <div class="result-row"><span class="pill pill-green">✓ PASS</span><span style="font-size:13px">Lightcone flip (α=1.0)</span></div>
    <div class="result-row"><span class="pill pill-green">✓ PASS</span><span style="font-size:13px">R-law r≈−1.0</span></div>
    <div class="result-row"><span class="pill pill-green">✓ PASS</span><span style="font-size:13px">Self-amplification effect</span></div>
    <div class="result-row"><span class="pill pill-blue">→ NEXT</span><span style="font-size:13px">Geodesic Adam</span></div>
    <div class="result-row"><span class="pill pill-blue">→ NEXT</span><span style="font-size:13px">GPT-2 Lanczos test</span></div>
  </div>

  <div class="sidebar-card">
    <div class="sidebar-title">Core Equation</div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--cyan); line-height:1.9; background:rgba(0,0,0,0.3); padding:10px; border-radius:6px;">
      η = I − 2α P_t<br>
      s = QηK^T/√d<br>
      g_t = P_t ∇L<br>
      R = λ‖(I−P_t)θ‖²
    </div>
  </div>

</aside>

</div><!-- .content -->
</div><!-- .page -->

</body>
</html>

## Citation

```bibtex
@article{li2026k1,
  author  = {Li, Y. Y. N.},
  title   = {K=1 Chronogeometrodynamics: Lorentzian Geometry from Information Time},
  year    = {2026},
  doi     = {10.5281/zenodo.18949565}
}
```

## License

[MIT](LICENSE)
