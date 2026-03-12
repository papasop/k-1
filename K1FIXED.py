# ============================================================
# K=1 CHRONOGEOMETRODYNAMICS
# EXTENDED CLEAN RATIO SCAN  (fixed η_K, wide η_σ grid)
# ============================================================
#
# FIXES vs k1_clean_ratio.py:
#   - Previous logspace(1e-4, 3e-3) → max ratio ≈ 2.4 (TOO LOW)
#   - This script covers r ∈ [0.5, 27] with 16 explicit points
#     that include all theoretically interesting values
#   - Two separate Adam optimisers (no shared momentum state)
#   - Final loss = mean of last 100 steps (smoother than 50)
#
# RATIO GRID (explicit, not logspace):
#   r = η_σ / η_K ∈ {0.5, 1, 1.5, 2, 3, 4, 6, 9, 12, 15, 18, 21, 27}
#   η_K = 3e-4 fixed throughout
#   η_σ = r × η_K
#
# WHAT WE ARE LOOKING FOR:
#   Shape A  ▼  interior minimum near r=9 → STRONG support for G⁻¹ ratio
#   Shape B  ╲  monotone decreasing past r=27 → NULL (no optimum in range)
#   Shape C  ╲╱ minimum somewhere but not near 9 → theory ratio wrong
#   Shape D  ─  flat → experiment underpowered
#
# ============================================================

import math
import random
import urllib.request

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# 0. CONFIG
# ============================================================

ETA_K = 3e-4   # FIXED for entire experiment

# Explicit ratio grid — covers 0.5 to 27, dense around theory=9
RATIO_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              9.0, 12.0, 15.0, 18.0, 21.0, 27.0]

ETA_SIGMA_GRID = [r * ETA_K for r in RATIO_GRID]

THEORY_RATIO = 9.0
SEEDS        = [42, 43, 44, 45, 46]   # 5 seeds
TRAIN_STEPS  = 600
BATCH_SIZE   = 32
BLOCK_SIZE   = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 68)
print("EXTENDED CLEAN RATIO SCAN")
print(f"η_K = {ETA_K:.1e}  (fixed)")
print(f"Ratios: {RATIO_GRID}")
print(f"Seeds: {SEEDS}  |  Steps: {TRAIN_STEPS}")
print(f"Theory prediction: r = {THEORY_RATIO}")
print("=" * 68)

# ============================================================
# 1. DATA
# ============================================================

def download_text(url, max_chars=300_000):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read(max_chars * 2).decode("utf-8", errors="ignore")[:max_chars]
    except Exception as e:
        print(f"  Download failed: {e}")
        return None

def load_corpus(name):
    if name == "synthetic":
        rng = random.Random(0)
        vocab, period, n = 22, 37, 150_000
        pat    = [rng.randint(0, vocab-1) for _ in range(period)]
        tokens = torch.tensor([pat[i % period] for i in range(n)], dtype=torch.long)
        print(f"\nCorpus: synthetic ({n:,} tokens, vocab={vocab}, period={period})")
        return tokens, vocab

    elif name == "tiny-shakespeare":
        url  = ("https://raw.githubusercontent.com/karpathy/char-rnn"
                "/master/data/tinyshakespeare/input.txt")
        text = download_text(url)
        if text is None:
            # Markov-chain fallback (non-repeating, harder)
            rng = random.Random(0); vocab = 75; c = 0; out = []
            for _ in range(200_000):
                c = (c + rng.randint(-6, 6)) % vocab \
                    if rng.random() < 0.7 else rng.randint(0, vocab-1)
                out.append(c)
            tokens = torch.tensor(out, dtype=torch.long)
            print(f"\nCorpus: fallback-markov ({len(tokens):,} tokens, vocab={vocab})")
            return tokens, vocab
        chars  = sorted(set(text))
        tok    = {c: i for i, c in enumerate(chars)}
        tokens = torch.tensor([tok[c] for c in text], dtype=torch.long)
        print(f"\nCorpus: tiny-shakespeare ({len(tokens):,} tokens, vocab={len(chars)})")
        return tokens, len(chars)

    raise ValueError(name)

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x  = torch.stack([data[i:i+block_size]     for i in ix])
    y  = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ============================================================
# 2. MODEL
# ============================================================

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
    def forward(self, x): return x + self.net(x)

class Transformer(nn.Module):
    def __init__(self, vocab, d=128, n_layers=2, block=64):
        super().__init__()
        self.tok    = nn.Embedding(vocab, d)
        self.pos    = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(n_layers)])
        self.ln     = nn.LayerNorm(d)
        self.out    = nn.Linear(d, vocab, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        logits = self.out(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

# ============================================================
# 3. TWO-GROUP ADAM  (independent momentum states)
# ============================================================

class TwoGroupAdam:
    """
    Completely separate Adam optimisers for embedding (K-group)
    and non-embedding (σ-group). No shared step counter or state.
    This ensures η_K and η_σ are truly independent.
    """
    def __init__(self, model, eta_k, eta_sigma):
        k_params, s_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            if "tok" in name or "pos" in name:
                k_params.append(p)
            else:
                s_params.append(p)
        self.opt_k = torch.optim.Adam(k_params, lr=eta_k,
                                       betas=(0.9, 0.999), eps=1e-8,
                                       weight_decay=0.0)
        self.opt_s = torch.optim.Adam(s_params, lr=eta_sigma,
                                       betas=(0.9, 0.999), eps=1e-8,
                                       weight_decay=0.0)
        self.n_k = sum(p.numel() for p in k_params)
        self.n_s = sum(p.numel() for p in s_params)

    def zero_grad(self):
        self.opt_k.zero_grad()
        self.opt_s.zero_grad()

    def step(self):
        self.opt_k.step()
        self.opt_s.step()

# ============================================================
# 4. SINGLE RUN
# ============================================================

def run_once(data, vocab_size, eta_sigma, seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    model = Transformer(vocab_size).to(DEVICE)
    opt   = TwoGroupAdam(model, eta_k=ETA_K, eta_sigma=eta_sigma)

    losses = []
    model.train()
    for _ in range(TRAIN_STEPS):
        x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))

    # Average last 100 steps (reduce step-to-step noise)
    tail   = losses[-100:]
    final  = float(np.mean(tail))
    # Early (steps 100-200) and late (last 100) to see trajectory
    early  = float(np.mean(losses[100:200])) if TRAIN_STEPS >= 200 else final
    return {"final": final, "early": early, "losses": losses}

# ============================================================
# 5. MAIN LOOP
# ============================================================

CORPORA = ["synthetic", "tiny-shakespeare"]
all_grid = {}   # corpus → list of dicts, one per ratio

for corpus_name in CORPORA:
    data, vocab_size = load_corpus(corpus_name)
    grid = []

    print(f"\n{'─'*68}")
    print(f"Scanning {len(RATIO_GRID)} ratios × {len(SEEDS)} seeds "
          f"= {len(RATIO_GRID)*len(SEEDS)} runs")
    print(f"{'─'*68}")
    print(f"  {'r':>5}  {'η_σ':>8}  {'mean±std':>14}  {'early':>7}  note")

    for ratio, eta_s in zip(RATIO_GRID, ETA_SIGMA_GRID):
        runs   = [run_once(data, vocab_size, eta_s, s) for s in SEEDS]
        finals = [r["final"] for r in runs]
        earlys = [r["early"] for r in runs]
        mean_f = float(np.mean(finals))
        std_f  = float(np.std(finals))
        mean_e = float(np.mean(earlys))
        note   = " ← THEORY" if ratio == THEORY_RATIO else ""
        print(f"  r={ratio:>4.1f}  η_σ={eta_s:.2e}  "
              f"{mean_f:.4f}±{std_f:.4f}  early={mean_e:.4f}{note}")
        grid.append({"ratio": ratio, "eta_sigma": eta_s,
                     "mean": mean_f, "std": std_f,
                     "early": float(np.mean(earlys)), "runs": runs})

    all_grid[corpus_name] = grid

# ============================================================
# 6. ANALYSIS
# ============================================================

print(f"\n{'='*68}")
print("ANALYSIS")
print(f"{'='*68}")

def analyse(grid, corpus_name):
    ratios = np.array([g["ratio"] for g in grid])
    means  = np.array([g["mean"]  for g in grid])
    stds   = np.array([g["std"]   for g in grid])

    best_idx   = int(np.argmin(means))
    best_r     = ratios[best_idx]
    best_mean  = means[best_idx]
    best_std   = stds[best_idx]

    # Theory point
    t_idx  = int(np.argmin(np.abs(ratios - THEORY_RATIO)))
    t_mean = means[t_idx]
    t_std  = stds[t_idx]
    gap    = t_mean - best_mean
    gap_sigma = gap / (best_std + 1e-9)

    # Interior minimum?
    has_interior = 0 < best_idx < len(grid)-1

    # Monotone check (ignoring noise < 0.5σ)
    diffs = np.diff(means)
    sig_up   = (diffs >  stds[:-1] * 0.5).sum()
    sig_down = (diffs < -stds[:-1] * 0.5).sum()

    # Flat zone: within 1σ of best
    flat = ratios[np.abs(means - best_mean) <= best_std].tolist()

    # Slope between r=9 and r=12 (sign tells direction past theory)
    if THEORY_RATIO in ratios:
        ti = list(ratios).index(THEORY_RATIO)
        slope_after = means[min(ti+1, len(means)-1)] - means[ti]
    else:
        slope_after = 0.0

    print(f"\n  [{corpus_name}]")
    print(f"  Best ratio     : r={best_r:.1f}  "
          f"loss={best_mean:.4f}±{best_std:.4f}  "
          f"({'interior' if has_interior else 'EDGE'})")
    print(f"  Theory r=9     : loss={t_mean:.4f}±{t_std:.4f}  "
          f"gap={gap:+.4f} ({gap_sigma:+.1f}σ)")
    print(f"  Flat zone (±1σ): r ∈ {[f'{r:.0f}' for r in flat]}")
    print(f"  Sig. increases : {sig_up}  |  sig. decreases: {sig_down}")
    if slope_after > best_std * 0.3:
        print(f"  After r=9      : loss RISES → optimum near r=9")
    elif slope_after < -best_std * 0.3:
        print(f"  After r=9      : loss still falls → no optimum at r=9")
    else:
        print(f"  After r=9      : flat (Δ={slope_after:+.4f})")

    # Verdict
    if has_interior and abs(best_r - THEORY_RATIO) / THEORY_RATIO <= 0.20:
        v = "STRONG"
        detail = (f"Interior min at r={best_r:.0f}, within 20% of "
                  f"theory r={THEORY_RATIO:.0f}. Quantitative prediction supported.")
    elif has_interior and THEORY_RATIO in flat:
        v = "WEAK"
        detail = (f"Interior min at r={best_r:.0f}; r=9 in flat zone. "
                  f"Consistent but landscape too flat to discriminate.")
    elif has_interior:
        v = "WRONG RATIO"
        detail = (f"Interior min at r={best_r:.0f}, outside 20% band "
                  f"of r=9. Direction right, magnitude wrong.")
    elif sig_up == 0:
        v = "NULL (monotone)"
        detail = ("Loss decreases monotonically across all tested ratios. "
                  "Extend grid beyond r=27, or experiment underpowered.")
    else:
        v = "INCONCLUSIVE"
        detail = f"Best at grid edge (r={best_r:.0f}); extend grid."

    print(f"\n  → VERDICT: {v}")
    print(f"     {detail}")
    return v

verdicts = {}
for corpus_name, grid in all_grid.items():
    verdicts[corpus_name] = analyse(grid, corpus_name)

# ============================================================
# 7. LANDSCAPE VISUALISATION
# ============================================================

def sparkline(vals, markers=None):
    """
    Draw a sparkline. markers = {index: char} for special positions.
    """
    bars = " ▁▂▃▄▅▆▇█"
    lo, hi = min(vals), max(vals)
    if hi == lo: return "─" * len(vals)
    chars = [bars[int((v - lo) / (hi - lo) * 8)] for v in vals]
    if markers:
        for idx, ch in markers.items():
            if 0 <= idx < len(chars):
                chars[idx] = ch
    return "".join(chars)

print(f"\n{'='*68}")
print("LANDSCAPE  (left=small η_σ, right=large η_σ)")
print(f"  '9' marks theory point (r=9), '*' marks empirical best")
print(f"  Shape ▼ = good (interior min) | Shape ╲ = null (monotone)")
print(f"{'='*68}")

for corpus_name, grid in all_grid.items():
    ratios = [g["ratio"] for g in grid]
    means  = [g["mean"]  for g in grid]
    best_i  = int(np.argmin(means))
    theory_i = min(range(len(ratios)), key=lambda i: abs(ratios[i]-9.0))
    markers = {theory_i: "9", best_i: "*"}
    if theory_i == best_i:
        markers = {best_i: "★"}
    spark = sparkline(means, markers)
    print(f"\n  {corpus_name}")
    print(f"  [{spark}]")
    # Ratio axis labels
    labels = [f"{r:.0f}" for r in ratios]
    axis   = "  " + "  ".join(
        f"r={labels[i]}" if i in [0, len(labels)//2, len(labels)-1] else ""
        for i in range(len(labels))
    ).strip()
    print(f"  r: {ratios[0]:.1f} {'─'*10} {ratios[len(ratios)//2]:.0f} "
          f"{'─'*10} {ratios[-1]:.0f}")

# ============================================================
# 8. SUMMARY AND NEXT STEPS
# ============================================================

print(f"\n{'='*68}")
print("SUMMARY")
print(f"{'='*68}")
for corpus_name, v in verdicts.items():
    print(f"  {corpus_name:<25} → {v}")

print(f"""
NEXT STEPS DEPENDING ON RESULT:

  If STRONG on both corpora:
    → This is the key result. Report: "Optimal η_σ/η_K ≈ 9, matching
      G⁻¹ = diag(1, -9) prediction." Update paper as primary finding.

  If STRONG on synthetic, NULL on shakespeare:
    → Consistent with embedding-bottleneck hypothesis.
      Test: larger model (d=256, 4 layers) on shakespeare where
      non-embedding params dominate more clearly.

  If WRONG RATIO (interior min but not at 9):
    → Compute the empirically optimal ratio and work backwards:
      What σ_ref gives G₂₂ = −1/r_opt² ?
      This may refine the theory's σ_ref convention.

  If NULL (monotone) past r=27:
    → The LR-ratio experiment cannot verify the quantitative prediction.
      The next clean experiment: implement the full J_G gradient coupling
      and test whether the coupling strength d_c = 0.2451 matches the
      empirically stable range.
""")
