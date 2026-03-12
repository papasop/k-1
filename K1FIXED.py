# ============================================================
# K=1 CHRONOGEOMETRODYNAMICS
# COMPREHENSIVE RATIO EXPERIMENT  v4
# ============================================================
#
# FOUR IMPROVEMENTS vs k1_extended_scan.py:
#
#   [1] HIGH-r GRID: r ∈ {0.5,1,2,3,4,6,9,12,18,27,36,54}
#       Sees whether loss rebounds after r=9 or keeps falling.
#
#   [2] MULTIPLE η_K VALUES: {1e-4, 2e-4, 3e-4, 5e-4}
#       Computes r* = argmin_r loss(r) for each η_K separately.
#       If r* is stable ≈9 across η_K → geometric ratio is real.
#       If r* drifts with η_K → not a constant ratio, theory wrong.
#
#   [3] VAL LOSS: data split 90/10 train/val.
#       Report both train and val loss at each ratio.
#       On synthetic (memorisable) train/val may decouple.
#       On shakespeare they should track.
#
#   [4] BPE TOKENIZER for tiny-shakespeare:
#       Uses a simple custom BPE (no external deps) that builds
#       ~512-token vocab, reducing vocab size relative to char-level
#       so embeddings occupy a smaller fraction of parameters.
#       This tests whether the directional effect holds when
#       embeddings are a smaller bottleneck.
#
# STRUCTURE:
#   Experiment A: high-r grid at η_K = 3e-4 (both corpora)
#   Experiment B: multi-η_K sweep (tiny-shakespeare only)
#   Results printed with sparklines + ASCII scatter of r* vs η_K
#
# ============================================================

import math
import random
import collections
import urllib.request
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# 0. CONFIG
# ============================================================

# [1] High-r grid
HIGH_R_GRID = [0.5, 1, 2, 3, 4, 6, 9, 12, 18, 27, 36, 54]

# [2] Multiple η_K values
ETA_K_LIST  = [1e-4, 2e-4, 3e-4, 5e-4]
ETA_K_FIXED = 3e-4          # used for Experiment A

THEORY_RATIO = 9.0
SEEDS        = [42, 43, 44, 45, 46]
TRAIN_STEPS  = 600
BATCH_SIZE   = 32
BLOCK_SIZE   = 64
VAL_FRAC     = 0.10          # [3] 10% held-out val set
VAL_BATCHES  = 20            # batches to estimate val loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 68)
print("COMPREHENSIVE RATIO EXPERIMENT v4")
print(f"Device: {DEVICE}")
print(f"r grid : {HIGH_R_GRID}")
print(f"η_K list: {[f'{x:.0e}' for x in ETA_K_LIST]}")
print(f"Theory : r* = {THEORY_RATIO}")
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

# ── [4] Minimal BPE tokenizer ──────────────────────────────
class MiniBPE:
    """
    A self-contained byte-pair encoding tokenizer.
    Builds a vocab of target_vocab merges from raw text.
    No external dependencies.
    """
    def __init__(self, target_vocab: int = 512):
        self.target_vocab = target_vocab
        self.merges: List[Tuple[int, int]] = []
        self.vocab_size: int = 0

    def _get_pairs(self, ids):
        cnt = collections.Counter()
        for seq in ids:
            for a, b in zip(seq, seq[1:]):
                cnt[(a, b)] += 1
        return cnt

    def fit(self, text: str):
        # Start from byte-level (256 base tokens)
        ids = [[b for b in text.encode("utf-8", errors="replace")]
               for text in text.split("\n") if text]
        vocab = 256
        for _ in range(self.target_vocab - 256):
            pairs = self._get_pairs(ids)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            new_ids = []
            for seq in ids:
                merged, i = [], 0
                while i < len(seq):
                    if i < len(seq)-1 and (seq[i], seq[i+1]) == best:
                        merged.append(vocab); i += 2
                    else:
                        merged.append(seq[i]); i += 1
                new_ids.append(merged)
            ids = new_ids
            vocab += 1
        self.vocab_size = vocab
        self._ids_cache = ids          # store for encode reuse
        return self

    def encode(self, text: str) -> List[int]:
        ids = [b for b in text.encode("utf-8", errors="replace")]
        for (a, b), new_tok in zip(self.merges,
                                   range(256, 256 + len(self.merges))):
            merged, i = [], 0
            while i < len(ids):
                if i < len(ids)-1 and ids[i] == a and ids[i+1] == b:
                    merged.append(new_tok); i += 2
                else:
                    merged.append(ids[i]); i += 1
            ids = merged
        return ids

def load_corpus(name, bpe=False):
    """Returns (train_tokens, val_tokens, vocab_size, desc)."""
    if name == "synthetic":
        rng = random.Random(0)
        vocab, period, n = 22, 37, 150_000
        pat    = [rng.randint(0, vocab-1) for _ in range(period)]
        tokens = torch.tensor([pat[i % period] for i in range(n)],
                              dtype=torch.long)
        split  = int(len(tokens) * (1 - VAL_FRAC))
        return tokens[:split], tokens[split:], vocab, "synthetic"

    elif name == "tiny-shakespeare":
        url  = ("https://raw.githubusercontent.com/karpathy/char-rnn"
                "/master/data/tinyshakespeare/input.txt")
        text = download_text(url)
        if text is None:
            # Markov fallback
            rng = random.Random(0); vocab = 75; c = 0; out = []
            for _ in range(200_000):
                c = (c + rng.randint(-6,6)) % vocab if rng.random()<.7 \
                    else rng.randint(0, vocab-1)
                out.append(c)
            tokens = torch.tensor(out, dtype=torch.long)
            split  = int(len(tokens) * (1 - VAL_FRAC))
            return tokens[:split], tokens[split:], vocab, "fallback-markov"

        if bpe:
            print("  Building BPE tokenizer (target_vocab=512)...")
            bpe_tok = MiniBPE(target_vocab=512).fit(text)
            ids     = bpe_tok.encode(text)
            vocab   = bpe_tok.vocab_size
            desc    = f"shakespeare-bpe (vocab={vocab})"
        else:
            chars = sorted(set(text))
            tok   = {c: i for i, c in enumerate(chars)}
            ids   = [tok[c] for c in text]
            vocab = len(chars)
            desc  = f"shakespeare-char (vocab={vocab})"

        tokens = torch.tensor(ids, dtype=torch.long)
        split  = int(len(tokens) * (1 - VAL_FRAC))
        print(f"  {desc}: {len(tokens):,} tokens")
        return tokens[:split], tokens[split:], vocab, desc

    raise ValueError(name)

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x  = torch.stack([data[i:i+block_size]     for i in ix])
    y  = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_val_loss(model, val_data, block_size, batch_size,
                      n_batches, device):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val_data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))

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
        for blk in self.blocks: x = blk(x)
        x = self.ln(x)
        logits = self.out(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def emb_frac(self):
        """Fraction of parameters in embedding layers."""
        total = sum(p.numel() for p in self.parameters())
        emb   = sum(p.numel() for n, p in self.named_parameters()
                   if "tok" in n or "pos" in n)
        return emb / total

# ============================================================
# 3. TWO-GROUP ADAM
# ============================================================

class TwoGroupAdam:
    def __init__(self, model, eta_k, eta_sigma):
        k_params, s_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            (k_params if ("tok" in name or "pos" in name) else s_params).append(p)
        self.opt_k = torch.optim.Adam(k_params, lr=eta_k,
                                       betas=(0.9,0.999), eps=1e-8,
                                       weight_decay=0.0)
        self.opt_s = torch.optim.Adam(s_params, lr=eta_sigma,
                                       betas=(0.9,0.999), eps=1e-8,
                                       weight_decay=0.0)

    def zero_grad(self): self.opt_k.zero_grad(); self.opt_s.zero_grad()
    def step(self):      self.opt_k.step();      self.opt_s.step()

# ============================================================
# 4. SINGLE RUN
# ============================================================

def run_once(train_data, val_data, vocab_size, eta_k, eta_sigma, seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    model = Transformer(vocab_size).to(DEVICE)
    opt   = TwoGroupAdam(model, eta_k=eta_k, eta_sigma=eta_sigma)

    train_losses, val_losses = [], []
    for step in range(TRAIN_STEPS):
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        train_losses.append(float(loss.item()))

        if step % 100 == 99:   # val every 100 steps
            vl = estimate_val_loss(model, val_data,
                                   BLOCK_SIZE, BATCH_SIZE,
                                   VAL_BATCHES, DEVICE)
            val_losses.append(vl)

    train_final = float(np.mean(train_losses[-100:]))
    val_final   = float(np.mean(val_losses[-3:])) if val_losses else float("nan")
    return {"train": train_final, "val": val_final,
            "train_hist": train_losses, "val_hist": val_losses}

# ============================================================
# 5. EXPERIMENT A: HIGH-r GRID
# ============================================================

def run_grid(train_data, val_data, vocab_size, eta_k,
             r_grid, seeds, label):
    print(f"\n  {'─'*60}")
    print(f"  {label}   η_K={eta_k:.1e}")
    print(f"  {'r':>6}  {'η_σ':>9}  {'train':>8}  {'val':>8}  "
          f"{'t±std':>9}  {'v±std':>9}")
    print(f"  {'─'*60}")

    grid = []
    for r in r_grid:
        eta_s = r * eta_k
        runs  = [run_once(train_data, val_data, vocab_size,
                          eta_k, eta_s, s) for s in seeds]
        t_arr = [x["train"] for x in runs]
        v_arr = [x["val"]   for x in runs]
        t_m, t_s = float(np.mean(t_arr)), float(np.std(t_arr))
        v_m, v_s = float(np.mean(v_arr)), float(np.std(v_arr))
        mark = " ← r=9" if r == 9 else ""
        print(f"  r={r:>4.0f}  η_σ={eta_s:.2e}  "
              f"{t_m:.4f}  {v_m:.4f}  "
              f"±{t_s:.4f}  ±{v_s:.4f}{mark}")
        grid.append({"r": r, "eta_s": eta_s,
                     "t_m": t_m, "t_s": t_s,
                     "v_m": v_m, "v_s": v_s})
    return grid

def find_optimum(grid, key="v_m"):
    vals   = np.array([g[key] for g in grid])
    ratios = np.array([g["r"] for g in grid])
    best_i = int(np.argmin(vals))
    has_interior = 0 < best_i < len(grid)-1
    # Check rebound after theory point
    t_i = list(ratios).index(9) if 9 in ratios else -1
    rebound = (t_i >= 0 and t_i < len(grid)-1
               and vals[t_i+1] > vals[t_i] + grid[t_i]["v_s"]*0.3)
    return {"best_r": ratios[best_i],
            "best_val": vals[best_i],
            "interior": has_interior,
            "rebound_after_9": rebound,
            "vals": vals,
            "ratios": ratios}

# ── sparkline ────────────────────────────────────────────────
def sparkline(vals, special=None):
    bars = " ▁▂▃▄▅▆▇█"
    lo, hi = min(vals), max(vals)
    chars  = [bars[int((v-lo)/(hi-lo)*8+0.5)
                   if hi > lo else 0] for v in vals]
    if special:
        for i, ch in special.items():
            if 0 <= i < len(chars): chars[i] = ch
    return "".join(chars)

def verdict(res, corpus):
    r = res["best_r"]
    interior = res["interior"]
    rebound  = res["rebound_after_9"]
    if interior and abs(r - 9) / 9 <= 0.20:
        return "STRONG", f"interior min at r={r:.0f} (within 20% of r=9)"
    if rebound and not interior:
        return "REBOUND-NEAR-9", f"loss rises after r=9 but min at edge r={r:.0f}"
    if interior:
        return "WRONG-RATIO", f"interior min at r={r:.0f}, not near r=9"
    return "NULL", f"monotone to r={r:.0f}; no interior min"

# ============================================================
# 6. RUN EXPERIMENT A
# ============================================================

print(f"\n{'='*68}")
print("EXPERIMENT A: HIGH-r GRID  (η_K = 3e-4 fixed)")
print(f"{'='*68}")

EXP_A_CORPORA = [
    ("synthetic",       False),
    ("tiny-shakespeare", False),   # char-level
    ("tiny-shakespeare", True),    # BPE  [4]
]

exp_a_results = {}
for corpus_name, use_bpe in EXP_A_CORPORA:
    key = f"{corpus_name}{'_bpe' if use_bpe else '_char'}"
    print(f"\n[{key}]")
    tr, va, vocab, desc = load_corpus(corpus_name, bpe=use_bpe)
    model_tmp = Transformer(vocab).to(DEVICE)
    print(f"  Embedding frac: {model_tmp.emb_frac():.1%}")
    del model_tmp
    grid = run_grid(tr, va, vocab, ETA_K_FIXED, HIGH_R_GRID, SEEDS, desc)
    exp_a_results[key] = (grid, tr, va, vocab, desc)

# ============================================================
# 7. RUN EXPERIMENT B: MULTIPLE η_K
# ============================================================

print(f"\n{'='*68}")
print("EXPERIMENT B: MULTIPLE η_K  (tiny-shakespeare char-level)")
print(f"{'='*68}")

tr_sh, va_sh, vocab_sh, desc_sh = load_corpus("tiny-shakespeare", bpe=False)
exp_b_results = {}

for eta_k in ETA_K_LIST:
    grid = run_grid(tr_sh, va_sh, vocab_sh, eta_k,
                    HIGH_R_GRID, SEEDS, desc_sh)
    exp_b_results[eta_k] = grid

# ============================================================
# 8. ANALYSIS AND VERDICT
# ============================================================

print(f"\n{'='*68}")
print("ANALYSIS")
print(f"{'='*68}")

# --- Experiment A ---
print(f"\n  EXPERIMENT A  (η_K={ETA_K_FIXED:.0e} fixed)")
print(f"  {'corpus':<28} {'best_r':>7} {'interior':>9} {'rebound@9':>10} verdict")
print(f"  {'─'*70}")

for key, (grid, *_) in exp_a_results.items():
    res  = find_optimum(grid, key="v_m")
    v, msg = verdict(res, key)
    # sparkline
    ratios = res["ratios"]
    t_i  = list(ratios).index(9) if 9 in ratios else -1
    b_i  = list(ratios).index(res["best_r"]) if res["best_r"] in ratios else -1
    sp   = {"9": t_i, "*": b_i}
    if t_i == b_i: sp = {"★": t_i}
    spark = sparkline(res["vals"],
                      special={v: k for k, v in sp.items()})
    print(f"  {key:<28} r={res['best_r']:>4.0f}  "
          f"{'YES' if res['interior'] else 'NO':>9}  "
          f"{'YES' if res['rebound_after_9'] else 'NO':>10}  {v}")
    print(f"  {'':28} [{spark}]  {msg}")

# --- Experiment B ---
print(f"\n  EXPERIMENT B  (multiple η_K, tiny-shakespeare)")
print(f"  {'η_K':>8}  {'best_r':>7}  {'interior':>9}  verdict")
print(f"  {'─'*50}")

b_rstar = []
for eta_k, grid in exp_b_results.items():
    res  = find_optimum(grid, key="v_m")
    v, msg = verdict(res, "shakespeare")
    b_rstar.append((eta_k, res["best_r"]))
    print(f"  η_K={eta_k:.0e}  r*={res['best_r']:>4.0f}  "
          f"{'YES' if res['interior'] else 'NO':>9}  {v}")

# Stability of r* across η_K
rstar_vals = [r for _, r in b_rstar]
rstar_std  = float(np.std(rstar_vals))
rstar_mean = float(np.mean(rstar_vals))
print(f"\n  r* across η_K: mean={rstar_mean:.1f}  std={rstar_std:.1f}")
if rstar_std < 2:
    print(f"  → r* is STABLE across η_K (std<2). "
          f"Ratio is approximately constant.")
    if abs(rstar_mean - 9) / 9 < 0.25:
        print(f"  → Mean r*={rstar_mean:.1f} within 25% of theory r=9. "
              f"STRONG SUPPORT.")
    else:
        print(f"  → Mean r*={rstar_mean:.1f} not near r=9. "
              f"Stable ratio but wrong value.")
else:
    print(f"  → r* VARIES with η_K (std={rstar_std:.1f}). "
          f"Ratio is not a geometric constant.")

# ASCII scatter: r* vs η_K
print(f"\n  Scatter: r* vs η_K")
print(f"  (each row = one η_K value)")
for eta_k, rstar in b_rstar:
    bar = "─" * int(rstar / 2)
    print(f"  η_K={eta_k:.0e}  r*={rstar:>4.0f}  |{bar}●  "
          f"{'← theory' if abs(rstar-9)<1 else ''}")

# ============================================================
# 9. OVERALL VERDICT
# ============================================================

print(f"\n{'='*68}")
print("OVERALL VERDICT")
print(f"{'='*68}")
print(f"""
Three questions, each with a binary answer:

  Q1: Is there an interior optimum (not monotone)?
  Q2: Is the optimum near r=9 (within 25%)?
  Q3: Is r* stable across η_K values?

  Q1+Q2+Q3 = YES → STRONG: ratio=9 prediction confirmed
  Q1+Q3    = YES, Q2=NO → ratio is real but J_G gives wrong value
  Q1       = NO  → monotone landscape; ratio experiment uninformative
  Q3       = NO  → r* depends on η_K; not a geometric constant

NEXT STEPS:
  If monotone: implement full J_G gradient controller and test d_c.
  If Q2=NO:    compute empirical r* and back-solve for σ_ref that
               gives G₂₂ = -1/r*². Refine theory.
  If Q1+Q2+Q3: update paper to report this as primary empirical result.
""")
