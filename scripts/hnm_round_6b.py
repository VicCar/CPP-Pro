"""One round of hard negative mining on frozen ESM-C 6B + DeepSet.

Design (approved):
  - Fixed screening set: 10,000 unused negatives, held out FOREVER (never mined,
    never trained). Measures false-positive rate, comparable across rounds.
  - Mining pool: the remaining FILLED unused negatives (currently ~15,785 of the
    25,785 embedded so far). Each round scores the pool and adds the top-K hardest
    (highest P(CPP)) to train, excluding any already mined.
  - v6 val (554) is FIXED for early stopping — never contaminated with hard negs.
  - v6 test (570) is the benchmark, measured every round.
  - 3-seed DeepSet ensemble (mean prob) for stable mining + FP-rate.

State persists in results/hnm_state.json. Run once per round:

    ~/miniconda3/bin/python CPPro/CPPro_current/scripts/hnm_round_6b.py          # round 0 = baseline
    ~/miniconda3/bin/python CPPro/CPPro_current/scripts/hnm_round_6b.py          # round 1
    ...                                                                              # repeat

Each call trains, mines, scores, appends a row to results/hnm_rounds.csv, and
advances the round counter. Round 0 mines nothing (baseline); rounds 1+ mine top-K.
"""

from __future__ import annotations
import argparse, json, time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from head_sweep_6b import DeepSetHead, HEADS, D, L_MAX

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / 'dataset' / 'splits_v6'
RESULTS = ROOT / 'results'

# --- backbone-configurable globals (default = 6B + DeepSet; configure() rebinds them) ---
SPLITS_6B = ROOT / 'embeddings' / 'frozen_6b.h5'           # split (train+val+test) embeddings
UNUSED_6B = ROOT / 'embeddings' / 'frozen_6b_unused.h5'    # unused-negative pool embeddings
STATE_FILE = RESULTS / 'hnm_state.json'
ROUNDS_CSV = RESULTS / 'hnm_rounds.csv'
CACHE_DIR = ROOT / 'embeddings' / 'hnm_score_cache'        # uncompressed score cache (set below too)
HEAD_CLS = DeepSetHead
DIM = D
BACKBONE = '6b'

_CFG = {
    '6b':   dict(splits='frozen_6b.h5',   unused='frozen_6b_unused.h5',
                 state='hnm_state.json',  rounds='hnm_rounds.csv',
                 cache='hnm_score_cache', head='deepset', dim=2560),
    '600m': dict(splits='frozen_600m.h5', unused='frozen_600m_unused.h5',
                 state='hnm_state_600m.json', rounds='hnm_rounds_600m.csv',
                 cache='hnm_score_cache_600m', head='seqcnn', dim=1152),
}


def configure(backbone):
    """Rebind module globals for the chosen backbone. Default 6B leaves behavior unchanged."""
    global SPLITS_6B, UNUSED_6B, STATE_FILE, ROUNDS_CSV, CACHE_DIR, HEAD_CLS, DIM, BACKBONE
    c = _CFG[backbone]
    BACKBONE = backbone
    SPLITS_6B = ROOT / 'embeddings' / c['splits']
    UNUSED_6B = ROOT / 'embeddings' / c['unused']
    STATE_FILE = RESULTS / c['state']
    ROUNDS_CSV = RESULTS / c['rounds']
    CACHE_DIR = ROOT / 'embeddings' / c['cache']
    HEAD_CLS = HEADS[c['head']]
    DIM = c['dim']

SCREENING_SIZE = 10_000
MINING_POOL_SIZE = 40_000                                  # uncapped: take all ~39.6k remaining filled negs
TOP_K = 300                                                # max hard negs added per round
CONF_FLOOR = 0.8                                           # only mine negatives with P(CPP) >= this
MAX_NEG_RATIO = 2.0                                        # never let total negatives exceed 2x positives
SEEDS = [0, 1, 2, 3, 4]                                    # 5 replicates per round
THRESHOLDS = [0.5]                                         # single operating point for FP/TPR


# ----------------------- embedding helpers -----------------------
def mask_from_lengths(lengths, L=L_MAX):
    m = np.zeros((len(lengths), L), dtype=np.float32)
    for i, n in enumerate(lengths):
        m[i, :int(n)] = 1.0
    return m


def load_split(split: str):
    """Return per_token, mask, labels, seqs for a v6 split (from frozen_6b.h5)."""
    with h5py.File(SPLITS_6B, 'r') as f:
        sp = np.array([s.decode() for s in f['split'][:]])
        sel = np.where(sp == split)[0]
        pt = f['per_token'][sel].astype(np.float32)
        ln = f['lengths'][sel].astype(np.int32)
        lab = f['labels'][sel].astype(np.int64)
        seqs = np.array([s.decode() for s in f['sequences'][sel]])
    return pt, mask_from_lengths(ln), lab, seqs


def filled_unused_indices():
    """Indices of unused negatives that actually have embeddings (mean_pool != 0)."""
    with h5py.File(UNUSED_6B, 'r') as f:
        mp = f['mean_pool'][:]
    return np.where(np.abs(mp).sum(axis=1) > 0)[0]


def load_unused(indices, dtype=np.float32):
    """Return per_token, mask, seqs for the given unused-pool row indices.
    dtype=np.float16 keeps memory low for large pools (score_array up-casts per batch)."""
    indices = np.sort(np.asarray(indices))
    with h5py.File(UNUSED_6B, 'r') as f:
        pt = f['per_token'][indices].astype(dtype)
        ln = f['lengths'][indices].astype(np.int32)
        seqs = np.array([s.decode() for s in f['sequences'][indices]])
    return pt, mask_from_lengths(ln), seqs


# ----------------------- one-time uncompressed score cache -----------------------
# The unused h5 stores per_token gzip-compressed with per-row chunks, so pulling the
# fixed screening (10k) + pool (20k) sets decompresses 30k chunks (~7.6 min). Those
# index sets are FIXED across rounds, so we decompress ONCE into uncompressed .npy
# (memmap-loaded thereafter) — every round then scores in seconds.
# CACHE_DIR is a configurable global (set in the header / by configure()).

def build_or_load_score_cache(screening_idx, pool_idx):
    """Return (screen_X, screen_M, pool_X, pool_M) as fp16 memmaps. Builds the cache
    from the gzip h5 on first call (slow, ~8 min); instant memmap load afterward."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    idx_path = CACHE_DIR / 'indices.npz'
    files = {k: CACHE_DIR / f'{k}.npy' for k in ('screen_X', 'screen_M', 'pool_X', 'pool_M')}

    # Validate existing cache against the requested index sets
    valid = idx_path.exists() and all(p.exists() for p in files.values())
    if valid:
        saved = np.load(idx_path)
        valid = (np.array_equal(saved['screening_idx'], np.asarray(screening_idx)) and
                 np.array_equal(saved['pool_idx'], np.asarray(pool_idx)))
    if valid:
        print(f"[cache] loading score cache from {CACHE_DIR} (memmap)")
        return (np.load(files['screen_X'], mmap_mode='r'), np.load(files['screen_M'], mmap_mode='r'),
                np.load(files['pool_X'], mmap_mode='r'), np.load(files['pool_M'], mmap_mode='r'))

    print(f"[cache] building score cache (one-time gzip-h5 decompress, batched to memmap)...")
    t0 = time.time()
    _decompress_to_memmap(screening_idx, files['screen_X'], files['screen_M'])
    _decompress_to_memmap(pool_idx, files['pool_X'], files['pool_M'])
    np.savez(idx_path, screening_idx=np.asarray(screening_idx), pool_idx=np.asarray(pool_idx))
    print(f"[cache] built in {time.time()-t0:.1f}s → {CACHE_DIR}")
    return (np.load(files['screen_X'], mmap_mode='r'), np.load(files['screen_M'], mmap_mode='r'),
            np.load(files['pool_X'], mmap_mode='r'), np.load(files['pool_M'], mmap_mode='r'))


def _decompress_to_memmap(indices, x_path, m_path, batch=4000):
    """Decompress per_token for `indices` from the gzip h5 into an on-disk fp16 memmap,
    reading in batches so peak RAM stays ~1.3 GB regardless of total size (avoids OOM)."""
    indices = np.sort(np.asarray(indices))
    N = len(indices)
    Xmm = np.lib.format.open_memmap(x_path, mode='w+', dtype=np.float16, shape=(N, L_MAX, DIM))
    lengths = np.zeros(N, dtype=np.int32)
    with h5py.File(UNUSED_6B, 'r') as f:
        pt, ln = f['per_token'], f['lengths']
        for s in range(0, N, batch):
            idx = indices[s:s + batch]
            Xmm[s:s + len(idx)] = pt[idx].astype(np.float16)
            lengths[s:s + len(idx)] = ln[idx]
            Xmm.flush()
    del Xmm                                            # close the memmap
    np.save(m_path, mask_from_lengths(lengths))


def v6_cluster_factor():
    """seq -> cluster down-weight factor (1 / id80_cluster_size) for original v6 train.
    This is the cluster-variety factor; the class-balance factor is recomputed per round."""
    df = pd.read_csv(SPLITS_DIR / 'train.csv')
    if 'id80_size' in df:
        return {s: 1.0 / max(int(n), 1) for s, n in zip(df.sequence, df.id80_size)}
    return {s: 1.0 for s in df.sequence}


def balanced_weights(base_w, labels):
    """Recompute class weights so loss-mass is equal across classes, while keeping
    each row's cluster factor. Returns final per-row weights.
        final_w[i] = base_w[i] * class_weight[label[i]]
        class_weight[c] = total_base / (2 * sum_base_in_class_c)
    => sum of final weights is equal for positives and negatives."""
    base_w = np.asarray(base_w, dtype=np.float64)
    labels = np.asarray(labels)
    sum_pos = base_w[labels == 1].sum()
    sum_neg = base_w[labels == 0].sum()
    total = sum_pos + sum_neg
    cw_pos = total / (2 * max(sum_pos, 1e-9))
    cw_neg = total / (2 * max(sum_neg, 1e-9))
    w = base_w * np.where(labels == 1, cw_pos, cw_neg)
    return w.astype(np.float32), float(cw_pos), float(cw_neg)


# ----------------------- DeepSet train / score -----------------------
def train_deepset(seed, Xtr, Mtr, ytr, wtr, Xva, Mva, yva,
                  device='cuda', batch_size=256, max_epochs=100, patience=10, lr=1e-3):
    """GPU-resident training: all train tensors are moved to the GPU ONCE, then
    batches are sliced by index — no per-batch host->device copies. The DeepSet head
    is tiny, so without this the GPU starves waiting on 21 MB/batch transfers."""
    torch.manual_seed(seed); np.random.seed(seed)
    # Preload everything to GPU once (~1.7-2.4 GB for train at fp16; fits in 8 GB)
    Xtr_t = torch.from_numpy(Xtr).to(device)
    Mtr_t = torch.from_numpy(Mtr).to(device)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(device)
    wtr_t = torch.from_numpy(wtr.astype(np.float32)).to(device)
    Xva_t = torch.from_numpy(Xva).to(device); Mva_t = torch.from_numpy(Mva).to(device)
    n = Xtr_t.shape[0]

    model = HEAD_CLS(in_dim=DIM).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    best_mcc, best_state, left = -1.0, None, patience
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            optim.zero_grad(set_to_none=True)
            loss = (bce(model(Xtr_t[idx], Mtr_t[idx]), ytr_t[idx]) * wtr_t[idx]).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva_t, Mva_t)).cpu().numpy()
        mcc = matthews_corrcoef(yva, (pv >= 0.5).astype(int))
        if mcc > best_mcc:
            best_mcc, left = mcc, patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            left -= 1
            if left <= 0: break
    model.load_state_dict(best_state)
    # free the resident train tensors before the next seed
    del Xtr_t, Mtr_t, ytr_t, wtr_t, Xva_t, Mva_t
    torch.cuda.empty_cache()
    return model, best_mcc


def score_batched(model, h5_path, indices, device='cuda', batch_size=256):
    """Score unused-pool rows by index, reading per_token from h5 in batches."""
    indices = np.sort(np.asarray(indices))
    probs = np.zeros(len(indices), dtype=np.float32)
    model.eval()
    with h5py.File(h5_path, 'r') as f:
        pt_ds = f['per_token']; ln_ds = f['lengths']
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            X = pt_ds[idx].astype(np.float32)
            M = mask_from_lengths(ln_ds[idx].astype(np.int32))
            with torch.no_grad():
                p = torch.sigmoid(model(torch.from_numpy(X).to(device),
                                        torch.from_numpy(M).to(device))).cpu().numpy()
            probs[start:start + len(idx)] = p
    return probs


def score_array(model, X, M, device='cuda', batch_size=512):
    """Score an in-RAM array. Up-casts fp16 → float32 per batch so the pool/screening
    arrays can be held in RAM at half size."""
    probs = np.zeros(len(X), dtype=np.float32)
    model.eval()
    for s in range(0, len(X), batch_size):
        with torch.no_grad():
            xb = torch.from_numpy(X[s:s+batch_size]).float().to(device)
            mb = torch.from_numpy(M[s:s+batch_size]).float().to(device)
            p = torch.sigmoid(model(xb, mb)).cpu().numpy()
        probs[s:s+len(p)] = p
    return probs


# ----------------------- state -----------------------
def init_state():
    # 600M reuses the EXACT 6B screening/pool split for a clean cross-backbone comparison.
    if BACKBONE != '6b':
        ref = RESULTS / 'hnm_state.json'
        if ref.exists():
            r = json.loads(ref.read_text())
            print(f"[init] reusing 6B screening/pool split from {ref.name}")
            return {'round': 0, 'screening_idx': r['screening_idx'],
                    'mining_pool_idx': r['mining_pool_idx'], 'mined_idx': [], 'history': []}
        print("[init] WARNING: 6B state not found; drawing a fresh split (seed 0, same filled set)")
    filled = filled_unused_indices()
    rng = np.random.default_rng(0)
    perm = rng.permutation(filled)
    screening = sorted(perm[:SCREENING_SIZE].tolist())
    pool = sorted(perm[SCREENING_SIZE:SCREENING_SIZE + MINING_POOL_SIZE].tolist())  # capped
    print(f"[init] filled unused embeddings: {len(filled)}")
    print(f"[init] screening set: {len(screening)}  |  mining pool: {len(pool)} "
          f"(capped at {MINING_POOL_SIZE}; {len(filled) - SCREENING_SIZE - len(pool)} unused)")
    return {'round': 0, 'screening_idx': screening, 'mining_pool_idx': pool,
            'mined_idx': [], 'history': []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top-k', type=int, default=TOP_K, help='Max hard negs added per round.')
    ap.add_argument('--conf-floor', type=float, default=CONF_FLOOR,
                    help='Only mine negatives with P(CPP) >= this.')
    ap.add_argument('--max-neg-ratio', type=float, default=MAX_NEG_RATIO,
                    help='Never let total negatives exceed this multiple of positives.')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--reset', action='store_true', help='Wipe state and start at round 0.')
    ap.add_argument('--backbone', default='6b', choices=list(_CFG.keys()),
                    help='6b (DeepSet) or 600m (seqcnn). Selects embeddings/head/state files.')
    args = ap.parse_args()
    configure(args.backbone)
    print(f"[backbone] {BACKBONE}  head={HEAD_CLS.__name__}  dim={DIM}  state={STATE_FILE.name}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else init_state()
    rnd = state['round']
    print(f"\n=== HNM round {rnd} ===")

    # ---- build train: v6 train + mined hard negatives so far ----
    Xtr, Mtr, ytr, seqtr = load_split('train')
    cluster_factor = v6_cluster_factor()
    base_w = np.array([cluster_factor.get(s, 1.0) for s in seqtr], dtype=np.float32)  # original train
    if state['mined_idx']:
        Xh, Mh, seqh = load_unused(state['mined_idx'])
        Xtr = np.concatenate([Xtr, Xh]); Mtr = np.concatenate([Mtr, Mh])
        ytr = np.concatenate([ytr, np.zeros(len(Xh), dtype=ytr.dtype)])
        base_w = np.concatenate([base_w, np.ones(len(Xh), dtype=np.float32)])  # hard negs = singleton clusters
    # Recompute class weights each round so the (growing) negative set never dominates the loss
    wtr, cw_pos, cw_neg = balanced_weights(base_w, ytr)
    n_pos = int((ytr == 1).sum()); n_neg = int((ytr == 0).sum())
    print(f"[train] {len(Xtr)} rows ({n_pos}+ / {n_neg}- = 1:{n_neg/max(n_pos,1):.2f}) "
          f"incl {len(state['mined_idx'])} mined hard negs")
    print(f"[loss]  class_weight  pos={cw_pos:.3f}  neg={cw_neg:.3f}  (loss-mass balanced)")

    Xva, Mva, yva, _ = load_split('val')
    Xte, Mte, yte, _ = load_split('test')

    # ---- score-prep: load the FIXED screening + full pool from the one-time cache ----
    screening_idx = np.array(state['screening_idx'])
    pool_idx_full = np.array(sorted(state['mining_pool_idx']))          # FULL pool (fixed across rounds)
    screen_X, screen_M, pool_X, pool_M = build_or_load_score_cache(screening_idx, pool_idx_full)
    mined_set = set(state['mined_idx'])
    not_mined = np.array([i not in mined_set for i in pool_idx_full])   # selection mask (exclude mined)

    # ---- train 3-seed ensemble ----
    test_probs, screen_probs, pool_probs, val_mccs = [], [], [], []
    for seed in SEEDS:
        t0 = time.time()
        model, vmcc = train_deepset(seed, Xtr, Mtr, ytr, wtr, Xva, Mva, yva, device=args.device)
        t_tr = time.time() - t0
        val_mccs.append(vmcc)
        ts = time.time()
        test_probs.append(score_array(model, Xte.astype(np.float16), Mte, device=args.device))
        screen_probs.append(score_array(model, screen_X, screen_M, device=args.device))
        pool_probs.append(score_array(model, pool_X, pool_M, device=args.device))
        print(f"  seed {seed}: val_mcc={vmcc:.4f}  train={t_tr:.1f}s  score={time.time()-ts:.1f}s")

    test_p = np.mean(test_probs, axis=0)
    screen_p = np.mean(screen_probs, axis=0)
    pool_p = np.mean(pool_probs, axis=0)

    # ---- metrics ----
    test_mcc = matthews_corrcoef(yte, (test_p >= 0.5).astype(int))
    test_f1 = f1_score(yte, (test_p >= 0.5).astype(int))
    test_auc = roc_auc_score(yte, test_p)
    pos_mask = (yte == 1)
    rec = {'round': rnd, 'n_train': len(Xtr), 'n_mined_total': len(state['mined_idx']),
           'val_mcc': float(np.mean(val_mccs)),
           'test_mcc': float(test_mcc), 'test_f1': float(test_f1), 'test_auc': float(test_auc)}
    # Ensemble metrics (mean prob across seeds) at threshold 0.5
    for thr in THRESHOLDS:
        rec[f'fp_rate_screen@{thr}'] = float((screen_p >= thr).mean())
        rec[f'tpr_test@{thr}'] = float((test_p[pos_mask] >= thr).mean())
    # Per-seed FP rate + TPR (each seed scored individually) → mean ± std for error bars
    fp_per_seed = [float((sp >= 0.5).mean()) for sp in screen_probs]
    tpr_per_seed = [float((tp[pos_mask] >= 0.5).mean()) for tp in test_probs]
    rec['fp_rate_seed_mean'] = float(np.mean(fp_per_seed))
    rec['fp_rate_seed_std'] = float(np.std(fp_per_seed))
    rec['tpr_seed_mean'] = float(np.mean(tpr_per_seed))
    rec['tpr_seed_std'] = float(np.std(tpr_per_seed))
    rec['fp_per_seed'] = ';'.join(f'{x:.4f}' for x in fp_per_seed)
    rec['tpr_per_seed'] = ';'.join(f'{x:.4f}' for x in tpr_per_seed)
    print(f"  test MCC={test_mcc:.4f}  F1={test_f1:.4f}  AUC={test_auc:.4f}")
    print(f"  FP rate (per-seed): {rec['fp_rate_seed_mean']:.4f} ± {rec['fp_rate_seed_std']:.4f}  "
          f"[{rec['fp_per_seed']}]")
    print(f"  TPR     (per-seed): {rec['tpr_seed_mean']:.4f} ± {rec['tpr_seed_std']:.4f}  "
          f"[{rec['tpr_per_seed']}]")

    # ---- mine hard negatives for NEXT round ----
    # Recipe: confidence FLOOR (>= CONF_FLOOR), capped at TOP_K, and a 1:MAX_NEG_RATIO
    # ceiling on total negatives. Naturally terminates when nothing qualifies / ceiling hit.
    # pool_p covers the FULL fixed pool; mask out already-mined so they can't be reselected.
    pool_p_sel = pool_p.copy()
    pool_p_sel[~not_mined] = -1.0
    order = np.argsort(-pool_p_sel)                              # positions into pool_idx_full, prob desc
    above_floor = order[pool_p_sel[order] >= args.conf_floor]    # only confident false positives
    capped = above_floor[:args.top_k]                            # cap batch size
    # 1:MAX_NEG_RATIO ceiling — how many more negatives can we add?
    max_total_neg = int(args.max_neg_ratio * n_pos)
    room = max(0, max_total_neg - n_neg)
    selected = capped[:room]
    new_mined = pool_idx_full[selected].tolist()

    rec['mined_this_round'] = len(new_mined)
    rec['n_above_floor'] = int(len(above_floor))
    rec['neg_room_to_ceiling'] = int(room)
    rec['mined_min_prob'] = float(pool_p[selected[-1]]) if len(selected) else float('nan')
    rec['mined_max_prob'] = float(pool_p[selected[0]]) if len(selected) else float('nan')
    print(f"[mine] {len(above_floor)} negs >= {args.conf_floor};  ceiling room={room} "
          f"(max total neg {max_total_neg} @ 1:{args.max_neg_ratio});  "
          f"selected {len(new_mined)} "
          f"(P(CPP) {rec['mined_min_prob']:.3f}-{rec['mined_max_prob']:.3f})")
    if len(new_mined) == 0:
        print("[mine] nothing left to mine (floor unmet or ceiling reached) — HNM has converged.")

    # save mined sequences for this round
    if len(new_mined):
        _, _, mined_seqs = load_unused(new_mined)
        pd.DataFrame({'unused_idx': new_mined, 'sequence': mined_seqs,
                      'pool_prob': pool_p[selected]}).to_csv(
            RESULTS / f'hnm_round{rnd}_hard_negatives.csv', index=False)

    # ---- advance state ----
    state['history'].append(rec)
    state['mined_idx'] = sorted(set(state['mined_idx']) | set(new_mined))
    state['round'] = rnd + 1
    STATE_FILE.write_text(json.dumps(state, indent=2))

    # append to rounds CSV
    hist_df = pd.DataFrame(state['history'])
    hist_df.to_csv(ROUNDS_CSV, index=False)
    print(f"\n[done] round {rnd} logged. next round will train on {len(Xtr) + len(new_mined)} rows.")
    print(f"       state: {STATE_FILE}")
    print(f"       rounds log: {ROUNDS_CSV}")


if __name__ == '__main__':
    main()
