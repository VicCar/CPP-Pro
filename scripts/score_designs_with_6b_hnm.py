"""Score BoltzGen designs with ESM-C 6B (Forge) + the HNM-trained DeepSet head.

Loads the saved Round-N HNM ensemble (5 seeds) from CPPro_current/checkpoints/hnm_round<N>/.
Round 2 is the documented best per CPPro_current/results/hnm_fp_rate.md:
  FP rate 3.8% (vs round 0 baseline 7.9%) and test MCC 0.877.

The checkpoints are produced ONCE by train_save_hnm_ensemble.py — this script then
just loads + scores (no training).

Pipeline:
  1. Read input CSV; select BG-passers (default).
  2. Forge-embed each unique sequence per-token (cached → re-runs free).
  3. Load 5 DeepSet checkpoints from checkpoints/hnm_round<N>/seed{0..4}.pt.
  4. Score designs; merge cppro_prob_hnm + cppro_std_hnm onto the input CSV.

Usage:
    python score_designs_with_6b_hnm.py \\
        --csv  /path/to/all_designs_metrics.csv \\
        --out  /path/to/all_designs_metrics_hnm.csv \\
        --cache /path/to/forge_6b_pertoken_cache.npz \\
        --round 2     # default; loads checkpoints/hnm_round2/
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from head_sweep_6b import DeepSetHead, D, L_MAX
from hnm_round_6b import mask_from_lengths, score_array
from extract_embeddings_6b_forge import load_key, init_forge_client, encode_with_retry

ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = ROOT / 'checkpoints'
SEEDS = [0, 1, 2, 3, 4]
SEQ_COL_CANDIDATES = ('sequence', 'designed_sequence', 'designed_chain_sequence')


def pick_seq_col(df: pd.DataFrame) -> str:
    for c in SEQ_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise SystemExit(f'no sequence column found; tried {SEQ_COL_CANDIDATES}')


def select_designs(df: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == 'all':
        return df.copy()
    if how == 'bg_pass':
        if 'pass_filters' not in df.columns:
            raise SystemExit('--select bg_pass requires a pass_filters column')
        return df[df['pass_filters'] == True].copy()
    raise ValueError(f'unknown --select {how!r}')


def forge_embed_pertoken_cached(seqs: list[str], cache_path: Path):
    """Return (per_token (N, L_MAX, D) fp16, mask (N, L_MAX) fp32).
    Cache stores per-sequence padded per-token + length, keyed by sequence."""
    cache = {}
    if cache_path.exists():
        z = np.load(cache_path)
        cseqs = [str(s) for s in z['sequences']]
        cpt = z['per_token']        # (n_cache, L_MAX, D) fp16
        clen = z['lengths']         # (n_cache,) int32
        for s, pt, ln in zip(cseqs, cpt, clen):
            cache[s] = (pt, int(ln))
        print(f'  cache: {len(cache)} sequences already embedded ({cache_path.name})')

    todo = [s for s in seqs if s not in cache]
    if todo:
        print(f'  Forge-embedding {len(todo)} new sequences (ESM-C 6B, per-token)…')
        client = init_forge_client(load_key())
        t0 = time.time()
        for i, s in enumerate(todo):
            emb = encode_with_retry(client, s)            # (L_with_special, D) fp16
            L = emb.shape[0]
            if L > L_MAX:
                # Sanity: HNM L_MAX=66 = 64+2 special. v6 caps seqs at 50 → +2 = 52. BG designs
                # here cap ~35 → +2 = 37. Anything over L_MAX is unexpected; truncate + warn.
                print(f'    WARN: seq len {len(s)} → embedding L={L} > L_MAX={L_MAX}; truncating')
                emb = emb[:L_MAX]; L = L_MAX
            pad = np.zeros((L_MAX, emb.shape[1]), dtype=np.float16)
            pad[:L] = emb
            cache[s] = (pad, L)
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f'    {i+1}/{len(todo)}  ({rate:.2f}/s)')
        # persist (fixed-width keys → no pickle on load)
        keys = list(cache.keys())
        max_seq_len = max(len(s) for s in keys)
        np.savez(cache_path,
                 sequences=np.array(keys, dtype=f'U{max_seq_len}'),
                 per_token=np.stack([cache[s][0] for s in keys]),
                 lengths=np.array([cache[s][1] for s in keys], dtype=np.int32))
        print(f'  cache updated → {cache_path.name} ({len(cache)} total)')

    X = np.stack([cache[s][0] for s in seqs]).astype(np.float16)
    M = mask_from_lengths([cache[s][1] for s in seqs])
    return X, M


def load_hnm_ensemble(round_n: int, device: str = 'cuda'):
    """Load all 5 seed checkpoints for the requested HNM round."""
    ckpt_dir = CKPT_ROOT / f'hnm_round{round_n}'
    if not ckpt_dir.exists():
        raise SystemExit(f'no HNM checkpoints at {ckpt_dir}; run train_save_hnm_ensemble.py first')
    models = []
    for seed in SEEDS:
        ck_path = ckpt_dir / f'seed{seed}.pt'
        if not ck_path.exists():
            raise SystemExit(f'missing checkpoint {ck_path}')
        ck = torch.load(ck_path, map_location=device, weights_only=True)
        m = DeepSetHead(in_dim=ck.get('in_dim', 2560), hidden=ck.get('hidden', 256)).to(device)
        m.load_state_dict(ck['state_dict']); m.eval()
        models.append((seed, m, ck.get('val_mcc'), ck.get('mined_cap')))
    return models, ckpt_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path, required=True)
    ap.add_argument('--select', default='bg_pass')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--round', type=int, default=2,
                    help='HNM round to load (default 2: best per hnm_fp_rate.md).')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    seq_col = pick_seq_col(df)
    sel = select_designs(df, args.select)
    seqs = sel[seq_col].tolist()
    uniq = sorted(set(seqs))
    print(f'Selected {len(sel)} / {len(df)} designs ({len(uniq)} unique seqs, col={seq_col!r})')

    # --- Forge per-token embeddings ---
    Xd_all, Md_all = forge_embed_pertoken_cached(uniq, args.cache)
    seq_to_idx = {s: i for i, s in enumerate(uniq)}
    Xd = np.stack([Xd_all[seq_to_idx[s]] for s in seqs])
    Md = np.stack([Md_all[seq_to_idx[s]] for s in seqs])

    # --- load HNM ensemble (pre-trained) ---
    print(f'\nLoading HNM round-{args.round} ensemble…')
    models, ckpt_dir = load_hnm_ensemble(args.round, device=args.device)
    print(f'  loaded {len(models)} seeds from {ckpt_dir}')
    for seed, _, vmcc, mcap in models:
        print(f'    seed {seed}: val_mcc={vmcc:.4f}  (trained on v6_train+{mcap} mined)')

    # --- score ---
    per_seed = np.zeros((len(models), len(seqs)), dtype=np.float32)
    Xd_fp16 = Xd.astype(np.float16)
    for i, (seed, model, _, _) in enumerate(models):
        per_seed[i] = score_array(model, Xd_fp16, Md, device=args.device)

    sel = sel.copy()
    sel['cppro_prob_hnm'] = per_seed.mean(axis=0)
    sel['cppro_std_hnm'] = per_seed.std(axis=0)
    # keep generic columns so downstream funnel uses HNM by default
    sel['cppro_prob'] = sel['cppro_prob_hnm']
    sel['cppro_std'] = sel['cppro_std_hnm']

    merged = df.merge(sel[['id', 'cppro_prob_hnm', 'cppro_std_hnm', 'cppro_prob', 'cppro_std']],
                      on='id', how='left')
    merged.to_csv(args.out, index=False)
    n_scored = sel['cppro_prob_hnm'].notna().sum()
    n_cpp = int((sel['cppro_prob_hnm'] >= 0.5).sum())
    print(f'\nwrote {args.out}  ({n_scored} designs scored, {n_cpp} ≥ 0.5)')
    print(f'mean per-design std across seeds: {sel["cppro_std_hnm"].mean():.4f}')


if __name__ == '__main__':
    main()
