"""Hierarchical paired bootstrap significance test between classifier heads.

Compares heads on the FIXED v6 test set using their per-sequence predictions
(results/head_sweep_<tag>_test_preds.csv, written by head_sweep_6b.py).

Method (Koehn-2004 paired bootstrap, made hierarchical to also fold in seed variance):
  For each of B iterations:
    1. resample SEEDS with replacement, independently per head, and ensemble that
       head's resampled seeds into one per-sequence probability   (seed variance)
    2. resample TEST PEPTIDES with replacement, the SAME indices for both heads
       (paired)                                                    (test-sampling variance)
    3. compute each head's MCC@0.5 on the resampled peptides, and the difference
  Report the winner's MCC CI, the paired MCC-difference CI, and P(A beats B).

The winner (highest 5-seed ensemble MCC) is tested against every other head, so we
also get a direct answer for, e.g., "could DeepSet actually win on 600M?".

Usage:
    python head_bootstrap.py --tag 600m
    python head_bootstrap.py --tag 6b --n-boot 10000
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'


def load_preds(tag):
    """Return y_true (n_seq,), {head: (n_seed, n_seq) prob array}, ordered seqs."""
    df = pd.read_csv(RESULTS / f'head_sweep_{tag}_test_preds.csv')
    seqs = sorted(df['sequence'].unique())
    y_true = (df.groupby('sequence')['y_true'].first().reindex(seqs).values).astype(int)
    P = {}
    for h, sub in df.groupby('head'):
        piv = sub.pivot_table(index='seed', columns='sequence', values='y_prob')
        P[h] = piv.reindex(columns=seqs).values            # (n_seed, n_seq)
    return y_true, P, seqs


def ensemble_mcc(y_true, prob_arr):
    """MCC@0.5 of the seed-mean ensemble (point estimate)."""
    p = prob_arr.mean(0)
    return matthews_corrcoef(y_true, (p >= 0.5).astype(int))


def hier_paired_bootstrap(y_true, A, B, n_boot, rng):
    n_seq = len(y_true)
    nsa, nsb = A.shape[0], B.shape[0]
    diffs = np.empty(n_boot)
    mA = np.empty(n_boot); mB = np.empty(n_boot)
    for b in range(n_boot):
        ea = A[rng.integers(0, nsa, nsa)].mean(0)          # resample seeds (model A)
        eb = B[rng.integers(0, nsb, nsb)].mean(0)          # resample seeds (model B)
        idx = rng.integers(0, n_seq, n_seq)                # resample peptides (paired)
        yb = y_true[idx]
        a = matthews_corrcoef(yb, (ea[idx] >= 0.5).astype(int))
        bb = matthews_corrcoef(yb, (eb[idx] >= 0.5).astype(int))
        mA[b], mB[b], diffs[b] = a, bb, a - bb
    return diffs, mA, mB


def ci(x, lo=2.5, hi=97.5):
    return np.percentile(x, lo), np.percentile(x, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, help='6b or 600m')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    y_true, P, seqs = load_preds(args.tag)
    point = {h: ensemble_mcc(y_true, P[h]) for h in P}
    order = sorted(point, key=lambda h: -point[h])
    n_seed = {h: P[h].shape[0] for h in P}
    print(f"=== {args.tag}: {len(seqs)} test peptides, "
          f"{n_seed[order[0]]} seeds/head, {args.n_boot} bootstrap draws ===")
    print("ensemble MCC@0.5 (point):")
    for h in order:
        print(f"  {h:11s} {point[h]:.4f}   ({n_seed[h]} seeds)")

    winner = order[0]
    print(f"\nwinner = {winner}.  Hierarchical paired bootstrap vs each other head:")
    print(f"{'comparison':28s} {'dMCC (95% CI)':28s} {'P(win>other)':>12s}  verdict")
    for other in order[1:]:
        diffs, mA, mB = hier_paired_bootstrap(y_true, P[winner], P[other], args.n_boot, rng)
        lo, hi = ci(diffs)
        p_win = float((diffs > 0).mean())
        # two-sided p-value from the paired bootstrap
        p_two = 2 * min(p_win, 1 - p_win)
        sig = 'SIGNIFICANT' if (lo > 0 or hi < 0) else 'not sig (CI spans 0)'
        print(f"{winner+' vs '+other:28s} "
              f"{diffs.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]   "
              f"{p_win:>11.3f}  {sig} (p2={p_two:.3f})")

    # winner's own MCC CI
    _, mWin, _ = hier_paired_bootstrap(y_true, P[winner], P[winner], args.n_boot, rng)
    lo, hi = ci(mWin)
    print(f"\n{winner} MCC@0.5 = {point[winner]:.4f}  (95% bootstrap CI [{lo:.4f}, {hi:.4f}])")


if __name__ == '__main__':
    main()
