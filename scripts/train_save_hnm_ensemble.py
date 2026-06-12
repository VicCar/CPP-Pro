"""Train Round-N HNM ensemble ONCE, save 5 checkpoints, validate the metric.

Recreates the round-2 training set (v6 train + first 600 mined hard negs) by default
— the best operating point per CPPro_current/results/hnm_fp_rate.md (FP 3.8%,
test MCC 0.877). Trains the 5-seed DeepSet ensemble, persists each seed's
state_dict, and reproduces the published screening FP rate / test MCC / TPR so
we can confirm HNM actually worked before using the checkpoints for screening.

This does NOT touch hnm_state.json or hnm_rounds.csv — it's a separate, idempotent
"freeze and validate" pass on top of the existing round runner's mined indices.

Outputs:
  CPPro/CPPro_current/checkpoints/hnm_round<N>/
    seed{0..4}.pt           — DeepSet head state_dicts
    train_info.json         — n_train, n_pos, n_neg, mined_cap, seeds, timestamp
    metrics.json            — per-seed + ensemble FP rate, TPR, test MCC, F1, AUC
                              alongside hnm_fp_rate.md's recorded round-N numbers
                              for direct comparison

Usage:
    python train_save_hnm_ensemble.py                    # default: round 2 (mined-cap 600)
    python train_save_hnm_ensemble.py --mined-cap 300 --round 1
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hnm_round_6b import (
    load_split, load_unused, balanced_weights, v6_cluster_factor,
    train_deepset, score_array, build_or_load_score_cache,
    STATE_FILE, RESULTS, ROUNDS_CSV,
)

ROOT = HERE.parent
CKPT_ROOT = ROOT / 'checkpoints'
SEEDS = [0, 1, 2, 3, 4]


def build_training(mined_cap: int):
    Xtr, Mtr, ytr, seqtr = load_split('train')
    cluster_factor = v6_cluster_factor()
    base_w = np.array([cluster_factor.get(s, 1.0) for s in seqtr], dtype=np.float32)
    if mined_cap > 0:
        state = json.loads(STATE_FILE.read_text())
        all_mined = state['mined_idx']
        if mined_cap > len(all_mined):
            raise SystemExit(f'requested {mined_cap} mined but only {len(all_mined)} in state')
        idx = all_mined[:mined_cap]
        Xh, Mh, _ = load_unused(idx)
        Xtr = np.concatenate([Xtr, Xh]); Mtr = np.concatenate([Mtr, Mh])
        ytr = np.concatenate([ytr, np.zeros(len(Xh), dtype=ytr.dtype)])
        base_w = np.concatenate([base_w, np.ones(len(Xh), dtype=np.float32)])
    wtr, _, _ = balanced_weights(base_w, ytr)
    return Xtr, Mtr, ytr, wtr


def load_rounds_csv_record(round_n: int) -> dict | None:
    if not ROUNDS_CSV.exists(): return None
    df = pd.read_csv(ROUNDS_CSV)
    rows = df[df['round'] == round_n]
    if len(rows) == 0: return None
    return rows.iloc[0].to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=int, default=2, help='Logical round label for output dir.')
    ap.add_argument('--mined-cap', type=int, default=600,
                    help='Number of mined hard negs from hnm_state.json (round 2 = 600).')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    out_dir = CKPT_ROOT / f'hnm_round{args.round}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output dir: {out_dir}')

    # --- training set ---
    Xtr, Mtr, ytr, wtr = build_training(args.mined_cap)
    n_pos = int((ytr == 1).sum()); n_neg = int((ytr == 0).sum())
    print(f'Train: {len(Xtr)} rows  ({n_pos}+ / {n_neg}- = 1:{n_neg/max(n_pos,1):.2f}) '
          f'incl {args.mined_cap} mined hard negs')

    Xva, Mva, yva, _ = load_split('val')
    Xte, Mte, yte, _ = load_split('test')

    # --- screening set for FP rate (10k held-out negs, fixed across rounds) ---
    state = json.loads(STATE_FILE.read_text())
    screening_idx = np.array(state['screening_idx'])
    pool_idx_full = np.array(sorted(state['mining_pool_idx']))
    print('Loading cached screening + pool tensors…')
    screen_X, screen_M, _, _ = build_or_load_score_cache(screening_idx, pool_idx_full)
    print(f'  screening set: {screen_X.shape[0]} negatives')

    # --- 5-seed train + save ---
    test_probs, screen_probs, val_mccs, train_times = [], [], [], []
    print(f'\nTraining {len(SEEDS)}-seed DeepSet ensemble…')
    for seed in SEEDS:
        t0 = time.time()
        model, vmcc = train_deepset(seed, Xtr, Mtr, ytr, wtr, Xva, Mva, yva, device=args.device)
        t_tr = time.time() - t0
        # Save
        ckpt_path = out_dir / f'seed{seed}.pt'
        torch.save({
            'state_dict': model.state_dict(),
            'arch': 'DeepSetHead',
            'in_dim': 2560, 'hidden': 256,
            'seed': seed, 'val_mcc': float(vmcc),
            'mined_cap': args.mined_cap, 'round': args.round,
        }, ckpt_path)
        # Score test + screening
        tp = score_array(model, Xte.astype(np.float16), Mte, device=args.device)
        sp = score_array(model, screen_X, screen_M, device=args.device)
        test_probs.append(tp); screen_probs.append(sp); val_mccs.append(vmcc); train_times.append(t_tr)
        fp = float((sp >= 0.5).mean())
        tpr = float((tp[yte == 1] >= 0.5).mean())
        tm = matthews_corrcoef(yte, (tp >= 0.5).astype(int))
        print(f'  seed {seed}: val_mcc={vmcc:.4f}  test_mcc={tm:.4f}  FP@.5={fp:.4f}  TPR@.5={tpr:.4f}  '
              f'({t_tr:.1f}s)  → {ckpt_path.name}')

    # --- ensemble metrics ---
    test_p = np.mean(test_probs, axis=0); screen_p = np.mean(screen_probs, axis=0)
    pos_mask = (yte == 1)
    test_mcc = float(matthews_corrcoef(yte, (test_p >= 0.5).astype(int)))
    test_f1  = float(f1_score(yte, (test_p >= 0.5).astype(int)))
    test_auc = float(roc_auc_score(yte, test_p))
    fp_ens   = float((screen_p >= 0.5).mean())
    tpr_ens  = float((test_p[pos_mask] >= 0.5).mean())
    fp_seeds = [float((sp >= 0.5).mean()) for sp in screen_probs]
    tpr_seeds = [float((tp[pos_mask] >= 0.5).mean()) for tp in test_probs]

    metrics = {
        'round': args.round, 'mined_cap': args.mined_cap,
        'n_train': int(len(Xtr)), 'n_pos': n_pos, 'n_neg': n_neg,
        'seeds': SEEDS,
        'val_mcc_mean': float(np.mean(val_mccs)),
        'val_mcc_per_seed': [float(v) for v in val_mccs],
        'test_mcc_ensemble': test_mcc, 'test_f1_ensemble': test_f1, 'test_auc_ensemble': test_auc,
        'fp_rate_ensemble@0.5': fp_ens, 'tpr_test_ensemble@0.5': tpr_ens,
        'fp_rate_seed_mean': float(np.mean(fp_seeds)),
        'fp_rate_seed_std':  float(np.std(fp_seeds)),
        'fp_per_seed': fp_seeds,
        'tpr_seed_mean': float(np.mean(tpr_seeds)),
        'tpr_seed_std':  float(np.std(tpr_seeds)),
        'tpr_per_seed': tpr_seeds,
        'train_seconds_per_seed': train_times,
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    info = {
        'recipe': 'v6 train + first <mined_cap> mined hard negs from hnm_state.json',
        'mined_cap': args.mined_cap, 'round_label': args.round,
        'n_train': int(len(Xtr)), 'n_pos': n_pos, 'n_neg': n_neg,
        'arch': 'DeepSetHead(in_dim=2560, hidden=256, dropout=0.3)',
        'backbone': 'frozen ESM-C 6B (per-token, L_MAX=66)',
        'seeds': SEEDS,
        'screening_set_size': int(screen_X.shape[0]),
    }
    (out_dir / 'train_info.json').write_text(json.dumps(info, indent=2))

    # --- comparison to published round-N numbers ---
    rec = load_rounds_csv_record(args.round)
    print(f'\n=== HNM round {args.round} validation ===')
    print(f'Test MCC (ensemble): {test_mcc:.4f}    F1: {test_f1:.4f}    AUC: {test_auc:.4f}')
    print(f'FP rate @0.5 (per-seed mean ± std): '
          f'{np.mean(fp_seeds):.4f} ± {np.std(fp_seeds):.4f}    [{", ".join(f"{x:.4f}" for x in fp_seeds)}]')
    print(f'TPR     @0.5 (per-seed mean ± std): '
          f'{np.mean(tpr_seeds):.4f} ± {np.std(tpr_seeds):.4f}    [{", ".join(f"{x:.4f}" for x in tpr_seeds)}]')
    if rec is not None:
        print(f'\nPublished round-{args.round} from hnm_rounds.csv:')
        print(f'  test_mcc            = {rec["test_mcc"]:.4f}')
        print(f'  fp_rate_seed_mean   = {rec["fp_rate_seed_mean"]:.4f} ± {rec["fp_rate_seed_std"]:.4f}')
        print(f'  tpr_seed_mean       = {rec["tpr_seed_mean"]:.4f} ± {rec["tpr_seed_std"]:.4f}')
        d_mcc = test_mcc - rec['test_mcc']
        d_fp  = np.mean(fp_seeds) - rec['fp_rate_seed_mean']
        d_tpr = np.mean(tpr_seeds) - rec['tpr_seed_mean']
        print(f'  Δ test_mcc          = {d_mcc:+.4f}')
        print(f'  Δ fp_rate_seed_mean = {d_fp:+.4f}')
        print(f'  Δ tpr_seed_mean     = {d_tpr:+.4f}')
        print('  (small non-zero deltas expected from non-deterministic cuDNN / GPU; same recipe)')
    print(f'\nSaved {len(SEEDS)} checkpoints + metrics.json + train_info.json to {out_dir}')


if __name__ == '__main__':
    main()
