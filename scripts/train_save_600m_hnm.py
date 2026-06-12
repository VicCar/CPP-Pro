"""Freeze + save the 600M seqcnn HNM ensemble (analogue of train_save_hnm_ensemble.py for 6B).

Trains the 5-seed seqcnn ensemble on v6 train + the hard negatives mined by the 600M HNM
rounds (hnm_state_600m.json), and saves it to checkpoints/frozen_600m_seqcnn_hnm/ so
score_designs_600m.py --head seqcnn_hnm can load it. Reports v6-test MCC + screening FP rate.

Usage:
    ~/miniconda3/bin/python scripts/train_save_600m_hnm.py                  # use all mined negs
    ~/miniconda3/bin/python scripts/train_save_600m_hnm.py --mined-cap 600  # cap to a chosen round
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hnm_round_6b as H

ROOT = HERE.parent
SEEDS = [0, 1, 2, 3, 4]


def build_training(mined_cap):
    Xtr, Mtr, ytr, seqtr = H.load_split('train')
    cf = H.v6_cluster_factor()
    base_w = np.array([cf.get(s, 1.0) for s in seqtr], dtype=np.float32)
    state = json.loads(H.STATE_FILE.read_text())
    all_mined = state['mined_idx']
    cap = len(all_mined) if mined_cap is None else min(mined_cap, len(all_mined))
    if cap > 0:
        Xh, Mh, _ = H.load_unused(all_mined[:cap])
        Xtr = np.concatenate([Xtr, Xh]); Mtr = np.concatenate([Mtr, Mh])
        ytr = np.concatenate([ytr, np.zeros(len(Xh), dtype=ytr.dtype)])
        base_w = np.concatenate([base_w, np.ones(len(Xh), dtype=np.float32)])
    wtr, _, _ = H.balanced_weights(base_w, ytr)
    return Xtr, Mtr, ytr, wtr, cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mined-cap', type=int, default=None, help='Cap mined negs (default: all).')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    H.configure('600m')
    print(f'[backbone] 600m  head={H.HEAD_CLS.__name__}  dim={H.DIM}')

    out_dir = ROOT / 'checkpoints' / 'frozen_600m_seqcnn_hnm'
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr, Mtr, ytr, wtr, cap = build_training(args.mined_cap)
    n_pos = int((ytr == 1).sum()); n_neg = int((ytr == 0).sum())
    print(f'Train: {len(Xtr)} rows ({n_pos}+/{n_neg}-) incl {cap} mined hard negs')

    Xva, Mva, yva, _ = H.load_split('val')
    Xte, Mte, yte, _ = H.load_split('test')
    state = json.loads(H.STATE_FILE.read_text())
    screen_X, screen_M, _, _ = H.build_or_load_score_cache(
        np.array(state['screening_idx']), np.array(sorted(state['mining_pool_idx'])))

    test_probs, screen_probs, val_mccs = [], [], []
    for seed in SEEDS:
        t0 = time.time()
        model, vmcc = H.train_deepset(seed, Xtr, Mtr, ytr, wtr, Xva, Mva, yva, device=args.device)
        torch.save({'state_dict': model.state_dict(), 'arch': H.HEAD_CLS.__name__,
                    'head': 'seqcnn', 'input_type': 'token', 'in_dim': H.DIM, 'hidden': 256,
                    'seed': seed, 'val_mcc': float(vmcc), 'mined_cap': cap,
                    'backbone': 'esmc_600m_frozen_hnm'}, out_dir / f'seed{seed}.pt')
        tp = H.score_array(model, Xte.astype(np.float16), Mte, device=args.device)
        sp = H.score_array(model, screen_X, screen_M, device=args.device)
        test_probs.append(tp); screen_probs.append(sp); val_mccs.append(vmcc)
        print(f'  seed {seed}: val={vmcc:.4f}  test={matthews_corrcoef(yte,(tp>=.5).astype(int)):.4f}  '
              f'FP={float((sp>=.5).mean()):.4f}  ({time.time()-t0:.1f}s)')

    test_p = np.mean(test_probs, axis=0); screen_p = np.mean(screen_probs, axis=0)
    pred = (test_p >= 0.5).astype(int)
    metrics = {'backbone': 'frozen ESM-C 600M', 'head': 'seqcnn', 'in_dim': H.DIM,
               'mined_cap': cap, 'n_train': int(len(Xtr)), 'seeds': SEEDS,
               'val_mcc_mean': float(np.mean(val_mccs)),
               'test_mcc_ensemble': float(matthews_corrcoef(yte, pred)),
               'test_f1_ensemble': float(f1_score(yte, pred)),
               'test_auc_ensemble': float(roc_auc_score(yte, test_p)),
               'fp_rate_ensemble@0.5': float((screen_p >= 0.5).mean()),
               'tpr_test_ensemble@0.5': float((test_p[yte == 1] >= 0.5).mean())}
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    sz = sum(f.stat().st_size for f in out_dir.glob('seed*.pt')) / 1e6
    print(f'\nensemble: test MCC {metrics["test_mcc_ensemble"]:.4f}  '
          f'FP {metrics["fp_rate_ensemble@0.5"]:.4f}  TPR {metrics["tpr_test_ensemble@0.5"]:.4f}'
          f'  | {len(SEEDS)} ckpts = {sz:.1f} MB -> {out_dir}')


if __name__ == '__main__':
    main()
