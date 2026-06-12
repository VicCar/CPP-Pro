"""Train + save an open-weight ESM-C 600M classifier ensemble (any head).

Default head is seqcnn (best on 600M per the head sweep + bootstrap: significantly
beats deepset/mlp, ties cnn on accuracy but ~10x smaller). Trains on frozen_600m.h5,
saves a 5-seed ensemble that score_designs_600m.py loads to score new peptides
locally with NO API.

Outputs:
  checkpoints/frozen_600m_<head>/
    seed{0..4}.pt    head state_dicts (arch, in_dim, input_type recorded)
    metrics.json     per-seed + ensemble v6-test MCC / F1 / AUC

Usage:
    ~/miniconda3/bin/python scripts/train_save_600m.py                 # seqcnn, 5 seeds
    ~/miniconda3/bin/python scripts/train_save_600m.py --head cnn
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from head_sweep_6b import HEADS, load_data, build_loaders

ROOT = HERE.parent
EMB_FILE = ROOT / 'embeddings' / 'frozen_600m.h5'
IN_DIM = 1152


def train_one(head_name, seed, splits, dim, device='cuda', max_epochs=100, patience=10, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    HeadCls = HEADS[head_name]
    itype = HeadCls.input_type
    model = HeadCls(in_dim=dim).to(device)
    train_loader, val_loader, _ = build_loaders(splits, itype, batch_size=64)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    best_mcc, best_state, left = -1.0, None, patience
    for _ in range(max_epochs):
        model.train()
        for batch in train_loader:
            if itype == 'mean':
                X, y, w = [t.to(device) for t in batch]; mask = None
            else:
                X, mask, y, w = [t.to(device) for t in batch]
            optim.zero_grad(set_to_none=True)
            (bce(model(X, mask), y) * w).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                if itype == 'mean':
                    X, y, w = [t.to(device) for t in batch]; mask = None
                else:
                    X, mask, y, w = [t.to(device) for t in batch]
                ps.append(torch.sigmoid(model(X, mask)).cpu().numpy()); ys.append(y.cpu().numpy())
        mcc = matthews_corrcoef(np.concatenate(ys), (np.concatenate(ps) >= 0.5).astype(int))
        if mcc > best_mcc:
            best_mcc, left = mcc, patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            left -= 1
            if left <= 0:
                break
    model.load_state_dict(best_state)
    return model, float(best_mcc), itype


def score_split(model, splits, split, itype, device='cuda'):
    model.eval()
    d = splits[split]
    X = torch.from_numpy(d['mean'] if itype == 'mean' else d['token']).to(device)
    mask = None if itype == 'mean' else torch.from_numpy(d['mask']).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(X, mask)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head', default='seqcnn', choices=list(HEADS.keys()))
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    out_dir = ROOT / 'checkpoints' / f'frozen_600m_{args.head}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[load] {EMB_FILE.name}  head={args.head}')
    splits, dim = load_data(EMB_FILE)
    assert dim == IN_DIM, f'expected {IN_DIM}-d embeddings, got {dim}'

    test_probs, val_mccs = [], []
    yte = splits['test']['label']
    for seed in args.seeds:
        t0 = time.time()
        model, vmcc, itype = train_one(args.head, seed, splits, dim, device=args.device)
        tp = score_split(model, splits, 'test', itype, device=args.device)
        test_probs.append(tp); val_mccs.append(vmcc)
        ck = out_dir / f'seed{seed}.pt'
        torch.save({'state_dict': model.state_dict(), 'arch': HEADS[args.head].__name__,
                    'head': args.head, 'in_dim': dim, 'input_type': itype,
                    'seed': seed, 'val_mcc': vmcc, 'backbone': 'esmc_600m_frozen'}, ck)
        tm = matthews_corrcoef(yte, (tp >= 0.5).astype(int))
        print(f'  seed {seed}: val={vmcc:.4f}  test={tm:.4f}  ({time.time()-t0:.1f}s)  -> {ck.name}')

    test_p = np.mean(test_probs, axis=0); pred = (test_p >= 0.5).astype(int)
    metrics = {'backbone': 'frozen ESM-C 600M (open-weight)', 'head': args.head,
               'arch': HEADS[args.head].__name__, 'in_dim': dim, 'seeds': args.seeds,
               'val_mcc_mean': float(np.mean(val_mccs)),
               'test_mcc_ensemble': float(matthews_corrcoef(yte, pred)),
               'test_f1_ensemble': float(f1_score(yte, pred)),
               'test_auc_ensemble': float(roc_auc_score(yte, test_p))}
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    sz = sum(f.stat().st_size for f in out_dir.glob('seed*.pt')) / 1e6
    print(f'\nensemble v6-test MCC: {metrics["test_mcc_ensemble"]:.4f}  '
          f'| {len(args.seeds)} checkpoints = {sz:.1f} MB -> {out_dir}')


if __name__ == '__main__':
    main()
