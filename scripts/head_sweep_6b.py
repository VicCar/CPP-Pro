"""Classifier-head sweep on frozen ESM-C 6B (v6 splits).

Heads tested:
  - mlp          — 2-layer MLP on mean-pooled (2560-d)
  - cnn          — pLM4CPPs CNN on mean-pooled (current baseline)
  - transformer  — 2-layer self-attention with learned [CLS] token on per-token (B, L, 2560)
  - deepset      — order-agnostic φ-sum-ρ on per-token

3 seeds per head → 12 runs total, ~10-15 min on GPU.

Outputs to results/:
    head_sweep_6b_per_seed.csv   — one row per (head, seed) with test/val MCC, F1, AUC
    head_sweep_6b_summary.csv    — one row per head with mean ± std across seeds
    head_sweep_6b.png/.pdf       — bar plot
    head_sweep_6b.md             — markdown leaderboard
"""

from __future__ import annotations
import argparse, time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
EMB_FILE = ROOT / 'embeddings' / 'frozen_6b.h5'
SPLITS_DIR = ROOT / 'dataset' / 'splits_v6'
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

D = 2560
L_MAX = 66          # per_token shape from extract script (50 aa max + 2 special tokens)


# ----------------------- heads -----------------------
class MLPHead(nn.Module):
    def __init__(self, in_dim=D, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, x, mask=None):
        return self.net(x).squeeze(-1)
    input_type = 'mean'


class CNNHead(nn.Module):
    """pLM4CPPs paper CNN on mean-pooled embedding (current baseline)."""
    def __init__(self, in_dim=D, hidden=256, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64); self.pool1 = nn.MaxPool1d(2); self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128); self.pool2 = nn.MaxPool1d(2); self.drop2 = nn.Dropout(dropout)
        flat_dim = 128 * (in_dim // 4)
        self.fc1 = nn.Linear(flat_dim, hidden); self.drop3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)
    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = self.drop1(self.pool1(torch.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(torch.relu(self.bn2(self.conv2(x)))))
        return self.fc2(self.drop3(torch.relu(self.fc1(x.flatten(1))))).squeeze(-1)
    input_type = 'mean'


class TransformerHead(nn.Module):
    """2-layer self-attention with a learned [CLS] token on per-token features."""
    def __init__(self, in_dim=D, model_dim=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, model_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, dim_feedforward=4 * model_dim,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
    def forward(self, x, mask):
        # x: (B, L, in_dim), mask: (B, L) where 1=real, 0=pad
        B = x.size(0)
        x = self.proj_in(x)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
        mask = torch.cat([cls_mask, mask], dim=1)
        key_padding_mask = (mask == 0)  # True where padding
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.head(out[:, 0]).squeeze(-1)
    input_type = 'token'


class DeepSetHead(nn.Module):
    """Order-agnostic per-token aggregation."""
    def __init__(self, in_dim=D, hidden=256, dropout=0.3):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
    def forward(self, x, mask):
        h = self.phi(x)
        m = mask.unsqueeze(-1).float()
        agg = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return self.rho(agg).squeeze(-1)
    input_type = 'token'


class SeqCNNHead(nn.Module):
    """CNN over the residue SEQUENCE (masked), not over a mean-pooled vector.

    Conv1d runs along length L with D input channels, so it learns local residue
    motifs (k-mer windows). Padded positions are re-masked after every conv block so
    they cannot leak into the final pooling (fixes the v4-era per-residue-CNN bug),
    then a masked max-pool over L summarises the strongest motif activations.
    """
    def __init__(self, in_dim=D, filters=128, kernel=5, hidden=256, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, filters, kernel, padding=kernel // 2)
        self.bn1 = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters, filters, kernel, padding=kernel // 2)
        self.bn2 = nn.BatchNorm1d(filters)
        self.fc1 = nn.Linear(filters, hidden); self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)
    def forward(self, x, mask):
        m = mask.unsqueeze(1)                                  # (B, 1, L), 1=real
        x = x.transpose(1, 2)                                  # (B, D, L)
        x = torch.relu(self.bn1(self.conv1(x))) * m            # re-mask pads
        x = torch.relu(self.bn2(self.conv2(x))) * m
        x = x.masked_fill(m == 0, float('-inf')).max(dim=2).values   # masked max-pool over L
        return self.fc2(self.drop(torch.relu(self.fc1(x)))).squeeze(-1)
    input_type = 'token'


HEADS = {'mlp': MLPHead, 'cnn': CNNHead, 'transformer': TransformerHead,
         'deepset': DeepSetHead, 'seqcnn': SeqCNNHead}


# ----------------------- data -----------------------
def load_data(emb_file=EMB_FILE):
    with h5py.File(emb_file, 'r') as f:
        mean_pool = f['mean_pool'][:].astype(np.float32)
        per_token = f['per_token'][:].astype(np.float32)
        lengths   = f['lengths'][:].astype(np.int32)
        labels    = f['labels'][:].astype(np.int64)
        split     = np.array([s.decode() for s in f['split'][:]])
        seqs      = np.array([s.decode() for s in f['sequences'][:]])

    # build mask from lengths: 1 for real tokens, 0 for padding
    mask = np.zeros((len(lengths), per_token.shape[1]), dtype=np.float32)
    for i, L in enumerate(lengths):
        mask[i, :L] = 1.0

    # weights from v6 CSVs
    w_map = {}
    for s in ['train', 'val', 'test']:
        df = pd.read_csv(SPLITS_DIR / f'{s}.csv')
        if 'sample_weight' not in df.columns:
            df['sample_weight'] = 1.0
        for seq, w in zip(df.sequence, df.sample_weight):
            w_map[(s, seq)] = float(w)
    weights = np.array([w_map.get((sp, sq), 1.0) for sp, sq in zip(split, seqs)], dtype=np.float32)

    splits = {}
    for s in ['train', 'val', 'test']:
        m = (split == s)
        splits[s] = dict(mean=mean_pool[m], token=per_token[m], mask=mask[m],
                         label=labels[m], weight=weights[m], seqs=seqs[m])
        print(f"  {s}: n={int(m.sum())}  pos={int((labels[m]==1).sum())}  neg={int((labels[m]==0).sum())}")
    return splits, int(mean_pool.shape[1])


def build_loaders(splits, input_type, batch_size):
    """Return train, val, test DataLoaders for the given input type."""
    def make(s, shuffle):
        d = splits[s]
        if input_type == 'mean':
            ds = TensorDataset(torch.from_numpy(d['mean']),
                               torch.from_numpy(d['label'].astype(np.float32)),
                               torch.from_numpy(d['weight']))
        else:
            ds = TensorDataset(torch.from_numpy(d['token']),
                               torch.from_numpy(d['mask']),
                               torch.from_numpy(d['label'].astype(np.float32)),
                               torch.from_numpy(d['weight']))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return make('train', True), make('val', False), make('test', False)


# ----------------------- train one (head, seed) cell -----------------------
def train_cell(head_name, seed, splits, device='cuda', dim=D,
               batch_size=64, max_epochs=100, patience=10, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)

    HeadCls = HEADS[head_name]
    model = HeadCls(in_dim=dim).to(device)
    input_type = HeadCls.input_type

    train_loader, val_loader, test_loader = build_loaders(splits, input_type, batch_size)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction='none')

    best_val_mcc = -1.0; best_state = None; left = patience
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            if input_type == 'mean':
                X, y, w = [t.to(device) for t in batch]; mask = None
            else:
                X, mask, y, w = [t.to(device) for t in batch]
            optim.zero_grad(set_to_none=True)
            logits = model(X, mask)
            loss = (bce(logits, y) * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        # val
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                if input_type == 'mean':
                    X, y, w = [t.to(device) for t in batch]; mask = None
                else:
                    X, mask, y, w = [t.to(device) for t in batch]
                p = torch.sigmoid(model(X, mask)).cpu().numpy()
                ps.append(p); ys.append(y.cpu().numpy())
        y_va = np.concatenate(ys); p_va = np.concatenate(ps)
        mcc = matthews_corrcoef(y_va, (p_va >= 0.5).astype(int))
        if mcc > best_val_mcc:
            best_val_mcc = mcc; left = patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            left -= 1
            if left <= 0: break

    # test with best weights
    model.load_state_dict(best_state)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in test_loader:
            if input_type == 'mean':
                X, y, w = [t.to(device) for t in batch]; mask = None
            else:
                X, mask, y, w = [t.to(device) for t in batch]
            p = torch.sigmoid(model(X, mask)).cpu().numpy()
            ps.append(p); ys.append(y.cpu().numpy())
    y_te = np.concatenate(ys); p_te = np.concatenate(ps)
    pred = (p_te >= 0.5).astype(int)
    result = dict(
        head=head_name, seed=seed,
        val_mcc=float(best_val_mcc),
        test_mcc=float(matthews_corrcoef(y_te, pred)),
        test_f1=float(f1_score(y_te, pred)),
        test_auc=float(roc_auc_score(y_te, p_te)),
        n_params=sum(p.numel() for p in model.parameters()),
    )
    # also return per-sequence test predictions (for paired-bootstrap significance testing)
    return result, y_te, p_te


# ----------------------- main + plot -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heads', nargs='+', default=list(HEADS.keys()))
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--emb-file', type=Path, default=EMB_FILE,
                    help='Embedding h5 to sweep (default frozen_6b.h5).')
    ap.add_argument('--tag', default='6b',
                    help='Output filename tag, e.g. "6b" → head_sweep_6b_*.csv.')
    args = ap.parse_args()

    print(f"=== head sweep on {args.tag} ({args.emb_file.name}) ===")
    print(f"heads: {args.heads}, seeds: {args.seeds}, device: {args.device}")
    print("[load] reading h5 ...")
    splits, dim = load_data(args.emb_file)
    print(f"[load] embedding dim = {dim}")

    rows = []
    pred_rows = []
    test_seqs = splits['test']['seqs']
    for head_name in args.heads:
        for seed in args.seeds:
            t0 = time.time()
            r, y_te, p_te = train_cell(head_name, seed, splits, device=args.device, dim=dim)
            r['elapsed_sec'] = time.time() - t0
            rows.append(r)
            for sq, yt, yp in zip(test_seqs, y_te, p_te):
                pred_rows.append((head_name, seed, sq, int(yt), float(yp)))
            print(f"  [{head_name:11s} seed={seed}]  val={r['val_mcc']:.4f}  "
                  f"test={r['test_mcc']:.4f}  F1={r['test_f1']:.4f}  AUC={r['test_auc']:.4f}  "
                  f"({r['elapsed_sec']:.1f}s, n_params={r['n_params']:,})")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f'head_sweep_{args.tag}_per_seed.csv', index=False)

    # per-sequence test predictions (one row per head x seed x test peptide) for the
    # paired-bootstrap significance test in head_bootstrap.py
    pd.DataFrame(pred_rows, columns=['head', 'seed', 'sequence', 'y_true', 'y_prob']).to_csv(
        RESULTS / f'head_sweep_{args.tag}_test_preds.csv', index=False)

    # summary per head
    summary = df.groupby('head').agg(
        test_mcc_mean=('test_mcc', 'mean'), test_mcc_std=('test_mcc', 'std'),
        test_f1_mean=('test_f1', 'mean'),   test_f1_std=('test_f1', 'std'),
        test_auc_mean=('test_auc', 'mean'), test_auc_std=('test_auc', 'std'),
        val_mcc_mean=('val_mcc', 'mean'),
        n_params=('n_params', 'first'),
    ).reset_index().sort_values('test_mcc_mean', ascending=False)
    summary.to_csv(RESULTS / f'head_sweep_{args.tag}_summary.csv', index=False)

    # plot
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(len(summary))
    colors = {'mlp': '#1f77b4', 'cnn': '#ff7f0e', 'transformer': '#2ca02c',
              'deepset': '#d62728', 'seqcnn': '#9467bd'}
    bar_colors = [colors.get(h, '#888') for h in summary['head']]
    ax.bar(xs, summary.test_mcc_mean, yerr=summary.test_mcc_std,
           color=bar_colors, edgecolor='black', capsize=4, width=0.6)
    for x, m, s in zip(xs, summary.test_mcc_mean, summary.test_mcc_std):
        ax.text(x, m + s + 0.005, f"{m:.3f}\n±{s:.3f}", ha='center', va='bottom', fontsize=9)
    ax.axhline(0.6126, ls='--', color='grey', alpha=0.7,
               label='pLM4CPPs published (v4 test, ref)')
    ax.set_xticks(xs); ax.set_xticklabels(summary['head'], fontsize=10)
    ax.set_ylabel('v6 test MCC'); ax.set_ylim(0, 1.0)
    ax.set_title(f'Head sweep on frozen {args.tag} (v6 splits, {len(args.seeds)} seeds per head)')
    ax.grid(axis='y', alpha=0.3); ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(RESULTS / f'head_sweep_{args.tag}.{ext}', dpi=150)

    # markdown
    md = [f'# Head sweep on frozen {args.tag} (v6)',
          f'\n3 seeds per head. pLM4CPPs published baseline = 0.6126 on v4 test.\n',
          '| Head | test MCC | test F1 | test AUC | val MCC | n_params |',
          '|---|---:|---:|---:|---:|---:|']
    for _, r in summary.iterrows():
        md.append(f"| {r['head']} | **{r.test_mcc_mean:.4f} ± {r.test_mcc_std:.4f}** | "
                  f"{r.test_f1_mean:.4f} ± {r.test_f1_std:.4f} | "
                  f"{r.test_auc_mean:.4f} ± {r.test_auc_std:.4f} | "
                  f"{r.val_mcc_mean:.4f} | {int(r.n_params):,} |")
    (RESULTS / f'head_sweep_{args.tag}.md').write_text('\n'.join(md))

    print(f"\n=== summary (sorted by test MCC) ===")
    print(summary.to_string(index=False))
    print(f"\nwrote results to {RESULTS}/")


if __name__ == '__main__':
    main()
