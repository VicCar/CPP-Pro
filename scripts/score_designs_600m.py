"""Score peptides for cell-penetration with the OPEN-WEIGHT ESM-C 600M.

Fully self-contained: NO API key, NO embedding cache. Downloads the open ESM-C 600M
weights from HuggingFace on first run, embeds each sequence locally, and runs the saved
ensemble from checkpoints/frozen_600m_<head>/ (produced by train_save_600m.py).

Default head is seqcnn (best on 600M: v6-test MCC ~0.74). The head is read from the
checkpoint, so this scorer works for any head (seqcnn uses per-token features; cnn/mlp
use the mean-pooled embedding).

Accuracy is lower than the hosted 6B model (~0.88 with HNM) but needs no external
dependency. Use for a fully open, reproducible scorer.

Usage:
    ~/miniconda3/bin/python scripts/score_designs_600m.py --seq RRRRRRRRR
    ~/miniconda3/bin/python scripts/score_designs_600m.py --csv designs.csv --out scored.csv
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from head_sweep_6b import HEADS

ROOT = HERE.parent
IN_DIM = 1152
L_MAX = 66                                      # 64 aa cap + 2 special tokens (matches frozen_600m.h5)
SEQ_COL_CANDIDATES = ('sequence', 'designed_sequence', 'designed_chain_sequence', 'seq')


def load_esmc600m(device):
    """Construct ESM-C 600M and load open EvolutionaryScale weights (mirrors extract_v6_esmc600m.py)."""
    from esm.models.esmc import ESMC
    import esm.tokenization as tk
    from huggingface_hub import snapshot_download
    for tokfile in (ROOT.parent / 'CPPro_legacy' / 'classifier_experiments' / 'HFToken.txt',
                    Path.home() / '.cppro_hf_token'):
        if tokfile.exists():
            from huggingface_hub import login
            login(token=tokfile.read_text().strip(), add_to_git_credential=False)
            break
    snap = snapshot_download('EvolutionaryScale/esmc-600m-2024-12')
    wpath = os.path.join(snap, 'data', 'weights', 'esmc_600m_2024_12_v0.pth')
    toks = tk.get_esmc_model_tokenizers()
    model = ESMC(d_model=1152, n_heads=18, n_layers=36, tokenizer=toks, use_flash_attn=False).eval()
    miss, unexp = model.load_state_dict(torch.load(wpath, map_location='cpu', weights_only=True), strict=False)
    assert not miss and not unexp, f'state_dict mismatch: missing={len(miss)} unexpected={len(unexp)}'
    return model.to(device), toks


def embed(model, toks, seqs, device):
    """Return per-token (N, L_MAX, D) fp32, mask (N, L_MAX), and mean-pool (N, D)."""
    N = len(seqs)
    Xtok = np.zeros((N, L_MAX, IN_DIM), dtype=np.float32)
    mask = np.zeros((N, L_MAX), dtype=np.float32)
    Xmean = np.zeros((N, IN_DIM), dtype=np.float32)
    with torch.no_grad():
        for i, s in enumerate(seqs):
            ids = toks.encode(s)
            seq_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            with torch.amp.autocast(device.type, dtype=torch.float16):
                h = model(seq_t).embeddings[0].float().cpu().numpy()       # (L+2, D)
            L = min(h.shape[0], L_MAX)
            Xtok[i, :L] = h[:L]; mask[i, :L] = 1.0
            Xmean[i] = h[:L].mean(0)
    return Xtok, mask, Xmean


def load_ensemble(head, device):
    ckpt_dir = ROOT / 'checkpoints' / f'frozen_600m_{head}'
    if not ckpt_dir.exists():
        raise SystemExit(f'no checkpoints at {ckpt_dir}; run scripts/train_save_600m.py --head {head} first')
    models = []
    for ck_path in sorted(ckpt_dir.glob('seed*.pt')):
        ck = torch.load(ck_path, map_location=device, weights_only=True)
        m = HEADS[ck['head']](in_dim=ck.get('in_dim', IN_DIM)).to(device)
        m.load_state_dict(ck['state_dict']); m.eval()
        models.append((m, ck.get('input_type', 'token')))
    if not models:
        raise SystemExit(f'no seed*.pt in {ckpt_dir}')
    return models, ckpt_dir


def score(models, Xtok, mask, Xmean, device):
    per = np.zeros((len(models), len(Xmean)), dtype=np.float32)
    Xtok_t = torch.from_numpy(Xtok).to(device); mask_t = torch.from_numpy(mask).to(device)
    Xmean_t = torch.from_numpy(Xmean).to(device)
    for i, (m, itype) in enumerate(models):
        with torch.no_grad():
            out = m(Xmean_t, None) if itype == 'mean' else m(Xtok_t, mask_t)
            per[i] = torch.sigmoid(out).cpu().numpy()
    return per.mean(0), per.std(0)


def pick_seq_col(df):
    for c in SEQ_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise SystemExit(f'no sequence column found; tried {SEQ_COL_CANDIDATES}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', help='Score a single peptide and print P(CPP).')
    ap.add_argument('--csv', type=Path, help='Score a CSV (auto-detects sequence column).')
    ap.add_argument('--out', type=Path, help='Output CSV (default: <csv stem>_600m.csv).')
    ap.add_argument('--head', default='seqcnn_hnm',
                    help='Which 600M ensemble to load (default seqcnn_hnm, the HNM-hardened model).')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    if not args.seq and not args.csv:
        raise SystemExit('provide --seq or --csv')

    device = torch.device(args.device)
    models, ckpt_dir = load_ensemble(args.head, device)
    print(f'loaded {len(models)}-seed ensemble from {ckpt_dir.name}')
    print('loading open-weight ESM-C 600M (first run downloads from HuggingFace)...')
    backbone, toks = load_esmc600m(device)

    if args.seq:
        Xtok, mask, Xmean = embed(backbone, toks, [args.seq], device)
        prob, std = score(models, Xtok, mask, Xmean, device)
        print(f'\n{args.seq}\n  P(CPP) = {prob[0]:.4f}   per-seed std = {std[0]:.4f}'
              f'   call = {"CPP" if prob[0] >= 0.5 else "non-CPP"}')
        return

    df = pd.read_csv(args.csv)
    col = pick_seq_col(df)
    uniq = sorted(set(df[col].astype(str)))
    print(f'embedding {len(uniq)} unique sequences (col={col!r})...')
    Xtok, mask, Xmean = embed(backbone, toks, uniq, device)
    prob, std = score(models, Xtok, mask, Xmean, device)
    pmap = dict(zip(uniq, prob)); smap = dict(zip(uniq, std))
    df['cppro_prob_600m'] = df[col].astype(str).map(pmap)
    df['cppro_std_600m'] = df[col].astype(str).map(smap)
    out = args.out or args.csv.with_name(args.csv.stem + '_600m.csv')
    df.to_csv(out, index=False)
    print(f'wrote {out}  ({len(df)} rows, {int((df.cppro_prob_600m >= 0.5).sum())} >= 0.5)')


if __name__ == '__main__':
    main()
