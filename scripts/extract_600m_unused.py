"""Extract ESM-C 600M per-token embeddings for the unused-negative pool (for 600M HNM).

Mirrors frozen_6b_unused.h5 row-for-row (same sequences, same order, same 51156 rows),
so the EXISTING HNM screening / mining-pool index split is reused unchanged. Only rows
that are filled in the 6B file are embedded (the rest stay zero). Open-weight ESM-C 600M,
runs locally, NO API. Resumable.

    -> embeddings/frozen_600m_unused.h5
"""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from pathlib import Path
import h5py, numpy as np, torch
from esm.models.esmc import ESMC
import esm.tokenization as tk

ROOT = Path(__file__).resolve().parent.parent
SRC6B = ROOT / 'embeddings' / 'frozen_6b_unused.h5'
OUT = ROOT / 'embeddings' / 'frozen_600m_unused.h5'
MAX_LEN, HID = 64, 1152
LMAX = MAX_LEN + 2

assert torch.cuda.is_available(), 'CUDA required'
device = torch.device('cuda')

tokfile = ROOT.parent / 'CPPro_legacy' / 'classifier_experiments' / 'HFToken.txt'
if tokfile.exists():
    from huggingface_hub import login
    login(token=tokfile.read_text().strip(), add_to_git_credential=False)

with h5py.File(SRC6B, 'r') as f:
    seqs = [s.decode() for s in f['sequences'][:]]
    labels = f['labels'][:] if 'labels' in f else np.zeros(len(seqs), np.int8)
    filled = np.abs(f['mean_pool'][:]).sum(1) > 0
N = len(seqs)
print(f'[unused] {N} rows, {int(filled.sum())} filled-in-6B to embed at 600M', flush=True)

print('Loading esmc_600m (open weights) ...', flush=True)
from huggingface_hub import snapshot_download
snap = snapshot_download('EvolutionaryScale/esmc-600m-2024-12')
wpath = os.path.join(snap, 'data', 'weights', 'esmc_600m_2024_12_v0.pth')
toks = tk.get_esmc_model_tokenizers()
m = ESMC(d_model=1152, n_heads=18, n_layers=36, tokenizer=toks, use_flash_attn=False).eval()
miss, unexp = m.load_state_dict(torch.load(wpath, map_location='cpu', weights_only=True), strict=False)
assert not miss and not unexp, f'state_dict mismatch: missing={len(miss)} unexpected={len(unexp)}'
m = m.to(device)

mode = 'a' if OUT.exists() else 'w'
with h5py.File(OUT, mode) as f:
    if 'per_token' not in f:
        f.create_dataset('per_token', shape=(N, LMAX, HID), dtype='float16',
                         chunks=(1, LMAX, HID), compression='gzip', compression_opts=1)
        f.create_dataset('mean_pool', shape=(N, HID), dtype='float16')
        f.create_dataset('lengths', shape=(N,), dtype='int32')
        f.create_dataset('sequences', data=np.array(seqs, dtype='S64'))
        f.create_dataset('labels', data=labels.astype(np.int8))
        f.attrs['model'] = 'esmc_600m'; f.attrs['hidden_dim'] = HID
    pt, mpd, ln = f['per_token'], f['mean_pool'], f['lengths']
    done = np.abs(mpd[:]).sum(1) > 0
    todo = [i for i in range(N) if filled[i] and not done[i]]
    print(f'[resume] {len(todo)} rows to embed', flush=True)
    t0 = time.time()
    with torch.no_grad():
        for j, i in enumerate(todo):
            ids = toks.encode(seqs[i])
            seq_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                h = m(seq_t).embeddings[0].float().cpu().numpy()
            L = min(h.shape[0], LMAX)
            pt[i, :L] = h[:L].astype(np.float16); pt[i, L:] = 0
            mpd[i] = h[:L].mean(0).astype(np.float16); ln[i] = L
            if (j + 1) % 500 == 0:
                f.flush(); r = (j + 1) / (time.time() - t0)
                print(f'  {j+1}/{len(todo)}  {r:.1f}/s  eta {(len(todo)-j-1)/r/60:.1f} min', flush=True)
print(f'[done] {OUT}  ({OUT.stat().st_size/1e6:.0f} MB)')
