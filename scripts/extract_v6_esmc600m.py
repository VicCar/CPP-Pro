"""Extract frozen ESM-C 600M embeddings for v6, matching the frozen_6b.h5 schema.

Usage:
    ~/miniconda3/bin/python benchmarks/extract_v6_esmc600m.py   # -> embeddings/frozen_600m.h5

Mean-pool convention MATCHES frozen_6b.h5 (mean over ALL tokens incl. BOS/EOS).
"""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch
from esm.models.esmc import ESMC
import esm.tokenization as tk

ROOT = Path(__file__).resolve().parent.parent          # CPPro_current/
SPLITS_DIR = ROOT / 'dataset' / 'splits_v6'
EMB_DIR = ROOT / 'embeddings'; EMB_DIR.mkdir(parents=True, exist_ok=True)
OUT = EMB_DIR / 'frozen_600m.h5'
MAX_LEN = 64
HID = 1152

tokfile = ROOT.parent / 'CPPro_legacy' / 'classifier_experiments' / 'HFToken.txt'
if tokfile.exists():
    from huggingface_hub import login
    login(token=tokfile.read_text().strip(), add_to_git_credential=False)

assert torch.cuda.is_available(), 'CUDA required'
device = torch.device('cuda')

dfs = []
for s in ['train', 'val', 'test']:
    d = pd.read_csv(SPLITS_DIR / f'{s}.csv'); d['split'] = s
    dfs.append(d[['sequence', 'label', 'split']])
df = pd.concat(dfs, ignore_index=True)
seqs = df.sequence.tolist(); N = len(seqs)
print(f"[v6] N={N} (train={(df.split=='train').sum()}, val={(df.split=='val').sum()}, "
      f"test={(df.split=='test').sum()})  max_len={max(len(s) for s in seqs)}", flush=True)

print("Loading esmc_600m (EvolutionaryScale open weights, direct load) ...", flush=True)
# The installed `esm` is the Biohub fork (for 6B-via-Forge); its from_pretrained('esmc_600m')
# routes to biohub/esmc-600m without local weights. Bypass it: construct the 600M arch and
# load the open EvolutionaryScale weights directly (verified clean: missing=0, unexpected=0).
from huggingface_hub import snapshot_download
_snap = snapshot_download("EvolutionaryScale/esmc-600m-2024-12")
_wpath = os.path.join(_snap, "data", "weights", "esmc_600m_2024_12_v0.pth")
m = ESMC(d_model=1152, n_heads=18, n_layers=36,
         tokenizer=tk.get_esmc_model_tokenizers(), use_flash_attn=False).eval()
_miss, _unexp = m.load_state_dict(torch.load(_wpath, map_location='cpu', weights_only=True), strict=False)
assert len(_miss) == 0 and len(_unexp) == 0, f"state_dict mismatch: missing={len(_miss)} unexpected={len(_unexp)}"
m = m.to(device)
toks = tk.get_esmc_model_tokenizers()
print(f"  hidden={HID}  (weights: {_wpath})", flush=True)

with h5py.File(OUT, 'w') as f:
    pt = f.create_dataset('per_token', shape=(N, MAX_LEN + 2, HID), dtype='float16',
                          chunks=(1, MAX_LEN + 2, HID), compression='gzip', compression_opts=1)
    mp = f.create_dataset('mean_pool', shape=(N, HID), dtype='float16')
    lengths = f.create_dataset('lengths', shape=(N,), dtype='int32')
    f.create_dataset('sequences', data=np.array(seqs, dtype='S64'))
    f.create_dataset('labels', data=df.label.values.astype(np.int8))
    f.create_dataset('split', data=np.array(df.split.tolist(), dtype='S5'))
    f.attrs['model'] = 'esmc_600m'; f.attrs['hidden_dim'] = HID; f.attrs['max_len_with_special'] = MAX_LEN + 2

    t0 = time.time(); peak = 0.0
    with torch.no_grad():
        for i, s in enumerate(seqs):
            ids = toks.encode(s)
            seq_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = m(seq_t)
            h = out.embeddings[0].float().cpu().numpy()    # (L+2, HID), incl BOS/EOS
            L = h.shape[0]
            if L > MAX_LEN + 2:
                h = h[:MAX_LEN + 2]; L = MAX_LEN + 2
            pt[i, :L] = h.astype(np.float16); pt[i, L:] = 0
            mp[i] = h.mean(0).astype(np.float16)            # mean over ALL tokens (matches frozen_6b.h5)
            lengths[i] = L
            peak = max(peak, torch.cuda.max_memory_allocated() / 1024**3)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{N} rate={(i+1)/(time.time()-t0):.1f}/s peak_vram={peak:.2f}GB", flush=True)

print(f"[done] wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)  peak_vram={peak:.2f}GB")
