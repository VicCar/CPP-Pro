"""Extract ESM-C 6B per-token embeddings via Biohub's hosted inference API.

Reads v6 train + val + test sequences, calls Biohub for each, caches per-token
+ mean-pool embeddings to:

    CPPro_current/embeddings/frozen_6b.h5

ESM-C 6B is NOT open-weight. Biohub's fork of the `esm` package + their hosted
endpoint is our access path. The script is RESUMABLE — if interrupted, re-running
picks up where it left off by checking which rows in the h5 are still
zero-filled. Costs only re-pay for sequences not yet cached.

PREREQUISITE: the `esm` package (provides esm.sdk.forge.ESMCForgeInferenceClient):

    pip install esm

Endpoint and key are configurable:
  - URL: environment variable CPPRO_FORGE_URL  (default https://biohub.ai, Biohub's
    hosted ESM-C 6B inference). Set this if your key is for a different provider.
  - key: Biohub_key.txt in CPPro_current/  (one line, not committed).

Run locally (no GPU needed; the 6B model runs on the hosted server):

    ~/miniconda3/bin/python CPPro/CPPro_current/scripts/extract_embeddings_6b_forge.py

Or to test on a small slice first (RECOMMENDED, confirms API + key + cost):

    ~/miniconda3/bin/python CPPro/CPPro_current/scripts/extract_embeddings_6b_forge.py --limit 5
"""

from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ----------------------- paths + config -----------------------
ROOT = Path(__file__).resolve().parent.parent          # CPPro_current/
SPLITS_DIR = ROOT / 'dataset' / 'splits_v6'
EMB_DIR = ROOT / 'embeddings'
OUT_PATH = EMB_DIR / 'frozen_6b.h5'
KEY_FILE = ROOT / 'Biohub_key.txt'

FORGE_MODEL = 'esmc-6b-2024-12'                         # ESM-C 6B model id
FORGE_URL = os.environ.get('CPPRO_FORGE_URL', 'https://biohub.ai')   # hosted inference endpoint
HIDDEN_D = 2560                                          # ESM-C 6B hidden dim
MAX_LEN = 64                                             # peptide cap (v6 is 5-50 + 2 special tokens)

DEFAULT_BATCH = 1                                        # Forge serves one sequence per call
RETRY_BACKOFF = [2, 5, 15, 60]                           # seconds — exponential backoff for rate limits


# ----------------------- helpers -----------------------
def load_key() -> str:
    if not KEY_FILE.exists():
        raise FileNotFoundError(
            f"Forge API key not found at {KEY_FILE}.\n"
            f"Save it (one line, no whitespace) and re-run."
        )
    return KEY_FILE.read_text().strip()


def collect_sequences() -> pd.DataFrame:
    """Concatenate v6 train+val+test in stable order. Returns df with columns:
    sequence, label, length, len_bin, split."""
    dfs = []
    for split in ['train', 'val', 'test']:
        d = pd.read_csv(SPLITS_DIR / f'{split}.csv')
        d['split'] = split
        dfs.append(d[['sequence', 'label', 'length', 'len_bin', 'split']])
    return pd.concat(dfs, ignore_index=True)


def init_forge_client(api_key: str):
    """Init the ESM-C 6B Biohub client. Requires the Biohub esm fork (see module docstring)."""
    from esm.sdk.forge import ESMCForgeInferenceClient
    return ESMCForgeInferenceClient(model=FORGE_MODEL, url=FORGE_URL, token=api_key)


def encode_one(client, seq: str) -> np.ndarray:
    """Two-call Forge round-trip → per-token embedding (L, HIDDEN_D) as float16.
    L = len(seq) + special tokens (typically +2 for BOS/EOS in ESM-C).

    Forge's logits endpoint takes a tokenised ESMProteinTensor, not a raw ESMProtein.
    Step 1: encode(ESMProtein)  → ESMProteinTensor (one call)
    Step 2: logits(tensor, cfg) → LogitsOutput with .embeddings (one call)
    """
    from esm.sdk.api import ESMProtein, LogitsConfig
    protein = ESMProtein(sequence=seq)
    tensor = client.encode(protein)
    cfg = LogitsConfig(sequence=True, return_embeddings=True)
    out = client.logits(tensor, cfg)
    # out.embeddings shape: (1, L, D) — squeeze the batch dim
    # Forge can return bf16 (numpy lacks bf16 dtype) — cast to fp32 first, then fp16
    emb = out.embeddings.squeeze(0).to(torch.float32).cpu().numpy().astype(np.float16)
    return emb


def encode_with_retry(client, seq: str, max_retries: int = 4) -> np.ndarray:
    """Wrap encode_one with exponential backoff on rate-limit / transient errors."""
    for attempt in range(max_retries):
        try:
            return encode_one(client, seq)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            msg = str(e)[:200]
            print(f"\n  [retry {attempt+1}/{max_retries}]  {msg}  → sleeping {wait}s")
            time.sleep(wait)


# ----------------------- main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None,
                    help='Only extract the first N sequences. Use for cost/api sanity testing.')
    ap.add_argument('--out', type=Path, default=OUT_PATH)
    ap.add_argument('--force', action='store_true',
                    help='Overwrite any existing cache instead of resuming.')
    args = ap.parse_args()

    api_key = load_key()
    print(f"[forge] key loaded from {KEY_FILE}  ({len(api_key)} chars)")

    seqs_df = collect_sequences()
    if args.limit:
        seqs_df = seqs_df.head(args.limit).copy()
    N = len(seqs_df)
    print(f"[v6] {N} sequences to extract  "
          f"(train={int((seqs_df.split=='train').sum())}, "
          f"val={int((seqs_df.split=='val').sum())}, "
          f"test={int((seqs_df.split=='test').sum())})")

    EMB_DIR.mkdir(parents=True, exist_ok=True)

    # Open / create h5 file. Resumable: zero-filled rows are re-attempted.
    mode = 'w' if args.force or not args.out.exists() else 'a'
    with h5py.File(args.out, mode) as f:
        if 'per_token' not in f:
            f.create_dataset('per_token', shape=(N, MAX_LEN + 2, HIDDEN_D), dtype='float16',
                             chunks=(1, MAX_LEN + 2, HIDDEN_D), compression='gzip', compression_opts=1)
            f.create_dataset('mean_pool', shape=(N, HIDDEN_D), dtype='float16')
            f.create_dataset('lengths', shape=(N,), dtype='int32')
            f.create_dataset('sequences', data=np.array(seqs_df.sequence.tolist(), dtype='S64'))
            f.create_dataset('labels', data=seqs_df.label.values.astype(np.int8))
            f.create_dataset('split', data=np.array(seqs_df.split.tolist(), dtype='S5'))
            f.attrs['model'] = FORGE_MODEL
            f.attrs['hidden_dim'] = HIDDEN_D
            f.attrs['max_len_with_special'] = MAX_LEN + 2

        per_token = f['per_token']
        mean_pool = f['mean_pool']
        lengths = f['lengths']

        # Resume: identify rows still un-filled (mean_pool all zeros)
        existing_mp = mean_pool[:]
        to_do_mask = (np.abs(existing_mp).sum(axis=1) == 0)
        to_do_idx = np.where(to_do_mask)[0]
        if len(to_do_idx) < N:
            print(f"[resume] {N - len(to_do_idx)} sequences already cached; "
                  f"{len(to_do_idx)} remaining.")

        if len(to_do_idx) == 0:
            print("[done] all sequences already cached. exiting.")
            return

        # Connect to Forge
        client = init_forge_client(api_key)
        print(f"[forge] connected to {FORGE_MODEL} at {FORGE_URL}")

        t0 = time.time()
        for j, i in enumerate(tqdm(to_do_idx, desc='Forge extract')):
            seq = seqs_df.sequence.iloc[i]
            emb = encode_with_retry(client, seq)             # (L_with_special, D)
            L = emb.shape[0]
            if L > MAX_LEN + 2:
                # Defensive — shouldn't happen since v6 sequences are ≤ 50 aa
                emb = emb[:MAX_LEN + 2]
                L = MAX_LEN + 2
            per_token[i, :L] = emb
            per_token[i, L:] = 0
            mean_pool[i] = emb.mean(axis=0)
            lengths[i] = L

            if (j + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (j + 1) / elapsed
                eta = (len(to_do_idx) - (j + 1)) / max(rate, 1e-9)
                f.flush()
                tqdm.write(f"  [{j+1}/{len(to_do_idx)}]  rate={rate:.2f}/s  "
                           f"eta={eta/60:.1f} min  flushed.")

    print(f"\n[done] wrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
