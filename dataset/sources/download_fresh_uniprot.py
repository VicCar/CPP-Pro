"""download_fresh_uniprot.py — pull a fresh batch of UniProt sequences for
hard-negative mining round 2 / fair generalisation eval.

Uses the /uniprotkb/stream endpoint (handles large batches in one request,
unlike /search which paginates at 500/page).

Output:
  CPPro/CPPro_dataset/sources/uniprotkb_fresh_5_50_<release>.fasta.gz   — raw FASTA
  CPPro/CPPro_dataset/sources/fresh_negatives_clean.csv                 — filtered
"""
from __future__ import annotations
import gzip, io, json, sys
from pathlib import Path
import requests
import pandas as pd

ROOT = Path('/home/victor/MRes_2/CPPro')
SRC  = ROOT / 'CPPro_dataset' / 'sources'
SPL  = ROOT / 'CPPro_dataset' / 'splits_v4'

# Existing pool we want to AVOID overlapping with
existing_seqs = set()
for csv_path in [SRC / 'uniprot_negatives.csv',                       # original 8,809
                 SPL / 'train.csv', SPL / 'val.csv', SPL / 'test.csv']:
    df = pd.read_csv(csv_path)
    existing_seqs |= set(df.sequence)
phC_train = pd.read_csv(ROOT / 'CPPro_dataset' / 'splits_v4_phC' / 'train.csv')
existing_seqs |= set(phC_train.sequence)
print(f'Existing-pool sequences to exclude: {len(existing_seqs)}')


def stream_uniprot(query: str, target_n: int):
    """Stream UniProt FASTA. Returns bytes (gzip)."""
    url = 'https://rest.uniprot.org/uniprotkb/stream'
    params = {
        'query':           query,
        'format':          'fasta',
        'compressed':      'true',
    }
    print(f'Streaming UniProt query: {query!r}')
    r = requests.get(url, params=params, stream=True, timeout=300)
    r.raise_for_status()
    total = r.headers.get('X-Total-Results', '?')
    print(f'  X-Total-Results: {total}')
    chunks = []
    bytes_read = 0
    for chunk in r.iter_content(chunk_size=64 * 1024):
        chunks.append(chunk)
        bytes_read += len(chunk)
        if bytes_read % (1024 * 1024) < 64 * 1024:
            print(f'  …{bytes_read//1024} KB streamed', flush=True)
    print(f'  Total bytes: {bytes_read}')
    return b''.join(chunks)


def parse_fasta_gz(blob: bytes):
    """Yields (accession, sequence) pairs from a gzipped FASTA blob."""
    with gzip.open(io.BytesIO(blob), 'rt') as f:
        cur_acc, cur = None, []
        for line in f:
            if line.startswith('>'):
                if cur_acc:
                    yield cur_acc, ''.join(cur)
                # parse "sp|ACC|NAME ..." or "tr|ACC|NAME ..."
                parts = line[1:].split('|')
                cur_acc = parts[1] if len(parts) >= 2 else line[1:].strip()
                cur = []
            else:
                cur.append(line.strip())
        if cur_acc:
            yield cur_acc, ''.join(cur)


# Strategy: query unreviewed (TrEMBL), reference proteome filter, length 5-50
# This yields fresh sequences not in our 13,173 reviewed pool.
QUERY = '(reviewed:false) AND (length:[5 TO 50]) AND (keyword:KW-1185)'
TARGET_RAW = 50000  # how many to pull before filtering

raw_path = SRC / 'uniprotkb_fresh_5_50_2026_05.fasta.gz'
if raw_path.exists():
    print(f'Reusing cached download: {raw_path}')
    blob = raw_path.read_bytes()
else:
    blob = stream_uniprot(QUERY, TARGET_RAW)
    raw_path.write_bytes(blob)
    print(f'Saved raw FASTA → {raw_path}')

# Parse, filter, dedupe
print('\nParsing + filtering...')
canonical = set('ACDEFGHIKLMNPQRSTVWY')
fresh_seqs = {}     # accession → sequence (deduped by sequence)
seen_seq = set()
n_total = 0
n_noncan = 0
n_existing = 0
for acc, seq in parse_fasta_gz(blob):
    n_total += 1
    if not (5 <= len(seq) <= 50): continue
    if not all(c in canonical for c in seq.upper()):
        n_noncan += 1
        continue
    seq = seq.upper()
    if seq in existing_seqs:
        n_existing += 1
        continue
    if seq in seen_seq:
        continue
    seen_seq.add(seq)
    fresh_seqs[acc] = seq
    if len(fresh_seqs) >= TARGET_RAW: break
print(f'  Streamed entries:                 {n_total}')
print(f'  Skipped non-canonical AA:         {n_noncan}')
print(f'  Skipped (already in our pool):    {n_existing}')
print(f'  Fresh unique kept:                {len(fresh_seqs)}')

# Length distribution
def bin_len(L):
    if L <= 10: return '(5, 10)'
    if L <= 20: return '(11, 20)'
    return '(21, 50)'

df = pd.DataFrame({'acc': list(fresh_seqs.keys()),
                   'sequence': list(fresh_seqs.values())})
df['length']  = df.sequence.str.len()
df['len_bin'] = df.length.apply(bin_len)
print('\nLength distribution of fresh pool:')
print(df.len_bin.value_counts().sort_index())

# Save initial fresh CSV (we'll apply annotation filter next, in a separate step)
out = SRC / 'fresh_negatives_pool.csv'
df.to_csv(out, index=False)
print(f'\nSaved fresh pool → {out}')
print(f'(Annotation filter — drop AMP/TM/toxin — will run in a separate step.)')
