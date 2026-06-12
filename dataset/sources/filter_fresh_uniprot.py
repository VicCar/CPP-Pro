"""filter_fresh_uniprot.py — apply annotation filter (AMP/TM/toxin) to the
fresh 50k UniProt pool. Caches the API responses, then writes a clean CSV.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path('/home/victor/MRes_2/CPPro')
SRC  = ROOT / 'CPPro_dataset' / 'sources'
EMB  = ROOT / 'classifier_experiments' / 'embeddings'

INPUT  = SRC / 'fresh_negatives_pool.csv'
CACHE  = EMB / 'uniprot_fresh50k_annotations.tsv'
OUTPUT = SRC / 'fresh_negatives_clean.csv'

DROP_KEYWORDS = [
    'Antimicrobial', 'Antibiotic', 'Defensin', 'Bacteriocin',
    'Cytolysis', 'Toxin', 'Hemolysis',
    'Transmembrane', 'Cell membrane',
]
DROP_NAME_TOKENS = ['cell-penetrating', 'cell penetrating', 'penetratin',
                    'membrane translocation']


def query_batch(accs):
    url = 'https://rest.uniprot.org/uniprotkb/search'
    params = {
        'query': ' OR '.join(f'accession:{a}' for a in accs),
        'fields': 'accession,keyword,protein_name',
        'format': 'tsv',
        'size': len(accs),
    }
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200: return r.text
        except Exception:
            pass
        time.sleep(2)
    return None


df = pd.read_csv(INPUT)
print(f'Fresh pool: {len(df)} sequences')

if CACHE.exists():
    cached = pd.read_csv(CACHE, sep='\t')
    print(f'Loaded cached annotations: {len(cached)}')
else:
    rows = []
    accs = df.acc.tolist()
    for i in range(0, len(accs), 100):
        batch = accs[i:i+100]
        text = query_batch(batch)
        if text is None:
            print(f'  batch {i//100}: FAILED', flush=True)
            continue
        for ln in text.strip().split('\n')[1:]:
            parts = ln.split('\t')
            if len(parts) >= 3:
                rows.append({'acc': parts[0], 'keywords': parts[1], 'protein_name': parts[2]})
        if (i // 100) % 50 == 49:
            print(f'  done {i+100}/{len(accs)} — {len(rows)} cached', flush=True)
    cached = pd.DataFrame(rows)
    cached.to_csv(CACHE, sep='\t', index=False)
    print(f'Saved annotations to {CACHE}')

# Apply filters
def has_drop_kw(kw):
    if pd.isna(kw): return False
    kws = [k.strip().lower() for k in kw.split(';')]
    return any(any(d.lower() in k for d in DROP_KEYWORDS) for k in kws)

def has_cpp_name(name):
    if pd.isna(name): return False
    s = name.lower()
    return any(t in s for t in DROP_NAME_TOKENS)

cached['drop_kw']   = cached.keywords.apply(has_drop_kw)
cached['drop_cpp']  = cached.protein_name.apply(has_cpp_name)
cached['exclude']   = cached.drop_kw | cached.drop_cpp

print(f'\nFilter breakdown:')
print(f'  Annotations retrieved:       {len(cached)} of {len(df)}  '
      f'({100*len(cached)/len(df):.1f}%)')
print(f'  Dropped by keyword:          {int(cached.drop_kw.sum())}')
print(f'  Dropped by name (CPP-like):  {int(cached.drop_cpp.sum())}')
print(f'  TOTAL dropped:               {int(cached.exclude.sum())}')

# Merge: any acc not in `cached` keeps default exclude=False (i.e., we keep it)
df_merged = df.merge(cached[['acc', 'exclude']], on='acc', how='left')
df_merged['exclude'] = df_merged['exclude'].fillna(False)
clean = df_merged[~df_merged['exclude']].copy()
print(f'\nClean fresh pool: {len(clean)} of {len(df)} ({100*len(clean)/len(df):.1f}%)')

# Length distribution
def bin_len(L):
    if L <= 10: return '(5, 10)'
    if L <= 20: return '(11, 20)'
    return '(21, 50)'
clean['len_bin'] = clean.length.apply(bin_len)
print('\nClean fresh pool by length bin:')
print(clean.len_bin.value_counts().sort_index())

clean[['acc', 'sequence', 'length', 'len_bin']].to_csv(OUTPUT, index=False)
print(f'\nSaved {OUTPUT}')
