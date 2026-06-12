"""inspect_novel.py — characterise the 149 POSEIDON peptides that are NOT in v4.

For each novel sequence, report length, AA composition, net charge at pH 7,
hydrophobic fraction, and the upstream POSEIDON metadata (cell line, cargo,
type) that justifies labelling it positive.

Also check:
  - exact-match overlap with v4 train/val/test
  - exact-match overlap with the 42,332 fresh TrEMBL negatives (would be a contradiction)
  - sequence-length / AA-composition distribution vs v4 train positives
  - any sequences with non-standard residues (drop these)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path('/home/victor/MRes_2/CPPro/CPPro_dataset')
POSEIDON = ROOT / 'sources' / 'poseidon'

STD_AA = set('ACDEFGHIKLMNPQRSTVWY')
POS_AA = set('KR')   # H excluded at pH 7
NEG_AA = set('DE')
HYDROPHOBIC = set('AILMFWV')


def net_charge_ph7(seq: str) -> float:
    return sum(1 for a in seq if a in POS_AA) - sum(1 for a in seq if a in NEG_AA)


def hydrophobic_frac(seq: str) -> float:
    return sum(1 for a in seq if a in HYDROPHOBIC) / max(1, len(seq))


def main():
    # Load POSEIDON ML CSV (deduped + featurised)
    pos_ml = pd.read_csv(POSEIDON / 'CPP_ML.csv', encoding='latin-1', low_memory=False)
    seqs = pos_ml['peptide_sequence'].dropna().unique().tolist()
    print(f'POSEIDON unique seqs: {len(seqs)}')

    # Load v4 splits + fresh-negative pool
    train = pd.read_csv(ROOT / 'splits_v4' / 'train.csv')
    val   = pd.read_csv(ROOT / 'splits_v4' / 'val.csv')
    test  = pd.read_csv(ROOT / 'splits_v4' / 'test.csv')
    fresh = pd.read_csv(ROOT / 'sources' / 'fresh_negatives_clean.csv')

    v4_all = set(train.sequence) | set(val.sequence) | set(test.sequence)
    fresh_seqs = set(fresh.sequence)

    novel = [s for s in seqs if s not in v4_all]
    print(f'Novel-to-v4 (any split): {len(novel)}')

    # Cross-check: do any of the novel POSEIDON seqs appear in the fresh-negative pool?
    contradictions = set(novel) & fresh_seqs
    print(f'CONTRADICTIONS with fresh TrEMBL negatives: {len(contradictions)}')
    if contradictions:
        print('  Will need to drop these from negatives if added as positives:')
        for s in list(contradictions)[:10]:
            print(f'    {s}')

    # Drop sequences with non-standard residues
    novel_clean = []
    rejected_nonstd = []
    for s in novel:
        if not isinstance(s, str): continue
        non_std = [a for a in s if a not in STD_AA]
        if non_std:
            rejected_nonstd.append((s, set(non_std)))
        else:
            novel_clean.append(s)
    print(f'\nAfter dropping non-standard-AA seqs: {len(novel_clean)} (rejected {len(rejected_nonstd)})')
    if rejected_nonstd:
        for s, bad in rejected_nonstd[:5]:
            print(f'   reject  {s[:60]}{"..." if len(s)>60 else ""}  bad={bad}')

    # Length distribution
    lens = pd.Series([len(s) for s in novel_clean])
    print(f'\nLength: min={lens.min()} median={lens.median():.0f} max={lens.max()}')
    print(f'  in (5,10):   {((lens>=5)&(lens<=10)).sum()}')
    print(f'  in (11,20):  {((lens>=11)&(lens<=20)).sum()}')
    print(f'  in (21,50):  {((lens>=21)&(lens<=50)).sum()}')
    print(f'  in (51,200): {((lens>=51)&(lens<=200)).sum()}')
    print(f'  <5 or >200:  {((lens<5)|(lens>200)).sum()}  (will need to drop, out-of-range)')

    # Charge + hydrophobic-fraction distribution vs v4 train positives
    train_pos = train[train.label==1]
    print(f'\n--- charge / hydrophobicity comparison ---')
    novel_charge = pd.Series([net_charge_ph7(s) for s in novel_clean])
    v4_charge    = pd.Series([net_charge_ph7(s) for s in train_pos.sequence])
    print(f'  POSEIDON novel  net charge: median={novel_charge.median():.1f} mean={novel_charge.mean():.2f}')
    print(f'  v4 train pos    net charge: median={v4_charge.median():.1f} mean={v4_charge.mean():.2f}')
    novel_hphob = pd.Series([hydrophobic_frac(s) for s in novel_clean])
    v4_hphob    = pd.Series([hydrophobic_frac(s) for s in train_pos.sequence])
    print(f'  POSEIDON novel  hydrophobic frac: median={novel_hphob.median():.2f}')
    print(f'  v4 train pos    hydrophobic frac: median={v4_hphob.median():.2f}')

    # POSEIDON metadata: cell lines & cargos these novel peptides were tested with
    print('\n--- upstream evidence for the novel positives ---')
    novel_rows = pos_ml[pos_ml['peptide_sequence'].isin(novel_clean)]
    print(f'  total POSEIDON rows backing these {len(novel_clean)} seqs: {len(novel_rows)}')
    # bring in the raw csv for cell line / cargo / type metadata
    raw = pd.read_csv(POSEIDON / 'CPP_dataset.csv', sep=';', encoding='latin-1')
    novel_raw = raw[raw['Sequence'].isin(novel_clean)]
    print(f'  raw-csv rows: {len(novel_raw)}')
    print(f'  distinct PubmedIDs: {novel_raw["PubmedID"].nunique()}')
    print(f'  distinct cell lines: {novel_raw["Cell line"].nunique()}')
    print(f'  distinct cargos: {novel_raw["Cargo"].nunique()}')
    print(f'  distinct Types: {novel_raw["Type"].nunique()}')
    print('\n  Type breakdown for novel:')
    print(novel_raw['Type'].value_counts().head(10).to_string())

    # How many distinct PubmedIDs back each peptide? (1 = single-study, no replication)
    studies_per = novel_raw.groupby('Sequence')['PubmedID'].nunique()
    print(f'\n  novel peptides backed by 1 study only: {(studies_per==1).sum()}')
    print(f'  novel peptides backed by 2+ studies:   {(studies_per>=2).sum()}')

    # Write final cleaned positives + diagnostic table
    out = pd.DataFrame({
        'sequence': novel_clean,
        'length':   [len(s) for s in novel_clean],
        'net_charge_pH7': [net_charge_ph7(s) for s in novel_clean],
        'hydrophobic_frac': [hydrophobic_frac(s) for s in novel_clean],
    })
    # Add metadata
    meta = novel_raw.groupby('Sequence').agg(
        n_studies=('PubmedID', 'nunique'),
        n_cell_lines=('Cell line', 'nunique'),
        first_pubmed=('PubmedID', 'first'),
        any_type=('Type', lambda x: x.iloc[0]),
        any_cargo=('Cargo', lambda x: x.iloc[0]),
    ).reset_index().rename(columns={'Sequence': 'sequence'})
    out = out.merge(meta, on='sequence', how='left')
    out_path = POSEIDON / 'novel_positives_candidates.csv'
    out.to_csv(out_path, index=False)
    print(f'\nWrote {out_path}: {len(out)} candidate novel positives')


if __name__ == '__main__':
    main()
