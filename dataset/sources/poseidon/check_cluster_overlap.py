"""check_cluster_overlap.py — id40 + id80 cluster overlap of POSEIDON novel positives
against v4 train / val / test.

Reports (does not enforce):
  - id40 overlap with TEST (to flag — matches v4 protocol of "report not enforce")
  - id40 overlap with VAL  (to ENFORCE — must be 0 by policy)
  - id80 overlap with TRAIN (to flag near-duplicates that would be downweighted)
"""
from __future__ import annotations
from pathlib import Path
import subprocess, tempfile, shutil
import pandas as pd

ROOT = Path('/home/victor/MRes_2/CPPro/CPPro_dataset')
MMSEQS = Path.home() / 'miniconda3/bin/mmseqs'
POSEIDON = ROOT / 'sources' / 'poseidon'


def mmseqs_cluster(seqs, min_id, cov=0.8, cov_mode=0):
    tmp = Path(tempfile.mkdtemp(prefix='mmcl_'))
    fa = tmp / 'in.fasta'
    with open(fa, 'w') as f:
        for i, s in enumerate(seqs):
            f.write(f'>{i}\n{s}\n')
    db = tmp / 'db'; clu = tmp / 'clu'; w = tmp / 'w'
    subprocess.run([str(MMSEQS), 'createdb', str(fa), str(db)], check=True,
                   capture_output=True)
    subprocess.run([str(MMSEQS), 'cluster', str(db), str(clu), str(w),
                    '--min-seq-id', str(min_id), '-c', str(cov),
                    '--cov-mode', str(cov_mode),
                    '--cluster-reassign', '1', '-v', '1'],
                   check=True, capture_output=True)
    tsv = tmp / 'cluster.tsv'
    subprocess.run([str(MMSEQS), 'createtsv', str(db), str(db), str(clu), str(tsv)],
                   check=True, capture_output=True)
    rep_of = {}
    with open(tsv) as f:
        for line in f:
            rep, mem = line.strip().split('\t')
            rep_of[int(mem)] = int(rep)
    out = {seqs[m]: seqs[rep_of[m]] for m in range(len(seqs))}
    shutil.rmtree(tmp)
    return out


def main():
    cand = pd.read_csv(POSEIDON / 'novel_positives_candidates.csv')
    # Drop out-of-range lengths (v4 supports 5..50 only)
    cand = cand[(cand.length >= 5) & (cand.length <= 50)].reset_index(drop=True)
    print(f'POSEIDON in-range novel candidates: {len(cand)}')

    train = pd.read_csv(ROOT / 'splits_v4' / 'train.csv')
    val   = pd.read_csv(ROOT / 'splits_v4' / 'val.csv')
    test  = pd.read_csv(ROOT / 'splits_v4' / 'test.csv')

    def overlap(min_id, label_target_df, target_name):
        all_seqs = cand.sequence.tolist() + label_target_df.sequence.tolist()
        rep_of = mmseqs_cluster(all_seqs, min_id=min_id)
        cand_reps   = {s: rep_of[s] for s in cand.sequence}
        target_reps = set(rep_of[s] for s in label_target_df.sequence)
        hits = [(s, cand_reps[s]) for s in cand.sequence if cand_reps[s] in target_reps]
        return hits

    print(f'\n=== id40 overlap (POSEIDON novel vs v4 splits) ===')
    test_hits  = overlap(0.4, test,  'test')
    val_hits   = overlap(0.4, val,   'val')
    train_hits = overlap(0.4, train, 'train')
    print(f'  vs TEST  ({len(test)} seqs):  {len(test_hits)} POSEIDON candidates cluster at id40')
    print(f'  vs VAL   ({len(val)} seqs):   {len(val_hits)} POSEIDON candidates cluster at id40   <-- MUST DROP')
    print(f'  vs TRAIN ({len(train)} seqs): {len(train_hits)} POSEIDON candidates cluster at id40  (OK — adds to train clusters)')

    print(f'\n=== id80 overlap (near-duplicates against train) ===')
    train_id80_hits = overlap(0.8, train, 'train')
    print(f'  vs TRAIN at id80: {len(train_id80_hits)} POSEIDON candidates are near-duplicate of an existing train seq')
    print(f'     these still get added but will be downweighted by 1/id80_size_in_split')

    # Save the decision table
    keep_mask = ~cand.sequence.isin(set(s for s, _ in val_hits))
    cand['flag_test_id40_hit']  = cand.sequence.isin(set(s for s, _ in test_hits))
    cand['flag_val_id40_hit']   = cand.sequence.isin(set(s for s, _ in val_hits))
    cand['flag_train_id40_hit'] = cand.sequence.isin(set(s for s, _ in train_hits))
    cand['flag_train_id80_hit'] = cand.sequence.isin(set(s for s, _ in train_id80_hits))
    cand['keep_for_v5'] = keep_mask

    out = POSEIDON / 'novel_positives_decision.csv'
    cand.to_csv(out, index=False)
    print(f'\nWrote {out}')
    print(f'  POSEIDON novel positives to KEEP for v5 train: {int(keep_mask.sum())}')
    print(f'  Dropped due to val id40 overlap:                {int((~keep_mask).sum())}')


if __name__ == '__main__':
    main()
