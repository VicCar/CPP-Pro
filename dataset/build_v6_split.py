"""build_v6_split.py — construct CPPro_dataset/splits_v6/{train,val,test}.csv.

Clean rebuild starting from sources/ only (no v3 / v4 / v5 inheritance).
Differences from v4:
  - All current source CSVs merged (cppsite3, endm, graphcpp, practicpp, pursuecpp, poseidon_novel).
  - UniProt negatives = v4-vintage `uniprot_negatives.csv` ∪ fresh `fresh_negatives_clean.csv`.
  - ACPP stripping: 7 cleavable-ACPP entries (polyE + protease linker + CPP arm) are stripped
    to their C-terminal CPP arm only. Raw → stripped sequences logged in V6_METHODS.md.
  - Length filter 5–50 (no (51, 200) bin — no untainted entries there anyway).
  - pLM4CPPs taint flags inherited from v4 all_assignments.csv (the canonical reference).
  - GraphCPP taint flags = sequence in graphcpp.csv.
  - No hard-negative mining — that runs as a separate post-build step on top of v6.

Pipeline:
  1. Load all positive sources; apply ACPP strip; dedup; collect taint flags.
  2. Load all negative sources; dedup against positives.
  3. Canonical AA + length 5–50 filter; drop label-conflicted sequences.
  4. TEST: 95 pos + 95 neg per length bin, untainted-by-both only, seed 0.
  5. 1:1 train+val pool: all remaining positives + same number of negatives, sampled.
  6. MMseqs2 id40 cluster-disjoint train↔val split.
  7. MMseqs2 id80 within each of train and val → sample_weight = (1/id80_size) × class_weight.
  8. Write splits_v6/{train,val,test}.csv + all_assignments.csv + build_report.md.

This script is deterministic given seed=0 and the source CSVs.
"""

from __future__ import annotations
import re, subprocess, tempfile, shutil
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------- config -----------------------
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'sources'
OUT_DIR = ROOT / 'splits_v6'
OUT_DIR.mkdir(exist_ok=True)
SEED = 0
RNG = np.random.default_rng(SEED)

LEN_BINS = [(5, 10), (11, 20), (21, 50)]
TEST_PER_CLASS_PER_BIN = {(5, 10): 95, (11, 20): 95, (21, 50): 95}
NEG_TO_POS_RATIO = 1.0
VAL_TARGET_FRAC = 0.10
LEN_MIN, LEN_MAX = 5, 50

CANON = set("ACDEFGHIKLMNPQRSTVWY")
def is_canon(s): return bool(s) and not bool(set(s) - CANON)
def length_bin(L):
    for lo, hi in LEN_BINS:
        if lo <= L <= hi: return (lo, hi)
    return None

MMSEQS = Path.home() / 'miniconda3/bin/mmseqs'

# ACPP cleavable-architecture detection: polyE block + known protease linker
ACPP_LINKER_PAT = re.compile(r'(PLGLAG|GPLGLA|GALGLP|PLGLAR|PLGVR)')
def strip_acpp(seq_raw: str) -> tuple[str, str | None, str | None]:
    """If seq_raw matches cleavable-ACPP architecture (^E{6+}...<linker>...), return
    (CPP_arm, linker, original_clean) — the C-terminal CPP arm after the linker.
    Otherwise return (seq_clean, None, None). Dashes are always stripped first."""
    seq = seq_raw.replace('-', '')
    if re.match(r'^E{6,}', seq):
        m = ACPP_LINKER_PAT.search(seq)
        if m:
            return seq[m.end():], m.group(1), seq
    return seq, None, None


# ----------------------- helpers -----------------------
def mmseqs_cluster(seqs, min_id, cov=0.8, cov_mode=0):
    if not seqs: return {}
    tmp = Path(tempfile.mkdtemp(prefix='mmclu_'))
    try:
        fa = tmp / 'in.fa'
        with open(fa, 'w') as f:
            for i, s in enumerate(seqs):
                f.write(f'>s_{i}\n{s}\n')
        db = tmp / 'db'; clu = tmp / 'clu'
        w = tmp / 'w'; w.mkdir()
        subprocess.run([str(MMSEQS), 'createdb', str(fa), str(db)], capture_output=True, check=True)
        subprocess.run([str(MMSEQS), 'cluster', str(db), str(clu), str(w),
                        '--min-seq-id', str(min_id), '-c', str(cov), '--cov-mode', str(cov_mode),
                        '-s', '7.5'], capture_output=True, check=True)
        tsv = tmp / 'cl.tsv'
        subprocess.run([str(MMSEQS), 'createtsv', str(db), str(db), str(clu), str(tsv)],
                       capture_output=True, check=True)
        cdf = pd.read_csv(tsv, sep='\t', header=None, names=['rep', 'mem'])
        rep_to_cid = {r: i for i, r in enumerate(cdf.rep.unique())}
        seq_to_cid = {}
        for _, row in cdf.iterrows():
            idx = int(row.mem.split('_')[1])
            seq_to_cid[seqs[idx]] = rep_to_cid[row.rep]
        return seq_to_cid
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------- 1. Load positives, apply ACPP strip -----------------------
print("[1/8] Loading positive sources and applying ACPP strip...")

POSITIVE_SOURCES = [
    ('CPPsite3-Natural', 'cppsite3_natural.csv', 'label'),
    ('EnDM-CPP',         'endm_cpp.csv',         'label'),
    ('GraphCPP',         'graphcpp.csv',         'label'),
    ('PractiCPP',        'practicpp.csv',        'label'),
    ('PursueCPP',        'pursuecpp.csv',        'label'),
]
pos_rows = []
acpp_log = []
for src_name, fn, label_col in POSITIVE_SOURCES:
    df = pd.read_csv(SRC_DIR / fn)
    df = df[df[label_col] == 1]
    for _, row in df.iterrows():
        seq_raw = str(row.sequence)
        seq, linker, orig = strip_acpp(seq_raw)
        pos_rows.append({'sequence': seq, 'label': 1, 'source': src_name,
                         'raw_sequence': seq_raw if seq_raw != seq else seq_raw})
        if linker is not None:
            acpp_log.append({'source': src_name, 'raw': orig, 'linker': linker, 'cpp_arm': seq})

# POSEIDON novel-positives (curated shortlist)
ps_decision = pd.read_csv(SRC_DIR / 'poseidon' / 'novel_positives_decision.csv')
# Column with sequence varies — try common ones
seq_col = next((c for c in ('sequence', 'peptide_sequence', 'standard_sequence') if c in ps_decision.columns), None)
assert seq_col, f"No sequence column found in POSEIDON; columns: {ps_decision.columns.tolist()}"
for _, row in ps_decision.iterrows():
    seq = str(row[seq_col]).replace('-', '')
    seq, linker, orig = strip_acpp(seq)
    pos_rows.append({'sequence': seq, 'label': 1, 'source': 'POSEIDON-novel',
                     'raw_sequence': seq})
    if linker is not None:
        acpp_log.append({'source': 'POSEIDON-novel', 'raw': orig, 'linker': linker, 'cpp_arm': seq})

pos_df = pd.DataFrame(pos_rows)
print(f"  raw positive rows: {len(pos_df)}; ACPP strips applied: {len(acpp_log)}")
for r in acpp_log:
    print(f"    {r['source']:18s}  {r['raw']!r:50s}  linker={r['linker']:7s}  -> {r['cpp_arm']}")


# ----------------------- 2. Load negatives -----------------------
print("\n[2/8] Loading negative sources...")
neg_rows = []
NEG_SOURCES_LABELED = [
    ('GraphCPP-neg',  'graphcpp.csv',  'label'),
    ('EnDM-CPP-neg',  'endm_cpp.csv',  'label'),
    ('PractiCPP-neg', 'practicpp.csv', 'label'),
    ('PursueCPP-neg', 'pursuecpp.csv', 'label'),
]
for src_name, fn, label_col in NEG_SOURCES_LABELED:
    df = pd.read_csv(SRC_DIR / fn)
    df = df[df[label_col] == 0]
    for _, row in df.iterrows():
        neg_rows.append({'sequence': str(row.sequence).replace('-', ''),
                         'label': 0, 'source': src_name, 'raw_sequence': str(row.sequence)})
# UniProt negatives (v4 vintage)
un4 = pd.read_csv(SRC_DIR / 'uniprot_negatives.csv')
for _, row in un4.iterrows():
    neg_rows.append({'sequence': str(row.sequence).replace('-', ''),
                     'label': 0, 'source': 'UniProt-v4', 'raw_sequence': str(row.sequence)})
# Fresh UniProt negatives (v5/v6 staged, keyword-filtered)
fn = pd.read_csv(SRC_DIR / 'fresh_negatives_clean.csv')
for _, row in fn.iterrows():
    neg_rows.append({'sequence': str(row.sequence).replace('-', ''),
                     'label': 0, 'source': 'UniProt-fresh', 'raw_sequence': str(row.sequence)})

neg_df = pd.DataFrame(neg_rows)
print(f"  raw negative rows: {len(neg_df)}")


# ----------------------- 3. Merge, filter, drop conflicts -----------------------
print("\n[3/8] Merging, filtering, resolving label conflicts...")
allf = pd.concat([pos_df, neg_df], ignore_index=True)
print(f"  raw rows merged: {len(allf)}")

allf = allf[allf.sequence.map(is_canon)]
allf['length'] = allf.sequence.str.len()
allf = allf[allf.length.between(LEN_MIN, LEN_MAX)]
print(f"  after canonical-AA + length-[{LEN_MIN},{LEN_MAX}] filter: {len(allf)}")

# Label conflict resolution: drop any sequence appearing as both 0 and 1
lbls = allf.groupby('sequence').label.agg(set)
clean_seqs = lbls[lbls.map(len) == 1].index
n_conflicts = int((lbls.map(len) > 1).sum())
pool = allf[allf.sequence.isin(clean_seqs)].drop_duplicates('sequence').reset_index(drop=True)
pool['label'] = pool.sequence.map(lambda s: list(lbls[s])[0])
pool['length'] = pool.sequence.str.len()
pool['len_bin'] = pool.length.map(length_bin)
print(f"  dropped {n_conflicts} label-conflicted sequences")
print(f"  final unique pool: {len(pool)} ({(pool.label==1).sum()}+ / {(pool.label==0).sum()}−)")


# ----------------------- 4. Taint flags -----------------------
print("\n[4/8] Computing taint flags (vs pLM4CPPs and GraphCPP training data)...")
# pLM4CPPs taint: inherit from v4 all_assignments.csv (the canonical reference set).
# After reorg into CPPro_current/, this file lives in taint_references/ next to build_v6_split.py.
v4_assign = pd.read_csv(ROOT / 'taint_references' / 'splits_v4_all_assignments.csv')
plm4cpps_tainted = set(v4_assign[v4_assign.tainted_plm4cpps].sequence)
print(f"  pLM4CPPs reference set: {len(plm4cpps_tainted)} sequences")

# GraphCPP taint: sequence is in graphcpp.csv (any split)
gcpp_all = pd.read_csv(SRC_DIR / 'graphcpp.csv')
graphcpp_tainted = set(gcpp_all.sequence.astype(str).str.replace('-', '', regex=False))
print(f"  GraphCPP reference set: {len(graphcpp_tainted)} sequences")

pool['tainted_plm4cpps'] = pool.sequence.isin(plm4cpps_tainted)
pool['tainted_graphcpp'] = pool.sequence.isin(graphcpp_tainted)
pool['tainted_either']   = pool.tainted_plm4cpps | pool.tainted_graphcpp
print(f"  tainted_plm4cpps: {int(pool.tainted_plm4cpps.sum())}")
print(f"  tainted_graphcpp: {int(pool.tainted_graphcpp.sum())}")
print(f"  tainted_either:   {int(pool.tainted_either.sum())}  (untainted: {int((~pool.tainted_either).sum())})")


# ----------------------- 5. TEST sampling (untainted, length-stratified) -----------------------
print(f"\n[5/8] Sampling test set (95+/95− per bin × 3 bins, untainted-by-both, seed={SEED})...")
test_chunks = []
test_summary = []
candidates = pool[~pool.tainted_either]
for (lo, hi), n_per_class in TEST_PER_CLASS_PER_BIN.items():
    bin_mask = candidates.length.between(lo, hi)
    pos_avail = candidates[bin_mask & (candidates.label == 1)]
    neg_avail = candidates[bin_mask & (candidates.label == 0)]
    n_pos = min(n_per_class, len(pos_avail))
    n_neg = min(n_per_class, len(neg_avail))
    take_pos = pos_avail.sample(n=n_pos, random_state=int(RNG.integers(0, 2**31)))
    take_neg = neg_avail.sample(n=n_neg, random_state=int(RNG.integers(0, 2**31)))
    test_chunks.extend([take_pos, take_neg])
    test_summary.append((f'({lo},{hi})', len(pos_avail), n_pos, len(neg_avail), n_neg))

test_df = pd.concat(test_chunks, ignore_index=True)
test_seqs = set(test_df.sequence)
print(f"  test: {len(test_df)} ({(test_df.label==1).sum()}+/{(test_df.label==0).sum()}−)")
for b, pa, pt, na, nt in test_summary:
    print(f"    bin {b}: pos {pt}/{pa}, neg {nt}/{na}")


# ----------------------- 6. Build 1:1 train+val pool -----------------------
print(f"\n[6/8] Sampling negatives at 1:{NEG_TO_POS_RATIO:.0f} ratio for train+val...")
remaining = pool[~pool.sequence.isin(test_seqs)].reset_index(drop=True)
rem_pos = remaining[remaining.label == 1]
rem_neg = remaining[remaining.label == 0]
n_neg_target = int(round(len(rem_pos) * NEG_TO_POS_RATIO))
rem_neg_sampled = rem_neg.sample(n=min(n_neg_target, len(rem_neg)),
                                 random_state=int(RNG.integers(0, 2**31)))
trainval_pool = pd.concat([rem_pos, rem_neg_sampled], ignore_index=True)
print(f"  positives in train+val pool: {len(rem_pos)}")
print(f"  negatives sampled: {len(rem_neg_sampled)} (of {len(rem_neg)} remaining)")
print(f"  train+val total: {len(trainval_pool)}")


# ----------------------- 7. id40 clustering + train/val split + id80 weights -----------------------
print("\n[7/8] MMseqs2 id40 clustering + train↔val cluster-disjoint split...")
seq_to_id40 = mmseqs_cluster(trainval_pool.sequence.tolist(), min_id=0.4)
trainval_pool['id40'] = trainval_pool.sequence.map(seq_to_id40)
n_clusters = trainval_pool.id40.nunique()
print(f"  id40 clusters: {n_clusters}")

n_val_target = int(round(VAL_TARGET_FRAC * len(trainval_pool)))
target_per_label = n_val_target // 2          # ~272 pos and ~273 neg for n_val_target=545
LABEL_SLACK = 1.05                              # allow ≤5% overshoot per label
BIN_SLACK = 1.30                                # softer per-(label,bin) cap (avoid pathological skew)

# Per-(label, length-bin) quotas — used as a SOFT cap so val isn't all in one length bin.
cells = [(L, b) for L in [0, 1] for b in LEN_BINS]
cell_quota = {(L, b): int(round(VAL_TARGET_FRAC *
                  ((trainval_pool.label == L) & (trainval_pool.len_bin == b)).sum()))
              for L, b in cells}
cell_filled = {c: 0 for c in cells}

clu_summary = (trainval_pool.groupby('id40').agg(size=('sequence', 'size')).reset_index())
clu_summary = clu_summary.sample(frac=1, random_state=int(RNG.integers(0, 2**31))).reset_index(drop=True)

val_clusters = set()
val_pos, val_neg = 0, 0

for _, c in clu_summary.iterrows():
    cid = c.id40
    cluster_rows = trainval_pool[trainval_pool.id40 == cid]
    cp = int((cluster_rows.label == 1).sum())
    cn = int((cluster_rows.label == 0).sum())

    # HARD per-label cap: enforces ~50/50 val balance.
    if val_pos + cp > target_per_label * LABEL_SLACK: continue
    if val_neg + cn > target_per_label * LABEL_SLACK: continue

    # SOFT per-(label, bin) cap: avoid filling val from one length bin only.
    proposed = dict(cell_filled)
    for _, row in cluster_rows.iterrows():
        proposed[(row.label, row.len_bin)] += 1
    if any(proposed[k] > cell_quota[k] * BIN_SLACK for k in cells):
        continue

    val_clusters.add(cid)
    cell_filled = proposed
    val_pos += cp
    val_neg += cn
    if val_pos + val_neg >= n_val_target:
        break

val_df = trainval_pool[trainval_pool.id40.isin(val_clusters)].reset_index(drop=True)
train_df = trainval_pool[~trainval_pool.id40.isin(val_clusters)].reset_index(drop=True)
print(f"  val:   {len(val_df)} ({(val_df.label==1).sum()}+/{(val_df.label==0).sum()}−)  "
      f"target ≈ {target_per_label}+ / {target_per_label}−")
print(f"  train: {len(train_df)} ({(train_df.label==1).sum()}+/{(train_df.label==0).sum()}−)")
overlap = set(train_df.id40) & set(val_df.id40)
assert len(overlap) == 0, f"id40 disjointness violated: {len(overlap)} shared clusters"
print(f"  id40 overlap train↔val: 0 ✓")

# id80 within each split → sample weights
def add_weights(split_df, name):
    seq_to_id80 = mmseqs_cluster(split_df.sequence.tolist(), min_id=0.8)
    split_df = split_df.copy()
    split_df['id80'] = split_df.sequence.map(seq_to_id80)
    split_df['id80_size'] = split_df.id80.map(split_df.id80.value_counts())
    split_df['id40_size'] = split_df.id40.map(split_df.id40.value_counts())
    n_pos = (split_df.label == 1).sum(); n_neg = (split_df.label == 0).sum()
    n = len(split_df)
    cw = {1: n / (2 * max(n_pos, 1)), 0: n / (2 * max(n_neg, 1))}
    split_df['sample_weight'] = (1.0 / split_df.id80_size) * split_df.label.map(cw)
    print(f"  {name}: id80 clusters={split_df.id80.nunique()}; "
          f"weight [{split_df.sample_weight.min():.4f}, {split_df.sample_weight.max():.4f}]; "
          f"class_weight pos={cw[1]:.3f} neg={cw[0]:.3f}")
    return split_df

train_df = add_weights(train_df, 'train')
val_df = add_weights(val_df, 'val')

# Test ↔ train id40 overlap (reported only)
all_for_clu = pd.concat([train_df.assign(split='train'),
                         val_df.assign(split='val'),
                         test_df.assign(split='test')], ignore_index=True)[['sequence', 'split']]
seq_to_id40_all = mmseqs_cluster(all_for_clu.sequence.tolist(), min_id=0.4)
all_for_clu['id40_global'] = all_for_clu.sequence.map(seq_to_id40_all)
train_clu = set(all_for_clu[all_for_clu.split == 'train'].id40_global)
val_clu = set(all_for_clu[all_for_clu.split == 'val'].id40_global)
test_in_train = (all_for_clu.split == 'test') & all_for_clu.id40_global.isin(train_clu)
test_in_val = (all_for_clu.split == 'test') & all_for_clu.id40_global.isin(val_clu)
print(f"  test sharing id40 with train: {int(test_in_train.sum())}/{len(test_df)} (REPORTED, not enforced)")
print(f"  test sharing id40 with val:   {int(test_in_val.sum())}/{len(test_df)}")

# Stamp test
test_id40_map = dict(zip(all_for_clu[all_for_clu.split == 'test'].sequence,
                         all_for_clu[all_for_clu.split == 'test'].id40_global))
test_df = test_df.copy()
test_df['id40'] = test_df.sequence.map(test_id40_map)
test_df['id80'] = -1
test_df['id80_size'] = 1
test_df['id40_size'] = 1
test_df['sample_weight'] = 1.0


# ----------------------- 8. Write outputs -----------------------
print("\n[8/8] Writing splits + report...")
COLS = ['sequence', 'label', 'length', 'len_bin',
        'tainted_plm4cpps', 'tainted_graphcpp',
        'id80', 'id40', 'id80_size', 'id40_size', 'sample_weight']
def fin(df):
    df = df.copy()
    df['len_bin'] = df.len_bin.astype(str)
    return df[COLS]
train_out = fin(train_df); val_out = fin(val_df); test_out = fin(test_df)
train_out.to_csv(OUT_DIR / 'train.csv', index=False)
val_out.to_csv(OUT_DIR / 'val.csv', index=False)
test_out.to_csv(OUT_DIR / 'test.csv', index=False)
print(f"  wrote {OUT_DIR}/train.csv: {len(train_out)} rows")
print(f"  wrote {OUT_DIR}/val.csv:   {len(val_out)} rows")
print(f"  wrote {OUT_DIR}/test.csv:  {len(test_out)} rows")

split_map = {**{s: 'train' for s in train_out.sequence},
             **{s: 'val'   for s in val_out.sequence},
             **{s: 'test'  for s in test_out.sequence}}
ap = pool[['sequence', 'label', 'length', 'len_bin',
           'tainted_plm4cpps', 'tainted_graphcpp']].copy()
ap['split'] = ap.sequence.map(split_map).fillna('unused')
ap['len_bin'] = ap.len_bin.astype(str)
ap.to_csv(OUT_DIR / 'all_assignments.csv', index=False)
print(f"  wrote {OUT_DIR}/all_assignments.csv: {len(ap)} rows  "
      f"(splits: {ap.split.value_counts().to_dict()})")

# Build report
piv = pd.crosstab([ap[ap.split.isin(['train','val','test'])].split,
                   ap[ap.split.isin(['train','val','test'])].len_bin],
                  ap[ap.split.isin(['train','val','test'])].label, margins=True)
acpp_lines = "\n".join(
    f"| {r['source']} | `{r['raw']}` | {r['linker']} | `{r['cpp_arm']}` |" for r in acpp_log
) if acpp_log else "| _(none)_ | | | |"

report = [
    "# splits_v6 build report",
    f"Generated by `build_v6_split.py` (seed={SEED}, ratio 1:{NEG_TO_POS_RATIO:.0f}, "
    f"split 80/{int(VAL_TARGET_FRAC*100)}/{int(VAL_TARGET_FRAC*100)}, length [{LEN_MIN},{LEN_MAX}]).",
    "",
    "## Pool composition",
    "- Positive sources: CPPsite3-Natural, EnDM-CPP, GraphCPP, PractiCPP, PursueCPP, POSEIDON-novel.",
    "- Negative sources: GraphCPP-neg, EnDM-CPP-neg, PractiCPP-neg, PursueCPP-neg, UniProt-v4, UniProt-fresh.",
    f"- Pool (canonical AA + length [{LEN_MIN},{LEN_MAX}] + drop-on-conflict): "
    f"**{len(pool)}** ({int((pool.label==1).sum())}+ / {int((pool.label==0).sum())}−).",
    f"- Label conflicts dropped: {n_conflicts}.",
    f"- Tainted by pLM4CPPs: {int(pool.tainted_plm4cpps.sum())}.",
    f"- Tainted by GraphCPP: {int(pool.tainted_graphcpp.sum())}.",
    f"- Untainted: {int((~pool.tainted_either).sum())}.",
    f"- Trainable subset (1:1 ratio, all positives + sampled negatives, post-test): {len(trainval_pool)}.",
    "",
    "## ACPP stripping (cleavable Tsien-style architectures)",
    "Cleavable ACPP architecture = `^E{6+}...<protease_linker>...<CPP_arm>`. "
    "Linkers recognised: PLGLAG, GPLGLA, GALGLP, PLGLAR, PLGVR. "
    "Pre-merge, the C-terminal CPP arm replaces the full ACPP sequence.",
    "",
    "| Source | Raw | Linker | CPP arm (kept) |",
    "|---|---|---|---|",
    acpp_lines,
    "",
    "## Splits",
    "| Split | n | pos | neg |",
    "|---|---:|---:|---:|",
    f"| train | {len(train_out)} | {int((train_out.label==1).sum())} | {int((train_out.label==0).sum())} |",
    f"| val   | {len(val_out)}   | {int((val_out.label==1).sum())}   | {int((val_out.label==0).sum())}   |",
    f"| test  | {len(test_out)}  | {int((test_out.label==1).sum())}  | {int((test_out.label==0).sum())}  |",
    "",
    "## By split × length bin × label",
    "```",
    piv.to_string(),
    "```",
    "",
    "## Test isolation checks (must all be 0)",
    f"- Test ∩ pLM4CPPs (exact): {int(test_out.tainted_plm4cpps.sum())}",
    f"- Test ∩ GraphCPP (exact): {int(test_out.tainted_graphcpp.sum())}",
    f"- Test ∩ train (exact):    {len(set(test_out.sequence) & set(train_out.sequence))}",
    f"- Test ∩ val (exact):      {len(set(test_out.sequence) & set(val_out.sequence))}",
    "",
    "## Train ↔ val cluster disjointness (must be 0)",
    f"- Shared id40 clusters: {len(set(train_out.id40)&set(val_out.id40))}",
    "",
    "## Test ↔ train id40 cluster overlap (REPORTED, not enforced)",
    f"- Test sequences in id40 cluster with ≥1 train sequence: "
    f"**{int(test_in_train.sum())}/{len(test_df)}**",
    "- Intentionally not enforced — pLM4CPPs and GraphCPP both train→test split via random "
    "shuffle without cluster-disjoint constraint. Holding our test to a stricter bar would "
    "handicap us in the head-to-head comparison. The 5-fold cluster-disjoint CV "
    "(run separately) provides the honest novel-CPP-design generalisation number.",
    "",
    "## Sample weights",
    "- Computed within each of train and val:",
    "  `sample_weight = (1 / id80_cluster_size_in_split) × class_weight[label]`.",
    f"- Train weight range: [{train_out.sample_weight.min():.4f}, {train_out.sample_weight.max():.4f}]",
    f"- Val weight range:   [{val_out.sample_weight.min():.4f}, {val_out.sample_weight.max():.4f}]",
    "- Test: no weights (uniform).",
]
(OUT_DIR / 'build_report.md').write_text('\n'.join(report))
print(f"  wrote {OUT_DIR}/build_report.md")
print("\nDONE.")
