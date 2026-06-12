# CPPro dataset v6

Canonical CPP classifier training corpus for the v6-era CPPro and DPLM-1 3B work.
Supersedes v4/v5.

## Quick stats

| Split | n | pos | neg |
|---|---:|---:|---:|
| train | 4,905 | 2,450 | 2,455 |
| val | 545 | 275 | 270 |
| test | 570 | 285 | 285 |

Pool: 57,176 unique sequences (3,010+ / 54,166−), canonical AA, length 5–50, drop-on-conflict.
Trainable: 1:1 ratio (5,450 trainable + 51,156 unused negatives reserved for HNM).

## Test guarantees

- Exact-disjoint from pLM4CPPs training data ✓
- Exact-disjoint from GraphCPP training data ✓
- Length-stratified: 95+/95− per bin × `(5,10), (11,20), (21,50)`
- id40 cluster overlap with train: **140/570** (intentional — matches pLM4CPPs/GraphCPP
  random-shuffle eval protocol for head-to-head comparability). Report v6 test MCC and
  5-fold cluster-disjoint CV MCC side-by-side; see [V6_METHODS.md §7](V6_METHODS.md#7-test--train-cluster-overlap--deliberate).

## Files

- [`train.csv`](train.csv) · [`val.csv`](val.csv) · [`test.csv`](test.csv)
- [`all_assignments.csv`](all_assignments.csv) — every pool sequence with split label
- [`build_report.md`](build_report.md) — auto-generated stats + checks
- [`V6_METHODS.md`](V6_METHODS.md) — thesis-grade methods document
- [`README.md`](README.md) — this file

## Regenerate

```bash
~/miniconda3/bin/python CPPro/CPPro_dataset/build_v6_split.py
```

Seed 0, deterministic.
