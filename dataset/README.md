# dataset/

The v6 CPP training corpus and everything needed to rebuild it.

## Contents

- [`splits_v6/`](splits_v6/) — canonical train (4,905) / val (545) / test (570) splits. **Read [`splits_v6/V6_METHODS.md`](splits_v6/V6_METHODS.md) for thesis-grade methods.**
- [`sources/`](sources/) — raw input CSVs and the UniProt-fetch+filter scripts that produced them. Inputs to `build_v6_split.py`.
- [`taint_references/`](taint_references/) — exact-set membership references for the `tainted_plm4cpps` / `tainted_graphcpp` flags. Currently just `splits_v4_all_assignments.csv` (the pLM4CPPs reference set). GraphCPP taint is computed against `sources/graphcpp.csv` directly.
- [`build_v6_split.py`](build_v6_split.py) — deterministic v6 build script (seed=0).

## Regenerate

```bash
~/miniconda3/bin/python CPPro/CPPro_current/dataset/build_v6_split.py
```

Outputs go to `splits_v6/`. Re-running produces byte-identical splits given unchanged source CSVs.

## Headline numbers (post-build)

| | n | pos | neg |
|---|---:|---:|---:|
| Pool (canonical, length 5–50, drop-on-conflict) | 57,176 | 3,010 | 54,166 |
| Train | 4,905 | 2,450 | 2,455 |
| Val | 545 | 275 | 270 |
| Test | 570 | 285 | 285 |
| Unused (HNM reserve) | 51,156 | — | 51,156 |

Test is exact-disjoint from both pLM4CPPs and GraphCPP training data. Train↔val id40 cluster-disjoint. Test↔train id40 overlap allowed (140/570) — intentional, matches pLM4CPPs/GraphCPP eval protocol; the honest novel-CPP number comes from cluster-disjoint CV run separately.
