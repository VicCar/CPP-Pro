# results/

**Output data** from scripts in [`../scripts/`](../scripts/).

Empty until the first benchmark runs.

## What goes here

- Per-seed / per-fold / per-row CSV outputs from benchmark scripts
- Summary CSVs / JSONs (means, stds, leaderboard rows)
- Headline plots (`.png`, `.pdf`) — **gitignored** (regeneratable from CSVs)

## What does NOT go here

- Hand-authored documents (those belong in [`../notes/`](../notes/))
- Embedding caches (those belong in [`../embeddings/`](../embeddings/), gitignored)
- Checkpoints (those belong in [`../checkpoints/`](../checkpoints/), gitignored)
- Per-epoch training logs (those belong in [`../logs/`](../logs/), gitignored)

## Naming convention

`<benchmark_short_name>_<aggregation>.<ext>`

Examples (none of these exist yet):

- `v6_test_per_seed.csv` — one row per (seed, test sequence) with `prob`, `label`, `pred`
- `v6_test_summary.csv` — one row per model (mean test MCC / F1 / AUC across seeds, plus std)
- `cluster_disjoint_cv_per_fold.csv` — one row per fold per seed
- `ood_vdomain_per_seed.csv` — V-domain set predictions; per-seed std is the load-bearing column
- `competitor_head_to_head.csv` — pLM4CPPs / GraphCPP / CPPro-v6 on v6 test
- `headline_v6.png` — main paper-grade plot (gitignored)

## Retention

Old result CSVs are kept indefinitely in git. Plots regenerate from CSVs as needed. If a benchmark changes its semantics, version the output filename (e.g. `v6_test_per_seed_v2.csv`) rather than overwriting.
