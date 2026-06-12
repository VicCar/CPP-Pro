# V6 Methods — CPPro dataset v6

Build script: [`build_v6_split.py`](../build_v6_split.py) · Seed: 0 · Generated: 2026-05-26

This is the canonical dataset for the v6-era CPPro classifier and DPLM-1 3B generator
work. v1–v15 (under dataset v4) are legacy. v5 lineage (v4 → phB → phC → phD → v5 → phE → phF)
is closed; v6 supersedes it.

---

## 1. Provenance — source-by-source

| Source | File | Positives kept | Negatives kept | Notes |
|---|---|---:|---:|---|
| CPPsite3-Natural | `sources/cppsite3_natural.csv` | 5,310 | 0 | Largest single positive contributor; canonical experimentally-verified CPPs |
| EnDM-CPP | `sources/endm_cpp.csv` | 473 | 473 | Balanced; partially overlaps PractiCPP via SiameseCPP upstream |
| GraphCPP | `sources/graphcpp.csv` | 950 | 1,570 | Includes both the GraphCPP repo CSV and the user CSV (union); all entries flagged `tainted_graphcpp` |
| PractiCPP | `sources/practicpp.csv` | 462 | 462 | Balanced; carries upstream taint flags |
| PursueCPP | `sources/pursuecpp.csv` | 242 | 2,282 | Lab-curated; ~9 novel-vs-v3 positives folded in |
| POSEIDON novel-positives | `sources/poseidon/novel_positives_decision.csv` | 130 | 0 | Curated novel-positive shortlist (post-cluster-overlap check) |
| UniProt-v4 | `sources/uniprot_negatives.csv` | 0 | 8,809 | v4-vintage UniProt-reviewed peptides, keyword-filtered |
| UniProt-fresh | `sources/fresh_negatives_clean.csv` | 0 | 42,332 | Fresh UniProt-reviewed peptides, post-keyword-filter (filter dropped 15.3% of raw 50k pool) |

Raw rows merged: 63,495. After canonical-AA + length [5,50] + drop-on-conflict + dedup:
**57,176 unique sequences** (3,010+ / 54,166−).

---

## 2. ACPP stripping policy

Cleavable Tsien-lab-style ACPPs have architecture `^E{6+} <protease_linker> <CPP_arm>`,
where polyE is a polyanionic blocking arm meant to be cleaved off *in vivo* by tumour-
associated proteases (MMP-2/9, uPA), exposing the polycationic CPP. Including the full
ACPP in training would bias the classifier to call uncleaved pro-drug architectures as
CPPs — but the *functional* CPP is the C-terminal arm only.

**Policy:** detect cleavable ACPPs and strip them to the CPP arm before dedup. Linkers
recognised: `PLGLAG`, `GPLGLA`, `GALGLP`, `PLGLAR`, `PLGVR` (canonical MMP-2/9 + uPA
cleavage sites used in published ACPP design).

7 entries matched the cleavable-ACPP architecture across the source CSVs; after the strip
they dedupe to 3 unique CPP arms (full table in [build_report.md](build_report.md#acpp-stripping-cleavable-tsien-style-architectures)):

| Raw sequence (representative) | Linker | CPP arm kept |
|---|---|---|
| `EEEEEEEEEE-PLGLAG-VSRRRRRRGGRRRR` | PLGLAG | `VSRRRRRRGGRRRR` |
| `EEEEEEEEPLGLAGRRRRRRRRN` | PLGLAG | `RRRRRRRRN` |
| `EEEEEEE-GALGLP-RRRRRRRRKKR` | GALGLP | `RRRRRRRRKKR` |

**Not stripped:** 16 polyE-tailed amphipathic peptides (`KWKWKWKWEEEEEEEE`, etc.) — these
have no cleavage linker and are intentional anionic-amphipathic CPP designs, not
pro-drug architectures. Retained as-is.

---

## 3. Filtering

In order:
1. **Canonical AA only:** retain only sequences over `{A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y}`.
   Dashes (used by CPPsite3 to mark modular constructs) are stripped pre-check.
2. **Length 5–50:** longer peptides excluded for compatibility with the per-residue
   embedding cache window and because v6 has zero untainted negatives in (51, 200).
3. **Drop-on-conflict label policy:** any sequence appearing as both positive and negative
   across sources is *dropped*, not relabelled. 306 sequences dropped (most from
   SiameseCPP-upstream cross-source disagreements).

Net unique pool after filtering: **57,176** sequences.

---

## 4. Taint flags vs. pLM4CPPs and GraphCPP training data

Two upstream models are used as external benchmarks (pLM4CPPs published checkpoint,
GraphCPP published checkpoint). For a fair head-to-head comparison the v6 test set must
be *exact-disjoint* from each of their training corpora.

- **`tainted_plm4cpps`**: sequence is in the canonical pLM4CPPs reference set (inherited
  from `splits_v4/all_assignments.csv`, which canonicalised this from the pLM4CPPs
  published embedded dataset). 5,373 unique pLM4CPPs-tainted sequences; **4,003** of them
  land in the v6 pool.
- **`tainted_graphcpp`**: sequence is in `graphcpp.csv` (every entry, regardless of
  GraphCPP's own train/val/test split, is part of GraphCPP's published corpus).
  **2,122** in the v6 pool.
- **`tainted_either`**: 4,296 (untainted: 52,880).

Test sampling draws *only* from untainted-by-both. Exact-disjointness is verified post-build
([build_report.md → Test isolation checks](build_report.md#test-isolation-checks-must-all-be-0)).

---

## 5. Test set construction

The headline benchmark. Designed to match the pLM4CPPs / GraphCPP evaluation protocol
so head-to-head numbers are fair.

- **Size:** 570 sequences (285+ / 285− balanced).
- **Length stratification:** 95 pos + 95 neg per bin across `(5,10)`, `(11,20)`, `(21,50)`.
  Bin `(51,200)` is empty by length filter.
- **Sampling pool:** untainted-by-both only.
- **Seed:** 0 (deterministic).
- **Cluster-disjointness from train:** *not* enforced (see §7).

---

## 6. Train + val construction

After the test set is held out:

1. **1:1 ratio trainable subset.** All remaining positives (2,725) plus 2,725 negatives
   sampled uniformly from the 53,881 remaining negatives (tainted or untainted — the
   distinction matters for test only). Trainable pool: 5,450 sequences.
2. **id40 cluster-disjoint train ↔ val split.** MMseqs2 sequence clustering at 40%
   identity. Each id40 cluster is assigned wholesale to either train or val, with
   greedy per-(label, length-bin) quota fill targeting val ≈ 10% of the trainable pool
   (within ≤120% slack to handle cluster size variance). **Hard constraint enforced:
   zero id40 cluster overlap between train and val.**
3. **id80 weighting within each split.** MMseqs2 clustering at 80% identity, computed
   independently within train and within val. Per-sequence loss weight:

   ```
   sample_weight = (1 / id80_cluster_size_in_split) × class_weight[label]
   ```

   where `class_weight[L] = n_split / (2 × max(n_split_L, 1))`. This downweights
   dense cluster regions (so the dominant cationic-amphipathic mode doesn't drown out
   rare modes like anionic / ACPP-arm / photo-activatable) while keeping class balance.

Resulting splits:

| Split | n | pos | neg | id80 clusters | sample_weight range |
|---|---:|---:|---:|---:|---|
| train | 4,905 | 2,450 | 2,455 | 3,815 | [0.0238, 1.0010] |
| val | 545 | 275 | 270 | 457 | [0.0918, 1.0093] |
| test | 570 | 285 | 285 | — | 1.0 (uniform) |

---

## 7. Test ↔ train cluster overlap — deliberate

**140/570 test sequences share an id40 cluster with ≥ 1 train sequence.**

This is *not* enforced because pLM4CPPs and GraphCPP both perform their train→test split
via random shuffle without any cluster-disjoint constraint. Enforcing one on v6's test
would handicap CPPro in the head-to-head comparison against published numbers.

The trade-off is that v6 test MCC includes a **memorisation premium** — the model
benefits when test sequences have close-identity neighbours in train. From v4 we measured
this as ≈ +0.04 MCC inflation (0.83 v4 test vs 0.79 5-fold cluster-disjoint CV); v6 is
expected to behave similarly.

**Reporting convention:** every v6 result is quoted with **both** numbers:

- **v6 test MCC** = headline, head-to-head-comparable with pLM4CPPs/GraphCPP.
- **v6 5-fold cluster-disjoint CV MCC** = honest novel-CPP-design generalisation.

The 5-fold CV is run as a separate post-build script and is not part of `build_v6_split.py`.

---

## 8. What's NOT in v6 (and why)

- **Hard negative mining (HNM):** deliberately omitted from the v6 build. HNM will be
  rerun fresh on top of v6 once the classifier is trained, as a separate post-build pass
  (round 1, round 2, etc., each appending a `splits_v6_phX/` directory in the same
  pattern as v5_phE/v5_phF).
- **Length bin (51, 200):** filtered out. v6 sources contain zero untainted negatives in
  this bin, so per-length MCC at long lengths remains untestable; no point including
  positives we can't test.
- **Non-canonical amino acids:** excluded (no NCAA negatives are available at scale, so
  binary classification over NCAA peptides is not trainable).
- **The 16 polyE-tailed amphipathic peptides:** retained, not stripped. These are not
  Tsien-lab cleavable pro-drugs; they're intentional anionic-amphipathic CPP designs.

---

## 9. Regeneration

```bash
~/miniconda3/bin/python CPPro/CPPro_dataset/build_v6_split.py
```

Deterministic given the source CSVs and `SEED = 0`. Re-running produces byte-identical
output.

---

## 10. Files in `splits_v6/`

- [`train.csv`](train.csv) · [`val.csv`](val.csv) · [`test.csv`](test.csv) — the splits
- [`all_assignments.csv`](all_assignments.csv) — every pool sequence with split label
- [`build_report.md`](build_report.md) — auto-generated stats + checks
- [`V6_METHODS.md`](V6_METHODS.md) — this document
- [`README.md`](README.md) — short overview + regeneration command
