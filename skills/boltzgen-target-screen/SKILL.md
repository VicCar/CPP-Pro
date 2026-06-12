---
name: boltzgen-target-screen
description: Score a completed BoltzGen peptide-design run for a receptor target — CPPro cell-penetration (ESM-C 6B via Forge + DeepSet HNM ensemble), filter funnel (BG filters → binding iptm → CPPro → novelty), and a pocket/cascade viz dashboard opened with `viz <target>`. Use when the user has new BoltzGen design results to analyze, says "score <target>", "run the pipeline on <target>", "make the viz for <target>", or adds a new receptor screen.
---

# BoltzGen target screening pipeline

End-to-end: a completed BoltzGen design run against a receptor pocket goes in, a ranked
set of cell-penetrating + novel binder candidates plus a viz dashboard comes out.

The pipeline deliberately spans two areas of the repo (CPPro owns the CPPro model + the
Forge key + the 6B embedding cache; BoltzGen_pipeline owns the design runs + the driver
scripts + the viz). This skill is the single source of truth that ties them together.

## What it does

1. **Score** each design for cell penetration with CPPro: ESM-C 6B embeddings via the
   Biohub/Forge API + the DeepSet head, loaded as the pre-trained 5-seed HNM round-2
   ensemble (`checkpoints/hnm_round2/`). Checkpoints are loaded, not trained per run.
2. **Filter funnel** (each gate a subset of the previous):
   `total → pass BG filters → CPPro ≥ floor → rank by binding composite (within pool) → novel → orthogonal panel`.
   CPPro is gated **first** because binding and penetration anti-correlate for these designs;
   the binding composite then **ranks** the penetrant survivors (it is not a hard cut). The
   composite is a **transparent equal-weight rank-mean** (`composite_rerank.py`) of all six:
   **ipSAE** (not ipTM) + interface PAE + ptm + salt bridges + H-bonds + buried SASA — three
   confidence terms balanced by three physical-contact terms (contacts are real binding signal,
   not just confidence). ipSAE replaces ipTM because it normalises by interface size and so
   discriminates small peptide interfaces that ipTM saturates over. It is ranked **within the
   candidate pool** (the CPPro-passers), because rank-mean is population-dependent. Survivors get
   μH + net-charge descriptors and are reduced to an **orthogonal panel** (one per cluster; prefer
   exact pairwise dedup over id40 — see gotchas) so the shortlist is not 3× the same motif.
3. **Novelty**: Smith-Waterman + BLOSUM62 bit-score vs the world CPP corpus
   (v6 train+val+test +CPPs), corpus-size-independent (`tools/novelty.py`). The bit-score
   `novel` label is a **first-pass screen, not the final call**: for each shortlisted
   candidate, manually inspect the actual top corpus match and its local identity (d3's
   `_0078` scored `novel` by bit-score yet is 64% locally identical to a known amphipathic CPP).
4. **Viz**: Kabsch-align designs to a reference receptor frame, render pocket-focused
   PyMOL PNGs per gate + the natural-ligand reference + a hero, build `index.html`,
   write to `BoltzGen_pipeline/targets/<TargetName>/viz/`.
5. **View**: `viz <TargetName>` opens the dashboard in a browser.
6. **Select**: the funnel returns an orthogonal *panel*, not a final answer. A human picks the
   shortlist by judgment and records it in `results/<run>/SELECTION.md` (the d3 pattern below).

## Prerequisites (check these first)

- **Biohub/Forge API key**: `CPPro/CPPro_current/Biohub_key.txt` (canonical). A convenience
  copy lives at `BoltzGen_pipeline/Biohub_key.txt` (gitignored — it is a secret, never commit).
- **DeepSet HNM checkpoints**: `CPPro/CPPro_current/checkpoints/hnm_round2/seed{0..4}.pt` — the
  5-seed ensemble the scorer loads. Pre-trained; (re)built once by
  `CPPro/CPPro_current/scripts/train_save_hnm_ensemble.py`. The frozen-6B v6 cache
  (`CPPro/CPPro_current/embeddings/frozen_6b.h5`) is only needed to *re-train* the ensemble,
  not to score; rebuild it with `CPPro/CPPro_current/scripts/extract_embeddings_6b_forge.py`.
- **esm package = the Biohub fork (reports version 3.3.0)**. This enables the Forge client.
  **Do NOT downgrade to 3.2.3** — that only mattered for the old local ESM-C 600M scorer,
  which we no longer use. Downgrading breaks Forge.
- **BoltzGen results downloaded** to `BoltzGen_pipeline/results/<run_name>/final_ranked_designs/`
  (must contain `final_designs_metrics_100.csv` + `final_100_designs/rank*.cif`).
- GPU available (DeepSet scoring is seconds — checkpoints are loaded, not trained). Forge embedding is remote (no local GPU needed).

## Run a NEW target (5 steps)

### 1. Add a target config
Edit `BoltzGen_pipeline/scripts/target_pipeline.py` → `TARGET_CONFIGS`:

```python
'<TargetName>': {
    'results_dir': BG / 'results' / '<run_name>',
    'pocket_residues': [...],        # receptor pocket residues — copy from the design yaml's `binding:` field
    'binding_topn': 60,              # novelty funnel: keep top-N by the binding composite
    'strong_bind_iptm': 0.85,        # LEGACY: still used by the viz stage's iptm gate (see note)
    'source_pdb': BG / 'jobs' / '<date>' / '<PDB>.pdb',
    'source_receptor_chain': 'A',    # chain in source PDB that is the receptor
    'source_ligand_chain': 'B',      # chain that is the natural ligand (shown as the reference pose)
    'natural_ligand_label': '<e.g. CHMP4 C-terminal helix>',
    'mechanism': 'activator',        # or 'inhibitor' — for the dashboard text
}
```

**Choosing `binding_topn`** (the novelty funnel's binding gate): the composite is a *rank*,
not a thresholdable scalar, so pick how many top-composite designs to carry into the
CPPro/novelty/diversity gates (default 60). Larger = more candidates downstream.

**`strong_bind_iptm` is now legacy**: the score/novelty funnel ranks by the composite, but
the **viz stage still gates on `iptm > strong_bind_iptm`** (`target_pipeline_viz.py`), so keep
the field set until viz is migrated. Check the iptm range with
`pd.read_csv(metrics_csv).design_to_target_iptm.describe()` (V-domain ~0.85, Bro1 ~0.60).

### 2. Score (CPPro 6B + DeepSet HNM via Forge)
```bash
~/miniconda3/bin/python BoltzGen_pipeline/scripts/target_pipeline.py <TargetName> --stage score
```
Forge-embeds the BG-passing designs per-token (~1.5 s each, cached to
`results/<run>/forge_6b_pertoken_cache.npz` so re-runs are free), loads the pre-trained 5-seed
DeepSet HNM round-2 ensemble (no training), writes `final_designs_metrics_100_cppro.csv` with
`cppro_prob_hnm` + `cppro_std_hnm` (mirrored into generic `cppro_prob` / `cppro_std`).

### 3. Novelty + funnel
```bash
~/miniconda3/bin/python BoltzGen_pipeline/scripts/target_pipeline.py <TargetName> --stage novelty
```
Prints the funnel and the top passers, writes `results/<run>/dual_passers_novelty_world.csv`.

### 4. Viz
```bash
~/miniconda3/bin/python BoltzGen_pipeline/scripts/target_pipeline_viz.py <TargetName>
```
Writes cascade CIFs + `figures/pocket_*.png` + `index.html` to `targets/<TargetName>/viz/`.

(steps 2-3 can be combined: `target_pipeline.py <TargetName> --stage all`.)

### 5. View
```bash
viz <TargetName>          # fuzzy match works: viz alix
```
`viz` with no argument lists all registered targets. (Defined in `tools/viz.sh`, sourced from
`~/.bashrc`. It runs a local HTTP/1.1 server on :8765; over SSH it prints a port-forward hint.)

### 6. Final selection (human judgment) → write SELECTION.md
The funnel/panel is the *input* to a hand-curated final pick, not the deliverable. This mirrors
how d3/S100A11 was done — see [`results/d3_s100a11_anxa1/SELECTION.md`](../../../BoltzGen_pipeline/results/d3_s100a11_anxa1/SELECTION.md)
as the template. Steps:
- **Sanity-check the ipSAE ceiling first.** Look at the penetrant pool's ipSAE max + median
  (d3: max 0.40, median 0.31). A low ceiling means *no high-confidence binder exists* — say so,
  and frame the picks as the best binding∩penetration compromise, not as strong binders.
- **Pick a small, mutually-distinct shortlist** from the orthogonal panel (d3 kept **2 of 53**),
  ranking by `composite_rank` then weighing CPPro, ipSAE (read against the ceiling), μH, charge.
- **Scrutinise novelty by hand** for each pick (see the novelty caveat above) — and prefer exact
  pairwise-identity dedup over id40, which leaks near-twins (d3 `_0078`/`_0368`, 67% identical).
- **Read the pocket vs the designs' composition.** If a charge/μH skew is a *generator bias*
  rather than a binding requirement (check the pocket's own charge/hydrophobicity), flag the
  target for a biased re-generation (d3's neutral 1QLS pocket → flagged for an R/K-biased regen).
- **Write `SELECTION.md`**: chosen ids + sequences, each one's CPPro / composite_rank / ipSAE /
  μH / charge / novelty, *why* it was picked, and the caveats to carry forward (ceiling, any
  borderline novelty, near-twin risk, regen recommendation).

## File map (the pieces, and where they live)

| piece | path |
|---|---|
| driver: score + novelty + funnel | `BoltzGen_pipeline/scripts/target_pipeline.py` (TARGET_CONFIGS here) |
| driver: viz (cascade + PyMOL + index) | `BoltzGen_pipeline/scripts/target_pipeline_viz.py` |
| binding composite (ipSAE) + μH + id40 diversity | `BoltzGen_pipeline/scripts/composite_rerank.py` |
| CPPro 6B scorer (called by --stage score) | `CPPro/CPPro_current/scripts/score_designs_with_6b_hnm.py` |
| DeepSet head definition (`DeepSetHead`) | `CPPro/CPPro_current/scripts/head_sweep_6b.py` |
| DeepSet HNM ensemble checkpoints (loaded at score time) | `CPPro/CPPro_current/checkpoints/hnm_round2/seed{0..4}.pt` |
| HNM round runner + checkpoint builder | `CPPro/CPPro_current/scripts/hnm_round_6b.py`, `train_save_hnm_ensemble.py` |
| Forge client + embedding extractor | `CPPro/CPPro_current/scripts/extract_embeddings_6b_forge.py` |
| frozen-6B v6 embeddings (only to re-train the head) | `CPPro/CPPro_current/embeddings/frozen_6b.h5` |
| Forge API key | `CPPro/CPPro_current/Biohub_key.txt` (+ gitignored copy in `BoltzGen_pipeline/`) |
| novelty (corpus + bit-score classifier) | `tools/novelty.py` |
| color system (family-aware palette) | `tools/colors.py` |
| viz shell command | `tools/viz.sh` (`source ~/MRes_2/tools/viz.sh`) |
| per-target outputs | `BoltzGen_pipeline/targets/<TargetName>/viz/` |
| design runs (input) | `BoltzGen_pipeline/results/<run_name>/` |

## Worked examples (already done)

> These funnels were computed with the earlier 6B **CNN** scorer. Re-scoring with the DeepSet
> HNM ensemble (delete the run's `*_cppro.csv` first) will shift the CPPro-gate counts.

- **ALIX-Vdomain** (activator, 2R02, V-domain pocket): funnel `100 → 56 → 11 → 7 → 6`.
  Hits are amphipathic α-helices (μH 0.4-0.55), modestly cationic (+3/+4).
- **ALIX-inhibitor-Bro1** (inhibitor, 3C3O, CHMP4 groove): funnel `100 → 66 → 39 → 9 → 9`.
  iptm threshold 0.60 (Bro1 is a harder pocket). Hits are acidic amphipathic helices.

## Methods notes + gotchas (so a re-run doesn't trip)

- **6B is much more stable than the old local v15** on OOD designs: mean per-seed std ~0.08
  (6B) vs ~0.34 (v15 local 600M multistream).
- **The head is DeepSet, not CNN.** The v6 head sweep picked DeepSet (test MCC 0.871 vs
  CNN 0.825, and lower seed variance) — see `CPPro/CPPro_current/results/head_sweep_6b.md`.
  Hard-negative mining on top cut the 10k-screening false-positive rate from 7.5% (round 0)
  to 3.0% (round 2, the deployed checkpoint), at a small recall cost (0.975 → 0.933) — see
  `CPPro/CPPro_current/results/hnm_fp_rate.md`. The scorer loads the saved 5-seed round-2
  ensemble; nothing is trained per run.
- **Novelty is corpus-size-independent** (bit-score anchored to a fixed `N_ref` in
  `tools/novelty.py`). Growing the corpus does not drift classifications. E-value is also
  reported but not used for the call. See `CPPro/CPPro_current/notes/INSTABILITY_BRIEF.md`.
- **Crystal-reference alignment matches by sequence position, not residue number.** BoltzGen
  renumbers the receptor (e.g. Bro1 is 23-379 in the design vs 2-358 in 3C3O, a +21 offset).
  `seq_aligned_pairs` in `target_pipeline_viz.py` handles this; if you see RMSD > 5 Å warning,
  the natural-ligand pose is unreliable and the source/receptor chains likely differ.
- **BoltzGen chain convention**: chain A = designed peptide, chain B = receptor.
- **CPPro is only run on BG-passers** (`--select bg_pass`) to limit Forge calls. The
  score/novelty funnel ranks binding by the **ipSAE composite** (`composite_rerank.py`,
  top-`binding_topn`), then applies CPPro + novelty + the orthogonal (best-per-id40) panel.
- **Known inconsistency (viz):** `target_pipeline_viz.py` still uses the old
  `iptm > strong_bind_iptm` gate, so the viz set can differ from the novelty funnel until viz
  is migrated to the composite. Deferred while viz is deprioritised.
- **id40 diversity is approximate (MMseqs `cluster` leaks):** MMseqs cluster is a greedy
  heuristic and does NOT guarantee all pairs above `--min-seq-id` co-cluster (set-dependent,
  worse for short peptides). On d3 it split a 66.6%-identical pair across clusters. For a
  trustworthy orthogonal panel, prefer **exact all-vs-all pairwise-identity dedup** over
  id40 clustering. (Same caveat applies to CPPro's id40/id80 dataset clustering — see
  `CPPro/CPPro_current/notes/DESIGN_DECISIONS.md`, 2026-06-10.)
- **The score stage skips if `*_cppro.csv` exists** (delete to re-score). CSVs written by
  the old CNN scorer have only `cppro_prob` (no `cppro_prob_hnm`) — delete and re-score to
  pick up the DeepSet HNM model.

## Repo reorg awareness (2026-05-27)

CPPro split into `CPPro_current/` (active v6 work, dataset, 6B) and `CPPro_legacy/` (v4-era).
`tools/novelty.py` and `target_pipeline.py` resolve paths across this split. The dataset is at
`CPPro/CPPro_current/dataset/splits_v6/`.
