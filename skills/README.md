# Skills

Two Claude Code skills for the CPP-Pro / BoltzGen peptide-design pipeline.

> **Tailored to the author's working layout.** These skills assume a single root containing
> `CPPro/CPPro_current/`, `BoltzGen_pipeline/`, and `tools/`. The repo-internal paths they
> reference will **not** resolve inside CPP-Pro alone, so treat them as a reference for the
> workflow and adapt the paths to your own structure. The BoltzGen pipeline scripts that
> Skill 2 drives live in the separate `VicCar/BoltzGen_pipeline` repo. Imperial HPC paths use
> a `<user>` placeholder. The CPP-Pro scorer needs a Biohub/Forge API key (gitignored, never
> committed).

## Skill 1 — `boltzgen-target-submit`  (co-crystal → submittable BoltzGen job)

Turn a receptor-ligand co-crystal (PDB/CIF) into a submittable BoltzGen peptide-design job:
identify the receptor vs the natural ligand, extract the 5 Å binding hotspots, write the
design `.yaml` + Imperial `.pbs`, and a PyMOL `.pse` to eyeball the site before submitting.
Ships two helpers:
- `extract_hotspots.py` — 5.0 Å heavy-atom contact extractor (PDB/CIF, inter- or intra-chain) → residue list + starter yaml.
- `build_verify_pse.py` — PyMOL session: pocket residues blue, natural ligand red, receptor grey.

## Skill 2 — `boltzgen-target-screen`  (BoltzGen output → top hits)

Score a completed design run and curate a shortlist:
CPP-Pro cell-penetration (ESM-C 6B via Forge + DeepSet HNM ensemble) → filter funnel
(BoltzGen filters → CPP-Pro gate → ipSAE binding composite → novelty → orthogonal panel) →
a hand-curated final selection written to `SELECTION.md`.
