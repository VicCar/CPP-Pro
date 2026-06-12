---
name: boltzgen-target-submit
description: Turn a receptor-ligand co-crystal (PDB/CIF) into a submittable BoltzGen peptide-design job — extract binding hotspots, write the design yaml + Imperial PBS, and a PyMOL .pse to eyeball the site before submitting. Use when starting a new BoltzGen target, the user says "design a peptide against <PDB>", "build a BoltzGen job for <receptor>", "set up a new design target", or "make the spec for <co-crystal>". The mirror of boltzgen-target-screen (which scores the run's output).
---

# BoltzGen target submission (co-crystal -> submittable job)

A receptor-ligand co-crystal goes in; a complete `jobs/<date>/<job>/` folder comes out
(design `.yaml` + Imperial `.pbs` + a verification `.pse` + a README), ready for the
user to push and `qsub` on Imperial. This is the input side; `boltzgen-target-screen`
is the return trip (scoring the downloaded designs).

**You author the files locally. The user pushes/qsubs on Imperial themselves — you
cannot run on Imperial.** Stop after the job folder is built and the `.pse` is verified.

## The one decision that matters most: target the RECEPTOR, not the ligand

A co-crystal has a receptor protein and a natural ligand (often a peptide). You design a
peptide to occupy the **receptor's** binding site. So:

- **Binding hotspots go on the RECEPTOR chain.** ✅
- **The natural ligand is EXCLUDED from the design `include` block.** ✅
- Putting hotspots on the ligand peptide ("ligand-decoy") is **wrong for almost every
  goal** — it was the mistake in 3 of 5 Round-1 yamls. If you catch yourself selecting the
  short peptide chain as the design target, stop: that is backwards.

Decide receptor vs ligand explicitly from the structure (the receptor is the larger
folded protein/domain; the ligand is the short peptide/helix sitting in its groove).
For an **intramolecular** co-crystal (e.g. a C-terminal domain auto-inhibiting an
N-terminal domain in one chain), the "receptor" is the domain you design against and the
"ligand" is the masking domain — split by residue range.

## Steps

### 1. Get the structure + identify chains
Download from RCSB (`.pdb` or `.cif`). Find the receptor chain and the natural-ligand
chain. If unsure, the contact extractor (step 2) reports atom counts per selection — a
selection with very few residues is the ligand.

### 2. Extract hotspots (5.0 Å heavy-atom) + a starter yaml
```bash
~/miniconda3/bin/python extract_hotspots.py STRUCTURE \
    --receptor A --ligand B --out-yaml <job>.yaml
```
- `--receptor` / `--ligand` accept `A`, multi-chain `A,B`, or a sub-range `A:1-275`
  (intramolecular).
- Cutoff is **5.0 Å** for design hotspots (broader than the 4 Å used for post-hoc
  interface labelling — captures second-shell residues a designed peptide may reach).
- Prints residue numbers + one-letter labels and writes a starter yaml.

### 3. Finish the yaml
Edit the starter yaml's header comment with the **target identity + mechanism rationale**
(what the peptide does: occupy / block / mimic, and why that matters). Set the designed
peptide length (`sequence: 20..50` is the CPP-compatible default). Confirm the `include`
block lists only receptor chains.

```yaml
entities:
  - protein:
      id: P
      sequence: 20..50          # designed peptide length range (CPP-compatible)
  - file:
      path: <PDB>.pdb
      include:
        - chain:
            id: A               # receptor chain(s) only — ligand excluded
binding_types:
  - chain:
      id: A
      binding: 93,96,97,...     # receptor hotspots from step 2
```

### 4. Write the PBS (Imperial)
**Critical infra constraints (these break the run if wrong):**
- **`gpu_type=L40S`, NOT A100** — BoltzGen dies on cx3 A100 nodes loading the
  cuequivariance kernel (cuBLAS < 12.5).
- **`--use_kernels false`** — after the bg-env rebuild, the kernel can't find
  `libcublas.so.12` even on L40S; the pure-torch path sidesteps it.
- Protocol `peptide-anything`, `--devices 1`.
- **Ask the user for `--num_designs` and `--budget`** — do not assume them. They trade off
  run time against how many candidates reach the screening step, and depend on pocket size
  and GPU budget. A typical starting point is `--num_designs 200 --budget 100`; offer that
  as a default but confirm before writing the PBS.
- Outputs to `$EPHEMERAL`; logs to the user's RDS `BoltzGen_pipeline/logs/`.

```bash
#!/bin/bash
#PBS -N bg_<job>
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /rds/general/user/<user>/home/BoltzGen_pipeline/logs/

module load Miniforge3/24.11.3-0
source activate bg
cd $HOME/BoltzGen_pipeline/jobs/<date>

boltzgen run <job>.yaml \
    --output $EPHEMERAL/results/<job> \
    --protocol peptide-anything \
    --num_designs <ASK USER> --budget <ASK USER> --devices 1 \
    --use_kernels false \
    --cache $EPHEMERAL/boltzgen_cache --reuse
```

### 5. Build the verification .pse (eyeball before submit)
```bash
~/miniconda3/bin/pymol -cq build_verify_pse.py -- STRUCTURE \
    --pocket 93,96,97,... --receptor A --ligand B --out <job>_site.pse
```
Pocket residues **blue**, natural ligand **red**, rest of receptor grey. Open in PyMOL
and confirm the blue patch is the cleft you intend and the red ligand sits in it. If the
blue is scattered or the ligand is elsewhere, the chain/residue selections are wrong —
fix before submitting.

### 6. Write the README + hand off
A short `README.md` in the job folder: target, PDB, receptor/ligand chains, pocket
residues, mechanism, designed length, protocol/budget, and any caveats (see below). Then
tell the user the folder is ready to push + `qsub`. After the run downloads, scoring is
`boltzgen-target-screen`.

## Caveats to record per target (so interpretation isn't blindsided)
- **Cofactor/conformational dependencies.** If the pocket only exists in a liganded or
  metal-loaded state (Ca²⁺ EF-hands, etc.), the template coords must already be in that
  state. Note whether ions are entered explicitly (convention so far: chain-only include,
  ions not entered — flag it as the first thing to try if hit geometry looks wrong).
- **Biological unit vs asymmetric unit.** If the real unit is a dimer/oligomer but all
  hotspots fall on one protomer, designing against that protomer is a valid simplification
  — state it.
- **`strong_bind_iptm` placeholder.** This belongs to the *screening* step
  (`boltzgen-target-screen`), NOT to the yaml/pbs you write here. Just record a placeholder
  (e.g. 0.75) in the README; it gets re-tuned from `design_to_target_iptm.describe()` after
  results download.

## Files in this skill
| file | what |
|---|---|
| `extract_hotspots.py` | 5.0 Å heavy-atom contact extractor (PDB/CIF, inter- or intra-chain) → residue list + starter yaml |
| `build_verify_pse.py` | PyMOL session: pocket blue, ligand red, receptor grey |

## Where things live (repo conventions)
- **Job folder layout (one convention):** write into a single dated folder
  `BoltzGen_pipeline/jobs/<date>/` (add a short tag for theme runs, e.g.
  `jobs/2026-06-05-fap-cleavage/`). Files are named by job: `<job>.yaml`, `<job>.pbs`,
  `<job>_site.pse`, plus one `README.md` for the folder. A dated folder may hold several
  related jobs. Do **not** nest a per-job subfolder; the `.pbs` `cd` target is the dated
  folder. The PDB/CIF lives alongside.
- The full methods log + prior worked examples: `BoltzGen_pipeline/METHODS.md`.
- Imperial paths assume user `<user>`; RDS home holds `BoltzGen_pipeline/`, `$EPHEMERAL`
  holds results + cache.
