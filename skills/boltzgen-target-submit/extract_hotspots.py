#!/usr/bin/env python
"""Extract receptor binding hotspots from a co-crystal and emit a starter BoltzGen yaml.

Given a structure (PDB or CIF, optionally .gz) and a receptor/ligand selection, find
every receptor residue with any heavy atom within `cutoff` A (default 5.0) of any
heavy atom of the natural ligand. Those residues become the `binding:` hotspots in a
receptor-targeted BoltzGen design spec.

This is the RECEPTOR-TARGETED direction (correct): hotspots on the receptor protein,
ligand excluded from the design. Do NOT put hotspots on the ligand peptide (the
"ligand-decoy" mistake caught in 3/5 of the Round-1 yamls).

Selections accept a chain, or a chain + residue range for intramolecular co-crystals
(e.g. a C-terminal autoinhibitory domain masking an N-terminal domain in one chain):
    A                 whole chain A
    A,B               chains A and B (multi-chain receptor)
    A:1-275           chain A residues 1..275 (sub-domain)

Usage:
    extract_hotspots.py STRUCTURE --receptor SEL --ligand SEL [--cutoff 5.0]
                        [--length 20..50] [--out-yaml PATH]

Examples:
    # inter-chain co-crystal (most common): Bcl-xL (A) vs BIM BH3 (B)
    extract_hotspots.py 4QVE.pdb --receptor A --ligand B --out-yaml d_bclxl.yaml

    # intramolecular: GSDMD NT pore domain (A:1-275) masked by CT (A:276-484)
    extract_hotspots.py 6N9O.pdb --receptor A:1-275 --ligand A:276-484
"""
import argparse
import gzip
import sys

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'MSE': 'M', 'SEC': 'U',
}


def parse_selection(sel):
    """'A:1-275' -> ('A', 1, 275);  'A' -> ('A', None, None). Comma-split upstream."""
    out = []
    for part in sel.split(','):
        part = part.strip()
        if ':' in part:
            chain, rng = part.split(':', 1)
            lo, hi = rng.split('-')
            out.append((chain, int(lo), int(hi)))
        else:
            out.append((part, None, None))
    return out


def _open(path):
    return gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path, 'rt')


def parse_atoms(path):
    """Yield (chain, resseq, resname, atomname, x, y, z) heavy ATOM records.

    Handles PDB. For CIF, parses the _atom_site loop. Skips hydrogens + HETATM
    (so ligands/ions/waters are ignored; the natural ligand here is a *protein*
    peptide chain, parsed via ATOM records).
    """
    is_cif = str(path).lower().replace('.gz', '').endswith(('.cif', '.mmcif'))
    if is_cif:
        yield from _parse_cif_atoms(path)
        return
    with _open(path) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            atomname = line[12:16].strip()
            if atomname.startswith('H'):
                continue
            resname = line[17:20].strip()
            chain = line[21].strip()
            try:
                resseq = int(line[22:26])
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            yield chain, resseq, resname, atomname, x, y, z


def _parse_cif_atoms(path):
    """Minimal _atom_site loop parser (auth_asym_id / auth_seq_id, ATOM group only)."""
    with _open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == 'loop_':
            cols, j = [], i + 1
            while j < len(lines) and lines[j].lstrip().startswith('_atom_site.'):
                cols.append(lines[j].strip().split('.', 1)[1])
                j += 1
            if cols:
                idx = {c: k for k, c in enumerate(cols)}
                need = ('group_PDB', 'label_atom_id', 'label_comp_id', 'Cartn_x', 'Cartn_y', 'Cartn_z')
                if all(c in idx for c in need):
                    chain_col = 'auth_asym_id' if 'auth_asym_id' in idx else 'label_asym_id'
                    seq_col = 'auth_seq_id' if 'auth_seq_id' in idx else 'label_seq_id'
                    k = j
                    while k < len(lines) and not lines[k].lstrip().startswith(('_', 'loop_', '#')):
                        row = lines[k].split()
                        k += 1
                        if len(row) < len(cols) or row[idx['group_PDB']] != 'ATOM':
                            continue
                        atomname = row[idx['label_atom_id']].strip('"')
                        if atomname.startswith('H'):
                            continue
                        try:
                            resseq = int(row[idx[seq_col]])
                            x, y, z = (float(row[idx['Cartn_x']]),
                                       float(row[idx['Cartn_y']]),
                                       float(row[idx['Cartn_z']]))
                        except (ValueError, KeyError):
                            continue
                        yield (row[idx[chain_col]], resseq, row[idx['label_comp_id']],
                               atomname, x, y, z)
                    i = k
                    continue
        i += 1


def in_sel(chain, resseq, sel):
    for c, lo, hi in sel:
        if chain == c and (lo is None or lo <= resseq <= hi):
            return True
    return False


def extract(path, receptor_sel, ligand_sel, cutoff):
    rec, lig = [], []
    resname_by = {}
    for chain, resseq, resname, atom, x, y, z in parse_atoms(path):
        r = in_sel(chain, resseq, receptor_sel)
        l = in_sel(chain, resseq, ligand_sel)
        if r and not l:
            rec.append((chain, resseq, x, y, z))
            resname_by[(chain, resseq)] = resname
        elif l and not r:
            lig.append((x, y, z))
    cut2 = cutoff * cutoff
    hits = {}  # chain -> set(resseq)
    for c, rs, rx, ry, rz in rec:
        for lx, ly, lz in lig:
            dx, dy, dz = rx - lx, ry - ly, rz - lz
            if dx * dx + dy * dy + dz * dz <= cut2:
                hits.setdefault(c, set()).add(rs)
                break
    return {c: sorted(v) for c, v in hits.items()}, resname_by, len(rec), len(lig)


def label(chain, residues, resname_by):
    return ' '.join(THREE_TO_ONE.get(resname_by.get((chain, r), 'XXX'), 'X') + str(r)
                    for r in residues)


def build_yaml(path, receptor_sel, hits, resname_by, length, cutoff, ligand_sel):
    rec_chains = []
    for c, _, _ in receptor_sel:
        if c not in rec_chains:
            rec_chains.append(c)
    include = '\n'.join(f'        - chain:\n            id: {c}' for c in rec_chains)
    lig_str = ','.join(c for c, _, _ in ligand_sel)
    blocks = []
    for c in rec_chains:
        res = hits.get(c, [])
        if not res:
            continue
        blocks.append(
            f'  # chain {c}: receptor residues within {cutoff} A of ligand ({lig_str})\n'
            f'  # {label(c, res, resname_by)}\n'
            f'  - chain:\n'
            f'      id: {c}\n'
            f'      binding: ' + ','.join(str(r) for r in res))
    sub = any(lo is not None for _, lo, _ in receptor_sel)
    warn = ('# NOTE: receptor selection used a residue sub-range. Confirm whether your\n'
            '# BoltzGen include block should restrict to that range (vs the whole chain).\n'
            if sub else '')
    return f"""# BoltzGen design spec (starter — EDIT the header with target + mechanism rationale)
# Receptor-targeted: hotspots on the receptor protein; natural ligand EXCLUDED from include.
# Hotspots = receptor residues within {cutoff} A (heavy-atom) of the natural ligand.
{warn}
entities:
  - protein:
      id: P
      sequence: {length}

  - file:
      path: {path.split('/')[-1]}
      include:
{include}

binding_types:
{chr(10).join(blocks)}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('structure')
    ap.add_argument('--receptor', required=True, help='e.g. A  or  A,B  or  A:1-275')
    ap.add_argument('--ligand', required=True, help='e.g. B  or  A:276-484')
    ap.add_argument('--cutoff', type=float, default=5.0, help='heavy-atom cutoff A (default 5.0)')
    ap.add_argument('--length', default='20..50', help='designed peptide length range')
    ap.add_argument('--out-yaml', help='write a starter yaml to this path')
    a = ap.parse_args()

    rsel, lsel = parse_selection(a.receptor), parse_selection(a.ligand)
    hits, resname_by, n_rec, n_lig = extract(a.structure, rsel, lsel, a.cutoff)

    if n_lig == 0:
        sys.exit(f'ERROR: 0 ligand heavy atoms for selection {a.ligand} — check chains '
                 f'(receptor parsed {n_rec} atoms). For CIF, auth chain IDs are used.')
    total = sum(len(v) for v in hits.values())
    print(f'receptor atoms: {n_rec}   ligand atoms: {n_lig}   cutoff: {a.cutoff} A')
    if total == 0:
        sys.exit('WARNING: 0 contact residues — receptor and ligand do not contact, '
                 'or chain assignment is reversed.')
    for c in sorted(hits):
        res = hits[c]
        print(f'\nchain {c}: {len(res)} hotspots')
        print('  resi:   ' + ','.join(str(r) for r in res))
        print('  labels: ' + label(c, res, resname_by))

    if a.out_yaml:
        with open(a.out_yaml, 'w') as f:
            f.write(build_yaml(a.structure, rsel, hits, resname_by, a.length, a.cutoff, lsel))
        print(f'\nwrote starter yaml: {a.out_yaml}')


if __name__ == '__main__':
    main()
