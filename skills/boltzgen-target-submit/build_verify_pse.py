#!/usr/bin/env python
"""Build a PyMOL .pse to eyeball a BoltzGen design site BEFORE submitting.

Loads the co-crystal and colours:
  - binding-pocket (hotspot) residues  -> BLUE  (sticks + cartoon)
  - natural ligand                     -> RED   (the pose a designed peptide must mimic/block)
  - rest of receptor                   -> grey cartoon

Open the saved .pse in PyMOL and confirm the blue patch is the cleft you intend to
design against and the red ligand sits in it. If it looks wrong, the chain/residue
selections are wrong — fix before writing the job.

Run with the project python (PyMOL must be importable):
    ~/miniconda3/bin/pymol -cq build_verify_pse.py -- STRUCTURE \\
        --pocket 93,96,97,... --receptor A --ligand B [--out site.pse]

Selections match extract_hotspots.py: chain 'A', multi 'A,B', or sub-range 'A:1-275'.
"""
import argparse
import sys
from pymol import cmd

SHARED = dict(ray_shadows=0, ambient=0.40, specular=0.2, stick_radius=0.22,
              antialias=2, cartoon_fancy_helices=1, cartoon_side_chain_helper=1)


def sel_expr(obj, sel):
    """Turn 'A' / 'A,B' / 'A:1-275' into a PyMOL selection string within `obj`."""
    parts = []
    for token in sel.split(','):
        token = token.strip()
        if ':' in token:
            chain, rng = token.split(':', 1)
            lo, hi = rng.split('-')
            parts.append(f'({obj} and chain {chain} and resi {lo}-{hi})')
        else:
            parts.append(f'({obj} and chain {token})')
    return ' or '.join(parts)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('structure')
    ap.add_argument('--pocket', required=True, help='comma-sep hotspot residue numbers')
    ap.add_argument('--receptor', required=True, help='e.g. A or A,B or A:1-275')
    ap.add_argument('--ligand', required=True, help='e.g. B or A:276-484')
    ap.add_argument('--receptor-chain', default=None,
                    help='chain the pocket residue numbers belong to (default: first receptor chain)')
    ap.add_argument('--out', default='verify_site.pse')
    a = ap.parse_args(argv)

    cmd.reinitialize()
    cmd.load(a.structure, 'm')
    cmd.remove('solvent'); cmd.remove('hydro')
    cmd.bg_color('white'); cmd.hide('everything')

    rec = sel_expr('m', a.receptor)
    lig = sel_expr('m', a.ligand)
    pchain = a.receptor_chain or a.receptor.split(',')[0].split(':')[0]
    pocket = f'(m and chain {pchain} and resi {a.pocket})'

    cmd.show('cartoon', rec); cmd.color('grey80', rec)
    cmd.color('marine', pocket)
    cmd.show('sticks', f'({pocket}) and not (name C+N+O)')

    cmd.show('cartoon', lig); cmd.color('red', lig)
    cmd.show('sticks', f'({lig}) and not (name C+N+O)')

    for k, v in SHARED.items():
        cmd.set(k, v)
    cmd.orient(f'({pocket}) or ({lig})')
    cmd.zoom(f'({pocket}) or ({lig})', 3)
    cmd.save(a.out)

    got = set()
    cmd.iterate(f'({pocket}) and name CA', 'got.add(int(resi))', space={'got': got})
    print(f'saved {a.out}  |  pocket CA shown: {len(got)}  |  ligand atoms: '
          f'{cmd.count_atoms(lig)}')


# PyMOL runs this via `pymol -cq build_verify_pse.py -- ...`, where __name__ == 'pymol'
# (not '__main__') and the post-`--` args land in sys.argv[1:]. Call main directly.
main(sys.argv[1:])
