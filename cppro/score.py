"""Score peptides for cell-penetration with the open-weight CPPro-600M.

Pulls the 5-seed SeqCNN ensemble from the Hugging Face Hub (mischievers/CPPro-600M)
and the ESM-C 600M backbone from EvolutionaryScale; no API key, no local checkpoints.

    cppro-score --seq RRRRRRRRR
    cppro-score --csv designs.csv --out scored.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .embed import IN_DIM, embed, load_esmc600m
from .heads import build_head

HF_REPO = "mischievers/CPPro-600M"
N_SEEDS = 5


def load_ensemble(device: str = "cpu", head: str = "seqcnn"):
    from huggingface_hub import hf_hub_download

    models = []
    for s in range(N_SEEDS):
        sd = torch.load(hf_hub_download(HF_REPO, f"seed{s}.pt"),
                        map_location=device, weights_only=True)
        m = build_head(head, in_dim=IN_DIM).to(device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models


def score_sequences(seqs: list[str], device: str = "cpu"):
    """Return (P(CPP) per sequence, per-seed std). High std flags an unstable/OOD call."""
    backbone, toks = load_esmc600m(device)
    xtok, mask = embed(backbone, toks, seqs, device)
    models = load_ensemble(device)
    xt = torch.from_numpy(xtok).to(device)
    mk = torch.from_numpy(mask).to(device)
    per = []
    with torch.no_grad():
        for m in models:
            per.append(torch.sigmoid(m(xt, mk)).cpu().numpy())
    per = np.stack(per)
    return per.mean(0), per.std(0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score peptides for cell-penetration (open-weight CPPro-600M).")
    ap.add_argument("--seq", help="score one peptide")
    ap.add_argument("--csv", help="score a CSV (auto-detects the sequence column)")
    ap.add_argument("--out", help="output CSV path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.seq:
        p, s = score_sequences([args.seq], args.device)
        print(f"P(CPP) = {p[0]:.4f}   (per-seed std {s[0]:.4f})")
    elif args.csv:
        import pandas as pd

        df = pd.read_csv(args.csv)
        col = next((c for c in ("sequence", "seq", "designed_sequence", "peptide")
                    if c in df.columns), None)
        if col is None:
            raise SystemExit("no sequence column (tried sequence/seq/designed_sequence/peptide)")
        p, s = score_sequences(df[col].astype(str).tolist(), args.device)
        df["cppro_prob_600m"] = p
        df["cppro_std_600m"] = s
        out = args.out or (args.csv.rsplit(".", 1)[0] + "_scored.csv")
        df.to_csv(out, index=False)
        print(f"wrote {out}  ({len(df)} peptides scored)")
    else:
        raise SystemExit("provide --seq or --csv")


if __name__ == "__main__":
    main()
