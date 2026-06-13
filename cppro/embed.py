"""Local ESM-C 600M embedding (open weights, no API key)."""

from __future__ import annotations

import os

import numpy as np
import torch

IN_DIM = 1152
L_MAX = 66  # 64 aa cap + 2 special tokens (matches the v6 training cache)


def load_esmc600m(device: str = "cpu"):
    """Build ESM-C 600M and load the open EvolutionaryScale weights from the Hub."""
    import esm.tokenization as tk
    from esm.models.esmc import ESMC
    from huggingface_hub import snapshot_download

    snap = snapshot_download("EvolutionaryScale/esmc-600m-2024-12")
    wpath = os.path.join(snap, "data", "weights", "esmc_600m_2024_12_v0.pth")
    toks = tk.get_esmc_model_tokenizers()
    model = ESMC(d_model=1152, n_heads=18, n_layers=36, tokenizer=toks, use_flash_attn=False).eval()
    miss, unexp = model.load_state_dict(
        torch.load(wpath, map_location="cpu", weights_only=True), strict=False)
    assert not miss and not unexp, f"state_dict mismatch: missing={len(miss)} unexpected={len(unexp)}"
    return model.to(device), toks


def embed(model, toks, seqs: list[str], device: str = "cpu"):
    """Per-token embeddings (N, L_MAX, 1152) + mask (N, L_MAX), matching the training cache."""
    n = len(seqs)
    xtok = np.zeros((n, L_MAX, IN_DIM), dtype=np.float32)
    mask = np.zeros((n, L_MAX), dtype=np.float32)
    with torch.no_grad():
        for i, s in enumerate(seqs):
            ids = torch.tensor(toks.encode(str(s).strip()), dtype=torch.long, device=device).unsqueeze(0)
            if device == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    h = model(ids).embeddings[0].float().cpu().numpy()
            else:
                h = model(ids).embeddings[0].float().cpu().numpy()
            length = min(h.shape[0], L_MAX)
            xtok[i, :length] = h[:length]
            mask[i, :length] = 1.0
    return xtok, mask
