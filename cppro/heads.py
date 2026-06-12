"""Classifier heads on frozen PLM embeddings (registry + factory).

input_type selects the feature view: 'mean' (B, in_dim) or 'token' (B, L, in_dim) + mask.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

HEAD_REGISTRY: dict[str, type[nn.Module]] = {}


def register_head(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    def deco(cls: type[nn.Module]) -> type[nn.Module]:
        HEAD_REGISTRY[name] = cls
        return cls

    return deco


@register_head("mlp")
class MLPHead(nn.Module):
    input_type = "mean"

    def __init__(self, in_dim: int = 1152, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, mask=None):
        return self.net(x).squeeze(-1)


@register_head("cnn")
class CNNHead(nn.Module):
    input_type = "mean"

    def __init__(self, in_dim: int = 1152, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * (in_dim // 4), hidden)
        self.drop3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = self.drop1(self.pool1(torch.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(torch.relu(self.bn2(self.conv2(x)))))
        return self.fc2(self.drop3(torch.relu(self.fc1(x.flatten(1))))).squeeze(-1)


@register_head("transformer")
class TransformerHead(nn.Module):
    input_type = "token"

    def __init__(self, in_dim=1152, model_dim=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, model_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(model_dim, n_heads, 4 * model_dim, dropout,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))

    def forward(self, x, mask):
        b = x.size(0)
        x = self.proj_in(x)
        x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1)
        cls_mask = torch.ones(b, 1, device=mask.device, dtype=mask.dtype)
        mask = torch.cat([cls_mask, mask], dim=1)
        out = self.encoder(x, src_key_padding_mask=(mask == 0))
        return self.head(out[:, 0]).squeeze(-1)


@register_head("deepset")
class DeepSetHead(nn.Module):
    input_type = "token"

    def __init__(self, in_dim=1152, hidden=256, dropout=0.3):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    def forward(self, x, mask):
        h = self.phi(x)
        m = mask.unsqueeze(-1).float()
        return self.rho((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


@register_head("seqcnn")
class SeqCNNHead(nn.Module):
    """Masked sequence-CNN (the portable 600M production head)."""

    input_type = "token"

    def __init__(self, in_dim=1152, filters=128, kernel=5, hidden=256, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, filters, kernel, padding=kernel // 2)
        self.bn1 = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters, filters, kernel, padding=kernel // 2)
        self.bn2 = nn.BatchNorm1d(filters)
        self.fc1 = nn.Linear(filters, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        m = mask.unsqueeze(1)
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x))) * m
        x = torch.relu(self.bn2(self.conv2(x))) * m
        x = x.masked_fill(m == 0, float("-inf")).max(dim=2).values
        return self.fc2(self.drop(torch.relu(self.fc1(x)))).squeeze(-1)


def build_head(name: str, in_dim: int = 1152, **kw) -> nn.Module:
    if name not in HEAD_REGISTRY:
        raise KeyError(f"unknown head '{name}'; have {sorted(HEAD_REGISTRY)}")
    return HEAD_REGISTRY[name](in_dim=in_dim, **kw)
