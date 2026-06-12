"""CPPro — open-weight cell-penetrating-peptide classifier (ESM-C 600M + SeqCNN)."""

from .heads import HEAD_REGISTRY, build_head
from .score import score_sequences

__version__ = "0.6.0"
__all__ = ["build_head", "HEAD_REGISTRY", "score_sequences", "__version__"]
