"""Prediction heads (spec §3.3): classifier MLP on the embedding ``z``.

Loss: BCE-with-logits with optional class weights (balanced => pos_weight
computed from source-period labels only).
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    """MLP producing the malware logit from the feature embedding z."""

    def __init__(self, in_dim: int = 64, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return malware logits with shape ``(batch,)``."""
        return self.net(z).squeeze(-1)


def balanced_pos_weight(y_source: np.ndarray) -> float:
    """pos_weight = n_neg / n_pos from SOURCE labels only (leakage rule)."""
    y = np.asarray(y_source)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0:
        raise ValueError("source labels contain no positive (malware) rows")
    return max(1e-3, n_neg / n_pos)


def classification_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    """BCE with optional class weighting."""
    y = y.float()
    weight = None
    if pos_weight is not None:
        weight = torch.tensor([float(pos_weight)], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=weight)


def malware_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """2-class softmax from the single malware logit: ``[P(benign), P(malware)]``.

    CDAN-style conditioning needs a proper 2-column probability matrix; the
    classifier emits one logit, so build the pair ``[1-sigmoid, sigmoid]``.
    """
    p = torch.sigmoid(logits)
    return torch.stack([1.0 - p, p], dim=1)


def build_pos_weight(cfg: Mapping, y_source: np.ndarray) -> Optional[float]:
    if str(cfg.get("class_weights", "none")).lower() == "balanced":
        return balanced_pos_weight(y_source)
    return None
