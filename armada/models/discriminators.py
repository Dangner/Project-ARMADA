"""Domain discriminators for adversarial domain adaptation (spec §3.4).

D1 (marginal): binary source/target discriminator on the embedding ``z``.
D2 (class-conditional, CDAN-style): discriminator on the outer product of
``z`` and the classifier softmax, optionally through a fixed randomised
multilinear map, with optional entropy conditioning (CDAN+E).

Both consume GRL-reversed features so the encoder is trained to confuse them.
"""

from __future__ import annotations

from typing import Optional, Tuple

import math

import torch
import torch.nn as nn

from .grl import grad_reverse


class MarginalDiscriminator(nn.Module):
    """D1: marginal domain discriminator on z (0 = source, 1 = target)."""

    def __init__(self, in_dim: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        """Domain logits ``(batch,)``; ``lambd`` reverses gradients into z."""
        return self.net(grad_reverse(z, lambd)).squeeze(-1)


def cdan_joint_features(
    z: torch.Tensor,
    class_probs: torch.Tensor,
    random_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multilinear conditioning features: outer product p (x) z.

    ``class_probs`` is the classifier softmax ``(batch, n_classes)``; the
    outer product ``(batch, n_classes, d)`` is flattened to
    ``(batch, n_classes * d)``.  When ``random_map`` is given (fixed
    ``(n_classes * d, map_dim)`` Gaussian matrix), the joint vector is
    projected down (randomised multilinear map, keeps D2 small).
    """
    if class_probs.dim() != 2 or z.dim() != 2:
        raise ValueError("z and class_probs must be 2D")
    if class_probs.shape[0] != z.shape[0]:
        raise ValueError("batch mismatch between z and class_probs")
    joint = torch.bmm(class_probs.unsqueeze(2), z.unsqueeze(1))  # (B, C, d)
    joint = joint.reshape(joint.shape[0], -1)
    if random_map is not None:
        joint = joint @ random_map.to(joint.dtype)
    return joint


class ConditionalDiscriminator(nn.Module):
    """D2: class-conditional (CDAN) domain discriminator."""

    def __init__(
        self,
        feat_dim: int = 64,
        n_classes: int = 2,
        hidden: int = 128,
        random_map_dim: Optional[int] = None,
        entropy_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.entropy_conditioning = entropy_conditioning
        in_dim = feat_dim * n_classes
        if random_map_dim is not None:
            gen = torch.Generator().manual_seed(0)  # fixed, not learned
            proj = torch.randn(in_dim, random_map_dim, generator=gen) / math.sqrt(in_dim)
            self.register_buffer("random_map", proj)
            in_dim = random_map_dim
        else:
            self.random_map = None
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def entropy_weight(class_probs: torch.Tensor) -> torch.Tensor:
        """CDAN+E confidence weight ``1 - H(p) / log(C)`` in [0, 1]."""
        C = class_probs.shape[-1]
        if C <= 1:
            return torch.ones(class_probs.shape[0], device=class_probs.device)
        H = -(class_probs.clamp_min(1e-8).log() * class_probs).sum(dim=-1)
        return (1.0 - H / math.log(C)).clamp(0.0, 1.0)

    def forward(
        self,
        z: torch.Tensor,
        class_probs: torch.Tensor,
        lambd: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (domain logits ``(batch,)``, per-sample entropy weights or None)."""
        joint = cdan_joint_features(
            z, class_probs, random_map=self.random_map if hasattr(self, "random_map") else None
        )
        logits = self.net(grad_reverse(joint, lambd)).squeeze(-1)
        weights = self.entropy_weight(class_probs) if self.entropy_conditioning else None
        return logits, weights
