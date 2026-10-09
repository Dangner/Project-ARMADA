"""FGSM feature-space attack with EMBER constraints (spec §3.6).

One-step signed-gradient perturbation in the model's processed input space,
followed by :class:`FeatureSpaceProjector` projection (count-like features
non-negative, per-column source max, frozen missingness).  Feature-space
only — not guaranteed to be valid PE files.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F

from .constraints import FeatureSpaceProjector


def fgsm_attack(
    model: torch.nn.Module,
    groups_clean: Mapping[str, torch.Tensor],
    y: torch.Tensor,
    eps: float,
    projector: Optional[FeatureSpaceProjector] = None,
    clip_min: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Return adversarial group blocks within an l∞ ball of radius ``eps``.

    Parameters
    ----------
    groups_clean: processed group tensors ``(batch, dim_g)``.
    y: true labels ``(batch,)`` in {0,1} (evaluation-only labels; attacks are
        white-box untargeted — standard robustness protocol).
    eps: perturbation radius in standardised feature units.
    """
    if eps <= 0:
        return {k: v.clone() for k, v in groups_clean.items()}
    was_training = model.training
    model.eval()
    adv = {k: v.clone().detach() for k, v in groups_clean.items()}
    for k in adv:
        adv[k].requires_grad_(True)

    logits, _ = model(adv)
    loss = F.binary_cross_entropy_with_logits(logits, y.float())
    grads = torch.autograd.grad(loss, [adv[k] for k in adv], allow_unused=True)

    with torch.no_grad():
        out: Dict[str, torch.Tensor] = {}
        free = projector.free_mask(groups_clean) if projector is not None else None
        for (name, x), g in zip(adv.items(), grads):
            if g is None:
                g = torch.zeros_like(x)
            delta = eps * g.sign()
            if free is not None:
                delta = delta * free[name].to(delta.dtype)
            x_adv = x.detach() + delta
            out[name] = x_adv
        if projector is not None:
            out = projector.project(out, groups_clean, clip_min=clip_min)
        # final safety: keep everything inside the l∞ ball around clean
        for name in out:
            out[name] = torch.max(
                torch.min(out[name], groups_clean[name] + eps), groups_clean[name] - eps
            )
    if was_training:
        model.train()
    return out
