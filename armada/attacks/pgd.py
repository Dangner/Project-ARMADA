"""PGD feature-space attack with EMBER constraints (spec §3.6).

Iterative projected signed-gradient descent in the model's processed input
space.  Each step is projected by :class:`FeatureSpaceProjector`
(count-like features non-negative, per-column source max, frozen
missingness) and clipped back into the l∞ ball around the clean input.
Feature-space only — not guaranteed to be valid PE files.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F

from .constraints import FeatureSpaceProjector
from .fgsm import fgsm_attack


def pgd_attack(
    model: torch.nn.Module,
    groups_clean: Mapping[str, torch.Tensor],
    y: torch.Tensor,
    eps: float,
    step_size: float = 0.01,
    steps: int = 7,
    projector: Optional[FeatureSpaceProjector] = None,
    clip_min: float = 0.0,
    random_start: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Return adversarial group blocks from ``steps`` PGD iterations."""
    if eps <= 0:
        return {k: v.clone() for k, v in groups_clean.items()}
    if steps <= 1:
        return fgsm_attack(model, groups_clean, y, eps, projector, clip_min)

    was_training = model.training
    model.eval()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=groups_clean[next(iter(groups_clean))].device)
        generator.manual_seed(seed)

    free = projector.free_mask(groups_clean) if projector is not None else None
    adv = {k: v.clone().detach() for k, v in groups_clean.items()}
    if random_start:
        with torch.no_grad():
            for name in adv:
                delta = torch.empty_like(adv[name]).uniform_(-eps, eps, generator=generator)
                if free is not None:
                    delta = delta * free[name].to(delta.dtype)
                adv[name] = adv[name] + delta
            if projector is not None:
                adv = projector.project(adv, groups_clean, clip_min=clip_min)

    for _ in range(int(steps)):
        for k in adv:
            adv[k] = adv[k].detach().requires_grad_(True)
        logits, _ = model(adv)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        grads = torch.autograd.grad(loss, [adv[k] for k in adv], allow_unused=True)
        with torch.no_grad():
            out: Dict[str, torch.Tensor] = {}
            for (name, x), g in zip(adv.items(), grads):
                if g is None:
                    g = torch.zeros_like(x)
                delta = step_size * g.sign()
                if free is not None:
                    delta = delta * free[name].to(delta.dtype)
                out[name] = x.detach() + delta
            if projector is not None:
                out = projector.project(out, groups_clean, clip_min=clip_min)
            for name in out:
                # stay inside the l∞ ball around the clean input
                out[name] = torch.max(
                    torch.min(out[name], groups_clean[name] + eps), groups_clean[name] - eps
                )
            adv = out

    if was_training:
        model.train()
    return {k: v.detach() for k, v in adv.items()}
