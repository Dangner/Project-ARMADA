"""Test-Time Training via masked group-feature reconstruction (spec §3.5).

Masking: a random subset of the 9 group tokens is replaced by a learned
``[MASK]`` token before the encoder.  Per-group decoder heads reconstruct
each masked group's *processed features* from the encoder output at that
position ("masked feature reconstruction").  The joint objective is
classification + domain + reconstruction (see :mod:`armada.train.adapt`).

TTT at inference: a few gradient steps on the reconstruction loss using the
UNLABELED rows of the current target window only, updating profiler/encoder
(and decoder) parameters including LayerNorm gains — the classifier stays
frozen.  Unless ``online`` is set, the source-trained weights are restored
before every new target window.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MaskedGroupDecoder(nn.Module):
    """Per-group decoder heads: encoder row -> reconstructed group features."""

    def __init__(self, group_dims: Mapping[str, int], d_model: int) -> None:
        super().__init__()
        self.group_names = list(group_dims)
        self.heads = nn.ModuleDict(
            {name: nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, dim))
             for name, dim in group_dims.items()}
        )

    def forward(self, z_rows: torch.Tensor, names: List[str]) -> Dict[str, torch.Tensor]:
        """``z_rows``: ``(batch, len(names), d)`` encoder outputs at group positions."""
        out = {}
        for i, name in enumerate(names):
            out[name] = self.heads[name](z_rows[:, i])
        return out


def sample_group_mask(
    batch: int,
    n_groups: int,
    mask_ratio: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Boolean ``(batch, n_groups)`` mask with at least one masked group/sample."""
    if not 0.0 < mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in (0, 1]")
    mask = torch.rand(batch, n_groups, device=device, generator=generator) < mask_ratio
    empty = ~mask.any(dim=1)
    if empty.any():
        fix = torch.randint(0, n_groups, (int(empty.sum()),), device=device, generator=generator)
        mask[empty, fix] = True
    return mask


def masked_reconstruction_loss(
    recon: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    group_names: List[str],
) -> torch.Tensor:
    """Mean MSE over masked (sample, group) entries only."""
    total = recon[group_names[0]].new_zeros(())
    n_elems = 0
    for i, name in enumerate(group_names):
        m = mask[:, i].float()  # (B,)
        if float(m.sum()) == 0:
            continue
        sq = F.mse_loss(recon[name], targets[name], reduction="none").mean(dim=-1)  # (B,)
        total = total + (sq * m).sum()
        n_elems += int(m.sum())
    if n_elems == 0:
        return total
    return total / n_elems


class TTTAdapter:
    """Per-window test-time training around an :class:`ArmadaDomainModel`.

    Always resets to the saved source-trained weights before a new window
    unless ``online=True`` (then adaptation accumulates across windows).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        steps: int = 5,
        mask_ratio: float = 0.3,
        online: bool = False,
        grad_clip: float = 1.0,
    ) -> None:
        self.model = model
        self.lr = float(lr)
        self.steps = int(steps)
        self.mask_ratio = float(mask_ratio)
        self.online = bool(online)
        self.grad_clip = float(grad_clip)
        self.base_state = copy.deepcopy(model.state_dict())

    def reset(self) -> None:
        self.model.load_state_dict(self.base_state)

    def adapt(self, unl_groups: Mapping[str, torch.Tensor], batch_size: int = 256) -> List[float]:
        """Run TTT steps on unlabeled target batches; returns step losses.

        ``unl_groups`` may hold numpy arrays (processed group blocks) or
        tensors; they are converted to model-device float32 tensors here.
        """
        if unl_groups is None or len(next(iter(unl_groups.values()))) == 0:
            return []
        device = next(self.model.parameters()).device
        groups = {
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in unl_groups.items()
        }
        params = [
            p
            for name, p in self.model.named_parameters()
            if p.requires_grad and not name.startswith("classifier.")
        ]
        optimizer = torch.optim.SGD(params, lr=self.lr)
        n = len(next(iter(groups.values())))
        losses: List[float] = []
        self.model.train()
        for _ in range(self.steps):
            idx = torch.randperm(n)[: min(batch_size, n)]
            batch = {k: v[idx] for k, v in groups.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = self.model.reconstruction_loss(batch, mask_ratio=self.mask_ratio)
            if not torch.isfinite(loss):
                # Never let one bad batch poison reported metrics: restore
                # source weights and stop adapting (prediction then degrades
                # gracefully to the source model).
                logger.warning("non-finite TTT loss; resetting to source weights")
                self.reset()
                return losses
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
        self.model.eval()
        return losses

    def state_changed(self) -> bool:
        """True when current params differ from the saved source state."""
        now = self.model.state_dict()
        for k, v in self.base_state.items():
            if not torch.equal(now[k], v):
                return True
        return False
