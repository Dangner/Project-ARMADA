"""EMBER-valid feature-space projection for adversarial attacks (spec §3.6).

Adversaries act in the model's *processed* input space (z-scored group
features).  After every perturbation step the implied RAW features are
projected back onto the EMBER-feasible region:

- count-like features stay non-negative (``clip_min = 0``);
- per-column upper bound = source-period max (leakage rule: bounds are
  derived from SOURCE rows only);
- missing-mask channels are frozen: the adversary cannot flip feature
  presence, and columns missing in the clean sample cannot be modified.

**Limitation (README §Limitations):** these are feature-space perturbations.
They are NOT guaranteed to correspond to valid PE executables; robustness
numbers measure the model in feature space only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import torch

from ..data.groups import GROUPS, LOG_TRANSFORM_GROUPS, TOTAL_DIM
from ..data.loader import SourceOnlyPreprocessor


@dataclass
class GroupStats:
    log_transform: bool
    mean: torch.Tensor  # (dim,)
    std: torch.Tensor  # (dim,)
    zero_var: torch.Tensor  # (dim,) bool
    raw_max: torch.Tensor  # (dim,) source-period per-column max


class FeatureSpaceProjector:
    """Projects processed-space candidates back onto the EMBER-feasible set.

    Built from SOURCE rows + the fitted :class:`SourceOnlyPreprocessor` only.
    """

    def __init__(
        self,
        preprocessor: SourceOnlyPreprocessor,
        source_raw: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.group_names = [g.name for g in GROUPS]
        self.stats: Dict[str, GroupStats] = {}
        for g in GROUPS:
            sc = preprocessor.scalers[g.name]
            block = np.asarray(source_raw[:, g.slice], dtype=np.float32)
            raw_max = block.max(axis=0)
            self.stats[g.name] = GroupStats(
                log_transform=g.name in LOG_TRANSFORM_GROUPS,
                mean=torch.as_tensor(sc.mean_, dtype=torch.float32, device=self.device),
                std=torch.as_tensor(sc.std_, dtype=torch.float32, device=self.device),
                zero_var=torch.as_tensor(sc.zero_var_, dtype=torch.bool, device=self.device),
                raw_max=torch.as_tensor(raw_max, dtype=torch.float32, device=self.device),
            )
        self.processed_dims = preprocessor.processed_dims

    # -- transform helpers (mirror GroupScaler, torch, differentiable) -----

    def _to_raw(self, feat: torch.Tensor, st: GroupStats) -> torch.Tensor:
        t = feat * st.std + st.mean
        if st.log_transform:
            t = torch.expm1(t)
        return t

    def _to_feat(self, raw: torch.Tensor, st: GroupStats) -> torch.Tensor:
        t = raw
        if st.log_transform:
            t = torch.log1p(torch.clamp(raw, min=0.0))
        z = (t - st.mean) / st.std
        z = torch.where(st.zero_var, torch.zeros_like(z), z)
        return z

    def free_mask(self, groups_clean: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Full-width boolean mask of cells the adversary may modify.

        Feature channels of PRESENT features are free; the missing-mask half
        and missing-feature channels are always False (frozen).
        """
        out = {}
        for name in self.group_names:
            block = groups_clean[name]
            width = self.processed_dims[name] // 2
            mask_ch = block[:, width:]
            free_feat = mask_ch == 0.0  # present cells only
            frozen = torch.zeros_like(mask_ch, dtype=torch.bool)
            out[name] = torch.cat([free_feat, frozen], dim=1)
        return out

    @torch.no_grad()
    def project(
        self,
        groups: Mapping[str, torch.Tensor],
        groups_clean: Mapping[str, torch.Tensor],
        clip_min: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Clamp ``groups`` onto the feasible set around ``groups_clean``.

        Upper bound per cell is ``max(source_max_j, clean_ij)`` so the clean
        sample is always feasible (even when a target row exceeds the source
        max — bounds are constraints, never fitted parameters).
        """
        out: Dict[str, torch.Tensor] = {}
        for name in self.group_names:
            st = self.stats[name]
            width = self.processed_dims[name] // 2
            clean = groups_clean[name]
            feat, mask_ch = groups[name][:, :width], clean[:, width:]
            clean_feat = clean[:, :width]
            raw = self._to_raw(feat, st)
            hi = torch.maximum(st.raw_max.unsqueeze(0), self._to_raw(clean_feat, st))
            raw = torch.clamp(raw, min=clip_min)
            raw = torch.minimum(raw, hi)
            fixed = self._to_feat(raw, st)
            free = (mask_ch == 0.0) & ~st.zero_var.unsqueeze(0)
            feat_out = torch.where(free, fixed, clean_feat)
            out[name] = torch.cat([feat_out, mask_ch], dim=1)
        return out
