"""Threat Profiler — the "dendritic cell" bridge (spec §3.1).

Splits the (pre-processed) 2381-dim EMBER vector into the nine named feature
groups, projects each group through its own small MLP
(``Linear -> LayerNorm -> GELU``) into a shared ``d_model`` token space, and
emits one token per group plus a learnable ``[CLS]`` profile token.

Ablation switch ``shared_group_projection=True`` replaces the nine per-group
MLPs with one shared projection over the flattened concatenation (the
"no threat-profiler projections" ablation of spec §5).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from ..data.groups import GROUP_NAMES


def _group_mlp(in_dim: int, d_model: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, d_model),
        nn.LayerNorm(d_model),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class ThreatProfiler(nn.Module):
    """Group-wise projection of raw feature blocks into d_model tokens."""

    def __init__(
        self,
        group_dims: Mapping[str, int],
        d_model: int = 64,
        dropout: float = 0.1,
        shared_group_projection: bool = False,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        if not group_dims:
            raise ValueError("group_dims must not be empty")
        self.group_names: List[str] = list(group_dims)
        if tuple(self.group_names) != GROUP_NAMES:
            raise ValueError(
                f"group_names must match the EMBER group order {GROUP_NAMES}, "
                f"got {tuple(self.group_names)}"
            )
        self.group_dims = dict(group_dims)
        self.d_model = d_model
        self.shared_group_projection = shared_group_projection
        self.use_cls_token = use_cls_token

        if shared_group_projection:
            n_groups = len(group_dims)
            self.shared_projection = nn.Sequential(
                nn.Linear(sum(group_dims.values()), n_groups * d_model),
                nn.LayerNorm(n_groups * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.projections = None
        else:
            self.projections = nn.ModuleDict(
                {name: _group_mlp(dim, d_model, dropout) for name, dim in group_dims.items()}
            )
            self.shared_projection = None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls_token else None
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    @property
    def n_tokens(self) -> int:
        return len(self.group_names) + (1 if self.use_cls_token else 0)

    def forward(
        self, groups: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project group blocks to the token sequence.

        Parameters
        ----------
        groups:
            ``name -> (batch, dim_g)`` tensors in EMBER group order.

        Returns
        -------
        tokens: ``(batch, n_tokens, d_model)`` — group tokens in group order,
            ``[CLS]`` prepended when enabled.
        group_names: convenience list of the token identities (index 0 is CLS).
        """
        for name in self.group_names:
            if name not in groups:
                raise KeyError(f"missing feature group '{name}'")
            expected = self.group_dims[name]
            if groups[name].shape[-1] != expected:
                raise ValueError(
                    f"group '{name}' has width {groups[name].shape[-1]}, expected {expected}"
                )

        if self.shared_group_projection is not None and self.shared_projection is not None:
            flat = torch.cat([groups[n] for n in self.group_names], dim=-1)
            tokens = self.shared_projection(flat)
            tokens = tokens.view(tokens.shape[0], len(self.group_names), self.d_model)
        else:
            tokens = torch.stack(
                [self.projections[n](groups[n]) for n in self.group_names], dim=1
            )

        if self.cls_token is not None:
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        return tokens, list(self.group_names)

    @torch.no_grad()
    def profile(self, groups: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Human-readable group-wise anomaly profile (spec §3.1).

        Anomaly score per group = mean absolute source-standardised deviation
        over the group's feature dims (the mask channels appended by the
        preprocessor are excluded).  Higher = more anomalous vs the source
        period.  Explainability aid only — never used for fitting.
        """
        scores: Dict[str, torch.Tensor] = {}
        for name in self.group_names:
            block = groups[name]
            width = self.group_dims[name]
            feat = block[..., : width // 2]  # standardised half (mask is the back half)
            if feat.shape[-1] == 0:
                feat = block
            scores[name] = feat.abs().mean(dim=-1)
        return scores

    @torch.no_grad()
    def profile_text(self, groups: Mapping[str, torch.Tensor], row: int = 0) -> List[str]:
        """One-line-per-group human-readable profile for a single sample."""
        scores = self.profile(groups)
        lines = []
        for name in self.group_names:
            lines.append(f"{name:<22} anomaly={float(scores[name][row]):.3f}")
        return lines
