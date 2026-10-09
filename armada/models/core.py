"""ARMADA core composition: Threat Profiler -> grouped encoder -> classifier.

Assembles the Phase-2 modules into one ``nn.Module``.  Later phases attach
the dual discriminators, TTT decoder, memory and decision engine around it.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .encoder import GroupedSelfAttentionEncoder
from .heads import ClassifierHead
from .profiler import ThreatProfiler


class ArmadaCore(nn.Module):
    """profiler + grouped self-attention encoder + classifier head."""

    def __init__(
        self,
        group_dims: Mapping[str, int],
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        cls_hidden: int = 128,
        shared_group_projection: bool = False,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.profiler = ThreatProfiler(
            group_dims,
            d_model=d_model,
            dropout=dropout,
            shared_group_projection=shared_group_projection,
            use_cls_token=use_cls_token,
        )
        self.encoder = GroupedSelfAttentionEncoder(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout
        )
        self.classifier = ClassifierHead(in_dim=d_model, hidden=cls_hidden, dropout=dropout)
        self.d_model = d_model

    def embed(
        self,
        groups: Mapping[str, torch.Tensor],
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        tokens, _ = self.profiler(groups)
        return self.encoder(tokens, need_weights=need_weights)

    def forward(
        self, groups: Mapping[str, torch.Tensor], need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Return (malware logits, attention weights-or-None)."""
        z, attn = self.embed(groups, need_weights=need_weights)
        return self.classifier(z), attn

    @torch.no_grad()
    def predict_proba(
        self, groups: Mapping[str, torch.Tensor], need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        self.eval()
        logits, attn = self(groups, need_weights=need_weights)
        return torch.sigmoid(logits), attn

    @classmethod
    def from_config(cls, cfg: Mapping, group_dims: Mapping[str, int]) -> "ArmadaCore":
        m = cfg.get("model", cfg)
        return cls(
            group_dims=group_dims,
            d_model=int(m.get("d_model", 64)),
            n_heads=int(m.get("n_heads", 4)),
            n_layers=int(m.get("n_layers", 2)),
            dropout=float(m.get("dropout", 0.1)),
            cls_hidden=int(m.get("cls_hidden", 128)),
            shared_group_projection=bool(m.get("shared_group_projection", False)),
            use_cls_token=bool(m.get("use_cls_token", True)),
        )


class ArmadaDomainModel(ArmadaCore):
    """Core + dual discriminators (D1/D2) + masked-reconstruction decoder.

    Used from Phase 3 on: Stage B trains this jointly (classification + domain
    + reconstruction), TTT adapts the encoder on masked reconstruction.
    """

    def __init__(
        self,
        group_dims: Mapping[str, int],
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        cls_hidden: int = 128,
        shared_group_projection: bool = False,
        use_cls_token: bool = True,
        use_d2: bool = True,
        entropy_conditioning: bool = False,
        cond_map_dim: Optional[int] = None,
    ) -> None:
        super().__init__(
            group_dims=group_dims,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            cls_hidden=cls_hidden,
            shared_group_projection=shared_group_projection,
            use_cls_token=use_cls_token,
        )
        from .discriminators import ConditionalDiscriminator, MarginalDiscriminator
        from .ttt import MaskedGroupDecoder

        self.group_names = list(group_dims)
        self.d1 = MarginalDiscriminator(in_dim=d_model)
        self.use_d2 = use_d2
        self.d2 = (
            ConditionalDiscriminator(
                feat_dim=d_model,
                n_classes=2,
                random_map_dim=cond_map_dim,
                entropy_conditioning=entropy_conditioning,
            )
            if use_d2
            else None
        )
        self.decoder = MaskedGroupDecoder(group_dims, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    # -- encoding ---------------------------------------------------------

    def encode_with_mask(
        self,
        groups: Mapping[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode with ``mask`` (B, n_groups) group positions replaced by MASK.

        Returns ``(z, z_rows)``: CLS embedding and per-group encoder rows
        (positions 1..9 of the encoded sequence).
        """
        tokens, _ = self.profiler(groups)
        group_tokens = torch.where(
            mask.unsqueeze(-1), self.mask_token.to(tokens.dtype), tokens[:, 1:]
        )
        seq = torch.cat([tokens[:, :1], group_tokens], dim=1)
        encoded, attn = self.encoder(seq)
        return encoded[:, 0], encoded[:, 1:]

    def reconstruction_loss(
        self,
        groups: Mapping[str, torch.Tensor],
        mask_ratio: float = 0.3,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Masked group-feature reconstruction loss (mean MSE on masked cells)."""
        from .ttt import masked_reconstruction_loss, sample_group_mask

        tokens, names = self.profiler(groups)
        B = tokens.shape[0]
        device = tokens.device
        if mask is None:
            mask = sample_group_mask(B, len(self.group_names), mask_ratio, device)
        group_tokens = torch.where(
            mask.unsqueeze(-1), self.mask_token.to(tokens.dtype), tokens[:, 1:]
        )
        seq = torch.cat([tokens[:, :1], group_tokens], dim=1)
        encoded, _ = self.encoder.forward_sequence(seq)
        # per-group decoder uses the encoder output AT the group's own position
        z_rows = encoded[:, 1:]  # (B, n_groups, d)
        recon = self.decoder(z_rows, self.group_names)
        return masked_reconstruction_loss(recon, groups, mask, self.group_names)

    # -- domain adaptation ------------------------------------------------

    def domain_loss_d1(self, z: torch.Tensor, is_target: bool, lambd: float) -> torch.Tensor:
        import torch.nn.functional as F

        label = torch.full(
            (z.shape[0],), 1.0 if is_target else 0.0, device=z.device, dtype=z.dtype
        )
        return F.binary_cross_entropy_with_logits(self.d1(z, lambd), label)

    def domain_loss_d2(
        self,
        z: torch.Tensor,
        class_probs: torch.Tensor,
        is_target: bool,
        lambd: float,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        if self.d2 is None:
            return z.new_zeros(())
        label = torch.full(
            (z.shape[0],), 1.0 if is_target else 0.0, device=z.device, dtype=z.dtype
        )
        logits, weights = self.d2(z, class_probs, lambd)
        loss = F.binary_cross_entropy_with_logits(logits, label, reduction="none")
        if weights is not None:
            loss = loss * weights
        return loss.mean()

    @classmethod
    def from_config(cls, cfg: Mapping, group_dims: Mapping[str, int]) -> "ArmadaDomainModel":
        m = cfg.get("model", cfg)
        t = cfg.get("train", {})
        return cls(
            group_dims=group_dims,
            d_model=int(m.get("d_model", 64)),
            n_heads=int(m.get("n_heads", 4)),
            n_layers=int(m.get("n_layers", 2)),
            dropout=float(m.get("dropout", 0.1)),
            cls_hidden=int(m.get("cls_hidden", 128)),
            shared_group_projection=bool(m.get("shared_group_projection", False)),
            use_cls_token=bool(m.get("use_cls_token", True)),
            use_d2=bool(t.get("use_d2", True)),
            entropy_conditioning=bool(t.get("entropy_conditioning", False)),
            cond_map_dim=t.get("cond_map_dim", None),
        )
