"""Grouped self-attention encoder — the "adaptive immune engine" (spec §3.2).

Pre-LayerNorm Transformer encoder over the 10 tokens (9 feature groups +
``[CLS]`` profile token).  The ``[CLS]`` output is the feature embedding ``z``.
Attention weights are returned for per-group importance analysis.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class PreLNEncoderLayer(nn.Module):
    """Transformer encoder layer with pre-LayerNorm and exposed attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, ffn_mult: int = 4) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.norm_attn(x)
        attn_out, weights = self.attn(
            h, h, h, need_weights=need_weights, average_attn_weights=False
        )
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.norm_ff(x))
        return x, (weights if need_weights else None)


class GroupedSelfAttentionEncoder(nn.Module):
    """Stack of pre-LN layers; ``z`` is the encoded ``[CLS]`` row."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not (1 <= n_layers <= 4):
            raise ValueError("n_layers must be in [1, 4] per the spec")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.layers = nn.ModuleList(
            [PreLNEncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.use_cls_token = True  # tokens always carry CLS at index 0

    def forward_sequence(
        self, tokens: torch.Tensor, need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Encode the token sequence and return the full encoded sequence.

        Returns
        -------
        encoded: ``(batch, n_tokens, d_model)`` post-final-norm sequence.
        attn: per-layer attention weights or ``None``.
        """
        if tokens.dim() != 3:
            raise ValueError(f"expected (batch, tokens, d_model), got {tuple(tokens.shape)}")
        x = tokens
        all_weights: List[torch.Tensor] = []
        for layer in self.layers:
            x, w = layer(x, need_weights=need_weights)
            if need_weights and w is not None:
                all_weights.append(w)
        x = self.final_norm(x)
        return x, (all_weights if need_weights else None)

    def forward(
        self, tokens: torch.Tensor, need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Encode the token sequence.

        Parameters
        ----------
        tokens: ``(batch, n_tokens, d_model)`` from :class:`ThreatProfiler`.
        need_weights: also return per-layer attention maps.

        Returns
        -------
        z: ``(batch, d_model)`` — the encoded ``[CLS]`` profile token.
        attn: list (over layers) of ``(batch, n_heads, n_tokens, n_tokens)``
            attention weights, or ``None`` when not requested.
        """
        encoded, attn = self.forward_sequence(tokens, need_weights=need_weights)
        z = encoded[:, 0]  # CLS row
        return z, attn
