"""Gradient Reversal Layer and the DANN lambda schedule (spec §3.4).

Forward: identity.  Backward: multiplies the gradient by ``-lambda``.
Schedule: ``lambda(p) = 2 / (1 + exp(-gamma * p)) - 1`` with progress
``p in [0, 1]`` over training (gamma = 10 by default).
"""

from __future__ import annotations

import math

import torch
from torch.autograd import Function


class GradReverse(Function):
    """Identity forward, gradient-flip * lambda backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


def grl_lambda(progress: float, gamma: float = 10.0) -> float:
    """Scheduled reversal strength for training progress ``progress`` in [0,1]."""
    p = min(1.0, max(0.0, float(progress)))
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


class GradientReversal(torch.nn.Module):
    """Module wrapper holding the current lambda as a buffer-free attribute."""

    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = float(lambd)

    def set_lambda(self, lambd: float) -> None:
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambd)
