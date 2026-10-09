"""Unit tests: Gradient Reversal Layer (spec §3.4, §8 GRL sign flip)."""

import math

import pytest
import torch

from armada.models.grl import GradReverse, GradientReversal, grad_reverse, grl_lambda


def test_gradient_sign_is_flipped_and_scaled():
    x = torch.randn(4, 8, requires_grad=True)
    lambd = 0.37
    y = grad_reverse(x, lambd)
    y.sum().backward()
    # d(sum(y))/dx = +1 forward, so with GRL the grad must be exactly -lambda.
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.full_like(x, -lambd), atol=1e-6)


def test_forward_is_identity():
    x = torch.randn(3, 5)
    y = grad_reverse(x, 0.9)
    torch.testing.assert_close(y, x)


def test_lambda_zero_still_negates():
    x = torch.randn(2, 2, requires_grad=True)
    grad_reverse(x, 0.0).sum().backward()
    assert torch.allclose(x.grad, torch.zeros_like(x))


def test_lambda_schedule_monotone_and_bounded():
    vals = [grl_lambda(p) for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert vals[0] == pytest.approx(0.0)
    assert vals[-1] == pytest.approx(2 / (1 + math.exp(-10)) - 1, rel=1e-6)
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    assert all(0.0 <= v <= 1.0 for v in vals)
    # out-of-range progress is clamped
    assert grl_lambda(-1.0) == 0.0
    assert grl_lambda(2.0) == vals[-1]
    # gamma knob
    assert grl_lambda(1.0, gamma=0.0) == pytest.approx(0.0)


def test_gradient_reversal_module_tracks_lambda():
    grl = GradientReversal(0.5)
    x = torch.randn(2, 3, requires_grad=True)
    grl(x).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.5), atol=1e-6)
    grl.set_lambda(1.0)
    x2 = torch.randn(2, 3, requires_grad=True)
    grl(x2).sum().backward()
    assert torch.allclose(x2.grad, torch.full_like(x2, -1.0), atol=1e-6)


def test_grad_reverse_rejects_wrong_lambda_type():
    x = torch.randn(2, 2, requires_grad=True)
    with pytest.raises((TypeError, RuntimeError, ValueError)):
        grad_reverse(x, "big")
