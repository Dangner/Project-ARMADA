"""Unit tests: domain discriminators D1/D2 (spec §3.4)."""

import numpy as np
import pytest
import torch

from armada.models.discriminators import (
    ConditionalDiscriminator,
    MarginalDiscriminator,
    cdan_joint_features,
)


def test_marginal_discriminator_shapes_and_grad_flow():
    d1 = MarginalDiscriminator(in_dim=16, hidden=32)
    z = torch.randn(6, 16, requires_grad=True)
    logits = d1(z, lambd=1.0)
    assert logits.shape == (6,)
    logits.sum().backward()
    # encoder side sees FLIPPED gradients through the discriminator
    assert z.grad is not None
    assert float(z.grad.abs().sum()) > 0


def test_cdan_joint_features_outer_product():
    z = torch.randn(4, 16)
    p = torch.softmax(torch.randn(4, 2), dim=1)
    joint = cdan_joint_features(z, p)
    assert joint.shape == (4, 32)
    # entry (c, j) of the outer product must equal p_c * z_j
    np.testing.assert_allclose(
        joint[0].reshape(2, 16)[1].numpy(), (p[0, 1] * z[0]).numpy(), rtol=1e-5
    )


def test_cdan_randomised_multilinear_map():
    z = torch.randn(3, 16)
    p = torch.softmax(torch.randn(3, 2), dim=1)
    R = torch.randn(32, 10)
    joint = cdan_joint_features(z, p, random_map=R)
    assert joint.shape == (3, 10)


def test_cdan_rejects_mismatched_batches():
    with pytest.raises(ValueError, match="batch mismatch"):
        cdan_joint_features(torch.randn(3, 8), torch.softmax(torch.randn(4, 2), dim=1))


def test_conditional_discriminator_shapes_and_entropy_weights():
    d2 = ConditionalDiscriminator(feat_dim=16, n_classes=2, hidden=32, entropy_conditioning=True)
    z = torch.randn(5, 16)
    p = torch.softmax(torch.randn(5, 2), dim=1)
    logits, weights = d2(z, p, lambd=0.5)
    assert logits.shape == (5,)
    assert weights is not None and weights.shape == (5,)
    assert torch.all((weights >= 0) & (weights <= 1))


def test_conditional_discriminator_without_entropy_conditioning():
    d2 = ConditionalDiscriminator(feat_dim=16, n_classes=2, hidden=32)
    z = torch.randn(5, 16)
    p = torch.softmax(torch.randn(5, 2), dim=1)
    logits, weights = d2(z, p)
    assert weights is None


def test_conditional_discriminator_random_map_option():
    d2 = ConditionalDiscriminator(
        feat_dim=16, n_classes=2, hidden=32, random_map_dim=12
    )
    z = torch.randn(4, 16)
    p = torch.softmax(torch.randn(4, 2), dim=1)
    logits, _ = d2(z, p)
    assert logits.shape == (4,)
    assert d2.random_map.shape == (32, 12)


def test_entropy_weight_is_extreme_for_confident_predictions():
    from armada.models.discriminators import ConditionalDiscriminator as CD

    confident = torch.tensor([[0.999, 0.001], [0.5, 0.5]])
    w = CD.entropy_weight(confident)
    assert float(w[0]) > 0.9
    assert float(w[1]) < 0.1
