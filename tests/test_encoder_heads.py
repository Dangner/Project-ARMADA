"""Unit tests: grouped self-attention encoder (spec §3.2) and heads (§3.3)."""

import numpy as np
import pytest
import torch

from armada.models.core import ArmadaCore
from armada.models.encoder import GroupedSelfAttentionEncoder, PreLNEncoderLayer
from armada.models.heads import (
    ClassifierHead,
    balanced_pos_weight,
    build_pos_weight,
    classification_loss,
)

from armada.data.groups import group_dims

PROC_DIMS = {name: 2 * dim for name, dim in group_dims().items()}


def _core(**kw):
    defaults = dict(d_model=16, n_heads=4, n_layers=2, dropout=0.0, cls_hidden=32)
    defaults.update(kw)
    return ArmadaCore(PROC_DIMS, **defaults)


def _groups(batch=5, seed=0):
    torch.manual_seed(seed)
    return {name: torch.randn(batch, dim) for name, dim in PROC_DIMS.items()}


def test_encoder_returns_cls_embedding():
    enc = GroupedSelfAttentionEncoder(d_model=16, n_heads=4, n_layers=2, dropout=0.0)
    tokens = torch.randn(5, 10, 16)
    z, attn = enc(tokens)
    assert z.shape == (5, 16)
    assert attn is None


def test_encoder_attention_weights_shapes():
    enc = GroupedSelfAttentionEncoder(d_model=16, n_heads=4, n_layers=3, dropout=0.0)
    tokens = torch.randn(2, 10, 16)
    z, attn = enc(tokens, need_weights=True)
    assert len(attn) == 3  # one map per layer
    for w in attn:
        assert w.shape == (2, 4, 10, 10)  # batch, heads, query, key
        np.testing.assert_allclose(
            w.sum(dim=-1).detach().numpy(), np.ones((2, 4, 10)), rtol=1e-4
        )


def test_encoder_pre_layer_norm_structure():
    layer = PreLNEncoderLayer(d_model=16, n_heads=4, dropout=0.0)
    x = torch.randn(3, 10, 16)
    out, _ = layer(x)
    assert out.shape == x.shape
    assert isinstance(layer.norm_attn, torch.nn.LayerNorm)
    assert isinstance(layer.norm_ff, torch.nn.LayerNorm)


def test_encoder_validates_shapes():
    with pytest.raises(ValueError, match="n_layers"):
        GroupedSelfAttentionEncoder(d_model=16, n_heads=4, n_layers=8)
    with pytest.raises(ValueError, match="divisible"):
        GroupedSelfAttentionEncoder(d_model=18, n_heads=4, n_layers=2)
    enc = GroupedSelfAttentionEncoder(d_model=16, n_heads=4, n_layers=1)
    with pytest.raises(ValueError, match="expected"):
        enc(torch.randn(10, 16))


def test_classifier_head_logit_shape():
    head = ClassifierHead(in_dim=16, hidden=32, dropout=0.0)
    z = torch.randn(7, 16)
    logits = head(z)
    assert logits.shape == (7,)


def test_core_forward_and_gradients():
    model = _core()
    logits, _ = model(_groups(batch=4))
    assert logits.shape == (4,)
    loss = classification_loss(logits, torch.tensor([0, 1, 1, 0], dtype=torch.float32))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_core_predict_proba_range_and_attention():
    model = _core()
    probs, attn = model.predict_proba(_groups(batch=3), need_weights=True)
    assert torch.all((probs >= 0) & (probs <= 1))
    assert len(attn) == 2


def test_balanced_pos_weight_from_source_labels():
    y = np.array([0, 0, 0, 1])
    assert balanced_pos_weight(y) == pytest.approx(3.0)
    assert build_pos_weight({"class_weights": "none"}, y) is None
    assert build_pos_weight({"class_weights": "balanced"}, y) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        balanced_pos_weight(np.array([0, 0, 0]))


def test_classification_loss_pos_weight_changes_value():
    logits = torch.tensor([2.0, -2.0])
    y = torch.tensor([1.0, 0.0])
    plain = classification_loss(logits, y, pos_weight=None)
    weighted = classification_loss(logits, y, pos_weight=5.0)
    assert float(weighted) != float(plain)
