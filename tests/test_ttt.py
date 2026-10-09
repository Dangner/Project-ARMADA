"""Unit tests: TTT via masked feature reconstruction (spec §3.5, §8 TTT reset)."""

import copy

import numpy as np
import pytest
import torch

from armada.data.groups import group_dims
from armada.models.core import ArmadaDomainModel
from armada.models.ttt import (
    MaskedGroupDecoder,
    TTTAdapter,
    masked_reconstruction_loss,
    sample_group_mask,
)

PROC_DIMS = {name: 2 * dim for name, dim in group_dims().items()}
GROUP_NAMES = list(PROC_DIMS)


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    defaults = dict(
        d_model=16, n_heads=4, n_layers=1, dropout=0.0, cls_hidden=32, use_d2=True
    )
    defaults.update(kw)
    return ArmadaDomainModel(PROC_DIMS, **defaults)


def _groups(batch=8, seed=0):
    torch.manual_seed(seed)
    return {name: torch.randn(batch, dim) for name, dim in PROC_DIMS.items()}


def test_sample_group_mask_covers_every_sample():
    mask = sample_group_mask(16, 9, 0.2, torch.device("cpu"))
    assert mask.shape == (16, 9)
    assert mask.any(dim=1).all()  # at least one masked group per sample
    with pytest.raises(ValueError):
        sample_group_mask(4, 9, 0.0, torch.device("cpu"))


def test_reconstruction_loss_only_on_masked_cells():
    torch.manual_seed(0)
    model = _model()
    groups = _groups(batch=6)
    mask = sample_group_mask(6, 9, 0.5, torch.device("cpu"))
    loss = model.reconstruction_loss(groups, mask=mask)
    assert torch.isfinite(loss)
    assert float(loss) > 0

    # masking nothing -> loss must be exactly 0 (empty set)
    full = torch.zeros(6, 9, dtype=torch.bool)
    loss0 = model.reconstruction_loss(groups, mask=full)
    assert float(loss0) == 0.0


def test_decoder_shapes():
    dec = MaskedGroupDecoder(PROC_DIMS, d_model=16)
    z_rows = torch.randn(4, 9, 16)
    out = dec(z_rows, GROUP_NAMES)
    assert set(out) == set(GROUP_NAMES)
    for name, rec in out.items():
        assert rec.shape == (4, PROC_DIMS[name])


def test_masked_reconstruction_loss_helper():
    recon = {n: torch.ones(3, PROC_DIMS[n]) for n in GROUP_NAMES}
    targets = {n: torch.zeros(3, PROC_DIMS[n]) for n in GROUP_NAMES}
    mask = torch.zeros(3, 9, dtype=torch.bool)
    mask[0, 0] = True
    mask[2, 3] = True
    loss = masked_reconstruction_loss(recon, targets, mask, GROUP_NAMES)
    assert float(loss) == pytest.approx(1.0)  # MSE(1, 0) = 1 over masked cells


def test_ttt_reset_restores_exact_source_weights():
    model = _model()
    groups = _groups(batch=12, seed=3)
    adapter = TTTAdapter(model, lr=1e-2, steps=3, mask_ratio=0.5, online=False)

    assert not adapter.state_changed()
    losses = adapter.adapt({k: v for k, v in groups.items()}, batch_size=6)
    assert len(losses) == 3
    assert all(np.isfinite(losses))
    assert adapter.state_changed()  # TTT actually moved the weights

    adapter.reset()
    assert not adapter.state_changed()  # byte-identical to saved source state


def test_ttt_online_mode_persists_across_windows():
    model = _model()
    adapter = TTTAdapter(model, lr=1e-2, steps=2, mask_ratio=0.5, online=True)
    adapter.adapt(_groups(seed=1), batch_size=6)
    changed_after_first = adapter.state_changed()
    adapter.adapt(_groups(seed=2), batch_size=6)
    assert changed_after_first  # online: no reset between windows
    assert adapter.state_changed()


def test_ttt_does_not_update_classifier():
    model = _model()
    adapter = TTTAdapter(model, lr=1e-2, steps=2, mask_ratio=0.5)
    before = copy.deepcopy(model.classifier.state_dict())
    adapter.adapt(_groups(seed=4), batch_size=6)
    after = model.classifier.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), "classifier must stay frozen during TTT"


def test_ttt_unlabeled_source_only_and_empty_batch():
    model = _model()
    adapter = TTTAdapter(model, lr=1e-2, steps=2, mask_ratio=0.5)
    assert adapter.adapt({k: v[:0] for k, v in _groups().items()}) == []
    assert adapter.adapt(None) == []
    assert not adapter.state_changed()


def test_ttt_recon_loss_decreases_with_more_steps():
    torch.manual_seed(7)
    model = _model(seed=7)
    groups = _groups(batch=16, seed=7)
    adapter = TTTAdapter(model, lr=5e-2, steps=8, mask_ratio=0.5)
    losses = adapter.adapt(groups, batch_size=16)
    assert losses[-1] < losses[0]
