"""Unit tests: Threat Profiler (spec §3.1)."""

import numpy as np
import pytest
import torch

from armada.data.groups import GROUP_NAMES, group_dims
from armada.models.profiler import ThreatProfiler

from conftest import make_dummy_features

# Processed widths: standardised half + missing-mask half per group.
PROC_DIMS = {name: 2 * dim for name, dim in group_dims().items()}


def _proc_groups(batch=6, seed=0):
    torch.manual_seed(seed)
    return {name: torch.randn(batch, dim) for name, dim in PROC_DIMS.items()}


def test_profiler_emits_one_token_per_group_plus_cls():
    prof = ThreatProfiler(PROC_DIMS, d_model=32, dropout=0.0)
    tokens, names = prof(_proc_groups(batch=4))
    assert tokens.shape == (4, 10, 32)  # 9 groups + CLS
    assert tuple(names) == GROUP_NAMES
    assert prof.n_tokens == 10


def test_each_group_has_its_own_projection():
    prof = ThreatProfiler(PROC_DIMS, d_model=32, dropout=0.0)
    assert prof.projections is not None
    assert len(prof.projections) == 9
    params = [id(p) for m in prof.projections.values() for p in m.parameters()]
    assert len(params) == len(set(params))  # no parameter sharing across groups


def test_cls_token_is_learnable():
    prof = ThreatProfiler(PROC_DIMS, d_model=32, dropout=0.0)
    assert prof.cls_token.requires_grad
    tokens, _ = prof(_proc_groups(batch=2))
    loss = tokens[:, 0].sum()
    loss.backward()
    assert prof.cls_token.grad is not None


def test_shared_projection_ablation():
    prof = ThreatProfiler(PROC_DIMS, d_model=32, dropout=0.0, shared_group_projection=True)
    assert prof.projections is None
    assert prof.shared_projection is not None
    tokens, _ = prof(_proc_groups(batch=3))
    assert tokens.shape == (3, 10, 32)


def test_missing_group_and_wrong_width_raise():
    prof = ThreatProfiler(PROC_DIMS, d_model=32, dropout=0.0)
    groups = _proc_groups(batch=2)
    bad = dict(groups)
    del bad["ImportsInfo"]
    with pytest.raises(KeyError):
        prof(bad)
    bad = dict(groups)
    bad["GeneralFileInfo"] = torch.randn(2, 3)
    with pytest.raises(ValueError, match="expected"):
        prof(bad)


def test_group_order_must_match_ember_schema():
    wrong = {"ByteHistogram": 10, "GeneralFileInfo": 4}
    with pytest.raises(ValueError, match="EMBER group order"):
        ThreatProfiler(wrong, d_model=8)


def test_profile_anomaly_scores():
    prof = ThreatProfiler(PROC_DIMS, d_model=32, dropout=0.0)
    groups = _proc_groups(batch=2)
    scores = prof.profile(groups)
    assert set(scores) == set(GROUP_NAMES)
    for name, s in scores.items():
        assert s.shape == (2,)
        assert float(s.min()) >= 0.0
    lines = prof.profile_text(groups, row=0)
    assert len(lines) == 9
    assert lines[0].startswith("ByteHistogram")
