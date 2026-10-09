"""Unit tests: FGSM/PGD feature-space attacks + EMBER constraints (spec §3.6)."""

import numpy as np
import pytest
import torch

from armada.attacks import FeatureSpaceProjector, fgsm_attack, pgd_attack
from armada.attacks.constraints import FeatureSpaceProjector as FP2
from armada.data.groups import group_dims
from armada.data.loader import SourceOnlyPreprocessor
from armada.models.core import ArmadaCore

PROC_DIMS = {name: 2 * dim for name, dim in group_dims().items()}
GROUP_NAMES = list(PROC_DIMS)


def _setup(seed=0, batch=8):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    source_raw = rng.lognormal(2.0, 1.0, size=(64, 2381)).astype(np.float32)
    # sprinkle sentinels so missingness exists
    source_raw[rng.random(source_raw.shape) < 0.1] = -1.0
    pre = SourceOnlyPreprocessor().fit(source_raw)
    projector = FeatureSpaceProjector(pre, source_raw)
    model = ArmadaCore(PROC_DIMS, d_model=16, n_heads=4, n_layers=1, dropout=0.0, cls_hidden=32)
    groups = {
        name: torch.as_tensor(pre.transform(source_raw[:batch])[name], dtype=torch.float32)
        for name in GROUP_NAMES
    }
    y = torch.randint(0, 2, (batch,))
    return model, groups, y, projector, pre


def _l_inf(groups_a, groups_b):
    return max(
        float((groups_a[n] - groups_b[n]).abs().max()) for n in groups_a
    )


def test_fgsm_stays_in_linf_ball_and_increases_loss():
    model, groups, y, projector, _ = _setup(seed=1)
    eps = 0.1
    adv = fgsm_attack(model, groups, y, eps, projector)
    assert _l_inf(adv, groups) <= eps + 1e-5

    model.eval()
    with torch.no_grad():
        loss_clean = torch.nn.functional.binary_cross_entropy_with_logits(
            model(groups)[0], y.float()
        )
        loss_adv = torch.nn.functional.binary_cross_entropy_with_logits(model(adv)[0], y.float())
    assert float(loss_adv) >= float(loss_clean) - 1e-6


def test_pgd_stays_in_linf_ball_and_constraints():
    model, groups, y, projector, _ = _setup(seed=2)
    eps, steps = 0.2, 5
    adv = pgd_attack(model, groups, y, eps, step_size=0.05, steps=steps, projector=projector)
    assert _l_inf(adv, groups) <= eps + 1e-5
    for name in GROUP_NAMES:
        assert torch.isfinite(adv[name]).all()
        width = PROC_DIMS[name] // 2
        # missing-mask channels frozen
        assert torch.equal(adv[name][:, width:], groups[name][:, width:])


def test_projection_keeps_count_features_valid():
    model, groups, y, projector, _ = _setup(seed=3)
    eps = 5.0  # huge radius: projection must dominate
    adv = fgsm_attack(model, groups, y, eps, projector, clip_min=0.0)
    for name in GROUP_NAMES:
        width = PROC_DIMS[name] // 2
        clean_mask = groups[name][:, width:]
        free = clean_mask == 0
        # un-standardize free cells and check raw >= 0 and <= source max (>= clean)
        st = projector.stats[name]
        feat = adv[name][:, :width]
        raw = projector._to_raw(feat, st)
        hi = torch.maximum(st.raw_max.unsqueeze(0), projector._to_raw(groups[name][:, :width], st))
        free_full = free & ~st.zero_var.unsqueeze(0)
        assert float(raw[free_full].min()) >= -1e-4
        assert float((raw[free_full] - hi[free_full]).max()) <= 1e-3


def test_freeze_missing_cells():
    model, groups, y, projector, _ = _setup(seed=4)
    eps = 1.0
    adv = fgsm_attack(model, groups, y, eps, projector)
    free = projector.free_mask(groups)
    for name in GROUP_NAMES:
        width = PROC_DIMS[name] // 2
        # masked (missing) cells' standardized values are untouched
        missing = ~free[name][:, :width]
        if missing.any():
            assert torch.equal(adv[name][:, :width][missing], groups[name][:, :width][missing])


def test_zero_eps_is_identity():
    model, groups, y, projector, _ = _setup(seed=5)
    adv = fgsm_attack(model, groups, y, 0.0, projector)
    for name in GROUP_NAMES:
        assert torch.equal(adv[name], groups[name])
    adv2 = pgd_attack(model, groups, y, 0.0)
    for name in GROUP_NAMES:
        assert torch.equal(adv2[name], groups[name])


def test_random_start_and_seeded_pgd_deterministic():
    model, groups, y, projector, _ = _setup(seed=6)
    a = pgd_attack(model, groups, y, 0.2, step_size=0.05, steps=3, projector=projector, seed=123)
    b = pgd_attack(model, groups, y, 0.2, step_size=0.05, steps=3, projector=projector, seed=123)
    for name in GROUP_NAMES:
        assert torch.allclose(a[name], b[name])


def test_pgd_one_step_falls_back_to_fgsm():
    model, groups, y, projector, _ = _setup(seed=7)
    a = pgd_attack(model, groups, y, 0.15, steps=1, projector=projector)
    b = fgsm_attack(model, groups, y, 0.15, projector)
    for name in GROUP_NAMES:
        assert torch.allclose(a[name], b[name])


def test_projector_upper_bound_includes_clean_target_rows():
    # target rows may exceed source max; projection must still leave clean fixed
    model, groups, y, projector, pre = _setup(seed=8)
    st = projector.stats["GeneralFileInfo"]
    hi = torch.maximum(st.raw_max.unsqueeze(0), projector._to_raw(groups["GeneralFileInfo"][:, :10], st))
    assert torch.all(hi >= projector._to_raw(groups["GeneralFileInfo"][:, :10], st) - 1e-4)
