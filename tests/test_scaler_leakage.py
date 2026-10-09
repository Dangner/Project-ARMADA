"""Unit tests: no target-time information may leak into fitted statistics
(spec §2/§0.6).  Scalers and thresholds are fit on the source period only."""

import numpy as np
import pytest

from armada.data.groups import TOTAL_DIM
from armada.data.loader import GroupScaler, SourceOnlyPreprocessor, stratified_cap_indices
from armada.data.drift_split import build_drift_split


def test_group_scaler_stats_come_from_fit_rows_only():
    rng = np.random.default_rng(0)
    X_source = rng.normal(loc=2.0, scale=3.0, size=(500, 16)).astype(np.float32)
    X_target = rng.normal(loc=100.0, scale=50.0, size=(200, 16)).astype(np.float32)

    sc = GroupScaler(log_transform=False).fit(X_source)

    # Stats must equal the source statistics...
    np.testing.assert_allclose(sc.mean_, X_source.mean(axis=0), rtol=1e-5, atol=1e-5)
    # ...and must NOT have been recomputed on anything else.
    assert not np.allclose(sc.mean_, X_target.mean(axis=0))

    Z = sc.transform(X_target)
    scaled = Z[:, :16]
    # Target values are centred with SOURCE stats: mean of Z ≈ (tgt_mean - src_mean)/src_std
    expected = (X_target.mean(axis=0) - sc.mean_) / sc.std_
    np.testing.assert_allclose(scaled.mean(axis=0), expected, rtol=1e-4, atol=1e-4)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        GroupScaler(log_transform=False).transform(np.zeros((2, 4)))


def test_zero_variance_columns_map_to_zero():
    X = np.ones((50, 4), dtype=np.float32) * 3.0  # all columns constant
    X[:, 2] = np.linspace(0, 1, 50)  # one informative column
    sc = GroupScaler(log_transform=False).fit(X)
    Z = sc.transform(X)[:, :4]
    assert np.all(Z[:, [0, 1, 3]] == 0.0)
    assert np.std(Z[:, 2]) > 0.1


def test_sentinel_becomes_zero_and_mask_channel():
    X = np.full((10, 5), 2.0, dtype=np.float32)
    X[0, 0] = -1.0
    sc = GroupScaler(log_transform=False).fit(X)
    Z = sc.transform(X)
    assert Z.shape == (10, 10)  # 5 scaled + 5 mask
    assert Z[0, 5] == 1.0  # missing-mask channel marks the sentinel
    assert Z[1, 5] == 0.0
    assert np.isfinite(Z).all()


def test_source_only_preprocessor_fit_records_rows_and_never_sees_target():
    rng = np.random.default_rng(1)
    X_src = rng.normal(size=(100, TOTAL_DIM)).astype(np.float32)
    X_tgt = rng.normal(loc=50.0, size=(40, TOTAL_DIM)).astype(np.float32)

    pre = SourceOnlyPreprocessor().fit(X_src)
    assert pre.n_fit_rows_ == 100

    src_mean_before = {k: v.mean_.copy() for k, v in pre.scalers.items()}
    _ = pre.transform(X_tgt)  # transforming target must not update anything
    for k, v in pre.scalers.items():
        np.testing.assert_array_equal(v.mean_, src_mean_before[k])

    proc = pre.transform(X_tgt)
    assert set(proc) == set(pre.processed_dims)
    for name, block in proc.items():
        assert block.shape == (40, pre.processed_dims[name])
    # full-width missing mask per group: processed width is 2 x raw width
    assert pre.processed_total_dim == 2 * TOTAL_DIM


def test_drift_split_source_val_comes_from_source_months_only():
    rng = np.random.default_rng(2)
    n = 900
    months = np.array([f"2017-{m:02d}" for m in (np.arange(n) * 6 // n + 1)])
    y = rng.choice([-1.0, 0.0, 1.0], size=n).astype(np.float32)

    split = build_drift_split(
        months,
        y,
        {"mode": "auto", "source_fraction": 0.5, "n_target_windows": 3, "val_fraction": 0.2},
        rng=rng,
    )
    src_months = set(split.source_months)
    assert src_months  # non-empty
    # every source train/val row lies in a source month
    for idx in (split.source_train_idx, split.source_val_idx):
        assert set(months[idx]) <= src_months
    # target windows never contain source months and are ordered in time
    all_target_months = []
    for w in split.windows:
        assert set(w.months).isdisjoint(src_months)
        all_target_months.extend(w.months)
    assert all_target_months == sorted(all_target_months)


def test_stratified_cap_indices_deterministic_and_stratified():
    y = np.array([0] * 300 + [1] * 100 + [-1] * 50)
    rng_a = np.random.default_rng(5)
    rng_b = np.random.default_rng(5)
    ia = stratified_cap_indices(y, 90, rng_a)
    ib = stratified_cap_indices(y, 90, rng_b)
    np.testing.assert_array_equal(ia, ib)
    assert len(ia) == 90
    counts = {v: int((y[ia] == v).sum()) for v in (0, 1, -1)}
    assert counts[1] > 15  # minority class preserved (~20 of 90)
    assert counts[-1] >= 5
