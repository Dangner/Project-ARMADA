"""Unit tests: EMBER loading, caps, caching and memory guards."""

import numpy as np
import pytest

from armada.data.groups import TOTAL_DIM
from armada.data.loader import (
    EmberPool,
    SyntheticDataError,
    cache_dir_for,
    check_memory,
    config_hash,
    estimate_bytes,
    load_ember_test,
    load_ember_train,
    load_metadata,
    prepare_seed_dataset,
    require_ember_artifacts,
    save_cached_arrays,
    load_cached_arrays,
)

from conftest import FIXTURE_MONTHS


def test_require_ember_artifacts_error_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="X_train.dat"):
        require_ember_artifacts(tmp_path / "missing")


def test_load_ember_train_alignment(ember_fixture_dir):
    split = load_ember_train(ember_fixture_dir)
    assert len(split) == 600
    assert split.X.shape == (600, TOTAL_DIM)
    assert set(np.unique(split.y)) <= {-1.0, 0.0, 1.0}
    assert split.appeared.shape == (600,)
    assert set(split.appeared) <= set(FIXTURE_MONTHS)
    assert split.provenance.is_real
    # chronological ordering like the official dump
    assert list(split.appeared[:20]).count(FIXTURE_MONTHS[0]) == 20


def test_metadata_aligns_with_test_split(ember_fixture_dir):
    meta = load_metadata(ember_fixture_dir)
    assert (meta["subset"] == "train").sum() == 600
    assert (meta["subset"] == "test").sum() == 120
    te = load_ember_test(ember_fixture_dir)
    assert len(te) == 120


def test_pool_gather_matches_row_order(ember_fixture_dir):
    pool = EmberPool([load_ember_train(ember_fixture_dir), load_ember_test(ember_fixture_dir)])
    assert len(pool) == 720
    idx = np.array([0, 5, 599, 600, 719])
    rows = pool.gather(idx)
    assert rows.shape == (5, TOTAL_DIM)
    np.testing.assert_array_equal(rows[0], np.asarray(pool.splits[0].X[0]))
    np.testing.assert_array_equal(rows[3], np.asarray(pool.splits[1].X[0]))
    np.testing.assert_array_equal(rows[4], np.asarray(pool.splits[1].X[119]))


def test_mismatched_metadata_raises(ember_fixture_dir):
    import pandas as pd

    meta = pd.read_csv(ember_fixture_dir / "metadata.csv", index_col=0)
    drop = meta[meta["subset"] == "train"].index[:3]
    meta = meta.drop(index=drop)
    meta.to_csv(ember_fixture_dir / "metadata.csv")
    with pytest.raises(ValueError, match="align 1:1"):
        load_ember_train(ember_fixture_dir)


def test_config_hash_stable_and_sensitive():
    a = {"data": {"n_train": 20000}, "split": {"mode": "auto"}, "seed": 0}
    b = {"data": {"n_train": 20000}, "split": {"mode": "auto"}, "seed": 0}
    c = {"data": {"n_train": 50000}, "split": {"mode": "auto"}, "seed": 0}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)
    d = {"data": {"n_train": 20000}, "split": {"mode": "auto"}, "seed": 1}
    assert config_hash(a) != config_hash(d)
    assert cache_dir_for(a, "cache").name == f"pre_{config_hash(a)}"


def test_cache_roundtrip(tmp_path):
    arrays = {"x": np.arange(6, dtype=np.float32).reshape(2, 3), "y": np.array([0, 1])}
    save_cached_arrays(tmp_path / "c", arrays, {"seed": 0, "note": "unit"})
    loaded = load_cached_arrays(tmp_path / "c")
    np.testing.assert_array_equal(loaded["x"], arrays["x"])
    np.testing.assert_array_equal(loaded["y"], arrays["y"])
    assert load_cached_arrays(tmp_path / "does_not_exist") is None


def test_memory_estimate_and_guard():
    assert estimate_bytes(10, 2381) == 10 * 2381 * 4
    check_memory(10, 2381)  # tiny: must pass
    with pytest.raises(MemoryError, match="Reduce n_train"):
        check_memory(10**12, 2381)


def test_prepare_seed_dataset_caps_and_cache(fixture_cfg):
    pool = EmberPool([load_ember_train(fixture_cfg["data"]["data_dir"])])
    ds0 = prepare_seed_dataset(pool, fixture_cfg, seed=0)
    assert len(ds0.source_train_y) == 200
    # val is the (capped) source-only holdout; the fixture pool only holds
    # ~36 validation rows after the drift split, so <= n_val is the contract.
    assert 0 < len(ds0.source_val_y) <= 60
    assert ds0.provenance.is_real
    names = [w["name"] for w in ds0.windows]
    assert names == ["T1", "T2"]
    for w in ds0.windows:
        assert len(w["y"]) <= 80
        assert set(np.unique(w["y"])) <= {0, 1}
        assert w["proc"][list(w["proc"])[0]].shape[0] == len(w["y"])
    # cache created and reload yields identical arrays
    ds0b = prepare_seed_dataset(pool, fixture_cfg, seed=0)
    np.testing.assert_array_equal(ds0.source_train_raw, ds0b.source_train_raw)
    np.testing.assert_array_equal(ds0.windows[0]["raw"], ds0b.windows[0]["raw"])
    # different seed -> different cache dir (subsample may differ)
    ds1 = prepare_seed_dataset(pool, fixture_cfg, seed=1)
    assert ds1.seed == 1
