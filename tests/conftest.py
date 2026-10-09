"""Shared fixtures.

Synthetic tensors live HERE (tests/) only.  Every dummy array is tagged with
``SYNTHETIC`` provenance so the results writers refuse it — the paper pipeline
only ever runs on real EMBER artifacts (see README §Data).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from armada.data.groups import TOTAL_DIM
from armada.data.loader import SYNTHETIC, Provenance

# Fixed month layout used by fixtures: source months then target months.
FIXTURE_MONTHS = ["2017-01", "2017-02", "2017-03", "2017-04", "2017-05", "2017-06"]
TEST_MONTHS = ["2017-11", "2017-12"]


def make_dummy_features(n_rows: int, rng: np.random.Generator, n_cols: int = TOTAL_DIM) -> np.ndarray:
    """Dummy feature block for tests.  Always tagged as synthetic."""
    X = rng.normal(loc=0.0, scale=1.0, size=(n_rows, n_cols)).astype(np.float32)
    X[X < -0.5] = -1.0  # sprinkle sentinels
    return X


def make_dummy_dataset(n_rows: int = 600, seed: int = 0, n_cols: int = TOTAL_DIM):
    """Aligned dummy (X, y, appeared) with labels {0,1,-1} over FIXTURE_MONTHS."""
    rng = np.random.default_rng(seed)
    X = make_dummy_features(n_rows, rng, n_cols)
    y = rng.choice(np.array([-1.0, 0.0, 1.0]), size=n_rows, p=[0.2, 0.4, 0.4]).astype(np.float32)
    months = np.array([FIXTURE_MONTHS[i * len(FIXTURE_MONTHS) // n_rows] for i in range(n_rows)])
    return X, y, months, SYNTHETIC


def write_fake_ember_artifacts(
    data_dir: Path,
    n_train: int = 600,
    n_test: int = 120,
    seed: int = 0,
) -> Path:
    """Build a miniature EMBER-2018-layout dataset on disk (tests only).

    Produces X_train.dat, y_train.dat, X_test.dat, y_test.dat, metadata.csv with
    the official row alignment (metadata subset rows == .dat rows).  Content is
    SYNTHETIC and must never reach results/.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    X_tr = make_dummy_features(n_train, rng)
    y_tr = rng.choice(np.array([-1.0, 0.0, 1.0]), size=n_train, p=[0.25, 0.375, 0.375]).astype(np.float32)
    months_tr = np.array([FIXTURE_MONTHS[min(i * len(FIXTURE_MONTHS) // n_train, len(FIXTURE_MONTHS) - 1)] for i in range(n_train)])
    # Chronological ordering (as in the real dump): sort by month.
    order = np.argsort(months_tr, kind="stable")
    X_tr, y_tr, months_tr = X_tr[order], y_tr[order], months_tr[order]

    X_te = make_dummy_features(n_test, rng)
    y_te = rng.choice(np.array([0.0, 1.0]), size=n_test, p=[0.5, 0.5]).astype(np.float32)
    months_te = np.array([TEST_MONTHS[i % len(TEST_MONTHS)] for i in range(n_test)])

    X_tr.tofile(data_dir / "X_train.dat")
    y_tr.tofile(data_dir / "y_train.dat")
    X_te.tofile(data_dir / "X_test.dat")
    y_te.tofile(data_dir / "y_test.dat")

    import pandas as pd

    rows = []
    for i in range(n_train):
        rows.append({"sha256": f"train{i:064d}", "appeared": months_tr[i], "label": y_tr[i], "subset": "train"})
    for i in range(n_test):
        rows.append({"sha256": f"test{i:064d}", "appeared": months_te[i], "label": y_te[i], "subset": "test"})
    pd.DataFrame(rows).to_csv(data_dir / "metadata.csv")
    return data_dir


@pytest.fixture
def dummy_arrays():
    X, y, months, prov = make_dummy_dataset(n_rows=400, seed=7)
    return X, y, months, prov


@pytest.fixture
def real_provenance() -> Provenance:
    """Ember-kind provenance as produced by the real loader.

    Used to exercise the writers' allow-path; the guard tests use
    ``SYNTHETIC`` for the refuse path.
    """
    return Provenance(kind="ember", data_dir="<fixture>", details="unit-test")


@pytest.fixture
def ember_fixture_dir(tmp_path: Path) -> Path:
    return write_fake_ember_artifacts(tmp_path / "ember2018", n_train=600, n_test=120, seed=11)


@pytest.fixture
def fixture_cfg(ember_fixture_dir: Path, tmp_path: Path) -> dict:
    return {
        "seed_list": [0, 1],
        "data": {
            "data_dir": str(ember_fixture_dir),
            "cache_dir": str(tmp_path / "cache"),
            "checkpoints_dir": str(tmp_path / "checkpoints"),
            "results_dir": str(tmp_path / "results"),
            "figures_dir": str(tmp_path / "figures"),
            "tables_dir": str(tmp_path / "tables"),
            "logs_dir": str(tmp_path / "logs"),
            "include_test_subset": True,
            "n_train": 200,
            "n_val": 60,
            "n_per_window": 80,
            "n_unlabeled_per_window": 80,
            "sentinel_value": -1.0,
        },
        "split": {
            "mode": "auto",
            "source_fraction": 0.5,
            "n_target_windows": 2,
            "val_fraction": 0.15,
        },
        "model": {
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 1,
            "dropout": 0.0,
            "cls_hidden": 32,
            "shared_group_projection": False,
            "use_cls_token": True,
        },
        "train": {
            "stage_a_epochs": 3,
            "batch_size": 64,
            "lr": 1.0e-3,
            "weight_decay": 1.0e-5,
            "grad_clip": 5.0,
            "scheduler": "cosine",
            "early_stopping_patience": 5,
            "class_weights": "balanced",
            "amp": False,
        },
        "baselines": {
            "models": ["LR", "RF"],
            "rf_n_estimators": 10,
            "rf_max_depth": 6,
            "lr_max_iter": 200,
        },
        "eval": {"threshold": 0.5, "fixed_fprs": [0.001, 0.01], "ece_bins": 10},
    }
