"""Smoke tests: classical baselines + metric suite on dummy data.

Everything here writes to pytest tmp dirs only — never to results/.
"""

import numpy as np
import pytest

from armada.data.groups import TOTAL_DIM
from armada.eval.baselines import make_model, run_baseline_seed, summarise
from armada.eval.metrics import (
    aggregate_mean_std,
    compute_metrics,
    expected_calibration_error,
    tpr_at_fpr,
)

from conftest import make_dummy_features


def _toy_binary(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = make_dummy_features(n, rng)
    y = (X[:, 0] > 0).astype(np.int64) + rng.integers(0, 2, n) * 0  # separable-ish
    y = np.clip(y + rng.integers(0, 2, n) // 2, 0, 1)  # add label noise
    return X, y


def test_make_model_all_classical():
    for name in ("LR", "RF", "GBDT"):
        m = make_model(name, seed=0, cfg={})
        assert hasattr(m, "fit")
    with pytest.raises(ValueError):
        make_model("Nope", 0, {})


@pytest.mark.parametrize("name", ["LR", "RF"])
def test_baseline_learns_signal_and_metrics_are_finite(name):
    X, y = _toy_binary(n=400, seed=1)
    model = make_model(name, 0, {"baselines": {"rf_n_estimators": 10, "rf_max_depth": 6}})
    model.fit(X[:300], y[:300])
    prob = model.predict_proba(X[300:])[:, 1]
    m = compute_metrics(y[300:], prob)
    for key in ("accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "ece"):
        assert np.isfinite(m[key]), key
    assert 0.0 <= m["auc_roc"] <= 1.0
    assert m["n"] == 100


def test_tpr_at_fpr_operating_points():
    # Positives all outrank negatives: at FPR<=0 we still capture everything.
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.55, 0.7, 0.9])
    r_strict = tpr_at_fpr(y, scores, 0.0)
    assert r_strict["tpr"] == 1.0
    r = tpr_at_fpr(y, scores, 0.25)  # one FP allowed
    assert r["tpr"] == 1.0

    # Interleaved ranking: the FPR cap genuinely binds.
    y2 = np.array([0, 1, 0, 1])
    scores2 = np.array([0.95, 0.9, 0.55, 0.5])
    assert tpr_at_fpr(y2, scores2, 0.4)["tpr"] == 0.0  # even 1 FP is over budget
    assert tpr_at_fpr(y2, scores2, 0.5)["tpr"] == 0.5  # top-ranked pair only


def test_ece_perfectly_calibrated_vs_confident_wrong():
    y = np.array([0, 0, 1, 1])
    good = np.array([0.49, 0.51, 0.51, 0.49])
    assert expected_calibration_error(y, good, n_bins=4) <= 0.51
    bad = np.array([0.99, 0.99, 0.01, 0.01])
    assert expected_calibration_error(y, bad, n_bins=4) > 0.4


def test_run_baseline_seed_and_summarise(tmp_path, real_provenance):
    from armada.eval.baselines import ResultsWriter

    writer = ResultsWriter(tmp_path / "results", real_provenance)
    rows = []
    for seed in (0, 1):
        X, y = _toy_binary(n=350, seed=seed)
        windows = [("T1", X[250:300], y[250:300]), ("T2", X[300:], y[300:])]
        rows += run_baseline_seed("LR", seed, X[:250], y[:250], windows, {}, writer)
    assert len(rows) == 4  # 2 seeds x 2 windows
    by_window, overall = summarise(rows)
    assert len(by_window) == 2
    assert {r["method"] for r in overall} == {"LR"}
    assert overall[0]["n_seeds"] == 2
    assert np.isfinite(overall[0]["auc_roc_mean"])
    agg = aggregate_mean_std(rows, group_keys=("method", "window"))
    assert len(agg) == 2


def test_run_baseline_skips_empty_window(tmp_path, real_provenance):
    from armada.eval.baselines import ResultsWriter

    writer = ResultsWriter(tmp_path / "results", real_provenance)
    X, y = _toy_binary(n=200, seed=2)
    rows = run_baseline_seed("RF", 0, X[:150], y[:150], [("T1", X[150:], y[150:]), ("T2", np.zeros((0, TOTAL_DIM)), np.zeros(0))], {"baselines": {"rf_n_estimators": 5}}, writer)
    assert [r["window"] for r in rows] == ["T1"]


def test_metric_suite_keys_present():
    X, y = _toy_binary(n=100, seed=3)
    m = compute_metrics(y, X[:, 0] * 0 + 0.5 * y + 0.1)
    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "auc_pr",
        "ece",
        "tpr_at_fpr_0p001",
        "tpr_at_fpr_0p01",
    ):
        assert key in m
