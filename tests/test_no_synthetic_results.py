"""Unit tests: synthetic data can never reach results/ (spec §0, §8)."""

import numpy as np
import pytest

from armada.data.loader import SYNTHETIC, Provenance, SyntheticDataError
from armada.eval.baselines import ResultsWriter, metrics_from_saved_predictions


def test_synthetic_provenance_refused_by_results_writer(tmp_path):
    with pytest.raises(SyntheticDataError, match="refusing to write results"):
        ResultsWriter(tmp_path / "results", SYNTHETIC)


def test_dummy_factory_output_is_tagged_synthetic(dummy_arrays):
    X, y, months, prov = dummy_arrays
    assert prov.kind == "synthetic"
    assert not prov.is_real
    with pytest.raises(SyntheticDataError):
        prov.assert_real("results")


def test_ember_provenance_allowed(real_provenance):
    assert real_provenance.is_real
    real_provenance.assert_real("results")  # must not raise


def test_synthetic_prediction_file_cannot_feed_metrics(tmp_path):
    pred = tmp_path / "fake.npz"
    np.savez_compressed(
        pred,
        y_true=np.array([0, 1, 1]),
        y_prob=np.array([0.1, 0.9, 0.8]),
        method=np.str_("dummy"),
        seed=np.int64(0),
        window=np.str_("T1"),
        provenance=np.str_("synthetic"),
    )
    with pytest.raises(SyntheticDataError):
        metrics_from_saved_predictions(pred)


def test_results_writer_saves_and_recomputes_with_ember_provenance(tmp_path, real_provenance):
    writer = ResultsWriter(tmp_path / "results", real_provenance)
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_prob = np.clip(y_true * 0.7 + rng.normal(0.2, 0.2, 200), 0, 1)
    path = writer.save_predictions("LR", 0, "T1", y_true, y_prob)
    metrics = metrics_from_saved_predictions(path)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["auc_roc"] <= 1.0
    assert np.isfinite(metrics["ece"])
