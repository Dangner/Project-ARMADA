"""Unit tests: decision engine — thresholds from SOURCE validation only
(spec §3.8, §8 threshold-selection test)."""

import numpy as np
import pytest

from armada.models.decision import (
    MALWARE,
    SAFE,
    SUSPICIOUS,
    DecisionEngine,
    threshold_at_fpr,
)


def _source_val(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # scores separate classes but with overlap
    scores = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.18, n), 0, 1)
    return y, scores


def test_threshold_at_fpr_respects_budget():
    y, s = _source_val()
    t = threshold_at_fpr(y, s, 0.01)
    pred = s >= t
    fpr = (pred & (y == 0)).sum() / max(1, (y == 0).sum())
    assert fpr <= 0.01 + 1e-9


def test_thresholds_are_data_derived_not_hardcoded():
    y1, s1 = _source_val(seed=1)
    y2, s2 = _source_val(seed=2)
    s2 = s2 * 0.3 + 0.05  # different score scale
    e1 = DecisionEngine().fit(y1, s1)
    e2 = DecisionEngine().fit(y2, s2)
    assert e1.malware_threshold != e2.malware_threshold  # no hardcoded 30/60
    assert e1.malware_threshold > e1.suspicious_threshold
    assert e2.malware_threshold > e2.suspicious_threshold


def test_engine_hits_fpr_targets_on_source_val():
    y, s = _source_val(n=2000, seed=3)
    e = DecisionEngine(malware_fpr_target=0.01, suspicious_fpr_target=0.1).fit(y, s)
    rep = e.report(y, s)
    fpr_mal = rep["conf_fp"] / max(1, rep["conf_fp"] + rep["conf_tn"])
    assert fpr_mal <= 0.01 + 1e-9


def test_verdict_partition_and_order():
    y, s = _source_val()
    e = DecisionEngine().fit(y, s)
    v = e.verdicts(s)
    assert set(v) <= {SAFE, SUSPICIOUS, MALWARE}
    # high score -> MALWARE, low score -> SAFE
    assert e.verdicts(np.array([1.0]))[0] == MALWARE
    assert e.verdicts(np.array([0.0]))[0] == SAFE
    mid = (e.suspicious_threshold + e.malware_threshold) / 2
    assert e.verdicts(np.array([mid]))[0] == SUSPICIOUS


def test_report_confusion_and_sandbox_rate():
    y = np.array([0, 0, 1, 1, 1, 0])
    s = np.array([0.1, 0.2, 0.95, 0.9, 0.55, 0.6])
    e = DecisionEngine(malware_fpr_target=0.2, suspicious_fpr_target=0.5).fit(
        np.array([0, 0, 1, 1]), np.array([0.05, 0.15, 0.85, 0.95])
    )
    rep = e.report(y, s)
    assert rep["n"] == 6
    assert rep["n_safe"] + rep["n_suspicious"] + rep["n_malware"] == 6
    assert rep["conf_tp"] + rep["conf_fn"] == 3  # all malware counted
    assert rep["conf_fp"] + rep["conf_tn"] == 3
    assert 0.0 <= rep["sandbox_rate"] <= 1.0
    # alert confusion covers everything too
    assert rep["alert_conf_tp"] + rep["alert_conf_fn"] == 3


def test_verdicts_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        DecisionEngine().verdicts(np.array([0.5]))


def test_threshold_at_fpr_needs_benign_rows():
    with pytest.raises(ValueError, match="benign"):
        threshold_at_fpr(np.array([1, 1, 1]), np.array([0.2, 0.4, 0.6]), 0.01)
    with pytest.raises(ValueError, match="empty"):
        threshold_at_fpr(np.array([]), np.array([]), 0.01)
