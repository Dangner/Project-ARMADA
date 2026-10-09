"""Unit tests: immune memory bank (spec §3.7)."""

import numpy as np
import pytest

from armada.models.memory import MemoryBank, MemoryCalibrator


def _z(*coords):
    return np.asarray(coords, dtype=np.float32).reshape(1, -1)


def test_max_malware_similarity_is_cosine_to_nearest_malware():
    bank = MemoryBank(capacity=16, store_benign=True)
    bank.add(np.stack([np.array([1.0, 0.0]), np.array([0.0, 1.0])]), np.array([1, 1]))
    q = np.stack([np.array([1.0, 0.1]), np.array([0.0, 2.0])])
    sims = bank.max_malware_similarity(q)
    assert sims.shape == (2,)
    assert sims[0] == pytest.approx(1.0 / np.sqrt(1.01), rel=1e-4)  # closest to [1,0]
    assert sims[1] == pytest.approx(1.0, rel=1e-4)  # colinear with [0,1]


def test_benign_entries_excluded_from_malware_similarity():
    bank = MemoryBank(capacity=16, store_benign=True)
    bank.add(np.stack([np.array([1.0, 0.0])]), np.array([0]))  # benign only
    sims = bank.max_malware_similarity(np.stack([np.array([1.0, 0.0])]))
    assert float(sims[0]) == 0.0  # no malware entries -> zero threat similarity
    assert bank.n_benign == 1 and bank.n_malware == 0


def test_store_benign_flag():
    bank = MemoryBank(capacity=16, store_benign=False)
    added = bank.add(np.stack([np.array([1.0, 0.0]), np.array([0.0, 1.0])]), np.array([0, 1]))
    assert added == 1
    assert len(bank) == 1


def test_capacity_eviction_keeps_newest():
    bank = MemoryBank(capacity=3, evict="age")
    for i in range(6):
        bank.add(_z(1.0, float(i)), np.array([1]))
        bank.step_window()
    assert len(bank) == 3
    assert bank.n_evicted == 3


def test_age_eviction_after_windows():
    bank = MemoryBank(capacity=64, evict="age", max_age_windows=2)
    bank.add(_z(1.0, 0.0), np.array([1]))
    bank.step_window()
    bank.step_window()
    bank.step_window()
    assert len(bank) == 0  # older than 2 windows
    assert bank.n_evicted == 1


def test_lru_eviction_prefers_unused_entries():
    bank = MemoryBank(capacity=2, evict="lru")
    bank.add(np.stack([np.array([1.0, 0.0]), np.array([0.0, 1.0])]), np.array([1, 1]))
    bank.max_malware_similarity(_z(1.0, 0.0))  # touches entry 0
    bank.add(_z(0.5, 0.5), np.array([1]))  # exceeds capacity -> evict entry 1
    assert len(bank) == 2
    assert float(bank.max_malware_similarity(_z(1.0, 0.0))[0]) == pytest.approx(1.0, rel=1e-4)


def test_update_from_detections_threshold_rule():
    bank = MemoryBank(capacity=64, store_benign=True)
    z = np.stack([np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])])
    scores = np.array([0.99, 0.5, 0.96])
    added = bank.update_from_detections(z, scores, threshold=0.95)
    assert added == 2  # 0.99 and 0.96 pass; 0.5 does not
    assert bank.n_malware == 2


def test_calibrator_learns_weights_and_scores():
    rng = np.random.default_rng(0)
    n = 200
    p = rng.uniform(0, 1, n)
    sim = rng.uniform(0, 1, n)
    y = ((p + sim) > 1.0).astype(np.int64)
    cal = MemoryCalibrator(use_memory=True).fit(p, sim, y)
    assert cal.model is not None
    coef = cal.model.coef_
    assert coef.shape == (1, 2)
    assert coef[0, 0] > 0 and coef[0, 1] > 0  # learned, positive contributions
    scores = cal.score(p, sim)
    assert scores.shape == (n,)
    assert np.all((scores >= 0) & (scores <= 1))


def test_calibrator_no_memory_ignores_similarity():
    rng = np.random.default_rng(1)
    n = 100
    p = rng.uniform(0, 1, n)
    sim = rng.uniform(0, 1, n)
    y = (p > 0.5).astype(np.int64)
    cal = MemoryCalibrator(use_memory=False).fit(p, sim, y)
    s1 = cal.score(p, sim)
    s2 = cal.score(p, np.zeros(n))
    np.testing.assert_allclose(s1, s2)  # similarity has no effect


def test_calibrator_requires_both_classes():
    cal = MemoryCalibrator()
    with pytest.raises(ValueError, match="both classes"):
        cal.fit(np.array([0.1, 0.9]), np.array([0.0, 1.0]), np.array([1, 1]))


def test_calibrator_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        MemoryCalibrator().score(np.array([0.5]), np.array([0.1]))
