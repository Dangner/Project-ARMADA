"""Smoke tests: ARMADA GUI scoring engine (no Tkinter required)."""

import numpy as np
import pytest

from armada.models.decision import MALWARE, SAFE, SUSPICIOUS


def test_scan_engine_requires_checkpoint(fixture_cfg, tmp_path):
    from armada.gui import ScanEngine

    fixture_cfg["train"] = dict(fixture_cfg["train"], stage_a_epochs=1, batch_size=64)
    with pytest.raises(FileNotFoundError, match="stage train"):
        ScanEngine(fixture_cfg, seed=0)


def test_scan_engine_random_scan_payload(fixture_cfg, tmp_path):
    from armada.gui import ScanEngine
    from armada.run import stage_data, stage_train

    fixture_cfg["train"] = dict(
        fixture_cfg["train"], stage_a_epochs=1, stage_b_epochs=1, batch_size=64
    )
    fixture_cfg["ttt"] = {"enabled": False}
    fixture_cfg["memory"] = {
        "capacity": 128, "store_benign": True, "evict": "age",
        "max_age_windows": 3, "add_confidence": 0.9,
    }
    fixture_cfg["decision"] = {
        "malware_fpr_target": 0.1, "suspicious_fpr_target": 0.3, "calibrator": "logistic",
    }
    stage_data(fixture_cfg)
    stage_train(fixture_cfg)

    np.random.seed(0)
    engine = ScanEngine(fixture_cfg, seed=0)
    payload = engine.random_scan()
    for key in ("window", "index", "true_label", "p_malware", "max_memory_similarity",
                "score", "verdict", "thresholds", "group_threat"):
        assert key in payload
    assert payload["verdict"] in (SAFE, SUSPICIOUS, MALWARE)
    assert 0.0 <= payload["score"] <= 1.0
    assert payload["thresholds"][1] <= payload["thresholds"][0]
    assert payload["group_threat"]  # per-group activations present
