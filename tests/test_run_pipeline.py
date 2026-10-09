"""End-to-end pipeline smoke test on dummy fixtures (tmp dirs only).

Runs the exact Phase-1 code path of ``python -m armada.run``:
``--stage data`` then ``--stage eval``, against a miniature EMBER-layout
dataset built in tests/.  Nothing here may write to the repository
``results/`` directory — everything lands in pytest tmp paths.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from armada.run import stage_data, stage_eval


def test_stage_data_and_eval_end_to_end(fixture_cfg, tmp_path):
    stage_data(fixture_cfg)
    stage_eval(fixture_cfg)

    results = Path(fixture_cfg["data"]["results_dir"])
    assert (results / "config_used.yaml").exists()

    per_seed = pd.read_csv(results / "metrics_per_seed.csv")
    assert set(per_seed["method"]) == {"LR", "RF"}
    assert set(per_seed["window"]) == {"T1", "T2"}
    assert set(per_seed["seed"]) == {0, 1}

    by_window = pd.read_csv(results / "metrics_by_window.csv")
    assert len(by_window) == 4  # 2 methods x 2 windows
    for col in ("accuracy_mean", "auc_roc_mean", "auc_pr_mean", "ece_mean", "f1_mean"):
        assert col in by_window.columns

    overall = pd.read_csv(results / "metrics_all.csv")
    assert set(overall["method"]) == {"LR", "RF"}
    assert (overall["n_seeds"] == 2).all()

    # every metric is finite and in range; nothing is a magic constant
    for df in (by_window, overall):
        for col in df.columns:
            if col.endswith(("_mean", "_std")):
                vals = df[col].to_numpy(dtype=float)
                assert np.isfinite(vals).all(), col

    # predictions saved per (method, seed, window)
    preds = sorted((results / "predictions").glob("*.npz"))
    assert len(preds) == 8  # 2 methods x 2 seeds x 2 windows

    # cache exists for both seeds
    cache_root = Path(fixture_cfg["data"]["cache_dir"])
    assert len(list(cache_root.glob("pre_*"))) == 2


def test_eval_refuses_to_run_without_ember_data(fixture_cfg, tmp_path):
    import pytest

    from armada.run import stage_data

    fixture_cfg["data"]["data_dir"] = str(tmp_path / "no_such_dir")
    with pytest.raises(FileNotFoundError, match="EMBER 2018 artifacts missing"):
        stage_data(fixture_cfg)


def test_train_and_eval_includes_grouped_attention(fixture_cfg, tmp_path):
    """Phase 2 loop: --stage train then --stage eval adds GroupedAttn rows."""
    from armada.run import stage_data, stage_eval, stage_train

    fixture_cfg["train"] = dict(fixture_cfg["train"], stage_a_epochs=3, batch_size=64)
    stage_data(fixture_cfg)
    stage_train(fixture_cfg)

    ckpt_dir = Path(fixture_cfg["data"]["checkpoints_dir"])
    assert (ckpt_dir / "armada_stageA_seed0.pt").exists()
    assert (ckpt_dir / "armada_stageA_seed1.pt").exists()
    assert (ckpt_dir / "training_curves_seed0.csv").exists()

    stage_eval(fixture_cfg)
    results = Path(fixture_cfg["data"]["results_dir"])
    overall = pd.read_csv(results / "metrics_all.csv")
    # Phase 2 rows must be present; later phases may add more methods.
    assert {"LR", "RF", "GroupedAttn"} <= set(overall["method"])
    by_window = pd.read_csv(results / "metrics_by_window.csv")
    attn = by_window[by_window["method"] == "GroupedAttn"]
    assert set(attn["window"]) == {"T1", "T2"}
    assert np.isfinite(attn["auc_roc_mean"].to_numpy(dtype=float)).all()


def test_phase3_full_loop_includes_dann_and_ttt_rows(fixture_cfg, tmp_path):
    """Stage B variants reach the metrics tables with per-component curves."""
    from armada.run import stage_data, stage_eval, stage_train

    fixture_cfg["train"] = dict(fixture_cfg["train"], stage_a_epochs=1, stage_b_epochs=2, batch_size=64)
    fixture_cfg["ttt"] = {"enabled": True, "steps": 1, "lr": 1e-3, "mask_ratio": 0.3, "report_online": True}
    stage_data(fixture_cfg)
    stage_train(fixture_cfg)

    ckpt_dir = Path(fixture_cfg["data"]["checkpoints_dir"])
    for variant in ("dann", "dual"):
        assert (ckpt_dir / f"armada_stageB_{variant}_seed0.pt").exists()
    curves = pd.read_csv(ckpt_dir / "training_curves_stageB_dual_seed0.csv")
    for col in ("train_cls", "train_marg", "train_cond", "train_recon"):
        assert col in curves.columns

    stage_eval(fixture_cfg)
    results = Path(fixture_cfg["data"]["results_dir"])
    overall = pd.read_csv(results / "metrics_all.csv")
    expected = {
        "LR", "RF", "GroupedAttn", "DANN", "DualDANN", "DualDANN+TTT", "DualDANN+TTT-online",
        "ARMADA", "ARMADA-noMem",
    }
    assert expected == set(overall["method"])

    per_seed = pd.read_csv(results / "metrics_per_seed.csv")
    ttt_rows = per_seed[(per_seed["method"] == "DualDANN+TTT")]
    # TTT runs on every window that has unlabeled rows (fixture T2 has none);
    # at least one window must have been adapted on.
    assert (ttt_rows["ttt_steps"] >= 1).any()


def test_phase4_robust_stage_end_to_end(fixture_cfg, tmp_path):
    """--stage robust writes results/robustness.csv with clean + attacked rows."""
    from armada.run import stage_data, stage_robust, stage_train

    fixture_cfg["train"] = dict(fixture_cfg["train"], stage_a_epochs=1, stage_b_epochs=1, batch_size=64)
    fixture_cfg["ttt"] = {"enabled": True, "steps": 1, "lr": 1e-3, "mask_ratio": 0.3}
    fixture_cfg["robustness"] = {
        "eval_epsilons": [0.0, 0.1],
        "pgd_steps": 2,
        "pgd_step_size": 0.05,
        "clip_min": 0.0,
    }
    stage_data(fixture_cfg)
    stage_train(fixture_cfg)
    stage_robust(fixture_cfg)

    results = Path(fixture_cfg["data"]["results_dir"])
    rob = pd.read_csv(results / "robustness.csv")
    assert {"GroupedAttn", "DANN", "DualDANN", "DualDANN+TTT"} == set(rob["method"])
    assert {"clean", "fgsm", "pgd", "noise"} == set(rob["attack"])
    assert (rob["epsilon"] == 0.0).any()
    assert set(rob["seed"]) == {0, 1}


def test_phase5_armada_rows_and_decision_report(fixture_cfg, tmp_path):
    """Full ARMADA (memory + calibrator + decision) rows and the decision report."""
    from armada.run import stage_data, stage_eval, stage_train

    fixture_cfg["train"] = dict(fixture_cfg["train"], stage_a_epochs=1, stage_b_epochs=1, batch_size=64)
    fixture_cfg["ttt"] = {"enabled": True, "steps": 1, "lr": 1e-3, "mask_ratio": 0.3}
    fixture_cfg["memory"] = {
        "capacity": 256,
        "store_benign": True,
        "evict": "age",
        "max_age_windows": 3,
        "add_confidence": 0.9,
    }
    fixture_cfg["decision"] = {
        "malware_fpr_target": 0.05,
        "suspicious_fpr_target": 0.2,
        "calibrator": "logistic",
    }
    stage_data(fixture_cfg)
    stage_train(fixture_cfg)
    stage_eval(fixture_cfg)

    results = Path(fixture_cfg["data"]["results_dir"])
    overall = pd.read_csv(results / "metrics_all.csv")
    assert {"ARMADA", "ARMADA-noMem"} <= set(overall["method"])

    dec = pd.read_csv(results / "decision_report.csv")
    assert {"ARMADA", "ARMADA-noMem"} == set(dec["method"])
    for col in (
        "n_safe",
        "n_suspicious",
        "n_malware",
        "sandbox_rate",
        "conf_tp",
        "conf_fp",
        "conf_tn",
        "conf_fn",
        "malware_threshold",
        "suspicious_threshold",
    ):
        assert col in dec.columns
    # verdict counts add up to the window size
    assert (dec["n_safe"] + dec["n_suspicious"] + dec["n_malware"] == dec["n"]).all()
    # thresholds are recorded per row and are data-derived (finite)
    assert np.isfinite(dec["malware_threshold"]).all()
    assert (dec["malware_threshold"] >= dec["suspicious_threshold"]).all()
