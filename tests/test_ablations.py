"""Unit tests: ablation variants, scale-study resolution, significance CSV."""

import numpy as np
import pandas as pd
import pytest

from armada.ablations import (
    REFERENCE,
    _scale_points,
    resolve_variant,
    run_significance,
)


def _tiny_cfg():
    return {
        "feature_groups": {"imports": [0, 10], "strings": [10, 20], "header": [20, 22]},
        "train": {"use_adversarial_training": True},
        "model": {"shared_group_projection": False},
        "ablation": {"significance": {"anchor": "ARMADA", "metrics": ["auc_roc", "f1"]}},
        "scale_study": {"n_train_values": [20000, 50000]},
    }


def test_resolve_variant_reference_unchanged():
    cfg = _tiny_cfg()
    v, label = resolve_variant(REFERENCE, cfg)
    assert label == REFERENCE
    assert v["train"]["use_adversarial_training"] is True
    assert v["model"]["shared_group_projection"] is False
    assert cfg["train"]["use_adversarial_training"] is True  # no mutation


def test_resolve_variant_no_at_and_shared_projection():
    v, _ = resolve_variant("no_adversarial_training", _tiny_cfg())
    assert v["train"]["use_adversarial_training"] is False
    v, _ = resolve_variant("shared_group_projection", _tiny_cfg())
    assert v["model"]["shared_group_projection"] is True


def test_resolve_variant_drop_group_masks_columns():
    v, label = resolve_variant("drop_ImportsInfo", _tiny_cfg())
    assert label == "drop_ImportsInfo"
    assert v["ablation"]["mask_groups"] == ["ImportsInfo"]
    assert "ablation" in v  # mask recorded, schema untouched


def test_resolve_variant_unknown_names_raise():
    with pytest.raises(ValueError, match="unknown feature group"):
        resolve_variant("drop_nothing", _tiny_cfg())
    with pytest.raises(ValueError, match="unknown ablation"):
        resolve_variant("bogus_variant", _tiny_cfg())


def test_apply_group_mask_zeroes_processed_blocks():
    import numpy as np

    from armada.ablations import _apply_group_mask

    class _FakeDS:
        source_train_proc = {"ImportsInfo": np.ones((3, 5), dtype=np.float32), "HeaderFileInfo": np.ones((3, 2), dtype=np.float32)}
        source_val_proc = {"ImportsInfo": np.ones((2, 5), dtype=np.float32), "HeaderFileInfo": np.ones((2, 2), dtype=np.float32)}
        windows = [
            {
                "proc": {"ImportsInfo": np.ones((4, 5), dtype=np.float32), "HeaderFileInfo": np.ones((4, 2), dtype=np.float32)},
                "unl_proc": {"ImportsInfo": np.ones((4, 5), dtype=np.float32), "HeaderFileInfo": np.ones((4, 2), dtype=np.float32)},
            }
        ]

        def ensure_processed(self):
            pass

    ds = _FakeDS()
    _apply_group_mask(ds, ["ImportsInfo"])
    for block in (ds.source_train_proc, ds.source_val_proc, ds.windows[0]["proc"], ds.windows[0]["unl_proc"]):
        assert float(np.abs(block["ImportsInfo"]).sum()) == 0.0
        assert float(block["HeaderFileInfo"].sum()) > 0.0  # untouched


def test_scale_points_from_either_key():
    assert _scale_points(_tiny_cfg()) == [20000, 50000]
    cfg = _tiny_cfg()
    cfg["ablation"] = {"scale_study": [1, 2]}
    assert _scale_points(cfg) == [1, 2]
    cfg["ablation"] = {}
    cfg["scale_study"] = {"n_train_values": []}
    assert _scale_points(cfg) == []


def test_run_significance_writes_csv(tmp_path):
    rows = []
    for seed in (0, 1, 2):
        rows.append({"method": "ARMADA", "seed": seed, "window": "T2", "auc_roc": 0.9, "f1": 0.8})
        rows.append({"method": "LR", "seed": seed, "window": "T2", "auc_roc": 0.7, "f1": 0.6})
        rows.append({"method": "DANN", "seed": seed, "window": "T2", "auc_roc": 0.85, "f1": 0.75})
    pd.DataFrame(rows).to_csv(tmp_path / "metrics_per_seed.csv", index=False)

    out = run_significance(tmp_path, _tiny_cfg())
    df = pd.read_csv(out)
    assert set(df["method_b"]) == {"LR", "DANN"}
    assert set(df["metric"]) == {"auc_roc", "f1"}  # configured subset only
    assert (df["n_pairs"] == 3).all()
    assert (df["test"] == "exact_sign_flip").all()
    # ARMADA > LR on every seed -> p = 2/8 exact
    lr_row = df[(df["method_b"] == "LR") & (df["metric"] == "auc_roc")].iloc[0]
    assert lr_row["mean_diff"] == pytest.approx(0.2)
    assert lr_row["p_value"] == pytest.approx(0.25)


def test_run_significance_requires_eval_output(tmp_path):
    with pytest.raises(FileNotFoundError, match="stage eval"):
        run_significance(tmp_path, _tiny_cfg())
