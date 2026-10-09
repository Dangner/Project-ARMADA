"""Tests: adversarial training in Stage B + robustness evaluation (spec §3.6)."""

import numpy as np
import pytest
import torch

from armada.attacks import FeatureSpaceProjector
from armada.data.loader import EmberPool, load_ember_train, prepare_seed_dataset
from armada.train.adapt import train_stage_b


@pytest.fixture
def fixture_ds(fixture_cfg):
    pool = EmberPool([load_ember_train(fixture_cfg["data"]["data_dir"])])
    return prepare_seed_dataset(pool, fixture_cfg, seed=0, use_cache=False)


def test_adversarial_training_adds_adv_loss_component(fixture_cfg, fixture_ds, tmp_path):
    cfg = dict(fixture_cfg)
    cfg["train"] = dict(
        cfg["train"],
        stage_b_epochs=1,
        batch_size=32,
        use_adversarial_training=True,
        adv_weight=0.5,
        adv_attack="fgsm",
    )
    cfg["ttt"] = {"mask_ratio": 0.3}
    cfg["robustness"] = {"fgsm_epsilon": 0.05, "clip_min": 0.0}
    result = train_stage_b(fixture_ds, cfg, variant="dual", out_dir=tmp_path)
    row = result.curves[-1]
    assert "train_adv" in row
    assert np.isfinite(row["train_adv"])
    assert row["train_adv"] > 0  # adversarial loss actually contributed


def test_adversarial_training_pgd_variant(fixture_cfg, fixture_ds):
    cfg = dict(fixture_cfg)
    cfg["train"] = dict(
        cfg["train"],
        stage_b_epochs=1,
        batch_size=32,
        use_adversarial_training=True,
        adv_attack="pgd",
    )
    cfg["ttt"] = {"mask_ratio": 0.3}
    cfg["robustness"] = {"pgd_epsilon": 0.05, "pgd_steps": 2, "pgd_step_size": 0.02, "clip_min": 0.0}
    result = train_stage_b(fixture_ds, cfg, variant="dual")
    assert result.epochs_run >= 1
    assert np.isfinite(result.curves[-1]["train_adv"])


def test_adversarial_training_off_by_default(fixture_cfg, fixture_ds):
    cfg = dict(fixture_cfg)
    cfg["train"] = dict(cfg["train"], stage_b_epochs=1, batch_size=32)
    cfg["ttt"] = {"mask_ratio": 0.3}
    result = train_stage_b(fixture_ds, cfg, variant="dual")
    assert result.curves[-1]["train_adv"] == 0.0


def test_robustness_eval_writes_clean_and_attacked_rows(fixture_cfg, fixture_ds, tmp_path):
    from armada.eval.baselines import ResultsWriter
    from armada.eval.robustness import evaluate_robustness_seed
    from armada.models.core import ArmadaDomainModel

    ds = fixture_ds
    writer = ResultsWriter(tmp_path / "results", ds.provenance)
    model = ArmadaDomainModel.from_config(
        {**fixture_cfg, "train": {**fixture_cfg["train"], "use_d2": True}},
        ds.preprocessor.processed_dims,
    )
    cfg = dict(fixture_cfg)
    cfg["robustness"] = {"eval_epsilons": [0.0, 0.1], "pgd_steps": 2, "pgd_step_size": 0.05, "clip_min": 0.0}
    rows = evaluate_robustness_seed("TESTM", model, ds, cfg, writer, torch.device("cpu"))

    # per window: 1 clean + 2 eps x 3 attacks
    attacks = {(r["attack"], r["epsilon"]) for r in rows}
    assert ("clean", 0.0) in attacks
    for attack in ("fgsm", "pgd", "noise"):
        assert (attack, 0.1) in attacks
    for r in rows:
        assert 0.0 <= r["accuracy"] <= 1.0
    # predictions were saved for every row
    preds = list((tmp_path / "results" / "predictions").glob("robust_TESTM_*.npz"))
    assert len(preds) == len(rows)


def test_noise_attack_respects_ball(fixture_cfg):
    from armada.eval.robustness import _noise_attack

    pool = EmberPool([load_ember_train(fixture_cfg["data"]["data_dir"])])
    ds = prepare_seed_dataset(pool, fixture_cfg, seed=0, use_cache=False)
    projector = FeatureSpaceProjector(ds.preprocessor, ds.source_train_raw)
    groups = {
        name: torch.as_tensor(block[:4], dtype=torch.float32)
        for name, block in ds.source_val_proc.items()
    }
    eps = 0.25
    adv = _noise_attack(groups, eps, projector)
    for name in groups:
        assert float((adv[name] - groups[name]).abs().max()) <= eps + 1e-5
