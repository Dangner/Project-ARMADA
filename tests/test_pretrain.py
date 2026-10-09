"""Tests: Stage A source pre-training (spec §4) — smoke, early stopping,
checkpoint round-trip.  Dummy data in tests/ only; writes go to tmp dirs."""

import numpy as np
import pytest
import torch

from armada.data.loader import EmberPool, load_ember_train, prepare_seed_dataset
from armada.train.pretrain import (
    evaluate_classifier,
    load_checkpoint,
    train_stage_a,
)
from armada.models.core import ArmadaCore


@pytest.fixture
def fixture_ds(fixture_cfg):
    pool = EmberPool([load_ember_train(fixture_cfg["data"]["data_dir"])])
    return prepare_seed_dataset(pool, fixture_cfg, seed=0, use_cache=False)


def test_stage_a_smoke_reduces_loss_and_saves_artifacts(fixture_cfg, fixture_ds, tmp_path):
    cfg = dict(fixture_cfg)
    cfg["train"] = dict(cfg["train"], stage_a_epochs=4, lr=5e-3, batch_size=32)
    result = train_stage_a(fixture_ds, cfg, out_dir=tmp_path)

    assert result.epochs_run >= 1
    assert np.isfinite(result.best_val_auc)
    assert result.curves[0]["train_loss"] > 0
    # the model fits the source period: final train loss below first epoch
    assert result.curves[-1]["train_loss"] < result.curves[0]["train_loss"]

    ckpt = tmp_path / "armada_stageA_seed0.pt"
    curves = tmp_path / "training_curves_seed0.csv"
    assert ckpt.exists() and curves.exists()
    assert result.checkpoint_path == ckpt

    # checkpoint round-trip reproduces the same probabilities
    model_a = ArmadaCore.from_config(cfg, fixture_ds.preprocessor.processed_dims)
    model_a.load_state_dict(result.state_dict)
    model_b, meta = load_checkpoint(ckpt)
    assert meta["seed"] == 0
    assert meta["best_val_auc"] == pytest.approx(result.best_val_auc)

    fixture_ds.ensure_processed()
    groups = {
        name: torch.as_tensor(block[:16], dtype=torch.float32)
        for name, block in fixture_ds.source_val_proc.items()
    }
    with torch.no_grad():
        pa, _ = model_a.predict_proba(groups)
        pb, _ = model_b.predict_proba(groups)
    np.testing.assert_allclose(pa.numpy(), pb.numpy(), rtol=1e-5)


def test_early_stopping_when_validation_does_not_improve(fixture_cfg, fixture_ds):
    cfg = dict(fixture_cfg)
    cfg["train"] = dict(
        cfg["train"],
        stage_a_epochs=30,
        early_stopping_patience=1,
        lr=1e-12,  # effectively no learning -> no val improvement
        batch_size=64,
    )
    result = train_stage_a(fixture_ds, cfg, out_dir=None)
    assert result.epochs_run < 30  # stopped early
    assert result.epochs_run <= 3


def test_evaluate_classifier_on_source_val(fixture_cfg, fixture_ds):
    cfg = dict(fixture_cfg)
    model = ArmadaCore.from_config(cfg, fixture_ds.preprocessor.processed_dims)
    fixture_ds.ensure_processed()
    stats = evaluate_classifier(
        model,
        fixture_ds.source_val_proc,
        fixture_ds.source_val_y,
        batch_size=32,
        device=torch.device("cpu"),
    )
    assert set(stats) == {"n", "auc", "bce"}
    assert stats["n"] == len(fixture_ds.source_val_y)
    assert np.isfinite(stats["bce"])


def test_train_stage_a_requires_source_labels(fixture_cfg, fixture_ds):
    cfg = dict(fixture_cfg)
    cfg["train"] = dict(cfg["train"], stage_a_epochs=1)
    fixture_ds.source_train_y = np.zeros_like(fixture_ds.source_train_y)
    with pytest.raises(ValueError, match="no positive"):
        train_stage_a(fixture_ds, cfg, out_dir=None)
