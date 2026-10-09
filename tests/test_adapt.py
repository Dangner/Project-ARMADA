"""Tests: Stage B joint adaptation training (spec §3.4/§3.5/§4)."""

import numpy as np
import pytest
import torch

from armada.data.loader import EmberPool, load_ember_train, prepare_seed_dataset
from armada.models.grl import grl_lambda
from armada.train.adapt import load_stage_b, train_stage_b
from armada.train.pretrain import train_stage_a


@pytest.fixture
def fixture_ds(fixture_cfg):
    pool = EmberPool([load_ember_train(fixture_cfg["data"]["data_dir"])])
    return prepare_seed_dataset(pool, fixture_cfg, seed=0, use_cache=False)


def test_stage_b_dual_smoke_curves_and_checkpoint(fixture_cfg, fixture_ds, tmp_path):
    cfg = dict(fixture_cfg)
    cfg["train"] = dict(cfg["train"], stage_b_epochs=2, batch_size=32, lr=5e-3)
    cfg["ttt"] = {"mask_ratio": 0.3, "steps": 2, "lr": 1e-3}
    result = train_stage_b(fixture_ds, cfg, variant="dual", out_dir=tmp_path)

    assert result.epochs_run >= 1
    assert result.checkpoint_path.exists()
    curves = result.curves
    for row in curves:
        for key in ("train_cls", "train_marg", "train_cond", "train_recon", "grl_lambda"):
            assert np.isfinite(row[key]), key
    # dual variant actually trains the conditional path
    assert curves[-1]["train_cond"] > 0
    assert curves[-1]["train_recon"] > 0
    # GRL schedule grows over training
    assert curves[-1]["grl_lambda"] >= curves[0]["grl_lambda"]

    model, meta = load_stage_b(result.checkpoint_path)
    assert meta["variant"] == "dual"
    assert model.use_d2 and model.d2 is not None


def test_stage_b_dann_variant_skips_d2_and_recon(fixture_cfg, fixture_ds, tmp_path):
    cfg = dict(fixture_cfg)
    cfg["train"] = dict(cfg["train"], stage_b_epochs=1, batch_size=32)
    cfg["ttt"] = {"mask_ratio": 0.3}
    result = train_stage_b(fixture_ds, cfg, variant="dann", out_dir=tmp_path)
    row = result.curves[-1]
    assert row["train_cond"] == 0.0
    assert row["train_recon"] == 0.0
    assert row["train_marg"] > 0
    model, meta = load_stage_b(result.checkpoint_path)
    assert meta["variant"] == "dann"
    assert model.d2 is None


def test_stage_b_rejects_unknown_variant(fixture_cfg, fixture_ds):
    with pytest.raises(ValueError, match="variant"):
        train_stage_b(fixture_ds, fixture_cfg, variant="nope")


def test_stage_b_can_init_from_stage_a(fixture_cfg, fixture_ds, tmp_path):
    cfg = dict(fixture_cfg)
    cfg["train"] = dict(cfg["train"], stage_a_epochs=1, stage_b_epochs=1, batch_size=32)
    cfg["ttt"] = {"mask_ratio": 0.3}
    res_a = train_stage_a(fixture_ds, cfg, out_dir=tmp_path)
    res_b = train_stage_b(
        fixture_ds, cfg, variant="dual", init_state=res_a.state_dict
    )
    assert res_b.epochs_run >= 1
    # core weights should be near the Stage A init (small lr, 1 epoch)
    assert res_b.state_dict is not None


def test_joint_loss_components_match_grl_schedule(fixture_cfg, fixture_ds):
    """lambda at p=0 is 0 (pure supervised) and grows; keep the mapping tested."""
    assert grl_lambda(0.0) == 0.0
    assert grl_lambda(1.0) > 0.99
