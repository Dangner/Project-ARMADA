"""Ablations + scale study + significance (spec §3.1, §3.9).

``--stage ablate`` runs, all from real EMBER:

- **training ablations** (one factor changed per variant): no adversarial
  training, shared group projection, and drop-one-feature-group for every
  configured group — each trained and evaluated like the reference model;
- **scale study**: the reference model at source-train budgets
  ``ablation.scale_study`` (total rows = 3 × per-class cap);
- **paired significance tests** over seeds (exact sign-flip) between ARMADA
  and every comparator, computed from the saved ``metrics_per_seed.csv``.

Ablation runs use their own checkpoint directories so they never overwrite
the main run's checkpoints.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REFERENCE = "reference"
TRAIN_ABLATIONS = ("no_adversarial_training", "shared_group_projection")


def resolve_variant(name: str, cfg: Mapping) -> Tuple[Dict, str]:
    """Return (variant cfg, short label) for one ablation name.

    - ``reference``: unchanged config;
    - ``no_adversarial_training``: ``train.use_adversarial_training=false``;
    - ``shared_group_projection``: ``model.shared_group_projection=true``;
    - ``drop_<group>``: the named feature group is zero-masked after
      source-only preprocessing (train and eval see no signal from it).
    """
    new_cfg = copy.deepcopy(dict(cfg))
    label = name
    if name == REFERENCE:
        return new_cfg, label
    if name == "no_adversarial_training":
        new_cfg.setdefault("train", {})["use_adversarial_training"] = False
        return new_cfg, label
    if name == "shared_group_projection":
        new_cfg.setdefault("model", {})["shared_group_projection"] = True
        return new_cfg, label
    if name.startswith("drop_"):
        group = name[len("drop_") :]
        from .data.groups import GROUP_NAMES

        if group not in GROUP_NAMES:
            raise ValueError(
                f"unknown feature group in ablation {name!r}: {group!r} "
                f"(canonical names: {GROUP_NAMES})"
            )
        # Information ablation: the group's processed block is zero-masked
        # after source-only preprocessing (train + eval), so the model keeps
        # its architecture but receives no signal from that group.
        new_cfg.setdefault("ablation", {})["mask_groups"] = [group]
        return new_cfg, label
    raise ValueError(f"unknown ablation variant: {name!r}")


def _apply_group_mask(ds, mask_groups: Sequence[str]) -> None:
    """Zero-mask processed blocks of ``mask_groups`` (train + val + windows)."""
    if not mask_groups:
        return
    ds.ensure_processed()
    blocks = [ds.source_train_proc, ds.source_val_proc]
    for w in ds.windows:
        blocks.append(w["proc"])
        if w.get("unl_proc"):
            blocks.append(w["unl_proc"])
    for block in blocks:
        for g in mask_groups:
            if g in block:
                block[g] = np.zeros_like(block[g])


def _train_and_eval_armada(cfg: Mapping, results_dir: Path, method: str) -> List[Dict[str, object]]:
    """Train Stage A+B (dual) per seed in an isolated checkpoint dir, then
    evaluate the full ARMADA pipeline for each seed."""
    from .data.loader import prepare_seed_dataset
    from .eval.baselines import ResultsWriter
    from .eval.neural import evaluate_armada_seed
    from .run import build_pool  # lazy: run.py imports ablations lazily too
    from .train.adapt import train_stage_b
    from .train.pretrain import train_stage_a
    from .utils import set_seed

    ckpt_dir = Path(cfg["data"].get("checkpoints_dir", "checkpoints"))
    pool = build_pool(cfg)
    mask_groups = list(cfg.get("ablation", {}).get("mask_groups", []))
    rows: List[Dict[str, object]] = []
    for seed in cfg["seed_list"]:
        set_seed(int(seed))
        ds = prepare_seed_dataset(pool, cfg, int(seed))
        _apply_group_mask(ds, mask_groups)
        writer = ResultsWriter(results_dir, ds.provenance)
        result_a = train_stage_a(ds, cfg, out_dir=ckpt_dir)
        result_b = train_stage_b(
            ds, cfg, "dual", out_dir=ckpt_dir, init_state=result_a.state_dict
        )
        if result_b.checkpoint_path is None:
            raise RuntimeError(f"ablation {method}: Stage B produced no checkpoint")
        m_rows, _ = evaluate_armada_seed(
            method=method,
            checkpoint_path=result_b.checkpoint_path,
            ds=ds,
            cfg=cfg,
            writer=writer,
            use_memory=True,
            use_ttt=bool(cfg.get("ttt", {}).get("enabled", True)),
        )
        rows.extend(m_rows)
    return rows


def run_ablations(cfg: Mapping, results_dir: Path) -> Path:
    """Train/evaluate every configured ablation + the reference, write CSV."""
    ab = cfg.get("ablation", {})
    variants: Sequence[str] = list(ab.get("variants", []))
    if not ab.get("run_variants", True) or not variants:
        logger.info("no ablation variants configured — skipping training ablations")
        return results_dir / "ablation.csv"
    names = [REFERENCE] + list(variants)
    all_rows: List[Dict[str, object]] = []
    ckpt_root = Path(cfg["data"].get("checkpoints_dir", "checkpoints"))
    for name in names:
        v_cfg, label = resolve_variant(name, cfg)
        # isolate checkpoints per variant
        v_cfg["data"] = dict(
            v_cfg["data"], checkpoints_dir=str(ckpt_root / f"ablate_{label}")
        )
        method = f"ABL:{label}"
        logger.info(
            "ablation run: %s (checkpoints -> %s)", method, v_cfg["data"]["checkpoints_dir"]
        )
        rows = _train_and_eval_armada(v_cfg, results_dir, method)
        for r in rows:
            r["ablation"] = label
        all_rows.extend(rows)

    out = results_dir / "ablation.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    logger.info("wrote %s (%d rows)", out, len(all_rows))
    return out


def _scale_points(cfg: Mapping) -> List[int]:
    """Scale-study budgets (TOTAL source-train rows), from either key."""
    pts = cfg.get("ablation", {}).get("scale_study")
    if pts is None:
        pts = cfg.get("scale_study", {}).get("n_train_values", [])
    return [int(p) for p in pts]


def run_scale_study(cfg: Mapping, results_dir: Path) -> Path:
    """Reference model at increasing source-train budgets.

    Each point is the TOTAL number of source-train rows used (labels -1/0/1
    combined); the per-class cap is ``point // 3``.  Skips quietly when no
    points are configured.
    """
    points = _scale_points(cfg)
    if not points:
        logger.info("no scale points configured — skipping scale study")
        return results_dir / "scale_study.csv"
    all_rows: List[Dict[str, object]] = []
    ckpt_root = Path(cfg["data"].get("checkpoints_dir", "checkpoints"))
    for point in points:
        per_class = max(8, point // 3)
        s_cfg = copy.deepcopy(dict(cfg))
        s_cfg["data"]["n_train"] = per_class
        if "n_test" in s_cfg["data"]:
            s_cfg["data"]["n_test"] = min(int(s_cfg["data"]["n_test"]), per_class)
        s_cfg["data"] = dict(
            s_cfg["data"], checkpoints_dir=str(ckpt_root / f"scale_{point}")
        )
        method = f"ARMADA@{point}"
        logger.info("scale study point %d rows (n_train=%d/label)", point, per_class)
        rows = _train_and_eval_armada(s_cfg, results_dir, method)
        for r in rows:
            r["scale_point"] = float(point)
            r["n_train_per_class"] = float(per_class)
        all_rows.extend(rows)

    out = results_dir / "scale_study.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    logger.info("wrote %s (%d rows)", out, len(all_rows))
    return out


def run_significance(results_dir: Path, cfg: Mapping) -> Path:
    """Paired sign-flip tests from the saved per-seed metrics.

    Compares ARMADA against every other method with matched seeds on each
    window and each configured metric.
    """
    from .eval.significance import paired_method_comparison

    src = results_dir / "metrics_per_seed.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found — run `--stage eval` before `--stage ablate` for significance tests"
        )
    df = pd.read_csv(src)
    sig_cfg = cfg.get("ablation", {}).get("significance", {})
    metrics = list(sig_cfg.get("metrics", ["auc_roc", "auc_pr", "f1"]))
    anchor = str(sig_cfg.get("anchor", "ARMADA"))
    if anchor not in set(df["method"]):
        raise ValueError(f"anchor method {anchor!r} missing from {src}")

    rows: List[Dict[str, object]] = []
    others = sorted(set(df["method"]) - {anchor})
    for window in sorted(set(df["window"])):
        wdf = df[df["window"] == window]
        for other in others:
            for metric in metrics:
                if metric not in wdf.columns:
                    continue
                a = wdf[wdf["method"] == anchor].sort_values("seed")
                b = wdf[wdf["method"] == other].sort_values("seed")
                merged = a[["seed", metric]].merge(
                    b[["seed", metric]], on="seed", suffixes=("_a", "_b")
                )
                if merged.empty:
                    continue
                r = paired_method_comparison(
                    merged[f"{metric}_a"].to_numpy(), merged[f"{metric}_b"].to_numpy()
                )
                rows.append(
                    {
                        "window": window,
                        "metric": metric,
                        "method_a": anchor,
                        "method_b": other,
                        "n_pairs": r["n_pairs"],
                        "mean_diff": r["mean_diff"],
                        "p_value": r["p_value"],
                        "test": "exact_sign_flip",
                    }
                )

    out = results_dir / "significance.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("wrote %s (%d comparisons)", out, len(rows))
    return out
