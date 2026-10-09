"""ARMADA single entry point.

    python -m armada.run --config configs/main.yaml --stage {data,train,eval,ablate,robust,figures,all}
    python -m armada.run --config configs/main.yaml --stage data --fast   # smoke run
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import yaml

logger = logging.getLogger("armada")

STAGES = ("data", "train", "eval", "ablate", "robust", "figures", "all")
PHASE_OF_STAGE = {
    "figures": "Phase 7 (figures, tables, REPORT.md)",
}


# ---------------------------------------------------------------------------
# Config / logging plumbing
# ---------------------------------------------------------------------------


def load_config(path: os.PathLike | str) -> Dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} must be a mapping")
    for key in ("data", "split", "model", "train", "eval"):
        if key not in cfg:
            raise ValueError(f"config {path} is missing required section '{key}'")
    return cfg


def setup_logging(logs_dir: Path, stage: str) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"run_{stage}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return log_path


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def log_peak_memory(stage: str) -> None:
    logger.info("peak RSS after stage '%s': %.2f GB", stage, peak_rss_gb())


def save_config_used(cfg: Mapping, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "config_used.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False)
    return path


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def build_pool(cfg: Mapping):
    from armada.data.loader import EmberPool, load_ember_test, load_ember_train, require_ember_artifacts

    data_dir = cfg["data"]["data_dir"]
    require_ember_artifacts(data_dir)
    splits = [load_ember_train(data_dir)]
    if cfg["data"].get("include_test_subset", False):
        splits.append(load_ember_test(data_dir))
    pool = EmberPool(splits)
    logger.info(
        "EMBER pool: %d rows (%d splits), data_dir=%s",
        len(pool),
        len(splits),
        data_dir,
    )
    return pool


def stage_data(cfg: Mapping) -> None:
    from armada.data.loader import prepare_seed_dataset

    t0 = time.perf_counter()
    pool = build_pool(cfg)
    for seed in cfg["seed_list"]:
        logger.info("== data stage: seed %s ==", seed)
        ds = prepare_seed_dataset(pool, cfg, int(seed))
        logger.info(
            "seed %s: source_train=%d source_val=%d windows=%s",
            seed,
            len(ds.source_train_y),
            len(ds.source_val_y),
            [(w["name"], len(w["y"]), len(w["unl_raw"])) for w in ds.windows],
        )
    logger.info("data stage finished in %.1fs", time.perf_counter() - t0)
    log_peak_memory("data")


def _load_or_build_datasets(cfg: Mapping):
    from armada.data.loader import prepare_seed_dataset

    pool = build_pool(cfg)
    return {int(s): prepare_seed_dataset(pool, cfg, int(s)) for s in cfg["seed_list"]}


def stage_train(cfg: Mapping) -> None:
    """Stage A supervised pre-training + Stage B adaptation, per seed.

    Stage B trains two variants: ``dann`` (naive single-discriminator DANN)
    and ``dual`` (dual-discriminator + reconstruction, TTT-ready).
    """
    from armada.train.adapt import train_stage_b
    from armada.train.pretrain import train_stage_a

    t0 = time.perf_counter()
    datasets = _load_or_build_datasets(cfg)
    ckpt_dir = Path(cfg["data"].get("checkpoints_dir", "checkpoints"))
    for seed, ds in datasets.items():
        # Paper outputs (checkpoints/curves) require real EMBER provenance.
        ds.provenance.assert_real("checkpoints/ and training curves")
        result_a = train_stage_a(ds, cfg, out_dir=ckpt_dir)
        logger.info(
            "seed %s Stage A: best val_auc=%.4f @ epoch %d (%d epochs)",
            seed,
            result_a.best_val_auc,
            result_a.best_epoch + 1,
            result_a.epochs_run,
        )
        for variant in ("dann", "dual"):
            result_b = train_stage_b(
                ds,
                cfg,
                variant=variant,
                out_dir=ckpt_dir,
                init_state=result_a.state_dict,
            )
            logger.info(
                "seed %s Stage B/%s: best val_auc=%.4f @ epoch %d (%d epochs)",
                seed,
                variant,
                result_b.best_val_auc,
                result_b.best_epoch + 1,
                result_b.epochs_run,
            )
    logger.info("train stage finished in %.1fs", time.perf_counter() - t0)
    log_peak_memory("train")


def stage_eval(cfg: Mapping) -> None:
    """Phase 1-2 scope: classical baselines + source-only grouped attention
    on the real EMBER windows.  DANN/TTT/memory variants join from Phase 3-5.
    """
    import pandas as pd

    from armada.eval.baselines import (
        ResultsWriter,
        available_baselines,
        run_baseline_seed,
        run_plain_mlp,
        summarise,
    )
    from armada.eval.neural import evaluate_armada_seed, evaluate_model_seed, evaluate_stage_b_seed

    t0 = time.perf_counter()
    results_dir = Path(cfg["data"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    save_config_used(cfg, results_dir)

    datasets = _load_or_build_datasets(cfg)
    models = available_baselines(cfg)
    logger.info("eval stage: baselines=%s", models)

    all_rows: List[Dict[str, object]] = []
    all_decision_rows: List[Dict[str, object]] = []
    for seed, ds in datasets.items():
        writer = ResultsWriter(results_dir, ds.provenance)
        windows = [(w["name"], w["raw"], w["y"]) for w in ds.windows if len(w["y"])]
        for name in models:
            rows = run_baseline_seed(
                name,
                seed,
                ds.source_train_raw,
                ds.source_train_y,
                windows,
                cfg,
                writer,
            )
            all_rows.extend(rows)
        if cfg.get("baselines", {}).get("plain_mlp_enabled", False):
            all_rows.extend(run_plain_mlp(ds, cfg, writer, seed))

        # Source-only grouped attention (Phase 2), when a checkpoint exists.
        ckpt = Path(cfg["data"].get("checkpoints_dir", "checkpoints")) / f"armada_stageA_seed{seed}.pt"
        if ckpt.exists():
            rows = evaluate_model_seed(
                method="GroupedAttn",
                checkpoint_path=ckpt,
                ds=ds,
                cfg=cfg,
                writer=writer,
            )
            all_rows.extend(rows)
        else:
            logger.warning(
                "no checkpoint %s — run --stage train first to include "
                "GroupedAttn rows in the metrics tables",
                ckpt,
            )

        # Stage B variants (Phase 3): DANN / dual-discriminator / +TTT.
        ckpt_dir = Path(cfg["data"].get("checkpoints_dir", "checkpoints"))
        ttt_cfg = cfg.get("ttt", {})
        plan = [
            ("dann", "DANN", False, False),
            ("dual", "DualDANN", False, False),
            ("dual", "DualDANN+TTT", True, False),
            ("dual", "DualDANN+TTT-online", True, True),
        ]
        for variant, method, use_ttt, online in plan:
            ckpt_b = ckpt_dir / f"armada_stageB_{variant}_seed{seed}.pt"
            if use_ttt and not ttt_cfg.get("enabled", True):
                continue
            if online and not ttt_cfg.get("report_online", True):
                continue
            if not ckpt_b.exists():
                logger.warning("no checkpoint %s — skipping %s", ckpt_b, method)
                continue
            rows = evaluate_stage_b_seed(
                method=method,
                checkpoint_path=ckpt_b,
                ds=ds,
                cfg=cfg,
                writer=writer,
                use_ttt=use_ttt,
                ttt_online=online,
            )
            all_rows.extend(rows)

        # Phase 5: full ARMADA (memory + learned calibrator + decision engine)
        # and its no-memory counterpart ("evaluate with and without memory").
        ckpt_dual = ckpt_dir / f"armada_stageB_dual_seed{seed}.pt"
        if ckpt_dual.exists():
            decision_rows: List[Dict[str, object]] = []
            for method, use_memory in (("ARMADA", True), ("ARMADA-noMem", False)):
                rows, dec = evaluate_armada_seed(
                    method=method,
                    checkpoint_path=ckpt_dual,
                    ds=ds,
                    cfg=cfg,
                    writer=writer,
                    use_memory=use_memory,
                    use_ttt=bool(cfg.get("ttt", {}).get("enabled", True)),
                )
                all_rows.extend(rows)
                decision_rows.extend(dec)
            all_decision_rows.extend(decision_rows)
        else:
            logger.warning(
                "no checkpoint %s — skipping ARMADA/ARMADA-noMem rows", ckpt_dual
            )

    if not all_rows:
        raise RuntimeError("eval produced no rows: check window configuration")

    by_window, overall = summarise(all_rows)
    # Raw per-seed rows also land in the CSV (git-friendly, recomputable).
    raw_df = pd.DataFrame(all_rows)
    raw_path = results_dir / "metrics_per_seed.csv"
    raw_df.to_csv(raw_path, index=False)

    win_df = pd.DataFrame(by_window)
    win_df.to_csv(results_dir / "metrics_by_window.csv", index=False)
    all_df = pd.DataFrame(overall)
    all_df.to_csv(results_dir / "metrics_all.csv", index=False)

    if all_decision_rows:
        dec_df = pd.DataFrame(all_decision_rows)
        dec_path = results_dir / "decision_report.csv"
        dec_df.to_csv(dec_path, index=False)
        logger.info("wrote %s (%d rows)", dec_path, len(dec_df))

    logger.info("wrote %s (%d rows)", raw_path, len(raw_df))
    logger.info("wrote %s (%d rows)", results_dir / "metrics_by_window.csv", len(win_df))
    logger.info("wrote %s (%d rows)", results_dir / "metrics_all.csv", len(all_df))
    for r in overall:
        logger.info(
            "OVERALL %s: acc=%.4f±%.4f auc_roc=%.4f±%.4f auc_pr=%.4f±%.4f f1=%.4f±%.4f",
            r["method"],
            r.get("accuracy_mean", float("nan")),
            r.get("accuracy_std", float("nan")),
            r.get("auc_roc_mean", float("nan")),
            r.get("auc_roc_std", float("nan")),
            r.get("auc_pr_mean", float("nan")),
            r.get("auc_pr_std", float("nan")),
            r.get("f1_mean", float("nan")),
            r.get("f1_std", float("nan")),
        )
    logger.info("eval stage finished in %.1fs", time.perf_counter() - t0)
    log_peak_memory("eval")


def stage_robust(cfg: Mapping) -> None:
    """Robustness evaluation: clean vs FGSM/PGD/noise at several epsilons
    (spec §3.6, Phase 4).  Feature-space attacks, EMBER-constrained."""
    import pandas as pd

    from armada.attacks import FeatureSpaceProjector  # noqa: F401  (documents intent)
    from armada.eval.baselines import ResultsWriter
    from armada.eval.neural import evaluate_stage_b_seed  # noqa: F401
    from armada.eval.robustness import evaluate_robustness_seed
    from armada.models.ttt import TTTAdapter
    from armada.train.adapt import load_stage_b
    from armada.train.pretrain import load_checkpoint
    from armada.utils import get_device

    t0 = time.perf_counter()
    results_dir = Path(cfg["data"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["data"].get("checkpoints_dir", "checkpoints"))
    device = get_device(cfg)

    datasets = _load_or_build_datasets(cfg)
    all_rows: List[Dict[str, object]] = []
    for seed, ds in datasets.items():
        writer = ResultsWriter(results_dir, ds.provenance)

        # 1) source-only grouped attention
        ckpt_a = ckpt_dir / f"armada_stageA_seed{seed}.pt"
        if ckpt_a.exists():
            model_a, _ = load_checkpoint(ckpt_a, device=device)
            all_rows.extend(
                evaluate_robustness_seed("GroupedAttn", model_a, ds, cfg, writer, device)
            )
        else:
            logger.warning("robust: missing %s — skipping GroupedAttn", ckpt_a)

        # 2) Stage B variants
        for variant, method, with_ttt in (
            ("dann", "DANN", False),
            ("dual", "DualDANN", False),
            ("dual", "DualDANN+TTT", True),
        ):
            ckpt_b = ckpt_dir / f"armada_stageB_{variant}_seed{seed}.pt"
            if not ckpt_b.exists():
                logger.warning("robust: missing %s — skipping %s", ckpt_b, method)
                continue
            model_b, _ = load_stage_b(ckpt_b, device=device)
            prepare_fn = None
            if with_ttt and cfg.get("ttt", {}).get("enabled", True):
                tcfg = cfg.get("ttt", {})
                adapter = TTTAdapter(
                    model_b,
                    lr=float(tcfg.get("lr", 1e-4)),
                    steps=int(tcfg.get("steps", 5)),
                    mask_ratio=float(tcfg.get("mask_ratio", 0.3)),
                    online=bool(tcfg.get("online", False)),
                )

                def prepare_fn(model, w, adapter=adapter, batch_size=None):  # noqa: E306
                    if not adapter.online:
                        adapter.reset()
                    unl = w.get("unl_proc") or {}
                    if unl and len(next(iter(unl.values()))):
                        adapter.adapt(unl, batch_size=int(cfg["train"].get("batch_size", 256)))

            all_rows.extend(
                evaluate_robustness_seed(
                    method, model_b, ds, cfg, writer, device, prepare_fn=prepare_fn
                )
            )

    if not all_rows:
        raise RuntimeError("robust stage produced no rows: train checkpoints first")

    df = pd.DataFrame(all_rows)
    out = results_dir / "robustness.csv"
    df.to_csv(out, index=False)
    logger.info("wrote %s (%d rows)", out, len(df))
    summary = (
        df.groupby(["method", "attack", "epsilon"], sort=True)[["accuracy", "auc_roc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    logger.info("robustness summary (mean over windows/seeds):\n%s", summary.to_string(index=False))
    logger.info("robust stage finished in %.1fs", time.perf_counter() - t0)
    log_peak_memory("robust")


def stage_ablate(cfg: Mapping) -> None:
    """Phase 6: training ablations + scale study + paired significance tests."""
    from armada.ablations import run_ablations, run_scale_study, run_significance

    t0 = time.perf_counter()
    results_dir = Path(cfg["data"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    save_config_used(cfg, results_dir)

    ab_cfg = cfg.get("ablation", {})
    run_ablations(cfg, results_dir)
    run_scale_study(cfg, results_dir)
    if ab_cfg.get("run_significance", True):
        try:
            run_significance(results_dir, cfg)
        except FileNotFoundError as exc:
            logger.warning("significance skipped: %s", exc)
    logger.info("ablate stage finished in %.1fs", time.perf_counter() - t0)
    log_peak_memory("ablate")


def stage_unavailable(cfg: Mapping, stage: str) -> None:
    raise NotImplementedError(
        f"--stage {stage} is delivered in {PHASE_OF_STAGE.get(stage, 'a later phase')}. "
        "Phases are reviewed incrementally; run --stage data / --stage eval so far."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="armada.run", description=__doc__)
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    parser.add_argument("--stage", type=str, default="all", choices=STAGES)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use configs/fast.yaml (small caps, smoke run, NOT for paper numbers)",
    )
    args = parser.parse_args(argv)

    cfg_path = args.config
    if args.fast:
        cfg_path = "configs/fast.yaml"
        logger_note = "FAST mode: using configs/fast.yaml (results are smoke-only)"
    else:
        logger_note = None

    try:
        cfg = load_config(cfg_path)
    except FileNotFoundError:
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2

    logs_dir = Path(cfg["data"].get("logs_dir", "logs"))
    setup_logging(logs_dir, args.stage)
    if logger_note:
        logger.warning(logger_note)
    logger.info("config: %s | stage: %s | cwd: %s", cfg_path, args.stage, os.getcwd())
    logger.info("device: cuda_available=%s (config=%s)", _cuda_available(), cfg.get("device", "auto"))

    stages = ["data", "train", "eval", "ablate", "robust", "figures"] if args.stage == "all" else [args.stage]
    for stage in stages:
        try:
            if stage == "data":
                stage_data(cfg)
            elif stage == "train":
                stage_train(cfg)
            elif stage == "eval":
                stage_eval(cfg)
            elif stage == "ablate":
                stage_ablate(cfg)
            elif stage == "robust":
                stage_robust(cfg)
            else:
                stage_unavailable(cfg, stage)
        except MemoryError as exc:
            logger.error("MEMORY ERROR: %s", exc)
            return 3
        except FileNotFoundError as exc:
            logger.error("DATA ERROR: %s", exc)
            return 4
        except NotImplementedError as exc:
            if args.stage == "all":
                logger.warning("stage '%s' skipped for now: %s", stage, exc)
                continue
            logger.error("%s", exc)
            return 5
    logger.info("all requested stages complete")
    return 0


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


if __name__ == "__main__":
    raise SystemExit(main())
