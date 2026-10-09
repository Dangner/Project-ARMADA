"""Classical baselines trained on the same source rows as ARMADA.

Baselines (spec §5): Logistic Regression, Random Forest, Gradient Boosting
(sklearn HistGradientBoosting; XGBoost used automatically when installed).
All models see the same stratified source subsample and the same target
windows as ARMADA; nothing is fit on target labels.

Every reported number is computed from saved predictions via
:mod:`armada.eval.metrics`; writers refuse synthetic provenance.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..data.loader import Provenance, SyntheticDataError
from .metrics import aggregate_mean_std, compute_metrics

logger = logging.getLogger(__name__)

# Baseline names for the paper tables.
CLASSICAL_BASELINES: Tuple[str, ...] = ("LR", "RF", "GBDT")
OPTIONAL_BASELINES: Tuple[str, ...] = ("XGBoost",)


def make_model(name: str, seed: int, cfg: Mapping):
    """Construct an unfitted baseline estimator from the config."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    bcfg = cfg.get("baselines", {})
    if name == "LR":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=int(bcfg.get("lr_max_iter", 1000)),
                C=float(bcfg.get("lr_C", 1.0)),
                class_weight=bcfg.get("lr_class_weight", "balanced"),
                random_state=seed,
            ),
        )
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=int(bcfg.get("rf_n_estimators", 200)),
            max_depth=bcfg.get("rf_max_depth", None),
            min_samples_leaf=int(bcfg.get("rf_min_samples_leaf", 2)),
            n_jobs=int(bcfg.get("rf_n_jobs", -1)),
            class_weight=bcfg.get("rf_class_weight", "balanced_subsample"),
            random_state=seed,
        )
    if name == "GBDT":
        return HistGradientBoostingClassifier(
            max_iter=int(bcfg.get("gbdt_max_iter", 200)),
            learning_rate=float(bcfg.get("gbdt_learning_rate", 0.1)),
            max_leaf_nodes=int(bcfg.get("gbdt_max_leaf_nodes", 31)),
            early_stopping=bool(bcfg.get("gbdt_early_stopping", True)),
            validation_fraction=float(bcfg.get("gbdt_validation_fraction", 0.1)),
            random_state=seed,
        )
    if name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError:  # pragma: no cover - optional dependency
            raise ImportError("xgboost is not installed")
        return XGBClassifier(
            n_estimators=int(bcfg.get("xgb_n_estimators", 200)),
            max_depth=int(bcfg.get("xgb_max_depth", 6)),
            learning_rate=float(bcfg.get("xgb_learning_rate", 0.1)),
            subsample=float(bcfg.get("xgb_subsample", 0.9)),
            colsample_bytree=float(bcfg.get("xgb_colsample_bytree", 0.9)),
            n_jobs=int(bcfg.get("xgb_n_jobs", -1)),
            random_state=seed,
            eval_metric="auc",
        )
    raise ValueError(f"unknown baseline '{name}'")


def available_baselines(cfg: Mapping) -> List[str]:
    names = list(cfg.get("baselines", {}).get("models", CLASSICAL_BASELINES))
    out = []
    for n in names:
        if n in OPTIONAL_BASELINES:
            try:
                make_model(n, 0, cfg)
            except ImportError:
                logger.warning("baseline %s skipped: xgboost not installed", n)
                continue
        out.append(n)
    return out


# ---------------------------------------------------------------------------
# Guarded results I/O: synthetic data may never touch results/
# ---------------------------------------------------------------------------


@dataclass
class ResultsWriter:
    """Writes predictions/metrics under ``results/`` with a provenance guard."""

    results_dir: Path
    provenance: Provenance

    def __post_init__(self) -> None:
        self.provenance.assert_real("results/")
        self.results_dir = Path(self.results_dir)
        (self.results_dir / "predictions").mkdir(parents=True, exist_ok=True)

    def save_predictions(
        self,
        method: str,
        seed: int,
        window: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Path:
        self.provenance.assert_real(f"predictions for {method}/{window}")
        path = self.results_dir / "predictions" / f"{method}_seed{seed}_{window}.npz"
        np.savez_compressed(
            path,
            y_true=np.asarray(y_true),
            y_prob=np.asarray(y_prob, dtype=np.float64),
            method=np.str_(method),
            seed=np.int64(seed),
            window=np.str_(window),
            provenance=np.str_(self.provenance.kind),
        )
        return path


def metrics_from_saved_predictions(
    pred_path: Path, threshold: float = 0.5
) -> Dict[str, float]:
    """Recompute the metric suite from a saved predictions file (paper rule)."""
    with np.load(pred_path, allow_pickle=False) as data:
        if str(data["provenance"]) != "ember":
            raise SyntheticDataError(
                f"{pred_path} was produced from synthetic data; refusing to "
                "derive reported metrics from it."
            )
        y_true, y_prob = data["y_true"], data["y_prob"]
    return compute_metrics(y_true, y_prob, threshold=threshold)


# ---------------------------------------------------------------------------
# Training / evaluation loop for one seed
# ---------------------------------------------------------------------------


def run_baseline_seed(
    name: str,
    seed: int,
    X_src: np.ndarray,
    y_src: np.ndarray,
    windows: Sequence[Tuple[str, np.ndarray, np.ndarray]],  # (name, X, y)
    cfg: Mapping,
    writer: ResultsWriter,
) -> List[Dict[str, object]]:
    """Fit one baseline on source rows and evaluate on every target window."""
    model = make_model(name, seed, cfg)
    t0 = time.perf_counter()
    logger.info("[%s seed=%d] fitting on %d source rows...", name, seed, len(y_src))
    model.fit(X_src, y_src)
    fit_s = time.perf_counter() - t0
    logger.info("[%s seed=%d] fit done in %.1fs", name, seed, fit_s)

    rows: List[Dict[str, object]] = []
    for win_name, X_win, y_win in windows:
        if len(y_win) == 0:
            logger.warning("[%s] window %s has no labeled rows; skipped", name, win_name)
            continue
        y_prob = model.predict_proba(X_win)[:, 1]
        writer.save_predictions(name, seed, win_name, y_win, y_prob)
        metrics = metrics_from_saved_predictions(
            writer.results_dir / "predictions" / f"{name}_seed{seed}_{win_name}.npz"
        )
        row: Dict[str, object] = {
            "method": name,
            "seed": seed,
            "window": win_name,
            "fit_seconds": round(fit_s, 2),
        }
        row.update(metrics)
        rows.append(row)
        logger.info(
            "[%s seed=%d %s] acc=%.4f auc_roc=%.4f auc_pr=%.4f f1=%.4f",
            name,
            seed,
            win_name,
            metrics["accuracy"],
            metrics["auc_roc"],
            metrics["auc_pr"],
            metrics["f1"],
        )
    return rows


def _feature_matrix(proc: Mapping[str, np.ndarray], group_dims: Mapping[str, int]) -> np.ndarray:
    """Flatten processed groups in deterministic (sorted) order."""
    names = sorted(group_dims)
    return np.concatenate([np.asarray(proc[n], dtype=np.float32) for n in names], axis=1)


def run_plain_mlp(
    ds: "SeedDataset",
    cfg: Mapping,
    writer: "ResultsWriter",
    seed: int,
) -> List[Dict[str, object]]:
    """Plain MLP baseline on concatenated standardised features (spec §3.4)."""
    from sklearn.neural_network import MLPClassifier

    b = cfg.get("baselines", {})
    if not b.get("plain_mlp_enabled", False):
        raise RuntimeError("run_plain_mlp: baselines.plain_mlp_enabled is false")
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(b.get("plain_mlp_hidden", [256, 128])),
        activation="relu",
        max_iter=int(b.get("plain_mlp_max_iter", 300)),
        early_stopping=True,
        n_iter_no_change=10,
        batch_size=min(256, max(32, int(cfg["train"].get("batch_size", 256)))),
        random_state=seed,
        learning_rate_init=float(b.get("plain_mlp_lr", 1e-3)),
    )
    ds.ensure_processed()
    names = sorted(ds.source_train_proc)
    X = np.concatenate([ds.source_train_proc[n] for n in names], axis=1)
    y = ds.source_train_y
    t0 = time.perf_counter()
    clf.fit(X, y)
    fit_s = time.perf_counter() - t0
    logger.info("PlainMLP seed=%d fitted on %s in %.1fs", seed, X.shape, fit_s)

    rows: List[Dict[str, object]] = []
    for w in ds.windows:
        y_true = np.asarray(w["y"])
        if len(y_true) == 0:
            continue
        Xw = np.concatenate([w["proc"][n] for n in names], axis=1)
        y_prob = clf.predict_proba(Xw)[:, list(clf.classes_).index(1)]
        writer.save_predictions("PlainMLP", seed, w["name"], y_true, y_prob)
        m = metrics_from_saved_predictions(
            writer.results_dir / "predictions" / f"PlainMLP_seed{seed}_{w['name']}.npz"
        )
        row: Dict[str, object] = {
            "method": "PlainMLP",
            "seed": seed,
            "window": w["name"],
            "fit_seconds": fit_s,
            "ttt_steps": 0,
        }
        row.update(m)
        rows.append(row)
        logger.info(
            "PlainMLP seed=%d %s acc=%.4f auc_roc=%.4f", seed, w["name"], m["accuracy"], m["auc_roc"]
        )
    return rows


def summarise(rows: Sequence[Mapping[str, object]]) -> Tuple[List[Dict], List[Dict]]:
    """Per-window mean±std over seeds and overall (windows averaged) mean±std."""
    by_window = aggregate_mean_std(rows, group_keys=("method", "window"))
    # Overall: average each metric across windows within a seed, then mean±std.
    per_seed: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    for r in rows:
        key = (str(r["method"]), int(r["seed"]))
        acc = per_seed.setdefault(key, {})
        for k, v in r.items():
            if isinstance(v, (int, float, np.floating, np.integer)) and k not in (
                "seed",
                "n",
                "n_malware",
                "n_benign",
                "fit_seconds",
            ):
                acc.setdefault(k, []).append(float(v))
    overall_rows: List[Dict[str, object]] = []
    methods = sorted({k[0] for k in per_seed})
    for method in methods:
        seeds = sorted(s for m, s in per_seed if m == method)
        agg: Dict[str, object] = {"method": method, "n_seeds": len(seeds), "n_windows": 0}
        metric_names: List[str] = []
        for s in seeds:
            metric_names = list(per_seed[(method, s)])
        for m in metric_names:
            vals = [float(np.mean(per_seed[(method, s)][m])) for s in seeds]
            agg[f"{m}_mean"] = float(np.mean(vals))
            agg[f"{m}_std"] = float(np.std(vals, ddof=0))
        if seeds:
            agg["n_windows"] = len(per_seed[(method, seeds[0])][metric_names[0]])
        overall_rows.append(agg)
    return by_window, overall_rows
