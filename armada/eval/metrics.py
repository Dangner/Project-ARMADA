"""Evaluation metrics for malware detection under concept drift.

Everything reported in the paper is computed here from saved predictions —
no metric is ever hand-entered.  Suite (per spec §5):

Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR (average precision),
TPR at fixed FPR (0.1% and 1%), and expected calibration error (ECE).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: TPR@FPR operating points reported in the paper.
FIXED_FPRS: Tuple[float, ...] = (0.001, 0.01)

N_ECE_BINS: int = 10


def _check_binary(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true/y_score length mismatch")
    if y_true.shape[0] == 0:
        raise ValueError("empty evaluation set")
    vals = set(np.unique(y_true).tolist())
    if not vals.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"y_true must be binary 0/1, got values {sorted(vals)}")
    return y_true.astype(np.int64), y_score


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_target: float) -> Dict[str, float]:
    """Largest TPR achievable at FPR <= ``fpr_target`` (score-threshold sweep)."""
    y_true, y_score = _check_binary(y_true, y_score)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {"tpr": float("nan"), "threshold": float("nan"), "fpr": float("nan")}
    order = np.argsort(-y_score, kind="stable")
    yt = y_true[order]
    tp = np.cumsum(yt == 1)
    fp = np.cumsum(yt == 0)
    fpr = fp / n_neg
    tpr = tp / n_pos
    ok = np.flatnonzero(fpr <= fpr_target)
    if len(ok) == 0:
        return {"tpr": 0.0, "threshold": float(y_score[order][0]), "fpr": float(fpr[0])}
    best = ok[-1]
    return {
        "tpr": float(tpr[best]),
        "threshold": float(y_score[order][best]),
        "fpr": float(fpr[best]),
    }


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_ECE_BINS
) -> float:
    """Standard equal-width ECE on the malware probability in [0, 1]."""
    y_true, y_prob = _check_binary(y_true, y_prob)
    if y_prob.min() < -1e-6 or y_prob.max() > 1 + 1e-6:
        raise ValueError("y_prob must lie in [0, 1]")
    y_prob = np.clip(y_prob, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not in_bin.any():
            continue
        conf = y_prob[in_bin].mean()
        acc = y_true[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(acc - conf)
    return float(ece)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    fixed_fprs: Sequence[float] = FIXED_FPRS,
) -> Dict[str, float]:
    """Full paper metric suite for one (window, method, seed) cell."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true, y_prob = _check_binary(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(np.int64)

    def _safe(fn, *a, **kw) -> float:
        try:
            return float(fn(*a, **kw))
        except ValueError:
            return float("nan")

    out: Dict[str, float] = {
        "accuracy": _safe(accuracy_score, y_true, y_pred),
        "precision": _safe(precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe(recall_score, y_true, y_pred, zero_division=0),
        "f1": _safe(f1_score, y_true, y_pred, zero_division=0),
        "auc_roc": _safe(roc_auc_score, y_true, y_prob),
        "auc_pr": _safe(average_precision_score, y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_malware": int((y_true == 1).sum()),
        "n_benign": int((y_true == 0).sum()),
    }
    for fpr_t in fixed_fprs:
        r = tpr_at_fpr(y_true, y_prob, fpr_t)
        key = f"tpr_at_fpr_{fpr_t:g}".replace(".", "p")
        out[key] = r["tpr"]
        out[f"{key}_threshold"] = r["threshold"]
    return out


def aggregate_mean_std(
    rows: Iterable[Mapping[str, object]],
    group_keys: Sequence[str] = ("method", "window"),
    metric_keys: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    """Mean ± std across seeds (rows must carry a 'seed' column).

    Returns one row per group with ``<metric>_mean`` / ``<metric>_std`` and
    the number of seeds aggregated.
    """
    rows = list(rows)
    if not rows:
        return []
    if metric_keys is None:
        skip = set(group_keys) | {"seed", "n", "n_malware", "n_benign"}
        metric_keys = sorted(
            k for k in rows[0] if k not in skip and isinstance(rows[0][k], (int, float, np.floating, np.integer))
        )
    groups: Dict[Tuple, List[Mapping[str, object]]] = {}
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        groups.setdefault(key, []).append(r)

    agg: List[Dict[str, object]] = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        members = groups[key]
        row: Dict[str, object] = {k: v for k, v in zip(group_keys, key)}
        row["n_seeds"] = len(members)
        for m in metric_keys:
            vals = np.array([float(mm[m]) for mm in members if mm.get(m) is not None], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                row[f"{m}_mean"] = float(finite.mean())
                row[f"{m}_std"] = float(finite.std(ddof=0))
            else:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
        agg.append(row)
    return agg
