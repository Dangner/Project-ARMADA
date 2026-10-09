"""Temporal concept-drift split (spec §2).

Rows are ordered by ``appeared`` month.  The earliest months form the SOURCE
period (training + validation); later months form chronological target
windows T1..Tk evaluated in time order.  No row ever appears in more than
one split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def unique_months(months: Sequence[str]) -> List[str]:
    """Sorted unique month labels (chronological)."""
    return sorted(set(str(m) for m in months))


@dataclass
class Window:
    """One target evaluation window."""

    name: str
    months: Tuple[str, ...]
    labeled_idx: np.ndarray
    unlabeled_idx: np.ndarray


@dataclass
class DriftSplit:
    """Source period + chronological target windows over row indices."""

    source_months: Tuple[str, ...]
    source_train_idx: np.ndarray
    source_val_idx: np.ndarray
    windows: List[Window]
    month_counts: List[Dict[str, object]] = field(default_factory=list)


def _month_counts(months: np.ndarray, y: np.ndarray) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for m in unique_months(months):
        sel = months == m
        yy = y[sel]
        rows.append(
            {
                "month": m,
                "n": int(sel.sum()),
                "malware": int((yy == 1).sum()),
                "benign": int((yy == 0).sum()),
                "unlabeled": int((yy == -1).sum()),
            }
        )
    return rows


def build_drift_split(
    months: Sequence[str],
    y: Sequence[float],
    split_cfg: Mapping,
    rng: np.random.Generator,
) -> DriftSplit:
    """Build the temporal split from row-level ``appeared`` months and labels.

    Parameters
    ----------
    months, y:
        Per-row month label and EMBER label (1 malware, 0 benign, -1 unlabeled).
    split_cfg:
        ``mode: auto|explicit``; auto uses ``source_fraction`` of the months
        for the source period and spreads the remaining months over
        ``n_target_windows`` windows (earlier windows take the remainder).
        ``val_fraction`` of the SOURCE labelled rows form the validation set.
    rng:
        Seeded generator (validation subsample).
    """
    months = np.asarray([str(m) for m in months])
    y = np.asarray(y, dtype=np.float64)
    if len(months) != len(y):
        raise ValueError("months and y must have the same length")
    mode = str(split_cfg.get("mode", "auto"))

    if mode == "auto":
        all_months = unique_months(months)
        if len(all_months) < 2:
            raise ValueError("not enough distinct months for a temporal split")
        frac = float(split_cfg.get("source_fraction", 0.5))
        n_source = int(round(frac * len(all_months)))
        if n_source < 1:
            n_source = 1
        if n_source >= len(all_months):
            raise ValueError(
                "empty target domain: source_fraction leaves no later months for evaluation"
            )
        source_months = tuple(all_months[:n_source])
        target_months = all_months[n_source:]
        if not target_months:
            raise ValueError("empty target domain: leave later months for evaluation")
        k = int(split_cfg.get("n_target_windows", 2))
        if k < 1:
            raise ValueError("n_target_windows must be >= 1")
        # Spread target months over k windows; earlier windows take remainder.
        base, extra = divmod(len(target_months), k)
        windows_months: List[Tuple[str, ...]] = []
        pos = 0
        for i in range(k):
            size = base + (1 if i < extra else 0)
            if size == 0:
                continue
            windows_months.append(tuple(target_months[pos : pos + size]))
            pos += size
    elif mode == "explicit":
        source_months = tuple(str(m) for m in split_cfg.get("source_months", ()))
        raw_windows = split_cfg.get("target_windows", ())
        windows_months = [tuple(str(m) for m in w) for w in raw_windows]
        if not source_months:
            raise ValueError("explicit mode needs a non-empty source_months list")
        if not windows_months:
            raise ValueError("empty target domain: target_windows is empty")
    else:
        raise ValueError(f"unknown split.mode {mode!r} (expected 'auto' or 'explicit')")

    src_mask = np.isin(months, list(source_months))
    labeled = y != -1

    # --- source train / val from SOURCE labelled rows only ---
    src_labeled = np.where(src_mask & labeled)[0]
    val_frac = float(split_cfg.get("val_fraction", 0.1))
    n_val = int(round(val_frac * len(src_labeled)))
    if n_val > 0 and len(src_labeled) > 0:
        perm = rng.permutation(len(src_labeled))
        val_idx = np.sort(src_labeled[perm[:n_val]])
        train_idx = np.sort(src_labeled[perm[n_val:]])
    else:
        val_idx = np.array([], dtype=np.int64)
        train_idx = np.sort(src_labeled)

    # --- target windows ---
    windows: List[Window] = []
    for i, w_months in enumerate(windows_months):
        w_mask = np.isin(months, list(w_months))
        labeled_idx = np.where(w_mask & labeled)[0]
        unlabeled_idx = np.where(w_mask & ~labeled)[0]
        windows.append(
            Window(
                name=f"T{i + 1}",
                months=tuple(w_months),
                labeled_idx=labeled_idx,
                unlabeled_idx=unlabeled_idx,
            )
        )

    logger.info(
        "drift split: source=%s (%d train / %d val rows), windows=%s",
        ",".join(source_months),
        len(train_idx),
        len(val_idx),
        "; ".join(f"{w.name}:{','.join(w.months)}" for w in windows),
    )
    return DriftSplit(
        source_months=tuple(source_months),
        source_train_idx=train_idx.astype(np.int64),
        source_val_idx=val_idx.astype(np.int64),
        windows=windows,
        month_counts=_month_counts(months, y),
    )
