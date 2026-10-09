"""Decision engine — SAFE / SUSPICIOUS / MALWARE (spec §3.8).

Two score thresholds are chosen on the **source validation split only**:

- ``malware_threshold``: smallest score threshold whose FPR on source-val is
  at most ``malware_fpr_target`` (default 0.1 %) — verdict MALWARE;
- ``suspicious_threshold``: the same criterion at ``suspicious_fpr_target``
  (default 5 %) — verdict SUSPICIOUS between the two thresholds ("sandbox"
  bucket), SAFE below.

No hardcoded 30/60-style cutoffs: thresholds are data-derived and recorded
in every report row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SAFE = "SAFE"
SUSPICIOUS = "SUSPICIOUS"
MALWARE = "MALWARE"
VERDICTS = (SAFE, SUSPICIOUS, MALWARE)


def threshold_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    """Smallest threshold with empirical FPR <= ``fpr_target`` on this data.

    Equals the score at the sweep point of :func:`armada.eval.metrics.tpr_at_fpr`
    restricted to (y_true, scores) — i.e. the most permissive decision rule
    meeting the false-positive budget.
    """
    from ..eval.metrics import tpr_at_fpr

    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) == 0:
        raise ValueError("threshold_at_fpr: empty source validation split")
    if (y == 0).sum() == 0:
        raise ValueError("threshold_at_fpr: no benign rows in the source validation split")
    r = tpr_at_fpr(y, s, fpr_target)
    t = float(r["threshold"])
    if not np.isfinite(t):  # single-class edge case
        t = float(np.max(s)) + 1.0
    return t


@dataclass
class DecisionEngine:
    """Three-way verdicts from calibrated malware scores."""

    malware_fpr_target: float = 0.001
    suspicious_fpr_target: float = 0.05
    malware_threshold: Optional[float] = None
    suspicious_threshold: Optional[float] = None

    def fit(self, y_val: np.ndarray, scores_val: np.ndarray) -> "DecisionEngine":
        """Choose both thresholds from the SOURCE validation split only."""
        y = np.asarray(y_val)
        s = np.asarray(scores_val, dtype=np.float64)
        self.malware_threshold = threshold_at_fpr(y, s, self.malware_fpr_target)
        self.suspicious_threshold = threshold_at_fpr(y, s, self.suspicious_fpr_target)
        if self.suspicious_threshold > self.malware_threshold:
            # small-sample edge: keep ordering MALWARE > SUSPICIOUS
            self.suspicious_threshold = self.malware_threshold
        logger.info(
            "DecisionEngine thresholds from source-val (n=%d): MALWARE>=%.6f (FPR<=%g), "
            "SUSPICIOUS>=%.6f (FPR<=%g)",
            len(y),
            self.malware_threshold,
            self.malware_fpr_target,
            self.suspicious_threshold,
            self.suspicious_fpr_target,
        )
        return self

    def verdicts(self, scores: np.ndarray) -> np.ndarray:
        if self.malware_threshold is None or self.suspicious_threshold is None:
            raise RuntimeError("DecisionEngine.fit must run on the source validation split first")
        s = np.asarray(scores, dtype=np.float64)
        out = np.full(s.shape, SAFE, dtype=object)
        out[s >= self.suspicious_threshold] = SUSPICIOUS
        out[s >= self.malware_threshold] = MALWARE
        return out

    def report(self, y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Confusion + routing summary for one evaluated window.

        - ``conf_*``: MALWARE-as-positive 2-class confusion counts;
        - ``alert_conf_*``: (MALWARE or SUSPICIOUS)-as-positive confusion;
        - ``sandbox_rate``: proportion routed to SUSPICIOUS.
        """
        y = np.asarray(y_true).astype(np.int64)
        s = np.asarray(scores, dtype=np.float64)
        v = self.verdicts(s)
        n = len(y)
        pred_mal = v == MALWARE
        pred_alert = (v == MALWARE) | (v == SUSPICIOUS)
        pos = y == 1

        out: Dict[str, float] = {
            "n": float(n),
            "n_safe": float((v == SAFE).sum()),
            "n_suspicious": float((v == SUSPICIOUS).sum()),
            "n_malware": float((v == MALWARE).sum()),
            "sandbox_rate": float((v == SUSPICIOUS).sum() / max(1, n)),
            "conf_tp": float((pred_mal & pos).sum()),
            "conf_fp": float((pred_mal & ~pos).sum()),
            "conf_tn": float((~pred_mal & ~pos).sum()),
            "conf_fn": float((~pred_mal & pos).sum()),
            "alert_conf_tp": float((pred_alert & pos).sum()),
            "alert_conf_fp": float((pred_alert & ~pos).sum()),
            "alert_conf_tn": float((~pred_alert & ~pos).sum()),
            "alert_conf_fn": float((~pred_alert & pos).sum()),
            "malware_threshold": float(self.malware_threshold),
            "suspicious_threshold": float(self.suspicious_threshold),
        }
        return out
