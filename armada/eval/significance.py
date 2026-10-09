"""Paired significance tests (spec §3.9).

Exact sign-flip permutation test on paired per-seed differences — no extra
dependencies, exact for the small n (3 seeds) used in the paper.  Two-sided
p-value: fraction of sign assignments whose |mean difference| is at least the
observed one.  With n pairs the smallest possible p is 2/2^n; this low power
at n=3 is reported honestly in the paper.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def exact_sign_flip_pvalue(diffs: np.ndarray) -> Tuple[float, float]:
    """Return ``(mean_diff, two-sided p)`` for paired differences ``diffs``."""
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    if n == 0:
        raise ValueError("exact_sign_flip_pvalue: no paired observations")
    obs = float(d.mean())
    if n > 16:  # Monte-Carlo fallback for large n (not expected here)
        rng = np.random.default_rng(0)
        signs = rng.choice([-1.0, 1.0], size=(4096, n))
        stats = (signs * d).mean(axis=1)
        p = float((np.abs(stats) >= abs(obs) - 1e-12).mean())
        return obs, p
    # exact enumeration of all 2^n sign assignments
    idx = np.arange(n)
    extreme = 0
    total = 2**n
    for mask in range(total):
        signs = np.where((mask >> idx) & 1, 1.0, -1.0)
        if abs((signs * d).mean()) >= abs(obs) - 1e-12:
            extreme += 1
    return obs, float(extreme / total)


def paired_method_comparison(
    per_seed_a: Iterable[float],
    per_seed_b: Iterable[float],
) -> Dict[str, float]:
    """Compare two methods over paired seeds (a − b)."""
    a = np.asarray(list(per_seed_a), dtype=np.float64)
    b = np.asarray(list(per_seed_b), dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("paired_method_comparison: unpaired inputs")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 2:
        return {
            "n_pairs": float(len(a)),
            "mean_diff": float("nan"),
            "p_value": float("nan"),
        }
    diffs = a - b
    mean_diff, p = exact_sign_flip_pvalue(diffs)
    return {"n_pairs": float(len(diffs)), "mean_diff": mean_diff, "p_value": p}
