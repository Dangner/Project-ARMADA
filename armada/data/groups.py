"""EMBER 2018 feature-group schema (feature_version=2, 2381 dims).

The 2381-dim EMBER feature vector is a concatenation of nine named,
contiguous groups.  Every module in ARMADA refers to groups through the
slice table defined here so the boundaries can never silently drift.

References
----------
- Anderson & Roth, "EMBER: An Open Dataset for Training Static PE Malware
  Machine Learning Models", arXiv:1804.04637, 2018.
- Official vectoriser: https://github.com/elastic/ember (ember/features.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

#: Total dimensionality of an EMBER 2018 (feature_version=2) vector.
TOTAL_DIM: int = 2381

#: Sentinel used by EMBER for "feature not present / not measured".
SENTINEL_VALUE: float = -1.0


@dataclass(frozen=True)
class FeatureGroup:
    """One named, contiguous slice of the EMBER vector."""

    name: str
    start: int
    stop: int
    description: str

    @property
    def dim(self) -> int:
        return self.stop - self.start

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


#: Ordered group table.  Order and boundaries exactly match the official
#: EMBER 2018 vectoriser (feature_version=2).
GROUPS: Tuple[FeatureGroup, ...] = (
    FeatureGroup("ByteHistogram", 0, 256, "byte-value histogram (256 bins)"),
    FeatureGroup("ByteEntropyHistogram", 256, 512, "2D byte/entropy histogram (16x16)"),
    FeatureGroup("StringExtractor", 512, 616, "printable-string statistics"),
    FeatureGroup("GeneralFileInfo", 616, 626, "size, entropy, has_* indicators"),
    FeatureGroup("HeaderFileInfo", 626, 688, "PE header / COFF fields"),
    FeatureGroup("SectionInfo", 688, 943, "section names, sizes, entropy stats"),
    FeatureGroup("ImportsInfo", 943, 2223, "hashed import names (1280 buckets)"),
    FeatureGroup("ExportsInfo", 2223, 2351, "hashed export names (128 buckets)"),
    FeatureGroup("DataDirectories", 2351, 2381, "15 data-directory size/virtual-address pairs"),
)

GROUP_NAMES: Tuple[str, ...] = tuple(g.name for g in GROUPS)

#: Groups whose raw values are counts/histograms: log1p is applied before
#: standardisation (kept as behaviour of the original Armada_train_eval.py).
LOG_TRANSFORM_GROUPS = frozenset(
    {"ByteHistogram", "ByteEntropyHistogram", "StringExtractor", "ImportsInfo", "ExportsInfo"}
)

# Module import time guard: the schema must always describe 2381 contiguous
# dims with no gaps or overlaps.
_slices = [(g.start, g.stop) for g in GROUPS]
assert _slices[0][0] == 0, "first group must start at 0"
assert _slices[-1][1] == TOTAL_DIM, "last group must end at TOTAL_DIM"
assert all(_slices[i][1] == _slices[i + 1][0] for i in range(len(_slices) - 1)), (
    "group slices must be contiguous with no gaps/overlaps"
)
assert sum(g.dim for g in GROUPS) == TOTAL_DIM, "group dims must sum to 2381"


def group_slices() -> Dict[str, slice]:
    """Mapping ``name -> slice(0..2381)`` for indexing raw feature matrices."""
    return {g.name: g.slice for g in GROUPS}


def group_dims() -> Dict[str, int]:
    """Mapping ``name -> dimensionality``."""
    return {g.name: g.dim for g in GROUPS}


def split_vector(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Split a ``(n, 2381)`` raw matrix into the nine named group blocks.

    Parameters
    ----------
    X:
        Array of shape ``(n, TOTAL_DIM)`` (or ``(TOTAL_DIM,)`` for one sample).

    Returns
    -------
    dict of ``name -> (n, dim)`` arrays (views where possible).
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[1] != TOTAL_DIM:
        raise ValueError(f"expected shape (n, {TOTAL_DIM}), got {X.shape}")
    return {g.name: np.ascontiguousarray(X[:, g.slice]) for g in GROUPS}


def verify_slice_boundaries() -> List[Tuple[str, int, int, int]]:
    """Return the (name, start, stop, dim) table after re-checking invariants.

    Used by unit tests and by the data loader as an explicit runtime check.
    """
    rows: List[Tuple[str, int, int, int]] = []
    for g in GROUPS:
        rows.append((g.name, g.start, g.stop, g.dim))
    assert sum(r[3] for r in rows) == TOTAL_DIM
    return rows
