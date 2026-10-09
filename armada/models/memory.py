"""Immune Memory — threat memory bank (spec §3.7).

Fixed-size bank of embeddings ``z`` of confirmed malware (and optionally
benign samples) with cosine nearest-neighbour lookup (numpy; no FAISS).
At inference the max cosine similarity to *malware* entries is combined with
the classifier probability by a **learned** calibrator
(:class:`MemoryCalibrator`, logistic regression fit on the SOURCE validation
split — never hand-set weights).

Update rule: high-confidence detections from a target window are added;
entries are evicted by age (``max_age_windows``) or least-recently-used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryBank:
    """Cosine-similarity memory of (embedding, label, age, use) entries."""

    capacity: int = 4096
    store_benign: bool = True
    evict: str = "age"  # "age" | "lru"
    max_age_windows: int = 3

    def __post_init__(self) -> None:
        self._z: List[np.ndarray] = []
        self._labels: List[int] = []
        self._age: List[int] = []
        self._last_used: List[int] = []
        self._tick = 0
        self._use_clock = 0
        self.n_added = 0
        self.n_evicted = 0

    def __len__(self) -> int:
        return len(self._z)

    @property
    def n_malware(self) -> int:
        return sum(1 for l in self._labels if l == 1)

    @property
    def n_benign(self) -> int:
        return sum(1 for l in self._labels if l == 0)

    def add(self, z: np.ndarray, labels: np.ndarray) -> int:
        """Add embeddings with labels (1 = malware, 0 = benign).

        Benign entries are only stored when ``store_benign`` is set.  Returns
        the number of entries actually inserted.
        """
        z = np.atleast_2d(np.asarray(z, dtype=np.float32))
        labels = np.atleast_1d(np.asarray(labels, dtype=np.int64))
        if z.shape[0] != labels.shape[0]:
            raise ValueError("z/labels length mismatch")
        added = 0
        for i in range(z.shape[0]):
            lab = int(labels[i])
            if lab == 0 and not self.store_benign:
                continue
            self._z.append(z[i].copy())
            self._labels.append(lab)
            self._age.append(0)
            self._last_used.append(self._tick)
            added += 1
        self.n_added += added
        self._enforce_capacity()
        return added

    def _malware_matrix(self) -> Optional[np.ndarray]:
        rows = [zz for zz, lab in zip(self._z, self._labels) if lab == 1]
        if not rows:
            return None
        return np.stack(rows)

    def max_malware_similarity(self, z: np.ndarray) -> np.ndarray:
        """Max cosine similarity of each query to MALWARE entries (0 if empty)."""
        z = np.atleast_2d(np.asarray(z, dtype=np.float32))
        M = self._malware_matrix()
        if M is None or len(self) == 0:
            return np.zeros(z.shape[0], dtype=np.float32)
        q = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-8, None)
        M_n = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-8, None)
        sims = q @ M_n.T  # (n_query, n_malware)
        best = sims.argmax(axis=1)
        # record usage for LRU eviction (dedicated usage clock)
        self._use_clock += 1
        mal_indices = [i for i, lab in enumerate(self._labels) if lab == 1]
        for b in np.unique(best):
            self._last_used[mal_indices[int(b)]] = self._use_clock
        return sims.max(axis=1).astype(np.float32)

    def step_window(self) -> None:
        """Advance the age clock at the end of a target window (for age eviction)."""
        self._tick += 1
        if self.evict == "age":
            keep = [i for i, a in enumerate(self._age) if a < self.max_age_windows]
            self.n_evicted += len(self._z) - len(keep)
            self._z = [self._z[i] for i in keep]
            self._labels = [self._labels[i] for i in keep]
            self._age = [self._age[i] + 1 for i in keep]
            self._last_used = [self._last_used[i] for i in keep]
        else:
            self._age = [a + 1 for a in self._age]

    def update_from_detections(
        self,
        z: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        label: int = 1,
    ) -> int:
        """Add high-confidence detections (score >= threshold) as memory entries."""
        z = np.atleast_2d(np.asarray(z, dtype=np.float32))
        scores = np.atleast_1d(np.asarray(scores, dtype=np.float64))
        keep = scores >= float(threshold)
        if not keep.any():
            return 0
        return self.add(z[keep], np.full(int(keep.sum()), label, dtype=np.int64))

    def _enforce_capacity(self) -> None:
        while len(self._z) > self.capacity:
            if self.evict == "lru":
                drop = int(np.argmin(self._last_used))
            else:
                # oldest (highest age) first; ties -> earliest inserted
                drop = int(np.argmax(self._age))
            self._z.pop(drop)
            self._labels.pop(drop)
            self._age.pop(drop)
            self._last_used.pop(drop)
            self.n_evicted += 1


@dataclass
class MemoryCalibrator:
    """Learned combination of classifier probability and memory similarity.

    Logistic regression on ``[p_malware, max_sim]`` (or ``[p_malware]`` when
    ``use_memory=False``), fit on the SOURCE validation split only.  The
    output is the deployed malware score consumed by the decision engine.
    """

    use_memory: bool = True

    def __post_init__(self) -> None:
        self.model = None  # sklearn LogisticRegression, created at fit

    def _features(self, probs: np.ndarray, sims: np.ndarray) -> np.ndarray:
        probs = np.atleast_1d(np.asarray(probs, dtype=np.float64))
        if not self.use_memory:
            return probs.reshape(-1, 1)
        sims = np.atleast_1d(np.asarray(sims, dtype=np.float64))
        return np.stack([probs, sims], axis=1)

    def fit(self, probs: np.ndarray, sims: np.ndarray, y: np.ndarray) -> "MemoryCalibrator":
        from sklearn.linear_model import LogisticRegression

        X = self._features(probs, sims)
        y = np.atleast_1d(np.asarray(y, dtype=np.int64))
        if len(np.unique(y)) < 2:
            raise ValueError("calibrator fit needs both classes in the source validation split")
        self.model = LogisticRegression(max_iter=1000, random_state=0)
        self.model.fit(X, y)
        logger.info(
            "MemoryCalibrator fit on %d source-val rows (use_memory=%s) coef=%s intercept=%.4f",
            len(y),
            self.use_memory,
            None if self.model.coef_ is None else self.model.coef_.round(4).tolist(),
            float(self.model.intercept_[0]),
        )
        return self

    def score(self, probs: np.ndarray, sims: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MemoryCalibrator.score called before fit")
        return self.model.predict_proba(self._features(probs, sims))[:, 1]
