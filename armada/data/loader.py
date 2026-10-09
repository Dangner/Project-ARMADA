"""EMBER 2018 loading, caps, preprocessing and caching.

Real-data rule: every array carries a :class:`Provenance`.  The official
EMBER vectorised dumps (``X_train.dat`` / ``y_train.dat`` / ``X_test.dat`` /
``y_test.dat`` / ``metadata.csv``) are read via memmap and checked 1:1
against the metadata rows exactly like the official toolkit.

No synthetic arrays are ever produced here — dummy tensors live in
``tests/conftest.py`` only and are tagged :data:`SYNTHETIC`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .groups import GROUPS, LOG_TRANSFORM_GROUPS, SENTINEL_VALUE, TOTAL_DIM, split_vector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provenance / synthetic-data guard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """Where an array came from — ``kind`` must be ``"ember"`` for real data."""

    kind: str
    data_dir: Optional[str] = None
    details: Optional[str] = None

    @property
    def is_real(self) -> bool:
        return self.kind == "ember"

    def assert_real(self, what: str = "results") -> None:
        if not self.is_real:
            raise SyntheticDataError(
                f"refusing to write {what}: dataset provenance is '{self.kind}' "
                f"({self.details or 'no details'}). Only real EMBER data may be "
                "used for reported results. Synthetic tensors belong in tests/ "
                "and must never touch results/."
            )


class SyntheticDataError(RuntimeError):
    """Raised when synthetic data would reach results/."""


SYNTHETIC = Provenance(kind="synthetic", details="explicitly marked synthetic")
REAL = Provenance(kind="ember")


# ---------------------------------------------------------------------------
# Memory guards
# ---------------------------------------------------------------------------


def estimate_bytes(n_rows: int, n_cols: int) -> int:
    """float32 bytes for an ``(n_rows, n_cols)`` block."""
    return int(n_rows) * int(n_cols) * 4


#: Conservative working budget so peak RSS stays below the 8 GB machine cap.
BUDGET_BYTES = 4 * 1024**3


def check_memory(n_rows: int, n_cols: int, budget_bytes: int = BUDGET_BYTES) -> None:
    """Raise a helpful :class:`MemoryError` before an allocation explodes."""
    need = estimate_bytes(n_rows, n_cols)
    if need > budget_bytes:
        raise MemoryError(
            f"estimated {need / 1e9:.1f} GB for a ({n_rows}, {n_cols}) float32 block "
            f"exceeds the {budget_bytes / 1e9:.1f} GB working budget. "
            "Reduce n_train / n_per_window in the config, or use smaller subset caps."
        )


# ---------------------------------------------------------------------------
# Artifact discovery + raw loading
# ---------------------------------------------------------------------------


def require_ember_artifacts(data_dir: Path | str, with_test: bool = False) -> Path:
    """Check the official EMBER vectorised artifacts exist."""
    data_dir = Path(data_dir)
    required = ["X_train.dat", "y_train.dat", "metadata.csv"]
    if with_test:
        required += ["X_test.dat", "y_test.dat"]
    for name in required:
        if not (data_dir / name).exists():
            raise FileNotFoundError(
                f"EMBER 2018 artifacts missing in {data_dir}: expected "
                f"{', '.join(required)} (e.g. X_train.dat). "
                "Vectorise the official dump first: "
                "python -m armada.data.vectorize --data_dir data/ember2018 --with-test"
            )
    return data_dir


@dataclass
class EmberSplit:
    """One aligned (X, y, appeared) block from the official dump."""

    X: np.ndarray  # (n, TOTAL_DIM) float32 — memmap-backed for the real dump
    y: np.ndarray  # (n,) float32 in {-1, 0, 1}
    appeared: np.ndarray  # (n,) month strings
    provenance: Provenance

    def __len__(self) -> int:
        return int(self.y.shape[0])


def load_metadata(data_dir: Path | str):
    """``metadata.csv`` (sha256, appeared, label, subset) as a DataFrame."""
    import pandas as pd

    path = Path(data_dir) / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"required EMBER artifact missing: {path}")
    return pd.read_csv(path, index_col=0)


def _load_split(data_dir: Path, split: str) -> EmberSplit:
    x_path = data_dir / f"X_{split}.dat"
    y_path = data_dir / f"y_{split}.dat"
    X = np.memmap(x_path, dtype=np.float32, mode="r").reshape(-1, TOTAL_DIM)
    y = np.asarray(np.memmap(y_path, dtype=np.float32, mode="r"))
    meta = load_metadata(data_dir)
    rows = meta[meta["subset"] == split]
    if len(rows) != X.shape[0] or len(rows) != y.shape[0]:
        raise ValueError(
            f"metadata rows and .dat rows must align 1:1 for subset '{split}': "
            f"metadata={len(rows)}, X={X.shape[0]}, y={y.shape[0]}"
        )
    appeared = rows["appeared"].astype(str).to_numpy()
    return EmberSplit(
        X=X,
        y=y,
        appeared=appeared,
        provenance=Provenance(kind="ember", data_dir=str(data_dir), details=f"X_{split}.dat"),
    )


def load_ember_train(data_dir: Path | str) -> EmberSplit:
    """Load ``X_train.dat`` aligned with metadata rows for subset ``train``."""
    return _load_split(Path(data_dir), "train")


def load_ember_test(data_dir: Path | str) -> EmberSplit:
    """Load ``X_test.dat`` aligned with metadata rows for subset ``test``."""
    return _load_split(Path(data_dir), "test")


class EmberPool:
    """Concatenated view over one or more :class:`EmberSplit` blocks."""

    def __init__(self, splits: Sequence[EmberSplit]):
        if not splits:
            raise ValueError("EmberPool needs at least one split")
        self.splits: List[EmberSplit] = list(splits)
        sizes = [len(s) for s in self.splits]
        self._offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self._y = np.concatenate([np.asarray(s.y) for s in self.splits])
        self._appeared = np.concatenate([np.asarray(s.appeared) for s in self.splits])
        self.provenance = self.splits[0].provenance

    def __len__(self) -> int:
        return int(self._offsets[-1])

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def appeared(self) -> np.ndarray:
        return self._appeared

    def _map(self, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= len(self)):
            raise IndexError("pool index out of range")
        split_ids = np.searchsorted(self._offsets, idx, side="right") - 1
        local = idx - self._offsets[split_ids]
        return split_ids, local

    def gather(self, idx: Sequence[int] | np.ndarray) -> np.ndarray:
        """Rows of ``X`` for global indices, in the requested order."""
        split_ids, local = self._map(np.asarray(idx))
        out = np.empty((len(local), TOTAL_DIM), dtype=np.float32)
        for sid in np.unique(split_ids):
            sel = split_ids == sid
            X = self.splits[int(sid)].X
            out[sel] = np.asarray(X[local[sel]], dtype=np.float32)
        return out

    def gather_labels(self, idx: Sequence[int] | np.ndarray) -> np.ndarray:
        return self._y[np.asarray(idx, dtype=np.int64)]

    def gather_months(self, idx: Sequence[int] | np.ndarray) -> np.ndarray:
        return self._appeared[np.asarray(idx, dtype=np.int64)]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def config_hash(cfg: Mapping) -> str:
    """Stable short hash of a nested config mapping (sort-key JSON)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:10]


def cache_dir_for(cfg: Mapping, cache_root: Path | str) -> Path:
    return Path(cache_root) / f"pre_{config_hash(cfg)}"


def save_cached_arrays(cache_dir: Path | str, arrays: Mapping[str, np.ndarray], meta: Mapping) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(cache_dir / f"{name}.npy", arr)
    (cache_dir / "meta.json").write_text(json.dumps(dict(meta), default=str, indent=1))
    return cache_dir


def load_cached_arrays(cache_dir: Path | str) -> Optional[Dict[str, np.ndarray]]:
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return None
    out: Dict[str, np.ndarray] = {}
    for path in sorted(cache_dir.glob("*.npy")):
        out[path.stem] = np.load(path)
    return out or None


# ---------------------------------------------------------------------------
# Scaling / preprocessing (source-only)
# ---------------------------------------------------------------------------


class GroupScaler:
    """Per-group standard scaler with optional log1p and sentinel handling.

    ``transform`` returns the scaled values concatenated with a missingness
    mask channel (``1`` where the raw value is the EMBER sentinel ``-1``),
    so the processed width is exactly ``2 * input_width``.
    """

    def __init__(self, log_transform: bool = False):
        self.log_transform = bool(log_transform)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.zero_var_: Optional[np.ndarray] = None
        self.n_fit_rows_: int = 0

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        V = X.copy()
        V[X == SENTINEL_VALUE] = np.nan
        if self.log_transform:
            V = np.log1p(np.clip(V, 0.0, None))
        return V

    def fit(self, X: np.ndarray) -> "GroupScaler":
        V = self._prepare(X)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(V, axis=0) if V.shape[0] else np.zeros(V.shape[1], np.float32)
            std = np.nanstd(V, axis=0) if V.shape[0] else np.ones(V.shape[1], np.float32)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std), std, 1.0)
        # zero-variance columns map to 0 via (x - mean) / 1
        self.zero_var_ = ~(std > 0)  # frozen features: no signal to perturb
        std = np.where(std > 0, std, 1.0)
        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)
        self.n_fit_rows_ = int(X.shape[0])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("GroupScaler.transform called before fit")
        X = np.asarray(X, dtype=np.float32)
        mask = (X == SENTINEL_VALUE).astype(np.float32)
        V = self._prepare(X)
        V = np.where(np.isnan(V), 0.0, V)
        scaled = (V - self.mean_) / self.std_
        scaled = np.where(mask == 1.0, 0.0, scaled)  # sentinel -> 0
        return np.concatenate([scaled.astype(np.float32), mask], axis=1)

    def to_dict(self) -> Dict[str, object]:
        """JSON-safe snapshot for checkpoints (provenance / deployment)."""
        return {
            "log_transform": self.log_transform,
            "mean_": np.asarray(self.mean_, dtype=float).tolist(),
            "std_": np.asarray(self.std_, dtype=float).tolist(),
            "zero_var_": np.asarray(self.zero_var_, dtype=bool).tolist(),
            "n_fit_rows_": self.n_fit_rows_,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "GroupScaler":
        sc = cls(log_transform=bool(payload.get("log_transform", False)))
        sc.mean_ = np.asarray(payload["mean_"], dtype=np.float32)
        sc.std_ = np.asarray(payload["std_"], dtype=np.float32)
        sc.zero_var_ = np.asarray(payload["zero_var_"], dtype=bool)
        sc.n_fit_rows_ = int(payload.get("n_fit_rows_", 0))
        return sc


class SourceOnlyPreprocessor:
    """Group-wise preprocessing fit on SOURCE rows only.

    ``transform`` yields one block per feature group; each block carries the
    scaled values followed by a full-width missingness mask (2x raw width).
    """

    def __init__(self) -> None:
        self.scalers: Dict[str, GroupScaler] = {}
        self.processed_dims: Dict[str, int] = {}
        self.n_fit_rows_: int = 0

    def fit(self, X_raw: np.ndarray) -> "SourceOnlyPreprocessor":
        check_memory(*X_raw.shape[:2])
        groups = split_vector(X_raw)
        self.scalers = {}
        self.processed_dims = {}
        for g in GROUPS:
            sc = GroupScaler(log_transform=g.name in LOG_TRANSFORM_GROUPS)
            sc.fit(groups[g.name])
            self.scalers[g.name] = sc
            self.processed_dims[g.name] = 2 * g.dim
        self.n_fit_rows_ = int(X_raw.shape[0])
        return self

    def transform(self, X_raw: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.scalers:
            raise RuntimeError("SourceOnlyPreprocessor.transform called before fit")
        groups = split_vector(X_raw)
        return {g.name: self.scalers[g.name].transform(groups[g.name]) for g in GROUPS}

    @property
    def processed_total_dim(self) -> int:
        return int(sum(self.processed_dims.values()))


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------


def stratified_cap_indices(
    y: Sequence[float], cap: int, rng: np.random.Generator
) -> np.ndarray:
    """Deterministic proportional stratified sample of at most ``cap`` rows."""
    y = np.asarray(y)
    n = len(y)
    if cap <= 0:
        return np.array([], dtype=np.int64)
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    classes, counts = np.unique(y, return_counts=True)
    quotas = counts / counts.sum() * cap
    base = np.floor(quotas).astype(int)
    # distribute the remainder to the largest fractional parts
    remainder = cap - int(base.sum())
    order = np.argsort(-(quotas - base), kind="stable")
    for i in range(remainder):
        base[order[i % len(order)]] += 1
    # at least one row per class when possible
    for i in range(len(base)):
        if base[i] == 0 and counts[i] > 0 and cap >= len(classes):
            take_from = int(np.argmax(base))
            if base[take_from] > 1:
                base[take_from] -= 1
                base[i] = 1
    chosen: List[np.ndarray] = []
    for c, k in zip(classes, base):
        idx = np.where(y == c)[0]
        if k <= 0:
            continue
        pick = rng.choice(idx, size=min(k, len(idx)), replace=False)
        chosen.append(pick)
    return np.sort(np.concatenate(chosen)).astype(np.int64)


# ---------------------------------------------------------------------------
# Seed dataset
# ---------------------------------------------------------------------------


@dataclass
class SeedDataset:
    """Everything one training/evaluation seed needs, raw + processed."""

    seed: int
    provenance: Provenance
    preprocessor: SourceOnlyPreprocessor
    source_train_raw: np.ndarray
    source_train_y: np.ndarray
    source_val_raw: np.ndarray
    source_val_y: np.ndarray
    windows: list  # list of dict(name, months, raw, y, unl_raw, unl_proc, proc)
    source_train_proc: Optional[Dict[str, np.ndarray]] = None
    source_val_proc: Optional[Dict[str, np.ndarray]] = None

    def ensure_processed(self) -> None:
        if self.source_train_proc is None:
            self.source_train_proc = self.preprocessor.transform(self.source_train_raw)
        if self.source_val_proc is None:
            self.source_val_proc = self.preprocessor.transform(self.source_val_raw)
        for w in self.windows:
            if w.get("proc") is None:
                w["proc"] = self.preprocessor.transform(w["raw"])
            if w.get("unl_proc") is None and len(w["unl_raw"]):
                w["unl_proc"] = self.preprocessor.transform(w["unl_raw"])
            elif w.get("unl_proc") is None:
                w["unl_proc"] = {}


def prepare_seed_dataset(
    pool: EmberPool,
    cfg: Mapping,
    seed: int,
    use_cache: bool = True,
) -> SeedDataset:
    """Build the capped, drift-split, source-normalised dataset for one seed.

    Sequence (no leakage):
    1. temporal drift split on months/labels (source = earliest months);
    2. caps: ``n_train`` / ``n_val`` on source labelled rows,
       ``n_per_window`` / ``n_unlabeled_per_window`` per target window;
    3. preprocessor fit on source TRAIN rows only;
    4. processed blocks are lazy (``ensure_processed``) and cached per config.
    """
    from .drift_split import build_drift_split

    data_cfg = cfg["data"]
    cache_key = dict(cfg)
    cache_key["seed"] = int(seed)
    cache_dir = cache_dir_for(cache_key, data_cfg.get("cache_dir", "cache"))

    arrays: Optional[Dict[str, np.ndarray]] = None
    meta: Optional[dict] = None
    if use_cache:
        arrays = load_cached_arrays(cache_dir)

    rng = np.random.default_rng(int(seed))
    if arrays is None:
        check_memory(len(pool), TOTAL_DIM)
        months = pool.appeared
        y_all = pool.y
        split = build_drift_split(months, y_all, cfg["split"], rng=rng)

        n_train = int(data_cfg.get("n_train", 20000))
        n_val = int(data_cfg.get("n_val", 2000))
        n_per_window = int(data_cfg.get("n_per_window", 2000))
        n_unl = int(data_cfg.get("n_unlabeled_per_window", 2000))

        tr_idx = split.source_train_idx
        if len(tr_idx) > n_train:
            tr_idx = tr_idx[stratified_cap_indices(y_all[tr_idx], n_train, rng)]
        va_idx = split.source_val_idx
        if len(va_idx) > n_val:
            va_idx = va_idx[stratified_cap_indices(y_all[va_idx], n_val, rng)]

        arrays = {
            "source_train_raw": pool.gather(tr_idx),
            "source_train_y": (y_all[tr_idx] == 1).astype(np.int64),
            "source_val_raw": pool.gather(va_idx),
            "source_val_y": (y_all[va_idx] == 1).astype(np.int64),
        }
        win_meta = []
        for w in split.windows:
            lab = w.labeled_idx
            if len(lab) > n_per_window:
                lab = lab[stratified_cap_indices(y_all[lab], n_per_window, rng)]
            unl = w.unlabeled_idx
            if len(unl) > n_unl:
                unl = unl[stratified_cap_indices(y_all[unl], n_unl, rng)]
            arrays[f"{w.name}_raw"] = pool.gather(lab)
            arrays[f"{w.name}_y"] = (y_all[lab] == 1).astype(np.int64)
            arrays[f"{w.name}_unl_raw"] = pool.gather(unl)
            win_meta.append({"name": w.name, "months": list(w.months)})
        meta = {
            "seed": int(seed),
            "config_hash": config_hash(cache_key),
            "windows": win_meta,
            "source_months": list(split.source_months),
        }
        if use_cache:
            save_cached_arrays(cache_dir, arrays, meta)
    elif meta is None:
        meta_path = Path(cache_dir) / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:  # cache without meta — recover window names from array keys
            names = sorted(
                k[:-4] for k in arrays if k.endswith("_raw") and not k.endswith("_unl_raw") and k.startswith("T")
            )
            meta = {
                "seed": int(seed),
                "windows": [{"name": n, "months": []} for n in names],
            }

    pre = SourceOnlyPreprocessor().fit(arrays["source_train_raw"])
    windows = []
    for w_meta in meta["windows"]:
        name = w_meta["name"]
        windows.append(
            {
                "name": name,
                "months": tuple(w_meta.get("months", ())),
                "raw": arrays[f"{name}_raw"],
                "y": arrays[f"{name}_y"],
                "unl_raw": arrays[f"{name}_unl_raw"],
                "proc": None,
                "unl_proc": None,
            }
        )

    provenance = Provenance(
        kind="ember",
        data_dir=pool.provenance.data_dir,
        details=f"seed={int(seed)} hash={config_hash(cache_key)}",
    )
    ds = SeedDataset(
        seed=int(seed),
        provenance=provenance,
        preprocessor=pre,
        source_train_raw=arrays["source_train_raw"],
        source_train_y=arrays["source_train_y"],
        source_val_raw=arrays["source_val_raw"],
        source_val_y=arrays["source_val_y"],
        windows=windows,
    )
    ds.ensure_processed()
    return ds
