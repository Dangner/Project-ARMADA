"""Vectorise the official EMBER 2018 dump into aligned .dat artifacts.

Wraps the official `ember` toolkit (feature_version=2):

- reads the raw JSONL corpus with ``ember.read_vectorized_features``;
- writes ``X_train.dat`` / ``y_train.dat`` / ``X_test.dat`` / ``y_test.dat``
  (float32 raw dumps, exactly as the official vectorised release);
- writes ``metadata.csv`` via ``ember.create_metadata`` so row alignment is
  the official one (each ``X_*.dat`` row matches the metadata row order).

Usage
-----
    python -m armada.data.vectorize --data_dir data/ember2018 --with-test

The official `ember` package must be installed (``pip install git+https://
github.com/elastic/ember.git``); this module never fabricates feature rows.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def vectorize(data_dir: Path | str, with_test: bool = True) -> None:
    """Create the aligned vectorised artifacts from the raw EMBER corpus."""
    data_dir = Path(data_dir)
    try:
        import ember  # official toolkit
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError(
            "the official 'ember' package is required to vectorise the raw "
            "corpus: pip install git+https://github.com/elastic/ember.git"
        ) from exc

    logger.info("reading raw EMBER corpus in %s (feature_version=2)", data_dir)
    if with_test:
        X_train, y_train, X_test, y_test = ember.read_vectorized_features(
            str(data_dir), feature_version=2
        )
    else:
        X_train, y_train = ember.read_vectorized_features(str(data_dir), feature_version=2)[:2]
        X_test = y_test = None

    for name, arr in (("X_train", X_train), ("y_train", y_train), ("X_test", X_test), ("y_test", y_test)):
        if arr is None:
            continue
        out = data_dir / f"{name}.dat"
        np.asarray(arr, dtype=np.float32).tofile(out)
        logger.info("wrote %s (%s %s)", out, arr.shape, arr.dtype)

    # Official row alignment: metadata.csv rows == .dat rows, in order.
    ember.create_metadata(str(data_dir))
    logger.info("wrote %s", data_dir / "metadata.csv")
    logger.info("done — artifacts are ready for `python -m armada.run --stage data`")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="armada.data.vectorize", description=__doc__)
    parser.add_argument("--data_dir", type=str, required=True, help="EMBER 2018 corpus directory")
    parser.add_argument("--with-test", action="store_true", help="also vectorise the test subset")
    args = parser.parse_args(argv)
    try:
        vectorize(args.data_dir, with_test=args.with_test)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
