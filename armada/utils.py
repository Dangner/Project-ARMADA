"""Small shared helpers: seeding, device selection, parameter counting."""

from __future__ import annotations

import logging
import os
import random
from typing import Mapping

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch for reproducible runs (no seed cherry-picking)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN (only matters when CUDA is present).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - torch optional for pure-sklearn stages
        pass


def get_device(cfg: Mapping):
    import torch

    want = str(cfg.get("device", "auto")).lower()
    if want == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config requests cuda but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if want == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def log_environment() -> None:
    logger.info("python=%s pid=%d cwd=%s", os.sys.version.split()[0], os.getpid(), os.getcwd())
    try:
        import torch

        logger.info(
            "torch=%s cuda_available=%s device_count=%d",
            torch.__version__,
            torch.cuda.is_available(),
            torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )
    except Exception:  # pragma: no cover
        logger.info("torch not importable")
