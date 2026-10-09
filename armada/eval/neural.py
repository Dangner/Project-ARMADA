"""Neural-model evaluation protocol (source-only for now).

Loads a Stage A checkpoint and evaluates it on every target window through
the exact same prediction/metrics pipeline as the classical baselines
(saved predictions -> recomputed metrics -> guarded results writer).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from ..data.loader import SeedDataset
from ..train.pretrain import load_checkpoint
from .baselines import ResultsWriter, metrics_from_saved_predictions

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict_proba_windows(
    model,
    windows: Sequence[Mapping],
    batch_size: int,
    device: torch.device,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Per-window (name, y_true, y_prob) from processed group blocks."""
    model.eval()
    out = []
    for w in windows:
        if len(w["y"]) == 0:
            continue
        proc = w["proc"]
        probs = []
        n = len(w["y"])
        for start in range(0, n, batch_size):
            sl = slice(start, start + batch_size)
            groups = {
                name: torch.as_tensor(block[sl], dtype=torch.float32, device=device)
                for name, block in proc.items()
            }
            p, _ = model.predict_proba(groups)
            probs.append(p.cpu().numpy())
        out.append((w["name"], np.asarray(w["y"]), np.concatenate(probs)))
    return out


def evaluate_model_seed(
    method: str,
    checkpoint_path: Path,
    ds: SeedDataset,
    cfg: Mapping,
    writer: ResultsWriter,
) -> List[Dict[str, object]]:
    """Evaluate one checkpoint on all target windows of one seed dataset."""
    from ..utils import get_device

    device = get_device(cfg)
    model, ckpt = load_checkpoint(Path(checkpoint_path), device=device)
    batch_size = int(cfg["train"].get("batch_size", 256))
    ds.ensure_processed()

    rows: List[Dict[str, object]] = []
    for win_name, y_true, y_prob in predict_proba_windows(model, ds.windows, batch_size, device):
        writer.save_predictions(method, ds.seed, win_name, y_true, y_prob)
        metrics = metrics_from_saved_predictions(
            writer.results_dir / "predictions" / f"{method}_seed{ds.seed}_{win_name}.npz"
        )
        row: Dict[str, object] = {
            "method": method,
            "seed": ds.seed,
            "window": win_name,
            "fit_seconds": float("nan"),
        }
        row.update(metrics)
        rows.append(row)
        logger.info(
            "[%s seed=%d %s] acc=%.4f auc_roc=%.4f auc_pr=%.4f f1=%.4f",
            method,
            ds.seed,
            win_name,
            metrics["accuracy"],
            metrics["auc_roc"],
            metrics["auc_pr"],
            metrics["f1"],
        )
    return rows


def evaluate_stage_b_seed(
    method: str,
    checkpoint_path: Path,
    ds: SeedDataset,
    cfg: Mapping,
    writer: ResultsWriter,
    use_ttt: bool = False,
    ttt_online: bool = False,
) -> List[Dict[str, object]]:
    """Evaluate a Stage B checkpoint, optionally with per-window TTT.

    TTT protocol (spec §3.5): before each target window the model resets to
    the source-trained weights (unless ``ttt_online``), runs ``ttt.steps``
    reconstruction steps on that window's UNLABELED rows only, then predicts
    the window's labeled rows.  The classifier is never updated by TTT.
    """
    from ..models.ttt import TTTAdapter
    from ..train.adapt import load_stage_b
    from ..utils import get_device

    device = get_device(cfg)
    model, ckpt = load_stage_b(Path(checkpoint_path), device=device)
    batch_size = int(cfg["train"].get("batch_size", 256))
    ds.ensure_processed()

    tcfg = cfg.get("ttt", {})
    adapter = None
    if use_ttt:
        adapter = TTTAdapter(
            model,
            lr=float(tcfg.get("lr", 1e-4)),
            steps=int(tcfg.get("steps", 5)),
            mask_ratio=float(tcfg.get("mask_ratio", 0.3)),
            online=bool(ttt_online),
        )

    rows: List[Dict[str, object]] = []
    for w in ds.windows:
        y_true = np.asarray(w["y"])
        if len(y_true) == 0:
            continue
        ttt_steps_run = 0
        if adapter is not None:
            if not adapter.online:
                adapter.reset()
            unl = w.get("unl_proc") or {}
            losses = adapter.adapt(unl, batch_size=batch_size) if unl else []
            ttt_steps_run = len(losses)

        proc = w["proc"]
        probs = []
        n = len(y_true)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                sl = slice(start, start + batch_size)
                groups = {
                    name: torch.as_tensor(block[sl], dtype=torch.float32, device=device)
                    for name, block in proc.items()
                }
                p, _ = model.predict_proba(groups)
                probs.append(p.cpu().numpy())
        y_prob = np.concatenate(probs)

        writer.save_predictions(method, ds.seed, w["name"], y_true, y_prob)
        metrics = metrics_from_saved_predictions(
            writer.results_dir / "predictions" / f"{method}_seed{ds.seed}_{w['name']}.npz"
        )
        row: Dict[str, object] = {
            "method": method,
            "seed": ds.seed,
            "window": w["name"],
            "fit_seconds": float("nan"),
            "ttt_steps": ttt_steps_run,
        }
        row.update(metrics)
        rows.append(row)
        logger.info(
            "[%s seed=%d %s] acc=%.4f auc_roc=%.4f auc_pr=%.4f f1=%.4f (ttt_steps=%d)",
            method,
            ds.seed,
            w["name"],
            metrics["accuracy"],
            metrics["auc_roc"],
            metrics["auc_pr"],
            metrics["f1"],
            ttt_steps_run,
        )
    return rows
