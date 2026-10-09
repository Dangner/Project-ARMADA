"""Stage A — supervised source pre-training (spec §4).

Trains encoder + classifier on the labeled SOURCE period only, with AdamW,
cosine/one-cycle LR schedule, gradient clipping, optional AMP on CUDA, early
stopping on the source validation split, checkpointing and per-component
training curves.  Target windows are never touched here.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from ..data.loader import Provenance, SeedDataset
from ..models.core import ArmadaCore
from ..models.heads import build_pos_weight, classification_loss
from ..utils import count_parameters, get_device, set_seed

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Everything Stage A produces."""

    best_epoch: int
    best_val_auc: float
    epochs_run: int
    curves: List[Dict[str, float]] = field(default_factory=list)
    checkpoint_path: Optional[Path] = None
    state_dict: Optional[Dict[str, torch.Tensor]] = None


class _GroupBatcher:
    """Iterates (processed group dict, labels) mini-batches with a seeded order."""

    def __init__(
        self,
        proc: Dict[str, np.ndarray],
        y: np.ndarray,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.proc = proc
        self.y = torch.as_tensor(y, dtype=torch.float32, device=device)
        self.n = len(y)
        self.batch_size = min(batch_size, max(1, self.n))
        self.device = device
        self.names = list(proc.keys())

    def epoch_batches(self, rng: torch.Generator):
        perm = torch.randperm(self.n, generator=rng).numpy()
        for start in range(0, self.n, self.batch_size):
            idx = perm[start : start + self.batch_size]
            groups = {
                name: torch.as_tensor(self.proc[name][idx], dtype=torch.float32, device=self.device)
                for name in self.names
            }
            yield groups, self.y[torch.as_tensor(idx, device=self.device)]


@torch.no_grad()
def evaluate_classifier(
    model: ArmadaCore,
    proc: Dict[str, np.ndarray],
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """Validation metrics on a labeled set (source val / any eval split)."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    probs: List[np.ndarray] = []
    n = len(y)
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        groups = {
            name: torch.as_tensor(block[sl], dtype=torch.float32, device=device)
            for name, block in proc.items()
        }
        p, _ = model.predict_proba(groups)
        probs.append(p.cpu().numpy())
    y_prob = np.concatenate(probs) if probs else np.zeros(0)
    y_true = np.asarray(y)
    out: Dict[str, float] = {"n": float(n)}
    if len(y_true) and len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["auc"] = float("nan")
    out["bce"] = float(
        torch.nn.functional.binary_cross_entropy(
            torch.as_tensor(y_prob, dtype=torch.float32),
            torch.as_tensor(y_true, dtype=torch.float32),
        ).item()
    ) if n else float("nan")
    return out


def _build_scheduler(optimizer, cfg: Mapping, total_steps: int):
    kind = str(cfg.get("scheduler", "cosine")).lower()
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    if kind == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=float(cfg.get("lr", 1e-4)), total_steps=max(1, total_steps)
        )
    return None


def train_stage_a(
    ds: SeedDataset,
    cfg: Mapping,
    out_dir: Optional[Path] = None,
    model: Optional[ArmadaCore] = None,
) -> TrainResult:
    """Supervised pre-training of profiler + encoder + classifier on source rows.

    ``ds`` must carry source train/val blocks (processed with source-only
    stats).  When ``out_dir`` is given, writes ``armada_stageA_seed<seed>.pt``
    and ``training_curves_seed<seed>.csv`` (paper artifacts; caller is
    responsible for provenance guarantees on real runs — tests pass tmp dirs).
    """
    tcfg = cfg["train"]
    set_seed(ds.seed)
    device = get_device(cfg)
    use_amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
    logger.info(
        "Stage A: seed=%d device=%s amp=%s train_rows=%d val_rows=%d",
        ds.seed,
        device,
        use_amp,
        len(ds.source_train_y),
        len(ds.source_val_y),
    )

    ds.ensure_processed()
    if model is None:
        model = ArmadaCore.from_config(cfg, ds.preprocessor.processed_dims).to(device)
    else:
        model = model.to(device)
    logger.info("model parameters: %d", count_parameters(model))

    pos_weight = build_pos_weight(tcfg, ds.source_train_y)
    if pos_weight is not None:
        logger.info("class weights: balanced pos_weight=%.3f (from source labels)", pos_weight)

    params = model.parameters()
    optimizer = torch.optim.AdamW(
        params,
        lr=float(tcfg.get("lr", 1e-4)),
        weight_decay=float(tcfg.get("weight_decay", 1e-5)),
    )
    epochs = int(tcfg.get("stage_a_epochs", 15))
    batch_size = int(tcfg.get("batch_size", 256))
    total_steps = max(1, (len(ds.source_train_y) // max(1, batch_size)) * epochs)
    scheduler = _build_scheduler(optimizer, tcfg, total_steps)
    grad_clip = float(tcfg.get("grad_clip", 5.0))
    patience = int(tcfg.get("early_stopping_patience", 5))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    batcher = _GroupBatcher(ds.source_train_proc, ds.source_train_y, batch_size, device)
    rng = torch.Generator(device="cpu").manual_seed(ds.seed)

    curves: List[Dict[str, float]] = []
    best_val_auc = -float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0

    from tqdm import tqdm

    for epoch in tqdm(range(epochs), desc=f"Stage A (seed {ds.seed})", leave=False):
        model.train()
        t0 = time.perf_counter()
        total_loss = 0.0
        n_batches = 0
        for groups, y_batch in batcher.epoch_batches(rng):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _ = model(groups)
                loss = classification_loss(logits, y_batch, pos_weight=pos_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "non-finite training loss; try lowering train.lr or batch size"
                )
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and isinstance(
                scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                scheduler.step()
            total_loss += float(loss.item())
            n_batches += 1
        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.OneCycleLR
        ):
            scheduler.step()

        val = evaluate_classifier(
            model, ds.source_val_proc, ds.source_val_y, batch_size, device
        )
        row = {
            "epoch": float(epoch),
            "train_loss": total_loss / max(1, n_batches),
            "val_bce": val["bce"],
            "val_auc": val["auc"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - t0,
        }
        curves.append(row)
        logger.info(
            "Stage A epoch %d/%d | train_loss=%.4f val_bce=%.4f val_auc=%.4f lr=%.2e (%.1fs)",
            epoch + 1,
            epochs,
            row["train_loss"],
            row["val_bce"],
            row["val_auc"],
            row["lr"],
            row["seconds"],
        )

        improved = val["auc"] > best_val_auc + 1e-5
        if np.isfinite(val["auc"]) and improved:
            best_val_auc = val["auc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info(
                    "early stopping at epoch %d (best val_auc=%.4f @ epoch %d)",
                    epoch + 1,
                    best_val_auc,
                    best_epoch + 1,
                )
                break

    model.load_state_dict(best_state)
    result = TrainResult(
        best_epoch=best_epoch,
        best_val_auc=float(best_val_auc),
        epochs_run=len(curves),
        curves=curves,
        state_dict=best_state,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"armada_stageA_seed{ds.seed}.pt"
        torch.save(
            {
                "stage": "A",
                "seed": ds.seed,
                "config": dict(cfg),
                "group_dims": ds.preprocessor.processed_dims,
                "model_state": best_state,
                "best_val_auc": result.best_val_auc,
                "best_epoch": result.best_epoch,
                "provenance": {
                    "kind": ds.provenance.kind,
                    "data_dir": ds.provenance.data_dir,
                    "details": ds.provenance.details,
                },
                "preprocessor": {k: v.to_dict() for k, v in ds.preprocessor.scalers.items()},
            },
            ckpt_path,
        )
        curves_path = out_dir / f"training_curves_seed{ds.seed}.csv"
        import pandas as pd

        pd.DataFrame(curves).to_csv(curves_path, index=False)
        result.checkpoint_path = ckpt_path
        logger.info("saved checkpoint %s and curves %s", ckpt_path, curves_path)

    return result


def load_checkpoint(path: Path, device=None) -> Tuple[ArmadaCore, Dict]:
    """Rebuild the model + config from a Stage A checkpoint."""
    ckpt = torch.load(path, map_location=device or "cpu")
    model = ArmadaCore.from_config(ckpt["config"], ckpt["group_dims"])
    model.load_state_dict(ckpt["model_state"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, ckpt
