"""Stage B — joint adaptation training (spec §3.4, §3.5, §4).

Per step: one labeled SOURCE batch + one unlabeled TARGET batch (labels of
target rows are never used).  Joint objective::

    total = w_cls * classification
          + w_marg * marginal_domain      (D1 through GRL)
          + w_cond * conditional_domain   (D2/CDAN through GRL)
          + w_recon * masked_reconstruction

The GRL reversal strength follows ``lambda(p) = 2/(1+exp(-gamma p)) - 1``
with ``p`` = step / total_steps.  Two variants are trained for the paper
tables: ``dann`` (naive single-discriminator DANN: D1 only, no
reconstruction) and ``dual`` (full dual-discriminator + reconstruction).
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from ..data.loader import SeedDataset
from ..models.core import ArmadaDomainModel
from ..models.grl import grl_lambda
from ..models.heads import build_pos_weight, classification_loss
from ..utils import get_device, set_seed

logger = logging.getLogger(__name__)

VARIANTS = ("dann", "dual")


@dataclass
class AdaptResult:
    variant: str
    best_epoch: int
    best_val_auc: float
    epochs_run: int
    curves: List[Dict[str, float]] = field(default_factory=list)
    checkpoint_path: Optional[Path] = None
    state_dict: Optional[Dict[str, torch.Tensor]] = None


def _stack_batches(
    proc: Mapping[str, np.ndarray], idx: np.ndarray, device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(block[idx], dtype=torch.float32, device=device)
        for name, block in proc.items()
    }


def _target_pool(ds: SeedDataset) -> Dict[str, np.ndarray]:
    """Concatenate processed unlabeled rows from every target window."""
    parts = []
    for w in ds.windows:
        if len(w["unl_raw"]):
            if w.get("unl_proc") is None:
                w["unl_proc"] = ds.preprocessor.transform(w["unl_raw"])
            parts.append(w["unl_proc"])
    if not parts:
        raise RuntimeError(
            "Stage B needs unlabeled target rows (config data.n_unlabeled_per_window "
            "> 0 and windows with y == -1 rows); none were found."
        )
    return {name: np.concatenate([p[name] for p in parts], axis=0) for name in parts[0]}


def train_stage_b(
    ds: SeedDataset,
    cfg: Mapping,
    variant: str = "dual",
    out_dir: Optional[Path] = None,
    init_state: Optional[Dict[str, torch.Tensor]] = None,
) -> AdaptResult:
    """Joint adaptation training for one seed and one variant.

    ``init_state``: optional Stage A state dict to start from (recommended).
    Writes ``armada_stageB_<variant>_seed<seed>.pt`` and a curves CSV with
    per-component losses when ``out_dir`` is given.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got '{variant}'")
    tcfg = cfg["train"]
    set_seed(ds.seed)
    device = get_device(cfg)
    use_amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
    use_d2 = variant == "dual"
    recon_weight = float(tcfg.get("recon_weight", 0.5)) if variant == "dual" else 0.0

    ds.ensure_processed()
    model = ArmadaDomainModel.from_config(
        {**cfg, "train": {**tcfg, "use_d2": use_d2}}, ds.preprocessor.processed_dims
    ).to(device)
    if init_state is not None:
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        # new submodules (d1/d2/decoder/mask) legitimately start fresh
        logger.info(
            "Stage B (%s) init from Stage A: %d tensors loaded (%d new modules)",
            variant,
            len(init_state) - len(missing),
            len(missing),
        )
    logger.info(
        "Stage B: seed=%d variant=%s device=%s amp=%s use_d2=%s recon_weight=%.2f",
        ds.seed,
        variant,
        device,
        use_amp,
        use_d2,
        recon_weight,
    )

    pos_weight = build_pos_weight(tcfg, ds.source_train_y)
    tgt_proc = _target_pool(ds)
    n_tgt = len(next(iter(tgt_proc.values())))
    logger.info("Stage B target pool: %d unlabeled rows", n_tgt)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg.get("lr", 1e-4)),
        weight_decay=float(tcfg.get("weight_decay", 1e-5)),
    )
    epochs = int(tcfg.get("stage_b_epochs", 15))
    batch_size = int(tcfg.get("batch_size", 256))
    steps_per_epoch = max(1, len(ds.source_train_y) // max(1, batch_size))
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    ) if str(tcfg.get("scheduler", "cosine")).lower() == "cosine" else None
    grad_clip = float(tcfg.get("grad_clip", 5.0))
    patience = int(tcfg.get("early_stopping_patience", 5))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    from .pretrain import _GroupBatcher, evaluate_classifier

    src_batcher = _GroupBatcher(ds.source_train_proc, ds.source_train_y, batch_size, device)
    rng = torch.Generator(device="cpu").manual_seed(ds.seed)

    curves: List[Dict[str, float]] = []
    best_val_auc = -float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    global_step = 0

    from tqdm import tqdm

    for epoch in tqdm(range(epochs), desc=f"Stage B/{variant} (seed {ds.seed})", leave=False):
        model.train()
        t0 = time.perf_counter()
        sums = {"cls": 0.0, "marg": 0.0, "cond": 0.0, "recon": 0.0, "total": 0.0}
        n_batches = 0
        for src_groups, y_batch in src_batcher.epoch_batches(rng):
            t_idx = torch.randperm(n_tgt).numpy()[:batch_size]
            tgt_groups = _stack_batches(tgt_proc, t_idx, device)

            lambd = grl_lambda(global_step / total_steps, float(tcfg.get("grl_gamma", 10.0)))
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                # classification + domain losses from the unmasked forward
                z_src, _ = model.embed(src_groups)
                logits_src = model.classifier(z_src)
                z_tgt, _ = model.embed(tgt_groups)
                from ..models.heads import malware_class_probs

                with torch.no_grad():
                    probs_tgt = malware_class_probs(model.classifier(z_tgt))
                probs_src = malware_class_probs(logits_src)

                cls_loss = classification_loss(logits_src, y_batch, pos_weight=pos_weight)
                marg_loss = model.domain_loss_d1(z_src, False, lambd) + model.domain_loss_d1(
                    z_tgt, True, lambd
                )
                if use_d2:
                    cond_loss = model.domain_loss_d2(
                        z_src, probs_src, False, lambd
                    ) + model.domain_loss_d2(z_tgt, probs_tgt, True, lambd)
                else:
                    cond_loss = cls_loss.new_zeros(())

                if recon_weight > 0:
                    recon_loss = model.reconstruction_loss(
                        src_groups, mask_ratio=float(cfg.get("ttt", {}).get("mask_ratio", 0.3))
                    ) + model.reconstruction_loss(
                        tgt_groups, mask_ratio=float(cfg.get("ttt", {}).get("mask_ratio", 0.3))
                    )
                else:
                    recon_loss = cls_loss.new_zeros(())

                total = (
                    float(tcfg.get("cls_weight", 1.0)) * cls_loss
                    + float(tcfg.get("marg_weight", 0.5)) * marg_loss
                    + float(tcfg.get("cond_weight", 0.5)) * cond_loss
                    + recon_weight * recon_loss
                )
            if not torch.isfinite(total):
                raise RuntimeError(
                    "non-finite Stage B loss; try lowering train.lr / marg_weight"
                )
            scaler.scale(total).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            sums["cls"] += float(cls_loss.item())
            sums["marg"] += float(marg_loss.item())
            sums["cond"] += float(cond_loss.item()) if torch.is_tensor(cond_loss) else float(cond_loss)
            sums["recon"] += float(recon_loss.item()) if torch.is_tensor(recon_loss) else float(recon_loss)
            sums["total"] += float(total.item())
            n_batches += 1
            global_step += 1

        val = evaluate_classifier(
            model, ds.source_val_proc, ds.source_val_y, batch_size, device
        )
        row = {
            "epoch": float(epoch),
            "train_cls": sums["cls"] / max(1, n_batches),
            "train_marg": sums["marg"] / max(1, n_batches),
            "train_cond": sums["cond"] / max(1, n_batches),
            "train_recon": sums["recon"] / max(1, n_batches),
            "train_total": sums["total"] / max(1, n_batches),
            "val_bce": val["bce"],
            "val_auc": val["auc"],
            "grl_lambda": lambd,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - t0,
        }
        curves.append(row)
        logger.info(
            "Stage B/%s epoch %d/%d | cls=%.4f marg=%.4f cond=%.4f recon=%.4f "
            "val_auc=%.4f lambda=%.3f (%.1fs)",
            variant,
            epoch + 1,
            epochs,
            row["train_cls"],
            row["train_marg"],
            row["train_cond"],
            row["train_recon"],
            row["val_auc"],
            row["grl_lambda"],
            row["seconds"],
        )

        improved = np.isfinite(val["auc"]) and val["auc"] > best_val_auc + 1e-5
        if improved:
            best_val_auc = val["auc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info(
                    "Stage B/%s early stop at epoch %d (best val_auc=%.4f @ %d)",
                    variant,
                    epoch + 1,
                    best_val_auc,
                    best_epoch + 1,
                )
                break

    model.load_state_dict(best_state)
    result = AdaptResult(
        variant=variant,
        best_epoch=best_epoch,
        best_val_auc=float(best_val_auc),
        epochs_run=len(curves),
        curves=curves,
        state_dict=best_state,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"armada_stageB_{variant}_seed{ds.seed}.pt"
        torch.save(
            {
                "stage": "B",
                "variant": variant,
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
        import pandas as pd

        curves_path = out_dir / f"training_curves_stageB_{variant}_seed{ds.seed}.csv"
        pd.DataFrame(curves).to_csv(curves_path, index=False)
        result.checkpoint_path = ckpt_path
        logger.info("saved Stage B checkpoint %s and curves %s", ckpt_path, curves_path)

    return result


def load_stage_b(path: Path, device=None) -> Tuple[ArmadaDomainModel, Dict]:
    """Rebuild an ArmadaDomainModel from a Stage B checkpoint."""
    ckpt = torch.load(path, map_location=device or "cpu")
    tcfg = ckpt["config"].get("train", {})
    use_d2 = ckpt.get("variant", "dual") == "dual"
    model = ArmadaDomainModel.from_config(
        {**ckpt["config"], "train": {**tcfg, "use_d2": use_d2}}, ckpt["group_dims"]
    )
    model.load_state_dict(ckpt["model_state"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, ckpt
