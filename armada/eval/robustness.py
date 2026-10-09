"""Robustness evaluation: clean vs adversarial accuracy (spec §3.6).

For every evaluated method and each epsilon in ``robustness.eval_epsilons``,
generates FGSM and PGD feature-space attacks (plus a random-noise baseline)
on every target window's labeled rows and recomputes the metric suite from
saved predictions.  MalGAN-style attacks are optional and not implemented.

**Limitation:** attacks live in EMBER feature space with valid-range
projection; perturbed vectors are not guaranteed to be realisable PE files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ..attacks import FeatureSpaceProjector, fgsm_attack, pgd_attack
from ..data.loader import SeedDataset
from .baselines import ResultsWriter, metrics_from_saved_predictions

logger = logging.getLogger(__name__)

ATTACKS = ("clean", "fgsm", "pgd", "noise")


def _noise_attack(
    groups_clean: Mapping[str, torch.Tensor],
    eps: float,
    projector: FeatureSpaceProjector,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Random-noise baseline: uniform l∞ noise of radius ``eps`` + projection."""
    free = projector.free_mask(groups_clean)
    with torch.no_grad():
        out = {}
        for name, x in groups_clean.items():
            delta = torch.empty_like(x).uniform_(-eps, eps, generator=generator)
            delta = delta * free[name].to(delta.dtype)
            out[name] = x + delta
        out = projector.project(out, groups_clean)
        for name in out:
            out[name] = torch.max(torch.min(out[name], groups_clean[name] + eps), groups_clean[name] - eps)
    return out


@torch.no_grad()
def _predict(model, groups: Mapping[str, torch.Tensor]) -> np.ndarray:
    p, _ = model.predict_proba(groups)
    return p.cpu().numpy()


def _window_attack_metrics(
    method: str,
    model: torch.nn.Module,
    w: Mapping,
    attack: str,
    eps: float,
    cfg: Mapping,
    projector: FeatureSpaceProjector,
    writer: ResultsWriter,
    seed: int,
    device: torch.device,
    pgd_steps: int,
    pgd_step_size: float,
    clip_min: float,
) -> Optional[Dict[str, object]]:
    y_true = np.asarray(w["y"])
    n = len(y_true)
    if n == 0:
        return None
    proc = w["proc"]
    batch_size = int(cfg["train"].get("batch_size", 256))
    probs: List[np.ndarray] = []
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        clean = {
            name: torch.as_tensor(block[sl], dtype=torch.float32, device=device)
            for name, block in proc.items()
        }
        y_t = torch.as_tensor(y_true[sl], dtype=torch.float32, device=device)
        if attack == "clean" or eps <= 0:
            adv = clean
        elif attack == "fgsm":
            adv = fgsm_attack(model, clean, y_t, eps, projector, clip_min)
        elif attack == "pgd":
            adv = pgd_attack(
                model,
                clean,
                y_t,
                eps,
                step_size=pgd_step_size,
                steps=pgd_steps,
                projector=projector,
                clip_min=clip_min,
                seed=seed,
            )
        elif attack == "noise":
            adv = _noise_attack(clean, eps, projector)
        else:
            raise ValueError(f"unknown attack '{attack}'")
        probs.append(_predict(model, adv))
    y_prob = np.concatenate(probs)

    tag = "clean" if attack == "clean" or eps <= 0 else f"{attack}_eps{eps:g}"
    pred_name = f"robust_{method}_{tag}"
    writer.save_predictions(pred_name, seed, w["name"], y_true, y_prob)
    metrics = metrics_from_saved_predictions(
        writer.results_dir / "predictions" / f"{pred_name}_seed{seed}_{w['name']}.npz"
    )
    row: Dict[str, object] = {
        "method": method,
        "seed": seed,
        "window": w["name"],
        "attack": "clean" if eps <= 0 else attack,
        "epsilon": float(eps if attack != "clean" else 0.0),
    }
    row.update(metrics)
    return row


def evaluate_robustness_seed(
    method: str,
    model: torch.nn.Module,
    ds: SeedDataset,
    cfg: Mapping,
    writer: ResultsWriter,
    device: torch.device,
    prepare_fn=None,
) -> List[Dict[str, object]]:
    """Robustness rows for one method/seed over all windows and epsilons.

    ``prepare_fn(model, w) -> None`` optionally mutates the model before a
    window is attacked (e.g. run TTT on the window's unlabeled rows first).
    """
    rcfg = cfg.get("robustness", {})
    epsilons = [float(e) for e in rcfg.get("eval_epsilons", [0.0, 0.01, 0.02, 0.05, 0.1])]
    pgd_steps = int(rcfg.get("pgd_steps", 7))
    pgd_step_size = float(rcfg.get("pgd_step_size", 0.01))
    clip_min = float(rcfg.get("clip_min", 0.0))
    projector = FeatureSpaceProjector(ds.preprocessor, ds.source_train_raw, device=device)
    ds.ensure_processed()

    rows: List[Dict[str, object]] = []
    for w in ds.windows:
        if len(w["y"]) == 0:
            continue
        if prepare_fn is not None:
            prepare_fn(model, w)
        # clean row
        r = _window_attack_metrics(
            method, model, w, "clean", 0.0, cfg, projector, writer, ds.seed, device,
            pgd_steps, pgd_step_size, clip_min,
        )
        rows.append(r)
        for eps in epsilons:
            if eps <= 0:
                continue
            for attack in ("fgsm", "pgd", "noise"):
                r = _window_attack_metrics(
                    method, model, w, attack, eps, cfg, projector, writer, ds.seed, device,
                    pgd_steps, pgd_step_size, clip_min,
                )
                rows.append(r)
                logger.info(
                    "[robust %s seed=%d %s %s eps=%g] acc=%.4f auc_roc=%.4f",
                    method,
                    ds.seed,
                    w["name"],
                    attack,
                    eps,
                    r["accuracy"],
                    r["auc_roc"],
                )
    return rows
