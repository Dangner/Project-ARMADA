"""Figure generation for the paper (spec §3.10).

Every figure is generated from the CSV artifacts written by the eval stages —
never from numbers typed in code.  When an input CSV is missing the figure is
skipped with a log line (the run is incomplete, not fabricated).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .style import (
    ATTACK_COLOR,
    METHOD_COLOR,
    SERIES_COLORS,
    VERDICT_COLOR,
    apply_style,
    method_color,
    new_figure,
    save_figure,
)

logger = logging.getLogger(__name__)


def _load(results_dir: Path, name: str) -> Optional[pd.DataFrame]:
    path = results_dir / name
    if not path.exists():
        logger.warning("figure input missing — skipping components of %s", path)
        return None
    return pd.read_csv(path)


def _mean_std(sub: pd.DataFrame, metric: str):
    """(mean, std) arrays for one metric from aggregated or raw rows."""
    if f"{metric}_mean" in sub.columns:
        mean = sub[f"{metric}_mean"].to_numpy(dtype=float)
        std = (
            sub[f"{metric}_std"].to_numpy(dtype=float)
            if f"{metric}_std" in sub.columns
            else None
        )
        return mean, std
    return sub[metric].to_numpy(dtype=float), None


def fig_auc_by_window(by_window: pd.DataFrame, out: Path) -> Optional[Path]:
    """Per-window AUC-ROC trajectory for every method (drift picture)."""
    metric = "auc_roc"
    if metric not in by_window.columns and f"{metric}_mean" not in by_window.columns:
        return None
    fig, ax = new_figure()
    methods = sorted(by_window["method"].unique())
    for i, m in enumerate(methods):
        sub = by_window[by_window["method"] == m].sort_values("window")
        if sub.empty:
            continue
        y, err = _mean_std(sub, metric)
        ax.errorbar(
            sub["window"],
            y,
            yerr=err,
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=m,
            color=method_color(m, i),
        )
    ax.set_xlabel("drift window (chronological)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Detection quality under drift (mean ± std over seeds)")
    ax.legend(loc="best", ncols=2)
    ax.set_ylim(0.4, 1.02)
    save_figure(fig, out)
    return out


def fig_overall_bars(overall: pd.DataFrame, out: Path) -> Optional[Path]:
    """Grouped bars: AUC-ROC / AUC-PR / F1 mean±std per method."""
    metrics = [
        m
        for m in ("auc_roc", "auc_pr", "f1")
        if m in overall.columns or f"{m}_mean" in overall.columns
    ]
    if not metrics:
        return None
    fig, ax = new_figure(figsize=(8.2, 4.6))
    methods = sorted(overall["method"].unique())
    x = np.arange(len(methods))
    width = 0.26
    colors = {"auc_roc": SERIES_COLORS[0], "auc_pr": SERIES_COLORS[1], "f1": SERIES_COLORS[2]}
    for j, metric in enumerate(metrics):
        means, stds = [], []
        for m in methods:
            sub = overall[overall["method"] == m]
            means.append(float(sub[f"{metric}_mean"].iloc[0]) if f"{metric}_mean" in sub else float("nan"))
            stds.append(float(sub[f"{metric}_std"].iloc[0]) if f"{metric}_std" in sub else 0.0)
        ax.bar(
            x + (j - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=metric.upper(),
            color=colors.get(metric, SERIES_COLORS[j]),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Overall performance (mean ± std over seeds)")
    ax.legend(loc="best")
    ax.set_ylim(0.0, 1.05)
    save_figure(fig, out)
    return out


def fig_robustness(rob: pd.DataFrame, out: Path) -> Optional[Path]:
    """Accuracy (or AUC) vs attack strength for FGSM / PGD / noise."""
    value_col = "auc_roc" if "auc_roc" in rob.columns else ("accuracy" if "accuracy" in rob.columns else None)
    if value_col is None or "epsilon" not in rob.columns:
        return None
    fig, ax = new_figure()
    for i, attack in enumerate([a for a in ("clean", "fgsm", "pgd", "noise") if a in set(rob["attack"])]):
        sub = rob[rob["attack"] == attack]
        curve = sub.groupby("epsilon")[value_col].agg(["mean", "std"]).reset_index()
        ax.plot(
            curve["epsilon"],
            curve["mean"],
            marker="o",
            linewidth=1.8,
            label=attack.upper(),
            color=ATTACK_COLOR.get(attack, SERIES_COLORS[i]),
        )
        if curve["std"].notna().any():
            ax.fill_between(
                curve["epsilon"],
                curve["mean"] - curve["std"].fillna(0),
                curve["mean"] + curve["std"].fillna(0),
                color=ATTACK_COLOR.get(attack, SERIES_COLORS[i]),
                alpha=0.15,
            )
    ax.set_xlabel("attack budget ε (feature space)")
    ax.set_ylabel(value_col.replace("_", "-").upper())
    ax.set_title("Robustness under feature-space attacks (mean ± std)")
    ax.legend(loc="best")
    save_figure(fig, out)
    return out


def fig_memory_ablation(by_window: pd.DataFrame, out: Path) -> Optional[Path]:
    """ARMADA vs ARMADA-noMem per window (value of immune memory)."""
    if "auc_roc" not in by_window.columns and "auc_roc_mean" not in by_window.columns:
        return None
    pair = by_window[by_window["method"].isin(["ARMADA", "ARMADA-noMem"])]
    if pair.empty:
        return None
    fig, ax = new_figure()
    for i, m in enumerate(["ARMADA", "ARMADA-noMem"]):
        sub = pair[pair["method"] == m].sort_values("window")
        if sub.empty:
            continue
        y, err = _mean_std(sub, "auc_roc")
        ax.errorbar(sub["window"], y, yerr=err, marker="o", linewidth=1.8,
                    capsize=3, label=m, color=METHOD_COLOR[m])
    ax.set_xlabel("drift window")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Immune memory contribution (with vs without)")
    ax.legend(loc="best")
    ax.set_ylim(0.4, 1.02)
    save_figure(fig, out)
    return out


def fig_scale_study(scale: pd.DataFrame, out: Path) -> Optional[Path]:
    """AUC-ROC vs source-train budget."""
    if "auc_roc" not in scale.columns or "scale_point" not in scale.columns:
        return None
    fig, ax = new_figure()
    curve = scale.groupby("scale_point")["auc_roc"].agg(["mean", "std"]).reset_index()
    curve = curve.sort_values("scale_point")
    ax.plot(curve["scale_point"], curve["mean"], marker="o", linewidth=1.8,
            color=METHOD_COLOR["ARMADA"])
    if curve["std"].notna().any():
        ax.fill_between(curve["scale_point"],
                        curve["mean"] - curve["std"].fillna(0),
                        curve["mean"] + curve["std"].fillna(0),
                        color=METHOD_COLOR["ARMADA"], alpha=0.15)
    ax.set_xlabel("source-train rows (total)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Scale study: data volume vs detection quality")
    ax.set_xscale("log")
    save_figure(fig, out)
    return out


def fig_decision_routing(dec: pd.DataFrame, out: Path) -> Optional[Path]:
    """Verdict routing: stacked SAFE/SUSPICIOUS/MALWARE shares per window."""
    need = {"n_safe", "n_suspicious", "n_malware", "n"}
    if not need.issubset(dec.columns):
        return None
    fig, ax = new_figure()
    for i, m in enumerate(sorted(dec["method"].unique())):
        sub = dec[dec["method"] == m].sort_values("window")
        if sub.empty:
            continue
        x = np.arange(len(sub))
        share = np.stack(
            [
                sub["n_safe"] / sub["n"],
                sub["n_suspicious"] / sub["n"],
                sub["n_malware"] / sub["n"],
            ],
            axis=1,
        )
        left = np.zeros(len(sub))
        for k, (verdict, color) in enumerate(VERDICT_COLOR.items()):
            ax.barh(x + i * 0.35, share[:, k], left=left, height=0.32,
                    color=color, label=verdict if i == 0 else None, alpha=0.9 if i == 0 else 0.55)
            left = left + share[:, k]
        ax.set_yticks(x + 0.17)
        ax.set_yticklabels(sub["window"])
    ax.set_xlabel("share of window")
    ax.set_title("Decision routing by window (SAFE / SUSPICIOUS / MALWARE)")
    ax.legend(loc="lower right")
    ax.set_xlim(0.0, 1.0)
    save_figure(fig, out)
    return out


def fig_ablation_deltas(abl: pd.DataFrame, out: Path) -> Optional[Path]:
    """Δ AUC-ROC of each ablation vs reference (feature-group importance)."""
    if "auc_roc" not in abl.columns or "ablation" not in abl.columns:
        return None
    means = abl.groupby("ablation")["auc_roc"].mean()
    if REF not in means:
        return None
    delta = (means - means[REF]).drop(labels=[REF], errors="ignore").sort_values()
    fig, ax = new_figure(figsize=(8.2, 4.8))
    colors = [VERDICT_COLOR["MALWARE"] if d < 0 else VERDICT_COLOR["SAFE"] for d in delta]
    ax.barh(range(len(delta)), delta.values, color=colors)
    ax.set_yticks(range(len(delta)))
    ax.set_yticklabels(delta.index)
    ax.axvline(0.0, color="#334155", linewidth=0.8)
    ax.set_xlabel("Δ AUC-ROC vs reference (mean over seeds)")
    ax.set_title("Ablations: what each component / feature group contributes")
    save_figure(fig, out)
    return out


REF = "reference"


FIGURE_SPECS = (
    ("fig1_auc_by_window", fig_auc_by_window, "metrics_by_window.csv"),
    ("fig2_overall_bars", fig_overall_bars, "metrics_all.csv"),
    ("fig3_robustness", fig_robustness, "robustness.csv"),
    ("fig4_memory_ablation", fig_memory_ablation, "metrics_by_window.csv"),
    ("fig5_scale_study", fig_scale_study, "scale_study.csv"),
    ("fig6_decision_routing", fig_decision_routing, "decision_report.csv"),
    ("fig7_ablation_deltas", fig_ablation_deltas, "ablation.csv"),
)


def generate_all(
    results_dir: Path,
    figures_dir: Path,
    formats: Sequence[str] = ("png", "pdf"),
) -> List[Path]:
    """Generate every figure whose input CSV exists. Never fabricates."""
    apply_style()
    written: List[Path] = []
    for name, fn, source in FIGURE_SPECS:
        df = _load(results_dir, source)
        if df is None or df.empty:
            logger.warning("figure %s skipped: %s empty or missing", name, source)
            continue
        # re-implement save with formats via wrapper
        out = figures_dir / name
        produced = _save_with_formats(fn, df, out, formats)
        if produced:
            written.append(produced)
            logger.info("wrote figure %s", produced.name)
    return written


def _save_with_formats(fn, df: pd.DataFrame, out: Path, formats: Sequence[str]):
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    try:
        produced = fn(df, out)
    finally:
        for num in set(plt.get_fignums()) - before:
            plt.close(num)
    return produced
