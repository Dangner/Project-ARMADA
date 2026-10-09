"""LaTeX + CSV tables for the paper, generated from results/*.csv only."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _fmt(value: float, digits: int = 4) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "--"
    return "--" if not (v == v) else f"{v:.{digits}f}"


def table_main_results(by_window: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """Method × window AUC-ROC / AUC-PR / F1 (mean ± std)."""
    if "auc_roc" not in by_window.columns and "auc_roc_mean" not in by_window.columns:
        return None
    rows = []
    for _, r in by_window.sort_values(["method", "window"]).iterrows():
        rows.append(
            {
                "method": r["method"],
                "window": r["window"],
                "auc_roc": _fmt(r.get("auc_roc_mean", r.get("auc_roc"))),
                "auc_pr": _fmt(r.get("auc_pr_mean", r.get("auc_pr"))),
                "f1": _fmt(r.get("f1_mean", r.get("f1"))),
                "tpr_at_fpr_0.001": _fmt(r.get("tpr_at_fpr_0.001_mean", r.get("tpr_at_fpr_0.001"))),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table1_main_results.csv", index=False)
    tex_path = out_dir / "table1_main_results.tex"
    tex_path.write_text(df.to_latex(index=False, caption="Main results per drift window", label="tab:main"))
    return tex_path


def table_robustness(rob: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    value = "auc_roc" if "auc_roc" in rob.columns else ("accuracy" if "accuracy" in rob.columns else None)
    if value is None:
        return None
    pivot = rob.pivot_table(index=["method", "attack"], columns="epsilon", values=value, aggfunc="mean")
    pivot = pivot.round(4)
    pivot.to_csv(out_dir / "table2_robustness.csv")
    tex_path = out_dir / "table2_robustness.tex"
    tex_path.write_text(pivot.to_latex(caption="Robustness under feature-space attacks", label="tab:robust"))
    return tex_path


def table_ablation(abl: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    if "auc_roc" not in abl.columns or "ablation" not in abl.columns:
        return None
    g = abl.groupby("ablation")["auc_roc"].agg(["mean", "std", "count"]).round(4).reset_index()
    g.to_csv(out_dir / "table3_ablation.csv", index=False)
    tex_path = out_dir / "table3_ablation.tex"
    tex_path.write_text(g.to_latex(index=False, caption="Ablation study", label="tab:ablation"))
    return tex_path


def table_decision(dec: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    need = {"n_safe", "n_suspicious", "n_malware", "sandbox_rate"}
    if not need.issubset(dec.columns):
        return None
    cols = ["method", "window", "n", "n_safe", "n_suspicious", "n_malware",
            "sandbox_rate", "conf_tp", "conf_fp", "conf_fn",
            "malware_threshold", "suspicious_threshold"]
    cols = [c for c in cols if c in dec.columns]
    g = dec[cols].copy()
    for c in ("sandbox_rate", "malware_threshold", "suspicious_threshold"):
        if c in g:
            g[c] = g[c].astype(float).round(4)
    g.to_csv(out_dir / "table4_decision.csv", index=False)
    tex_path = out_dir / "table4_decision.tex"
    tex_path.write_text(g.to_latex(index=False, caption="Decision engine routing", label="tab:decision"))
    return tex_path


def generate_all(results_dir: Path, tables_dir: Path) -> List[Path]:
    """Generate every table whose input CSV exists."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    specs = (
        (table_main_results, "metrics_by_window.csv"),
        (table_robustness, "robustness.csv"),
        (table_ablation, "ablation.csv"),
        (table_decision, "decision_report.csv"),
    )
    written: List[Path] = []
    for fn, source in specs:
        path = results_dir / source
        if not path.exists():
            logger.warning("table input missing — skipping %s", source)
            continue
        df = pd.read_csv(path)
        if df.empty:
            logger.warning("table input empty — skipping %s", source)
            continue
        out = fn(df, tables_dir)
        if out:
            written.append(out)
            logger.info("wrote table %s", out.name)
    return written
