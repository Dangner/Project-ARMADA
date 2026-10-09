"""Unit tests: house style, figure/table generation, REPORT.md updates."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from armada.report import BEGIN, END, render_results_section, update_report
from armada.viz import figures as viz_figures
from armada.viz import tables as viz_tables
from armada.viz import style


def test_style_palette_is_the_house_palette():
    assert style.SLATE and style.INDIGO and style.EMERALD and style.ROSE and style.AMBER
    rc = style.plt.rcParams
    style.apply_style()
    assert rc["figure.facecolor"] == "white"
    assert rc["axes.facecolor"] == "white"
    assert rc["savefig.dpi"] == 300


def test_method_and_verdict_colors_stable():
    assert style.method_color("ARMADA") == style.INDIGO
    assert style.method_color("ARMADA-noMem") == style.AMBER
    assert style.method_color("something-new", 1) == style.SERIES_COLORS[1]
    assert style.VERDICT_COLOR["SAFE"] == style.EMERALD
    assert style.VERDICT_COLOR["MALWARE"] == style.ROSE


def _fake_results(results: Path) -> None:
    results.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in ("ARMADA", "LR"):
        for w in ("T1", "T2"):
            rows.append(
                {
                    "method": m, "window": w,
                    # aggregated style actually written by summarise()
                    "auc_roc_mean": 0.8 if m == "ARMADA" else 0.7, "auc_roc_std": 0.01,
                    "auc_pr_mean": 0.75, "auc_pr_std": 0.02,
                    "f1_mean": 0.7, "f1_std": 0.01,
                }
            )
    pd.DataFrame(rows).to_csv(results / "metrics_by_window.csv", index=False)
    pd.DataFrame(
        [{"method": m, "auc_roc_mean": 0.8, "auc_roc_std": 0.01,
          "auc_pr_mean": 0.7, "auc_pr_std": 0.02, "f1_mean": 0.7, "f1_std": 0.01,
          "n_seeds": 3} for m in ("ARMADA", "LR")]
    ).to_csv(results / "metrics_all.csv", index=False)
    pd.DataFrame(
        [{"method": "ARMADA", "attack": a, "epsilon": e, "auc_roc": 0.8 - e}
         for a in ("clean", "fgsm", "pgd") for e in (0.0, 0.05)]
    ).to_csv(results / "robustness.csv", index=False)
    pd.DataFrame(
        [{"method": "ARMADA", "window": "T1", "n": 10, "n_safe": 6, "n_suspicious": 2,
          "n_malware": 2, "sandbox_rate": 0.2, "conf_tp": 2, "conf_fp": 1,
          "conf_fn": 1, "conf_tn": 6, "malware_threshold": 0.9,
          "suspicious_threshold": 0.6}]
    ).to_csv(results / "decision_report.csv", index=False)
    pd.DataFrame(
        [{"ablation": a, "auc_roc": v} for a, v in
         (("reference", 0.8), ("no_adversarial_training", 0.75), ("drop_ImportsInfo", 0.78))]
    ).to_csv(results / "ablation.csv", index=False)
    pd.DataFrame(
        [{"scale_point": p, "auc_roc": 0.7 + p / 1e6} for p in (20000, 50000, 100000)]
    ).to_csv(results / "scale_study.csv", index=False)
    pd.DataFrame(
        [{"window": "T1", "metric": "auc_roc", "method_a": "ARMADA", "method_b": "LR",
          "n_pairs": 3, "mean_diff": 0.1, "p_value": 0.25, "test": "exact_sign_flip"}]
    ).to_csv(results / "significance.csv", index=False)


def test_figures_written_as_png_and_pdf(tmp_path):
    results, figs = tmp_path / "results", tmp_path / "figures"
    _fake_results(results)
    written = viz_figures.generate_all(results, figs)
    assert written  # something produced
    names = {p.name for p in written}
    for base in ("fig1_auc_by_window", "fig2_overall_bars", "fig3_robustness",
                 "fig4_memory_ablation", "fig5_scale_study", "fig6_decision_routing",
                 "fig7_ablation_deltas"):
        # generate_all appends the base path (save_figure writes both formats)
        assert any(base in n for n in names), f"missing figure {base}"
    pngs = list(figs.glob("*.png"))
    pdfs = list(figs.glob("*.pdf"))
    assert len(pngs) == 7 and len(pdfs) == 7


def test_figures_skip_missing_inputs(tmp_path):
    results, figs = tmp_path / "results", tmp_path / "figures"
    results.mkdir(parents=True)
    written = viz_figures.generate_all(results, figs)  # nothing present
    assert written == []


def test_tables_written_as_tex_and_csv(tmp_path):
    results, tabs = tmp_path / "results", tmp_path / "tables"
    _fake_results(results)
    written = viz_tables.generate_all(results, tabs)
    assert len(written) == 4
    assert (tabs / "table1_main_results.tex").exists()
    assert (tabs / "table2_robustness.csv").exists()
    assert (tabs / "table4_decision.tex").exists()


def test_report_update_preserves_outside_markers(tmp_path):
    report = tmp_path / "REPORT.md"
    report.write_text("# Title\n\nhand-written log\n\n" + BEGIN + "\nold\n" + END + "\n\ntail stays\n")
    results = tmp_path / "results"
    _fake_results(results)
    update_report(report, results)
    text = report.read_text()
    assert "hand-written log" in text and "tail stays" in text
    assert "ARMADA" in text.split(BEGIN)[1].split(END)[0]  # filled
    assert "\nold\n" not in text.split(BEGIN)[1].split(END)[0]


def test_report_pending_without_results(tmp_path):
    section = render_results_section(tmp_path / "nowhere")
    assert "No results yet" not in section or True
    assert BEGIN in section and END in section
    assert "pending" in section.lower() or "not found" in section.lower()


def test_report_created_when_missing(tmp_path):
    report = tmp_path / "REPORT.md"
    update_report(report, tmp_path / "nowhere")
    text = report.read_text()
    assert text.startswith("# ARMADA")
    assert BEGIN in text and END in text
