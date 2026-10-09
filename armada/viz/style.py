"""Publication figure style (paper house style).

White background, slate text, series colours from the indigo / emerald /
rose / amber family.  All paper figures are produced at 300 dpi in both PNG
and PDF via :func:`save_figure`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe (WSL2 / CI)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# --- palette ---------------------------------------------------------------
SLATE = "#334155"          # text / axes
SLATE_LIGHT = "#64748b"    # secondary text
INDIGO = "#4f46e5"         # primary series (ARMADA)
EMERALD = "#059669"        # positive / memory-enabled
ROSE = "#e11d48"           # attacks / negatives
AMBER = "#d97704"          # warnings / suspicious
GRID = "#e2e8f0"

#: Default cycle for method series (paper order).
SERIES_COLORS: Sequence[str] = (INDIGO, EMERALD, AMBER, ROSE, SLATE, SLATE_LIGHT)

DPI = 300
FIGSIZE = (7.2, 4.6)

METHOD_COLOR = {
    "ARMADA": INDIGO,
    "ARMADA-noMem": AMBER,
    "DualDANN+TTT": EMERALD,
    "DualDANN+TTT-online": EMERALD,
    "DualDANN": SLATE_LIGHT,
    "DANN": SLATE,
    "GroupedAttn": "#7c3aed",
    "PlainMLP": "#0ea5e9",
    "LR": "#94a3b8",
    "RF": "#64748b",
    "GBDT": "#475569",
    "XGBoost": "#475569",
    "ABL:reference": INDIGO,
}

VERDICT_COLOR = {"SAFE": EMERALD, "SUSPICIOUS": AMBER, "MALWARE": ROSE}
ATTACK_COLOR = {"clean": EMERALD, "fgsm": AMBER, "pgd": ROSE, "noise": SLATE_LIGHT}


def apply_style() -> None:
    """Apply the house style globally (white bg, slate text, grid)."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": DPI,
            "figure.dpi": 100,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": SLATE,
            "axes.labelcolor": SLATE,
            "axes.titlecolor": SLATE,
            "xtick.color": SLATE_LIGHT,
            "ytick.color": SLATE_LIGHT,
            "text.color": SLATE,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "sans-serif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def method_color(name: str, index: int = 0) -> str:
    """Colour for one method series (stable across all figures)."""
    if name in METHOD_COLOR:
        return METHOD_COLOR[name]
    if name.startswith("ABL:"):
        return SERIES_COLORS[index % len(SERIES_COLORS)]
    return SERIES_COLORS[index % len(SERIES_COLORS)]


def save_figure(fig: Figure, out_base: Path, formats: Sequence[str] = ("png", "pdf")) -> None:
    """Save a figure at 300 dpi; ``out_base`` without extension."""
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=DPI, bbox_inches="tight")


def new_figure(figsize=FIGSIZE):
    """Create a styled figure + single axes."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax
