"""ARMADA Threat Intelligence Scanner (GUI).

Refactored from the legacy ``mini_project_gui.py``: same flow — header,
scanner controls, scan-a-random-file, verdict display with a feature-group
bar chart, status bar — but running the **ARMADA pipeline** (profiler +
encoder + classifier + immune memory + calibrated decision engine) instead
of a LightGBM model, and using the house palette from ``viz.style``.

Requires a real EMBER dataset and trained checkpoints; refuses to invent
scan results without them (same behaviour as the legacy GUI's "model file
not found" guard).  Tkinter is imported lazily so headless environments can
import this module safely.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

WINDOW_TITLE = "ARMADA Threat Intelligence Scanner"
CANVAS_TITLE = "🛡️ ARMADA AI SCANNER"


class ScanEngine:
    """Non-GUI scoring: dataset + checkpoint → calibrated verdict for one row."""

    def __init__(self, cfg: dict, seed: int = 0):
        from .data.loader import prepare_seed_dataset
        from .models.decision import DecisionEngine
        from .models.memory import MemoryBank, MemoryCalibrator
        from .run import build_pool  # lazy: run.py imports gui lazily too
        from .train.adapt import load_stage_b
        from .utils import get_device

        self.cfg = cfg
        self.seed = seed
        self.device = get_device(cfg)
        pool = build_pool(cfg)
        self.ds = prepare_seed_dataset(pool, cfg, seed)
        self.ds.ensure_processed()

        ckpt_dir = Path(cfg["data"].get("checkpoints_dir", "checkpoints"))
        ckpt = ckpt_dir / f"armada_stageB_dual_seed{seed}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"model checkpoint not found: {ckpt}. Train first: "
                "python -m armada.run --config configs/main.yaml --stage train"
            )
        self.model, _ = load_stage_b(ckpt, device=self.device)

        mem_cfg = cfg.get("memory", {})
        self.memory = MemoryBank(
            capacity=int(mem_cfg.get("capacity", 4096)),
            store_benign=bool(mem_cfg.get("store_benign", True)),
            evict=str(mem_cfg.get("evict", "age")),
            max_age_windows=int(mem_cfg.get("max_age_windows", 3)),
        )
        from .eval.neural import _embed_and_prob

        bs = int(cfg["train"].get("batch_size", 256))
        z_tr, _ = _embed_and_prob(self.model, self.ds.source_train_proc, bs, self.device)
        self.memory.add(z_tr, self.ds.source_train_y)
        z_val, p_val = _embed_and_prob(self.model, self.ds.source_val_proc, bs, self.device)
        sim_val = self.memory.max_malware_similarity(z_val)
        self.calibrator = MemoryCalibrator(use_memory=True).fit(
            p_val, sim_val, self.ds.source_val_y
        )
        scores_val = self.calibrator.score(p_val, sim_val)
        dec_cfg = cfg.get("decision", {})
        self.engine = DecisionEngine(
            malware_fpr_target=float(dec_cfg.get("malware_fpr_target", 0.001)),
            suspicious_fpr_target=float(dec_cfg.get("suspicious_fpr_target", 0.05)),
        ).fit(self.ds.source_val_y, scores_val)

    def random_scan(self) -> dict:
        """Score one random target-window row; returns display payload."""
        from .eval.neural import _embed_and_prob

        windows = [w for w in self.ds.windows if len(w["y"])]
        if not windows:
            raise RuntimeError("no labelled target windows available for scanning")
        w = windows[int(np.random.randint(len(windows)))]
        i = int(np.random.randint(len(w["y"])))
        proc_i = {name: block[i : i + 1] for name, block in w["proc"].items()}
        z, p = _embed_and_prob(self.model, proc_i, 1, self.device)
        sim = self.memory.max_malware_similarity(z)
        score = float(self.calibrator.score(p, sim)[0])
        verdict = str(self.engine.verdicts(np.array([score]))[0])
        group_threat = {
            name: float(np.abs(block[0]).mean()) for name, block in proc_i.items()
        }
        return {
            "window": w["name"],
            "index": i,
            "true_label": int(w["y"][i]),
            "p_malware": float(p[0]),
            "max_memory_similarity": float(sim[0]),
            "score": score,
            "verdict": verdict,
            "thresholds": (
                float(self.engine.malware_threshold),
                float(self.engine.suspicious_threshold),
            ),
            "group_threat": group_threat,
        }


def run_gui(cfg: dict, seed: int = 0) -> None:
    """Launch the Tkinter scanner (blocking)."""
    import tkinter as tk
    from tkinter import messagebox

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    from .viz.style import SLATE, VERDICT_COLOR, apply_style

    apply_style()

    class MalwareScannerApp:
        def __init__(self, root: "tk.Tk"):
            self.root = root
            self.root.title(WINDOW_TITLE)
            self.root.geometry("860x620")
            self.root.configure(bg="white")
            self.engine: Optional[ScanEngine] = None
            self.create_header()
            self.create_main_panel()
            self.create_status_bar()
            self.root.after(100, self.load_resources)

        def create_header(self):
            header = tk.Frame(self.root, bg=SLATE, height=80)
            header.pack(fill=tk.X)
            tk.Label(
                header,
                text=CANVAS_TITLE,
                font=("Helvetica", 22, "bold"),
                bg=SLATE,
                fg="white",
            ).pack(pady=18)

        def create_main_panel(self):
            controls = tk.Frame(self.root, bg="white", width=280, padx=12, pady=12)
            controls.pack(side=tk.LEFT, fill=tk.Y)
            tk.Label(
                controls, text="Scanner Controls", font=("Arial", 13, "bold"),
                bg="white", fg=SLATE,
            ).pack(pady=(0, 12))
            self.btn_scan = tk.Button(
                controls,
                text="SCAN RANDOM FILE",
                font=("Arial", 11, "bold"),
                bg=SLATE,
                fg="white",
                state=tk.DISABLED,
                command=self.scan_file,
                height=2,
            )
            self.btn_scan.pack(fill=tk.X, pady=6)
            self.lbl_info = tk.Label(
                controls,
                text="Loads EMBER + ARMADA checkpoints\n(real data only).",
                font=("Arial", 9), bg="white", fg=SLATE, justify=tk.LEFT,
            )
            self.lbl_info.pack(pady=12)

            viz = tk.Frame(self.root, bg="white")
            viz.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            self.lbl_result = tk.Label(
                viz, text="System Ready.", font=("Courier", 16, "bold"),
                bg="white", fg=SLATE,
            )
            self.lbl_result.pack(pady=8)
            self.fig = Figure(figsize=(5.6, 3.6), dpi=100)
            self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.fig, master=viz)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def create_status_bar(self):
            self.status_var = tk.StringVar(value="Initializing System...")
            bar = tk.Label(
                self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN,
                anchor=tk.W, bg="#f1f5f9", fg=SLATE,
            )
            bar.pack(side=tk.BOTTOM, fill=tk.X)

        def load_resources(self):
            try:
                self.status_var.set("Loading AI Brain (EMBER + ARMADA checkpoints)...")
                self.engine = ScanEngine(cfg, seed=seed)
                self.btn_scan.config(state=tk.NORMAL)
                self.status_var.set("System Ready — model loaded.")
            except FileNotFoundError as exc:
                self.status_var.set("Model unavailable.")
                messagebox.showerror("Error", str(exc))
            except Exception as exc:  # pragma: no cover - GUI-only path
                logger.exception("GUI load failed")
                self.status_var.set("Load failed.")
                messagebox.showerror("Error", f"Load failed: {exc}")

        def scan_file(self):
            if self.engine is None:
                return
            self.status_var.set("Scanning...")
            payload = self.engine.random_scan()
            verdict = payload["verdict"]
            color = VERDICT_COLOR.get(verdict, SLATE)
            true_txt = "MALWARE" if payload["true_label"] == 1 else "BENIGN"
            self.lbl_result.config(
                text=(
                    f"{verdict}   (score {payload['score']:.3f} | "
                    f"P(mal) {payload['p_malware']:.3f} | sim {payload['max_memory_similarity']:.3f})"
                ),
                fg=color,
            )
            self.ax.clear()
            names = sorted(payload["group_threat"])
            vals = [payload["group_threat"][n] for n in names]
            self.ax.barh(names, vals, color=color)
            self.ax.set_title(
                f"{payload['window']} row {payload['index']} — true label: {true_txt}",
                fontsize=10,
            )
            self.ax.set_xlabel("mean |z-scored| group activation")
            self.fig.tight_layout()
            self.canvas.draw()
            self.status_var.set(
                f"Scan complete — verdict {verdict} (thresholds {payload['thresholds'][1]:.3f}/"
                f"{payload['thresholds'][0]:.3f})"
            )

    root = tk.Tk()
    MalwareScannerApp(root)
    root.mainloop()


def main(argv=None) -> int:
    import argparse

    from .run import load_config

    parser = argparse.ArgumentParser(prog="armada.gui", description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_gui(cfg, args.seed)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
