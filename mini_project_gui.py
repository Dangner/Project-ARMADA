"""Launcher kept for compatibility with the original asset name.

The scanner now lives in :mod:`armada.gui` and runs the ARMADA pipeline
(real EMBER + trained checkpoints only).  Behaviour is preserved: header,
"SCAN RANDOM FILE", verdict + feature-group chart, status bar.
"""

from armada.gui import main

if __name__ == "__main__":
    raise SystemExit(main())
