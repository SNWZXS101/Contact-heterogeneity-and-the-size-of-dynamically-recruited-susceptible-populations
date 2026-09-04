#!/usr/bin/env python3
"""Build final S10 counterfactual outputs, figures, and table."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

commands = [
    [sys.executable, str(SCRIPTS / "21_recompute_counterfactuals_final.py")],
    [sys.executable, str(SCRIPTS / "22_FigureS7_network_states_final.py")],
    [sys.executable, str(SCRIPTS / "23_FigureS8_counterfactual_heatmaps_final.py")],
    [sys.executable, str(SCRIPTS / "24_generate_S10_table.py")],
]

for command in commands:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

print("S10 assets completed.")
