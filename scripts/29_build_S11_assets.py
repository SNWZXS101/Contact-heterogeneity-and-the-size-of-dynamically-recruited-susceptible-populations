#!/usr/bin/env python3
"""Build final S11 activation-probability sensitivity outputs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

commands = [
    [sys.executable, str(SCRIPTS / "26_recompute_activation_sensitivity_final.py")],
    [sys.executable, str(SCRIPTS / "27_FigureS9_activation_sensitivity_final.py")],
    [sys.executable, str(SCRIPTS / "28_generate_S11_sensitivity_table.py")],
]

for command in commands:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

print("S11 sensitivity assets completed.")
