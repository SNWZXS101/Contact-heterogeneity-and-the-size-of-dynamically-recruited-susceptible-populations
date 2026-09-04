#!/usr/bin/env python3
"""Build all final S9 policy-analysis assets from frozen outputs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

commands = [
    [sys.executable, str(SCRIPTS / "17_recompute_policy_analysis.py")],
    [sys.executable, str(SCRIPTS / "18_FigureS5_policy_covariates_final.py")],
    [sys.executable, str(SCRIPTS / "19_FigureS6_policy_regression_final.py")],
]

for command in commands:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

print("S9 policy assets completed.")
