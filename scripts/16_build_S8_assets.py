#!/usr/bin/env python3
"""Build all final S8 tables and supplementary figures from frozen outputs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

commands = [
    [sys.executable, str(SCRIPTS / "12_generate_S8_tables.py")],
    [sys.executable, str(SCRIPTS / "13_FigureS2_all_international_fits_final.py")],
    [sys.executable, str(SCRIPTS / "14_FigureS3_network_profiles_final.py")],
    [sys.executable, str(SCRIPTS / "15_FigureS4_holdout_predictions_final.py")],
]

for command in commands:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

print("S8 assets completed.")
