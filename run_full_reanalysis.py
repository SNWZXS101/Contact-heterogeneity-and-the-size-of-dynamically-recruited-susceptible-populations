#!/usr/bin/env python3
"""Run the full 42-wave fit, 111-wave analysis, profiles, figures, and tables."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--quick", action="store_true", help="smoke-test optimisation budgets")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--china-mode", choices=("reproduce", "refit-all"), default="reproduce")
args = parser.parse_args()

common = []
if args.quick:
    common.append("--quick")
if args.overwrite:
    common.append("--overwrite")

solver_args = ["--steps", "8", "--h-grid", "0,0.25,0.5,1,2,4"]

commands = [
    [sys.executable, str(SCRIPTS / "01_fit_42_international_waves.py"), *solver_args, *common],
    [sys.executable, str(SCRIPTS / "02_analyse_china_111_waves.py"), "--mode", args.china_mode, *solver_args, *common],
    [sys.executable, str(SCRIPTS / "03_Figure1_framework.py")],
    [sys.executable, str(SCRIPTS / "04_Figure2_multicountry_trajectories.py"), "--results-source", "outputs"],
    [sys.executable, str(SCRIPTS / "05_Figure3_representative_fits.py"), "--results-source", "outputs"],
    [sys.executable, str(SCRIPTS / "06_Figure4_model_evidence.py"), "--results-source", "outputs"],
    [sys.executable, str(SCRIPTS / "07_profile_heterogeneity_appendix.py"), "--results-source", "outputs",
     "--steps", "8", "--h-grid", "0,0.25,0.5,1,2,4", *( ["--quick"] if args.quick else [] )],
    [sys.executable, str(SCRIPTS / "08_generate_complete_153_wave_tables.py"), "--results-source", "outputs"],
]

for command in commands:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
print("\nFull pipeline completed. See outputs/.")
