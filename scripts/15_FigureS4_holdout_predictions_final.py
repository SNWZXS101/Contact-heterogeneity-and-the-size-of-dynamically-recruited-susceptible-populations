#!/usr/bin/env python3
"""Plot final guarded retrospective within-wave tail extrapolations."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
DEFAULT_OUT = ROOT / "outputs" / "figures" / "FigureS4_holdout_predictions.jpg"
REPRESENTATIVES = ["IT04", "JP06", "KR06", "UK03", "US04", "ZA04"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    results = args.results_dir.expanduser().resolve()
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    cur = pd.read_csv(results / "international_holdout_curves.csv")
    met = pd.read_csv(results / "international_holdout_metrics.csv")

    needed = {
        "n_train_fitted", "n_train_nominal", "test_start_day",
        "reservoir_family_selected_train", "reservoir_family_test_log_rmse",
    }
    missing = needed.difference(met.columns)
    if missing:
        raise ValueError(f"Holdout metrics missing final guarded-design fields: {sorted(missing)}")

    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.8))

    for k, (ax, wid) in enumerate(zip(axs.ravel(), REPRESENTATIVES)):
        g = cur.loc[cur["wave_id"].astype(str).eq(wid)].sort_values("day")
        r = met.loc[met["wave_id"].astype(str).eq(wid)].iloc[0]

        fit_end = int(r["n_train_fitted"])
        nominal = int(r["n_train_nominal"])
        test_start = int(r["test_start_day"])
        last_day = int(g["day"].max())

        ax.axvspan(-.5, fit_end-.5, color="#E8E8E8", alpha=.70)
        ax.axvspan(fit_end-.5, test_start-.5, color="#F4F4F4", alpha=.95, hatch="////", edgecolor="#B0B0B0")
        ax.axvspan(test_start-.5, last_day+.5, color="#F8F8F8", alpha=.55)
        ax.axvline(nominal-.5, ls=":", lw=.85, c="black")
        ax.axvline(test_start-.5, ls="--", lw=.90, c="black")

        ax.scatter(g["day"], g["observed"], s=9, c="black", alpha=.66, zorder=4)
        ax.plot(g["day"], g["classic_pred"], color="#777777", lw=1.0)
        ax.plot(g["day"], g["reservoir_pred"], color="#0072B2", lw=1.1)
        ax.plot(g["day"], g["network_pred"], color="#D55E00", lw=1.2)

        family = "M1" if str(r["reservoir_family_selected_train"]) == "reservoir" else "M2"
        ax.set_yscale("symlog", linthresh=1)
        ax.set_title(
            f"{chr(65+k)}  {r['country']}: {r['variant']}\n"
            f"test RMSE M0/M1/M2="
            f"{r['classic_test_log_rmse']:.2f}/"
            f"{r['reservoir_test_log_rmse']:.2f}/"
            f"{r['network_test_log_rmse']:.2f}; "
            f"train-selected family={family} ({r['reservoir_family_test_log_rmse']:.2f})",
            loc="left", fontsize=7.8, weight="bold",
        )
        ax.set_xlabel("day")
        ax.set_ylabel("reported cases")
        ax.grid(alpha=.15)

    legend_items = [
        Patch(facecolor="#E8E8E8", alpha=.70, label="fitted centred observations"),
        Patch(facecolor="#F4F4F4", alpha=.95, hatch="////", edgecolor="#B0B0B0", label="smoothing buffer"),
        Patch(facecolor="#F8F8F8", alpha=.55, label="scored held-out tail"),
        Line2D([], [], marker="o", ls="", c="black", markersize=4, label="observed"),
        Line2D([], [], color="#777777", lw=1.2, label="M0 classic SEIR"),
        Line2D([], [], color="#0072B2", lw=1.2, label="M1 homogeneous RA-SEIR"),
        Line2D([], [], color="#D55E00", lw=1.2, label="M2 activity-stratified RA-SEIR"),
        Line2D([], [], c="black", ls=":", lw=.9, label="nominal 70% split"),
        Line2D([], [], c="black", ls="--", lw=.9, label="test scoring begins"),
    ]
    fig.legend(
        handles=legend_items, ncol=5, frameon=False, loc="lower center",
        fontsize=7.5, bbox_to_anchor=(.5, .005),
    )
    fig.suptitle(
        "Guarded retrospective within-wave tail extrapolation",
        fontsize=13.5, weight="bold", y=.995,
    )
    fig.tight_layout(rect=[0, .09, 1, .965])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
