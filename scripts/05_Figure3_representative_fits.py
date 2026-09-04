#!/usr/bin/env python3
"""Generate manuscript Figure 3 independently from bundled fit results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.plot_utils import configure_matplotlib  # noqa: E402
from common.project_paths import output_path, resolve_result_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-source", choices=("reference", "outputs", "auto"), default="reference")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = args.output.expanduser().resolve() if args.output else output_path("figures", "Figure3_representative_fits.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    configure_matplotlib(8)

    representatives = ["W072", "IT04", "JP06", "KR06", "UK03", "US04", "ZA04"]
    intl_summary = pd.read_csv(resolve_result_file("international_fit_summary.csv", args.results_source))
    intl_curves = pd.read_csv(resolve_result_file("international_fit_curves.csv", args.results_source))
    china_summary = pd.read_csv(resolve_result_file("china_fit_summary_111_three_models.csv", args.results_source))
    china_curves = pd.read_csv(resolve_result_file("china_fit_curves_111_three_models.csv", args.results_source))

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2)); flat = axes.ravel()
    for index, wave_id in enumerate(representatives):
        axis = flat[index]
        if wave_id.startswith("W"):
            row = china_summary[china_summary.wave_id == wave_id].iloc[0]
            data = china_curves[china_curves.wave_id == wave_id].sort_values("day")
            label = "China (Shanghai)"
        else:
            row = intl_summary[intl_summary.wave_id == wave_id].iloc[0]
            data = intl_curves[intl_curves.wave_id == wave_id].sort_values("day")
            label = row.country
        axis.scatter(data.day, data.observed, s=10, color="black", alpha=.65,
                     label="observed 7-day mean" if index == 0 else None, zorder=3)
        axis.plot(data.day, data.classic_pred, color="#777777", linewidth=1.1,
                  label="classic SEIR" if index == 0 else None)
        axis.plot(data.day, data.reservoir_pred, color="#0072B2", linewidth=1.3,
                  label="AR-SEIR" if index == 0 else None)
        axis.plot(data.day, data.network_pred, color="#D55E00", linewidth=1.4,
                  label="network AR-SEIR" if index == 0 else None)
        axis.set_yscale("symlog", linthresh=1); axis.grid(alpha=.15)
        axis.set_title(f"{chr(65+index)}  {label}: {row.variant}", loc="left", weight="bold", fontsize=9)
        axis.text(.03, .94,
                  f"h={row.network_h_cv2:g}; ΔAICc$_{{R-N}}$={row.delta_aicc_reservoir_minus_network:.1f}",
                  transform=axis.transAxes, va="top", fontsize=7)
        axis.set_xlabel("day"); axis.set_ylabel("daily reported cases")
    flat[7].axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    flat[7].legend(handles, labels, loc="center", frameon=False, fontsize=9)
    flat[7].text(.5, .22,
                 "The fitted scale Q is a reporting-scale\naccessible population, not census population.",
                 ha="center", fontsize=8)
    fig.suptitle(
        "Representative fits show when dynamic recruitment and contact heterogeneity alter epidemic shape",
        fontsize=14, weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
