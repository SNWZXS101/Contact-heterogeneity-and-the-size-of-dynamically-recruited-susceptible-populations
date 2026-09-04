#!/usr/bin/env python3
"""Generate manuscript Figure 4 independently from bundled model results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    out = args.output.expanduser().resolve() if args.output else output_path("figures", "Figure4_model_evidence.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    configure_matplotlib(8)

    international = pd.read_csv(resolve_result_file("international_fit_summary.csv", args.results_source))
    china = pd.read_csv(resolve_result_file("china_fit_summary_111_three_models.csv", args.results_source))
    holdout = pd.read_csv(resolve_result_file("international_holdout_metrics.csv", args.results_source))
    china["network_peak_head_fraction_Q"] = china.network_max_S_head / china.network_Q
    china["network_peak_edge_fraction_Q"] = china.network_max_S_edge / china.network_Q
    colors = {"classic": "#777777", "reservoir": "#0072B2", "network": "#D55E00"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axis = axes[0, 0]
    scopes = [("China local waves", china), ("International national waves", international)]
    bottom = np.zeros(2)
    for model in ("classic", "reservoir", "network"):
        values = [(data.winner == model).sum() for _, data in scopes]
        axis.bar([0, 1], values, bottom=bottom, label=model, color=colors[model])
        for x, value, base in zip([0, 1], values, bottom):
            axis.text(x, base + value/2, str(value), ha="center", va="center", color="white", weight="bold")
        bottom += values
    axis.set_xticks([0, 1], [label for label, _ in scopes]); axis.set_ylabel("number of waves")
    axis.set_title("A  AICc/AIC winner by spatial scale", loc="left", weight="bold")
    axis.legend(frameon=False, ncol=3, fontsize=7)

    axis = axes[0, 1]
    axis.axhline(2, color="#999999", linestyle="--", linewidth=.8); axis.axhline(0, color="#BBBBBB", linewidth=.6)
    axis.scatter(international.duration_days, international.delta_aicc_classic_minus_reservoir,
                 s=25, alpha=.65, label="classic - AR-SEIR", color="#0072B2")
    axis.scatter(international.duration_days, international.delta_aicc_reservoir_minus_network,
                 s=25, alpha=.65, label="AR-SEIR - network", color="#D55E00")
    axis.set_xlabel("wave duration (days)"); axis.set_ylabel("ΔAICc (positive favours richer model)")
    axis.set_title("B  Long national waves expose non-closed susceptible pools", loc="left", weight="bold")
    axis.legend(frameon=False, fontsize=7); axis.grid(alpha=.15)

    axis = axes[1, 0]
    countries = list(international.country.unique()); y = np.arange(len(countries))
    for index, country in enumerate(countries):
        data = international[international.country == country]
        head = data.network_peak_head_fraction_Q.median()
        edge = data.network_peak_edge_fraction_Q.median()
        axis.plot([head, edge], [index, index], color="#999999", linewidth=1.5)
        axis.scatter(head, index, color="#0072B2", s=42, label="headcount fraction" if index == 0 else None)
        axis.scatter(edge, index, color="#D55E00", s=42, label="edge-weighted fraction" if index == 0 else None)
    axis.set_yticks(y, countries); axis.set_xlim(0, 1)
    axis.set_xlabel("median peak fraction of fitted Q")
    axis.set_title("C  People and transmission edges are not interchangeable", loc="left", weight="bold")
    axis.legend(frameon=False, fontsize=7); axis.grid(axis="x", alpha=.15)

    axis = axes[1, 1]
    country_holdout = holdout.groupby("country")[[f"{model}_test_log_rmse" for model in colors]].mean()
    x = np.arange(len(country_holdout)); width = .25
    for offset, model in enumerate(("classic", "reservoir", "network")):
        axis.bar(x + (offset-1)*width, country_holdout[f"{model}_test_log_rmse"], width,
                 label=model, color=colors[model])
    axis.set_xticks(x, country_holdout.index, rotation=25, ha="right")
    axis.set_ylabel("mean 30% holdout log-RMSE")
    axis.set_title("D  Predictive gains are heterogeneous across countries", loc="left", weight="bold")
    axis.legend(frameon=False, fontsize=7); axis.grid(axis="y", alpha=.15)

    fig.suptitle(
        "Model evidence differs between rapid local containment and prolonged national transmission",
        fontsize=14, weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
