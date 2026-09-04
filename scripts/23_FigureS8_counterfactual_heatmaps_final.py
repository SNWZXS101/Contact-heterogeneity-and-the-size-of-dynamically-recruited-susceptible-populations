#!/usr/bin/env python3
"""Plot final S10 mechanism-counterfactual heatmaps."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
DEFAULT_OUT = ROOT / "outputs" / "figures" / "FigureS8_counterfactual_heatmaps.jpg"

BASELINE = "Baseline observed-response fit"
SCENARIO_ORDER = [
    "Reservoir gating (a -50%)",
    "Broad contact reduction (beta -30%)",
    "Faster detection/isolation (gamma +50%)",
    "Target highest-activity quartile (top 3 classes -50%)",
    "Immune protection (25% susceptible protected)",
    "Adaptive combined package",
]
SHORT = ["gating", "contact", "isolation", "target top quartile", "immunity", "combined"]
METRICS = [
    ("percent_reduction_peak_incidence", "Peak incidence"),
    ("percent_reduction_cumulative_incidence", "Cumulative incidence"),
    ("percent_reduction_max_S_head", r"Maximum $S_{\rm head}$"),
    ("percent_reduction_max_S_edge", r"Maximum $S_{\rm edge}$"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    results = args.results_dir.expanduser().resolve()
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(results / "policy_counterfactual_summary_final.csv")
    d = d.loc[~d["scenario"].eq(BASELINE)].copy()
    countries = list(dict.fromkeys(d["country"].astype(str).tolist()))

    fig, axs = plt.subplots(2, 2, figsize=(13.2, 9.1))
    for ax, (column, title) in zip(axs.ravel(), METRICS):
        p = (
            d.pivot(index="country", columns="scenario", values=column)
            .reindex(index=countries, columns=SCENARIO_ORDER)
        )
        im = ax.imshow(p.values, aspect="auto", vmin=-20, vmax=100, cmap="RdYlBu")
        ax.set_yticks(range(len(countries)), countries)
        ax.set_xticks(range(len(SCENARIO_ORDER)), SHORT, rotation=28, ha="right")
        ax.set_title(title, loc="left", weight="bold")
        for i in range(len(countries)):
            for j in range(len(SCENARIO_ORDER)):
                value = p.iloc[i, j]
                ax.text(
                    j, i, f"{value:.0f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if value > 75 else "black",
                )
        fig.colorbar(im, ax=ax, fraction=.035, pad=.02, label="% reduction")

    fig.suptitle(
        "Conditional mechanism counterfactuals across seven representative fitted waves",
        fontsize=13, weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
