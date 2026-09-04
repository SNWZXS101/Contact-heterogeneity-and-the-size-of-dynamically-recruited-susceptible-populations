#!/usr/bin/env python3
"""Plot final all-42 international fits from frozen Phase-2 results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
DEFAULT_OUT = ROOT / "outputs" / "figures" / "FigureS2_all_international_fits.jpg"

COUNTRY_ORDER = [
    "Italy", "Japan", "South Africa", "South Korea",
    "United Kingdom", "United States",
]
MODEL_LABEL = {"classic": "M0", "reservoir": "M1", "network": "M2"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    results = args.results_dir.expanduser().resolve()
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    cur = pd.read_csv(results / "international_fit_curves.csv")
    summ = pd.read_csv(results / "international_fit_summary.csv")

    if summ["wave_id"].nunique() != 42:
        raise ValueError("Expected 42 international waves")

    summ = summ.copy()
    summ["country_order"] = pd.Categorical(
        summ["country"], categories=COUNTRY_ORDER, ordered=True
    )
    if "start" in summ.columns:
        summ["_start"] = pd.to_datetime(summ["start"], errors="coerce")
        summ = summ.sort_values(["country_order", "_start", "wave_id"])
    else:
        summ = summ.sort_values(["country_order", "wave_id"])

    order = summ["wave_id"].astype(str).tolist()
    fig, axs = plt.subplots(7, 6, figsize=(15.5, 17.3))
    axs = axs.ravel()

    for ax, wid in zip(axs, order):
        g = cur.loc[cur["wave_id"].astype(str).eq(wid)].sort_values("day")
        r = summ.loc[summ["wave_id"].astype(str).eq(wid)].iloc[0]
        scale = max(float(g["observed"].max()), 1.0)
        h_col = "network_h_grid" if "network_h_grid" in r.index else "network_h_cv2"
        h = float(r[h_col])
        winner = MODEL_LABEL.get(str(r["winner"]), str(r["winner"]))

        ax.scatter(g["day"], g["observed"]/scale, s=3.2, c="black", alpha=.62, zorder=4)
        ax.plot(g["day"], g["classic_pred"]/scale, color="#777777", lw=.55, alpha=.80)
        ax.plot(g["day"], g["reservoir_pred"]/scale, color="#0072B2", lw=.70)
        ax.plot(g["day"], g["network_pred"]/scale, color="#D55E00", lw=.78)
        ax.set_ylim(-.025, 1.16)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{wid} · {r['country']}\n{r['variant']} · h={h:g} · {winner}",
            fontsize=6.35, loc="left", pad=2.0
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axs[len(order):]:
        ax.axis("off")

    handles = [
        Line2D([], [], marker="o", ls="", markersize=3.5, c="black", label="observed centred 7-day mean"),
        Line2D([], [], color="#777777", lw=1.2, label="M0 classic SEIR"),
        Line2D([], [], color="#0072B2", lw=1.2, label="M1 homogeneous RA-SEIR"),
        Line2D([], [], color="#D55E00", lw=1.2, label="M2 activity-stratified RA-SEIR"),
    ]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="lower center", fontsize=8)
    fig.suptitle("All 42 international wave fits", fontsize=14, weight="bold", y=.995)
    fig.text(
        .5, .027,
        "Each panel is normalised by its observed peak; h is the final fitted activity-heterogeneity profile value.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=[0, .045, 1, .975], h_pad=.82, w_pad=.58)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
