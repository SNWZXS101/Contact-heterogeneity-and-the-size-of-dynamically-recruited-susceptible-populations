#!/usr/bin/env python3
"""Plot final fitted network-state trajectories for six international representatives."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
DEFAULT_OUT = ROOT / "outputs" / "figures" / "FigureS7_network_states.jpg"
REPS = ["IT04", "JP06", "KR06", "UK03", "US04", "ZA04"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    results = args.results_dir.expanduser().resolve()
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    states = pd.read_csv(results / "international_network_states.csv")
    summary = pd.read_csv(results / "international_fit_summary.csv")

    fig, axs = plt.subplots(2, 3, figsize=(13.2, 7.6))
    for k, (ax, wid) in enumerate(zip(axs.ravel(), REPS)):
        g = states.loc[states["wave_id"].astype(str).eq(wid)].sort_values("day")
        r = summary.loc[summary["wave_id"].astype(str).eq(wid)].iloc[0]
        Q = float(r["network_Q"])
        h_col = "network_h_grid" if "network_h_grid" in r.index else "network_h_cv2"
        h = float(r[h_col])

        ax.plot(g["day"], g["U_head"]/Q, color="#4C78A8", lw=1.2)
        ax.plot(g["day"], g["S_head"]/Q, color="#0072B2", lw=1.25)
        ax.plot(g["day"], g["S_edge"]/Q, color="#D55E00", lw=1.25)
        ax.set_ylim(bottom=0)
        ax.set_title(
            f"{chr(65+k)}  {r['country']}: {r['variant']}; h={h:g}",
            loc="left", weight="bold", fontsize=8,
        )
        ax.set_xlabel("day")
        ax.set_ylabel("fraction of fitted Q")
        ax.grid(alpha=.15)

    handles = [
        plt.Line2D([], [], color="#4C78A8", lw=1.3, label="remaining reservoir U/Q"),
        plt.Line2D([], [], color="#0072B2", lw=1.3, label=r"high-risk headcount $S_{\rm head}/Q$"),
        plt.Line2D([], [], color="#D55E00", lw=1.3, label=r"edge-weighted burden $S_{\rm edge}/Q$"),
    ]
    fig.legend(handles=handles, ncol=3, frameon=False, loc="lower center")
    fig.suptitle(
        "Fitted activity-stratified states separate remaining reservoir, high-risk people, and high-risk edges",
        fontsize=13, weight="bold",
    )
    fig.tight_layout(rect=[0, .055, 1, .96])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
