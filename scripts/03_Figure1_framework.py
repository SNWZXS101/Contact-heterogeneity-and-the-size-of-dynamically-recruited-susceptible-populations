#!/usr/bin/env python3
"""Generate manuscript Figure 1 independently from bundled policy data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.plot_utils import configure_matplotlib  # noqa: E402
from common.project_paths import INTERNATIONAL_DATA_DIR, output_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = args.output.expanduser().resolve() if args.output else output_path("figures", "Figure1_framework.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    configure_matplotlib(9)

    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1], wspace=.23, hspace=.30)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    def box(x, y, w, h, text, face, fontsize=8.5):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=face, edgecolor="#333333", linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", weight="bold", fontsize=fontsize)

    def arrow(x1, y1, x2, y2, label="", radius=0, linestyle="-"):
        patch = FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
            linewidth=1.2, color="#444444", connectionstyle=f"arc3,rad={radius}",
            linestyle=linestyle,
        )
        ax.add_patch(patch)
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2+.15, label, ha="center", va="bottom", fontsize=7.5)

    box(.35, 7.9, 2.45, 1.05, "Latent susceptible\nreservoir, $U$", "#E8F2FF")
    box(3.45, 7.9, 2.45, 1.05, "High-risk accessible\nsusceptibles, $S$", "#FFF1D6")
    box(7.05, 7.9, 1.75, 1.05, "Exposed, $E$", "#FCE2E2")
    box(7.05, 5.95, 1.75, 1.05, "Infectious, $I$", "#F7C5C5")
    box(7.05, 4.0, 1.75, 1.05, "Removed, $R$", "#E7E7E7")
    arrow(2.8, 8.43, 3.45, 8.43); arrow(5.9, 8.43, 7.05, 8.43)
    ax.text(3.12, 9.02, r"$(1-p)a(t)U\Theta_I$", ha="center", fontsize=7.2)
    ax.text(6.48, 9.02, r"$\beta(t)S\Theta_I$", ha="center", fontsize=7.2)
    arrow(7.93, 7.9, 7.93, 7.0, r"$\sigma$")
    arrow(7.93, 5.95, 7.93, 5.05, r"$\gamma+\delta(t)$")
    arrow(2.2, 7.92, 7.05, 8.05, radius=.28)
    ax.text(4.8, 7.18, r"$p\,a(t)U\Theta_I$", ha="center", fontsize=7.2)

    box(.45, 5.25, 2.45, .95, "Reservoir gating\ntravel and gatherings\ncluster containment", "#DDEBF7", 7.0)
    box(3.45, 5.25, 2.45, .95, "Contact suppression\nmasks and distancing\nworkplace measures", "#E2F0D9", 7.0)
    box(.45, 2.85, 2.45, .95, "Rapid detection\ntesting and tracing\nisolation", "#FFF2CC", 7.0)
    box(3.45, 2.85, 2.45, .95, "Targeted protection\nvaccination\nhub shielding", "#F4CCCC", 7.0)
    arrow(1.68, 6.2, 1.68, 7.85, linestyle="--"); ax.text(1.42, 7.0, r"$\downarrow a$", fontsize=7.5)
    arrow(4.68, 6.2, 4.68, 7.85, linestyle="--"); ax.text(4.42, 7.0, r"$\downarrow \beta$", fontsize=7.5)
    arrow(2.9, 3.32, 7.0, 6.35, radius=.10, linestyle="--"); ax.text(5.35, 4.65, r"$\uparrow \delta$", fontsize=7.5)
    arrow(5.9, 3.32, 6.95, 7.95, radius=-.10, linestyle="--"); ax.text(6.45, 5.1, r"$\downarrow U,S,z$", fontsize=7.5)
    ax.text(.25, 9.65, "A", weight="bold", fontsize=13)
    ax.text(.55, 9.35, "Reservoir-activated network SEIR and intervention entry points", weight="bold", fontsize=11)
    ax.text(.5, 1.35, "Two estimands must be separated:", weight="bold")
    ax.text(.65, .88, r"Headcount: $S_{head}=Q\sum_j S_j$  (people)", fontsize=9)
    ax.text(.65, .43, r"Edge-weighted risk: $S_{edge}=Q\sum_j z_jS_j$  (transmission-equivalent contacts)", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_axis_off(); ax2.set_xlim(0, 10); ax2.set_ylim(0, 6)
    ax2.text(.05, 5.7, "B", weight="bold", fontsize=13)
    ax2.text(.55, 5.72, "Contact heterogeneity changes who enters the active pool", weight="bold")
    xs = np.linspace(1, 9, 12); sizes = np.linspace(70, 420, 12); activities = np.linspace(.2, 3.0, 12)
    for x, size, activity in zip(xs, sizes, activities):
        ax2.scatter(x, 2.7, s=size, alpha=.75, edgecolor="#333333", linewidth=.5,
                    color=plt.cm.viridis((activity-.2)/2.8))
        ax2.plot([5, x], [4.75, 2.7], linewidth=.5, alpha=.35, color="#666666")
    ax2.scatter(5, 4.75, s=450, marker="*", c="#C44E52", edgecolor="black", linewidth=.8)
    ax2.text(5, 5.3, "infectious contact source", ha="center", fontsize=8)
    ax2.text(1, 1.25, "low activity", ha="center"); ax2.text(9, 1.25, "high activity", ha="center")
    ax2.annotate(
        "few high-activity people can carry\na large fraction of transmission edges",
        xy=(8.5, 3.2), xytext=(5.7, .25), arrowprops=dict(arrowstyle="->"), fontsize=8,
    )

    ax3 = fig.add_subplot(gs[1, 1])
    policy = pd.read_csv(INTERNATIONAL_DATA_DIR / "international_policy_daily.csv", parse_dates=["date"])
    policy = policy[(policy.date >= "2020-01-01") & (policy.date <= "2022-12-31")]
    columns = ["reservoir_gating_index", "detection_index", "barrier_information_index", "targeted_protection_index"]
    labels = ["Reservoir\ngating", "Detection\nand tracing", "Barrier and\ninformation", "Targeted\nprotection"]
    order = ["China", "South Korea", "Japan", "Italy", "United Kingdom", "United States", "South Africa"]
    matrix = policy.groupby("country")[columns].mean().reindex(order)
    image = ax3.imshow(matrix.values, aspect="auto", vmin=0, vmax=100, cmap="YlGnBu")
    ax3.set_yticks(range(len(order)), order); ax3.set_xticks(range(4), labels)
    for i in range(len(order)):
        for j in range(4):
            value = matrix.iloc[i, j]
            ax3.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8,
                     color="white" if value > 62 else "black")
    ax3.set_title("C  Enacted national policy profiles, 2020-22 (OxCGRT scale)", loc="left", weight="bold")
    colorbar = fig.colorbar(image, ax=ax3, fraction=.035, pad=.02)
    colorbar.set_label("mean normalised policy score")
    fig.suptitle(
        "A dynamic susceptible reservoir links epidemic waves, contact networks, and policy levers",
        fontsize=14, weight="bold", y=.995,
    )
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
