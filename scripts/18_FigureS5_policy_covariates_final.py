#!/usr/bin/env python3
"""Generate final Figure S5 from the recomputed S9 policy/model merge."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "policy" / "international_model_policy_merged_final.csv"
DEFAULT_OUT = ROOT / "outputs" / "figures" / "FigureS5_policy_covariates.jpg"

COLS = [
    "gating",
    "detection",
    "barrier",
    "vaccination",
    "peak_Shead_over_Q",
    "h",
]
LABELS = [
    "gating",
    "detection",
    "barrier-\ninformation",
    "fully\nvaccinated",
    "peak\n$S_{head}/Q$",
    "$h$",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    d = pd.read_csv(args.input.expanduser().resolve())
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(
        1, 2, figsize=(13.2, 6.2),
        gridspec_kw={"width_ratios": [1.15, 1.05]}
    )

    # Country medians of the actual predictors used in the regression.
    mat = d.groupby("country")[COLS[:4]].median()
    im = axs[0].imshow(mat.values, aspect="auto", vmin=0, vmax=100, cmap="YlGnBu")
    axs[0].set_yticks(range(len(mat)), mat.index)
    axs[0].set_xticks(range(4), LABELS[:4], rotation=18, ha="right")
    axs[0].set_title("A  Median predictor context across analysed waves", loc="left", weight="bold")
    for i in range(len(mat)):
        for j in range(4):
            value = mat.iloc[i, j]
            axs[0].text(
                j, i, f"{value:.0f}", ha="center", va="center", fontsize=8,
                color="white" if value > 60 else "black"
            )
    fig.colorbar(im, ax=axs[0], fraction=.035, pad=.02, label="policy or vaccination scale")

    # Variant-era medians of predictors plus two fitted outcomes, standardized
    # column-wise for visual comparison only.
    v = d.groupby("variant_era")[COLS].median()
    sd = v.std(ddof=0).replace(0, 1)
    z = (v - v.mean()) / sd
    im2 = axs[1].imshow(z.values, aspect="auto", vmin=-2, vmax=2, cmap="RdBu_r")
    axs[1].set_yticks(range(len(v)), v.index)
    axs[1].set_xticks(range(len(COLS)), LABELS, rotation=28, ha="right")
    axs[1].set_title("B  Variant-era medians (column-standardised)", loc="left", weight="bold")
    fig.colorbar(im2, ax=axs[1], fraction=.035, pad=.02, label="standard deviations")

    fig.suptitle(
        "Policy predictors are strongly patterned by country, calendar time, vaccination, and variant era",
        fontsize=13, weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
