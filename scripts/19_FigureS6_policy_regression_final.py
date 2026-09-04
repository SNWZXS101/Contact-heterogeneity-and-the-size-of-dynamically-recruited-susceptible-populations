#!/usr/bin/env python3
"""Generate final Figure S6 from recomputed policy regression coefficients."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "policy" / "policy_regression_coefficients_final.csv"
DEFAULT_OUT = ROOT / "outputs" / "figures" / "FigureS6_policy_regression.jpg"

OUTCOME_ORDER = [
    "peak_Shead_over_Q",
    "accessed_reservoir_fraction_Q",
    "q",
    "h",
    "network_AICc_support",
    "reservoir_AICc_support",
]
OUTCOME_LABEL = {
    "peak_Shead_over_Q": r"Peak $S_{\mathrm{head}}/Q$",
    "accessed_reservoir_fraction_Q": "Reservoir accessed fraction",
    "q": r"Transmission-decay rate $q$",
    "h": r"Activity heterogeneity $h$",
    "network_AICc_support": "AICc support for M2 over M1",
    "reservoir_AICc_support": "AICc support for M1 over M0",
}
PREDICTOR_ORDER = ["gating", "detection", "barrier", "vaccination"]
PREDICTOR_LABEL = ["gating", "detection", "barrier-information", "fully vaccinated"]
COLORS = {
    "gating": "#0072B2",
    "detection": "#009E73",
    "barrier": "#D55E00",
    "vaccination": "#CC79A7",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    d = pd.read_csv(args.input.expanduser().resolve())
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 3, figsize=(13.4, 7.6))
    for ax, outcome in zip(axs.ravel(), OUTCOME_ORDER):
        g = d.loc[d["outcome"].eq(outcome)].set_index("predictor").reindex(PREDICTOR_ORDER)
        y = np.arange(len(PREDICTOR_ORDER))
        ax.axvline(0, color="#777777", lw=.8)
        ax.errorbar(
            g["estimate"], y,
            xerr=[g["estimate"] - g["ci_low"], g["ci_high"] - g["estimate"]],
            fmt="none", ecolor="#555555", capsize=2, lw=1.0,
        )
        for i, predictor in enumerate(PREDICTOR_ORDER):
            ax.scatter(
                g.loc[predictor, "estimate"], i,
                color=COLORS[predictor], s=35, zorder=3
            )
        ax.set_yticks(y, PREDICTOR_LABEL)
        ax.invert_yaxis()
        ax.set_title(OUTCOME_LABEL[outcome], fontsize=8.5, weight="bold")
        ax.set_xlabel("standardised coefficient (HC3 95% CI)")
        ax.grid(axis="x", alpha=.15)

    fig.suptitle(
        "Descriptive policy associations after country and variant-era fixed effects",
        fontsize=13, weight="bold",
    )
    fig.text(
        .5, .012,
        "Twenty-four coefficients were examined; multiplicity and leave-one-country-out diagnostics are reported in the accompanying table.",
        ha="center", style="italic", fontsize=8.5,
    )
    fig.tight_layout(rect=[0, .04, 1, .96])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
