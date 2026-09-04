#!/usr/bin/env python3
"""Plot final S11 sensitivity to infection probability at first reservoir contact."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
DEFAULT_INPUT = RESULTS / "activation_probability_sensitivity_multicountry_final.csv"
DEFAULT_OUT = ROOT / "outputs" / "figures" / "FigureS9_activation_sensitivity.jpg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    d = pd.read_csv(args.input.expanduser().resolve())
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    waves = list(dict.fromkeys(d["wave_id"].astype(str).tolist()))
    fig, axs = plt.subplots(1, 3, figsize=(14.3, 4.9))

    for wid in waves:
        g = d.loc[d["wave_id"].astype(str).eq(wid)].sort_values("p_activation_infection")
        label = f"{g['country'].iloc[0]} ({wid})"
        axs[0].plot(
            g["p_activation_infection"], g["network_h_grid"], "o-", label=label
        )
        axs[1].plot(
            g["p_activation_infection"], g["peak_S_head_fraction_Q"], "o-"
        )
        axs[2].plot(
            g["p_activation_infection"], g["accessed_reservoir_fraction_Q"], "o-"
        )

    axs[0].set_title("A  Selected heterogeneity profile")
    axs[1].set_title("B  Peak high-risk headcount")
    axs[2].set_title("C  Reservoir accessed during wave")

    for ax in axs:
        ax.set_xlabel(r"infection probability on first reservoir contact, $p$")
        ax.set_xticks([0, .05, .10, .20, .30])
        ax.axvline(.10, ls="--", lw=.8, alpha=.65)
        ax.grid(alpha=.15)

    axs[0].set_ylabel(r"$h=\mathrm{CV}^2(Z)$")
    axs[0].set_yticks([0, .25, .5, 1, 2, 4])
    axs[1].set_ylabel(r"peak $S_{\mathrm{head}}/Q$")
    axs[2].set_ylabel("reservoir accessed fraction / Q")
    axs[0].legend(frameon=False, fontsize=6, loc="upper left")

    fig.suptitle(
        "Sensitivity to the assumed probability of infection on first reservoir contact",
        fontsize=13, weight="bold",
    )
    fig.text(
        .5, .015,
        r"Dashed line marks the frozen baseline assumption $p=0.10$.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=[0, .04, 1, .94])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
