#!/usr/bin/env python3
"""Generate final Figure S10 China-specific context from frozen Phase-2 outputs.

This script does not refit any model. It reads the final frozen China summary
and curves from outputs/results.

Run from project root:
    python scripts/30_FigureS10_china_context_final.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
DEFAULT_OUT = ROOT / "outputs" / "figures" / "FigureS10_china_context.jpg"

MODEL_ORDER = ["classic", "reservoir", "network"]
MODEL_LABEL = {
    "classic": "M0 classic",
    "reservoir": "M1 reservoir",
    "network": "M2 activity-stratified",
}
MODEL_COLOR = {
    "classic": "#777777",
    "reservoir": "#0072B2",
    "network": "#D55E00",
}
VARIANT_COLOR = {
    "Alpha": "#4C78A8",
    "Delta": "#F28E2B",
    "Omicron": "#59A14F",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    results = args.results_dir.expanduser().resolve()
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(results / "china_fit_summary_111_three_models.csv")
    curves = pd.read_csv(results / "china_fit_curves_111_three_models.csv")

    if summary["wave_id"].nunique() != 111:
        raise ValueError("Expected 111 unique Chinese waves")

    fig, axs = plt.subplots(2, 2, figsize=(13.2, 8.7))

    # A. Winners by variant era.
    ct = (
        pd.crosstab(summary["variant"], summary["winner"])
        .reindex(columns=MODEL_ORDER, fill_value=0)
    )
    variant_order = [v for v in ["Alpha", "Delta", "Omicron"] if v in ct.index]
    variant_order += [v for v in ct.index if v not in variant_order]
    ct = ct.reindex(variant_order)

    bottom = np.zeros(len(ct), dtype=float)
    for model in MODEL_ORDER:
        axs[0, 0].bar(
            ct.index,
            ct[model].to_numpy(),
            bottom=bottom,
            label=MODEL_LABEL[model],
            color=MODEL_COLOR[model],
        )
        bottom += ct[model].to_numpy()

    axs[0, 0].set_title(
        "A  Model winners across 111 Chinese waves",
        loc="left",
        weight="bold",
    )
    axs[0, 0].set_ylabel("waves")
    axs[0, 0].legend(frameon=False, fontsize=7)

    # B. Information-criterion support for M1 over M0 versus duration.
    aicc = pd.to_numeric(
        summary["delta_aicc_classic_minus_reservoir"], errors="coerce"
    )
    aic = pd.to_numeric(
        summary["delta_aic_classic_minus_reservoir"], errors="coerce"
    )
    use_aicc = np.isfinite(aicc)
    support = aicc.where(use_aicc, aic)

    plot_df = summary.assign(_support=support, _aicc=use_aicc)
    for variant, group in plot_df.groupby("variant"):
        color = VARIANT_COLOR.get(str(variant), None)
        axs[0, 1].scatter(
            group["duration_days"],
            group["_support"],
            s=22,
            alpha=.65,
            label=str(variant),
            color=color,
        )
        fallback = group.loc[~group["_aicc"]]
        if len(fallback):
            axs[0, 1].scatter(
                fallback["duration_days"],
                fallback["_support"],
                s=42,
                marker="x",
                color=color,
                linewidths=1.2,
            )

    axs[0, 1].axhline(2, color="#999999", ls="--", lw=.9)
    axs[0, 1].set_xlabel("duration (days)")
    axs[0, 1].set_ylabel("information-criterion support: M0 minus M1")
    axs[0, 1].set_title(
        "B  Recruitment support versus wave duration",
        loc="left",
        weight="bold",
    )
    axs[0, 1].legend(frameon=False, fontsize=7)
    axs[0, 1].grid(alpha=.15)

    # C. W072 fit.
    row = summary.loc[summary["wave_id"].astype(str).eq("W072")].iloc[0]
    g = curves.loc[curves["wave_id"].astype(str).eq("W072")].sort_values("day")
    h_col = "network_h_grid" if "network_h_grid" in row.index else "network_h_cv2"
    h = float(row[h_col])

    ax = axs[1, 0]
    ax.scatter(
        g["day"],
        g["observed"],
        s=10,
        color="black",
        alpha=.75,
        zorder=4,
    )
    ax.plot(g["day"], g["classic_pred"], color=MODEL_COLOR["classic"], lw=1.0)
    ax.plot(g["day"], g["reservoir_pred"], color=MODEL_COLOR["reservoir"], lw=1.2)
    ax.plot(g["day"], g["network_pred"], color=MODEL_COLOR["network"], lw=1.25)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title(
        f"C  Shanghai Omicron W072: h={h:g}; winner=M2",
        loc="left",
        weight="bold",
    )
    ax.set_xlabel("day")
    ax.set_ylabel("reported cases")
    ax.grid(alpha=.15)

    # D. Peak headcount versus peak edge-weighted burden.
    ax = axs[1, 1]
    if "network_peak_head_fraction_Q" in summary.columns:
        head = pd.to_numeric(
            summary["network_peak_head_fraction_Q"], errors="coerce"
        )
        edge = pd.to_numeric(
            summary["network_peak_edge_fraction_Q"], errors="coerce"
        )
    else:
        head = (
            pd.to_numeric(summary["network_max_S_head"], errors="coerce")
            / pd.to_numeric(summary["network_Q"], errors="coerce")
        )
        edge = (
            pd.to_numeric(summary["network_max_S_edge"], errors="coerce")
            / pd.to_numeric(summary["network_Q"], errors="coerce")
        )

    h_values = pd.to_numeric(
        summary["network_h_grid"]
        if "network_h_grid" in summary.columns
        else summary["network_h_cv2"],
        errors="coerce",
    )

    sc = ax.scatter(
        head,
        edge,
        c=h_values,
        cmap="viridis",
        s=25,
        alpha=.78,
        edgecolors="none",
    )

    lim = max(1.0, float(np.nanmax([head.max(), edge.max()])))
    ax.plot([0, lim], [0, lim], color="#777777", ls="--", lw=.9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    # IMPORTANT: Matplotlib mathtext does not know manuscript macros such as
    # \Shead or \Sedge. Use explicit mathtext-compatible notation here.
    ax.set_xlabel(r"peak $S_{\mathrm{head}}/Q$")
    ax.set_ylabel(r"peak $S_{\mathrm{edge}}/Q$")
    ax.set_title(
        "D  Headcount and edge-weighted burden diverge with heterogeneity",
        loc="left",
        weight="bold",
    )
    ax.grid(alpha=.15)

    cbar = fig.colorbar(sc, ax=ax, fraction=.045, pad=.025)
    cbar.set_label(r"selected $h=\mathrm{CV}^2(Z)$")

    fig.suptitle(
        "China provides a rapid-containment contrast to prolonged national waves",
        fontsize=13.5,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
