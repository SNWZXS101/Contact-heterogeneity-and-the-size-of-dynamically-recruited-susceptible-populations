#!/usr/bin/env python3
"""Refit and plot appendix network-heterogeneity profiles.

By default this script reproduces the 12-wave international profile panel used
for the appendix. It can also profile selected Chinese waves or both datasets.
The profile is recomputed from observed wave data; it is not merely replotted
from the bundled reference CSV.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.data_loaders import load_china_inputs, load_international_inputs  # noqa: E402
from common.io_utils import atomic_to_csv, parse_float_grid  # noqa: E402
from common.model_core import (  # noqa: E402
    FitResult,
    P_ACT_DEFAULT,
    fit_network_profile,
    inverse_s0_transform,
    parameter_summary,
)
from common.plot_utils import configure_matplotlib  # noqa: E402
from common.project_paths import output_path, resolve_result_file  # noqa: E402


def _fit_from_summary(row: pd.Series) -> np.ndarray:
    return np.asarray([
        math.log(float(row.network_Q)),
        math.log(float(row.network_beta0)),
        math.log(float(row.network_q)),
        math.log(float(row.network_a)),
        inverse_s0_transform(float(row.network_s0)),
    ], dtype=float)


def _select_default_international(summary: pd.DataFrame) -> List[str]:
    selected = pd.concat([
        summary.nlargest(8, "delta_aicc_reservoir_minus_network"),
        summary.nsmallest(4, "delta_aicc_reservoir_minus_network"),
    ]).drop_duplicates("wave_id").head(12)
    return selected["wave_id"].astype(str).tolist()


def _load_requested_data(scope: str, result_source: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rows: List[pd.DataFrame] = []
    y_map: Dict[str, np.ndarray] = {}
    if scope in {"international", "both"}:
        daily, _ = load_international_inputs()
        summary = pd.read_csv(resolve_result_file("international_fit_summary.csv", result_source))
        summary = summary.copy(); summary["profile_scope"] = "international"
        summary["location"] = summary["country"]
        rows.append(summary)
        for wave_id, group in daily.groupby("wave_id"):
            y_map[str(wave_id)] = group.sort_values("date")["observed_cases"].to_numpy(dtype=float)
    if scope in {"china", "both"}:
        _, china_y, _ = load_china_inputs()
        summary = pd.read_csv(resolve_result_file("china_fit_summary_111_three_models.csv", result_source))
        summary = summary.copy(); summary["profile_scope"] = "china"
        summary["location"] = summary["province"]
        rows.append(summary)
        y_map.update(china_y)
    return pd.concat(rows, ignore_index=True, sort=False), y_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("international", "china", "both"), default="international")
    parser.add_argument("--wave-ids", default="",
                        help="comma-separated IDs; otherwise manuscript selections are used")
    parser.add_argument("--h-grid", default="0,0.25,0.5,1,2,4")
    parser.add_argument("--p", type=float, default=P_ACT_DEFAULT)
    parser.add_argument("--classes", type=int, default=12)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--results-source", choices=("reference", "outputs", "auto"), default="reference")
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--figure-output", type=Path, default=None)
    args = parser.parse_args()

    if not 0.0 < args.p < 1.0:
        raise SystemExit("--p must be between 0 and 1")
    h_grid = parse_float_grid(args.h_grid)
    summary, y_map = _load_requested_data(args.scope, args.results_source)

    if args.wave_ids:
        selected = [item.strip() for item in args.wave_ids.split(",") if item.strip()]
    elif args.scope == "international":
        selected = _select_default_international(summary)
    elif args.scope == "china":
        selected = [wave for wave in ("W057", "W074", "W095", "X003", "X009")
                    if wave in set(summary.wave_id.astype(str))]
    else:
        international = summary[summary.profile_scope == "international"]
        selected = _select_default_international(international)
        selected += [wave for wave in ("W057", "W074", "W095", "X003", "X009")
                     if wave in set(summary.wave_id.astype(str))]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise SystemExit("No waves selected")

    summary_index = summary.set_index("wave_id", drop=False)
    missing = [wave for wave in selected if wave not in summary_index.index or wave not in y_map]
    if missing:
        raise KeyError(f"Selected waves not found in data/results: {missing}")

    rows: List[dict] = []
    for index, wave_id in enumerate(selected, 1):
        row = summary_index.loc[wave_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        y = y_map[wave_id]
        x = _fit_from_summary(row)
        dummy = FitResult(
            "reservoir", x, np.empty(0), np.nan, np.nan, np.nan,
            5, True, 0,
        )
        best, profiles = fit_network_profile(
            y, dummy, h_grid=h_grid, p_act=args.p, m=args.classes,
            starts_per_h=1 if args.quick else 2,
            max_nfev=45 if args.quick else 220,
            steps=args.steps, seed=31000 + index,
        )
        finite_aicc = [fit.aicc for fit in profiles if np.isfinite(fit.aicc)]
        use_aicc = bool(finite_aicc)
        minimum = min(finite_aicc) if use_aicc else min(fit.aic for fit in profiles)
        for fit in profiles:
            params = parameter_summary(fit, y, args.p)
            criterion = fit.aicc if use_aicc else fit.aic
            rows.append({
                "profile_scope": row.profile_scope,
                "wave_id": wave_id,
                "location": row.location,
                "variant": row.variant,
                "h_grid": fit.h_target,
                "h_realized": fit.h_realized,
                "H_second_moment": fit.z_second_moment,
                "sse_log1p": fit.sse,
                "aic": fit.aic,
                "aicc": fit.aicc,
                "profile_criterion": "AICc" if use_aicc else "AIC",
                "delta_information_criterion": criterion - minimum,
                "Q": params["Q"], "beta0": params["beta0"], "q": params["q"],
                "a": params["a"], "s0": params["s0"], "u0": params["u0"],
                "R0_network": params["R0"],
                "best_h_this_profile": best.h_target,
            })
        print(f"[{index:02d}/{len(selected)}] {wave_id} best h={best.h_target:g}", flush=True)

    profile_df = pd.DataFrame(rows)
    csv_out = args.csv_output.expanduser().resolve() if args.csv_output else output_path(
        "results", "appendix_heterogeneity_profiles.csv"
    )
    figure_out = args.figure_output.expanduser().resolve() if args.figure_output else output_path(
        "figures", "FigureS3_network_profiles.jpg"
    )
    atomic_to_csv(profile_df, csv_out)

    configure_matplotlib(8)
    n = len(selected)
    ncols = min(4, max(1, n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1*ncols, 2.8*nrows), squeeze=False)
    flat = axes.ravel()
    for axis, wave_id in zip(flat, selected):
        group = profile_df[profile_df.wave_id == wave_id].sort_values("h_grid")
        meta = group.iloc[0]
        axis.plot(group.h_grid, group.delta_information_criterion, "o-", color="#D55E00")
        axis.axhline(2, color="#999999", linestyle="--", linewidth=.7)
        axis.set_xticks(group.h_grid)
        axis.set_ylim(bottom=-.2)
        axis.set_title(f"{wave_id} {meta.location}\n{meta.variant}", fontsize=8, loc="left")
        axis.set_xlabel(r"$h=CV^2$")
        axis.set_ylabel(f"Δ{meta.profile_criterion} from best")
        axis.grid(alpha=.15)
    for axis in flat[n:]:
        axis.axis("off")
    fig.suptitle(
        "Network heterogeneity profiles: sharp minima indicate information; boundary minima indicate weak identification",
        fontsize=13, weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, .95])
    figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(csv_out)
    print(figure_out)


if __name__ == "__main__":
    main()
