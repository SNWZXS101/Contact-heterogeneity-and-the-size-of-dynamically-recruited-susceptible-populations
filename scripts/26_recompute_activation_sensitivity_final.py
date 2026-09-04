#!/usr/bin/env python3
"""Final sensitivity of representative-wave fits to activation-infection probability p.

This script uses the frozen ``common.model_core`` implementation:
- positivity-safe RK4 stage-flow evaluation;
- eight substeps/day;
- final common h grid {0,0.25,0.5,1,2,4};
- adaptive optimizer continuation;
- exact M1 / M2(h=0) nesting refinement.

The p=0.10 row is taken directly from the frozen Phase-2 fit rather than
refitted, so the central sensitivity point exactly reproduces the manuscript
source of truth.

Run from project root:
    python scripts/26_recompute_activation_sensitivity_final.py

Outputs:
    outputs/results/activation_probability_sensitivity_multicountry_final.csv
    outputs/results/activation_probability_sensitivity_manifest_final.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from common.fit_helpers import fitresult_from_profile_h0  # noqa: E402
from common.model_core import (  # noqa: E402
    FitResult,
    P_ACT_DEFAULT,
    fit_network_profile,
    fit_reservoir,
    inverse_s0_transform,
    parameter_summary,
    simulate_network_detailed,
)

RESULTS = ROOT / "outputs" / "results"
REPS = ["W072", "IT04", "JP06", "KR06", "UK03", "US04", "ZA04"]
PGRID = (0.0, 0.05, 0.10, 0.20, 0.30)
HGRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
STEPS = 8
M = 12


def warm_x(row: pd.Series) -> np.ndarray:
    return np.asarray([
        math.log(float(row.network_Q)),
        math.log(float(row.network_beta0)),
        math.log(float(row.network_q)),
        math.log(float(row.network_a)),
        inverse_s0_transform(float(row.network_s0)),
    ], dtype=float)


def frozen_network_fit(row: pd.Series) -> FitResult:
    h_target = float(row.get("network_h_grid", row.network_h_cv2))
    h_realized = float(row.network_h_cv2)
    return FitResult(
        model="network",
        x=warm_x(row),
        pred=np.empty(0),
        sse=float(row.network_sse_log1p),
        aic=float(row.network_aic),
        aicc=float(row.network_aicc),
        k=6,
        success=bool(row.network_success),
        nfev=int(row.network_nfev),
        h_target=h_target,
        h_realized=h_realized,
        z_second_moment=1.0 + h_realized,
    )


def load_wave_inputs(results: Path):
    intl = pd.read_csv(results / "international_fit_summary.csv")
    icur = pd.read_csv(results / "international_fit_curves.csv")
    china = pd.read_csv(results / "china_fit_summary_111_three_models.csv")
    ccur = pd.read_csv(results / "china_fit_curves_111_three_models.csv")

    out = {}
    for wid in REPS:
        if wid.startswith("W"):
            row = china.loc[china["wave_id"].astype(str).eq(wid)].iloc[0]
            curve = ccur.loc[ccur["wave_id"].astype(str).eq(wid)].sort_values("day")
            country = "China"
        else:
            row = intl.loc[intl["wave_id"].astype(str).eq(wid)].iloc[0]
            curve = icur.loc[icur["wave_id"].astype(str).eq(wid)].sort_values("day")
            country = str(row.country)
        out[wid] = {
            "row": row,
            "country": country,
            "variant": str(row.variant),
            "y": curve["observed"].to_numpy(dtype=float),
        }
    return out


def diagnostics(fit: FitResult, y: np.ndarray, p: float):
    params = parameter_summary(fit, y, p_act=p)
    detail = simulate_network_detailed(
        fit, y, h=fit.h_target, p_act=p, m=M, steps=STEPS
    )
    Q = float(params["Q"])
    recruited = np.asarray(detail["cumulative_recruited_to_S"], dtype=float)
    direct = np.asarray(detail["cumulative_direct_from_U"], dtype=float)
    return params, {
        "peak_S_head_fraction_Q": float(np.max(detail["S_head"]) / Q),
        "peak_S_edge_fraction_Q": float(np.max(detail["S_edge"]) / Q),
        "accessed_reservoir_fraction_Q": float((recruited[-1] + direct[-1]) / Q),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    results = args.results_dir.expanduser().resolve()

    waves = load_wave_inputs(results)
    rows = []

    for iw, wid in enumerate(REPS, 1):
        info = waves[wid]
        row = info["row"]
        y = info["y"]
        frozen_x = warm_x(row)
        previous_x = frozen_x.copy()

        print(f"[{iw}/{len(REPS)}] {wid} {info['country']}", flush=True)

        for p in PGRID:
            if abs(p - P_ACT_DEFAULT) < 1e-12:
                # Exact frozen central point.
                network = frozen_network_fit(row)
                reservoir_aicc = float(row.reservoir_aicc)
                reservoir_aic = float(row.reservoir_aic)
                reservoir_sse = float(row.reservoir_sse_log1p)
                reservoir_source = str(row.get("reservoir_fit_source", "frozen"))
                source = "frozen_phase2_fit"
            else:
                reservoir = fit_reservoir(
                    y,
                    p_act=p,
                    starts=4,
                    max_nfev=220,
                    steps=STEPS,
                    seed=50000 + 1000*iw + int(round(100*p)),
                    warm=[frozen_x, previous_x],
                )
                network, profile = fit_network_profile(
                    y,
                    reservoir,
                    h_grid=HGRID,
                    p_act=p,
                    m=M,
                    starts_per_h=1,
                    max_nfev=120,
                    steps=STEPS,
                    seed=60000 + 1000*iw + int(round(100*p)),
                )

                # Exact h=0 nesting refinement for the M1 comparator.
                reservoir_h0 = fitresult_from_profile_h0(profile, y)
                reservoir_source = "standalone"
                if reservoir_h0.sse < reservoir.sse:
                    reservoir = reservoir_h0
                    reservoir_source = "profile_h0"

                reservoir_aicc = float(reservoir.aicc)
                reservoir_aic = float(reservoir.aic)
                reservoir_sse = float(reservoir.sse)
                previous_x = network.x.copy()
                source = "refitted_sensitivity"

            params, diag = diagnostics(network, y, p)

            rows.append({
                "wave_id": wid,
                "country": info["country"],
                "variant": info["variant"],
                "p_activation_infection": p,
                "fit_source": source,
                "reservoir_fit_source": reservoir_source,
                "reservoir_sse_log1p": reservoir_sse,
                "reservoir_aic": reservoir_aic,
                "reservoir_aicc": reservoir_aicc,
                "network_sse_log1p": float(network.sse),
                "network_aic": float(network.aic),
                "network_aicc": float(network.aicc),
                "delta_reservoir_minus_network": reservoir_aicc - float(network.aicc),
                "network_h_grid": float(network.h_target),
                "network_h_cv2": float(network.h_realized),
                "Q": params["Q"],
                "beta0": params["beta0"],
                "q": params["q"],
                "a": params["a"],
                "s0": params["s0"],
                "u0": params["u0"],
                "Rinit": params["R0"],
                **diag,
            })
            print(
                f"  p={p:.2f} h={network.h_target:g} "
                f"peakS={diag['peak_S_head_fraction_Q']:.3f} "
                f"accessed={diag['accessed_reservoir_fraction_Q']:.3f}",
                flush=True,
            )

    out = pd.DataFrame(rows)
    output_csv = results / "activation_probability_sensitivity_multicountry_final.csv"
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    if not out["accessed_reservoir_fraction_Q"].between(-1e-8, 1.000001).all():
        bad = out.loc[
            ~out["accessed_reservoir_fraction_Q"].between(-1e-8, 1.000001),
            ["wave_id", "p_activation_infection", "accessed_reservoir_fraction_Q"]
        ]
        raise ValueError(f"Physically invalid reservoir-access diagnostics:\n{bad}")

    manifest = {
        "representative_waves": REPS,
        "p_grid": list(PGRID),
        "h_grid": list(HGRID),
        "activity_classes": M,
        "rk4_substeps_per_day": STEPS,
        "solver_stage_positivity": "nonnegative_flow_evaluation",
        "central_p_0p10_source": "exact frozen Phase-2 fit",
        "m1_h0_nesting_refinement": True,
        "n_rows": int(len(out)),
        "all_accessed_reservoir_fractions_physical": True,
    }
    (results / "activation_probability_sensitivity_manifest_final.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(output_csv)


if __name__ == "__main__":
    main()
