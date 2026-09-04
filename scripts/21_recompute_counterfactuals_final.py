#!/usr/bin/env python3
"""Recompute final mechanism and heterogeneity counterfactuals from frozen fits.

This script deliberately does NOT import the legacy ``network_models.py``.
It uses the final activity-class constructor from ``common.model_core`` and a
counterfactual RK4 implementation with the same positivity-safe stage-flow
convention and eight substeps/day as the frozen fitting pipeline.

Run from project root:
    python scripts/21_recompute_counterfactuals_final.py

Inputs:
    outputs/results/international_fit_summary.csv
    outputs/results/international_fit_curves.csv
    outputs/results/china_fit_summary_111_three_models.csv
    outputs/results/china_fit_curves_111_three_models.csv

Outputs:
    outputs/results/policy_counterfactual_summary_final.csv
    outputs/results/policy_counterfactual_curves_final.csv
    outputs/results/network_heterogeneity_counterfactual_summary_final.csv
    outputs/results/network_heterogeneity_counterfactual_curves_final.csv
    outputs/results/policy_counterfactual_manifest_final.json
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

from common.model_core import (  # noqa: E402
    GAMMA,
    P_ACT_DEFAULT,
    SIGMA,
    make_activity_classes,
)

RESULTS = ROOT / "outputs" / "results"

REPS = ["W072", "IT04", "JP06", "KR06", "UK03", "US04", "ZA04"]
STRUCTURAL_ANCHOR = "X003"
H_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
STEPS = 8
N_CLASSES = 12
TOP_CLASS_COUNT = 3  # 3/12 = highest-activity quartile.

BASELINE = "Baseline observed-response fit"
GATING = "Reservoir gating (a -50%)"
CONTACT = "Broad contact reduction (beta -30%)"
ISOLATION = "Faster detection/isolation (gamma +50%)"
TARGET = "Target highest-activity quartile (top 3 classes -50%)"
IMMUNITY = "Immune protection (25% susceptible protected)"
COMBINED = "Adaptive combined package"

SCENARIOS = {
    BASELINE: dict(a=1.0, beta=1.0, gamma=1.0, top=1.0, susceptible=1.0),
    GATING: dict(a=0.5, beta=1.0, gamma=1.0, top=1.0, susceptible=1.0),
    CONTACT: dict(a=1.0, beta=0.7, gamma=1.0, top=1.0, susceptible=1.0),
    ISOLATION: dict(a=1.0, beta=1.0, gamma=1.5, top=1.0, susceptible=1.0),
    TARGET: dict(a=1.0, beta=1.0, gamma=1.0, top=0.5, susceptible=1.0),
    IMMUNITY: dict(a=1.0, beta=1.0, gamma=1.0, top=1.0, susceptible=0.75),
    COMBINED: dict(a=0.5, beta=0.75, gamma=1.5, top=0.5, susceptible=0.85),
}


def initial_state(params: dict, y0: float, weights: np.ndarray, susceptible_factor: float):
    """Construct the fitted baseline initial state, then apply protection.

    The fitted initial I state is constructed using the baseline GAMMA even in
    the faster-isolation counterfactual. This makes gamma the dynamical lever;
    it does not silently alter the starting epidemic state.
    """
    Q = float(params["Q"])
    obs0 = max(float(y0), 0.5)
    e0 = obs0 / (Q * SIGMA)
    i0 = obs0 / (Q * GAMMA)
    if e0 + i0 > 0.5:
        fac = 0.5 / (e0 + i0)
        e0 *= fac
        i0 *= fac

    s0_raw = float(params["s0"])
    s0 = min(s0_raw, max(1e-6, 1.0 - e0 - i0 - 1e-6))
    u0 = max(1e-8, 1.0 - s0 - e0 - i0)

    protected = (1.0 - susceptible_factor) * (u0 + s0)
    u0 *= susceptible_factor
    s0 *= susceptible_factor

    U = u0 * weights.copy()
    S = s0 * weights.copy()
    E = e0 * weights.copy()
    I = i0 * weights.copy()
    R = protected * weights.copy()
    return U, S, E, I, R, u0, s0, e0, i0, protected


def targeted_activity(z: np.ndarray, multiplier: float):
    z_eff = np.asarray(z, dtype=float).copy()
    targeted = np.zeros(z_eff.size, dtype=bool)
    if multiplier < 1.0:
        idx = np.argsort(z_eff)[-TOP_CLASS_COUNT:]
        z_eff[idx] *= multiplier
        targeted[idx] = True
    return z_eff, targeted


def simulate(
    params: dict,
    observed: np.ndarray,
    *,
    scenario: dict,
    h_override: float | None = None,
    initial_rinit_match: bool = False,
    steps: int = STEPS,
):
    h_target = float(params["h"] if h_override is None else h_override)
    z, w, h_real = make_activity_classes(h_target, N_CLASSES)
    z_eff, targeted = targeted_activity(z, float(scenario["top"]))

    # For the structural experiment only: because E[z^2]=1+h before
    # targeting, dividing both a and beta0 by 1+h holds the initial
    # rank-one reproduction quantity constant across h.
    structural_scale = 1.0 / (1.0 + h_real) if initial_rinit_match else 1.0

    Q = float(params["Q"])
    beta0 = float(params["beta0"]) * float(scenario["beta"]) * structural_scale
    q = float(params["q"])
    a = float(params["a"]) * float(scenario["a"]) * structural_scale
    gamma = GAMMA * float(scenario["gamma"])
    p_act = P_ACT_DEFAULT

    U, S, E, I, R, u0, s0, e0, i0, protected = initial_state(
        params, observed[0], w, float(scenario["susceptible"])
    )

    dt = 1.0 / steps
    t = 0.0
    cumulative_incidence_state = 0.0
    cumulative_infections = 0.0
    cumulative_recruited = 0.0
    cumulative_direct = 0.0
    rows = []

    def rhs(U_, S_, E_, I_, R_, tt):
        Up = np.maximum(U_, 0.0)
        Sp = np.maximum(S_, 0.0)
        Ep = np.maximum(E_, 0.0)
        Ip = np.maximum(I_, 0.0)

        theta = max(0.0, float(np.dot(z_eff, Ip)))
        beta = beta0 * math.exp(-q * tt)
        act = a * z_eff * Up * theta
        inf = beta * z_eff * Sp * theta

        return (
            -act,
            (1.0 - p_act) * act - inf,
            p_act * act + inf - SIGMA * Ep,
            SIGMA * Ep - gamma * Ip,
            gamma * Ip,
            float(SIGMA * Ep.sum()),
            float((p_act * act + inf).sum()),
            float(((1.0 - p_act) * act).sum()),
            float((p_act * act).sum()),
        )

    for day in range(len(observed)):
        c0 = cumulative_incidence_state
        for _ in range(steps):
            k1 = rhs(U, S, E, I, R, t)
            k2 = rhs(
                U + 0.5*dt*k1[0], S + 0.5*dt*k1[1],
                E + 0.5*dt*k1[2], I + 0.5*dt*k1[3],
                R + 0.5*dt*k1[4], t + 0.5*dt
            )
            k3 = rhs(
                U + 0.5*dt*k2[0], S + 0.5*dt*k2[1],
                E + 0.5*dt*k2[2], I + 0.5*dt*k2[3],
                R + 0.5*dt*k2[4], t + 0.5*dt
            )
            k4 = rhs(
                U + dt*k3[0], S + dt*k3[1],
                E + dt*k3[2], I + dt*k3[3],
                R + dt*k3[4], t + dt
            )

            U += dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6.0
            S += dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6.0
            E += dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6.0
            I += dt*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) / 6.0
            R += dt*(k1[4] + 2*k2[4] + 2*k3[4] + k4[4]) / 6.0

            for arr in (U, S, E, I, R):
                np.maximum(arr, 0.0, out=arr)

            mass = U + S + E + I + R
            factor = np.divide(w, mass, out=np.ones_like(w), where=mass > 0)
            U *= factor
            S *= factor
            E *= factor
            I *= factor
            R *= factor

            cumulative_incidence_state += dt*(k1[5] + 2*k2[5] + 2*k3[5] + k4[5]) / 6.0
            cumulative_infections += dt*(k1[6] + 2*k2[6] + 2*k3[6] + k4[6]) / 6.0
            cumulative_recruited += dt*(k1[7] + 2*k2[7] + 2*k3[7] + k4[7]) / 6.0
            cumulative_direct += dt*(k1[8] + 2*k2[8] + 2*k3[8] + k4[8]) / 6.0
            t += dt

        rows.append({
            "day": day,
            "incidence": Q * (cumulative_incidence_state - c0),
            "U_head": Q * U.sum(),
            "S_head": Q * S.sum(),
            "S_edge": Q * float(np.dot(z_eff, S)),
            "I_edge": Q * float(np.dot(z_eff, I)),
            "cumulative_infections": Q * cumulative_infections,
            "cumulative_recruited": Q * cumulative_recruited,
            "cumulative_direct": Q * cumulative_direct,
        })

    df = pd.DataFrame(rows)
    H_eff = float(np.dot(w, z_eff*z_eff))
    mean_z_eff = float(np.dot(w, z_eff))
    rinit = H_eff * (p_act*a*u0 + beta0*s0) / gamma

    meta = {
        "Q": Q,
        "h_target": h_target,
        "h_realized": h_real,
        "H_effective": H_eff,
        "mean_activity_effective": mean_z_eff,
        "targeted_class_count": int(targeted.sum()),
        "targeted_population_fraction": float(targeted.mean()),
        "Rinit_initial": rinit,
        "a_effective": a,
        "beta0_effective": beta0,
        "gamma_effective": gamma,
        "u0_effective": u0,
        "s0_effective": s0,
        "e0_initial": e0,
        "i0_initial": i0,
        "protected_initial": protected,
    }
    return df, meta


def load_representatives(results: Path):
    intl = pd.read_csv(results / "international_fit_summary.csv")
    icur = pd.read_csv(results / "international_fit_curves.csv")
    china = pd.read_csv(results / "china_fit_summary_111_three_models.csv")
    ccur = pd.read_csv(results / "china_fit_curves_111_three_models.csv")

    records = []
    requested = REPS + [STRUCTURAL_ANCHOR]
    for wid in requested:
        if wid.startswith(("W", "X")):
            r = china.loc[china["wave_id"].astype(str).eq(wid)].iloc[0]
            g = ccur.loc[ccur["wave_id"].astype(str).eq(wid)].sort_values("day")
            country = "China"
        else:
            r = intl.loc[intl["wave_id"].astype(str).eq(wid)].iloc[0]
            g = icur.loc[icur["wave_id"].astype(str).eq(wid)].sort_values("day")
            country = str(r["country"])

        h_col = "network_h_grid" if "network_h_grid" in r.index else "network_h_cv2"
        records.append({
            "wave_id": wid,
            "country": country,
            "variant": str(r["variant"]),
            "observed": g["observed"].to_numpy(dtype=float),
            "stored_network_pred": g["network_pred"].to_numpy(dtype=float),
            "params": {
                "Q": float(r["network_Q"]),
                "beta0": float(r["network_beta0"]),
                "q": float(r["network_q"]),
                "a": float(r["network_a"]),
                "s0": float(r["network_s0"]),
                "h": float(r[h_col]),
            },
        })
    return records


def percent_reduction(value: float, baseline: float) -> float:
    return 100.0 * (1.0 - value / max(float(baseline), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    results = args.results_dir.expanduser().resolve()

    records = load_representatives(results)
    by_id = {r["wave_id"]: r for r in records}

    summary_rows = []
    curve_rows = []
    baseline_qc = []

    for wid in REPS:
        rr = by_id[wid]
        baseline_metrics = None

        for scenario_name, scenario in SCENARIOS.items():
            df, meta = simulate(rr["params"], rr["observed"], scenario=scenario)

            metrics = {
                "peak_incidence": float(df["incidence"].max()),
                "peak_day": int(df["incidence"].idxmax()),
                "cumulative_incidence": float(df["incidence"].sum()),
                "max_S_head": float(df["S_head"].max()),
                "max_S_edge": float(df["S_edge"].max()),
                "final_recruited": float(df["cumulative_recruited"].iloc[-1]),
                "final_direct": float(df["cumulative_direct"].iloc[-1]),
            }
            if baseline_metrics is None:
                baseline_metrics = metrics.copy()

                # QC against the final frozen fitted M2 prediction.
                pred = df["incidence"].to_numpy(dtype=float)
                stored = rr["stored_network_pred"]
                baseline_qc.append({
                    "wave_id": wid,
                    "max_abs_log1p_difference_vs_frozen_network_pred": float(
                        np.max(np.abs(np.log1p(pred) - np.log1p(stored)))
                    ),
                })

            row = {
                "wave_id": wid,
                "country": rr["country"],
                "variant": rr["variant"],
                "scenario": scenario_name,
                **meta,
                **metrics,
            }
            for key in ("peak_incidence", "cumulative_incidence", "max_S_head", "max_S_edge"):
                row[f"relative_{key}"] = metrics[key] / max(baseline_metrics[key], 1e-12)
                row[f"percent_reduction_{key}"] = percent_reduction(
                    metrics[key], baseline_metrics[key]
                )

            summary_rows.append(row)
            temp = df.copy()
            temp["wave_id"] = wid
            temp["country"] = rr["country"]
            temp["variant"] = rr["variant"]
            temp["scenario"] = scenario_name
            curve_rows.append(temp)

    summary = pd.DataFrame(summary_rows)
    curves = pd.concat(curve_rows, ignore_index=True)

    # Structural heterogeneity experiment anchored to final X003 fit.
    anchor = by_id[STRUCTURAL_ANCHOR]
    structure_rows = []
    structure_curves = []
    for mode in ("fixed_parameters", "initial_Rinit_matched"):
        for h in H_GRID:
            df, meta = simulate(
                anchor["params"],
                anchor["observed"],
                scenario=SCENARIOS[BASELINE],
                h_override=h,
                initial_rinit_match=(mode == "initial_Rinit_matched"),
            )
            structure_rows.append({
                "wave_id": STRUCTURAL_ANCHOR,
                "mode": mode,
                "h": h,
                **meta,
                "peak_incidence": float(df["incidence"].max()),
                "cumulative_incidence": float(df["incidence"].sum()),
                "max_S_head": float(df["S_head"].max()),
                "max_S_edge": float(df["S_edge"].max()),
            })
            temp = df.copy()
            temp["wave_id"] = STRUCTURAL_ANCHOR
            temp["mode"] = mode
            temp["h"] = h
            structure_curves.append(temp)

    structure = pd.DataFrame(structure_rows)
    structure_curve = pd.concat(structure_curves, ignore_index=True)

    summary.to_csv(results / "policy_counterfactual_summary_final.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(results / "policy_counterfactual_curves_final.csv", index=False, encoding="utf-8-sig")
    structure.to_csv(
        results / "network_heterogeneity_counterfactual_summary_final.csv",
        index=False, encoding="utf-8-sig"
    )
    structure_curve.to_csv(
        results / "network_heterogeneity_counterfactual_curves_final.csv",
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(baseline_qc).to_csv(
        results / "counterfactual_baseline_reproduction_qc.csv",
        index=False, encoding="utf-8-sig"
    )

    nonbaseline = summary.loc[~summary["scenario"].eq(BASELINE)]
    median = (
        nonbaseline.groupby("scenario")[
            [
                "percent_reduction_peak_incidence",
                "percent_reduction_cumulative_incidence",
                "percent_reduction_max_S_head",
                "percent_reduction_max_S_edge",
            ]
        ]
        .median()
        .round(3)
    )

    matched = structure.loc[structure["mode"].eq("initial_Rinit_matched")].sort_values("h")
    h0 = matched.loc[np.isclose(matched["h"], 0.0)].iloc[0]
    h4 = matched.loc[np.isclose(matched["h"], 4.0)].iloc[0]
    structural_summary = {
        "anchor_wave": STRUCTURAL_ANCHOR,
        "h0_to_h4_percent_reduction_max_S_head": percent_reduction(
            h4["max_S_head"], h0["max_S_head"]
        ),
        "h0_to_h4_percent_reduction_max_S_edge": percent_reduction(
            h4["max_S_edge"], h0["max_S_edge"]
        ),
        "Rinit_range_initial_matched": [
            float(matched["Rinit_initial"].min()),
            float(matched["Rinit_initial"].max()),
        ],
    }

    manifest = {
        "representative_waves": REPS,
        "structural_anchor": STRUCTURAL_ANCHOR,
        "scenarios": list(SCENARIOS),
        "rk4_substeps_per_day": STEPS,
        "solver_stage_positivity": "nonnegative_flow_evaluation",
        "targeting_definition": (
            "highest three of 12 equal-mass activity classes (25% of model population) "
            "have activity halved; activity values are not renormalised"
        ),
        "faster_isolation_initial_state": (
            "initial fitted state held fixed; gamma changes only the post-baseline dynamics"
        ),
        "structural_h_grid": list(H_GRID),
        "structural_initial_Rinit_matching": "a and beta0 divided by 1+h",
        "median_percent_reductions": median.to_dict(orient="index"),
        "structural_summary": structural_summary,
        "caveat": "conditional mechanism counterfactuals, not historical causal effects",
    }
    (results / "policy_counterfactual_manifest_final.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
