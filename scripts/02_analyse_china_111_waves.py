#!/usr/bin/env python3
"""Analyse all 111 Chinese COVID-19 waves with classic, AR-SEIR, and network AR-SEIR.

The default ``reproduce`` mode retains the archived classic/AR-SEIR fits for
Tang's original 101-wave cohort, then refines the nested network profile and
fits all three models anew to the ten post-May-2022 waves. This is the workflow
used for the manuscript. ``--mode refit-all`` independently refits all three
models to all 111 observed series.

Direct full run
---------------
    python scripts/02_analyse_china_111_waves.py --overwrite

Fast smoke test
---------------
    python scripts/02_analyse_china_111_waves.py --quick --limit 1 --overwrite --skip-holdout
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.data_loaders import load_china_inputs  # noqa: E402
from common.fit_helpers import (  # noqa: E402
    MODEL_NAMES,
    choose_winner,
    fitresult_from_profile_h0,
    prediction_metrics,
)
from common.io_utils import atomic_to_csv, atomic_write_json, parse_float_grid  # noqa: E402
from common.model_core import (  # noqa: E402
    FitResult,
    P_ACT_DEFAULT,
    fit_classic,
    fit_network_profile,
    fit_reservoir,
    information_criteria,
    inverse_s0_transform,
    make_activity_classes,
    parameter_summary,
    simulate_classic,
    simulate_network,
    simulate_network_detailed,
    simulate_reservoir,
)
from common.project_paths import OUTPUT_RESULTS_DIR  # noqa: E402

OUTPUT_FILES = {
    "summary": "china_fit_summary_111_three_models.csv",
    "curves": "china_fit_curves_111_three_models.csv",
    "profiles": "china_network_profile_111.csv",
}


def old_reservoir_x(row: pd.Series) -> np.ndarray:
    return np.asarray([
        math.log(float(row.Q_accessible_scale)),
        math.log(float(row.beta0_per_day)),
        math.log(float(row.control_decay_q_per_day)),
        math.log(float(row.activation_a_per_day)),
        inverse_s0_transform(float(row.initial_active_fraction_s0)),
    ], dtype=float)


def _load_records(path: Path) -> List[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict("records")


def _drop_wave(records: List[dict], wave_id: str) -> List[dict]:
    return [row for row in records if str(row.get("wave_id", "")) != wave_id]


def _save_checkpoint(store: Dict[str, List[dict]], output_dir: Path) -> None:
    for key, filename in OUTPUT_FILES.items():
        atomic_to_csv(pd.DataFrame(store[key]), output_dir / filename)


def _archived_classic(wave: dict, y: np.ndarray) -> FitResult:
    sse = float(wave["old_classic_sse"])
    aic, aicc = information_criteria(sse, len(y), 3)
    return FitResult(
        "classic", np.full(3, np.nan), np.asarray(wave["old_classic_pred"], dtype=float),
        sse, aic, aicc, 3, True, 0,
    )


def _fit_models(
    wave: dict,
    y: np.ndarray,
    *,
    mode: str,
    h_grid: Sequence[float],
    p_act: float,
    classes: int,
    steps: int,
    quick: bool,
    seed: int,
) -> Tuple[FitResult, FitResult, FitResult, List[FitResult]]:
    if quick:
        c_starts, r_starts, n_starts = 1, 1, 1
        c_nfev, r_nfev, n_nfev = 35, 45, 35
    else:
        c_starts, r_starts, n_starts = 3, 4, 1
        c_nfev, r_nfev, n_nfev = 160, 200, 110

    reproduce_archived = mode == "reproduce" and wave["cohort"] == "Tang101"
    if reproduce_archived:
        warm_x = old_reservoir_x(wave["old_meta"])
        old = wave["old_meta"]
        dummy = FitResult(
            "reservoir", warm_x, np.empty_like(y), float(old.reservoir_sse_log1p),
            float(old.reservoir_aic), float("nan"), 5, True, 0,
        )
        network, profiles = fit_network_profile(
            y, dummy, h_grid=h_grid, p_act=p_act, m=classes,
            starts_per_h=n_starts, max_nfev=n_nfev,
            steps=steps, seed=seed + 1000,
        )
        reservoir = fitresult_from_profile_h0(profiles, y)
        classic = _archived_classic(wave, y)
        return classic, reservoir, network, profiles

    classic = fit_classic(
        y, starts=c_starts, max_nfev=c_nfev, steps=steps, seed=seed + 11,
    )
    reservoir0 = fit_reservoir(
        y, p_act=p_act, starts=r_starts, max_nfev=r_nfev,
        steps=steps, seed=seed + 23,
    )
    network, profiles = fit_network_profile(
        y, reservoir0, h_grid=h_grid, p_act=p_act, m=classes,
        starts_per_h=n_starts, max_nfev=n_nfev,
        steps=steps, seed=seed + 37,
    )
    reservoir = fitresult_from_profile_h0(profiles, y)
    if reservoir0.sse < reservoir.sse:
        reservoir = reservoir0
    return classic, reservoir, network, profiles


def _parameter_dict(fit: FitResult, y: np.ndarray, p_act: float) -> Dict[str, float]:
    if fit.model == "classic" and not np.all(np.isfinite(fit.x)):
        return {
            "Q": np.nan, "beta0": np.nan, "q": np.nan, "a": np.nan,
            "s0": 1.0, "u0": 0.0, "R0": np.nan,
        }
    return parameter_summary(fit, y, p_act)


def _add_fit_fields(prefix: str, fit: FitResult, params: Dict[str, float], row: dict) -> None:
    row.update({
        f"{prefix}_sse_log1p": fit.sse,
        f"{prefix}_aic": fit.aic,
        f"{prefix}_aicc": fit.aicc,
        f"{prefix}_Q": params.get("Q", np.nan),
        f"{prefix}_beta0": params.get("beta0", np.nan),
        f"{prefix}_q": params.get("q", np.nan),
        f"{prefix}_a": params.get("a", np.nan),
        f"{prefix}_s0": params.get("s0", np.nan),
        f"{prefix}_u0": params.get("u0", np.nan),
        f"{prefix}_R0": params.get("R0", np.nan),
        f"{prefix}_success": bool(fit.success),
        f"{prefix}_nfev": int(fit.nfev),
    })


def _summary_to_network_fit(row: pd.Series) -> FitResult:
    x = np.asarray([
        math.log(float(row.network_Q)),
        math.log(float(row.network_beta0)),
        math.log(float(row.network_q)),
        math.log(float(row.network_a)),
        inverse_s0_transform(float(row.network_s0)),
    ])
    h = float(row.network_h_cv2)
    return FitResult(
        "network", x, np.empty(0), float(row.network_sse_log1p),
        float(row.network_aic), float(row.network_aicc), 6, True, 0,
        h_target=h, h_realized=h, z_second_moment=1.0 + h,
    )


def _run_external_holdout(
    waves: List[dict],
    y_by_id: Dict[str, np.ndarray],
    *,
    output_dir: Path,
    h_grid: Sequence[float],
    p_act: float,
    classes: int,
    steps: int,
    quick: bool,
) -> None:
    metric_rows: List[dict] = []
    curve_rows: List[dict] = []
    for index, wave in enumerate([w for w in waves if w["cohort"] == "External10"], 1):
        y = y_by_id[wave["wave_id"]]
        n_train = max(14, int(math.ceil(0.70 * len(y))))
        n_train = min(n_train, len(y) - 5)
        train = y[:n_train]
        c = fit_classic(train, starts=1 if quick else 3, max_nfev=45 if quick else 130,
                        steps=steps, seed=22000 + index)
        r0 = fit_reservoir(train, p_act=p_act, starts=1 if quick else 4,
                           max_nfev=55 if quick else 160, steps=steps, seed=23000 + index)
        nfit, profiles = fit_network_profile(
            train, r0, h_grid=h_grid, p_act=p_act, m=classes,
            starts_per_h=1, max_nfev=40 if quick else 110,
            steps=steps, seed=24000 + index,
        )
        r = fitresult_from_profile_h0(profiles, train)
        if r0.sse < r.sse:
            r = r0
        pred_c = simulate_classic(c.x, y, steps)
        pred_r = simulate_reservoir(r.x, y, p_act, steps)
        z, weights, _ = make_activity_classes(nfit.h_target, classes)
        pred_n = simulate_network(nfit.x, y, z, weights, p_act, steps)
        prediction_map = {"classic": pred_c, "reservoir": pred_r, "network": pred_n}
        local_metrics = []
        for model, prediction in prediction_map.items():
            metric = prediction_metrics(y[n_train:], prediction[n_train:])
            local_metrics.append((model, metric["log_rmse"]))
            metric_rows.append({
                "wave_id": wave["wave_id"], "province": wave["province"],
                "n_total": len(y), "n_train": n_train, "n_test": len(y) - n_train,
                "model": model, "test_log_rmse": metric["log_rmse"],
                "test_mae_cases": metric["mae"],
                "network_h": nfit.h_target if model == "network" else np.nan,
            })
        winner = min(local_metrics, key=lambda item: item[1])[0]
        for row in metric_rows[-3:]:
            row["holdout_winner"] = winner
        for day in range(len(y)):
            curve_rows.append({
                "wave_id": wave["wave_id"], "province": wave["province"],
                "day": day, "observed": y[day], "is_train": int(day < n_train),
                "classic_pred": pred_c[day], "reservoir_pred": pred_r[day],
                "network_pred": pred_n[day],
            })
        print(f"  holdout {index:02d}/10 {wave['province']:<16} h={nfit.h_target:g}")
    atomic_to_csv(pd.DataFrame(metric_rows), output_dir / "external_10_holdout_metrics.csv")
    atomic_to_csv(pd.DataFrame(curve_rows), output_dir / "external_10_holdout_curves.csv")


def _generate_state_and_structure_outputs(
    summary: pd.DataFrame,
    y_by_id: Dict[str, np.ndarray],
    *,
    output_dir: Path,
    p_act: float,
    classes: int,
    steps: int,
    h_grid: Sequence[float],
) -> None:
    summary_by_id = summary.set_index("wave_id")
    state_rows: List[dict] = []
    for wave_id in [x for x in ("W091", "X005", "X008", "X009") if x in summary_by_id.index]:
        row = summary_by_id.loc[wave_id]
        fit = _summary_to_network_fit(row)
        y = y_by_id[wave_id]
        detail = simulate_network_detailed(fit, y, p_act=p_act, m=classes, steps=max(8, steps))
        for day in range(len(y)):
            state_rows.append({
                "wave_id": wave_id, "day": day, "observed": y[day],
                "network_pred": detail["pred"][day],
                "U_head": detail["U_head"][day], "S_head": detail["S_head"][day],
                "S_edge": detail["S_edge"][day], "theta_I": detail["theta_I"][day],
                "mean_z_S": detail["mean_z_S"][day],
                "top20_active_share": detail["top20_active_share"][day],
                "cumulative_recruited_to_S": detail["cumulative_recruited_to_S"][day],
                "cumulative_direct_from_U": detail["cumulative_direct_from_U"][day],
                "h_cv2": fit.h_target, "H_second_moment": fit.z_second_moment,
            })
    atomic_to_csv(pd.DataFrame(state_rows), output_dir / "china_network_state_examples.csv")

    external = summary.loc[summary["cohort"].eq("External10")].copy()
    if external.empty:
        return
    representative = external.sort_values(
        "delta_aicc_reservoir_minus_network", ascending=False
    ).iloc[0]
    wave_id = str(representative.wave_id)
    base_fit = _summary_to_network_fit(representative)
    base_x = base_fit.x.copy()
    y = y_by_id[wave_id]
    scenario_rows: List[dict] = []
    class_rows: List[dict] = []
    for h in h_grid:
        z, weights, _ = make_activity_classes(h, classes)
        for class_index, (activity, weight) in enumerate(zip(z, weights), 1):
            class_rows.append({
                "h_cv2": h, "class": class_index, "activity_z": activity,
                "weight": weight, "H_second_moment": 1.0 + h,
            })
        for scenario in ("fixed_edge_rates", "R0_normalized"):
            x = base_x.copy()
            if scenario == "R0_normalized":
                x[1] -= math.log1p(h)
                x[3] -= math.log1p(h)
            detail = simulate_network_detailed(
                x, y, h=h, p_act=p_act, m=classes, steps=max(8, steps),
            )
            for day in range(len(y)):
                scenario_rows.append({
                    "representative_wave_id": wave_id, "scenario": scenario,
                    "h_cv2": h, "H_second_moment": 1.0 + h, "day": day,
                    "pred": detail["pred"][day], "U_head": detail["U_head"][day],
                    "S_head": detail["S_head"][day], "S_edge": detail["S_edge"][day],
                    "mean_z_S": detail["mean_z_S"][day],
                    "top20_active_share": detail["top20_active_share"][day],
                    "cumulative_recruited_to_S": detail["cumulative_recruited_to_S"][day],
                })
    atomic_to_csv(pd.DataFrame(scenario_rows), output_dir / "china_network_structure_scenarios.csv")
    atomic_to_csv(pd.DataFrame(class_rows), output_dir / "china_network_degree_classes.csv")
    atomic_write_json({
        "wave_id": wave_id,
        "selection": "largest External10 AICc gain of network over homogeneous reservoir",
    }, output_dir / "china_network_structure_representative.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_RESULTS_DIR)
    parser.add_argument("--mode", choices=("reproduce", "refit-all"), default="reproduce")
    parser.add_argument("--h-grid", default="0,0.25,0.5,1,2,4")
    parser.add_argument("--p", type=float, default=P_ACT_DEFAULT)
    parser.add_argument("--classes", type=int, default=12)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.p < 1.0:
        raise SystemExit("--p must be between 0 and 1")
    h_grid = parse_float_grid(args.h_grid)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: output_dir / name for key, name in OUTPUT_FILES.items()}
    if args.overwrite:
        for path in paths.values():
            if path.exists():
                path.unlink()

    store = {key: _load_records(path) for key, path in paths.items()}
    completed = {
        str(row["wave_id"]) for row in store["summary"]
        if row.get("winner") in MODEL_NAMES
    }
    waves, y_by_id, dates_by_id = load_china_inputs()
    if args.limit:
        waves = waves[:args.limit]
    started = time.time()

    for index, wave in enumerate(waves, 1):
        wave_id = wave["wave_id"]
        if wave_id in completed:
            print(f"[{index:03d}/{len(waves)}] {wave_id}: checkpoint already complete; skipped")
            continue
        for key in store:
            store[key] = _drop_wave(store[key], wave_id)
        y = y_by_id[wave_id]
        print(f"[{index:03d}/{len(waves)}] {wave_id} {wave['province']:<16} n={len(y)}", flush=True)
        try:
            classic, reservoir, network, profiles = _fit_models(
                wave, y, mode=args.mode, h_grid=h_grid, p_act=args.p,
                classes=args.classes, steps=args.steps, quick=args.quick,
                seed=10000 + index,
            )
            pc = _parameter_dict(classic, y, args.p)
            pr = _parameter_dict(reservoir, y, args.p)
            pn = _parameter_dict(network, y, args.p)
            winner, criterion, weights = choose_winner([classic, reservoir, network])
            detail = simulate_network_detailed(
                network, y, p_act=args.p, m=args.classes, steps=max(8, args.steps),
            )
            row = {key: value for key, value in wave.items()
                   if key not in {"old_meta", "old_classic_pred", "old_classic_sse"}}
            row.update({
                "p_activation_infection": args.p,
                "analysis_mode": args.mode,
                "comparison_criterion": criterion,
                "winner": winner,
            })
            _add_fit_fields("classic", classic, pc, row)
            _add_fit_fields("reservoir", reservoir, pr, row)
            _add_fit_fields("network", network, pn, row)
            row.update({
                "network_h_cv2": network.h_target,
                "network_second_moment_H": network.z_second_moment,
                "classic_akaike_weight": float(weights[0]),
                "reservoir_akaike_weight": float(weights[1]),
                "network_akaike_weight": float(weights[2]),
                "delta_aic_classic_minus_reservoir": classic.aic - reservoir.aic,
                "delta_aic_reservoir_minus_network": reservoir.aic - network.aic,
                "delta_aic_classic_minus_network": classic.aic - network.aic,
                "delta_aicc_classic_minus_reservoir": classic.aicc - reservoir.aicc,
                "delta_aicc_reservoir_minus_network": reservoir.aicc - network.aicc,
                "delta_aicc_classic_minus_network": classic.aicc - network.aicc,
                "network_initial_S_head": pn["Q"] * pn["s0"],
                "network_max_S_head": float(np.max(detail["S_head"])),
                "network_max_S_edge": float(np.max(detail["S_edge"])),
                "network_peak_head_fraction_Q": float(np.max(detail["S_head"]) / pn["Q"]),
                "network_peak_edge_fraction_Q": float(np.max(detail["S_edge"]) / pn["Q"]),
                "network_max_edge_to_head_ratio": float(np.max(
                    np.asarray(detail["S_edge"]) / np.maximum(np.asarray(detail["S_head"]), 1e-12)
                )),
                "network_max_mean_activity_in_S": float(np.max(detail["mean_z_S"])),
                "network_max_top20_share_in_S": float(np.max(detail["top20_active_share"])),
                "network_recruited_to_S_final": float(detail["cumulative_recruited_to_S"][-1]),
                "network_direct_from_U_final": float(detail["cumulative_direct_from_U"][-1]),
                "network_accessed_reservoir_fraction_Q": float(
                    (detail["cumulative_recruited_to_S"][-1] + detail["cumulative_direct_from_U"][-1]) / pn["Q"]
                ),
            })
            store["summary"].append(row)
            for day, (date, observed, pred_c, pred_r, pred_n) in enumerate(zip(
                dates_by_id[wave_id], y, classic.pred, reservoir.pred, network.pred
            )):
                store["curves"].append({
                    "cohort": wave["cohort"], "wave_id": wave_id,
                    "province": wave["province"], "variant": wave["variant"],
                    "date": date, "day": day, "observed": observed,
                    "classic_pred": pred_c, "reservoir_pred": pred_r,
                    "network_pred": pred_n,
                })
            for fit in profiles:
                params = parameter_summary(fit, y, args.p)
                store["profiles"].append({
                    "cohort": wave["cohort"], "wave_id": wave_id,
                    "province": wave["province"], "h_cv2": fit.h_target,
                    "H_second_moment": fit.z_second_moment,
                    "sse_log1p": fit.sse, "aic": fit.aic, "aicc": fit.aicc,
                    "Q": params["Q"], "beta0": params["beta0"],
                    "q": params["q"], "a": params["a"],
                    "s0": params["s0"], "u0": params["u0"],
                    "R0_network": params["R0"],
                })
            print(
                f"  winner={winner:<9} h={network.h_target:g} "
                f"deltaAIC(R-N)={reservoir.aic-network.aic:.2f}",
                flush=True,
            )
        except Exception as exc:
            store["summary"].append({
                **{key: value for key, value in wave.items() if not key.startswith("old_")},
                "fit_error": f"{type(exc).__name__}: {exc}",
            })
            print(f"ERROR {wave_id}: {type(exc).__name__}: {exc}", flush=True)
        _save_checkpoint(store, output_dir)

    summary = pd.DataFrame(store["summary"])
    completed_summary = summary.loc[summary.get("winner", pd.Series(index=summary.index, dtype=str)).isin(MODEL_NAMES)].copy()
    if not args.limit and len(completed_summary) == 111:
        _generate_state_and_structure_outputs(
            completed_summary, y_by_id, output_dir=output_dir,
            p_act=args.p, classes=args.classes, steps=args.steps, h_grid=h_grid,
        )
        if not args.skip_holdout:
            _run_external_holdout(
                waves, y_by_id, output_dir=output_dir, h_grid=h_grid,
                p_act=args.p, classes=args.classes, steps=args.steps, quick=args.quick,
            )

    manifest = {
        "requested_waves": len(waves),
        "completed_waves_in_checkpoint": int(len(completed_summary)),
        "analysis_mode": args.mode,
        "h_grid": list(h_grid),
        "p_activation_infection": args.p,
        "activity_classes": args.classes,
        "rk4_substeps_per_day": args.steps,
        "solver_stage_positivity": "nonnegative_flow_evaluation",
        "quick_mode": bool(args.quick),
        "elapsed_seconds_this_invocation": float(time.time() - started),
    }
    atomic_write_json(manifest, output_dir / "china_111_analysis_manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
