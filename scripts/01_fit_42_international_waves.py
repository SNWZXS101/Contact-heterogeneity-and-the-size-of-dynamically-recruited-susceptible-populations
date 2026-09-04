#!/usr/bin/env python3
"""Fit M0/M1/M2 to all 42 international COVID-19 waves.

Direct full run
---------------
    python scripts/01_fit_42_international_waves.py

Fast smoke test
---------------
    python scripts/01_fit_42_international_waves.py --quick --limit 1 --overwrite

Outputs are written to ``outputs/results`` by default. The script checkpoints
all six result tables after every wave, so an interrupted run can be resumed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.data_loaders import load_international_inputs  # noqa: E402
from common.fit_helpers import (  # noqa: E402
    MODEL_NAMES,
    add_fit_fields,
    choose_winner,
    fitresult_from_profile_h0,
    prediction_metrics,
    stable_seed,
)
from common.io_utils import atomic_to_csv, atomic_write_json, parse_float_grid  # noqa: E402
from common.model_core import (  # noqa: E402
    P_ACT_DEFAULT,
    fit_classic,
    fit_network_profile,
    fit_reservoir,
    make_activity_classes,
    parameter_summary,
    simulate_classic,
    simulate_network,
    simulate_network_detailed,
    simulate_reservoir,
)
from common.project_paths import OUTPUT_RESULTS_DIR  # noqa: E402

OUTPUT_FILES = {
    "summary": "international_fit_summary.csv",
    "curves": "international_fit_curves.csv",
    "profiles": "international_network_profiles.csv",
    "states": "international_network_states.csv",
    "holdout": "international_holdout_metrics.csv",
    "holdout_curves": "international_holdout_curves.csv",
}

# The international outcome is a centred 7-day mean. A centred value at day t
# depends on raw daily incidence from t-3 through t+3. Around the nominal 70/30
# split we therefore leave three centred observations unused on each side of
# the boundary. This guarantees that the raw observations supporting the fitted
# training series and the scored test series do not overlap.
CENTERED_SMOOTHING_HALF_WINDOW_DAYS = 3


def _load_records(path: Path) -> List[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict("records")


def _drop_wave(records: List[dict], wave_id: str) -> List[dict]:
    return [row for row in records if str(row.get("wave_id", "")) != wave_id]


def _save_checkpoint(store: Dict[str, List[dict]], output_dir: Path) -> None:
    for key, filename in OUTPUT_FILES.items():
        atomic_to_csv(pd.DataFrame(store[key]), output_dir / filename)


def _fit_one_wave(
    wave_id: str,
    group: pd.DataFrame,
    catalog_row: pd.Series,
    *,
    h_grid: Sequence[float],
    p_act: float,
    classes: int,
    steps: int,
    quick: bool,
    holdout: bool,
) -> Tuple[dict, List[dict], List[dict], List[dict], Optional[dict], List[dict]]:
    y = group["observed_cases"].to_numpy(dtype=float)
    seed = stable_seed(wave_id)
    if quick:
        classic_starts, reservoir_starts, network_starts = 1, 1, 1
        classic_nfev, reservoir_nfev, network_nfev = 35, 45, 30
    else:
        classic_starts, reservoir_starts, network_starts = 3, 4, 1
        classic_nfev, reservoir_nfev, network_nfev = 180, 220, 120

    started = time.time()
    classic = fit_classic(
        y, starts=classic_starts, max_nfev=classic_nfev,
        steps=steps, seed=seed + 1,
    )
    reservoir = fit_reservoir(
        y, p_act=p_act, starts=reservoir_starts,
        max_nfev=reservoir_nfev, steps=steps, seed=seed + 2,
    )
    network, profile = fit_network_profile(
        y, reservoir, h_grid=h_grid, p_act=p_act, m=classes,
        starts_per_h=network_starts, max_nfev=network_nfev,
        steps=steps, seed=seed + 3,
    )
    # Exact nesting QC: h=0 in the M2 profile is the same dynamical model as M1.
    # Retain whichever five-parameter M1 optimum (standalone or profile-h0)
    # achieves the smaller transformed-scale SSE.
    reservoir_h0 = fitresult_from_profile_h0(profile, y)
    reservoir_source = "standalone"
    if reservoir_h0.sse < reservoir.sse:
        reservoir = reservoir_h0
        reservoir_source = "profile_h0"

    winner, criterion, weights = choose_winner([classic, reservoir, network])
    row: Dict[str, object] = catalog_row.to_dict()
    row.update({
        "p_activation_infection": p_act,
        "comparison_criterion": criterion,
        "winner": winner,
        "fit_seconds": float(time.time() - started),
        "classic_akaike_weight": float(weights[0]),
        "reservoir_akaike_weight": float(weights[1]),
        "network_akaike_weight": float(weights[2]),
        "reservoir_fit_source": reservoir_source,
    })
    add_fit_fields("classic", classic, y, row, p_act)
    add_fit_fields("reservoir", reservoir, y, row, p_act)
    add_fit_fields("network", network, y, row, p_act)
    row.update({
        "network_h_cv2": network.h_realized,
        "network_h_grid": network.h_target,
        "network_second_moment_H": network.z_second_moment,
        "delta_aicc_classic_minus_reservoir": classic.aicc - reservoir.aicc,
        "delta_aicc_reservoir_minus_network": reservoir.aicc - network.aicc,
        "delta_aicc_classic_minus_network": classic.aicc - network.aicc,
        "delta_aic_classic_minus_reservoir": classic.aic - reservoir.aic,
        "delta_aic_reservoir_minus_network": reservoir.aic - network.aic,
        "delta_aic_classic_minus_network": classic.aic - network.aic,
    })

    detail = simulate_network_detailed(
        network, y, h=network.h_target, p_act=p_act,
        m=classes, steps=max(6, steps),
    )
    q_scale = float(detail["Q"])
    s_head = np.asarray(detail["S_head"], dtype=float)
    s_edge = np.asarray(detail["S_edge"], dtype=float)
    recruited = np.asarray(detail["cumulative_recruited_to_S"], dtype=float)
    direct = np.asarray(detail["cumulative_direct_from_U"], dtype=float)
    row.update({
        "network_initial_S_head": float(s_head[0]),
        "network_max_S_head": float(s_head.max()),
        "network_max_S_edge": float(s_edge.max()),
        "network_peak_head_fraction_Q": float(s_head.max() / q_scale),
        "network_peak_edge_fraction_Q": float(s_edge.max() / q_scale),
        "network_max_edge_to_head_ratio": float(np.max(s_edge / np.maximum(s_head, 1e-12))),
        "network_max_mean_activity_in_S": float(np.max(detail["mean_z_S"])),
        "network_max_top20_share_in_S": float(np.max(detail["top20_active_share"])),
        "network_recruited_to_S_final": float(recruited[-1]),
        "network_direct_from_U_final": float(direct[-1]),
        "network_recruited_fraction_Q": float(recruited[-1] / q_scale),
        "network_direct_fraction_Q": float(direct[-1] / q_scale),
        "network_accessed_reservoir_fraction_Q": float((recruited[-1] + direct[-1]) / q_scale),
    })

    curve_rows: List[dict] = []
    state_rows: List[dict] = []
    reset_group = group.reset_index(drop=True)
    for day, data_row in reset_group.iterrows():
        curve_rows.append({
            "wave_id": wave_id,
            "country": catalog_row["country"],
            "variant": catalog_row["variant"],
            "date": data_row["date"],
            "day": day,
            "observed": y[day],
            "classic_pred": classic.pred[day],
            "reservoir_pred": reservoir.pred[day],
            "network_pred": network.pred[day],
        })
        state_rows.append({
            "wave_id": wave_id,
            "country": catalog_row["country"],
            "date": data_row["date"],
            "day": day,
            "U_head": detail["U_head"][day],
            "S_head": detail["S_head"][day],
            "S_edge": detail["S_edge"][day],
            "theta_I": detail["theta_I"][day],
            "mean_activity_S": detail["mean_z_S"][day],
            "top20_share_S": detail["top20_active_share"][day],
            "cum_recruited_S": recruited[day],
            "cum_direct_U": direct[day],
        })

    profile_rows: List[dict] = []
    for fit in profile:
        params = parameter_summary(fit, y, p_act)
        profile_rows.append({
            "wave_id": wave_id,
            "country": catalog_row["country"],
            "variant": catalog_row["variant"],
            "h_grid": fit.h_target,
            "h_realized": fit.h_realized,
            "aic": fit.aic,
            "aicc": fit.aicc,
            "sse_log1p": fit.sse,
            "Q": params["Q"],
            "beta0": params["beta0"],
            "q": params["q"],
            "a": params["a"],
            "s0": params["s0"],
            "u0": params["u0"],
            "R0": params["R0"],
        })

    holdout_row: Optional[dict] = None
    holdout_curve_rows: List[dict] = []
    if holdout and len(y) >= 35:
        # Nominal split is kept at 70% of the centred series. To avoid overlap
        # in the raw daily observations supporting the centred 7-day values,
        # we fit only through split-4 and begin scoring at split+3.
        nominal_split_day = max(25, int(math.floor(0.70 * len(y))))
        smoothing_half_window_days = CENTERED_SMOOTHING_HALF_WINDOW_DAYS
        n_train_fitted = nominal_split_day - smoothing_half_window_days
        test_start_day = nominal_split_day + smoothing_half_window_days

        if n_train_fitted < 10:
            raise ValueError(
                f"{wave_id}: insufficient fitted training observations after "
                f"centred-smoothing boundary buffer"
            )
        if test_start_day >= len(y):
            raise ValueError(
                f"{wave_id}: no scored test observations after centred-smoothing "
                f"boundary buffer"
            )

        y_train = y[:n_train_fitted]
        hold_classic = fit_classic(
            y_train, starts=1 if quick else 2,
            max_nfev=45 if quick else 120, steps=steps, seed=seed + 11,
        )
        hold_reservoir = fit_reservoir(
            y_train, p_act=p_act, starts=1 if quick else 3,
            max_nfev=55 if quick else 150, steps=steps, seed=seed + 12,
        )
        hold_network, hold_profile = fit_network_profile(
            y_train, hold_reservoir, h_grid=h_grid, p_act=p_act, m=classes,
            starts_per_h=1, max_nfev=35 if quick else 80,
            steps=steps, seed=seed + 13,
        )
        hold_reservoir_h0 = fitresult_from_profile_h0(hold_profile, y_train)
        hold_reservoir_source = "standalone"
        if hold_reservoir_h0.sse < hold_reservoir.sse:
            hold_reservoir = hold_reservoir_h0
            hold_reservoir_source = "profile_h0"

        pred_classic = simulate_classic(hold_classic.x, y, steps=steps)
        pred_reservoir = simulate_reservoir(hold_reservoir.x, y, p_act=p_act, steps=steps)
        z, weights_h, _ = make_activity_classes(hold_network.h_target, m=classes)
        pred_network = simulate_network(
            hold_network.x, y, z, weights_h, p_act=p_act, steps=steps,
        )

        test_slice = slice(test_start_day, None)
        family_selected_train, family_train_criterion, _ = choose_winner(
            [hold_reservoir, hold_network]
        )
        holdout_row = {
            "wave_id": wave_id,
            "country": catalog_row["country"],
            "variant": catalog_row["variant"],
            "n_total": len(y),
            # Backwards-compatible field: number of centred observations
            # actually used in fitting after the boundary buffer.
            "n_train": n_train_fitted,
            "n_train_nominal": nominal_split_day,
            "n_train_fitted": n_train_fitted,
            "nominal_split_day": nominal_split_day,
            "smoothing_half_window_days": smoothing_half_window_days,
            "pre_split_guard_days": smoothing_half_window_days,
            "post_split_guard_days": smoothing_half_window_days,
            "test_start_day": test_start_day,
            "n_test": len(y) - test_start_day,
            "network_h_cv2": hold_network.h_realized,
            "classic_train_aic": hold_classic.aic,
            "classic_train_aicc": hold_classic.aicc,
            "reservoir_train_aic": hold_reservoir.aic,
            "reservoir_train_aicc": hold_reservoir.aicc,
            "network_train_aic": hold_network.aic,
            "network_train_aicc": hold_network.aicc,
            "reservoir_fit_source_train": hold_reservoir_source,
            "reservoir_family_selected_train": family_selected_train,
            "reservoir_family_train_criterion": family_train_criterion,
        }
        predictions = {
            "classic": pred_classic,
            "reservoir": pred_reservoir,
            "network": pred_network,
        }
        for name, prediction in predictions.items():
            metric = prediction_metrics(y[test_slice], prediction[test_slice])
            holdout_row.update({
                f"{name}_test_log_rmse": metric["log_rmse"],
                f"{name}_test_mae": metric["mae"],
                f"{name}_test_smape": metric["smape"],
                f"{name}_test_peak_day_error": metric["peak_day_error"],
            })
        holdout_row["winner_test"] = min(
            MODEL_NAMES, key=lambda model: holdout_row[f"{model}_test_log_rmse"]
        )

        # Implementable reservoir-family predictor: choose M1 vs M2 using only
        # the training information criterion, then evaluate that fixed choice
        # on the scored held-out tail.
        selected_prediction = (
            pred_reservoir if family_selected_train == "reservoir" else pred_network
        )
        selected_metric = prediction_metrics(
            y[test_slice], selected_prediction[test_slice]
        )
        holdout_row["reservoir_family_test_log_rmse"] = selected_metric["log_rmse"]
        holdout_row["reservoir_family_test_mae"] = selected_metric["mae"]
        holdout_row["reservoir_family_test_smape"] = selected_metric["smape"]
        holdout_row["reservoir_family_beats_classic_test"] = int(
            selected_metric["log_rmse"] < holdout_row["classic_test_log_rmse"]
        )

        # Diagnostic upper bound only: this uses the scored test set to choose
        # the better family member and must never be reported as a deployable
        # predictive procedure.
        holdout_row["oracle_reservoir_family_test_log_rmse"] = min(
            holdout_row["reservoir_test_log_rmse"],
            holdout_row["network_test_log_rmse"],
        )

        for day, data_row in reset_group.iterrows():
            is_train = day < n_train_fitted
            is_pre_guard = n_train_fitted <= day < nominal_split_day
            is_post_guard = nominal_split_day <= day < test_start_day
            is_test = day >= test_start_day
            if is_train:
                holdout_phase = "train"
            elif is_pre_guard:
                holdout_phase = "pre_split_guard"
            elif is_post_guard:
                holdout_phase = "post_split_guard"
            else:
                holdout_phase = "test"

            holdout_curve_rows.append({
                "wave_id": wave_id,
                "country": catalog_row["country"],
                "date": data_row["date"],
                "day": day,
                "is_train": is_train,
                "is_pre_split_guard": is_pre_guard,
                "is_post_split_guard": is_post_guard,
                "is_guard": is_pre_guard or is_post_guard,
                "is_test": is_test,
                "holdout_phase": holdout_phase,
                "observed": y[day],
                "classic_pred": pred_classic[day],
                "reservoir_pred": pred_reservoir[day],
                "network_pred": pred_network[day],
            })

    return row, curve_rows, profile_rows, state_rows, holdout_row, holdout_curve_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_RESULTS_DIR)
    parser.add_argument("--h-grid", default="0,0.25,0.5,1,2,4")
    parser.add_argument("--p", type=float, default=P_ACT_DEFAULT,
                        help="probability of infection on first activation contact")
    parser.add_argument("--classes", type=int, default=12)
    parser.add_argument("--steps", type=int, default=8, help="positivity-safe RK4 substeps per day")
    parser.add_argument("--limit", type=int, default=0,
                        help="fit only the first N waves; 0 means all 42")
    parser.add_argument("--quick", action="store_true",
                        help="small optimisation budgets for a smoke test")
    parser.add_argument("--no-holdout", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="remove any existing checkpoint files first")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.p < 1.0:
        raise SystemExit("--p must be between 0 and 1")
    h_grid = parse_float_grid(args.h_grid)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_paths = {key: output_dir / filename for key, filename in OUTPUT_FILES.items()}
    if args.overwrite:
        for path in target_paths.values():
            if path.exists():
                path.unlink()

    store = {key: _load_records(path) for key, path in target_paths.items()}
    completed = {
        str(row["wave_id"]) for row in store["summary"]
        if row.get("winner") in MODEL_NAMES
    }

    daily, catalog = load_international_inputs()
    if args.limit:
        catalog = catalog.head(args.limit).copy()
    started = time.time()
    for index, catalog_row in catalog.reset_index(drop=True).iterrows():
        wave_id = str(catalog_row["wave_id"])
        if wave_id in completed:
            print(f"[{index + 1:02d}/{len(catalog)}] {wave_id}: checkpoint already complete; skipped")
            continue
        group = daily.loc[daily["wave_id"].astype(str).eq(wave_id)].sort_values("date")
        print(
            f"[{index + 1:02d}/{len(catalog)}] {wave_id} "
            f"{catalog_row['country']} {catalog_row['variant']} n={len(group)}",
            flush=True,
        )
        for key in store:
            store[key] = _drop_wave(store[key], wave_id)
        try:
            result = _fit_one_wave(
                wave_id, group, catalog_row, h_grid=h_grid, p_act=args.p,
                classes=args.classes, steps=args.steps, quick=args.quick,
                holdout=not args.no_holdout,
            )
            summary, curves, profiles, states, holdout, holdout_curves = result
            store["summary"].append(summary)
            store["curves"].extend(curves)
            store["profiles"].extend(profiles)
            store["states"].extend(states)
            if holdout is not None:
                store["holdout"].append(holdout)
                store["holdout_curves"].extend(holdout_curves)
        except Exception as exc:
            error_row = catalog_row.to_dict()
            error_row.update({
                "wave_id": wave_id,
                "fit_error": f"{type(exc).__name__}: {exc}",
            })
            store["summary"].append(error_row)
            _save_checkpoint(store, output_dir)
            print(f"ERROR {wave_id}: {type(exc).__name__}: {exc}", flush=True)
            continue
        _save_checkpoint(store, output_dir)

    summary_df = pd.DataFrame(store["summary"])
    manifest = {
        "requested_waves": int(len(catalog)),
        "fitted_waves_in_checkpoint": int(summary_df.get("winner", pd.Series(dtype=str)).isin(MODEL_NAMES).sum()),
        "h_grid": list(h_grid),
        "p_activation_infection": args.p,
        "activity_classes": args.classes,
        "rk4_substeps_per_day": args.steps,
        "solver_stage_positivity": "nonnegative_flow_evaluation",
        "latent_period_days": 3,
        "infectious_period_days": 5,
        "quick_mode": bool(args.quick),
        "holdout_enabled": not args.no_holdout,
        "holdout_nominal_train_fraction": 0.70,
        "holdout_centered_smoothing_half_window_days": CENTERED_SMOOTHING_HALF_WINDOW_DAYS,
        "holdout_raw_support_overlap": False,
        "holdout_design": (
            "retrospective within-wave tail extrapolation with symmetric "
            "3-day boundary buffers for centred 7-day smoothing"
        ),
        "elapsed_seconds_this_invocation": float(time.time() - started),
    }
    atomic_write_json(manifest, output_dir / "international_fit_manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
