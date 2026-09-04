#!/usr/bin/env python3
"""Generate complete machine-readable and LaTeX tables for all 153 waves.

The script combines the 111 Chinese and 42 international fit summaries. It
writes a wide all-parameter table, a 459-row model-long table, a compact LaTeX
longtable, model-winner summaries, and a validation manifest.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.data_loaders import load_china_inputs, load_international_inputs  # noqa: E402
from common.io_utils import atomic_to_csv, atomic_write_json, latex_escape  # noqa: E402
from common.model_core import FitResult, inverse_s0_transform, simulate_network_detailed  # noqa: E402
from common.project_paths import OUTPUT_TABLES_DIR, resolve_result_file  # noqa: E402

MODELS = ("classic", "reservoir", "network")


def _network_fit_from_row(row: pd.Series) -> FitResult:
    x = np.asarray([
        math.log(float(row.network_Q)),
        math.log(float(row.network_beta0)),
        math.log(float(row.network_q)),
        math.log(float(row.network_a)),
        inverse_s0_transform(float(row.network_s0)),
    ], dtype=float)
    h = float(row.network_h_cv2)
    return FitResult(
        "network", x, np.empty(0), float(row.network_sse_log1p),
        float(row.network_aic), float(row.network_aicc), 6, True, 0,
        h_target=h, h_realized=h, z_second_moment=1.0+h,
    )


def _recompute_network_diagnostics(wide: pd.DataFrame) -> pd.DataFrame:
    """Recompute state-derived metrics using non-negative RK4 stage flows.

    Early archived result files contained a few physically impossible negative
    cumulative-recruitment diagnostics caused by RK4 intermediate-stage
    overshoot. Model fits and incidence curves were unaffected. This function
    preserves stored values in ``*_stored`` columns and replaces state-derived
    diagnostics with a robust re-simulation from the fitted parameters.
    """
    data = wide.copy()
    diagnostic_columns = [
        "network_initial_S_head", "network_max_S_head", "network_max_S_edge",
        "network_peak_head_fraction_Q", "network_peak_edge_fraction_Q",
        "network_max_edge_to_head_ratio", "network_max_mean_activity_in_S",
        "network_max_top20_share_in_S", "network_recruited_to_S_final",
        "network_direct_from_U_final", "network_recruited_fraction_Q",
        "network_direct_fraction_Q", "network_accessed_reservoir_fraction_Q",
    ]
    for column in diagnostic_columns:
        if column in data.columns:
            data[column + "_stored"] = data[column]

    international_daily, _ = load_international_inputs()
    y_map = {
        str(wave_id): group.sort_values("date")["observed_cases"].to_numpy(dtype=float)
        for wave_id, group in international_daily.groupby("wave_id")
    }
    _, china_y, _ = load_china_inputs()
    y_map.update(china_y)

    status = []
    for index, row in data.iterrows():
        wave_id = str(row.wave_id)
        try:
            fit = _network_fit_from_row(row)
            y = y_map[wave_id]
            p_act = float(row.get("p_activation_infection", 0.1))
            detail = simulate_network_detailed(fit, y, p_act=p_act, m=12, steps=8)
            q_scale = float(detail["Q"])
            s_head = np.asarray(detail["S_head"], dtype=float)
            s_edge = np.asarray(detail["S_edge"], dtype=float)
            recruited = np.asarray(detail["cumulative_recruited_to_S"], dtype=float)
            direct = np.asarray(detail["cumulative_direct_from_U"], dtype=float)
            data.at[index, "network_initial_S_head"] = float(s_head[0])
            data.at[index, "network_max_S_head"] = float(s_head.max())
            data.at[index, "network_max_S_edge"] = float(s_edge.max())
            data.at[index, "network_peak_head_fraction_Q"] = float(s_head.max()/q_scale)
            data.at[index, "network_peak_edge_fraction_Q"] = float(s_edge.max()/q_scale)
            data.at[index, "network_max_edge_to_head_ratio"] = float(np.max(s_edge/np.maximum(s_head,1e-12)))
            data.at[index, "network_max_mean_activity_in_S"] = float(np.max(detail["mean_z_S"]))
            data.at[index, "network_max_top20_share_in_S"] = float(np.max(detail["top20_active_share"]))
            data.at[index, "network_recruited_to_S_final"] = float(recruited[-1])
            data.at[index, "network_direct_from_U_final"] = float(direct[-1])
            data.at[index, "network_recruited_fraction_Q"] = float(recruited[-1]/q_scale)
            data.at[index, "network_direct_fraction_Q"] = float(direct[-1]/q_scale)
            data.at[index, "network_accessed_reservoir_fraction_Q"] = float((recruited[-1]+direct[-1])/q_scale)
            status.append("recomputed")
        except Exception as exc:
            status.append(f"stored_after_error:{type(exc).__name__}")
    data["network_diagnostic_source"] = status
    data["network_diagnostic_physically_valid"] = (
        data["network_recruited_to_S_final"].ge(-1e-8)
        & data["network_direct_from_U_final"].ge(-1e-8)
        & data["network_accessed_reservoir_fraction_Q"].between(-1e-8, 1.000001)
    ).astype(int)
    return data


def _harmonise_china(china: pd.DataFrame) -> pd.DataFrame:
    data = china.copy()
    data["scope"] = "China local/provincial wave"
    data["country"] = "China"
    data["country_code"] = "CHN"
    data["setting"] = data["province"]
    data["strategy"] = "Dynamic zero / rapid local containment"
    data["variant_era"] = data["variant"].map({
        "Alpha": "pre-variant/Alpha", "Delta": "Delta", "Omicron": "Omicron",
    }).fillna(data["variant"])
    data["peak_cases_7d"] = data.get("peak_cases", np.nan)
    data["total_cases_7d"] = data.get("total_cases", np.nan)
    data["peak_cases_per_million"] = np.nan
    data["population"] = np.nan
    data["source"] = "Tang 101-wave data plus ten post-May-2022 Chinese provincial waves"
    data["network_peak_head_fraction_Q"] = (
        data["network_max_S_head"] / data["network_Q"]
    )
    data["network_peak_edge_fraction_Q"] = (
        data["network_max_S_edge"] / data["network_Q"]
    )
    data["network_recruited_fraction_Q"] = (
        data["network_recruited_to_S_final"] / data["network_Q"]
    )
    data["network_direct_fraction_Q"] = (
        data["network_direct_from_U_final"] / data["network_Q"]
    )
    data["network_accessed_reservoir_fraction_Q"] = (
        data["network_recruited_to_S_final"] + data["network_direct_from_U_final"]
    ) / data["network_Q"]
    if "network_h_grid" not in data:
        data["network_h_grid"] = data["network_h_cv2"]
    return data


def _harmonise_international(international: pd.DataFrame) -> pd.DataFrame:
    data = international.copy()
    data["scope"] = "International national wave"
    data["setting"] = data["country"]
    return data


def _wide_column_order(columns: List[str]) -> List[str]:
    metadata = [
        "scope", "wave_id", "setting", "country", "country_code", "province", "province_zh",
        "cohort", "strategy", "variant", "variant_era", "start", "end", "peak_date",
        "duration_days", "peak_cases", "total_cases", "peak_cases_7d", "total_cases_7d",
        "peak_cases_per_million", "population", "source", "p_activation_infection",
        "comparison_criterion", "winner",
    ]
    model_columns: List[str] = []
    suffixes = [
        "sse_log1p", "aic", "aicc", "akaike_weight", "Q", "beta0", "q", "a",
        "s0", "u0", "R0", "success", "nfev",
    ]
    for model in MODELS:
        model_columns.extend([f"{model}_{suffix}" for suffix in suffixes])
    comparison = [
        "network_h_cv2", "network_h_grid", "network_second_moment_H",
        "delta_aic_classic_minus_reservoir", "delta_aic_reservoir_minus_network",
        "delta_aic_classic_minus_network", "delta_aicc_classic_minus_reservoir",
        "delta_aicc_reservoir_minus_network", "delta_aicc_classic_minus_network",
        "network_initial_S_head", "network_max_S_head", "network_max_S_edge",
        "network_peak_head_fraction_Q", "network_peak_edge_fraction_Q",
        "network_max_edge_to_head_ratio", "network_max_mean_activity_in_S",
        "network_max_top20_share_in_S", "network_recruited_to_S_final",
        "network_direct_from_U_final", "network_recruited_fraction_Q",
        "network_direct_fraction_Q", "network_accessed_reservoir_fraction_Q", "fit_seconds",
    ]
    ordered = [column for column in metadata + model_columns + comparison if column in columns]
    ordered += sorted(set(columns) - set(ordered))
    return ordered


def _make_long_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for _, wave in wide.iterrows():
        for model in MODELS:
            rows.append({
                "scope": wave.get("scope"),
                "wave_id": wave.get("wave_id"),
                "setting": wave.get("setting"),
                "country": wave.get("country"),
                "variant": wave.get("variant"),
                "variant_era": wave.get("variant_era"),
                "start": wave.get("start"),
                "end": wave.get("end"),
                "duration_days": wave.get("duration_days"),
                "winner": wave.get("winner"),
                "comparison_criterion": wave.get("comparison_criterion"),
                "model": model,
                "is_winner": int(wave.get("winner") == model),
                "sse_log1p": wave.get(f"{model}_sse_log1p", np.nan),
                "aic": wave.get(f"{model}_aic", np.nan),
                "aicc": wave.get(f"{model}_aicc", np.nan),
                "akaike_weight": wave.get(f"{model}_akaike_weight", np.nan),
                "Q": wave.get(f"{model}_Q", np.nan),
                "beta0": wave.get(f"{model}_beta0", np.nan),
                "q": wave.get(f"{model}_q", np.nan),
                "a": wave.get(f"{model}_a", np.nan),
                "s0": wave.get(f"{model}_s0", np.nan),
                "u0": wave.get(f"{model}_u0", np.nan),
                "R0": wave.get(f"{model}_R0", np.nan),
                "success": wave.get(f"{model}_success", np.nan),
                "nfev": wave.get(f"{model}_nfev", np.nan),
                "h_cv2": wave.get("network_h_cv2", np.nan) if model == "network" else 0.0,
                "peak_S_head_fraction_Q": wave.get("network_peak_head_fraction_Q", np.nan) if model == "network" else np.nan,
                "peak_S_edge_fraction_Q": wave.get("network_peak_edge_fraction_Q", np.nan) if model == "network" else np.nan,
                "accessed_reservoir_fraction_Q": wave.get("network_accessed_reservoir_fraction_Q", np.nan) if model == "network" else np.nan,
            })
    return pd.DataFrame(rows)


def _format_number(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(number):
        return "--"
    return f"{number:.{digits}f}"


def _write_latex(wide: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{landscape}",
        r"\begin{longtable}{p{1.0cm}p{2.3cm}p{2.5cm}rrrrrrrr}",
        r"\caption{Model comparison and network high-risk susceptible metrics for all 153 waves}\label{tab:all153}\\",
        r"\toprule",
        r"ID & Setting & Variant & Days & Winner & $h$ & $\Delta$ C--R & $\Delta$ R--N & peak $S_h/Q$ & peak $S_e/Q$ & accessed $U/Q$\\",
        r"\midrule\endfirsthead",
        r"\toprule ID & Setting & Variant & Days & Winner & $h$ & $\Delta$ C--R & $\Delta$ R--N & peak $S_h/Q$ & peak $S_e/Q$ & accessed $U/Q$\\\midrule\endhead",
    ]
    for _, row in wide.iterrows():
        lines.append(
            f"{latex_escape(row.wave_id)} & {latex_escape(row.setting)} & {latex_escape(row.variant)} & "
            f"{int(row.duration_days)} & {latex_escape(row.winner)} & "
            f"{_format_number(row.network_h_cv2, 2)} & "
            f"{_format_number(row.delta_aicc_classic_minus_reservoir, 1)} & "
            f"{_format_number(row.delta_aicc_reservoir_minus_network, 1)} & "
            f"{_format_number(row.network_peak_head_fraction_Q, 3)} & "
            f"{_format_number(row.network_peak_edge_fraction_Q, 3)} & "
            f"{_format_number(row.network_accessed_reservoir_fraction_Q, 3)}\\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\multicolumn{11}{p{25cm}}{\footnotesize C--R=classic minus homogeneous reservoir AR-SEIR; R--N=homogeneous reservoir minus network AR-SEIR. Positive values favour the model to the right. Fractions are relative to fitted $Q$, not administrative population. For very short waves AIC is used when AICc is undefined; consult the machine-readable table for the criterion.}\\",
        r"\end{longtable}",
        r"\end{landscape}",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-source", choices=("reference", "outputs", "auto"), default="reference")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_TABLES_DIR)
    parser.add_argument("--use-stored-network-diagnostics", action="store_true",
                        help="do not robustly recompute state-derived network metrics")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    china = pd.read_csv(resolve_result_file("china_fit_summary_111_three_models.csv", args.results_source))
    international = pd.read_csv(resolve_result_file("international_fit_summary.csv", args.results_source))
    if china.wave_id.nunique() != 111:
        raise ValueError(f"Expected 111 Chinese waves, found {china.wave_id.nunique()}")
    if international.wave_id.nunique() != 42:
        raise ValueError(f"Expected 42 international waves, found {international.wave_id.nunique()}")

    china = _harmonise_china(china)
    international = _harmonise_international(international)
    all_columns = sorted(set(china.columns).union(international.columns))
    for column in all_columns:
        if column not in china:
            china[column] = np.nan
        if column not in international:
            international[column] = np.nan
    wide = pd.concat([china[all_columns], international[all_columns]], ignore_index=True, sort=False)
    wide = wide.sort_values(["scope", "country", "start", "wave_id"], kind="mergesort").reset_index(drop=True)
    if not args.use_stored_network_diagnostics:
        wide = _recompute_network_diagnostics(wide)
    if len(wide) != 153 or wide.wave_id.nunique() != 153:
        raise ValueError(f"Expected exactly 153 unique waves; got rows={len(wide)}, unique={wide.wave_id.nunique()}")
    wide = wide[_wide_column_order(wide.columns.tolist())]
    long = _make_long_table(wide)
    if len(long) != 459:
        raise ValueError(f"Expected 459 wave-model rows; found {len(long)}")

    wide_path = output_dir / "complete_153_wave_table_wide.csv"
    long_path = output_dir / "complete_153_wave_table_model_long.csv"
    atomic_to_csv(wide, wide_path)
    atomic_to_csv(long, long_path)

    scope_summary = (
        wide.groupby(["scope", "winner"]).size().unstack(fill_value=0)
        .reindex(columns=list(MODELS), fill_value=0).reset_index()
    )
    scope_summary["n_waves"] = scope_summary[list(MODELS)].sum(axis=1)
    overall = {"scope": "All 153 waves", "n_waves": len(wide)}
    for model in MODELS:
        overall[model] = int((wide.winner == model).sum())
    scope_summary = pd.concat([scope_summary, pd.DataFrame([overall])], ignore_index=True)
    atomic_to_csv(scope_summary, output_dir / "complete_153_wave_model_winner_summary.csv")

    country_summary = (
        wide.groupby(["country", "winner"]).size().unstack(fill_value=0)
        .reindex(columns=list(MODELS), fill_value=0).reset_index()
    )
    country_summary["n_waves"] = country_summary[list(MODELS)].sum(axis=1)
    atomic_to_csv(country_summary, output_dir / "complete_153_wave_country_summary.csv")

    latex_path = output_dir / "TableS8_all_153_waves.tex"
    _write_latex(wide, latex_path)
    manifest: Dict[str, object] = {
        "n_china": 111,
        "n_international": 42,
        "n_total": 153,
        "n_model_long_rows": 459,
        "unique_wave_ids": int(wide.wave_id.nunique()),
        "winner_counts": {model: int((wide.winner == model).sum()) for model in MODELS},
        "results_source": args.results_source,
        "network_diagnostics": "stored" if args.use_stored_network_diagnostics else "robustly recomputed from fitted parameters",
        "physically_valid_network_diagnostics": int(wide["network_diagnostic_physically_valid"].sum()) if "network_diagnostic_physically_valid" in wide else None,
        "files": [
            wide_path.name, long_path.name,
            "complete_153_wave_model_winner_summary.csv",
            "complete_153_wave_country_summary.csv",
            latex_path.name,
        ],
    }
    atomic_write_json(manifest, output_dir / "complete_153_wave_table_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()
