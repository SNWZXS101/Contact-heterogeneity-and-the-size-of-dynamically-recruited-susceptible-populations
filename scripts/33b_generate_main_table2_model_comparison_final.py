#!/usr/bin/env python3
"""Generate final main-text Table 2: model evidence and guarded temporal validation.

Inputs
------
outputs/results/international_fit_summary.csv
outputs/results/china_fit_summary_111_three_models.csv
outputs/results/international_holdout_metrics.csv

Outputs
-------
outputs/tables/main_table2_model_comparison_final.csv
outputs/tables/main_table2_holdout_final.csv
outputs/tables/main_table2_model_comparison_final.tex

Run
---
python scripts/33b_generate_main_table2_model_comparison_final.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
TABLES = ROOT / "outputs" / "tables"

MODELS = ["classic", "reservoir", "network"]


def winner_counts(df: pd.DataFrame) -> dict[str, int]:
    return {model: int(df["winner"].astype(str).eq(model).sum()) for model in MODELS}


def count_gt(df: pd.DataFrame, column: str, threshold: float) -> int:
    x = pd.to_numeric(df[column], errors="coerce")
    return int((x > threshold).sum())


def mean_numeric(df: pd.DataFrame, column: str) -> float:
    x = pd.to_numeric(df[column], errors="coerce")
    return float(x[np.isfinite(x)].mean())


def median_numeric(df: pd.DataFrame, column: str) -> float:
    x = pd.to_numeric(df[column], errors="coerce")
    return float(x[np.isfinite(x)].median())


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=TABLES)
    args = parser.parse_args()

    results = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    intl = pd.read_csv(results / "international_fit_summary.csv")
    china = pd.read_csv(results / "china_fit_summary_111_three_models.csv")
    hold = pd.read_csv(results / "international_holdout_metrics.csv")
    all153 = pd.concat([china, intl], ignore_index=True, sort=False)

    if len(china) != 111 or len(intl) != 42 or len(all153) != 153:
        raise ValueError("Expected China=111, international=42, all=153.")

    comparison_rows = []
    for label, df in [
        ("China local/provincial", china),
        ("International national", intl),
        ("All waves", all153),
    ]:
        wins = winner_counts(df)
        comparison_rows.append({
            "analysis_set": label,
            "n_waves": int(len(df)),
            "M0_winner": wins["classic"],
            "M1_winner": wins["reservoir"],
            "M2_winner": wins["network"],
            "delta_AICc_M0_minus_M1_gt2": count_gt(
                df, "delta_aicc_classic_minus_reservoir", 2
            ),
            "delta_AICc_M1_minus_M2_gt2": count_gt(
                df, "delta_aicc_reservoir_minus_network", 2
            ),
            "delta_AICc_M1_minus_M2_gt10": count_gt(
                df, "delta_aicc_reservoir_minus_network", 10
            ),
        })

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        output_dir / "main_table2_model_comparison_final.csv",
        index=False, encoding="utf-8-sig",
    )

    # Holdout summaries.
    holdout_rows = []
    holdout_rows.append({
        "model": "M0 classic SEIR",
        "test_winner_count": int(hold["winner_test"].astype(str).eq("classic").sum()),
        "mean_RMSElog": mean_numeric(hold, "classic_test_log_rmse"),
        "median_RMSElog": median_numeric(hold, "classic_test_log_rmse"),
        "waves_lower_error_than_M0": np.nan,
    })
    holdout_rows.append({
        "model": "M1 homogeneous RA-SEIR",
        "test_winner_count": int(hold["winner_test"].astype(str).eq("reservoir").sum()),
        "mean_RMSElog": mean_numeric(hold, "reservoir_test_log_rmse"),
        "median_RMSElog": median_numeric(hold, "reservoir_test_log_rmse"),
        "waves_lower_error_than_M0": int(
            (
                pd.to_numeric(hold["reservoir_test_log_rmse"], errors="coerce")
                <
                pd.to_numeric(hold["classic_test_log_rmse"], errors="coerce")
            ).sum()
        ),
    })
    holdout_rows.append({
        "model": "M2 activity-stratified RA-SEIR",
        "test_winner_count": int(hold["winner_test"].astype(str).eq("network").sum()),
        "mean_RMSElog": mean_numeric(hold, "network_test_log_rmse"),
        "median_RMSElog": median_numeric(hold, "network_test_log_rmse"),
        "waves_lower_error_than_M0": int(
            (
                pd.to_numeric(hold["network_test_log_rmse"], errors="coerce")
                <
                pd.to_numeric(hold["classic_test_log_rmse"], errors="coerce")
            ).sum()
        ),
    })
    holdout_rows.append({
        "model": "Training-selected RA-SEIR family",
        "test_winner_count": np.nan,
        "mean_RMSElog": mean_numeric(hold, "reservoir_family_test_log_rmse"),
        "median_RMSElog": median_numeric(hold, "reservoir_family_test_log_rmse"),
        "waves_lower_error_than_M0": int(
            pd.to_numeric(
                hold["reservoir_family_beats_classic_test"], errors="coerce"
            ).fillna(0).sum()
        ),
    })

    holdout = pd.DataFrame(holdout_rows)
    holdout.to_csv(
        output_dir / "main_table2_holdout_final.csv",
        index=False, encoding="utf-8-sig",
    )

    # Selected-family training counts.
    family_m1 = int(
        hold["reservoir_family_selected_train"].astype(str).eq("reservoir").sum()
    )
    family_m2 = int(
        hold["reservoir_family_selected_train"].astype(str).eq("network").sum()
    )

    tex = [
        r"\begin{table}[p]",
        r"\centering",
        r"\caption{Model-comparison evidence and guarded temporal validation}",
        r"\label{tab:modelcomparison}",
        r"\small",
        r"\textbf{A. Full-wave model comparison}\\[0.35em]",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Analysis set & Waves & M0 winner & M1 winner & M2 winner & $\Delta\AICc_{\rm M0-M1}>2$ & $\Delta\AICc_{\rm M1-M2}>2$ & $\Delta\AICc_{\rm M1-M2}>10$\\",
        r"\midrule",
    ]

    for _, row in comparison.iterrows():
        tex.append(
            f"{row['analysis_set']} & {int(row['n_waves'])} & "
            f"{int(row['M0_winner'])} & {int(row['M1_winner'])} & {int(row['M2_winner'])} & "
            f"{int(row['delta_AICc_M0_minus_M1_gt2'])} & "
            f"{int(row['delta_AICc_M1_minus_M2_gt2'])} & "
            f"{int(row['delta_AICc_M1_minus_M2_gt10'])}\\\\"
        )

    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\medskip",
        r"\textbf{B. Guarded retrospective tail extrapolation in 42 international waves}\\[0.35em]",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model or procedure & Test winner count & Mean $\RMSElog$ & Median $\RMSElog$ & Waves with lower error than M0\\",
        r"\midrule",
    ]

    for _, row in holdout.iterrows():
        winner = "--" if pd.isna(row["test_winner_count"]) else str(int(row["test_winner_count"]))
        beats = "--" if pd.isna(row["waves_lower_error_than_M0"]) else str(int(row["waves_lower_error_than_M0"]))
        tex.append(
            f"{row['model']} & {winner} & "
            f"{fmt(row['mean_RMSElog'])} & {fmt(row['median_RMSElog'])} & {beats}\\\\"
        )

    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\medskip",
        r"\footnotesize M0=classic SEIR; M1=homogeneous \RASEIR; M2=activity-stratified \RASEIR. "
        r"Positive $\Delta\AICc_{\rm M0-M1}$ favours M1 and positive $\Delta\AICc_{\rm M1-M2}$ favours M2. "
        r"Three very short Chinese waves used the prespecified AIC fallback when AICc was undefined; the $\Delta$AICc columns therefore summarise only the corrected-information-criterion contrasts recorded in the final fit outputs. "
        rf"For temporal validation, the RA-SEIR family was selected using training information only (M1 in {family_m1} waves and M2 in {family_m2}) and then evaluated on the guarded held-out tail. "
        r"The nominal 70\% split was separated from fitting and test scoring by three centred observations on each side so that the raw daily observations supporting fitted and scored centred 7-day values did not overlap. "
        r"The test-set oracle choice between M1 and M2 is not shown because it is not a deployable prediction procedure.",
        r"\end{table}",
        "",
    ]

    output_tex = output_dir / "main_table2_model_comparison_final.tex"
    output_tex.write_text("\n".join(tex), encoding="utf-8")

    print("Full-wave comparison:")
    print(comparison.to_string(index=False))
    print("\nHoldout:")
    print(holdout.to_string(index=False))
    print(f"\nTraining-selected family: M1={family_m1}, M2={family_m2}")
    print(output_tex)


if __name__ == "__main__":
    main()
