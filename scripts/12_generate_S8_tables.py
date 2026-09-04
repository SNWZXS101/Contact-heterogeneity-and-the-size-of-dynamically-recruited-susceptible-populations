#!/usr/bin/env python3
"""Generate final S8 country, variant, and holdout tables from frozen outputs.

Run:
    python scripts/12_generate_S8_tables.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "outputs" / "results"
DEFAULT_TABLES = ROOT / "outputs" / "tables"
MODELS = ("classic", "reservoir", "network")


def latex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text


def num(value: object, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def canonical_h(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(8)


def winner_counts(group: pd.DataFrame) -> dict[str, int]:
    return {m: int(group["winner"].astype(str).eq(m).sum()) for m in MODELS}


def fit_summary_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for value, group in df.groupby(group_col, dropna=False, sort=False):
        counts = winner_counts(group)
        h_col = "network_h_grid" if "network_h_grid" in group.columns else "network_h_cv2"
        rows.append({
            group_col: value,
            "n_waves": int(len(group)),
            "classic_wins": counts["classic"],
            "reservoir_wins": counts["reservoir"],
            "network_wins": counts["network"],
            "median_h": float(canonical_h(group[h_col]).median()),
            "median_peak_Shead_over_Q": float(
                pd.to_numeric(group["network_peak_head_fraction_Q"], errors="coerce").median()
            ),
            "median_peak_Sedge_over_Q": float(
                pd.to_numeric(group["network_peak_edge_fraction_Q"], errors="coerce").median()
            ),
            "median_delta_AICc_M0_minus_M1": float(
                pd.to_numeric(group["delta_aicc_classic_minus_reservoir"], errors="coerce").median()
            ),
            "median_delta_AICc_M1_minus_M2": float(
                pd.to_numeric(group["delta_aicc_reservoir_minus_network"], errors="coerce").median()
            ),
        })
    return pd.DataFrame(rows)


def write_fit_table_tex(df: pd.DataFrame, group_col: str, caption: str, label: str, path: Path) -> None:
    heading = "Country" if group_col == "country" else "Variant-era label"
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        rf"{heading} & $n$ & M0 & M1 & M2 & median $h$ & median $\Shead/Q$ & median $\Sedge/Q$ & median $\Delta\AICc_{{\rm M1-M2}}$\\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{latex_escape(row[group_col])} & "
            f"{int(row['n_waves'])} & {int(row['classic_wins'])} & "
            f"{int(row['reservoir_wins'])} & {int(row['network_wins'])} & "
            f"{num(row['median_h'],2)} & "
            f"{num(row['median_peak_Shead_over_Q'],3)} & "
            f"{num(row['median_peak_Sedge_over_Q'],3)} & "
            f"{num(row['median_delta_AICc_M1_minus_M2'],1)}\\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\medskip",
        r"\footnotesize M0=classic SEIR; M1=homogeneous \RASEIR; M2=activity-stratified \RASEIR. "
        r"Positive $\Delta\AICc_{\rm M1-M2}$ favours M2. "
        r"$h=\CV^2(Z)$ is the fitted contact-activity heterogeneity profile value. "
        r"$\Shead/Q$ and $\Sedge/Q$ are ratios to the fitted observation/accessibility scale $Q$, not census fractions.",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def holdout_summary(hold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = list(dict.fromkeys(hold["country"].astype(str).tolist()))
    for country in order + ["Overall"]:
        group = hold if country == "Overall" else hold.loc[hold["country"].astype(str).eq(country)]
        wins = {m: int(group["winner_test"].astype(str).eq(m).sum()) for m in MODELS}
        m1 = int(group["reservoir_family_selected_train"].astype(str).eq("reservoir").sum())
        m2 = int(group["reservoir_family_selected_train"].astype(str).eq("network").sum())
        beats = int(pd.to_numeric(
            group["reservoir_family_beats_classic_test"], errors="coerce"
        ).fillna(0).sum())
        rows.append({
            "country": country,
            "n_waves": int(len(group)),
            "classic_test_wins": wins["classic"],
            "reservoir_test_wins": wins["reservoir"],
            "network_test_wins": wins["network"],
            "mean_RMSElog_classic": float(pd.to_numeric(group["classic_test_log_rmse"], errors="coerce").mean()),
            "mean_RMSElog_reservoir": float(pd.to_numeric(group["reservoir_test_log_rmse"], errors="coerce").mean()),
            "mean_RMSElog_network": float(pd.to_numeric(group["network_test_log_rmse"], errors="coerce").mean()),
            "family_selected_M1": m1,
            "family_selected_M2": m2,
            "mean_RMSElog_training_selected_family": float(
                pd.to_numeric(group["reservoir_family_test_log_rmse"], errors="coerce").mean()
            ),
            "family_beats_classic_n": beats,
        })
    return pd.DataFrame(rows)


def write_holdout_tex(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        r"\caption{Guarded retrospective tail-extrapolation performance by country}",
        r"\label{tab:app-holdout-summary}",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"Country & $n$ & M0 wins & M1 wins & M2 wins & mean M0 & mean M1 & mean M2 & family M1/M2 & family mean & family beats M0\\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{latex_escape(row['country'])} & {int(row['n_waves'])} & "
            f"{int(row['classic_test_wins'])} & {int(row['reservoir_test_wins'])} & "
            f"{int(row['network_test_wins'])} & "
            f"{num(row['mean_RMSElog_classic'],3)} & "
            f"{num(row['mean_RMSElog_reservoir'],3)} & "
            f"{num(row['mean_RMSElog_network'],3)} & "
            f"{int(row['family_selected_M1'])}/{int(row['family_selected_M2'])} & "
            f"{num(row['mean_RMSElog_training_selected_family'],3)} & "
            f"{int(row['family_beats_classic_n'])}\\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\medskip",
        r"\footnotesize Mean M0, M1, and M2 are mean test $\RMSElog$ values. "
        r"Family M1/M2 gives the number of waves in which the training information criterion selected M1 or M2. "
        r"The family mean is therefore an implementable training-selected reservoir-family result. "
        r"Three centred observations on each side of the nominal 70\% split were excluded from fitting/scoring.",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TABLES)
    args = parser.parse_args()

    results = args.results_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    fit = pd.read_csv(results / "international_fit_summary.csv")
    hold = pd.read_csv(results / "international_holdout_metrics.csv")
    if fit["wave_id"].nunique() != 42 or hold["wave_id"].nunique() != 42:
        raise ValueError("Expected 42 unique international waves in fit and holdout files")

    country = fit_summary_by_group(fit, "country")
    variant_col = "variant_era" if "variant_era" in fit.columns else "variant"
    variant = fit_summary_by_group(fit, variant_col).rename(columns={variant_col: "variant_group"})
    holdout = holdout_summary(hold)

    country.to_csv(out / "appendix_table_country_summary_final.csv", index=False, encoding="utf-8-sig")
    variant.to_csv(out / "appendix_table_variant_summary_final.csv", index=False, encoding="utf-8-sig")
    holdout.to_csv(out / "appendix_table_holdout_summary_final.csv", index=False, encoding="utf-8-sig")

    write_fit_table_tex(
        country, "country", "Final international model-comparison summaries by country",
        "tab:app-country-summary", out / "appendix_table_country_summary_final.tex"
    )
    write_fit_table_tex(
        variant, "variant_group",
        "Final international model-comparison summaries by epidemiological variant-era label",
        "tab:app-variant-summary", out / "appendix_table_variant_summary_final.tex"
    )
    write_holdout_tex(holdout, out / "appendix_table_holdout_summary_final.tex")

    manifest = {
        "n_international_waves": 42,
        "n_holdout_waves": 42,
        "variant_grouping_source_column": variant_col,
        "country_rows": int(len(country)),
        "variant_rows": int(len(variant)),
        "holdout_rows_including_overall": int(len(holdout)),
    }
    (out / "S8_table_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
