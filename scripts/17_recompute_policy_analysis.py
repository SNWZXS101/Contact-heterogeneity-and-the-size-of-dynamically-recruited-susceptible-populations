#!/usr/bin/env python3
"""Recompute final S9 policy associations from frozen Phase-2 results.

This script rebuilds the wave-level policy/model merge, unadjusted Spearman
associations, standardized OLS models with country and variant-era fixed
effects and HC3 robust covariance, multiplicity diagnostics across 24 tests,
and leave-one-country-out sensitivity.

Run from project root:
    python scripts/17_recompute_policy_analysis.py

Expected inputs:
    outputs/results/international_fit_summary.csv
    data/international/international_wave_policy_covariates.csv

Outputs:
    outputs/policy/international_model_policy_merged_final.csv
    outputs/policy/policy_spearman_final.csv
    outputs/policy/policy_regression_coefficients_final.csv
    outputs/policy/policy_regression_leave_one_country_out_final.csv
    outputs/policy/policy_regression_loco_summary_final.csv
    outputs/tables/appendix_table_policy_regression_final.tex
    outputs/tables/policy_highlight_sentence_final.tex
    outputs/policy/policy_analysis_manifest.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIT = ROOT / "outputs" / "results" / "international_fit_summary.csv"
DEFAULT_POLICY = ROOT / "data" / "international" / "international_wave_policy_covariates.csv"
DEFAULT_OUT = ROOT / "outputs" / "policy"
DEFAULT_TABLES = ROOT / "outputs" / "tables"

PREDICTOR_CANDIDATES = {
    "gating": [
        "early_reservoir_gating_index",
        "reservoir_gating_index_early",
        "early_gating_index",
        "reservoir_gating_index",
    ],
    "detection": [
        "early_detection_index",
        "detection_index_early",
        "detection_index",
    ],
    "barrier": [
        "early_barrier_information_index",
        "barrier_information_index_early",
        "barrier_information_index",
    ],
    "vaccination": [
        "peak_people_fully_vaccinated_per_hundred",
        "people_fully_vaccinated_per_hundred_peak",
        "peak_fully_vaccinated_per_hundred",
    ],
}

OUTCOME_CANDIDATES = {
    "peak_Shead_over_Q": ["network_peak_head_fraction_Q"],
    "accessed_reservoir_fraction_Q": ["network_accessed_reservoir_fraction_Q"],
    "q": ["network_q"],
    "h": ["network_h_grid", "network_h_cv2"],
    "network_AICc_support": ["delta_aicc_reservoir_minus_network"],
    "reservoir_AICc_support": ["delta_aicc_classic_minus_reservoir"],
}

OUTCOME_LABEL_TEX = {
    "peak_Shead_over_Q": r"peak $\Shead/Q$",
    "accessed_reservoir_fraction_Q": r"reservoir accessed fraction",
    "q": r"$q$",
    "h": r"$h$",
    "network_AICc_support": r"network-model $\Delta\AICc$ support",
    "reservoir_AICc_support": r"reservoir-model $\Delta\AICc$ support",
}

PREDICTOR_LABEL_TEX = {
    "gating": "gating",
    "detection": "detection",
    "barrier": "barrier--information",
    "vaccination": "fully vaccinated coverage",
}


def first_existing(columns: Iterable[str], candidates: List[str], label: str) -> str:
    columns = set(columns)
    for name in candidates:
        if name in columns:
            return name
    raise KeyError(f"Could not find {label}; tried {candidates}")


def coalesce_column(df: pd.DataFrame, base: str) -> pd.Series:
    if base in df.columns:
        return df[base]
    for suffix in ("_fit", "_policy", "_x", "_y"):
        name = base + suffix
        if name in df.columns:
            return df[name]
    raise KeyError(base)


def zscore(series: pd.Series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(x)
    return (x - mean) / sd


def design_matrix(df: pd.DataFrame, predictors: List[str]) -> Tuple[np.ndarray, List[str]]:
    cols = [np.ones(len(df), dtype=float)]
    names = ["Intercept"]

    for predictor in predictors:
        cols.append(zscore(df[predictor]))
        names.append(predictor)

    for factor in ("country", "variant_era"):
        values = df[factor].astype(str)
        levels = sorted(values.unique().tolist())
        for level in levels[1:]:
            cols.append(values.eq(level).astype(float).to_numpy())
            names.append(f"{factor}[{level}]")

    return np.column_stack(cols), names


def hc3_ols(df: pd.DataFrame, outcome: str, predictors: List[str]) -> Tuple[pd.DataFrame, dict]:
    required = [outcome, *predictors, "country", "variant_era"]
    data = df[required].copy()
    for col in [outcome, *predictors]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna().reset_index(drop=True)

    y = zscore(data[outcome])
    X, names = design_matrix(data, predictors)

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta

    # HC3 leverage correction.
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    denom = np.maximum(1.0 - leverage, 1e-8)
    adj = resid / denom
    meat = X.T @ ((adj ** 2)[:, None] * X)
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    zstat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvals = np.array([math.erfc(abs(v) / math.sqrt(2.0)) for v in zstat])
    ci_low = beta - 1.959963984540054 * se
    ci_high = beta + 1.959963984540054 * se

    rank = int(np.linalg.matrix_rank(X))
    result = pd.DataFrame({
        "term": names,
        "estimate": beta,
        "se_hc3": se,
        "z": zstat,
        "p": pvals,
        "ci_low": ci_low,
        "ci_high": ci_high,
    })
    meta = {
        "n": int(len(data)),
        "design_columns": int(X.shape[1]),
        "design_rank": rank,
    }
    return result, meta


def holm_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for rev_rank in range(m - 1, -1, -1):
        idx = order[rev_rank]
        rank = rev_rank + 1
        value = min(1.0, p[idx] * m / rank)
        running = min(running, value)
        adjusted[idx] = running
    return adjusted


def latex_escape(text: object) -> str:
    s = str(text)
    for old, new in [
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
        ("{", r"\{"), ("}", r"\}"),
    ]:
        s = s.replace(old, new)
    return s


def fmt(x: float, digits: int = 3) -> str:
    return "--" if not np.isfinite(x) else f"{x:.{digits}f}"


def write_regression_table(coef: pd.DataFrame, loco_summary: pd.DataFrame, path: Path) -> None:
    merged = coef.merge(
        loco_summary,
        on=["outcome", "predictor"],
        how="left",
        validate="one_to_one",
    )
    lines = [
        r"\begin{landscape}",
        r"\begin{longtable}{p{4.1cm}p{2.7cm}rrrrrr}",
        r"\caption{Descriptive policy associations with final frozen model outcomes}\label{tab:app-policy-regression}\\",
        r"\toprule",
        r"Outcome & Predictor & Estimate & 95\% CI & $p$ & Holm $p$ & LOCO range & Same-sign LOCO\\",
        r"\midrule\endfirsthead",
        r"\toprule",
        r"Outcome & Predictor & Estimate & 95\% CI & $p$ & Holm $p$ & LOCO range & Same-sign LOCO\\",
        r"\midrule\endhead",
    ]
    for _, row in merged.iterrows():
        outcome = OUTCOME_LABEL_TEX[str(row["outcome"])]
        predictor = PREDICTOR_LABEL_TEX[str(row["predictor"])]
        ci = f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]"
        loco = f"[{fmt(row['loco_min'])}, {fmt(row['loco_max'])}]"
        same = f"{int(row['same_sign_n'])}/{int(row['loco_n'])}"
        lines.append(
            f"{outcome} & {predictor} & {fmt(row['estimate'])} & "
            f"{ci} & {fmt(row['p'],3)} & {fmt(row['p_holm'],3)} & "
            f"{loco} & {same}\\\\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{8}{p{24cm}}{\footnotesize "
        r"Continuous predictors and outcomes were standardised before fitting. "
        r"All models included country and epidemiological variant-era fixed effects and HC3 robust covariance. "
        r"Twenty-four outcome--predictor coefficients were examined. Holm-adjusted values are shown as a multiplicity sensitivity analysis; "
        r"LOCO=leave one country out. These ecological associations are descriptive and are not causal intervention-effect estimates.}\\",
        r"\end{longtable}",
        r"\end{landscape}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_highlight_sentence(coef: pd.DataFrame, loco_summary: pd.DataFrame, path: Path) -> None:
    merged = coef.merge(
        loco_summary,
        on=["outcome", "predictor"],
        how="left",
        validate="one_to_one",
    ).sort_values("p")
    row = merged.iloc[0]
    outcome = OUTCOME_LABEL_TEX[str(row["outcome"])]
    predictor = PREDICTOR_LABEL_TEX[str(row["predictor"])]
    sentence = (
        "The smallest nominal adjusted-model association was observed for "
        f"{predictor} with {outcome}: the standardised coefficient was "
        f"{row['estimate']:.3f} (95\\% CI {row['ci_low']:.3f} to {row['ci_high']:.3f}; "
        f"$p={row['p']:.3f}$; Holm-adjusted $p={row['p_holm']:.3f}$). "
        f"Across leave-one-country-out analyses, the coefficient ranged from "
        f"{row['loco_min']:.3f} to {row['loco_max']:.3f} and retained the same "
        f"sign in {int(row['same_sign_n'])} of {int(row['loco_n'])} exclusions."
    )
    path.write_text(sentence + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-summary", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--policy-covariates", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    args = parser.parse_args()

    fit = pd.read_csv(args.fit_summary.expanduser().resolve())
    policy = pd.read_csv(args.policy_covariates.expanduser().resolve())

    if fit["wave_id"].nunique() != 42 or policy["wave_id"].nunique() != 42:
        raise ValueError("Expected 42 unique international waves in both inputs")

    merged = fit.merge(policy, on="wave_id", how="inner", suffixes=("_fit", "_policy"))
    if merged["wave_id"].nunique() != 42:
        raise ValueError("Merge did not retain exactly 42 unique waves")

    merged["country"] = coalesce_column(merged, "country").astype(str)
    try:
        merged["variant_era"] = coalesce_column(merged, "variant_era").astype(str)
    except KeyError:
        merged["variant_era"] = coalesce_column(merged, "variant").astype(str)

    predictor_source: Dict[str, str] = {}
    for short, candidates in PREDICTOR_CANDIDATES.items():
        source = first_existing(merged.columns, candidates, f"policy predictor {short}")
        predictor_source[short] = source
        merged[short] = pd.to_numeric(merged[source], errors="coerce")

    outcome_source: Dict[str, str] = {}
    for short, candidates in OUTCOME_CANDIDATES.items():
        source = first_existing(merged.columns, candidates, f"model outcome {short}")
        outcome_source[short] = source
        merged[short] = pd.to_numeric(merged[source], errors="coerce")

    out = args.output_dir.expanduser().resolve()
    tables = args.tables_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    merged.to_csv(out / "international_model_policy_merged_final.csv", index=False, encoding="utf-8-sig")

    predictors = list(PREDICTOR_CANDIDATES)
    outcomes = list(OUTCOME_CANDIDATES)

    # Unadjusted Spearman.
    spearman_rows = []
    for outcome in outcomes:
        for predictor in predictors:
            pair = merged[[outcome, predictor]].dropna()
            rho, p = spearmanr(pair[outcome], pair[predictor])
            spearman_rows.append({
                "outcome": outcome,
                "predictor": predictor,
                "n": int(len(pair)),
                "rho": float(rho),
                "p": float(p),
            })
    spearman = pd.DataFrame(spearman_rows)
    spearman.to_csv(out / "policy_spearman_final.csv", index=False, encoding="utf-8-sig")

    # Adjusted OLS/HC3.
    coef_rows = []
    model_meta = []
    for outcome in outcomes:
        fit_table, meta = hc3_ols(merged, outcome, predictors)
        model_meta.append({"outcome": outcome, **meta})
        for predictor in predictors:
            row = fit_table.loc[fit_table["term"].eq(predictor)].iloc[0]
            coef_rows.append({
                "outcome": outcome,
                "predictor": predictor,
                "n": meta["n"],
                "estimate": float(row["estimate"]),
                "se_hc3": float(row["se_hc3"]),
                "z": float(row["z"]),
                "p": float(row["p"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
            })
    coef = pd.DataFrame(coef_rows)
    coef["p_holm"] = holm_adjust(coef["p"].to_numpy())
    coef["p_bh"] = bh_adjust(coef["p"].to_numpy())
    coef.to_csv(out / "policy_regression_coefficients_final.csv", index=False, encoding="utf-8-sig")

    # Leave-one-country-out.
    loco_rows = []
    countries = sorted(merged["country"].unique().tolist())
    for omitted in countries:
        subset = merged.loc[~merged["country"].eq(omitted)].copy()
        for outcome in outcomes:
            fit_table, meta = hc3_ols(subset, outcome, predictors)
            for predictor in predictors:
                row = fit_table.loc[fit_table["term"].eq(predictor)].iloc[0]
                loco_rows.append({
                    "omitted_country": omitted,
                    "outcome": outcome,
                    "predictor": predictor,
                    "n": meta["n"],
                    "estimate": float(row["estimate"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                    "p": float(row["p"]),
                })
    loco = pd.DataFrame(loco_rows)
    loco.to_csv(
        out / "policy_regression_leave_one_country_out_final.csv",
        index=False, encoding="utf-8-sig"
    )

    full_lookup = coef.set_index(["outcome", "predictor"])["estimate"]
    loco_summary_rows = []
    for (outcome, predictor), group in loco.groupby(["outcome", "predictor"]):
        full = float(full_lookup.loc[(outcome, predictor)])
        est = pd.to_numeric(group["estimate"], errors="coerce")
        same = int((np.sign(est) == np.sign(full)).sum()) if full != 0 else 0
        loco_summary_rows.append({
            "outcome": outcome,
            "predictor": predictor,
            "loco_n": int(est.notna().sum()),
            "loco_min": float(est.min()),
            "loco_max": float(est.max()),
            "same_sign_n": same,
        })
    loco_summary = pd.DataFrame(loco_summary_rows)
    loco_summary.to_csv(
        out / "policy_regression_loco_summary_final.csv",
        index=False, encoding="utf-8-sig"
    )

    write_regression_table(
        coef, loco_summary, tables / "appendix_table_policy_regression_final.tex"
    )
    write_highlight_sentence(
        coef, loco_summary, tables / "policy_highlight_sentence_final.tex"
    )

    manifest = {
        "n_merged_waves": int(merged["wave_id"].nunique()),
        "predictor_source_columns": predictor_source,
        "outcome_source_columns": outcome_source,
        "predictors": predictors,
        "outcomes": outcomes,
        "n_adjusted_tests": int(len(coef)),
        "country_levels": countries,
        "variant_era_levels": sorted(merged["variant_era"].unique().tolist()),
        "fixed_effects": ["country", "variant_era"],
        "covariance": "HC3",
        "multiplicity_sensitivity": ["Holm", "Benjamini-Hochberg"],
    }
    (out / "policy_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
