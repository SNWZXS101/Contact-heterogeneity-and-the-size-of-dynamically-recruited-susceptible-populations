#!/usr/bin/env python3
"""Audit the Phase-2 final numerical freeze against bundled reference results.

Run from project root:
    python scripts/09_audit_final_freeze.py

Inputs
------
reference_results/
    international_fit_summary.csv
    china_fit_summary_111_three_models.csv
    international_holdout_metrics.csv   (old/original reference)
outputs/results/
    international_fit_summary.csv
    china_fit_summary_111_three_models.csv
    international_holdout_metrics.csv
    international_fit_manifest.json
    china_111_analysis_manifest.json
outputs/tables/
    complete_153_wave_table_wide.csv
    complete_153_wave_table_manifest.json

Outputs
-------
outputs/audit/
    final_freeze_audit_summary.json
    final_freeze_audit.txt
    winner_changes.csv
    h_changes.csv
    representative_wave_comparison.csv
    numerical_sensitive_wave_comparison.csv
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

REF = ROOT / "reference_results"
NEW = ROOT / "outputs" / "results"
TABLES = ROOT / "outputs" / "tables"
OUT = ROOT / "outputs" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ("classic", "reservoir", "network")
REPRESENTATIVES = ["W072", "X003", "IT04", "JP06", "KR06", "UK03", "US04", "ZA04"]
NUMERICALLY_SENSITIVE_CHINA = [
    "W004", "W029", "W033", "W034", "W037", "W050", "W055", "W061",
    "W062", "W063", "W064", "W065", "W093", "W095", "W097", "W098", "X002",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def int_counts(series: pd.Series, values: Iterable[str] = MODELS) -> Dict[str, int]:
    return {name: int(series.astype(str).eq(name).sum()) for name in values}


def pct_counts(counts: Dict[str, int], n: int) -> Dict[str, float]:
    return {key: 100.0 * value / n for key, value in counts.items()}


def finite_mean(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    return float(x[np.isfinite(x)].mean())


def finite_median(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    return float(x[np.isfinite(x)].median())


def safe_count_gt(df: pd.DataFrame, column: str, threshold: float) -> int:
    if column not in df:
        return 0
    x = pd.to_numeric(df[column], errors="coerce")
    return int((x > threshold).sum())


def success_failures(df: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for model in MODELS:
        column = f"{model}_success"
        if column in df:
            s = df[column]
            bad = df.loc[~s.astype(str).str.lower().isin(["true", "1", "1.0"]), "wave_id"]
            out[model] = bad.astype(str).tolist()
    return out


def h_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Count h values after numerical canonicalisation.

    Some stored finite-class values differ from their target grid only at
    floating-point round-off (for example 0.2499999999999999 vs 0.25).
    Canonicalise before grouping so combined China+international counts are
    summed rather than overwritten after string formatting.
    """
    if "network_h_cv2" not in df:
        return {}
    h = pd.to_numeric(df["network_h_cv2"], errors="coerce").round(8)
    counts = h.value_counts(dropna=False).sort_index()
    result: Dict[str, int] = {}
    for key, value in counts.items():
        label = "NA" if pd.isna(key) else f"{float(key):g}"
        result[label] = int(value)
    return result


def compare_scope(old: pd.DataFrame, new: pd.DataFrame, label: str) -> dict:
    old_counts = int_counts(old["winner"])
    new_counts = int_counts(new["winner"])
    return {
        "label": label,
        "n_old": int(len(old)),
        "n_new": int(len(new)),
        "old_winner_counts": old_counts,
        "new_winner_counts": new_counts,
        "old_winner_percent": pct_counts(old_counts, len(old)),
        "new_winner_percent": pct_counts(new_counts, len(new)),
        "net_change": {m: new_counts[m] - old_counts[m] for m in MODELS},
        "old_h_distribution": h_distribution(old),
        "new_h_distribution": h_distribution(new),
        "old_delta_C_R_gt2": safe_count_gt(old, "delta_aicc_classic_minus_reservoir", 2),
        "new_delta_C_R_gt2": safe_count_gt(new, "delta_aicc_classic_minus_reservoir", 2),
        "old_delta_R_N_gt2": safe_count_gt(old, "delta_aicc_reservoir_minus_network", 2),
        "new_delta_R_N_gt2": safe_count_gt(new, "delta_aicc_reservoir_minus_network", 2),
        "old_delta_R_N_gt10": safe_count_gt(old, "delta_aicc_reservoir_minus_network", 10),
        "new_delta_R_N_gt10": safe_count_gt(new, "delta_aicc_reservoir_minus_network", 10),
        "new_optimizer_failures": success_failures(new),
    }


def make_change_table(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    old2 = old.copy()
    new2 = new.copy()
    keep = [
        "wave_id", "winner", "network_h_cv2",
        "delta_aicc_classic_minus_reservoir",
        "delta_aicc_reservoir_minus_network",
        "classic_aicc", "reservoir_aicc", "network_aicc",
        "reservoir_fit_source",
    ]
    old_keep = [c for c in keep if c in old2.columns]
    new_keep = [c for c in keep if c in new2.columns]
    a = old2[old_keep].copy().add_prefix("old_")
    a = a.rename(columns={"old_wave_id": "wave_id"})
    b = new2[new_keep].copy().add_prefix("new_")
    b = b.rename(columns={"new_wave_id": "wave_id"})
    return a.merge(b, on="wave_id", how="outer", validate="one_to_one")


def summarize_holdout(new_hold: pd.DataFrame, old_hold: pd.DataFrame | None) -> dict:
    result: dict = {"n_waves": int(len(new_hold))}
    result["new_winner_counts"] = int_counts(new_hold["winner_test"])
    result["new_mean_log_rmse"] = {
        model: finite_mean(new_hold[f"{model}_test_log_rmse"]) for model in MODELS
    }
    result["new_median_log_rmse"] = {
        model: finite_median(new_hold[f"{model}_test_log_rmse"]) for model in MODELS
    }

    if "reservoir_family_selected_train" in new_hold:
        result["training_selected_family_counts"] = int_counts(
            new_hold["reservoir_family_selected_train"], ("reservoir", "network")
        )
    if "reservoir_family_test_log_rmse" in new_hold:
        result["training_selected_family_mean_log_rmse"] = finite_mean(
            new_hold["reservoir_family_test_log_rmse"]
        )
        result["training_selected_family_median_log_rmse"] = finite_median(
            new_hold["reservoir_family_test_log_rmse"]
        )
    if "reservoir_family_beats_classic_test" in new_hold:
        result["training_selected_family_beats_classic_n"] = int(
            pd.to_numeric(new_hold["reservoir_family_beats_classic_test"], errors="coerce").fillna(0).sum()
        )
    if "oracle_reservoir_family_test_log_rmse" in new_hold:
        result["oracle_family_mean_log_rmse"] = finite_mean(
            new_hold["oracle_reservoir_family_test_log_rmse"]
        )
        result["oracle_family_median_log_rmse"] = finite_median(
            new_hold["oracle_reservoir_family_test_log_rmse"]
        )

    guard_cols = [
        "n_train_nominal", "n_train_fitted", "smoothing_half_window_days",
        "pre_split_guard_days", "post_split_guard_days", "test_start_day", "n_test",
    ]
    result["guard_columns_present"] = {c: bool(c in new_hold.columns) for c in guard_cols}
    if "pre_split_guard_days" in new_hold:
        result["pre_guard_unique"] = sorted(
            pd.to_numeric(new_hold["pre_split_guard_days"], errors="coerce").dropna().unique().tolist()
        )
    if "post_split_guard_days" in new_hold:
        result["post_guard_unique"] = sorted(
            pd.to_numeric(new_hold["post_split_guard_days"], errors="coerce").dropna().unique().tolist()
        )

    if old_hold is not None:
        result["old_winner_counts"] = int_counts(old_hold["winner_test"])
        result["old_mean_log_rmse"] = {
            model: finite_mean(old_hold[f"{model}_test_log_rmse"]) for model in MODELS
        }
        result["old_median_log_rmse"] = {
            model: finite_median(old_hold[f"{model}_test_log_rmse"]) for model in MODELS
        }
    return result


def select_wave_rows(
    old_china: pd.DataFrame,
    new_china: pd.DataFrame,
    old_intl: pd.DataFrame,
    new_intl: pd.DataFrame,
    wave_ids: List[str],
) -> pd.DataFrame:
    old = pd.concat([old_china, old_intl], ignore_index=True, sort=False)
    new = pd.concat([new_china, new_intl], ignore_index=True, sort=False)
    cmp = make_change_table(old, new)
    return cmp[cmp["wave_id"].astype(str).isin(wave_ids)].copy()


def main() -> None:
    old_intl = read_csv(REF / "international_fit_summary.csv")
    old_china = read_csv(REF / "china_fit_summary_111_three_models.csv")
    new_intl = read_csv(NEW / "international_fit_summary.csv")
    new_china = read_csv(NEW / "china_fit_summary_111_three_models.csv")
    new_hold = read_csv(NEW / "international_holdout_metrics.csv")

    old_hold_path = REF / "international_holdout_metrics.csv"
    old_hold = read_csv(old_hold_path) if old_hold_path.exists() else None

    wide = read_csv(TABLES / "complete_153_wave_table_wide.csv")
    intl_manifest = read_json(NEW / "international_fit_manifest.json")
    china_manifest = read_json(NEW / "china_111_analysis_manifest.json")
    table_manifest = read_json(TABLES / "complete_153_wave_table_manifest.json")

    if old_intl["wave_id"].nunique() != 42 or new_intl["wave_id"].nunique() != 42:
        raise ValueError("International summary must contain 42 unique waves in old and new results.")
    if old_china["wave_id"].nunique() != 111 or new_china["wave_id"].nunique() != 111:
        raise ValueError("China summary must contain 111 unique waves in old and new results.")
    if wide["wave_id"].nunique() != 153:
        raise ValueError("Final wide table must contain 153 unique waves.")

    intl_audit = compare_scope(old_intl, new_intl, "International 42")
    china_audit = compare_scope(old_china, new_china, "China 111")

    old_all = pd.concat([old_china, old_intl], ignore_index=True, sort=False)
    new_all = pd.concat([new_china, new_intl], ignore_index=True, sort=False)
    all_audit = compare_scope(old_all, new_all, "All 153")

    intl_changes = make_change_table(old_intl, new_intl)
    china_changes = make_change_table(old_china, new_china)
    winner_changes = pd.concat(
        [
            intl_changes.assign(scope="international"),
            china_changes.assign(scope="china"),
        ],
        ignore_index=True,
        sort=False,
    )
    winner_changes = winner_changes.loc[
        winner_changes["old_winner"].astype(str) != winner_changes["new_winner"].astype(str)
    ].copy()
    winner_changes.to_csv(OUT / "winner_changes.csv", index=False, encoding="utf-8-sig")

    h_changes = pd.concat(
        [
            intl_changes.assign(scope="international"),
            china_changes.assign(scope="china"),
        ],
        ignore_index=True,
        sort=False,
    )
    if {"old_network_h_cv2", "new_network_h_cv2"}.issubset(h_changes.columns):
        old_h = pd.to_numeric(h_changes["old_network_h_cv2"], errors="coerce")
        new_h = pd.to_numeric(h_changes["new_network_h_cv2"], errors="coerce")
        h_changes = h_changes.loc[~np.isclose(old_h, new_h, equal_nan=True)].copy()
    else:
        h_changes = h_changes.iloc[0:0].copy()
    h_changes.to_csv(OUT / "h_changes.csv", index=False, encoding="utf-8-sig")

    reps = select_wave_rows(old_china, new_china, old_intl, new_intl, REPRESENTATIVES)
    reps.to_csv(OUT / "representative_wave_comparison.csv", index=False, encoding="utf-8-sig")

    sensitive = select_wave_rows(
        old_china, new_china, old_intl, new_intl, NUMERICALLY_SENSITIVE_CHINA
    )
    sensitive.to_csv(
        OUT / "numerical_sensitive_wave_comparison.csv", index=False, encoding="utf-8-sig"
    )

    holdout_audit = summarize_holdout(new_hold, old_hold)

    accessed = pd.to_numeric(
        wide.get("network_accessed_reservoir_fraction_Q", pd.Series(dtype=float)),
        errors="coerce",
    )
    physical = pd.to_numeric(
        wide.get("network_diagnostic_physically_valid", pd.Series(dtype=float)),
        errors="coerce",
    )
    diagnostics = {
        "n_rows": int(len(wide)),
        "unique_waves": int(wide["wave_id"].nunique()),
        "physically_valid_n": int((physical == 1).sum()) if len(physical) else None,
        "accessed_fraction_min": float(accessed.min()) if accessed.notna().any() else None,
        "accessed_fraction_max": float(accessed.max()) if accessed.notna().any() else None,
        "accessed_fraction_negative_n": int((accessed < -1e-8).sum()) if accessed.notna().any() else None,
        "accessed_fraction_gt1_n": int((accessed > 1.000001).sum()) if accessed.notna().any() else None,
    }

    audit = {
        "international": intl_audit,
        "china": china_audit,
        "all_153": all_audit,
        "holdout": holdout_audit,
        "network_diagnostics": diagnostics,
        "manifests": {
            "international_fit_manifest": intl_manifest,
            "china_111_analysis_manifest": china_manifest,
            "complete_153_wave_table_manifest": table_manifest,
        },
        "winner_change_n": int(len(winner_changes)),
        "h_change_n": int(len(h_changes)),
        "representative_wave_ids": REPRESENTATIVES,
        "numerically_sensitive_china_wave_ids": NUMERICALLY_SENSITIVE_CHINA,
    }

    (OUT / "final_freeze_audit_summary.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines: List[str] = []
    lines.append("PHASE-2 FINAL NUMERICAL FREEZE AUDIT")
    lines.append("=" * 72)
    for scope_key in ("all_153", "china", "international"):
        block = audit[scope_key]
        lines.append("")
        lines.append(block["label"])
        lines.append(f"  old winners: {block['old_winner_counts']}")
        lines.append(f"  new winners: {block['new_winner_counts']}")
        lines.append(f"  net change : {block['net_change']}")
        lines.append(f"  old h: {block['old_h_distribution']}")
        lines.append(f"  new h: {block['new_h_distribution']}")
        lines.append(
            "  ΔAICc>2 C-R old/new: "
            f"{block['old_delta_C_R_gt2']}/{block['new_delta_C_R_gt2']}"
        )
        lines.append(
            "  ΔAICc>2 R-N old/new: "
            f"{block['old_delta_R_N_gt2']}/{block['new_delta_R_N_gt2']}"
        )
        lines.append(
            "  ΔAICc>10 R-N old/new: "
            f"{block['old_delta_R_N_gt10']}/{block['new_delta_R_N_gt10']}"
        )
        lines.append(f"  optimizer failures (new): {block['new_optimizer_failures']}")

    lines.append("")
    lines.append("Holdout")
    lines.append(f"  new winner counts: {holdout_audit.get('new_winner_counts')}")
    lines.append(f"  new mean log-RMSE: {holdout_audit.get('new_mean_log_rmse')}")
    lines.append(
        "  training-selected family counts: "
        f"{holdout_audit.get('training_selected_family_counts')}"
    )
    lines.append(
        "  training-selected family mean log-RMSE: "
        f"{holdout_audit.get('training_selected_family_mean_log_rmse')}"
    )
    lines.append(
        "  training-selected family beats classic: "
        f"{holdout_audit.get('training_selected_family_beats_classic_n')}"
    )
    lines.append(
        "  oracle family mean log-RMSE (diagnostic only): "
        f"{holdout_audit.get('oracle_family_mean_log_rmse')}"
    )

    lines.append("")
    lines.append("Network diagnostics")
    for key, value in diagnostics.items():
        lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append(f"Winner changes: {len(winner_changes)}")
    if len(winner_changes):
        for _, row in winner_changes.iterrows():
            lines.append(
                f"  {row['wave_id']}: {row.get('old_winner')} -> {row.get('new_winner')}, "
                f"h {row.get('old_network_h_cv2')} -> {row.get('new_network_h_cv2')}"
            )
    lines.append("")
    lines.append(f"h changes: {len(h_changes)}")
    lines.append("See CSV outputs for full wave-level comparison.")

    (OUT / "final_freeze_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nAudit outputs written to: {OUT}")


if __name__ == "__main__":
    main()
