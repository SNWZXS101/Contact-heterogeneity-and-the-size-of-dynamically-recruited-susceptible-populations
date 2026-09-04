#!/usr/bin/env python3
"""Generate final S10 counterfactual summary tables and result sentence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
TABLES = ROOT / "outputs" / "tables"

BASELINE = "Baseline observed-response fit"
SCENARIO_ORDER = [
    "Reservoir gating (a -50%)",
    "Broad contact reduction (beta -30%)",
    "Faster detection/isolation (gamma +50%)",
    "Target highest-activity quartile (top 3 classes -50%)",
    "Immune protection (25% susceptible protected)",
    "Adaptive combined package",
]
SHORT = {
    SCENARIO_ORDER[0]: "Reservoir gating",
    SCENARIO_ORDER[1]: "Broad contact reduction",
    SCENARIO_ORDER[2]: "Faster isolation",
    SCENARIO_ORDER[3]: "Highest-activity quartile targeting",
    SCENARIO_ORDER[4]: "Immune protection",
    SCENARIO_ORDER[5]: "Combined package",
}


def f3(x):
    return f"{float(x):.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--tables-dir", type=Path, default=TABLES)
    args = parser.parse_args()

    results = args.results_dir.expanduser().resolve()
    tables = args.tables_dir.expanduser().resolve()
    tables.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(results / "policy_counterfactual_summary_final.csv")
    manifest = json.loads(
        (results / "policy_counterfactual_manifest_final.json").read_text(encoding="utf-8")
    )

    rows = []
    for scenario in SCENARIO_ORDER:
        g = d.loc[d["scenario"].eq(scenario)]
        row = {"scenario": scenario}
        for col in (
            "percent_reduction_peak_incidence",
            "percent_reduction_cumulative_incidence",
            "percent_reduction_max_S_head",
            "percent_reduction_max_S_edge",
        ):
            row[f"{col}_median"] = float(g[col].median())
            row[f"{col}_q25"] = float(g[col].quantile(.25))
            row[f"{col}_q75"] = float(g[col].quantile(.75))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(
        tables / "appendix_table_counterfactual_summary_final.csv",
        index=False, encoding="utf-8-sig"
    )

    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        r"\caption{Conditional mechanism counterfactuals across seven representative fitted waves}",
        r"\label{tab:app-counterfactual-summary}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Scenario & Peak incidence & Cumulative incidence & max $\Shead$ & max $\Sedge$\\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{SHORT[row['scenario']]} & "
            f"{f3(row['percent_reduction_peak_incidence_median'])}\\% & "
            f"{f3(row['percent_reduction_cumulative_incidence_median'])}\\% & "
            f"{f3(row['percent_reduction_max_S_head_median'])}\\% & "
            f"{f3(row['percent_reduction_max_S_edge_median'])}\\%\\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\medskip",
        r"\footnotesize Entries are median percentage reductions relative to each wave's frozen M2 baseline. "
        r"The seven representatives are W072, IT04, JP06, KR06, UK03, US04, and ZA04. "
        r"Highest-activity targeting halves activity in the top three of 12 equal-mass activity classes (25\% of the model population) without renormalising the remaining activity values. "
        r"All experiments are conditional structural perturbations, not estimates of historical policy effects.",
        r"\end{table}",
        "",
    ]
    (tables / "appendix_table_counterfactual_summary_final.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    med = summary.set_index("scenario")
    st = manifest["structural_summary"]
    sentence = (
        "Across the seven representative waves, the median reductions in peak "
        "and cumulative incidence were "
        f"{f3(med.loc[SCENARIO_ORDER[0],'percent_reduction_peak_incidence_median'])}\\% and "
        f"{f3(med.loc[SCENARIO_ORDER[0],'percent_reduction_cumulative_incidence_median'])}\\% for reservoir gating; "
        f"{f3(med.loc[SCENARIO_ORDER[1],'percent_reduction_peak_incidence_median'])}\\% and "
        f"{f3(med.loc[SCENARIO_ORDER[1],'percent_reduction_cumulative_incidence_median'])}\\% for broad contact reduction; "
        f"{f3(med.loc[SCENARIO_ORDER[2],'percent_reduction_peak_incidence_median'])}\\% and "
        f"{f3(med.loc[SCENARIO_ORDER[2],'percent_reduction_cumulative_incidence_median'])}\\% for faster isolation; "
        f"{f3(med.loc[SCENARIO_ORDER[3],'percent_reduction_peak_incidence_median'])}\\% and "
        f"{f3(med.loc[SCENARIO_ORDER[3],'percent_reduction_cumulative_incidence_median'])}\\% for highest-activity-quartile targeting; and "
        f"{f3(med.loc[SCENARIO_ORDER[4],'percent_reduction_peak_incidence_median'])}\\% and "
        f"{f3(med.loc[SCENARIO_ORDER[4],'percent_reduction_cumulative_incidence_median'])}\\% for 25\\% immune protection. "
        f"In the X003 initial-$\\Rinit$-matched structural experiment, increasing $h$ from 0 to 4 changed maximum $\\Shead$ by "
        f"{f3(st['h0_to_h4_percent_reduction_max_S_head'])}\\% and maximum $\\Sedge$ by "
        f"{f3(st['h0_to_h4_percent_reduction_max_S_edge'])}\\%."
    )
    (tables / "counterfactual_result_sentence_final.tex").write_text(
        sentence + "\n", encoding="utf-8"
    )

    print(summary[[
        "scenario",
        "percent_reduction_peak_incidence_median",
        "percent_reduction_cumulative_incidence_median",
        "percent_reduction_max_S_head_median",
        "percent_reduction_max_S_edge_median",
    ]].to_string(index=False))
    print(json.dumps(st, indent=2))


if __name__ == "__main__":
    main()
