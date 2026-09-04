#!/usr/bin/env python3
"""Generate final main-text Table 1: study settings and wave architecture.

This table is the natural companion to Figure 2. It is generated from final
wave catalogues rather than maintained by hand.

Inputs
------
data/international/international_wave_catalog.csv
outputs/results/china_fit_summary_111_three_models.csv

Outputs
-------
outputs/tables/main_table1_settings_final.csv
outputs/tables/main_table1_settings_final.tex

Run
---
python scripts/31b_generate_main_table1_settings_final.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INTERNATIONAL = ROOT / "data" / "international"
RESULTS = ROOT / "outputs" / "results"
TABLES = ROOT / "outputs" / "tables"

COUNTRY_ORDER = [
    "China",
    "Italy",
    "Japan",
    "South Africa",
    "South Korea",
    "United Kingdom",
    "United States",
]


def fmt_date(value) -> str:
    return pd.Timestamp(value).strftime("%d %b %Y")


def variant_eras(values: pd.Series) -> str:
    items = [str(value) for value in values.dropna().tolist()]
    joined = " ".join(items).lower()

    eras = []
    if any(token in joined for token in ["ancestral", "pre-alpha", "pre alpha", "alpha"]):
        eras.append("pre-Alpha/Alpha")
    if "delta" in joined:
        eras.append("Delta")
    if "omicron" in joined or "ba." in joined:
        eras.append("Omicron")

    if not eras:
        # Fall back to preserving the actual labels if unexpected input appears.
        eras = list(dict.fromkeys(items))
    return "; ".join(eras)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--international-catalog",
        type=Path,
        default=INTERNATIONAL / "international_wave_catalog.csv",
    )
    parser.add_argument(
        "--china-summary",
        type=Path,
        default=RESULTS / "china_fit_summary_111_three_models.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=TABLES)
    args = parser.parse_args()

    catalog = pd.read_csv(
        args.international_catalog.expanduser().resolve(),
        parse_dates=["start", "end", "peak_date"],
    )
    china = pd.read_csv(
        args.china_summary.expanduser().resolve(),
        parse_dates=["start", "end"],
    )

    if catalog["wave_id"].nunique() != 42:
        raise ValueError("Expected 42 international waves.")
    if china["wave_id"].nunique() != 111:
        raise ValueError("Expected 111 Chinese waves.")

    rows = [{
        "setting": "China",
        "n_waves": 111,
        "analysis_unit": "Local or provincial epidemic waves",
        "period": f"{fmt_date(china['start'].min())} to {fmt_date(china['end'].max())}",
        "variant_eras": variant_eras(china["variant"]),
        "incidence_series": "Confirmed cases; source wave definitions and later provincial extensions",
    }]

    for country in COUNTRY_ORDER[1:]:
        group = catalog.loc[catalog["country"].eq(country)].sort_values("start")
        if group.empty:
            raise ValueError(f"No international catalogue rows for {country}.")
        rows.append({
            "setting": country,
            "n_waves": int(group["wave_id"].nunique()),
            "analysis_unit": "National epidemic waves",
            "period": f"{fmt_date(group['start'].min())} to {fmt_date(group['end'].max())}",
            "variant_eras": variant_eras(group["variant"]),
            "incidence_series": "National confirmed cases; centred 7-day mean",
        })

    out = pd.DataFrame(rows)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    out.to_csv(
        output_dir / "main_table1_settings_final.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tex = [
        r"\begin{table}[p]",
        r"\centering",
        r"\caption{Study settings and analysed epidemic waves}",
        r"\label{tab:settings}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{l r >{\raggedright\arraybackslash}p{2.8cm} >{\raggedright\arraybackslash}p{3.1cm} X}",
        r"\toprule",
        r"Setting & Waves & Analysis unit & Analysed period & Epidemiological variant eras\\",
        r"\midrule",
    ]

    for _, row in out.iterrows():
        tex.append(
            f"{row['setting']} & {int(row['n_waves'])} & "
            f"{row['analysis_unit']} & {row['period']} & "
            f"{row['variant_eras']}\\\\"
        )

    tex += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\par\medskip",
        r"\footnotesize China comprises 101 previously published local/provincial waves and ten later provincial extensions. "
        r"The 42 international waves comprise six from the United States, eight from the United Kingdom, seven from Japan, "
        r"eight from South Korea, eight from Italy, and five from South Africa. International cumulative confirmed cases were "
        r"converted to non-negative daily increments and analysed as centred 7-day means. Variant labels denote epidemiological "
        r"eras around nominated peaks and are not individual genomic assignments. The different spatial scales are descriptive "
        r"features of the study design and should not be interpreted as direct country-policy comparisons.",
        r"\end{table}",
        "",
    ]

    (output_dir / "main_table1_settings_final.tex").write_text(
        "\n".join(tex),
        encoding="utf-8",
    )

    print(out.to_string(index=False))
    print(output_dir / "main_table1_settings_final.tex")


if __name__ == "__main__":
    main()
