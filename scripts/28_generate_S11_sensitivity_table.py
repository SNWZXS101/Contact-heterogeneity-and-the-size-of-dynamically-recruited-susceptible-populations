#!/usr/bin/env python3
"""Generate final S11 sensitivity summary table and result sentence."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
TABLES = ROOT / "outputs" / "tables"


def fmt(x, digits=3):
    return f"{float(x):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=RESULTS / "activation_probability_sensitivity_multicountry_final.csv"
    )
    parser.add_argument("--tables-dir", type=Path, default=TABLES)
    args = parser.parse_args()

    d = pd.read_csv(args.input.expanduser().resolve())
    tables = args.tables_dir.expanduser().resolve()
    tables.mkdir(parents=True, exist_ok=True)

    rows = []
    for wid, g in d.groupby("wave_id", sort=False):
        g = g.sort_values("p_activation_infection")
        baseline = g.loc[g["p_activation_infection"].eq(.10)].iloc[0]
        rows.append({
            "wave_id": wid,
            "country": g["country"].iloc[0],
            "variant": g["variant"].iloc[0],
            "baseline_h": baseline["network_h_grid"],
            "h_min": g["network_h_grid"].min(),
            "h_max": g["network_h_grid"].max(),
            "baseline_peak_Shead_Q": baseline["peak_S_head_fraction_Q"],
            "peak_Shead_Q_min": g["peak_S_head_fraction_Q"].min(),
            "peak_Shead_Q_max": g["peak_S_head_fraction_Q"].max(),
            "baseline_accessed_Q": baseline["accessed_reservoir_fraction_Q"],
            "accessed_Q_min": g["accessed_reservoir_fraction_Q"].min(),
            "accessed_Q_max": g["accessed_reservoir_fraction_Q"].max(),
            "same_h_as_baseline_n": int((g["network_h_grid"] == baseline["network_h_grid"]).sum()),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(
        tables / "appendix_table_activation_sensitivity_final.csv",
        index=False, encoding="utf-8-sig"
    )

    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        r"\caption{Sensitivity of representative-wave latent structure to the assumed probability of infection on first reservoir contact}",
        r"\label{tab:app-activation-sensitivity}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Wave & Setting & baseline $h$ & $h$ range & baseline peak $\Shead/Q$ & peak $\Shead/Q$ range & baseline accessed & accessed range\\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['wave_id']} & {row['country']} & "
            f"{fmt(row['baseline_h'],2)} & "
            f"{fmt(row['h_min'],2)}--{fmt(row['h_max'],2)} & "
            f"{fmt(row['baseline_peak_Shead_Q'])} & "
            f"{fmt(row['peak_Shead_Q_min'])}--{fmt(row['peak_Shead_Q_max'])} & "
            f"{fmt(row['baseline_accessed_Q'])} & "
            f"{fmt(row['accessed_Q_min'])}--{fmt(row['accessed_Q_max'])}\\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\medskip",
        r"\footnotesize The baseline column is the exact frozen \(p=0.10\) Phase-2 fit. "
        r"Ranges span \(p\in\{0,0.05,0.10,0.20,0.30\}\). "
        r"Headcount and reservoir-access quantities are relative to the fitted observation/accessibility scale \(Q\), not census population.",
        r"\end{table}",
        "",
    ]
    (tables / "appendix_table_activation_sensitivity_final.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    changed_h = int((summary["same_h_as_baseline_n"] < 5).sum())
    peak_fold = float(
        (summary["peak_Shead_Q_max"] / summary["peak_Shead_Q_min"].clip(lower=1e-12)).max()
    )
    accessed_fold = float(
        (summary["accessed_Q_max"] / summary["accessed_Q_min"].clip(lower=1e-12)).max()
    )
    sentence = (
        f"Across the seven representative waves, selected \\(h\\) changed somewhere "
        f"on the tested \\(p\\)-grid in {changed_h} waves. "
        f"The largest within-wave ratio of maximum to minimum peak \\(\\Shead/Q\\) "
        f"across the grid was {peak_fold:.2f}, and the corresponding largest ratio "
        f"for the reservoir accessed fraction was {accessed_fold:.2f}. "
        f"These ranges quantify structural sensitivity rather than uncertainty intervals."
    )
    (tables / "activation_sensitivity_result_sentence_final.tex").write_text(
        sentence + "\n", encoding="utf-8"
    )

    print(summary.to_string(index=False))
    print(sentence)


if __name__ == "__main__":
    main()
