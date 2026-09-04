#!/usr/bin/env python3
"""Generate manuscript Figure 2 independently from bundled country data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.plot_utils import configure_matplotlib  # noqa: E402
from common.project_paths import (  # noqa: E402
    INTERNATIONAL_DATA_DIR, SOURCE_DATA_DIR, output_path, resolve_result_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-source", choices=("reference", "outputs", "auto"), default="reference")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = args.output.expanduser().resolve() if args.output else output_path("figures", "Figure2_multicountry_trajectories.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    configure_matplotlib(8)

    owid = pd.read_csv(SOURCE_DATA_DIR / "owid_selected_countries.csv", parse_dates=["date"])
    catalog = pd.read_csv(INTERNATIONAL_DATA_DIR / "international_wave_catalog.csv",
                          parse_dates=["start", "end", "peak_date"])
    china = pd.read_csv(resolve_result_file("china_fit_summary_111_three_models.csv", args.results_source),
                        parse_dates=["start", "end"])
    order = ["China", "United States", "United Kingdom", "Japan", "South Korea", "Italy", "South Africa"]
    fig, axes = plt.subplots(7, 1, figsize=(13, 12), sharex=True)
    for index, (axis, country) in enumerate(zip(axes, order)):
        data = owid[(owid.country == country) & (owid.date >= "2020-01-01") & (owid.date <= "2022-12-31")].copy()
        cases = data.new_cases_smoothed_per_million.fillna(0).clip(lower=0)
        axis.fill_between(data.date, 0, cases, color="#4C78A8", alpha=.35, linewidth=0)
        axis.plot(data.date, cases, color="#2F5D8A", linewidth=.8)
        axis.set_yscale("symlog", linthresh=.1); axis.set_ylabel("cases\nper million")
        right = axis.twinx()
        right.plot(data.date, data.stringency_index, color="#D55E00", linewidth=.7, alpha=.8)
        right.plot(data.date, data.people_fully_vaccinated_per_hundred, color="#009E73", linewidth=.8, alpha=.9)
        right.set_ylim(0, 105); right.set_yticks([0, 50, 100]); right.tick_params(axis="y", labelsize=7)
        if country != "China":
            for _, row in catalog[catalog.country == country].iterrows():
                axis.axvspan(row.start, row.end, color="#888888", alpha=.045)
                axis.axvline(row.peak_date, color="#555555", linewidth=.35, alpha=.5)
        else:
            for _, row in china.iterrows():
                axis.axvspan(row.start, row.end, color="#888888", alpha=.018)
        axis.text(.004, .86, chr(65+index) + f"  {country}", transform=axis.transAxes, weight="bold", fontsize=9)
        if index == 0:
            right.plot([], [], color="#D55E00", label="stringency index")
            right.plot([], [], color="#009E73", label="fully vaccinated (%)")
            right.legend(loc="upper right", ncol=2, frameon=False, fontsize=7)
        axis.grid(axis="y", alpha=.15)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.suptitle(
        "National epidemic trajectories, enacted policy intensity, vaccination, and analysed wave windows",
        fontsize=14, weight="bold", y=.995,
    )
    fig.text(.995, .5, "policy or vaccination (%)", rotation=90, va="center", ha="right", fontsize=8)
    fig.tight_layout(rect=[0, .01, .985, .98])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
