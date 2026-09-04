#!/usr/bin/env python3
"""Plot final S8 heterogeneity profiles without refitting models."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
DEFAULT_OUT = ROOT / "outputs" / "figures" / "FigureS3_network_profiles.jpg"


def load_profiles(results: Path) -> pd.DataFrame:
    path = results / "appendix_heterogeneity_profiles.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/07_profile_heterogeneity_appendix.py "
            "from the final frozen outputs first."
        )
    df = pd.read_csv(path)
    required = {"wave_id", "h_grid", "delta_information_criterion"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Profile CSV missing columns: {sorted(missing)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    results = args.results_dir.expanduser().resolve()
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    p = load_profiles(results)
    selected = list(dict.fromkeys(p["wave_id"].astype(str).tolist()))
    if len(selected) != 12:
        raise ValueError(f"Expected 12 selected profile waves, found {len(selected)}")

    fig, axs = plt.subplots(3, 4, figsize=(12.5, 8.8))
    for ax, wid in zip(axs.ravel(), selected):
        g = p.loc[p["wave_id"].astype(str).eq(wid)].sort_values("h_grid")
        meta = g.iloc[0]
        y = pd.to_numeric(g["delta_information_criterion"], errors="coerce")
        best_idx = int(np.nanargmin(y.to_numpy(dtype=float)))

        ax.plot(g["h_grid"], y, "o-", color="#D55E00", lw=1.25, ms=4.6)
        ax.scatter(
            [g.iloc[best_idx]["h_grid"]], [y.iloc[best_idx]],
            s=58, marker="*", color="#D55E00", zorder=5,
        )
        ax.axhline(2, ls="--", lw=.75, alpha=.7)
        ax.set_xticks(sorted(pd.to_numeric(g["h_grid"], errors="coerce").unique()))
        ax.set_ylim(bottom=-.25)
        location = meta.get("location", meta.get("country", ""))
        ax.set_title(f"{wid} · {location}\n{meta['variant']}", fontsize=8.2, loc="left")
        ax.set_xlabel(r"$h=\mathrm{CV}^2(Z)$")
        criterion = str(meta.get("profile_criterion", "AICc"))
        ax.set_ylabel(rf"$\Delta${criterion} from best")
        ax.grid(alpha=.15)

    fig.suptitle(
        "Conditional heterogeneity profiles in 12 representative international waves",
        fontsize=13.5, weight="bold", y=.995,
    )
    fig.tight_layout(rect=[0, 0, 1, .965])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
