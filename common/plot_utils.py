"""Shared plotting configuration."""
from __future__ import annotations

import matplotlib.pyplot as plt


def configure_matplotlib(font_size: float = 8.0) -> None:
    plt.rcParams.update({
        "font.size": font_size,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
    })
