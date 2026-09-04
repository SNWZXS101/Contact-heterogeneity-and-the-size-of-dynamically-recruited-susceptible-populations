"""Shared input/output, validation, and command-line helpers."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


def atomic_to_csv(df: pd.DataFrame, path: Path, *, encoding: str = "utf-8-sig") -> None:
    """Write a CSV atomically to avoid corrupting checkpoints on interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding=encoding, newline="", suffix=".tmp", delete=False,
        dir=str(path.parent)
    ) as handle:
        tmp = Path(handle.name)
        df.to_csv(handle, index=False)
    os.replace(str(tmp), str(path))


def atomic_write_json(obj: object, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".tmp", delete=False,
        dir=str(path.parent)
    ) as handle:
        tmp = Path(handle.name)
        handle.write(text)
    os.replace(str(tmp), str(path))


def parse_float_grid(text: str) -> Tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise ValueError("The grid must contain at least one number")
    if any(x < 0 for x in values):
        raise ValueError("Heterogeneity grid values must be non-negative")
    return values


def validate_unique_count(df: pd.DataFrame, column: str, expected: int, label: str) -> None:
    actual = int(df[column].nunique())
    if actual != expected:
        raise ValueError(f"{label}: expected {expected} unique {column} values, found {actual}")


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def model_winner_from_values(values: Sequence[float], names: Sequence[str]) -> str:
    a = np.asarray(values, dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        return "unresolved"
    idx = np.where(finite)[0][int(np.argmin(a[finite]))]
    return str(names[idx])


def latex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
