"""Data loaders for the 42 international and 111 Chinese waves."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .project_paths import CHINA_DATA_DIR, INTERNATIONAL_DATA_DIR, require_files


def load_international_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily_path = INTERNATIONAL_DATA_DIR / "international_wave_daily.csv"
    catalog_path = INTERNATIONAL_DATA_DIR / "international_wave_catalog.csv"
    require_files([daily_path, catalog_path])
    daily = pd.read_csv(daily_path, parse_dates=["date"])
    catalog = pd.read_csv(catalog_path)
    if daily["wave_id"].nunique() != 42 or catalog["wave_id"].nunique() != 42:
        raise ValueError("The bundled international panel must contain exactly 42 waves")
    return daily, catalog


def load_china_inputs() -> Tuple[List[dict], Dict[str, np.ndarray], Dict[str, List[str]]]:
    paths = [
        CHINA_DATA_DIR / "fit_summary_101.csv",
        CHINA_DATA_DIR / "wave_daily_101.csv",
        CHINA_DATA_DIR / "fit_curves_101.csv",
        CHINA_DATA_DIR / "external_10_wave_catalog.csv",
        CHINA_DATA_DIR / "external_10_wave_daily.csv",
    ]
    require_files(paths)
    old_meta = pd.read_csv(paths[0])
    old_daily = pd.read_csv(paths[1])
    old_curves = pd.read_csv(paths[2])
    ext_meta = pd.read_csv(paths[3])
    ext_daily = pd.read_csv(paths[4])

    waves: List[dict] = []
    y_by_id: Dict[str, np.ndarray] = {}
    dates_by_id: Dict[str, List[str]] = {}

    old_curve_groups = {key: group.sort_values("day") for key, group in old_curves.groupby("wave_id")}
    old_daily_groups = {key: group.sort_values("day") for key, group in old_daily.groupby("wave_id")}
    old_meta_by_id = old_meta.set_index("wave_id")
    for wave_id, group in old_daily_groups.items():
        meta = old_meta_by_id.loc[wave_id]
        curve = old_curve_groups[wave_id]
        y = group["reported_cases"].to_numpy(dtype=float)
        waves.append({
            "cohort": "Tang101",
            "wave_id": str(wave_id),
            "province": str(meta.province),
            "variant": str(meta.variant),
            "start": str(meta.start),
            "end": str(meta.end),
            "duration_days": int(meta.duration_days),
            "peak_cases": float(meta.peak_cases),
            "total_cases": float(meta.total_cases),
            "old_meta": meta,
            "old_classic_pred": curve["classic_pred"].to_numpy(dtype=float),
            "old_classic_sse": float(meta.classic_sse_log1p),
        })
        y_by_id[str(wave_id)] = y
        dates_by_id[str(wave_id)] = group["date"].astype(str).tolist()

    ext_groups = {key: group.sort_values("day") for key, group in ext_daily.groupby("wave_id")}
    for _, meta in ext_meta.iterrows():
        wave_id = str(meta.wave_id)
        group = ext_groups[wave_id]
        y = group["reported_confirmed"].to_numpy(dtype=float)
        waves.append({
            "cohort": "External10",
            "wave_id": wave_id,
            "province": str(meta.province),
            "province_zh": str(meta.get("province_zh", "")),
            "variant": str(meta.get("variant", "Omicron")),
            "start": str(meta.start),
            "end": str(meta.end),
            "duration_days": int(meta.duration_days),
            "peak_cases": float(meta.peak_cases),
            "total_cases": float(meta.total_cases),
            "old_meta": None,
        })
        y_by_id[wave_id] = y
        dates_by_id[wave_id] = group["date"].astype(str).tolist()

    waves.sort(key=lambda item: (item["cohort"] != "Tang101", item["wave_id"]))
    if len(waves) != 111:
        raise ValueError(f"The bundled China panel must contain 111 waves; found {len(waves)}")
    return waves, y_by_id, dates_by_id
