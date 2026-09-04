"""Reusable fitting, model-comparison, and metric helpers."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .model_core import (
    FitResult,
    akaike_weights,
    information_criteria,
    parameter_summary,
)

MODEL_NAMES = ("classic", "reservoir", "network")


def fitresult_from_profile_h0(profile: Sequence[FitResult], y: np.ndarray) -> FitResult:
    """Convert the exact h=0 network profile into a five-parameter AR-SEIR fit."""
    h0 = min(profile, key=lambda result: abs(result.h_target))
    aic, aicc = information_criteria(h0.sse, len(y), 5)
    return FitResult(
        model="reservoir",
        x=h0.x.copy(),
        pred=h0.pred.copy(),
        sse=h0.sse,
        aic=aic,
        aicc=aicc,
        k=5,
        success=h0.success,
        nfev=h0.nfev,
    )


def choose_winner(fits: Sequence[FitResult]) -> Tuple[str, str, np.ndarray]:
    """Choose a model by AICc, falling back to AIC for very short waves."""
    aicc = np.asarray([fit.aicc for fit in fits], dtype=float)
    if np.all(np.isfinite(aicc)):
        values = aicc
        criterion = "AICc"
    else:
        values = np.asarray([fit.aic for fit in fits], dtype=float)
        criterion = "AIC"
    weights = akaike_weights(values)
    return fits[int(np.argmin(values))].model, criterion, weights


def prediction_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "log_rmse": float(np.sqrt(np.mean((np.log1p(predicted) - np.log1p(observed)) ** 2))),
        "mae": float(np.mean(np.abs(predicted - observed))),
        "smape": float(np.mean(2.0 * np.abs(predicted - observed) /
                                  (np.abs(observed) + np.abs(predicted) + 1.0))),
        "peak_day_error": int(np.argmax(predicted) - np.argmax(observed)),
    }


def add_fit_fields(prefix: str, fit: FitResult, y: np.ndarray, row: Dict[str, object], p_act: float) -> None:
    params = parameter_summary(fit, y, p_act)
    row.update({
        f"{prefix}_sse_log1p": fit.sse,
        f"{prefix}_aic": fit.aic,
        f"{prefix}_aicc": fit.aicc,
        f"{prefix}_Q": params["Q"],
        f"{prefix}_beta0": params["beta0"],
        f"{prefix}_q": params["q"],
        f"{prefix}_a": params["a"],
        f"{prefix}_s0": params["s0"],
        f"{prefix}_u0": params["u0"],
        f"{prefix}_R0": params["R0"],
        f"{prefix}_success": bool(fit.success),
        f"{prefix}_nfev": int(fit.nfev),
    })


def stable_seed(text: str, offset: int = 0) -> int:
    """Deterministic seed that does not depend on Python hash randomisation."""
    return int(sum((i + 1) * ord(char) for i, char in enumerate(text)) + offset)
