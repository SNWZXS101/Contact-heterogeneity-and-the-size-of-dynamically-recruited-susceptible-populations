#!/usr/bin/env python3
"""Apply the final adaptive optimizer-continuation patch to common/model_core.py.

Run once from the project root:
    python scripts/10_apply_optimizer_retry_patch.py

The patch:
- leaves every already-successful multistart optimum unchanged;
- only continues optimisation when the selected best start returned success=False;
- retries from that best point with larger evaluation budgets;
- records total function evaluations in FitResult.nfev;
- creates common/model_core.pre_optimizer_retry_backup.py before editing;
- syntax-compiles the patched file before replacing the original.
"""
from __future__ import annotations

import py_compile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "common" / "model_core.py"
BACKUP = ROOT / "common" / "model_core.pre_optimizer_retry_backup.py"

if not TARGET.exists():
    raise FileNotFoundError(f"Target not found: {TARGET}")

text = TARGET.read_text(encoding="utf-8")

if "_continue_least_squares_if_needed" in text:
    print("Optimizer retry patch already present; no changes made.")
    raise SystemExit(0)

marker = """    return out[:max(starts, len(warm) if warm else 0, 1)]


def fit_classic(
"""

helper = """    return out[:max(starts, len(warm) if warm else 0, 1)]


def _continue_least_squares_if_needed(
    residual,
    opt,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_nfev: int,
    xtol: float,
    ftol: float,
    gtol: float,
):
    \"""Continue only a selected best fit that did not converge.

    The initial multistart search is left unchanged. If its best objective
    value comes from an OptimizeResult with success=False, optimisation is
    continued from that point with two progressively larger evaluation
    budgets. Already-successful fits are returned unchanged in parameter space.

    Returns
    -------
    opt_final:
        Best retained OptimizeResult after any continuation.
    total_nfev:
        Function evaluations for the selected initial run plus continuation
        runs. This is stored in FitResult.nfev for QC.
    \"""
    total_nfev = int(opt.nfev)
    if bool(opt.success):
        return opt, total_nfev

    best = opt
    best_sse = float(np.dot(opt.fun, opt.fun))
    current_x = np.asarray(opt.x, dtype=float).copy()

    # Fresh budgets for each continuation run. This has zero extra cost for
    # fits that already converged.
    for multiplier in (4, 12):
        retry_budget = max(int(max_nfev * multiplier), int(max_nfev + 200))
        x0 = np.clip(current_x, lower + 1e-10, upper - 1e-10)
        retry = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            max_nfev=retry_budget,
            xtol=xtol,
            ftol=ftol,
            gtol=gtol,
            x_scale="jac",
        )
        total_nfev += int(retry.nfev)
        retry_sse = float(np.dot(retry.fun, retry.fun))

        if retry_sse <= best_sse:
            best = retry
            best_sse = retry_sse
            current_x = np.asarray(retry.x, dtype=float).copy()
        else:
            current_x = np.asarray(best.x, dtype=float).copy()

        if bool(retry.success) and best is retry:
            return retry, total_nfev

    return best, total_nfev


def fit_classic(
"""

if marker not in text:
    raise RuntimeError("Could not locate _start_points -> fit_classic insertion point.")
text = text.replace(marker, helper, 1)

old_classic = """    assert best is not None
    sse, opt = best
    pred = simulate_classic(opt.x, y, steps)
    aic, aicc = information_criteria(sse, y.size, 3)
    return FitResult("classic", opt.x.copy(), pred, sse, aic, aicc, 3, bool(opt.success), int(opt.nfev))
"""
new_classic = """    assert best is not None
    sse, opt = best
    opt, total_nfev = _continue_least_squares_if_needed(
        residual, opt, lower, upper,
        max_nfev=max_nfev, xtol=1e-7, ftol=1e-7, gtol=1e-7,
    )
    sse = float(np.dot(opt.fun, opt.fun))
    pred = simulate_classic(opt.x, y, steps)
    aic, aicc = information_criteria(sse, y.size, 3)
    return FitResult(
        "classic", opt.x.copy(), pred, sse, aic, aicc, 3,
        bool(opt.success), int(total_nfev),
    )
"""
if old_classic not in text:
    raise RuntimeError("Could not locate fit_classic finalisation block.")
text = text.replace(old_classic, new_classic, 1)

old_reservoir = """    assert best is not None
    sse, opt = best
    pred = simulate_reservoir(opt.x, y, p_act, steps)
    aic, aicc = information_criteria(sse, y.size, 5)
    return FitResult("reservoir", opt.x.copy(), pred, sse, aic, aicc, 5, bool(opt.success), int(opt.nfev))
"""
new_reservoir = """    assert best is not None
    sse, opt = best
    opt, total_nfev = _continue_least_squares_if_needed(
        residual, opt, lower, upper,
        max_nfev=max_nfev, xtol=1e-7, ftol=1e-7, gtol=1e-7,
    )
    sse = float(np.dot(opt.fun, opt.fun))
    pred = simulate_reservoir(opt.x, y, p_act, steps)
    aic, aicc = information_criteria(sse, y.size, 5)
    return FitResult(
        "reservoir", opt.x.copy(), pred, sse, aic, aicc, 5,
        bool(opt.success), int(total_nfev),
    )
"""
if old_reservoir not in text:
    raise RuntimeError("Could not locate fit_reservoir finalisation block.")
text = text.replace(old_reservoir, new_reservoir, 1)

old_network = """        assert best is not None
        sse, opt = best
        pred = simulate_network(opt.x, y, z, w, p_act, steps)
        # Count h as a sixth estimated structural parameter, including the
        # h=0 boundary profile, for conservative comparison to M1.
        aic, aicc = information_criteria(sse, y.size, 6)
        result = FitResult("network", opt.x.copy(), pred, sse, aic, aicc, 6,
                           bool(opt.success), int(opt.nfev), float(h), h_real, second)
"""
new_network = """        assert best is not None
        sse, opt = best
        opt, total_nfev = _continue_least_squares_if_needed(
            residual, opt, lower, upper,
            max_nfev=max_nfev, xtol=2e-7, ftol=2e-7, gtol=2e-7,
        )
        sse = float(np.dot(opt.fun, opt.fun))
        pred = simulate_network(opt.x, y, z, w, p_act, steps)
        # Count h as a sixth estimated structural parameter, including the
        # h=0 boundary profile, for conservative comparison to M1.
        aic, aicc = information_criteria(sse, y.size, 6)
        result = FitResult(
            "network", opt.x.copy(), pred, sse, aic, aicc, 6,
            bool(opt.success), int(total_nfev), float(h), h_real, second,
        )
"""
if old_network not in text:
    raise RuntimeError("Could not locate fit_network_profile finalisation block.")
text = text.replace(old_network, new_network, 1)

required_tokens = [
    "_continue_least_squares_if_needed",
    "for multiplier in (4, 12):",
    "int(total_nfev)",
    "steps: int = 8",
]
for token in required_tokens:
    if token not in text:
        raise RuntimeError(f"Patched source missing expected token: {token}")

if not BACKUP.exists():
    shutil.copy2(TARGET, BACKUP)

tmp = TARGET.with_suffix(".optimizer_retry_tmp.py")
tmp.write_text(text, encoding="utf-8")

py_compile.compile(str(tmp), doraise=True)
tmp.replace(TARGET)

print(f"Patched: {TARGET}")
print(f"Backup : {BACKUP}")
print("Syntax : PASS")
print("Policy : retry selected best fit only when success=False; budgets 4x then 12x")
