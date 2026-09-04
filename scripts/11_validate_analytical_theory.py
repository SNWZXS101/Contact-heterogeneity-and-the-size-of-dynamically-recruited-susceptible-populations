#!/usr/bin/env python3
"""Numerical validation of the analytical theory in Supplementary Section S5.

This script is data-independent. It validates:
1) the 12-class activity calibration and second-moment identity;
2) the rank-one next-generation spectral-radius formula;
3) exact h=0 nesting of the homogeneous and activity-stratified dynamics;
4) the Gamma fixed-edge-capacity formula against the fitted 12-class profile;
5) the local final-size sign-reversal derivative.

Outputs are written to outputs/theory_validation/.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtri

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "theory_validation"
OUT.mkdir(parents=True, exist_ok=True)

H_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
M = 12


def make_activity_classes(h: float, m: int = M):
    w = np.full(m, 1.0 / m, dtype=float)
    if h <= 1e-14:
        return np.ones(m), w, 0.0
    scores = ndtri((np.arange(m, dtype=float) + 0.5) / m)

    def classes(log_sd: float):
        raw = np.exp(log_sd * scores - np.max(log_sd * scores))
        z = raw / float(np.dot(w, raw))
        cv2 = float(np.dot(w, (z - 1.0) ** 2))
        return z, cv2

    lo, hi = 0.0, 1.0
    _, cv_hi = classes(hi)
    while cv_hi < h:
        hi *= 2.0
        if hi > 64.0:
            raise RuntimeError(f"Could not construct activity classes for h={h}")
        _, cv_hi = classes(hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        _, cv_mid = classes(mid)
        if cv_mid < h:
            lo = mid
        else:
            hi = mid
    z, h_real = classes(0.5 * (lo + hi))
    return z, w, h_real


def validate_activity_and_rank_one():
    rows = []
    base_factor = 0.73
    for h in H_GRID:
        z, w, h_real = make_activity_classes(h)
        mean_z = float(np.dot(w, z))
        second = float(np.dot(w, z * z))
        K = base_factor * np.outer(w * z, z)
        # Power iteration avoids dependence on a dense eigensolver while still
        # checking the rank-one numerical map directly.
        v = np.ones(len(z), dtype=float)
        v /= math.sqrt(float(np.dot(v, v)))
        rho = 0.0
        for _ in range(50):
            Kv = K @ v
            norm = math.sqrt(float(np.dot(Kv, Kv)))
            if norm == 0.0:
                rho = 0.0
                break
            v = Kv / norm
            rho = float(np.dot(v, K @ v) / np.dot(v, v))
        rho = abs(rho)
        expected = base_factor * (1.0 + h)
        rows.append({
            "h_target": h,
            "h_realized": h_real,
            "mean_z": mean_z,
            "second_moment": second,
            "mean_error": abs(mean_z - 1.0),
            "h_error": abs((second - 1.0) - h),
            "spectral_radius_numeric": rho,
            "spectral_radius_expected": expected,
            "spectral_radius_abs_error": abs(rho - expected),
        })
    return pd.DataFrame(rows)


def m1_rhs(state, t, *, a, beta0, q, p, sigma, gamma):
    U, S, E, I, R = state
    U = max(U, 0.0); S = max(S, 0.0)
    E = max(E, 0.0); I = max(I, 0.0)
    beta = beta0 * math.exp(-q * t)
    act = a * U * I
    inf = beta * S * I
    return np.array([
        -act,
        (1.0 - p) * act - inf,
        p * act + inf - sigma * E,
        sigma * E - gamma * I,
        gamma * I,
    ], dtype=float)


def m2_rhs(state, t, z, w, *, a, beta0, q, p, sigma, gamma):
    U, S, E, I, R = state
    Up = np.maximum(U, 0.0); Sp = np.maximum(S, 0.0)
    Ep = np.maximum(E, 0.0); Ip = np.maximum(I, 0.0)
    theta = max(0.0, float(np.dot(z, Ip)))
    beta = beta0 * math.exp(-q * t)
    act = a * z * Up * theta
    inf = beta * z * Sp * theta
    return np.array([
        -act,
        (1.0 - p) * act - inf,
        p * act + inf - sigma * Ep,
        sigma * Ep - gamma * Ip,
        gamma * Ip,
    ])


def validate_h0_nesting(days=120, steps=8):
    params = dict(a=0.8, beta0=0.9, q=0.025, p=0.10,
                  sigma=1/3, gamma=1/5)
    init = np.array([0.82, 0.10, 0.04, 0.04, 0.0], dtype=float)
    z, w, _ = make_activity_classes(0.0)
    state1 = init.copy()
    state2 = np.vstack([x * w for x in init])
    dt = 1.0 / steps
    t = 0.0
    max_state_error = 0.0

    for _day in range(days):
        for _ in range(steps):
            k1 = m1_rhs(state1, t, **params)
            k2 = m1_rhs(state1 + .5*dt*k1, t + .5*dt, **params)
            k3 = m1_rhs(state1 + .5*dt*k2, t + .5*dt, **params)
            k4 = m1_rhs(state1 + dt*k3, t + dt, **params)
            state1 += dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
            state1 = np.maximum(state1, 0.0)
            state1 /= state1.sum()

            q1 = m2_rhs(state2, t, z, w, **params)
            q2 = m2_rhs(state2 + .5*dt*q1, t + .5*dt, z, w, **params)
            q3 = m2_rhs(state2 + .5*dt*q2, t + .5*dt, z, w, **params)
            q4 = m2_rhs(state2 + dt*q3, t + dt, z, w, **params)
            state2 += dt * (q1 + 2*q2 + 2*q3 + q4) / 6.0
            state2 = np.maximum(state2, 0.0)
            mass = state2.sum(axis=0)
            factor = np.divide(w, mass, out=np.ones_like(w), where=mass > 0)
            state2 *= factor
            t += dt

        agg = state2.sum(axis=1)
        max_state_error = max(max_state_error, float(np.max(np.abs(state1 - agg))))

    return {
        "days": days,
        "rk4_substeps_per_day": steps,
        "max_abs_aggregate_state_error": max_state_error,
    }


def discrete_head_for_edge(h: float, b: float):
    z, w, _ = make_activity_classes(h)

    def edge_fraction(x):
        return 1.0 - float(np.dot(w, z * np.exp(-z * x)))

    hi = 1.0
    while edge_fraction(hi) < b:
        hi *= 2.0
    x = brentq(lambda xx: edge_fraction(xx) - b, 0.0, hi)
    head = 1.0 - float(np.dot(w, np.exp(-z * x)))
    return head, x


def validate_edge_capacity():
    rows = []
    for b in (0.01, 0.10, 0.25):
        for h in H_GRID:
            discrete_head, pressure = discrete_head_for_edge(h, b)
            gamma_head = (
                b if h == 0
                else 1.0 - (1.0 - b) ** (1.0 / (1.0 + h))
            )
            rows.append({
                "edge_target_b": b,
                "h": h,
                "discrete_head_fraction": discrete_head,
                "gamma_head_fraction": gamma_head,
                "discrete_minus_gamma": discrete_head - gamma_head,
                "relative_difference": (
                    (discrete_head - gamma_head) / gamma_head
                    if gamma_head > 0 else 0.0
                ),
                "activation_pressure_x": pressure,
            })
    return pd.DataFrame(rows)


def F(h, lam):
    if h == 0:
        return 1.0 - math.exp(-lam)
    return 1.0 - (1.0 + h * lam) ** (-1.0 / h)


def G(h, lam):
    if h == 0:
        return 1.0 - math.exp(-lam)
    return 1.0 - (1.0 + h * lam) ** (-(1.0 + h) / h)


def H_for_C(h, Rb, C):
    if h == 0:
        lam = Rb * C
        return C / F(0.0, lam), lam

    def psi(lam):
        return lam - Rb * C * G(h, lam) / F(h, lam)

    left = 1e-8
    right = max(10.0, 10.0 * Rb * C + 5.0)
    fleft = psi(left)
    fright = psi(right)
    if fleft == 0.0:
        left = 1e-6
        fleft = psi(left)
    if fleft * fright > 0:
        raise RuntimeError("Could not bracket final-size root")
    root = brentq(psi, left, right)
    return C / F(h, root), root


def analytic_H_derivative_at_zero(Rb, C):
    lam0 = Rb * C
    F0 = 1.0 - math.exp(-lam0)
    return (
        C * math.exp(-lam0) * lam0**2 / F0**2
        * (0.5 - math.exp(-lam0) / F0)
    )


def validate_final_size_sign_reversal():
    Rb = 3.0
    eps = 1e-4
    rows = []
    for C in (0.10, 0.50):
        H0, _ = H_for_C(0.0, Rb, C)
        Heps, _ = H_for_C(eps, Rb, C)
        H1, lam1 = H_for_C(1.0, Rb, C)
        derivative_exact = analytic_H_derivative_at_zero(Rb, C)
        derivative_fd = (Heps - H0) / eps
        rows.append({
            "Rb": Rb,
            "C": C,
            "Rb_times_C": Rb*C,
            "log3": math.log(3.0),
            "H_h0": H0,
            "H_h1": H1,
            "relative_change_h1_vs_h0": H1/H0 - 1.0,
            "lambda_h1": lam1,
            "analytic_derivative_at_h0": derivative_exact,
            "finite_difference_derivative_eps_1e-4": derivative_fd,
            "derivative_abs_error": abs(derivative_exact - derivative_fd),
        })
    return pd.DataFrame(rows)


def main():
    print("[1/4] validating activity moments and rank-one threshold ...", flush=True)
    activity = validate_activity_and_rank_one()

    print("[2/4] validating exact h=0 numerical nesting ...", flush=True)
    nesting = validate_h0_nesting()

    print("[3/4] validating finite-class edge-capacity equivalence ...", flush=True)
    edge = validate_edge_capacity()

    print("[4/4] validating final-size sign reversal ...", flush=True)
    final = validate_final_size_sign_reversal()

    activity.to_csv(OUT / "activity_rankone_validation.csv", index=False)
    edge.to_csv(OUT / "edge_capacity_validation.csv", index=False)
    final.to_csv(OUT / "final_size_sign_reversal_validation.csv", index=False)

    summary = {
        "activity_grid": list(H_GRID),
        "classes": M,
        "max_mean_activity_error": float(activity["mean_error"].max()),
        "max_h_calibration_error": float(activity["h_error"].max()),
        "max_rankone_spectral_radius_abs_error": float(
            activity["spectral_radius_abs_error"].max()
        ),
        "h0_nesting": nesting,
        "edge_capacity": {
            "max_abs_relative_difference_discrete_vs_gamma_b_0p01": float(
                edge.loc[np.isclose(edge["edge_target_b"], .01), "relative_difference"].abs().max()
            ),
            "max_abs_relative_difference_discrete_vs_gamma_b_0p10": float(
                edge.loc[np.isclose(edge["edge_target_b"], .10), "relative_difference"].abs().max()
            ),
            "max_abs_relative_difference_discrete_vs_gamma_b_0p25": float(
                edge.loc[np.isclose(edge["edge_target_b"], .25), "relative_difference"].abs().max()
            ),
            "h4_discrete_head_ratio_at_b_0p01": float(
                edge.loc[
                    np.isclose(edge["edge_target_b"], .01) & np.isclose(edge["h"], 4.0),
                    "discrete_head_fraction"
                ].iloc[0] / .01
            ),
            "h4_gamma_head_ratio_at_b_0p01": float(
                edge.loc[
                    np.isclose(edge["edge_target_b"], .01) & np.isclose(edge["h"], 4.0),
                    "gamma_head_fraction"
                ].iloc[0] / .01
            ),
        },
        "threshold_h4_exact_head_ratio": 1.0 / 5.0,
        "threshold_h4_exact_reduction": 0.8,
        "final_size_sign_reversal": final.to_dict("records"),
    }

    (OUT / "theory_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
