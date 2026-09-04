#!/usr/bin/env python3
"""Numerical models and fitting utilities for the complete 153-wave analysis.

Models
------
M0: classic SEIR with exponentially decaying transmission.
M1: reservoir-activated SEIR (U,S,E,I,R).
M2: degree/activity-structured reservoir SEIR with proportionate mixing.

The network model is a finite-dimensional heterogeneous mean-field ODE. It is
nested: at heterogeneity h=0 all activity classes have z=1 and aggregate
exactly to M1. At h>0, class j experiences force proportional to z_j and
Theta_I=sum_j z_j I_j because E[z]=1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
try:
    from numba import njit
except ImportError:  # pragma: no cover - slower pure-Python fallback
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        def decorator(func):
            return func
        return decorator
from scipy.optimize import least_squares
from scipy.special import ndtri

SIGMA = 1.0 / 3.0
GAMMA = 1.0 / 5.0
P_ACT_DEFAULT = 0.10


@dataclass
class FitResult:
    model: str
    x: np.ndarray
    pred: np.ndarray
    sse: float
    aic: float
    aicc: float
    k: int
    success: bool
    nfev: int
    h_target: float = 0.0
    h_realized: float = 0.0
    z_second_moment: float = 1.0


@njit(cache=True)
def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def inverse_s0_transform(s0: float) -> float:
    """Inverse of s0=.001+.998*sigmoid(x), with safe clipping."""
    u = (float(s0) - 0.001) / 0.998
    u = min(max(u, 1e-8), 1.0 - 1e-8)
    return math.log(u / (1.0 - u))


def unpack_reservoir_x(x: np.ndarray) -> tuple[float, float, float, float, float]:
    Q, beta0, q, a = map(math.exp, x[:4])
    s0 = 0.001 + 0.998 / (1.0 + math.exp(-float(x[4])))
    return Q, beta0, q, a, s0


def make_activity_classes(h: float, m: int = 12) -> tuple[np.ndarray, np.ndarray, float]:
    """Equal-mass positive activity classes with exactly E[z]=1 and CV(z)^2=h.

    A fixed set of normal quantile scores is exponentiated and normalized. The
    log-scale is solved by bisection so that the finite-class second moment is
    exact, avoiding quadrature drift in the interpretation of ``h``.
    """
    if m < 2:
        raise ValueError("m must be at least 2")
    if h < 0 or h >= m - 1:
        raise ValueError(f"h must satisfy 0 <= h < m-1; got h={h}, m={m}")
    w = np.full(m, 1.0 / m, dtype=np.float64)
    if h <= 1e-14:
        z = np.ones(m, dtype=np.float64)
        return z, w, 0.0

    scores = ndtri((np.arange(m, dtype=np.float64) + 0.5) / m)

    def classes(log_sd: float) -> tuple[np.ndarray, float]:
        raw = np.exp(log_sd * scores - np.max(log_sd * scores))
        zz = raw / float(np.dot(w, raw))
        cv2 = float(np.dot(w, (zz - 1.0) ** 2))
        return zz, cv2

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
    z, h_realized = classes(0.5 * (lo + hi))
    return z.astype(np.float64), w, h_realized


@njit(cache=True)
def _initial_totals(x: np.ndarray, y0: float) -> tuple[float, float, float, float, float, float]:
    Q = math.exp(x[0])
    beta0 = math.exp(x[1])
    q = math.exp(x[2])
    a = math.exp(x[3]) if x.size >= 5 else 0.0
    s0 = 0.001 + 0.998 * _sigmoid(x[4]) if x.size >= 5 else 0.0
    obs0 = max(y0, 0.5)
    e0 = obs0 / (Q * SIGMA)
    i0 = obs0 / (Q * GAMMA)
    if e0 + i0 > 0.5:
        scale = 0.5 / (e0 + i0)
        e0 *= scale
        i0 *= scale
    if x.size >= 5:
        s0 = min(s0, max(1e-6, 1.0 - e0 - i0 - 1e-6))
    return Q, beta0, q, a, s0, e0 + i0


@njit(cache=True)
def simulate_classic(x: np.ndarray, y: np.ndarray, steps: int = 8) -> np.ndarray:
    n = y.size
    Q = math.exp(x[0])
    beta0 = math.exp(x[1])
    q = math.exp(x[2])
    y0 = max(y[0], 0.5)
    e = y0 / (Q * SIGMA)
    i = y0 / (Q * GAMMA)
    if e + i > 0.5:
        scale = 0.5 / (e + i)
        e *= scale
        i *= scale
    state = np.array([max(1e-8, 1.0 - e - i), e, i, 0.0, 0.0])
    pred = np.empty(n, dtype=np.float64)
    dt = 1.0 / steps
    t = 0.0
    for day in range(n):
        c0 = state[4]
        for _ in range(steps):
            beta1 = beta0 * math.exp(-q * t)
            inf1 = beta1 * state[0] * state[2]
            k1 = np.array([-inf1, inf1 - SIGMA * state[1], SIGMA * state[1] - GAMMA * state[2], GAMMA * state[2], SIGMA * state[1]])

            z2 = state + 0.5 * dt * k1
            beta2 = beta0 * math.exp(-q * (t + 0.5 * dt))
            inf2 = beta2 * z2[0] * z2[2]
            k2 = np.array([-inf2, inf2 - SIGMA * z2[1], SIGMA * z2[1] - GAMMA * z2[2], GAMMA * z2[2], SIGMA * z2[1]])

            z3 = state + 0.5 * dt * k2
            inf3 = beta2 * z3[0] * z3[2]
            k3 = np.array([-inf3, inf3 - SIGMA * z3[1], SIGMA * z3[1] - GAMMA * z3[2], GAMMA * z3[2], SIGMA * z3[1]])

            z4 = state + dt * k3
            beta4 = beta0 * math.exp(-q * (t + dt))
            inf4 = beta4 * z4[0] * z4[2]
            k4 = np.array([-inf4, inf4 - SIGMA * z4[1], SIGMA * z4[1] - GAMMA * z4[2], GAMMA * z4[2], SIGMA * z4[1]])

            state += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            for j in range(4):
                if state[j] < 0.0:
                    state[j] = 0.0
            mass = state[0] + state[1] + state[2] + state[3]
            if mass > 0.0 and abs(mass - 1.0) > 1e-10:
                for j in range(4):
                    state[j] /= mass
            t += dt
        pred[day] = max(0.0, Q * (state[4] - c0))
    return pred


@njit(cache=True)
def simulate_reservoir(
    x: np.ndarray,
    y: np.ndarray,
    p_act: float = P_ACT_DEFAULT,
    steps: int = 8,
) -> np.ndarray:
    """Simulate M1 with positivity-safe RK4 stage-flow evaluation."""
    n = y.size
    Q = math.exp(x[0])
    beta0 = math.exp(x[1])
    q = math.exp(x[2])
    a = math.exp(x[3])
    s0 = 0.001 + 0.998 * _sigmoid(x[4])
    y0 = max(y[0], 0.5)
    e = y0 / (Q * SIGMA)
    i = y0 / (Q * GAMMA)
    if e + i > 0.5:
        scale = 0.5 / (e + i)
        e *= scale
        i *= scale
    s = min(s0, max(1e-6, 1.0 - e - i - 1e-6))
    u = max(1e-8, 1.0 - s - e - i)
    state = np.array([u, s, e, i, 0.0, 0.0])
    pred = np.empty(n, dtype=np.float64)
    dt = 1.0 / steps
    t = 0.0

    for day in range(n):
        c0 = state[5]
        for _ in range(steps):
            u1 = max(state[0], 0.0)
            s1 = max(state[1], 0.0)
            e1 = max(state[2], 0.0)
            i1 = max(state[3], 0.0)
            beta1 = beta0 * math.exp(-q * t)
            act1 = a * u1 * i1
            inf1 = beta1 * s1 * i1
            k1 = np.array([-act1, (1.0-p_act)*act1-inf1,
                           p_act*act1+inf1-SIGMA*e1,
                           SIGMA*e1-GAMMA*i1, GAMMA*i1, SIGMA*e1])

            z2 = state + 0.5 * dt * k1
            u2 = max(z2[0], 0.0); s2 = max(z2[1], 0.0)
            e2 = max(z2[2], 0.0); i2 = max(z2[3], 0.0)
            beta2 = beta0 * math.exp(-q * (t + 0.5 * dt))
            act2 = a * u2 * i2
            inf2 = beta2 * s2 * i2
            k2 = np.array([-act2, (1.0-p_act)*act2-inf2,
                           p_act*act2+inf2-SIGMA*e2,
                           SIGMA*e2-GAMMA*i2, GAMMA*i2, SIGMA*e2])

            z3 = state + 0.5 * dt * k2
            u3 = max(z3[0], 0.0); s3 = max(z3[1], 0.0)
            e3 = max(z3[2], 0.0); i3 = max(z3[3], 0.0)
            act3 = a * u3 * i3
            inf3 = beta2 * s3 * i3
            k3 = np.array([-act3, (1.0-p_act)*act3-inf3,
                           p_act*act3+inf3-SIGMA*e3,
                           SIGMA*e3-GAMMA*i3, GAMMA*i3, SIGMA*e3])

            z4 = state + dt * k3
            u4 = max(z4[0], 0.0); s4 = max(z4[1], 0.0)
            e4 = max(z4[2], 0.0); i4 = max(z4[3], 0.0)
            beta4 = beta0 * math.exp(-q * (t + dt))
            act4 = a * u4 * i4
            inf4 = beta4 * s4 * i4
            k4 = np.array([-act4, (1.0-p_act)*act4-inf4,
                           p_act*act4+inf4-SIGMA*e4,
                           SIGMA*e4-GAMMA*i4, GAMMA*i4, SIGMA*e4])

            state += dt * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
            for j in range(5):
                if state[j] < 0.0:
                    state[j] = 0.0
            mass = state[0] + state[1] + state[2] + state[3] + state[4]
            if mass > 0.0 and abs(mass - 1.0) > 1e-10:
                for j in range(5):
                    state[j] /= mass
            t += dt
        pred[day] = max(0.0, Q * (state[5] - c0))
    return pred

@njit(cache=True)
def simulate_network(
    x: np.ndarray,
    y: np.ndarray,
    z_activity: np.ndarray,
    weights: np.ndarray,
    p_act: float = P_ACT_DEFAULT,
    steps: int = 8,
) -> np.ndarray:
    """Simulate M2 with positivity-safe RK4 stage-flow evaluation."""
    n = y.size
    m = z_activity.size
    Q = math.exp(x[0]); beta0 = math.exp(x[1]); q = math.exp(x[2]); a = math.exp(x[3])
    s0 = 0.001 + 0.998 * _sigmoid(x[4])
    y0 = max(y[0], 0.5)
    e0 = y0 / (Q * SIGMA); i0 = y0 / (Q * GAMMA)
    if e0 + i0 > 0.5:
        scale = 0.5 / (e0 + i0); e0 *= scale; i0 *= scale
    s0 = min(s0, max(1e-6, 1.0 - e0 - i0 - 1e-6))
    u0 = max(1e-8, 1.0 - s0 - e0 - i0)

    U = u0 * weights.copy(); S = s0 * weights.copy()
    E = e0 * weights.copy(); I = i0 * weights.copy(); R = np.zeros(m, dtype=np.float64)
    pred = np.empty(n, dtype=np.float64)
    dt = 1.0 / steps; t = 0.0; cumulative = 0.0

    k1 = np.empty((5, m), dtype=np.float64); k2 = np.empty((5, m), dtype=np.float64)
    k3 = np.empty((5, m), dtype=np.float64); k4 = np.empty((5, m), dtype=np.float64)
    U2=np.empty(m); S2=np.empty(m); E2=np.empty(m); I2=np.empty(m); R2=np.empty(m)
    U3=np.empty(m); S3=np.empty(m); E3=np.empty(m); I3=np.empty(m); R3=np.empty(m)
    U4=np.empty(m); S4=np.empty(m); E4=np.empty(m); I4=np.empty(m); R4=np.empty(m)

    for day in range(n):
        c0 = cumulative
        for _ in range(steps):
            theta1 = 0.0
            for j in range(m):
                theta1 += z_activity[j] * max(I[j], 0.0)
            theta1 = max(theta1, 0.0)
            beta1 = beta0 * math.exp(-q * t); dc1 = 0.0
            for j in range(m):
                uj=max(U[j],0.0); sj=max(S[j],0.0); ej=max(E[j],0.0); ij=max(I[j],0.0)
                act=a*z_activity[j]*uj*theta1; inf=beta1*z_activity[j]*sj*theta1
                k1[0,j]=-act; k1[1,j]=(1.0-p_act)*act-inf
                k1[2,j]=p_act*act+inf-SIGMA*ej; k1[3,j]=SIGMA*ej-GAMMA*ij; k1[4,j]=GAMMA*ij
                dc1 += SIGMA*ej
                U2[j]=U[j]+0.5*dt*k1[0,j]; S2[j]=S[j]+0.5*dt*k1[1,j]
                E2[j]=E[j]+0.5*dt*k1[2,j]; I2[j]=I[j]+0.5*dt*k1[3,j]; R2[j]=R[j]+0.5*dt*k1[4,j]

            theta2 = 0.0
            for j in range(m):
                theta2 += z_activity[j] * max(I2[j], 0.0)
            theta2=max(theta2,0.0)
            beta2=beta0*math.exp(-q*(t+0.5*dt)); dc2=0.0
            for j in range(m):
                uj=max(U2[j],0.0); sj=max(S2[j],0.0); ej=max(E2[j],0.0); ij=max(I2[j],0.0)
                act=a*z_activity[j]*uj*theta2; inf=beta2*z_activity[j]*sj*theta2
                k2[0,j]=-act; k2[1,j]=(1.0-p_act)*act-inf
                k2[2,j]=p_act*act+inf-SIGMA*ej; k2[3,j]=SIGMA*ej-GAMMA*ij; k2[4,j]=GAMMA*ij
                dc2 += SIGMA*ej
                U3[j]=U[j]+0.5*dt*k2[0,j]; S3[j]=S[j]+0.5*dt*k2[1,j]
                E3[j]=E[j]+0.5*dt*k2[2,j]; I3[j]=I[j]+0.5*dt*k2[3,j]; R3[j]=R[j]+0.5*dt*k2[4,j]

            theta3 = 0.0
            for j in range(m):
                theta3 += z_activity[j] * max(I3[j], 0.0)
            theta3=max(theta3,0.0); dc3=0.0
            for j in range(m):
                uj=max(U3[j],0.0); sj=max(S3[j],0.0); ej=max(E3[j],0.0); ij=max(I3[j],0.0)
                act=a*z_activity[j]*uj*theta3; inf=beta2*z_activity[j]*sj*theta3
                k3[0,j]=-act; k3[1,j]=(1.0-p_act)*act-inf
                k3[2,j]=p_act*act+inf-SIGMA*ej; k3[3,j]=SIGMA*ej-GAMMA*ij; k3[4,j]=GAMMA*ij
                dc3 += SIGMA*ej
                U4[j]=U[j]+dt*k3[0,j]; S4[j]=S[j]+dt*k3[1,j]
                E4[j]=E[j]+dt*k3[2,j]; I4[j]=I[j]+dt*k3[3,j]; R4[j]=R[j]+dt*k3[4,j]

            theta4 = 0.0
            for j in range(m):
                theta4 += z_activity[j] * max(I4[j], 0.0)
            theta4=max(theta4,0.0)
            beta4=beta0*math.exp(-q*(t+dt)); dc4=0.0
            for j in range(m):
                uj=max(U4[j],0.0); sj=max(S4[j],0.0); ej=max(E4[j],0.0); ij=max(I4[j],0.0)
                act=a*z_activity[j]*uj*theta4; inf=beta4*z_activity[j]*sj*theta4
                k4[0,j]=-act; k4[1,j]=(1.0-p_act)*act-inf
                k4[2,j]=p_act*act+inf-SIGMA*ej; k4[3,j]=SIGMA*ej-GAMMA*ij; k4[4,j]=GAMMA*ij
                dc4 += SIGMA*ej

            for j in range(m):
                U[j]+=dt*(k1[0,j]+2*k2[0,j]+2*k3[0,j]+k4[0,j])/6.0
                S[j]+=dt*(k1[1,j]+2*k2[1,j]+2*k3[1,j]+k4[1,j])/6.0
                E[j]+=dt*(k1[2,j]+2*k2[2,j]+2*k3[2,j]+k4[2,j])/6.0
                I[j]+=dt*(k1[3,j]+2*k2[3,j]+2*k3[3,j]+k4[3,j])/6.0
                R[j]+=dt*(k1[4,j]+2*k2[4,j]+2*k3[4,j]+k4[4,j])/6.0
                if U[j] < 0.0: U[j]=0.0
                if S[j] < 0.0: S[j]=0.0
                if E[j] < 0.0: E[j]=0.0
                if I[j] < 0.0: I[j]=0.0
                if R[j] < 0.0: R[j]=0.0
                mass=U[j]+S[j]+E[j]+I[j]+R[j]
                if mass > 0.0 and abs(mass-weights[j]) > 1e-10:
                    sc=weights[j]/mass
                    U[j]*=sc; S[j]*=sc; E[j]*=sc; I[j]*=sc; R[j]*=sc

            cumulative += dt*(dc1+2*dc2+2*dc3+dc4)/6.0
            t += dt
        pred[day]=max(0.0,Q*(cumulative-c0))
    return pred

def information_criteria(sse: float, n: int, k: int) -> tuple[float, float]:
    aic = n * math.log(max(float(sse) / n, 1e-12)) + 2.0 * k
    if n <= k + 1:
        return aic, float("inf")
    aicc = aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)
    return aic, aicc


def parameter_bounds(y: np.ndarray, reservoir: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = max(float(np.sum(y)), 1.0)
    peak = max(float(np.max(y)), 1.0)
    q_low = max(total * 1.01, peak * 2.0, 30.0)
    q_high = max(1e8, q_low * 2.0)
    if reservoir:
        lower = np.array([math.log(q_low), math.log(0.03), math.log(1e-4), math.log(1e-3), -7.0])
        upper = np.array([math.log(q_high), math.log(10.0), math.log(1.0), math.log(1e4), 7.0])
        center = np.array([math.log(max(total*3.0, q_low*1.2)), math.log(1.0), math.log(0.04), math.log(5.0), -1.0])
    else:
        lower = np.array([math.log(q_low), math.log(0.03), math.log(1e-4)])
        upper = np.array([math.log(q_high), math.log(10.0), math.log(1.0)])
        center = np.array([math.log(max(total*3.0, q_low*1.2)), math.log(1.0), math.log(0.04)])
    return lower, upper, center


def _start_points(
    lower: np.ndarray,
    upper: np.ndarray,
    center: np.ndarray,
    starts: int,
    seed: int,
    warm: Sequence[np.ndarray] | None = None,
) -> list[np.ndarray]:
    eps = 1e-8
    out: list[np.ndarray] = []
    if warm:
        for x in warm:
            arr = np.asarray(x, dtype=float)
            if arr.shape == center.shape and np.all(np.isfinite(arr)):
                out.append(np.clip(arr, lower+eps, upper-eps))
    out.append(np.clip(center, lower+eps, upper-eps))
    rng = np.random.default_rng(seed)
    while len(out) < max(1, starts):
        # Half log-uniform global starts, half local perturbations around center.
        if len(out) % 2:
            xx = lower + (upper-lower) * rng.uniform(size=lower.size)
        else:
            xx = center + rng.normal(scale=0.30, size=lower.size) * (upper-lower)
        out.append(np.clip(xx, lower+eps, upper-eps))
    return out[:max(starts, len(warm) if warm else 0, 1)]


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
    """Continue only a selected best fit that did not converge.

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
    """
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
    y: np.ndarray,
    starts: int = 5,
    max_nfev: int = 260,
    steps: int = 8,
    seed: int = 1,
    warm: Sequence[np.ndarray] | None = None,
) -> FitResult:
    y = np.asarray(y, dtype=np.float64)
    lower, upper, center = parameter_bounds(y, reservoir=False)
    x0s = _start_points(lower, upper, center, starts, seed, warm)

    def residual(x: np.ndarray) -> np.ndarray:
        return np.log1p(simulate_classic(x, y, steps)) - np.log1p(y)

    best = None
    for x0 in x0s:
        opt = least_squares(residual, x0, bounds=(lower, upper), max_nfev=max_nfev,
                            xtol=1e-7, ftol=1e-7, gtol=1e-7, x_scale="jac")
        sse = float(np.dot(opt.fun, opt.fun))
        if best is None or sse < best[0]:
            best = (sse, opt)
    assert best is not None
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


def fit_reservoir(
    y: np.ndarray,
    p_act: float = P_ACT_DEFAULT,
    starts: int = 6,
    max_nfev: int = 320,
    steps: int = 8,
    seed: int = 2,
    warm: Sequence[np.ndarray] | None = None,
) -> FitResult:
    y = np.asarray(y, dtype=np.float64)
    lower, upper, center = parameter_bounds(y, reservoir=True)
    x0s = _start_points(lower, upper, center, starts, seed, warm)

    def residual(x: np.ndarray) -> np.ndarray:
        return np.log1p(simulate_reservoir(x, y, p_act, steps)) - np.log1p(y)

    best = None
    for x0 in x0s:
        opt = least_squares(residual, x0, bounds=(lower, upper), max_nfev=max_nfev,
                            xtol=1e-7, ftol=1e-7, gtol=1e-7, x_scale="jac")
        sse = float(np.dot(opt.fun, opt.fun))
        if best is None or sse < best[0]:
            best = (sse, opt)
    assert best is not None
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


def fit_network_profile(
    y: np.ndarray,
    reservoir_fit: FitResult | np.ndarray,
    h_grid: Iterable[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    p_act: float = P_ACT_DEFAULT,
    m: int = 12,
    starts_per_h: int = 2,
    max_nfev: int = 180,
    steps: int = 8,
    seed: int = 3,
) -> tuple[FitResult, list[FitResult]]:
    """Profile the network heterogeneity grid and return best plus all fits."""
    y = np.asarray(y, dtype=np.float64)
    x_res = reservoir_fit.x if isinstance(reservoir_fit, FitResult) else np.asarray(reservoir_fit, dtype=float)
    lower, upper, center = parameter_bounds(y, reservoir=True)
    profiles: list[FitResult] = []
    prev_x = x_res.copy()

    for ih, h in enumerate(h_grid):
        z, w, h_real = make_activity_classes(float(h), m=m)
        second = float(np.dot(w, z*z))
        # At h=0 the model is exactly nested; retain reservoir optimum as a
        # guaranteed baseline, but still allow a local refinement.
        warm = [x_res, prev_x]
        x0s = _start_points(lower, upper, center, starts_per_h, seed + 101*ih, warm)

        def residual(x: np.ndarray) -> np.ndarray:
            return np.log1p(simulate_network(x, y, z, w, p_act, steps)) - np.log1p(y)

        best = None
        for x0 in x0s:
            opt = least_squares(residual, x0, bounds=(lower, upper), max_nfev=max_nfev,
                                xtol=2e-7, ftol=2e-7, gtol=2e-7, x_scale="jac")
            sse = float(np.dot(opt.fun, opt.fun))
            if best is None or sse < best[0]:
                best = (sse, opt)
        assert best is not None
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
        profiles.append(result)
        prev_x = opt.x.copy()

    best_result = min(profiles, key=lambda r: (r.aicc, r.sse))
    return best_result, profiles


def parameter_summary(
    fit: FitResult,
    y: np.ndarray,
    p_act: float = P_ACT_DEFAULT,
) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    if fit.model == "classic":
        Q, beta0, q = map(math.exp, fit.x[:3])
        return {
            "Q": Q,
            "beta0": beta0,
            "q": q,
            "a": float("nan"),
            "s0": 1.0,
            "u0": 0.0,
            "R0": beta0 / GAMMA,
            "heterogeneity_multiplier": 1.0,
        }
    Q, beta0, q, a, s0_raw = unpack_reservoir_x(fit.x)
    obs0 = max(float(y[0]), 0.5)
    e0 = obs0/(Q*SIGMA); i0 = obs0/(Q*GAMMA)
    if e0+i0 > 0.5:
        sc=0.5/(e0+i0); e0*=sc; i0*=sc
    s0 = min(s0_raw, max(1e-6, 1.0-e0-i0-1e-6))
    u0 = max(1e-8, 1.0-s0-e0-i0)
    H = fit.z_second_moment if fit.model == "network" else 1.0
    R0 = H * (p_act*a*u0 + beta0*s0) / GAMMA
    return {
        "Q": Q,
        "beta0": beta0,
        "q": q,
        "a": a,
        "s0": s0,
        "u0": u0,
        "R0": R0,
        "heterogeneity_multiplier": H,
    }


def simulate_network_detailed(
    fit_or_x: FitResult | np.ndarray,
    y: np.ndarray,
    h: float | None = None,
    p_act: float = P_ACT_DEFAULT,
    m: int = 12,
    steps: int = 8,
) -> dict[str, np.ndarray | float]:
    """Pure-Python detailed daily network states and recruitment diagnostics."""
    if isinstance(fit_or_x, FitResult):
        x = fit_or_x.x
        h_use = fit_or_x.h_target if h is None else h
    else:
        x = np.asarray(fit_or_x, dtype=float)
        if h is None:
            raise ValueError("h is required when fit_or_x is an array")
        h_use = h
    y = np.asarray(y, dtype=float)
    z, w, h_real = make_activity_classes(float(h_use), m=m)
    Q, beta0, q, a, s0_raw = unpack_reservoir_x(x)
    obs0=max(float(y[0]),.5); e0=obs0/(Q*SIGMA); i0=obs0/(Q*GAMMA)
    if e0+i0>.5:
        sc=.5/(e0+i0); e0*=sc; i0*=sc
    s0=min(s0_raw,max(1e-6,1-e0-i0-1e-6)); u0=max(1e-8,1-s0-e0-i0)
    U=u0*w.copy(); S=s0*w.copy(); E=e0*w.copy(); I=i0*w.copy(); R=np.zeros(m)
    n=y.size; dt=1/steps; t=0.0; C=0.0
    pred=np.zeros(n); Uhead=np.zeros(n); Shead=np.zeros(n); Sedge=np.zeros(n); thetaI=np.zeros(n)
    mean_z_S=np.zeros(n); top20_share=np.zeros(n); recruited=np.zeros(n); direct=np.zeros(n)
    order=np.argsort(z); top_idx=order[int(math.floor(.8*m)):]
    cum_recr=0.0; cum_direct=0.0

    def rhs(U_,S_,E_,I_,R_,tt):
        # RK4 intermediate stages can become slightly negative when parameters
        # are near numerical bounds. Evaluate epidemiological flows from the
        # non-negative orthant so cumulative recruitment and infection counts
        # remain physically meaningful.
        U_pos=np.maximum(U_,0.0); S_pos=np.maximum(S_,0.0)
        E_pos=np.maximum(E_,0.0); I_pos=np.maximum(I_,0.0)
        theta=max(0.0,float(np.dot(z,I_pos))); beta=beta0*math.exp(-q*tt)
        act=a*z*U_pos*theta; inf=beta*z*S_pos*theta
        return (-act,(1-p_act)*act-inf,p_act*act+inf-SIGMA*E_pos,SIGMA*E_pos-GAMMA*I_pos,GAMMA*I_pos,float(SIGMA*E_pos.sum()),float(((1-p_act)*act).sum()),float((p_act*act).sum()))

    for day in range(n):
        c0=C
        for _ in range(steps):
            k1=rhs(U,S,E,I,R,t)
            k2=rhs(U+.5*dt*k1[0],S+.5*dt*k1[1],E+.5*dt*k1[2],I+.5*dt*k1[3],R+.5*dt*k1[4],t+.5*dt)
            k3=rhs(U+.5*dt*k2[0],S+.5*dt*k2[1],E+.5*dt*k2[2],I+.5*dt*k2[3],R+.5*dt*k2[4],t+.5*dt)
            k4=rhs(U+dt*k3[0],S+dt*k3[1],E+dt*k3[2],I+dt*k3[3],R+dt*k3[4],t+dt)
            U += dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
            S += dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
            E += dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6
            I += dt*(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6
            R += dt*(k1[4]+2*k2[4]+2*k3[4]+k4[4])/6
            for arr in (U,S,E,I,R): np.maximum(arr,0,out=arr)
            mass=U+S+E+I+R; factor=np.divide(w,mass,out=np.ones_like(w),where=mass>0)
            U*=factor; S*=factor; E*=factor; I*=factor; R*=factor
            C += dt*(k1[5]+2*k2[5]+2*k3[5]+k4[5])/6
            cum_recr += dt*(k1[6]+2*k2[6]+2*k3[6]+k4[6])/6
            cum_direct += dt*(k1[7]+2*k2[7]+2*k3[7]+k4[7])/6
            t += dt
        pred[day]=Q*(C-c0); Uhead[day]=Q*U.sum(); Shead[day]=Q*S.sum(); Sedge[day]=Q*np.dot(z,S)
        thetaI[day]=Q*np.dot(z,I)
        mean_z_S[day]=float(np.dot(z,S)/max(S.sum(),1e-15))
        top20_share[day]=float(S[top_idx].sum()/max(S.sum(),1e-15))
        recruited[day]=Q*cum_recr; direct[day]=Q*cum_direct
    return {
        "pred":pred,"U_head":Uhead,"S_head":Shead,"S_edge":Sedge,"theta_I":thetaI,
        "mean_z_S":mean_z_S,"top20_active_share":top20_share,
        "cumulative_recruited_to_S":recruited,"cumulative_direct_from_U":direct,
        "z":z,"weights":w,"h_realized":h_real,"second_moment":float(np.dot(w,z*z)),
        "Q":Q,"beta0":beta0,"q":q,"a":a,"s0":s0,"u0":u0,
    }


def akaike_weights(aicc_values: Sequence[float]) -> np.ndarray:
    a = np.asarray(aicc_values, dtype=float)
    finite = np.isfinite(a)
    out = np.zeros_like(a)
    if not np.any(finite):
        return out
    delta = a[finite] - np.min(a[finite])
    w = np.exp(-0.5 * delta)
    out[finite] = w / w.sum()
    return out
