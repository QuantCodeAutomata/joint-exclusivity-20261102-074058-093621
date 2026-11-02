"""
Generalised jointly exclusive (G-JE) family with distorted face severities.

Implements Section 3.3 of the paper:
  - Distortion functions G_i such that face severities have survival G_i(bar{F}_i(x))
  - Capacity G_i* = inf ratio defining the admissibility bound per component
  - G-JE existence condition (formula 3.14/3.18)
  - Simulation from G-JE laws
  - Verification of confinement property for the linear truncation special case
  - Trivariate Example 3.13
"""

from __future__ import annotations

from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from src.canonical_je import (
    enumerate_faces,
    faces_containing,
    build_active_set_pmf,
)


# ---------------------------------------------------------------------------
# Capacity computation (formula 3.14)
# ---------------------------------------------------------------------------

def compute_capacity(
    G_i: Callable[[float], float],
    F_bar_i: Callable[[float], float],
    x_grid: np.ndarray,
) -> float:
    """
    Numerically approximate the capacity G_i* (formula 3.14 in Section 3.3).

    G_i* = inf_{0 <= x < y, G_i(bar{F}_i(x)) > G_i(bar{F}_i(y))}
               (bar{F}_i(x) - bar{F}_i(y)) / (G_i(bar{F}_i(x)) - G_i(bar{F}_i(y)))

    Evaluates the infimum over all pairs (x, y) with x < y in x_grid where the
    distorted survival is strictly decreasing.

    Parameters
    ----------
    G_i : callable float -> float
        Distortion function G_i: [0,1] -> [0,1], nondecreasing.
    F_bar_i : callable float -> float
        Marginal survival function bar{F}_i(x) = P(X_i > x).
    x_grid : ndarray
        Dense grid of x values on which to evaluate the infimum.

    Returns
    -------
    float
        G_i* (capacity).  Returns inf (no constraint) if the distortion is
        everywhere constant (no improvement in tightness possible).
    """
    sf = np.array([F_bar_i(x) for x in x_grid])
    gf = np.array([G_i(s) for s in sf])

    ratios = []
    m = len(x_grid)
    for j in range(m):
        for k in range(j + 1, m):
            d_gf = gf[j] - gf[k]
            d_sf = sf[j] - sf[k]
            if d_gf > 1e-14 and d_sf > 0:
                ratios.append(d_sf / d_gf)

    if not ratios:
        return float("inf")
    return float(np.min(ratios))


def compute_linear_truncation_capacity(a: float, b: float) -> float:
    """
    Compute G_i* analytically for the linear truncation distortion.

    Linear truncation: G_i(u) = 0 for u <= a, (u - a) / (b - a) for u in [a, b], 1 for u >= b.
    Capacity (closed form derived from Proposition 3.11 / Example 3.13):
        G_i* = b - a

    Parameters
    ----------
    a : float
        Lower truncation point.
    b : float
        Upper truncation point (= bar{F}_i(0) in the example).

    Returns
    -------
    float
    """
    return float(b - a)


# ---------------------------------------------------------------------------
# G-JE admissibility condition (formula 3.14 / 3.18)
# ---------------------------------------------------------------------------

def check_gje_admissibility(
    q: Sequence[float],
    capacities: Sequence[float],
) -> dict:
    """
    Check the G-JE existence condition (Section 3.3, formula 3.14).

    Condition:
        tau - 1 <= min(
            (n-2)/(n-1) * sum_i G_i*,
            sum_i G_i* - max_i G_i*
        )

    Parameters
    ----------
    q : sequence of float
        Claim probabilities q_i.
    capacities : sequence of float
        G_i* values.

    Returns
    -------
    dict with keys:
        'admissible': bool
        'lhs': tau - 1
        'rhs_1': (n-2)/(n-1) * sum G_i*
        'rhs_2': sum G_i* - max G_i*
        'rhs': min(rhs_1, rhs_2)
    """
    q = np.asarray(q, dtype=float)
    caps = np.asarray(capacities, dtype=float)
    n = len(q)
    tau = float(q.sum())

    lhs = tau - 1.0
    sum_caps = float(caps.sum())
    max_cap = float(caps.max())
    rhs_1 = (n - 2) / (n - 1) * sum_caps if n > 1 else 0.0
    rhs_2 = sum_caps - max_cap
    rhs = min(rhs_1, rhs_2)

    return {
        "admissible": lhs <= rhs + 1e-10,
        "lhs": lhs,
        "rhs_1": rhs_1,
        "rhs_2": rhs_2,
        "rhs": rhs,
        "tau": tau,
        "sum_caps": sum_caps,
        "max_cap": max_cap,
    }


# ---------------------------------------------------------------------------
# G-JE admissibility constraints for face weights (formula 3.8 with G_i*)
# ---------------------------------------------------------------------------

def check_gje_face_weight_admissibility(
    q: Sequence[float],
    capacities: Sequence[float],
    face_weights: Dict[FrozenSet[int], float],
    faces: Optional[List[FrozenSet[int]]] = None,
) -> dict:
    """
    Check face-weight constraints for G-JE (Section 3.3).

    For G-JE, the upper constraint per component is:
        sum_{I in M_i} p_I <= G_i*
    and the lower constraint is:
        sum_{I in M} (|I| - 1) p_I >= tau - 1.

    Parameters
    ----------
    q : sequence of float
    capacities : sequence of float
        G_i* values.
    face_weights : dict
    faces : list of frozensets, optional

    Returns
    -------
    dict
    """
    q_arr = np.asarray(q, dtype=float)
    n = len(q_arr)
    tau = float(q_arr.sum())
    caps = np.asarray(capacities, dtype=float)

    if faces is None:
        faces = enumerate_faces(n)

    upper_violations = []
    for i in range(n):
        M_i = faces_containing(i, faces)
        pi_i = sum(face_weights.get(I, 0.0) for I in M_i)
        if pi_i > caps[i] + 1e-10:
            upper_violations.append((i, float(pi_i), float(caps[i])))

    lower_sum = sum((len(I) - 1) * face_weights.get(I, 0.0) for I in faces)
    lower_bound = max(0.0, tau - 1.0)
    lower_ok = lower_sum >= lower_bound - 1e-10

    return {
        "admissible": len(upper_violations) == 0 and lower_ok,
        "upper_violations": upper_violations,
        "lower_sum": float(lower_sum),
        "lower_bound": float(lower_bound),
        "lower_ok": lower_ok,
    }


# ---------------------------------------------------------------------------
# G-JE sampler
# ---------------------------------------------------------------------------

def sample_gje(
    q: Sequence[float],
    marginal_survival_fns: Sequence[Callable[[float], float]],
    distortions: Sequence[Callable[[float], float]],
    capacities: Sequence[float],
    face_weights: Dict[FrozenSet[int], float],
    face_copulas: Dict[FrozenSet[int], object],
    distorted_severity_samplers: Sequence[Callable[[np.ndarray, np.random.Generator], np.ndarray]],
    axis_severity_samplers: Sequence[Callable[[np.ndarray, np.random.Generator], np.ndarray]],
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample from the G-JE family (Section 3.3).

    The face law for intermediate face I:
        P(X_i > x_i for i in I; X_j = 0 for j not in I)
            = p_I * C_I(G_i(bar{F}_i(x_i)) / G_i*, i in I)
        (actually: C_I(G_i(bar{F}_i(x_i)), i in I) directly,
         with severity distribution having survival G_i(bar{F}_i(.)))

    Axis law for component i:
        P(X_i > x_i; X_j = 0 for j != i)
            = bar{F}_i(x_i) - pi_i * G_i(bar{F}_i(x_i))

    Parameters
    ----------
    q : sequence of float
    marginal_survival_fns : list of callables x -> float
    distortions : list of callables u -> float
        G_i(u) for u in [0,1].
    capacities : sequence of float
        G_i* values.
    face_weights : dict frozenset -> float
    face_copulas : dict frozenset -> copula object
    distorted_severity_samplers : list of callables (u, rng) -> ndarray
        Maps U(0,1) uniforms to positive severities via quantile transform of the
        distorted zero-truncated distribution with survival G_i(bar{F}_i(.)) / G_i*.
        Specifically, sampler_i(u) = Q_i(u) where Q_i is the quantile of the
        distorted severity conditional on X_i > 0 on the face.
    axis_severity_samplers : list of callables (u, rng) -> ndarray
        Maps U(0,1) uniforms to positive severities via quantile transform of the
        axis-specific severity distribution bar{F}_i(x) - pi_i * G_i(bar{F}_i(x)).
    n_samples : int
    rng : np.random.Generator, optional

    Returns
    -------
    ndarray of shape (n_samples, n)
    """
    from src.config import RANDOM_SEED

    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    q_arr = np.asarray(q, dtype=float)
    n = len(q_arr)
    faces = enumerate_faces(n)

    # Build PMF: for G-JE, face mass p_I, axis mass q_i - pi_i, origin mass
    patterns, probs = build_active_set_pmf(q_arr, face_weights, faces)
    X = np.zeros((n_samples, n), dtype=float)

    counts = rng.multinomial(n_samples, probs)
    start = 0
    for pattern, cnt in zip(patterns, counts):
        if cnt == 0:
            start += cnt
            continue
        idx = slice(start, start + cnt)
        start += cnt
        if pattern is None:
            pass  # origin: all zeros
        elif len(pattern) == 1:
            i = next(iter(pattern))
            U = rng.uniform(size=cnt)
            X[idx, i] = axis_severity_samplers[i](U, rng)
        else:
            I = pattern
            I_sorted = sorted(I)
            copula = face_copulas[I]
            U_mat = copula.sample(cnt, rng)  # (cnt, |I|)
            for j_local, j_global in enumerate(I_sorted):
                X[idx, j_global] = distorted_severity_samplers[j_global](U_mat[:, j_local], rng)

    perm = rng.permutation(n_samples)
    return X[perm]


# ---------------------------------------------------------------------------
# Linear truncation helpers (Example 3.13)
# ---------------------------------------------------------------------------

def make_linear_truncation(a: float, b: float) -> Callable[[float], float]:
    """
    Return the linear truncation distortion G_i.

    G_i(u) = 0 for u in [0, a],
             (u - a) / (b - a) for u in [a, b],
             1 for u in [b, 1].

    Parameters
    ----------
    a : float
        Lower truncation threshold.
    b : float
        Upper truncation threshold.
    """
    def G_i(u: float) -> float:
        u = float(u)
        if u <= a:
            return 0.0
        if u >= b:
            return 1.0
        return (u - a) / (b - a)
    return G_i


def make_example313_distorted_sampler(
    a: float,
    b: float,
    q: float,
) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """
    Return a distorted face severity sampler for Example 3.13 (linear truncation).

    The distorted face severity has survival G_i(bar{F}_i(x)) / G_i*
    where G_i* = b - a = 3/8.

    For Example 3.13: bar{F}_i(x) = (1/2)(1-x), so
        G_i(bar{F}_i(x)) = G_i((1-x)/2)
    With linear truncation (a=1/8, b=1/2):
        G_i(u) = (u - 1/8)/(3/8) on [1/8, 1/2]
    So G_i(bar{F}_i(x)) = ((1-x)/2 - 1/8)/(3/8) = (3/8 - x/2)/(3/8) for x in [0, 3/4]
    The support of the face severity is x in (0, 3/4) (confinement property).

    The distorted CDF of the face severity conditional on face activation:
        P(X_i <= x | face I active) = 1 - G_i(bar{F}_i(x)) / G_i*
        = 1 - (3/8 - x/2) / (3/8) = x * (4/3)  for x in [0, 3/4]

    So the distorted face severity ~ Uniform(0, 3/4) * 1{positive}.

    Inverse CDF: x = u * (3/4).

    Parameters
    ----------
    a : float
    b : float
    q : float

    Returns
    -------
    callable (u, rng) -> ndarray
    """
    capacity = b - a
    upper = (q - a) / (q / (1.0))  # upper limit of face severity support
    # Derivation: G_i(bar{F}_i(x)) = 0 iff bar{F}_i(x) <= a, i.e. q*(1-x) <= a
    # => x >= 1 - a/q = 1 - (1/8)/(1/2) = 3/4
    # So support of face severity is x in (0, 1 - a/q)
    face_support_upper = 1.0 - a / q  # = 3/4 for a=1/8, q=1/2

    def sampler(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Quantile of Uniform(0, face_support_upper)
        return u * face_support_upper
    return sampler


def make_example313_axis_sampler(
    a: float,
    b: float,
    q: float,
    pi: float,
) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """
    Return an axis severity sampler for Example 3.13.

    Axis survival: bar{H}_i(x) = bar{F}_i(x) - pi * G_i(bar{F}_i(x))
    In [0, 3/4): bar{H}_i(x) = q*(1-x) - pi*((q*(1-x) - a)/(b-a))
    In [3/4, 1): bar{H}_i(x) = bar{F}_i(x) - pi (since G_i(bar{F}_i(x))=0 for x >= 3/4)

    For Example 3.13 with q=1/2, a=1/8, b=1/2, pi=lambda/4+1/8:
    the axis severity distribution includes the upper tail.

    This function uses numerical inversion for generality.

    Parameters
    ----------
    a : float
    b : float
    q : float
    pi : float
        pi_i = sum_{I in M_i} p_I

    Returns
    -------
    callable (u, rng) -> ndarray
    """
    capacity = b - a

    def axis_survival(x: float) -> float:
        s = q * max(1.0 - x, 0.0)  # bar{F}_i(x) for U(0,1) positive part
        G_val = max(0.0, min(1.0, (s - a) / capacity)) if s > a else 0.0
        return max(0.0, s - pi * G_val)

    def sampler(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Numerical inversion of the axis CDF
        # axis_CDF(x) = 1 - axis_survival(x), x in [0, 1]
        x_grid = np.linspace(0.0, 1.0 - 1e-9, 1000)
        surv_grid = np.array([axis_survival(x) for x in x_grid])
        # axis mass = axis_survival(0) = q - pi * G_i(q)
        # G_i(q) = (q - a) / (b - a) = 1 (since b = q)
        axis_mass = axis_survival(0.0)
        if axis_mass <= 1e-14:
            return np.zeros(len(u))
        # u maps to quantiles: u in [0,1] -> x values via 1 - axis_survival(x) / axis_mass
        cdf_grid = 1.0 - surv_grid / max(axis_mass, 1e-14)
        result = np.interp(u, cdf_grid, x_grid)
        return result

    return sampler


def verify_confinement(
    samples: np.ndarray,
    face_labels: np.ndarray,
    confinement_upper: float,
) -> dict:
    """
    Verify that joint losses on intermediate faces are confined to [0, confinement_upper).

    Parameters
    ----------
    samples : ndarray of shape (n_samples, n)
    face_labels : ndarray of shape (n_samples,)
        0 = origin, 1,...,n = axis, n+1,... = intermediate faces.
        Any value > 0 for intermediate face samples.
    confinement_upper : float

    Returns
    -------
    dict
    """
    on_face = face_labels > 0
    face_samples = samples[on_face]
    if len(face_samples) == 0:
        return {"confinement_verified": True, "max_face_value": float("nan")}

    max_val = float(np.max(face_samples[face_samples > 0])) if np.any(face_samples > 0) else 0.0
    return {
        "confinement_verified": max_val < confinement_upper,
        "max_face_value": max_val,
        "confinement_upper": confinement_upper,
    }
