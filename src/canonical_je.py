"""
Canonical jointly exclusive (JE) family with face-by-face specification.

Implements Section 3.2 of the paper:
  - Intermediate-face parameterization via face weights p_I and survival copulas C_I
  - Admissibility polytope for face weights
  - Simulation from canonical JE laws
  - Closed-form CDF (formula 3.10)
  - Trivariate Example 3.9
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Face enumeration
# ---------------------------------------------------------------------------

def enumerate_faces(n: int) -> List[FrozenSet[int]]:
    """
    Enumerate intermediate faces M = {I subset {0,...,n-1} : 2 <= |I| <= n-1}.

    Parameters
    ----------
    n : int
        Number of components.

    Returns
    -------
    list of frozensets
        Each frozenset is an intermediate face I.
    """
    indices = list(range(n))
    faces = []
    for k in range(2, n):
        for combo in combinations(indices, k):
            faces.append(frozenset(combo))
    return faces


def faces_containing(i: int, faces: List[FrozenSet[int]]) -> List[FrozenSet[int]]:
    """Return all faces in `faces` that contain component i."""
    return [I for I in faces if i in I]


# ---------------------------------------------------------------------------
# Admissibility check
# ---------------------------------------------------------------------------

def check_canonical_admissibility(
    q: Sequence[float],
    face_weights: Dict[FrozenSet[int], float],
    faces: Optional[List[FrozenSet[int]]] = None,
) -> dict:
    """
    Check whether face weights satisfy the canonical JE admissibility constraints.

    Constraints (Section 3.2):
      (i)  For all i: sum_{I in M_i} p_I <= bar{F}_i(0) = q_i
      (ii) sum_{I in M} (|I| - 1) * p_I >= tau - 1 = sum_i q_i - 1

    Parameters
    ----------
    q : sequence of float
        Claim probabilities q_i.
    face_weights : dict
        Mapping frozenset -> weight p_I >= 0.
    faces : list of frozensets, optional
        Intermediate faces; computed from n = len(q) if not provided.

    Returns
    -------
    dict with keys:
        'admissible': bool
        'upper_violations': list of (i, sum, q_i) for violated upper constraints
        'lower_sum': float
        'lower_bound': float
        'lower_ok': bool
    """
    q = np.asarray(q, dtype=float)
    n = len(q)
    tau = float(q.sum())

    if faces is None:
        faces = enumerate_faces(n)

    upper_violations = []
    for i in range(n):
        M_i = faces_containing(i, faces)
        pi_i = sum(face_weights.get(I, 0.0) for I in M_i)
        if pi_i > q[i] + 1e-10:
            upper_violations.append((i, float(pi_i), float(q[i])))

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
# Copula primitives (bivariate and general)
# ---------------------------------------------------------------------------

class IndependenceCopula:
    """Bivariate independence survival copula: C(u1, u2) = u1 * u2."""

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return (n, 2) array of independent U(0,1) samples."""
        return rng.uniform(size=(n, 2))


class ComonotonicCopula:
    """Comonotonic (upper Frechet) survival copula: C(u1, u2) = min(u1, u2)."""

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return (n, 2) array where both coordinates equal a single U(0,1)."""
        U = rng.uniform(size=n)
        return np.column_stack([U, U])


class CountermonotonicCopula:
    """Countermonotonic (lower Frechet) copula: C(u1, u2) = max(u1 + u2 - 1, 0)."""

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return (n, 2) array where second coordinate = 1 - first."""
        U = rng.uniform(size=n)
        return np.column_stack([U, 1.0 - U])


class IndependenceHighDimCopula:
    """d-dimensional independence copula: all coordinates independent U(0,1)."""

    def __init__(self, d: int) -> None:
        self.d = d

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(size=(n, self.d))


# ---------------------------------------------------------------------------
# Canonical JE sampler
# ---------------------------------------------------------------------------

def build_active_set_pmf(
    q: np.ndarray,
    face_weights: Dict[FrozenSet[int], float],
    faces: List[FrozenSet[int]],
) -> Tuple[List, np.ndarray]:
    """
    Build the probability mass function over active sets.

    Mass on each intermediate face I: p_I.
    Mass on axis {i}: q_i - pi_i = q_i - sum_{I in M_i} p_I.
    Mass at origin: 1 - sum_i q_i + sum_I (|I|-1) p_I  (formula 3.6).

    Parameters
    ----------
    q : ndarray
    face_weights : dict
    faces : list of frozensets

    Returns
    -------
    patterns : list of frozensets (None for origin)
    probs : ndarray of corresponding probabilities
    """
    n = len(q)
    tau = q.sum()
    patterns = []
    probs = []

    # Intermediate faces
    for I in faces:
        p_I = face_weights.get(I, 0.0)
        if p_I > 0:
            patterns.append(I)
            probs.append(p_I)

    # Axes {i}
    for i in range(n):
        M_i = faces_containing(i, faces)
        pi_i = sum(face_weights.get(I, 0.0) for I in M_i)
        axis_mass = float(q[i] - pi_i)
        if axis_mass > 1e-15:
            patterns.append(frozenset([i]))
            probs.append(axis_mass)

    # Origin — formula (3.6): p_0 = 1 - tau + sum_I (|I|-1) p_I
    lower_sum = sum((len(I) - 1) * face_weights.get(I, 0.0) for I in faces)
    origin_mass = 1.0 - tau + lower_sum
    if origin_mass > 1e-15:
        patterns.append(None)  # None represents origin
        probs.append(origin_mass)

    probs = np.array(probs, dtype=float)
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()
    return patterns, probs


def sample_canonical_je(
    q: Sequence[float],
    face_weights: Dict[FrozenSet[int], float],
    face_copulas: Dict[FrozenSet[int], object],
    severity_samplers: Sequence[Callable[[int, np.random.Generator], np.ndarray]],
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample from the canonical JE family (Section 3.2).

    For each sample:
    1. Draw active face A from the PMF over {origin, axes, intermediate faces}.
    2. Conditional on A = I, draw U from copula C_I.
    3. Map U to severities via quantile transforms of zero-truncated marginals.
    4. Set non-active coordinates to zero.

    Parameters
    ----------
    q : sequence of float
        Claim probabilities.
    face_weights : dict frozenset -> float
        Weights p_I for intermediate faces.
    face_copulas : dict frozenset -> copula object
        Copula for each intermediate face; must have a `sample(n, rng)` method
        returning an (n, |I|) array of U(0,1) values.
    severity_samplers : list of callables
        severity_samplers[i](u, rng) maps uniform u array to positive severities
        via quantile transform of zero-truncated F_i (i.e., F_i^{-1}(F_i(0) + u*(1-F_i(0)))).
    n_samples : int
    rng : np.random.Generator, optional

    Returns
    -------
    ndarray of shape (n_samples, n)
    """
    from src.config import RANDOM_SEED

    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    q = np.asarray(q, dtype=float)
    n = len(q)
    faces = enumerate_faces(n)

    patterns, probs = build_active_set_pmf(q, face_weights, faces)
    X = np.zeros((n_samples, n), dtype=float)

    # Assign each sample to an active face via multinomial draw
    counts = rng.multinomial(n_samples, probs)
    start = 0
    for pattern, cnt in zip(patterns, counts):
        if cnt == 0:
            continue
        idx = slice(start, start + cnt)
        start += cnt
        if pattern is None:
            # Origin: all zeros — already set
            pass
        elif len(pattern) == 1:
            # Axis {i}: only component i is active
            i = next(iter(pattern))
            # Sample U ~ U(0,1) and invert the zero-truncated marginal
            U = rng.uniform(size=cnt)
            X[idx, i] = severity_samplers[i](U, rng)
        else:
            # Intermediate face I: use copula C_I for joint uniforms
            I = pattern
            I_sorted = sorted(I)
            copula = face_copulas[I]
            U_mat = copula.sample(cnt, rng)  # shape (cnt, |I|)
            for j_local, j_global in enumerate(I_sorted):
                X[idx, j_global] = severity_samplers[j_global](U_mat[:, j_local], rng)

    # Shuffle rows since multinomial draw fills them in block order
    perm = rng.permutation(n_samples)
    return X[perm]


# ---------------------------------------------------------------------------
# CDF formula (3.10)
# ---------------------------------------------------------------------------

def canonical_je_cdf_survival(
    x: Sequence[float],
    q: Sequence[float],
    face_weights: Dict[FrozenSet[int], float],
    face_survival_copulas: Dict[FrozenSet[int], Callable[[Sequence[float]], float]],
    marginal_survival_fns: Sequence[Callable[[float], float]],
    faces: Optional[List[FrozenSet[int]]] = None,
) -> float:
    """
    Evaluate the joint survival probability P(X_i > x_i for i in I; X_j=0 for j not in I)
    summed correctly to give the CDF, based on formula (3.10) of the paper.

    For a single point x, formula (3.10) gives
        P(X_1 <= x_1, ..., X_n <= x_n)
    by inclusion-exclusion over all subsets, using the face-by-face representation.

    This implementation evaluates the survival function for a subset representation:
    P(X_i > x_i for all i in I, X_j = 0 for j not in I) for each face, then
    aggregates.

    Parameters
    ----------
    x : sequence of float
        Point at which to evaluate CDF; each x_i >= 0.
    q : sequence of float
    face_weights : dict
    face_survival_copulas : dict frozenset -> callable
        face_survival_copulas[I](u_list) evaluates the survival copula C_I(u_1,...,u_|I|)
        where u_i = bar{F}_i(x_i) / bar{F}_i(0) are the scaled survivals on [0,1].
    marginal_survival_fns : list of callables
        marginal_survival_fns[i](x) = P(X_i > x).
    faces : list of frozensets, optional

    Returns
    -------
    float
        P(X_1 <= x_1, ..., X_n <= x_n)
    """
    q_arr = np.asarray(q, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    n = len(q_arr)

    if faces is None:
        faces = enumerate_faces(n)

    # Compute scaled survival values s_i = bar{F}_i(x_i) / q_i for x_i >= 0
    s = np.zeros(n)
    for i in range(n):
        if q_arr[i] > 0:
            surv_i = float(marginal_survival_fns[i](x_arr[i]))
            s[i] = min(1.0, surv_i / q_arr[i])

    # Compute P(X=0) = 1 - tau + lower_sum  (origin mass, formula 3.6)
    lower_sum = sum((len(I) - 1) * face_weights.get(I, 0.0) for I in faces)
    tau = float(q_arr.sum())
    p_origin = 1.0 - tau + lower_sum

    # Accumulate CDF via sum over all events
    # CDF = P(all X_i <= x_i) = P(origin) + sum_i P(axis i, X_i <= x_i)
    #       + sum_{I in M} P(face I, X_i <= x_i for i in I)
    cdf = max(0.0, p_origin)

    # Axis contributions
    for i in range(n):
        M_i = faces_containing(i, faces)
        pi_i = sum(face_weights.get(I, 0.0) for I in M_i)
        axis_mass_i = q_arr[i] - pi_i
        if axis_mass_i > 0:
            # P(axis {i}, X_i <= x_i) = (q_i - pi_i) * P(Y_i^axis <= x_i)
            # where Y_i^axis has survival bar{F}_i(x_i) / (q_i - pi_i)
            # but here we need P(X_i <= x_i | active on axis) = 1 - s[i]
            # P(on axis {i} and X_i <= x_i)
            cdf += axis_mass_i * (1.0 - s[i])

    # Intermediate face contributions
    for I in faces:
        p_I = face_weights.get(I, 0.0)
        if p_I <= 0:
            continue
        I_sorted = sorted(I)
        u_list = [s[i] for i in I_sorted]
        # C_I(u_i for i in I) is the survival copula evaluated at scaled survivals
        copula_val = float(face_survival_copulas[I](u_list))
        # P(face I, X_i <= x_i for all i in I) = p_I * (1 - C_I(...)) [complement]
        # Actually, P(face I, X_i <= x_i) = p_I * (1 - P(X_i > x_i for all i in I | face I))
        #   = p_I * (1 - C_I(s[i], i in I))
        cdf += p_I * (1.0 - copula_val)

    return float(np.clip(cdf, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Trivariate Example 3.9 helpers
# ---------------------------------------------------------------------------

def make_example39_marginals(q: float = 0.5) -> dict:
    """
    Build marginals for Example 3.9 (Section 3.2):
    X_i = 0 w.p. (1-q), U(0,1) w.p. q.

    Parameters
    ----------
    q : float
        Claim probability (default 0.5).

    Returns
    -------
    dict with:
        'q_list': [q, q, q]
        'severity_samplers': list of callables (u, rng) -> x mapping uniform to severity
        'marginal_survival_fns': list of callables x -> P(X_i > x)
    """
    def surv(x: float) -> float:
        # P(X > x) = q*(1-x) for x in [0,1], 0 for x >= 1
        return float(q * max(0.0, 1.0 - max(x, 0.0)))

    def sev_sampler(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Zero-truncated U(0,1): sample directly as U(0,1) since it's already uniform
        return u  # u is U(0,1) from copula; uniform[0,1] quantile is identity

    return {
        "q_list": [q, q, q],
        "severity_samplers": [sev_sampler, sev_sampler, sev_sampler],
        "marginal_survival_fns": [surv, surv, surv],
    }


def make_example39_face_weights(lam: float) -> Dict[FrozenSet[int], float]:
    """
    Face weights for Example 3.9: p_I = lambda / 2 for each pair I.

    Parameters
    ----------
    lam : float
        lambda parameter in [1/3, 1/2].

    Returns
    -------
    dict frozenset -> float
    """
    # n=3: intermediate faces are the three pairs {0,1}, {0,2}, {1,2}
    pairs = [frozenset([0, 1]), frozenset([0, 2]), frozenset([1, 2])]
    return {I: lam / 2.0 for I in pairs}


def make_example39_copulas() -> Dict[FrozenSet[int], object]:
    """
    Face copulas for Example 3.9 (Section 3.2):
    C_{12} = independence, C_{13} = comonotonic, C_{23} = countermonotonic.
    """
    # Indices: 0-based (0=1, 1=2, 2=3 in paper's 1-based notation)
    from src.canonical_je import IndependenceCopula, ComonotonicCopula, CountermonotonicCopula
    return {
        frozenset([0, 1]): IndependenceCopula(),   # C_{12} independence
        frozenset([0, 2]): ComonotonicCopula(),    # C_{13} comonotonic
        frozenset([1, 2]): CountermonotonicCopula(),  # C_{23} countermonotonic
    }


def theoretical_correlations_example39(lam: float) -> dict:
    """
    Closed-form correlations for Example 3.9 (Section 3.2).

    Corr(X1, X2) = (6*lambda - 3) / 5
    Corr(X1, X3) = (8*lambda - 3) / 5
    Corr(X2, X3) = (4*lambda - 3) / 5

    Parameters
    ----------
    lam : float

    Returns
    -------
    dict with keys 'corr_12', 'corr_13', 'corr_23'.
    """
    return {
        "corr_12": (6.0 * lam - 3.0) / 5.0,
        "corr_13": (8.0 * lam - 3.0) / 5.0,
        "corr_23": (4.0 * lam - 3.0) / 5.0,
    }


def compute_empirical_correlations(samples: np.ndarray) -> dict:
    """
    Compute pairwise Pearson correlations from a sample matrix.

    Parameters
    ----------
    samples : ndarray of shape (n_samples, 3)

    Returns
    -------
    dict with keys 'corr_12', 'corr_13', 'corr_23'.
    """
    corr_mat = np.corrcoef(samples.T)
    return {
        "corr_12": float(corr_mat[0, 1]),
        "corr_13": float(corr_mat[0, 2]),
        "corr_23": float(corr_mat[1, 2]),
    }
