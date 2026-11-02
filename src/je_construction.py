"""
Wrap-around construction of jointly exclusive (JE) random vectors.

Implements Theorem 3.2 (Section 3.1) of the paper:
  - Existence condition tau = sum q_i <= n - 1
  - Explicit circular-arc construction X_i* = Y_i * 1{Theta in R_i}
  - Verification of marginals, JE property, claim-count support, and convex order
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


def compute_arcs(q: Sequence[float]) -> List[Tuple[float, float]]:
    """
    Compute the start and length of each circular arc R_i on [0, 1).

    The wrap-around construction (Theorem 3.2) places component i on the arc
    of length q_i beginning at sigma_{i-1} = q_1 + ... + q_{i-1}, taken modulo 1.

    Parameters
    ----------
    q : sequence of float
        Claim probabilities q_i = P(X_i > 0) in [0, 1].

    Returns
    -------
    list of (start, length) pairs
        Each pair defines the arc R_i; start is in [0, 1).
    """
    q = np.asarray(q, dtype=float)
    assert np.all(q >= 0) and np.all(q <= 1), "q_i must lie in [0, 1]"
    sigma = np.concatenate([[0.0], np.cumsum(q)])
    arcs = []
    for i, qi in enumerate(q):
        start = sigma[i] % 1.0
        arcs.append((float(start), float(qi)))
    return arcs


def in_arc(theta: np.ndarray, start: float, length: float) -> np.ndarray:
    """
    Test membership Theta in R_i for the wrap-around arc.

    R_i is the interval of length `length` starting at `start`, taken modulo 1.
    When start + length < 1, membership is [start, start + length).
    When start + length >= 1, the arc wraps and membership is
    [start, 1) union [0, start + length - 1).

    Parameters
    ----------
    theta : ndarray of float in [0, 1)
        Uniform samples.
    start : float
        Arc start in [0, 1).
    length : float
        Arc length (= q_i).

    Returns
    -------
    ndarray of bool
    """
    if length == 0.0:
        return np.zeros(len(theta), dtype=bool)
    # Modular formula: (theta - start) mod 1 < length.
    # This avoids floating-point boundary errors in the end = start + length
    # computation (e.g. 0.8 + 0.4 == 1.2000000000000002).
    return (theta - start) % 1.0 < length


def sample_je_wraparound(
    q: Sequence[float],
    severity_samplers: Sequence[Callable[[int, np.random.Generator], np.ndarray]],
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample from the wrap-around JE construction X_i* = Y_i * 1{Theta in R_i}.

    Implements the construction of Theorem 3.2 (Section 3.1).  Each severity
    sampler must return strictly positive values (samples from the zero-truncated
    positive part of F_i).

    Parameters
    ----------
    q : sequence of float
        Claim probabilities q_i = P(X_i > 0).
    severity_samplers : list of callables
        severity_samplers[i](n, rng) draws n i.i.d. samples from the
        zero-truncated distribution of X_i conditioned on X_i > 0.
    n_samples : int
        Number of Monte Carlo samples.
    rng : numpy Generator, optional
        Random number generator; created from RANDOM_SEED if not given.

    Returns
    -------
    ndarray of shape (n_samples, n)
        Simulated X* matrix; rows are samples, columns are components.
    """
    from src.config import RANDOM_SEED

    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    q = np.asarray(q, dtype=float)
    n = len(q)
    arcs = compute_arcs(q)

    theta = rng.uniform(0.0, 1.0, size=n_samples)
    X = np.zeros((n_samples, n), dtype=float)

    for i in range(n):
        qi = q[i]
        if qi == 0.0:
            continue
        mask = in_arc(theta, arcs[i][0], qi)
        n_active = int(mask.sum())
        if n_active > 0:
            Y = severity_samplers[i](n_active, rng)
            assert np.all(Y > 0), f"Severity sampler {i} must return positive values"
            X[mask, i] = Y

    return X


def verify_marginals(
    samples: np.ndarray,
    marginal_survival_fns: Sequence[Callable[[np.ndarray], np.ndarray]],
    x_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare empirical survival functions to theoretical targets on a grid.

    Parameters
    ----------
    samples : ndarray of shape (n_samples, n)
    marginal_survival_fns : list of callables
        marginal_survival_fns[i](x) returns P(X_i > x) for array x.
    x_grid : ndarray
        Grid of x values at which to compare.

    Returns
    -------
    empirical : ndarray of shape (n, len(x_grid))
    theoretical : ndarray of shape (n, len(x_grid))
    """
    n = samples.shape[1]
    empirical = np.zeros((n, len(x_grid)))
    theoretical = np.zeros((n, len(x_grid)))
    for i in range(n):
        for j, x in enumerate(x_grid):
            empirical[i, j] = np.mean(samples[:, i] > x)
            theoretical[i, j] = marginal_survival_fns[i](np.array([x]))[0]
    return empirical, theoretical


def verify_je_property(samples: np.ndarray) -> float:
    """
    Estimate the probability that all components are simultaneously positive.

    For a valid JE vector, this must be zero (up to Monte Carlo noise).

    Parameters
    ----------
    samples : ndarray of shape (n_samples, n)

    Returns
    -------
    float
        Estimated P(min_i X_i > 0).
    """
    return float(np.mean(np.all(samples > 0, axis=1)))


def compute_claim_counts(samples: np.ndarray) -> np.ndarray:
    """
    Compute the number of positive components in each sample (kappa*).

    Parameters
    ----------
    samples : ndarray of shape (n_samples, n)

    Returns
    -------
    ndarray of shape (n_samples,) with integer values
    """
    return (samples > 0).sum(axis=1)


def verify_claim_count_support(samples: np.ndarray, tau: float) -> dict:
    """
    Check that the empirical support of kappa* is exactly {floor(tau), ceil(tau)}.

    Theorem 3.2 (Section 3.1) states that the count kappa* = |A(X_N*)| takes
    values only in {floor(tau), ceil(tau)} where tau = sum_i bar{F}_i(0).

    Parameters
    ----------
    samples : ndarray of shape (n_samples, n)
    tau : float
        Sum of claim probabilities.

    Returns
    -------
    dict with keys:
        'expected_support': {floor(tau), ceil(tau)}
        'empirical_support': set of observed counts
        'support_correct': bool
        'kappa_distribution': dict count -> empirical probability
    """
    counts = compute_claim_counts(samples)
    floor_tau = int(np.floor(tau))
    ceil_tau = int(np.ceil(tau))
    expected = {floor_tau, ceil_tau}

    observed = set(np.unique(counts).tolist())
    values, freqs = np.unique(counts, return_counts=True)
    kappa_dist = {int(v): float(f / len(counts)) for v, f in zip(values, freqs)}

    return {
        "expected_support": expected,
        "empirical_support": observed,
        "support_correct": observed == expected,
        "kappa_distribution": kappa_dist,
        "kappa_mean": float(counts.mean()),
        "kappa_var": float(counts.var()),
        "tau": tau,
    }


def convex_order_comparison(
    samples_star: np.ndarray,
    alt_samples_list: List[np.ndarray],
    test_fn_names: List[str],
) -> dict:
    """
    Compare E[f(kappa*)] against E[f(kappa_alt)] for convex test functions f.

    For convex order kappa* <=_cx kappa_alt, we need E[f(kappa*)] <= E[f(kappa_alt)]
    for all convex f.  This function evaluates several convex test functions and
    checks this inequality empirically.

    Parameters
    ----------
    samples_star : ndarray of shape (n_mc, n)
        Samples from the wrap-around (minimal) construction.
    alt_samples_list : list of ndarray of shape (n_mc, n)
        Samples from alternative dependence structures.
    test_fn_names : list of str
        Names of test functions to evaluate ('k2', 'k3', 'exp05', 'exp1', 'exp2').

    Returns
    -------
    dict with empirical expectations for star and each alternative.
    """
    def apply_fn(counts: np.ndarray, name: str) -> float:
        if name == "k2":
            return float(np.mean(counts**2))
        elif name == "k3":
            return float(np.mean(counts**3))
        elif name == "exp05":
            return float(np.mean(np.exp(0.5 * counts)))
        elif name == "exp1":
            return float(np.mean(np.exp(counts)))
        elif name == "exp2":
            return float(np.mean(np.exp(2.0 * counts)))
        raise ValueError(f"Unknown test function: {name}")

    counts_star = compute_claim_counts(samples_star)
    results = {"star": {fn: apply_fn(counts_star, fn) for fn in test_fn_names}}
    for k, alt_samples in enumerate(alt_samples_list):
        counts_alt = compute_claim_counts(alt_samples)
        results[f"alt_{k}"] = {fn: apply_fn(counts_alt, fn) for fn in test_fn_names}

    # Verify convex order: star should be <= each alternative for all test fns
    dominated = {}
    for k in range(len(alt_samples_list)):
        key = f"alt_{k}"
        dominated[key] = all(
            results["star"][fn] <= results[key][fn] + 1e-6
            for fn in test_fn_names
        )
    results["star_dominates"] = dominated

    return results


def make_exponential_sampler(rate: float) -> Callable[[int, np.random.Generator], np.ndarray]:
    """
    Return a severity sampler drawing from Exponential(rate).

    Parameters
    ----------
    rate : float
        Rate parameter (mean = 1/rate).
    """
    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.exponential(1.0 / rate, size=n)
    return sampler


def make_uniform_truncated_sampler() -> Callable[[int, np.random.Generator], np.ndarray]:
    """
    Return a sampler for U(0, 1) conditioned on positivity (all values positive).
    """
    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=n)
    return sampler


def marginal_survival_exponential(rate: float, q: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return survival function of zero-inflated Exponential: P(X > x) = q * exp(-rate * x).

    Parameters
    ----------
    rate : float
        Rate of positive part.
    q : float
        Claim probability P(X > 0).
    """
    def surv(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return q * np.exp(-rate * x)
    return surv


def marginal_survival_zero_inflated_uniform(q: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return survival function: X=0 w.p. (1-q), X~Uniform(0,1) w.p. q.
    P(X > x) = q * (1 - x) for x in [0, 1]; 0 for x >= 1.

    Parameters
    ----------
    q : float
        Claim probability P(X > 0).
    """
    def surv(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where(x >= 1.0, 0.0, q * np.maximum(1.0 - x, 0.0))
    return surv


def sample_alternative_activity_patterns(
    q: Sequence[float],
    n_samples: int,
    rng: np.random.Generator,
    method: str = "independent",
) -> np.ndarray:
    """
    Sample binary activity indicators from alternative dependence structures.

    Used to construct alternative X vectors for convex-order comparison.

    Parameters
    ----------
    q : sequence of float
        Marginal claim probabilities.
    n_samples : int
    rng : np.random.Generator
    method : str
        'independent' - independent Bernoulli indicators,
        'comonotonic' - all on/off together using single uniform.

    Returns
    -------
    ndarray of shape (n_samples, n) with 0/1 entries
    """
    q = np.asarray(q, dtype=float)
    n = len(q)
    B = np.zeros((n_samples, n), dtype=float)

    if method == "independent":
        for i in range(n):
            B[:, i] = (rng.uniform(size=n_samples) < q[i]).astype(float)
    elif method == "comonotonic":
        U = rng.uniform(size=n_samples)
        for i in range(n):
            B[:, i] = (U < q[i]).astype(float)
    else:
        raise ValueError(f"Unknown method: {method}")

    return B
