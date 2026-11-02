"""
m-exclusivity hierarchy for non-negative random vectors.

Implements Sections 3.4 and 4.1 of the paper:
  - m-exclusivity definition: P(|A(X)| >= m) = 0
  - Existence condition tau <= m - 1
  - Wrap-around construction gives the strongest admissible level
  - Convex-order bounds: kappa* <=_cx |A(X)| <=_cx kappa_m#
  - Threshold exceedance exclusivity (Section 4.1)
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

from src.je_construction import (
    compute_arcs,
    in_arc,
    sample_je_wraparound,
    compute_claim_counts,
)


# ---------------------------------------------------------------------------
# m-exclusivity existence and hierarchy
# ---------------------------------------------------------------------------

def strongest_admissible_m(q: Sequence[float]) -> int:
    """
    Return the strongest (smallest) m such that an m-exclusive vector exists.

    A non-negative vector with marginals having claim probabilities q_i exists
    in the m-exclusive class if and only if tau = sum_i q_i <= m - 1.
    The smallest admissible m is ceil(tau) + 1 when tau is not an integer,
    or tau + 1 when tau is an integer.

    Actually, from the existence theorem: tau <= m-1, so m >= ceil(tau + 1) = floor(tau) + 2
    when tau is non-integer, and m >= tau + 1 when tau is integer.

    Equivalently, the smallest m is floor(tau) + 2 for non-integer tau (since
    tau <= floor(tau) + 1 = m - 1), and tau + 1 for integer tau.

    Parameters
    ----------
    q : sequence of float

    Returns
    -------
    int
        Smallest m for which m-exclusive vectors with marginals q exist.
    """
    tau = float(sum(q))
    return int(np.floor(tau)) + 2 if tau != int(tau) else int(tau) + 1


def check_m_exclusive_existence(q: Sequence[float], m: int) -> dict:
    """
    Check whether m-exclusive vectors with marginals q_i exist.

    Existence condition (Section 3.4): tau = sum_i q_i <= m - 1.

    Parameters
    ----------
    q : sequence of float
    m : int
        Exclusivity level.

    Returns
    -------
    dict with keys:
        'exists': bool
        'tau': float
        'bound': m - 1
        'strongest_m': int (smallest admissible m)
    """
    tau = float(sum(q))
    admissible = tau <= m - 1 + 1e-10
    return {
        "exists": admissible,
        "admissible": admissible,  # alias for exists
        "tau": tau,
        "bound": m - 1,
        "m": m,
        "strongest_m": strongest_admissible_m(q),
    }


def verify_m_exclusive(samples: np.ndarray, m: int) -> dict:
    """
    Estimate the probability that m or more components are simultaneously positive.

    For a valid m-exclusive vector, P(|A(X)| >= m) = 0.

    Parameters
    ----------
    samples : ndarray of shape (n_samples, n)
    m : int

    Returns
    -------
    dict with keys:
        'violation_rate': float (should be 0)
        'count_distribution': dict count -> empirical probability
    """
    counts = compute_claim_counts(samples)
    violation_rate = float(np.mean(counts >= m))
    values, freqs = np.unique(counts, return_counts=True)
    count_dist = {int(v): float(f / len(counts)) for v, f in zip(values, freqs)}
    return {
        "violation_rate": violation_rate,
        "count_distribution": count_dist,
        "max_count": int(counts.max()),
    }


# ---------------------------------------------------------------------------
# Upper extremal count distribution kappa_m# (Section 3.4)
# ---------------------------------------------------------------------------

def upper_extremal_count_distribution(tau: float, m: int) -> dict:
    """
    Compute the distribution of the upper extremal count kappa_m#.

    Kappa_m# = m - 1 with probability tau / (m - 1), and 0 otherwise.
    (Section 3.4, Proposition/Theorem on upper convex-order bound.)

    Parameters
    ----------
    tau : float
    m : int

    Returns
    -------
    dict with keys:
        'kappa_values': [0, m-1]
        'probabilities': [1 - tau/(m-1), tau/(m-1)]
        'mean': tau
        'variance': float
    """
    c = m - 1
    p_high = tau / c
    p_low = 1.0 - p_high
    mean = p_high * c  # = tau
    variance = p_high * c**2 - mean**2  # Var[kappa_m#] = p*c^2 - (p*c)^2 = tau*c - tau^2
    variance = max(0.0, tau * c - tau**2)
    return {
        "kappa_values": [0, c],
        "probabilities": [float(p_low), float(p_high)],
        "mean": float(mean),
        "variance": float(variance),
    }


def sample_upper_extremal_count(tau: float, m: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample from the upper extremal count distribution kappa_m#.

    Parameters
    ----------
    tau : float
    m : int
    n_samples : int
    rng : np.random.Generator

    Returns
    -------
    ndarray of shape (n_samples,) with values in {0, m-1}
    """
    c = m - 1
    p_high = tau / c
    return np.where(rng.uniform(size=n_samples) < p_high, c, 0)


def convex_order_bounds_check(
    q: Sequence[float],
    m: int,
    samples_je: np.ndarray,
    samples_alt: np.ndarray,
    test_fns: Optional[List[str]] = None,
) -> dict:
    """
    Empirically verify kappa* <=_cx |A(X)| <=_cx kappa_m# (Section 3.4).

    Parameters
    ----------
    q : sequence of float
    m : int
    samples_je : ndarray of shape (n_mc, n)
        Samples from the wrap-around (minimal) construction.
    samples_alt : ndarray of shape (n_mc, n)
        Samples from an alternative m-exclusive structure.
    test_fns : list of str, optional

    Returns
    -------
    dict
    """
    from src.config import RANDOM_SEED

    if test_fns is None:
        test_fns = ["k2", "k3", "exp05", "exp1", "exp2"]

    tau = float(sum(q))
    rng = np.random.default_rng(RANDOM_SEED + 99)
    n_mc = len(samples_je)

    # Upper extremal count samples
    kappa_upper = sample_upper_extremal_count(tau, m, n_mc, rng)

    def eval_fn(counts: np.ndarray, name: str) -> float:
        if name == "k2":
            return float(np.mean(counts**2))
        elif name == "k3":
            return float(np.mean(counts**3))
        elif name == "exp05":
            return float(np.mean(np.exp(0.5 * counts)))
        elif name == "exp1":
            return float(np.mean(np.exp(1.0 * counts)))
        elif name == "exp2":
            return float(np.mean(np.exp(2.0 * counts)))
        raise ValueError(f"Unknown: {name}")

    counts_je = compute_claim_counts(samples_je)
    counts_alt = compute_claim_counts(samples_alt)

    results = {}
    for fn in test_fns:
        e_je = eval_fn(counts_je, fn)
        e_alt = eval_fn(counts_alt, fn)
        e_upper = eval_fn(kappa_upper, fn)
        results[fn] = {
            "kappa_star": e_je,
            "alternative": e_alt,
            "kappa_upper": e_upper,
            "lower_bound_holds": e_je <= e_alt + 1e-4,
            "upper_bound_holds": e_alt <= e_upper + 1e-4,
        }

    return results


# ---------------------------------------------------------------------------
# Threshold exceedance exclusivity (Section 4.1)
# ---------------------------------------------------------------------------

def compute_exceedance_probabilities(
    marginal_survival_fns: Sequence[Callable[[float], float]],
    thresholds: Sequence[float],
) -> np.ndarray:
    """
    Compute P(Z_i > l_i) for each component via the survival function.

    Parameters
    ----------
    marginal_survival_fns : list of callables
        Survival function S_i(x) = P(Z_i > x) for each component i.
    thresholds : sequence of float
        l_i values.

    Returns
    -------
    ndarray of exceedance probabilities p_i = P(Z_i > l_i).
    """
    return np.array([float(sf(l)) for sf, l in zip(marginal_survival_fns, thresholds)])


def sample_threshold_exceedance_je(
    marginal_survival_fns: Sequence[Callable[[float], float]],
    thresholds: Sequence[float],
    excess_severity_samplers: Sequence[Callable[[int, np.random.Generator], np.ndarray]],
    subthreshold_samplers: Sequence[Callable[[int, np.random.Generator], np.ndarray]],
    m: int,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Construct m-exclusive threshold exceedance vector Z_tilde (Section 4.1).

    Construction (Proposition 4.1):
    1. Compute p_i = P(Z_i > l_i).
    2. Build excess X_i^+ = (Z_i - l_i)_+ using wrap-around with marginals of excesses.
    3. Sample sub-threshold Z_i^- from conditional law of Z_i | Z_i <= l_i, independently.
    4. Set Z_tilde_i = (l_i + X_i^+) * 1{X_i^+ > 0} + Z_i^- * 1{X_i^+ = 0}.

    Parameters
    ----------
    marginal_survival_fns : list of callables x -> float
        P(Z_i > x) for each component.
    thresholds : sequence of float
        l_i values.
    excess_severity_samplers : list of callables (n, rng) -> ndarray
        Samples from the excess distribution (Z_i - l_i | Z_i > l_i).
    subthreshold_samplers : list of callables (n, rng) -> ndarray
        Samples from the distribution of Z_i | Z_i <= l_i.
    m : int
        Exclusivity level.
    n_samples : int
    rng : np.random.Generator, optional

    Returns
    -------
    ndarray of shape (n_samples, n)
    """
    from src.config import RANDOM_SEED

    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    thresholds = np.asarray(thresholds, dtype=float)
    n = len(thresholds)

    # Compute exceedance probabilities p_i = P(Z_i > l_i) = bar{F}_i(l_i)
    p = np.array([float(marginal_survival_fns[i](thresholds[i])) for i in range(n)])

    # Check m-exclusive existence
    tau_excess = float(p.sum())
    assert tau_excess <= m - 1 + 1e-10, (
        f"Exceedance probabilities sum to {tau_excess:.4f} > m-1 = {m-1}: "
        "m-exclusive above thresholds does not exist"
    )

    # Use wrap-around with excess marginals
    excess_samples = sample_je_wraparound(p, excess_severity_samplers, n_samples, rng)

    # Assemble Z_tilde
    Z_tilde = np.zeros((n_samples, n), dtype=float)
    for i in range(n):
        active = excess_samples[:, i] > 0
        n_active = int(active.sum())
        n_inactive = n_samples - n_active

        if n_active > 0:
            Z_tilde[active, i] = thresholds[i] + excess_samples[active, i]
        if n_inactive > 0:
            Z_tilde[~active, i] = subthreshold_samplers[i](n_inactive, rng)

    return Z_tilde


def verify_threshold_exceedance_je(
    samples: np.ndarray,
    thresholds: Sequence[float],
    m: int,
) -> dict:
    """
    Verify that fewer than m components simultaneously exceed their thresholds.

    Parameters
    ----------
    samples : ndarray of shape (n_samples, n)
    thresholds : sequence of float
    m : int

    Returns
    -------
    dict
    """
    thresholds = np.asarray(thresholds, dtype=float)
    exceedances = samples > thresholds[None, :]  # (n_samples, n)
    exceed_counts = exceedances.sum(axis=1)
    violation_rate = float(np.mean(exceed_counts >= m))
    max_exceed = int(exceed_counts.max())
    return {
        "violation_rate": violation_rate,
        "max_simultaneous_exceedances": max_exceed,
        "je_property_holds": violation_rate < 1e-10,
        "m": m,
    }


def make_normal_subthreshold_sampler(
    mu: float, sigma: float, threshold: float
) -> Callable[[int, np.random.Generator], np.ndarray]:
    """
    Return a sampler for Normal(mu, sigma) conditioned on being <= threshold.

    Uses inverse CDF (truncated normal) via scipy.

    Parameters
    ----------
    mu, sigma : float
    threshold : float
    """
    from scipy.stats import truncnorm

    # Truncated normal on (-inf, threshold] standardized
    b_std = (threshold - mu) / sigma

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        U = rng.uniform(size=n)
        # CDF of truncated normal at threshold is 1; F_{trunc}(x) = Phi((x-mu)/sigma)/Phi(b_std)
        # Quantile: x = mu + sigma * Phi^{-1}(u * Phi(b_std))
        from scipy.special import ndtri
        phi_b = norm.cdf(b_std)
        return mu + sigma * ndtri(U * phi_b)

    return sampler


def make_normal_excess_sampler(
    mu: float, sigma: float, threshold: float
) -> Callable[[int, np.random.Generator], np.ndarray]:
    """
    Return a sampler for (Z - threshold | Z > threshold) where Z ~ Normal(mu, sigma).

    Samples the excess (Z - threshold) conditioned on Z > threshold.
    """
    from scipy.stats import truncnorm

    a_std = (threshold - mu) / sigma  # lower bound in standard units

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        # Truncated normal on (threshold, inf)
        U = rng.uniform(size=n)
        from scipy.special import ndtri
        phi_a = norm.cdf(a_std)
        raw = mu + sigma * ndtri(phi_a + U * (1.0 - phi_a))
        return np.maximum(raw - threshold, 0.0)

    return sampler
