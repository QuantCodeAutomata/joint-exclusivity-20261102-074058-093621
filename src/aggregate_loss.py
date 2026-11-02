"""
Aggregate-loss extremal structures, stop-loss premiums, and TVaR bounds.

Implements Section 4.2 of the paper:
  - Safest aggregate loss S* from the wrap-around construction
  - Stop-loss premium formula for S* (exchangeable frequency-severity model)
  - Upper bound within the m-exclusive frequency-severity class
  - Most dangerous structure S# over the Frechet class (Theorem 4.6)
  - TVaR computation
  - Benchmark table (Example 4.8): n=10, q=0.15, unit-mean exponential
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import integrate
from scipy.stats import gamma as gamma_dist
from scipy.special import gammainc, gammaincc

from src.je_construction import (
    compute_arcs,
    in_arc,
    sample_je_wraparound,
    make_exponential_sampler,
    sample_alternative_activity_patterns,
)
from src.m_exclusivity import sample_upper_extremal_count


# ---------------------------------------------------------------------------
# Stop-loss premium for Gamma distribution
# ---------------------------------------------------------------------------

def stop_loss_gamma(d: float, k: int, rate: float) -> float:
    """
    Compute the stop-loss premium E[(T - d)_+] for T ~ Gamma(k, rate).

    Formula (using incomplete gamma):
        E[(T - d)_+] = (k/rate) * (1 - F_{Gamma(k+1, rate)}(d)) - d * (1 - F_{Gamma(k, rate)}(d))

    Parameters
    ----------
    d : float
        Retention level.
    k : int or float
        Shape parameter (number of summands).
    rate : float
        Rate parameter (mean = k / rate).
    d : float
        Retention level.

    Returns
    -------
    float
    """
    if k <= 0:
        return 0.0
    if d <= 0:
        return float(k / rate)
    scale = 1.0 / rate
    mean = k * scale
    # E[(T-d)_+] = mean * (1 - F_{Gamma(k+1,rate)}(d)) - d * (1 - F_{Gamma(k,rate)}(d))
    surv_k = float(gamma_dist.sf(d, a=k, scale=scale))
    surv_k1 = float(gamma_dist.sf(d, a=k + 1, scale=scale))
    return float(mean * surv_k1 - d * surv_k)


def stop_loss_safest(
    d: float,
    tau: float,
    rate: float,
) -> float:
    """
    Compute stop-loss premium for the safest aggregate S* (Section 4.2, formula).

    Pi_{S*}(d) = (1 - tau + floor(tau)) * Pi_{T_{floor(tau)}}(d)
               + (tau - floor(tau)) * Pi_{T_{floor(tau)+1}}(d)

    where T_k = sum of k i.i.d. Exponential(rate) ~ Gamma(k, rate).

    Parameters
    ----------
    d : float
    tau : float
        Sum of claim probabilities.
    rate : float
        Exponential severity rate parameter.

    Returns
    -------
    float
    """
    k_lo = int(np.floor(tau))
    k_hi = k_lo + 1
    alpha = tau - k_lo  # weight on k_hi

    pi_lo = stop_loss_gamma(d, k_lo, rate) if k_lo >= 1 else max(0.0, -d)
    pi_hi = stop_loss_gamma(d, k_hi, rate)

    return float((1.0 - alpha) * pi_lo + alpha * pi_hi)


def stop_loss_class_upper_bound(
    d: float,
    tau: float,
    m: int,
    rate: float,
) -> float:
    """
    Upper bound on stop-loss premium within the m-exclusive frequency-severity class.

    Pi_S(d) <= (tau / (m-1)) * Pi_{T_{m-1}}(d)  (Section 4.2, formula after Proposition 3.16)

    Parameters
    ----------
    d : float
    tau : float
    m : int
    rate : float

    Returns
    -------
    float
    """
    c = m - 1
    return float((tau / c) * stop_loss_gamma(d, c, rate))


# ---------------------------------------------------------------------------
# Sampling structures for benchmark comparison
# ---------------------------------------------------------------------------

def sample_safest_aggregate(
    q: Sequence[float],
    severity_samplers: Sequence[Callable[[int, np.random.Generator], np.ndarray]],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample aggregate losses S* = sum_i X_i* from the wrap-around (safest) structure.

    Parameters
    ----------
    q : sequence of float
    severity_samplers : list of callables (n, rng) -> ndarray
    n_samples : int
    rng : np.random.Generator

    Returns
    -------
    ndarray of shape (n_samples,)
    """
    X = sample_je_wraparound(q, severity_samplers, n_samples, rng)
    return X.sum(axis=1)


def sample_independent_aggregate(
    q: Sequence[float],
    severity_sampler: Callable[[int, np.random.Generator], np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample aggregate losses under independent claim occurrences.

    B_i ~ Bernoulli(q_i) independently; S = sum_i B_i * Y_i.

    Parameters
    ----------
    q : sequence of float
    severity_sampler : callable (n, rng) -> ndarray (common severity)
    n_samples : int
    rng : np.random.Generator

    Returns
    -------
    ndarray of shape (n_samples,)
    """
    q_arr = np.asarray(q, dtype=float)
    n = len(q_arr)
    B = sample_alternative_activity_patterns(q_arr, n_samples, rng, method="independent")
    Y = severity_sampler(n * n_samples, rng).reshape(n_samples, n)
    return (B * Y).sum(axis=1)


def sample_comonotonic_aggregate(
    q: Sequence[float],
    severity_sampler: Callable[[int, np.random.Generator], np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample aggregate losses under co-monotonic claim occurrences in the
    frequency-severity class.

    All indicators B_i = 1{U < q_i} for a single U ~ U(0,1); S = sum_i B_i Y_i.

    Parameters
    ----------
    q : sequence of float
    severity_sampler : callable (n, rng) -> ndarray
    n_samples : int
    rng : np.random.Generator

    Returns
    -------
    ndarray of shape (n_samples,)
    """
    q_arr = np.asarray(q, dtype=float)
    n = len(q_arr)
    B = sample_alternative_activity_patterns(q_arr, n_samples, rng, method="comonotonic")
    Y = severity_sampler(n * n_samples, rng).reshape(n_samples, n)
    return (B * Y).sum(axis=1)


def sample_m_exclusive_upper_bound_aggregate(
    q: Sequence[float],
    tau: float,
    m: int,
    severity_sampler: Callable[[int, np.random.Generator], np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample aggregate losses from the upper extremal m-exclusive structure.

    Kappa_m# = m-1 with prob tau/(m-1), 0 otherwise.
    S = Kappa_m# * Y where Y ~ severity (i.i.d. copy for each active claim).

    Parameters
    ----------
    q : sequence of float
    tau : float
    m : int
    severity_sampler : callable
    n_samples : int
    rng : np.random.Generator

    Returns
    -------
    ndarray of shape (n_samples,)
    """
    kappa = sample_upper_extremal_count(tau, m, n_samples, rng)
    S = np.zeros(n_samples)
    for i in range(n_samples):
        k = int(kappa[i])
        if k > 0:
            S[i] = severity_sampler(k, rng).sum()
    return S


def sample_frechet_worst_case(
    q_sorted: np.ndarray,
    m: int,
    G_quantile: Callable[[np.ndarray], np.ndarray],
    marginal_quantiles: Sequence[Callable[[np.ndarray], np.ndarray]],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample from the Frechet-class worst case S# (Theorem 4.6, Section 4.2).

    Assumes q_sorted is in non-increasing order: q_1 >= ... >= q_n.
    With c = m - 1 and the common severity law G, the construction is:

        S# = sum_{i=1}^r F_i^{-1}(U) + (c-r)*G^{-1}((U - 1 + t)/t) * 1{U > 1-t}

    where r and t are determined by the q values (see the paper's formula in Theorem 4.6).

    Parameters
    ----------
    q_sorted : ndarray
        Claim probabilities in non-increasing order.
    m : int
    G_quantile : callable
        Quantile function of the common severity law G (for the shared part).
    marginal_quantiles : list of callables
        Quantile functions of F_i (marginal of X_i = q_i * G), one per component.
        For the equal-q case with common G, marginal_quantile[i](u) = G_quantile(u/q_i)
        restricted to u < q_i (i.e. X_i has atom at 0).
    n_samples : int
    rng : np.random.Generator

    Returns
    -------
    ndarray of shape (n_samples,)
    """
    n = len(q_sorted)
    c = m - 1
    tau = float(q_sorted.sum())

    # Paper formula (1-indexed q_1 >= ... >= q_n, Section 4.2 Theorem 4.6):
    #   t_k = (1/(c-k)) * sum_{j=k+1}^n q_j
    #   r   = min{k in {0,...,c-1} : q_{k+1} <= t_k}
    # In 0-indexed q_sorted[0..n-1]:
    #   sum_{j=k+1}^n q_j  (1-indexed) = sum(q_sorted[k:])  (0-indexed, k to n-1)
    #   q_{k+1}            (1-indexed) = q_sorted[k]         (0-indexed)
    r = c - 1        # fallback: last feasible k
    t = 0.0
    for k in range(c):
        remaining = c - k
        t_k = float(q_sorted[k:].sum()) / remaining
        if q_sorted[k] <= t_k + 1e-12:
            r = k
            t = t_k
            break

    # For the equal-q case with q_i = q for all i:
    #   t_k = (n-k)*q / (c-k); q <= t_k iff c-k <= n-k iff c <= n, which holds.
    #   So r = 0 and t = tau/c whenever the attainability condition max_i q_i <= tau/(m-1) holds.

    U = rng.uniform(size=n_samples)

    S = np.zeros(n_samples)

    # First r components: each contributes F_i^{-1}(U)
    for i in range(r):
        # F_i has atom at 0 of mass (1-q_i); for X_i = q_i * G(u):
        # quantile of X_i at u: 0 if u <= 1-q_i, G_quantile((u - 1+q_i)/q_i) if u > 1-q_i
        xi = marginal_quantiles[i](U)
        S += xi

    # Last (c - r) components: each contributes G^{-1}((U - 1 + t)/t) * 1{U > 1-t}
    active = U > 1.0 - t
    if active.any():
        u_rescaled = np.where(active, (U - (1.0 - t)) / t, 0.0)
        sev = G_quantile(np.clip(u_rescaled, 0.0, 1.0))
        S += (c - r) * np.where(active, sev, 0.0)

    return S


# ---------------------------------------------------------------------------
# TVaR computation
# ---------------------------------------------------------------------------

def tvar_from_samples(samples: np.ndarray, alpha: float) -> float:
    """
    Compute TVaR_alpha = E[X | X > VaR_alpha(X)] from samples.

    TVaR_alpha(X) = (1/(1-alpha)) * E[X * 1{X > VaR_alpha}]
                  = VaR_alpha + E[(X - VaR_alpha)_+ ] / (1 - alpha)

    Parameters
    ----------
    samples : ndarray of shape (n_samples,)
    alpha : float
        Confidence level in (0, 1).

    Returns
    -------
    float
    """
    q = float(np.quantile(samples, alpha))
    excess = np.maximum(samples - q, 0.0)
    return float(q + excess.mean() / (1.0 - alpha))


def tvar_from_stop_loss(
    var: float,
    stop_loss_fn: Callable[[float], float],
    alpha: float,
) -> float:
    """
    Compute TVaR_alpha from VaR and stop-loss premium:
        TVaR_alpha = VaR_alpha + Pi(VaR_alpha) / (1 - alpha)

    Parameters
    ----------
    var : float
        VaR at level alpha.
    stop_loss_fn : callable d -> Pi(d)
    alpha : float

    Returns
    -------
    float
    """
    return float(var + stop_loss_fn(var) / (1.0 - alpha))


# ---------------------------------------------------------------------------
# Benchmark table (Example 4.8, Section 4.2)
# ---------------------------------------------------------------------------

def compute_benchmark_table(
    n: int,
    q: float,
    rate: float,
    alpha: float,
    n_samples: int,
    rng: np.random.Generator,
) -> dict:
    """
    Reproduce the benchmark table of Example 4.8 (Section 4.2).

    Parameters (Example 4.8):
        n = 10, q_i = 0.15 for all i, unit-mean exponential severities (rate=1),
        TVaR at alpha = 0.99.

    Structures compared:
        1. Safest JE (S*): wrap-around construction
        2. Independent claim occurrences
        3. Most dangerous 3-exclusive upper bound
        4. Most dangerous JE (= n-exclusive upper bound)
        5. Co-monotonic claim occurrences
        6. Frechet-class worst case (Theorem 4.6)

    Parameters
    ----------
    n : int
    q : float
        Common claim probability.
    rate : float
        Exponential severity rate.
    alpha : float
        TVaR confidence level.
    n_samples : int
    rng : np.random.Generator

    Returns
    -------
    dict mapping structure name -> {'mean', 'variance', 'tvar'}
    """
    q_vec = [q] * n
    tau = n * q

    sev_sampler = make_exponential_sampler(rate)

    # Structure 1: Safest JE (S*)
    sev_samplers = [sev_sampler] * n
    S_star = sample_safest_aggregate(q_vec, sev_samplers, n_samples, rng)

    # Structure 2: Independent
    S_indep = sample_independent_aggregate(q_vec, sev_sampler, n_samples, rng)

    # Structure 3: Most dangerous 3-exclusive (m=3)
    m_3 = 3
    S_3excl = sample_m_exclusive_upper_bound_aggregate(
        q_vec, tau, m_3, sev_sampler, n_samples, rng
    )

    # Structure 4: Most dangerous JE (m = n)
    m_je = n
    S_je_worst = sample_m_exclusive_upper_bound_aggregate(
        q_vec, tau, m_je, sev_sampler, n_samples, rng
    )

    # Structure 5: Co-monotonic
    S_comono = sample_comonotonic_aggregate(q_vec, sev_sampler, n_samples, rng)

    # Structure 6: Frechet worst case (Theorem 4.6)
    # For equal q sorted in non-increasing order, with common G = Exp(rate)
    # G_quantile(u) = -log(1-u) / rate  (Exponential quantile)
    def G_quantile(u: np.ndarray) -> np.ndarray:
        return -np.log(1.0 - np.clip(u, 0.0, 1.0 - 1e-10)) / rate

    # Marginal quantile of X_i: 0 if U <= 1-q, G^{-1}((U-(1-q))/q) if U > 1-q
    def marginal_quantile_i(u: np.ndarray) -> np.ndarray:
        active = u > (1.0 - q)
        u_scaled = np.where(active, (u - (1.0 - q)) / q, 0.0)
        sev = G_quantile(np.clip(u_scaled, 0.0, 1.0))
        return np.where(active, sev, 0.0)

    q_sorted = np.array(sorted([q] * n, reverse=True))
    marginal_quantiles = [marginal_quantile_i] * n
    S_frechet = sample_frechet_worst_case(
        q_sorted, m_3, G_quantile, marginal_quantiles, n_samples, rng
    )

    structures = {
        "Safest (S*)": S_star,
        "Independent": S_indep,
        "3-exclusive worst": S_3excl,
        "JE worst": S_je_worst,
        "Co-monotonic": S_comono,
        "Frechet worst (m=3)": S_frechet,
    }

    results = {}
    for name, S in structures.items():
        results[name] = {
            "mean": float(S.mean()),
            "variance": float(S.var()),
            "tvar": tvar_from_samples(S, alpha),
        }

    return results


def compute_analytical_safest_stats(
    n: int,
    q: float,
    rate: float,
    alpha: float,
) -> dict:
    """
    Compute analytical mean, variance, and stop-loss premiums for S*.

    For i.i.d. exponential severities with rate, tau = n*q:
        E[S*] = tau / rate  (same as mean under any fixed-marginals coupling)
        Var[S*] = (floor(tau) + (tau - floor(tau))^2 ...) -- computed numerically

    Parameters
    ----------
    n : int
    q : float
    rate : float
    alpha : float

    Returns
    -------
    dict
    """
    tau = n * q
    mean_S = tau / rate

    # Variance of S*: Var[S*] = E[S*^2] - (E[S*])^2
    # S* is either T_{k_lo} or T_{k_hi}, kappa* is {k_lo, k_hi}
    k_lo = int(np.floor(tau))
    k_hi = k_lo + 1
    alpha_hi = tau - k_lo

    # E[T_k^2] = (k/rate^2) * (k + 1) for Gamma(k,rate)? No:
    # T_k ~ Gamma(k, rate): E[T_k] = k/rate, Var[T_k] = k/rate^2
    # E[T_k^2] = Var[T_k] + E[T_k]^2 = k/rate^2 + k^2/rate^2 = k(k+1)/rate^2
    e_sq_lo = k_lo * (k_lo + 1) / rate**2 if k_lo >= 1 else 0.0
    e_sq_hi = k_hi * (k_hi + 1) / rate**2

    e_s_sq = (1 - alpha_hi) * e_sq_lo + alpha_hi * e_sq_hi
    variance = e_s_sq - mean_S**2

    return {
        "mean": mean_S,
        "variance": float(variance),
        "tau": tau,
    }
