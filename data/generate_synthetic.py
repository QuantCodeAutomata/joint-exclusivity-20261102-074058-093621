"""
Synthetic marginal generators for all Joint Exclusivity experiments.

Provides factory functions that return the sampler callables and survival functions
needed by src/ modules.  All randomness is passed in via numpy Generator objects;
no data is loaded from disk or external APIs.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from src.config import (
    EXP1_RATES,
    EXP1_Q,
    EXP2_Q,
    EXP3_Q,
    EXP3_A,
    EXP3_B,
    EXP5_Q,
    EXP5_SEVERITY_RATE,
)
from src.je_construction import (
    make_exponential_sampler,
    make_uniform_truncated_sampler,
    marginal_survival_exponential,
    marginal_survival_zero_inflated_uniform,
)


# ---------------------------------------------------------------------------
# Zero-inflated exponential marginals (used in exp_1)
# ---------------------------------------------------------------------------

def make_zero_inflated_exponential_marginals(
    q: Sequence[float],
    rates: Sequence[float],
) -> dict:
    """
    Build marginals for zero-inflated exponential distributions.

    Each X_i = 0 with prob 1 - q_i, and Exponential(rate_i) with prob q_i.

    Parameters
    ----------
    q : sequence of float
    rates : sequence of float

    Returns
    -------
    dict with keys:
        'q_list': list of float
        'severity_samplers': list of callables (n, rng) -> ndarray
        'marginal_survival_fns': list of callables x_array -> ndarray
    """
    n = len(q)
    severity_samplers = [make_exponential_sampler(rates[i]) for i in range(n)]
    survival_fns = [
        marginal_survival_exponential(rates[i], q[i]) for i in range(n)
    ]
    return {
        "q_list": list(q),
        "severity_samplers": severity_samplers,
        "marginal_survival_fns": survival_fns,
    }


# ---------------------------------------------------------------------------
# Zero-inflated uniform marginals (used in exp_2, exp_3, exp_4 partial)
# ---------------------------------------------------------------------------

def make_zero_inflated_uniform_marginals(
    q: Sequence[float],
) -> dict:
    """
    Build marginals for Example 3.9 (Section 3.2): X_i=0 w.p. 1-q_i, Uniform(0,1) w.p. q_i.

    The zero-truncated positive part is simply Uniform(0,1).
    Severity sampler takes u ~ U(0,1) (from copula) and returns u directly.

    Parameters
    ----------
    q : sequence of float

    Returns
    -------
    dict with keys:
        'q_list': list of float
        'severity_samplers': list of callables (u, rng) -> ndarray
            These map copula uniforms u to positive severities.
        'severity_samplers_by_count': list of callables (n, rng) -> ndarray
            These draw n i.i.d. samples for the wrap-around construction.
        'marginal_survival_fns': list of callables ndarray -> ndarray
    """
    n = len(q)

    def make_sev_by_count(qi: float) -> Callable[[int, np.random.Generator], np.ndarray]:
        def sampler(n_: int, rng: np.random.Generator) -> np.ndarray:
            return rng.uniform(0.0, 1.0, size=n_)
        return sampler

    def make_sev_from_copula(qi: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
        # For Uniform(0,1) positive part: quantile is identity
        def sampler(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            return np.asarray(u, dtype=float)
        return sampler

    severity_samplers_count = [make_sev_by_count(q[i]) for i in range(n)]
    severity_samplers_copula = [make_sev_from_copula(q[i]) for i in range(n)]
    survival_fns = [marginal_survival_zero_inflated_uniform(q[i]) for i in range(n)]

    return {
        "q_list": list(q),
        "severity_samplers": severity_samplers_count,  # for wrap-around (n, rng)
        "severity_samplers_copula": severity_samplers_copula,  # for canonical JE (u, rng)
        "marginal_survival_fns": survival_fns,
    }


# ---------------------------------------------------------------------------
# Common exponential severity (used in exp_5)
# ---------------------------------------------------------------------------

def make_common_exponential_marginals(
    q_list: Sequence[float],
    rate: float,
) -> dict:
    """
    Build marginals for the frequency-severity model of Example 4.8:
    each X_i has marginal bar{F}_i(x) = q_i * exp(-rate * x).

    The positive severity is Exponential(rate) for all components.

    Parameters
    ----------
    q_list : sequence of float
    rate : float

    Returns
    -------
    dict
    """
    n = len(q_list)
    severity_sampler = make_exponential_sampler(rate)
    survival_fns = [
        marginal_survival_exponential(rate, q_list[i]) for i in range(n)
    ]
    return {
        "q_list": list(q_list),
        "severity_sampler": severity_sampler,
        "severity_samplers": [severity_sampler] * n,
        "marginal_survival_fns": survival_fns,
    }


# ---------------------------------------------------------------------------
# Normal marginals (used in exp_4 threshold exceedance illustration)
# ---------------------------------------------------------------------------

def make_normal_marginals(
    n: int,
    mu_list: Sequence[float],
    sigma_list: Sequence[float],
) -> dict:
    """
    Build standard normal marginals for threshold exceedance experiment.

    Parameters
    ----------
    n : int
    mu_list : sequence of float
    sigma_list : sequence of float

    Returns
    -------
    dict
    """
    from scipy.stats import norm

    survival_fns = []
    for i in range(n):
        mu, sigma = mu_list[i], sigma_list[i]
        def make_surv(m_, s_):
            def surv(x: float) -> float:
                return float(norm.sf(x, loc=m_, scale=s_))
            return surv
        survival_fns.append(make_surv(mu, sigma))

    return {
        "n": n,
        "mu_list": list(mu_list),
        "sigma_list": list(sigma_list),
        "marginal_survival_fns": survival_fns,
    }


# ---------------------------------------------------------------------------
# Convenience: default setups for each experiment
# ---------------------------------------------------------------------------

def default_exp1_marginals() -> dict:
    """Return default marginals for experiment 1."""
    return make_zero_inflated_exponential_marginals(EXP1_Q, EXP1_RATES)


def default_exp2_marginals() -> dict:
    """Return default marginals for experiment 2 (Example 3.9)."""
    return make_zero_inflated_uniform_marginals(EXP2_Q)


def default_exp3_marginals() -> dict:
    """Return default marginals for experiment 3 (Example 3.13)."""
    return make_zero_inflated_uniform_marginals(EXP3_Q)


def default_exp5_marginals() -> dict:
    """Return default marginals for experiment 5 (Example 4.8)."""
    q_list = [EXP5_Q] * 10
    return make_common_exponential_marginals(q_list, EXP5_SEVERITY_RATE)
