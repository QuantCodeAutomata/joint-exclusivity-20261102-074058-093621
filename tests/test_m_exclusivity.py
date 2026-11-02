"""
Tests for src/m_exclusivity.py — m-exclusivity hierarchy (Sections 3.4 and 4.1).

Paper properties tested:
  - Existence condition tau <= m-1 (Section 3.4).
  - Strongest admissible exclusivity level from wrap-around.
  - Wrap-around gives |A(X*)| <= m-1 almost surely when tau <= m-1.
  - Upper extremal count: kappa_m# = m-1 w.p. tau/(m-1), 0 otherwise.
  - Mean of kappa_m# equals tau.
  - Convex-order bounds: E[f(kappa*)] <= E[f(kappa)] <= E[f(kappa_m#)].
  - Threshold exceedance JE: no m simultaneous exceedances.
  - Constructed Z_tilde has the target marginals.
"""

import numpy as np
import pytest
from scipy.stats import expon

from src.m_exclusivity import (
    check_m_exclusive_existence,
    strongest_admissible_m,
    upper_extremal_count_distribution,
    sample_upper_extremal_count,
    convex_order_bounds_check,
    sample_threshold_exceedance_je,
    verify_threshold_exceedance_je,
    make_normal_subthreshold_sampler,
    make_normal_excess_sampler,
)
from src.je_construction import (
    sample_je_wraparound,
    compute_claim_counts,
    verify_je_property,
)


# ---------------------------------------------------------------------------
# Existence condition (Section 3.4)
# ---------------------------------------------------------------------------

class TestExistenceCondition:
    def test_admissible_when_tau_le_m_minus_1(self):
        """tau = 1.5 <= m-1 = 2 for m=3: admissible."""
        q = [0.3, 0.4, 0.35, 0.25, 0.2]  # tau = 1.5
        result = check_m_exclusive_existence(q, m=3)
        assert result["admissible"]
        assert abs(result["tau"] - 1.5) < 1e-10

    def test_inadmissible_when_tau_gt_m_minus_1(self):
        """tau = 2.5 > m-1 = 2 for m=3: inadmissible."""
        q = [0.5, 0.5, 0.5, 0.5, 0.5]  # tau = 2.5
        result = check_m_exclusive_existence(q, m=3)
        assert not result["admissible"]

    def test_boundary_tau_equals_m_minus_1(self):
        """tau = 2.0 = m-1 = 2 for m=3: admissible (equality is OK)."""
        q = [1.0, 0.5, 0.5]  # tau = 2.0
        result = check_m_exclusive_existence(q, m=3)
        assert result["admissible"]

    def test_je_is_n_exclusive(self):
        """JE (m=n) exists whenever tau <= n-1."""
        q = [0.3, 0.3, 0.3, 0.3]  # tau=1.2 <= 3 = n-1 for n=4
        result = check_m_exclusive_existence(q, m=4)
        assert result["admissible"]

    def test_mutual_exclusivity_is_2_exclusive(self):
        """Mutual exclusivity (m=2) requires tau <= 1."""
        q = [0.4, 0.4, 0.4]  # tau=1.2 > 1 → not mutually exclusive
        result = check_m_exclusive_existence(q, m=2)
        assert not result["admissible"]

        q_me = [0.3, 0.3, 0.4]  # tau=1.0 <= 1 → admissible
        result_me = check_m_exclusive_existence(q_me, m=2)
        assert result_me["admissible"]


class TestStrongestAdmissibleM:
    def test_tau_1_5_strongest_m_is_3(self):
        """tau=1.5 → ceil(tau+1) = ceil(2.5) = 3, i.e. smallest m=3."""
        q = [0.3, 0.4, 0.35, 0.25, 0.2]  # tau=1.5
        m = strongest_admissible_m(q)
        assert m == 3, f"Expected 3, got {m}"

    def test_tau_2_strongest_m_is_3(self):
        """tau=2.0 integer → m = tau + 1 = 3."""
        q = [1.0, 0.5, 0.5]  # tau=2.0
        m = strongest_admissible_m(q)
        assert m == 3

    def test_tau_0_9_strongest_m_is_2(self):
        """tau=0.9 → floor(0.9)+2=2: ME class is achievable."""
        q = [0.3, 0.3, 0.3]  # tau=0.9
        m = strongest_admissible_m(q)
        assert m == 2


# ---------------------------------------------------------------------------
# Wrap-around gives strongest admissible level (Section 3.4)
# ---------------------------------------------------------------------------

class TestWraparoundLevel:
    def test_wraparound_satisfies_m_exclusive(self, rng):
        """
        Wrap-around vector with tau=1.5 is at most 2-exclusive (|A| <= 2).
        Since m=3 is the smallest admissible m, |A(X*)| <= m-1 = 2 a.s.
        """
        q = [0.3, 0.4, 0.35, 0.25, 0.2]  # tau=1.5, m=3
        m = strongest_admissible_m(q)

        def make_exp_sampler(rate: float):
            return lambda n, r: r.exponential(1.0 / rate, size=n)

        samplers = [make_exp_sampler(1.0) for _ in q]
        samples = sample_je_wraparound(q, samplers, 50_000, rng)
        counts = compute_claim_counts(samples)

        assert counts.max() <= m - 1, (
            f"Wrap-around violated m-exclusive: max count = {counts.max()}, m-1 = {m-1}"
        )

    def test_wraparound_exactly_supported_on_floor_ceil_tau(self, rng):
        """
        Wrap-around claim count kappa* is supported on {floor(tau), ceil(tau)}.
        tau=1.5 → support on {1, 2}.
        """
        q = [0.3, 0.4, 0.35, 0.25, 0.2]  # tau=1.5
        tau = sum(q)
        floor_tau = int(np.floor(tau))
        ceil_tau = int(np.ceil(tau))

        def make_exp_sampler():
            return lambda n, r: r.exponential(1.0, size=n)

        samplers = [make_exp_sampler() for _ in q]
        samples = sample_je_wraparound(q, samplers, 100_000, rng)
        counts = compute_claim_counts(samples)

        observed_support = set(counts.tolist())
        expected_support = {floor_tau, ceil_tau}
        assert observed_support == expected_support, (
            f"Observed support {observed_support}, expected {expected_support}"
        )


# ---------------------------------------------------------------------------
# Upper extremal count (Section 3.4)
# ---------------------------------------------------------------------------

class TestUpperExtremalCount:
    def test_distribution_mean_equals_tau(self):
        """E[kappa_m#] = tau (Section 3.4 Proposition)."""
        tau = 1.5
        m = 3
        dist = upper_extremal_count_distribution(tau, m)
        assert abs(dist["mean"] - tau) < 1e-10

    def test_support_is_zero_and_m_minus_one(self):
        tau = 1.5
        m = 3
        dist = upper_extremal_count_distribution(tau, m)
        assert dist["kappa_values"] == [0, m - 1]

    def test_probabilities_sum_to_one(self):
        for tau, m in [(0.6, 2), (1.5, 3), (2.0, 4), (0.9, 2)]:
            dist = upper_extremal_count_distribution(tau, m)
            total = sum(dist["probabilities"])
            assert abs(total - 1.0) < 1e-10, f"tau={tau}, m={m}: probs={dist['probabilities']}"

    def test_variance_formula(self):
        """Var[kappa_m#] = tau*(m-1) - tau^2 = tau*(m-1-tau)."""
        tau, m = 1.5, 3
        dist = upper_extremal_count_distribution(tau, m)
        expected_var = tau * (m - 1) - tau**2  # = 1.5*2 - 1.5^2 = 3 - 2.25 = 0.75
        assert abs(dist["variance"] - expected_var) < 1e-10

    def test_sampling_mean_approx_tau(self, rng):
        tau, m = 1.5, 3
        samples = sample_upper_extremal_count(tau, m, 200_000, rng)
        assert abs(np.mean(samples) - tau) < 0.03


# ---------------------------------------------------------------------------
# Convex-order bounds (Section 3.4): kappa* <=_cx |A(X)| <=_cx kappa_m#
# ---------------------------------------------------------------------------

class TestConvexOrderBounds:
    def test_lower_bound_holds_for_all_test_functions(self, rng):
        """
        E[f(kappa*)] <= E[f(kappa_alt)] for convex f and kappa_alt from the
        independent structure (which is m-exclusive when tau <= m-1).
        """
        q = [0.3, 0.4, 0.35, 0.25, 0.2]  # tau=1.5, m=3

        def make_exp_sampler():
            return lambda n, r: r.exponential(1.0, size=n)

        samplers = [make_exp_sampler() for _ in q]
        samples_je = sample_je_wraparound(q, samplers, 100_000, rng)

        # Independent structure: same marginals, independent indicators
        n = len(q)
        q_arr = np.array(q)
        indicators_ind = rng.uniform(size=(100_000, n)) < q_arr[None, :]
        sev_ind = np.column_stack([
            make_exp_sampler()(100_000, rng) for _ in q
        ])
        samples_ind = indicators_ind.astype(float) * sev_ind

        result = convex_order_bounds_check(q, m=3, samples_je=samples_je, samples_alt=samples_ind)
        for fn, vals in result.items():
            assert vals["lower_bound_holds"], (
                f"Lower convex-order bound fails for f={fn}: "
                f"E[f(kappa*)]={vals['kappa_star']:.4f} > E[f(kappa_alt)]={vals['alternative']:.4f}"
            )

    def test_upper_bound_holds_for_all_test_functions(self, rng):
        """
        E[f(kappa_alt)] <= E[f(kappa_m#)] for convex f, where kappa_alt is m-exclusive.

        The upper convex-order bound (Section 3.4) applies only to m-exclusive
        alternatives.  We use a second draw from the wrap-around construction
        (which is 3-exclusive by construction when tau <= m-1 = 2) as a valid
        m-exclusive alternative distinct from the extremal kappa*.
        """
        q = [0.3, 0.4, 0.35, 0.25, 0.2]  # tau=1.5 <= 2 = m-1

        def make_exp_sampler():
            return lambda n, r: r.exponential(1.0, size=n)

        samplers = [make_exp_sampler() for _ in q]
        # Both draws are 3-exclusive (tau=1.5 <= m-1=2); a second independent draw
        # from the same wrap-around distribution is a valid non-trivial alternative.
        rng2 = np.random.default_rng(rng.integers(2**31))
        samples_je = sample_je_wraparound(q, samplers, 100_000, rng)
        samples_alt = sample_je_wraparound(q, [make_exp_sampler() for _ in q], 100_000, rng2)

        result = convex_order_bounds_check(q, m=3, samples_je=samples_je, samples_alt=samples_alt)
        for fn, vals in result.items():
            assert vals["upper_bound_holds"], (
                f"Upper convex-order bound fails for f={fn}: "
                f"E[f(kappa_alt)]={vals['alternative']:.4f} > E[f(kappa_m#)]={vals['kappa_upper']:.4f}"
            )


# ---------------------------------------------------------------------------
# Threshold exceedance exclusivity (Section 4.1)
# ---------------------------------------------------------------------------

class TestThresholdExceedanceJE:
    @pytest.fixture
    def normal_threshold_setup(self, rng):
        """
        Normal marginals with VaR thresholds.
        Z_i ~ N(0, 1), threshold l_i = 1.0, P(Z_i > 1) ≈ 0.159.
        n=3: tau ≈ 0.477 <= 1 = m-1 for m=2 (ME above threshold).
        """
        n = 3
        mu, sigma = 0.0, 1.0
        threshold = 1.0  # P(Z > 1) ≈ 0.1587 each, tau ≈ 0.476 <= 1

        from scipy.stats import norm as scipy_norm
        p = float(scipy_norm.sf(threshold))
        marginal_survival_fns = [lambda x: float(scipy_norm.sf(x))] * n
        thresholds = [threshold] * n
        excess_samplers = [make_normal_excess_sampler(mu, sigma, threshold)] * n
        subthreshold_samplers = [
            make_normal_subthreshold_sampler(mu, sigma, threshold)
        ] * n
        return {
            "marginal_survival_fns": marginal_survival_fns,
            "thresholds": thresholds,
            "excess_samplers": excess_samplers,
            "subthreshold_samplers": subthreshold_samplers,
            "p_each": p,
            "m": 2,
        }

    def test_no_simultaneous_exceedances_for_me(self, normal_threshold_setup, rng):
        """
        m=2: never two components simultaneously exceed threshold (mutual exclusivity above l).
        """
        s = normal_threshold_setup
        samples = sample_threshold_exceedance_je(
            s["marginal_survival_fns"], s["thresholds"],
            s["excess_samplers"], s["subthreshold_samplers"],
            m=s["m"], n_samples=50_000, rng=rng,
        )
        result = verify_threshold_exceedance_je(samples, s["thresholds"], m=s["m"])
        assert result["je_property_holds"], (
            f"ME above threshold violated: {result['violation_rate']:.2e}"
        )

    def test_marginals_approximately_preserved(self, normal_threshold_setup, rng):
        """
        Z_tilde_i should have the same marginal as Z_i ~ N(0,1), so
        P(Z_tilde_i > threshold) should equal p_i.
        """
        s = normal_threshold_setup
        samples = sample_threshold_exceedance_je(
            s["marginal_survival_fns"], s["thresholds"],
            s["excess_samplers"], s["subthreshold_samplers"],
            m=s["m"], n_samples=100_000, rng=rng,
        )
        n = len(s["thresholds"])
        for i in range(n):
            emp = np.mean(samples[:, i] > s["thresholds"][i])
            assert abs(emp - s["p_each"]) < 0.02, (
                f"Component {i}: empirical exceed prob={emp:.4f}, target={s['p_each']:.4f}"
            )

    def test_je_holds_for_m3(self, rng):
        """m=3 threshold exceedance: never three simultaneous exceedances."""
        from scipy.stats import norm as scipy_norm
        n = 4
        threshold = 0.5  # P(Z > 0.5) ≈ 0.3085; tau = 4*0.3085 ≈ 1.23 <= 2 for m=3
        p = float(scipy_norm.sf(threshold))
        tau = n * p
        assert tau <= 2.0 + 1e-6, f"tau={tau:.4f} > m-1=2, setup invalid"

        marginal_survival_fns = [lambda x: float(scipy_norm.sf(x))] * n
        thresholds = [threshold] * n
        excess_samplers = [make_normal_excess_sampler(0.0, 1.0, threshold)] * n
        sub_samplers = [make_normal_subthreshold_sampler(0.0, 1.0, threshold)] * n

        samples = sample_threshold_exceedance_je(
            marginal_survival_fns, thresholds, excess_samplers, sub_samplers,
            m=3, n_samples=50_000, rng=rng,
        )
        result = verify_threshold_exceedance_je(samples, thresholds, m=3)
        assert result["je_property_holds"], (
            f"3-exclusive above threshold violated: {result['violation_rate']:.2e}"
        )
