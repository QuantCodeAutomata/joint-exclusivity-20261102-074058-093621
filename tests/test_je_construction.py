"""
Tests for src/je_construction.py — wrap-around JE construction (Section 3.1).

Paper properties tested:
  - Existence: JE with prescribed marginals exists iff tau <= n-1 (Theorem 3.2).
  - Arc membership uses modular arithmetic (Theorem 3.2 construction).
  - Marginal preservation: empirical P(X_i > x) ≈ bar{F}_i(x).
  - JE property: P(all X_i > 0) = 0 when admissible.
  - Claim-count support: kappa* takes values only in {floor(tau), ceil(tau)}.
  - Convex-order minimality: E[f(kappa*)] <= E[f(kappa)] for convex f.
"""

import numpy as np
import pytest

from src.je_construction import (
    compute_arcs,
    in_arc,
    sample_je_wraparound,
    verify_marginals,
    verify_je_property,
    compute_claim_counts,
    verify_claim_count_support,
    convex_order_comparison,
    make_exponential_sampler,
    make_uniform_truncated_sampler,
    marginal_survival_exponential,
    marginal_survival_zero_inflated_uniform,
    sample_alternative_activity_patterns,
)


# ---------------------------------------------------------------------------
# Arc construction
# ---------------------------------------------------------------------------

class TestComputeArcs:
    def test_arc_lengths_equal_q(self):
        q = [0.3, 0.4, 0.35, 0.25, 0.2]
        arcs = compute_arcs(q)
        for (start, length), qi in zip(arcs, q):
            assert abs(length - qi) < 1e-10, f"Arc length {length} != q_i {qi}"

    def test_arc_starts_are_cumulative_mod1(self):
        q = [0.3, 0.4, 0.35]
        arcs = compute_arcs(q)
        assert abs(arcs[0][0] - 0.0) < 1e-10
        assert abs(arcs[1][0] - 0.3) < 1e-10
        # third arc starts at 0.7; but 0.7+0.35=1.05 → wraps, start still 0.7
        assert abs(arcs[2][0] - 0.7) < 1e-10

    def test_zero_q_produces_zero_length_arc(self):
        q = [0.0, 0.5, 0.3]
        arcs = compute_arcs(q)
        assert arcs[0][1] == 0.0

    def test_single_component(self):
        q = [0.7]
        arcs = compute_arcs(q)
        assert len(arcs) == 1
        assert abs(arcs[0][0] - 0.0) < 1e-10
        assert abs(arcs[0][1] - 0.7) < 1e-10


class TestInArc:
    def test_simple_non_wrapping(self):
        # arc [0.2, 0.4) for start=0.2, length=0.2
        theta = np.array([0.0, 0.2, 0.3, 0.39, 0.4, 0.9])
        result = in_arc(theta, start=0.2, length=0.2)
        assert not result[0]
        assert result[1]
        assert result[2]
        assert result[3]
        assert not result[4]
        assert not result[5]

    def test_wrapping_arc(self):
        # start=0.8, length=0.4 → covers [0.8, 1.0) and [0.0, 0.2).
        # Avoid testing at the exact endpoint 0.2; use 0.21 (safely outside).
        theta = np.array([0.0, 0.1, 0.19, 0.21, 0.79, 0.8, 0.9, 0.99])
        result = in_arc(theta, start=0.8, length=0.4)
        assert result[0]        # 0.0 in arc
        assert result[1]        # 0.1 in arc
        assert result[2]        # 0.19 in arc
        assert not result[3]    # 0.21 not in arc (strictly outside right end 0.2)
        assert not result[4]    # 0.79 not in arc
        assert result[5]        # 0.8 in arc (left endpoint)
        assert result[6]        # 0.9 in arc
        assert result[7]        # 0.99 in arc

    def test_zero_length_arc(self):
        theta = np.array([0.0, 0.5, 1.0 - 1e-15])
        result = in_arc(theta, start=0.5, length=0.0)
        assert not any(result)

    def test_full_arc(self):
        theta = np.array([0.0, 0.5, 0.999])
        result = in_arc(theta, start=0.0, length=1.0)
        assert all(result)


# ---------------------------------------------------------------------------
# Existence condition (Theorem 3.2)
# ---------------------------------------------------------------------------

class TestExistenceCondition:
    """Verify that JE is possible iff tau <= n-1."""

    def test_admissible_case_no_je_violation(self, rng):
        q = [0.3, 0.4, 0.25, 0.2]  # tau = 1.15 <= 3
        rates = [1.0, 0.5, 1.5, 2.0]
        samplers = [make_exponential_sampler(r) for r in rates]
        samples = sample_je_wraparound(q, samplers, n_samples=50_000, rng=rng)
        je_rate = verify_je_property(samples)
        assert je_rate < 1e-3, f"JE violation rate {je_rate} should be ~0"

    def test_inadmissible_case_has_je_violation(self, rng):
        # tau = 4.3 > n-1 = 4; wrap-around still runs but is not JE
        q = [0.9, 0.9, 0.9, 0.9, 0.7]  # tau = 4.3, n = 5
        samplers = [make_exponential_sampler(1.0)] * 5
        samples = sample_je_wraparound(q, samplers, n_samples=50_000, rng=rng)
        je_rate = verify_je_property(samples)
        assert je_rate > 0.05, (
            f"Inadmissible case should have JE violations; got {je_rate}"
        )

    def test_boundary_tau_equals_n_minus_1(self, rng):
        # tau = 3 = n-1, exactly at boundary; JE should hold
        q = [0.75, 0.75, 0.75, 0.75]  # n=4, tau=3.0 = n-1
        samplers = [make_exponential_sampler(1.0)] * 4
        samples = sample_je_wraparound(q, samplers, n_samples=50_000, rng=rng)
        je_rate = verify_je_property(samples)
        assert je_rate < 1e-3, f"Boundary case JE rate {je_rate} should be ~0"


# ---------------------------------------------------------------------------
# Marginal preservation
# ---------------------------------------------------------------------------

class TestMarginalPreservation:
    def test_exponential_marginals_preserved(self, rng):
        q = [0.4, 0.6]
        rates = [1.0, 2.0]
        samplers = [make_exponential_sampler(r) for r in rates]
        samples = sample_je_wraparound(q, samplers, n_samples=100_000, rng=rng)

        survivals = [marginal_survival_exponential(r, qi) for r, qi in zip(rates, q)]
        x_grid = np.linspace(0, 3, 30)
        empirical, theoretical = verify_marginals(samples, survivals, x_grid)
        max_error = float(np.abs(empirical - theoretical).max())
        assert max_error < 0.02, f"Max marginal error {max_error:.4f} exceeds tolerance"

    def test_uniform_marginals_preserved(self, rng):
        q = [0.5, 0.5, 0.5]
        samplers = [make_uniform_truncated_sampler()] * 3
        samples = sample_je_wraparound(q, samplers, n_samples=100_000, rng=rng)

        survivals = [marginal_survival_zero_inflated_uniform(qi) for qi in q]
        x_grid = np.linspace(0, 1, 20)
        empirical, theoretical = verify_marginals(samples, survivals, x_grid)
        max_error = float(np.abs(empirical - theoretical).max())
        assert max_error < 0.02, f"Max marginal error {max_error:.4f} exceeds tolerance"

    def test_marginal_at_zero_matches_q(self, rng):
        """bar{F}_i(0) = q_i must match empirically."""
        q = [0.3, 0.6, 0.4]
        samplers = [make_exponential_sampler(1.0)] * 3
        samples = sample_je_wraparound(q, samplers, n_samples=100_000, rng=rng)
        for i, qi in enumerate(q):
            empirical = np.mean(samples[:, i] > 0)
            assert abs(empirical - qi) < 0.02, (
                f"Component {i}: empirical q = {empirical:.4f}, target = {qi}"
            )


# ---------------------------------------------------------------------------
# Claim-count support (Theorem 3.2)
# ---------------------------------------------------------------------------

class TestClaimCountSupport:
    def test_support_on_two_adjacent_integers(self, rng):
        q = [0.3, 0.4, 0.35, 0.25, 0.2]  # tau = 1.5
        samplers = [make_exponential_sampler(1.0)] * 5
        samples = sample_je_wraparound(q, samplers, n_samples=100_000, rng=rng)
        result = verify_claim_count_support(samples, tau=sum(q))
        assert result["support_correct"], (
            f"Claim count support {result['empirical_support']} != "
            f"expected {result['expected_support']}"
        )

    def test_kappa_mean_equals_tau(self, rng):
        q = [0.4, 0.5, 0.3]  # tau = 1.2
        samplers = [make_exponential_sampler(1.0)] * 3
        samples = sample_je_wraparound(q, samplers, n_samples=100_000, rng=rng)
        kappa = compute_claim_counts(samples)
        tau = sum(q)
        assert abs(np.mean(kappa) - tau) < 0.02, (
            f"E[kappa*] = {np.mean(kappa):.4f} but tau = {tau}"
        )

    def test_integer_tau_gives_single_support_point(self, rng):
        q = [0.5, 0.5]  # tau = 1.0; floor == ceil
        samplers = [make_exponential_sampler(1.0)] * 2
        samples = sample_je_wraparound(q, samplers, n_samples=50_000, rng=rng)
        kappa = compute_claim_counts(samples)
        assert set(kappa) == {1}, f"Expected support {{1}}, got {set(kappa)}"

    def test_kappa_variance_minimal(self, rng):
        """Claim count of wrap-around has smaller variance than alternatives."""
        q = [0.4, 0.5, 0.3]
        samplers = [make_exponential_sampler(1.0)] * 3
        samples_star = sample_je_wraparound(q, samplers, n_samples=80_000, rng=rng)
        kappa_star = compute_claim_counts(samples_star)

        # Independence coupling
        rng_ind = np.random.default_rng(99)
        n = len(q)
        q_arr = np.array(q)
        samples_ind = np.zeros((80_000, n))
        for i in range(n):
            active = rng_ind.random(80_000) < q_arr[i]
            sev = np.random.default_rng(i).exponential(1.0, 80_000)
            samples_ind[:, i] = np.where(active, sev, 0.0)
        kappa_ind = compute_claim_counts(samples_ind)

        assert np.var(kappa_star) < np.var(kappa_ind), (
            f"Var(kappa*) = {np.var(kappa_star):.4f} should be < "
            f"Var(kappa_ind) = {np.var(kappa_ind):.4f}"
        )


# ---------------------------------------------------------------------------
# Convex order minimality
# ---------------------------------------------------------------------------

class TestConvexOrderMinimality:
    def test_star_dominates_alternatives_convex_order(self, rng):
        """kappa* <=_cx kappa_alt for independence and co-monotonic alternatives."""
        q = [0.3, 0.4, 0.35, 0.25, 0.2]
        n, n_mc = len(q), 80_000
        q_arr = np.array(q)
        samplers = [make_exponential_sampler(1.0)] * n
        samples_star = sample_je_wraparound(q, samplers, n_samples=n_mc, rng=rng)
        sev = np.column_stack([rng.exponential(1.0, n_mc) for _ in q])

        # Independent alternative
        ind_B = (rng.uniform(size=(n_mc, n)) < q_arr).astype(float)
        samples_ind = ind_B * sev

        # Co-monotonic alternative: single U drives all indicators
        U_comono = rng.uniform(size=n_mc)
        comono_B = (U_comono[:, None] < q_arr[None, :]).astype(float)
        samples_comono = comono_B * sev

        test_fns = ["k2", "exp05", "exp1"]
        result = convex_order_comparison(samples_star, [samples_ind, samples_comono], test_fns)
        all_hold = all(result["star_dominates"][k] for k in ["alt_0", "alt_1"])
        assert all_hold, (
            "Wrap-around kappa* should dominate all alternatives in convex order; "
            f"dominates={result['star_dominates']}"
        )

    def test_star_has_smaller_second_moment_than_independence(self, rng):
        q = [0.4, 0.5, 0.3]
        samplers = [make_exponential_sampler(1.0)] * 3
        samples_star = sample_je_wraparound(q, samplers, n_samples=80_000, rng=rng)
        kappa_star = compute_claim_counts(samples_star)

        # Independence coupling
        n = len(q)
        rng2 = np.random.default_rng(999)
        samples_ind = np.zeros((80_000, n))
        for i in range(n):
            active = rng2.random(80_000) < q[i]
            samples_ind[:, i] = np.where(active, 1.0, 0.0)
        kappa_ind = compute_claim_counts(samples_ind)

        e_k2_star = np.mean(kappa_star ** 2)
        e_k2_ind = np.mean(kappa_ind ** 2)
        assert e_k2_star < e_k2_ind, (
            f"E[kappa*^2] = {e_k2_star:.4f} should be < E[kappa_ind^2] = {e_k2_ind:.4f}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_n_equals_2_admissible(self, rng):
        q = [0.3, 0.5]  # tau = 0.8 <= 1
        samplers = [make_exponential_sampler(1.0)] * 2
        samples = sample_je_wraparound(q, samplers, n_samples=30_000, rng=rng)
        je_rate = verify_je_property(samples)
        assert je_rate < 1e-3

    def test_all_zero_q_produces_zero_matrix(self, rng):
        q = [0.0, 0.0, 0.0]
        samplers = [make_exponential_sampler(1.0)] * 3
        samples = sample_je_wraparound(q, samplers, n_samples=1_000, rng=rng)
        assert np.all(samples == 0)

    def test_output_shape(self, rng):
        q = [0.2, 0.3, 0.4]
        samplers = [make_exponential_sampler(1.0)] * 3
        samples = sample_je_wraparound(q, samplers, n_samples=500, rng=rng)
        assert samples.shape == (500, 3)

    def test_samples_non_negative(self, rng):
        q = [0.3, 0.5, 0.2]
        samplers = [make_exponential_sampler(1.0)] * 3
        samples = sample_je_wraparound(q, samplers, n_samples=5_000, rng=rng)
        assert np.all(samples >= 0)
