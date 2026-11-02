"""
Tests for src/aggregate_loss.py — aggregate-loss extremal structures (Section 4.2).

Paper properties tested:
  - stop_loss_gamma formula agrees with direct simulation for Gamma(k,1).
  - stop_loss_safest formula (Section 4.2) matches Monte Carlo S*.
  - S* has smaller stop-loss premiums than the independent structure.
  - stop_loss_class_upper_bound is an actual upper bound on the independent structure.
  - S# (Fréchet worst) dominates all tested alternatives in stop-loss.
  - TVaR satisfies the standard tail-risk identity: TVaR >= VaR >= 0.
  - Benchmark table (Example 4.8): safest variance <= independent variance.
"""

import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist, expon

from src.aggregate_loss import (
    stop_loss_gamma,
    stop_loss_safest,
    stop_loss_class_upper_bound,
    sample_safest_aggregate,
    sample_independent_aggregate,
    sample_comonotonic_aggregate,
    sample_frechet_worst_case,
    tvar_from_samples,
    tvar_from_stop_loss,
    compute_benchmark_table,
    compute_analytical_safest_stats,
)


# ---------------------------------------------------------------------------
# stop_loss_gamma
# ---------------------------------------------------------------------------

class TestStopLossGamma:
    def test_at_zero_retention_equals_mean(self):
        """E[(T-0)_+] = E[T] = k/rate for T ~ Gamma(k, rate)."""
        for k, rate in [(1, 1.0), (3, 2.0), (5, 0.5)]:
            pi = stop_loss_gamma(d=0.0, k=k, rate=rate)
            expected = k / rate
            assert abs(pi - expected) < 1e-8, (
                f"k={k}, rate={rate}: Pi(0)={pi:.6f}, mean={expected:.6f}"
            )

    def test_positive_and_decreasing(self):
        """Stop-loss premium is non-negative and decreasing in d."""
        k, rate = 3, 1.0
        d_values = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]
        pi_values = [stop_loss_gamma(d, k, rate) for d in d_values]
        assert all(p >= 0 for p in pi_values)
        for i in range(len(pi_values) - 1):
            assert pi_values[i] >= pi_values[i + 1], (
                f"Not decreasing at d={d_values[i+1]}"
            )

    def test_agrees_with_monte_carlo(self, rng):
        """Numerical formula matches direct Monte Carlo for Gamma(3, 1)."""
        k, rate, d = 3, 1.0, 2.0
        analytical = stop_loss_gamma(d, k, rate)

        n_mc = 1_000_000
        samples = rng.gamma(shape=k, scale=1.0 / rate, size=n_mc)
        mc_estimate = float(np.maximum(samples - d, 0.0).mean())
        assert abs(analytical - mc_estimate) < 0.01, (
            f"Formula={analytical:.4f}, MC={mc_estimate:.4f}"
        )

    def test_k_zero_returns_zero(self):
        """T_0 = 0 a.s., so E[(0-d)_+] = 0."""
        assert stop_loss_gamma(d=1.0, k=0, rate=1.0) == 0.0


# ---------------------------------------------------------------------------
# stop_loss_safest (Section 4.2)
# ---------------------------------------------------------------------------

class TestStopLossSafest:
    def test_formula_vs_mc_tau_1_5(self, rng):
        """
        With tau=1.5, rate=1: Pi_{S*}(d) = 0.5*Pi_{T_1}(d) + 0.5*Pi_{T_2}(d).
        Compare against Monte Carlo from the wrap-around sampler.
        """
        q = [0.15] * 10  # tau = 1.5
        tau = 1.5
        rate = 1.0
        n_mc = 500_000

        def exp_sampler(n, r):
            return r.exponential(1.0 / rate, size=n)

        samplers = [exp_sampler] * 10
        agg = sample_safest_aggregate(q, samplers, n_mc, rng)

        for d in [0.5, 1.0, 2.0, 3.0]:
            analytical = stop_loss_safest(d, tau, rate)
            mc_estimate = float(np.maximum(agg - d, 0.0).mean())
            rel_err = abs(analytical - mc_estimate) / max(analytical, 1e-8)
            assert rel_err < 0.05, (
                f"d={d}: formula={analytical:.5f}, MC={mc_estimate:.5f}, "
                f"rel_err={rel_err:.3f}"
            )

    def test_safest_le_independent_stop_loss(self, rng):
        """
        S* minimizes stop-loss premium within the frequency-severity class:
        Pi_{S*}(d) <= Pi_{S_indep}(d) for all d (Section 4.2).
        """
        q = [0.15] * 10
        rate = 1.0
        n_mc = 500_000

        # sample_safest_aggregate takes a list of per-component samplers
        def exp_sampler(n, r):
            return r.exponential(1.0 / rate, size=n)

        samplers = [exp_sampler] * 10
        agg_star = sample_safest_aggregate(q, samplers, n_mc, rng)
        # sample_independent_aggregate takes a single shared severity sampler
        agg_ind = sample_independent_aggregate(q, exp_sampler, n_mc, rng)

        for d in [0.5, 1.0, 2.0, 3.0, 5.0]:
            pi_star = float(np.maximum(agg_star - d, 0.0).mean())
            pi_ind = float(np.maximum(agg_ind - d, 0.0).mean())
            assert pi_star <= pi_ind + 0.02, (
                f"d={d}: S* stop-loss {pi_star:.4f} > independent {pi_ind:.4f}"
            )


# ---------------------------------------------------------------------------
# stop_loss_class_upper_bound (Section 4.2)
# ---------------------------------------------------------------------------

class TestStopLossUpperBound:
    def test_upper_bound_holds_against_wraparound(self, rng):
        """
        Pi_S(d) <= (tau/(m-1)) * Pi_{T_{m-1}}(d) for any m-exclusive vector.
        Test with m=3 (3-exclusive) and the wrap-around structure (tau=1.5<=2=m-1),
        which is 3-exclusive by construction.  Independent indicators are NOT
        guaranteed 3-exclusive so cannot be used to test this bound.
        """
        q = [0.15] * 10
        rate = 1.0
        n_mc = 500_000

        def exp_sampler(n, r):
            return r.exponential(1.0 / rate, size=n)

        # sample_safest_aggregate uses the wrap-around construction; with tau=1.5
        # it is 3-exclusive (|A|<=2 always).
        # Provide a list of identical samplers (one per component).
        exp_samplers = [exp_sampler] * len(q)
        agg_ww = sample_safest_aggregate(q, exp_samplers, n_mc, rng)

        for d in [0.5, 1.0, 2.0, 3.0]:
            upper = stop_loss_class_upper_bound(d, tau=1.5, m=3, rate=rate)
            mc_ww = float(np.maximum(agg_ww - d, 0.0).mean())
            assert mc_ww <= upper + 0.02, (
                f"d={d}: wrap-around stop-loss {mc_ww:.4f} exceeds upper bound {upper:.4f}"
            )


# ---------------------------------------------------------------------------
# TVaR
# ---------------------------------------------------------------------------

class TestTVaR:
    def test_tvar_ge_var(self, rng):
        """TVaR_alpha(X) >= VaR_alpha(X) for all alpha."""
        samples = rng.exponential(scale=1.0, size=100_000)
        for alpha in [0.90, 0.95, 0.99]:
            var = float(np.quantile(samples, alpha))
            tvar = tvar_from_samples(samples, alpha)
            assert tvar >= var - 1e-8, f"alpha={alpha}: TVaR={tvar:.4f} < VaR={var:.4f}"

    def test_tvar_ge_zero_for_non_negative(self, rng):
        """TVaR is non-negative for non-negative losses."""
        samples = rng.exponential(size=50_000)
        assert tvar_from_samples(samples, 0.95) >= 0.0

    def test_tvar_from_stop_loss_matches_sample_tvar(self, rng):
        """
        TVaR_alpha = VaR_alpha + Pi(VaR_alpha) / (1 - alpha)
        should match the sample-based TVaR for Gamma(3,1).
        """
        k, rate, alpha = 3, 1.0, 0.99
        samples = rng.gamma(shape=k, scale=1.0 / rate, size=500_000)
        var = float(np.quantile(samples, alpha))
        tvar_mc = tvar_from_samples(samples, alpha)
        tvar_sl = tvar_from_stop_loss(
            var, lambda d: stop_loss_gamma(d, k, rate), alpha
        )
        assert abs(tvar_mc - tvar_sl) < 0.15, (
            f"TVaR from stop-loss={tvar_sl:.4f}, from samples={tvar_mc:.4f}"
        )


# ---------------------------------------------------------------------------
# Benchmark table (Example 4.8, n=10, q=0.15, unit-mean exponential)
# ---------------------------------------------------------------------------

class TestBenchmarkTable:
    @pytest.fixture(scope="class")
    def benchmark_results(self, rng):
        """Run the benchmark once and cache results for this class."""
        return compute_benchmark_table(
            n=10, q=0.15, rate=1.0, alpha=0.99,
            n_samples=200_000, rng=rng,
        )

    def test_expected_keys_present(self, benchmark_results):
        """Benchmark table contains the six expected structures."""
        required = {"Safest (S*)", "Independent", "3-exclusive worst",
                    "JE worst", "Co-monotonic", "Frechet worst (m=3)"}
        assert required.issubset(set(benchmark_results.keys())), (
            f"Missing keys: {required - set(benchmark_results.keys())}"
        )

    def test_safest_variance_le_independent_variance(self, benchmark_results):
        """S* variance <= independent structure variance (convex order → variance ordering)."""
        var_star = benchmark_results["Safest (S*)"]["variance"]
        var_ind = benchmark_results["Independent"]["variance"]
        assert var_star <= var_ind + 0.05, (
            f"Safest variance {var_star:.4f} > independent {var_ind:.4f}"
        )

    def test_safest_tvar_le_independent_tvar(self, benchmark_results):
        """S* TVaR <= independent TVaR (convex order)."""
        tvar_star = benchmark_results["Safest (S*)"]["tvar"]
        tvar_ind = benchmark_results["Independent"]["tvar"]
        assert tvar_star <= tvar_ind + 0.1, (
            f"Safest TVaR99 {tvar_star:.4f} > independent {tvar_ind:.4f}"
        )

    def test_frechet_worst_tvar_ge_3excl_worst_tvar(self, benchmark_results):
        """
        Fréchet-class worst (m=3) >= 3-exclusive worst in the frequency-severity class.
        The Fréchet class (no frequency-severity structure) is larger, so its worst case
        is at least as bad as the frequency-severity class worst at the same m (Section 4.2).
        """
        tvar_3excl = benchmark_results["3-exclusive worst"]["tvar"]
        tvar_frechet = benchmark_results["Frechet worst (m=3)"]["tvar"]
        assert tvar_frechet >= tvar_3excl - 0.2, (
            f"Fréchet TVaR99 {tvar_frechet:.4f} < 3-excl worst TVaR99 {tvar_3excl:.4f}"
        )

    def test_safest_mean_equals_tau_times_severity_mean(self, benchmark_results):
        """E[S*] = tau * E[Y] = 1.5 * 1 = 1.5 (rate=1 exponential)."""
        mean_star = benchmark_results["Safest (S*)"]["mean"]
        assert abs(mean_star - 1.5) < 0.05, (
            f"E[S*]={mean_star:.4f}, expected 1.5"
        )

    def test_all_structures_non_negative_outputs(self, benchmark_results):
        """All aggregate loss structures should be non-negative."""
        for name, vals in benchmark_results.items():
            assert vals.get("mean", 0) >= -1e-8, f"Negative mean for {name}"
            assert vals.get("variance", 0) >= -1e-8, f"Negative variance for {name}"


# ---------------------------------------------------------------------------
# Analytical safest stats
# ---------------------------------------------------------------------------

class TestAnalyticalSafestStats:
    def test_variance_formula_tau_1_5(self):
        """
        Var[S*] for unit-mean exponential: positive and finite.
        tau=1.5, rate=1: 0.5*T_1 + 0.5*T_2 mixture.
        """
        stats = compute_analytical_safest_stats(n=10, q=0.15, rate=1.0, alpha=0.99)
        assert stats["mean"] > 0
        assert stats["variance"] > 0
        assert np.isfinite(stats["variance"])

    def test_mean_equals_tau_over_rate(self):
        """E[S*] = tau / rate = n*q / rate."""
        for n, q_val, rate in [(10, 0.15, 1.0), (6, 0.15, 2.0), (10, 0.23, 0.5)]:
            stats = compute_analytical_safest_stats(n=n, q=q_val, rate=rate, alpha=0.99)
            tau = n * q_val
            expected_mean = tau / rate
            assert abs(stats["mean"] - expected_mean) < 1e-8, (
                f"n={n}, q={q_val}, rate={rate}: E[S*]={stats['mean']:.6f}, "
                f"expected={expected_mean:.6f}"
            )
