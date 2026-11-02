"""
Tests for src/generalised_je.py — G-JE family (Section 3.3).

Paper properties tested:
  - Capacity G_i* for linear truncation equals b - a = 3/8 (Example 3.13).
  - G-JE admissibility condition (formula 3.18).
  - G-JE sampler produces valid JE vectors with correct marginals.
  - Confinement: joint-face severities are confined to [0, 3/4) in Example 3.13.
  - Axis mass is positive and concentrated in the upper part of the support.
"""

import numpy as np
import pytest

from src.generalised_je import (
    compute_capacity,
    compute_linear_truncation_capacity,
    check_gje_admissibility,
    check_gje_face_weight_admissibility,
    make_linear_truncation,
    make_example313_distorted_sampler,
    make_example313_axis_sampler,
    sample_gje,
)
from src.canonical_je import (
    enumerate_faces,
    make_example39_copulas,
    make_example39_face_weights,
)


# ---------------------------------------------------------------------------
# Capacity computation (formula 3.14)
# ---------------------------------------------------------------------------

class TestComputeCapacity:
    def test_linear_truncation_capacity_matches_closed_form(self):
        """G_i* = b - a = 3/8 for linear truncation with a=1/8, b=1/2 (Example 3.13)."""
        a, b, q = 1.0 / 8.0, 1.0 / 2.0, 0.5

        def G_i(u: float) -> float:
            return float(make_linear_truncation(a, b)(u))

        def F_bar_i(x: float) -> float:
            return q * max(0.0, 1.0 - x) if x <= 1.0 else 0.0

        x_grid = np.linspace(0.0, 1.0 - 1e-6, 500)
        cap = compute_capacity(G_i, F_bar_i, x_grid)
        assert abs(cap - (b - a)) < 1e-3, (
            f"Numerical capacity {cap:.6f} should equal b-a={b-a:.6f}"
        )

    def test_analytical_capacity_example313(self):
        """Analytical formula: G_i* = b - a (Section 3.3)."""
        cap = compute_linear_truncation_capacity(a=1.0 / 8.0, b=1.0 / 2.0)
        assert abs(cap - 3.0 / 8.0) < 1e-12, f"Expected 3/8, got {cap}"

    def test_capacity_constant_distortion_is_inf(self):
        """If distortion is constant, capacity is unbounded (inf)."""
        def G_flat(u: float) -> float:
            return 0.0  # constant → no constraint

        def F_bar_i(x: float) -> float:
            return max(0.0, 1.0 - x)

        x_grid = np.linspace(0, 1, 200)
        cap = compute_capacity(G_flat, F_bar_i, x_grid)
        assert cap == float("inf") or cap > 1e10

    def test_capacity_non_negative(self):
        a, b, q = 1.0 / 8.0, 1.0 / 2.0, 0.5
        G_i = make_linear_truncation(a, b)
        F_bar_i = lambda x: q * max(0.0, 1.0 - x)
        x_grid = np.linspace(0, 1, 300)
        cap = compute_capacity(G_i, F_bar_i, x_grid)
        assert cap >= 0


class TestLinearTruncation:
    def test_zero_below_a(self):
        G = make_linear_truncation(a=0.125, b=0.5)
        assert G(0.0) == 0.0
        assert G(0.1) == 0.0
        assert abs(G(0.125)) < 1e-10

    def test_one_above_b(self):
        G = make_linear_truncation(a=0.125, b=0.5)
        assert G(0.5) == 1.0
        assert G(0.7) == 1.0
        assert G(1.0) == 1.0

    def test_linear_in_between(self):
        G = make_linear_truncation(a=0.125, b=0.5)
        # At u = 0.3 (= a + 0.175): G = (0.3 - 0.125) / (0.5 - 0.125) = 0.175/0.375
        expected = (0.3 - 0.125) / (0.5 - 0.125)
        assert abs(G(0.3) - expected) < 1e-10


# ---------------------------------------------------------------------------
# G-JE admissibility (formula 3.18 from Section 3.3)
# ---------------------------------------------------------------------------

class TestGJEAdmissibility:
    def test_example313_admissible(self):
        """
        Example 3.13: n=3, q_i=0.5, G_i*=3/8.
        Condition: tau - 1 = 0.5 <= min((1/2)*(9/8), (9/8)-(3/8)) = min(9/16, 3/4) = 9/16.
        """
        q = [0.5, 0.5, 0.5]
        caps = [3.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0]
        result = check_gje_admissibility(q, caps)
        assert result["admissible"], (
            f"Example 3.13 should be admissible; lhs={result['lhs']}, rhs={result['rhs']}"
        )
        assert abs(result["lhs"] - 0.5) < 1e-10

    def test_inadmissible_when_tau_too_large(self):
        """tau - 1 > min(rhs) → inadmissible."""
        q = [0.9, 0.9, 0.9]  # tau = 2.7, tau-1 = 1.7
        caps = [0.3, 0.3, 0.3]  # sum_caps = 0.9, rhs = min(0.45, 0.6) = 0.45
        result = check_gje_admissibility(q, caps)
        assert not result["admissible"]

    def test_admissible_with_large_caps(self):
        q = [0.3, 0.3, 0.3]  # tau-1 = -0.1 <= 0; always admissible
        caps = [0.3, 0.3, 0.3]
        result = check_gje_admissibility(q, caps)
        assert result["admissible"]


class TestGJEFaceWeightAdmissibility:
    def test_example313_lambda_min_admissible(self):
        """p_I(1/6) = 1/6*1/4 + 1/8 = 1/8; pi_i = 3/8 = G_i*."""
        q = [0.5, 0.5, 0.5]
        caps = [3.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0]
        lam = 1.0 / 6.0
        fw = {I: lam / 4.0 + 1.0 / 8.0 for I in [
            frozenset([0, 1]), frozenset([0, 2]), frozenset([1, 2])
        ]}
        result = check_gje_face_weight_admissibility(q, caps, fw)
        assert result["admissible"], (
            f"lambda=1/6 should be admissible; {result}"
        )

    def test_example313_lambda_max_admissible(self):
        """p_I(1/4) = 1/16 + 1/8 = 3/16; pi_i = 9/16 > G_i* = 3/8? No, 3*3/16 = 9/16..."""
        # Wait: pi_i = sum_{I in M_i} p_I = 2 * p_I (each i is in 2 pairs)
        # = 2 * (1/4 * 1/4 + 1/8) = 2 * 3/16 = 3/8 = G_i*
        q = [0.5, 0.5, 0.5]
        caps = [3.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0]
        lam = 1.0 / 4.0
        fw = {I: lam / 4.0 + 1.0 / 8.0 for I in [
            frozenset([0, 1]), frozenset([0, 2]), frozenset([1, 2])
        ]}
        result = check_gje_face_weight_admissibility(q, caps, fw)
        assert result["admissible"], f"lambda=1/4 should be admissible; {result}"


# ---------------------------------------------------------------------------
# G-JE sampling: JE property and marginals for Example 3.13
# ---------------------------------------------------------------------------

class TestGJESampling:
    @pytest.fixture
    def example313_setup(self):
        """Build all components for Example 3.13 (Section 3.3)."""
        q = [0.5, 0.5, 0.5]
        a, b = 1.0 / 8.0, 1.0 / 2.0
        cap = 3.0 / 8.0
        capacities = [cap, cap, cap]
        G = make_linear_truncation(a, b)
        distortions = [G, G, G]

        def F_bar_i(x):
            return 0.5 * max(0.0, 1.0 - x)

        marginal_survival_fns = [F_bar_i, F_bar_i, F_bar_i]

        lam = 0.2  # in [1/6, 1/4]
        face_weights = {I: lam / 4.0 + 1.0 / 8.0 for I in [
            frozenset([0, 1]), frozenset([0, 2]), frozenset([1, 2])
        ]}
        face_copulas = make_example39_copulas()

        pi = 2.0 * face_weights[frozenset([0, 1])]  # each i is in 2 pairs
        dist_samplers = [make_example313_distorted_sampler(a, b, 0.5)] * 3
        axis_samplers = [make_example313_axis_sampler(a, b, 0.5, pi)] * 3

        return {
            "q": q,
            "capacities": capacities,
            "distortions": distortions,
            "marginal_survival_fns": marginal_survival_fns,
            "face_weights": face_weights,
            "face_copulas": face_copulas,
            "dist_samplers": dist_samplers,
            "axis_samplers": axis_samplers,
        }

    def test_je_property(self, example313_setup, rng):
        """No sample should have all three components simultaneously positive."""
        s = example313_setup
        samples = sample_gje(
            s["q"], s["marginal_survival_fns"], s["distortions"],
            s["capacities"], s["face_weights"], s["face_copulas"],
            s["dist_samplers"], s["axis_samplers"],
            n_samples=50_000, rng=rng,
        )
        all_positive = np.all(samples > 0, axis=1)
        assert np.sum(all_positive) == 0, (
            f"JE violated: {np.sum(all_positive)} samples have all components positive"
        )

    def test_marginals_match_target_q(self, example313_setup, rng):
        s = example313_setup
        samples = sample_gje(
            s["q"], s["marginal_survival_fns"], s["distortions"],
            s["capacities"], s["face_weights"], s["face_copulas"],
            s["dist_samplers"], s["axis_samplers"],
            n_samples=100_000, rng=rng,
        )
        for i, qi in enumerate([0.5, 0.5, 0.5]):
            emp = np.mean(samples[:, i] > 0)
            assert abs(emp - qi) < 0.03, (
                f"Component {i}: empirical P(X>0)={emp:.4f}, target={qi}"
            )

    def test_non_negative_outputs(self, example313_setup, rng):
        s = example313_setup
        samples = sample_gje(
            s["q"], s["marginal_survival_fns"], s["distortions"],
            s["capacities"], s["face_weights"], s["face_copulas"],
            s["dist_samplers"], s["axis_samplers"],
            n_samples=5_000, rng=rng,
        )
        assert np.all(samples >= 0)

    def test_face_severity_confinement(self, example313_setup, rng):
        """
        Joint-face losses confined to [0, 3/4) per coordinate (Example 3.13, Section 3.3).
        """
        s = example313_setup
        samples = sample_gje(
            s["q"], s["marginal_survival_fns"], s["distortions"],
            s["capacities"], s["face_weights"], s["face_copulas"],
            s["dist_samplers"], s["axis_samplers"],
            n_samples=100_000, rng=rng,
        )
        # Identify samples on intermediate faces (exactly one active coordinate per column pair)
        # On faces: exactly 2 coordinates positive at once
        n_active = np.sum(samples > 0, axis=1)
        face_mask = n_active >= 2
        if np.sum(face_mask) > 0:
            face_samples = samples[face_mask]
            positive_vals = face_samples[face_samples > 0]
            confinement_upper = 3.0 / 4.0
            assert np.all(positive_vals < confinement_upper + 1e-6), (
                f"Face severity max = {positive_vals.max():.4f} exceeds {confinement_upper}"
            )


# ---------------------------------------------------------------------------
# Distorted sampler: support check
# ---------------------------------------------------------------------------

class TestDistortedSampler:
    def test_face_severity_in_zero_to_three_quarters(self, rng):
        """Distorted sampler for Example 3.13 maps U(0,1) to [0, 3/4)."""
        sampler = make_example313_distorted_sampler(a=0.125, b=0.5, q=0.5)
        u = rng.uniform(size=10_000)
        x = sampler(u, rng)
        assert np.all(x >= 0)
        assert np.all(x <= 0.75 + 1e-10), f"Max face severity = {x.max():.4f}"
        # Uniform on [0, 3/4]: mean should be ~3/8
        assert abs(np.mean(x) - 3.0 / 8.0) < 0.02
