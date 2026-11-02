"""
Tests for src/canonical_je.py — canonical JE family (Section 3.2).

Paper properties tested:
  - Admissibility polytope constraints (upper and lower bounds on face weights).
  - Canonical JE sampler produces valid JE vectors with correct marginals.
  - All-active event has zero probability for admissible weights.
  - Closed-form correlations for Example 3.9 match simulation.
  - Correlations vary linearly with lambda, spanning negative to positive values.
"""

import numpy as np
import pytest
from itertools import combinations

from src.canonical_je import (
    enumerate_faces,
    faces_containing,
    check_canonical_admissibility,
    sample_canonical_je,
    make_example39_marginals,
    make_example39_face_weights,
    make_example39_copulas,
    theoretical_correlations_example39,
    compute_empirical_correlations,
)


# ---------------------------------------------------------------------------
# Face enumeration
# ---------------------------------------------------------------------------

class TestEnumerateFaces:
    def test_n3_intermediate_faces_are_pairs(self):
        faces = enumerate_faces(3)
        # For n=3, M = all subsets of size 2 (n-1=2, n-1=2, so M={2<=|I|<=2})
        assert len(faces) == 3  # {0,1}, {0,2}, {1,2}
        for f in faces:
            assert 2 <= len(f) <= 2

    def test_n4_intermediate_faces(self):
        faces = enumerate_faces(4)
        # Pairs: C(4,2)=6, triples: C(4,3)=4 → 10 faces
        assert len(faces) == 10

    def test_n2_no_intermediate_faces(self):
        faces = enumerate_faces(2)
        # For n=2: 2<=|I|<=1 is empty
        assert len(faces) == 0


class TestFacesContaining:
    def test_component_0_in_pairs(self):
        faces = enumerate_faces(3)
        M0 = faces_containing(0, faces)
        for f in M0:
            assert 0 in f
        assert len(M0) == 2  # {0,1} and {0,2}

    def test_all_components_symmetric_in_example39(self):
        faces = enumerate_faces(3)
        for i in range(3):
            M_i = faces_containing(i, faces)
            assert len(M_i) == 2


# ---------------------------------------------------------------------------
# Admissibility
# ---------------------------------------------------------------------------

class TestCanonicalAdmissibility:
    def test_admissible_example39_lambda_min(self):
        q = [0.5, 0.5, 0.5]
        lam = 1.0 / 3.0
        fw = make_example39_face_weights(lam)
        result = check_canonical_admissibility(q, fw)
        assert result["admissible"], (
            f"lambda=1/3 should be admissible; lower_sum={result['lower_sum']}, "
            f"lower_bound={result['lower_bound']}"
        )

    def test_admissible_example39_lambda_max(self):
        q = [0.5, 0.5, 0.5]
        lam = 0.5
        fw = make_example39_face_weights(lam)
        result = check_canonical_admissibility(q, fw)
        assert result["admissible"], "lambda=1/2 should be admissible"

    def test_inadmissible_upper_violation(self):
        q = [0.3, 0.5, 0.5]
        # pi_0 = p_{01} + p_{02} = 0.3 > q_0 = 0.3 (just at boundary)
        # Set them above q_0
        fw = {frozenset([0, 1]): 0.2, frozenset([0, 2]): 0.2, frozenset([1, 2]): 0.1}
        result = check_canonical_admissibility(q, fw)
        # pi_0 = 0.4 > q_0 = 0.3
        assert not result["admissible"]
        assert len(result["upper_violations"]) > 0

    def test_inadmissible_lower_bound_violation(self):
        q = [0.5, 0.5, 0.5]  # tau=1.5, need lower_sum >= 0.5
        # All face weights zero → lower_sum=0 < 0.5
        fw = {frozenset([0, 1]): 0.0, frozenset([0, 2]): 0.0, frozenset([1, 2]): 0.0}
        result = check_canonical_admissibility(q, fw)
        assert not result["admissible"]
        assert not result["lower_ok"]


# ---------------------------------------------------------------------------
# Canonical JE sampling: JE property and marginals
# ---------------------------------------------------------------------------

class TestCanonicalJESampling:
    def test_je_property_all_positive_absent(self, rng):
        """No sample should have all three components positive (JE)."""
        q = [0.5, 0.5, 0.5]
        lam = 0.4
        fw = make_example39_face_weights(lam)
        copulas = make_example39_copulas()
        marg = make_example39_marginals(q=0.5)
        samplers = marg["severity_samplers"]

        samples = sample_canonical_je(q, fw, copulas, samplers, n_samples=50_000, rng=rng)
        all_positive = np.all(samples > 0, axis=1)
        assert np.sum(all_positive) == 0, (
            f"JE violated: {np.sum(all_positive)} samples have all components positive"
        )

    def test_marginals_match_target(self, rng):
        """P(X_i > 0) should equal q_i = 0.5 for each component."""
        q = [0.5, 0.5, 0.5]
        lam = 0.4
        fw = make_example39_face_weights(lam)
        copulas = make_example39_copulas()
        marg = make_example39_marginals(q=0.5)
        samplers = marg["severity_samplers"]

        samples = sample_canonical_je(q, fw, copulas, samplers, n_samples=100_000, rng=rng)
        for i, qi in enumerate(q):
            emp = np.mean(samples[:, i] > 0)
            assert abs(emp - qi) < 0.02, (
                f"Component {i}: empirical P(X>0)={emp:.4f}, target={qi}"
            )

    def test_uniform_severity_conditional_on_active(self, rng):
        """Positive severities should be approximately Uniform(0,1)."""
        q = [0.5, 0.5, 0.5]
        lam = 0.4
        fw = make_example39_face_weights(lam)
        copulas = make_example39_copulas()
        marg = make_example39_marginals(q=0.5)
        samplers = marg["severity_samplers"]

        samples = sample_canonical_je(q, fw, copulas, samplers, n_samples=100_000, rng=rng)
        for i in range(3):
            pos = samples[:, i][samples[:, i] > 0]
            # Median of U(0,1) is 0.5; check within reasonable range
            assert abs(np.median(pos) - 0.5) < 0.05, (
                f"Component {i}: median of positive part = {np.median(pos):.4f}, expected ~0.5"
            )

    def test_output_non_negative(self, rng):
        q = [0.5, 0.5, 0.5]
        lam = 0.4
        fw = make_example39_face_weights(lam)
        copulas = make_example39_copulas()
        marg = make_example39_marginals(q=0.5)
        samples = sample_canonical_je(q, fw, copulas, marg["severity_samplers"],
                                      n_samples=10_000, rng=rng)
        assert np.all(samples >= 0)


# ---------------------------------------------------------------------------
# Example 3.9: correlation formulas (Section 3.2)
# ---------------------------------------------------------------------------

class TestExample39Correlations:
    """
    From the paper (Section 3.2, Example 3.9):
      Corr(X1, X2) = (6λ - 3) / 5
      Corr(X1, X3) = (8λ - 3) / 5
      Corr(X2, X3) = (4λ - 3) / 5
    """

    @pytest.mark.parametrize("lam", [1.0 / 3.0, 0.4, 0.5])
    def test_correlations_match_theory(self, lam, rng):
        q = [0.5, 0.5, 0.5]
        fw = make_example39_face_weights(lam)
        copulas = make_example39_copulas()
        marg = make_example39_marginals(q=0.5)
        samples = sample_canonical_je(
            q, fw, copulas, marg["severity_samplers"], n_samples=200_000, rng=rng
        )

        emp = compute_empirical_correlations(samples)
        theo = theoretical_correlations_example39(lam)

        tol = 0.05  # Monte Carlo tolerance
        for key in ["corr_12", "corr_13", "corr_23"]:
            assert abs(emp[key] - theo[key]) < tol, (
                f"λ={lam:.3f}, {key}: empirical={emp[key]:.4f}, "
                f"theoretical={theo[key]:.4f}"
            )

    def test_correlation_signs_at_lambda_min(self):
        """At lambda=1/3: Corr_12 = (2-3)/5 = -0.2, Corr_13=(8/3-3)/5 < 0, Corr_23<0."""
        lam = 1.0 / 3.0
        theo = theoretical_correlations_example39(lam)
        assert theo["corr_12"] < 0
        assert theo["corr_13"] < 0
        assert theo["corr_23"] < 0

    def test_correlation_linear_in_lambda(self):
        """Correlations increase linearly with lambda."""
        lam1, lam2 = 1.0 / 3.0, 0.5
        t1 = theoretical_correlations_example39(lam1)
        t2 = theoretical_correlations_example39(lam2)
        for key in ["corr_12", "corr_13", "corr_23"]:
            assert t2[key] > t1[key], (
                f"{key} should increase from lambda={lam1} to lambda={lam2}"
            )

    def test_positive_correlation_corr13_at_lambda_max(self):
        """At lambda=1/2: Corr(X1,X3) = (8*0.5-3)/5 = 1/5 > 0."""
        theo = theoretical_correlations_example39(0.5)
        assert theo["corr_13"] > 0
