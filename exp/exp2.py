"""
Experiment 2: Canonical JE family – trivariate Example 3.9 (Section 3.2).

Reproduces:
- Example 3.9: trivariate canonical JE with Uniform(0,1) positive parts.
- Correlation sweep over lambda in [1/3, 1/2] with three distinct face copulas.
- Verification of JE property, marginals, and pairwise correlations.
- Joint CDF comparison (empirical vs closed-form formula 3.10).

Run as: python -m exp.exp2
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, FrozenSet, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    EXP2_N,
    EXP2_Q,
    EXP2_LAMBDA_MIN,
    EXP2_LAMBDA_MAX,
    EXP2_N_LAMBDA,
    N_MC_SAMPLES,
    RANDOM_SEED,
    MARGINAL_TOLERANCE,
)
from src.canonical_je import (
    enumerate_faces,
    faces_containing,
    check_canonical_admissibility,
    IndependenceCopula,
    ComonotonicCopula,
    CountermonotonicCopula,
    sample_canonical_je,
    make_example39_marginals,
    make_example39_face_weights,
    make_example39_copulas,
    theoretical_correlations_example39,
    compute_empirical_correlations,
    canonical_je_cdf_survival,
)
from src.je_construction import verify_je_property

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Tolerance for empirical vs theoretical correlations (Monte Carlo noise)
CORRELATION_TOLERANCE = 0.03

# Tolerance for JE property (should be identically zero, allow tiny MC noise)
JE_TOLERANCE = 1e-4

# Lambda value chosen for CDF / marginal plots (midpoint of grid)
_LAM_CHOSEN_IDX = EXP2_N_LAMBDA // 2


def _face_survival_copulas_example39() -> Dict[FrozenSet[int], object]:
    """Analytical survival copula callables for Example 3.9."""
    return {
        frozenset([0, 1]): lambda u: u[0] * u[1],                       # independence
        frozenset([0, 2]): lambda u: min(u[0], u[1]),                    # comonotonic
        frozenset([1, 2]): lambda u: max(u[0] + u[1] - 1.0, 0.0),       # countermonotonic
    }


def _theoretical_joint_cdf(
    x_vals: np.ndarray,
    lam: float,
    q: float = 0.5,
) -> np.ndarray:
    """
    Compute the theoretical joint CDF P(X0<=x, X1<=x, X2<=x) on a diagonal grid.

    Uses `canonical_je_cdf_survival` with the formula from Section 3.2.
    """
    marg = make_example39_marginals(q=q)
    face_weights = make_example39_face_weights(lam)
    face_survival_copulas = _face_survival_copulas_example39()
    marginal_fns = marg["marginal_survival_fns"]

    cdfs = np.empty(len(x_vals))
    for k, x in enumerate(x_vals):
        cdfs[k] = canonical_je_cdf_survival(
            x=[x, x, x],
            q=marg["q_list"],
            face_weights=face_weights,
            face_survival_copulas=face_survival_copulas,
            marginal_survival_fns=marginal_fns,
        )
    return cdfs


def run_admissibility_check(lam_grid: np.ndarray) -> List[dict]:
    """Verify admissibility for each lambda in the grid."""
    results = []
    for lam in lam_grid:
        face_weights = make_example39_face_weights(lam)
        info = check_canonical_admissibility(EXP2_Q, face_weights)
        results.append({
            "lambda": float(lam),
            "admissible": bool(info["admissible"]),
            "lower_sum": float(info["lower_sum"]),
            "lower_bound": float(info["lower_bound"]),
        })
    return results


def run_correlation_sweep(
    lam_grid: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[List[dict], List[np.ndarray]]:
    """
    For each lambda, simulate the canonical JE model and compute:
    - JE property verification
    - Marginal survival function check
    - Empirical vs theoretical pairwise correlations

    Returns a list of result dicts and a list of sample arrays.
    """
    marg = make_example39_marginals(q=EXP2_Q[0])
    face_copulas = make_example39_copulas()

    results = []
    all_samples = []

    for k, lam in enumerate(lam_grid):
        face_weights = make_example39_face_weights(lam)

        samples = sample_canonical_je(
            q=marg["q_list"],
            face_weights=face_weights,
            face_copulas=face_copulas,
            severity_samplers=marg["severity_samplers"],
            n_samples=n_samples,
            rng=rng,
        )

        je_rate = verify_je_property(samples)
        emp_corr = compute_empirical_correlations(samples)
        theo_corr = theoretical_correlations_example39(lam)

        results.append({
            "lambda": float(lam),
            "je_rate": float(je_rate),
            "je_ok": bool(je_rate < JE_TOLERANCE),
            "empirical_corr_12": float(emp_corr["corr_12"]),
            "empirical_corr_13": float(emp_corr["corr_13"]),
            "empirical_corr_23": float(emp_corr["corr_23"]),
            "theoretical_corr_12": float(theo_corr["corr_12"]),
            "theoretical_corr_13": float(theo_corr["corr_13"]),
            "theoretical_corr_23": float(theo_corr["corr_23"]),
            "corr_12_err": float(abs(emp_corr["corr_12"] - theo_corr["corr_12"])),
            "corr_13_err": float(abs(emp_corr["corr_13"] - theo_corr["corr_13"])),
            "corr_23_err": float(abs(emp_corr["corr_23"] - theo_corr["corr_23"])),
            "corr_ok": bool(
                abs(emp_corr["corr_12"] - theo_corr["corr_12"]) < CORRELATION_TOLERANCE
                and abs(emp_corr["corr_13"] - theo_corr["corr_13"]) < CORRELATION_TOLERANCE
                and abs(emp_corr["corr_23"] - theo_corr["corr_23"]) < CORRELATION_TOLERANCE
            ),
        })
        all_samples.append(samples)

    return results, all_samples


def check_marginals(
    samples: np.ndarray,
    q: float = 0.5,
) -> dict:
    """
    Verify marginal survival P(Xi > x) = q*(1-x) for x in {0, 0.1, ..., 1}.

    Returns max absolute error and per-component errors.
    """
    x_grid = np.linspace(0.0, 1.0, 11)
    n = samples.shape[1]
    errors = []
    for i in range(n):
        emp = np.array([np.mean(samples[:, i] > x) for x in x_grid])
        theo = q * np.maximum(1.0 - x_grid, 0.0)
        errors.append(float(np.abs(emp - theo).max()))

    return {
        "x_grid": x_grid.tolist(),
        "max_error_per_component": errors,
        "max_error": float(max(errors)),
        "marginals_ok": bool(max(errors) < MARGINAL_TOLERANCE),
    }


def plot_correlations(
    lam_grid: np.ndarray,
    sweep_results: List[dict],
    output_path: Path,
) -> None:
    """Plot empirical vs theoretical pairwise correlations over lambda."""
    emp_12 = [r["empirical_corr_12"] for r in sweep_results]
    emp_13 = [r["empirical_corr_13"] for r in sweep_results]
    emp_23 = [r["empirical_corr_23"] for r in sweep_results]

    theo_12 = [r["theoretical_corr_12"] for r in sweep_results]
    theo_13 = [r["theoretical_corr_13"] for r in sweep_results]
    theo_23 = [r["theoretical_corr_23"] for r in sweep_results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    pairs = [
        ("Corr(X1,X2)", emp_12, theo_12, "steelblue"),
        ("Corr(X1,X3)", emp_13, theo_13, "tomato"),
        ("Corr(X2,X3)", emp_23, theo_23, "seagreen"),
    ]
    for ax, (title, emp, theo, color) in zip(axes, pairs):
        ax.plot(lam_grid, theo, color=color, lw=2, label="Theoretical")
        ax.scatter(lam_grid, emp, color=color, s=20, zorder=5, label="Empirical")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("Correlation")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Pairwise Correlations vs $\\lambda$ (Canonical JE, Example 3.9)\n"
        r"$C_{12}$=independence, $C_{13}$=comonotonic, $C_{23}$=countermonotonic"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_cdf(
    samples: np.ndarray,
    lam: float,
    output_path: Path,
) -> None:
    """
    Compare empirical joint CDF P(X0<=x, X1<=x, X2<=x) to theoretical formula.

    Also shows marginal CDFs for each component.
    """
    x_grid = np.linspace(0.0, 1.0, 50)

    # Joint CDF on diagonal
    emp_joint = np.array([np.mean(np.all(samples <= x, axis=1)) for x in x_grid])
    theo_joint = _theoretical_joint_cdf(x_grid, lam)

    # Marginal CDFs
    q = 0.5
    theo_marginal = 0.5 + 0.5 * x_grid  # P(Xi <= x) = 0.5 + 0.5*x for x in [0,1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: joint CDF
    ax = axes[0]
    ax.plot(x_grid, theo_joint, lw=2, color="steelblue", label="Theoretical (formula 3.10)")
    ax.plot(x_grid, emp_joint, lw=1.5, linestyle="--", color="tomato", label="Empirical")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$P(X_0 \leq x,\, X_1 \leq x,\, X_2 \leq x)$")
    ax.set_title(f"Joint CDF on Diagonal ($\\lambda={lam:.3f}$)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: marginal CDFs for each component
    ax = axes[1]
    colors = ["steelblue", "tomato", "seagreen"]
    for i, color in enumerate(colors):
        emp_marg = np.array([np.mean(samples[:, i] <= x) for x in x_grid])
        ax.plot(x_grid, theo_marginal, lw=2, color=color,
                label=f"Theoretical $X_{i+1}$" if i == 0 else "_")
        ax.plot(x_grid, emp_marg, lw=1.5, linestyle="--", color=color,
                label=f"Empirical $X_{i+1}$")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$P(X_i \leq x)$")
    ax.set_title(f"Marginal CDFs ($\\lambda={lam:.3f}$)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("CDF Verification: Empirical vs Closed-Form (Example 3.9)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def main(n_samples: int = N_MC_SAMPLES) -> None:
    """Run experiment 2 and save all results."""
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 60)
    print("Experiment 2: Canonical JE Family (Example 3.9, Section 3.2)")
    print("=" * 60)

    n = EXP2_N
    q = EXP2_Q[0]
    tau = sum(EXP2_Q)
    lam_grid = np.linspace(EXP2_LAMBDA_MIN, EXP2_LAMBDA_MAX, EXP2_N_LAMBDA)
    lam_chosen = float(lam_grid[_LAM_CHOSEN_IDX])

    print(f"\nParameters: n={n}, q_i={q}, tau={tau:.3f}, n-1={n-1}")
    print(f"Lambda grid: [{EXP2_LAMBDA_MIN:.4f}, {EXP2_LAMBDA_MAX:.4f}], {EXP2_N_LAMBDA} points")
    print(f"Face copulas: C_{{01}}=independence, C_{{02}}=comonotonic, C_{{12}}=countermonotonic")
    print(f"n_samples={n_samples:,}")

    # Part 1: Admissibility
    print("\nChecking admissibility for all lambda values...")
    adm_results = run_admissibility_check(lam_grid)
    all_admissible = all(r["admissible"] for r in adm_results)
    print(f"  All admissible: {all_admissible}")
    assert all_admissible, "Some lambda values are not admissible!"

    # Part 2: Correlation sweep over all lambda values
    print(f"\nRunning correlation sweep ({EXP2_N_LAMBDA} lambda values, "
          f"{n_samples:,} samples each)...")
    t0 = time.time()
    sweep_results, all_samples = run_correlation_sweep(lam_grid, n_samples, rng)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Print summary
    print(f"\n{'Lambda':>8} | {'Corr12_emp':>10} {'Corr12_th':>10} | "
          f"{'Corr13_emp':>10} {'Corr13_th':>10} | "
          f"{'Corr23_emp':>10} {'Corr23_th':>10} | {'JE_ok':>5}")
    print("-" * 100)
    for r in sweep_results:
        print(f"  {r['lambda']:.4f} | "
              f"{r['empirical_corr_12']:>10.4f} {r['theoretical_corr_12']:>10.4f} | "
              f"{r['empirical_corr_13']:>10.4f} {r['theoretical_corr_13']:>10.4f} | "
              f"{r['empirical_corr_23']:>10.4f} {r['theoretical_corr_23']:>10.4f} | "
              f"{str(r['je_ok']):>5}")

    # Part 3: Marginal verification at chosen lambda
    chosen_samples = all_samples[_LAM_CHOSEN_IDX]
    print(f"\nMarginal verification at lambda={lam_chosen:.4f}:")
    marg_check = check_marginals(chosen_samples, q=q)
    print(f"  Max marginal error: {marg_check['max_error']:.6f} "
          f"({'OK' if marg_check['marginals_ok'] else 'FAIL'})")
    for i, err in enumerate(marg_check["max_error_per_component"]):
        print(f"  Component {i+1} max error: {err:.6f}")

    # Part 4: Print JE property summary
    je_violations = [r["je_rate"] for r in sweep_results]
    max_je_violation = max(je_violations)
    all_je_ok = all(r["je_ok"] for r in sweep_results)
    print(f"\nJE property: max violation rate = {max_je_violation:.2e} "
          f"({'OK' if all_je_ok else 'FAIL'})")

    # Part 5: Print correlation errors
    max_corr_err = max(
        max(r["corr_12_err"], r["corr_13_err"], r["corr_23_err"])
        for r in sweep_results
    )
    all_corr_ok = all(r["corr_ok"] for r in sweep_results)
    print(f"\nCorrelation verification: max error = {max_corr_err:.6f} "
          f"({'OK' if all_corr_ok else 'FAIL'})")

    # Part 6: Plots
    plot_correlations(lam_grid, sweep_results, RESULTS_DIR / "exp2_correlations.png")
    plot_cdf(chosen_samples, lam_chosen, RESULTS_DIR / "exp2_cdf.png")
    print("\nPlots saved: results/exp2_correlations.png, results/exp2_cdf.png")

    # Part 7: Save JSON results
    output = {
        "parameters": {
            "n": n,
            "q_i": q,
            "tau": tau,
            "lambda_min": float(EXP2_LAMBDA_MIN),
            "lambda_max": float(EXP2_LAMBDA_MAX),
            "n_lambda": EXP2_N_LAMBDA,
            "n_samples": n_samples,
            "chosen_lambda": lam_chosen,
        },
        "admissibility": {"all_admissible": all_admissible, "details": adm_results},
        "correlation_sweep": sweep_results,
        "marginal_verification": {
            "lambda": lam_chosen,
            **marg_check,
        },
        "je_property": {
            "max_violation_rate": float(max_je_violation),
            "all_ok": all_je_ok,
        },
        "correlation_errors": {
            "max_error": float(max_corr_err),
            "all_ok": all_corr_ok,
            "tolerance": CORRELATION_TOLERANCE,
        },
    }
    with open(RESULTS_DIR / "exp2_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("Numerical results saved to results/exp2_results.json")

    # Assertions
    assert all_admissible, "Admissibility violated for some lambda."
    assert all_je_ok, (
        f"JE property violated: max rate = {max_je_violation:.2e}, "
        f"tolerance = {JE_TOLERANCE:.2e}"
    )
    assert marg_check["marginals_ok"], (
        f"Marginal mismatch: max error = {marg_check['max_error']:.6f}, "
        f"tolerance = {MARGINAL_TOLERANCE}"
    )
    assert all_corr_ok, (
        f"Correlation mismatch: max error = {max_corr_err:.6f}, "
        f"tolerance = {CORRELATION_TOLERANCE}"
    )

    print("\nAll assertions passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
