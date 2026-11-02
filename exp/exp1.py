"""
Experiment 1: Existence and wrap-around construction of jointly exclusive random vectors.

Reproduces the following from Section 3.1 of the paper:
- Theorem 3.2: JE vectors exist if and only if tau = sum q_i <= n - 1.
- Explicit wrap-around construction X_i* = Y_i * 1{Theta in R_i}.
- Verification that the construction preserves marginals.
- Verification of the JE property (no all-positive event).
- Verification that kappa* = |A(X*)| is supported on {floor(tau), ceil(tau)}.
- Empirical confirmation that kappa* is minimal in convex order.

Run as: python -m exp.exp1
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    EXP1_N,
    EXP1_Q,
    EXP1_RATES,
    EXP1_X_GRID_POINTS,
    EXP1_JE_TOLERANCE,
    EXP1_N_ALTERNATIVES,
    N_MC_SAMPLES,
    RANDOM_SEED,
    MARGINAL_TOLERANCE,
)
from src.je_construction import (
    compute_arcs,
    sample_je_wraparound,
    verify_marginals,
    verify_je_property,
    verify_claim_count_support,
    convex_order_comparison,
    make_exponential_sampler,
    marginal_survival_exponential,
    sample_alternative_activity_patterns,
)
from data.generate_synthetic import default_exp1_marginals

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_admissibility_scan(n: int) -> dict:
    """
    Scan tau vs n-1 to illustrate the existence boundary (Section 3.1).

    Tests three configurations with uniform q_i = tau/n:
      1. Admissible (tau = n-1 - 0.5)
      2. Boundary (tau = n-1, JE is just possible)
      3. Inadmissible (tau = n-1 + 0.3)
    """
    results = []
    for offset, label in [(-0.5, "admissible"), (0.0, "boundary"), (0.3, "inadmissible")]:
        tau_target = (n - 1) + offset
        q_uniform = tau_target / n
        q_scaled = [q_uniform] * n
        tau_scaled = sum(q_scaled)
        admissible = tau_scaled <= n - 1 + 1e-9
        results.append({
            "label": label,
            "tau": round(tau_scaled, 4),
            "n_minus_1": n - 1,
            "admissible": admissible,
        })
    return {"admissibility_scan": results}


def run_wrap_around_verification(
    q_list: list,
    rates: list,
    n_samples: int,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate the wrap-around construction and verify all properties.
    """
    severity_samplers = [make_exponential_sampler(r) for r in rates]
    survival_fns = [marginal_survival_exponential(rates[i], q_list[i]) for i in range(len(q_list))]

    # Simulate
    t0 = time.time()
    samples = sample_je_wraparound(q_list, severity_samplers, n_samples, rng)
    elapsed = time.time() - t0

    n = len(q_list)
    tau = sum(q_list)

    # Verify marginals on a grid
    x_grid = np.linspace(0.0, 3.0, EXP1_X_GRID_POINTS)
    empirical, theoretical = verify_marginals(samples, survival_fns, x_grid)
    max_marginal_error = float(np.abs(empirical - theoretical).max())

    # Verify JE property
    je_violation_rate = verify_je_property(samples)

    # Verify claim count support
    count_results = verify_claim_count_support(samples, tau)

    # Arcs
    arcs = compute_arcs(q_list)

    return {
        "n": n,
        "q_list": q_list,
        "tau": tau,
        "arcs": [(round(a, 6), round(b, 6)) for a, b in arcs],
        "admissible": tau <= n - 1,
        "n_samples": n_samples,
        "simulation_time_s": round(elapsed, 2),
        "max_marginal_error": round(max_marginal_error, 6),
        "marginal_ok": max_marginal_error < MARGINAL_TOLERANCE,
        "je_violation_rate": round(je_violation_rate, 8),
        "je_ok": je_violation_rate < EXP1_JE_TOLERANCE,
        "kappa_support_correct": count_results["support_correct"],
        "kappa_distribution": count_results["kappa_distribution"],
        "kappa_mean": round(count_results["kappa_mean"], 6),
        "kappa_var": round(count_results["kappa_var"], 6),
        "expected_support": str(count_results["expected_support"]),
        "empirical_support": str(count_results["empirical_support"]),
    }, samples


def run_convex_order_comparison(
    q_list: list,
    rates: list,
    samples_star: np.ndarray,
    n_alternatives: int,
    rng: np.random.Generator,
) -> dict:
    """
    Compare kappa* against alternative dependence structures in convex order.
    """
    severity_samplers = [make_exponential_sampler(r) for r in rates]
    q_arr = np.asarray(q_list)
    n_samples = len(samples_star)

    alt_samples_list = []
    for k in range(n_alternatives):
        # Independent Bernoulli indicators with same marginals
        B = sample_alternative_activity_patterns(q_arr, n_samples, rng, method="independent")
        Y = np.column_stack([
            severity_samplers[i](n_samples, rng) for i in range(len(q_list))
        ])
        alt = B * Y
        alt_samples_list.append(alt)

    # Also add co-monotonic
    B_comono = sample_alternative_activity_patterns(q_arr, n_samples, rng, method="comonotonic")
    Y_comono = np.column_stack([
        severity_samplers[i](n_samples, rng) for i in range(len(q_list))
    ])
    alt_samples_list.append(B_comono * Y_comono)

    test_fns = ["k2", "k3", "exp05", "exp1", "exp2"]
    co_results = convex_order_comparison(samples_star, alt_samples_list, test_fns)

    return co_results


def plot_marginal_verification(
    q_list: list,
    rates: list,
    samples: np.ndarray,
    survival_fns: list,
    output_path: Path,
) -> None:
    """Plot empirical vs theoretical marginal survival functions."""
    n = len(q_list)
    x_grid = np.linspace(0.0, 4.0, 100)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        emp = np.array([np.mean(samples[:, i] > x) for x in x_grid])
        theo = survival_fns[i](x_grid)
        ax.plot(x_grid, theo, label="Theoretical", lw=2, color="steelblue")
        ax.plot(x_grid, emp, label="Empirical", lw=1.5, linestyle="--", color="tomato")
        ax.set_title(f"Component {i+1}\n(q={q_list[i]:.2f}, rate={rates[i]:.1f})")
        ax.set_xlabel("x")
        ax.set_ylabel("P(X > x)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Marginal Survival Functions: Empirical vs Theoretical\n(Wrap-Around JE Construction)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_claim_count_distribution(
    count_dist: dict,
    tau: float,
    output_path: Path,
) -> None:
    """Plot the empirical distribution of kappa*."""
    counts = sorted(count_dist.keys())
    probs = [count_dist[k] for k in counts]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts, probs, color="steelblue", alpha=0.8, edgecolor="navy")

    # Mark floor and ceil of tau
    ax.axvline(x=np.floor(tau), color="tomato", linestyle="--", lw=2,
               label=f"floor(tau)={int(np.floor(tau))}")
    ax.axvline(x=np.ceil(tau), color="green", linestyle="--", lw=2,
               label=f"ceil(tau)={int(np.ceil(tau))}")

    ax.set_xlabel("kappa* = Number of positive components")
    ax.set_ylabel("Probability")
    ax.set_title(f"Claim Count Distribution (tau={tau:.3f})\nSupport on {{floor, ceil}}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def main(n_samples: int = N_MC_SAMPLES) -> None:
    """Run experiment 1 and save all results."""
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 60)
    print("Experiment 1: Wrap-around JE Construction")
    print("=" * 60)

    q_list = EXP1_Q
    rates = EXP1_RATES
    n = EXP1_N

    # Part 1: Admissibility scan
    adm = run_admissibility_scan(n)
    print("\nAdmissibility scan:")
    for row in adm["admissibility_scan"]:
        print(f"  {row['label']:15s}: tau={row['tau']:.3f}, n-1={row['n_minus_1']}, "
              f"admissible={row['admissible']}")

    # Part 2: Wrap-around verification
    print(f"\nRunning wrap-around construction (n={n}, n_samples={n_samples:,})...")
    wrap_results, samples = run_wrap_around_verification(q_list, rates, n_samples, rng)
    tau = wrap_results["tau"]

    print(f"  tau = {tau:.4f}, n-1 = {n-1}, admissible: {wrap_results['admissible']}")
    print(f"  Max marginal error:   {wrap_results['max_marginal_error']:.6f} "
          f"({'OK' if wrap_results['marginal_ok'] else 'FAIL'})")
    print(f"  JE violation rate:    {wrap_results['je_violation_rate']:.2e} "
          f"({'OK' if wrap_results['je_ok'] else 'FAIL'})")
    print(f"  kappa* support OK:    {wrap_results['kappa_support_correct']}")
    print(f"  kappa* distribution:  {wrap_results['kappa_distribution']}")
    print(f"  kappa* mean:          {wrap_results['kappa_mean']:.4f} (expected {tau:.4f})")
    print(f"  kappa* variance:      {wrap_results['kappa_var']:.4f}")

    # Part 3: Convex order comparison
    print(f"\nConvex order comparison vs {EXP1_N_ALTERNATIVES} alternative structures...")
    conv_results = run_convex_order_comparison(q_list, rates, samples, EXP1_N_ALTERNATIVES, rng)

    all_dominated = all(conv_results["star_dominates"].values())
    print(f"  kappa* dominates all alternatives in convex order: {all_dominated}")
    for fn in ["k2", "k3", "exp1"]:
        e_star = conv_results["star"].get(fn, float("nan"))
        vals = [conv_results[f"alt_{k}"][fn] for k in range(EXP1_N_ALTERNATIVES)]
        print(f"  E[{fn}(kappa*)]: star={e_star:.4f}, "
              f"alt range=[{min(vals):.4f}, {max(vals):.4f}]")

    # Part 4: Non-admissible case (tau > n-1): use uniform q_i = (n-1+0.3)/n
    print("\nNon-admissible case (tau > n-1):")
    tau_na = (n - 1) + 0.3  # 4.3 for n=5
    q_large = [tau_na / n] * n  # uniform, tau = n-1 + 0.3 > n-1
    tau_large = sum(q_large)
    print(f"  tau_large = {tau_large:.4f} > {n-1} = n-1 (inadmissible)")
    sev_large = [make_exponential_sampler(1.0)] * n  # unit exponential
    samples_large = sample_je_wraparound(q_large, sev_large, n_samples, rng)
    je_rate_large = verify_je_property(samples_large)
    expected_rate = max(0.0, tau_large - (n - 1))
    print(f"  JE violation rate: {je_rate_large:.4f} (expected ~{expected_rate:.4f})")

    # Plots
    survival_fns = [marginal_survival_exponential(rates[i], q_list[i]) for i in range(n)]
    plot_marginal_verification(
        q_list, rates, samples, survival_fns,
        RESULTS_DIR / "exp1_marginals.png",
    )
    plot_claim_count_distribution(
        wrap_results["kappa_distribution"], tau,
        RESULTS_DIR / "exp1_claim_count.png",
    )
    print("\nPlots saved to results/exp1_marginals.png and results/exp1_claim_count.png")

    # Save numerical results
    output = {
        "admissibility_scan": adm["admissibility_scan"],
        "wrap_around_verification": {k: v for k, v in wrap_results.items()},
        "convex_order": {
            "star_dominates_all": all_dominated,
            "test_function_values": {
                k: v for k, v in conv_results.items()
                if k not in ("star_dominates",)
            },
        },
        "non_admissible_je_violation_rate": round(je_rate_large, 6),
        "non_admissible_tau": round(tau_large, 4),
    }
    with open(RESULTS_DIR / "exp1_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("Numerical results saved to results/exp1_results.json")

    # Assertions for correctness
    assert wrap_results["marginal_ok"], (
        f"Marginal error {wrap_results['max_marginal_error']:.6f} exceeds tolerance {MARGINAL_TOLERANCE}"
    )
    assert wrap_results["je_ok"], (
        f"JE violation rate {wrap_results['je_violation_rate']:.2e} exceeds tolerance"
    )
    assert wrap_results["kappa_support_correct"], (
        f"kappa* support {wrap_results['empirical_support']} != expected {wrap_results['expected_support']}"
    )
    assert all_dominated, "kappa* does not dominate all alternatives in convex order"
    assert je_rate_large > 0.0, "Expected non-zero JE violation in non-admissible case"

    print("\nAll assertions passed.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: JE wrap-around construction")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run with paper-scale sample size (same as default)",
    )
    args = parser.parse_args()
    main()
