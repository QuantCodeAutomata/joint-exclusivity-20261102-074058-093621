"""
Experiment 5: Aggregate-loss extremal structures, stop-loss premiums, and TVaR bounds.

Reproduces the following from Section 4.2 of the paper:
- Stop-loss premiums for the safest aggregate S* (formula from Section 4.2).
- Upper bound on stop-loss within the m-exclusive frequency-severity class.
- Frechet-class worst case S# (Theorem 4.6).
- Benchmark table (Example 4.8): n=10, q=0.15, unit-mean exponential severities,
  comparing Variance and TVaR_0.99 across six structures.

Run as: python -m exp.exp5
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    EXP5_N,
    EXP5_Q,
    EXP5_SEVERITY_RATE,
    EXP5_M_3EXCL,
    EXP5_M_JE,
    EXP5_TVAR_ALPHA,
    EXP5_D_MAX,
    EXP5_D_POINTS,
    N_MC_SAMPLES,
    RANDOM_SEED,
)
from src.aggregate_loss import (
    stop_loss_gamma,
    stop_loss_safest,
    stop_loss_class_upper_bound,
    sample_safest_aggregate,
    sample_independent_aggregate,
    sample_comonotonic_aggregate,
    sample_m_exclusive_upper_bound_aggregate,
    sample_frechet_worst_case,
    tvar_from_samples,
    compute_benchmark_table,
    compute_analytical_safest_stats,
)
from src.je_construction import make_exponential_sampler

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def compute_stop_loss_curves(
    tau: float,
    rate: float,
    m_3: int,
    m_je: int,
    d_values: np.ndarray,
) -> dict:
    """Compute analytical stop-loss curves for safest and upper bounds."""
    pi_star = np.array([stop_loss_safest(d, tau, rate) for d in d_values])
    pi_3excl = np.array([stop_loss_class_upper_bound(d, tau, m_3, rate) for d in d_values])
    pi_je = np.array([stop_loss_class_upper_bound(d, tau, m_je, rate) for d in d_values])
    return {
        "d": d_values,
        "pi_star": pi_star,
        "pi_3excl_bound": pi_3excl,
        "pi_je_bound": pi_je,
    }


def plot_stop_loss_curves(curves: dict, output_path: Path) -> None:
    """Plot stop-loss premium curves."""
    d = curves["d"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d, curves["pi_star"], label="Safest S* (formula)", lw=2, color="steelblue")
    ax.plot(d, curves["pi_3excl_bound"], label="3-excl upper bound", lw=2,
            linestyle="--", color="tomato")
    ax.plot(d, curves["pi_je_bound"], label="JE upper bound", lw=2,
            linestyle=":", color="green")
    ax.set_xlabel("Retention d")
    ax.set_ylabel("Stop-loss premium Pi(d)")
    ax.set_title("Stop-Loss Premiums: Safest vs Upper Bounds\n(n=10, q=0.15, unit-exp)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, d[-1])
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_aggregate_distributions(
    structures: dict,
    alpha: float,
    output_path: Path,
) -> None:
    """Plot empirical CDFs of aggregate losses for all structures."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["steelblue", "green", "tomato", "purple", "orange", "black"]
    for (name, S), color in zip(structures.items(), colors):
        S_sorted = np.sort(S)
        cdf = np.arange(1, len(S) + 1) / len(S)
        ax.plot(S_sorted, cdf, label=name, lw=1.5, color=color, alpha=0.85)
    ax.axhline(y=alpha, color="gray", linestyle="--", lw=1, label=f"alpha={alpha}")
    ax.set_xlabel("Aggregate loss S")
    ax.set_ylabel("CDF")
    ax.set_title("Aggregate Loss Distributions: Six Structures\n(n=10, q=0.15, unit-exp)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def main(n_samples: int = N_MC_SAMPLES) -> None:
    """Run experiment 5 and save all results."""
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 60)
    print("Experiment 5: Aggregate Loss and Stop-Loss Premiums")
    print("=" * 60)

    n = EXP5_N
    q = EXP5_Q
    rate = EXP5_SEVERITY_RATE
    alpha = EXP5_TVAR_ALPHA
    tau = n * q
    q_vec = [q] * n

    print(f"\nBenchmark parameters: n={n}, q={q}, rate={rate}, tau={tau:.3f}, alpha={alpha}")

    # Part 1: Stop-loss curves (analytical)
    print("\nComputing analytical stop-loss curves...")
    d_values = np.linspace(0.0, EXP5_D_MAX, EXP5_D_POINTS)
    curves = compute_stop_loss_curves(tau, rate, EXP5_M_3EXCL, EXP5_M_JE, d_values)

    # Verify at d=0: Pi_S*(0) = E[S*] = tau/rate
    pi_star_0 = stop_loss_safest(0.0, tau, rate)
    print(f"  Pi_S*(0) = E[S*] = {pi_star_0:.4f}, expected = {tau/rate:.4f}")

    # Part 2: Benchmark table (Monte Carlo)
    print(f"\nRunning Monte Carlo benchmark (n_samples={n_samples:,})...")
    t0 = time.time()
    table = compute_benchmark_table(n, q, rate, alpha, n_samples, rng)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Also compute analytical stats for S*
    analytical = compute_analytical_safest_stats(n, q, rate, alpha)

    print("\nBenchmark Table (Example 4.8):")
    print(f"{'Structure':<25} {'Mean':>8} {'Variance':>10} {'TVaR_0.99':>12}")
    print("-" * 60)
    for name, stats in table.items():
        print(f"{name:<25} {stats['mean']:>8.4f} {stats['variance']:>10.4f} {stats['tvar']:>12.4f}")
    print("-" * 60)
    print(f"{'Analytical S* mean':<25} {analytical['mean']:>8.4f} "
          f"{analytical['variance']:>10.4f}")

    # Part 3: Verify orderings
    print("\nVerifying expected orderings:")
    s_star_var = table["Safest (S*)"]["variance"]
    s_indep_var = table["Independent"]["variance"]
    s_comono_var = table["Co-monotonic"]["variance"]
    s_star_tvar = table["Safest (S*)"]["tvar"]
    s_indep_tvar = table["Independent"]["tvar"]
    s_comono_tvar = table["Co-monotonic"]["tvar"]

    print(f"  Var(S*) = {s_star_var:.4f} <= Var(Indep) = {s_indep_var:.4f}: "
          f"{'OK' if s_star_var <= s_indep_var + 0.05 else 'WARN'}")
    print(f"  Var(Indep) = {s_indep_var:.4f} <= Var(Comono) = {s_comono_var:.4f}: "
          f"{'OK' if s_indep_var <= s_comono_var + 0.05 else 'WARN'}")
    print(f"  TVaR(S*) = {s_star_tvar:.4f} <= TVaR(Indep) = {s_indep_tvar:.4f}: "
          f"{'OK' if s_star_tvar <= s_indep_tvar + 0.05 else 'WARN'}")

    # Part 4: Upper bound check
    pi_ub_3excl_0 = stop_loss_class_upper_bound(0.0, tau, EXP5_M_3EXCL, rate)
    pi_star_0_check = stop_loss_safest(0.0, tau, rate)
    print(f"\n  E[S*] = {pi_star_0_check:.4f} <= 3-excl upper bound Pi_0 = {pi_ub_3excl_0:.4f}: "
          f"{'OK' if pi_star_0_check <= pi_ub_3excl_0 + 1e-4 else 'WARN'}")

    # Attainability check: max q_i <= tau / (m-1)
    max_q = max(q_vec)
    attain_3 = max_q <= tau / (EXP5_M_3EXCL - 1) + 1e-10
    attain_je = max_q <= tau / (EXP5_M_JE - 1) + 1e-10
    print(f"  Attainability (m=3): max_q={max_q:.3f} <= tau/(m-1)={tau/(EXP5_M_3EXCL-1):.3f}: {attain_3}")
    print(f"  Attainability (m=10): max_q={max_q:.3f} <= tau/(m-1)={tau/(EXP5_M_JE-1):.4f}: {attain_je}")

    # Plots
    rng2 = np.random.default_rng(RANDOM_SEED + 1)
    sev_sampler = make_exponential_sampler(rate)
    from src.aggregate_loss import (
        sample_safest_aggregate, sample_independent_aggregate,
        sample_comonotonic_aggregate, sample_m_exclusive_upper_bound_aggregate,
        sample_frechet_worst_case,
    )

    sev_samplers = [sev_sampler] * n
    q_sorted = np.array(sorted([q] * n, reverse=True))

    def G_quantile(u: np.ndarray) -> np.ndarray:
        return -np.log(1.0 - np.clip(u, 0.0, 1.0 - 1e-10)) / rate

    def marginal_q_i(u: np.ndarray) -> np.ndarray:
        active = u > (1.0 - q)
        u_scaled = np.where(active, (u - (1.0 - q)) / q, 0.0)
        sev = G_quantile(np.clip(u_scaled, 0.0, 1.0))
        return np.where(active, sev, 0.0)

    n_plot = min(n_samples, 50_000)
    structures_plot = {
        "Safest (S*)": sample_safest_aggregate(q_vec, sev_samplers, n_plot, rng2),
        "Independent": sample_independent_aggregate(q_vec, sev_sampler, n_plot, rng2),
        "3-excl worst": sample_m_exclusive_upper_bound_aggregate(
            q_vec, tau, EXP5_M_3EXCL, sev_sampler, n_plot, rng2),
        "JE worst": sample_m_exclusive_upper_bound_aggregate(
            q_vec, tau, EXP5_M_JE, sev_sampler, n_plot, rng2),
        "Co-monotonic": sample_comonotonic_aggregate(q_vec, sev_sampler, n_plot, rng2),
        "Frechet worst": sample_frechet_worst_case(
            q_sorted, EXP5_M_3EXCL, G_quantile, [marginal_q_i] * n, n_plot, rng2),
    }

    plot_stop_loss_curves(curves, RESULTS_DIR / "exp5_stop_loss.png")
    plot_aggregate_distributions(
        structures_plot, alpha, RESULTS_DIR / "exp5_distributions.png"
    )
    print("\nPlots saved to results/exp5_stop_loss.png and results/exp5_distributions.png")

    # Save results
    output = {
        "parameters": {"n": n, "q": q, "rate": rate, "tau": tau, "alpha": alpha},
        "analytical_safest": analytical,
        "benchmark_table": table,
        "stop_loss_at_d0": {
            "pi_star": float(pi_star_0),
            "expected_mean": float(tau / rate),
            "pi_3excl_bound": float(pi_ub_3excl_0),
        },
        "attainability": {
            "m3": attain_3,
            "m_je": attain_je,
        },
        "orderings_verified": {
            "var_star_le_indep": bool(s_star_var <= s_indep_var + 0.05),
            "var_indep_le_comono": bool(s_indep_var <= s_comono_var + 0.05),
            "tvar_star_le_indep": bool(s_star_tvar <= s_indep_tvar + 0.05),
        },
    }
    with open(RESULTS_DIR / "exp5_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("Numerical results saved to results/exp5_results.json")

    # Assertions
    assert pi_star_0 <= pi_ub_3excl_0 + 1e-6, "S* mean exceeds upper bound"
    assert s_star_var <= s_indep_var + 0.1, "Var(S*) should be <= Var(Indep)"

    print("\nAll assertions passed.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 5: Aggregate loss and TVaR")
    parser.add_argument("--full", action="store_true",
                        help="Run with paper-scale configuration")
    args = parser.parse_args()
    main()
