"""
Experiment 4: m-exclusivity hierarchy and threshold exceedance exclusivity.

Reproduces the following from Sections 3.4 and 4.1 of the paper:
- m-exclusivity hierarchy: existence condition tau <= m-1 (Section 3.4)
- Wrap-around construction gives the strongest admissible exclusivity level
- Convex-order bounds: kappa* <=_cx |A(X_indep)| <=_cx kappa_m#
- Threshold exceedance JE construction for standard normal marginals (Section 4.1)

Run as: python -m exp.exp4
"""

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from src.config import (
    EXP4_M_VALUES,
    EXP4_N,
    EXP4_Q,
    EXP4_THRESH_ALPHA,
    EXP4_THRESH_M,
    EXP4_THRESH_N,
    N_MC_SAMPLES,
    RANDOM_SEED,
)
from src.je_construction import (
    compute_claim_counts,
    make_exponential_sampler,
    sample_alternative_activity_patterns,
    sample_je_wraparound,
    verify_je_property,
)
from src.m_exclusivity import (
    check_m_exclusive_existence,
    convex_order_bounds_check,
    make_normal_excess_sampler,
    make_normal_subthreshold_sampler,
    sample_threshold_exceedance_je,
    sample_upper_extremal_count,
    strongest_admissible_m,
    upper_extremal_count_distribution,
    verify_m_exclusive,
    verify_threshold_exceedance_je,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Part A helpers
# ---------------------------------------------------------------------------

def run_m_hierarchy(
    q: list[float],
    m_values: list[int],
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], np.ndarray]:
    """
    Run the m-exclusivity hierarchy experiment.

    For each m, checks existence (tau <= m-1), verifies that the wrap-around
    satisfies the m-exclusive property, computes kappa_m#, and checks the
    convex-order sandwich kappa* <=_cx kappa_indep <=_cx kappa_m#.

    Returns
    -------
    results : dict
    samples_je : ndarray of shape (n_samples, n) from the wrap-around
    """
    tau = float(sum(q))
    n = len(q)
    s_m = strongest_admissible_m(q)

    # Severity samplers: unit-mean exponential for all components.
    severity_samplers = [make_exponential_sampler(1.0) for _ in range(n)]

    # Wrap-around depends only on q, so one simulation suffices for all m.
    samples_je = sample_je_wraparound(q, severity_samplers, n_samples, rng)

    results: dict[str, Any] = {
        "n": n,
        "q": q,
        "tau": tau,
        "strongest_admissible_m": s_m,
        "m_results": {},
    }

    for m in m_values:
        existence = check_m_exclusive_existence(q, m)
        m_result: dict[str, Any] = {
            "m": m,
            "exists": existence["exists"],
            "tau": tau,
            "bound": m - 1,
            "condition": f"tau={tau:.4f} <= m-1={m - 1}",
        }

        if existence["exists"]:
            # Verify m-exclusive property: max simultaneous active components < m.
            verify = verify_m_exclusive(samples_je, m)
            m_result["violation_rate"] = verify["violation_rate"]
            m_result["max_count"] = verify["max_count"]
            m_result["m_exclusive_ok"] = verify["violation_rate"] < 1e-6
            m_result["count_distribution"] = verify["count_distribution"]

            # Upper extremal distribution kappa_m#.
            kappa_dist = upper_extremal_count_distribution(tau, m)
            m_result["kappa_m_sharp"] = {
                "kappa_values": kappa_dist["kappa_values"],
                "probabilities": kappa_dist["probabilities"],
                "mean": kappa_dist["mean"],
                "variance": kappa_dist["variance"],
            }

            # Independent Bernoulli alternative with the same marginal q_i.
            samples_indep = sample_alternative_activity_patterns(
                q, n_samples, rng, method="independent"
            )

            bounds = convex_order_bounds_check(q, m, samples_je, samples_indep)
            m_result["convex_order_bounds"] = bounds
            m_result["lower_bound_holds_all"] = all(
                v["lower_bound_holds"] for v in bounds.values()
            )
            m_result["upper_bound_holds_all"] = all(
                v["upper_bound_holds"] for v in bounds.values()
            )

        results["m_results"][str(m)] = m_result

    return results, samples_je


# ---------------------------------------------------------------------------
# Part B helpers
# ---------------------------------------------------------------------------

def run_threshold_exceedance(
    n: int,
    alpha_levels: list[float],
    m: int,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], np.ndarray]:
    """
    Construct the threshold exceedance JE vector and verify its properties.

    Returns
    -------
    results : dict
    samples : ndarray of shape (n_samples, n)
    """
    thresholds = [norm.ppf(a) for a in alpha_levels]
    p_exceed = [1.0 - a for a in alpha_levels]
    tau_exceed = float(sum(p_exceed))

    marginal_survival_fns = [lambda x, _i=i: float(norm.sf(x)) for i in range(n)]
    excess_samplers = [
        make_normal_excess_sampler(0.0, 1.0, thresholds[i]) for i in range(n)
    ]
    sub_samplers = [
        make_normal_subthreshold_sampler(0.0, 1.0, thresholds[i]) for i in range(n)
    ]

    t0 = time.time()
    samples = sample_threshold_exceedance_je(
        marginal_survival_fns=marginal_survival_fns,
        thresholds=thresholds,
        excess_severity_samplers=excess_samplers,
        subthreshold_samplers=sub_samplers,
        m=m,
        n_samples=n_samples,
        rng=rng,
    )
    elapsed = time.time() - t0

    # Verify JE property above thresholds.
    verify = verify_threshold_exceedance_je(samples, thresholds, m)

    # Verify marginals match N(0,1).
    x_grid = np.linspace(-3.0, 3.0, 60)
    marginal_errors: list[float] = []
    for i in range(n):
        emp_sf = np.array([float(np.mean(samples[:, i] > x)) for x in x_grid])
        theo_sf = norm.sf(x_grid)
        marginal_errors.append(float(np.abs(emp_sf - theo_sf).max()))

    max_marginal_error = max(marginal_errors)

    return {
        "n": n,
        "m": m,
        "alpha_levels": alpha_levels,
        "thresholds": [round(float(t), 6) for t in thresholds],
        "p_exceed": [round(float(p), 6) for p in p_exceed],
        "tau_exceed": round(tau_exceed, 6),
        "existence_condition_holds": tau_exceed <= m - 1 + 1e-10,
        "n_samples": n_samples,
        "simulation_time_s": round(elapsed, 2),
        "je_verification": {
            k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
            for k, v in verify.items()
        },
        "marginal_errors": [round(e, 6) for e in marginal_errors],
        "max_marginal_error": round(max_marginal_error, 6),
        "marginal_ok": bool(max_marginal_error < 0.02),
    }, samples


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_m_hierarchy(
    m_results: dict[str, Any],
    samples_je: np.ndarray,
    q: list[float],
    tau: float,
    output_path: Path,
) -> None:
    """Plot count distributions from wrap-around and kappa_m# for each admissible m."""
    admissible_ms = [int(m) for m, r in m_results.items() if r["exists"]]
    if not admissible_ms:
        return

    n_mc = len(samples_je)
    counts_je = compute_claim_counts(samples_je)
    plot_rng = np.random.default_rng(RANDOM_SEED + 77)

    n_cols = len(admissible_ms)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]

    all_counts = np.arange(0, int(counts_je.max()) + 2)
    bins = np.arange(-0.5, int(counts_je.max()) + 1.5, 1.0)

    for ax, m in zip(axes, admissible_ms):
        r = m_results[str(m)]
        kappa_upper_samples = sample_upper_extremal_count(tau, m, n_mc, plot_rng)
        counts_indep = compute_claim_counts(
            sample_alternative_activity_patterns(
                q, n_mc, np.random.default_rng(RANDOM_SEED + 88), method="independent"
            )
        )

        max_bin = max(int(counts_je.max()), m, int(counts_indep.max())) + 1
        plot_bins = np.arange(-0.5, max_bin + 0.5, 1.0)

        ax.hist(
            counts_je, bins=plot_bins, density=True, alpha=0.65,
            color="steelblue", label="kappa* (wrap-around)", zorder=3,
        )
        ax.hist(
            counts_indep, bins=plot_bins, density=True, alpha=0.45,
            color="green", label="|A(X_indep)|", zorder=2,
        )
        # kappa_m# as vertical bars
        kd = r["kappa_m_sharp"]
        for val, prob in zip(kd["kappa_values"], kd["probabilities"]):
            ax.bar(
                val, prob, width=0.25, color="tomato", alpha=0.8,
                label=f"kappa_m# (m={m})" if val == kd["kappa_values"][0] else None,
                zorder=4,
            )

        ax.axvline(x=tau, color="navy", linestyle=":", lw=2, label=f"tau={tau:.2f}")
        ax.set_xlabel("Count |A(X)|")
        ax.set_ylabel("Probability")
        ax.set_title(
            f"m={m}  (tau={tau:.2f} \u2264 m\u22121={m - 1}: \u2713 Admissible)\n"
            f"Wrap-around: max_count={r['max_count']} < {m} \u2713"
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"m-Exclusivity Hierarchy  (n={len(q)}, tau={tau:.2f})\n"
        "kappa* \u2264_cx |A(X_indep)| \u2264_cx kappa_m#",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_threshold_exceedance(
    samples: np.ndarray,
    thresholds: list[float],
    alpha_levels: list[float],
    output_path: Path,
) -> None:
    """Plot marginal densities vs N(0,1) and threshold markers."""
    n = samples.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    x_grid = np.linspace(-3.5, 3.5, 300)
    theo_pdf = norm.pdf(x_grid)

    for i, ax in enumerate(axes):
        ax.hist(
            samples[:, i], bins=70, density=True, alpha=0.55,
            color="steelblue", label="Simulated Z_tilde_i", range=(-4.0, 4.0),
        )
        ax.plot(x_grid, theo_pdf, color="tomato", lw=2, label="N(0,1)")
        ax.axvline(
            x=thresholds[i], color="green", linestyle="--", lw=2,
            label=f"l_i={thresholds[i]:.3f}  (\u03b1={alpha_levels[i]:.2f})",
        )
        ax.set_xlabel("z")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Component {i + 1}\n"
            f"P(Z>l)={1 - alpha_levels[i]:.2f}"
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Threshold Exceedance JE: Marginal Verification\n"
        "(Z_tilde_i ~ N(0,1), no 2 components simultaneously exceed thresholds)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_samples: int = N_MC_SAMPLES) -> None:
    """Run experiment 4 and save all results."""
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 65)
    print("Experiment 4: m-Exclusivity Hierarchy and Threshold Exceedance")
    print("=" * 65)

    q = EXP4_Q
    n = EXP4_N
    m_values = EXP4_M_VALUES
    tau = float(sum(q))

    print(f"\n[Parameters] n={n}, tau={tau:.4f}, q={q}")
    print(f"[Parameters] m values tested: {m_values}")
    print(f"Strongest admissible m: {strongest_admissible_m(q)}")

    # ------------------------------------------------------------------
    # Part A: m-exclusivity hierarchy
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("Part A: m-Exclusivity Hierarchy")
    print("-" * 65)

    t0 = time.time()
    hier_results, samples_je = run_m_hierarchy(q, m_values, n_samples, rng)
    elapsed_a = time.time() - t0
    print(f"Wrap-around simulation: {elapsed_a:.1f}s  (n_samples={n_samples:,})")

    # Summary table header
    print(
        f"\n{'m':>4}  {'Exists':>7}  {'Condition':>26}  "
        f"{'ViolRate':>10}  {'MaxCnt':>7}  {'LoBnd':>7}  {'UpBnd':>7}"
    )
    print("-" * 75)
    for m in m_values:
        r = hier_results["m_results"][str(m)]
        cond = r["condition"]
        exists = r["exists"]
        if exists:
            viol = f"{r['violation_rate']:.2e}"
            maxc = str(r["max_count"])
            lo = str(r["lower_bound_holds_all"])
            up = str(r["upper_bound_holds_all"])
        else:
            viol = maxc = lo = up = "N/A"
        print(f"{m:>4}  {str(exists):>7}  {cond:>26}  {viol:>10}  {maxc:>7}  {lo:>7}  {up:>7}")

    # Detailed output per m
    for m in m_values:
        r = hier_results["m_results"][str(m)]
        print(f"\n  m={m}: exists={r['exists']}, {r['condition']}")
        if not r["exists"]:
            print(f"    => m-exclusive vectors with these marginals do NOT exist for m={m}.")
            continue

        mc_ok = r["m_exclusive_ok"]
        print(
            f"    Wrap-around satisfies {m}-exclusivity: "
            f"max_count={r['max_count']} < {m}  =>  {'OK' if mc_ok else 'FAIL'}"
        )
        kd = r["kappa_m_sharp"]
        print(
            f"    kappa_m# distribution: "
            f"P(0)={kd['probabilities'][0]:.4f}, "
            f"P({m - 1})={kd['probabilities'][1]:.4f}, "
            f"mean={kd['mean']:.4f}, var={kd['variance']:.4f}"
        )
        print(f"    Convex-order bounds (kappa* <=_cx indep <=_cx kappa_m#):")
        for fn, bres in r["convex_order_bounds"].items():
            lo_ok = bres["lower_bound_holds"]
            up_ok = bres["upper_bound_holds"]
            print(
                f"      {fn:6s}: kappa*={bres['kappa_star']:8.4f}  "
                f"indep={bres['alternative']:8.4f}  "
                f"kappa_m#={bres['kappa_upper']:8.4f}  "
                f"[lo={lo_ok}, up={up_ok}]"
            )
        print(
            f"    All lower bounds hold: {r['lower_bound_holds_all']}  "
            f"| All upper bounds hold: {r['upper_bound_holds_all']}"
        )
        if not r["upper_bound_holds_all"]:
            print(
                f"    NOTE: upper bound fails because independent Bernoulli is not "
                f"{m}-exclusive (count may exceed m-1={m - 1})."
            )

    # ------------------------------------------------------------------
    # Part B: threshold exceedance exclusivity
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("Part B: Threshold Exceedance Exclusivity")
    print("-" * 65)

    thresh_n = EXP4_THRESH_N
    thresh_alpha = EXP4_THRESH_ALPHA
    thresh_m = EXP4_THRESH_M
    thresholds_val = [float(norm.ppf(a)) for a in thresh_alpha]
    p_exceed_val = [round(1.0 - a, 6) for a in thresh_alpha]

    print(f"\n[Parameters] n={thresh_n}, m={thresh_m}")
    print(f"  alpha_levels = {thresh_alpha}")
    print(f"  thresholds   = {[round(t, 4) for t in thresholds_val]}")
    print(f"  p_exceed     = {p_exceed_val}  (sum={sum(p_exceed_val):.4f})")
    print(
        f"  Existence: sum(p_exceed)={sum(p_exceed_val):.4f} <= m-1={thresh_m - 1}  "
        f"=> {sum(p_exceed_val) <= thresh_m - 1 + 1e-10}"
    )

    t0 = time.time()
    thresh_results, samples_thresh = run_threshold_exceedance(
        thresh_n, thresh_alpha, thresh_m, n_samples, rng
    )
    elapsed_b = time.time() - t0
    print(f"\nThreshold exceedance simulation: {elapsed_b:.1f}s")

    ve = thresh_results["je_verification"]
    print("\nJE verification (no 2 components exceed thresholds simultaneously):")
    print(f"  Violation rate (>= 2 exceedances): {ve['violation_rate']:.2e}")
    print(f"  Max simultaneous exceedances:       {ve['max_simultaneous_exceedances']}")
    print(f"  JE property holds:                  {ve['je_property_holds']}")

    print("\nMarginal verification (Z_tilde_i vs N(0,1)):")
    for i, err in enumerate(thresh_results["marginal_errors"]):
        status = "OK" if err < 0.02 else "FAIL"
        print(f"  Component {i + 1}: max |P_emp - P_theo| = {err:.6f}  [{status}]")
    print(f"  Overall marginal OK: {thresh_results['marginal_ok']}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")
    plot_m_hierarchy(
        hier_results["m_results"],
        samples_je,
        q,
        tau,
        RESULTS_DIR / "exp4_mhierarchy.png",
    )
    plot_threshold_exceedance(
        samples_thresh,
        thresholds_val,
        thresh_alpha,
        RESULTS_DIR / "exp4_threshold.png",
    )
    print("  Saved: results/exp4_mhierarchy.png")
    print("  Saved: results/exp4_threshold.png")

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    output: dict[str, Any] = {
        "part_a": {
            "parameters": {
                "n": n,
                "q": q,
                "tau": tau,
                "m_values": m_values,
                "strongest_admissible_m": hier_results["strongest_admissible_m"],
                "n_samples": n_samples,
            },
            "m_results": hier_results["m_results"],
        },
        "part_b": thresh_results,
    }
    json_path = RESULTS_DIR / "exp4_results.json"
    with open(json_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\nNumerical results saved to {json_path}")

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    # m-exclusive property must hold for all admissible m values.
    for m in m_values:
        r = hier_results["m_results"][str(m)]
        if r["exists"]:
            assert r["m_exclusive_ok"], (
                f"m={m}: wrap-around violation rate {r['violation_rate']:.2e} is non-zero"
            )
            # Lower convex-order bound kappa* <=_cx indep must always hold.
            assert r["lower_bound_holds_all"], (
                f"m={m}: lower convex-order bound kappa* <=_cx indep fails"
            )

    # Threshold exceedance: JE property must hold exactly.
    assert ve["je_property_holds"], (
        f"Threshold exceedance JE violated: rate={ve['violation_rate']:.2e}"
    )
    assert thresh_results["marginal_ok"], (
        f"Marginal error {thresh_results['max_marginal_error']:.6f} exceeds tolerance 0.02"
    )

    print("\nAll assertions passed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
