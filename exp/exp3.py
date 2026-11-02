"""
Experiment 3: Generalised JE with linear truncation distortion (Section 3.3).

Reproduces Example 3.13 (trivariate G-JE, linear truncation):
- Marginals X_i = 0 w.p. 1/2, Uniform(0,1) w.p. 1/2  (q_i = 0.5, tau = 1.5).
- Linear truncation distortion G_i with a=1/8, b=1/2, giving G_i* = 3/8.
- G-JE admissibility condition verification.
- Face weights p_I(lambda) = lambda/4 + 1/8 over lambda in [1/6, 1/4].
- Simulation and verification of JE property, marginals, confinement, and axis range.

Run as: python -m exp.exp3
"""

import json
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    EXP3_A,
    EXP3_B,
    EXP3_CAPACITY,
    EXP3_CAPACITY_GRID,
    EXP3_CONFINEMENT_UPPER,
    EXP3_LAMBDA_MAX,
    EXP3_LAMBDA_MIN,
    EXP3_N,
    EXP3_Q,
    MARGINAL_TOLERANCE,
    N_MC_SAMPLES,
    RANDOM_SEED,
)
from src.generalised_je import (
    check_gje_admissibility,
    check_gje_face_weight_admissibility,
    compute_capacity,
    compute_linear_truncation_capacity,
    make_example313_axis_sampler,
    make_example313_distorted_sampler,
    make_linear_truncation,
    sample_gje,
)
from src.canonical_je import (
    ComonotonicCopula,
    CountermonotonicCopula,
    IndependenceCopula,
    enumerate_faces,
    faces_containing,
)
from src.je_construction import (
    verify_je_property,
    verify_marginals,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

N_LAMBDA = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_marginal_surv(q: float) -> Callable[[np.ndarray], np.ndarray]:
    """P(X > x) = q*(1-x) for x in [0,1], 0 for x >= 1."""
    def surv(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return np.where(x_arr >= 1.0, 0.0, q * np.maximum(1.0 - np.maximum(x_arr, 0.0), 0.0))
    return surv


def make_marginal_surv_scalar(q: float) -> Callable[[float], float]:
    """Scalar version for sample_gje interface."""
    def surv(x: float) -> float:
        return float(q * max(1.0 - max(float(x), 0.0), 0.0)) if float(x) < 1.0 else 0.0
    return surv


def face_weights_from_lambda(lam: float) -> Dict[FrozenSet[int], float]:
    """p_I(lambda) = lambda/4 + 1/8 for each of the three pairs."""
    p = lam / 4.0 + 1.0 / 8.0
    pairs = [frozenset([0, 1]), frozenset([0, 2]), frozenset([1, 2])]
    return {I: p for I in pairs}


def pi_from_lambda(lam: float) -> float:
    """
    pi_i = sum_{I containing i} p_I.
    Each component appears in exactly 2 pairs, so pi_i = 2*(lam/4 + 1/8).
    """
    return 2.0 * (lam / 4.0 + 1.0 / 8.0)


# ---------------------------------------------------------------------------
# Step 1: Capacity computation
# ---------------------------------------------------------------------------

def compute_capacities(
    a: float,
    b: float,
    q: float,
    n: int,
    capacity_grid_points: int,
) -> Tuple[float, float, float]:
    """
    Return (analytical_capacity, numerical_capacity, expected_capacity).

    The analytical formula gives G_i* = b - a.
    The numerical computation uses compute_capacity over a dense x-grid.
    """
    G_i = make_linear_truncation(a, b)
    surv_scalar = make_marginal_surv_scalar(q)

    analytical = compute_linear_truncation_capacity(a, b)
    x_grid = np.linspace(0.0, 1.0 - 1e-6, capacity_grid_points)
    numerical = compute_capacity(G_i, surv_scalar, x_grid)

    return analytical, numerical, EXP3_CAPACITY


# ---------------------------------------------------------------------------
# Step 2: Admissibility checks
# ---------------------------------------------------------------------------

def run_admissibility_checks(
    q_list: List[float],
    capacities: List[float],
    n_lambda: int,
) -> Tuple[dict, List[dict]]:
    """
    Check G-JE admissibility condition and face-weight admissibility over lambda grid.
    """
    adm = check_gje_admissibility(q_list, capacities)

    faces = enumerate_faces(EXP3_N)
    lambda_grid = np.linspace(EXP3_LAMBDA_MIN, EXP3_LAMBDA_MAX, n_lambda)
    face_adm_results = []
    for lam in lambda_grid:
        fw = face_weights_from_lambda(float(lam))
        res = check_gje_face_weight_admissibility(q_list, capacities, fw, faces)
        face_adm_results.append({
            "lambda": float(lam),
            "p_I": float(list(fw.values())[0]),
            "pi_i": float(pi_from_lambda(float(lam))),
            "admissible": bool(res["admissible"]),
            "lower_sum": float(res["lower_sum"]),
            "lower_bound": float(res["lower_bound"]),
            "lower_ok": bool(res["lower_ok"]),
        })

    return adm, face_adm_results


# ---------------------------------------------------------------------------
# Step 3: Simulation
# ---------------------------------------------------------------------------

def run_simulation(
    q_list: List[float],
    capacities: List[float],
    lam: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate G-JE samples for a given lambda."""
    n = len(q_list)
    a, b = EXP3_A, EXP3_B
    q = q_list[0]  # same for all components

    G_i = make_linear_truncation(a, b)
    distortions = [G_i] * n

    pi_i = pi_from_lambda(lam)
    dist_sampler = make_example313_distorted_sampler(a, b, q)
    axis_sampler_fn = make_example313_axis_sampler(a, b, q, pi_i)

    distorted_samplers = [dist_sampler] * n
    axis_samplers = [axis_sampler_fn] * n

    marginal_surv_scalar = make_marginal_surv_scalar(q)
    marginal_survival_fns = [marginal_surv_scalar] * n

    face_weights = face_weights_from_lambda(lam)
    face_copulas: Dict[FrozenSet[int], object] = {
        frozenset([0, 1]): IndependenceCopula(),
        frozenset([0, 2]): ComonotonicCopula(),
        frozenset([1, 2]): CountermonotonicCopula(),
    }

    samples = sample_gje(
        q=q_list,
        marginal_survival_fns=marginal_survival_fns,
        distortions=distortions,
        capacities=capacities,
        face_weights=face_weights,
        face_copulas=face_copulas,
        distorted_severity_samplers=distorted_samplers,
        axis_severity_samplers=axis_samplers,
        n_samples=n_samples,
        rng=rng,
    )
    return samples


# ---------------------------------------------------------------------------
# Step 4: Verification
# ---------------------------------------------------------------------------

def verify_all(
    samples: np.ndarray,
    q_list: List[float],
    confinement_upper: float,
) -> dict:
    """
    Run all verification checks:
    (a) JE property, (b) marginals, (c) confinement, (d) axis range.
    """
    n = samples.shape[1]
    q = q_list[0]

    # (a) JE property
    je_violation_rate = verify_je_property(samples)

    # (b) Marginals
    x_grid = np.linspace(0.0, 0.99, 40)
    surv_array_fns = [make_marginal_surv(q)] * n
    empirical, theoretical = verify_marginals(samples, surv_array_fns, x_grid)
    max_marginal_err = float(np.abs(empirical - theoretical).max())

    # Classify samples by active set
    n_positive = (samples > 0).sum(axis=1)
    is_face = n_positive >= 2         # intermediate face: 2+ positive
    is_axis = n_positive == 1         # axis: exactly 1 positive

    # (c) Confinement: face samples confined to x < 3/4
    face_samples = samples[is_face]
    if len(face_samples) > 0:
        positive_face_vals = face_samples[face_samples > 0]
        max_face_val = float(positive_face_vals.max()) if len(positive_face_vals) > 0 else 0.0
    else:
        max_face_val = 0.0
    confinement_ok = max_face_val < confinement_upper

    # (d) Axis losses not confined to < 3/4 (can exceed confinement_upper)
    axis_samples = samples[is_axis]
    if len(axis_samples) > 0:
        positive_axis_vals = axis_samples[axis_samples > 0]
        max_axis_val = float(positive_axis_vals.max()) if len(positive_axis_vals) > 0 else 0.0
        min_axis_val = float(positive_axis_vals.min()) if len(positive_axis_vals) > 0 else 0.0
        axis_exceeds_upper = bool((positive_axis_vals >= confinement_upper).any())
    else:
        max_axis_val = min_axis_val = 0.0
        axis_exceeds_upper = False

    return {
        "je_violation_rate": float(je_violation_rate),
        "je_ok": bool(je_violation_rate < 1e-3),
        "max_marginal_error": float(max_marginal_err),
        "marginal_ok": bool(max_marginal_err < MARGINAL_TOLERANCE),
        "n_face_samples": int(is_face.sum()),
        "n_axis_samples": int(is_axis.sum()),
        "n_origin_samples": int((n_positive == 0).sum()),
        "max_face_val": max_face_val,
        "confinement_ok": bool(confinement_ok),
        "confinement_upper": confinement_upper,
        "max_axis_val": max_axis_val,
        "min_axis_val": min_axis_val,
        "axis_exceeds_upper": bool(axis_exceeds_upper),
    }


# ---------------------------------------------------------------------------
# Step 5: Plot
# ---------------------------------------------------------------------------

def plot_confinement(
    samples: np.ndarray,
    confinement_upper: float,
    output_path: Path,
) -> None:
    """
    Plot histogram of face severity values and axis severity values,
    with a vertical line at the confinement upper bound.
    """
    n_positive = (samples > 0).sum(axis=1)
    is_face = n_positive >= 2
    is_axis = n_positive == 1

    face_vals = samples[is_face][samples[is_face] > 0]
    axis_vals = samples[is_axis][samples[is_axis] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Face severity distribution
    ax = axes[0]
    bins = np.linspace(0.0, 1.0, 61)
    ax.hist(face_vals, bins=bins, density=True, color="steelblue", alpha=0.75,
            label=f"Face severities (n={len(face_vals):,})")
    ax.axvline(x=confinement_upper, color="crimson", lw=2, linestyle="--",
               label=f"Confinement bound = {confinement_upper:.2f}")
    ax.set_xlabel("Severity value")
    ax.set_ylabel("Density")
    ax.set_title("Intermediate-face severity distribution\n"
                 "(confined to $(0,\\, 3/4)$, Example 3.13)")
    ax.legend(fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.3)

    # Axis severity distribution
    ax = axes[1]
    ax.hist(axis_vals, bins=bins, density=True, color="darkorange", alpha=0.75,
            label=f"Axis severities (n={len(axis_vals):,})")
    ax.axvline(x=confinement_upper, color="crimson", lw=2, linestyle="--",
               label=f"Confinement bound = {confinement_upper:.2f}")
    ax.set_xlabel("Severity value")
    ax.set_ylabel("Density")
    ax.set_title("Axis severity distribution\n"
                 "(not confined, spans beyond $3/4$, Example 3.13)")
    ax.legend(fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "G-JE Confinement: Face vs Axis Severities (Example 3.13, $\\lambda=1/4$)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_samples: int = N_MC_SAMPLES) -> None:
    """Run Experiment 3 and save all results."""
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 65)
    print("Experiment 3: G-JE with Linear Truncation Distortion (Example 3.13)")
    print("=" * 65)

    n = EXP3_N
    q_list = EXP3_Q
    a, b = EXP3_A, EXP3_B
    q = q_list[0]

    # ------------------------------------------------------------------
    # 1. Capacity
    # ------------------------------------------------------------------
    print(f"\n--- Capacity (G_i* = b - a = {b} - {a} = {b-a}) ---")
    analytical_cap, numerical_cap, expected_cap = compute_capacities(
        a, b, q, n, EXP3_CAPACITY_GRID
    )
    print(f"  Analytical G_i*   = {analytical_cap:.6f}")
    print(f"  Numerical G_i*    = {numerical_cap:.6f}  (grid={EXP3_CAPACITY_GRID} pts)")
    print(f"  Expected (config) = {expected_cap:.6f}")
    cap_match = abs(analytical_cap - expected_cap) < 1e-10
    cap_num_close = abs(numerical_cap - expected_cap) < 5e-4
    print(f"  Analytical matches expected: {cap_match}")
    print(f"  Numerical close to expected: {cap_num_close}")

    capacities = [analytical_cap] * n

    # ------------------------------------------------------------------
    # 2. G-JE admissibility condition
    # ------------------------------------------------------------------
    print("\n--- G-JE Admissibility Condition ---")
    adm = check_gje_admissibility(q_list, capacities)
    tau = adm["tau"]
    print(f"  n={n}, tau=sum(q_i)={tau:.4f}, tau-1={adm['lhs']:.4f}")
    print(f"  sum G_i* = {adm['sum_caps']:.6f}, max G_i* = {adm['max_cap']:.6f}")
    print(f"  (n-2)/(n-1)*sum G_i*  = rhs_1 = {adm['rhs_1']:.6f}")
    print(f"  sum G_i* - max G_i*   = rhs_2 = {adm['rhs_2']:.6f}")
    print(f"  rhs = min(rhs_1, rhs_2) = {adm['rhs']:.6f}")
    print(f"  Condition tau-1 <= rhs: {adm['lhs']:.4f} <= {adm['rhs']:.4f}  ->  {adm['admissible']}")

    # ------------------------------------------------------------------
    # 3. Face-weight admissibility over lambda grid
    # ------------------------------------------------------------------
    print(f"\n--- Face-Weight Admissibility  (lambda in [{EXP3_LAMBDA_MIN:.4f}, {EXP3_LAMBDA_MAX:.4f}], {N_LAMBDA} pts) ---")
    _, face_adm_list = run_admissibility_checks(q_list, capacities, N_LAMBDA)
    all_face_adm = all(r["admissible"] for r in face_adm_list)
    print(f"  All {N_LAMBDA} lambda values admissible: {all_face_adm}")
    print(f"  p_I range: [{face_adm_list[0]['p_I']:.6f}, {face_adm_list[-1]['p_I']:.6f}]")
    print(f"  pi_i range: [{face_adm_list[0]['pi_i']:.6f}, {face_adm_list[-1]['pi_i']:.6f}]  "
          f"(G_i* = {analytical_cap:.6f})")

    # ------------------------------------------------------------------
    # 4. Simulation  (lambda = EXP3_LAMBDA_MAX = 1/4)
    # ------------------------------------------------------------------
    lam_sim = EXP3_LAMBDA_MAX
    pi_sim = pi_from_lambda(lam_sim)
    p_I_sim = lam_sim / 4.0 + 1.0 / 8.0
    print(f"\n--- Simulation  (lambda={lam_sim:.4f}, p_I={p_I_sim:.4f}, pi_i={pi_sim:.4f}) ---")
    print(f"  n_samples={n_samples:,}")
    samples = run_simulation(q_list, capacities, lam_sim, n_samples, rng)
    print(f"  Samples shape: {samples.shape}")

    # ------------------------------------------------------------------
    # 5. Verification
    # ------------------------------------------------------------------
    print("\n--- Verification ---")
    ver = verify_all(samples, q_list, EXP3_CONFINEMENT_UPPER)

    print(f"  (a) JE property violation rate: {ver['je_violation_rate']:.2e}  "
          f"({'OK' if ver['je_ok'] else 'FAIL'})")
    print(f"  (b) Max marginal error:         {ver['max_marginal_error']:.6f}  "
          f"({'OK' if ver['marginal_ok'] else 'FAIL'})")

    total_pos = ver["n_face_samples"] + ver["n_axis_samples"]
    frac_face = ver["n_face_samples"] / n_samples
    frac_axis = ver["n_axis_samples"] / n_samples
    frac_orig = ver["n_origin_samples"] / n_samples
    print(f"  Sample composition:  origin={frac_orig:.4f},  "
          f"axis={frac_axis:.4f},  face={frac_face:.4f}")

    print(f"  (c) Confinement: max face value = {ver['max_face_val']:.6f}  "
          f"< {EXP3_CONFINEMENT_UPPER} ?  {ver['confinement_ok']}")
    print(f"  (d) Axis range:  min={ver['min_axis_val']:.6f}, max={ver['max_axis_val']:.6f}  "
          f"(exceeds {EXP3_CONFINEMENT_UPPER}: {ver['axis_exceeds_upper']})")

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    plot_path = RESULTS_DIR / "exp3_confinement.png"
    plot_confinement(samples, EXP3_CONFINEMENT_UPPER, plot_path)
    print(f"\nPlot saved to {plot_path}")

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    results = {
        "description": "Example 3.13: Trivariate G-JE with linear truncation distortion",
        "parameters": {
            "n": n,
            "q_list": q_list,
            "tau": tau,
            "a": a,
            "b": b,
            "capacity_analytical": analytical_cap,
            "capacity_numerical": numerical_cap,
            "capacity_expected": expected_cap,
        },
        "gje_admissibility": {
            k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
            for k, v in adm.items()
        },
        "face_weight_admissibility": {
            "all_admissible": bool(all_face_adm),
            "n_lambda": N_LAMBDA,
            "lambda_min": EXP3_LAMBDA_MIN,
            "lambda_max": EXP3_LAMBDA_MAX,
            "grid": face_adm_list,
        },
        "simulation": {
            "lambda": lam_sim,
            "p_I": p_I_sim,
            "pi_i": pi_sim,
            "n_samples": n_samples,
        },
        "verification": {
            k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
            for k, v in ver.items()
        },
    }

    json_path = RESULTS_DIR / "exp3_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {json_path}")

    # ------------------------------------------------------------------
    # 8. Assertions
    # ------------------------------------------------------------------
    assert cap_match, f"Analytical capacity {analytical_cap} != expected {expected_cap}"
    assert cap_num_close, (
        f"Numerical capacity {numerical_cap:.6f} not close to expected {expected_cap:.6f}"
    )
    assert adm["admissible"], "G-JE admissibility condition failed"
    assert all_face_adm, "Face-weight admissibility failed for some lambda"
    assert ver["je_ok"], f"JE violation rate {ver['je_violation_rate']:.2e} too high"
    assert ver["marginal_ok"], f"Max marginal error {ver['max_marginal_error']:.6f} exceeds tolerance"
    assert ver["confinement_ok"], (
        f"Confinement failed: max face value {ver['max_face_val']:.6f} >= {EXP3_CONFINEMENT_UPPER}"
    )
    assert ver["axis_exceeds_upper"], (
        f"Axis losses did not exceed confinement bound {EXP3_CONFINEMENT_UPPER}"
    )

    print("\nAll assertions passed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
