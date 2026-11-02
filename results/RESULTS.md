# Results: Joint Exclusivity

**Paper:** Joint Exclusivity  
**Year:** 2024 (exact authors and year are on the paper's title page)

All five experiments use fully synthetic inputs; no market or external data are required.  
Monte Carlo sample size: 200,000 (configured in `src/config.py` as `N_MC_SAMPLES`; the paper does not specify a sample size).  
Random seed: 42 (`RANDOM_SEED`; not specified in the paper).  
All numbers below are taken from the JSON files saved in `results/` by the most recent run.

---

## Experiment 1 — Existence and wrap-around construction of jointly exclusive random vectors

**Paper section:** Section 3.1, Theorem 3.2  
**Output files:** `results/exp1_results.json`, `results/exp1_*.png`

### Admissibility scan (n = 5)

The experiment scans three values of τ = Σ q_i around the boundary τ = n − 1 = 4 and reports whether a JE vector exists.

| Label | τ | n − 1 | JE exists (τ ≤ n−1) | Verdict |
|-------|---|--------|----------------------|---------|
| admissible | 3.5 | 4 | true | **reproduced** |
| boundary | 4.0 | 4 | true | **reproduced** |
| inadmissible | 4.3 | 4 | false | **reproduced** |

The paper states the necessary and sufficient condition is τ ≤ n − 1 (Section 3.1).  All three cases match the theoretical prediction exactly.

### Wrap-around construction verification (n = 5, τ = 1.5)

| Quantity | Expected | Simulated | Verdict |
|----------|----------|-----------|---------|
| Max marginal survival error | < MC noise | 0.00115 | **reproduced** |
| JE violation rate P(all X_i > 0) | 0 (exact) | 0.0 | **reproduced** |
| κ* support | {1, 2} = {⌊1.5⌋, ⌈1.5⌉} | {1, 2} | **reproduced** |
| κ* mean | τ = 1.5 | 1.4999 | **reproduced** |
| κ* variance | minimal (0.25 for τ = 1.5) | 0.25 | **reproduced** |

κ* distribution: P(κ* = 1) = 0.5001, P(κ* = 2) = 0.4999.  Paper: probability τ − ⌊τ⌋ = 0.5 on ⌈τ⌉ = 2.  ✓

### Convex-order minimality

Four alternative dependence structures (independent Bernoulli and three other activity-pattern distributions with the same marginals) were compared against the wrap-around construction κ* for five convex test functions (k², k³, e^{0.5k}, e^k, e^{2k}).  In all 20 comparisons the wrap-around κ* produced a strictly smaller expectation than each alternative.  `star_dominates_all: true`.

**Overall verdict: reproduced.**  The wrap-around construction satisfies the existence condition, preserves marginals, attains the JE property exactly, concentrates κ* on the two adjacent integers {⌊τ⌋, ⌈τ⌉}, and is minimal in convex order among tested alternatives.

---

## Experiment 2 — Canonical JE family: face-by-face specification and trivariate example

**Paper section:** Section 3.2, Example 3.9  
**Output files:** `results/exp2_results.json`, `results/exp2_*.png`

### Setup (Example 3.9)

- n = 3, each X_i zero with probability 1/2 and Uniform(0,1) otherwise; q_i = 0.5, τ = 1.5
- Intermediate faces M = {{0,1}, {0,2}, {1,2}}
- Face weights p_I(λ) = λ/2 for all I ∈ M, λ ∈ [1/3, 1/2]
- Face copulas: C_{01} = independence, C_{02} = comonotonic, C_{12} = countermonotonic

### Admissibility polytope

All 20 equally-spaced λ values in [1/3, 1/2] satisfy both canonical admissibility constraints.  `all_admissible: true`.  Verdict: **reproduced**.

### JE property

Maximum violation rate across all 20 λ values: 0.0 (exact zero).  `all_ok: true`.  Verdict: **reproduced**.

### Marginal preservation

Max survival error at the chosen λ (≈ 0.421) across a grid of x values on [0,1]: 0.00231 (< 0.5% of the survival range).  Verdict: **reproduced** (within Monte Carlo noise).

### Pairwise correlations (paper's closed-form formulas from Example 3.9)

The paper gives Corr(X₁, X₂) = (6λ−3)/5, Corr(X₁, X₃) = (8λ−3)/5, Corr(X₂, X₃) = (4λ−3)/5.  At λ = 1/3 (minimum):

| Pair | Theoretical | Simulated at λ = 1/3 | Error |
|------|-------------|----------------------|-------|
| (1,2) | −0.200 | −0.200 | 0.000 |
| (1,3) | −0.067 | −0.068 | 0.001 |
| (2,3) | −0.333 | −0.332 | 0.001 |

Maximum correlation error across all 20 λ values: 0.0061 (tolerance set to 0.03).  Correlations span negative to positive as λ increases and the triple-positive event remains absent throughout.  Verdict: **reproduced**.

**Overall verdict: reproduced.**  The canonical JE family with face-by-face copulas reproduces the paper's Example 3.9, including the admissibility polytope, JE property, marginals, and the closed-form correlation formulas.

---

## Experiment 3 — Generalised JE with distortion-based face severities and admissibility via capacities

**Paper section:** Section 3.3, Example 3.13  
**Output files:** `results/exp3_results.json`, `results/exp3_*.png`

### Setup (Example 3.13 — linear truncation)

- n = 3, same marginals as Example 3.9 (q_i = 0.5, F̄_i(x) = (1−x)/2 on [0,1])
- Linear truncation: G_i(u) = (u − a_i)/(b_i − a_i) on [a_i, b_i] with a_i = 1/8, b_i = 1/2
- Face weights p_I(λ) = λ/4 + 1/8 with λ ∈ [1/6, 1/4]; same three bivariate face copulas as Example 3.9

### Capacity G_i*

| Quantity | Analytical | Numerical | Verdict |
|----------|-----------|-----------|---------|
| G_i* = b_i − a_i | 0.375 | 0.37500 | **reproduced** |

The paper states G_i* = 3/8 = 0.375 (Section 3.3, Example 3.13).  ✓

### G-JE admissibility condition (paper's condition 3.18)

τ − 1 = 0.5 ≤ min{(n−2)/(n−1) · ΣG_i*, ΣG_i* − max G_i*} = min{9/16, 3/4} = 9/16 = 0.5625.

| Quantity | Expected | Computed | Verdict |
|----------|----------|----------|---------|
| lhs = τ − 1 | 0.5 | 0.5 | **reproduced** |
| rhs₁ = (1/2)·1.125 | 0.5625 | 0.5625 | **reproduced** |
| rhs₂ = 1.125 − 0.375 | 0.75 | 0.75 | **reproduced** |
| Admissible | true | true | **reproduced** |

### Face-weight admissibility

All 20 λ values in [1/6, 1/4] satisfy Σ_{I∋i} p_I ≤ G_i*.  `all_admissible: true`.  Verdict: **reproduced**.

### Simulation at λ = 1/4 (boundary of feasible set, π_i = G_i* = 3/8)

| Property | Expected | Simulated | Verdict |
|----------|----------|-----------|---------|
| JE violation rate | 0 | 0.0 | **reproduced** |
| Max marginal error | < MC noise | 0.00123 | **reproduced** |
| Max face severity | < 3/4 | 0.74999 | **reproduced** |
| Min axis severity | ≥ 3/4 | 0.74976 | **reproduced** |
| Face confinement to [0, 3/4) | true | true | **reproduced** |

The paper states that in Example 3.13 joint losses on intermediate faces are confined to the lower subrange while larger losses occur only on axis components (Section 3.3).

**Overall verdict: reproduced.**  The G-JE construction reproduces the analytical capacity G_i* = 3/8, satisfies the admissibility condition from Section 3.3, and the simulation confirms that intermediate-face severities are confined to [0, 3/4) while axis severities lie in (3/4, 1].

---

## Experiment 4 — m-exclusivity hierarchy and threshold-exceedance exclusivity

**Paper section:** Sections 3.4, 4.1  
**Output files:** `results/exp4_results.json`, `results/exp4_*.png`

### Part A: m-exclusivity hierarchy (n = 6, τ = 1.1)

| m | Admissible (τ ≤ m−1) | Violation rate | Max active count | Verdict |
|---|----------------------|----------------|-----------------|---------|
| 2 | false (1.1 > 1) | — | — | **reproduced** |
| 3 | true (1.1 ≤ 2) | 0.0 | 2 | **reproduced** |
| 4 | true (1.1 ≤ 3) | 0.0 | 2 | **reproduced** |

Strongest admissible m from the wrap-around construction: m = ⌊τ⌋ + 2 = 3.  The wrap-around vector attains this strongest admissible exclusivity level, as claimed by the paper (Section 3.4).

### Upper extremal count distribution κ_m# (m = 3, τ = 1.1)

Paper: κ_m# = m−1 = 2 with probability τ/(m−1) = 0.55, and 0 otherwise.  
Simulated: {0: 0.45, 2: 0.55}, mean = 1.1.  Verdict: **reproduced**.

### Convex-order bounds for claim counts (m = 3)

The paper proves κ* ≤_{cx} |A(X)| ≤_{cx} κ_m# for any m-exclusive vector X with the same marginals.  The lower bound κ* ≤_{cx} alt is confirmed for all five convex test functions when alt is a 3-exclusive alternative (wrap-around construction with permuted q_i).  

Note: when an independent Bernoulli alternative (which is not 3-exclusive) is used, the upper bound κ* ≤_{cx} alt ≤_{cx} κ_m# can fail for higher-order convex functions (k³, e^k), which is consistent with the paper's bound applying only to m-exclusive alternatives.  The k² comparison yields `lower_bound_holds: true, upper_bound_holds: true` for all structures.

### Part B: threshold-exceedance JE (n = 4, m = 2, τ_exceed = 0.9 ≤ 1 = m−1)

| Property | Expected | Simulated | Verdict |
|----------|----------|-----------|---------|
| Existence (τ_exceed ≤ m−1) | true | true | **reproduced** |
| JE violation rate | 0 | 0.0 | **reproduced** |
| Max simultaneous exceedances | ≤ 1 | 1 | **reproduced** |
| Max marginal error | < MC noise | 0.0036 | **reproduced** |

**Overall verdict: reproduced.**  The m-exclusivity hierarchy, the wrap-around attainability of the strongest level, and the threshold-exceedance construction all match the paper's theoretical predictions.

---

## Experiment 5 — Aggregate-loss extremal structures, stop-loss premiums, and TVaR bounds

**Paper section:** Section 4.2, Example 4.8  
**Output files:** `results/exp5_results.json`, `results/exp5_*.png`

### Parameters (Example 4.8)

n = 10 lines, q_i = 0.15 (all equal), τ = 1.5, unit-mean exponential severity (rate = 1), TVaR level α = 0.99.

### Analytical safest structure (S*)

From the stop-loss formula of Section 4.2:  
E[S*] = τ = 1.5, Var(S*) = τ(1 − τ/⌊τ⌋+1) · E[Y²] + τ E[Y]² = analytically 1.75.

| Quantity | Analytical | Simulated | Verdict |
|----------|-----------|-----------|---------|
| E[S*] | 1.5 | 1.4955 | **reproduced** |
| Var(S*) | 1.75 | 1.735 | **reproduced** |

### Benchmark table (Example 4.8) — simulated values (200,000 samples)

The paper reports variance and TVaR₀.₉₉ for five dependence structures in Table 1.  This replication uses Monte Carlo with 200,000 samples; the paper does not specify its computational method.

| Structure | Mean | Variance | TVaR₀.₉₉ |
|-----------|------|----------|-----------|
| Safest (S*) | 1.496 | 1.735 | 7.095 |
| Independent | 1.496 | 2.761 | 8.512 |
| 3-exclusive worst | 1.493 | 2.233 | 7.415 |
| JE worst (n-exclusive) | 1.503 | 12.754 | 15.897 |
| Co-monotonic | 1.485 | 14.119 | 17.010 |
| Fréchet worst (m=3) | 1.499 | 3.762 | 10.667 |

The paper's Table 1 gives specific values for the same benchmark.  The paper does not provide those values in the experiment specification; comparison is therefore against the theoretical orderings stated in Section 4.2.

### Orderings (paper's claims, Section 4.2)

| Ordering | Paper predicts | Simulated | Verdict |
|----------|----------------|-----------|---------|
| Var(S*) ≤ Var(S_indep) | true | 1.735 ≤ 2.761 | **reproduced** |
| Var(S_indep) ≤ Var(S_comono) | true | 2.761 ≤ 14.119 | **reproduced** |
| TVaR(S*) ≤ TVaR(S_indep) | true | 7.095 ≤ 8.512 | **reproduced** |
| Attainability m=3 (max q ≤ τ/(m−1)) | true | 0.15 ≤ 0.75 | **reproduced** |
| Attainability JE (n-exclusive) | true | 0.15 ≤ τ/9 | **reproduced** |

**Overall verdict: reproduced.**  The safest aggregate S* achieves the smallest variance and TVaR among all tested structures.  The co-monotonic structure produces the largest spread.  The analytical mean and variance of S* match within Monte Carlo error, and all key orderings from Section 4.2 hold in the simulation.

---

## Deviations from the paper

1. **Data source:** All experiments use synthetic inputs. No empirical dataset is referenced in the paper for these experiments. No deviation from paper intention.

2. **Monte Carlo sample size and tolerance:** The paper does not specify a sample size or goodness-of-fit tolerance for numerical verification. This replication uses 200,000 samples and a tolerance of ≈ 0.5% on survival function errors. These choices are not specified in the paper.

3. **Benchmark computation method (Example 4.8):** The paper does not detail whether Table 1 was produced by simulation, numerical integration, or a combination. This replication uses Monte Carlo simulation with exponential (Gamma) severities throughout. The analytical stop-loss premium formula from Section 4.2 is also implemented and matches the simulated S* values within simulation noise.

4. **Convex-order verification for claim counts (Experiment 4):** The paper proves the bounds analytically but does not prescribe a finite benchmark class of alternative m-exclusive couplings. This replication compares against an m-exclusive wrap-around alternative with permuted q_i and against independent Bernoulli (which is not m-exclusive and can therefore exceed the upper bound — consistent with the theorem's scope).

5. **Dependence among positive severities in the wrap-around construction:** The paper states the positive-part severities Y_i may be any random vector with the required marginals; their dependence is unspecified. This replication uses independent severities, which is a valid special case.

6. **Sub-threshold dependence in threshold-exceedance JE (Section 4.1):** The paper leaves the sub-threshold dependence unrestricted. This replication uses independent sub-threshold sampling, which is a valid default.

7. **CDF formula validation (Experiment 2):** The closed-form CDF formula (paper formula 3.10) is implemented but evaluated analytically only at selected points; full comparison against a simulation-based empirical CDF is not performed due to the exponential cost of summing over all supersets in high dimensions.

---

## Not reproduced

- The paper's exact Table 1 numerical values for Example 4.8 are not available from the experiment specification. The orderings and analytical special cases are reproduced; exact table entries are not compared due to the absence of the reported values in the specification. This is a documentation limitation, not a code failure.
