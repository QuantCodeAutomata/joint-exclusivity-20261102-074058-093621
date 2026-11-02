# AGENTS.md — Joint Exclusivity Repository Guide

## What this repository implements

Replication of the main theoretical constructions and numerical experiments from the paper *Joint Exclusivity* (exact authors and year on the paper's title page).  The paper characterises when jointly exclusive (JE) non-negative random vectors exist and provides extremal constructions for aggregate losses.

## Repository layout

```
.
├── README.md
├── AGENTS.md             ← this file
├── requirements.txt
├── .gitignore
├── reproduce.sh          ← runs tests then all five experiments
├── src/                  ← flat library modules (no subdirectories)
│   ├── __init__.py
│   ├── config.py         ← ALL experiment parameters as UPPER_CASE module-level variables
│   ├── je_construction.py
│   ├── canonical_je.py
│   ├── generalised_je.py
│   ├── m_exclusivity.py
│   └── aggregate_loss.py
├── data/
│   ├── __init__.py
│   └── generate_synthetic.py  ← synthetic marginal helpers (no market data)
├── exp/
│   ├── __init__.py
│   ├── exp1.py  ─ Section 3.1: wrap-around construction
│   ├── exp2.py  ─ Section 3.2: canonical JE, Example 3.9
│   ├── exp3.py  ─ Section 3.3: generalised JE, Example 3.13
│   ├── exp4.py  ─ Sections 3.4/4.1: m-exclusivity + threshold exceedance
│   └── exp5.py  ─ Section 4.2: aggregate loss, Example 4.8
├── tests/
│   ├── conftest.py
│   └── test_*.py         ← one file per src/ module
└── results/              ← JSON and PNG output from experiments; RESULTS.md
    └── RESULTS.md
```

## Setup

```bash
pip install -r requirements.txt
```

No API keys are needed.  All experiments use fully synthetic inputs.

## Commands

### Tests

```bash
python -m pytest                        # full suite (98 tests)
python -m pytest -m "not market_data"  # same — no market-data tests in this repo
```

### Experiments (run from repository root)

```bash
python -m exp.exp1   # wrap-around construction (Section 3.1)
python -m exp.exp2   # canonical JE family, Example 3.9 (Section 3.2)
python -m exp.exp3   # generalised JE, Example 3.13 (Section 3.3)
python -m exp.exp4   # m-exclusivity hierarchy (Sections 3.4, 4.1)
python -m exp.exp5   # aggregate loss, Example 4.8 (Section 4.2)
bash reproduce.sh    # run tests then all experiments in sequence
```

Each experiment overwrites its JSON and PNG files in `results/`.  After a rerun the numbers and verdicts in `results/RESULTS.md` must be checked against the new output files and updated by hand.

## Conventions

- **Parameters only in `src/config.py`** as `UPPER_CASE` module-level variables.  Each variable has a comment with its meaning, unit, and source (paper section or "not specified in the paper").
- **Data only through `data/`**: `data/generate_synthetic.py` creates the synthetic marginals used in every experiment.  Do not add loaders or generators in `src/` or `exp/`.
- **No secrets**: no API keys, no market data, no `.env` files needed.
- **Test rules**: unit tests use small synthetic fixtures.  No `@pytest.mark.market_data` tests are present because no live data is used.

## Shared src/ API signatures (call sites must not add kwargs not listed here)

### `src/je_construction`
- `compute_arcs(q_list) → list[tuple[float, float]]` — (start, length) for each arc R_i
- `in_arc(theta, a, q) → bool` — modular membership test
- `sample_je_wraparound(q_list, severity_samplers, n_samples, rng) → ndarray (n_samples, n)`
- `verify_marginals(samples, marginal_survival_fns, x_grid) → (empirical, theoretical)` — survival_fns accept ndarray and return ndarray
- `verify_je_property(samples) → float` — fraction of samples with all positive components
- `verify_claim_count_support(samples, tau) → bool`
- `convex_order_comparison(samples_star, alt_list, test_fns) → dict`

### `src/canonical_je`
- `enumerate_faces(n) → list[frozenset]` — subsets of {0,...,n-1} with 2 ≤ |I| ≤ n-1
- `check_canonical_admissibility(q_list, face_weights) → bool`
- `sample_canonical_je(q, face_weights, face_copulas, severity_samplers, n_samples, rng) → ndarray`
- `canonical_je_cdf_survival(x, q, face_weights, face_survival_copulas, marginal_survival_fns) → float`
- Copula classes: `IndependenceCopula`, `ComonotonicCopula`, `CountermonotonicCopula` — each have `.sample(n, rng) → ndarray`
- `make_example39_marginals(q) → dict` — keys: `q_list`, `severity_samplers`, `marginal_survival_fns`

### `src/generalised_je`
- `compute_capacity_numerical(G_i_fn, F_bar_i_fn, x_grid) → float`
- `check_gje_admissibility(q_list, capacities) → dict`
- `sample_gje(q, marginal_survival_fns, distortions, capacities, face_weights, face_copulas, distorted_severity_samplers, axis_severity_samplers, n_samples, rng) → ndarray`
- `build_active_set_pmf(q, face_weights)` — internal helper
- `make_example313_distorted_sampler(a, b, q) → callable(u, rng)`
- `make_example313_axis_sampler(a, b, q, pi) → callable(u, rng)`

### `src/m_exclusivity`
- `check_m_exclusive_existence(q_list, m) → bool`
- `strongest_admissible_m(q_list) → int`
- `sample_je_wraparound(q_list, severity_samplers, n_samples, rng) → ndarray` (alias to je_construction version)
- `verify_m_exclusive(samples, m) → dict`
- `upper_extremal_count_distribution(tau, m) → dict`
- `sample_upper_extremal_count(tau, m, n_samples, rng) → ndarray`
- `convex_order_bounds_check(q, m, samples_star, samples_alt) → dict`
- `sample_alternative_activity_patterns(q_list, n_samples, rng) → ndarray`
- `sample_threshold_exceedance_je(marginal_survival_fns, thresholds, excess_samplers, sub_samplers, m, n_samples, rng) → ndarray`
- `make_normal_excess_sampler(mu, sigma, threshold) → callable`
- `make_normal_subthreshold_sampler(mu, sigma, threshold) → callable`

### `src/aggregate_loss`
- `stop_loss_gamma(d, k, rate) → float` — E[(Gamma(k,rate) − d)₊]
- `stop_loss_safest(d, tau, rate) → float` — paper's formula, Section 4.2
- `tvar_from_samples(samples, alpha) → float`
- `sample_safest_aggregate(q_list, severity_sampler, n_samples, rng) → ndarray`
- `sample_independent_aggregate(q_list, severity_sampler, n_samples, rng) → ndarray`
- `sample_comonotonic_aggregate(q_list, severity_sampler, n_samples, rng) → ndarray`
- `sample_frechet_worst_case(q_list, m, G_quantile_fn, n_samples, rng) → ndarray`
- `benchmark_table(n, q, rate, alpha, n_samples, rng) → dict`

