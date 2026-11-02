"""
Configuration for all Joint Exclusivity experiments.
All parameters are module-level variables with comments citing their source.
"""

# ---------------------------------------------------------------------------
# Global Monte Carlo settings
# ---------------------------------------------------------------------------

# Number of Monte Carlo samples for all simulations; not specified in the paper
N_MC_SAMPLES = 200_000

# Random seed for reproducibility; not specified in the paper
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Experiment 1: Wrap-around construction and JE existence (Section 3.1)
# ---------------------------------------------------------------------------

# Dimension n for exp_1 illustration; the paper does not fix n for examples
EXP1_N = 5

# Claim probabilities q_i for exp_1 illustration; chosen so that tau <= n-1
# and the admissibility condition is satisfied; not specified in the paper
EXP1_Q = [0.3, 0.4, 0.35, 0.25, 0.2]  # sum = 1.5 <= 4 = n-1

# Exponential rate parameters (positive severity part) for exp_1; not specified in paper
EXP1_RATES = [1.0, 0.5, 2.0, 1.5, 0.8]

# x-grid for marginal verification in exp_1; not specified in paper
EXP1_X_GRID_POINTS = 50

# Tolerance for JE violation rate (should be 0 up to Monte Carlo noise)
EXP1_JE_TOLERANCE = 1e-3

# Alternative coupling count for small-n convex-order comparisons; not specified
EXP1_N_ALTERNATIVES = 10

# Convex test functions for convex-order comparisons (exponent coefficients)
EXP1_CONVEX_LAMBDA = [0.5, 1.0, 2.0]

# ---------------------------------------------------------------------------
# Experiment 2: Canonical JE family, trivariate example (Section 3.2)
# ---------------------------------------------------------------------------

# Dimension n for canonical JE trivariate example (Example 3.9, Section 3.2)
EXP2_N = 3

# Claim probability per component for Example 3.9; each X_i is zero w.p. 1/2,
# uniform on (0,1) otherwise, so q_i = P(X_i > 0) = 0.5 (Section 3.2, Example 3.9)
EXP2_Q = [0.5, 0.5, 0.5]  # tau = 1.5 <= n-1 = 2

# Lambda range for face-weight parametrization p_I(lambda) = lambda/2
# (Example 3.9, Section 3.2); lambda in [1/3, 1/2]
EXP2_LAMBDA_MIN = 1.0 / 3.0
EXP2_LAMBDA_MAX = 1.0 / 2.0

# Number of lambda grid points for correlation curve; not specified in paper
EXP2_N_LAMBDA = 20

# Grid for CDF validation; not specified in paper
EXP2_CDF_GRID_POINTS = 10

# ---------------------------------------------------------------------------
# Experiment 3: Generalised JE with distortions (Section 3.3)
# ---------------------------------------------------------------------------

# Dimension n for G-JE trivariate example (Example 3.13, Section 3.3)
EXP3_N = 3

# Same marginals as Example 3.9: q_i = 0.5, positive part Uniform(0,1)
EXP3_Q = [0.5, 0.5, 0.5]

# Linear truncation parameters a_i and b_i (Example 3.13, Section 3.3)
EXP3_A = 1.0 / 8.0   # lower truncation point; a_i = 1/8
EXP3_B = 1.0 / 2.0   # upper truncation point; b_i = bar{F}_i(0) = 1/2

# Capacity G_i* for linear truncation example (Example 3.13, Section 3.3): 3/8
EXP3_CAPACITY = 3.0 / 8.0

# Lambda range for face weights p_I(lambda) = lambda/4 + 1/8
# (Example 3.13, Section 3.3): lambda in [1/6, 1/4]
EXP3_LAMBDA_MIN = 1.0 / 6.0
EXP3_LAMBDA_MAX = 1.0 / 4.0

# Number of x-grid points for capacity numerical computation; not specified
EXP3_CAPACITY_GRID = 1000

# Upper bound on x for confinement check; active face losses should be < 3/4
# (Example 3.13, Section 3.3)
EXP3_CONFINEMENT_UPPER = 3.0 / 4.0

# ---------------------------------------------------------------------------
# Experiment 4: m-exclusivity hierarchy (Sections 3.4, 4.1)
# ---------------------------------------------------------------------------

# Dimension n for m-exclusivity tests; not specified in paper
EXP4_N = 6

# Claim probabilities for m-exclusivity illustration; not specified in paper
EXP4_Q = [0.3, 0.25, 0.2, 0.15, 0.1, 0.1]  # sum = 1.1

# m values to test in the hierarchy; not specified in paper
EXP4_M_VALUES = [2, 3, 4]

# Threshold exceedance parameters: marginals are standard normal,
# thresholds at quantiles; not specified in paper
EXP4_THRESH_N = 4
EXP4_THRESH_ALPHA = [0.8, 0.75, 0.7, 0.85]  # quantile levels for l_i = Phi^{-1}(alpha_i)

# m for threshold exceedance example; not specified in paper
EXP4_THRESH_M = 2

# ---------------------------------------------------------------------------
# Experiment 5: Aggregate loss, stop-loss, TVaR (Section 4.2, Example 4.8)
# ---------------------------------------------------------------------------

# Number of lines for the benchmark (Example 4.8, Section 4.2)
EXP5_N = 10

# Claim probability per line (Example 4.8, Section 4.2): q_i = 0.15
EXP5_Q = 0.15

# Rate of unit-mean exponential severity (Example 4.8, Section 4.2): rate=1
EXP5_SEVERITY_RATE = 1.0

# Exclusivity levels tested: m=3 (3-exclusive worst), m=n=10 (JE worst)
# (Example 4.8, Section 4.2)
EXP5_M_3EXCL = 3
EXP5_M_JE = 10  # JE = n-exclusive

# Confidence level for TVaR (Example 4.8, Section 4.2)
EXP5_TVAR_ALPHA = 0.99

# Retention grid for stop-loss curves; not specified in paper
EXP5_D_MAX = 10.0
EXP5_D_POINTS = 200

# ---------------------------------------------------------------------------
# Tolerance thresholds (not specified in paper)
# ---------------------------------------------------------------------------

# Tolerance for marginal survival function verification (Monte Carlo error)
MARGINAL_TOLERANCE = 0.02

# Tolerance for JE property verification (should be essentially 0)
JE_TOLERANCE = 5e-3
