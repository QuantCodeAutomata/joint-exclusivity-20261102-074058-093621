#!/usr/bin/env bash
# Run the full test suite and all five experiments from the repository root.
set -u

echo "=== Installing dependencies ==="
pip install -r requirements.txt --quiet

echo "=== Running test suite ==="
python -m pytest -m "not market_data" -v

echo "=== Experiment 1: Wrap-around construction (Section 3.1) ==="
python -m exp.exp1

echo "=== Experiment 2: Canonical JE family, Example 3.9 (Section 3.2) ==="
python -m exp.exp2

echo "=== Experiment 3: Generalised JE, Example 3.13 (Section 3.3) ==="
python -m exp.exp3

echo "=== Experiment 4: m-exclusivity hierarchy (Sections 3.4, 4.1) ==="
python -m exp.exp4

echo "=== Experiment 5: Aggregate loss, Example 4.8 (Section 4.2) ==="
python -m exp.exp5

echo "=== Done. Results written to results/ ==="
