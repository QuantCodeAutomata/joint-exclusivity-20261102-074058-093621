"""
Shared pytest configuration and fixtures for the joint exclusivity test suite.
All tests use synthetic data; no market-data API keys are required.
"""

import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "market_data: mark test as requiring a live market-data API key"
    )


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Fixed-seed generator shared across the session."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def small_q() -> list:
    """Admissible claim probabilities, tau = 1.5 <= n-1 = 4 for n = 5."""
    return [0.3, 0.4, 0.35, 0.25, 0.2]


@pytest.fixture(scope="session")
def trivariate_q() -> list:
    """Example 3.9 / 3.13 marginals: n=3, q_i=0.5, tau=1.5."""
    return [0.5, 0.5, 0.5]
