"""Shared fixtures. The tiny hand-computed problem is the one from the day-1 lecture."""

import numpy as np
import pytest

from optlab.types import Mat, Vec


@pytest.fixture
def hand_X() -> Mat:
    """X = [[1], [2]] — small enough to differentiate on paper."""
    return np.array([[1.0], [2.0]])


@pytest.fixture
def hand_y_linear() -> Vec:
    """y = (2, 4). With w = 0: value 5, gradient -5."""
    return np.array([2.0, 4.0])


@pytest.fixture
def hand_y_logistic() -> Vec:
    """y = (0, 1). With w = 0: value log 2, gradient -0.25."""
    return np.array([0.0, 1.0])


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded, so a failure is reproducible."""
    return np.random.default_rng(20250921)
