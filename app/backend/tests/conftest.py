import pytest

from aeronet import limits


@pytest.fixture(autouse=True)
def _fresh_limits():
    """The per-address and concurrency counters of the expensive endpoints start empty in every test."""
    limits.reset()
    yield
    limits.reset()
