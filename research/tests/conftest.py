import os

os.environ["MPLBACKEND"] = "Agg"  # must be set before anything imports matplotlib

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "NNfoil", REPO / "research", REPO / "app" / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(scope="session")
def repo():
    return REPO


@pytest.fixture(scope="session")
def mesh_50x25(repo):
    import metrics

    return metrics.load_mesh(repo / "NNfoil" / "mesh" / "NACA0012_50_25.x")
