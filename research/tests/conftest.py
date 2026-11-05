import os

os.environ["MPLBACKEND"] = "Agg"  # must be set before anything imports matplotlib

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "NNfoil", REPO / "research"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_mesh_inline(path):
    # Same convention as the upstream main script. Replaced by metrics.load_mesh in A7.
    xxyy = np.loadtxt(path, skiprows=1)
    h = xxyy.shape[0] // 2
    xx = xxyy[:h].T
    yy = xxyy[h:].T
    xx = np.r_[xx, xx[0:1]]
    yy = np.r_[yy, yy[0:1]]
    return xx, yy


@pytest.fixture(scope="session")
def repo():
    return REPO


@pytest.fixture(scope="session")
def mesh_50x25(repo):
    return _load_mesh_inline(repo / "NNfoil" / "mesh" / "NACA0012_50_25.x")
