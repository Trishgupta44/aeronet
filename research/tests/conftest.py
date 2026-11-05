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


@pytest.fixture(scope="session")
def tiny_run(repo, tmp_path_factory):
    """A 2-round run of a tiny untrained-quality network on 50x25 (plumbing only, numbers are meaningless)."""
    import run_case

    out = tmp_path_factory.mktemp("runs") / "tiny"
    rc = run_case.main(["--mesh", str(repo / "NNfoil" / "mesh" / "NACA0012_50_25.x"), "--shape", "NACA0012",
                        "--mach", "0.4", "--alpha", "4", "--layers", "2,8,8,4", "--lbfgs-iter", "5",
                        "--rounds", "2", "--threads", "2", "--out", str(out)])
    assert rc == 0
    return out
