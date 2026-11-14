"""Beat-the-critical-Mach optimisation (aeronet/optimize.py, POST /api/optimize): inviscid, bounded, never worse than the start."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aeronet import optimize
from aeronet.geometry import naca4
from aeronet.main import app


def test_continuous_geometry_matches_the_standard_code():
    af, mine = naca4("2412", 50), optimize.build(0.02, 0.4, 0.12, 50)
    assert np.allclose(af.x, mine.x) and np.allclose(af.y, mine.y)


@pytest.mark.parametrize("target", [0.0, 0.4, 0.9])
def test_design_reaches_the_target_lift(target):
    d = optimize.design_at_cl(0.02, 0.4, 0.10, target)
    assert d["cl"] == pytest.approx(target, abs=2e-6)


def test_digits_gives_the_nearest_standard_code():
    assert optimize._digits(0.0381, 0.69, 0.0905) == "4709"
    assert optimize._digits(0.0, 0.4, 0.12) == "0012"
    assert optimize._digits(0.02, 0.04, 0.1) == "2110"  # camber position is at least 0.1 when there is camber


@pytest.fixture(scope="module")
def result():
    return optimize.optimise(0.6, 0.09, "0012")


def test_optimised_shape_respects_the_bounds_and_the_minimum_thickness(result):
    a = result["after"]
    b = result["bounds"]
    assert b["m"][0] <= a["m"] <= b["m"][1] and b["p"][0] - 1e-9 <= a["p"] <= b["p"][1] and 0.09 - 1e-9 <= a["t"] <= b["t"][1]
    assert abs(a["alpha"]) <= optimize.ALPHA_LIMIT


def test_optimised_shape_gives_the_target_lift_and_a_higher_critical_mach(result):
    assert result["after"]["cl"] == pytest.approx(0.6, abs=1e-5) and result["before"]["cl"] == pytest.approx(0.6, abs=1e-5)
    assert result["after"]["cp_min"] > result["before"]["cp_min"]      # less suction
    assert result["gain_pg"] > 0.05                                    # NACA 0012 at Cl 0.6 has Mcr about 0.48
    assert result["after"]["mach_crit_kt"] < result["after"]["mach_crit_pg"]
    assert "no drag" in result["label"].lower()


def test_history_never_gets_worse_and_the_rounded_code_is_reported(result):
    h = result["history"]
    assert len(h) == result["evaluations"] and all(b >= a - 1e-12 for a, b in zip(h, h[1:]))
    assert result["rounded"]["name"].startswith("NACA ") and len(result["rounded"]["name"]) == len("NACA 4709")


def test_a_thicker_minimum_cannot_beat_a_thinner_one():
    thin = optimize.optimise(0.4, 0.08, "0012")["after"]["cp_min"]
    thick = optimize.optimise(0.4, 0.14, "0012")["after"]["cp_min"]
    assert thin >= thick - 1e-3


def test_endpoint_and_validation():
    c = TestClient(app)
    r = c.post("/api/optimize", json={"target_cl": 0.5, "min_thickness": 0.1, "code": "2412"})
    assert r.status_code == 200 and r.json()["before"]["name"] == "NACA 2412"
    assert c.post("/api/optimize", json={"target_cl": 5, "min_thickness": 0.1}).status_code == 422
    assert c.post("/api/optimize", json={"target_cl": 0.5, "min_thickness": 0.01}).status_code == 422
    assert c.post("/api/optimize", json={"target_cl": 0.5, "min_thickness": 0.1, "code": "abc"}).status_code == 422
