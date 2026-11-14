"""Critical pressure coefficient and critical Mach number (aeronet/compressibility.py)."""

import math

import pytest
from fastapi.testclient import TestClient

from aeronet import compressibility as c
from aeronet import panel
from aeronet.main import app


def test_cp_crit_at_mach_half_matches_the_isentropic_formula():
    # 2/(1.4*0.25) * ((1 + 0.2*0.25)/1.2)^3.5 - 1)  (hand-evaluated: -2.1334)
    expect = 2.0 / (1.4 * 0.25) * (((1.0 + 0.2 * 0.25) / 1.2) ** 3.5 - 1.0)
    assert c.cp_crit(0.5) == pytest.approx(expect, rel=1e-12)
    assert c.cp_crit(0.5) == pytest.approx(-2.1334, abs=5e-4)


def test_cp_crit_other_points():
    assert c.cp_crit(1.0) == pytest.approx(0.0, abs=1e-12)  # at Mach 1 the sonic point is the free stream
    assert c.cp_crit(0.7) == pytest.approx(-0.7791, abs=5e-4)
    assert c.cp_crit(0.0) is None
    values = [c.cp_crit(m) for m in (0.2, 0.4, 0.6, 0.8, 0.95)]
    assert all(a < b for a, b in zip(values, values[1:]))  # rises toward 0 as Mach grows


@pytest.mark.parametrize("rule", [c.prandtl_glauert, c.karman_tsien])
@pytest.mark.parametrize("cp0", [-0.05, -0.4127, -1.0, -3.0])
def test_critical_mach_is_where_the_corrected_cp_meets_cp_crit(rule, cp0):
    m = c.critical_mach(cp0, rule)
    assert 0.0 < m < 1.0
    assert rule(cp0, m) == pytest.approx(c.cp_crit(m), abs=1e-9)


def test_karman_tsien_gives_the_lower_critical_mach():
    for cp0 in (-0.2, -0.4127, -1.0):
        assert c.critical_mach(cp0, c.karman_tsien) < c.critical_mach(cp0, c.prandtl_glauert)


def test_no_suction_means_no_critical_mach():
    assert c.critical_mach(0.1) is None
    assert c.critical_mach(0.0) is None


def test_naca0012_at_zero_degrees_has_mcr_near_0_73_by_prandtl_glauert():
    """Panel Cp0_min of NACA 0012 at 0 deg is -0.41; Prandtl-Glauert then crosses Cp* at Mach 0.74 and Karman-Tsien at 0.73.

    Cross-checks (not exact references): a student wind-tunnel report quotes Mcr about 0.73 for this airfoil, and NASA
    quotes a drag-rise Mach number of about 0.70 (a different quantity, always above Mcr). The textbook Mcr value was not
    verified from the book itself, so only a band is asserted here.
    """
    s = panel.solve("0012", 0.0, 0.0)
    v = c.validity(s.cp, s.beta)
    assert v["cp0_min"] == pytest.approx(-0.41, abs=0.02)
    assert 0.70 < v["mach_crit_pg"] < 0.78
    assert 0.68 < v["mach_crit_kt"] < v["mach_crit_pg"]


def test_validity_flags_supersonic_flow_only_above_mcr():
    low = panel.solve("0012", 4.0, 0.4)   # Mcr (PG) about 0.53, so still fine at 0.4
    high = panel.solve("0012", 4.0, 0.6)  # above it
    assert c.validity(low.cp, low.beta)["supersonic"] is False
    v = c.validity(high.cp, high.beta)
    assert v["supersonic"] is True and v["cp_min"] < v["cp_crit"]
    assert v["mach_crit_pg"] == pytest.approx(c.validity(low.cp, low.beta)["mach_crit_pg"], abs=1e-6)  # Mcr does not depend on the Mach asked for


def test_validity_at_mach_zero_has_no_cp_crit_and_no_warning():
    s = panel.solve("0012", 4.0, 0.0)
    v = c.validity(s.cp, s.beta)
    assert v["cp_crit"] is None and v["supersonic"] is False and math.isfinite(v["mach_crit_pg"])


def test_api_solve_returns_validity():
    r = TestClient(app).post("/api/solve", json={"code": "0012", "alpha": 4, "mach": 0.6})
    assert r.status_code == 200
    v = r.json()["validity"]
    assert v["supersonic"] is True and v["cp_crit"] == pytest.approx(c.cp_crit(0.6))
    assert 0.5 < v["mach_crit_pg"] < 0.56 and v["mach_crit_kt"] < v["mach_crit_pg"]
