"""Sanity tests of the panel-method solver (port of prototype/test_solver.js) and parity checks
against values produced by the JavaScript prototype."""

import math
import time
from pathlib import Path

import numpy as np
import pytest

from aeronet.compare import compare_cp, parse_reference, thin_airfoil_cl
from aeronet.field import field_grid, inside_body
from aeronet.geometry import naca4, validate_code
from aeronet.panel import solve

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "reference_naca0012_200x100.dat"


def test_symmetric_airfoil_zero_lift_at_zero_alpha():
    assert abs(solve("0012", 0, 0).cl) < 1e-3


def test_naca0012_lift_at_4_degrees():
    r = solve("0012", 4, 0)
    assert 0.44 < r.cl < 0.50
    assert r.cl == pytest.approx(0.4822, abs=1e-3)  # value from the JS prototype
    assert thin_airfoil_cl(r.airfoil, 4, 0) == pytest.approx(0.4386, abs=1e-3)


def test_dalembert_drag_is_near_zero():
    assert abs(solve("0012", 4, 0).cd) < 0.01


def test_cambered_airfoil_lifts_at_zero_alpha():
    r = solve("2412", 0, 0)
    assert 0.20 < r.cl < 0.30
    assert r.cl == pytest.approx(0.2555, abs=1e-3)


def test_antisymmetry():
    assert solve("0012", 6, 0).cl == pytest.approx(-solve("0012", -6, 0).cl, abs=1e-6)


def test_prandtl_glauert_scaling():
    ratio = solve("0012", 4, 0.4).cl / solve("0012", 4, 0).cl
    assert ratio == pytest.approx(1 / math.sqrt(1 - 0.16), abs=1e-6)


def test_kutta_condition():
    r = solve("0012", 4, 0)
    assert abs(r.cp[0] - r.cp[-1]) < 0.05


def test_stagnation_pressure_near_one():
    assert 0.95 < solve("0012", 4, 0).cp.max() < 1.02


def test_solver_is_fast():
    solve("2412", 3, 0.3)  # warm-up
    t = time.perf_counter()
    for _ in range(10):
        solve("2412", 3, 0.3)
    assert (time.perf_counter() - t) / 10 < 0.15


def test_prototype_parity_at_mach_04():
    assert solve("0012", 4, 0.4, 80).cl == pytest.approx(0.526, abs=1e-3)


def test_geometry_is_counter_clockwise_polygon():
    af = naca4("0012", 80)
    assert af.x.size == 160 and af.y.size == 160
    assert af.x[0] == pytest.approx(1.0) and af.y[0] == pytest.approx(0.0, abs=1e-9)
    area = 0.5 * np.sum(af.x * np.roll(af.y, -1) - np.roll(af.x, -1) * af.y)
    assert area > 0  # counter-clockwise: interior lies to the left of every panel


@pytest.mark.parametrize("code,ok", [("0012", True), ("2412", True), ("2315", True), ("0003", False), ("2015", False), ("12", False), ("abcd", False)])
def test_code_validation(code, ok):
    if ok:
        assert validate_code(code) == code
    else:
        with pytest.raises(ValueError):
            validate_code(code)


def test_reference_parse_and_shift():
    pts = parse_reference(SAMPLE.read_text(encoding="utf8"))
    assert pts.shape == (200, 3)
    assert 0.0 <= pts[:, 0].min() and pts[:, 0].max() <= 1.0  # shifted from -0.5..0.5
    assert parse_reference("header only\nnot numbers here\n").size == 0


def test_comparison_with_bundled_reference_matches_prototype():
    ref = parse_reference(SAMPLE.read_text(encoding="utf8"))
    c = compare_cp(solve("0012", 4, 0.4, 80), ref)
    # prototype (JS) gave RMS 0.019, max 0.136 on 162 points for this file
    assert c["n"] == 162
    assert c["rms"] == pytest.approx(0.019, abs=0.002)
    assert c["max"] == pytest.approx(0.136, abs=0.01)


def test_field_grid_masks_body_and_is_finite_outside():
    g = field_grid(solve("0012", 4, 0.4), 100)
    assert g["width"] == 100 and g["height"] == 56 and len(g["cp"]) == 100 * 56
    inside = [v is None for v in g["cp"]]
    assert 0 < sum(inside) < 0.1 * len(inside)
    vals = np.array([v for v in g["cp"] if v is not None])
    assert np.isfinite(vals).all() and vals.min() < 0 < vals.max()


def test_inside_body_point_tests():
    af = naca4("0012", 70)
    assert inside_body(af.x, af.y, np.array([0.5, 0.5, 1.2, -0.1]), np.array([0.0, 0.5, 0.0, 0.0])).tolist() == [True, False, False, False]
