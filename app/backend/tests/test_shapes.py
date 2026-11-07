"""Generic-shape solver and outline preparation (Plan B, task B1), then the shape library (B6)."""

from pathlib import Path

import numpy as np
import pytest

from aeronet.compare import compare_cp_nearest, parse_reference
from aeronet.geometry import naca4
from aeronet.panel import solve, solve_xy
from aeronet.shapes import chaikin, normalize_shape, prepare_shape, self_intersects, signed_area

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "reference_naca0012_200x100.dat"


def ellipse(a, b, n, cx=0.0, cy=0.0, clockwise=False):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if clockwise:
        t = -t
    return np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)])


def figure_eight(n=200):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.sin(t), 0.3 * np.sin(2 * t)])


def test_normalize_clockwise_offset_ellipse():
    pts = ellipse(3.0, 0.75, 77, cx=5.0, cy=-2.0, clockwise=True)  # clockwise, offset, scaled
    X, Y = normalize_shape(pts, n=120)
    assert X.size == Y.size == 120
    assert X.min() == pytest.approx(0.0, abs=1e-12) and X.max() == pytest.approx(1.0, abs=1e-12)
    assert (X[0], Y[0]) == (pytest.approx(1.0), pytest.approx(0.0, abs=1e-12))
    assert signed_area(X, Y) > 0


def test_circle_non_lifting_matches_potential_flow_theory():
    X, Y = normalize_shape(ellipse(0.5, 0.5, 300), n=200)
    s = solve_xy(X, Y, 0.0, 0.0, lifting=False)
    assert s.cp.max() == pytest.approx(1.0, abs=0.01)  # measured 0.9989
    assert s.cp.min() == pytest.approx(-3.0, abs=0.01)  # measured -2.9989
    assert abs(s.cl) < 1e-6 and abs(s.cd) < 1e-3
    assert s.airfoil is None and s.ground_y is None


def test_reversed_naca0012_points_match_the_code_solver():
    af = naca4("0012", 120)
    X, Y = normalize_shape(np.column_stack([af.x, af.y])[::-1], n=240)
    s = solve_xy(X, Y, 4.0, 0.4)
    assert s.cl == pytest.approx(solve("0012", 4, 0.4, 120).cl, abs=0.005)  # measured 0.5247 vs 0.5263


def test_upper_flag_marks_the_first_half_of_the_panels():
    up = solve("0012", 4, 0, 70).upper
    assert up.dtype == bool and up.size == 140
    assert up[:70].all() and not up[70:].any()


def test_self_intersection_detection():
    X, Y = figure_eight().T
    assert self_intersects(X, Y)
    X, Y = ellipse(1.0, 0.3, 200).T
    assert not self_intersects(X, Y)


def test_prepare_shape_errors():
    with pytest.raises(ValueError, match="at least 8"):
        prepare_shape([(0, 0), (1, 0), (0.5, 0.3)])
    with pytest.raises(ValueError, match="crosses itself"):
        prepare_shape(figure_eight())
    line = [(i / 9, 0.0) for i in range(10)]
    with pytest.raises(ValueError, match="too thin"):
        prepare_shape(line)
    with pytest.raises(ValueError, match="does not fit"):
        prepare_shape(ellipse(0.05, 0.5, 100))  # 0.1 wide, 1 high
    with pytest.raises(ValueError):
        prepare_shape([(0, 0)] * 7 + [(float("nan"), 1)] + [(1, 1), (2, 3)])


def test_prepare_shape_info_and_kutta_warning():
    X, Y, info, warnings = prepare_shape(ellipse(2.0, 0.5, 300), lifting=True)
    assert X.size == 200
    assert info["chord"] == pytest.approx(4.0, abs=0.01)
    assert info["thickness"] == pytest.approx(0.25, abs=0.005)
    assert info["n_input"] == 300
    assert info["area"] == pytest.approx(np.pi * 0.5 * 0.125, abs=0.005)
    assert any("Kutta" in w for w in warnings)
    # lift off: no warning
    assert prepare_shape(ellipse(2.0, 0.5, 300), lifting=False)[3] == []
    af = naca4("0012", 80)
    assert prepare_shape(np.column_stack([af.x, af.y]))[3] == []


def test_chaikin_cuts_the_corners_of_a_square():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    out = chaikin(sq, 2)
    assert out.shape == (16, 2)
    assert out.min() >= 0 and out.max() <= 1
    for c in sq:
        assert (np.hypot(*(out - c).T) > 0.05).all()


def test_compare_cp_nearest_on_the_bundled_sample():
    ref = parse_reference(SAMPLE.read_text(encoding="utf8"))
    c = compare_cp_nearest(solve("0012", 4, 0.4, 80), ref)
    assert c["n"] == 162
    assert c["rms"] == pytest.approx(0.0231, abs=0.002)
    assert compare_cp_nearest(solve("0012", 4, 0.4, 80), np.empty((0, 3))) == {"rms": None, "max": None, "n": 0}
