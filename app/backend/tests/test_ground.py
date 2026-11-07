"""Ground effect by the method of images (Plan B, task B2; the field-mask test comes with B4)."""

import numpy as np
import pytest

from aeronet.field import field_uv
from aeronet.geometry import naca4
from aeronet.panel import solve, solve_xy
from aeronet.shapes import normalize_shape


def _ellipse(a=0.5, b=0.1, n=300):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return normalize_shape(np.column_stack([a * np.cos(t), b * np.sin(t)]), n=200)


ELLIPSE = _ellipse()  # 1 x 0.2, normalised to 200 points


def _naca(code="0012", n_half=70):
    af = naca4(code, n_half)
    return af.x, af.y


def test_road_is_a_streamline():
    s = solve_xy(*ELLIPSE, 0.0, 0.0, lifting=False, ground=0.05)
    px = np.linspace(-0.3, 1.3, 9)
    py = np.full(9, s.ground_y)
    _, v = field_uv(s, px, py)
    assert np.abs(v).max() < 1e-9  # measured exactly 0


def test_far_road_changes_nothing():
    X, Y = _naca()
    free = solve_xy(X, Y, 4.0, 0.0).cl
    assert solve_xy(X, Y, 4.0, 0.0, ground=200.0).cl == pytest.approx(free, abs=1e-3)  # 0.4821 vs 0.4822
    assert abs(solve_xy(*ELLIPSE, 0.0, 0.0, lifting=False, ground=50.0).cl) < 1e-4


def test_ellipse_is_sucked_down_near_the_road():
    assert abs(solve_xy(*ELLIPSE, 0.0, 0.0, lifting=False).cl) < 1e-9
    assert solve_xy(*ELLIPSE, 0.0, 0.0, lifting=False, ground=0.05).cl < -0.1  # measured -0.233


def test_freestream_stays_horizontal_and_road_sits_below_the_body():
    s = solve_xy(*ELLIPSE, 5.0, 0.0, lifting=False, ground=0.1)
    assert s.ground_y == pytest.approx(s.Y.min() - 0.1)
    assert s.vx_inf == 1.0 and s.vy_inf == 0.0


def test_alpha_rotates_the_body_nose_up():
    X, Y = _naca()
    s = solve_xy(X, Y, 6.0, 0.0, ground=0.5)
    le, te = np.argmin(s.X), 0
    assert s.Y[le] > s.Y[te]
    # rotation about the quarter chord keeps the chord length
    assert np.hypot(s.X[le] - s.X[te], s.Y[le] - s.Y[te]) == pytest.approx(1.0, abs=1e-3)
    assert solve_xy(X, Y, 0.0, 0.0, ground=0.5).ground_y == pytest.approx(Y.min() - 0.5)


def test_ground_effect_reference_numbers():
    X, Y = _naca(n_half=80)  # the reference values were measured with 160 panels
    cl = {gap: solve_xy(X, Y, 4.0, 0.0, ground=gap).cl for gap in (1.0, 0.3, 0.1, 0.05)}
    assert cl[1.0] == pytest.approx(0.4898, abs=5e-4)
    assert cl[0.3] == pytest.approx(0.5300, abs=5e-4)
    assert cl[0.1] == pytest.approx(0.5321, abs=5e-4)
    assert cl[0.05] == pytest.approx(0.4511, abs=5e-4)


def test_no_ground_keeps_the_old_behaviour():
    assert solve("0012", 4, 0).ground_y is None
