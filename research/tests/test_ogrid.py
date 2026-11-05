import time

import numpy as np
import pytest

import metrics
import ogrid

DEEP_C = ([0.5, -0.5, -0.5, 0.5, 0.5, -0.3, -0.3, 0.5], [0.3, 0.3, -0.3, -0.3, -0.2, -0.2, 0.2, 0.2])
SEDAN = ([0.5, 0.45, 0.2, 0.05, -0.2, -0.42, -0.5, -0.48, 0.5], [0, 0.08, 0.1, 0.2, 0.2, 0.08, 0.02, -0.06, -0.06])


def _circle(n=400, r=0.5):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return r * np.cos(t), r * np.sin(t)


def _polyline_resampled(shape, n=50):
    return ogrid.resample_closed(np.array(shape[0], float), np.array(shape[1], float), n)


def test_upstream_coefficient_reproduces_bundled_mesh(mesh_50x25):
    xx, yy = mesh_50x25  # (N+1, M), NNfoil frame
    X, Y, iters, change = ogrid.ogrid(xx[:-1, 0], yy[:-1, 0], 25, cross_coeff=0.5)
    assert X.shape == Y.shape == (25, 50)
    assert change < 1e-6
    err = max(np.abs(X - xx[:-1].T).max(), np.abs(Y - yy[:-1].T).max())
    assert err < 1e-3  # measured 3.2e-4: the shipped meshes used the upstream coefficient 0.5


def test_corrected_coefficient_moves_the_nodes_but_stays_valid(mesh_50x25):
    xx, yy = mesh_50x25
    X, Y, _, _ = ogrid.ogrid(xx[:-1, 0], yy[:-1, 0], 25, cross_coeff=2.0)
    err = max(np.abs(X - xx[:-1].T).max(), np.abs(Y - yy[:-1].T).max())
    assert 1e-3 < err < 0.05
    assert ogrid.is_valid(X, Y)


def test_cosine_wall_from_the_200x100_wall_equals_the_50x25_wall(repo):
    x2, y2 = metrics.load_mesh(repo / "NNfoil" / "mesh" / "NACA0012_200_100.x")
    x5, y5 = metrics.load_mesh(repo / "NNfoil" / "mesh" / "NACA0012_50_25.x")
    X, Y = ogrid.cosine_wall(x2[:-1, 0], y2[:-1, 0], 50)
    assert X.shape == Y.shape == (50,)
    assert np.abs(X - x5[:-1, 0]).max() < 1e-12
    assert np.abs(Y - y5[:-1, 0]).max() < 1e-12


def test_cosine_wall_naca2412_is_cambered_and_keeps_orientation(repo):
    x, y = metrics.load_mesh(repo / "NNfoil" / "mesh" / "NACA2412_200_100.x")
    X, Y = ogrid.cosine_wall(x[:-1, 0], y[:-1, 0], 100)
    assert np.allclose(X, 0.5 * np.cos(2 * np.pi * np.arange(100) / 100))
    assert Y[:50].max() > 0.05 and Y[50:].min() < -0.01  # upper branch first (CCW from the trailing edge)
    assert Y[:50].max() + Y[50:].min() > 0.02  # cambered: not symmetric


def test_resample_closed_normalises_a_shape():
    x, y = np.array(SEDAN[0]), np.array(SEDAN[1])
    X, Y = ogrid.resample_closed(x, y, 60)
    assert X.shape == Y.shape == (60,)
    assert X[0] == pytest.approx(0.5) and Y[0] == pytest.approx(0.0)  # starts at the rightmost point, TE at y = 0
    assert X.min() >= -0.5 - 1e-12 and X.max() <= 0.5 + 1e-12
    area = 0.5 * np.sum(X * np.roll(Y, -1) - np.roll(X, -1) * Y)
    assert area > 0  # counter-clockwise
    seg = np.hypot(np.diff(np.r_[X, X[0]]), np.diff(np.r_[Y, Y[0]]))
    assert seg.std() / seg.mean() < 0.15  # (nearly) uniform arc length; corners shorten chords


def test_resample_closed_accepts_reversed_order_duplicate_end_point_and_other_scale():
    t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    x, y = 3.0 * np.cos(t) + 7.0, 1.5 * np.sin(t) - 2.0  # an ellipse of chord 6, off centre
    a = ogrid.resample_closed(x, y, 40)
    b = ogrid.resample_closed(x[::-1], y[::-1], 40)
    c = ogrid.resample_closed(np.r_[x, x[0]], np.r_[y, y[0]], 40)  # duplicated end point
    for X, Y in (b, c):
        assert np.allclose(X, a[0], atol=1e-9) and np.allclose(Y, a[1], atol=1e-9)
    assert a[0].max() == pytest.approx(0.5) and a[0].min() > -0.5 - 1e-9
    assert abs(a[1][0]) < 1e-12


def test_circle_gives_a_valid_grid_quickly():
    X0, Y0 = ogrid.resample_closed(*_circle(), 50)
    t0 = time.perf_counter()
    X, Y, iters, change = ogrid.ogrid(X0, Y0, 25)
    dt = time.perf_counter() - t0
    assert dt < 5.0, dt  # 50x25 about 1 s; the loop mesh_gen needs about 10 s
    assert X.shape == (25, 50)
    assert ogrid.is_valid(X, Y)
    assert change < 1e-6
    assert np.allclose(np.hypot(X[0], Y[0]), 0.5, atol=1e-3)  # wall row 0
    assert np.allclose(np.hypot(X[-1], Y[-1]), 15.0)  # far circle, last row
    assert (ogrid.cell_jacobian(X, Y) < 0).all()  # with this ordering valid grids are all negative


def test_gate_folded_shape_is_rejected_and_sedan_passes():
    X0, Y0 = _polyline_resampled(DEEP_C)
    X, Y, *_ = ogrid.ogrid(X0, Y0, 25)
    assert not ogrid.is_valid(X, Y)
    X0, Y0 = _polyline_resampled(SEDAN)
    X, Y, *_ = ogrid.ogrid(X0, Y0, 25)
    assert ogrid.is_valid(X, Y)


def test_is_valid_rejects_a_hand_folded_grid_and_non_finite_values():
    X0, Y0 = ogrid.resample_closed(*_circle(), 20)
    X, Y, *_ = ogrid.ogrid(X0, Y0, 8, tol=1e-4)
    assert ogrid.is_valid(X, Y)
    Xf = X.copy()
    Xf[3, :] = Xf[5, :]  # row 3 jumps past row 4: the cells between them flip sign
    Yf = Y.copy()
    Yf[3, :] = Yf[5, :]
    assert not ogrid.is_valid(Xf, Yf)
    Xn = X.copy()
    Xn[2, 2] = np.nan
    assert not ogrid.is_valid(Xn, Y)


def test_write_x_round_trips_through_load_mesh(tmp_path):
    X0, Y0 = ogrid.resample_closed(*_circle(), 20)
    X, Y, *_ = ogrid.ogrid(X0, Y0, 8, tol=1e-4)
    p = tmp_path / "c.x"
    ogrid.write_x(p, X, Y)
    assert p.read_text().splitlines()[0] == "20\t8"
    xx, yy = metrics.load_mesh(p)
    assert xx.shape == yy.shape == (21, 8)
    assert np.array_equal(xx[-1], xx[0]) and np.array_equal(yy[-1], yy[0])  # seam row equals row 0
    assert np.abs(xx[:-1] - X.T).max() < 1e-15 and np.abs(yy[:-1] - Y.T).max() < 1e-15
