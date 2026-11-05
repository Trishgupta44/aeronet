import csv

import numpy as np
import pytest

import metrics
from aeronet import panel


@pytest.fixture(scope="module")
def panel_sol():
    return panel.solve("0012", 4, 0.4, 100)


def test_panel_cp_through_forces_matches_panel_cl_cd(panel_sol):
    P = panel_sol.geom
    xn = np.r_[P.x0, P.x0[0]]
    yn = np.r_[P.y0, P.y0[0]]
    cl, cd = metrics.forces_from_faces(xn, yn, panel_sol.cp, 4)
    assert cl == pytest.approx(panel_sol.cl, abs=1e-9)
    assert cd == pytest.approx(panel_sol.cd, abs=1e-9)
    assert cl == pytest.approx(0.526233, abs=1e-6)


def test_uniform_cp_gives_zero_forces(mesh_50x25):
    xx, yy = mesh_50x25
    cl, cd = metrics.forces_from_faces(xx[:, 0], yy[:, 0], np.full(xx.shape[0] - 1, 0.7), 4)
    assert abs(cl) < 1e-12 and abs(cd) < 1e-12


def test_cp_error_panel_vs_fvm_800(repo, panel_sol):
    P = panel_sol.geom
    ref = metrics.load_fvm(repo / "NNfoil" / "FVM_Cp" / "NACA0012_800_400_Cp.dat")
    e = metrics.cp_error(P.cx, P.cy, panel_sol.cp, ref)
    assert e["rms"] == pytest.approx(0.0099, abs=0.0005)
    assert e["n"] == 654


@pytest.mark.parametrize(
    "rel, alpha, expected",
    [
        ("FVM_Cp/NACA0012_800_400_Cp.dat", 4, 0.5193),
        ("FVM_Cp/NACA0012_50_25_Cp.dat", 4, 0.4265),
        ("FVM_Cp_NACA2412/NACA2412_800_400_AoA4_Cp.dat", 4, 0.7605),
    ],
)
def test_fvm_forces_cl(repo, rel, alpha, expected):
    cl, _ = metrics.fvm_forces(repo / "NNfoil" / rel, alpha)
    assert cl == pytest.approx(expected, abs=5e-4)


def test_order_contour_sorts_by_angle():
    th = np.array([3.0, 0.5, 5.0, 1.5, 4.0, 2.0])
    pts = np.c_[np.cos(th), np.sin(th), th]
    out = metrics.order_contour(pts[::-1].copy())
    assert np.allclose(out[:, 2], np.sort(th))


def test_load_mesh_wall_and_load_mesh(repo):
    path = repo / "NNfoil" / "mesh" / "NACA0012_50_25.x"
    xx, yy = metrics.load_mesh(path)
    assert xx.shape == yy.shape and xx.shape[0] > 1
    assert np.allclose(xx[0], xx[-1]) and np.allclose(yy[0], yy[-1])
    x, y = metrics.load_mesh_wall(path)
    assert x.shape == (xx.shape[0] - 1,)
    assert np.allclose(x, xx[:-1, 0] + 0.5)
    assert x[0] == pytest.approx(1.0, abs=1e-6) and y[0] == pytest.approx(0.0, abs=1e-6)
    # counter-clockwise: positive signed area (shoelace), about 0.0819
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert area == pytest.approx(0.0819, abs=1e-3)


def test_append_summary_replaces_same_key(tmp_path):
    path = tmp_path / "summary.csv"
    key = dict(experiment="E0", case="NACA0012_a4", method="fvm", mesh="50x25")
    metrics.append_summary({**key, "rms_cp": 0.1, "source": "measured"}, path)
    metrics.append_summary({**key, "rms_cp": 0.2, "source": "measured"}, path)
    metrics.append_summary({**key, "method": "panel", "rms_cp": 0.3}, path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == metrics.SUMMARY_COLUMNS
    assert len(rows) == 2
    fvm = [r for r in rows if r["method"] == "fvm"]
    assert len(fvm) == 1 and float(fvm[0]["rms_cp"]) == 0.2
    assert fvm[0]["cl"] == ""  # None / missing stays empty, never 0


def test_append_summary_rejects_unknown_column(tmp_path):
    with pytest.raises(ValueError):
        metrics.append_summary({"experiment": "E0", "bogus": 1}, tmp_path / "s.csv")
