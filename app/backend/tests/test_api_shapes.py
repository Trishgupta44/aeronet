"""HTTP endpoints for drawn shapes: solve_shape (B3), field velocity (B4), polar (B5), library (B6)."""

import numpy as np
from fastapi.testclient import TestClient

from aeronet.main import app

client = TestClient(app)


def circle(n=120, r=0.5):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [[float(r * np.cos(a)), float(r * np.sin(a))] for a in t]


def eight(n=200):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [[float(np.sin(a)), float(0.3 * np.sin(2 * a))] for a in t]


def post_shape(**kw):
    body = {"points": circle(), "alpha": 0, "mach": 0, "lifting": False, "grid_width": 90}
    body.update(kw)
    return client.post("/api/solve_shape", json=body)


# ---- B3: /api/solve_shape ----

def test_solve_shape_circle():
    r = post_shape()
    assert r.status_code == 200
    d = r.json()
    assert d["code"] == "custom" and d["cl_thin"] is None and d["ground_y"] is None
    assert abs(d["cl"]) < 1e-6
    assert d["shape"]["n_input"] == 120 and d["warnings"] == []
    assert d["n_panels"] == 200 and len(d["x"]) == 200 and len(d["surface"]["cp"]) == 200
    assert d["field"]["width"] == 90 and len(d["field"]["cp"]) == 90 * d["field"]["height"]
    assert d["alpha"] == 0 and d["mach"] == 0 and d["solve_ms"] > 0 and d["total_ms"] >= d["solve_ms"]


def test_solve_shape_rejects_bad_input():
    assert post_shape(points=[[0, 0], [1, 0], [0.5, 0.3]]).status_code == 422
    r = post_shape(points=eight())
    assert r.status_code == 422 and "crosses itself" in r.json()["detail"]
    assert post_shape(ground=0.001).status_code == 422
    assert post_shape(alpha=30).status_code == 422


def test_solve_shape_with_ground_reports_the_road():
    d = post_shape(ground=0.1, grid_width=0).json()
    assert d["ground_y"] is not None and d["ground_y"] < min(d["y"])
    assert d["field"] is None


def test_solve_shape_warns_about_a_rounded_rear_when_lifting():
    pts = [[2 * np.cos(a), 0.5 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 120, endpoint=False)]
    d = post_shape(points=pts, lifting=True, alpha=3).json()
    assert any("Kutta" in w for w in d["warnings"])


def test_solve_shape_compares_with_a_reference_by_nearest_point():
    ref = client.get("/api/reference/sample").json()["points"]
    pts = client.post("/api/solve", json={"code": "0012", "alpha": 4, "mach": 0.4, "n_half": 100}).json()
    d = post_shape(points=list(zip(pts["x"], pts["y"])), alpha=4, mach=0.4, lifting=True, reference=ref, grid_width=0).json()
    assert d["comparison"]["n"] == 162 and d["comparison"]["rms"] < 0.06


# ---- B4: velocity components in the field ----

def test_solve_has_u_and_v_in_the_field():
    f = client.post("/api/solve", json={"code": "0012", "alpha": 4, "mach": 0.4, "grid_width": 60}).json()["field"]
    n = f["width"] * f["height"]
    assert len(f["u"]) == n and len(f["v"]) == n and len(f["cp"]) == n


def test_masks_agree_between_cp_u_and_v():
    f = post_shape().json()["field"]
    assert [c is None for c in f["cp"]] == [u is None for u in f["u"]] == [v is None for v in f["v"]]
    assert any(c is None for c in f["cp"])


def test_velocity_is_consistent_with_cp():
    f = post_shape(mach=0.0).json()["field"]
    for c, u, v in zip(f["cp"], f["u"], f["v"]):
        if c is not None:
            assert abs(c - (1 - u * u - v * v)) < 0.01  # rounding of the three values to 3 decimals


# ---- B5: /api/polar ----

def test_polar_for_a_naca_code():
    r = client.post("/api/polar", json={"code": "2412", "alphas": [-4, 0, 4, 8], "mach": 0})
    assert r.status_code == 200
    d = r.json()
    assert d["alpha"] == [-4, 0, 4, 8]
    assert d["cl"] == sorted(d["cl"]) and len(set(d["cl"])) == 4
    assert abs(d["cl"][1] - 0.2555) < 2e-3
    assert len(d["cl_thin"]) == 4 and all(isinstance(v, float) for v in d["cl_thin"])


def test_polar_for_points_has_no_thin_airfoil_values():
    d = client.post("/api/polar", json={"points": circle(), "alphas": [0, 5], "lifting": False}).json()
    assert d["cl_thin"] is None and max(abs(c) for c in d["cl"]) < 1e-6


def test_polar_with_ground_uses_the_road():
    free = client.post("/api/polar", json={"code": "0012", "alphas": [4]}).json()["cl"][0]
    near = client.post("/api/polar", json={"code": "0012", "alphas": [4], "ground": 0.05}).json()["cl"][0]
    assert abs(near - free) > 0.02


def test_polar_validation():
    assert client.post("/api/polar", json={"alphas": [0]}).status_code == 422  # neither code nor points
    assert client.post("/api/polar", json={"code": "0012", "points": circle(), "alphas": [0]}).status_code == 422
    assert client.post("/api/polar", json={"code": "0012", "alphas": [30]}).status_code == 422
    assert client.post("/api/polar", json={"code": "0012", "alphas": [-10 + 0.25 * i for i in range(62)]}).status_code == 422
    assert client.post("/api/polar", json={"code": "0012", "alphas": []}).status_code == 422
    assert client.post("/api/polar", json={"code": "0003", "alphas": [0]}).status_code == 422
    r = client.post("/api/polar", json={"points": eight(), "alphas": [0]})
    assert r.status_code == 422 and "crosses itself" in r.json()["detail"]


def test_polar_accepts_61_angles():
    r = client.post("/api/polar", json={"code": "0012", "alphas": [-10 + 0.25 * i for i in range(61)]})
    assert r.status_code == 200 and len(r.json()["cl"]) == 61


# ---- B6: shape library and .dat import ----

IDS = ["circle", "ellipse", "teardrop", "plate", "bird", "sedan", "sports"]


def _library():
    r = client.get("/api/shapes")
    assert r.status_code == 200
    return {s["id"]: s for s in r.json()}


def _solve_preset(p, alpha):
    return client.post(
        "/api/solve_shape",
        json={"points": p["points"], "alpha": alpha, "mach": 0, "lifting": p["lifting"], "ground": p["ground"], "grid_width": 0},
    )


def test_library_ids_in_order_and_fields():
    r = client.get("/api/shapes").json()
    assert [s["id"] for s in r] == IDS
    for s in r:
        assert set(s) == {"id", "label", "points", "lifting", "ground", "note"}
        assert len(s["points"]) >= 8 and s["label"]
        assert all(abs(c - round(c, 5)) < 1e-12 for p in s["points"] for c in p)  # rounded to 5 decimals


def test_every_preset_solves_with_its_own_settings():
    lib = _library()
    for p in lib.values():
        r = _solve_preset(p, 0 if not p["lifting"] else 4)
        assert r.status_code == 200, (p["id"], r.text)
        assert r.json()["warnings"] == [], p["id"]


def test_preset_settings_and_notes():
    lib = _library()
    assert lib["circle"]["lifting"] is False and "Prandtl" in lib["circle"]["note"]
    assert lib["ellipse"]["lifting"] is False
    assert all(lib[k]["lifting"] is True and lib[k]["ground"] is None for k in ("teardrop", "plate", "bird"))
    for k, gap in (("sedan", 0.03), ("sports", 0.02)):
        assert lib[k]["ground"] == gap and lib[k]["lifting"] is False and "Idealized" in lib[k]["note"]


def test_preset_lift_values_are_the_measured_ones():
    lib = _library()
    expected = {"teardrop": (4, 0.508), "plate": (4, 0.457), "bird": (4, 1.204), "sedan": (0, -0.496), "sports": (0, -0.368)}
    for k, (alpha, cl) in expected.items():
        assert abs(_solve_preset(lib[k], alpha).json()["cl"] - cl) < 0.01, k
    circ = _solve_preset(lib["circle"], 0).json()
    assert abs(circ["cl"]) < 1e-6


def _naca0012_dat(style):
    from aeronet.geometry import naca4

    af = naca4("0012", 80)
    pts = list(zip(af.x, af.y))  # TE -> upper -> LE (index 80) -> lower -> (TE)
    if style == "selig":
        rows = pts + [(1.0, 0.0)]
        return "NACA 0012\n" + "\n".join(f"{x:.6f}  {y:.6f}" for x, y in rows) + "\n"
    upper = pts[:81][::-1]
    lower = [pts[80]] + pts[81:] + [(1.0, 0.0)]
    fmt = lambda rows: "\n".join(f"{x:.6f}  {y:.6f}" for x, y in rows)
    return f"NACA 0012\n 81. 81.\n\n{fmt(upper)}\n\n{fmt(lower)}\n"


def test_parse_dat_selig_and_lednicer_agree_with_the_code_solver():
    from aeronet.panel import solve
    from aeronet.shapes import parse_dat, prepare_shape
    from aeronet.panel import solve_xy

    cls = []
    for style in ("selig", "lednicer"):
        name, pts = parse_dat(_naca0012_dat(style))
        assert name == "NACA 0012"
        X, Y, _, _ = prepare_shape(pts, n=160)
        cls.append(solve_xy(X, Y, 4.0, 0.0).cl)
    assert abs(cls[0] - cls[1]) < 1e-6  # same outline, same Cl
    assert abs(cls[0] - solve("0012", 4, 0, 80).cl) < 0.005  # measured 0.4802 vs 0.4822


def test_parse_dat_rejects_files_without_enough_rows():
    import pytest

    from aeronet.shapes import parse_dat

    with pytest.raises(ValueError):
        parse_dat("only a name\n1 2")
    with pytest.raises(ValueError):
        parse_dat("")


def test_parse_dat_accepts_commas_and_extra_columns():
    from aeronet.shapes import parse_dat

    text = "my shape\n" + "\n".join(f"{np.cos(a):.4f}, {0.3 * np.sin(a):.4f}, 0.0" for a in np.linspace(0, 6.0, 12))
    name, pts = parse_dat(text)
    assert name == "my shape" and len(pts) == 12 and len(pts[0]) == 2


def test_parse_dat_endpoint():
    rows = "\n".join(f"{0.5 + 0.5 * np.cos(a):.5f} {0.1 * np.sin(a):.5f}" for a in np.linspace(0, 2 * np.pi, 120, endpoint=False))
    d = client.post("/api/shapes/parse_dat", json={"text": "Test foil\n" + rows, "name": "x.dat"}).json()
    assert d["name"] == "Test foil" and len(d["points"]) == 120
    r = client.post("/api/shapes/parse_dat", json={"text": "nothing", "name": "x.dat"})
    assert r.status_code == 400 and r.json()["detail"]
