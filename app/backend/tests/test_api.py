from fastapi.testclient import TestClient

from aeronet.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_solvers_and_presets():
    solvers = {s["id"]: s for s in client.get("/api/solvers").json()}
    assert solvers["panel"]["available"] is True
    assert solvers["pinn"]["available"] is False and "checkpoint" in solvers["pinn"]["note"]
    assert [p["code"] for p in client.get("/api/airfoils").json()] == ["0012", "2412", "4412"]


def test_solve_basic():
    r = client.post("/api/solve", json={"code": "0012", "alpha": 4, "mach": 0.4})
    assert r.status_code == 200
    d = r.json()
    assert abs(d["cl"] - 0.526) < 1e-3 and abs(d["cd"]) < 0.01
    assert abs(d["cl_thin"] - 0.479) < 1e-3
    assert d["n_panels"] == 140 and len(d["surface"]["cp"]) == 140 and len(d["x"]) == 140
    assert d["field"] is None and d["comparison"] is None
    assert d["solve_ms"] > 0


def test_solve_with_field_and_reference():
    sample = client.get("/api/reference/sample").json()
    assert sample["count"] == 200 and sample["condition"] == {"code": "0012", "alpha": 4.0, "mach": 0.4}
    r = client.post("/api/solve", json={"code": "0012", "alpha": 4, "mach": 0.4, "grid_width": 100, "reference": sample["points"]})
    d = r.json()
    assert d["field"]["width"] == 100 and len(d["field"]["cp"]) == d["field"]["width"] * d["field"]["height"]
    assert d["comparison"]["n"] > 100 and d["comparison"]["rms"] < 0.05


def test_validation_errors():
    assert client.post("/api/solve", json={"code": "0003"}).status_code == 422
    assert client.post("/api/solve", json={"code": "12"}).status_code == 422
    assert client.post("/api/solve", json={"code": "0012", "alpha": 45}).status_code == 422
    assert client.post("/api/solve", json={"code": "0012", "mach": 0.95}).status_code == 422


def test_unavailable_and_unknown_solver():
    assert client.post("/api/solve", json={"code": "0012", "solver": "pinn"}).status_code == 409
    assert client.post("/api/solve", json={"code": "0012", "solver": "nope"}).status_code == 404


def test_parse_reference_endpoint():
    text = "nodenumber x y Cp\n1 -0.4 0.05 -0.2\n2 -0.3 0.06 -0.3\n"
    d = client.post("/api/reference/parse", json={"text": text, "name": "t.dat"}).json()
    assert d["count"] == 2 and abs(d["points"][0][0] - 0.1) < 1e-9
    assert client.post("/api/reference/parse", json={"text": "nothing useful"}).status_code == 400
