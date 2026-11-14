"""Deployment settings: the default mode (NNfoil reference data served, with permission), the strict mode (AERONET_WITHHOLD_FVM_DATA=1: none of it),
the stripped data copy of deploy/make_data.py, CORS origins, and the limits on the expensive endpoints."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from aeronet import config, limits, optimize, pinn_cases
from aeronet.main import app
from conftest import SUMMARY_HEADER, csv_rows, make_case, write_case, write_results

client = TestClient(app)
REPO = Path(__file__).resolve().parents[3]
FVM_PTS = [[0.1, 0.05, 0.3], [0.5, 0.06, -0.2], [0.9, 0.01, 0.1], [0.5, -0.06, -0.1]]  # (x_app, y, Cp): upper and lower wall
CASE_ID = "naca0012_100x50_a4_m04"


def _make_data_module():
    spec = importlib.util.spec_from_file_location("deploy_make_data", REPO / "deploy" / "make_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def full_data(pinn_dir):
    """Exported data in the committed form: the case CARRIES the FVM reference and same-mesh points (as app/backend/data does)."""
    write_results(pinn_dir, csv_rows(
        SUMMARY_HEADER
        + "B0,NACA0012_a4,fvm,100x50,0.4,4,0.045985,0.260987,0.480267,0.519329,0.01,,,measured,ref\n"
        + "E2,NACA0012_a4,pinn,100x50,0.4,4,0.014451,0.082595,0.516286,0.519329,0.0006,577.6,,measured,final round; rounds=20; failed_rounds=0\n"
    ))
    x = [0.05 + 0.1 * i for i in range(10)]
    write_case(pinn_dir, make_case(
        CASE_ID,
        surface={"x": x, "y": [0.03] * 5 + [-0.03] * 5, "cp": [0.1] * 10},
        snapshots=[{"round": 1, "cp": [0.4] * 10}, {"round": 2, "cp": [0.1] * 10}],
        reference={"name": "FVM 800×400", "points": FVM_PTS},
        fvm_same_mesh={"name": "FVM 100×50", "points": FVM_PTS, "rms": 0.046, "max": 0.26, "cl": 0.48},
    ))
    return pinn_dir


@pytest.fixture
def strict(full_data, monkeypatch):
    """Strict mode (AERONET_WITHHOLD_FVM_DATA=1) with the FVM points still ON DISK: the server alone must keep them back."""
    monkeypatch.setenv("AERONET_WITHHOLD_FVM_DATA", "1")
    return full_data


# ---- default mode: the author gave permission, so the reference data is served (with credit on the pages) -----------------------------

def test_default_mode_serves_the_reference_data(full_data, monkeypatch):
    monkeypatch.delenv("AERONET_WITHHOLD_FVM_DATA", raising=False)
    assert config.withhold_fvm_data() is False
    c = client.get(f"/api/pinn/cases/{CASE_ID}").json()
    assert c["reference"]["points"] == FVM_PTS and c["fvm_same_mesh"]["points"] == FVM_PTS and "fvm_curves_withheld" not in c
    d = client.get("/api/showdown").json()
    assert d["fvm_curves_withheld"] is False and d["reference"]["curve"] is not None
    assert next(m for m in d["meshes"] if m["mesh"] == "100x50")["fvm"]["curve"] is not None


def test_default_build_keeps_the_sample_and_the_curves(monkeypatch):
    monkeypatch.delenv("AERONET_WITHHOLD_FVM_DATA", raising=False)
    caps = client.get("/api/capabilities").json()
    assert caps["public_build"] is False and caps["fvm_curves"] is True
    assert caps["fvm_sample"] is True  # the sample file ships with the repository
    assert client.get("/api/reference/sample").status_code == 200


def test_the_committed_data_holds_fvm_curves_so_a_strict_image_must_be_built_from_a_stripped_copy():
    problems = _make_data_module().find_fvm_data(REPO / "app" / "backend" / "data")
    assert any(".dat" in p for p in problems) and any("reference has FVM points" in p for p in problems)


# ---- strict mode: nothing of it is served, and a stripped copy holds none of it -----------------------------------------------------------

def test_strict_mode_serves_numbers_but_no_fvm_curves(strict):
    d = client.get("/api/showdown").json()
    assert d["fvm_curves_withheld"] is True and d["reference"]["curve"] is None
    m = next(m for m in d["meshes"] if m["mesh"] == "100x50")
    assert m["fvm"]["rms"] == 0.045985 and m["fvm"]["curve"] is None  # the error number stays, the curve does not
    assert m["pinn"][0]["curve"]["cp"] == [0.1] * 10                    # our own PINN result stays


def test_strict_showdown_needs_no_fvm_reference_at_all(pinn_dir, monkeypatch):
    """A stripped data folder (the strict image) has no reference points anywhere: the showdown still answers."""
    monkeypatch.setenv("AERONET_WITHHOLD_FVM_DATA", "1")
    write_results(pinn_dir, csv_rows(SUMMARY_HEADER + "B0,NACA0012_a4,fvm,100x50,0.4,4,0.045985,0.260987,0.480267,0.519329,0.01,,,measured,ref\n"))
    write_case(pinn_dir, pinn_cases.strip_fvm(make_case(CASE_ID, reference={"name": "FVM 800×400", "points": FVM_PTS})))
    d = client.get("/api/showdown").json()
    assert d["reference"]["curve"] is None and d["fvm_curves_withheld"] is True


def test_normal_showdown_still_demands_a_reference(pinn_dir, monkeypatch):
    monkeypatch.delenv("AERONET_WITHHOLD_FVM_DATA", raising=False)
    write_results(pinn_dir, [])
    write_case(pinn_dir, make_case(CASE_ID, reference=None))
    assert client.get("/api/showdown").status_code == 503


def test_strict_case_endpoint_has_no_fvm_curve_but_keeps_the_numbers(strict):
    r = client.get(f"/api/pinn/cases/{CASE_ID}")
    assert r.status_code == 200
    raw = r.text
    assert '"reference"' not in raw and '"fvm_same_mesh"' not in raw and '"points"' not in raw
    c = r.json()
    assert c["fvm_curves_withheld"] is True and c["fvm_ref_name"] == "FVM 800×400"
    assert c["fvm_same_errors"] == {"name": "FVM 100×50", "rms": 0.046, "max": 0.26, "cl": 0.48}
    assert c["metrics"]["rms"] == 0.0145 and c["panel"]["cl"] == 0.5          # the project's own numbers
    assert c["surface"]["cp"] == [0.1] * 10 and len(c["snapshots"]) == 2 and c["field"]["width"] == 3  # PINN curve, replay, flow field
    assert len(c["rms_by_round"]) == 2 and c["rms_by_round"][0] > c["rms_by_round"][1]


def test_strict_mode_strips_every_route_that_serves_a_case(strict):
    for path in ("/api/pinn/cases", "/api/showdown", "/api/results", f"/api/pinn/cases/{CASE_ID}", "/api/results/e1_rounds", "/api/tradeoff"):
        t = client.get(path).text.replace(" ", "")
        assert '"fvm_same_mesh"' not in t and '"reference":{"name"' not in t and "[0.1,0.05,0.3]" not in t, path
    assert "reference" not in pinn_cases.get_case_meta(CASE_ID) and "reference" not in pinn_cases.get_case(CASE_ID)


def test_strict_mode_hides_the_bundled_fvm_sample(strict):
    assert client.get("/api/reference/sample").status_code == 404
    caps = client.get("/api/capabilities").json()
    assert caps["fvm_sample"] is False and caps["fvm_curves"] is False and caps["public_build"] is True


def test_strip_fvm_removes_only_the_curves_and_is_idempotent():
    case = make_case(CASE_ID, reference={"name": "FVM 800×400", "points": FVM_PTS}, fvm_same_mesh={"name": "FVM 100×50", "points": FVM_PTS, "rms": 0.1, "max": 0.2, "cl": 0.3})
    out = pinn_cases.strip_fvm(case)
    assert "reference" not in out and "points" not in json.dumps(out) and case["reference"]["points"] == FVM_PTS  # the input is untouched
    assert out["fvm_same_errors"] == {"name": "FVM 100×50", "rms": 0.1, "max": 0.2, "cl": 0.3}
    for k in ("surface", "panel", "metrics", "timing", "log", "snapshots", "field", "config", "source"):
        assert out[k] == case[k]
    assert pinn_cases.strip_fvm(out) == out
    bare = pinn_cases.strip_fvm(make_case(CASE_ID, reference=None, fvm_same_mesh=None))
    assert bare["fvm_curves_withheld"] is True and "fvm_ref_name" not in bare and "fvm_same_errors" not in bare


def test_round_rms_matches_the_definition_of_the_exported_error():
    sx, sy = [0.25, 0.5, 0.75, 0.5], [0.04, 0.06, 0.03, -0.06]  # reference points exactly on the faces: the nearest face is itself
    ref = [[x, y, cp + 0.1] for x, y, cp in zip(sx, sy, [0.0, -0.2, 0.1, 0.3])]
    case = make_case(CASE_ID, surface={"x": sx, "y": sy, "cp": [0.0, -0.2, 0.1, 0.3]}, reference={"name": "r", "points": ref},
                     snapshots=[{"round": 1, "cp": [0.0, 0.0, 0.0, 0.0]}, {"round": 2, "cp": [0.0, -0.2, 0.1, 0.3]}])
    r = pinn_cases.round_rms(case)
    assert r[1] == pytest.approx(0.1, abs=1e-6) and r[0] > r[1]
    assert pinn_cases.round_rms(make_case(CASE_ID, reference=None)) == [None, None]
    near_edge = make_case(CASE_ID, reference={"name": "r", "points": [[0.01, 0.0, 1.0]]})  # outside 0.02 <= x <= 0.98: nothing to compare
    assert pinn_cases.round_rms(near_edge) == [None, None]


def test_make_data_writes_a_stripped_copy_that_holds_no_fvm_data(tmp_path):
    md = _make_data_module()
    out = tmp_path / "public"
    manifest = md.build(REPO / "app" / "backend" / "data", out)
    assert manifest["cases"] == 14 and manifest["cases_stripped"] == 14
    assert md.find_fvm_data(out) == []
    assert not list(out.rglob("*.dat")) and not (out / "reference_naca0012_200x100.dat").exists()
    c = json.loads((out / "pinn_cases" / "naca0012_100x50_a4_m04_gs.json").read_text(encoding="utf8"))
    assert "reference" not in c and "fvm_same_mesh" not in c and c["fvm_same_errors"]["rms"] > 0 and len(c["rms_by_round"]) == len(c["snapshots"])
    assert md.main(["--check", str(out)]) == 0 and md.main(["--check", str(REPO / "app" / "backend" / "data")]) == 1


def test_find_fvm_data_flags_each_form(tmp_path):
    md = _make_data_module()
    (tmp_path / "pinn_cases").mkdir()
    (tmp_path / "pinn_cases" / "a.json").write_text(json.dumps({"reference": {"name": "r", "points": [[0, 0, 0]]}}), encoding="utf8")
    (tmp_path / "pinn_cases" / "b.json").write_text(json.dumps({"x": {"deep": [{"points": [[0, 0, 0]]}]}}), encoding="utf8")
    (tmp_path / "pinn_cases" / "c.json").write_text(json.dumps({"fvm_same_mesh": {"points": [[1, 1, 1]], "rms": 0.1}}), encoding="utf8")
    (tmp_path / "sample_Cp.dat").write_text("1 0 0 0", encoding="utf8")
    p = md.find_fvm_data(tmp_path)
    assert len(p) == 4 and any("a.json" in q for q in p) and any("b.json" in q for q in p) and any("c.json" in q for q in p) and any(".dat" in q or "_Cp" in q for q in p)
    (tmp_path / "sample_Cp.dat").unlink()
    for n in "abc":
        (tmp_path / "pinn_cases" / f"{n}.json").write_text(json.dumps({"fvm_same_errors": {"rms": 0.1}, "rms_by_round": [0.1, 0.2]}), encoding="utf8")
    assert md.find_fvm_data(tmp_path) == []


def test_cors_origins_from_the_environment(monkeypatch):
    monkeypatch.setenv("AERONET_CORS_ORIGINS", "https://demo.example.org/, https://other.example.org")
    assert config.cors_origins() == config.DEV_ORIGINS + ["https://demo.example.org", "https://other.example.org"]
    monkeypatch.delenv("AERONET_CORS_ORIGINS")
    assert config.cors_origins() == config.DEV_ORIGINS


def test_bad_numbers_in_the_environment_fall_back_to_the_defaults(monkeypatch):
    monkeypatch.setenv("AERONET_MAX_HEAVY", "many")
    monkeypatch.setenv("AERONET_HEAVY_PER_MIN", "0")
    assert config.max_heavy() == 2 and config.heavy_per_minute() == 1  # not a number: default; below the minimum: the minimum


def _request(host="203.0.113.7"):
    return SimpleNamespace(client=SimpleNamespace(host=host))


def test_per_address_limit_answers_429(monkeypatch):
    monkeypatch.setattr(optimize, "optimise", lambda *a, **k: {"ok": True})
    monkeypatch.setenv("AERONET_HEAVY_PER_MIN", "2")
    body = {"target_cl": 0.5, "min_thickness": 0.1}
    assert [client.post("/api/optimize", json=body).status_code for _ in range(3)] == [200, 200, 429]
    assert "wait a minute" in client.post("/api/optimize", json=body).json()["detail"]


def test_concurrency_limit_answers_503_and_frees_the_slot(monkeypatch):
    monkeypatch.setenv("AERONET_MAX_HEAVY", "1")
    first = limits.heavy_guard(_request("198.51.100.1"))
    next(first)  # takes the only slot
    second = limits.heavy_guard(_request("198.51.100.2"))
    with pytest.raises(HTTPException) as busy:
        next(second)
    assert busy.value.status_code == 503 and "busy" in busy.value.detail
    first.close()  # the first request finishes
    third = limits.heavy_guard(_request("198.51.100.3"))
    next(third)
    third.close()


def test_a_refused_request_does_not_use_up_the_budget_of_the_address(monkeypatch):
    monkeypatch.setenv("AERONET_MAX_HEAVY", "1")
    monkeypatch.setenv("AERONET_HEAVY_PER_MIN", "2")
    holder = limits.heavy_guard(_request("198.51.100.9"))
    next(holder)
    refused = limits.heavy_guard(_request("203.0.113.5"))
    with pytest.raises(HTTPException):
        next(refused)
    holder.close()
    ok = limits.heavy_guard(_request("203.0.113.5"))
    next(ok)  # still has its full allowance
    ok.close()
