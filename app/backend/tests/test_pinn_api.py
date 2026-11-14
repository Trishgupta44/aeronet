"""PINN case endpoints (C7): the lists, the case file, the results rows, availability, the 409, 404 and the path-traversal guard.

The first group uses a temporary data folder with one fixture case; the second checks the real exported data that is committed
in app/backend/data (size, flow-field mask, every showdown run present)."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aeronet import field as field_mod
from aeronet import pinn_cases, solvers
from aeronet.main import app
from conftest import SUMMARY_HEADER, csv_rows, make_case, write_case, write_results

client = TestClient(app)


def _pinn(solvers_json):
    return next(s for s in solvers_json if s["id"] == "pinn")


# ------------------------------------------------------------------------------------------------ fixture data
def test_case_list_has_one_item_with_the_listed_fields(pinn_dir):
    write_case(pinn_dir, make_case("fixture_case"))
    r = client.get("/api/pinn/cases")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 1
    assert items[0] == {"id": "fixture_case", "label": "Fixture case", "shape": "NACA0012", "mesh": "100x50", "mach": 0.4, "alpha": 4.0,
                        "rms": 0.0145, "cl": 0.516, "collapsed": False, "source": {"results": "measured", "condition": "quoted"}}


def test_case_list_is_sorted_by_label_and_flags_collapsed_runs(pinn_dir):
    write_case(pinn_dir, make_case("run_b", label="B case"))
    bad = make_case("run_a", label="A case")
    bad["metrics"].update(rms=15.6, cl=0.0, collapsed=True)
    write_case(pinn_dir, bad)
    items = client.get("/api/pinn/cases").json()
    assert [i["id"] for i in items] == ["run_a", "run_b"]
    assert items[0]["collapsed"] is True and items[0]["cl"] == 0.0 and items[1]["collapsed"] is False


def test_case_list_is_empty_without_a_cases_folder(pinn_dir, monkeypatch):
    monkeypatch.setattr(pinn_cases, "CASES_DIR", pinn_dir / "nope")
    assert client.get("/api/pinn/cases").json() == []


def test_one_case_comes_back_whole_with_its_field(pinn_dir):
    case = make_case("fixture_case")
    write_case(pinn_dir, case)
    r = client.get("/api/pinn/cases/fixture_case")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    d = r.json()
    assert d == case and d["field"]["cp"][1] is None and len(d["field"]["cp"]) == d["field"]["width"] * d["field"]["height"]


def test_unknown_id_is_404(pinn_dir):
    write_case(pinn_dir, make_case("fixture_case"))
    r = client.get("/api/pinn/cases/no_such_case")
    assert r.status_code == 404 and "no_such_case" in r.json()["detail"]


@pytest.mark.parametrize("bad", ["../x", "..%2Fx", "%2e%2e%2fresults_summary", "..\\x", "fixture_case.json", "Fixture_Case", "fixture-case"])
def test_path_traversal_and_malformed_ids_are_404(pinn_dir, bad):
    write_case(pinn_dir, make_case("fixture_case"))
    (pinn_dir / "secret.json").write_text("{}")  # a file next to the cases folder must stay out of reach
    r = client.get(f"/api/pinn/cases/{bad}")
    assert r.status_code == 404


def test_loader_checks_the_id_against_the_list(pinn_dir):
    write_case(pinn_dir, make_case("fixture_case"))
    (pinn_dir / "pinn_cases" / "Not_An_Id.json").write_text("{}")  # a file whose name is not a valid id is not a case
    assert pinn_cases.case_ids() == ["fixture_case"]
    for bad in ("../results_summary", "..", "a/b", "fixture_case/", None, 5, "Not_An_Id"):
        with pytest.raises(KeyError):
            pinn_cases.get_case(bad)
    assert pinn_cases.get_case("fixture_case")["id"] == "fixture_case"


def test_results_is_empty_when_the_file_is_missing_and_the_rows_when_present(pinn_dir):
    assert client.get("/api/results").json() == []
    rows = csv_rows(SUMMARY_HEADER + "E2,NACA0012_a4,pinn,100x50,0.4,4,0.014451,0.082595,0.516286,0.519329,0.0006,577.6,,measured,rounds=20\n"
                    + "E2,NACA0012_a4,pinn_gs_nopos,200x100,0.4,4,0.009213,0.035,0.5118,0.519329,0.0005,,,measured,train_s withheld: slept\n")
    write_results(pinn_dir, rows)
    got = client.get("/api/results").json()
    assert got == rows and got[0]["rms_cp"] == 0.014451 and got[1]["train_s"] is None and got[0]["source"] == "measured"


def test_e1_rounds_is_a_503_when_missing_and_the_file_when_present(pinn_dir):
    r = client.get("/api/results/e1_rounds")
    assert r.status_code == 503 and "not available" in r.json()["detail"]
    data = {"source": "results/e1_rounds.csv", "runs": [{"run_id": "a", "method": "pinn", "positivity": True, "seed": 0, "mesh": "50x25",
                                                          "rounds": [1, 2], "cp_rms": [0.19, 0.037], "cl": [0.3, 0.48]}]}
    (pinn_dir / "rounds_e1.json").write_text(json.dumps(data), encoding="utf8")
    assert client.get("/api/results/e1_rounds").json() == data


def test_real_e1_rounds_has_six_equal_length_runs():
    if not pinn_cases.ROUNDS_E1_FILE.exists():
        pytest.skip("no export here")
    d = client.get("/api/results/e1_rounds").json()
    assert len(d["runs"]) == 6
    for r in d["runs"]:
        assert len(r["rounds"]) == len(r["cp_rms"]) == len(r["cl"]) and r["mesh"] == "50x25"


def test_cases_are_reread_when_the_file_changes(pinn_dir):
    write_case(pinn_dir, make_case("fixture_case"))
    assert client.get("/api/pinn/cases").json()[0]["rms"] == 0.0145
    changed = make_case("fixture_case")
    changed["metrics"]["rms"] = 0.5  # a different size changes the cache stamp even within the same clock tick
    changed["label"] = "Fixture case, changed"
    write_case(pinn_dir, changed)
    assert client.get("/api/pinn/cases").json()[0]["rms"] == 0.5


# ------------------------------------------------------------------------------------------------ solvers
def test_pinn_is_available_only_when_cases_exist(pinn_dir):
    assert _pinn(client.get("/api/solvers").json())["available"] is False  # empty folder
    assert "export" in _pinn(client.get("/api/solvers").json())["note"]
    write_case(pinn_dir, make_case("fixture_case"))
    p = _pinn(client.get("/api/solvers").json())
    assert p["available"] is True and "PINN Lab" in p["note"]
    assert next(s for s in client.get("/api/solvers").json() if s["id"] == "panel")["available"] is True


def test_solve_with_the_pinn_is_refused_with_409_even_when_cases_exist(pinn_dir):
    write_case(pinn_dir, make_case("fixture_case"))
    r = client.post("/api/solve", json={"code": "0012", "alpha": 4, "mach": 0.4, "solver": "pinn"})
    assert r.status_code == 409
    msg = r.json()["detail"]
    assert "trained condition" in msg and "PINN Lab" in msg
    # an unknown solver is still a 404, and the panel method still answers
    assert client.post("/api/solve", json={"code": "0012", "solver": "nope"}).status_code == 404
    assert client.post("/api/solve", json={"code": "0012"}).status_code == 200


def test_no_interpolation_between_cases(pinn_dir):
    """A condition between two trained cases gets the same refusal: there is no code path that blends networks."""
    write_case(pinn_dir, make_case("case_a4", alpha=4.0))
    write_case(pinn_dir, make_case("case_a0", alpha=0.0))
    r = client.post("/api/solve", json={"code": "0012", "alpha": 2, "mach": 0.4, "solver": "pinn"})
    assert r.status_code == 409
    with pytest.raises(LookupError):
        solvers.get_solver("pinn")


# ------------------------------------------------------------------------------------------------ the real exported data
REAL = sorted(pinn_cases.CASES_DIR.glob("*.json")) if pinn_cases.CASES_DIR.is_dir() else []


def test_the_committed_data_makes_pinn_available_and_lists_every_case():
    if not REAL:
        pytest.skip("no exported cases in app/backend/data")
    assert _pinn(client.get("/api/solvers").json())["available"] is True
    items = client.get("/api/pinn/cases").json()
    assert {i["id"] for i in items} == {p.stem for p in REAL}
    assert len({i["label"] for i in items}) == len(items)  # labels tell the cases apart
    for i in items:
        assert i["source"]["results"] == "measured" and i["source"]["condition"] == "quoted"
    assert sum(i["collapsed"] for i in items) == 1  # the NACA 2412 4-degree seed-0 run, shown honestly
    assert next(i for i in items if i["collapsed"])["id"] == "naca2412_100x50_a4_m03"


def test_every_showdown_run_has_an_exported_case():
    from aeronet import pinn_data

    if not REAL:
        pytest.skip("no exported cases in app/backend/data")
    assert set(pinn_data.E2_RUNS.values()) <= {p.stem for p in REAL}


@pytest.mark.parametrize("path", REAL, ids=[p.stem for p in REAL])
def test_real_case_is_under_3_mb_has_the_5_7_keys_and_a_correct_field_mask(path):
    assert path.stat().st_size < 3_000_000
    c = json.loads(path.read_text(encoding="utf8"))
    for k in ("id", "label", "shape", "mesh", "mach", "alpha", "x", "y", "surface", "reference", "panel", "metrics", "timing", "log",
              "snapshots", "field", "source"):
        assert k in c, k
    assert c["id"] == path.stem and set(c["surface"]) == {"x", "y", "cp"}
    assert len(c["surface"]["cp"]) == len(c["surface"]["x"]) == len(c["snapshots"][-1]["cp"])
    assert {"rms", "max", "n", "cl", "cd", "cl_ref", "collapsed"} <= set(c["metrics"])
    f = c["field"]
    assert (f["xmin"], f["xmax"], f["ymin"], f["ymax"]) == (field_mod.XMIN, field_mod.XMAX, field_mod.YMIN, field_mod.YMAX)
    assert f["height"] == round(f["width"] / 1.8)
    n = f["width"] * f["height"]
    assert len(f["cp"]) == len(f["u"]) == len(f["v"]) == n
    # null exactly inside the body (the wall polygon), finite everywhere else (same grid as the panel field: row-major, top row first)
    xs = f["xmin"] + (np.arange(f["width"]) + 0.5) / f["width"] * (f["xmax"] - f["xmin"])
    ys = f["ymax"] - (np.arange(f["height"]) + 0.5) / f["height"] * (f["ymax"] - f["ymin"])
    gx, gy = np.meshgrid(xs, ys)
    inside = field_mod.inside_body(np.asarray(c["x"]), np.asarray(c["y"]), gx.ravel(), gy.ravel())
    for key in ("cp", "u", "v"):
        null = np.array([v is None for v in f[key]])
        assert abs(int(null.sum()) - int(inside.sum())) <= 5 and int((null != inside).sum()) <= 5  # only cells on the wall may differ
        assert all(np.isfinite(v) for v in f[key] if v is not None)
    assert inside.sum() > 100 and not inside.all()


def test_the_collapsed_case_keeps_its_failure_visible():
    path = pinn_cases.CASES_DIR / "naca2412_100x50_a4_m03.json"
    if not path.exists():
        pytest.skip("not exported here")
    c = client.get("/api/pinn/cases/naca2412_100x50_a4_m03").json()
    assert c["metrics"]["collapsed"] is True and abs(c["metrics"]["cl"]) < 1e-3 and c["metrics"]["cl_ref"] > 0.7
    assert "collapsed" in c["label"] and c["config"]["failed_rounds"] == 7 and c["source"]["condition"] == "quoted"
    assert "quoted" in c["source"]["condition_note"] and "0.3" in c["source"]["condition_note"]


def test_the_withheld_200x100_time_is_null_with_a_note():
    path = pinn_cases.CASES_DIR / "naca0012_200x100_a4_m04_gs_nopos.json"
    if not path.exists():
        pytest.skip("not exported here")
    c = json.loads(path.read_text(encoding="utf8"))
    assert c["timing"]["train_s"] is None and "withheld" in c["timing"]["note"]
    assert all(t is None for t in c["log"]["time_s"])


def test_exported_results_rows_equal_summary_csv():
    from pathlib import Path

    src = Path(__file__).resolve().parents[3] / "results" / "summary.csv"
    if not src.exists() or not pinn_cases.RESULTS_FILE.exists():
        pytest.skip("no results/summary.csv or no export here")
    import csv

    want = list(csv.DictReader(src.open(encoding="utf8")))
    got = client.get("/api/results").json()
    assert len(got) == len(want)
    for w, g in zip(want, got):
        for k, v in w.items():
            if v == "":
                assert g[k] is None
            elif k in ("mach", "alpha_deg", "rms_cp", "max_cp", "cl", "cl_ref", "cd", "train_s", "infer_ms"):
                assert g[k] == float(v)
            else:
                assert g[k] == v
