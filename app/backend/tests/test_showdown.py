"""Same-mesh showdown: numbers must be the ones in the exported results (= results/summary.csv), final round only, pending shown
as pending. The showdown reads only through aeronet.pinn_cases (exported case JSON, results rows, seed runs): no results/ or NNfoil/."""

import csv

import pytest
from fastapi.testclient import TestClient

from aeronet import pinn_cases, pinn_data
from aeronet.main import app
from conftest import SUMMARY_HEADER, csv_rows, make_case, write_case, write_results, write_seed_runs

FVM_PTS = [[0.1, 0.05, 0.3], [0.5, 0.06, -0.2], [0.9, 0.01, 0.1], [0.5, -0.06, -0.1]]  # (x_app, y, Cp): upper and lower wall


@pytest.fixture
def fake(pinn_dir):
    write_results(pinn_dir, csv_rows(
        SUMMARY_HEADER
        + "B0,NACA0012_a4,fvm,100x50,0.4,4,0.045985,0.260987,0.480267,0.519329,0.01,,,measured,ref\n"
        + "E2,NACA0012_a4,pinn,100x50,0.4,4,0.014451,0.082595,0.516286,0.519329,0.0006,577.6,,measured,final round; rounds=20; failed_rounds=0\n"
    ))
    x = [0.05 + 0.1 * i for i in range(10)]
    done = make_case(
        "naca0012_100x50_a4_m04",
        surface={"x": x, "y": [0.03] * 5 + [-0.03] * 5, "cp": [0.1] * 10},  # the FINAL round
        snapshots=[{"round": 1, "cp": [9.0] * 10}, {"round": 2, "cp": [0.1] * 10}],
        reference={"name": "FVM 800×400", "points": FVM_PTS},
        fvm_same_mesh={"name": "FVM 100×50", "points": FVM_PTS, "rms": 0.046, "max": 0.26, "cl": 0.48},
    )
    pend = make_case("naca0012_200x100_a4_m04_gs_nopos", mesh="200x100", config={"rounds": 30, "rounds_done": 3},
                     log={"round": [1, 2, 3], "time_s": [1.0, 2.0, 3.0]}, fvm_same_mesh=None)
    write_case(pinn_dir, done)
    write_case(pinn_dir, pend)
    return pinn_dir


def test_numbers_come_from_the_summary_and_the_final_round(fake):
    d = pinn_data.showdown()
    m = next(m for m in d["meshes"] if m["mesh"] == "100x50")
    assert m["fvm"]["rms"] == 0.045985 and m["fvm"]["source"] == "measured"
    p = m["pinn"][0]
    assert p["status"] == "done" and p["rms"] == 0.014451 and p["max"] == 0.082595 and p["train_s"] == 577.6
    assert p["rounds"] == 20 and p["failed_rounds"] == 0
    assert p["final_round"] == 2 and set(p["curve"]["cp"]) == {0.1}  # final round, not the first and not a "best" round
    assert sum(p["curve"]["upper"]) == 5


def test_seed_spread_comes_from_the_seed_runs_and_collapsed_seeds_count_but_give_no_range(fake):
    x = [0.05 + 0.1 * i for i in range(10)]
    y = [0.03] * 5 + [-0.03] * 5

    def run(seed, run_id, rms, cl, rounds, failed, collapsed, cp):
        return {"seed": seed, "run_id": run_id, "rms": rms, "cl": cl, "rounds": rounds, "failed_rounds": failed, "collapsed": collapsed,
                "curve": None if cp is None else {"x": x, "y": y, "cp": [cp] * 10}}

    write_seed_runs(fake, {
        "100x50:pinn": {"source": "results/seed_check.csv", "runs": [
            run(0, "naca0012_100x50_a4_m04", 0.014451, 0.516286, 20, 0, False, 0.1),
            run(1, "naca0012_100x50_a4_m04_s1", 0.015379, 0.506552, 5, 0, False, 0.3),
            run(2, "naca0012_100x50_a4_m04_s2", None, None, 0, 5, True, None)]},
        "100x50:pinn_gs": {"source": "results/seed_check.csv", "runs": [run(0, "some_other_run", 0.0161, 0.5, 4, 0, False, 0.9)]},
    })
    p = next(m for m in pinn_data.showdown()["meshes"] if m["mesh"] == "100x50")["pinn"][0]
    s = p["seeds"]
    assert s["source"] == "results/seed_check.csv" and s["n"] == 3 and s["n_with_result"] == 2 and s["n_without_result"] == 1
    assert s["cl"] == {"min": 0.506552, "max": 0.516286} and s["rms"] == {"min": 0.014451, "max": 0.015379}
    assert s["rounds_differ"] is True
    assert s["band"]["n_curves"] == 2 and s["band"]["cp_min"] == [0.1] * 10 and s["band"]["cp_max"] == [0.3] * 10
    assert all("curve" not in r for r in s["runs"]) and s["runs"][2]["collapsed"] is True
    assert p["validity"]["supersonic"] is False and p["validity"]["cp_crit"] < -3  # other variant (pinn_gs) must not leak in


def test_note_parsing_handles_both_note_formats():
    assert pinn_data._first_int("rounds=19; failed_rounds=1", "failed_rounds", "restarts") == 1
    assert pinn_data._first_int("rounds=19/20; seed=0; restarts=2; nan_rounds=0", "failed_rounds", "restarts") == 2
    assert pinn_data._first_int("rounds=20", "failed_rounds", "restarts") is None
    assert pinn_data._withheld_note("a; train_s withheld: the laptop slept during this run; b") == "the laptop slept during this run"
    assert pinn_data._withheld_note("nothing here") is None


def test_no_seed_data_means_no_spread_is_claimed(fake):
    p = next(m for m in pinn_data.showdown()["meshes"] if m["mesh"] == "100x50")["pinn"][0]
    assert p["seeds"] is None


def test_run_without_a_summary_row_is_pending_and_has_no_curve(fake):
    d = pinn_data.showdown()
    p = next(m for m in d["meshes"] if m["mesh"] == "200x100")["pinn"][0]
    assert p == {"method": "pinn_gs_nopos", "label": pinn_data.METHOD_LABELS["pinn_gs_nopos"], "run_id": "naca0012_200x100_a4_m04_gs_nopos",
                 "status": "pending", "rounds_done": 3, "rounds_target": 30}


def test_mesh_without_rows_has_no_fvm_and_no_pinn(fake):
    d = pinn_data.showdown()
    m = next(m for m in d["meshes"] if m["mesh"] == "50x25")
    assert m["fvm"] is None and m["pinn"] == []


def test_endpoint(fake):
    r = TestClient(app).get("/api/showdown")
    assert r.status_code == 200 and r.json()["reference"]["label"] == "FVM 800x400"


def test_endpoint_reports_missing_data(pinn_dir):
    assert TestClient(app).get("/api/showdown").status_code == 503  # no case with a reference


def test_a_summary_row_without_an_exported_case_leaves_the_run_out(fake):
    write_results(fake, csv_rows(
        SUMMARY_HEADER
        + "B0,NACA0012_a4,fvm,100x50,0.4,4,0.045985,0.260987,0.480267,0.519329,0.01,,,measured,ref\n"
        + "E2,NACA0012_a4,pinn,100x50,0.4,4,0.014451,0.082595,0.516286,0.519329,0.0006,577.6,,measured,rounds=20\n"
        + "E2,NACA0012_a4,pinn_gs,100x50,0.4,4,0.0162,0.083,0.5,0.519329,0.0006,121.1,,measured,rounds=4\n"))
    m = next(m for m in pinn_data.showdown()["meshes"] if m["mesh"] == "100x50")
    assert [p["method"] for p in m["pinn"]] == ["pinn"]  # pinn_gs has a row but no case file: no curve, so it is not shown


def test_the_showdown_has_no_file_access_of_its_own():
    """One loader: pinn_data no longer opens files or knows the results/ and NNfoil/ folders (pinn_cases is the only reader)."""
    for gone in ("results_dir", "fvm_dir", "load_fvm_curve", "REPO_ROOT", "os", "csv", "json", "Path"):
        assert not hasattr(pinn_data, gone), gone


def test_exported_data_headline_matches_results_summary_csv():
    """On a checkout with the real exported data: every E2/B0 number shown equals results/summary.csv (the export source)."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[3] / "results" / "summary.csv"
    if not src.exists() or not pinn_cases.has_cases():
        pytest.skip("no results/ or no exported cases here")
    rows = list(csv.DictReader(src.open(encoding="utf8")))
    want = {(r["experiment"], r["method"], r["mesh"]): float(r["rms_cp"]) for r in rows if r["case"] == "NACA0012_a4" and r["rms_cp"] != ""}
    d = pinn_data.showdown()
    m = next(m for m in d["meshes"] if m["mesh"] == "100x50")
    assert m["fvm"]["rms"] == want[("B0", "fvm", "100x50")]
    assert next(p for p in m["pinn"] if p["method"] == "pinn")["rms"] == want[("E2", "pinn", "100x50")]
    assert next(p for p in m["pinn"] if p["method"] == "pinn")["final_round"] == 20  # the final round of the 20-round run
    for mesh in d["meshes"]:  # every run of the card has its exported case and a curve
        assert mesh["fvm"]["curve"] is not None and all(p["status"] == "done" and p["curve"]["x"] for p in mesh["pinn"])
    assert sum(len(m["pinn"]) for m in d["meshes"]) == len(pinn_data.E2_RUNS)
