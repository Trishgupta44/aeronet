"""Same-mesh showdown loader: numbers must be the ones in results/summary.csv, final round only, pending shown as pending."""

import csv
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aeronet import pinn_data
from aeronet.main import app

HEADER = "experiment,case,method,mesh,mach,alpha_deg,rms_cp,max_cp,cl,cl_ref,cd,train_s,infer_ms,source,notes\n"
FVM_HEADER = "nodenumber x-coordinate y-coordinate pressure-coefficient\n"


def _fvm_file(path, scale):
    ang = np.linspace(0.1, 2 * np.pi - 0.1, 24)
    x, y = 0.5 * np.cos(ang), 0.06 * np.sin(ang)
    rows = [f"{i + 1} {x[i]:.6e} {y[i]:.6e} {scale * (1 - 4 * x[i] ** 2):.6e}" for i in range(24)]
    path.write_text(FVM_HEADER + "\n".join(rows) + "\n")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    res, fvm = tmp_path / "results", tmp_path / "fvm"
    (res / "runs").mkdir(parents=True)
    fvm.mkdir()
    for name in ("50_25", "100_50", "200_100", "800_400"):
        _fvm_file(fvm / f"NACA0012_{name}_Cp.dat", 1.0)
    (res / "summary.csv").write_text(
        HEADER
        + "B0,NACA0012_a4,fvm,100x50,0.4,4,0.045985,0.260987,0.480267,0.519329,0.01,,,measured,ref\n"
        + "E2,NACA0012_a4,pinn,100x50,0.4,4,0.014451,0.082595,0.516286,0.519329,0.0006,577.6,,measured,final round; rounds=20; failed_rounds=0\n"
    )
    done = res / "runs" / "naca0012_100x50_a4_m04"
    done.mkdir()
    x = np.linspace(0.05, 0.95, 10).tolist()
    y = [0.03] * 5 + [-0.03] * 5
    (done / "snapshots.json").write_text(json.dumps({"x": x, "y": y, "rounds": [{"round": 1, "cp": [9.0] * 10}, {"round": 2, "cp": [0.1] * 10}]}))
    pend = res / "runs" / "naca0012_200x100_a4_m04_gs_nopos"
    pend.mkdir()
    (pend / "log.json").write_text(json.dumps({"round": [1, 2, 3]}))
    (pend / "config.json").write_text(json.dumps({"rounds": 30}))
    monkeypatch.setenv("AERONET_RESULTS_DIR", str(res))
    monkeypatch.setenv("AERONET_FVM_DIR", str(fvm))
    return res


def test_numbers_come_from_the_summary_and_the_final_round(fake):
    d = pinn_data.showdown()
    m = next(m for m in d["meshes"] if m["mesh"] == "100x50")
    assert m["fvm"]["rms"] == 0.045985 and m["fvm"]["source"] == "measured"
    p = m["pinn"][0]
    assert p["status"] == "done" and p["rms"] == 0.014451 and p["max"] == 0.082595 and p["train_s"] == 577.6
    assert p["rounds"] == 20 and p["failed_rounds"] == 0
    assert p["final_round"] == 2 and set(p["curve"]["cp"]) == {0.1}  # final round, not the first and not a "best" round
    assert sum(p["curve"]["upper"]) == 5


def test_seed_spread_comes_from_seed_check_and_collapsed_seeds_count_but_give_no_range(fake):
    head = "mode,method,seed,run_id,cl_after_transfer,final_cl,cp_rms,rounds,failed_rounds,seconds,collapsed,complete\n"
    (fake / "seed_check.csv").write_text(
        head
        + "pos,cold,0,naca0012_100x50_a4_m04,,0.516286,0.014451,20,0,577.6,no,yes\n"
        + "pos,cold,1,naca0012_100x50_a4_m04_s1,,0.506552,0.015379,5,0,324.6,no,yes\n"
        + "pos,cold,2,naca0012_100x50_a4_m04_s2,,,,0,5,,yes,yes\n"
        + "pos,gs,0,some_other_run,0.4,0.5,0.0161,4,0,120.0,no,yes\n"  # other variant: must not leak into the pinn spread
    )
    s1 = fake / "runs" / "naca0012_100x50_a4_m04_s1"
    s1.mkdir()
    x = np.linspace(0.05, 0.95, 10).tolist()
    (s1 / "snapshots.json").write_text(json.dumps({"x": x, "y": [0.03] * 5 + [-0.03] * 5, "rounds": [{"round": 5, "cp": [0.3] * 10}]}))
    p = next(m for m in pinn_data.showdown()["meshes"] if m["mesh"] == "100x50")["pinn"][0]
    s = p["seeds"]
    assert s["source"] == "results/seed_check.csv" and s["n"] == 3 and s["n_with_result"] == 2 and s["n_without_result"] == 1
    assert s["cl"] == {"min": 0.506552, "max": 0.516286} and s["rms"] == {"min": 0.014451, "max": 0.015379}
    assert s["rounds_differ"] is True
    assert s["band"]["n_curves"] == 2 and s["band"]["cp_min"] == [0.1] * 10 and s["band"]["cp_max"] == [0.3] * 10
    assert p["validity"]["supersonic"] is False and p["validity"]["cp_crit"] < -3


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


def test_endpoint_reports_missing_data(tmp_path, monkeypatch):
    monkeypatch.setenv("AERONET_RESULTS_DIR", str(tmp_path / "none"))
    monkeypatch.setenv("AERONET_FVM_DIR", str(tmp_path / "none"))
    assert TestClient(app).get("/api/showdown").status_code == 503


def test_real_summary_headline_if_present():
    """On a checkout that has the real results: the 100x50 numbers shown equal results/summary.csv."""
    path = pinn_data.results_dir() / "summary.csv"
    if not path.exists() or not (pinn_data.fvm_dir() / "NACA0012_100_50_Cp.dat").exists():
        pytest.skip("no results/ here")
    rows = list(csv.DictReader(path.open(encoding="utf8")))
    want = {(r["experiment"], r["method"], r["mesh"]): float(r["rms_cp"]) for r in rows if r["case"] == "NACA0012_a4" and r["rms_cp"] != ""}
    m = next(m for m in pinn_data.showdown()["meshes"] if m["mesh"] == "100x50")
    assert m["fvm"]["rms"] == want[("B0", "fvm", "100x50")]
    if ("E2", "pinn", "100x50") in want:
        assert next(p for p in m["pinn"] if p["method"] == "pinn")["rms"] == want[("E2", "pinn", "100x50")]
