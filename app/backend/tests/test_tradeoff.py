"""Accuracy-against-cost data: errors and times come from summary.csv; grid sequencing is charged for its coarse run too."""

import pytest
from fastapi.testclient import TestClient

from aeronet import neural, tradeoff
from aeronet.main import app
from conftest import csv_rows, write_results

HEADER = "experiment,case,method,mesh,mach,alpha_deg,rms_cp,max_cp,cl,cl_ref,cd,train_s,infer_ms,source,notes\n"


def _row(exp, method, mesh, rms, train, notes=""):
    return f"{exp},NACA0012_a4,{method},{mesh},0.4,4,{rms},0.1,0.5,0.52,0,{train},,measured,\"{notes}\"\n"


@pytest.fixture
def summary(pinn_dir, monkeypatch):
    write_results(pinn_dir, csv_rows(
        HEADER
        + _row("B0", "fvm", "50x25", 0.108, "")
        + _row("B0", "fvm", "100x50", 0.046, "")
        + _row("E2", "panel", "140 panels", 0.0135, "")
        + _row("E2", "pinn", "50x25", 0.07, 300.0, "rounds=19/20")
        + _row("E2", "pinn", "100x50", 0.0145, 600.0, "rounds=20/20")
        + _row("E2", "pinn_gs", "100x50", 0.0162, 120.0, "rounds=4/4; init_from=naca0012_50x25_a4_m04")
        + _row("E2", "pinn_gs_nopos", "200x100", 0.0092, "", "init_from=naca0012_100x50_a4_m04_gs_nopos; train_s withheld: the laptop slept")
    ))
    monkeypatch.setattr(tradeoff, "measure_panel_seconds", lambda: 0.0012)
    return pinn_dir


def test_errors_and_times_are_the_summary_values(summary):
    d = tradeoff.tradeoff()
    by_id = {p["id"]: p for p in d["points"]}
    assert by_id["100x50:pinn"]["rms"] == 0.0145 and by_id["100x50:pinn"]["seconds"] == 600.0 and by_id["100x50:pinn"]["total_seconds"] == 600.0
    assert d["panel"]["rms"] == 0.0135 and d["panel"]["seconds"] == 0.0012
    assert [(f["mesh"], f["rms"], f["paper_seconds"]) for f in d["fvm"]] == [("50x25", 0.108, 9.0), ("100x50", 0.046, 24.0)]


def test_grid_sequencing_total_includes_the_coarse_run(summary):
    gs = {p["id"]: p for p in tradeoff.tradeoff()["points"]}["100x50:pinn_gs"]
    assert gs["seconds"] == 120.0 and gs["total_seconds"] == 420.0  # 120 s fine stage + 300 s coarse run


def test_withheld_time_is_not_invented(summary):
    p = {p["id"]: p for p in tradeoff.tradeoff()["points"]}["200x100:pinn_gs_nopos"]
    assert p["seconds"] is None and p["total_seconds"] is None and "slept" in p["seconds_note"]


def test_fvm_times_are_labelled_as_quoted(summary):
    assert all("quoted" in f["time_source"] for f in tradeoff.tradeoff()["fvm"])


def test_neuralfoil_has_no_point_only_a_time(summary):
    nf = tradeoff.tradeoff()["neuralfoil"]
    assert nf["available"] == neural.available()
    if nf["available"]:
        assert nf["call_ms"] > 0 and "not a pressure distribution" in nf["note"]


def test_endpoint_and_missing_data(summary):
    assert TestClient(app).get("/api/tradeoff").status_code == 200
    (summary / "results_summary.json").unlink()
    r = TestClient(app).get("/api/tradeoff")
    assert r.status_code == 200 and r.json()["points"] == [] and r.json()["panel"] is None


def test_measured_panel_time_is_a_positive_small_number():
    assert 0.0 < tradeoff.measure_panel_seconds() < 1.0
