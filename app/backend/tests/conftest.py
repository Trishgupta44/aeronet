"""Shared helpers for the tests that need exported PINN data (cases, results rows, seed runs) in a temporary folder."""

import csv
import io
import json

import pytest

from aeronet import pinn_cases

SUMMARY_HEADER = "experiment,case,method,mesh,mach,alpha_deg,rms_cp,max_cp,cl,cl_ref,cd,train_s,infer_ms,source,notes\n"
NUMERIC = ("mach", "alpha_deg", "rms_cp", "max_cp", "cl", "cl_ref", "cd", "train_s", "infer_ms")


def csv_rows(text: str) -> list[dict]:
    """summary.csv text -> the row dicts of results_summary.json (numbers as float, empty cells None)."""
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        rows.append({k: (None if v == "" else float(v) if k in NUMERIC else v) for k, v in r.items()})
    return rows


def make_case(case_id="fixture_case", **over) -> dict:
    """A minimal valid case (roadmap 5.7): a 3 x 2 field, 4 surface points, 2 snapshots."""
    case = {
        "id": case_id, "label": "Fixture case", "shape": "NACA0012", "mesh": "100x50", "mach": 0.4, "alpha": 4.0,
        "source": {"results": "measured", "condition": "quoted"},
        "config": {"rounds": 2, "rounds_done": 2, "failed_rounds": 0, "positivity": True, "seed": 0},
        "x": [1.0, 0.5, 0.0, 0.5], "y": [0.0, 0.06, 0.0, -0.06],
        "surface": {"x": [0.75, 0.25, 0.25, 0.75], "y": [0.03, 0.04, -0.04, -0.03], "cp": [0.1, 0.2, 0.2, 0.1]},
        "reference": {"name": "FVM 800×400", "points": [[0.1, 0.05, 0.3], [0.5, -0.06, 0.1]]},
        "fvm_same_mesh": None,
        "panel": {"cl": 0.5, "surface": {"x": [0.5], "y": [0.0], "cp": [0.0]}},
        "metrics": {"rms": 0.0145, "max": 0.08, "n": 654, "cl": 0.516, "cd": 0.0007, "cl_ref": 0.519, "collapsed": False},
        "timing": {"train_s": 577.6, "infer_ms": 3.75, "threads": 8, "cpu": "test cpu"},
        "log": {"round": [1, 2], "time_s": [10.0, 20.0], "loss": [1e-3, 5e-4], "loss_f": [1e-3, 5e-4], "loss_b": [1e-3, 5e-4],
                "cl": [0.4, 0.5], "cd": [0.01, 0.001], "failed_rounds": 0},
        "snapshots": [{"round": 1, "cp": [0.0, 0.0, 0.0, 0.0]}, {"round": 2, "cp": [0.1, 0.2, 0.2, 0.1]}],
        "field": {"width": 3, "height": 2, "xmin": -0.4, "xmax": 1.4, "ymin": -0.5, "ymax": 0.5,
                  "cp": [0.1, None, 0.2, 0.3, 0.1, None], "u": [1.0, None, 1.0, 1.0, 1.0, None], "v": [0.0, None, 0.0, 0.0, 0.0, None]},
    }
    case.update(over)
    return case


@pytest.fixture
def pinn_dir(tmp_path, monkeypatch):
    """An empty, isolated data folder: the loader reads nothing from the real app/backend/data while a test uses it."""
    d = tmp_path / "data"
    (d / "pinn_cases").mkdir(parents=True)
    monkeypatch.setattr(pinn_cases, "CASES_DIR", d / "pinn_cases")
    monkeypatch.setattr(pinn_cases, "RESULTS_FILE", d / "results_summary.json")
    monkeypatch.setattr(pinn_cases, "SEED_RUNS_FILE", d / "seed_runs.json")
    monkeypatch.setattr(pinn_cases, "ROUNDS_E1_FILE", d / "rounds_e1.json")
    return d


def write_case(d, case: dict) -> None:
    (d / "pinn_cases" / f"{case['id']}.json").write_text(json.dumps(case), encoding="utf8")


def write_results(d, rows) -> None:
    (d / "results_summary.json").write_text(json.dumps(rows), encoding="utf8")


def write_seed_runs(d, groups) -> None:
    (d / "seed_runs.json").write_text(json.dumps({"groups": groups}), encoding="utf8")
