"""The ONE loader for trained-PINN results (roadmap C6 to C8 will replace it with exported pinn_cases JSON).

Reads, without recomputing anything:
  * ``results/summary.csv``   -> errors, Cl, training seconds (experiment B0 = FVM rows, E2 = PINN rows, both vs FVM 800x400)
  * ``NNfoil/FVM_Cp/*.dat``   -> FVM wall Cp on each mesh and the 800x400 reference
  * ``results/runs/<run>/``   -> final-round wall Cp (``snapshots.json``), round counts (``log.json``, ``config.json``)

Folders can be overridden with the environment variables ``AERONET_RESULTS_DIR`` and ``AERONET_FVM_DIR`` (used when
the app runs from a checkout that does not contain the run files).

Honesty: the PINN curve is always the FINAL round, never the "best" round (picking the best would use the FVM answer).
A run with no summary row is reported as pending and no curve is shown for it.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE = {"shape": "NACA0012", "alpha_deg": 4.0, "mach": 0.4, "case_id": "NACA0012_a4"}
REFERENCE_MESH = "800x400"
MESHES = ("50x25", "100x50", "200x100")

# (mesh, summary method) -> run folder name. Mirrors research/experiments.py experiment E2.
E2_RUNS = {
    ("50x25", "pinn"): "naca0012_50x25_a4_m04",
    ("50x25", "pinn_nopos"): "naca0012_50x25_a4_m04_nopos",
    ("100x50", "pinn"): "naca0012_100x50_a4_m04",
    ("100x50", "pinn_gs"): "naca0012_100x50_a4_m04_gs",
    ("100x50", "pinn_gs_nopos"): "naca0012_100x50_a4_m04_gs_nopos",
    ("200x100", "pinn_gs_nopos"): "naca0012_200x100_a4_m04_gs_nopos",
}
METHOD_LABELS = {
    "pinn": "PINN, cold start",
    "pinn_nopos": "PINN, cold start, no positivity",
    "pinn_gs": "PINN, grid-sequenced (from 50x25)",
    "pinn_gs_nopos": "PINN, grid-sequenced, no positivity",
}


def results_dir() -> Path:
    return Path(os.environ.get("AERONET_RESULTS_DIR") or REPO_ROOT / "results")


def fvm_dir() -> Path:
    return Path(os.environ.get("AERONET_FVM_DIR") or REPO_ROOT / "NNfoil" / "FVM_Cp")


def _num(text: str | None) -> float | None:
    try:
        return float(text) if text not in (None, "") else None
    except ValueError:
        return None


def _curve(x, y, cp) -> dict:
    """Wall Cp as plain lists in the app frame; ``upper`` is y >= 0 (the demo shape is NACA 0012)."""
    x, y, cp = (np.asarray(a, dtype=float) for a in (x, y, cp))
    order = np.argsort(x, kind="stable")
    return {
        "x": [round(float(v), 5) for v in x[order]],
        "cp": [round(float(v), 5) for v in cp[order]],
        "upper": [bool(v >= 0.0) for v in y[order]],
    }


def load_fvm_curve(mesh: str) -> dict:
    """FVM wall Cp of NACA0012 on ``mesh`` ("100x50") from the bundled NNfoil file (x_app = x + 0.5)."""
    m, n = mesh.split("x")
    path = fvm_dir() / f"NACA0012_{m}_{n}_Cp.dat"
    d = np.loadtxt(path, skiprows=1)
    return _curve(d[:, 1] + 0.5, d[:, 2], d[:, 3])


def _read_summary() -> list[dict]:
    path = results_dir() / "summary.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def _summary_row(rows, experiment: str, method: str, mesh: str) -> dict | None:
    for r in rows:
        if (r["experiment"], r["case"], r["method"], r["mesh"]) == (experiment, CASE["case_id"], method, mesh):
            return r
    return None


def _note_int(notes: str, key: str) -> int | None:
    m = re.search(rf"{key}=(\d+)", notes or "")
    return int(m.group(1)) if m else None


def _read_json(path: Path):
    with path.open(encoding="utf8") as f:
        return json.load(f)


def _pinn_entry(rows, mesh: str, method: str, run_id: str) -> dict | None:
    run = results_dir() / "runs" / run_id
    row = _summary_row(rows, "E2", method, mesh)
    base = {"method": method, "label": METHOD_LABELS[method], "run_id": run_id}
    if row is None:
        if not (run / "log.json").exists():
            return None
        log = _read_json(run / "log.json")
        target = _read_json(run / "config.json").get("rounds") if (run / "config.json").exists() else None
        return {**base, "status": "pending", "rounds_done": len(log.get("round", [])), "rounds_target": target}
    snap = _read_json(run / "snapshots.json")
    last = snap["rounds"][-1]
    return {
        **base,
        "status": "done",
        "source": "measured",
        "rms": _num(row["rms_cp"]),
        "max": _num(row["max_cp"]),
        "cl": _num(row["cl"]),
        "train_s": _num(row["train_s"]),
        "rounds": _note_int(row["notes"], "rounds"),
        "failed_rounds": _note_int(row["notes"], "failed_rounds"),
        "final_round": int(last["round"]),
        "curve": _curve(snap["x"], snap["y"], last["cp"]),
    }


def showdown() -> dict:
    """Everything the same-mesh showdown card needs (see roadmap feature 1)."""
    rows = _read_summary()
    meshes = []
    for mesh in MESHES:
        fvm_row = _summary_row(rows, "B0", "fvm", mesh)
        fvm = None
        if fvm_row is not None:
            fvm = {
                "source": "measured",
                "rms": _num(fvm_row["rms_cp"]),
                "max": _num(fvm_row["max_cp"]),
                "cl": _num(fvm_row["cl"]),
                "curve": load_fvm_curve(mesh),
            }
        pinn = [e for (m, meth), run_id in E2_RUNS.items() if m == mesh and (e := _pinn_entry(rows, m, meth, run_id))]
        meshes.append({"mesh": mesh, "fvm": fvm, "pinn": pinn})
    ref_row = _summary_row(rows, "B0", "fvm", REFERENCE_MESH)
    return {
        "case": CASE,
        "reference": {
            "label": f"FVM {REFERENCE_MESH}",
            "cl": _num(ref_row["cl"]) if ref_row else None,
            "curve": load_fvm_curve(REFERENCE_MESH),
        },
        "error_definition": "RMS and max |Cp - Cp_ref| at the reference points with 0.02 <= x/c <= 0.98, nearest wall point (research/metrics.py)",
        "meshes": meshes,
    }
