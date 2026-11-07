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

from . import compressibility

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


def _summary_row(rows, experiment: str, method: str, mesh: str, case: str | None = None) -> dict | None:
    for r in rows:
        if (r["experiment"], r["case"], r["method"], r["mesh"]) == (experiment, case or CASE["case_id"], method, mesh):
            return r
    return None


def _note_int(notes: str, key: str) -> int | None:
    m = re.search(rf"{key}=(\d+)", notes or "")
    return int(m.group(1)) if m else None


def _first_int(notes: str, *keys: str) -> int | None:
    for k in keys:
        v = _note_int(notes, k)
        if v is not None:
            return v
    return None


def _withheld_note(notes: str) -> str | None:
    """Why a training time is missing, when the summary row says so ("train_s withheld: <reason>")."""
    m = re.search(r"train_s withheld: ([^;]+)", notes or "")
    return m.group(1).strip() if m else None


def _read_json(path: Path):
    with path.open(encoding="utf8") as f:
        return json.load(f)


# Seed-to-seed spread. 100x50: results/seed_check.csv (mode, method) rows. 50x25: experiment E1 (summary.csv rows
# NACA0012_a4_s<seed>); seed 0 is the E2 run, seeds 1 and 2 are the 20-round re-runs.
SEED_CHECK_KEYS = {("100x50", "pinn"): ("pos", "cold"), ("100x50", "pinn_gs"): ("pos", "gs"), ("100x50", "pinn_gs_nopos"): ("nopos", "gs")}
E1_RUNS = {
    ("50x25", "pinn"): {0: "naca0012_50x25_a4_m04", 1: "naca0012_50x25_a4_m04_s1_r20", 2: "naca0012_50x25_a4_m04_s2_r20"},
    ("50x25", "pinn_nopos"): {0: "naca0012_50x25_a4_m04_nopos", 1: "naca0012_50x25_a4_m04_nopos_s1_r20", 2: "naca0012_50x25_a4_m04_nopos_s2_r20"},
}


def _final_curve(run_id: str) -> dict | None:
    path = results_dir() / "runs" / run_id / "snapshots.json"
    if not path.exists():
        return None
    snap = _read_json(path)
    if not snap.get("rounds"):
        return None
    return _curve(snap["x"], snap["y"], snap["rounds"][-1]["cp"])


def _seed_runs(rows, mesh: str, method: str) -> tuple[str, list[dict]] | None:
    """(source label, one dict per seed with seed, run_id, rms, cl, rounds, failed_rounds, collapsed) or None."""
    if (mesh, method) in SEED_CHECK_KEYS:
        mode, meth = SEED_CHECK_KEYS[(mesh, method)]
        path = results_dir() / "seed_check.csv"
        if not path.exists():
            return None
        with path.open(newline="", encoding="utf8") as f:
            sel = [r for r in csv.DictReader(f) if r["mode"] == mode and r["method"] == meth]
        runs = [{"seed": int(r["seed"]), "run_id": r["run_id"], "rms": _num(r["cp_rms"]), "cl": _num(r["final_cl"]),
                 "rounds": int(r["rounds"] or 0), "failed_rounds": int(r["failed_rounds"] or 0), "collapsed": r["collapsed"] == "yes"}
                for r in sel]
        return "results/seed_check.csv", sorted(runs, key=lambda d: d["seed"])
    if (mesh, method) in E1_RUNS:
        runs = []
        for seed, run_id in E1_RUNS[(mesh, method)].items():
            row = _summary_row(rows, "E1", method, mesh, case=f"{CASE['case_id']}_s{seed}")
            if row is None:
                continue
            runs.append({"seed": seed, "run_id": run_id, "rms": _num(row["rms_cp"]), "cl": _num(row["cl"]),
                         "rounds": _note_int(row["notes"], "rounds") or 0, "failed_rounds": None, "collapsed": False})
        return ("results/summary.csv (experiment E1)", runs) if runs else None
    return None


def _range(values: list[float]) -> dict | None:
    return {"min": min(values), "max": max(values)} if values else None


def _seed_spread(rows, mesh: str, method: str) -> dict | None:
    """Min-max of Cl and Cp RMS over the seeds that produced a result, and a per-point Cp band over their final rounds."""
    found = _seed_runs(rows, mesh, method)
    if found is None:
        return None
    source, runs = found
    done = [r for r in runs if r["rms"] is not None]
    band = None
    curves = [c for c in (_final_curve(r["run_id"]) for r in done) if c is not None]
    if len(curves) >= 2 and all(c["x"] == curves[0]["x"] for c in curves):
        cps = np.array([c["cp"] for c in curves])
        band = {"x": curves[0]["x"], "upper": curves[0]["upper"], "cp_min": cps.min(axis=0).round(5).tolist(), "cp_max": cps.max(axis=0).round(5).tolist(),
                "n_curves": len(curves)}
    return {
        "source": source,
        "n": len(runs),
        "n_with_result": len(done),
        "n_without_result": len(runs) - len(done),
        "cl": _range([r["cl"] for r in done if r["cl"] is not None]),
        "rms": _range([r["rms"] for r in done]),
        "rounds_differ": len({r["rounds"] for r in done}) > 1,
        "runs": runs,
        "band": band,
    }


def _pinn_validity(curve: dict) -> dict:
    """Is the PINN's own wall Cp compatible with subsonic flow at the case Mach number? (Cp_min against Cp*, see compressibility.py)"""
    cp_min = min(curve["cp"])
    star = compressibility.cp_crit(CASE["mach"])
    return {"cp_min": cp_min, "cp_crit": star, "supersonic": bool(star is not None and cp_min < star)}


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
    curve = _curve(snap["x"], snap["y"], last["cp"])
    return {
        **base,
        "status": "done",
        "source": "measured",
        "rms": _num(row["rms_cp"]),
        "max": _num(row["max_cp"]),
        "cl": _num(row["cl"]),
        "train_s": _num(row["train_s"]),
        "rounds": _note_int(row["notes"], "rounds"),
        # older rows say failed_rounds=N, newer ones say restarts=N (a failed round is restarted): same count
        "failed_rounds": _first_int(row["notes"], "failed_rounds", "restarts"),
        "train_s_note": _withheld_note(row["notes"]) if _num(row["train_s"]) is None else None,
        "final_round": int(last["round"]),
        "curve": curve,
        "validity": _pinn_validity(curve),
        "seeds": _seed_spread(rows, mesh, method),
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
