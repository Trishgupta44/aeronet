"""The same-mesh showdown (FVM vs PINN on each mesh), assembled from the exported data. This module reads NO file itself.

The single loader is :mod:`aeronet.pinn_cases`, which serves what ``research/export_results.py`` wrote to ``app/backend/data``:

  * ``results_summary.json`` -> errors, Cl, training seconds (experiment B0 = FVM rows, E2 = PINN rows, both vs FVM 800x400),
  * ``pinn_cases/<run>.json`` -> the PINN's final-round wall Cp, round counts, the FVM wall Cp on the same mesh and the 800x400 reference,
  * ``seed_runs.json``        -> the seed-to-seed spread groups with their final-round curves.

So the showdown works from a clean clone (no ``NNfoil/``, no ``results/runs/``). Earlier versions of this file read
``results/summary.csv``, ``NNfoil/FVM_Cp/`` and the run folders directly (with the ``AERONET_RESULTS_DIR`` and ``AERONET_FVM_DIR``
overrides); that second path is gone, so there is one place that knows where the numbers live.

Honesty: the PINN curve is always the FINAL round, never the "best" round (picking the best would use the FVM answer).
A run with no summary row is reported as pending and no curve is shown for it.
"""

from __future__ import annotations

import re

import numpy as np

from . import compressibility, config, pinn_cases

CASE = {"shape": "NACA0012", "alpha_deg": 4.0, "mach": 0.4, "case_id": "NACA0012_a4"}
REFERENCE_MESH = "800x400"
MESHES = ("50x25", "100x50", "200x100")

# (mesh, summary method) -> case id (= run folder name). Mirrors research/experiments.py experiment E2.
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


def _num(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
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


def _points_curve(points) -> dict:
    p = np.asarray(points, dtype=float)
    return _curve(p[:, 0], p[:, 1], p[:, 2])


def _read_summary() -> list[dict]:
    return pinn_cases.load_results()


def _summary_row(rows, experiment: str, method: str, mesh: str, case: str | None = None) -> dict | None:
    for r in rows:
        if (r["experiment"], r["case"], r["method"], r["mesh"]) == (experiment, case or CASE["case_id"], method, mesh):
            return r
    return None


def _note_int(notes: str | None, key: str) -> int | None:
    m = re.search(rf"{key}=(\d+)", notes or "")
    return int(m.group(1)) if m else None


def _first_int(notes: str | None, *keys: str) -> int | None:
    for k in keys:
        v = _note_int(notes, k)
        if v is not None:
            return v
    return None


def _withheld_note(notes: str | None) -> str | None:
    """Why a training time is missing, when the summary row says so ("train_s withheld: <reason>")."""
    m = re.search(r"train_s withheld: ([^;]+)", notes or "")
    return m.group(1).strip() if m else None


def _case(run_id: str) -> dict | None:
    """The exported case (without its field) of a run, or None if it was not exported."""
    try:
        return pinn_cases.get_case_meta(run_id)
    except KeyError:
        return None


def _range(values: list[float]) -> dict | None:
    return {"min": min(values), "max": max(values)} if values else None


def _seed_spread(mesh: str, method: str) -> dict | None:
    """Min-max of Cl and Cp RMS over the seeds that produced a result, and a per-point Cp band over their final rounds."""
    group = pinn_cases.load_seed_runs().get(f"{mesh}:{method}")
    if not group or not group.get("runs"):
        return None
    runs = group["runs"]
    done = [r for r in runs if r["rms"] is not None]
    band = None
    curves = [_curve(r["curve"]["x"], r["curve"]["y"], r["curve"]["cp"]) for r in done if r.get("curve")]
    if len(curves) >= 2 and all(c["x"] == curves[0]["x"] for c in curves):
        cps = np.array([c["cp"] for c in curves])
        band = {"x": curves[0]["x"], "upper": curves[0]["upper"], "cp_min": cps.min(axis=0).round(5).tolist(),
                "cp_max": cps.max(axis=0).round(5).tolist(), "n_curves": len(curves)}
    return {
        "source": group.get("source"),
        "n": len(runs),
        "n_with_result": len(done),
        "n_without_result": len(runs) - len(done),
        "cl": _range([r["cl"] for r in done if r["cl"] is not None]),
        "rms": _range([r["rms"] for r in done]),
        "rounds_differ": len({r["rounds"] for r in done}) > 1,
        "runs": [{k: v for k, v in r.items() if k != "curve"} for r in runs],
        "band": band,
    }


def _pinn_validity(curve: dict) -> dict:
    """Is the PINN's own wall Cp compatible with subsonic flow at the case Mach number? (Cp_min against Cp*, see compressibility.py)"""
    cp_min = min(curve["cp"])
    star = compressibility.cp_crit(CASE["mach"])
    return {"cp_min": cp_min, "cp_crit": star, "supersonic": bool(star is not None and cp_min < star)}


def _pinn_entry(rows, mesh: str, method: str, run_id: str) -> dict | None:
    case = _case(run_id)
    row = _summary_row(rows, "E2", method, mesh)
    base = {"method": method, "label": METHOD_LABELS[method], "run_id": run_id}
    if case is None:  # not exported (a row without a case cannot show a curve, so the run is left out)
        return None
    if row is None:
        cfg = case.get("config") or {}
        return {**base, "status": "pending", "rounds_done": len(case["log"].get("round", [])), "rounds_target": cfg.get("rounds")}
    s = case["surface"]
    curve = _curve(s["x"], s["y"], s["cp"])
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
        "final_round": int(case["snapshots"][-1]["round"]),
        "curve": curve,
        "validity": _pinn_validity(curve),
        "seeds": _seed_spread(mesh, method),
    }


def showdown() -> dict:
    """Everything the same-mesh showdown card needs (see roadmap feature 1)."""
    rows = _read_summary()
    withheld = config.withhold_fvm_data()  # public deployments: the error numbers yes, NNfoil's FVM Cp curves no (it has no licence)
    cases = {run_id: _case(run_id) for run_id in E2_RUNS.values()}  # in public mode the cases arrive without any FVM curve
    ref_points = next((c["reference"]["points"] for c in cases.values() if c and c.get("reference")), None)
    if ref_points is None and not withheld:
        raise ValueError("no exported PINN case with an FVM reference (run research/export_results.py)")
    meshes = []
    for mesh in MESHES:
        fvm_row = _summary_row(rows, "B0", "fvm", mesh)
        fvm = None
        if fvm_row is not None:
            same = None if withheld else next((c["fvm_same_mesh"] for (m, _), rid in E2_RUNS.items() if m == mesh and (c := cases[rid]) and c.get("fvm_same_mesh")), None)
            fvm = {
                "source": "measured",
                "rms": _num(fvm_row["rms_cp"]),
                "max": _num(fvm_row["max_cp"]),
                "cl": _num(fvm_row["cl"]),
                "curve": _points_curve(same["points"]) if same else None,
            }
        pinn = [e for (m, meth), run_id in E2_RUNS.items() if m == mesh and (e := _pinn_entry(rows, m, meth, run_id))]
        meshes.append({"mesh": mesh, "fvm": fvm, "pinn": pinn})
    ref_row = _summary_row(rows, "B0", "fvm", REFERENCE_MESH)
    return {
        "case": CASE,
        "reference": {
            "label": f"FVM {REFERENCE_MESH}",
            "cl": _num(ref_row["cl"]) if ref_row else None,
            "curve": None if withheld or ref_points is None else _points_curve(ref_points),
        },
        "fvm_curves_withheld": withheld,
        "error_definition": "RMS and max |Cp - Cp_ref| at the reference points with 0.02 <= x/c <= 0.98, nearest wall point (research/metrics.py)",
        "meshes": meshes,
    }
