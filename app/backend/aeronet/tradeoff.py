"""Accuracy against cost: Cp error (vs the FVM 800x400 solution) against seconds spent on THIS laptop.

Sources (nothing is retyped except the quoted paper times, which are labelled):
  * ``results/summary.csv`` experiment E2: Cp RMS and training seconds of every trained PINN (measured), the panel method's Cp RMS,
    and experiment B0: the Cp RMS of the FVM meshes. Training seconds are wall-clock seconds on the project laptop.
  * the panel method's time is measured NOW, on the server answering the request (median of several solves).
  * the FVM times are QUOTED from the NNfoil paper (Table 1) and come from different hardware, so they are never put on the
    time axis; the chart shows the FVM meshes as horizontal error lines and gives the quoted time only as text.
  * NeuralFoil outputs Cl and Cd but no pressure distribution, so it has no Cp error and no point on this chart; its measured
    time per call is returned for the table.

Grid sequencing is reported with BOTH costs: the fine stage alone, and the total including the coarse run it started from
(a run's ``init_from``, followed back to a cold start).
"""

from __future__ import annotations

import re
import statistics
from time import perf_counter

from . import neural, pinn_data
from .panel import solve as panel_solve

# NNfoil paper, Table 1 (NACA 0012, 4 degrees, Mach 0.4), as quoted in docs/verification.md. Different hardware; the paper says
# the table is "not meant as a rigorous comparison".
FVM_PAPER_SECONDS = {"50x25": 9.0, "100x50": 24.0, "200x100": 53.0, "400x200": 285.0}
FVM_MESHES = ("50x25", "100x50", "200x100", "400x200")
_PANEL_RUNS = 7


def measure_panel_seconds() -> float:
    """Median wall time of one panel solve (NACA 0012, 4 deg, Mach 0.4, 140 panels), after one warm-up solve."""
    panel_solve("0012", 4.0, 0.4, 70)
    times = []
    for _ in range(_PANEL_RUNS):
        t0 = perf_counter()
        panel_solve("0012", 4.0, 0.4, 70)
        times.append(perf_counter() - t0)
    return statistics.median(times)


def _num(text):
    return pinn_data._num(text)


def _total_seconds(rows, key, run_to_key, depth=0):
    """Training seconds of this run plus those of the run it was initialised from, recursively; None if any is withheld."""
    mesh, method = key
    row = pinn_data._summary_row(rows, "E2", method, mesh)
    own = _num(row["train_s"]) if row else None
    if own is None:
        return None
    m = re.search(r"init_from=([\w.]+)", row["notes"] or "")
    if not m:
        return own
    parent = run_to_key.get(m.group(1))
    if parent is None or depth > 4:
        return None
    up = _total_seconds(rows, parent, run_to_key, depth + 1)
    return None if up is None else own + up


def tradeoff() -> dict:
    rows = pinn_data._read_summary()
    run_to_key = {run: key for key, run in pinn_data.E2_RUNS.items()}
    points = []
    for (mesh, method), run_id in pinn_data.E2_RUNS.items():
        row = pinn_data._summary_row(rows, "E2", method, mesh)
        if row is None:
            continue
        seconds = _num(row["train_s"])
        points.append({
            "id": f"{mesh}:{method}",
            "kind": "pinn",
            "mesh": mesh,
            "method": method,
            "label": pinn_data.METHOD_LABELS[method],
            "grid_sequenced": "_gs" in method,
            "rms": _num(row["rms_cp"]),
            "seconds": seconds,  # this run's own training time
            "total_seconds": _total_seconds(rows, (mesh, method), run_to_key),  # including the coarse run(s) it started from
            "seconds_note": pinn_data._withheld_note(row["notes"]) if seconds is None else None,
            "source": "measured",
        })

    fvm = []
    for mesh in FVM_MESHES:
        row = pinn_data._summary_row(rows, "B0", "fvm", mesh)
        if row is not None:
            fvm.append({"mesh": mesh, "rms": _num(row["rms_cp"]), "paper_seconds": FVM_PAPER_SECONDS.get(mesh), "source": "measured",
                        "time_source": "quoted: NNfoil paper, Table 1 (other hardware)"})

    panel = None
    prow = next((r for r in rows if r["experiment"] == "E2" and r["case"] == pinn_data.CASE["case_id"] and r["method"] == "panel"), None)
    if prow is not None:
        panel = {"label": "Panel method (140 panels, Prandtl-Glauert)", "rms": _num(prow["rms_cp"]), "seconds": measure_panel_seconds(),
                 "source": "rms: measured (summary.csv); time: measured now on this server"}

    nf = {"available": neural.available()}
    if nf["available"]:
        nf.update({"call_ms": neural.time_call_ms(), "version": neural.version(),
                   "note": "NeuralFoil gives Cl and Cd, not a pressure distribution, so it has no Cp error and no point on this chart."})

    return {
        "case": pinn_data.CASE,
        "reference": "FVM 800x400",
        "points": points,
        "fvm": fvm,
        "panel": panel,
        "neuralfoil": nf,
    }
