"""Export finished runs to the app (Plan C, task C6; roadmap section 5.7).

    research/.venv/Scripts/python research/export_results.py [--out app/backend/data] [--only <run_id> ...] [--width 340]

Writes, next to each other under ``--out`` (default ``app/backend/data``):

* ``pinn_cases/<run_id>.json``  one self-contained case per run in ``CASES`` (the format of roadmap 5.7, plus a few
  extra keys: ``config``, ``source``, ``fvm_same_mesh``, ``notes``),
* ``results_summary.json``      the rows of ``results/summary.csv`` (numeric columns as numbers, empty cells as null),
* ``seed_runs.json``            seed-to-seed spread groups of the same-mesh showdown (final-round curves of the seed runs),
* ``rounds_e1.json``            the round-by-round Cp RMS and Cl of the six E1 runs (``results/e1_rounds.csv``, about 10 KB), for
  the training-trajectory chart of the Results page.

The backend only serves these files (``aeronet/pinn_cases.py``): no torch, no ``NNfoil/`` and no ``results/runs/`` are
needed at run time. Nothing is recomputed that a run folder or the summary already holds, except the Cp error
(``metrics.cp_error``, the one definition of roadmap 5.5) and the panel solution (``aeronet.panel``).

Honesty rules baked in:

* the PINN curve and the field are the FINAL round (never the "best" round: that would use the FVM answer);
* a run whose training time the summary withheld (the laptop slept) gets ``timing.train_s = null`` and a note, and its
  ``log.time_s`` is nulled too (it carries the same unusable wall time);
* ``source`` says what each number is: the results are **measured** here, the flow condition (Mach, angle) is **quoted**
  from the paper (arXiv 2401.08705); the NACA 2412 Mach number is quoted too (the panel-method fit prefers about 0.25);
* a run that collapsed to the trivial solution is exported like any other, flagged ``metrics.collapsed = true``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT / "app" / "backend"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import metrics  # noqa: E402
import runs  # noqa: E402

APP_DATA = ROOT / "app" / "backend" / "data"
SUMMARY_CSV = ROOT / "results" / "summary.csv"
SEED_CHECK_CSV = ROOT / "results" / "seed_check.csv"
E1_ROUNDS_CSV = ROOT / "results" / "e1_rounds.csv"
E4_TIMING = ROOT / "results" / "e4_timing.json"
FVM_DIR = ROOT / "NNfoil" / "FVM_Cp"

MAX_BYTES = 3_000_000          # roadmap 5.1: a case JSON must stay under 3 MB
WIDTHS = (340, 240, 180, 120, 80)  # field widths tried in turn until the file fits
PANELS_HALF = 70               # 140 panels, the same as the panel rows of the Milestone 3 tables
COLLAPSE_RMS = 1.0             # Cp RMS above this (the Cp range of a plausible answer is about 5) = collapsed

NUMERIC_COLUMNS = ("mach", "alpha_deg", "rms_cp", "max_cp", "cl", "cl_ref", "cd", "train_s", "infer_ms")

# ---------------------------------------------------------------------------------------------- what is exported
# run id, the words that tell this run from its siblings in the label, and its row in results/summary.csv
# (experiment, case, method, mesh): the row supplies the training time (it may be withheld) and is cross-checked.
_P = "NACA0012_a4"
CASES = [
    {"run": "naca0012_50x25_a4_m04", "variant": "positivity on", "row": ("E2", _P, "pinn", "50x25")},
    {"run": "naca0012_50x25_a4_m04_nopos", "variant": "positivity off", "row": ("E2", _P, "pinn_nopos", "50x25")},
    {"run": "naca0012_100x50_a4_m04", "variant": "cold start, positivity on", "row": ("E2", _P, "pinn", "100x50")},
    {"run": "naca0012_100x50_a4_m04_gs", "variant": "grid-seq., positivity on", "row": ("E2", _P, "pinn_gs", "100x50")},
    {"run": "naca0012_100x50_a4_m04_gs_nopos", "variant": "grid-seq., positivity off",
     "row": ("E2", _P, "pinn_gs_nopos", "100x50")},
    {"run": "naca0012_200x100_a4_m04_gs_nopos", "variant": "grid-seq., positivity off",
     "row": ("E2", _P, "pinn_gs_nopos", "200x100")},
    {"run": "cylinder_50x25_a0_m04", "variant": "positivity on", "row": ("E5", "Cylinder_a0", "pinn", "50x25")},
    {"run": "cylinder_50x25_a0_m04_nopos", "variant": "positivity off", "row": ("E5", "Cylinder_a0", "pinn_nopos", "50x25")},
    {"run": "naca2412_100x50_am4_m03", "variant": "positivity on", "row": ("E3", "NACA2412_a-4", "pinn", "100x50")},
    {"run": "naca2412_100x50_a0_m03", "variant": "positivity on", "row": ("E3", "NACA2412_a0", "pinn", "100x50")},
    {"run": "naca2412_100x50_a4_m03", "variant": "positivity on, seed 0", "row": ("E3", "NACA2412_a4", "pinn", "100x50")},
    {"run": "naca2412_100x50_a4_m03_s1", "variant": "positivity on, seed 1 (retry)",
     "row": ("E3", "NACA2412_a4_s1", "pinn", "100x50")},
    {"run": "naca2412_100x50_a4_m03_nopos", "variant": "positivity off, seed 0 (retry)",
     "row": ("E3", "NACA2412_a4", "pinn_nopos", "100x50")},
    {"run": "naca2412_100x50_a8_m03", "variant": "positivity on", "row": ("E3", "NACA2412_a8", "pinn", "100x50")},
]

CONDITION_SOURCE = {
    "NACA0012": "Mach 0.4 and 4 deg are the condition of the bundled FVM files, quoted from arXiv 2401.08705 section 3.2.",
    "Cylinder": ("Mach 0.4 and 0 deg quoted from arXiv 2401.08705 section 3.1; confirmed by the FVM stagnation point at "
                 "180 deg and its Cp 1.044."),
    "NACA2412": ("Mach 0.3 quoted from arXiv 2401.08705 section 3.4 (not stored in the Cp files); the panel-method fit "
                 "of the files prefers about 0.25 (fitted)."),
}

# Seed groups of the showdown: (mesh, method) -> where the seed runs and their numbers come from.
SEED_CHECK_GROUPS = {("100x50", "pinn"): ("pos", "cold"), ("100x50", "pinn_gs"): ("pos", "gs"),
                     ("100x50", "pinn_gs_nopos"): ("nopos", "gs")}
E1_GROUPS = {
    ("50x25", "pinn"): {0: "naca0012_50x25_a4_m04", 1: "naca0012_50x25_a4_m04_s1_r20", 2: "naca0012_50x25_a4_m04_s2_r20"},
    ("50x25", "pinn_nopos"): {0: "naca0012_50x25_a4_m04_nopos", 1: "naca0012_50x25_a4_m04_nopos_s1_r20",
                              2: "naca0012_50x25_a4_m04_nopos_s2_r20"},
}


# ---------------------------------------------------------------------------------------------- small helpers
def _round_list(values, nd) -> list:
    return [None if not np.isfinite(v) else round(float(v), nd) for v in np.asarray(values, dtype=float).ravel()]


def _points(arr, nd=5) -> list:
    return [[round(float(v), nd) for v in row] for row in np.asarray(arr, dtype=float)]


def _num(text):
    try:
        return float(text) if text not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _note_int(notes, key):
    m = re.search(rf"{key}=(\d+)", notes or "")
    return int(m.group(1)) if m else None


def read_summary_rows(path=SUMMARY_CSV) -> list[dict]:
    """The rows of ``summary.csv``: numeric columns become float, empty cells become ``None``."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if v in (None, ""):
                    row[k] = None
                elif k in NUMERIC_COLUMNS:
                    row[k] = _num(v) if _num(v) is not None else v
                else:
                    row[k] = v
            rows.append(row)
    return rows


def find_row(rows, key):
    return next((r for r in rows if (r["experiment"], r["case"], r["method"], r["mesh"]) == tuple(key)), None)


def _dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def write_summary_json(csv_path=SUMMARY_CSV, out=APP_DATA / "results_summary.json") -> Path:
    """``results/summary.csv`` as a list of row dicts (numeric columns float, empty cells null)."""
    rows = read_summary_rows(csv_path)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_dumps(rows), encoding="utf8")
    return out


# ---------------------------------------------------------------------------------------------- field resampling
def window():
    """The app's viewing window and its grid: (xmin, xmax, ymin, ymax), the same constants as ``aeronet.field``."""
    from aeronet import field

    return field.XMIN, field.XMAX, field.YMIN, field.YMAX


def grid_points(width):
    """Cell centres of the app grid: row-major, top row first; ``height = round(width / 1.8)``."""
    xmin, xmax, ymin, ymax = window()
    height = max(1, round(width / 1.8))
    xs = xmin + (np.arange(width) + 0.5) / width * (xmax - xmin)
    ys = ymax - (np.arange(height) + 0.5) / height * (ymax - ymin)
    gx, gy = np.meshgrid(xs, ys)
    return gx, gy, height


def resample_field(X, Y, arrays: dict, width: int, nd: int = 3) -> dict:
    """Curvilinear NNfoil arrays (shape (N+1, M), duplicated seam row, app frame) -> the app's regular grid.

    A periodic triangulation of the mesh (the seam row dropped first) is interpolated linearly at the cell centres of
    the app window; cells outside the triangulation (the inside of the body) are ``None``. ``scipy.griddata`` is not
    used: its Delaunay triangles bridge across the body.
    """
    import matplotlib.tri as mtri

    X = np.asarray(X, dtype=float)[:-1]
    Y = np.asarray(Y, dtype=float)[:-1]
    n, m = X.shape
    idx = np.arange(n * m).reshape(n, m)
    nxt = np.roll(idx, -1, axis=0)  # periodic in xi
    a, b, c, d = idx[:, :-1].ravel(), nxt[:, :-1].ravel(), nxt[:, 1:].ravel(), idx[:, 1:].ravel()
    # Split every quad along a-c, or along b-d where a-c would make a flipped triangle (a non-convex cell next to a sharp
    # trailing edge: matplotlib then rejects the triangulation as invalid).
    P = np.c_[X.ravel(), Y.ravel()]

    def area(i, j, k):
        u, v = P[j] - P[i], P[k] - P[i]
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    s = np.sign(np.median(area(a, b, c)))
    ok = (np.sign(area(a, b, c)) == s) & (np.sign(area(a, c, d)) == s)
    t1 = np.where(ok[:, None], np.c_[a, b, c], np.c_[a, b, d])
    t2 = np.where(ok[:, None], np.c_[a, c, d], np.c_[b, c, d])
    tri = mtri.Triangulation(P[:, 0], P[:, 1], np.r_[t1, t2])
    finder = tri.get_trifinder()
    gx, gy, height = grid_points(width)
    xmin, xmax, ymin, ymax = window()
    out = {"width": int(width), "height": int(height), "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    for name, arr in arrays.items():
        z = np.asarray(arr, dtype=float)[:-1].ravel()
        f = mtri.LinearTriInterpolator(tri, z, trifinder=finder)(gx, gy)
        vals = np.ma.filled(np.ma.masked_invalid(f), np.nan)
        out[name] = _round_list(vals, nd)  # NaN (masked or not finite) -> None
    return out


# ---------------------------------------------------------------------------------------------- reference and panel
def shape_code(shape: str):
    m = re.fullmatch(r"NACA(\d{4})", shape or "")
    return m.group(1) if m else None


def reference_path(shape: str, alpha) -> Path | None:
    if shape == "NACA0012":
        return runs.FVM["NACA0012"]
    if shape == "Cylinder":
        return runs.FVM["Cylinder"]
    if shape == "NACA2412":
        return runs.fvm_2412(alpha)
    return None


def fvm_same_mesh_path(shape: str, mesh: str) -> Path | None:
    """The bundled FVM Cp file of the same shape on the same mesh (50x25 -> ``NACA0012_50_25_Cp.dat``), if any."""
    if shape not in ("NACA0012", "Cylinder") or mesh == "800x400":
        return None
    n, m = mesh.split("x")
    p = FVM_DIR / f"{shape}_{n}_{m}_Cp.dat"
    return p if p.exists() else None


def _circle_nodes(n=PANELS_HALF * 2):
    """``n`` nodes of the circle of radius 0.5 centred at (0.5, 0): counter-clockwise from (1, 0), last not repeated."""
    t = 2.0 * np.pi * np.arange(n) / n
    return 0.5 + 0.5 * np.cos(t), 0.5 * np.sin(t)


def panel_solution(shape: str, alpha, mach, lifting=True):
    """The panel method of the app on this shape: NACA codes with 140 panels (lifting, Prandtl-Glauert), the circle
    without lift (sources only, Prandtl-Glauert); ``None`` for a shape with neither."""
    from aeronet import panel

    code = shape_code(shape)
    if code is not None:
        sol = panel.solve(code, alpha, mach, PANELS_HALF)
        what = f"Hess-Smith panel method with Prandtl-Glauert, {sol.n} panels"
    elif shape == "Cylinder":
        x, y = _circle_nodes()
        sol = panel.solve_xy(x, y, alpha, mach, lifting=False)
        what = f"Hess-Smith sources-only panel method (no lift) with Prandtl-Glauert, {sol.n} panels"
    else:
        return None
    g = sol.geom
    return {"cl": round(float(sol.cl), 5), "name": what,
            "surface": {"x": _round_list(g.cx, 5), "y": _round_list(g.cy, 5), "cp": _round_list(sol.cp, 4)}}


# ---------------------------------------------------------------------------------------------- one case
def detect_collapse(rms, cl, cl_ref):
    """``(collapsed, reason)``. Collapsed = the trivial solution: Cp RMS above 1.0, or Cl about 0 where the reference
    has real lift (a cylinder's Cl is 0, so there Cl alone says nothing)."""
    if rms is not None and rms > COLLAPSE_RMS:
        return True, f"Cp RMS {rms:.2f} against the reference (above {COLLAPSE_RMS})"
    if cl_ref is not None and cl is not None and abs(cl_ref) > 0.1 and abs(cl) < 0.01 * abs(cl_ref):
        return True, f"Cl {cl:.1e} against a reference Cl of {cl_ref:.3f}"
    return False, None


def pretty_shape(shape: str) -> str:
    code = shape_code(shape)
    return f"NACA {code}" if code else shape


def make_label(shape, mesh, alpha, mach, variant=None, collapsed=False) -> str:
    parts = [pretty_shape(shape), mesh.replace("x", "×"), f"{float(alpha):g}°", f"Ma {float(mach):g}"]
    if variant:
        parts.append(variant)
    if collapsed:
        parts.append("collapsed")
    return " · ".join(parts)


def _cpu_name():
    try:
        return json.loads(E4_TIMING.read_text(encoding="utf8"))["hardware"]["cpu"]
    except (OSError, KeyError, ValueError):
        import platform

        return platform.processor() or "unknown"


def e4_infer_ms(run_id):
    """Full-field forward pass of the trained network in ms (8 threads, mean of 100 calls, idle CPU): measured in E4."""
    try:
        d = json.loads(E4_TIMING.read_text(encoding="utf8"))["pinn"][run_id]["threads"]["8"]["field"]
        return round(float(d["mean"]), 3)
    except (OSError, KeyError, ValueError):
        return None


def _withheld(notes):
    m = re.search(r"train_s withheld: ([^;]+)", notes or "")
    return m.group(1).strip() if m else None


def build_case(run_dir, label=None, ref_path=None, lifting=True, width=340, *, row=None, variant=None, infer_ms=None,
               cpu=None, keep_snapshots=None, fvm_same=None, source=None, with_field=True) -> dict:
    """The case dict of roadmap 5.7 for one run folder (see the module docstring for the extra keys).

    ``row`` is the run's summary row (supplies ``train_s``, which may be withheld); ``fvm_same`` is the path of the FVM
    file on the run's own mesh; ``keep_snapshots`` limits the number of snapshots kept (first, middle, last, ...).
    """
    run = runs.load_run(run_dir)
    cfg, log = run["config"], run["log"]
    shape, mesh = cfg["shape"], cfg["mesh"]
    alpha, mach = float(cfg["alpha_deg"]), float(cfg["mach"])
    mesh_file = Path(cfg["mesh_file"])
    mesh_file = mesh_file if mesh_file.is_absolute() else ROOT / mesh_file
    wx, wy = metrics.load_mesh_wall(mesh_file)
    fx, fy, fcp = runs.final_surface(run)

    ref = metrics.load_fvm(ref_path) if ref_path else None
    err = metrics.cp_error(fx, fy, fcp, ref) if ref is not None else {"rms": None, "max": None, "n": 0}
    cl = float(log["cl"][-1]) if log.get("cl") else None
    cd = float(log["cd"][-1]) if log.get("cd") else None
    cl_ref = round(metrics.fvm_forces(ref_path, alpha)[0], 5) if ref_path else None
    collapsed, reason = detect_collapse(err["rms"], cl, cl_ref)

    # ---- timing: the summary row decides (it may withhold the time: the laptop slept)
    notes = []
    time_s = list(log.get("time_s", []))
    if row is not None:
        train_s = row.get("train_s")
        wh = _withheld(row.get("notes"))
        if train_s is None:
            time_s = [None] * len(time_s)  # the same unusable wall time sits in the log
            notes.append(f"Training time withheld: {wh or 'not recorded in results/summary.csv'}.")
    else:
        train_s = round(float(time_s[-1]), 1) if time_s else None
    if cfg.get("init_from"):
        notes.append(f"Grid-sequenced: weights moved from {cfg['init_from']}; the time is this stage only and "
                     "excludes that parent run.")
    timing = {"train_s": train_s, "infer_ms": infer_ms, "threads": int(cfg.get("threads", 8)), "cpu": cpu or _cpu_name(),
              "note": " ".join(notes) or None,
              "infer_note": ("full-field forward pass over every mesh node, mean of 100 calls on an idle CPU "
                             "(results/e4_timing.json, experiment E4)") if infer_ms is not None else "not timed"}

    snaps = [{"round": int(r["round"]), "cp": _round_list(r["cp"], 4)} for r in run["snapshots"]["rounds"]]
    if keep_snapshots and len(snaps) > keep_snapshots:
        pick = sorted({round(i * (len(snaps) - 1) / (keep_snapshots - 1)) for i in range(keep_snapshots)})
        snaps = [snaps[i] for i in pick]

    log_out = {k: (_round_list(v, 6) if k in ("loss", "loss_f", "loss_b", "cl", "cd") else v)
               for k, v in log.items() if isinstance(v, list)}
    log_out["time_s"] = [None if t is None else round(float(t), 2) for t in time_s]
    log_out["failed_rounds"] = int(log.get("failed_rounds", 0))
    for k in ("cl_after_transfer",):
        if k in log:
            log_out[k] = log[k]

    panel = panel_solution(shape, alpha, mach, lifting)
    if panel is not None and ref is not None:
        e = metrics.cp_error(panel["surface"]["x"], panel["surface"]["y"], panel["surface"]["cp"], ref)
        panel["metrics"] = {"rms": round(e["rms"], 6), "max": round(e["max"], 6), "n": e["n"]}

    same = None
    if fvm_same:
        pts = metrics.load_fvm(fvm_same)
        e = metrics.cp_error(pts[:, 0], pts[:, 1], pts[:, 2], ref) if ref is not None else {"rms": None, "max": None}
        n, m = mesh.split("x")
        same = {"name": f"FVM {n}×{m}", "points": _points(pts), "cl": round(metrics.fvm_forces(fvm_same, alpha)[0], 5),
                "rms": None if e["rms"] is None else round(e["rms"], 6),
                "max": None if e["max"] is None else round(e["max"], 6)}

    case = {
        "id": cfg["run_id"],
        "label": label or make_label(shape, mesh, alpha, mach, variant, collapsed),
        "shape": shape, "mesh": mesh, "mach": mach, "alpha": alpha,
        "source": source or {"results": "measured", "condition": "quoted", "condition_note": CONDITION_SOURCE.get(shape)},
        "config": {"rounds": cfg.get("rounds"), "rounds_done": len(log.get("round", [])),
                   "failed_rounds": int(log.get("failed_rounds", 0)), "layers": cfg.get("layers"),
                   "lbfgs_max_iter": cfg.get("lbfgs_max_iter"), "positivity": bool(cfg.get("positivity", True)),
                   "seed": cfg.get("seed"), "init_from": cfg.get("init_from")},
        "x": _round_list(wx, 5), "y": _round_list(wy, 5),
        "surface": {"x": _round_list(fx, 5), "y": _round_list(fy, 5), "cp": _round_list(fcp, 4)},
        "reference": None if ref is None else {"name": "FVM 800×400", "points": _points(ref)},
        "fvm_same_mesh": same,
        "panel": panel,
        "metrics": {"rms": None if err["rms"] is None else round(err["rms"], 6),
                    "max": None if err["max"] is None else round(err["max"], 6), "n": err["n"],
                    "cl": None if cl is None else round(cl, 5), "cd": None if cd is None else round(cd, 5),
                    "cl_ref": cl_ref, "collapsed": collapsed, "collapsed_reason": reason},
        "timing": timing,
        "log": log_out,
        "snapshots": snaps,
        "field": None,
    }
    if with_field and run["field"] is not None:
        f = run["field"]
        case["field"] = resample_field(f["X"], f["Y"], {k: f[k] for k in ("cp", "u", "v")}, width)
    return case


def export_case(run_dir, label=None, ref_path=None, lifting=True, width=340, out_dir=APP_DATA / "pinn_cases", *,
                max_bytes=MAX_BYTES, **kw) -> Path:
    """Write ``<out_dir>/<run_id>.json``. Over ``max_bytes``: the field width steps down (340, 240, 180, 120, 80), then
    only the first, middle and last snapshots are kept."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    widths = [width] + [w for w in WIDTHS if w < width]
    text = None
    for keep in (None, 3):
        for w in widths:
            case = build_case(run_dir, label, ref_path, lifting, w, keep_snapshots=keep, **kw)
            text = _dumps(case)
            if len(text.encode("utf8")) <= max_bytes:
                break
        else:
            continue
        break
    else:
        raise ValueError(f"{case['id']}: still {len(text.encode('utf8'))} bytes with 3 snapshots and width {widths[-1]}")
    path = out_dir / f"{case['id']}.json"
    path.write_text(text, encoding="utf8")
    return path


# ---------------------------------------------------------------------------------------------- seed runs (showdown)
def _final_curve(run_dir):
    p = Path(run_dir) / "snapshots.json"
    if not p.exists():
        return None
    snap = json.loads(p.read_text(encoding="utf8"))
    if not snap.get("rounds"):
        return None
    x, y = runs.wall_faces(snap)
    return {"x": _round_list(x, 5), "y": _round_list(y, 5), "cp": _round_list(snap["rounds"][-1]["cp"], 5)}


def seed_groups(rows, runs_root=runs.RUNS, seed_check=SEED_CHECK_CSV) -> dict:
    """``{"<mesh>:<method>": {"source", "runs": [{seed, run_id, rms, cl, rounds, failed_rounds, collapsed, curve}]}}``."""
    groups = {}
    sc = Path(seed_check)
    if sc.exists():
        with sc.open(newline="", encoding="utf8") as f:
            sc_rows = list(csv.DictReader(f))
        for (mesh, method), (mode, meth) in SEED_CHECK_GROUPS.items():
            sel = sorted((r for r in sc_rows if r["mode"] == mode and r["method"] == meth), key=lambda r: int(r["seed"]))
            if not sel:
                continue
            groups[f"{mesh}:{method}"] = {"source": "results/seed_check.csv", "runs": [
                {"seed": int(r["seed"]), "run_id": r["run_id"], "rms": _num(r["cp_rms"]), "cl": _num(r["final_cl"]),
                 "rounds": int(r["rounds"] or 0), "failed_rounds": int(r["failed_rounds"] or 0),
                 "collapsed": r["collapsed"] == "yes"} for r in sel]}
    for (mesh, method), by_seed in E1_GROUPS.items():
        out = []
        for seed, run_id in by_seed.items():
            row = find_row(rows, ("E1", f"NACA0012_a4_s{seed}", method, mesh))
            if row is None:
                continue
            out.append({"seed": seed, "run_id": run_id, "rms": row["rms_cp"], "cl": row["cl"],
                        "rounds": _note_int(row["notes"], "rounds") or 0, "failed_rounds": None, "collapsed": False})
        if out:
            groups[f"{mesh}:{method}"] = {"source": "results/summary.csv (experiment E1)", "runs": out}
    for g in groups.values():
        for r in g["runs"]:
            r["curve"] = _final_curve(Path(runs_root) / r["run_id"]) if r["rms"] is not None else None
    return groups


def write_seed_runs(rows, out=APP_DATA / "seed_runs.json", runs_root=runs.RUNS, seed_check=SEED_CHECK_CSV) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_dumps({"groups": seed_groups(rows, runs_root, seed_check)}), encoding="utf8")
    return out


def e1_rounds(path=E1_ROUNDS_CSV) -> dict:
    """``{"source", "runs": [{run_id, method, positivity, seed, mesh, rounds, cp_rms, cl}]}``: the E1 trajectories, in file order.
    Only the columns the chart needs are kept; ``cp_rms`` is None for a round without one."""
    by_run: dict = {}
    with Path(path).open(newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            run = by_run.setdefault(r["run_id"], {
                "run_id": r["run_id"], "method": r["method"], "positivity": r["method"] == "pinn",
                "seed": int(re.search(r"_s(\d+)$", r["case"]).group(1)), "mesh": r["mesh"], "rounds": [], "cp_rms": [], "cl": []})
            run["rounds"].append(int(r["round"]))
            run["cp_rms"].append(_num(r["cp_rms"]))
            run["cl"].append(_num(r["cl"]))
    return {"source": "results/e1_rounds.csv", "runs": list(by_run.values())}


def write_e1_rounds(out=APP_DATA / "rounds_e1.json", path=E1_ROUNDS_CSV) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_dumps(e1_rounds(path)), encoding="utf8")
    return out


# ---------------------------------------------------------------------------------------------- everything
def export_all(out=APP_DATA, only=None, width=340, runs_root=runs.RUNS, summary_csv=SUMMARY_CSV, verbose=True) -> list[Path]:
    out = Path(out)
    rows = read_summary_rows(summary_csv)
    written = []
    cpu = _cpu_name()
    for spec in CASES:
        rid = spec["run"]
        if only and rid not in only:
            continue
        run_dir = Path(runs_root) / rid
        if not (run_dir / "snapshots.json").exists():
            print(f"skip {rid}: no run folder", file=sys.stderr)
            continue
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf8"))
        row = find_row(rows, spec["row"])
        same = fvm_same_mesh_path(cfg["shape"], cfg["mesh"])
        path = export_case(run_dir, None, reference_path(cfg["shape"], cfg["alpha_deg"]), cfg["shape"] != "Cylinder", width,
                           out / "pinn_cases", row=row, variant=spec["variant"], infer_ms=e4_infer_ms(rid), cpu=cpu,
                           fvm_same=same)
        case = json.loads(path.read_text(encoding="utf8"))
        if row is not None and row.get("rms_cp") is not None and case["metrics"]["rms"] is not None:
            if abs(row["rms_cp"] - case["metrics"]["rms"]) > 1e-4:
                print(f"WARNING {rid}: exported Cp RMS {case['metrics']['rms']} differs from summary.csv {row['rms_cp']}",
                      file=sys.stderr)
        if verbose:
            m, f = case["metrics"], case["field"]
            print(f"{rid}: {path.stat().st_size / 1e6:.2f} MB, field {f['width']}x{f['height']}, rms {m['rms']}, "
                  f"cl {m['cl']}{' COLLAPSED' if m['collapsed'] else ''}")
        written.append(path)
    if not only:
        written.append(write_summary_json(summary_csv, out / "results_summary.json"))
        written.append(write_seed_runs(rows, out / "seed_runs.json", runs_root))
        if E1_ROUNDS_CSV.exists():
            written.append(write_e1_rounds(out / "rounds_e1.json"))
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export finished runs to app/backend/data (case JSON, summary, seed runs).")
    ap.add_argument("--out", default=str(APP_DATA), help="data folder (default app/backend/data)")
    ap.add_argument("--only", nargs="*", help="export only these run ids (no summary / seed files)")
    ap.add_argument("--width", type=int, default=340, help="field grid width (default 340; reduced if the file is over 3 MB)")
    args = ap.parse_args(argv)
    written = export_all(Path(args.out), only=set(args.only) if args.only else None, width=args.width)
    total = sum(p.stat().st_size for p in written)
    print(f"{len(written)} files, {total / 1e6:.2f} MB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
