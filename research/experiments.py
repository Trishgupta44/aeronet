"""Experiment drivers E1-E3 (Plan C): print the training commands, then turn finished run folders into summary rows.

    research/.venv/Scripts/python research/experiments.py commands e1|e2|e3   # one shell line per run, nothing is run
    research/.venv/Scripts/python research/experiments.py collect  e1|e2|e3   # finished runs -> results/summary.csv

``commands`` only prints (it skips runs that already have the requested number of attempted rounds, and prints a
``--resume`` line for a partial run). Run the printed lines one after another (parallel runs spoil the timing numbers),
then run ``collect``. Every figure is made by ``make_figures.py``, never here.

E1  positivity on/off, NACA 0012 50x25, Ma 0.4, 4 deg, seeds 0-2, 20 rounds (seed 0 = the Plan A runs).
E2  mesh size vs Cp error against FVM 800x400: FVM, panel and PINN (the 200x100 run is chained from 50x25 -> 100x50).
E3  NACA 2412 at Ma 0.3 (quoted from the paper): FVM / panel / thin-airfoil polar points and PINN runs at 4 angles.
E4  speed table: needs ``results/e4_timing.json`` from ``research/time_inference.py --e4`` (run alone, on an idle CPU);
    ``collect e4`` writes the summary rows and ``results/e4_speed.md`` (measured and quoted times kept apart).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT / "app" / "backend"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import metrics  # noqa: E402
import ogrid  # noqa: E402
import runs  # noqa: E402
import validate  # noqa: E402

REF0012 = runs.FVM["NACA0012"]
MESH_DIR = ROOT / "NNfoil" / "mesh"
RESULTS_MESH_DIR = ROOT / "results" / "meshes"
FVM_DIR = ROOT / "NNfoil" / "FVM_Cp"
MACH, ALPHA = 0.4, 4
E1_ROUNDS = 20
E3_ROUNDS = 10
BUDGET_S = 3 * 3600  # roadmap rule 4
FVM_MESHES = [("50x25", "50_25"), ("100x50", "100_50"), ("200x100", "200_100"), ("400x200", "400_200")]

E3_MACH = 0.3
E3_MACH_NOTE = ("Ma 0.3 quoted from arXiv 2401.08705 sec 3.4 (not stored in the Cp files; the panel-method fit "
                "of the files prefers about 0.25)")
E3_ALPHAS = list(range(-10, 11, 2))
E3_PINN_ALPHAS = [-4, 0, 4, 8]
E3_MESH_NOTE = ("mesh results/meshes/NACA2412_100_50.x: cosine wall from the bundled 200x100 wall, O-grid with the "
                "upstream cross coefficient 0.5 so that it matches how the bundled NACA 0012 meshes were made")


def _r(v, nd=6):
    return None if v is None else round(float(v), nd)


def _rel(p) -> str:
    """Repo-relative posix path when inside the repo, absolute posix path otherwise."""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def shell_line(cmd, out, append=False) -> str:
    """``mkdir -p <out> && <command> > <out>/train.log 2>&1`` with repo-relative paths (run it from the repo root in bash)."""
    args = []
    for a in cmd:
        a = str(a)
        args.append(_rel(a) if (Path(a).is_absolute() and Path(a).exists()) else a)
    return f"mkdir -p {out} && {shlex.join(args)} {'>>' if append else '>'} {out}/train.log 2>&1"


def rounds_for_budget(t_round, budget_s=BUDGET_S, cap=20) -> int:
    """Roadmap rule 4: ``min(cap, floor(budget / seconds per round))``."""
    if t_round <= 0:
        return cap
    return int(min(cap, math.floor(budget_s / t_round)))


# ---------------------------------------------------------------- specs and commands
def _spec(runs_root, *, shape, mesh_tag, mesh, alpha, mach, positivity, seed=0, rounds, mesh_file, grid_seq=False,
          init_from=None, dir_suffix=""):
    name = runs.run_id(shape, mesh, alpha, mach, positivity, seed, grid_seq=grid_seq) + dir_suffix
    return {"name": name, "dir": Path(runs_root) / name, "shape": shape, "mesh": mesh, "mach": mach, "alpha": alpha,
            "positivity": positivity, "seed": seed, "rounds": rounds, "mesh_file": mesh_file, "init_from": init_from}


def e1_specs(runs_root=runs.RUNS):
    """Six runs: seed 0 reuses the Plan A runs; seeds 1 and 2 are NEW folders (``..._r20``).

    The short Plan A seed-check runs ``naca0012_50x25_a4_m04[_nopos]_s1..s3`` (5 rounds) are the parents of the
    grid-sequenced seed-check runs, so they are left untouched.
    """
    out = []
    for positivity in (True, False):
        for seed in (0, 1, 2):
            out.append(_spec(runs_root, shape="NACA0012", mesh_tag="50_25", mesh="50x25", alpha=ALPHA, mach=MACH,
                             positivity=positivity, seed=seed, rounds=E1_ROUNDS,
                             mesh_file="NNfoil/mesh/NACA0012_50_25.x", dir_suffix="_r20" if seed else ""))
    return out


def e2_200_spec(runs_root, mode="nopos", rounds=1):
    pos = mode == "pos"
    parent = Path(runs_root) / runs.run_id("NACA0012", "100x50", ALPHA, MACH, pos, grid_seq=True)
    return _spec(runs_root, shape="NACA0012", mesh_tag="200_100", mesh="200x100", alpha=ALPHA, mach=MACH, positivity=pos,
                 rounds=rounds, mesh_file="NNfoil/mesh/NACA0012_200_100.x", grid_seq=True, init_from=_rel(parent)), parent


def e3_specs(runs_root=runs.RUNS, mesh_dir=None, rounds=E3_ROUNDS):
    mesh_dir = Path(mesh_dir) if mesh_dir is not None else RESULTS_MESH_DIR
    return [_spec(runs_root, shape="NACA2412", mesh_tag="100_50", mesh="100x50", alpha=a, mach=E3_MACH, positivity=True,
                  rounds=rounds, mesh_file=_rel(mesh_dir / "NACA2412_100_50.x")) for a in E3_PINN_ALPHAS]


def _command_for(spec, threads, notes):
    """The shell line for one run, or None when it is finished (or cannot continue)."""
    d = spec["dir"]
    cmd_kw = dict(positivity=spec["positivity"], seed=spec["seed"], threads=threads)
    out = _rel(d)
    if (d / "log.json").exists():
        run = runs.load_run(d)
        att = runs.attempts(run)
        if att >= spec["rounds"]:
            return None
        if not (d / "model.pt").exists():
            notes.append(f"{spec['name']}: {att} rounds attempted, none finished; delete the folder to start again")
            return None
        cmd = runs.train_cmd(spec["mesh_file"], spec["shape"], spec["mach"], spec["alpha"], spec["rounds"] - att, out,
                             resume=True, **cmd_kw)
        return shell_line(cmd, out, append=True)
    cmd = runs.train_cmd(spec["mesh_file"], spec["shape"], spec["mach"], spec["alpha"], spec["rounds"], out,
                         init_from=spec["init_from"], **cmd_kw)
    return shell_line(cmd, out)


def plan_commands(exp, runs_root=runs.RUNS, mode="nopos", mesh_dir=None, threads=8, rounds=None):
    """``(lines, notes)``: the shell lines still to run for experiment ``exp`` and messages for the person running them."""
    runs_root = Path(runs_root)
    notes: list[str] = []
    if exp == "e1":
        specs = e1_specs(runs_root)
        if rounds:
            specs = [{**s, "rounds": rounds} for s in specs]
    elif exp == "e2":
        spec, parent = e2_200_spec(runs_root, mode)
        if not ((parent / "model.pt").exists() and (parent / "config.json").exists()):
            return [], [f"{_rel(parent)} is missing: the 200x100 run is chained from it (50x25 -> 100x50 -> 200x100)"]
        d = spec["dir"]
        if (d / "log.json").exists():
            run = runs.load_run(d)
            good = len(run["log"]["round"])
            if good == 0:
                return [], [f"{spec['name']}: no finished round yet"]
            t_round = run["log"]["time_s"][-1] / good
            total = rounds_for_budget(t_round)
            if total < 3:
                return [], [f"{spec['name']}: {t_round:.0f} s per round: fewer than 3 rounds fit in 3 h, skip 200x100 "
                            f"(say so in the M3 doc)"]
            spec = {**spec, "rounds": total}
            notes.append(f"{spec['name']}: {t_round:.0f} s per round -> {total} rounds in total (3 h rule)")
        else:
            notes.append(f"{spec['name']}: probe of 1 round; when it is done run `commands e2` again for the rest "
                         f"(rounds = min(20, floor(3 h / time of round 1)), at least 3, else skip)")
        specs = [spec]
    elif exp == "e3":
        paths = ensure_e3_meshes(mesh_dir)
        specs = e3_specs(runs_root, Path(paths["upstream"]).parent, rounds or E3_ROUNDS)
    else:
        raise SystemExit(f"unknown experiment {exp!r}")
    lines = []
    for s in specs:
        line = _command_for(s, threads, notes)
        if line:
            lines.append(line)
    return lines, notes


# ---------------------------------------------------------------- meshes for E3
def ensure_e3_meshes(out_dir=None, force=False) -> dict:
    """NACA 2412 100x50 meshes: ``upstream`` (cross coefficient 0.5, used by E3) and ``winslow2`` (2.0, for comparison)."""
    out_dir = Path(out_dir) if out_dir is not None else RESULTS_MESH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"upstream": out_dir / "NACA2412_100_50.x", "winslow2": out_dir / "NACA2412_100_50_winslow2.x"}
    if force or not all(p.exists() for p in paths.values()):
        xx, yy = metrics.load_mesh(MESH_DIR / "NACA2412_200_100.x")
        xw, yw = ogrid.cosine_wall(xx[:-1, 0], yy[:-1, 0], 100)
        for key, cc in (("upstream", 0.5), ("winslow2", 2.0)):
            X, Y, iters, change = ogrid.ogrid(xw, yw, 50, cross_coeff=cc)
            if not (ogrid.is_valid(X, Y) and change < 1e-5):
                raise SystemExit(f"NACA2412 100x50 O-grid (cross_coeff {cc}) is not valid (change {change:.2g})")
            ogrid.write_x(paths[key], X, Y)
    return paths


# ---------------------------------------------------------------- rows
def trajectory_notes(rows) -> str:
    """Early / best / late Cp RMS of a run's rounds. The best round is chosen with the FVM error, so it is labelled post-hoc."""
    rms = [(r["round"], r["cp_rms"]) for r in rows if r["cp_rms"] is not None and np.isfinite(r["cp_rms"])]
    if not rms:
        return "rms_round3=n/a; best_round_rms=n/a"
    r3 = next((v for k, v in rms if k == 3), None)
    kb, vb = min(rms, key=lambda t: t[1])
    last = [v for _, v in rms[-5:]]
    return (f"rms_round3={'n/a' if r3 is None else format(r3, '.4g')}; best_round_rms={vb:.4g}@round{kb} "
            f"(post-hoc: picked using the FVM error); mean_rms_last5={float(np.mean(last)):.4g}")


def _nan_rounds(run, rows) -> int:
    log = run["log"]
    bad = 0
    cps = {int(r["round"]): np.asarray(r["cp"], dtype=float) for r in run["snapshots"]["rounds"]}
    for i, r in enumerate(log["round"]):
        vals = [log["loss"][i], log["cl"][i], log["cd"][i]]
        if not np.isfinite(vals).all() or not np.isfinite(cps[int(r)]).all():
            bad += 1
    return bad


def pinn_row(run_dir, ref_path, experiment, case=None, extra="", with_trajectory=False):
    """``(summary row, per-round rows)`` of one run: final-round Cp error against the FVM file, Cl, Cd, training time."""
    run = runs.load_run(run_dir)
    cfg, log, snap = run["config"], run["log"], run["snapshots"]
    if not log["round"]:
        raise SystemExit(f"{run_dir}: no finished round in log.json")
    alpha, mach = float(cfg["alpha_deg"]), float(cfg["mach"])
    ref = metrics.load_fvm(ref_path)
    traj = validate.rounds_table(cfg, log, snap, ref)
    last = traj[-1]
    cl_ref, _ = metrics.fvm_forces(ref_path, alpha)
    att = runs.attempts(run)
    restarts = runs.count_restarts(run_dir)
    notes = [f"rounds={len(log['round'])}/{att}", f"seed={cfg.get('seed', 0)}", f"restarts={restarts}",
             f"nan_rounds={_nan_rounds(run, traj)}", f"final round; ref: {Path(ref_path).name}"]
    if att < int(cfg.get("rounds", att)):
        notes.insert(0, f"INCOMPLETE ({att} of {cfg['rounds']} rounds attempted)")
    if cfg.get("init_from"):
        notes.append(f"init_from={cfg['init_from']}")
        if log.get("cl_after_transfer") is not None:
            notes.append(f"cl_after_transfer={float(log['cl_after_transfer']):.4f}")
    notes.append(trajectory_notes(traj))
    if extra:
        notes.append(extra)
    row = {"experiment": experiment, "case": case or validate.case_name(cfg), "method": validate.method_name(cfg),
           "mesh": cfg["mesh"], "mach": validate._num(mach), "alpha_deg": validate._num(alpha),
           "rms_cp": _r(last["cp_rms"]), "max_cp": _r(last["cp_max"]), "cl": _r(log["cl"][-1]), "cl_ref": _r(cl_ref),
           "cd": _r(log["cd"][-1]), "train_s": round(log["time_s"][-1], 1), "infer_ms": None, "source": "measured",
           "notes": "; ".join(notes)}
    tag = {"run_id": cfg.get("run_id", Path(run_dir).name), "method": row["method"], "case": row["case"],
           "mesh": cfg["mesh"]}
    return row, [{**tag, **t} for t in traj]


TRAJ_COLUMNS = ["run_id", "method", "case", "mesh", "round", "cp_rms", "cp_max", "cl", "cd", "loss", "time_s"]


def _write_traj(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=TRAJ_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items() if k in TRAJ_COLUMNS})


def _fvm_row(experiment, case, mesh, mach, alpha, path, ref_path, notes, ref=None):
    ref = metrics.load_fvm(ref_path) if ref is None else ref
    pts = metrics.load_fvm(path)
    err = metrics.cp_error(pts[:, 0], pts[:, 1], pts[:, 2], ref)
    cl, cd = metrics.fvm_forces(path, alpha)
    cl_ref, _ = metrics.fvm_forces(ref_path, alpha)
    return {"experiment": experiment, "case": case, "method": "fvm", "mesh": mesh, "mach": mach, "alpha_deg": alpha,
            "rms_cp": _r(err["rms"]), "max_cp": _r(err["max"]), "cl": _r(cl), "cl_ref": _r(cl_ref), "cd": _r(cd),
            "train_s": None, "infer_ms": None, "source": "measured", "notes": notes}


def fvm_mesh_rows():
    """E2 FVM rows: each coarse FVM file's own Cp points are the 'prediction', the 800x400 file is the reference."""
    ref = metrics.load_fvm(REF0012)
    return [_fvm_row("E2", "NACA0012_a4", label, MACH, ALPHA, FVM_DIR / f"NACA0012_{tag}_Cp.dat", REF0012,
                     "reference: FVM 800x400; coarse FVM points used as the prediction", ref)
            for label, tag in FVM_MESHES]


def panel_row():
    """E2 panel row: the app's panel solver (Prandtl-Glauert) at the panel centres against FVM 800x400."""
    from aeronet import panel

    sol = panel.solve("0012", ALPHA, MACH, 70)
    ref = metrics.load_fvm(REF0012)
    err = metrics.cp_error(sol.geom.cx, sol.geom.cy, sol.cp, ref)
    cl_ref, _ = metrics.fvm_forces(REF0012, ALPHA)
    return {"experiment": "E2", "case": "NACA0012_a4", "method": "panel", "mesh": f"{sol.n} panels", "mach": MACH,
            "alpha_deg": ALPHA, "rms_cp": _r(err["rms"]), "max_cp": _r(err["max"]), "cl": _r(sol.cl),
            "cl_ref": _r(cl_ref), "cd": _r(sol.cd), "train_s": None, "infer_ms": None, "source": "measured",
            "notes": "reference: FVM 800x400; Hess-Smith panel method with Prandtl-Glauert"}


def e3_fvm_rows():
    return [{"experiment": "E3", "case": f"NACA2412_a{a}", "method": "fvm", "mesh": "800x400", "mach": E3_MACH,
             "alpha_deg": a, "rms_cp": None, "max_cp": None, "cl": _r((cl := metrics.fvm_forces(runs.fvm_2412(a), a))[0]),
             "cl_ref": _r(cl[0]), "cd": _r(cl[1]), "train_s": None, "infer_ms": None, "source": "measured",
             "notes": f"the reference itself (Cl from the ordered Cp file); {E3_MACH_NOTE}"} for a in E3_ALPHAS]


def e3_panel_rows():
    from aeronet import panel

    rows = []
    for a in E3_ALPHAS:
        path = runs.fvm_2412(a)
        ref = metrics.load_fvm(path)
        sol = panel.solve("2412", a, E3_MACH, 70)
        err = metrics.cp_error(sol.geom.cx, sol.geom.cy, sol.cp, ref)
        cl_f, _ = metrics.fvm_forces(path, a)
        rows.append({"experiment": "E3", "case": f"NACA2412_a{a}", "method": "panel", "mesh": f"{sol.n} panels",
                     "mach": E3_MACH, "alpha_deg": a, "rms_cp": _r(err["rms"]), "max_cp": _r(err["max"]),
                     "cl": _r(sol.cl), "cl_ref": _r(cl_f), "cd": _r(sol.cd), "train_s": None, "infer_ms": None,
                     "source": "measured", "notes": f"reference: FVM 800x400; {E3_MACH_NOTE}"})
    return rows


def e3_thin_rows():
    from aeronet import compare, geometry

    af = geometry.naca4("2412")
    return [{"experiment": "E3", "case": f"NACA2412_a{a}", "method": "thin_airfoil", "mesh": None, "mach": E3_MACH,
             "alpha_deg": a, "rms_cp": None, "max_cp": None, "cl": _r(compare.thin_airfoil_cl(af, a, E3_MACH)),
             "cl_ref": _r(metrics.fvm_forces(runs.fvm_2412(a), a)[0]), "cd": None, "train_s": None, "infer_ms": None,
             "source": "measured", "notes": f"Prandtl-Glauert corrected; reference: FVM 800x400; {E3_MACH_NOTE}"}
            for a in E3_ALPHAS]


# ---------------------------------------------------------------- collect
def _existing(spec):
    return (spec["dir"] / "log.json").exists() and (spec["dir"] / "snapshots.json").exists()


def _print_rows(rows):
    for r in rows:
        print(f"{r['experiment']:3s} {r['case']:16s} {r['method']:14s} {str(r['mesh']):11s} "
              f"rms {str(r['rms_cp']):9s} cl {str(r['cl']):9s} (ref {str(r['cl_ref'])})")


def collect(exp, runs_root=runs.RUNS, summary="results/summary.csv", rounds_dir=None):
    """Compute the rows of experiment ``exp`` from finished run folders and append them to ``summary``."""
    runs_root = Path(runs_root)
    rounds_dir = Path(rounds_dir) if rounds_dir is not None else ROOT / "results"
    rows, traj = [], []

    def take(spec, ref_path, experiment, case=None, extra=""):
        if not _existing(spec):
            print(f"skip {spec['name']}: no finished round", file=sys.stderr)
            return None
        row, t = pinn_row(spec["dir"], ref_path, experiment, case, extra)
        rows.append(row)
        traj.extend(t)
        return row

    if exp == "e1":
        for s in e1_specs(runs_root):
            take(s, REF0012, "E1", case=f"NACA0012_a4_s{s['seed']}",
                 extra="E1: positivity " + ("on" if s["positivity"] else "off"))
    elif exp == "e2":
        fvm = fvm_mesh_rows()
        fvm_rms = {r["mesh"]: r["rms_cp"] for r in fvm}
        names = [
            ("50x25", "NACA0012_50_25", runs.run_id("NACA0012", "50x25", ALPHA, MACH, True),
             "50x25 seed 0, positivity on (the E1 seed-0 run)"),
            ("50x25", "NACA0012_50_25", runs.run_id("NACA0012", "50x25", ALPHA, MACH, False),
             "50x25 seed 0, positivity off (the E1 seed-0 run)"),
            ("100x50", "", runs.run_id("NACA0012", "100x50", ALPHA, MACH, True),
             "100x50 from scratch (no grid sequencing), seed 0, positivity on"),
            ("100x50", "", runs.run_id("NACA0012", "100x50", ALPHA, MACH, True, grid_seq=True),
             "100x50 grid-sequenced from the 50x25 positivity-on run, seed 0"),
            ("100x50", "", runs.run_id("NACA0012", "100x50", ALPHA, MACH, False, grid_seq=True),
             "100x50 grid-sequenced from the 50x25 positivity-off run, seed 0"),
            ("200x100", "", runs.run_id("NACA0012", "200x100", ALPHA, MACH, False, grid_seq=True),
             "200x100 chained 50x25 -> 100x50 -> 200x100, positivity off, seed 0"),
            ("200x100", "", runs.run_id("NACA0012", "200x100", ALPHA, MACH, True, grid_seq=True),
             "200x100 chained 50x25 -> 100x50 -> 200x100, positivity on, seed 0"),
        ]
        for mesh, _, name, what in names:
            spec = {"name": name, "dir": runs_root / name}
            if not _existing(spec):
                if mesh != "200x100":
                    print(f"skip {name}: no finished round", file=sys.stderr)
                continue
            row, t = pinn_row(spec["dir"], REF0012, "E2", extra=what)
            better = row["rms_cp"] < fvm_rms[mesh]
            row["notes"] += (f"; PINN RMS < FVM RMS at {mesh}: {'yes' if better else 'no'} "
                             f"({row['rms_cp']:.4f} vs {fvm_rms[mesh]:.4f}, final round, same 800x400 reference)")
            rows.append(row)
            traj.extend(t)
        rows = fvm + [panel_row()] + rows
    elif exp == "e3":
        rows += e3_fvm_rows() + e3_panel_rows() + e3_thin_rows()
        for s in e3_specs(runs_root):
            take(s, runs.fvm_2412(s["alpha"]), "E3", case=f"NACA2412_a{s['alpha']}",
                 extra=f"{E3_MESH_NOTE}; {E3_MACH_NOTE}; seed 0, positivity on, cold 100x50 start; not an E1 verdict"
                       + ("; THIS RUN COLLAPSED to the trivial solution (Cl about 0): see the two retries (rows with "
                          "_s1 and pinn_nopos). In the NACA0012 seed check a cold positivity-on start did not "
                          "collapse in 4 of 4 seeds, so a collapse is possible with positivity on too"
                          if s["alpha"] == 4 else ""))
        # retries of the collapsed 4-degree case (run by hand after the chain): another seed, and positivity off
        for name, case, what in [
            ("naca2412_100x50_a4_m03_s1", "NACA2412_a4_s1", "retry: seed 1, positivity on"),
            ("naca2412_100x50_a4_m03_nopos", "NACA2412_a4", "retry: seed 0, positivity off"),
        ]:
            take({"name": name, "dir": runs_root / name}, runs.fvm_2412(4), "E3", case=case,
                 extra=f"{E3_MESH_NOTE}; {E3_MACH_NOTE}; {what} (after the seed-0 positivity-on run collapsed)")
    else:
        raise SystemExit(f"unknown experiment {exp!r}")
    for r in rows:
        metrics.append_summary(r, summary)
    _write_traj(rounds_dir / f"{exp}_rounds.csv", traj)
    return rows


# ---------------------------------------------------------------- E4: speed table (measured vs quoted)
E4_CASE = "NACA0012_a4"
# the paper's Table 1 FVM wall times for NACA 0012, 4 deg, Ma 0.4 (arXiv 2401.08705), in seconds; and the NNfoil times
FVM_QUOTED_S = {"50x25": 9, "100x50": 24, "200x100": 53, "400x200": 285}
NNFOIL_PAPER_S = {"50x25": 384, "100x50": 266, "200x100": 262, "400x200": 271}
E4_PAPER_CAVEAT = ("quoted from Table 1 of arXiv 2401.08705 (NACA 0012, 4 deg, Ma 0.4), measured by the paper's authors "
                   "on another machine; the paper says the table is \"not meant as a rigorous comparison\"")
# train_s is taken from the run logs of runs measured on an idle CPU in Plan A; the 200x100 run is withheld (the laptop slept)
E4_PINN_RUNS = [
    {"run": "naca0012_50x25_a4_m04", "method": "pinn", "mesh": "50x25", "train_ok": True},
    {"run": "naca0012_50x25_a4_m04_nopos", "method": "pinn_nopos", "mesh": "50x25", "train_ok": True},
    {"run": "naca0012_100x50_a4_m04", "method": "pinn", "mesh": "100x50", "train_ok": True},
    {"run": "naca0012_100x50_a4_m04_gs", "method": "pinn_gs", "mesh": "100x50", "train_ok": True},
    {"run": "naca0012_100x50_a4_m04_gs_nopos", "method": "pinn_gs_nopos", "mesh": "100x50", "train_ok": True},
    {"run": "naca0012_200x100_a4_m04_gs_nopos", "method": "pinn_gs_nopos", "mesh": "200x100", "train_ok": False},
]


def _ms(st, nd=3) -> str:
    return f"{st['mean']:.{nd}f} +/- {st['sd']:.{nd}f} ms (n={st['n']})"


def _pinn_e4_row(spec, t, runs_root):
    th = t["threads"]
    f8, f1 = th["8"], th["1"]
    notes = [f"infer_ms = full-field forward pass model(XE) over all {t['n_points']} mesh nodes (u, v, p, rho everywhere, "
             f"not just Cl), torch.no_grad, 8 threads, mean over {f8['field']['n']} calls after warm-up: "
             f"{_ms(f8['field'])}",
             f"1 thread: {_ms(f1['field'])}",
             f"wall-only evaluate_wall (Cp, Cl, Cd): 8 threads {_ms(f8['wall'])}; 1 thread {_ms(f1['wall'])}"]
    train = None
    run_dir = Path(runs_root) / spec["run"]
    if spec["train_ok"] and (run_dir / "log.json").exists():
        run = runs.load_run(run_dir)
        train = round(run["log"]["time_s"][-1], 1)
        notes.append(f"train_s = log.time_s[-1] of {spec['run']} (8 threads, idle CPU, measured in Plan A)")
        init = run["config"].get("init_from")
        if init:
            parent = Path(runs_root) / Path(str(init)).name
            ptxt = ""
            if (parent / "log.json").exists():
                ptxt = f" ({runs.load_run(parent)['log']['time_s'][-1]:.1f} s)"
            notes.append(f"grid-sequenced: train_s is this stage only and excludes the parent run {init}{ptxt}")
    elif not spec["train_ok"]:
        notes.append("train_s withheld: the laptop slept during this run, so its wall time is not valid")
    return {"experiment": "E4", "case": E4_CASE, "method": spec["method"], "mesh": spec["mesh"], "mach": MACH,
            "alpha_deg": ALPHA, "rms_cp": None, "max_cp": None, "cl": None, "cl_ref": None, "cd": None,
            "train_s": train, "infer_ms": _r(f8["field"]["mean"], 4), "source": "measured", "notes": "; ".join(notes)}


def e4_rows(timing, runs_root=runs.RUNS) -> list[dict]:
    """Summary rows (experiment E4) from a timing dict: PINN, panel and NeuralFoil (measured), FVM (quoted)."""
    rows = []
    for spec in E4_PINN_RUNS:
        t = timing["pinn"].get(spec["run"])
        if t is not None:
            rows.append(_pinn_e4_row(spec, t, runs_root))
    pl = timing.get("panel")
    if pl:
        rows.append({"experiment": "E4", "case": E4_CASE, "method": "panel", "mesh": f"{pl['n_panels']} panels",
                     "mach": MACH, "alpha_deg": ALPHA, "rms_cp": None, "max_cp": None, "cl": None, "cl_ref": None,
                     "cd": None, "train_s": None, "infer_ms": _r(pl["solve"]["mean"], 4), "source": "measured",
                     "notes": (f"infer_ms = panel.solve('0012', 4, 0.4, 70) only, mean over {pl['solve']['n']} calls "
                               f"after warm-up: {_ms(pl['solve'])}, median {pl['solve']['median']:.3f} ms, max {pl['solve']['max']:.1f} ms (includes building the geometry; no training; "
                               f"BLAS threads: {pl.get('blas_threads', 'default')}); solve plus field_grid("
                               f"{pl['field_width']}) (Cp on the whole window): {_ms(pl['solve_field170'], 2)}")})
    nf = timing.get("neuralfoil")
    if nf is not None:
        base = {"experiment": "E4", "case": E4_CASE, "method": "neuralfoil", "mesh": "large", "mach": None,
                "alpha_deg": ALPHA, "rms_cp": None, "max_cp": None, "cl_ref": None, "cd": None, "train_s": None,
                "source": "measured"}
        if "error" in nf:
            rows.append({**base, "cl": None, "infer_ms": None,
                         "notes": f"not measured: neuralfoil unavailable ({nf['error']})"})
        else:
            rows.append({**base, "cl": _r(nf["cl"], 4), "cd": _r(nf["cd"], 5), "infer_ms": _r(nf["time"]["mean"], 4),
                         "notes": (f"infer_ms = neuralfoil {nf['version']} get_aero_from_coordinates, model_size "
                                   f"{nf['model_size']}, NACA 0012 at 4 deg, Re 6e6, mean over {nf['time']['n']} calls "
                                   f"after warm-up: {_ms(nf['time'])} (the Kulfan fit is inside the time); cl is "
                                   f"viscous, Re 6e6, incompressible: not comparable with the inviscid Ma 0.4 "
                                   f"cl_ref 0.5193 of the FVM, so cl_ref is left empty; analysis_confidence "
                                   f"{nf['confidence']:.3f}")})
    for mesh, sec in FVM_QUOTED_S.items():
        rows.append({"experiment": "E4", "case": E4_CASE, "method": "fvm", "mesh": mesh, "mach": MACH,
                     "alpha_deg": ALPHA, "rms_cp": None, "max_cp": None, "cl": None, "cl_ref": None, "cd": None,
                     "train_s": None, "infer_ms": sec * 1000, "source": "quoted",
                     "notes": f"infer_ms = FVM wall time of one solve, {sec} s, {E4_PAPER_CAVEAT}"})
    return rows


def _md_e4(timing, rows) -> str:
    hw = timing["hardware"]
    load = timing.get("cpu_load_before", {})
    peak = max((max(v) for v in load.values() if v), default=None)
    L = ["# E4: speed table (measured vs quoted)", "",
         "Condition: NACA 0012, 4 degrees, Ma 0.4 (the FVM files' condition). Generated by `research/experiments.py "
         "collect e4` from `results/e4_timing.json` (made by `research/time_inference.py --e4`).", "",
         "## Hardware and method (measured here)", "",
         f"- CPU: {hw['cpu']} ({hw['logical_threads']} logical threads); {hw['platform']}",
         f"- Python {hw['python']}, torch {hw['torch']}, numpy {hw['numpy']}, neuralfoil {hw.get('neuralfoil') or 'n/a'}",
         f"- Every timing: {timing['warmup']} untimed warm-up calls, then the mean and sample standard deviation "
         f"(n - 1) of {timing['calls']} calls timed one by one with `time.perf_counter`; one timing block at a time, "
         "nothing else running. PINN calls run under `torch.no_grad()`.",
         "- Total CPU load seen before each timing block (percent): "
         + ("highest sample " + str(peak) + "; " if peak is not None else "")
         + "; ".join(f"{k.replace('naca0012_', '').replace('_a4_m04', '')}: {v}" for k, v in load.items()), "",
         "## PINN inference (measured)", "",
         "Full field = one forward pass `model(XE)` over every mesh node (u, v, p, rho at all nodes). Wall only = "
         "`evaluate_wall()` (Cp on the wall plus Cl and Cd).", "",
         "| Trained run | Mesh | Nodes | Threads | Full field, ms (mean ± sd) | Wall only, ms (mean ± sd) | n |",
         "|---|---|---|---|---|---|---|"]
    for spec in E4_PINN_RUNS:
        t = timing["pinn"].get(spec["run"])
        if not t:
            continue
        for th in ("1", "8"):
            f = t["threads"][th]
            L.append(f"| `{spec['run']}` | {spec['mesh']} | {t['n_points']} | {th} | "
                     f"{f['field']['mean']:.3f} ± {f['field']['sd']:.3f} | {f['wall']['mean']:.3f} ± "
                     f"{f['wall']['sd']:.3f} | {f['field']['n']} |")
    L += ["", "## PINN training time (measured in Plan A: 8 threads, idle CPU)", "",
          "| Trained run | Mesh | train_s | Note |", "|---|---|---|---|"]
    for spec in E4_PINN_RUNS:
        r = next((r for r in rows if r["method"] == spec["method"] and r["mesh"] == spec["mesh"]), None)
        if r is None:
            continue
        note = ("withheld: the laptop slept during this run" if r["train_s"] is None
                else "this stage only, excludes the parent run" if "gs" in r["method"] else "")
        L.append(f"| `{spec['run']}` | {r['mesh']} | {'' if r['train_s'] is None else format(r['train_s'], '.1f')} | {note} |")
    pl, nf = timing.get("panel"), timing.get("neuralfoil")
    L += ["", "## Other methods", "", "| Method | Setting | What is timed | Time (mean ± sd) | n | Source |",
          "|---|---|---|---|---|---|"]
    if pl:
        L.append(f"| Panel method (Hess-Smith + Prandtl-Glauert) | {pl['n_panels']} panels | `solve()` only | "
                 f"{pl['solve']['mean']:.3f} ± {pl['solve']['sd']:.3f} ms (median {pl['solve']['median']:.3f}, max {pl['solve']['max']:.1f}) | {pl['solve']['n']} | measured |")
        L.append(f"| Panel method | {pl['n_panels']} panels | `solve()` + `field_grid({pl['field_width']})` | "
                 f"{pl['solve_field170']['mean']:.2f} ± {pl['solve_field170']['sd']:.2f} ms | "
                 f"{pl['solve_field170']['n']} | measured |")
    if nf is not None:
        if "error" in nf:
            L.append(f"| NeuralFoil | large | per call | not measured: {nf['error']} | | |")
        else:
            L.append(f"| NeuralFoil {nf['version']} | large, Re 6e6 | `get_aero_from_coordinates` (includes the Kulfan "
                     f"fit) | {nf['time']['mean']:.3f} ± {nf['time']['sd']:.3f} ms | {nf['time']['n']} | measured |")
    for mesh, sec in FVM_QUOTED_S.items():
        L.append(f"| FVM | {mesh} | one solve, wall time | {sec} s | | quoted (arXiv 2401.08705 Table 1) |")
    L.append("")
    if nf is not None and "error" not in nf:
        L.append(f"NeuralFoil returned CL = {nf['cl']:.3f} (viscous, Re 6e6, incompressible): **not comparable** with the "
                 "inviscid Ma 0.4 FVM Cl of 0.519. Only its time is reported; it is used in no accuracy statement.")
        L.append("")
    L += ["For context, the paper's own NNfoil training wall times in Table 1 (quoted, another machine) are "
          + ", ".join(f"{v} s at {k}" for k, v in NNFOIL_PAPER_S.items())
          + "; the paper notes that L-BFGS often stops early.", "",
          "## What this does and does not show", "",
          "- **A PINN is trained per condition.** The inference time above is the cost of evaluating a network that "
          "was already trained for ONE shape, Mach number and angle of attack. It is a lookup of that one "
          "condition, not a solver. A new condition (another Mach number, angle or shape) means retraining, and the "
          "training times in the table are what that costs on this laptop.",
          "- **The panel method needs no training.** Each new condition is a fresh solve of a few milliseconds. It "
          "is inviscid and linear (Prandtl-Glauert), so it does not capture the non-linear compressible effects the "
          "Euler PINN is built for.",
          "- **The FVM times are quoted from another machine.** They come from Table 1 of the paper, are one solve "
          "each, were run on the authors' hardware and are not re-measured here; the paper itself says the table is "
          "\"not meant as a rigorous comparison\".",
          "- **NeuralFoil solves a different problem** (viscous, incompressible, Re 6e6), so its time is a speed "
          "reference only and its Cl is not compared with ours.",
          "- **No fair-speed claim is made beyond what is measured.** Accuracy at equal cost is a different question: "
          "it is answered by E2 (Cp error against the 800x400 FVM), not by this table. All times were measured on one "
          "laptop CPU with PyTorch; GPUs and other batch sizes were not measured."]
    return "\n".join(L) + "\n"


def collect_e4(runs_root=runs.RUNS, summary="results/summary.csv", timing_path=None, md_path=None):
    """Timing JSON -> E4 rows in ``summary`` and the human-readable table ``results/e4_speed.md``."""
    timing_path = Path(timing_path) if timing_path is not None else ROOT / "results" / "e4_timing.json"
    md_path = Path(md_path) if md_path is not None else ROOT / "results" / "e4_speed.md"
    if not timing_path.exists():
        raise SystemExit(f"{timing_path} is missing: run `research/time_inference.py --e4` first (on an idle CPU)")
    timing = json.loads(timing_path.read_text(encoding="utf8"))
    rows = e4_rows(timing, runs_root)
    for r in rows:
        metrics.append_summary(r, summary)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_md_e4(timing, rows), encoding="utf8", newline="\n")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands", help="print the training commands (runs nothing)")
    c.add_argument("exp", choices=["e1", "e2", "e3"])
    c.add_argument("--mode", choices=["nopos", "pos"], default="nopos", help="e2: positivity mode of the 200x100 chain")
    c.add_argument("--rounds", type=int, default=None, help="override the number of rounds (e1, e3)")
    c.add_argument("--threads", type=int, default=8)
    k = sub.add_parser("collect", help="finished run folders -> results/summary.csv rows")
    k.add_argument("exp", choices=["e1", "e2", "e3", "e4"])
    k.add_argument("--summary", default="results/summary.csv")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.cmd == "commands":
        lines, notes = plan_commands(a.exp, mode=a.mode, threads=a.threads, rounds=a.rounds)
        for n in notes:
            print(f"# {n}", file=sys.stderr)
        for ln in lines:
            print(ln)
        if not lines:
            print(f"# nothing to run for {a.exp}", file=sys.stderr)
        return 0
    rows = collect_e4(summary=a.summary) if a.exp == "e4" else collect(a.exp, summary=a.summary)
    _print_rows(rows)
    extra = "results/e4_speed.md" if a.exp == "e4" else f"results/{a.exp}_rounds.csv"
    print(f"wrote {len(rows)} rows to {a.summary} and {extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
