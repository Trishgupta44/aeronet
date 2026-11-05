"""Experiment drivers (Plan C): print the training commands, then turn finished run folders into summary rows.

    research/.venv/Scripts/python research/experiments.py commands e1   # one shell line per run, nothing is run
    research/.venv/Scripts/python research/experiments.py collect  e1   # finished runs -> results/summary.csv

``commands`` only prints (it skips runs that already have the requested number of attempted rounds, and prints a
``--resume`` line for a partial run). Run the printed lines one after another (parallel runs spoil the timing numbers),
then run ``collect``. Every figure is made by ``make_figures.py``, never here.

E1  positivity on/off, NACA 0012 50x25, Ma 0.4, 4 deg, seeds 0-2, 20 rounds (seed 0 = the Plan A runs).
"""

from __future__ import annotations

import argparse
import csv
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
import runs  # noqa: E402
import validate  # noqa: E402

REF0012 = runs.FVM["NACA0012"]
MACH, ALPHA = 0.4, 4
E1_ROUNDS = 20
BUDGET_S = 3 * 3600  # roadmap rule 4


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


def plan_commands(exp, runs_root=runs.RUNS, threads=8, rounds=None):
    """``(lines, notes)``: the shell lines still to run for experiment ``exp`` and messages for the person running them."""
    runs_root = Path(runs_root)
    notes: list[str] = []
    if exp == "e1":
        specs = e1_specs(runs_root)
        if rounds:
            specs = [{**s, "rounds": rounds} for s in specs]
    else:
        raise SystemExit(f"unknown experiment {exp!r}")
    lines = []
    for s in specs:
        line = _command_for(s, threads, notes)
        if line:
            lines.append(line)
    return lines, notes


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
    else:
        raise SystemExit(f"unknown experiment {exp!r}")
    for r in rows:
        metrics.append_summary(r, summary)
    _write_traj(rounds_dir / f"{exp}_rounds.csv", traj)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands", help="print the training commands (runs nothing)")
    c.add_argument("exp", choices=["e1"])
    c.add_argument("--rounds", type=int, default=None, help="override the number of rounds")
    c.add_argument("--threads", type=int, default=8)
    k = sub.add_parser("collect", help="finished run folders -> results/summary.csv rows")
    k.add_argument("exp", choices=["e1"])
    k.add_argument("--summary", default="results/summary.csv")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.cmd == "commands":
        lines, notes = plan_commands(a.exp, threads=a.threads, rounds=a.rounds)
        for n in notes:
            print(f"# {n}", file=sys.stderr)
        for ln in lines:
            print(ln)
        if not lines:
            print(f"# nothing to run for {a.exp}", file=sys.stderr)
        return 0
    rows = collect(a.exp, summary=a.summary)
    _print_rows(rows)
    print(f"wrote {len(rows)} rows to {a.summary} and results/{a.exp}_rounds.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
