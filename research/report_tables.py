"""Markdown tables for the Milestone 3 report, built from the files in ``results/`` (no number is typed here).

    research/.venv/Scripts/python research/report_tables.py [name ...] [--summary results/summary.csv]

Prints every table (or the named ones: e1 headline seed_check e3 e4 e5 e6 e6_fixed) so that they can be pasted into
``docs/milestones/M3-pinn-results.md``. The functions return plain strings, which the tests also use. Labels follow the
roadmap rule: every row carries its ``source`` (measured, quoted or fitted) from ``summary.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import results_io as rio  # noqa: E402
from results_io import num, pick, rounds_of, select  # noqa: E402

RUNS_DIR = rio.RESULTS / "runs"

METHOD_LABEL = {
    "pinn": "PINN, positivity on",
    "pinn_nopos": "PINN, positivity off",
    "pinn_gs": "PINN grid-seq., positivity on",
    "pinn_gs_nopos": "PINN grid-seq., positivity off",
    "fvm": "FVM",
    "panel": "Panel method (Hess-Smith + Prandtl-Glauert)",
    "thin_airfoil": "Thin-airfoil theory (PG corrected)",
    "neuralfoil": "NeuralFoil 0.3.3",
}


# ---------------------------------------------------------------- formatting
def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def f(v, nd=4, empty="-") -> str:
    v = num(v) if not isinstance(v, (int, float)) else v
    return empty if v is None else f"{v:.{nd}f}"


def pct(cl, ref, nd=1, empty="-") -> str:
    cl, ref = num(cl), num(ref)
    if cl is None or ref is None or ref == 0:
        return empty
    return f"{100.0 * (cl - ref) / ref:+.{nd}f}%"


def dcl(cl, ref, nd=3) -> str:
    """Signed Cl difference (prediction minus reference): used where a percentage would blow up near Cl = 0."""
    cl, ref = num(cl), num(ref)
    return "-" if cl is None or ref is None else f"{cl - ref:+.{nd}f}"


def rounds_cell(notes) -> str:
    r = rounds_of(notes)
    return "-" if r is None else f"{r[0]} ({r[1] - r[0]})"


def _train_cell(row) -> str:
    t = num(row.get("train_s"))
    if t is not None:
        return f"{t:.1f}"
    return "withheld (laptop slept)" if "withheld" in (row.get("notes") or "") else "-"


# ---------------------------------------------------------------- E2: the headline table
def headline_table(summary=None) -> str:
    """NACA 0012, Mach 0.4, 4 degrees: FVM, panel, thin-airfoil and every PINN run, against FVM 800x400."""
    s = rio.read_summary(summary)
    rows = []
    e2 = select(s, experiment="E2")
    for mesh in ("50x25", "100x50", "200x100", "400x200"):
        r = pick(e2, method="fvm", mesh=mesh)
        rows.append(["FVM", mesh, "-", f(r["cl"]), pct(r["cl"], r["cl_ref"]), f(r["rms_cp"]), f(r["max_cp"], 3), "-",
                     r["source"]])
    for r in select(e2, method="panel"):
        rows.append([METHOD_LABEL["panel"], r["mesh"], "-", f(r["cl"]), pct(r["cl"], r["cl_ref"]), f(r["rms_cp"]),
                     f(r["max_cp"], 3), "-", r["source"]])
    b0 = select(s, experiment="B0", case="NACA0012_a4", method="panel")
    for r in b0:
        rows.append([METHOD_LABEL["panel"] + " (B0 baseline)", "200 panels", "-", f(r["cl"]), pct(r["cl"], r["cl_ref"]),
                     f(r["rms_cp"]), f(r["max_cp"], 3), "-", r["source"]])
    for r in select(s, experiment="B0", case="NACA0012_a4", method="thin_airfoil"):
        rows.append([METHOD_LABEL["thin_airfoil"], "-", "-", f(r["cl"]), pct(r["cl"], r["cl_ref"]), "-", "-", "-",
                     r["source"]])
    for r in select(e2, source="measured"):
        if not r["method"].startswith("pinn"):
            continue
        label = METHOD_LABEL[r["method"]]
        if r["method"] == "pinn" and r["mesh"] == "100x50":
            label = "PINN cold start, positivity on"
        rows.append([label, r["mesh"], rounds_cell(r["notes"]), f(r["cl"]), pct(r["cl"], r["cl_ref"]), f(r["rms_cp"]),
                     f(r["max_cp"], 3), _train_cell(r), r["source"]])
    head = ["Method", "Mesh", "Rounds (failed)", "Cl", "Cl error", "Cp RMS", "Cp max error", "Train s", "Source"]
    return md_table(head, rows)


# ---------------------------------------------------------------- E1
def e1_table(summary=None, rounds_csv=None) -> str:
    """Six 50x25 runs (positivity on/off x seeds 0-2): Cp RMS at rounds 3, 4, 5, the final round and the worst round."""
    s = rio.read_summary(summary)
    traj = rio.read_csv(rounds_csv if rounds_csv is not None else rio.RESULTS / "e1_rounds.csv")
    rows = []
    for r in select(s, experiment="E1"):
        case = r["case"]
        mine = [t for t in traj if t["case"] == case and t["method"] == r["method"]]
        by = {int(t["round"]): t for t in mine}
        rms = {k: num(by[k]["cp_rms"]) for k in (3, 4, 5)}
        worst = max(num(t["cp_rms"]) for t in mine)
        rows.append(["on" if r["method"] == "pinn" else "off", case.split("_s")[-1], rounds_cell(r["notes"]),
                     f(rms[3], 3), f(rms[4], 3), f(rms[5], 3), f(r["rms_cp"], 3), f(worst, 3), f(r["cl"], 3),
                     f(r["train_s"], 1)])
    rows.sort(key=lambda x: (x[0] != "on", x[1]))
    head = ["Positivity", "Seed", "Rounds (failed)", "RMS r3", "RMS r4", "RMS r5", "RMS final", "RMS worst round",
            "Cl final", "Train s"]
    return md_table(head, rows)


# ---------------------------------------------------------------- seed check
def seed_check_table(path=None) -> str:
    rows = []
    d = rio.read_csv(path if path is not None else rio.RESULTS / "seed_check.csv")
    order = [("nopos", "gs"), ("nopos", "cold"), ("pos", "gs"), ("pos", "cold")]
    for mode, method in order:
        mine = {int(r["seed"]): r for r in d if r["mode"] == mode and r["method"] == method}
        cells = []
        for seed in (0, 1, 2, 3):
            r = mine.get(seed)
            if r is None:
                cells.append("not run")
            elif r["collapsed"] == "yes":
                cells.append(f"**collapsed: all {r['failed_rounds']} rounds failed**")
            else:
                extra = f", {r['failed_rounds']} failed round" if int(r["failed_rounds"]) else ""
                extra += f", {r['rounds']} rounds" if method == "cold" and int(r["rounds"]) > 5 else ""
                cells.append(f"Cl {f(r['final_cl'], 3)} (RMS {f(r['cp_rms'], 4)}){extra}")
        rows.append([("off" if mode == "nopos" else "on"), "grid-sequenced" if method == "gs" else "cold start"] + cells)
    return md_table(["Positivity", "Method", "Seed 0", "Seed 1", "Seed 2", "Seed 3"], rows)


# ---------------------------------------------------------------- E3
def e3_table(summary=None) -> str:
    """NACA 2412 at Mach 0.3 (quoted): Cl by angle for FVM, panel, thin-airfoil and every PINN run (diff = minus FVM)."""
    s = rio.read_summary(summary)
    e3 = select(s, experiment="E3")
    alphas = sorted({int(num(r["alpha_deg"])) for r in e3})
    rows = []
    for a in alphas:
        fv = pick(e3, method="fvm", alpha_deg=a)
        pn = pick(e3, method="panel", alpha_deg=a)
        th = pick(e3, method="thin_airfoil", alpha_deg=a)
        base = [str(a), f(fv["cl"]), f"{f(pn['cl'])} ({dcl(pn['cl'], fv['cl_ref'])})",
                f"{f(th['cl'])} ({dcl(th['cl'], th['cl_ref'])})"]
        pinn = [r for r in e3 if r["method"].startswith("pinn") and int(num(r["alpha_deg"])) == a]
        if not pinn:
            rows.append(base + ["-", "-", "-", "-"])
        for r in pinn:
            collapsed = abs(num(r["cl"])) < 0.1 < abs(num(r["cl_ref"]))
            cl_cell = (f"**{abs(num(r['cl'])):.4f} collapsed** ({dcl(abs(num(r['cl'])), r['cl_ref'])})" if collapsed
                       else f"{f(r['cl'])} ({dcl(r['cl'], r['cl_ref'])})")
            seed = "1" if r["case"].endswith("_s1") else "0"
            pos = "off" if r["method"] == "pinn_nopos" else "on"
            rms = f"{f(r['rms_cp'], 2)} (collapsed)" if collapsed else f(r["rms_cp"])
            rows.append(base + [f"seed {seed}, positivity {pos}", cl_cell, rms, f(r["train_s"], 1)])
    head = ["alpha (deg)", "FVM 800x400 Cl", "Panel Cl (diff)", "Thin-airfoil Cl (diff)", "PINN run", "PINN Cl (diff)",
            "PINN Cp RMS", "PINN train s"]
    return md_table(head, rows)


# ---------------------------------------------------------------- E4
def e4_table(summary=None, timing=None) -> str:
    """Speed table: measured query times and the FVM times quoted from the paper, kept apart by the Source column."""
    s = rio.read_summary(summary)
    t = rio.read_json(timing if timing is not None else rio.RESULTS / "e4_timing.json")
    e4 = select(s, experiment="E4")
    rows = []
    for r in e4:
        if r["method"] in ("pinn", "pinn_nopos", "pinn_gs", "pinn_gs_nopos"):
            run = next(k for k in t["pinn"] if _run_method(k) == r["method"] and _run_mesh(k) == r["mesh"])
            th = t["pinn"][run]["threads"]
            label = METHOD_LABEL[r["method"]]
            rows.append([f"{label}: full field", r["mesh"], f"{th['1']['field']['mean']:.2f}",
                         f"{th['8']['field']['mean']:.2f}", f"{th['8']['wall']['mean']:.2f}", r["source"]])
    for r in e4:
        if r["method"] == "panel":
            pl = t["panel"]
            rows.append(["Panel method: solve()", r["mesh"], "n/a", f"{pl['solve']['mean']:.2f}", "-", r["source"]])
            rows.append(["Panel method: solve() + 170-wide field grid", r["mesh"], "n/a",
                         f"{pl['solve_field170']['mean']:.1f}", "-", r["source"]])
        elif r["method"] == "neuralfoil":
            rows.append(["NeuralFoil 0.3.3 (viscous, Re 6e6: not comparable)", "large", "n/a",
                         f"{t['neuralfoil']['time']['mean']:.2f}", "-", r["source"]])
    for r in e4:
        if r["method"] == "fvm":
            rows.append(["FVM solve (another machine, quoted from the paper)", r["mesh"], "n/a",
                         f"{num(r['infer_ms']):,.0f}", "-", r["source"]])
    head = ["What is timed", "Mesh", "1 thread, ms", "8 threads (or default), ms", "Wall only, 8 threads, ms", "Source"]
    return md_table(head, rows)


def _run_mesh(run_id: str) -> str:
    import re

    return re.search(r"_(\d+x\d+)_", run_id).group(1)


def _run_method(run_id: str) -> str:
    return "pinn" + ("_gs" if "_gs" in run_id else "") + ("_nopos" if "nopos" in run_id else "")


# ---------------------------------------------------------------- E5
def e5_table(summary=None) -> str:
    """Cylinder at Mach 0.4, alpha 0: Cp error against FVM 800x400 and the suction peak."""
    s = rio.read_summary(summary)
    e5 = select(s, experiment="E5")
    rows = []

    def ext(r, key):
        v = rio.note_value(r["notes"], key)
        return "-" if v is None else f"{v:.2f}"

    order = []
    for r in e5:
        if r["method"].startswith("pinn"):
            order.append((0, r["method"], r))
    for r in e5:
        if r["method"] == "panel":
            order.append((1, r["mesh"], r))
    for r in e5:
        if r["method"] == "fvm":
            order.append((2, num(r["mesh"].split("x")[0]), r))
    order.sort(key=lambda x: (x[0], str(x[1]) if x[0] < 2 else x[1]))
    for kind, _, r in order:
        if kind == 0:
            label = METHOD_LABEL[r["method"]] + ", 10 rounds"
        elif kind == 1:
            label = "Panel + Prandtl-Glauert (Mach 0.4)" if "Ma 0" not in r["mesh"] else "Incompressible circle (panel, Mach 0)"
        else:
            label = "FVM" + (" (the reference)" if r["mesh"] == "800x400" else "")
        rows.append([label, r["mesh"], f(r["rms_cp"]) if num(r["rms_cp"]) is not None else "-",
                     f(r["max_cp"], 3) if num(r["max_cp"]) is not None else "-", ext(r, "Cp_min"), ext(r, "Cp_stag"),
                     r["source"]])
    return md_table(["Method", "Mesh", "Cp RMS", "Cp max error", "Cp min", "Cp at stagnation", "Source"], rows)


# ---------------------------------------------------------------- E6
GROUPS = [("development", "DEVELOPMENT: E1 runs (thresholds were set after looking at them)"),
          ("held-out-new", "HELD-OUT: six new 50x25 seeds"),
          ("held-out", "HELD-OUT: other finished runs (cylinder, NACA 2412, 100x50, 200x100)")]


def _truthy(v) -> bool:
    return str(v).strip().lower() == "true"


def e6_table(replay=None) -> str:
    """Plateau stop rule, replayed on finished runs: counts per group (from ``results/e6_replay.csv``)."""
    d = rio.read_csv(replay if replay is not None else rio.RESULTS / "e6_replay.csv")
    rows = []

    def row(label, rs):
        n = len(rs)
        drifted = [r for r in rs if _truthy(r["drifted"])]
        stopped = [r for r in rs if r["stop_round"] != "never"]
        drifted_stopped = [r for r in drifted if r["stop_round"] != "never"]
        bad = [r for r in stopped if _truthy(r["bad_stop"])]
        best = np.mean([num(r["rms_best"]) for r in rs])
        return [label, n, len(drifted), f"{len(drifted_stopped)} of {len(drifted)}", len(stopped), n - len(stopped),
                f"{len(bad)} of {len(stopped)}", f"{best:.4f}"]

    for key, label in GROUPS:
        rows.append(row(label, [r for r in d if r["group"] == key]))
    rows.append(row("HELD-OUT, all", [r for r in d if r["group"].startswith("held-out")]))
    return md_table(["Group", "Runs", "Drifted (final RMS > 2x best)", "Drifted runs the rule stopped", "Rule stopped",
                     "Never stopped", "Bad stops (> 1.5x best)", "Mean best-round RMS (post-hoc)"], rows)


def e6_fixed_table(replay=None, runs_dir=None) -> str:
    """Cp RMS if training were stopped at a fixed round 3, 4 or 5, against the final round (50x25 NACA 0012 groups)."""
    d = rio.read_csv(replay if replay is not None else rio.RESULTS / "e6_replay.csv")
    runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    rows = []
    for key, label in (("development", "E1 runs (development)"), ("held-out-new", "six new seeds (held-out)")):
        series = [rio.round_series(runs_dir / r["run_id"], rio.REF_0012) for r in d if r["group"] == key]
        cells = []
        for k in (3, 4, 5, None):
            vals = []
            for s_ in series:
                i = s_["round"].index(k) if k is not None else len(s_["round"]) - 1
                vals.append(s_["cp_rms"][i])
            cells.append(f"{min(vals):.3f}-{max(vals):.3f} (mean {np.mean(vals):.3f})")
        rows.append([label, len(series)] + cells)
    return md_table(["Group", "Runs", "Round 3", "Round 4", "Round 5", "Final round"], rows)


# ---------------------------------------------------------------- command line
TABLES = {"e1": e1_table, "headline": headline_table, "seed_check": seed_check_table, "e3": e3_table, "e4": e4_table,
          "e5": e5_table, "e6": e6_table, "e6_fixed": e6_fixed_table}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help=f"tables to print (default: all of {', '.join(TABLES)})")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    for name in a.names or list(TABLES):
        if name not in TABLES:
            raise SystemExit(f"unknown table {name!r}; choose from {', '.join(TABLES)}")
        print(f"### {name}\n\n{TABLES[name]()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
