"""E6: does the reference-free plateau stop rule stop training near the good answer? (replay on finished runs)

    research/.venv/Scripts/python research/stop_rule_eval.py replay             # every finished run -> results/e6_replay.csv
    research/.venv/Scripts/python research/stop_rule_eval.py replay --summary   # also add the E6 rows to results/summary.csv
    research/.venv/Scripts/python research/stop_rule_eval.py new-runs           # the 6 new E6 seed runs -> summary rows
    research/.venv/Scripts/python research/stop_rule_eval.py tables             # markdown tables for results/e6_stop_rule.md

The rule is the one in ``NNfoil/run_case.py`` (``--stop-rule plateau``): stop at the first logged round where the
last ``window`` rounds all have |dCl| < ``dcl`` and a wall-Cp RMS change < ``dcp``. This script replays it on the
logs of finished runs with the SAME functions (``run_case.round_changes`` and ``run_case.plateau_reached``) and
compares the answer at the stop round with the answer at the final round and with the best round.

HONESTY: the thresholds (dCl 0.002, dCp 0.005, window 2) were set after looking at the E1 runs, so E1 is the
DEVELOPMENT set. Only the other runs can support a claim. The thresholds below are fixed and are not changed after
seeing results. The "best round" is chosen with the FVM error: it is post-hoc and only a reference point.
The FVM reference is used to JUDGE the stop round, never to find it.
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
for _p in (HERE, ROOT / "NNfoil", ROOT / "app" / "backend"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import metrics  # noqa: E402
import runs  # noqa: E402

# Fixed before this analysis (the defaults of run_case.py), set after looking at E1.
DCL_TOL, DCP_TOL, WINDOW = 0.002, 0.005, 2
MIN_GOOD_ROUNDS = 6
EARLY_FACTOR = 1.5   # "stopped at a bad answer": Cp RMS at the stop round above this times the best-round RMS
JUMP_FACTOR = 2.0    # "drifted": final Cp RMS above this times the best-round RMS; the jump round is the first round above it
SLEEP_FACTOR, SLEEP_MIN_S = 5.0, 300.0  # a round that took > 5x the median round time and > 300 s: laptop sleep, timing discarded

DEV_RUNS = frozenset({
    "naca0012_50x25_a4_m04", "naca0012_50x25_a4_m04_nopos",
    "naca0012_50x25_a4_m04_s1_r20", "naca0012_50x25_a4_m04_s2_r20",
    "naca0012_50x25_a4_m04_nopos_s1_r20", "naca0012_50x25_a4_m04_nopos_s2_r20",
})
NEW_RE = re.compile(r"^naca0012_50x25_a4_m04(_nopos)?_s([345])_e6$")
ONLINE_RUN = "naca0012_50x25_a4_m04_s3_e6_online"
# two held-out runs were already replayed (not tuned on) in the drift Step 1 of docs/nnfoil_improvements.md
SEEN_IN_STEP1 = frozenset({"naca0012_100x50_a4_m04", "naca0012_200x100_a4_m04_gs_nopos"})

REPLAY_NOTE_DEV = ("replay of the plateau rule on a finished run; thresholds fixed before this analysis: dCl 0.002, dCp 0.005, "
                   "window 2; set after looking at E1 (DEVELOPMENT run, not a claim)")
REPLAY_NOTE_HELD = ("replay of the plateau rule on a finished run; thresholds fixed before this analysis: dCl 0.002, dCp 0.005, "
                    "window 2; held-out (not used to set the thresholds)")

COLUMNS = ["run_id", "group", "shape", "mesh", "mach", "alpha_deg", "positivity", "seed", "n_rounds", "stop_round",
           "rms_stop", "max_stop", "rms_final", "final_round", "best_round", "rms_best", "cl_ref", "cl_stop", "cl_final",
           "cl_best_round", "cl_best", "train_s_stop", "train_s_final", "saved_s", "timing_suspect", "drifted",
           "jump_round", "stopped_before_jump", "bad_stop", "bad_stop_kind", "seen_in_step1", "plateau_margin"]


def _rc():
    """The two stop-rule functions from run_case.py (imported lazily: it imports torch)."""
    import run_case

    return run_case.round_changes, run_case.plateau_reached


def _r(v, nd=6):
    return None if v is None else round(float(v), nd)


# ---------------------------------------------------------------- which run is which
def group_of(run_id: str) -> str:
    if run_id in DEV_RUNS:
        return "development"
    if NEW_RE.match(run_id):
        return "held-out-new"
    return "held-out"


def reference_for(cfg: dict):
    """FVM 800x400 file for a run (the finest reference of that shape and condition), or None for an unknown shape."""
    shape = cfg.get("shape")
    if shape in runs.FVM:
        return runs.FVM[shape]
    if shape == "NACA2412":
        p = runs.fvm_2412(cfg["alpha_deg"])
        return p if p.exists() else None
    return None


# ---------------------------------------------------------------- the replay
def replay_stop(d_cl, d_cp, dcl=DCL_TOL, dcp=DCP_TOL, window=WINDOW):
    """Index (into the per-round lists) of the first round where ``plateau_reached`` is true on the prefix, else None."""
    _, plateau_reached = _rc()
    for k in range(len(d_cl)):
        if plateau_reached(d_cl[:k + 1], d_cp[:k + 1], dcl, dcp, window):
            return k
    return None


def trajectory(run_dir, ref_path, alpha):
    """Per logged round: round, time_s, cl, cp_rms, cp_max, d_cl, d_cp (changes use ``run_case.round_changes``)."""
    round_changes, _ = _rc()
    d = Path(run_dir)
    log = json.loads((d / "log.json").read_text(encoding="utf8"))
    snap = json.loads((d / "snapshots.json").read_text(encoding="utf8"))
    by_round = {int(r["round"]): r["cp"] for r in snap["rounds"]}
    ref = metrics.load_fvm(ref_path)
    out, prev_cl, prev_cp = [], None, None
    for i, r in enumerate(log["round"]):
        r = int(r)
        cp = by_round[r]
        err = metrics.cp_error(snap["x"], snap["y"], cp, ref)
        d_cl, d_cp = round_changes(log["cl"][i], cp, prev_cl, prev_cp)
        out.append({"round": r, "time_s": float(log["time_s"][i]), "cl": float(log["cl"][i]), "cp_rms": err["rms"],
                    "cp_max": err["max"], "d_cl": d_cl, "d_cp": d_cp})
        prev_cl, prev_cp = log["cl"][i], cp
    return out, log


def timing_suspect(times, upto=None) -> bool:
    """True when a round up to index ``upto`` (inclusive) took > 5x the median round time and > 300 s (a laptop sleep)."""
    t = np.asarray(times, float)
    dt = np.diff(np.r_[0.0, t])
    if upto is not None:
        dt = dt[:upto + 1]
    if dt.size == 0:
        return False
    return bool(dt.max() > max(SLEEP_MIN_S, SLEEP_FACTOR * float(np.median(np.diff(np.r_[0.0, t])))))


def plateau_margin(traj, dcl=DCL_TOL, dcp=DCP_TOL, window=WINDOW):
    """Closest approach to the rule: min over windows of the largest ratio (dCl/dcl, dCp/dcp). The rule fires iff it is < 1."""
    ratio = [None if t["d_cl"] is None else max(t["d_cl"] / dcl, t["d_cp"] / dcp) for t in traj]
    best = None
    for k in range(window - 1, len(ratio)):
        w = ratio[k - window + 1:k + 1]
        if any(v is None for v in w):
            continue
        m = max(w)
        best = m if best is None else min(best, m)
    return best


def evaluate_run(run_dir, ref_path=None):
    """One result dict for a finished run, or None when it does not qualify.

    Skipped: missing ``config.json`` / ``log.json`` / ``snapshots.json``, fewer than ``MIN_GOOD_ROUNDS`` logged rounds,
    an unknown shape (no reference), or a run that was itself trained with a stop rule (online confirmation runs).
    """
    d = Path(run_dir)
    if not all((d / f).exists() for f in ("config.json", "log.json", "snapshots.json")):
        return None
    cfg = json.loads((d / "config.json").read_text(encoding="utf8"))
    if cfg.get("stop_rule", "none") != "none":
        return None
    ref_path = ref_path or reference_for(cfg)
    if ref_path is None:
        return None
    alpha = float(cfg["alpha_deg"])
    traj, log = trajectory(d, ref_path, alpha)
    if len(traj) < MIN_GOOD_ROUNDS:
        return None
    cl_ref, _ = metrics.fvm_forces(ref_path, alpha)
    k = replay_stop([t["d_cl"] for t in traj], [t["d_cp"] for t in traj])
    last = traj[-1]
    best = min(traj, key=lambda t: t["cp_rms"])
    best_cl = min(traj, key=lambda t: abs(t["cl"] - cl_ref))
    at = traj[k] if k is not None else last          # a run that never triggers is judged at its final round
    rms_best, rms_stop = best["cp_rms"], at["cp_rms"]
    jump = next((t["round"] for t in traj if t["round"] > best["round"] and t["cp_rms"] > JUMP_FACTOR * rms_best), None)
    drifted = last["cp_rms"] > JUMP_FACTOR * rms_best
    bad = k is not None and rms_stop > EARLY_FACTOR * rms_best
    kind = ""
    if bad:
        kind = "early" if at["round"] < best["round"] else "late"
    suspect = timing_suspect([t["time_s"] for t in traj], upto=k if k is not None else None)
    suspect_final = timing_suspect([t["time_s"] for t in traj])
    row = {
        "run_id": d.name, "group": group_of(d.name), "shape": cfg["shape"], "mesh": cfg["mesh"], "mach": cfg["mach"],
        "alpha_deg": alpha, "positivity": bool(cfg.get("positivity", True)), "seed": int(cfg.get("seed", 0)),
        "n_rounds": len(traj), "stop_round": at["round"] if k is not None else None,
        "rms_stop": _r(at["cp_rms"]), "max_stop": _r(at["cp_max"]), "rms_final": _r(last["cp_rms"]),
        "final_round": last["round"], "best_round": best["round"], "rms_best": _r(rms_best), "cl_ref": _r(cl_ref),
        "cl_stop": _r(at["cl"]), "cl_final": _r(last["cl"]), "cl_best_round": best_cl["round"],
        "cl_best": _r(best_cl["cl"]), "train_s_stop": _r(at["time_s"], 1), "train_s_final": _r(last["time_s"], 1),
        "saved_s": None if k is None else _r(last["time_s"] - at["time_s"], 1),
        "timing_suspect": bool(suspect or suspect_final), "drifted": bool(drifted), "jump_round": jump,
        "stopped_before_jump": None if (k is None or jump is None) else bool(at["round"] < jump),
        "bad_stop": bool(bad), "bad_stop_kind": kind, "seen_in_step1": d.name in SEEN_IN_STEP1,
        "plateau_margin": _r(plateau_margin(traj), 3),
        "_cfg": cfg, "_traj": traj, "_log": log,
    }
    return row


def evaluate_all(runs_root=runs.RUNS, exclude=None):
    """Result rows of every qualifying run folder under ``runs_root`` (sorted by group, then name)."""
    out = []
    for d in sorted(Path(runs_root).iterdir()):
        if not d.is_dir() or (exclude and re.search(exclude, d.name)):
            continue
        r = evaluate_run(d)
        if r is not None:
            out.append(r)
    order = {"development": 0, "held-out": 1, "held-out-new": 2}
    out.sort(key=lambda r: (order[r["group"]], r["run_id"]))
    return out


# ---------------------------------------------------------------- aggregate
def aggregate(rows):
    """Counts and means for a list of result rows (one group)."""
    n = len(rows)
    stopped = [r for r in rows if r["stop_round"] is not None]
    drifted = [r for r in rows if r["drifted"]]
    d_stop = [r for r in drifted if r["stop_round"] is not None]
    mean = lambda v: float(np.mean(v)) if len(v) else None  # noqa: E731
    clean_saved = [r["saved_s"] for r in stopped if not r["timing_suspect"] and r["saved_s"] is not None]
    return {
        "n": n, "n_stopped": len(stopped), "n_never": n - len(stopped),
        "n_drifted": len(drifted), "n_drifted_stopped": len(d_stop),
        "n_drifted_stopped_good": sum(1 for r in d_stop if not r["bad_stop"]),
        "n_drifted_stopped_before_jump": sum(1 for r in d_stop if r["stopped_before_jump"]),
        "n_bad_stop": sum(1 for r in stopped if r["bad_stop"]),
        "n_too_early": sum(1 for r in stopped if r["bad_stop_kind"] == "early"),
        "n_late_bad": sum(1 for r in stopped if r["bad_stop_kind"] == "late"),
        "mean_stop_round": mean([r["stop_round"] for r in stopped]),
        "mean_rms_stop_stopped": mean([r["rms_stop"] for r in stopped]),
        "mean_rms_final_stopped": mean([r["rms_final"] for r in stopped]),
        "mean_rms_stop_all": mean([r["rms_stop"] for r in rows]),
        "mean_rms_final_all": mean([r["rms_final"] for r in rows]),
        "mean_rms_best_all": mean([r["rms_best"] for r in rows]),
        "saved_s_total": float(sum(clean_saved)) if clean_saved else 0.0, "n_saved_s": len(clean_saved),
    }


# ---------------------------------------------------------------- exploratory sensitivity (NOT a claim)
SENS_GRID = [(0.002, 0.005, 2), (0.005, 0.01, 2), (0.01, 0.02, 2), (0.02, 0.03, 2), (0.002, 0.005, 3), (0.005, 0.01, 3),
             (0.01, 0.02, 3)]


def sensitivity(rows, grid=SENS_GRID):
    """Per threshold set: for the development runs and for all held-out runs, how many stop, and how many of the stops are bad."""
    out = []
    for dcl, dcp, win in grid:
        rec = {"dcl": dcl, "dcp": dcp, "window": win}
        for name, sel in (("dev", lambda r: r["group"] == "development"), ("held", lambda r: r["group"] != "development")):
            sub = [r for r in rows if sel(r)]
            n_stop = n_bad = 0
            rms_stop, rms_final = [], []
            for r in sub:
                tr = r["_traj"]
                k = replay_stop([t["d_cl"] for t in tr], [t["d_cp"] for t in tr], dcl, dcp, win)
                at = tr[k] if k is not None else tr[-1]
                rms_stop.append(at["cp_rms"])
                rms_final.append(tr[-1]["cp_rms"])
                if k is not None:
                    n_stop += 1
                    n_bad += at["cp_rms"] > EARLY_FACTOR * r["rms_best"]
            rec.update({f"{name}_n": len(sub), f"{name}_stopped": n_stop, f"{name}_bad": int(n_bad),
                        f"{name}_mean_rms_judged": float(np.mean(rms_stop)) if sub else None,
                        f"{name}_mean_rms_final": float(np.mean(rms_final)) if sub else None})
        out.append(rec)
    return out


def sensitivity_md(rows) -> str:
    head = ("| dCl < | dCp < | window | DEV runs stopped (of n) | DEV bad stops | DEV mean RMS (stop, else final) | DEV mean RMS final "
            "| HELD-OUT runs stopped (of n) | HELD-OUT bad stops | HELD-OUT mean RMS (stop, else final) | HELD-OUT mean RMS final |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    out = []
    for s in sensitivity(rows):
        out.append(f"| {s['dcl']} | {s['dcp']} | {s['window']} | {s['dev_stopped']} of {s['dev_n']} | {s['dev_bad']} | "
                   f"{_f(s['dev_mean_rms_judged'])} | {_f(s['dev_mean_rms_final'])} | {s['held_stopped']} of {s['held_n']} | "
                   f"{s['held_bad']} | {_f(s['held_mean_rms_judged'])} | {_f(s['held_mean_rms_final'])} |")
    return head + "\n".join(out) + "\n"


# ---------------------------------------------------------------- csv and summary rows
def write_replay_csv(rows, path):
    with Path(path).open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("never" if k == "stop_round" and r[k] is None else ("" if r[k] is None else r[k]))
                        for k in COLUMNS})


def _stop_text(r) -> str:
    return "never triggered (numbers are the final round)" if r["stop_round"] is None else f"stopped at round {r['stop_round']} of {r['n_rounds']}"


def replay_summary_row(r) -> dict:
    """``results/summary.csv`` row (E6, ``pinn_stoprule_replay``): numbers at the stop round (final round if never)."""
    cfg = r["_cfg"]
    from validate import _num, method_name

    notes = [REPLAY_NOTE_DEV if r["group"] == "development" else REPLAY_NOTE_HELD, f"group={r['group']}",
             f"run={r['run_id']}", f"training method {method_name(cfg)}", _stop_text(r),
             f"rms_final={r['rms_final']:.4g}@round{r['final_round']}",
             f"best_round_rms={r['rms_best']:.4g}@round{r['best_round']} (post-hoc: picked using the FVM error)",
             f"cl_final={r['cl_final']:.4f}", f"cl_best_round={r['cl_best']:.4f}@round{r['cl_best_round']} (post-hoc)",
             f"ref: FVM 800x400 file of the case"]
    if r["seen_in_step1"]:
        notes.append("this run was also replayed in drift Step 1 (docs/nnfoil_improvements.md 3c); thresholds not changed since")
    if cfg.get("init_from"):
        notes.append("grid-sequenced: train_s counts only the fine-mesh stage")
    if r["timing_suspect"]:
        notes.append("train_s DISCARDED (a round took far longer than the median: laptop sleep)")
    return {"experiment": "E6", "case": r["run_id"], "method": "pinn_stoprule_replay", "mesh": r["mesh"],
            "mach": _num(r["mach"]), "alpha_deg": _num(r["alpha_deg"]), "rms_cp": r["rms_stop"], "max_cp": r["max_stop"],
            "cl": r["cl_stop"], "cl_ref": r["cl_ref"], "cd": None,
            "train_s": None if r["timing_suspect"] else r["train_s_stop"], "infer_ms": None, "source": "measured",
            "notes": "; ".join(notes)}


def stoprule_summary_row(r) -> dict:
    """E6 ``pinn_stoprule`` / ``pinn_nopos_stoprule`` row of a new seed run, case ``NACA0012_a4_s<seed>``."""
    cfg = r["_cfg"]
    from validate import _num, case_name

    pos = r["positivity"]
    last_cd = r["_log"]["cd"]
    # cd at the stop round
    idx = next(i for i, t in enumerate(r["_traj"]) if t["round"] == (r["stop_round"] or r["final_round"]))
    notes = [f"plateau rule replayed on the finished {r['n_rounds']}-round run {r['run_id']} (no stop rule was used in training; "
             f"online check: pinn_stoprule_online); thresholds dCl 0.002, dCp 0.005, window 2, fixed in advance; "
             f"held-out (new seed {r['seed']})", _stop_text(r), f"rms_final={r['rms_final']:.4g}@round{r['final_round']}",
             f"best_round_rms={r['rms_best']:.4g}@round{r['best_round']} (post-hoc: picked using the FVM error)",
             f"cl_final={r['cl_final']:.4f}", "train_s = cumulative training time at the stop round",
             "positivity " + ("on" if pos else "off")]
    if r["timing_suspect"]:
        notes.append("train_s DISCARDED (a round took far longer than the median: laptop sleep)")
    return {"experiment": "E6", "case": case_name(cfg), "method": "pinn_stoprule" if pos else "pinn_nopos_stoprule",
            "mesh": r["mesh"], "mach": _num(r["mach"]), "alpha_deg": _num(r["alpha_deg"]), "rms_cp": r["rms_stop"],
            "max_cp": r["max_stop"], "cl": r["cl_stop"], "cl_ref": r["cl_ref"], "cd": _r(last_cd[idx]),
            "train_s": None if r["timing_suspect"] else r["train_s_stop"], "infer_ms": None, "source": "measured",
            "notes": "; ".join(notes)}


# ---------------------------------------------------------------- markdown tables
def _f(v, nd=4):
    return "" if v is None else f"{v:.{nd}f}"


def run_table_md(rows) -> str:
    head = ("| run | pos | seed | rounds | stop round | Cp RMS at stop | Cp RMS final | best round (post-hoc) | best RMS (post-hoc) "
            "| Cl at stop | Cl final | Cl ref | verdict |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    out = []
    for r in rows:
        if r["stop_round"] is None:
            verdict = "never stops" + (" (drifted)" if r["drifted"] else " (no drift)")
        elif r["bad_stop"]:
            verdict = f"bad stop ({r['bad_stop_kind']})"
        else:
            verdict = "good stop" + (" before the jump" if r["stopped_before_jump"] else (" (no drift in this run)" if not r["drifted"] else ""))
        out.append(f"| {r['run_id']} | {'on' if r['positivity'] else 'off'} | {r['seed']} | {r['n_rounds']} | "
                   f"{'never' if r['stop_round'] is None else r['stop_round']} | {_f(r['rms_stop'])} | {_f(r['rms_final'])} | "
                   f"{r['best_round']} | {_f(r['rms_best'])} | {_f(r['cl_stop'])} | {_f(r['cl_final'])} | {_f(r['cl_ref'])} | {verdict} |")
    return head + "\n".join(out) + "\n"


def aggregate_table_md(groups: dict) -> str:
    head = ("| group | runs | drifted (final RMS > 2x best) | rule stopped | never stopped | stopped at a good answer "
            "(RMS <= 1.5x best) | bad stop, too early | bad stop, after the jump | mean RMS at stop (stopped runs) "
            "| mean RMS final (same runs) | mean RMS best (post-hoc, all runs) |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    out = []
    for name, rows in groups.items():
        a = aggregate(rows)
        out.append(f"| {name} | {a['n']} | {a['n_drifted']} | {a['n_stopped']} | {a['n_never']} | "
                   f"{a['n_stopped'] - a['n_bad_stop']} | {a['n_too_early']} | {a['n_late_bad']} | "
                   f"{_f(a['mean_rms_stop_stopped'])} | {_f(a['mean_rms_final_stopped'])} | {_f(a['mean_rms_best_all'])} |")
    return head + "\n".join(out) + "\n"


# ---------------------------------------------------------------- the new E6 seed runs
def new_run_summary_rows(runs_root=runs.RUNS, rows=None):
    """E6 rows of the six new seed runs: full-run row (``pinn`` / ``pinn_nopos``) and the stop-rule row, side by side."""
    import experiments as ex

    rows = rows if rows is not None else evaluate_all(runs_root)
    out = []
    for r in rows:
        if r["group"] != "held-out-new":
            continue
        row, _ = ex.pinn_row(Path(runs_root) / r["run_id"], runs.FVM["NACA0012"], "E6", case=None,
                             extra=f"E6 held-out seed {r['seed']}, 20 planned rounds with --diagnostics and no stop rule; "
                                   "full-run numbers (final round), next to the pinn_stoprule row")
        out.append(row)
        out.append(stoprule_summary_row(r))
    return out


def online_summary_row(runs_root=runs.RUNS, predicted=None):
    """E6 ``pinn_stoprule_online`` row of the run trained with ``--stop-rule plateau`` (numbers at its last round)."""
    import experiments as ex

    d = Path(runs_root) / ONLINE_RUN
    log = json.loads((d / "log.json").read_text(encoding="utf8"))
    stop = log.get("stop")
    row, _ = ex.pinn_row(d, runs.FVM["NACA0012"], "E6", case=None, extra="")
    row["method"] = "pinn_stoprule_online"
    where = (f"stopped by the rule at round {stop['round']}" if stop else "the rule did not trigger within the planned rounds")
    row["notes"] = (f"online confirmation: trained with --stop-rule plateau --diagnostics (seed 3, positivity on, new folder {ONLINE_RUN}); "
                    f"{where}; replay of the earlier full run predicted round {predicted}; thresholds dCl 0.002, dCp 0.005, window 2; "
                    f"held-out; numbers are the last round; " + row["notes"])
    return row


# ---------------------------------------------------------------- cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("replay", "new-runs", "tables"):
        p = sub.add_parser(name)
        p.add_argument("--runs", default=str(runs.RUNS))
        p.add_argument("--exclude", default=None, help="regex of run folder names to leave out")
        p.add_argument("--out", default=str(ROOT / "results" / "e6_replay.csv"))
        p.add_argument("--summary-path", default=str(ROOT / "results" / "summary.csv"))
        if name == "replay":
            p.add_argument("--summary", action="store_true", help="also write the E6 rows to results/summary.csv")
        if name == "new-runs":
            p.add_argument("--online", action="store_true", help="also add the online-confirmation row")
            p.add_argument("--predicted", type=int, default=None, help="stop round predicted by the replay of the full run")
    a = ap.parse_args(argv)
    rows = evaluate_all(a.runs, a.exclude)
    if a.cmd == "replay":
        write_replay_csv(rows, a.out)
        print(f"{len(rows)} runs with >= {MIN_GOOD_ROUNDS} rounds -> {a.out}")
        for r in rows:
            print(f"{r['group']:13s} {r['run_id']:40s} stop {str(r['stop_round']):>5s}  rms stop {r['rms_stop']:.4f}  "
                  f"final {r['rms_final']:.4f}  best {r['rms_best']:.4f}@{r['best_round']}")
        if a.summary:
            n = 0
            for r in rows:
                if r["group"] == "held-out-new":   # these get pinn_stoprule rows via ``new-runs``
                    continue
                metrics.append_summary(replay_summary_row(r), a.summary_path)
                n += 1
            print(f"{n} rows (E6, pinn_stoprule_replay) -> {a.summary_path}")
    elif a.cmd == "new-runs":
        out = new_run_summary_rows(a.runs, rows)
        if a.online:
            out.append(online_summary_row(a.runs, a.predicted))
        for row in out:
            metrics.append_summary(row, a.summary_path)
            print(f"{row['case']:16s} {row['method']:22s} rms {row['rms_cp']} cl {row['cl']} train_s {row['train_s']}")
        print(f"{len(out)} rows -> {a.summary_path}")
    else:
        groups = {"DEVELOPMENT (E1, thresholds set on these)": [r for r in rows if r["group"] == "development"],
                  "HELD-OUT, finished runs (existing)": [r for r in rows if r["group"] == "held-out"],
                  "HELD-OUT, new seeds (E6)": [r for r in rows if r["group"] == "held-out-new"]}
        print(aggregate_table_md(groups))
        print("\nSENSITIVITY (tuned on held-out data: not a claim)\n")
        print(sensitivity_md(rows))
        for name, g in groups.items():
            print(f"\n### {name}\n\n{run_table_md(g)}")
            a_ = aggregate(g)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in a_.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
