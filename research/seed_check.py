"""Seed check (plan task A12b): does grid sequencing keep Cl > 0.49 on 100x50, and does the cold start collapse?

    research/.venv/Scripts/python research/seed_check.py [--runs results/runs] [--out results/seed_check.csv]

Scans the run folders under ``--runs`` (NACA0012, mesh 100x50; the 50x25 coarse stage of a grid-sequenced chain is
skipped) and writes one row per (mode, method, seed) to ``results/seed_check.csv``:

* mode: ``pos`` (positivity on) or ``nopos`` (``--no-positivity``), from ``config.json``
* method: ``gs`` (``init_from`` set) or ``cold`` (trained from scratch on 100x50)
* seed: ``config.json`` seed. Seed 0 is the earlier single-seed run: it is listed (``seed`` 0) but the verdicts use
  seeds 1-3 only
* ``cl_after_transfer`` (gs only), ``final_cl``, ``cp_rms`` (last round vs FVM 800x400, the section 5.5 metric),
  ``rounds`` logged, ``failed_rounds``, ``seconds`` (cumulative training time), ``collapsed`` (|final Cl| < 0.1)
* ``complete``: logged + failed rounds reach the requested ``rounds``. A run that is still training is listed with
  ``complete = no`` and is not counted in the verdicts (the verdict then says "pending").

The verdict lines are plain language, one pair per mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import metrics  # noqa: E402

CSV_COLUMNS = ["mode", "method", "seed", "run_id", "cl_after_transfer", "final_cl", "cp_rms", "rounds",
               "failed_rounds", "seconds", "collapsed", "complete"]
GS_CL_MIN = 0.49          # grid sequencing passes when the final Cl is above this for every seed
COLLAPSE_CL = 0.1         # a run collapsed when |final Cl| is below this
SEEDS = (1, 2, 3)
DEFAULT_REF = REPO / "NNfoil" / "FVM_Cp" / "NACA0012_800_400_Cp.dat"


def _load(run: Path):
    files = [run / n for n in ("config.json", "log.json", "snapshots.json")]
    if not all(f.exists() for f in files):
        return None
    try:
        return tuple(json.loads(f.read_text(encoding="utf8")) for f in files)
    except json.JSONDecodeError:  # a file caught mid-write by a running chain
        return None


def collect(runs_dir, ref_path=DEFAULT_REF) -> list[dict]:
    """One row per 100x50 NACA0012 run folder, sorted by mode, method, seed."""
    ref = metrics.load_fvm(ref_path)
    rows = []
    for run in sorted(Path(runs_dir).iterdir()):
        if not run.is_dir():
            continue
        loaded = _load(run)
        if loaded is None:
            continue
        cfg, log, snap = loaded
        if cfg.get("shape") != "NACA0012" or cfg.get("mesh") != "100x50":
            continue
        n_ok, n_failed = len(log["round"]), int(log.get("failed_rounds", 0))
        complete = n_ok + n_failed >= int(cfg["rounds"])
        if n_ok == 0 and not complete:
            continue  # not started yet
        row = {
            "mode": "pos" if cfg.get("positivity", True) else "nopos",
            "method": "gs" if cfg.get("init_from") else "cold",
            "seed": int(cfg.get("seed", 0)), "run_id": cfg.get("run_id", run.name),
            "cl_after_transfer": log.get("cl_after_transfer"),
            "final_cl": None, "cp_rms": None, "rounds": n_ok, "failed_rounds": n_failed,
            "seconds": round(log["time_s"][-1], 1) if n_ok else None,
            "collapsed": "yes" if n_ok == 0 else None, "complete": "yes" if complete else "no",
        }
        if n_ok:
            row["final_cl"] = log["cl"][-1]
            row["cp_rms"] = metrics.cp_error(snap["x"], snap["y"], snap["rounds"][-1]["cp"], ref)["rms"]
            row["collapsed"] = "yes" if abs(row["final_cl"]) < COLLAPSE_CL else "no"
        rows.append(row)
    rows.sort(key=lambda r: (r["mode"] != "nopos", r["method"] != "gs", r["seed"]))
    return rows


def verdicts(rows: list[dict]) -> list[str]:
    """Plain-language verdict lines per mode, from the complete runs with seeds 1-3 only."""
    n = len(SEEDS)
    lines = []
    for mode in ("nopos", "pos"):
        gs = [r for r in rows if r["mode"] == mode and r["method"] == "gs" and r["seed"] in SEEDS
              and r["complete"] == "yes"]
        cold = [r for r in rows if r["mode"] == mode and r["method"] == "cold" and r["seed"] in SEEDS
                and r["complete"] == "yes"]
        if any(r["final_cl"] is None or r["final_cl"] <= GS_CL_MIN for r in gs):
            gs_text = "no"
        elif len(gs) == n:
            gs_text = "yes"
        else:
            gs_text = f"pending ({len(gs)} of {n} seeds complete so far)"
        n_col = sum(r["collapsed"] == "yes" for r in cold)
        if n_col >= 1:
            cold_text = f"yes ({n_col} of {n})" if len(cold) == n else f"yes ({n_col} of {len(cold)} complete so far)"
        elif len(cold) == n:
            cold_text = f"no (0 of {n})"
        else:
            cold_text = f"pending ({len(cold)} of {n} seeds complete so far)"
        lines.append(f"[{mode}] grid sequencing gives Cl > {GS_CL_MIN} for all {n} seeds: {gs_text}")
        lines.append(f"[{mode}] cold start collapses in at least 1 of {n} seeds: {cold_text}")
    return lines


def write_csv(rows: list[dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def cell(k, v):
        if v is None:
            return ""
        if k in ("cp_rms", "final_cl", "cl_after_transfer"):
            return f"{v:.6f}"
        return v

    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(k, r[k]) for k in CSV_COLUMNS})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=str(REPO / "results" / "runs"))
    ap.add_argument("--ref", default=str(DEFAULT_REF), help="FVM 800x400 Cp file")
    ap.add_argument("--out", default=str(REPO / "results" / "seed_check.csv"))
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    rows = collect(a.runs, a.ref)
    write_csv(rows, a.out)
    print(f"wrote {a.out} ({len(rows)} rows)")
    for line in verdicts(rows):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
