"""Late-round drift analysis (Step 1): which logged quantity tracks the Cp error across training rounds?

    research/.venv/Scripts/python research/drift_analysis.py [--runs results/runs] [--out results/drift_step1.csv]

Reads only existing run folders (no training): ``log.json`` (per round: Cl, loss, loss_f, loss_b) and
``snapshots.json`` (wall Cp per round; ``cp_rms`` against the FVM 800x400 file is computed from it). For every run with at least ``MIN_ROUNDS``
rounds it computes these candidate **reference-free** signals per round, and the reference-based target
``cp_rms`` (the section 5.5 metric, used only to judge the signals, never as a stopping input):

* ``loss``, ``loss_f`` (PDE), ``loss_b`` (boundary: far field + wall slip + seam)
* ``d_cl``      |Cl_k - Cl_{k-1}|
* ``d_cp``      RMS(Cp_k - Cp_{k-1}) over the wall face centres

and writes, per run and signal, three numbers:

* ``rho_all``    Spearman rho between the signal and ``cp_rms`` over all rounds (rounds with a previous round for
                 the two ``d_*`` signals)
* ``rho_drift``  the same over the rounds from the best round (lowest ``cp_rms``) onward, the phase where the
                 answer gets worse. It needs at least 4 such rounds, else it is empty
* ``argmin_round`` the round where the signal is lowest, next to ``best_round`` (lowest ``cp_rms``)

A positive ``rho`` means the signal goes up when the error goes up, which is what a stopping rule needs. A negative
``rho`` means the signal says "better" while the answer gets worse.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import metrics  # noqa: E402

FVM_REF = REPO / "NNfoil" / "FVM_Cp" / "NACA0012_800_400_Cp.dat"
MIN_ROUNDS = 6
SIGNALS = ("loss", "loss_f", "loss_b", "d_cl", "d_cp")
OUT_COLUMNS = ["run_id", "mesh", "n_rounds", "best_round", "best_cp_rms", "last_cp_rms", "signal", "rho_all",
               "rho_drift", "argmin_round", "signal_at_best", "signal_at_last"]


def spearman(a, b) -> float | None:
    """Spearman rank correlation with average ranks for ties; None when it is undefined (n < 3 or a constant input)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or a.size != b.size:
        return None

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size)
        r[order] = np.arange(v.size, dtype=float)
        for u in np.unique(v):           # average the ranks of ties
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def load_run(run_dir: Path):
    """(config, rows) for a run folder, or None when ``config.json`` / ``log.json`` / ``snapshots.json`` are missing.

    ``cp_rms`` is computed here for every logged round with ``metrics.cp_error`` against the finest FVM file of the
    case (NACA0012 only; other shapes are skipped), the same definition as ``experiments.pinn_row``.
    """
    lg, sn, cf = run_dir / "log.json", run_dir / "snapshots.json", run_dir / "config.json"
    if not (lg.exists() and sn.exists() and cf.exists()):
        return None
    cfg = json.load(open(cf, encoding="utf8"))
    if cfg.get("shape") != "NACA0012" or abs(float(cfg.get("alpha_deg", 0)) - 4.0) > 1e-9:
        return None
    log = json.load(open(lg, encoding="utf8"))
    snap = json.load(open(sn, encoding="utf8"))
    x, y = np.asarray(snap["x"], float), np.asarray(snap["y"], float)
    by_round = {int(r["round"]): np.asarray(r["cp"], float) for r in snap["rounds"]}
    ref = metrics.load_fvm(FVM_REF)
    out = []
    for i, k in enumerate(log["round"]):
        k = int(k)
        if k not in by_round:
            continue
        err = metrics.cp_error(x, y, by_round[k], ref)
        out.append({"round": k, "cp_rms": err["rms"], "cl": float(log["cl"][i]), "loss": float(log["loss"][i]),
                    "loss_f": float(log["loss_f"][i]), "loss_b": float(log["loss_b"][i]), "cp": by_round[k]})
    out.sort(key=lambda d: d["round"])
    return cfg, out


def signals_by_round(rows):
    """List of dicts: round, cp_rms and every signal (the ``d_*`` ones are None for the first round)."""
    out = []
    for i, r in enumerate(rows):
        s = {"round": r["round"], "cp_rms": r["cp_rms"], "loss": r["loss"], "loss_f": r["loss_f"],
             "loss_b": r["loss_b"], "d_cl": None, "d_cp": None}
        if i > 0:
            p = rows[i - 1]
            s["d_cl"] = abs(r["cl"] - p["cl"])
            s["d_cp"] = float(np.sqrt(np.mean((r["cp"] - p["cp"]) ** 2)))
        out.append(s)
    return out


def analyse_run(run_id: str, cfg: dict, rows) -> list[dict]:
    sig = signals_by_round(rows)
    best = min(sig, key=lambda s: s["cp_rms"])
    out = []
    for name in SIGNALS:
        use = [s for s in sig if s[name] is not None]
        x = [s[name] for s in use]
        y = [s["cp_rms"] for s in use]
        drift = [s for s in use if s["round"] >= best["round"]]
        rho_d = spearman([s[name] for s in drift], [s["cp_rms"] for s in drift]) if len(drift) >= 4 else None
        amin = min(use, key=lambda s: s[name])
        at_best = next((s[name] for s in use if s["round"] == best["round"]), None)
        out.append({"run_id": run_id, "mesh": cfg.get("mesh", ""), "n_rounds": len(rows),
                    "best_round": best["round"], "best_cp_rms": round(best["cp_rms"], 5),
                    "last_cp_rms": round(sig[-1]["cp_rms"], 5), "signal": name,
                    "rho_all": _r(spearman(x, y)), "rho_drift": _r(rho_d), "argmin_round": amin["round"],
                    "signal_at_best": at_best, "signal_at_last": sig[-1][name]})
    return out


def _r(v):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else round(v, 3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=str(REPO / "results" / "runs"))
    ap.add_argument("--out", default=str(REPO / "results" / "drift_step1.csv"))
    args = ap.parse_args(argv)
    table = []
    for d in sorted(Path(args.runs).iterdir()):
        if not d.is_dir():
            continue
        loaded = load_run(d)
        if loaded is None or len(loaded[1]) < MIN_ROUNDS:
            continue
        table.extend(analyse_run(d.name, loaded[0], loaded[1]))
    with open(args.out, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        w.writerows(table)
    runs = sorted({t["run_id"] for t in table})
    print(f"{len(runs)} runs with >= {MIN_ROUNDS} rounds -> {args.out}")
    for sname in SIGNALS:
        print(f"\n{sname}:  rho(all rounds)  rho(from best round on)  argmin vs best round")
        for rid in runs:
            t = next(x for x in table if x["run_id"] == rid and x["signal"] == sname)
            print(f"  {rid:42s} {str(t['rho_all']):>7s} {str(t['rho_drift']):>9s}   r{t['argmin_round']:<3d} vs r{t['best_round']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
