"""Validate one NNfoil run folder against an FVM Cp file: metrics row in summary.csv plus a Cp figure.

    research/.venv/Scripts/python research/validate.py results/runs/<id> --ref NNfoil/FVM_Cp/NACA0012_800_400_Cp.dat \
        --experiment E2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "app" / "backend"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import metrics  # noqa: E402

_NACA4 = re.compile(r"^NACA(\d{4})$")


def _num(v):
    """4.0 -> 4, 0.4 -> 0.4 (so csv cells and case names stay short)."""
    f = float(v)
    return int(f) if f == int(f) else f


def case_name(cfg: dict) -> str:
    """``<Shape>_a<alpha>[_s<seed>]`` (roadmap 5.4); seed 0 is the default and has no suffix."""
    name = f"{cfg['shape']}_a{float(cfg['alpha_deg']):g}"
    if cfg.get("seed"):
        name += f"_s{int(cfg['seed'])}"
    return name


def method_name(cfg: dict) -> str:
    return "pinn" if cfg.get("positivity", True) else "pinn_nopos"


def validate(run_dir, ref_path, experiment, summary="results/summary.csv", ref_label=None, notes=""):
    run = Path(run_dir)
    cfg = json.loads((run / "config.json").read_text(encoding="utf8"))
    log = json.loads((run / "log.json").read_text(encoding="utf8"))
    snap = json.loads((run / "snapshots.json").read_text(encoding="utf8"))
    if not log["round"]:
        raise SystemExit(f"{run}: no finished round in log.json")

    alpha, mach = float(cfg["alpha_deg"]), float(cfg["mach"])
    x, y = snap["x"], snap["y"]
    cp = snap["rounds"][-1]["cp"]
    ref = metrics.load_fvm(ref_path)
    err = metrics.cp_error(x, y, cp, ref)
    cl_ref, _ = metrics.fvm_forces(ref_path, alpha)

    # figure: Cp vs x/c, inverted axis; panel line only for NACA 4-digit codes
    fig, ax = plt.subplots(figsize=(7, 4.6))
    label_ref = ref_label or "FVM reference"
    ax.plot(ref[:, 0], ref[:, 2], "o", ms=3, color="0.35", label=label_ref)
    m = _NACA4.match(cfg["shape"])
    if m:
        from aeronet import panel

        sol = panel.solve(m.group(1), alpha, mach, 100)
        ax.plot(sol.geom.cx, sol.cp, "-", lw=1.0, color="tab:green", label="panel method")
    ax.plot(x, cp, "-", lw=1.6, color="tab:red", label=f"PINN {cfg['mesh']}, round {log['round'][-1]}")
    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel("Cp")
    ax.set_title(f"{cfg['shape']}  Ma {mach:g}  alpha {alpha:g} deg   RMS {err['rms']:.4f}  max {err['max']:.4f}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run / "cp_vs_fvm.png", dpi=150)
    plt.close(fig)

    notes_out = "; ".join(s for s in ([f"ref: {ref_label}"] if ref_label else []) + ([notes] if notes else []))
    row = {
        "experiment": experiment, "case": case_name(cfg), "method": method_name(cfg), "mesh": cfg["mesh"],
        "mach": _num(mach), "alpha_deg": _num(alpha),
        "rms_cp": round(err["rms"], 6), "max_cp": round(err["max"], 6),
        "cl": round(log["cl"][-1], 6), "cl_ref": round(cl_ref, 6), "cd": round(log["cd"][-1], 6),
        "train_s": round(log["time_s"][-1], 1), "infer_ms": None, "source": "measured", "notes": notes_out,
    }
    metrics.append_summary(row, summary)
    return row, err


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--ref", required=True, help="FVM *_Cp.dat file")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--ref-label", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--summary", default="results/summary.csv")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    row, err = validate(a.run_dir, a.ref, a.experiment, a.summary, a.ref_label, a.notes)
    print(f"{row['case']} {row['method']} {row['mesh']}: RMS {err['rms']:.4f}  max {err['max']:.4f}  n {err['n']}  "
          f"Cl {row['cl']} (ref {row['cl_ref']})  Cd {row['cd']}  train {row['train_s']} s")
    print(f"wrote {Path(a.run_dir) / 'cp_vs_fvm.png'} and a row in {a.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
