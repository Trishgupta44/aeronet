"""Milestone 1 figures: PINN vs FVM Cp (50x25 and 100x50) and the per-round trajectories.

    research/.venv/Scripts/python research/make_m1_figures.py [--runs results/runs] [--out results/figures]

Reads the committed run folders (``snapshots.json``, ``rounds.csv``, ``log.json``), the FVM files in
``NNfoil/FVM_Cp`` and ``results/summary.csv`` (for the same-mesh FVM RMS and the reference Cl). Writes
``m1_cp_50x25.png``, ``m1_cp_100x50.png`` and ``m1_trajectory.png``. Nothing is hard-coded except the run names.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import metrics  # noqa: E402

FVM_DIR = REPO / "NNfoil" / "FVM_Cp"

# (run folder, legend label, colour, line style, mesh)
RUNS = [
    ("naca0012_50x25_a4_m04", "PINN 50x25 (positivity on)", "tab:blue", "-", "50x25"),
    ("naca0012_50x25_a4_m04_nopos", "PINN 50x25 (positivity off)", "tab:orange", "-", "50x25"),
    ("naca0012_100x50_a4_m04", "PINN 100x50 plain (positivity on)", "tab:green", "-", "100x50"),
    ("naca0012_100x50_a4_m04_gs_nopos", "PINN 100x50 grid-seq. (positivity off)", "tab:red", "-", "100x50"),
    ("naca0012_100x50_a4_m04_gs", "PINN 100x50 grid-seq. (positivity on)", "tab:purple", "-", "100x50"),
]
_LABEL = {r[0]: r for r in RUNS}


def _final_cp(run_dir: Path):
    snap = json.loads((run_dir / "snapshots.json").read_text(encoding="utf8"))
    return snap["x"], snap["rounds"][-1]["cp"], snap["rounds"][-1]["round"]


def _read_rounds(run_dir: Path) -> dict:
    with (run_dir / "rounds.csv").open(newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def _summary_value(summary: Path, method: str, mesh: str, col: str) -> float:
    """B0 FVM row (reference: FVM 800x400) for this mesh, e.g. its rms_cp."""
    with summary.open(newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["experiment"] == "B0" and r["method"] == method and r["mesh"] == mesh and r["case"] == "NACA0012_a4":
                return float(r[col])
    raise SystemExit(f"{summary}: no B0 {method} row for NACA0012_a4 {mesh}")


def _cp_figure(out: Path, runs_dir: Path, mesh: str, names: list[str], summary: Path) -> Path:
    n_xi, n_eta = mesh.split("x")
    ref800 = metrics.load_fvm(FVM_DIR / "NACA0012_800_400_Cp.dat")
    same = metrics.load_fvm(FVM_DIR / f"NACA0012_{n_xi}_{n_eta}_Cp.dat")
    fvm_rms = _summary_value(summary, "fvm", mesh, "rms_cp")
    fvm_cl = _summary_value(summary, "fvm", mesh, "cl")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ref800[:, 0], ref800[:, 2], "o", ms=2.5, color="0.55", label="FVM 800x400 (reference)", zorder=1)
    ax.plot(same[:, 0], same[:, 2], "s", ms=4, mfc="none", color="black",
            label=f"FVM {mesh} (RMS {fvm_rms:.3f}, Cl {fvm_cl:.3f})", zorder=2)
    for name in names:
        _, label, colour, style, _ = _LABEL[name]
        x, cp, last = _final_cp(runs_dir / name)
        err = metrics.cp_error(x, json.loads((runs_dir / name / "snapshots.json").read_text(encoding="utf8"))["y"],
                               cp, ref800)
        ax.plot(x, cp, style, lw=1.5, color=colour, label=f"{label}, round {last} (RMS {err['rms']:.3f})", zorder=3)
    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel("Cp")
    ax.set_title(f"NACA 0012, Mach 0.4, alpha 4 deg, {mesh} mesh: final-round Cp"
                 " (RMS = Cp error vs FVM 800x400)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = out / f"m1_cp_{mesh}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _trajectory_figure(out: Path, runs_dir: Path, summary: Path) -> Path:
    cl_ref = _summary_value(summary, "fvm", "50x25", "cl_ref")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, label, colour, style, _ in RUNS:
        d = _read_rounds(runs_dir / name)
        a1.plot(d["round"], d["cp_rms"], style, marker="o", ms=3, color=colour, label=label)
        a2.plot(d["round"], d["cl"], style, marker="o", ms=3, color=colour, label=label)
        cfg = json.loads((runs_dir / name / "config.json").read_text(encoding="utf8"))
        if cfg.get("init_from"):
            log = json.loads((runs_dir / name / "log.json").read_text(encoding="utf8"))
            a2.plot([0], [log["cl_after_transfer"]], marker="D", ms=6, mfc="none", color=colour, ls="none")
    for mesh, ls in (("50x25", ":"), ("100x50", "--")):
        v = _summary_value(summary, "fvm", mesh, "rms_cp")
        a1.axhline(v, color="black", ls=ls, lw=1.0, label=f"FVM {mesh} on its own mesh: RMS {v:.3f}")
        c = _summary_value(summary, "fvm", mesh, "cl")
        a2.axhline(c, color="black", ls=ls, lw=1.0, label=f"FVM {mesh}: Cl {c:.4f}")
    a2.axhline(cl_ref, color="tab:gray", ls="-", lw=1.5, label=f"FVM 800x400: Cl {cl_ref:.4f}")
    a1.set_yscale("log")
    a1.set_xlabel("training round")
    a1.set_ylabel("Cp RMS error vs FVM 800x400 (log scale)")
    a1.set_title("Cp error by round")
    a2.set_xlabel("training round (diamonds, round 0: Cl right after the grid transfer)")
    a2.set_ylabel("Cl")
    a2.set_title("Lift coefficient by round")
    for a in (a1, a2):
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=7)
    fig.suptitle("NACA 0012, Mach 0.4, alpha 4 deg, seed 0: every round of the five runs")
    fig.tight_layout()
    path = out / "m1_trajectory.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_figures(runs_dir, out_dir, summary) -> list[Path]:
    runs_dir, out_dir, summary = Path(runs_dir), Path(out_dir), Path(summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        _cp_figure(out_dir, runs_dir, "50x25", ["naca0012_50x25_a4_m04", "naca0012_50x25_a4_m04_nopos"], summary),
        _cp_figure(out_dir, runs_dir, "100x50", ["naca0012_100x50_a4_m04", "naca0012_100x50_a4_m04_gs_nopos"],
                   summary),
        _trajectory_figure(out_dir, runs_dir, summary),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=str(REPO / "results" / "runs"))
    ap.add_argument("--out", default=str(REPO / "results" / "figures"))
    ap.add_argument("--summary", default=str(REPO / "results" / "summary.csv"))
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    for p in make_figures(a.runs, a.out, a.summary):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
