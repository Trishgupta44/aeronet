"""Milestone 3 figures (``m3_*.png``), all drawn from the files in ``results/`` (no result number is typed here).

    research/.venv/Scripts/python research/make_figures.py [--out results/figures] [--only e1 e2 ...]

Inputs: ``results/summary.csv``, ``e1_rounds.csv``, ``e4_timing.json``, ``e6_replay.csv``, ``seed_check.csv``, the
committed run folders (``log.json`` and ``snapshots.json``) and the bundled FVM files. The cylinder figure also solves the
panel method (it is a computation, not a stored number). A missing input raises an error that names it: nothing is
skipped silently. ``python research/make_figures.py`` prints one ``saved`` line per figure.

Every figure carries a source line and the label of each series (measured, quoted or fitted). Positivity on is blue,
off is orange; line styles and marker shapes differ as well, so the figures stay readable in greyscale.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (HERE, REPO / "app" / "backend"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import metrics  # noqa: E402
import results_io as rio  # noqa: E402
from results_io import num, pick, select  # noqa: E402

DPI = 300
POS, NOPOS = "tab:blue", "tab:orange"
SEED_STYLE = {0: "-", 1: "--", 2: ":", 3: "-.", 4: (0, (5, 1)), 5: (0, (1, 1))}
NAMES = ["e1", "e2", "e3", "e4", "e5", "e6", "seed_check"]
FILES = {"e1": "m3_e1_trajectories.png", "e2": "m3_e2_mesh_study.png", "e3": "m3_e3_polar.png",
         "e4": "m3_e4_speed.png", "e5": "m3_e5_cylinder.png", "e6": "m3_e6_stop_rule.png",
         "seed_check": "m3_seed_check.png"}


def _finish(fig, out: Path, name: str, source: str) -> Path:
    fig.text(0.005, 0.001, textwrap.fill(source, 170), fontsize=6.5, color="0.35", ha="left", va="bottom")
    path = Path(out) / FILES[name]
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _summary(path=None):
    return rio.read_summary(path)


def _fvm_same_mesh(s, mesh, col):
    return num(pick(select(s, experiment="B0"), case="NACA0012_a4", method="fvm", mesh=mesh)[col])


# ---------------------------------------------------------------- E1
def fig_e1(out, summary=None) -> Path:
    s = _summary(summary)
    traj = rio.read_csv(rio.RESULTS / "e1_rounds.csv")
    rms_fvm = _fvm_same_mesh(s, "50x25", "rms_cp")
    cl_ref = _fvm_same_mesh(s, "50x25", "cl_ref")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    final, best = {}, {}
    for r in select(s, experiment="E1"):
        seed = int(re.search(r"_s(\d+)$", r["case"]).group(1))
        pos = r["method"] == "pinn"
        mine = sorted((t for t in traj if t["case"] == r["case"] and t["method"] == r["method"]),
                      key=lambda t: int(t["round"]))
        rounds = [int(t["round"]) for t in mine]
        rms = [num(t["cp_rms"]) for t in mine]
        cl = [num(t["cl"]) for t in mine]
        kw = dict(color=POS if pos else NOPOS, ls=SEED_STYLE[seed], marker="o" if pos else "s", ms=3, lw=1.3,
                  label=f"positivity {'on' if pos else 'off'}, seed {seed}")
        a1.plot(rounds, rms, **kw)
        a2.plot(rounds, cl, **kw)
        final[(pos, seed)], best[(pos, seed)] = rms[-1], min(rms)
    a1.axhline(rms_fvm, color="black", ls=(0, (1, 2)), lw=1.4, label=f"FVM 50x25 on its own mesh: RMS {rms_fvm:.3f}")
    a2.axhline(cl_ref, color="0.4", lw=1.6, label=f"FVM 800x400 reference: Cl {cl_ref:.4f}")
    for a in (a1, a2):
        a.axvspan(3, 5, color="0.85", zorder=0)
        a.set_xlabel("training round (grey band: rounds 3-5)")
        a.grid(alpha=0.3, which="both")
    a1.set_yscale("log")
    a1.set_ylabel("Cp RMS error vs FVM 800x400 (log scale)")
    a2.set_ylabel("Cl")
    a1.set_title("Cp error by round")
    a2.set_title("Lift coefficient by round")
    h, lab = a1.get_legend_handles_labels()
    h2, lab2 = a2.get_legend_handles_labels()
    fig.legend(h + h2[-1:], lab + lab2[-1:], fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.085))
    degraded = {p: sum(1 for k in final if k[0] == p and final[k] > 2 * best[k]) for p in (True, False)}
    mean_final = {p: np.mean([v for k, v in final.items() if k[0] == p]) for p in (True, False)}
    txt = (f"Neither setting wins.\nFinal-round RMS, mean of 3 seeds: on {mean_final[True]:.3f}, off {mean_final[False]:.3f}.\n"
           f"Runs ending above 2x their best RMS: on {degraded[True]} of 3, off {degraded[False]} of 3.")
    fig.text(0.5, 0.05, txt.replace(chr(10), "  "), ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    fig.suptitle("E1, NACA 0012, Mach 0.4, 4 deg, 50x25 mesh, positivity on/off x seeds 0-2: accuracy is reached at rounds 3-5, "
                 "then 5 of 6 runs degrade", fontsize=10)
    fig.tight_layout(rect=(0, 0.18, 1, 0.95))
    return _finish(fig, out, "e1", "Source (measured): results/e1_rounds.csv, results/summary.csv (E1, B0). Mach 0.4 and 4 deg are "
                   "the FVM files' condition (quoted from arXiv 2401.08705 sec 3.2).")


# ---------------------------------------------------------------- E2
def fig_e2(out, summary=None) -> Path:
    s = _summary(summary)
    e2 = select(s, experiment="E2")
    meshes = ["50x25", "100x50", "200x100", "400x200"]
    fvm = {m: num(pick(e2, method="fvm", mesh=m)["rms_cp"]) for m in meshes}
    panel = pick(e2, method="panel")
    style = {"pinn": ("tab:blue", "o", "PINN, positivity on, cold start"),
             "pinn_nopos": ("tab:orange", "s", "PINN, positivity off, cold start"),
             "pinn_gs": ("tab:blue", "^", "PINN grid-sequenced, positivity on"),
             "pinn_gs_nopos": ("tab:orange", "v", "PINN grid-sequenced, positivity off")}
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    xs = np.arange(len(meshes))
    ax.bar(xs - 0.2, [fvm[m] for m in meshes], width=0.3, color="0.65", edgecolor="black", hatch="//",
           label="FVM on its own mesh (measured)", zorder=2)
    for x, m in zip(xs, meshes):
        ax.text(x - 0.2, fvm[m] * 1.08, f"{fvm[m]:.4f}", ha="center", fontsize=7)
    seen = set()
    for i, m in enumerate(meshes):
        pr = [r for r in e2 if r["method"].startswith("pinn") and r["mesh"] == m]
        offs = np.linspace(0.0, 0.34, max(len(pr), 2))[: len(pr)] + 0.02 if len(pr) > 1 else [0.17]
        for off, r in zip(offs, pr):
            c, mk, lab = style[r["method"]]
            ax.plot(i + off, num(r["rms_cp"]), mk, color=c, ms=9, mec="black", zorder=3,
                    label=None if lab in seen else lab)
            seen.add(lab)
            ax.text(i + off + 0.03, num(r["rms_cp"]) * 1.1, f"{num(r['rms_cp']):.4f}", fontsize=7, rotation=40)
    ax.axhline(num(panel["rms_cp"]), color="tab:green", ls="--", lw=1.4,
               label=f"panel method, {panel['mesh']} (measured): {num(panel['rms_cp']):.4f}")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(meshes)
    ax.set_xlabel("mesh (the PINN is trained on that mesh; FVM is the solver on that mesh)")
    ax.set_ylabel("Cp RMS error vs FVM 800x400 (log scale; lower is better)")
    ax.set_ylim(0.004, 0.5)
    ax.text(3.17, 0.0072, "no PINN\nrun at 400x200", ha="center", fontsize=8, color="0.3")
    ax.grid(alpha=0.3, which="both", axis="y")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("E2, NACA 0012, Mach 0.4, 4 deg: PINN vs FVM on the same mesh (PINN markers: final round, seed 0)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _finish(fig, out, "e2", "Source (measured): results/summary.csv, experiment E2. Reference: FVM 800x400 file, "
                   "section 5.5 metric. 200x100 training time withheld (laptop slept); 400x200 PINN not run.")


# ---------------------------------------------------------------- E3
def fig_e3(out, summary=None) -> Path:
    s = _summary(summary)
    e3 = select(s, experiment="E3")
    alphas = sorted({int(num(r["alpha_deg"])) for r in e3})

    def series(method):
        return np.array([num(pick(e3, method=method, alpha_deg=a)["cl"]) for a in alphas])

    fv, pn, th = series("fvm"), series("panel"), series("thin_airfoil")
    fig, (ax, ad) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    ax.plot(alphas, fv, "o-", color="black", ms=4, lw=1.2, label="FVM 800x400 (reference; Cl integrated from the Cp files)")
    ax.plot(alphas, pn, "--", color="tab:green", lw=1.5, label="panel method + Prandtl-Glauert (measured)")
    ax.plot(alphas, th, ":", color="0.4", lw=1.8, label="thin-airfoil theory (measured)")
    ad.plot(alphas, pn - fv, "--", color="tab:green", lw=1.5)
    ad.plot(alphas, th - fv, ":", color="0.4", lw=1.8)
    ad.axhline(0, color="black", lw=0.8)

    spec = {("pinn", "NACA2412_a4_s1"): ("tab:blue", "D", "PINN seed 1, positivity on (retry at 4 deg)", -0.25),
            ("pinn_nopos", None): ("tab:orange", "s", "PINN seed 0, positivity off (retry at 4 deg)", 0.25)}
    lo = -0.16
    ad.set_ylim(lo, 0.06)
    used = set()
    for r in e3:
        if not r["method"].startswith("pinn"):
            continue
        a, cl = int(num(r["alpha_deg"])), num(r["cl"])
        ref = num(r["cl_ref"])
        collapsed = abs(cl) < 0.1 < abs(ref)
        if collapsed:
            ax.plot(a, cl, "X", color="red", ms=13, mec="black", zorder=5, label="collapsed (seed 0, positivity on)")
            ax.annotate("collapsed (seed 0)", (a, cl), xytext=(a + 1.5, cl - 0.55), fontsize=9, color="red",
                        arrowprops=dict(arrowstyle="->", color="red"))
            ad.annotate(f"collapsed: {cl - ref:+.2f}\n(off scale)", (a, lo), xytext=(a - 4.3, lo + 0.045), fontsize=8,
                        color="red", arrowprops=dict(arrowstyle="->", color="red"))
            ad.plot(a, lo, "v", color="red", ms=8, clip_on=False)
            continue
        key = ("pinn_nopos", None) if r["method"] == "pinn_nopos" else ("pinn", r["case"] if r["case"].endswith("_s1") else None)
        if key[0] == "pinn" and key[1] is None:
            c, mk, lab, dx = "tab:blue", "o", "PINN seed 0, positivity on (10 rounds, 100x50)", 0.0
        else:
            c, mk, lab, dx = spec[key]
        ax.plot(a + dx, cl, mk, color=c, ms=8, mec="black", zorder=4, label=None if lab in used else lab)
        ad.plot(a + dx, cl - ref, mk, color=c, ms=7, mec="black", zorder=4)
        used.add(lab)
    ax.set_ylabel("Cl")
    ad.set_ylabel("Cl minus FVM")
    ad.set_xlabel("angle of attack (deg)  [the two 4-degree retries are drawn 0.25 deg apart for legibility]")
    for a_ in (ax, ad):
        a_.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("E3, NACA 2412, Mach 0.3 (QUOTED from the paper, arXiv 2401.08705 sec 3.4): Cl vs angle of attack", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _finish(fig, out, "e3", "Source: results/summary.csv, experiment E3 (FVM, panel, thin-airfoil, PINN: measured). Mach number "
                   "quoted, not stored in the Cp files (the panel-method fit preferred about 0.25, fitted).")


# ---------------------------------------------------------------- E4
def fig_e4(out, summary=None) -> Path:
    s = _summary(summary)
    t = rio.read_json(rio.RESULTS / "e4_timing.json")
    e4 = select(s, experiment="E4")
    by_mesh: dict[str, list[dict]] = {}
    for run, v in t["pinn"].items():
        by_mesh.setdefault(re.search(r"_(\d+x\d+)_", run).group(1), []).append(v["threads"])
    bars = []  # (label, ms, kind, lo, hi)
    for mesh in ("50x25", "100x50", "200x100"):
        if mesh not in by_mesh:
            raise LookupError(f"results/e4_timing.json has no PINN timing for {mesh}")
        n = len(by_mesh[mesh])
        for th, kind in (("8", "pinn8"), ("1", "pinn1")):
            vals = [r[th]["field"]["mean"] for r in by_mesh[mesh]]
            bars.append((f"PINN full field, {mesh}, {th} thread{'s' if th == '8' else ''}  (mean of {n} trained run{'s' if n > 1 else ''})",
                         float(np.mean(vals)), kind, min(vals), max(vals)))
    bars.append(("Panel method: solve() only, 140 panels", t["panel"]["solve"]["mean"], "panel", None, None))
    bars.append(("Panel method: solve() + 170-wide field grid", t["panel"]["solve_field170"]["mean"], "panel", None, None))
    bars.append(("NeuralFoil 0.3.3 (viscous, Re 6e6: a different problem)", t["neuralfoil"]["time"]["mean"], "nf", None, None))
    for r in select(e4, method="fvm", source="quoted"):
        bars.append((f"FVM, {r['mesh']}: one solve (QUOTED from the paper, another machine)", num(r["infer_ms"]), "fvm", None, None))
    color = {"pinn8": "tab:blue", "pinn1": "lightsteelblue", "panel": "tab:green", "nf": "0.8", "fvm": "0.8"}
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ys = np.arange(len(bars))[::-1]
    for y, (lab, v, kind, lo, hi) in zip(ys, bars):
        hatch = "//" if kind in ("nf", "fvm") else None
        ax.barh(y, v, color=color[kind], edgecolor="black", hatch=hatch, height=0.72)
        if lo is not None and hi > lo:
            ax.plot([lo, hi], [y, y], color="black", lw=1.5)
        ax.text(v * 1.12, y, f"{v:,.2f} ms" if v < 100 else f"{v:,.0f} ms", va="center", fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 4e6)
    ax.set_xlabel("time per query, milliseconds (log scale)")
    ax.grid(alpha=0.3, which="both", axis="x")
    ax.legend(handles=[Patch(fc="tab:blue", ec="black", label="PINN, trained network evaluated (measured, 8 threads)"),
                       Patch(fc="lightsteelblue", ec="black", label="PINN, 1 thread (measured)"),
                       Patch(fc="tab:green", ec="black", label="panel method, no training (measured)"),
                       Patch(fc="0.8", ec="black", hatch="//", label="hatched: NOT comparable (viscous model, or quoted from another machine)")],
              fontsize=7.5, loc="center right")
    ax.set_title("E4: inference / query time, not a like-for-like comparison\n"
                 "(a PINN must be trained once per condition first; the panel method needs no training)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _finish(fig, out, "e4", "Source: results/e4_timing.json and results/summary.csv, experiment E4. Measured: this laptop, AMD Ryzen 9 6900HS, "
                   "100 calls after warm-up (black line = range over runs on the same mesh). Quoted: arXiv 2401.08705 Table 1.")


# ---------------------------------------------------------------- E5
def _theta(x, y):
    """Angle about the circle centre (0.5, 0) in degrees, 0 at the rear point, 180 at the front stagnation point."""
    return np.degrees(np.arctan2(np.asarray(y, float), np.asarray(x, float) - 0.5)) % 360.0


def fig_e5(out, summary=None) -> Path:
    import experiments
    import runs

    s = _summary(summary)
    e5 = select(s, experiment="E5")
    ref = metrics.load_fvm(rio.require(rio.REF_CYL))
    th_ref = _theta(ref[:, 0], ref[:, 1])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(th_ref, ref[:, 2], "o", ms=2.2, color="0.55", label=f"FVM 800x400 (reference, measured; Cp min {ref[:, 2].min():.2f})", zorder=1)
    xw, yw = experiments.cylinder_wall(experiments.E5_PANELS)
    for mach, col, ls, lab in ((experiments.E5_MACH, "tab:green", "--", "panel + Prandtl-Glauert, Mach 0.4"),
                               (0.0, "0.35", ":", "incompressible circle (panel, Mach 0)")):
        sol = experiments.panel_nonlifting(xw + 0.5, yw, 0.0, mach)
        order = np.argsort(_theta(sol["cx"], sol["cy"]))
        rms = num(pick(select(e5, method="panel"), mach=mach)["rms_cp"])
        ax.plot(_theta(sol["cx"], sol["cy"])[order], sol["cp"][order], ls, color=col, lw=1.7,
                label=f"{lab} (measured; RMS {rms:.3f}; Cp min {sol['cp'].min():.2f})")
    for method, col, ls, lab in (("pinn", POS, "-", "PINN 50x25, positivity on"), ("pinn_nopos", NOPOS, "-.", "PINN 50x25, positivity off")):
        run = runs.load_run(runs.RUNS / experiments.e5_specs()[0 if method == "pinn" else 1]["name"])
        x, y, cp = runs.final_surface(run)
        order = np.argsort(_theta(x, y))
        rms = num(pick(e5, method=method)["rms_cp"])
        ax.plot(_theta(x, y)[order], cp[order], ls, color=col, lw=1.8,
                label=f"{lab}, round {run['snapshots']['rounds'][-1]['round']} (measured; RMS {rms:.3f}; Cp min {cp.min():.2f})")
    ax.invert_yaxis()
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 45))
    ax.set_xlabel("angle about the cylinder centre, degrees (0 = rear point, 180 = front stagnation point)")
    ax.set_ylabel("Cp  (axis inverted: suction up)")
    ax.grid(alpha=0.3)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.045))
    ax.set_title("E5, cylinder, Mach 0.4, 0 deg: Cp around the circle (RMS = Cp error vs FVM 800x400)", fontsize=10)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    return _finish(fig, out, "e5", "Source (measured): results/summary.csv experiment E5, results/runs/cylinder_50x25_a0_m04[_nopos]/snapshots.json, "
                   "NNfoil/FVM_Cp/Cylinder_800_400_Cp.dat; panel curves recomputed by research/experiments.py. Mach 0.4 quoted (arXiv sec 3.1).")


# ---------------------------------------------------------------- E6
def fig_e6(out, summary=None) -> Path:
    d = rio.read_csv(rio.RESULTS / "e6_replay.csv")
    runs_dir = rio.RESULTS / "runs"
    drifted = [r for r in d if str(r["drifted"]).lower() == "true"]
    stopped = [r for r in d if r["stop_round"] != "never" and r["shape"] == "NACA0012" and r["mesh"] == "50x25"]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axvspan(3, 5, color="0.88", zorder=0)
    ax.text(4, 0.92, "fixed early stop\nat rounds 3-5", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="0.25")
    for r in drifted + stopped:
        if r["shape"] != "NACA0012":
            raise ValueError(f"{r['run_id']}: the E6 figure uses the NACA 0012 reference only")
        ser = rio.round_series(runs_dir / r["run_id"], rio.REF_0012)
        pos = str(r["positivity"]).lower() == "true"
        is_drift = r in drifted
        c = POS if pos else NOPOS
        lab = f"positivity {'on' if pos else 'off'}, seed {int(num(r['seed']))}" + ("" if is_drift else " (the one run the rule stopped; not a drifter)")
        ax.plot(ser["round"], ser["cp_rms"], ls=SEED_STYLE[int(num(r["seed"]))], color=c if is_drift else "0.45",
                marker="o" if pos else "s", ms=3, lw=1.2 if is_drift else 1.0, alpha=1.0 if is_drift else 0.8, label=lab)
        if is_drift:
            ax.plot(ser["round"][-1], ser["cp_rms"][-1], marker=">", ms=9, mfc="white", mec=c, mew=1.6, ls="none", zorder=4)
            if r["jump_round"]:
                j = int(float(r["jump_round"]))
                ax.plot(j, ser["cp_rms"][ser["round"].index(j)], "x", ms=8, mew=2, color="black", zorder=5)
        else:
            sr = int(float(r["stop_round"]))
            ax.plot(sr, ser["cp_rms"][ser["round"].index(sr)], marker="*", ms=16, color="gold", mec="black", zorder=6)
    n_dr = len(drifted)
    n_never = sum(1 for r in drifted if r["stop_round"] == "never")
    ax.plot([], [], marker=">", ms=9, mfc="white", mec="black", ls="none", label="end of run: plateau rule never fired")
    ax.plot([], [], "x", ms=8, mew=2, color="black", label="round where the error jumps (replay file)")
    ax.plot([], [], marker="*", ms=14, color="gold", mec="black", ls="none", label="round where the plateau rule would stop")
    ax.set_yscale("log")
    ax.set_xlabel("training round")
    ax.set_ylabel("Cp RMS error vs FVM 800x400 (log scale)")
    ax.grid(alpha=0.3, which="both")
    fig.legend(*ax.get_legend_handles_labels(), fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.115))
    fig.text(0.5, 0.075, f"Plateau rule (|dCl| < 0.002, dCp < 0.005, 2 rounds): never fired in {n_never} of {n_dr} drifted runs.\n"
            "Thresholds were set after looking at the E1 runs; the fixed early stop is the practice.",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_title("E6, NACA 0012, Mach 0.4, 4 deg, 50x25: the reference-free plateau stop rule does not catch the drift", fontsize=10)
    fig.tight_layout(rect=(0, 0.27, 1, 1))
    return _finish(fig, out, "e6", "Source (measured): results/e6_replay.csv and results/runs/<run>/{log,snapshots}.json (Cp RMS recomputed with research/metrics.py). "
                   "Eight drifted runs: five E1 (development) and three new held-out seeds.")


# ---------------------------------------------------------------- seed check
def fig_seed_check(out, summary=None) -> Path:
    import seed_check

    d = rio.read_csv(rio.RESULTS / "seed_check.csv")
    s = _summary(summary)
    cl_ref = _fvm_same_mesh(s, "100x50", "cl_ref")
    fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, mode, colour, title in ((axs[0], "nopos", NOPOS, "positivity off (upstream behaviour)"),
                                    (axs[1], "pos", POS, "positivity on")):
        for method, mk, dx, lab in (("gs", "o", -0.12, "grid-sequenced (50x25, then 100x50)"), ("cold", "s", 0.12, "cold start on 100x50")):
            for r in (r for r in d if r["mode"] == mode and r["method"] == method):
                seed = int(r["seed"])
                if r["collapsed"] == "yes" or num(r["final_cl"]) is None:
                    ax.plot(seed + dx, 0.39, "X", color="red", ms=14, mec="black", clip_on=False, zorder=5)
                    ax.annotate(f"collapsed:\nall {r['failed_rounds']} rounds failed,\nno Cl produced", (seed + dx, 0.39),
                                xytext=(seed - 1.1, 0.405), fontsize=7.5, color="red")
                else:
                    ax.plot(seed + dx, num(r["final_cl"]), mk, color=colour, ms=9, mec="black", zorder=3,
                            mfc=colour if method == "gs" else "white", label=lab)
        ax.axhline(cl_ref, color="0.4", lw=1.4, label=f"FVM 800x400 reference: Cl {cl_ref:.4f}")
        ax.axhline(seed_check.GS_CL_MIN, color="tab:red", ls=":", lw=1.2, label=f"pass line for grid sequencing: Cl > {seed_check.GS_CL_MIN}")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlim(-0.7, 3.7)
        ax.set_xlabel("seed")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        h, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, h))
        ax.legend(uniq.values(), uniq.keys(), fontsize=7, loc="center left")
    axs[0].set_ylabel("final Cl (NACA 0012, Mach 0.4, 4 deg, 100x50)")
    axs[0].set_ylim(0.37, 0.53)
    cold = {(r["mode"], r["seed"]): r["rounds"] for r in d if r["method"] == "cold"}
    longer = [f"seed {k[1]} positivity {'on' if k[0] == 'pos' else 'off'}: {v} rounds" for k, v in cold.items() if int(v) > 5]
    note = ("Grid-sequenced runs: 4 rounds on 100x50 after the 50x25 stage. Cold starts: 5 rounds"
            + (f" (except {', '.join(longer)})" if longer else "") + ". Seed 0 cold start without positivity was not run.")
    fig.suptitle("Seed check: cold-start collapse in 1 of 3 seeds without positivity; grid sequencing reaches Cl 0.50-0.52 in all seeds", fontsize=10)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.text(0.005, 0.07, note, fontsize=7, ha="left")
    return _finish(fig, out, "seed_check", "Source (measured): results/seed_check.csv. Cl reference from results/summary.csv (B0). A collapsed run has no Cl: its "
                   "marker is drawn near the bottom edge only as a flag, not at a Cl value.")


FIGS = {"e1": fig_e1, "e2": fig_e2, "e3": fig_e3, "e4": fig_e4, "e5": fig_e5, "e6": fig_e6, "seed_check": fig_seed_check}


def make_figures(out=None, summary=None, only=None) -> list[Path]:
    out = Path(out) if out is not None else rio.RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return [FIGS[n](out, summary) for n in (only or NAMES)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(rio.RESULTS / "figures"))
    ap.add_argument("--summary", default=None)
    ap.add_argument("--only", nargs="*", choices=NAMES)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    for p in make_figures(a.out, a.summary, a.only):
        print(f"saved {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
