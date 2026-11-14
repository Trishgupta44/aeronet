"""Block diagram and flowcharts for the report (boxes and arrows, matplotlib Agg): writes docs/deliverables_src/img/*.png.

    research/.venv/Scripts/python research/docs_build/make_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent.parent / "docs" / "deliverables_src" / "img"
NAVY, GREY, PALE, AMBER, GREEN = "#1b3a63", "#4a4a4a", "#e8eef7", "#fff3d6", "#e3f3e6"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, text, fc=PALE, ec=NAVY, size=8.5, bold=False, shape="box"):
    style = "round,pad=0.02,rounding_size=0.08" if shape == "box" else "round,pad=0.02,rounding_size=0.25"
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=style, fc=fc, ec=ec, lw=1.1))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color="black",
            fontweight="bold" if bold else None, linespacing=1.15)
    return (cx, cy, w, h)


def line(ax, pts, arrow_end=True):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] if arrow_end else xs, ys[:-1] if arrow_end else ys, color=GREY, lw=1.2, solid_capstyle="butt")
    if arrow_end:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2,
                                                                       shrinkA=0, shrinkB=0))
        ax.plot([pts[-2][0], pts[-1][0]], [pts[-2][1], pts[-1][1]], color=GREY, lw=1.2)


def down(ax, a, b, label=None, dx=0.0):
    (x1, y1, w1, h1), (x2, y2, w2, h2) = a, b
    line(ax, [(x1, y1 - h1 / 2), (x2, y2 + h2 / 2)])
    if label:
        ax.text((x1 + x2) / 2 + dx, (y1 - h1 / 2 + y2 + h2 / 2) / 2, label, fontsize=7.5, color=GREY, style="italic")


def right(ax, a, b, label=None, dy=0.12):
    (x1, y1, w1, h1), (x2, y2, w2, h2) = a, b
    line(ax, [(x1 + w1 / 2, y1), (x2 - w2 / 2, y2)])
    if label:
        ax.text((x1 + w1 / 2 + x2 - w2 / 2) / 2, y1 + dy, label, fontsize=7.5, color=GREY, style="italic", ha="center")


def left(ax, a, b, label=None, dy=0.12):
    (x1, y1, w1, h1), (x2, y2, w2, h2) = a, b
    line(ax, [(x1 - w1 / 2, y1), (x2 + w2 / 2, y2)])
    if label:
        ax.text((x1 - w1 / 2 + x2 + w2 / 2) / 2, y1 + dy, label, fontsize=7.5, color=GREY, style="italic", ha="center")


def system_architecture():
    fig, ax = canvas(6.3, 6.3)
    box(ax, 3.15, 6.05, 6.0, 0.4, "AeroNet system architecture (one laptop CPU, no GPU)", fc="white", ec="white",
        size=9.5, bold=True)
    fvm = box(ax, 1.0, 5.1, 1.8, 0.75, "FVM reference data\n(bundled, 800x400)", fc=AMBER)
    nn = box(ax, 3.15, 5.1, 2.2, 0.75, "NNfoil (PINN, PyTorch)\nvendored, 11 findings\nchecked", bold=True)
    mesh = box(ax, 5.3, 5.1, 1.8, 0.75, "O-grid mesh\n(bundled or ogrid.py)", fc=AMBER)
    run = box(ax, 3.15, 3.8, 3.4, 0.8, "Run folder (one per case)\nconfig, log, snapshots per round,\nfield, weights")
    exp = box(ax, 2.6, 2.6, 3.4, 0.7, "export_results.py: case JSON\n(<= 3 MB), summary.csv, figures")
    doc = box(ax, 5.45, 2.6, 1.6, 0.7, "docs_build: report,\nslides (numbers\nfrom summary.csv)", fc=GREEN, size=7.5)
    api = box(ax, 1.5, 1.2, 2.7, 0.9, "FastAPI backend (no PyTorch)\n/api/solve_shape, /api/polar,\n/api/pinn/cases, /api/results",
              bold=True)
    pan = box(ax, 4.9, 1.2, 2.6, 0.9, "Panel solver\nHess-Smith + Prandtl-Glauert,\nmethod of images (road)")
    ui = box(ax, 1.5, 0.0, 2.7, 0.6, "React front end: Explore,\nDraw, PINN Lab, Results", fc=GREEN, bold=True)
    right(ax, fvm, nn)
    left(ax, mesh, nn)
    down(ax, nn, run)
    line(ax, [(3.15, 3.4), (2.6, 2.95)])
    right(ax, exp, doc)
    line(ax, [(1.5, 2.25), (1.5, 1.65)])
    left(ax, pan, api)
    down(ax, api, ui)
    ax.set_ylim(-0.4, 6.4)
    fig.savefig(OUT / "system_architecture.png", bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


def training_flow():
    fig, ax = canvas(6.3, 7.4)
    s = box(ax, 3.4, 7.0, 3.6, 0.5, "Start: shape, Mach number, angle of attack", shape="round", fc=GREEN)
    m = box(ax, 3.4, 6.05, 3.8, 0.6, "Load or build the O-grid mesh\n(fold check: refuse if a cell folds)")
    i = box(ax, 3.4, 5.0, 4.4, 0.8, "Initialise the 5 x 128 network: cold start, or\nweights moved from a coarser mesh\n(grid sequencing)")
    r = box(ax, 3.4, 3.9, 4.0, 0.7, "One round: L-BFGS, up to 1000 iterations\non the Euler residual + boundary terms",
            bold=True)
    q = box(ax, 3.4, 2.85, 2.8, 0.6, "Round failed?", fc=AMBER, shape="round")
    e = box(ax, 3.4, 1.8, 3.8, 0.7, "evaluate_wall(): Cp, Cl, Cd;\nwrite log.json and the round snapshot")
    d = box(ax, 3.4, 0.75, 3.2, 0.7, "Stop? fixed round (3 to 5)\nor round limit", fc=AMBER, shape="round")
    x = box(ax, 5.75, 0.75, 1.0, 0.6, "Export\ncase JSON", fc=GREEN, shape="round", size=7.5)
    rs = box(ax, 5.7, 2.85, 1.1, 0.6, "Restart the\noptimiser", fc=AMBER, size=8)
    down(ax, s, m)
    down(ax, m, i)
    down(ax, i, r)
    down(ax, r, q)
    down(ax, q, e, "no", dx=0.1)
    down(ax, e, d)
    right(ax, d, x, "yes")
    right(ax, q, rs, "yes")
    line(ax, [(5.7, 3.15), (5.7, 3.9), (5.4, 3.9)])
    line(ax, [(1.8, 0.75), (0.15, 0.75), (0.15, 3.9), (1.4, 3.9)])
    ax.text(0.2, 0.5, "no: next round", fontsize=7.5, color=GREY, style="italic")
    fig.savefig(OUT / "training_flowchart.png", bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


def draw_flow():
    fig, ax = canvas(6.3, 6.8)
    steps = [
        ("User draws a closed outline (mouse, pen, finger),\nor picks a preset, or uploads a .dat file", GREEN, "round"),
        ("Clean up: smooth, close, fix direction, start at the\nrear point, resample, scale to chord 1", PALE, "box"),
        ("Outline crosses itself?", AMBER, "round"),
        ("POST /api/solve_shape: Hess-Smith panel solve\n+ Prandtl-Glauert (Mach number)", PALE, "box"),
        ("Optional road: method of images\n(the body tilts, not the wind)", PALE, "box"),
        ("Cp, Cl and the velocity field on a grid (JSON)", PALE, "box"),
        ("Browser canvas: colour map, isobars, particles,\nforce arrows, polar, pin and compare", GREEN, "round"),
    ]
    ys = [6.4, 5.35, 4.3, 3.3, 2.3, 1.4, 0.4]
    boxes = [box(ax, 2.4, y, 4.4, 0.68, txt, fc=fc, shape=shp) for (txt, fc, shp), y in zip(steps, ys)]
    for a, b in zip(boxes[:-1], boxes[1:]):
        down(ax, a, b, "no" if a is boxes[2] else None, dx=0.12)
    refuse = box(ax, 5.65, 4.3, 1.25, 1.0, "Refused with\na clear message:\n\"The outline\ncrosses itself\"", fc=AMBER, size=7.5)
    right(ax, boxes[2], refuse, "yes")
    fig.savefig(OUT / "draw_flowchart.png", bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    system_architecture()
    training_flow()
    draw_flow()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
