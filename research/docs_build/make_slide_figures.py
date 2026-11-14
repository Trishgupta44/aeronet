"""Three explanatory figures for the slides, drawn from repository data (no numbers typed by hand).

    research/.venv/Scripts/python research/docs_build/make_slide_figures.py

Writes ``docs/deliverables_src/slides_img/{pinn_diagram,le_gradient,mesh_transformation}.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "research"))
import metrics  # noqa: E402

OUT = REPO / "docs" / "deliverables_src" / "slides_img"
NAVY, INDIGO, MUTED, RED = "#0a1a5c", "#1a0f9e", "#566070", "#d4502a"


def _box(ax, x, y, w, h, text, fc="#eff4f7", ec=NAVY, size=13, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=NAVY,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.8))


def pinn_diagram(path):
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=200)
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.4, 4.6)
    ax.axis("off")
    _box(ax, 0.2, 1.7, 2.0, 1.2, "mesh position\n(xi, eta)", size=14)
    _box(ax, 2.9, 1.5, 2.3, 1.6, "neural network\n5 layers x 128", fc="#e6e8fb", size=14, bold=True)
    _box(ax, 5.9, 1.7, 2.0, 1.2, "flow state\nu, v, p, rho", size=14)
    _box(ax, 8.6, 2.9, 2.2, 1.1, "Euler residual\n(autodiff)", fc="#fdeee8", ec=RED, size=13)
    _box(ax, 8.6, 0.6, 2.2, 1.1, "wall and far-field\nconditions", fc="#fdeee8", ec=RED, size=13)
    _arrow(ax, 2.2, 2.3, 2.9, 2.3)
    _arrow(ax, 5.2, 2.3, 5.9, 2.3)
    _arrow(ax, 7.9, 2.5, 8.6, 3.35)
    _arrow(ax, 7.9, 2.1, 8.6, 1.15)
    ax.annotate("", xy=(4.05, 1.5), xytext=(9.7, 0.6),
                arrowprops=dict(arrowstyle="-|>", color=INDIGO, lw=1.8, ls="--", connectionstyle="arc3,rad=-0.15"))
    ax.text(5.6, -0.3, "loss = residual + conditions: adjust the weights (L-BFGS)", ha="center", fontsize=12, color=INDIGO)
    ax.text(0.2, 4.2, "No labelled data: the equations are the teacher", fontsize=15, color=NAVY, fontweight="bold")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def le_gradient(path):
    ref = metrics.load_fvm(REPO / "NNfoil" / "FVM_Cp" / "NACA0012_800_400_Cp.dat")
    wx, _ = metrics.load_mesh_wall(REPO / "NNfoil" / "mesh" / "NACA0012_50_25.x")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=200, gridspec_kw={"width_ratios": [1.2, 1]})
    for ax, xmax in ((a, 1.02), (b, 0.06)):
        m = ref[:, 0] <= xmax
        ax.plot(ref[m, 0], ref[m, 2], ".", ms=3 if xmax > 0.5 else 5, color=INDIGO, label="FVM 800x400 (reference file)")
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        ax.set_xlabel("x/c")
        ax.set_ylabel("Cp")
    xs = wx[(wx >= 0) & (wx <= 0.06)]
    b.vlines(xs, *b.get_ylim(), colors=RED, lw=1.2, label="wall nodes of the 50x25 mesh")
    a.set_title("Cp, NACA 0012, Mach 0.4, 4 deg", fontsize=12, color=NAVY)
    b.set_title("Zoom on the first 6% of the chord", fontsize=12, color=NAVY)
    b.legend(loc="lower right", fontsize=9)
    a.legend(loc="lower right", fontsize=9)
    fig.text(0.01, -0.02, "Data: NNfoil/FVM_Cp/NACA0012_800_400_Cp.dat (bundled FVM data, measured by the NNfoil authors) and "
             "NNfoil/mesh/NACA0012_50_25.x.", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mesh_transformation(path):
    xx, yy = metrics.load_mesh(REPO / "NNfoil" / "mesh" / "NACA0012_50_25.x")  # (N+1, M)
    n1, m = xx.shape
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200, gridspec_kw={"width_ratios": [1.15, 1]})
    for j in range(m):
        a.plot(xx[:, j], yy[:, j], color=INDIGO, lw=0.5 if j else 1.6, alpha=0.8 if j else 1)
    for i in range(0, n1, 2):
        a.plot(xx[i, :], yy[i, :], color=INDIGO, lw=0.4, alpha=0.7)
    a.plot(xx[:, 0], yy[:, 0], color=RED, lw=1.8)
    a.set_xlim(-1.3, 1.3)
    a.set_ylim(-1.0, 1.0)
    a.set_aspect("equal")
    a.set_title("Physical space (x, y): curved mesh, airfoil in the middle", fontsize=11, color=NAVY)
    xi, eta = np.arange(n1), np.arange(m)
    for j in eta:
        b.plot([0, n1 - 1], [j, j], color=INDIGO, lw=0.5 if j else 1.6, alpha=0.8)
    for i in range(0, n1, 2):
        b.plot([i, i], [0, m - 1], color=INDIGO, lw=0.4, alpha=0.7)
    b.plot([0, n1 - 1], [0, 0], color=RED, lw=2.2)
    b.text(n1 / 2, -2.2, "wall (the airfoil) is the bottom edge", ha="center", fontsize=10, color=RED)
    b.text(n1 / 2, m + 0.4, "far field is the top edge", ha="center", fontsize=10, color=NAVY)
    b.set_xlim(-1, n1)
    b.set_ylim(-3, m + 2)
    b.set_title("Computational space (xi, eta): a plain rectangle", fontsize=11, color=NAVY)
    for ax in (a, b):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.01, -0.02, "Mesh: NNfoil/mesh/NACA0012_50_25.x (every second xi line drawn). The network works on the right-hand "
             "picture; the equations are mapped back with the Jacobian.", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pinn_diagram(OUT / "pinn_diagram.png")
    le_gradient(OUT / "le_gradient.png")
    mesh_transformation(OUT / "mesh_transformation.png")
    for p in sorted(OUT.glob("*.png")):
        print("wrote", p)


if __name__ == "__main__":
    main()
