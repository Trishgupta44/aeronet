"""Finding 3: the cross-term factor in NNfoil/mesh_generation/mesh_gen.py.

mesh_gen.py (lines 66-70) updates the grid with `- 0.5 * beta * X_xi_eta`. The Winslow/Thompson elliptic grid
equation  alpha*x_xixi - 2*beta*x_xieta + gamma*x_etaeta = 0  needs, with differ.diff2d's central stencils, a factor 2.
This script regenerates the 50x25 NACA0012 O-grid twice (factor 0.5 = upstream, 2.0 = Winslow) and measures, for each:
iterations, time, max |true Winslow residual| over interior points, minimum angle between grid lines, whether the
Jacobian is single-signed, and the largest distance to the bundled NNfoil/mesh/NACA0012_50_25.x.
It does NOT import mesh_gen.py (a script that writes a file at import); the iteration is copied into `generate`.
All numbers are measured; the CSV is results/finding3_mesh_smoothing.csv.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "NNfoil" / "mesh_generation"))
from differ import diff2d  # noqa: E402  (pure functions, no side effects)

NODES = [-1, 0, 1]
# CST hyper-parameters and NACA0012 coefficients, copied from mesh_gen.py
DEGREE, N_1, N_2 = 5, 0.5, 1
B_UP = [0.171787, 0.155338, 0.161996, 0.137638, 0.145718, 0.143815]
B_DOWN = [-b for b in B_UP]
UPSTREAM_COEFF, WINSLOW_COEFF = 0.5, 2.0
COLUMNS = ["variant", "cross_coeff", "n_xi", "n_eta", "iterations", "time_s", "max_abs_winslow_residual",
           "min_grid_angle_deg", "jacobian_single_signed", "max_dist_to_bundled", "source"]


def cal_mx(x_cor):
    """CST basis matrix (mesh_gen.py Cal_mx with math.factorial)."""
    x_cor = np.asarray(x_cor).reshape(-1)
    mx = np.empty([x_cor.shape[0], DEGREE + 1])
    for i in range(DEGREE + 1):
        f = math.factorial(DEGREE) / (math.factorial(i) * math.factorial(DEGREE - i))
        mx[:, i] = f * np.power(x_cor, i + N_1) * np.power(1 - x_cor, DEGREE - i + N_2)
    return mx


def generate(n_xi, n_eta, cross_coeff, radius=15.0, tol=1e-6, max_iter=500000):
    """mesh_gen.py's O-grid iteration with the cross-term factor as an argument. Returns (X, Y, iterations).

    X, Y have shape (n_eta, n_xi): rows are radial (wall = row 0), columns run round the airfoil (periodic).
    """
    theta = np.linspace(0, 2 * np.pi, n_xi + 1).reshape(-1, 1)[:-1]
    x_wall = 0.5 * np.cos(theta)
    mx = cal_mx(x_wall.reshape(-1) + 0.5)
    h = n_xi // 2
    y_wall = np.r_[mx[:h] @ np.array(B_UP).reshape(-1, 1), mx[h:] @ np.array(B_DOWN).reshape(-1, 1)]
    X0, Y0 = x_wall.flatten(), y_wall.flatten()
    X1, Y1 = (radius * np.cos(theta)).flatten(), (radius * np.sin(theta)).flatten()

    X, Y = np.zeros((n_eta, n_xi)), np.zeros((n_eta, n_xi))
    X[0], Y[0], X[-1], Y[-1] = X0, Y0, X1, Y1
    for i in range(1, n_eta - 1):
        X[i] = X[0] + i / (n_eta - 1) * (X[-1] - X[0])
        Y[i] = Y[0] + i / (n_eta - 1) * (Y[-1] - Y[0])

    tem = X + 1
    k = 0
    for k in range(max_iter):
        alpha = diff2d(X, NODES, 0) ** 2 + diff2d(Y, NODES, 0) ** 2
        beta = diff2d(X, NODES, 1, 1, 2) * diff2d(X, NODES, 0) + diff2d(Y, NODES, 1, 1, 2) * diff2d(Y, NODES, 0)
        gamma = diff2d(X, NODES, 1, 1, 2) ** 2 + diff2d(Y, NODES, 1, 1, 2) ** 2
        dXdxi, dYdxi = diff2d(X, NODES, 1, 1, 2), diff2d(Y, NODES, 1, 1, 2)
        X = 0.5 * (alpha * diff2d(X, NODES, 1, 2, 2) - cross_coeff * beta * diff2d(dXdxi, NODES, 0)
                   + gamma * diff2d(X, NODES, 0, 2) + (alpha + gamma) * 2 * X) / (alpha + gamma)
        Y = 0.5 * (alpha * diff2d(Y, NODES, 1, 2, 2) - cross_coeff * beta * diff2d(dYdxi, NODES, 0)
                   + gamma * diff2d(Y, NODES, 0, 2) + (alpha + gamma) * 2 * Y) / (alpha + gamma)
        X[0], Y[0], X[-1], Y[-1] = X0, Y0, X1, Y1
        if abs(tem - X).max() < tol:
            break
        tem = X.copy()
    return X, Y, k


def quality(X, Y):
    """(max |Winslow residual| over interior rows, min grid-line angle in degrees, Jacobian single-signed?).

    The residual is (alpha x_xixi - 2 beta x_xieta + gamma x_etaeta) / (alpha + gamma) with the same central stencils
    the generator uses, so a converged Winslow grid gives ~0. The Jacobian is x_xi y_eta - x_eta y_xi on all nodes.
    """
    Xe, Ye = diff2d(X, NODES, 0), diff2d(Y, NODES, 0)  # radial (eta)
    Xx, Yx = diff2d(X, NODES, 1, 1, 2), diff2d(Y, NODES, 1, 1, 2)  # round the airfoil (xi), periodic
    a = Xe**2 + Ye**2
    b = Xx * Xe + Yx * Ye
    g = Xx**2 + Yx**2
    res = (a * diff2d(X, NODES, 1, 2, 2) - 2 * b * diff2d(Xx, NODES, 0) + g * diff2d(X, NODES, 0, 2))[1:-1]
    res = res / (a + g)[1:-1]
    cosang = (b / np.sqrt(a * g))[1:-1]
    angle = np.degrees(np.arccos(np.clip(np.abs(cosang), 0, 1))).min()
    jac = Xx * Ye - Xe * Yx
    return float(np.abs(res).max()), float(angle), bool((jac < 0).all() or (jac > 0).all())


def load_bundled(path):
    """Bundled NNfoil .x file -> (X, Y) with shape (n_eta, n_xi), as written by mesh_gen.py."""
    d = np.loadtxt(path, skiprows=1)
    h = d.shape[0] // 2
    return d[:h], d[h:]


def run(n_xi=50, n_eta=25, bundled_path=None):
    """Both variants plus the bundled mesh itself. Returns a list of row dicts (CSV columns)."""
    bundled_path = bundled_path or REPO / "NNfoil" / "mesh" / f"NACA0012_{n_xi}_{n_eta}.x"
    bx, by = load_bundled(bundled_path)
    rows = []
    for variant, coeff in (("upstream", UPSTREAM_COEFF), ("winslow", WINSLOW_COEFF)):
        t0 = time.perf_counter()
        X, Y, iters = generate(n_xi, n_eta, coeff)
        dt = time.perf_counter() - t0
        res, ang, ok = quality(X, Y)
        rows.append({"variant": variant, "cross_coeff": coeff, "n_xi": n_xi, "n_eta": n_eta, "iterations": iters,
                     "time_s": dt, "max_abs_winslow_residual": res, "min_grid_angle_deg": ang,
                     "jacobian_single_signed": ok, "max_dist_to_bundled": float(np.hypot(X - bx, Y - by).max()),
                     "source": "measured"})
    res, ang, ok = quality(bx, by)
    rows.append({"variant": "bundled", "cross_coeff": "", "n_xi": n_xi, "n_eta": n_eta, "iterations": "",
                 "time_s": "", "max_abs_winslow_residual": res, "min_grid_angle_deg": ang,
                 "jacobian_single_signed": ok, "max_dist_to_bundled": 0.0, "source": "measured"})
    return rows


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    rows = run()
    write_csv(rows, REPO / "results" / "finding3_mesh_smoothing.csv")
    for r in rows:
        print({k: (round(v, 8) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
