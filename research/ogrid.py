"""O-grid generator for any closed curve (Winslow elliptic smoothing, vectorised), with a Jacobian validity check.

All coordinates are in the NNfoil frame (chord in [-0.5, 0.5], trailing edge on the right at y = 0).

    wall -> ogrid(xw, yw, M) -> (X, Y, iters, last_change) -> is_valid(X, Y) -> write_x(path, X, Y)

Grid layout: ``X`` and ``Y`` have shape ``(M, N)``. Row 0 is the wall, row ``M-1`` is the far circle of radius ``R``;
column ``j`` sits at angle ``2*pi*j/N`` on the far circle. Columns are periodic (xi direction, axis 1); rows run from
the wall outwards (eta direction, axis 0). With this ordering a valid grid has a negative cell Jacobian everywhere.

The smoother is the Jacobi iteration of upstream ``NNfoil/mesh_generation/mesh_gen.py`` written with array slices.
``cross_coeff`` is the weight of the cross-derivative term: the Winslow equations
``alpha*x_xixi - 2*beta*x_xieta + gamma*x_etaeta = 0`` need **2.0** (default). Upstream uses 0.5, and the bundled
meshes (``NNfoil/mesh/*.x``) were made with it: pass ``cross_coeff=0.5`` to reproduce them (roadmap finding 3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- wall curves
def resample_closed(x, y, n):
    """Turn any closed curve into ``n`` wall nodes in the NNfoil frame.

    Drops a duplicated end point, makes the curve counter-clockwise, starts at the rightmost point (the trailing
    edge) and resamples uniformly by arc length. The chord (x extent) is scaled to 1 and centred on x = 0, and the
    start node is moved to y = 0. Use it for drawn shapes and the cylinder; airfoils use :func:`cosine_wall`.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape or x.size < 3:
        raise ValueError("need at least 3 points, with x and y of the same length")
    if np.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-12 * max(1.0, np.ptp(x)):
        x, y = x[:-1], y[:-1]
    if 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        x, y = x[::-1], y[::-1]
    chord = float(np.ptp(x))
    if chord <= 0:
        raise ValueError("the curve has no extent in x")
    start = int(np.argmax(x))
    xs, ys = np.roll(x, -start), np.roll(y, -start)
    xc, yc = np.r_[xs, xs[0]], np.r_[ys, ys[0]]
    seg = np.hypot(np.diff(xc), np.diff(yc))
    s = np.r_[0.0, np.cumsum(seg)]
    t = np.arange(n) * (s[-1] / n)
    X = np.interp(t, s, xc)
    Y = np.interp(t, s, yc)
    return (X - x.min()) / chord - 0.5, (Y - ys[0]) / chord


def cosine_wall(xw, yw, n):
    """Airfoil-style wall nodes ``x = 0.5*cos(theta)``, ``theta = 2*pi*j/n``, from a finer wall of the same airfoil.

    The wall must run counter-clockwise from the trailing edge (upper branch first, as the bundled meshes do). y is
    interpolated on the upper (TE -> LE) and lower (LE -> TE) branches separately. The bundled meshes use this
    spacing, so a coarse wall made from a finer one reproduces the bundled coarse wall.
    """
    xw = np.asarray(xw, dtype=float).ravel()
    yw = np.asarray(yw, dtype=float).ravel()
    le = int(np.argmin(xw))
    xu, yu = xw[: le + 1][::-1], yw[: le + 1][::-1]  # increasing x, from the LE to the TE
    xl, yl = np.r_[xw[le:], xw[0]], np.r_[yw[le:], yw[0]]  # increasing x, LE -> TE (closing at node 0)
    theta = 2.0 * np.pi * np.arange(n) / n
    X = 0.5 * np.cos(theta)
    Y = np.where(theta <= np.pi, np.interp(X, xu, yu), np.interp(X, xl, yl))
    return X, Y


# ---------------------------------------------------------------- grid
def ogrid(xw, yw, M, R=15.0, cross_coeff=2.0, tol=1e-6, max_iter=200_000):
    """Winslow O-grid around the wall ``(xw, yw)``: returns ``(X, Y, iters, last_change)``.

    Starts from linear interpolation between the wall and the far circle (radius ``R``, node j at angle
    ``2*pi*j/N``) and applies Jacobi sweeps until both X and Y change by less than ``tol`` (or ``max_iter``).
    ``last_change`` is the largest node move of the last sweep; a value above ``tol`` means it did not converge.
    """
    xw = np.asarray(xw, dtype=float).ravel()
    yw = np.asarray(yw, dtype=float).ravel()
    N = xw.size
    theta = 2.0 * np.pi * np.arange(N) / N
    s = (np.arange(M) / (M - 1))[:, None]
    X = xw[None, :] + s * (R * np.cos(theta) - xw)[None, :]
    Y = yw[None, :] + s * (R * np.sin(theta) - yw)[None, :]

    cc = float(cross_coeff)
    change = np.inf
    it = 0
    for it in range(1, int(max_iter) + 1):
        xp, xm = np.roll(X, -1, axis=1), np.roll(X, 1, axis=1)  # xi neighbours (periodic)
        yp, ym = np.roll(Y, -1, axis=1), np.roll(Y, 1, axis=1)
        x_xi, y_xi = 0.5 * (xp - xm), 0.5 * (yp - ym)
        x_eta, y_eta = 0.5 * (X[2:] - X[:-2]), 0.5 * (Y[2:] - Y[:-2])  # interior rows only
        x_xi, y_xi = x_xi[1:-1], y_xi[1:-1]
        a = x_eta * x_eta + y_eta * y_eta  # multiplies the xi second difference
        g = x_xi * x_xi + y_xi * y_xi  # multiplies the eta second difference
        b = x_xi * x_eta + y_xi * y_eta
        den = 2.0 * (a + g)
        x_xieta = 0.25 * (xp[2:] - xm[2:] - xp[:-2] + xm[:-2])
        y_xieta = 0.25 * (yp[2:] - ym[2:] - yp[:-2] + ym[:-2])
        Xn = (a * (xp[1:-1] + xm[1:-1]) + g * (X[2:] + X[:-2]) - cc * b * x_xieta) / den
        Yn = (a * (yp[1:-1] + ym[1:-1]) + g * (Y[2:] + Y[:-2]) - cc * b * y_xieta) / den
        change = max(np.abs(Xn - X[1:-1]).max(), np.abs(Yn - Y[1:-1]).max())
        X[1:-1], Y[1:-1] = Xn, Yn
        if not np.isfinite(change) or change < tol:
            break
    return X, Y, it, float(change)


def cell_jacobian(X, Y):
    """Jacobian ``x_xi*y_eta - x_eta*y_xi`` of every cell, shape ``(M-1, N)`` (cell j spans columns j and j+1, periodic)."""
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    Xr, Yr = np.roll(X, -1, axis=1), np.roll(Y, -1, axis=1)
    x_xi = 0.5 * ((Xr[:-1] - X[:-1]) + (Xr[1:] - X[1:]))
    y_xi = 0.5 * ((Yr[:-1] - Y[:-1]) + (Yr[1:] - Y[1:]))
    x_eta = 0.5 * ((X[1:] - X[:-1]) + (Xr[1:] - Xr[:-1]))
    y_eta = 0.5 * ((Y[1:] - Y[:-1]) + (Yr[1:] - Yr[:-1]))
    return x_xi * y_eta - x_eta * y_xi


def is_valid(X, Y) -> bool:
    """True when every cell Jacobian is finite, non-zero and has the same sign (no folded or collapsed cell)."""
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        return False
    j = cell_jacobian(X, Y)
    return bool((j < 0).all() or (j > 0).all())


def write_x(path, X, Y) -> None:
    """Write NNfoil's mesh format: first line ``N<TAB>M``, then the ``2M`` rows of ``np.r_[X, Y]`` (``%.16f``)."""
    M, N = X.shape
    with Path(path).open("w", newline="\n") as f:
        f.write(f"{N}\t{M}\n")
        np.savetxt(f, np.r_[X, Y], fmt="%.16f")
