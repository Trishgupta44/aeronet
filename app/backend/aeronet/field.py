"""Pressure-coefficient field around the airfoil, sampled on a regular grid."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .panel import PanelSolution

# Fixed viewing window used by the interface (aspect ratio 1.8 : 1).
XMIN, XMAX = -0.4, 1.4
YMIN, YMAX = -0.5, 0.5
_CHUNK = 2048
_WORKERS = min(4, os.cpu_count() or 1)


def inside_body(X: np.ndarray, Y: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Ray-casting point-in-polygon test, vectorised over the points."""
    x_i, y_i = X, Y
    x_j, y_j = np.roll(X, 1), np.roll(Y, 1)
    inside = np.zeros(px.shape, dtype=bool)
    for s in range(0, px.size, _CHUNK):
        qx, qy = px[s : s + _CHUNK, None], py[s : s + _CHUNK, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            crosses = (y_i > qy) != (y_j > qy)
            x_at = (x_j - x_i) * (qy - y_i) / (y_j - y_i) + x_i
            hit = crosses & (qx < x_at)
        inside[s : s + _CHUNK] = (hit.sum(axis=1) % 2) == 1
    return inside


def _cp_chunk(px, py, g, weights, freestream, beta, dtype):
    a_x, a_y, b_x, b_y = weights
    rx = px[:, None] - g.x0
    ry = py[:, None] - g.y0
    xl = rx * g.tx + ry * g.ty
    yl = ry * g.tx - rx * g.ty
    xe = xl - g.length
    with np.errstate(divide="ignore", invalid="ignore"):
        us = np.log((xl * xl + yl * yl) / (xe * xe + yl * yl)) * dtype(1 / (4 * np.pi))
        vs = np.arctan2(yl * g.length, xl * xe + yl * yl) * dtype(1 / (2 * np.pi))  # angle from panel start to end
    u = freestream[0] + us @ a_x + vs @ b_x
    v = freestream[1] + us @ a_y + vs @ b_y
    return (1.0 - (u * u + v * v)) / beta


def field_cp(sol: PanelSolution, px: np.ndarray, py: np.ndarray, dtype=np.float64, workers: int = 1) -> np.ndarray:
    """Cp at arbitrary points (outside the body), including the Prandtl-Glauert factor.

    Same result as summing ``panel.influence`` over all panels, but faster: a unit source and a
    unit vortex on a panel both depend only on the two local terms ``us`` (log of the distance
    ratio) and ``vs`` (angle subtended by the panel), so the sums over the panels reduce to four
    matrix-vector products with per-panel weights. ``dtype=np.float32`` and ``workers > 1`` trade a
    little precision / extra threads for speed (used for the picture, not for the forces).
    """
    g = sol.geom
    g = type(g)(*(np.asarray(a, dtype=dtype) for a in (g.x0, g.y0, g.length, g.tx, g.ty, g.nx, g.ny, g.cx, g.cy)))
    sigma, gamma = sol.sigma.astype(dtype), dtype(sol.gamma)
    ux, uy = -g.ty, g.tx  # local y axis in global coordinates
    # velocity = us * (sigma*t + gamma*u) + vs * (sigma*u - gamma*t), summed over panels
    weights = (sigma * g.tx + gamma * ux, sigma * g.ty + gamma * uy, sigma * ux - gamma * g.tx, sigma * uy - gamma * g.ty)
    freestream = (dtype(sol.vx_inf), dtype(sol.vy_inf))
    px, py = np.asarray(px, dtype=dtype), np.asarray(py, dtype=dtype)
    spans = [slice(s, s + _CHUNK) for s in range(0, px.size, _CHUNK)]
    job = lambda sl: _cp_chunk(px[sl], py[sl], g, weights, freestream, dtype(sol.beta), dtype)
    if workers > 1 and len(spans) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(job, spans))
    else:
        parts = [job(sl) for sl in spans]
    return np.concatenate(parts).astype(np.float64) if parts else np.empty(0)


def field_grid(sol: PanelSolution, width: int) -> dict:
    """Cp on a ``width`` x ``round(width / 1.8)`` grid of cell centres, ``None`` inside the body.

    Values are row-major, first row at the top (y = YMAX), matching how the canvas is drawn.
    """
    height = max(1, round(width / 1.8))
    xs = XMIN + (np.arange(width) + 0.5) / width * (XMAX - XMIN)
    ys = YMAX - (np.arange(height) + 0.5) / height * (YMAX - YMIN)
    gx, gy = np.meshgrid(xs, ys)
    px, py = gx.ravel(), gy.ravel()

    inside = inside_body(sol.airfoil.x, sol.airfoil.y, px, py)
    cp = np.full(px.size, np.nan)
    cp[~inside] = field_cp(sol, px[~inside], py[~inside], dtype=np.float32, workers=_WORKERS)
    values = np.round(cp, 3).astype(object)
    values[np.isnan(cp)] = None
    return {
        "width": int(width),
        "height": int(height),
        "xmin": XMIN,
        "xmax": XMAX,
        "ymin": YMIN,
        "ymax": YMAX,
        "cp": values.tolist(),
    }
