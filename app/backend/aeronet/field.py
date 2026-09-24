"""Pressure-coefficient field around the airfoil, sampled on a regular grid."""

from __future__ import annotations

import numpy as np

from .panel import PanelSolution, influence

# Fixed viewing window used by the interface (aspect ratio 1.8 : 1).
XMIN, XMAX = -0.4, 1.4
YMIN, YMAX = -0.5, 0.5
_CHUNK = 4096


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


def field_cp(sol: PanelSolution, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Cp at arbitrary points (outside the body), including the Prandtl-Glauert factor."""
    out = np.empty(px.size)
    for s in range(0, px.size, _CHUNK):
        sx, sy, vx, vy = influence(sol.geom, px[s : s + _CHUNK], py[s : s + _CHUNK])
        u = sol.vx_inf + sx @ sol.sigma + sol.gamma * vx.sum(axis=1)
        v = sol.vy_inf + sy @ sol.sigma + sol.gamma * vy.sum(axis=1)
        out[s : s + _CHUNK] = (1.0 - (u * u + v * v)) / sol.beta
    return out


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
    cp[~inside] = field_cp(sol, px[~inside], py[~inside])
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
