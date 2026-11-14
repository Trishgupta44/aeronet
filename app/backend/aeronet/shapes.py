"""Outlines for the shape solver: clean, normalise and validate a closed curve.

App frame: chord 1, leading edge at x = 0, trailing edge (the rightmost point) at (1, 0). Nodes run
counter-clockwise starting at the trailing edge, with the last node not repeated.
"""

from __future__ import annotations

import numpy as np

_TOL = 1e-9
MIN_POINTS = 8
MIN_AREA = 2e-4
Y_LIMIT = 0.5  # normalised outline must stay inside the view window (y in [-0.5, 0.5])


def signed_area(x, y) -> float:
    """Shoelace area; positive for counter-clockwise order."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def chaikin(points, iterations: int = 2) -> np.ndarray:
    """Corner-cutting smoothing of a closed polygon: every pass doubles the number of points."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    for _ in range(iterations):
        q = np.roll(p, -1, axis=0)
        out = np.empty((2 * len(p), 2))
        out[0::2] = 0.75 * p + 0.25 * q
        out[1::2] = 0.25 * p + 0.75 * q
        p = out
    return p


def _clean(points) -> np.ndarray:
    """Drop consecutive duplicates and a repeated closing point; reject non-finite values."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("Points must be a list of (x, y) pairs.")
    if not np.isfinite(p).all():
        raise ValueError("The outline contains values that are not finite numbers.")
    keep = [0] if len(p) else []
    for i in range(1, len(p)):
        if np.hypot(*(p[i] - p[keep[-1]])) > _TOL:
            keep.append(i)
    p = p[keep]
    while len(p) > 1 and np.hypot(*(p[-1] - p[0])) <= _TOL:
        p = p[:-1]
    return p


def _resample(p: np.ndarray, n: int) -> np.ndarray:
    """``n`` points spread uniformly by arc length over the closed loop, starting at ``p[0]``."""
    closed = np.vstack([p, p[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
    t = np.arange(n) * (s[-1] / n)
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


def _normalize_clean(p: np.ndarray, n: int):
    if signed_area(p[:, 0], p[:, 1]) < 0:
        p = p[::-1]
    p = np.roll(p, -int(np.argmax(p[:, 0])), axis=0)  # rightmost point first
    p = _resample(p, n)
    X, Y = p[:, 0].copy(), p[:, 1].copy()
    chord = X.max() - X.min()
    if chord <= _TOL:
        raise ValueError("The shape has no length along x.")
    return (X - X.min()) / chord, (Y - Y[0]) / chord


def normalize_shape(points, n: int = 200):
    """Counter-clockwise, rightmost point first, ``n`` points equally spaced along the curve, chord 1."""
    p = _clean(points)
    if len(p) < 3:
        raise ValueError("A shape needs at least 3 points.")
    return _normalize_clean(p, n)


def self_intersects(X, Y) -> bool:
    """True if any two non-adjacent edges of the closed polygon cross (proper crossings)."""
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    n = X.size
    ax, ay = X, Y
    bx, by = np.roll(X, -1), np.roll(Y, -1)

    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    o1 = orient(ax[:, None], ay[:, None], bx[:, None], by[:, None], ax[None, :], ay[None, :])
    o2 = orient(ax[:, None], ay[:, None], bx[:, None], by[:, None], bx[None, :], by[None, :])
    o3 = orient(ax[None, :], ay[None, :], bx[None, :], by[None, :], ax[:, None], ay[:, None])
    o4 = orient(ax[None, :], ay[None, :], bx[None, :], by[None, :], bx[:, None], by[:, None])
    cross = (o1 * o2 < 0) & (o3 * o4 < 0)
    i, j = np.indices((n, n))
    d = np.abs(i - j)
    return bool((cross & (d > 1) & (d < n - 1)).any())


def _tangent_angle_deg(X, Y) -> float:
    """Angle between the first and the last panel direction (small for a rounded rear end)."""
    t0 = np.array([X[1] - X[0], Y[1] - Y[0]])
    t1 = np.array([X[0] - X[-1], Y[0] - Y[-1]])
    c = np.dot(t0, t1) / (np.linalg.norm(t0) * np.linalg.norm(t1))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def prepare_shape(points, n: int = 200, lifting: bool = True):
    """Validate and normalise an outline. Returns ``(X, Y, info, warnings)``; raises ``ValueError``."""
    p = _clean(points)
    if len(p) < MIN_POINTS:
        raise ValueError(f"A shape needs at least {MIN_POINTS} distinct points (got {len(p)}).")
    X, Y = _normalize_clean(p, n)
    if self_intersects(X, Y):
        raise ValueError("The outline crosses itself. Draw a single loop without crossings.")
    area = signed_area(X, Y)
    if area < MIN_AREA:
        raise ValueError("The shape is too thin or has no inside area.")
    if Y.min() < -Y_LIMIT - 1e-6 or Y.max() > Y_LIMIT + 1e-6:
        raise ValueError(
            "After scaling to length 1 the shape does not fit the view (its height must stay within half a "
            "length above and below the tail). Draw a longer, flatter shape."
        )
    info = {
        "chord": float(p[:, 0].max() - p[:, 0].min()),
        "thickness": float(Y.max() - Y.min()),
        "area": float(area),
        "n_input": int(len(p)),
    }
    warnings: list[str] = []
    if lifting and _tangent_angle_deg(X, Y) < 60.0:
        warnings.append(
            "The rear end is rounded, so the Kutta condition (which fixes the lift) is not meaningful there. "
            "For bluff bodies turn lift off."
        )
    return X, Y, info, warnings


def parse_dat(text: str):
    """Read a UIUC airfoil ``.dat`` file (Selig or Lednicer). Returns ``(name, points)``.

    The first non-empty line is the name; every later line with two or more numbers (spaces or commas)
    is a point. Selig files run trailing edge -> upper surface -> leading edge -> lower surface and are
    used as they are. Lednicer files start with the two point counts (both above 1.5), then the upper
    surface and the lower surface, each from the leading edge to the trailing edge.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("The file is empty.")
    name = lines[0]
    rows: list[tuple[float, float]] = []
    for ln in lines[1:]:
        parts = ln.replace(",", " ").split()
        try:
            x, y = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            continue
        if np.isfinite([x, y]).all():
            rows.append((x, y))
    if rows and rows[0][0] > 1.5 and rows[0][1] > 1.5:  # Lednicer: first row holds the counts
        nu, nl = int(round(rows[0][0])), int(round(rows[0][1]))
        body = rows[1:]
        if nu < 2 or nl < 2 or len(body) < nu + nl:
            raise ValueError("The Lednicer header announces more points than the file contains.")
        upper, lower = body[:nu], body[nu : nu + nl]
        rows = upper[::-1] + lower[1:]
    if len(rows) < MIN_POINTS:
        raise ValueError(f"Found only {len(rows)} coordinate rows; a shape needs at least {MIN_POINTS}.")
    return name, rows


# ---- preset library -------------------------------------------------------------------------------

SEDAN = [(1.00, 0.10), (0.99, 0.16), (0.97, 0.20), (0.80, 0.215), (0.74, 0.23), (0.66, 0.33), (0.58, 0.355),
         (0.42, 0.36), (0.33, 0.33), (0.24, 0.235), (0.10, 0.20), (0.03, 0.17), (0.00, 0.12), (0.005, 0.07),
         (0.03, 0.03), (0.10, 0.00), (0.90, 0.00), (0.97, 0.03), (0.995, 0.06)]
SPORTS_CAR = [(1.00, 0.12), (0.98, 0.17), (0.92, 0.19), (0.80, 0.20), (0.62, 0.26), (0.52, 0.275), (0.42, 0.27),
              (0.32, 0.21), (0.15, 0.13), (0.04, 0.08), (0.00, 0.05), (0.02, 0.015), (0.08, 0.00), (0.92, 0.00),
              (0.98, 0.04)]

CIRCLE_NOTE = "Compare with NNfoil's FVM cylinder (Mach 0.4): Cp min \u22123.70 there vs \u22123.27 here (Prandtl\u2013Glauert)."
CAR_NOTE = (
    "Idealized potential flow: no separation and no wake, so the rear pressure is not realistic and drag is "
    "not shown. Use it to see where the flow speeds up and how the road changes lift."
)


def _ellipse(a: float, b: float, n: int, cx: float = 0.0):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([cx + a * np.cos(t), b * np.sin(t)])


def _symmetric(half_thickness, n: int = 81):
    """Symmetric outline from a half-thickness function of x: cosine-spaced x from 1 to 0, upper then lower."""
    x = 0.5 * (1.0 + np.cos(np.pi * np.arange(n) / (n - 1)))  # 1 -> 0
    yt = half_thickness(x)
    upper = np.column_stack([x, yt])
    lower = np.column_stack([x[::-1], -yt[::-1]])[1:-1]  # leading and trailing points already in `upper`
    return np.vstack([upper, lower])


def library() -> list[dict]:
    """Preset shapes: ``[{id, label, points, lifting, ground, note}]`` (points rounded to 5 decimals)."""
    from .geometry import naca4  # local import keeps shapes.py importable on its own

    bird = naca4("6409", 60)
    plate = lambda x: 0.015 * np.sqrt(np.minimum(1.0, x / 0.02)) * np.minimum(1.0, (1.0 - x) / 0.1)
    teardrop = lambda x: 0.25 * np.sqrt(x) * (1.0 - x)
    entries = [
        ("circle", "Circle", _ellipse(0.5, 0.5, 96, cx=0.5), False, None, CIRCLE_NOTE),
        ("ellipse", "Ellipse 3:1", _ellipse(0.5, 0.5 / 3.0, 96, cx=0.5), False, None, ""),
        ("teardrop", "Teardrop", _symmetric(teardrop), True, None, ""),
        ("plate", "Flat plate", _symmetric(plate), True, None, ""),
        ("bird", "Bird wing", np.column_stack([bird.x, bird.y]), True, None, ""),
        ("sedan", "Sedan", chaikin(SEDAN, 3), False, 0.03, CAR_NOTE),
        ("sports", "Sports car", chaikin(SPORTS_CAR, 3), False, 0.02, CAR_NOTE),
    ]
    return [
        {
            "id": id_,
            "label": label,
            "points": np.round(pts, 5).tolist(),
            "lifting": lifting,
            "ground": ground,
            "note": note,
        }
        for id_, label, pts, lifting, ground, note in entries
    ]
