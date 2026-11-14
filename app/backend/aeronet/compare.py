"""Thin-airfoil theory and comparison of a solution with reference Cp data."""

from __future__ import annotations

import numpy as np

from .geometry import Airfoil
from .panel import PanelSolution


def thin_airfoil_cl(airfoil: Airfoil, alpha_deg: float, mach: float) -> float:
    """Cl = 2*pi*(alpha - alpha_L0) / sqrt(1 - Mach^2), with alpha_L0 from the camber line."""
    n = 400
    theta = (np.arange(n) + 0.5) * np.pi / n
    x = 0.5 * (1 - np.cos(theta))
    integral = np.sum(airfoil.dcamber(x) * (np.cos(theta) - 1) * (np.pi / n))
    alpha_l0 = -integral / np.pi
    return float(2 * np.pi * (np.radians(alpha_deg) - alpha_l0) / np.sqrt(1 - mach * mach))


def parse_reference(text: str) -> np.ndarray:
    """Parse an NNfoil ``*_Cp.dat`` file: a header line, then ``node x y Cp`` rows.

    The files centre the chord on x = 0 (-0.5 .. 0.5); x is shifted to 0 .. 1. Returns an (K, 3)
    array of (x, y, Cp); empty if no valid rows were found.
    """
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            x, y, cp = (float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            continue
        if np.isfinite([x, y, cp]).all():
            rows.append((x, y, cp))
    pts = np.array(rows, dtype=float).reshape(-1, 3)
    if pts.size and pts[:, 0].min() < -0.1:
        pts[:, 0] += 0.5
    return pts


def compare_cp(sol: PanelSolution, ref: np.ndarray) -> dict:
    """RMS and maximum Cp difference between the solution and reference points (x, y, Cp).

    Only meaningful when the airfoil, angle of attack and Mach number match the reference.
    """
    if ref.size == 0:
        return {"rms": None, "max": None, "n": 0}
    af, g = sol.airfoil, sol.geom
    is_upper = g.cy >= af.camber(g.cx)

    def side(mask):
        order = np.argsort(g.cx[mask])
        return g.cx[mask][order], sol.cp[mask][order]

    up_x, up_cp = side(is_upper)
    lo_x, lo_cp = side(~is_upper)

    x, y, cp_ref = ref[:, 0], ref[:, 1], ref[:, 2]
    keep = (x >= 0.02) & (x <= 0.98)  # skip the very tips where the two surfaces are ambiguous
    ref_upper = y >= af.camber(x)
    pred = np.full(x.shape, np.nan)
    for is_up, (xs, cps) in ((True, (up_x, up_cp)), (False, (lo_x, lo_cp))):
        if xs.size == 0:
            continue
        sel = keep & (ref_upper == is_up) & (x >= xs[0]) & (x <= xs[-1])
        pred[sel] = np.interp(x[sel], xs, cps)
    valid = ~np.isnan(pred)
    if not valid.any():
        return {"rms": None, "max": None, "n": 0}
    err = pred[valid] - cp_ref[valid]
    return {"rms": float(np.sqrt(np.mean(err**2))), "max": float(np.max(np.abs(err))), "n": int(valid.sum())}


def compare_cp_nearest(sol: PanelSolution, ref: np.ndarray) -> dict:
    """RMS and maximum Cp difference, taking for each reference point the nearest panel centre.

    Roadmap section 5.5 rule, used for drawn shapes (they have no camber line). Reference points with
    0.02 <= x <= 0.98 are used.
    """
    if ref.size == 0:
        return {"rms": None, "max": None, "n": 0}
    x, y, cp_ref = ref[:, 0], ref[:, 1], ref[:, 2]
    keep = (x >= 0.02) & (x <= 0.98)
    if not keep.any():
        return {"rms": None, "max": None, "n": 0}
    g = sol.geom
    d2 = (x[keep, None] - g.cx[None, :]) ** 2 + (y[keep, None] - g.cy[None, :]) ** 2
    err = sol.cp[np.argmin(d2, axis=1)] - cp_ref[keep]
    return {"rms": float(np.sqrt(np.mean(err**2))), "max": float(np.max(np.abs(err))), "n": int(keep.sum())}
