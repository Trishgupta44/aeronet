"""Shared metrics and file loaders for AeroNet research scripts (one definition of every reported number)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SUMMARY_COLUMNS = ["experiment", "case", "method", "mesh", "mach", "alpha_deg", "rms_cp", "max_cp", "cl", "cl_ref",
                   "cd", "train_s", "infer_ms", "source", "notes"]
_KEY = ("experiment", "case", "method", "mesh")


def load_fvm(path) -> np.ndarray:
    """NNfoil *_Cp.dat (header, then node x y Cp) -> (K, 3) array of (x_app, y, cp), x_app = x + 0.5."""
    d = np.loadtxt(path, skiprows=1)
    return np.c_[d[:, 1] + 0.5, d[:, 2], d[:, 3]]


def load_mesh(path):
    """NNfoil .x mesh -> closed (xx, yy) of shape (N+1, M) in the NNfoil frame, exactly as the upstream main script."""
    d = np.loadtxt(path, skiprows=1)
    h = d.shape[0] // 2
    xx, yy = d[:h].T, d[h:].T
    return np.r_[xx, xx[0:1]], np.r_[yy, yy[0:1]]


def load_mesh_wall(path):
    """Wall nodes (N,) of an NNfoil mesh in the app frame (x + 0.5)."""
    xx, yy = load_mesh(path)
    return xx[:-1, 0] + 0.5, yy[:-1, 0].copy()


def cp_error(x, y, cp, ref) -> dict:
    """RMS / max |Cp_pred - Cp_ref| using, for each reference point with 0.02 <= x <= 0.98, the nearest predicted point."""
    x, y, cp, ref = (np.asarray(a, dtype=float) for a in (x, y, cp, ref))
    keep = (ref[:, 0] >= 0.02) & (ref[:, 0] <= 0.98)
    r = ref[keep]
    if r.size == 0 or x.size == 0:
        return {"rms": None, "max": None, "n": 0}
    d2 = (x[None, :] - r[:, 0:1]) ** 2 + (y[None, :] - r[:, 1:2]) ** 2
    err = cp[d2.argmin(axis=1)] - r[:, 2]
    return {"rms": float(np.sqrt(np.mean(err**2))), "max": float(np.max(np.abs(err))), "n": int(err.size)}


def forces_from_faces(xn, yn, cp_faces, alpha_deg):
    """(Cl, Cd) from face Cp on a closed wall polyline (N+1 nodes, last == first), NNfoil's Calculate_Force convention."""
    xn, yn, cp = (np.asarray(a, dtype=float) for a in (xn, yn, cp_faces))
    dx, dy = np.diff(xn), np.diff(yn)
    dl = np.hypot(dx, dy)
    tan_x, tan_y = -dx / dl, -dy / dl
    nor_x, nor_y = -tan_y, tan_x
    fx, fy = cp * dl * nor_x, cp * dl * nor_y
    a = np.radians(alpha_deg)
    cl = -(np.cos(a) * fy - np.sin(a) * fx).sum()
    cd = -(np.sin(a) * fy + np.cos(a) * fx).sum()
    return float(cl), float(cd)


def order_contour(points) -> np.ndarray:
    """Sort (x, y, cp) rows counter-clockwise by atan2(y - ybar, x - xbar) mod 2pi about their centroid.

    The bundled FVM Cp files are NOT stored in contour order; integrating them unsorted gives a wrong Cl.
    """
    pts = np.asarray(points, dtype=float)
    ang = np.mod(np.arctan2(pts[:, 1] - pts[:, 1].mean(), pts[:, 0] - pts[:, 0].mean()), 2.0 * np.pi)
    return pts[np.argsort(ang, kind="stable")]


def fvm_forces(path, alpha_deg):
    """(Cl, Cd) from an FVM Cp file: load, order the contour, face Cp = mean of neighbouring nodes (wrap), integrate."""
    pts = order_contour(load_fvm(path))
    x, y, cp = pts[:, 0], pts[:, 1], pts[:, 2]
    cp_faces = 0.5 * (cp + np.roll(cp, -1))
    return forces_from_faces(np.r_[x, x[0]], np.r_[y, y[0]], cp_faces, alpha_deg)


def append_summary(row: dict, path="results/summary.csv") -> None:
    """Add (or replace, matched on experiment/case/method/mesh) one row of results/summary.csv."""
    unknown = set(row) - set(SUMMARY_COLUMNS)
    if unknown:
        raise ValueError(f"unknown summary columns: {sorted(unknown)}")
    path = Path(path)
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf8") as f:
            rows = [r for r in csv.DictReader(f) if tuple(r[k] for k in _KEY) != tuple(str(row.get(k, "")) for k in _KEY)]
    rows.append({k: ("" if row.get(k) is None else row.get(k)) for k in SUMMARY_COLUMNS})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(rows)
