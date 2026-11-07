"""Hess-Smith panel method (incompressible, inviscid, 2D) with a Prandtl-Glauert Mach correction.

Each flat panel carries a constant source strength; one vortex strength is shared by all panels.
The unknowns follow from (a) zero normal velocity at every panel centre and (b) the Kutta
condition at the trailing edge. This is a *stand-in* for the NNfoil PINN: real physics, but not
the neural network.

Local panel frame: x along the panel, y = left of the panel direction (the airfoil interior for
counter-clockwise node order). The outward normal is the right-hand normal (ty, -tx).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .geometry import Airfoil, naca4

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class PanelGeometry:
    x0: np.ndarray
    y0: np.ndarray
    length: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    cx: np.ndarray
    cy: np.ndarray

    @property
    def n(self) -> int:
        return self.x0.size


def panel_geometry(X: np.ndarray, Y: np.ndarray) -> PanelGeometry:
    x1, y1 = np.roll(X, -1), np.roll(Y, -1)
    dx, dy = x1 - X, y1 - Y
    length = np.hypot(dx, dy)
    tx, ty = dx / length, dy / length
    return PanelGeometry(X, Y, length, tx, ty, ty, -tx, 0.5 * (X + x1), 0.5 * (Y + y1))


def influence(P: PanelGeometry, px: np.ndarray, py: np.ndarray, self_diag: bool = False):
    """Velocity induced at points (px, py) by every panel, for unit source and unit vortex strength.

    Returns four (M, N) arrays: source (x, y) and vortex (x, y) velocity components in global axes.
    With ``self_diag`` the points are the panel centres and the diagonal is set to the limit on the
    exterior (fluid) side of each panel.
    """
    rx = px[:, None] - P.x0[None, :]
    ry = py[:, None] - P.y0[None, :]
    xl = rx * P.tx + ry * P.ty
    yl = -rx * P.ty + ry * P.tx
    with np.errstate(divide="ignore", invalid="ignore"):
        us = np.log((xl**2 + yl**2) / ((xl - P.length) ** 2 + yl**2)) / (2 * TWO_PI)
        vs = (np.arctan2(yl, xl - P.length) - np.arctan2(yl, xl)) / TWO_PI
    if self_diag:
        idx = np.arange(P.n)
        us[idx, idx] = 0.0
        vs[idx, idx] = -0.5
    ux, uy = -P.ty, P.tx  # local y axis in global coordinates
    return (
        us * P.tx + vs * ux,  # source, x
        us * P.ty + vs * uy,  # source, y
        -vs * P.tx + us * ux,  # vortex, x
        -vs * P.ty + us * uy,  # vortex, y
    )


def influence_with_ground(P: PanelGeometry, px, py, ground_y: float | None, self_diag: bool = False):
    """``influence`` plus the image flow of a flat road at height ``ground_y`` (method of images).

    Reflecting the whole flow in the road line gives the image flow, whose velocity at (x, y) is
    (u, -v) of the real flow at the mirrored point (x, 2g - y); the road then has zero normal velocity.
    This holds for sources and for vortices alike (a reflected vortex reverses its circulation).
    """
    sx, sy, vx, vy = influence(P, px, py, self_diag=self_diag)
    if ground_y is None:
        return sx, sy, vx, vy
    mx, my, mvx, mvy = influence(P, px, 2.0 * ground_y - py)  # real panels seen from the mirrored point
    return sx + mx, sy - my, vx + mvx, vy - mvy  # image flow = (u, -v) at the mirror point


@dataclass(frozen=True)
class PanelSolution:
    airfoil: Airfoil | None  # None for drawn shapes
    geom: PanelGeometry
    sigma: np.ndarray
    gamma: float
    cp: np.ndarray  # surface Cp at panel centres
    cl: float
    cd: float
    beta: float
    alpha_deg: float
    vx_inf: float
    vy_inf: float
    ms: float  # time spent in solve()
    X: np.ndarray  # body nodes actually solved (rotated when a road is present)
    Y: np.ndarray
    upper: np.ndarray  # True for panels between the trailing edge and the leading-edge (min-x) node
    ground_y: float | None = None  # height of the road line, if any

    @property
    def n(self) -> int:
        return self.geom.n


def solve_xy(
    X: np.ndarray,
    Y: np.ndarray,
    alpha_deg: float,
    mach: float,
    lifting: bool = True,
    ground: float | None = None,
    airfoil: Airfoil | None = None,
) -> PanelSolution:
    """Solve the flow around the closed outline (``X``, ``Y``) at ``alpha_deg`` degrees and ``mach``.

    The nodes are taken as given: counter-clockwise, starting at the trailing edge, last node not
    repeated. ``lifting=False`` drops the vortex and the Kutta condition (sources only, so no
    circulation). ``ground`` is the gap to a road below the body (see ``influence_with_ground``).
    """
    t0 = perf_counter()
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    upper = np.arange(X.size) < int(np.argmin(X))  # panels between the trailing edge and the leading-edge node
    flow = np.radians(alpha_deg)  # freestream angle
    ground_y = None
    if ground is not None:
        # The freestream must be parallel to the road: tilt the body nose-up about the quarter chord instead.
        c, s = np.cos(-flow), np.sin(-flow)
        dx = X - 0.25
        X, Y = 0.25 + dx * c - Y * s, dx * s + Y * c
        ground_y = float(Y.min() - ground)
        flow = 0.0
    P = panel_geometry(X, Y)
    N = P.n
    vx_inf, vy_inf = np.cos(flow), np.sin(flow)

    sx, sy, vx, vy = influence_with_ground(P, P.cx, P.cy, ground_y, self_diag=True)
    nx, ny, tx, ty = P.nx[:, None], P.ny[:, None], P.tx[:, None], P.ty[:, None]
    normal_s = sx * nx + sy * ny
    normal_v = vx * nx + vy * ny
    tang_s = sx * tx + sy * ty
    tang_v = vx * tx + vy * ty
    rhs = -(vx_inf * P.nx + vy_inf * P.ny)

    if lifting:
        A = np.zeros((N + 1, N + 1))
        b = np.zeros(N + 1)
        A[:N, :N] = normal_s
        A[:N, N] = normal_v.sum(axis=1)
        b[:N] = rhs
        # Kutta condition: tangential speeds at the two trailing-edge panels cancel
        A[N, :N] = tang_s[0] + tang_s[-1]
        A[N, N] = (tang_v[0] + tang_v[-1]).sum()
        b[N] = -(vx_inf * (P.tx[0] + P.tx[-1]) + vy_inf * (P.ty[0] + P.ty[-1]))
        unknowns = np.linalg.solve(A, b)
        sigma, gamma = unknowns[:N], float(unknowns[N])
    else:
        sigma, gamma = np.linalg.solve(normal_s, rhs), 0.0

    vt = vx_inf * P.tx + vy_inf * P.ty + tang_s @ sigma + gamma * tang_v.sum(axis=1)
    beta = float(np.sqrt(1.0 - mach * mach))
    cp = (1.0 - vt**2) / beta

    fx = -np.sum(cp * P.nx * P.length)
    fy = -np.sum(cp * P.ny * P.length)
    cl = float(-fx * np.sin(flow) + fy * np.cos(flow))
    cd = float(fx * np.cos(flow) + fy * np.sin(flow))  # ideal value 0 (d'Alembert)

    ms = (perf_counter() - t0) * 1000.0
    return PanelSolution(
        airfoil, P, sigma, gamma, cp, cl, cd, beta, alpha_deg, float(vx_inf), float(vy_inf), ms, X, Y, upper, ground_y
    )


def solve(code: str, alpha_deg: float, mach: float, n_half: int = 70) -> PanelSolution:
    """Solve the flow around NACA ``code`` at ``alpha_deg`` degrees and Mach number ``mach``."""
    t0 = perf_counter()
    af = naca4(code, n_half)
    sol = solve_xy(af.x, af.y, alpha_deg, mach, airfoil=af)
    return dataclasses.replace(sol, ms=(perf_counter() - t0) * 1000.0)  # timing includes the geometry
