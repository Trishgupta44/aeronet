"""Beat the critical Mach number: shape a NACA 4-digit-family airfoil for the least suction at a given lift.

What is optimised: the three parameters of the family (camber m, camber position p, thickness t) so that, at the angle of attack
that gives the target Cl, the lowest surface Cp is as high (least negative) as possible. The critical Mach number of an airfoil
rises when its suction peak falls, so this raises Mcr (by the Prandtl-Glauert rule, and by Karman-Tsien as a second estimate).

What it is NOT: it is an INVISCID panel-method optimisation. There is no drag, no boundary layer, no stall, no structural or
manufacturing limit and no check of shock strength above Mcr; the result is a screening idea, not a finished design.

Method: the panel solver is linear in the angle of attack for the lift, so two solves give the angle that hits the target Cl and
a third gives the Cp there (about 10 ms per candidate). A bounded Nelder-Mead search runs from several starting shapes.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from . import compressibility
from .geometry import Airfoil, _thickness, validate_code
from .panel import solve_xy

LABEL = "Inviscid panel-method optimisation, no drag"
BOUNDS = {"m": (0.0, 0.06), "p": (0.2, 0.7), "t": (0.06, 0.18)}  # t's lower bound is replaced by the user's minimum thickness
ALPHA_LIMIT = 15.0  # degrees: designs that need more to reach the target lift are rejected
N_HALF_SEARCH = 50
N_HALF_REPORT = 70
MAX_EVALS_PER_START = 90
TIME_LIMIT_S = 12.0


def build(m: float, p: float, t: float, n_half: int) -> Airfoil:
    """Same construction as geometry.naca4 but with continuous parameters (checked against it in the tests)."""
    proto = Airfoil("custom", m, p, t, np.empty(0), np.empty(0))
    k = np.arange(n_half + 1)
    x = 0.5 * (1 + np.cos(np.pi * k / n_half))
    yc, phi, yt = proto.camber(x), np.arctan(proto.dcamber(x)), _thickness(t, x)
    xu, yu = x - yt * np.sin(phi), yc + yt * np.cos(phi)
    xl, yl = x + yt * np.sin(phi), yc - yt * np.cos(phi)
    lower = slice(n_half - 1, 0, -1)
    return Airfoil("custom", m, p, t, np.concatenate([xu, xl[lower]]), np.concatenate([yu, yl[lower]]))


def design_at_cl(m: float, p: float, t: float, target_cl: float, n_half: int = N_HALF_SEARCH) -> dict:
    """Angle of attack that gives ``target_cl`` (incompressible, Kutta condition) and the surface Cp there."""
    af = build(m, p, t, n_half)
    s0 = solve_xy(af.x, af.y, 0.0, 0.0, airfoil=af)
    s5 = solve_xy(af.x, af.y, 5.0, 0.0, airfoil=af)
    slope = (s5.cl - s0.cl) / 5.0  # Cl is linear in alpha for the panel method
    alpha = (target_cl - s0.cl) / slope
    sol = solve_xy(af.x, af.y, alpha, 0.0, airfoil=af)
    for _ in range(2):  # Cl from the pressure integral is only almost linear in alpha: one or two corrections reach the target
        err = target_cl - sol.cl
        if abs(err) < 1e-6:
            break
        alpha += err / slope
        sol = solve_xy(af.x, af.y, alpha, 0.0, airfoil=af)
    return {"af": af, "alpha": float(alpha), "cl": sol.cl, "cp_min": float(sol.cp.min()), "sol": sol}


def _objective(x, target_cl, tmin) -> float:
    m, p, t = x
    d = design_at_cl(m, p, t, target_cl)
    pen = max(0.0, abs(d["alpha"]) - ALPHA_LIMIT) * 0.5
    return -d["cp_min"] + pen  # minimise the suction (Cp_min is negative)


def _clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def _nelder_mead(f, x0, lo, hi, max_evals, step, deadline):
    """Small bounded Nelder-Mead (points are clipped into the box). Returns (best x, best f, [(x, f) per evaluation])."""
    n = len(x0)
    pts = [np.array(x0, float)]
    for i in range(n):
        q = np.array(x0, float)
        q[i] = q[i] + step[i] if q[i] + step[i] <= hi[i] else q[i] - step[i]
        pts.append(_clip(q, lo, hi))
    trace = []

    def ev(q):
        q = _clip(q, lo, hi)
        v = f(q)
        trace.append((q.copy(), v))
        return v

    vals = [ev(q) for q in pts]
    while len(trace) < max_evals and perf_counter() < deadline:
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = ev(xr)
        if fr < vals[0]:
            xe = c + 2.0 * (c - pts[-1])
            fe = ev(xe)
            pts[-1], vals[-1] = (_clip(xe, lo, hi), fe) if fe < fr else (_clip(xr, lo, hi), fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = _clip(xr, lo, hi), fr
        else:
            xc = c + 0.5 * (pts[-1] - c) if fr >= vals[-1] else c + 0.5 * (xr - c)
            fc = ev(xc)
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = _clip(xc, lo, hi), fc
            else:  # shrink towards the best point
                for i in range(1, n + 1):
                    pts[i] = _clip(pts[0] + 0.5 * (pts[i] - pts[0]), lo, hi)
                    vals[i] = ev(pts[i])
    best = int(np.argmin(vals))
    return pts[best], vals[best], trace


def describe(m: float, p: float, t: float, target_cl: float, name: str) -> dict:
    """Full report of one design at the target lift: parameters, angle, Cp_min, Mcr, outline and surface Cp."""
    d = design_at_cl(m, p, t, target_cl, N_HALF_REPORT)
    sol, af = d["sol"], d["af"]
    cp0 = d["cp_min"]
    return {
        "name": name,
        "m": m,
        "p": p,
        "t": t,
        "alpha": d["alpha"],
        "cl": d["cl"],
        "cp_min": cp0,
        "mach_crit_pg": compressibility.critical_mach(cp0, compressibility.prandtl_glauert),
        "mach_crit_kt": compressibility.critical_mach(cp0, compressibility.karman_tsien),
        "shape": {"x": af.x.tolist(), "y": af.y.tolist()},
        "surface": {"cx": sol.geom.cx.tolist(), "cp": sol.cp.tolist(), "upper": sol.upper.tolist()},
    }


def _digits(m: float, p: float, t: float) -> str:
    """Nearest standard 4-digit code (the closest real NACA airfoil to the optimised shape)."""
    md = int(round(m * 100))
    pd = int(round(p * 10)) if md > 0 else 0
    if md > 0 and pd < 1:
        pd = 1
    return f"{md}{pd}{max(4, int(round(t * 100))):02d}"


def optimise(target_cl: float, min_thickness: float, baseline_code: str = "0012") -> dict:
    validate_code(baseline_code)
    bm, bp, bt = int(baseline_code[0]) / 100, int(baseline_code[1]) / 10, int(baseline_code[2:]) / 100
    lo = np.array([BOUNDS["m"][0], BOUNDS["p"][0], min_thickness])
    hi = np.array([BOUNDS["m"][1], BOUNDS["p"][1], BOUNDS["t"][1]])
    t0 = perf_counter()
    deadline = t0 + TIME_LIMIT_S
    f = lambda x: _objective(x, target_cl, min_thickness)  # noqa: E731
    starts = [
        np.array([bm, max(bp, 0.2), bt]),
        np.array([0.02, 0.4, min_thickness]),
        np.array([0.04, 0.5, min_thickness]),
        np.array([0.0, 0.4, min_thickness + 0.02]),
    ]
    step = np.array([0.01, 0.1, 0.015])
    best, best_f, history, evals = None, np.inf, [], 0
    for s in starts:
        x, v, trace = _nelder_mead(f, _clip(s, lo, hi), lo, hi, MAX_EVALS_PER_START, step, deadline)
        for _, val in trace:
            evals += 1
            history.append(min(val, history[-1]) if history else val)
        if v < best_f:
            best, best_f = x, v
        if perf_counter() > deadline:
            break
    m, p, t = (float(v) for v in best)
    if m < 0.002:  # no camber: its position means nothing
        m, p = 0.0, 0.4
    before = describe(bm, bp, bt, target_cl, f"NACA {baseline_code}")
    after = describe(m, p, t, target_cl, "Optimised")
    code = _digits(m, p, t)
    rm, rp, rt = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    rounded = describe(rm, rp, rt, target_cl, f"NACA {code}")
    for d in (before, after, rounded):  # the outlines of the three designs are the same size class; keep the payload small
        d["shape"] = {k: [round(v, 5) for v in vs] for k, vs in d["shape"].items()}
        d["surface"] = {"cx": [round(v, 5) for v in d["surface"]["cx"]], "cp": [round(v, 5) for v in d["surface"]["cp"]], "upper": d["surface"]["upper"]}
    gain = None
    if before["mach_crit_pg"] is not None and after["mach_crit_pg"] is not None:
        gain = after["mach_crit_pg"] - before["mach_crit_pg"]
    return {
        "label": LABEL,
        "target_cl": target_cl,
        "min_thickness": min_thickness,
        "bounds": {"m": BOUNDS["m"], "p": BOUNDS["p"], "t": (min_thickness, BOUNDS["t"][1])},
        "before": before,
        "after": after,
        "rounded": rounded,
        "gain_pg": gain,
        "evaluations": evals,
        "elapsed_s": perf_counter() - t0,
        "history": [-v for v in history],  # best Cp_min found so far, per evaluation (rises towards 0)
        "limits": ("Inviscid panel method at Mach 0: no drag, boundary layer, stall, pitching-moment or structural limits. The critical Mach "
                   "number uses the Prandtl-Glauert rule (optimistic) and, as a second estimate, Karman-Tsien."),
        "active_bounds": [k for k, v, (a, b) in (("m", m, BOUNDS["m"]), ("p", p, BOUNDS["p"]), ("t", t, (min_thickness, BOUNDS["t"][1])))
                          if abs(v - a) < 1e-3 or abs(v - b) < 1e-3],
    }
