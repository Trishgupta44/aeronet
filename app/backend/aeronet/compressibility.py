"""Where the inviscid subsonic model stops being valid: critical pressure coefficient and critical Mach number.

Everything here is computed from the panel solution (Cp_min) with textbook isentropic relations (gamma = 1.4):

* Cp*(M) is the pressure coefficient at which the local flow just reaches Mach 1 when the free stream is Mach M:
      Cp*(M) = 2 / (gamma M^2) * [ ((1 + (gamma-1)/2 M^2) / ((gamma+1)/2))^(gamma/(gamma-1)) - 1 ]
* If the lowest Cp on the surface is below Cp*(M), the flow is locally supersonic (shocks are possible) and the
  subsonic, shock-free model of this app no longer applies.
* The critical Mach number Mcr is the free-stream Mach number where the (compressibility-corrected) Cp_min meets Cp*.
  Two corrections are used, both from the incompressible Cp_min:
      Prandtl-Glauert   Cp = Cp0 / beta
      Karman-Tsien      Cp = Cp0 / (beta + M^2 / (1 + beta) * Cp0 / 2)
  Prandtl-Glauert gives the smallest growth of suction with Mach number; Karman-Tsien gives larger suction, so its
  Mcr is lower. The app's solver uses Prandtl-Glauert, so its Mcr is the less conservative of the two.

Source for the relations: J. D. Anderson, Fundamentals of Aerodynamics, chapter 11 (compressible flow over airfoils).
"""

from __future__ import annotations

import math

import numpy as np

GAMMA = 1.4
_M_LO, _M_HI = 0.01, 0.999


def cp_crit(mach: float) -> float | None:
    """Cp* for free-stream Mach ``mach`` (None for mach ~ 0, where it tends to minus infinity)."""
    if mach < 1e-6:
        return None
    g = GAMMA
    ratio = (1.0 + 0.5 * (g - 1.0) * mach * mach) / (0.5 * (g + 1.0))
    return 2.0 / (g * mach * mach) * (ratio ** (g / (g - 1.0)) - 1.0)


def prandtl_glauert(cp0: float, mach: float) -> float:
    return cp0 / math.sqrt(1.0 - mach * mach)


def karman_tsien(cp0: float, mach: float) -> float:
    beta = math.sqrt(1.0 - mach * mach)
    return cp0 / (beta + mach * mach / (1.0 + beta) * cp0 / 2.0)


def critical_mach(cp0_min: float, correction=prandtl_glauert) -> float | None:
    """Free-stream Mach at which the corrected Cp_min equals Cp* (bisection). None when there is no suction
    (Cp0_min >= 0) or the crossing lies above Mach 0.999."""
    if cp0_min >= 0.0:
        return None

    def gap(m: float) -> float:  # > 0 while the flow is still subsonic everywhere
        try:
            cp = correction(cp0_min, m)
        except ZeroDivisionError:
            return -math.inf
        if cp > 0.0:  # Karman-Tsien's denominator changed sign: the rule has broken down, count it as crossed
            return -math.inf
        return cp - cp_crit(m)

    # Coarse scan for the first Mach number at which the gap is no longer positive, then bisect inside that step.
    grid = np.linspace(_M_LO, _M_HI, 200)
    lo = None
    for m in grid:
        if gap(float(m)) <= 0.0:
            break
        lo = float(m)
    else:
        return None  # still subsonic everywhere up to Mach 0.999
    if lo is None:
        return _M_LO
    hi = float(m)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) > 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


def validity(cp: np.ndarray, beta: float) -> dict:
    """Validity numbers for a panel solution: ``cp`` is the corrected surface Cp, ``beta`` = sqrt(1 - M^2)."""
    mach = math.sqrt(max(0.0, 1.0 - beta * beta))
    cp_min = float(np.min(cp))
    cp0_min = cp_min * beta  # undo the Prandtl-Glauert factor the solver applied
    star = cp_crit(mach)
    return {
        "cp_min": cp_min,
        "cp0_min": cp0_min,
        "cp_crit": star,
        "supersonic": bool(star is not None and cp_min < star),
        "mach_crit_pg": critical_mach(cp0_min, prandtl_glauert),
        "mach_crit_kt": critical_mach(cp0_min, karman_tsien),
    }
