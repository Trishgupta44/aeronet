"""Viscous airfoil estimate from NeuralFoil (Peter Sharpe, MIT licence): a neural network trained on XFOIL results.

This is NOT the app's own solver. It is shown next to the inviscid panel result to explain what the inviscid model leaves
out: skin friction and boundary-layer displacement, which give real drag and a lower lift slope. NeuralFoil is optional:
without ``pip install neuralfoil`` (see requirements-viscous.txt) the API answers 503 and the page says so.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .geometry import naca4
from .panel import solve as panel_solve

try:  # optional dependency
    import neuralfoil as _nf
except ImportError:  # pragma: no cover - depends on the environment
    _nf = None

LABEL = "NeuralFoil estimate (learned from XFOIL), viscous"
POLAR_ALPHAS = np.arange(-6.0, 14.01, 1.0)
_TIMING_CALLS = 5


def available() -> bool:
    return _nf is not None


def version() -> str | None:
    return getattr(_nf, "__version__", None) if _nf is not None else None


def _scalar(result: dict, key: str) -> float:
    return float(np.asarray(result[key]).ravel()[0])


def viscous(code: str, alpha: float, re: float, model_size: str = "large") -> dict:
    """NeuralFoil Cl, Cd, Cm, transition and confidence for NACA ``code`` at ``alpha`` degrees and Reynolds number ``re``,
    a Cl/Cd polar over -6..14 degrees, the inviscid panel result for comparison, and the measured time per NeuralFoil call.

    The panel comparison is at Mach 0 because NeuralFoil is an incompressible model (no Mach input is used).
    """
    if _nf is None:
        raise RuntimeError("NeuralFoil is not installed")
    af = naca4(code, 70)
    xy = np.c_[af.x, af.y]  # trailing edge, upper surface, leading edge, lower surface: the order NeuralFoil expects

    def call(a):
        return _nf.get_aero_from_coordinates(xy, alpha=a, Re=re, model_size=model_size)

    one = call(alpha)  # first call also warms the model up
    t0 = perf_counter()
    for _ in range(_TIMING_CALLS):
        call(alpha)
    call_ms = (perf_counter() - t0) / _TIMING_CALLS * 1000.0

    pol = call(POLAR_ALPHAS)
    inviscid = panel_solve(code, alpha, 0.0, 50)
    cl_inv = [panel_solve(code, float(a), 0.0, 50).cl for a in POLAR_ALPHAS]
    cd = _scalar(one, "CD")
    return {
        "label": LABEL,
        "version": version(),
        "model_size": model_size,
        "code": code,
        "alpha": alpha,
        "re": re,
        "cl": _scalar(one, "CL"),
        "cd": cd,
        "cm": _scalar(one, "CM"),
        "cl_cd": _scalar(one, "CL") / cd if cd > 0 else None,
        "confidence": _scalar(one, "analysis_confidence"),
        "top_xtr": _scalar(one, "Top_Xtr"),
        "bot_xtr": _scalar(one, "Bot_Xtr"),
        "call_ms": call_ms,
        "inviscid": {"cl": inviscid.cl, "cd": inviscid.cd, "mach": 0.0},
        "polar": {
            "alpha": POLAR_ALPHAS.tolist(),
            "cl": np.asarray(pol["CL"]).ravel().tolist(),
            "cd": np.asarray(pol["CD"]).ravel().tolist(),
            "confidence": np.asarray(pol["analysis_confidence"]).ravel().tolist(),
            "cl_inviscid": cl_inv,
        },
    }
