"""Registry of the solvers the API can use.

The panel method solves any airfoil on request. The NNfoil PINN does not: one network is trained per flow condition
(shape, Mach number, angle), so it cannot be asked for another condition. It is listed as available when trained cases
exist (``data/pinn_cases/``, see ``pinn_cases.py``) and is used by picking a case in the PINN Lab (``GET /api/pinn/cases``);
``POST /api/solve`` with ``solver=pinn`` is refused with a clear message instead of faking an interpolation between cases.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import pinn_cases
from .panel import PanelSolution, solve as panel_solve
from .schemas import AirfoilPreset, SolverInfo

PINN_NOTE = "Trained cases only: open the PINN Lab tab."
PINN_REFUSAL = (
    "The trained network answers only its trained condition (one model is trained per shape, Mach number and angle); "
    "pick a case in the PINN Lab (GET /api/pinn/cases) instead of solving an arbitrary input."
)


@dataclass(frozen=True)
class Solver:
    info: SolverInfo
    solve: Callable[[str, float, float, int], PanelSolution] | None


SOLVERS: dict[str, Solver] = {
    "panel": Solver(SolverInfo(id="panel", label="Panel method", available=True), panel_solve),
    "pinn": Solver(
        SolverInfo(
            id="pinn",
            label="NNfoil PINN",
            available=False,  # replaced at request time by solver_infos(): it depends on the exported data
            note=PINN_NOTE,
        ),
        None,
    ),
}

PRESETS = [
    AirfoilPreset(code="0012", label="NACA 0012"),
    AirfoilPreset(code="2412", label="NACA 2412"),
    AirfoilPreset(code="4412", label="NACA 4412"),
]


def solver_infos() -> list[SolverInfo]:
    """The solvers as ``GET /api/solvers`` reports them. ``pinn`` is available when at least one case file exists."""
    out = []
    for s in SOLVERS.values():
        if s.info.id == "pinn":
            has = pinn_cases.has_cases()
            note = PINN_NOTE if has else "No trained cases found (python research/export_results.py writes them to app/backend/data/pinn_cases)."
            out.append(s.info.model_copy(update={"available": has, "note": note}))
        else:
            out.append(s.info)
    return out


def get_solver(solver_id: str) -> Solver:
    solver = SOLVERS.get(solver_id)
    if solver is None:
        raise KeyError(f"Unknown solver '{solver_id}'.")
    if solver_id == "pinn":  # even with cases present: the network cannot solve an arbitrary input
        raise LookupError(PINN_REFUSAL)
    if not solver.info.available or solver.solve is None:
        raise LookupError(f"Solver '{solver_id}' is not available yet: {solver.info.note}")
    return solver
