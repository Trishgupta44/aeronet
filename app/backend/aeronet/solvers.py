"""Registry of the solvers the API can use.

Only the panel method is available now. A trained NNfoil PINN would be added here as a second
entry (``available=True`` plus a ``solve`` function returning a ``PanelSolution``-like object);
the frontend reads this list from ``GET /api/solvers`` and needs no change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .panel import PanelSolution, solve as panel_solve
from .schemas import AirfoilPreset, SolverInfo


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
            available=False,
            note="PINN option unlocks when a trained checkpoint is added (plan Day 5).",
        ),
        None,
    ),
}

PRESETS = [
    AirfoilPreset(code="0012", label="NACA 0012"),
    AirfoilPreset(code="2412", label="NACA 2412"),
    AirfoilPreset(code="4412", label="NACA 4412"),
]


def get_solver(solver_id: str) -> Solver:
    solver = SOLVERS.get(solver_id)
    if solver is None:
        raise KeyError(f"Unknown solver '{solver_id}'.")
    if not solver.info.available or solver.solve is None:
        raise LookupError(f"Solver '{solver_id}' is not available yet: {solver.info.note}")
    return solver
