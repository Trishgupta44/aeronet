"""FastAPI application: solver API, plus the built React frontend when it exists.

Run from ``app/backend``:  ``uvicorn aeronet.main:app --reload``
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import __version__
from .compare import compare_cp, parse_reference, thin_airfoil_cl
from .field import field_grid
from .schemas import (
    AirfoilPreset,
    Comparison,
    FieldData,
    ParseRequest,
    ReferenceData,
    SolveRequest,
    SolveResponse,
    SolverInfo,
    SurfaceData,
)
from .solvers import PRESETS, SOLVERS, get_solver

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"
SAMPLE_FILE = DATA_DIR / "reference_naca0012_200x100.dat"

app = FastAPI(title="AeroNet API", version=__version__, description="Airfoil flow solver (panel-method stand-in for the NNfoil PINN).")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _points(arr: np.ndarray) -> list[tuple[float, float, float]]:
    return [(float(x), float(y), float(c)) for x, y, c in arr]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/solvers", response_model=list[SolverInfo])
def list_solvers() -> list[SolverInfo]:
    return [s.info for s in SOLVERS.values()]


@app.get("/api/airfoils", response_model=list[AirfoilPreset])
def list_airfoils() -> list[AirfoilPreset]:
    return PRESETS


@app.post("/api/solve", response_model=SolveResponse)
def solve(req: SolveRequest) -> SolveResponse:
    t0 = perf_counter()
    try:
        solver = get_solver(req.solver)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'\""))
    except LookupError as e:
        raise HTTPException(status_code=409, detail=str(e))

    sol = solver.solve(req.code, req.alpha, req.mach, req.n_half)
    g = sol.geom

    comparison = None
    if req.reference:
        comparison = Comparison(**compare_cp(sol, np.asarray(req.reference, dtype=float)))

    field = FieldData(**field_grid(sol, req.grid_width)) if req.grid_width > 0 else None

    return SolveResponse(
        solver=req.solver,
        code=req.code,
        alpha=req.alpha,
        mach=req.mach,
        n_panels=sol.n,
        x=sol.X.tolist(),
        y=sol.Y.tolist(),
        surface=SurfaceData(cx=g.cx.tolist(), cy=g.cy.tolist(), cp=sol.cp.tolist(), upper=sol.upper.tolist()),
        cl=sol.cl,
        cd=sol.cd,
        cl_thin=thin_airfoil_cl(sol.airfoil, req.alpha, req.mach),
        beta=sol.beta,
        solve_ms=sol.ms,
        total_ms=(perf_counter() - t0) * 1000.0,
        field=field,
        comparison=comparison,
    )


@app.post("/api/reference/parse", response_model=ReferenceData)
def parse_reference_file(req: ParseRequest) -> ReferenceData:
    pts = parse_reference(req.text)
    if pts.size == 0:
        raise HTTPException(status_code=400, detail='Could not read any "node x y Cp" rows from that file.')
    return ReferenceData(name=req.name, count=len(pts), points=_points(pts))


@app.get("/api/reference/sample", response_model=ReferenceData)
def sample_reference() -> ReferenceData:
    pts = parse_reference(SAMPLE_FILE.read_text(encoding="utf8"))
    return ReferenceData(
        name="NACA0012 200×100 FVM sample",
        count=len(pts),
        points=_points(pts),
        condition={"code": "0012", "alpha": 4.0, "mach": 0.4},
    )


# The built frontend (npm run build) is served at "/" when present; mounted last so /api wins.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
