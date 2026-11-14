"""FastAPI application: solver API, plus the built React frontend when it exists.

Run from ``app/backend``:  ``uvicorn aeronet.main:app --reload``
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, compressibility, config, limits, neural, optimize, pinn_cases, pinn_data, tradeoff
from .compare import compare_cp, compare_cp_nearest, parse_reference, thin_airfoil_cl
from .field import field_grid
from .geometry import naca4
from .panel import PanelSolution, solve as panel_solve, solve_xy
from .shapes import library, parse_dat, prepare_shape
from .schemas import (
    AirfoilPreset,
    DatData,
    Comparison,
    FieldData,
    ParseRequest,
    PolarRequest,
    PolarResponse,
    ReferenceData,
    ShapeInfo,
    ShapePreset,
    ShapeSolveRequest,
    ShapeSolveResponse,
    SolveRequest,
    SolveResponse,
    SolverInfo,
    SurfaceData,
    OptimizeRequest,
    Validity,
    ViscousRequest,
    ViscousResponse,
)
from .solvers import PRESETS, get_solver, solver_infos

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = pinn_cases.DATA_DIR  # app/backend/data, or AERONET_DATA_DIR (the public build points it at the stripped copy)
FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"
SAMPLE_FILE = DATA_DIR / "reference_naca0012_200x100.dat"

SHAPES = library()  # presets are built once, at import time

app = FastAPI(title="AeroNet API", version=__version__, description="Airfoil flow solver (panel-method stand-in for the NNfoil PINN).")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _points(arr: np.ndarray) -> list[tuple[float, float, float]]:
    return [(float(x), float(y), float(c)) for x, y, c in arr]


def _response_fields(sol: PanelSolution, grid_width: int, comparison: Comparison | None, t0: float) -> dict:
    """Fields shared by /api/solve and /api/solve_shape."""
    g = sol.geom
    field = FieldData(**field_grid(sol, grid_width)) if grid_width > 0 else None
    return dict(
        n_panels=sol.n,
        x=sol.X.tolist(),
        y=sol.Y.tolist(),
        surface=SurfaceData(cx=g.cx.tolist(), cy=g.cy.tolist(), cp=sol.cp.tolist(), upper=sol.upper.tolist()),
        cl=sol.cl,
        cd=sol.cd,
        beta=sol.beta,
        solve_ms=sol.ms,
        total_ms=(perf_counter() - t0) * 1000.0,
        field=field,
        comparison=comparison,
        ground_y=sol.ground_y,
        validity=Validity(**compressibility.validity(sol.cp, sol.beta)),
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/solvers", response_model=list[SolverInfo])
def list_solvers() -> list[SolverInfo]:
    return solver_infos()


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
    comparison = None
    if req.reference:
        comparison = Comparison(**compare_cp(sol, np.asarray(req.reference, dtype=float)))

    return SolveResponse(
        solver=req.solver,
        code=req.code,
        alpha=req.alpha,
        mach=req.mach,
        cl_thin=thin_airfoil_cl(sol.airfoil, req.alpha, req.mach),
        **_response_fields(sol, req.grid_width, comparison, t0),
    )


@app.post("/api/solve_shape", response_model=ShapeSolveResponse)
def solve_shape(req: ShapeSolveRequest) -> ShapeSolveResponse:
    """Panel solution for a closed outline of any shape (see roadmap section 5.6)."""
    t0 = perf_counter()
    try:
        X, Y, info, warnings = prepare_shape(req.points, n=req.n_panels, lifting=req.lifting)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    sol = solve_xy(X, Y, req.alpha, req.mach, lifting=req.lifting, ground=req.ground)
    comparison = None
    if req.reference:
        comparison = Comparison(**compare_cp_nearest(sol, np.asarray(req.reference, dtype=float)))

    return ShapeSolveResponse(
        solver="panel",
        code="custom",
        alpha=req.alpha,
        mach=req.mach,
        cl_thin=None,
        shape=ShapeInfo(**info),
        warnings=warnings,
        **_response_fields(sol, req.grid_width, comparison, t0),
    )


@app.post("/api/polar", response_model=PolarResponse)
def polar(req: PolarRequest) -> PolarResponse:
    """Cl for a list of angles of attack (no field is computed)."""
    if req.code is not None:
        if req.ground is None and req.lifting:
            sols = [panel_solve(req.code, a, req.mach, req.n_panels // 2) for a in req.alphas]
        else:  # road and/or no Kutta condition: same NACA nodes through the generic solver
            af = naca4(req.code, req.n_panels // 2)
            sols = [solve_xy(af.x, af.y, a, req.mach, lifting=req.lifting, ground=req.ground, airfoil=af) for a in req.alphas]
        cl_thin = [thin_airfoil_cl(s.airfoil, a, req.mach) for s, a in zip(sols, req.alphas)]
    else:
        try:
            X, Y, _, _ = prepare_shape(req.points, n=req.n_panels, lifting=req.lifting)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        sols = [solve_xy(X, Y, a, req.mach, lifting=req.lifting, ground=req.ground) for a in req.alphas]
        cl_thin = None
    return PolarResponse(alpha=req.alphas, cl=[s.cl for s in sols], cl_thin=cl_thin)


@app.get("/api/capabilities")
def capabilities() -> dict:
    """What this deployment offers, so the page can hide what is missing (see config.py for the switches)."""
    withheld = config.withhold_fvm_data()
    return {
        "fvm_sample": (not withheld) and SAMPLE_FILE.exists(),  # the bundled NNfoil FVM sample and the overlay that uses it
        "fvm_curves": not withheld,  # FVM Cp curves in the PINN showdown (the error numbers are always shown)
        "neuralfoil": neural.available(),
        "public_build": withheld,
    }


@app.post("/api/viscous", response_model=ViscousResponse, dependencies=[Depends(limits.heavy_guard)])
def solve_viscous(req: ViscousRequest) -> dict:
    """NeuralFoil viscous Cl/Cd for a NACA airfoil, next to the inviscid panel result (optional dependency)."""
    if not neural.available():
        raise HTTPException(status_code=503, detail="NeuralFoil is not installed on this server (pip install -r requirements-viscous.txt).")
    return neural.viscous(req.code, req.alpha, req.re, req.model_size)


@app.post("/api/optimize", dependencies=[Depends(limits.heavy_guard)])
def optimize_shape(req: OptimizeRequest) -> dict:
    """Shape a NACA 4-digit-family airfoil for the least suction at the target Cl (raises the critical Mach number). Inviscid."""
    return optimize.optimise(req.target_cl, req.min_thickness, req.code)


@app.get("/api/tradeoff")
def get_tradeoff() -> dict:
    """Accuracy (Cp RMS vs FVM 800x400) against cost (seconds on this laptop) for panel, PINN and FVM (read from results/)."""
    try:
        return tradeoff.tradeoff()
    except (OSError, KeyError, ValueError) as e:
        raise HTTPException(status_code=503, detail=f"Trade-off data is not available here: {e}")


@app.get("/api/showdown")
def get_showdown() -> dict:
    """Same-mesh showdown: FVM vs PINN on each mesh, both against the FVM 800x400 reference (read from results/)."""
    try:
        return pinn_data.showdown()
    except (OSError, KeyError, ValueError) as e:
        raise HTTPException(status_code=503, detail=f"PINN results are not available here: {e}")


@app.get("/api/pinn/cases")
def list_pinn_cases() -> list[dict]:
    """The trained PINN cases (one network per flow condition): id, label, shape, mesh, Mach, angle, Cp RMS, Cl, collapsed flag, source."""
    return pinn_cases.list_cases()


@app.get("/api/pinn/cases/{case_id}")
def get_pinn_case(case_id: str):
    """One case (roadmap 5.7): surface Cp, reference, panel solution, metrics, timing, log, per-round snapshots and the flow field.
    Public mode (AERONET_WITHHOLD_FVM_DATA=1): the same without the FVM Cp curves (``reference``, ``fvm_same_mesh`` points); the numbers stay."""
    try:
        path = pinn_cases.case_path(case_id)  # checked against the files that exist: no path can be reached from the id
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'\""))
    if config.withhold_fvm_data():
        return JSONResponse(pinn_cases.get_case(case_id))
    return FileResponse(path, media_type="application/json")


@app.get("/api/results")
def get_results() -> list[dict]:
    """The rows of results/summary.csv as exported (every number measured, quoted or fitted: see each row's ``source``)."""
    return pinn_cases.load_results()


@app.get("/api/results/e1_rounds")
def get_e1_rounds() -> dict:
    """Round-by-round Cp RMS and Cl of the six E1 runs (50x25, positivity on or off, seeds 0 to 2), from results/e1_rounds.csv."""
    data = pinn_cases.load_rounds_e1()
    if not data:
        raise HTTPException(status_code=503, detail="The E1 round-by-round data is not available on this server.")
    return data


@app.get("/api/shapes", response_model=list[ShapePreset])
def list_shapes() -> list[dict]:
    return SHAPES


@app.post("/api/shapes/parse_dat", response_model=DatData)
def parse_dat_file(req: ParseRequest) -> DatData:
    """Read a UIUC Selig or Lednicer airfoil file into an outline for /api/solve_shape."""
    try:
        name, points = parse_dat(req.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return DatData(name=name or req.name, points=points)


@app.post("/api/reference/parse", response_model=ReferenceData)
def parse_reference_file(req: ParseRequest) -> ReferenceData:
    pts = parse_reference(req.text)
    if pts.size == 0:
        raise HTTPException(status_code=400, detail='Could not read any "node x y Cp" rows from that file.')
    return ReferenceData(name=req.name, count=len(pts), points=_points(pts))


@app.get("/api/reference/sample", response_model=ReferenceData)
def sample_reference() -> ReferenceData:
    if config.withhold_fvm_data() or not SAMPLE_FILE.exists():
        raise HTTPException(status_code=404, detail="The bundled FVM sample is not available in this deployment. You can still load your own *_Cp.dat file.")
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
