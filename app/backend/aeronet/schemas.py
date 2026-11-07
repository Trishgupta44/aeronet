"""Request and response models of the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .geometry import validate_code

Point = tuple[float, float, float]  # (x/c, y/c, Cp)


class SolveRequest(BaseModel):
    solver: str = Field("panel", description="Solver id, see GET /api/solvers")
    code: str = Field(..., description="NACA 4-digit code, e.g. 2412")
    alpha: float = Field(4.0, ge=-20, le=20, description="Angle of attack in degrees")
    mach: float = Field(0.4, ge=0.0, le=0.7, description="Free-stream Mach number (subsonic only)")
    n_half: int = Field(70, ge=20, le=200, description="Panels per surface")
    grid_width: int = Field(0, ge=0, le=400, description="Pressure-field grid width in cells; 0 = no field")
    reference: list[Point] | None = Field(None, max_length=20000, description="Reference Cp points to compare with")

    @field_validator("code")
    @classmethod
    def _check_code(cls, v: str) -> str:
        return validate_code(v)


class SurfaceData(BaseModel):
    cx: list[float]
    cy: list[float]
    cp: list[float]
    upper: list[bool]


class FieldData(BaseModel):
    width: int
    height: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    cp: list[float | None]
    u: list[float | None]
    v: list[float | None]


class Comparison(BaseModel):
    rms: float | None
    max: float | None
    n: int


class SolveResponse(BaseModel):
    solver: str
    code: str
    alpha: float
    mach: float
    n_panels: int
    x: list[float]
    y: list[float]
    surface: SurfaceData
    cl: float
    cd: float
    cl_thin: float | None = Field(None, description="Thin-airfoil Cl; null for drawn shapes (no camber line)")
    beta: float
    solve_ms: float = Field(..., description="Time spent inside the solver (server side)")
    total_ms: float = Field(..., description="Time to build the whole response (server side)")
    field: FieldData | None = None
    comparison: Comparison | None = None
    ground_y: float | None = Field(None, description="Height of the road line when ground effect is on")


class ShapeSolveRequest(BaseModel):
    points: list[tuple[float, float]] = Field(..., min_length=8, max_length=4000, description="Closed outline (x, y), any size")
    alpha: float = Field(4.0, ge=-20, le=20, description="Angle of attack in degrees")
    mach: float = Field(0.0, ge=0.0, le=0.7, description="Free-stream Mach number (subsonic only)")
    lifting: bool = Field(True, description="Apply the Kutta condition (off for bluff bodies)")
    ground: float | None = Field(None, ge=0.02, le=1.0, description="Gap between the lowest body point and the road, in chords")
    n_panels: int = Field(200, ge=40, le=400, description="Panels around the outline")
    grid_width: int = Field(0, ge=0, le=400, description="Pressure-field grid width in cells; 0 = no field")
    reference: list[Point] | None = Field(None, max_length=20000, description="Reference Cp points to compare with")


class ShapeInfo(BaseModel):
    chord: float = Field(..., description="Length of the outline along x before scaling to 1")
    thickness: float = Field(..., description="Height range after scaling to length 1")
    area: float
    n_input: int


class ShapeSolveResponse(SolveResponse):
    shape: ShapeInfo
    warnings: list[str] = []


class ParseRequest(BaseModel):
    text: str = Field(..., max_length=2_000_000)
    name: str = "uploaded file"


class ReferenceData(BaseModel):
    name: str
    count: int
    points: list[Point]
    condition: dict | None = None


class SolverInfo(BaseModel):
    id: str
    label: str
    available: bool
    note: str = ""


class AirfoilPreset(BaseModel):
    code: str
    label: str
