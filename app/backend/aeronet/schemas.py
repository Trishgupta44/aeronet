"""Request and response models of the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

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


class Validity(BaseModel):
    """Is the inviscid subsonic model still valid for this solution? (computed from the panel Cp, see compressibility.py)"""

    cp_min: float = Field(..., description="Lowest surface Cp of this solution (Mach-corrected)")
    cp0_min: float = Field(..., description="The same, incompressible (Mach 0)")
    cp_crit: float | None = Field(None, description="Cp* at this free-stream Mach: local flow is sonic when Cp reaches it; null at Mach 0")
    supersonic: bool = Field(..., description="True when cp_min < cp_crit")
    mach_crit_pg: float | None = Field(None, description="Critical Mach number with the Prandtl-Glauert rule (the one the solver uses)")
    mach_crit_kt: float | None = Field(None, description="Critical Mach number with the Karman-Tsien rule (usually lower)")


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
    validity: Validity | None = None


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


class PolarRequest(BaseModel):
    code: str | None = Field(None, description="NACA 4-digit code")
    points: list[tuple[float, float]] | None = Field(None, min_length=8, max_length=4000, description="Closed outline (x, y)")
    alphas: list[float] = Field(..., min_length=1, max_length=61, description="Angles of attack in degrees")
    mach: float = Field(0.0, ge=0.0, le=0.7)
    lifting: bool = True
    ground: float | None = Field(None, ge=0.02, le=1.0, description="Gap to the road, in chords")
    n_panels: int = Field(160, ge=40, le=400)

    @field_validator("alphas")
    @classmethod
    def _check_alphas(cls, v: list[float]) -> list[float]:
        if any(not -20 <= a <= 20 for a in v):
            raise ValueError("Every angle must be between -20 and 20 degrees.")
        return v

    @model_validator(mode="after")
    def _one_of_code_or_points(self):
        if (self.code is None) == (self.points is None):
            raise ValueError("Give exactly one of 'code' or 'points'.")
        if self.code is not None:
            validate_code(self.code)
        return self


class PolarResponse(BaseModel):
    alpha: list[float]
    cl: list[float]
    cl_thin: list[float] | None = None


class OptimizeRequest(BaseModel):
    target_cl: float = Field(0.6, ge=0.0, le=1.5, description="Lift coefficient the design must give")
    min_thickness: float = Field(0.09, ge=0.06, le=0.15, description="Smallest allowed thickness, fraction of chord")
    code: str = Field("0012", description="NACA 4-digit airfoil to compare against")

    @field_validator("code")
    @classmethod
    def _check_code(cls, v: str) -> str:
        return validate_code(v)


class ViscousRequest(BaseModel):
    code: str = Field(..., description="NACA 4-digit code (airfoils only)")
    alpha: float = Field(4.0, ge=-20, le=20, description="Angle of attack in degrees")
    re: float = Field(1.0e6, ge=1.0e4, le=1.0e8, description="Chord Reynolds number")
    model_size: str = Field("large", pattern="^(xxsmall|xsmall|small|medium|large|xlarge|xxlarge|xxxlarge)$")

    @field_validator("code")
    @classmethod
    def _check_code(cls, v: str) -> str:
        return validate_code(v)


class ViscousPolar(BaseModel):
    alpha: list[float]
    cl: list[float]
    cd: list[float]
    confidence: list[float]
    cl_inviscid: list[float] = Field(..., description="Panel-method Cl (Mach 0) at the same angles")


class ViscousInviscid(BaseModel):
    cl: float
    cd: float
    mach: float


class ViscousResponse(BaseModel):
    label: str
    version: str | None
    model_size: str
    code: str
    alpha: float
    re: float
    cl: float
    cd: float
    cm: float
    cl_cd: float | None
    confidence: float = Field(..., description="NeuralFoil's own analysis_confidence, 0 to 1")
    top_xtr: float
    bot_xtr: float
    call_ms: float = Field(..., description="Measured mean time of one NeuralFoil call on the server")
    inviscid: ViscousInviscid
    polar: ViscousPolar


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


class ShapePreset(BaseModel):
    id: str
    label: str
    points: list[tuple[float, float]]
    lifting: bool
    ground: float | None = None
    note: str = ""


class DatData(BaseModel):
    name: str
    points: list[tuple[float, float]]
