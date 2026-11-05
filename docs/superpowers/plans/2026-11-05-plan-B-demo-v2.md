# Plan B: Demo v2 (draw any shape, live animated flow) Implementation Guide

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to carry out this plan task by task. Steps use checkbox (`- [ ]`) syntax.
> This is a **guide**: it gives the goal, files, interfaces, approach, tests and acceptance check per task.
> You write the code with TDD: write the listed tests first, watch them fail, implement, then watch them pass.
> The interfaces in `docs/superpowers/plans/2026-11-05-aeronet-v2-roadmap.md` §5 are binding.

**Goal:** turn the React + FastAPI demo into a sellable flow explorer. Users can draw any closed 2D shape
(or pick a circle, car or bird wing) and instantly see an animated, CFD-looking flow with honest numbers.

**Architecture:** the backend gets a generic-shape panel solver (`solve_xy`), optional ground effect by the
method of images, velocity in the field, a polar endpoint and a shape library. The frontend replaces the
static field picture with a layered canvas (colour map, isobars, particles, forces, probe) and adds a Draw tab.

**Tech stack:** Python 3.13, NumPy, FastAPI, pydantic 2, pytest; React 18, Vite 5, Vitest 2, plain Canvas 2D
(**no new npm or pip dependencies**).

**Verified before writing (2026-11-05, in a scratch copy of `app/`):** every backend algorithm below was
implemented and run. The 29 existing tests plus 24 new ones passed (53 total), and 14 new frontend utility
tests passed (21 total). The expected values in the tests below are those measured results.

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `app/backend/aeronet/panel.py` | modify | `solve_xy` core (any polygon, lifting toggle, ground); `solve(code)` wraps it |
| `app/backend/aeronet/shapes.py` | create | clean, normalise and validate outlines; Chaikin; `.dat` parser; preset library |
| `app/backend/aeronet/field.py` | modify | `field_uv` (u, v); `field_grid` adds `u`/`v`, masks body and road |
| `app/backend/aeronet/compare.py` | modify | add `compare_cp_nearest` (roadmap §5.5 rule) |
| `app/backend/aeronet/schemas.py` | modify | `FieldData.u/v`; `cl_thin` nullable; `ground_y`; shape, polar, preset and dat models |
| `app/backend/aeronet/main.py` | modify | shared response builder; `/api/solve_shape`, `/api/polar`, `/api/shapes`, `/api/shapes/parse_dat` |
| `app/backend/tests/test_shapes.py`, `test_ground.py`, `test_api_shapes.py` | create | new tests |
| `app/frontend/src/utils/colormap.js` | modify | new perceptual diverging stops (same API) |
| `app/frontend/src/utils/field.js` | create | grid↔world, bilinear `sampleField`, `sampleVelocity` |
| `app/frontend/src/utils/particles.js` | create | seeded PRNG, create, RK2 step, respawn |
| `app/frontend/src/utils/contours.js` | create | marching-squares isobars |
| `app/frontend/src/utils/outline.js` | create | `chaikin`, `thin`, `mirrorClosed`, `toDat`, `surfaceCsv` |
| `app/frontend/src/utils/urlState.js` | create | read and write shareable query state |
| `app/frontend/src/utils/flow.test.js` | create | tests for all new utilities |
| `app/frontend/src/api.js`, `hooks/useSolve.js` | modify | new endpoints; solve for a NACA code or for points |
| `app/frontend/src/components/FlowCanvas.jsx` | create (replaces `FieldCard.jsx`, which is deleted) | layered field view |
| `app/frontend/src/hooks/useDrawPointer.js`, `components/DrawPad.jsx` | create | pointer drawing; Draw tab controls |
| `app/frontend/src/components/PolarCard.jsx` | create | Cl–α chart |
| `app/frontend/src/components/Tabs.jsx` | create | Explore, Draw and PINN Lab (disabled slot for Plan C) |
| `app/frontend/src/App.jsx`, `CpCard.jsx`, `InfoCards.jsx`, `ControlCards.jsx`, `styles.css`, `index.html` | modify | layout, pin/ghost, exports, present mode, credit footer, new look |
| `.claude/launch.json` | modify | add an `aeronet-app` entry (port 8000) |
| `docs/milestones/M2-demo-v2.md`, `docs/milestones/img/*.png` | create | Milestone 2 pack |

Backend tests: `cd app/backend && .venv/Scripts/python -m pytest -q`. Frontend: `cd app/frontend && npm test`. One commit per task.

---

### Task B0: Baseline

- [ ] `cd app/frontend && npm install`, then `npm test`. **Expected: 7 passed** (measured 2026-11-05).
- [ ] If `app/backend/.venv` is missing, create it:
  `python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt`. Then run
  `.venv/Scripts/python -m pytest -q`. **Expected: 29 passed.**
- [ ] `npm run build` must succeed. No commit (nothing changes; `node_modules/` and `dist/` are git-ignored).

---

### Task B1: Generic-shape solver and shape preparation

**Goal:** solve the flow around any closed outline, not just NACA codes, without changing any existing result.

**Files:** modify `panel.py`, `compare.py`; create `shapes.py`, `tests/test_shapes.py`.

**Interface** (the order and defaults matter, because Plans C and D call it):
```python
# panel.py
def solve_xy(X, Y, alpha_deg: float, mach: float, lifting: bool = True,
             ground: float | None = None, airfoil: Airfoil | None = None) -> PanelSolution: ...
def solve(code: str, alpha_deg: float, mach: float, n_half: int = 70) -> PanelSolution:
    # t0 = perf_counter(); af = naca4(code, n_half)
    # sol = solve_xy(af.x, af.y, alpha_deg, mach, airfoil=af)
    # return dataclasses.replace(sol, ms=(perf_counter() - t0) * 1000)   # keep timing incl. geometry
@dataclass(frozen=True)
class PanelSolution:  # existing fields unchanged, then ADD at the end, in this order:
    ...; X: np.ndarray; Y: np.ndarray; upper: np.ndarray; ground_y: float | None = None
```
- `shapes.py`: `signed_area(x, y)`, `chaikin(points, iterations=2) -> ndarray`,
  `normalize_shape(points, n=200) -> (X, Y)`, `self_intersects(X, Y) -> bool`,
  `prepare_shape(points, n=200, lifting=True) -> (X, Y, info, warnings)`,
  `parse_dat(text) -> (name, points)` and `library() -> list[dict]` (library in B6).
- `compare.py`: `compare_cp_nearest(sol, ref) -> {"rms","max","n"}`.

**Approach:**
- `solve_xy` is the body of the old `solve` with these changes:
  - It takes the nodes as given (CCW, starting at the trailing edge, last node not repeated).
  - `lifting=False` solves the N×N source-only system `normal_s @ sigma = rhs` with `gamma = 0` (no Kutta row).
  - The forces use the freestream angle `flow` (equal to alpha in free air, 0 with ground; see B2).
  - `upper = np.arange(N) < argmin(X)` (panels between the TE and the min-x node).
  - It stores `X`, `Y` and `airfoil` (None for drawn shapes). **Do not change the influence maths.**
- `main.py` and `field.py` switch from `sol.airfoil.x/y` to `sol.X/sol.Y`, and from the camber-based upper
  split to `sol.upper`. This matches the old split on 0012 and 2412. On 4412 and 6409 it differs on one
  leading-edge panel (the index rule is the geometrically correct one).
- `normalize_shape` steps:
  1. Drop consecutive duplicates and a repeated closing point (tolerance 1e-9), and reject non-finite values.
  2. Reverse the order if the signed area is negative.
  3. `np.roll` so the rightmost point (argmax x) comes first.
  4. Resample to `n` points uniformly by arc length over the closed loop (`np.interp` on the cumulative length).
  5. Scale and shift: `X = (X - X.min()) / chord`, `Y = (Y - Y[0]) / chord`, where chord is the x-extent.
     The result has the LE at x = 0 and the TE at (1, 0), as in roadmap §5.2.
- `self_intersects`: vectorised orientation test over all edge pairs, ignoring adjacent pairs (|i−j| ≤ 1 or
  = n−1). Run it on the resampled curve (n = 200 means 40k pairs, which is fast).
- `prepare_shape` raises `ValueError`, whose message is shown to the user, for:
  - fewer than 8 distinct points: "A shape needs at least 8 distinct points (got k)."
  - a crossing: "The outline crosses itself. Draw a single loop without crossings."
  - signed area < 2e-4: "The shape is too thin or has no inside area."
  - normalised Y outside ±0.5: "After scaling to length 1 the shape does not fit the view…"

  `info = {chord: raw x-extent, thickness: Y-range, area, n_input}`. Add a warning when lifting is on and the
  rear is rounded: the angle between the first and last panel tangents is under 60°. The warning text
  mentions "Kutta" and advises turning lift off for bluff bodies.
- `compare_cp_nearest`: reference points with 0.02 ≤ x ≤ 0.98 take the Cp of the nearest panel centre.

**Tests** (`tests/test_shapes.py`):
- Normalise a clockwise ellipse that is offset and scaled: 120 points, x in [0, 1], first point (1, 0), signed area > 0.
- Circle (300 input points, n = 200), non-lifting at 0°: Cp_max within 0.01 of 1 (measured 0.9989),
  Cp_min within 0.01 of −3 (measured −2.9989), |Cl| < 1e-6, |Cd| < 1e-3.
- NACA 0012 points given in reverse order (`naca4('0012',120)` reversed, n = 240), at 4° and Mach 0.4: Cl within
  0.005 of `solve('0012',4,0.4,120).cl` (measured 0.5247 against 0.5263).
- `solve('0012',4,0,70).upper`: exactly the first 70 panels are True.
- A figure eight (`x = sin t`, `y = 0.3 sin 2t`) self-intersects; a 1 × 0.3 ellipse does not.
- `prepare_shape` raises with a message matching each case: 3 points ("at least 8"), figure eight ("crosses
  itself"), a straight line ("too thin"), a 0.1 × 1 ellipse ("does not fit").
- `prepare_shape` on a 4 × 1 ellipse: chord 4.0, thickness 0.25, n_input 300, a warning containing "Kutta".
  On NACA 0012: no warnings.
- `chaikin` on a unit square, 2 passes: 16 points, all corners cut (every point more than 0.05 from each
  corner), all inside [0, 1].
- `compare_cp_nearest(solve('0012',4,0.4,80), bundled sample)`: n = 162, RMS ≈ 0.0231 ± 0.002. (The camber
  method gives 0.0187 on the same data; nearest-point is a little stricter.)
- All 29 old tests still pass, unchanged.

**Acceptance:** `.venv/Scripts/python -m pytest -q` reports 29 old tests + the new ones, all passed.
**Commit:** `feat(backend): generic-shape panel solver and outline preparation`

---

### Task B2: Ground effect (road under the car)

**Goal:** an optional road line under the body. It gives downforce for cars and ground effect for wings.

**Files:** modify `panel.py`, `field.py`; create `tests/test_ground.py`. (B1's `prepare_shape` is unchanged.)

**The formula** (method of images, by reflecting the evaluation point). Get this exactly right:
```python
def influence_with_ground(P, px, py, ground_y, self_diag=False):
    sx, sy, vx, vy = influence(P, px, py, self_diag=self_diag)
    if ground_y is None:
        return sx, sy, vx, vy
    mx, my, mvx, mvy = influence(P, px, 2.0 * ground_y - py)      # real panels seen from the mirrored point
    return sx + mx, sy - my, vx + mvx, vy - mvy                     # image flow = (u, -v) at the mirror point
```
- Use it in `solve_xy` in place of `influence` for the panel-centre matrices.
- **With ground, the freestream must be parallel to the road.** So in `solve_xy`, when `ground is not None`:
  1. Rotate the body nose-up by `alpha` about the pivot (0.25, 0):
     `c, s = cos(-a), sin(-a)`, `dx = X - 0.25`,
     `X, Y = 0.25 + dx*c - Y*s, dx*s + Y*c`.
  2. Set `ground_y = Y.min() - ground`.
  3. Set `flow = 0`, so the freestream is (1, 0).

  Then `Cl = -fx sin(flow) + fy cos(flow)`. Store the rotated `X`, `Y` and `ground_y` in the solution.
- In `field.py`, split the induced-velocity part out of the old chunk function (`_induced_chunk`). When
  `ground_y` is set, add the image term: `u += u_ind(x, 2g−y)` and `v -= v_ind(x, 2g−y)`. Then add the
  freestream. `field_uv(sol, px, py, dtype, workers) -> (u, v)` is the new public function, and
  `field_cp` is rebuilt on top of it.

**Tests** (`tests/test_ground.py`; ELLIPSE = a 1 × 0.2 ellipse normalised to 200 points):
- Non-lifting ELLIPSE with ground 0.05: v on 9 points along the road line is below 1e-9 (measured exactly 0).
- NACA 0012 at 4° with ground 200: Cl within 1e-3 of free air (measured 0.4821 against 0.4822). ELLIPSE with
  ground 50: |Cl| < 1e-4.
- ELLIPSE in free air: |Cl| < 1e-9. ELLIPSE with ground 0.05: Cl < −0.1 (measured −0.233, "sucked down").
- With ground 0.1: `ground_y == Y.min() - 0.1`, `vx_inf == 1`, `vy_inf == 0`. In `field_grid(s, 90)`, every row
  below the road is None and every row with y > 0.3 is finite (this test passes after B4).
- NACA 0012 at 6° with ground 0.5: the leading-edge node is higher than the trailing-edge node (nose up).
- Reference numbers for the report (measured, NACA 0012 at 4°): Cl is 0.4898 at gap 1.0, 0.5300 at 0.3,
  0.5321 at 0.1 and 0.4511 at 0.05.

**Acceptance:** all backend tests pass. **Commit:** `feat(backend): ground effect by the method of images`

---

### Task B3: `POST /api/solve_shape`

**Goal:** the HTTP endpoint for drawn shapes (roadmap §5.6).

**Files:** modify `schemas.py`, `main.py`; create `tests/test_api_shapes.py`.

**Interface:**
- `ShapeSolveRequest`:
  - `points: list[tuple[float,float]]` (min 8, max 4000)
  - `alpha` (±20, default 4), `mach` (0 to 0.7, default 0.0), `lifting=True`
  - `ground: float|None` (0.02 to 1.0)
  - `n_panels` (40 to 400, default 200), `grid_width` (0 to 400), `reference` (as in `SolveRequest`)
- `ShapeSolveResponse(SolveResponse)` adds `shape: {chord, thickness, area, n_input}` and `warnings: list[str]`.
- `SolveResponse` changes: `cl_thin: float | None`, new `ground_y: float | None = None`. The response has
  `code="custom"` and `cl_thin=None` for shapes.

**Approach:**
- Build the shared response fields in one helper, `_response_fields(sol, grid_width, comparison, t0) -> dict`,
  used by both `/api/solve` and `/api/solve_shape` (DRY).
- `prepare_shape` errors become `HTTPException(422, detail=message)`. Pydantic range errors are 422 already.
- If a reference is given, use `compare_cp_nearest`.

**Tests:**
- A 120-point circle, non-lifting, `grid_width` 90: 200, code "custom", cl_thin null, |Cl| < 1e-6, n_input 120, no
  warnings, ground_y null.
- 3 points: 422. A figure eight: 422, with the detail containing "crosses itself". `ground` 0.001: 422.
- The old `test_api.py` passes unchanged.

**Acceptance:** all tests pass. Measured on the sedan with field 170 and ground: about 220 ms per request,
330 KB of JSON. **Commit:** `feat(api): solve_shape endpoint for drawn outlines`

---

### Task B4: Velocity in the field (both endpoints)

**Goal:** `field.u` and `field.v` drive the particles and the probe.

**Files:** `field.py` (`field_grid`), `schemas.py` (`FieldData.u`, `FieldData.v: list[float|None]`), and the
tests in `test_api_shapes.py` and `test_ground.py`.

**Approach:**
- `blocked = inside_body(sol.X, sol.Y, …) | (py < ground_y if set)`.
- Compute u and v once with float32 and threads, then `cp = (1 − u² − v²)/β`.
- Round each array to 3 decimals and turn NaN into None.
- u and v are incompressible velocities (freestream speed 1); Cp includes Prandtl-Glauert. Say this in the docstring.

**Tests:**
- `/api/solve` with 0012 and `grid_width` 60: `u` and `v` exist, each with width·height entries.
- Circle via `solve_shape`: `cp[k] is None` exactly when `u[k] is None`.
- The ground mask test from B2.

**Acceptance:** all tests pass. The 0012 field at width 170 takes about 130 ms (measured 128 ms).
**Commit:** `feat(api): velocity components in the field grid`

---

### Task B5: `POST /api/polar`

**Goal:** Cl against α in one call (data for the polar card).

**Interface:**
- `PolarRequest`: exactly one of `code` or `points` (a `model_validator` raises "Give exactly one of 'code' or
  'points'." and validates the code); `alphas: list[float]` (each ±20, 1 to 61 items); `mach`, `lifting`,
  `ground`, `n_panels=160`.
- `PolarResponse`: `{alpha, cl, cl_thin|null}`.

**Approach:**
- For a code: call `solve(code, a, mach, n_panels//2)` for each angle, plus `thin_airfoil_cl` for each angle.
- For points: run `prepare_shape` once, then `solve_xy` for each angle. Return 422 on a bad shape.
- No field is computed.

**Tests:**
- 2412 at [−4, 0, 4, 8] and Mach 0: Cl increases with α, Cl(0°) within 2e-3 of 0.2555, 4 cl_thin values.
- Circle, non-lifting, at [0, 5]: cl_thin null, max |Cl| < 1e-6.
- Neither code nor points: 422. α = 30: 422. 62 angles: 422.

**Acceptance:** all tests pass (23 angles for the sedan with ground measured at about 200 ms).
**Commit:** `feat(api): lift polar endpoint`

---

### Task B6: Shape library and `.dat` import

**Goal:** one-click presets including cars, and import of UIUC airfoil files.

**Interface:**
- `GET /api/shapes` → `[{id, label, points, lifting, ground, note}]`, ids in this order: `circle, ellipse, teardrop,
  plate, bird, sedan, sports`.
- `POST /api/shapes/parse_dat {text, name}` → `{name, points}`, or 400 with a message. (This endpoint is an
  addition to roadmap §5.6.)

**Approach:**
- `library()` builds the presets once; `main.py` caches them as `SHAPES` at import time. Round the points to
  5 decimals.
  - circle: ellipse(0.5, 0.5), 96 points, lifting False. Note: "Compare with NNfoil's FVM cylinder (Mach 0.4):
    Cp min −3.70 there vs −3.27 here (Prandtl–Glauert)."
  - ellipse: 3:1, lifting False.
  - teardrop: half-thickness `0.25·√x·(1−x)`, lifting True.
  - plate: half-thickness `0.015·√min(1, x/0.02)·min(1, (1−x)/0.1)` (3 % thick, sharp tail), lifting True.
  - bird: `naca4('6409', 60)`, lifting True.

  Symmetric shapes come from a helper that samples x with cosine spacing from 1 to 0 (81 points), then
  stacks the upper side and the mirrored lower side.
- Cars: `chaikin(points, 3)`, lifting False, ground 0.03 (sedan) and 0.02 (sports). Note:
  "Idealized potential flow: no separation and no wake, so the rear pressure is not realistic and drag is not
  shown. Use it to see where the flow speeds up and how the road changes lift." Use exactly these outlines
  (nose at x = 0, underbody at y = 0, rear-most point first, CCW):
```python
SEDAN = [(1.00, 0.10), (0.99, 0.16), (0.97, 0.20), (0.80, 0.215), (0.74, 0.23), (0.66, 0.33), (0.58, 0.355),
         (0.42, 0.36), (0.33, 0.33), (0.24, 0.235), (0.10, 0.20), (0.03, 0.17), (0.00, 0.12), (0.005, 0.07),
         (0.03, 0.03), (0.10, 0.00), (0.90, 0.00), (0.97, 0.03), (0.995, 0.06)]
SPORTS_CAR = [(1.00, 0.12), (0.98, 0.17), (0.92, 0.19), (0.80, 0.20), (0.62, 0.26), (0.52, 0.275), (0.42, 0.27),
              (0.32, 0.21), (0.15, 0.13), (0.04, 0.08), (0.00, 0.05), (0.02, 0.015), (0.08, 0.00), (0.92, 0.00),
              (0.98, 0.04)]
```
- `parse_dat`:
  - The first non-empty line is the name. Then read every row with 2 or more numbers (commas allowed).
  - Fewer than 8 rows raises `ValueError`.
  - **Lednicer format:** if the first row has both values above 1.5, it holds the point counts (nu, nl).
    Then `points = upper[::-1] + lower[1:]`.
  - Selig format is used as is.

**Tests:**
- Library ids in order. Every preset solves through `/api/solve_shape` with its own `lifting`/`ground` (status
  200). Both cars have `ground` set and a note containing "Idealized".
- Measured values (0° for non-lifting presets, 4° for lifting, Mach 0): teardrop Cl 0.508, plate 0.457,
  bird 1.204, sedan −0.496, sports −0.368. Each solve takes 10 to 20 ms.
- `parse_dat`: Selig and Lednicer versions of NACA 0012 (81 points) give identical Cl, within 0.005 of
  `solve('0012',4,0,80)` (measured 0.4802 against 0.4822). "only a name\n1 2" raises.
- Endpoint: a 120-row Selig text gives its name and 120 points; "nothing" gives 400.

**Acceptance:** all tests pass (53 backend tests in total when run in the scratch check).
**Commit:** `feat(api): shape library with cars and UIUC .dat import`

---

### Task B7: `FlowCanvas.jsx` (the new field view)

**Goal:** a field view that looks like CFD software and comes alive.

**Files:** create `utils/field.js`, `utils/particles.js`, `utils/contours.js`, `utils/flow.test.js`,
`components/FlowCanvas.jsx`; modify `utils/colormap.js`; delete `components/FieldCard.jsx` (update the import in `App.jsx`).

**Interfaces:**
- `field.js`: `toGrid(f,x,y)->[i,j]`, `toWorld(f,i,j)->[x,y]` (cell centres at integer i and j; j = 0 is the top
  row); `sampleField(f,key,x,y)` (bilinear, null outside the grid or if any of the 4 neighbours is null);
  `sampleVelocity(f,x,y)->[u,v]|null`.
- `particles.js`: `mulberry32(seed)`; `createParticles(n,f,rand)`; `stepParticles(ps,f,dt,rand)`;
  `respawn(p,f,rand,fromLeft)`. A particle is `{x,y,px,py,age,life}`.
- `contours.js`: `ISO_LEVELS = [-1.5…0.75 step 0.25, without 0]`; `isoSegments(f, level, key='cp') -> [[x1,y1,x2,y2]]`.
- `colormap.js`: keep `CP_MIN`, `CP_MAX`, `STOPS`, `cmap` and `gradientCss`. The new 8 stops (RdBu-like) are
  −2 [23,37,99], −1.4 [33,84,160], −0.8 [67,147,195], −0.3 [166,206,227], 0 [247,247,244], 0.35 [253,204,138],
  0.7 [227,105,70], 1 [165,15,38]. `gradientCss` places each stop at its Cp position, not evenly.
- `<FlowCanvas result={...} layers={{field, isobars, particles, forces}} paused={bool} drawing={bool} onDrawComplete={fn} />`.

**Approach:**
- Two stacked canvases in `.canvas-box`:
  - Base canvas, redrawn on result, layer or size change through `useCanvasDraw`. It draws the colour map
    (offscreen image, smoothed), then the isobars (thin lines, `rgba(0,0,0,.28)`, dashed for negative
    levels), then the road band (below `ground_y`, hatched, labelled "road"), then the body (fill `#20242c`),
    then the force arrows, then the freestream arrow.
  - Overlay canvas for the particles, which runs a `requestAnimationFrame` loop.
- Particles:
  - About 900 particles at full width (scale with canvas area). `dt = 0.006` world units per frame.
  - RK2 midpoint step: `v1 = V(p)`, `m = p + dt/2·v1`, `p += dt·V(m)`.
  - Respawn on the inflow edge (the left 4 % of the window) when a particle leaves, hits a null cell or
    reaches its life (80 to 200 frames).
  - Trails: each frame, fade the overlay with `globalCompositeOperation='destination-out'` and a fill of
    `rgba(0,0,0,0.10)`. Then draw each segment (px,py)→(x,y) twice: 2.2 px `rgba(10,14,22,.35)`, then
    1 px `rgba(255,255,255,.9)`. That stays visible on both the blue and the white colours.
  - When `result` changes, keep the particles and only re-validate them. Don't reset, or the picture flickers.
- Measure fps: keep a moving average of the frame time in `window.__aeronetFps` (for B10).
- Respect `matchMedia('(prefers-reduced-motion: reduce)')`: start paused. Add a Pause/Play button.
- Hover probe: on mousemove, convert to world coordinates and show a small tooltip with
  `Cp`, `|V|/V∞` and `x/c` from `sampleField`. Hide it over the body.
- Force arrows (layer "forces"):
  - For every 3rd panel, an arrow from the panel centre along the outward normal `(ty, −tx)`, with length
    `−cp × 0.06` (suction points outward, pressure inward). Use the `--up` colour for the upper surface and
    `--lo` for the lower.
  - Plus a lift arrow at (0.25, 0), perpendicular to the freestream, length `0.15·cl`.
- Layer toggles render as a `.seg` button group above the canvas: Pressure, Isobars, Flow, Forces.
  Show the colour bar under the canvas (−2 suction … +1 stagnation).

**Tests** (`flow.test.js`; fixture: 18×10 grid in the app window with cp = x, u = 1, v = 0, and optional null cells):
- toWorld/toGrid round trip. Bilinear sample of cp = x at 0.333 gives 0.333. Null outside the grid and next to a
  null cell. `sampleVelocity` in a free area gives [1, 0].
- One RK2 step with dt = 0.01 in the uniform flow moves x by exactly 0.01, y unchanged, and sets px to the old x.
- A particle at x = 1.3 with dt = 0.5 respawns with x < −0.3 and age 0.
- `createParticles(50, f, mulberry32(7))` twice gives equal arrays, and every particle is in a free cell.
- `isoSegments(f, 0.5)`: more than 5 segments, all with x1 = x2 = 0.5 (to 9 decimals). Fewer segments when a null cell is added.
- The old colour-map tests still pass with the new stops.

**Acceptance:** `npm test` passes, `npm run build` passes, and the browser shows particles flowing around 0012
(checked in B10). **Commit:** `feat(frontend): animated flow canvas with isobars, probe and force arrows`

---

### Task B8: Draw tab (`DrawPad.jsx`)

**Goal:** draw a closed shape with the mouse, pen or finger and get the flow in under 0.5 s.

**Files:** create `utils/outline.js`, `hooks/useDrawPointer.js`, `components/DrawPad.jsx`, `components/Tabs.jsx`;
modify `api.js`, `hooks/useSolve.js`, `App.jsx`, `FlowCanvas.jsx`; add tests to `flow.test.js`.

**Interfaces:**
- `api.js` adds `solveShape(body, signal)`, `getShapes()`, `parseDat(text, name)` and `getPolar(body, signal)`.
- `useSolve(input)`, where `input` is `{kind:'naca', solver, code, alpha, mach, reference}` or
  `{kind:'shape', points, alpha, mach, lifting, ground, reference}`. It keeps the existing behaviour: a quick
  request with grid 100, then a full request with grid 170 after 180 ms, aborting stale requests. Memoise
  `points` by identity, since a new array means a new shape.
- `outline.js`: `chaikin(points, iters)`, `thin(points, minDist)`, `mirrorClosed(stroke)`,
  `toDat(points, name)` (Selig format, closed) and `surfaceCsv(result)` (`x,y,cp,upper` with 6 decimals).
- `useDrawPointer(canvasRef, enabled, onComplete)` turns pointer events into world points using the same
  window mapping as FlowCanvas.
- `<DrawPad shapes value onChange />` holds the controls; `<Tabs value onChange tabs=[{id,label,disabled,hint}] />`.

**Approach:**
- Tabs: `Explore` (the existing NACA controls), `Draw`, and `PINN Lab`. PINN Lab is disabled with the hint
  "Arrives with Plan C (trained PINN cases)". Plan C enables it.
- While drawing, FlowCanvas shows the live stroke. On pointer-up:
  1. `thin` with minDist 0.004.
  2. Fewer than 8 points: show "Draw a bigger loop".
  3. Mirror mode: `mirrorClosed` of the stroke, where the user draws the upper half above the dashed
     centre line y = 0.
  4. Smooth with `chaikin` (2 passes).
  5. Set the shape, which re-solves through `useSolve`.

  Use `setPointerCapture` and `touch-action: none` on the canvas.
- Controls: Draw (arms the pen), Undo (keep a stack of the last 10 shapes), Clear, a Mirror toggle, the
  library buttons from `GET /api/shapes` (which also set `lifting`/`ground` from the preset), "Lift (Kutta)",
  and a "Road" toggle with a gap slider (0.02 to 0.3). Also import `.dat` (file input → `parseDat`) and
  export `.dat` (Blob download, named `aeronet-shape.dat`).
- Show backend 422 messages and `warnings` in a status line under the canvas. Show the preset `note` as a
  yellow pill.
- Results card in shape mode: show "Thin-airfoil" as "n/a (not an airfoil code)". **Hide Cd when `ground_y`
  is set (cars)** and write "not shown: inviscid flow has no drag".

**Tests** (`flow.test.js`):
- `chaikin` on a square gives 16 points.
- `thin([[0,0],[0.001,0],[0.1,0]], 0.01)` gives `[[0,0],[0.1,0]]`.
- `mirrorClosed([[0,0.1],[1,0.2]])` gives `[[0,0.1],[1,0.2],[1,-0.2],[0,-0.1]]`.
- `toDat` writes a name line plus n+1 rows.
- `surfaceCsv` of one point gives `'x,y,cp,upper\n0.500000,0.100000,-0.300000,1\n'`.

**Acceptance:** `npm test` and `npm run build` pass. In the browser, drawing a blob shows the field and Cl,
and drawing a figure eight shows the "crosses itself" message. **Commit:** `feat(frontend): draw-any-shape tab with library, mirror, undo and .dat import/export`

---

### Task B9: Polish (new look, polar, pin, exports, URL, present mode, credit)

**Goal:** a page that looks like a product, plus the S-tier features from roadmap §3.

**Files:** create `components/PolarCard.jsx`, `utils/urlState.js`; modify `App.jsx`, `CpCard.jsx`, `InfoCards.jsx`,
`ControlCards.jsx`, `styles.css`, `index.html`; add tests to `flow.test.js`.

**Approach:**
- **Look:**
  - A dark "wind-tunnel" hero header (always dark, even in the light theme) with the title, a one-line pitch,
    and scope pills ("Inviscid · subsonic · 2D", "Panel method now; PINN in PINN Lab").
  - Keep the light and dark CSS tokens. Add `--hero-bg`, `--hero-ink` and `--particle`.
  - Make the field card full width above the two-column layout. The cards keep the same structure.
  - Replace the long yellow "Read this first" notice with a collapsible `<details>` titled "What is real here".
  - Set `<title>` in `index.html` to "AeroNet: airflow explorer".
  - Run the frontend-design skill for typography and spacing choices, without adding font or CSS libraries.
- **PolarCard:** α from −10 to 12 in steps of 1 at the current Mach, for the current code or shape (debounce
  300 ms; reuse the AbortController pattern from `useSolve`). Draw on canvas: the panel Cl line, the
  thin-airfoil dashed line (codes only), a dot at the current α, and axes labelled "α (deg)" and "Cl".
  Plan C adds FVM and PINN points, so accept an optional `extraSeries=[{label, points:[[a,cl]], style}]` prop now.
- **Pin and ghost:** a "Pin result" button stores the current result. CpCard draws the pinned Cp as grey
  dashed lines, labelled "pinned: <name>". The Results card shows ΔCl against the pinned result. Clear it with "Unpin".
- **Exports:**
  - PNG: composite the base and overlay canvases into one canvas, then `toBlob`, saved as `aeronet-flow.png`.
  - CSV: `surfaceCsv`, saved as `aeronet-cp.csv`.
  - `.dat` (Draw tab).
- **URL state:** `readState(location.search)` on load and `history.replaceState(null,'',writeState(location.search, s))`
  on change. Freehand outlines are not stored; library shapes are stored by id. Keep the `?light`, `?sample`
  and `?present` flags.
- **`?present`:** add class `present` on `<html>` (base font 18 px, notices and pipeline hidden, larger canvas).
  Auto-cycle every 8 s through 0012 at 4°, 2412 at 6°, the bird wing, the sedan with road, and the circle.
  Stop cycling on the first user input.
- **Credit footer:** "Flow solver for this view: Hess–Smith panel method. PINN: NNfoil by Cao, Song & Zhang
  (arXiv 2401.08705), used with credit; no licence, so this repository stays private."
- **InfoCards:** update the `PipelineCard` "Not built yet" list (the drawn shapes are built now). Leave
  `TrainingCard` for Plan C.

**Tests** (urlState, in `flow.test.js`):
- Round trip of the explore state `{tab:'explore',code:'2412',alpha:6,mach:0.3,shape:'',lifting:true,ground:''}`
  written onto `'?present'` (which must keep `present`).
- Round trip of the draw state `{tab:'draw',code:'0012',alpha:0,mach:0,shape:'sedan',lifting:false,ground:0.03}`.
- `readState('?tab=evil&code=12&alpha=99&mach=-1&shape=<x>&ground=9')` gives
  `{tab:'explore',code:'0012',alpha:12,mach:0,shape:'x',ground:1}`.

**Acceptance:** `npm test` (21 tests measured with the B7 to B9 utilities) and `npm run build` pass. All the
features work in the browser (checked in B10). **Commit:** `feat(frontend): new look, polar chart, pin/compare, exports, URL state and present mode`

---

### Task B10: Milestone 2 pack

**Goal:** proof for the mentor that Demo v2 works.

**Files:** modify `.claude/launch.json`; create `docs/milestones/M2-demo-v2.md` and `docs/milestones/img/m2-*.png`.

**Approach:**
1. Add this entry to the `configurations` list in `.claude/launch.json`:
   `{"name":"aeronet-app","runtimeExecutable":"app/backend/.venv/Scripts/python.exe","runtimeArgs":["-m","uvicorn","aeronet.main:app","--port","8000","--app-dir","app/backend"],"port":8000}`.
   Run `npm run build` first, so FastAPI serves `frontend/dist`.
2. Start it with `preview_start {name:"aeronet-app"}`. Check the console for errors. Read
   `window.__aeronetFps` after 5 s; **30 or more required** at grid 170 (record the measured value).
3. Save 4 screenshots to `docs/milestones/img/` (resize to 1280×800):
   - `m2-naca.png`: 2412 at 6°, isobars and flow on.
   - `m2-drawn.png`: a hand-drawn blob, with its Cl visible.
   - `m2-car.png`: the sedan with road, with the note pill visible.
   - `m2-polar.png`: the polar card for 2412.
4. Write the M2 doc:
   - what changed (feature list)
   - how to run it (two commands)
   - the 2-minute live demo script: Explore 0012, then sweep α, then Draw a blob, then mirror-draw a wing,
     then the sedan with road (downforce, honest note), then pin and compare, then the polar
   - the numbers table, with every value labelled **measured**: circle Cp extremes, the reversed-0012 Cl
     check, the ground-effect Cl at gaps 1.0/0.3/0.1/0.05, the preset Cl values, solve and field timings, and fps
   - the test counts
   - limits (inviscid, no separation, Cd not physical, freehand shapes not in URLs)

**Acceptance:** the doc exists and the 4 images exist, every number in it is measured in this task or
quoted from the B1 to B6 test outputs, and both test suites pass.
**Commit:** `docs: milestone 2 pack (demo v2 screenshots, demo script, numbers)`

---

**Final check (end of B10):** Plan B owns `solve_shape`, `polar`, `shapes` and field `u`/`v` (`pinn/*` and `results` are Plan C's). NACA `/api/solve` numbers are unchanged, no dependencies were added, and cars never show a Cd.
