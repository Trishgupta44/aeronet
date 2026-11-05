# Plan C: PINN Results in the Demo (Experiments E1–E5, PINN Lab, Train-on-Your-Shape) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax.
> This is a **guide**: you write the code. Interfaces in `2026-11-05-aeronet-v2-roadmap.md` §5 are binding.
> Where this plan and the roadmap disagree, the roadmap wins.

**Goal:** run the experiments the docs promise (E1–E5 and the speed table), put real PINN results into the
app (PINN Lab with training replay and a numbers dashboard), and let a user train a PINN on a drawn shape.

**Architecture:** research scripts in `research/` (run in `research/.venv`, the only venv with torch) turn
run folders (`results/runs/<id>/`, roadmap §5.3) into rows of `results/summary.csv` (§5.4) and into case
JSONs in `app/backend/data/pinn_cases/` (§5.7). The FastAPI backend only serves those files: **no torch in
the app**. The one exception is C10, which starts `research/.venv` as a subprocess.

**Tech stack:** PyTorch CPU, NumPy, SciPy, Matplotlib (`matplotlib.tri`), NeuralFoil 0.3.3, FastAPI,
React 18 with canvas, pytest, vitest.

**Prerequisites:** Plan A is done (`NNfoil/run_case.py`, `research/metrics.py`, `research/validate.py`,
`research/baselines.py`, and runs `naca0012_50x25_a4_m04` and `naca0012_100x50_a4_m04`). Plan B is done
(`/api/shapes`, `/api/solve_shape`, `/api/polar`, `FlowCanvas.jsx`, the tab layout, `panel.solve_xy`).
Before starting, run `research/.venv/Scripts/python -m pytest research/tests -q` and
`app/backend/.venv/Scripts/python -m pytest -q` in `app/backend`. Both must pass.

---

## Measured facts to build on (2026-11-05, this laptop, 16 threads)

| Fact | Value | Source |
|---|---|---|
| NNfoil 50×25, upstream net 5×128, L-BFGS 1000, 8 threads, no positivity | 70.4, 73.1, 76.0 s per round; Cl 0.5156, 0.5118, 0.5140 | measured |
| Same, **with** positivity (softplus+1e-3) | round 1 printed `Failed, Restart`; rounds 2–3: Cl 0.307 → 0.478, RMS vs FVM800 0.193 → 0.037 | measured |
| FVM Cl (NACA0012, 4°, Ma 0.4) from `fvm_forces` | 50×25 0.4265 · 100×50 0.4803 · 200×100 0.5046 · 400×200 0.5146 · 800×400 0.5193 | measured |
| FVM coarse vs FVM800, `cp_error` RMS | 50×25 0.1081 · 100×50 0.0460 · 200×100 0.0186 · 400×200 0.0059; panel vs FVM800 0.0099 | measured |
| NACA2412 FVM800 Cl | AoA −10: −0.9558 · −4: −0.2301 · 0: 0.2651 · 4: 0.7605 · 8: 1.2516 · 10: 1.4929 | measured |
| NACA2412 Mach | panel fit ≈ 0.25; stagnation-Cp estimate 0.22–0.29 (biased low: the same method gives 0.385 for the known 0.4 case); **not 0.4** | fitted |
| Cylinder FVM | stagnation at 180° → **α = 0°**; stagnation Cp 1.044 ≈ isentropic value at Ma 0.4 (1.0406) → **Ma 0.4** | fitted |
| Random airfoils 1–5 | the stagnation-Cp Mach varies per airfoil (0.0, 0.24, 0.31, 0.39, 0.64): conditions differ per airfoil | fitted |
| Vectorised O-grid (`ogrid.py` design below) | 50×25: about 3,000 iterations, 0.6–0.9 s · 100×50: about 11,000 iterations, 4–6 s, for NACA0012, circle, a concave "kidney" and a sedan outline; no folds. A deep "C" shape folds (0.7% of cells) and the check catches it | measured |
| Upstream (loop) mesh_gen | 50×25 about 10 s · 100×50 18.9 s · 200×100 280 s | measured |
| `ogrid` with `cross_coeff=0.5` vs bundled `NACA0012_50_25.x` | max node difference 3.2e-4 (so the bundled meshes used the upstream coefficient); with 2.0 it is 0.011 | measured |
| `cosine_wall()` from the 200×100 wall, resampled to 50 nodes | reproduces the bundled 50×25 wall to 1e-16 | measured |
| Field resampling (periodic triangulation, 340×189 grid) | 0.15 s; the mask equals the body exactly; cp+u+v JSON 1.40 MB; max interpolation error 0.004 on a smooth test field | measured |
| Upstream NNfoil (no positivity) on an `ogrid` **cylinder** 50×25, 3 rounds. **Caveat:** run at the hard-coded α 4°, not the FVM's 0° | RMS vs FVM800 1.48 → 0.268 → 0.258; Cp_min −3.57 (FVM −3.70, PG panel −3.27); Cl 0.029. Rounds took 167–535 s because other jobs were competing for the CPU. Full-mesh inference 2.9–3.3 ms | measured |
| NeuralFoil 0.3.3, `get_aero_from_coordinates`, NACA0012, α 4, Re 6e6 | CL 0.421 (large) and 0.415 (xxxlarge); 3.5 ms and 16.7 ms per call | measured |

---

## File map

| Path | Create/Modify | Responsibility |
|---|---|---|
| `research/runs.py` | create | run ids, train-command builder, `load_run`, wall faces, restart count, `build_model` |
| `research/experiments.py` | create | `commands <e>` prints the training commands; `collect <e>` appends rows to `summary.csv` |
| `research/ogrid.py` | create | `resample_closed`, `cosine_wall`, `ogrid`, `cell_jacobian`, `is_valid`, `write_x` |
| `research/time_inference.py` | create | PINN inference timing, panel timing, NeuralFoil timing |
| `research/export_results.py` | create | run folder → case JSON (§5.7); `--summary` → `results_summary.json` |
| `research/make_figures.py`, `research/report_tables.py` | create | every figure → `results/figures/`; Markdown tables → `results/tables.md` |
| `research/train_shape.py` | create | C10 pipeline: points → mesh → train → export, with `status.json` |
| `research/tests/test_runs.py`, `test_ogrid.py`, `test_export.py`, `test_experiments.py` | create | unit tests (no training) |
| `app/backend/aeronet/pinn_cases.py`, `jobs.py` | create | serve case files and results; the C10 job runner |
| `app/backend/aeronet/main.py`, `solvers.py` | modify | new endpoints; dynamic PINN availability |
| `app/backend/tests/test_pinn_api.py`, `test_jobs.py`, `tests/fake_train_shape.py` | create | API tests with fixture files |
| `app/backend/tests/test_api.py` | modify | `test_solvers_and_presets` no longer hard-codes `available=False` |
| `app/frontend/src/api.js` | modify | `getPinnCases`, `getPinnCase`, `getResults`, `startJob`, `getJob`, `getJobCapability` |
| `app/frontend/src/utils/pinn.js` (+ `pinn.test.js`) | create | case JSON → SolveResponse-shaped object; upper/lower split; snapshot surface |
| `app/frontend/src/components/PinnLab.jsx`, `PinnCpChart.jsx`, `ResultsTable.jsx`, `TrainShapeButton.jsx` | create | PINN Lab tab, Cp chart with replay, numbers dashboard, C10 button |
| `app/frontend/src/components/InfoCards.jsx` | modify | `TrainingCard({log})` draws curves; `ComparisonCard({result, rows})` fills measured rows |
| `app/frontend/src/App.jsx` | modify | add the `pinn` tab; the PINN solver button switches to it |
| `docs/milestones/M3-pinn-results.md` (+ `img/`) | create | Milestone 3 brief |

**Conventions (follow exactly):**
- `case` column in `summary.csv`: `<Shape>_a<alpha>` (e.g. `NACA2412_a4`, `Cylinder_a0`). For E1, which
  has two seeds, use `NACA0012_a4_s<seed>`. Together with (experiment, method, mesh) this keeps rows unique.
- `train_s` is the cumulative `log.time_s[-1]`. `infer_ms` is filled only in E4 rows.
- Training always runs with 8 threads. At most **2 runs at the same time** (16 threads in total), each
  started with Bash `run_in_background`. Don't wait idle: continue with the next task.
- Every figure is made by `make_figures.py` (C9), never inside `experiments.py`.

---

### Task C1: Run helpers plus E1 (positivity before vs after)

**Goal:** a reusable run toolkit, and the E1 result. Does the softplus positivity fix help or hurt?

**Files:** create `research/runs.py`, `research/experiments.py` and `research/tests/test_runs.py`.

**Interface (`research/runs.py`):**
- `ROOT` (repo root), `RUNS = ROOT/"results/runs"`, `FVM = {...}` reference paths (see below).
- `run_id(shape, mesh, alpha, mach, positivity=True, seed=None) -> str`, following roadmap §5.8. **Mach
  token:** if `mach*10` is an integer, use 2 digits (`0.4 → m04`); otherwise use 3 digits of `mach*100`
  (`0.25 → m025`). The alpha token: `4 → a4`, `-4 → am4`, `2.5 → a2p5`.
- `train_cmd(mesh_file, shape, mach, alpha, rounds, out, positivity=True, seed=0, threads=8, layers=None, lbfgs_iter=None) -> list[str]`
  returns the `NNfoil/run_case.py` command (§5.8) using `sys.executable`.
- `load_run(run_dir) -> {"dir","config","log","snapshots","field"}` (field = `dict(np.load(...))` or None).
- `wall_faces(snapshots) -> (x, y)`: if `len(x) == len(cp)+1`, return the midpoints of the nodes,
  otherwise x and y as given. This absorbs the node vs face ambiguity in §5.3.
- `final_surface(run) -> (x, y, cp)`, using the last snapshot.
- `count_restarts(run_dir) -> int`: counts lines starting with `Failed` in `train.log`.
- `build_model(run_dir, threads=8)`: rebuilds Plan A's `NNfoil(layers, torch.device("cpu"), xx, yy, Ma=, alpha_deg=, positivity=, ckpt_path=<run_dir>/_tmp.pth)`
  from `config.json` plus `metrics.load_mesh(config["mesh_file"])`, then loads `model.pt` with
  `torch.load(..., map_location="cpu", weights_only=True)` and calls `.eval()`. Import torch and the
  NNfoil module inside the function, after `sys.path.insert(0, str(ROOT/"NNfoil"))`, because NNfoil's
  modules do `from util import ...`.
- `FVM`: `{"NACA0012": "NNfoil/FVM_Cp/NACA0012_800_400_Cp.dat", "Cylinder": "NNfoil/FVM_Cp/Cylinder_800_400_Cp.dat"}`, plus
  `fvm_2412(alpha) -> "NNfoil/FVM_Cp_NACA2412/NACA2412_800_400_AoA{int(alpha)}_Cp.dat"`.

**Interface (`research/experiments.py`):** `python research/experiments.py commands e1|e2|e3|e5` prints one
command per line, skipping runs whose `log.json` already has the requested rounds.
`python research/experiments.py collect e1|...` computes the metrics with `metrics.cp_error` and
`metrics.fvm_forces` and calls `metrics.append_summary` once per row.

**E1 design:** mesh `NNfoil/mesh/NACA0012_50_25.x`, Ma 0.4, α 4, **20 rounds**, seeds 0 and 1, positivity
on and off: 4 runs, about 25 min each. `naca0012_50x25_a4_m04` from A12 counts as (on, seed 0) if it has
20 rounds. Row fields: method `pinn` or `pinn_nopos`, case `NACA0012_a4_s<seed>`, rms/max (final round),
cl, cl_ref (FVM800 0.5193), cd, train_s, and `notes = "rounds=20 seed=<s> restarts=<k> nan_rounds=<j> best_round_rms=<r>"`.

**Approach:**
- Print the four commands, start two in the background, then the other two when those finish.
- If positivity is worse, say so plainly; it is still a verification result. In `notes` and in the M3
  doc, list the likely cause and two remedies, without implementing them unless time allows. (1) The
  initial outputs start near softplus(0)+1e-3 ≈ 0.69, far from p∞ = 1/(1.4·0.16) = 4.46, which makes the
  first round unstable. Remedy A: shift, `p = softplus(raw) + 1e-3`, with the last layer's bias for p and
  rho initialised to `log(expm1(target))` (inverse softplus of p∞ and ρ∞ = 1). Remedy B: warm-start the
  positivity run from the non-positivity checkpoint.
- Stretch E1b: implement remedy A behind a `--pos-init` flag in `run_case.py` and add method `pinn_posinit`
  (add that word to the `method` list in the roadmap §5.4 if you do).

**Tests (`test_runs.py`, no torch needed):**
- `run_id("NACA0012","100x50",4,0.4) == "naca0012_100x50_a4_m04"`; `run_id("NACA2412","100x50",-4,0.25,positivity=False) == "naca2412_100x50_am4_m025_nopos"`.
- `wall_faces` returns midpoints when given N+1 nodes and N cp values, and passes through when lengths match.
- `count_restarts` counts 2 on a fake `train.log` containing two `Failed, ...` lines.
- `train_cmd` contains `--no-positivity` only when `positivity=False`.

**Acceptance:** `research/.venv/Scripts/python -m pytest research/tests/test_runs.py -q` passes. After the
runs, `collect e1` adds 4 rows with `experiment=E1`.

**Commit:** `feat(research): run helpers and E1 positivity experiment`. After collecting, a second commit:
`results: E1 positivity before/after`.

---

### Task C2: E2, mesh size vs error (the paper's sparse-mesh claim)

**Goal:** one table with, for each mesh, the FVM error and the PINN error against FVM 800×400.

**Files:** modify `research/experiments.py` (add `e2`).

**Design:**
- FVM rows (method `fvm`, case `NACA0012_a4`): meshes 50x25, 100x50, 200x100 and 400x200;
  rms/max = `cp_error(load_fvm(coarse), ref=load_fvm(800))` (use the coarse points as the "prediction");
  cl from `fvm_forces`. Expected RMS: 0.1081, 0.0460, 0.0186, 0.0059.
- PINN rows (method `pinn`): 50×25 (reuse the E1 seed-0 run with positivity on, or off if E1 showed it is
  better; state which in `notes`), 100×50 (A12 run), and 200×100 (new, `NNfoil/mesh/NACA0012_200_100.x`).
  For 200×100, first run **1 round**, measure its time, then pick `rounds = min(20, floor(3 h / t_round))`.
  If that is fewer than 3, skip it and write the reason in the M3 doc. Record the rounds in `notes`.
- Panel row (method `panel`, mesh `140 panels`): `cp_error` of the app's panel solution
  (`sys.path.insert(0, "app/backend")`, `from aeronet.panel import solve`; `solve("0012",4,0.4,70)`), using
  the panel centres `geom.cx` and `geom.cy`. Expected about 0.0099.

**Tests:** `test_experiments.py::test_e2_fvm_rows`. Call the E2 FVM-row builder (a pure function,
`fvm_mesh_rows()`, which returns dicts) and assert 4 rows, with the 50×25 RMS in [0.10, 0.12] and the
400×200 RMS < 0.01.

**Acceptance:** `collect e2` adds at least 7 rows. The headline check, written into `notes` of the PINN
50×25 row: "PINN RMS < FVM RMS at 50×25: yes/no".

**Commit:** `feat(research): E2 mesh study (FVM vs PINN against FVM 800x400)`.

---

### Task C3: O-grid generator plus E3 (NACA 2412 and the angle sweep)

**Goal:** a mesh for any closed curve (needed for E3, E5 and C10), then NACA 2412 PINN results and polar points.

**Files:** create `research/ogrid.py` and `research/tests/test_ogrid.py`; modify `research/experiments.py` (`e3`).

**Interface (`research/ogrid.py`):** all coordinates are in the **NNfoil frame** (chord in [−0.5, 0.5]).
- `resample_closed(x, y, n) -> (X, Y)`: drop a duplicated end point; make it CCW (signed area > 0); start
  at the rightmost point; resample uniformly by arc length; scale the chord to 1 into [−0.5, 0.5]; put the
  TE at y = 0. Use it for drawn shapes and the cylinder.
- `cosine_wall(xw, yw, n) -> (X, Y)`: airfoil-style nodes at `x = 0.5*cos(θ)`, θ ∈ [0, 2π) with n nodes;
  interpolate y on the upper (TE→LE) and lower (LE→TE) branches separately. Use it for airfoils, because
  the bundled meshes use this spacing.
- `ogrid(xw, yw, M, R=15.0, cross_coeff=2.0, tol=1e-6, max_iter=200_000) -> (X, Y, iters, last_change)`:
  X and Y have shape (M, N); row 0 is the wall and row M−1 is the far circle of radius R, with node j at
  angle 2πj/N. Columns are periodic.
- `cell_jacobian(X, Y) -> (M-1, N)` and `is_valid(X, Y) -> bool` (**all cells have the same sign**; with
  this ordering, valid grids are all negative).
- `write_x(path, X, Y)`: NNfoil's format. The first line is `f"{N}\t{M}"`, then `np.savetxt` of
  `np.r_[X, Y]` with `%.16f`. Plan A's `metrics.load_mesh` must read it back unchanged.

**Approach (key details):**
- Start from linear interpolation, wall → far circle, as upstream does. Iterate Jacobi on the Winslow
  equations with central differences (periodic in ξ via `np.roll`; η interior only). The ξ direction runs
  along columns (axis 1) and η along rows (axis 0):
```python
# a = x_eta^2+y_eta^2, g = x_xi^2+y_xi^2, b = x_xi*x_eta + y_xi*y_eta  (central differences)
# U_new = (a*(U[j+1]+U[j-1]) + g*(U[i+1]+U[i-1]) - cross_coeff*b*U_xieta) / (2*(a+g))
# U_xieta = 0.25*(U[i+1,j+1]-U[i+1,j-1]-U[i-1,j+1]+U[i-1,j-1]);  rows 0 and M-1 stay fixed
```
- `cross_coeff=2.0` is the correct Winslow coefficient. **Upstream `mesh_gen.py` uses 0.5** (finding 3,
  confirmed in Plan A). Keep the argument so the upstream meshes can be reproduced.

**Tests (`test_ogrid.py`):**
- With `cross_coeff=0.5`, starting from the `NACA0012_50_25.x` wall, `ogrid` reproduces that file's nodes
  to < 1e-3 (measured 3.2e-4).
- `cosine_wall` on the 200×100 wall with n=50 equals the 50×25 wall to < 1e-12.
- A circle at 50×25 is valid and finishes in < 5 s. A deep C shape (`x=[.5,-.5,-.5,.5,.5,-.3,-.3,.5]`,
  `y=[.3,.3,-.3,-.3,-.2,-.2,.2,.2]`, after `resample_closed`) is **not** valid.
- `write_x` followed by `metrics.load_mesh` round-trips: shapes (N+1, M), with the seam row equal to row 0.

**E3 design:**
- **Mach first.** Use the `web-fetch` agent or WebFetch on `https://arxiv.org/abs/2401.08705` and the HTML
  version (`https://arxiv.org/html/2401.08705`) to find the NACA 2412 sweep conditions and the conditions
  of random airfoils 1–5. Quote the sentence and the table in `docs/verification.md` under "Reference
  data conditions". If the paper doesn't say, use Ma 0.25, labelled `fitted` (source column), with a note.
- Mesh: `cosine_wall` from the `NACA2412_200_100.x` wall at N=100, then `ogrid(M=50, cross_coeff=0.5)`.
  The upstream coefficient keeps the mesh comparable with the NACA0012 runs. Write it to
  `results/meshes/NACA2412_100_50.x` and commit it.
- PINN runs: α = −4, 0, 4, 8 at 100×50, 10 rounds each (measure round 1 first and apply the 3-hour rule).
- Polar rows (experiment `E3`, case `NACA2412_a<α>`): `fvm` (mesh `800x400`, cl from `fvm_forces`, every
  angle file −10..10), `panel` (Plan B's `/api/polar` maths: import `aeronet.panel.solve("2412", α, Ma)`),
  `thin_airfoil` (`aeronet.compare.thin_airfoil_cl`) and `pinn` (rms, max, cl, cd).

**Acceptance:** `pytest research/tests/test_ogrid.py -q` passes. `collect e3` adds at least 11 `fvm` rows
plus 4 `pinn` rows. FVM Cl at 4° is 0.7605 ± 0.0005.

**Commits:** `feat(research): O-grid generator for any closed curve (Winslow, cross coefficient 2)`, then
`feat(research): E3 NACA2412 sweep and polar points`.

---

### Task C4: E4, speed table (measured vs quoted)

**Goal:** an honest speed table, following NeuralFoil's rule that speed is compared at stated accuracy.

**Files:** create `research/time_inference.py`; modify `research/experiments.py` (`collect e4`).

**Interface:** `python research/time_inference.py <run_dir> [--calls 100] [--threads 8]` prints JSON
`{"infer_ms","infer_ms_std","n_points","threads"}`. `collect e4` writes rows with experiment `E4`, case
`NACA0012_a4`:

| method | mesh | train_s | infer_ms | source |
|---|---|---|---|---|
| `pinn` | 50x25, 100x50 | log time | full-field forward pass | measured |
| `panel` | 140 panels | (empty) | `solve()` only; notes give solve+`field_grid(170)` ms | measured |
| `neuralfoil` | `large` | (empty) | per call | measured |
| `fvm` | 800x400 | (empty) | from the paper, converted to ms | quoted |

**Approach:**
- PINN inference: `nn = runs.build_model(run)`, then under `torch.no_grad()` do 10 warm-up calls of
  `nn.model(nn.XE)` (every mesh point), then time 100 calls with `time.perf_counter()`. Report mean and
  std. State in `notes` that this gives the whole field (u, v, p, ρ at every node), not just Cl.
- Panel: the median of 100 calls to `solve("0012",4,0.4,70)`. Also time solve plus `field_grid(sol,170)`.
- NeuralFoil: `pip install neuralfoil` into `research/.venv` and add it to `research/requirements.txt`.
  The API is `nf.get_aero_from_coordinates(coordinates=(K,2) array, alpha=4.0, Re=6e6, model_size="large")`,
  which returns a dict with `CL`, `CD` and `analysis_confidence`. The coordinates must be in the **app frame
  (x 0..1), Selig order TE→upper→LE→lower→TE**. Each call includes a Kulfan fit, which is part of the
  measured time; say so. NeuralFoil is **viscous and incompressible**: CL 0.421 vs inviscid FVM 0.519 is
  expected, not an error. Put that in `notes`. If the import fails, write the row with empty numbers and
  `notes="neuralfoil unavailable: <error>"`.
- FVM: find a stated FVM wall time in the paper (same fetch as C3). Quote it with page or table. If there
  is none, leave the row empty with `notes="not stated in paper"`.
- Hardware line for M3: CPU name (`wmic cpu get name`, or `platform.processor()`), torch version, threads.

**Tests:** `test_experiments.py::test_selig_order`: the coordinates builder for NeuralFoil starts and ends
at x ≈ 1 and has its minimum x in the middle of the array.

**Acceptance:** `collect e4` adds at least 5 rows, and each has `source` set.

**Commit:** `feat(research): E4 measured speed table (PINN, panel, NeuralFoil; FVM quoted)`.

---

### Task C5: E5, cylinder (where Prandtl-Glauert fails) plus extra shapes

**Goal:** show the full-Euler PINN beating the linearised panel method on a bluff body.

**Files:** modify `research/experiments.py` (`e5`).

**Design:**
- Mesh: `resample_closed` of a circle with radius 0.5, then `ogrid(cross_coeff=2.0)` at 50×25 and 100×50,
  written to `results/meshes/Cylinder_50_25.x` and `Cylinder_100_50.x`.
- Condition: **Ma 0.4, α 0** (see the facts table). Reference: `FVM_Cp/Cylinder_800_400_Cp.dat`.
  Rounds: 20 at 50×25 and 10 at 100×50.
- Rows (case `Cylinder_a0`): `pinn` at both meshes; `fvm` coarse meshes 50×25..400×200 against 800×400;
  `panel` using Plan B's `aeronet.panel.solve_xy(X, Y, 0, 0.4, lifting=False)` on the circle (expected
  RMS ≈ 0.30; Prandtl-Glauert can't reach the FVM's Cp_min of −3.70). Also add the cylinder's Cp_min to
  `notes`.
- E5b (Stretch): the mesh-correction effect. Rerun NACA0012 at 50×25 on `ogrid(cross_coeff=2.0)` from the
  same wall (10 rounds, seed 0) and compare RMS with the bundled-mesh run (method `pinn`, mesh
  `50x25-winslow2`).
- E5c (Stretch): random airfoils **only if** C3 found their conditions in the paper. Use the bundled
  `Airfoil<k>_200_100.x` meshes, or `cosine_wall` plus `ogrid` at 100×50 with `cross_coeff=0.5`.
  Otherwise write one line in M3: "skipped: reference conditions not published".

**Acceptance:** `collect e5` adds at least 4 rows. M3 states "PINN RMS vs panel+PG RMS on the cylinder".

**Commit:** `feat(research): E5 cylinder (PINN vs Prandtl-Glauert panel) and extras`.

---

### Task C6: Export runs to the app (case JSON, §5.7)

**Goal:** each chosen run becomes one self-contained JSON that the app can show without torch.

**Files:** create `research/export_results.py` and `research/tests/test_export.py`.

**Interface:**
- `export_case(run_dir, label, ref_path=None, lifting=True, width=340, out_dir=APP/"backend/data/pinn_cases") -> Path`
- `write_summary_json(csv="results/summary.csv", out=APP/"backend/data/results_summary.json")`: a list of
  row dicts; numeric columns become float, and empty cells become null.
- CLI: `export_results.py <run_dir> --label ... [--ref ...] [--non-lifting]`, `export_results.py --all`
  (the runs listed in a `CASES` constant at the top of the file: run id, label, ref, lifting) and
  `export_results.py --summary`.

**Approach:**
- `id` is the run id, and `shape`, `mesh`, `mach`, `alpha` come from `config.json`. `x`, `y` are the wall
  nodes in the app frame (open, N). `surface` is the final faces (`runs.final_surface`).
- `reference`: `{"name":"FVM 800×400","points": load_fvm(ref).tolist()}`, or `null` when there is no
  reference (drawn shapes, C10). Note for the roadmap: §5.7 should allow `null` here.
- `panel`: `aeronet.panel.solve_xy` on the wall nodes (app frame) with `lifting`. Store `cl` and the
  surface (`geom.cx`, `geom.cy`, `cp`).
- `metrics`: `cp_error` (if there is a ref), the final Cl and Cd from the log, and `cl_ref = fvm_forces(ref, alpha)`.
  `timing`: `train_s` from the log, `infer_ms` from `time_inference` (20 calls), `threads`, and the CPU name.
- `log`: copy the §5.3 keys. `snapshots`: `[{"round","cp"}]`, with Cp rounded to 4 decimals.
- **Field resampling:** `field.npz` arrays have shape (N+1, M) with a duplicated seam row. **Drop the
  last row first**, otherwise matplotlib rejects the triangulation ("Triangulation is invalid").
```python
idx = np.arange(N*M).reshape(N, M); nxt = np.roll(idx, -1, axis=0)        # periodic in xi
a, b, c, d = idx[:, :-1].ravel(), nxt[:, :-1].ravel(), nxt[:, 1:].ravel(), idx[:, 1:].ravel()
tri = matplotlib.tri.Triangulation(X.ravel(), Y.ravel(), np.r_[np.c_[a, b, c], np.c_[a, c, d]])
f = matplotlib.tri.LinearTriInterpolator(tri, cp.ravel())(gx, gy)   # masked exactly inside the body
```
  The grid matches the app's window: `xs = -0.4 + (i+0.5)/W*1.8`, `ys = 0.5 - (j+0.5)/H*1.0`,
  `H = round(W/1.8)`, row-major with the top row first. Masked cells become `None`. Round to 3 decimals.
  Do the same for u and v. Don't use `scipy.griddata`: its Delaunay triangulation bridges the body and
  fills its inside.
- Size guard: if the JSON is over 3 MB, retry with `width=240`; if still over, drop to 3 snapshots
  (first, middle, last).
- Note on u and v: NNfoil velocities are normalised by |V∞| = 1, the same as the panel field, so the
  particles from Plan B work unchanged.

**Tests (`test_export.py`, using a synthetic run folder built in `tmp_path` from the bundled 50×25 mesh
and analytic fields, no torch):**
- Field length = W·H. The `None` count equals the number of grid cells inside the wall polygon
  (`matplotlib.path.Path.contains_points`).
- The JSON has every §5.7 key, and `reference` may be null.
- Interpolating `cp = sin(3x)·exp(−r)` has max error < 0.01.
- File size < 3 MB.

**Acceptance:** `export_results.py --all` and then `--summary` write at least 2 case files (0012 50×25 and
100×50) and `results_summary.json`.

**Commit:** `feat(research): export trained runs and results to the app`.

---

### Task C7: Backend serves PINN cases and results

**Goal:** the app lists the trained cases and the results, and the PINN button unlocks when cases exist.

**Files:** create `app/backend/aeronet/pinn_cases.py` and `app/backend/tests/test_pinn_api.py`; modify
`app/backend/aeronet/main.py`, `solvers.py` and `app/backend/tests/test_api.py`.

**Interface:**
- `pinn_cases.CASES_DIR`, `RESULTS_FILE` (module attributes, so tests can monkeypatch them).
- `list_cases() -> list[dict]`: one summary per file: `{"id","label","shape","mach","alpha","mesh","rms","cl"}`,
  sorted by label. Cache by file mtime.
- `get_case(case_id) -> dict`: validate `^[a-z0-9_]+$` (path-traversal guard), else `KeyError`.
- `load_results() -> list[dict]` (empty list if the file is missing). `has_cases() -> bool`.
- Endpoints (§5.6): `GET /api/pinn/cases`, `GET /api/pinn/cases/{id}` (404 for an unknown or invalid id),
  `GET /api/results`. Return plain dicts (no `response_model`; the case JSON is large).
- `GET /api/solvers`: for `pinn`, `available = has_cases()` at request time, with note `"Trained cases
  only: open the PINN Lab tab."`. `POST /api/solve` with `solver=pinn` **still returns 409** with the detail
  "PINN results are per trained case: use GET /api/pinn/cases" (the PINN can't solve arbitrary inputs).
- Add the new routes **above** the `StaticFiles` mount at the end of `main.py`.

**Tests:**
- With a tmp `CASES_DIR` holding one fixture case (write a minimal valid dict in the test):
  `/api/pinn/cases` returns 1 item, `/api/pinn/cases/<id>` returns 200 with a `field`, an unknown id gives
  404, and `../x` gives 404.
- With an empty dir, `/api/solvers` has pinn `available=False`; with the fixture, `True`.
- `/api/results` returns `[]` when the file is missing, and the rows when present.
- Change `test_api.py::test_solvers_and_presets` to only check that `pinn` exists and has a note
  (availability now depends on data). `test_unavailable_and_unknown_solver` stays as it is (409 and 404).

**Acceptance:** `cd app/backend && .venv/Scripts/python -m pytest -q`: every test passes (29 + Plan B's +
these).

**Commit:** `feat(api): serve PINN cases and results; PINN availability from data`.

---

### Task C8: Frontend PINN Lab, training replay and numbers dashboard

**Goal:** the research results are live in the demo: PINN vs FVM vs panel, error numbers, loss curves,
"watch it learn", and the speed table.

**Files:** see the file map (frontend rows).

**Interface:**
- `api.js`: `getPinnCases()`, `getPinnCase(id)`, `getResults()`.
- `utils/pinn.js`:
  - `caseToResult(c)` returns a **SolveResponse-shaped** object
    `{solver:"pinn", code:c.shape, alpha:c.alpha, mach:c.mach, x:c.x, y:c.y, surface:{cx,cy,cp,upper}, cl:c.metrics.cl, cd:c.metrics.cd, solve_ms:c.timing.infer_ms, field:c.field}`.
  - `upper` is true for faces before the minimum-x face (the faces run CCW from the TE).
  - `snapshotResult(c, k)` is the same, with `surface.cp = c.snapshots[k].cp`.
  - Reuse the existing components unchanged. Before using `FlowCanvas` and `CpCard`, open those files and
    check that they read `result.field`, `result.x`, `result.y` and `result.surface`. If the prop names
    differ, adapt `caseToResult`, not the components.
- `PinnLab.jsx`:
  - Case picker (`.seg` buttons from `getPinnCases`).
  - `FlowCanvas` with `caseToResult(case)`.
  - `PinnCpChart`: canvas with a PINN line split upper/lower, a dashed panel line from `case.panel.surface`
    and hollow FVM circles from `case.reference.points`.
  - Metric tiles: RMS, max, Cl vs Cl_ref, Cd ("error only"), train time and inference ms, each with a
    small "measured" label.
  - Replay: a range input over `snapshots`, a ▶/⏸ button (600 ms per step) and the text "round k of n".
    The chart shows the snapshot Cp at that round and the FVM stays fixed.
- `TrainingCard({log})`: if `log` is missing, keep the current placeholder text. Otherwise draw two small
  canvases: log10(loss), log10(loss_f) and log10(loss_b) vs time (s); and Cl vs round with a horizontal
  Cl_ref line.
- `ResultsTable.jsx` (numbers dashboard): `getResults()`, grouped by `experiment` (E1..E5 headings with
  one-line explanations), columns method · mesh · case · RMS · Cl · train · infer · source badge.
  Empty cells show "–". Never show a number without its source.
- `ComparisonCard({result, rows})`: the PINN and FVM rows take their numbers from E4 rows when present
  (and keep the "measure Day N" text otherwise). Label quoted values "(quoted)".
- `App.jsx`: add a `pinn` tab labelled "PINN Lab" to Plan B's tab list (grep `App.jsx` for the tab array
  first and follow its pattern). Choosing the "NNfoil PINN" solver button switches to that tab. The PINN
  Lab tab also renders `TrainingCard`, `ComparisonCard` and `ResultsTable`.
- If a case has `reference: null` (drawn shapes from C10), hide the FVM series and the RMS tile.

**Tests (vitest, `utils/pinn.test.js`):**
- `caseToResult` on a 6-face toy case gives `upper = [T,T,T,F,F,F]` when the minimum x is at face 3.
- `snapshotResult(c, 0).surface.cp` equals `c.snapshots[0].cp`.
- `caseToResult` keeps `field` by reference (no copy).

**Acceptance:** `npm test` and `npm run build` pass. In the browser (use the preview tools with the
launch.json entry from Plan B), the PINN Lab shows at least 2 cases, the replay slider changes the curve,
and the console has no errors. Save 2 screenshots to `docs/milestones/img/m3-pinn-lab.png` and
`m3-replay.png`.

**Commit:** `feat(ui): PINN Lab with FVM overlay, training replay, numbers dashboard`.

---

### Task C9: Figures, tables and the Milestone 3 brief

**Goal:** every report figure and table is regenerated by one command, plus the M3 document for the mentor.

**Files:** create `research/make_figures.py`, `research/report_tables.py` and `docs/milestones/M3-pinn-results.md`.

**Figures (`results/figures/`, 300 dpi PNG, readable in greyscale, Cp axis inverted):**
1. `cp_naca0012.png`: FVM800 points, PINN 50×25 and 100×50 lines, panel dashed.
2. `mesh_study.png`: RMS vs number of mesh cells, log-log, FVM and PINN lines (E2).
3. `e1_loss.png`: log10 loss vs wall time, positivity on/off, both seeds.
4. `polar_2412.png`: Cl vs α. FVM points, panel line, thin-airfoil line, PINN markers. Title includes the
   Mach and its source label.
5. `cylinder.png`: Cp vs θ for FVM, PINN and panel+PG (E5).
6. `speed.png`: horizontal bars of per-query time on a log axis. Quoted bars are hatched, and the label
   says "quoted".
7. `training_replay.png`: PINN Cp after round 1, the middle round and the last round vs FVM.

Each figure function reads only `results/summary.csv` and `results/runs/*`. If its data is missing it
prints `SKIP <name>: missing <file>` and continues. The milestone check requires **zero SKIPs**.

**`report_tables.py`** writes `results/tables.md`: one Markdown table per experiment, from `summary.csv`,
with a source column. Plan D reuses it.

**M3 brief:** what was done; E1–E5 tables (paste them from `tables.md`); figures 1–7; the speed table and
hardware line; findings (positivity effect and remedies, sparse-mesh claim yes/no, Mach provenance for
2412, cylinder result); limits; screenshots from C8. Every number carries measured, quoted or fitted.

**Acceptance:** `research/.venv/Scripts/python research/make_figures.py` prints 7 `saved` lines and no
`SKIP`. The M3 doc has no "pending" left for E1–E4.

**Commit:** `docs: Milestone 3 brief, figures and tables from results`.

---

### Task C10: Train a PINN on your drawn shape (Tier S)

**Goal:** on the Draw tab, "Train a PINN on this shape" runs mesh → 50×25 PINN → export in a few minutes,
and the result opens in the PINN Lab.

**Feasibility, measured:** O-grid at 50×25 takes under 1 s (vectorised); one 50×25 training round takes
about 72 s. **Gate:** the mesh must be built in under 60 s **and** `is_valid`. Otherwise fail with
"shape too concave for the mesh generator: smooth it or make it more convex".

**Files:** create `research/train_shape.py`, `app/backend/aeronet/jobs.py`, `app/backend/tests/test_jobs.py`,
`app/backend/tests/fake_train_shape.py` and `app/frontend/src/components/TrainShapeButton.jsx`; modify
`main.py`, `api.js` and the Draw tab component from Plan B.

**Interface:**
- `research/train_shape.py --points <json file of [[x,y],...] app frame> --mach 0.3 --alpha 4 --rounds 4 --out results/runs/<id> --case-dir <dir> --label "Drawn shape"`.
  It writes `<out>/status.json` = `{"stage": "meshing|training|exporting|done|failed", "message": str, "case_id": str|null, "mesh_s": float}`
  at each stage. Steps: convert to the NNfoil frame (`x-0.5`); `resample_closed(…, 50)`;
  `ogrid(M=25, cross_coeff=2.0)`; gate; `write_x(<out>/mesh.x)`; run `run_case.py` as a subprocess with
  `--shape Drawn` and `train.log` redirection; then `export_case(out, label, ref_path=None, lifting=True)`.
  Exit code 0 on success, 3 on a mesh failure, 1 otherwise.
- `jobs.py`:
  - `research_python() -> Path|None`: the env var `AERONET_RESEARCH_PYTHON`, else
    `<repo>/research/.venv/Scripts/python.exe` (or `bin/python`), if it exists.
  - `TRAIN_SCRIPT`: env `AERONET_TRAIN_SCRIPT`, else `<repo>/research/train_shape.py`.
  - `start(points, mach, alpha, rounds) -> job_id` raises `RuntimeError("busy")` when a job is running.
    It saves `results/jobs/<job_id>/points.json` and starts `subprocess.Popen([...], cwd=repo, stdout=log file, stderr=STDOUT)`
    with Windows `creationflags=subprocess.CREATE_NO_WINDOW`.
  - `status(job_id) -> {"id","stage","round","rounds","cl","case_id","message","elapsed_s"}` reads
    `status.json` and the run's `log.json` (`round = len(log["round"])`, `cl = log["cl"][-1]`). If the
    process exited non-zero and the stage isn't `failed`, report `failed` with the last 5 lines of the log.
- Endpoints: `GET /api/pinn/jobs/capability` → `{"available": bool, "note"}`; `POST /api/pinn/jobs` with
  `{points (8..4000), mach (0..0.6), alpha (-10..10), rounds (1..6, default 4)}` → `{"id"}` (409 when busy,
  503 when there's no research python); `GET /api/pinn/jobs/{id}` → status (404 when unknown).
- `TrainShapeButton.jsx`: disabled with a tooltip when the capability is false. On click, show "about N
  min" (N = rounds × 1.3). Poll `GET` every 3 s, showing a stage, round k/n and the live Cl. When done,
  show "Open in PINN Lab", which switches the tab and selects `case_id`. Show the failure message
  verbatim on failure.

**Tests:**
- `test_jobs.py` (no torch): monkeypatch the env vars so they point to `sys.executable` and
  `fake_train_shape.py`. The fake script writes the stages, a 2-round `log.json` and a tiny case file,
  then exits 0. Assert the sequence ends with `done` and `case_id`. A second POST while running gives 409.
  A fake that exits 1 gives `failed` with a message. Capability is false when the python path is missing.
- `research/tests/test_ogrid.py::test_gate`: the deep C shape fails `is_valid`, and a sedan outline
  (`x=[.5,.45,.2,.05,-.2,-.42,-.5,-.48,.5]`, `y=[0,.08,.1,.2,.2,.08,.02,-.06,-.06]`) passes.
- Manual: draw a blob, train 3 rounds, and check that the case appears in the PINN Lab with a field.
  Record the wall time in M3.

**Acceptance:** backend tests pass. One real drawn-shape job completes in under 8 minutes. Add a
screenshot `docs/milestones/img/m3-train-your-shape.png`.

**Commit:** `feat: train a PINN on a drawn shape (background job, 50x25 O-grid)`.

---

## Self-review checklist (run before handing back)
- [ ] Roadmap promises P9–P16 and P19 each map to a task: E1 C1 · E2 C2 · E3 C3 · E4 C4 · E5 C5 · export C6 ·
  PINN mode C7+C8 · training history C8 · measured speed table C4+C8 · trained-shape list C8 · AoA sweep C3 ·
  figures C9.
- [ ] Every `summary.csv` row has `source`. No number appears in the UI or docs without a source.
- [ ] `pytest research/tests`, the backend pytest and `npm test` all pass. `npm run build` succeeds.
