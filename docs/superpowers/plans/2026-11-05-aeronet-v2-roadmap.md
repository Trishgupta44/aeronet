# AeroNet v2: Roadmap (read this first)

> **For agentic workers:** this is the master plan. It fixes the order of the work, the shared interfaces
> (file paths, JSON formats, API endpoints) and the milestone checks. The four phase plans in section 4 are
> task guides (goal, files, interface, approach, tests, acceptance check, commit). They are not full code:
> you write the code, test-first. Execute those one at a time with
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans.
> When a phase plan and this roadmap disagree on an interface, **this roadmap wins**: fix the phase plan.

**Goal:** turn AeroNet into a demo we can sell (good-looking live flow, hand-drawn shapes, real PINN
results) and a project with solid measured numbers. Deliver everything the existing docs promise.

**Written:** 2026-11-05, after reading the whole repo and running measurements on this laptop.

---

## 1. Where the project is now (measured on 2026-11-05)

| Item | State |
|---|---|
| `app/` React + FastAPI demo | Works. Backend tests: **29 passed**. Frontend tests are not run yet (`node_modules` missing). |
| Solver in the demo | Classical Hess-Smith panel method (stand-in). NACA 4-digit only. 2.2 ms at 140 panels, 18 ms at 400. |
| `NNfoil/` | Upstream code was added on 2026-11-05 as an **untracked folder with its own nested `.git`** (origin `github.com/Cao-WenBo/NNfoil`, head `14f12c5`). No licence file. 52 MB, of which 36 MB is meshes (the largest file is 12.5 MB). |
| PyTorch | Not in any project environment. `torch 2.15.0+cpu` installs and imports on this machine's Python 3.13.7. |
| CPU | 16 logical threads (the old plan assumed 4). |
| PINN timings | **Measured 2026-11-05** (NACA 0012, mesh 50×25, upstream network 5×128, L-BFGS `max_iter=1000`, 8 torch threads, minimal fixes only): **70.4 s, 73.1 s and 76.0 s** for rounds 1–3. Cl after each round: **0.5156, 0.5118, 0.5140**, with Cd 0.0032 → 0.0015 (panel 0.525; thin-airfoil 0.479). So 20 rounds at 50×25 take about 25 min. 100×50 and 200×100 are **not measured yet**. Estimates from the point count: about 5 min and about 20 min per round, which A12 must measure. |
| `deliverables/` (report, slides, synopsis, progress report, papers guide) | The root `README.md` lists them, but **they are not in the repo**. Phase D handles this. |

### 1a. Findings from the measurements (feed these into the report)

1. **The panel method matches the bundled FVM NACA 0012 data at Mach 0.4 and 4°.** RMS Cp error against
   the 800×400 file is **0.0094** with a quick nearest-panel match, and **0.0099** with the official §5.5
   metric (`cp_error`; report this one). It is 0.0113 against 400×200 and 0.0193 against 200×100. Against the
   coarse 100×50 and 50×25 FVM files it is 0.025 and 0.056. That gap is the FVM's own coarse-mesh error,
   which is exactly the paper's point: on sparse meshes FVM is inaccurate.
2. **The NACA 2412 sweep files (`FVM_Cp_NACA2412/*AoA*`) do NOT fit Mach 0.4.** At 4° the RMS is 0.036 at
   Mach 0.4, but 0.0108 at **Mach ≈ 0.25** (best fit, scanned in steps of 0.05). The same holds at 0°.
   Their Mach number must be confirmed from the NNfoil paper (arXiv 2401.08705). Until then, label it
   "fitted Mach 0.25".
3. **The random airfoils (`FVM_Cp_random_airfoil/Airfoil1..5`) are not all at Mach 0.4 and 4°.** The
   geometry matches the meshes exactly, but the best-fit conditions differ per airfoil (Airfoil1 ≈ Mach 0.25
   at 6°, Airfoil5 ≈ Mach 0.45 at 4.5°, and Airfoils 2 and 3 fit poorly inside the scanned range). Their
   conditions must come from the paper. Do not compare against them at a guessed condition.
4. **The cylinder FVM files (`FVM_Cp/Cylinder_*`) are at α = 0 and Mach ≈ 0.4** (the stagnation point
   sits at 180°, and its Cp of 1.044 matches Mach 0.4). Their minimum Cp is −3.70. Incompressible potential
   flow gives −3.00, and Prandtl-Glauert at Mach 0.4 gives −3.27. So the RMS error of the panel method
   with Prandtl-Glauert is about **0.30**: the linear Mach correction fails on bluff bodies. This is a
   strong "why the PINN (full Euler) is needed" result. Measure the PINN on the cylinder in Phase C.
5. **The PINN already beats the FVM on the sparse mesh (headline result, measured).** FVM Cl (from the
   ordered FVM Cp files) for NACA 0012 at 4° and Mach 0.4 is 0.4265 at 50×25, 0.4803 at 100×50, 0.5046 at
   200×100, 0.5146 at 400×200 and **0.5193 at 800×400**. The upstream PINN on the **50×25** mesh gave
   **Cl = 0.514** after one 72 s round. FVM on the same mesh is 18% low. After 3 rounds (no positivity)
   the PINN's Cp **RMS vs FVM 800×400 was 0.0271** (max 0.160), against **0.108** for FVM 50×25: about 4×
   more accurate on the same sparse mesh. A first 100×50 round (with positivity) gave Cl 0.372 and RMS 0.096.
   Its time (196 s) was measured while other jobs shared the CPU, so A12 must re-measure it on an idle machine. FVM coarse-mesh Cp error against
   800×400: RMS 0.108 at 50×25, 0.046 at 100×50, 0.019 at 200×100 and 0.006 at 400×200.
6. **Finding 3 is confirmed, with low impact.** `NNfoil/mesh_generation/mesh_gen.py` uses `-0.5*beta` on the
   cross-derivative term, while Winslow's equation needs `-2*beta` (4× too small). The bundled
   `NACA0012_50_25.x` matches the upstream coefficient to within 3.3e-4, so the shipped meshes were made with
   it. With `2*beta` the true Winslow residual falls from 1.95e-4 to 2.0e-6 and nodes move by up to 0.011
   chord. Both grids are valid (single-signed Jacobian, minimum grid angle 86.0° vs 86.8°).
7. **Mesh generation is fast enough for hand-drawn shapes.** 50×25 takes about 10 s, 100×50 takes 19 s
   (a concave kidney shape took 17 s) and 200×100 takes 280 s, with no folded cells in any case.
8a. **New finding 10: wrong-sign Jacobian at the trailing-edge seam.** NNfoil's `jacobian_trans(...,
   boundary=(1,1))` uses one-sided differences across the periodic ξ seam. This gives a near-zero det J of
   the wrong sign at the trailing-edge node on every bundled mesh (1–2 cells). Periodic differencing gives 0.
   The impact is low, since those cells get almost no weight in the J²-weighted loss. Plan A documents it and
   adds an opt-in flag.
8b. **Positivity slows early training (measured, 50×25).** With softplus on p and rho, round 1 hit
   "Failed, Restart" and Cl went 0.307 → 0.478 over rounds 2–3 (RMS 0.193 → 0.037). Without it, Cl was 0.516
   after round 1. E1 (Plan C) must use ≥ 10 rounds and 2 seeds before drawing any conclusion.
8d. **Plain NNfoil collapses on 100×50, and grid sequencing fixes it (separate research session, RAN, one
   seed).** From scratch on 100×50, every variant tried (original, float64, line search) fell into a trivial
   solution (PDE loss exactly 0, Cl about 0). Training on 50×25 and moving the weights to 100×50 reached
   **Cl 0.5065 (2.4% error) and Cp relative-L1 0.027 in 302 s total**, against 7.5% Cl error for the FVM on
   the same mesh. It is not in the NNfoil paper. Full details and further ideas: `docs/nnfoil_improvements.md`.
   Plan A task A12a implements it; Plan C uses it for the 100×50 and 200×100 results (C2) and adds ablations
   (C11).
8e. **Accuracy peaks early, then drifts while the loss keeps falling (measured, 50×25, positivity on, 19
   good rounds, 5.5 min total on an idle CPU).** Cp RMS vs FVM 800×400 by round: 0.193, 0.037, 0.030, 0.0275,
   0.030, 0.029, 0.034, then 0.068, 0.056 ... **0.073 at round 19**. Cl: 0.307, 0.478, 0.497, 0.499, 0.493,
   0.501, 0.482, then 0.41–0.44. The residual loss fell the whole time (1.1e-4 at round 3 against 7e-5 at
   round 19). The Euler residual does not pin the circulation (nothing enforces a Kutta condition, finding
   N22 in `docs/nnfoil_improvements.md`), so lower loss does not mean a better Cl. The final round is still
   better in Cp than the FVM on the same mesh (0.073 against 0.108), but its Cl (0.412) is no better than the
   FVM's (0.4265). Report the whole trajectory, not only the best round. Choosing a round after seeing the FVM
   error is post-hoc: label it so. Most rounds after round 3 stopped after 2–26 s (L-BFGS tolerance), so the
   run did far fewer iterations than the paper's 20,000. Candidate remedies (Plan C, E6/C11): a panel-method
   prior that carries the right circulation (idea I2 in the research doc), periodic input, and averaging or
   early stopping on a held-out criterion that does not use the FVM.
8f. **E1 first evidence (measured, 50×25, one seed each): the softplus positivity fix made training WORSE.**
   Without it (upstream behaviour, 18 good rounds, 2 failed, 416 s): Cp RMS vs FVM 800×400 stays at
   0.026–0.033 from round 2 on (best 0.0262), Cl 0.48–0.52, final loss 1.3e-5. With it (8e above): round 1
   fails, RMS drifts to 0.073, Cl to 0.41, loss stays near 7e-5. The upstream code never produced a negative p
   or rho in these runs, so there was no failure for the fix to cure, and softplus (+1e-3) slows the early
   fit. Finding 2 stays a valid *safety* concern but not a benefit on this case. The flag remains, default on
   in the code. **Use `--no-positivity` for the grid-sequenced and seed-check runs** (it matches the research
   doc's measurements), and let E1 (Plan C, 2 seeds, 20 rounds) decide the default. Do not claim the fix
   helps.
8c. **The FVM Cp files are not stored in contour order.** Integrating Cl directly from them gives nonsense
   (0.355). Sort the points by angle about the centroid first (`metrics.order_contour`).
9. **The "draw any shape" route works.** A prototype that resamples any closed curve and runs the panel
   method gave Cp_max 0.999 and Cp_min −2.999 for a circle (exact values 1 and −3). It reproduced
   NACA 0012's Cl (0.5247 against 0.5263 from the code path) even when the points were given in reverse
   order. The prototype code is in phase plan B, Task B1.

---

## 2. Everything that was promised (inventory) and where it is delivered

Sources: `docs/AeroNet_7_Day_Plan.md` (7DP), `docs/superpowers/plans/2026-09-24-aeronet-one-week-plan.md`
(1WP), `README.md`, `app/README.md`, and the text the app shows on screen.

| # | Promise | Source | Delivered in |
|---|---|---|---|
| P1 | NNfoil runs on CPU: device check, threads, NumPy 2, `torch.load` (findings 1, 4, 5, 6) | 7DP §2, Day 1 | A2 to A4 |
| P2 | Positivity of rho and p via softplus (finding 2), with a test that rho and p stay above 0.001 | 7DP Day 2, §9 | A5 |
| P3 | Cl and Cd computed and printed (finding 7) | 7DP Day 2 | A6 |
| P4 | Verdict on the mesh-smoothing coefficient (finding 3) | 7DP Day 2 | A9 |
| P5 | `requirements.txt` for the research code; each fix as its own commit | 7DP Day 1, 1WP §2 | A1, every A task |
| P6 | `docs/verification.md`: theory vs code vs verdict for the Euler equations, mesh transform and Jacobian, BCs and force integration | 7DP Day 2 | A9 |
| P7 | Small tests: freestream gives near-zero residual; surface mapping is correct | 7DP Day 2 | A8 |
| P8 | `validate.py`: PINN vs FVM RMS and max error and Cl, plus a Cp plot | 7DP Day 3 | A11 |
| P9 | E1 before vs after positivity; E2 mesh size vs error; E3 NACA 2412; E4 speed table incl. NeuralFoil; E5 UIUC shapes | 7DP Day 4, 1WP Day 4 | C1 to C5 |
| P10 | `results/summary.csv` (generated, not retyped) | 7DP Day 4 | A10 (format), C1 to C5 |
| P11 | `export_results.py`: one JSON per trained case | 7DP Day 5 | C6 |
| P12 | "NNfoil PINN" mode in the demo: dropdown of trained cases, FVM overlay, error shown | 7DP Day 5, app README | C7, C8 |
| P13 | Training-history panel filled from the training log | 7DP Day 5, app UI | C8 |
| P14 | Speed table with measured numbers (measured vs quoted labelled) | 7DP Day 5, app UI | C4, C8 |
| P15 | "Final demo lists shapes that have a trained model (0012, 2412, random airfoils)" | app UI | C7 |
| P16 | Angle-of-attack sweep (Cl vs α) | 1WP E3 | B5 (panel), C3 (FVM and PINN points) |
| P17 | Demo runs offline from a clean copy in under 2 minutes of setup | 7DP Day 5 | D5 |
| P18 | Report, slides with a demo script, synopsis, progress report, papers guide (with real numbers) | README, 7DP Days 6 and 7 | D2 to D4 |
| P19 | `make_figures.py`: all figures generated from scripts | 7DP Day 6, 1WP Day 6 | C9 |
| P20 | Viva Q&A sheet | 1WP §7 | D4 |
| P21 | Clean repo, tag `v1.0`, README quick start | 7DP Day 7 | D6 |
| P22 | Backup demo video (2 minutes) | 7DP Day 7 | D5 (script only; **a human records it**) |

**Deliberate change from 1WP:** 1WP planned a Streamlit demo. The React + FastAPI app replaced it, and it
stays. Say so in the report.

---

## 3. New features (beyond the docs), chosen to make the demo sellable

Ranked by value for cost. Tier: **M** = Must, **S** = Should, **X** = Stretch (cut from the bottom).

| Feature | Why it sells | Tier | Task |
|---|---|---|---|
| **Draw any 2D shape by hand** → instant Cp field, Cl, Cd, Cp curve | The user's main request; people love drawing | M | B1 to B3, B8 |
| **Animated flow particles** (wind-tunnel smoke) driven by the real velocity field | Makes the page come alive; the biggest visual upgrade | M | B4, B7 |
| **Isobar contour lines** and a perceptual colour map, plus a **hover probe** (Cp, speed at the cursor) | Looks like real CFD software | M | B7 |
| **Shape library**: circle, ellipse, teardrop, flat plate, sedan car, sports car, bird wing, UIUC `.dat` import | Instant "wow" presets; the car idea in 2D | M | B6, B8 |
| **Ground effect** (road under the car, method of images) | Makes the car shapes meaningful (downforce) | S | B2 |
| **Cl–α polar chart**, plus thin-airfoil line, plus FVM and PINN points | Engineering credibility in one picture | S | B5, B9, C3 |
| **Surface force arrows and lift vector** on the shape | Shows where the lift comes from | S | B7 |
| **Pin a result** (ghost overlay of the previous Cp curve) for A/B comparison | Turns the demo into a design tool | S | B9 |
| **PINN Lab**: trained cases, PINN vs FVM vs panel, error numbers, loss curve | The research result, live | M | C7, C8 |
| **Training replay**: slider or play button that shows the PINN's Cp curve after each training round | "Watch the network learn the physics" (very strong in a viva) | S | C6, C8 |
| **Numbers dashboard**: the validation tables from `results/summary.csv` inside the app | Mentor and buyers see the evidence | S | C8 |
| Export PNG, Cp CSV, `.dat` of a drawn shape; shareable URL state | Practical | S | B9 |
| **Train a PINN on your drawn shape** (background job: O-grid mesh at 50×25, 3–5 rounds ≈ 4–6 min, then the result appears) | The ultimate demo. Feasible: one 50×25 round measured at about 72 s on this CPU | S (cut to X if the O-grid for drawn shapes folds too often) | C10 |
| Presenter mode (`?present`: large fonts, no notices, auto-cycling presets) | Smooth live demo | X | B9 |

**Honesty rules for new features** (put them on screen): drawn shapes and cars use inviscid potential
flow, so there is no separation, drag is about 0 by d'Alembert, and the pressure at the rear of bluff
bodies is not realistic. Car results are labelled "idealized flow". Never show a car Cd as a real number.

---

## 4. Phases, plans and milestones

| Phase plan file | Contents | Ends with |
|---|---|---|
| `2026-11-05-plan-A-nnfoil-verified.md` | A1 to A12: vendor NNfoil, CPU fixes, positivity, Cl/Cd, tests, `run_case.py` CLI, verification table, metrics, FVM and panel baselines, first PINN runs | **Milestone 1** |
| `2026-11-05-plan-B-demo-v2.md` | B1 to B10: generic-shape panel solver, ground effect, velocity field, polar, shape library, new flow canvas (particles, isobars, probe), draw pad, polish | **Milestone 2** |
| `2026-11-05-plan-C-pinn-results.md` | C1 to C10: experiments E1 to E5, NeuralFoil timing, export to the app, PINN Lab, training replay, numbers dashboard, figures; optional train-on-your-shape | **Milestone 3** |
| `2026-11-05-plan-D-final-package.md` | D1 to D6: README and docs, report, slides, synopsis, progress report, papers guide, viva sheet, offline clean-copy test, tag `v1.0` | Final delivery |

**Order and parallelism.**
- A and B touch different files (A: `NNfoil/`, `research/`, `results/`, `docs/verification.md`; B: `app/`).
  They can run at the same time in two sessions. If you run them in one session, do **A1 to A6 first**,
  then start the long training runs in the background (A12) and do B while they train.
- C needs A finished (trained runs) and B finished (the canvas and API it extends).
- D needs C.

### Milestone 1: "NNfoil verified and running on a laptop CPU" (show to the mentor)
Deliverable: `docs/milestones/M1-nnfoil-verified.md` (written in Task A12). It contains:
- the bug and finding table, each with file:line, the fix, the commit hash and before/after evidence
- CPU timing per training round for 50×25 and 100×50
- a PINN vs FVM Cp figure for NACA 0012 at 4°, Mach 0.4 (50×25 and 100×50) with RMS, max and Cl
- the baseline table: FVM coarse vs FVM 800×400, panel vs FVM, and a row for the PINN
- the flow-condition findings from §1a (2412 and random airfoils)

Pass/fail check: `pytest research/tests` passes, `results/summary.csv` has at least 8 rows, both figures
exist, and every number in the M1 doc is labelled measured, quoted or fitted.

### Milestone 2: "Demo v2: draw any shape, live animated flow" (show to the mentor)
Deliverable: `docs/milestones/M2-demo-v2.md` (written in Task B10), with 4 screenshots (NACA preset, drawn
blob, car with ground, polar chart) in `docs/milestones/img/`, a 2-minute live demo script, and the
panel-method validation numbers shown in the app.

Pass/fail check: backend and frontend tests pass, `npm run build` succeeds, and the drawn-shape flow works
in the browser (verified by screenshots). The page keeps 30 fps or more for particles at 170-cell field
width (check with the browser's performance panel, or by logging frame times).

### Milestone 3: "The PINN in the demo, and the experiment results" (show to the mentor)
Deliverable: `docs/milestones/M3-pinn-results.md` (written in Task C9) with E1 to E4 tables and figures, the
measured speed table, and PINN Lab screenshots including training replay.

Pass/fail check: the PINN button in the app is enabled and shows at least 2 trained cases with FVM overlay
and error, `results/summary.csv` holds every experiment row, and `research/make_figures.py` regenerates
every figure.

---

## 5. Shared interfaces (contracts). Every phase plan must use exactly these.

### 5.1 Folder layout (new parts)
```
NNfoil/                     vendored upstream (nested .git moved out; see A1), then fixed in place
  run_case.py               CLI: train one case, write a run folder (A10)
research/
  requirements.txt          torch (CPU), numpy, scipy, matplotlib, neuralfoil
  metrics.py                cp_error(), cl_from_cp(), load_fvm(), load_mesh_wall()
  baselines.py              FVM coarse-vs-fine and panel-vs-FVM rows -> results/summary.csv
  validate.py               run folder vs FVM file -> metrics row + Cp figure
  experiments.py            E1..E5 drivers (Phase C)
  export_results.py         run folder -> app/backend/data/pinn_cases/<case_id>.json (Phase C)
  make_figures.py           all report figures -> results/figures/ (Phase C)
  ogrid.py                  O-grid generator for any closed curve, with a Jacobian check (Phase C, X)
  tests/                    pytest for all of the above
results/
  runs/<run_id>/            config.json, log.json, snapshots.json, model.pt, field.npz
  summary.csv               every number we report
  figures/                  PNG figures
docs/verification.md
docs/milestones/M1-…md, M2-…md, M3-…md, img/
app/backend/data/pinn_cases/<case_id>.json
app/backend/data/results_summary.json   (copy of summary.csv as JSON; written by export_results.py)
```
`results/runs/*/model.pt` is git-ignored. Everything else in `results/` is committed. A case JSON must
stay under 3 MB; downsample if needed.

### 5.2 Coordinates and units
- **App frame:** chord 1, leading edge at x = 0, trailing edge (rightmost point) at (1, 0).
- **NNfoil and FVM files:** chord in x ∈ [−0.5, 0.5]. Convert with `x_app = x + 0.5`, `y_app = y`.
- Angles are degrees in every API and JSON field (NNfoil uses radians internally).
- `Cp = (p − p∞) / (½ ρ∞ V∞²)`. In NNfoil units `Cp = (p − 1/(1.4 Ma²)) / 0.5`.

### 5.3 Run folder (written by `NNfoil/run_case.py`)
- `config.json`:
  `{"run_id","mesh_file","mesh":"100x50","shape":"NACA0012","mach","alpha_deg","layers":[...],"rounds","lbfgs_max_iter","threads","positivity":true,"seed","torch_version","started_utc"}`
- `log.json`: `{"round":[1..],"time_s":[cumulative],"loss":[],"loss_f":[],"loss_b":[],"cl":[],"cd":[]}`
- `snapshots.json`: `{"x":[...],"y":[...],"rounds":[{"round":1,"cp":[...]}, ...]}`. `x` and `y` are the
  wall **face centres** (`wall['X_f']`, N values, app frame), the same length as each `cp` list. There is
  one entry per round.
- `field.npz`: arrays `X, Y` (mesh nodes, app frame) and `u, v, p, rho, cp` with shape `(N_xi+1, N_eta)`.
- `model.pt`: `torch.save(model.state_dict())` (not the whole module).

### 5.4 `results/summary.csv` columns (exact order)
`experiment,case,method,mesh,mach,alpha_deg,rms_cp,max_cp,cl,cl_ref,cd,train_s,infer_ms,source,notes`
- `case` is `<Shape>_a<alpha>[_s<seed>]`, e.g. `NACA0012_a4_s1`. This keeps rows unique across seeds and angles.
- `method` is one of `pinn`, `pinn_nopos`, `pinn_posinit` (E1b only), `pinn_gs` (grid-sequenced),
  `pinn_robust`, `pinn_periodic`, `pinn_hardfar` (E6 ablations), `fvm`, `panel`, `neuralfoil`, `thin_airfoil`.
- `source` is one of `measured`, `quoted`, `fitted`.
- The reference for `rms_cp` is always the finest FVM file for that shape and condition (800×400), unless
  `notes` says otherwise. Empty cells stay empty, never 0.

### 5.5 Error metric (one definition for everything)
`research/metrics.py::cp_error(x, y, cp, ref)`: for each reference point with 0.02 ≤ x_app ≤ 0.98, take
the Cp of the **nearest** predicted surface point (Euclidean distance in the app frame). Return
`{"rms","max","n"}`. The app's existing `compare.py` uses a camber split instead. It stays for NACA codes,
so the app tests don't change. Drawn shapes use the nearest-point rule (B1). Report numbers come only from
`research/metrics.py`. Note for the report: the nearest-point rule is slightly harsher than interpolating
along the surface. On the bundled 200×100 sample it gives RMS 0.0231, against 0.0187 for the camber split.
It is applied the same way to every method, so the comparisons are fair.

### 5.6 HTTP API (backend, `app/backend/aeronet/main.py`)
Existing endpoints stay unchanged, except that the `field` object gains `u` and `v`.
- `field`: `{"width","height","xmin","xmax","ymin","ymax","cp":[...],"u":[...],"v":[...]}`. Row-major, top row
  first, `null` inside the body and below the ground.
- `POST /api/solve_shape`:
  `{"points":[[x,y],...] (8..4000), "alpha":4, "mach":0.0, "lifting":true, "ground":null|gap, "n_panels":200, "grid_width":170, "reference":null}`.
  Returns the `SolveResponse` fields plus `"shape":{"chord","thickness","area","n_input"}` and `"warnings":[str]`.
  `ground` is the gap between the lowest body point and the road, in chords (0.02..1.0).
  Returns 422 for a self-intersecting curve or one with fewer than 8 points.
  **With `ground` set**, the freestream stays horizontal (parallel to the road), and α rotates the body
  nose-up about (0.25, 0) instead. The response then also carries `"ground_y"` so the UI can draw the road.
  `cl_thin` is `null` for drawn shapes (they have no camber line). A shape that doesn't fit the view window
  after normalisation is rejected with a 422 and a clear message.
- `POST /api/shapes/parse_dat`: `{"text","name"}` returns `{"name","points"}`. It parses UIUC Selig and
  Lednicer `.dat` files.
- `POST /api/polar`: `{"code":"2412"} or {"points":[...]}` plus `"alphas":[...] (≤61), "mach", "lifting", "ground"`.
  Returns `{"alpha":[],"cl":[],"cl_thin":[]|null}`.
- `GET /api/shapes`: `[{"id","label","points","lifting","ground","note"}]`.
- `GET /api/pinn/cases`: `[{"id","label","shape","mach","alpha","mesh","rms","cl"}]`.
- `GET /api/pinn/cases/{id}`: the case JSON in 5.7.
- `GET /api/results`: the rows of `results_summary.json`.
- `GET /api/solvers`: `pinn.available` is true when `data/pinn_cases/` holds at least one JSON.

### 5.7 PINN case JSON (`app/backend/data/pinn_cases/<case_id>.json`)
```json
{
  "id": "naca0012_100x50_a4_m04", "label": "NACA 0012 · 100×50 · 4° · Ma 0.4",
  "shape": "NACA0012", "mesh": "100x50", "mach": 0.4, "alpha": 4.0,
  "x": [...], "y": [...],
  "surface": {"x": [...], "y": [...], "cp": [...]},
  "reference": {"name": "FVM 800×400", "points": [[x, y, cp], ...]}  /* or null for drawn shapes (C10) */,
  "panel": {"cl": 0.525, "surface": {"x": [...], "y": [...], "cp": [...]}},
  "metrics": {"rms": 0.0, "max": 0.0, "n": 0, "cl": 0.0, "cd": 0.0, "cl_ref": 0.0},
  "timing": {"train_s": 0.0, "infer_ms": 0.0, "threads": 8, "cpu": "…"},
  "log": {"round": [], "time_s": [], "loss": [], "loss_f": [], "loss_b": [], "cl": []},
  "snapshots": [{"round": 1, "cp": [...]}],
  "field": {"width": 340, "height": 189, "xmin": -0.4, "xmax": 1.4, "ymin": -0.5, "ymax": 0.5, "cp": [], "u": [], "v": []}
}
```
`x`, `y` and `surface` are in the app frame. The field is resampled from the curvilinear mesh onto the
app's regular grid (same window and row order as 5.6).

### 5.8 Research CLI and helper signatures (written in Plan A, used by Plans C and D)
- Training:
  `research/.venv/Scripts/python NNfoil/run_case.py --mesh NNfoil/mesh/NACA0012_100_50.x --shape NACA0012 --mach 0.4 --alpha 4 --rounds 20 --threads 8 --out results/runs/naca0012_100x50_a4_m04`
  with optional `--no-positivity`, `--layers 2,128,128,128,128,128,4`, `--lbfgs-iter 1000`, `--seed 0` and
  `--resume`. It writes the run folder of 5.3 **after every round**, so a killed run still has results.
- Run id: `<shape lowercase>_<NxM>_a<alpha>_m<mach>[_nopos][_s<seed>]`.
  - alpha: "m" prefix for minus and "p" for a decimal point, e.g. `a4`, `am4`, `a2p5`.
  - mach: Mach×10 as 2 digits when that is an integer (`m04`), otherwise Mach×100 as 3 digits (`m025`).
  - Examples: `naca0012_100x50_a4_m04`, `naca2412_100x50_am4_m025_nopos`.
- `research/metrics.py`:
  - `load_fvm(path) -> np.ndarray (K,3)`: (x_app, y, cp), header skipped.
  - `load_mesh(path) -> (xx, yy)`: closed arrays of shape (N+1, M) in the NNfoil frame, exactly like the upstream main script.
  - `load_mesh_wall(path) -> (x, y)`: wall nodes (N,) in the app frame.
  - `cp_error(x, y, cp, ref) -> {"rms","max","n"}` (5.5).
  - `forces_from_faces(xn, yn, cp_faces, alpha_deg) -> (cl, cd)`: xn and yn are N+1 closed wall nodes, and
    cp_faces has N face values. It uses the same normal and sign convention as NNfoil's `Calculate_Force`.
  - `order_contour(points) -> points`: sorts (x, y, cp) rows by angle about their centroid (CCW).
  - `fvm_forces(path, alpha_deg) -> (cl, cd)`: `load_fvm`, then `order_contour`, then face Cp = mean of adjacent nodes, then `forces_from_faces`.
  - `append_summary(row: dict, path="results/summary.csv")`: writes the header if the file is new, and
    replaces an existing row with the same (experiment, case, method, mesh).
- Validation:
  `research/.venv/Scripts/python research/validate.py results/runs/<id> --ref <FVM file> --experiment E2`
  prints the metrics, writes `results/runs/<id>/cp_vs_fvm.png` and calls `append_summary`.

---

## 6. Rules for the executing model (Sonnet)

1. **One task, one commit**, with the message given in the task. Run the task's tests before committing.
   Never use `--no-verify`.
2. **No fake numbers.** Every number in a doc, the UI or the report comes from a file in `results/` or a
   test output, and carries a source label: measured, quoted or fitted. If a run has not finished, write
   "pending", not an estimate.
3. **Long runs go in the background** and are logged to `results/runs/<id>/train.log`. While they run,
   continue with other tasks. Never wait idle on training.
4. **Budget rule** (from 7DP): if a run is predicted to take more than 3 hours on this laptop, shrink it
   (smaller mesh, or `layers=[2,64,64,64,64,4]`) and record the change in `notes`.
5. **Don't touch the panel-method maths** unless a task says so. The 29 existing backend tests must stay
   green throughout.
6. **Credit:** NNfoil is by Cao, Song and Zhang (arXiv 2401.08705) and has no licence. Keep the repo
   private. Credit them in the README, the app footer and the report.
7. **Windows:** use `python -m ...` from the right venv. The backend venv is `app/backend/.venv`, and the
   research venv is `research/.venv` (torch lives only there). Paths in commands use forward slashes.
8. **Stop and ask the user** if a checkpoint fails twice, if an interface in §5 seems wrong, or before
   deleting anything.
