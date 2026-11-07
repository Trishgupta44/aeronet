# AeroNet v2: Design Report (read before handing to Sonnet)

*Written 2026-11-05. It explains what will be built, what each module does, and why each design choice
was made (with the alternative we rejected). The step-by-step instructions for Sonnet are in
`docs/superpowers/plans/` (roadmap plus Plans A–D). This report is for you and your mentor.*

---

## 1. The idea in one paragraph

AeroNet becomes two things at once. First, a **verified research result**: NNfoil (a physics-informed
neural network for airfoil flow) fixed, run on a laptop CPU, and compared against the FVM reference data
with one honest metric. Second, a **sellable live demo**: draw any 2D shape, or pick an airfoil or a car,
and instantly see an animated, CFD-looking flow with pressure, lift and forces. There is also a "PINN Lab"
that shows the trained neural network's results, how it learned round by round, and how it compares with
the FVM. Every number on screen or in the report comes from a file in `results/` and is labelled
**measured**, **quoted** (from a paper) or **fitted**.

## 2. What we already found today (real measurements, 2026-11-05)

| Finding | Number | Why it matters |
|---|---|---|
| **The PINN beats the FVM on a sparse mesh** | On a 50×25 mesh: PINN Cl **0.514** after one 72 s training round. FVM on the same mesh: **0.426**. Fine-mesh FVM truth: **0.519**. After 3 rounds the PINN's Cp error was **RMS 0.027**, against **0.108** for the FVM on the same mesh (about **4× better**) | The paper's headline claim, reproduced on your laptop. This is the core result for M1 |
| NNfoil runs on CPU | about **72 s per training round** at 50×25 (8 threads); 20 rounds ≈ 25 min | Training is feasible without a GPU |
| FVM error grows on coarse meshes | Cp RMS vs 800×400: 0.108 (50×25), 0.046 (100×50), 0.019 (200×100), 0.006 (400×200) | The baseline the PINN must beat |
| Panel method (current demo) vs FVM | RMS 0.0099 on NACA 0012 | The stand-in is accurate for slender airfoils… |
| …but it is weaker on a cylinder | RMS 0.188 (corrected 2026-11-08; an earlier 0.30 was the incompressible circle); its Cp_min is −3.27 vs FVM's −3.70. The PINN reached RMS 0.134–0.140 and Cp_min −3.54 to −3.67 | A clear "why we need the PINN" result |
| Finding 3 (mesh smoothing) | **Confirmed**: the cross term is 4× too small (−0.5β instead of −2β). Low impact: both grids are valid | Settles a claim from the proposal with evidence |
| **New finding 10** | Wrong-sign Jacobian at the trailing-edge seam on every bundled mesh (1–2 cells). Periodic differencing fixes it | A bug nobody had listed |
| Positivity fix (finding 2) | It works (p and rho ≥ 0.001), but it **slowed early training** in a first test | An interesting result for experiment E1. Needs ≥ 10 rounds and 2 seeds |
| Data surprise | The NACA 2412 FVM files fit Mach ≈ 0.25, not 0.4. The random airfoils have different conditions each | We must confirm these in the paper before comparing |
| Mesh generation for drawn shapes | about 1 s at 50×25 and 5 s at 100×50, with no folds for normal shapes | Training a PINN on **your own drawn shape** takes a few minutes, so it's in the plan |

## 3. The three milestones (show these to your mentor)

| Milestone | What you show | Proof |
|---|---|---|
| **M1: NNfoil verified on a laptop CPU** (Plan A) | The bug and findings table with before/after evidence, CPU timings, the PINN vs FVM Cp figure, and the headline table (PINN vs FVM on the same sparse mesh) | `docs/milestones/M1-nnfoil-verified.md`, the tests pass, `results/summary.csv` |
| **M2: Demo v2** (Plan B) | Draw any shape, animated particles, isobars, the car with a road under it, the polar chart, pin-and-compare | `docs/milestones/M2-demo-v2.md` with 4 screenshots and a 2-minute demo script |
| **M3: PINN in the demo, plus experiments** (Plan C) | The PINN Lab with training replay, experiments E1–E5, the measured speed table, train-your-own-shape | `docs/milestones/M3-pinn-results.md`, the figures, the numbers dashboard in the app |
| Final (Plan D) | Report (.docx), 17 slides (.pptx), synopsis, progress report, papers guide, viva sheet, offline test, `v1.0` tag | `deliverables/` |

Plans A and B touch different files, so they can run **at the same time** in two Sonnet sessions. C needs
both, and D needs C.

---

## 4. Module by module: what it does and why it's designed this way

### 4.1 Repository and NNfoil (Plan A1)
**What:** NNfoil's folder becomes part of your repo. The first commit is the untouched upstream code, then
each fix is its own commit.
- **Decision:** vendor NNfoil (move its `.git` out, to `Downloads/NNfoil-upstream.git`).
  **Why:** `git log NNfoil/` then *is* the bug-fix evidence your mentor can read, one diff per finding.
  **Rejected:** a git submodule (your fixes would live in a separate fork, which is harder to show), and
  copying the code into a new folder (the diff against upstream would be lost).
- **Decision:** keep the repo private, and credit Cao, Song and Zhang everywhere.
  **Why:** the upstream repo has no licence.
- **Decision:** use a separate research environment (`research/.venv`) with CPU PyTorch.
  **Why:** the web app stays light and offline (no 200 MB torch install for the demo).

### 4.2 NNfoil fixes (Plan A2–A6): one commit each, each with a test
| # | Problem | Fix | Design note |
|---|---|---|---|
| 1 | Always picks the GPU, so it crashes on a laptop | `pick_device()`: GPU if present, else CPU | Testable by faking "no CUDA" |
| 2 | p and rho can go ≤ 0 (division by rho) | softplus + 0.001 on the p and rho outputs, **behind a flag** | The flag lets E1 measure before vs after on the same code |
| 4 | Uses 1 CPU thread | Up to 8 threads | 8 of the 16, so the laptop stays usable |
| 5 | `np.math` removed in NumPy 2 | `math.factorial` | Done first, because nothing runs without it |
| 6 | Checkpoint save/load breaks on new PyTorch; file name shared between runs | Save `state_dict` to a per-run path | Safer (`weights_only`) and parallel runs don't overwrite each other |
| 7 | Lift is never computed | `evaluate_wall()` returns Cl and Cd | Computed every round, so the Cl-vs-round curve is free |
| 8 | Mach and angle hard-coded | Constructor arguments | **One model per flow condition is kept.** The paper's "continuous angle" is not in this code, and faking it would be dishonest |
| 3 | Mesh-smoothing coefficient | **Documented, not changed** in `mesh_gen.py` | The bundled meshes and FVM data were made with the old one. The corrected version lives in the new `ogrid.py` |
| 10 | Wrong-sign Jacobian at the seam | Opt-in flag only | Changing the default would make our runs incomparable with upstream. Plan C can ablate it |

### 4.3 `NNfoil/run_case.py`: the training command (Plan A10)
**What:** one command trains one case and writes a standard "run folder": settings, a loss and Cl log, the
wall Cp after every round, the flow field, and the model.
- **Decision:** rewrite every file **after each round**. **Why:** a run killed after 2 hours still has
  2 hours of results. **Rejected:** saving only at the end.
- **Decision:** store the Cp after every round (snapshots). **Why:** it powers the "watch the network learn"
  replay in the demo at almost no cost.
- **Decision:** fixed run names (`naca0012_100x50_a4_m04`). **Why:** every script can find any run
  without a database.

### 4.4 `research/metrics.py`: one definition for every number (Plan A7)
**What:** the loaders for meshes and FVM files, the Cp error, lift from pressure, and the results table writer.
- **Decision:** **one error metric for everything** (nearest surface point, ignoring the 2% at each tip).
  **Why:** PINN, FVM, panel and NeuralFoil are all judged the same way, so the comparisons are fair. It is
  slightly stricter than the app's old metric (0.0231 vs 0.0187); the report says so.
- **Decision:** reorder FVM points before integrating lift. **Why:** we found the FVM files are not stored
  in surface order (an unordered integration gives a wrong Cl of 0.355 instead of 0.519).
- **Decision:** one `results/summary.csv` that every script appends to. **Why:** the report, slides and
  app tables are all generated from it, so no number is retyped by hand.

### 4.5 Validation and baselines (Plan A11)
**What:** `validate.py` compares a run with the FVM data (numbers plus a figure). `baselines.py` writes the
FVM coarse-vs-fine rows, panel vs FVM, and thin-airfoil theory.
- **Decision:** the reference is always the finest FVM file (800×400). **Why:** it is the closest thing to
  "truth" we have. Same-mesh comparisons are extra rows, clearly labelled.

### 4.6 Generic-shape solver (Plan B1–B2)
**What:** the panel solver now accepts *any* closed outline, not just NACA codes, with an optional road
under the shape.
- **Decision:** one core `solve_xy()`, with the old `solve(code)` as a thin wrapper. **Why:** existing
  results stay identical (the 29 old tests stay green). **Rejected:** a second, separate solver.
- **Decision:** clean up every drawn shape (fix direction, start at the rear point, resample evenly, scale
  to chord 1). **Why:** a hand drawing is messy. After this the solver always sees a tidy outline.
- **Decision:** a self-intersecting outline is refused with a clear message. **Why:** physics is undefined
  for it. Better to say so than to show garbage.
- **Decision:** a "lifting" switch (Kutta condition on or off). **Why:** a circle or a bluff body should not
  get artificial lift. Airfoils need the Kutta condition.
- **Decision:** the road uses the **method of images** (reflect the evaluation point). **Why:** it is
  exact (no flow through the road, checked to 0.0) and a tiny code change. With a road, the angle of
  attack tilts the *body*, not the wind, otherwise the road would leak. **Rejected:** extra mirrored panels.

### 4.7 API additions (Plan B3–B6, C7)
**What:** `solve_shape`, `polar`, `shapes`, `parse_dat`, `pinn/cases` and `results` endpoints, plus a
velocity field (u, v) for the animation.
- **Decision:** the backend computes the velocity on the same grid as the pressure. **Why:** the browser
  can animate thousands of particles from it with no extra requests. **Rejected:** asking the server per
  particle.
- **Decision:** `.dat` airfoil files are parsed in Python. **Why:** testable with pytest and reusable by the
  research scripts.
- **Decision:** **no PyTorch inside the app.** The trained PINN results are exported once to small JSON
  files. **Why:** the demo stays fast, light and offline-safe. Live inference adds nothing, because each
  model only knows one flow condition. **Rejected:** loading models in FastAPI.

### 4.8 The new flow view: `FlowCanvas` (Plan B7)
**What:** the pressure picture becomes a layered live canvas: a smooth colour map, isobar contour lines,
animated "smoke" particles, a hover probe (Cp and speed under the cursor), surface force arrows and a lift
vector, with toggles for each.
- **Decision:** plain Canvas 2D, **no new libraries**. **Why:** the offline demo stays small and nothing
  can break on install day. **Rejected:** WebGL or charting libraries.
- **Decision:** particles drawn as a dark halo with a white core. **Why:** visible on both deep blue and
  near-white regions.
- **Decision:** a 30 fps target, plus a "reduced motion" setting and a pause button. **Why:** smooth on a
  normal laptop and accessible, and pause is useful for screenshots.

### 4.9 Draw tab: `DrawPad` (Plan B8)
**What:** draw with the mouse, pen or finger. The outline is smoothed, auto-closed and solved when you lift
the pen. There are presets (circle, ellipse, teardrop, flat plate, bird wing, sedan, sports car), a
mirror-symmetry toggle, undo, and `.dat` import and export.
- **Decision:** the car presets use a road (ground effect) and are labelled "idealized flow". **Why:**
  honesty. Inviscid flow has no wake, so a car's drag is not real, and we never show it as a number.
- **Decision:** drawings are not stored in the URL. **Why:** short, safe links. Presets and settings are
  stored instead.

### 4.10 Polish (Plan B9)
The new visual design (a wind-tunnel look with light and dark themes), the Cl–α polar chart, pin and
compare (a ghost of the previous result), PNG and CSV export, a shareable URL, `?present` mode for the
live demo, and an NNfoil credit footer.

### 4.11 Experiments E1–E5 (Plan C1–C5)
| Exp | Question | Design choice |
|---|---|---|
| E1 | Does the positivity fix help? | Same mesh, 2 seeds, 20 rounds, on vs off. We already saw it slows the start, so 2 seeds avoid a lucky or unlucky single run |
| E2 | Does the PINN beat the FVM on sparse meshes? (the paper's claim) | 50×25, 100×50 (and 200×100 if under 3 h), all against 800×400 |
| E3 | A second airfoil plus an angle sweep | NACA 2412 at −4°, 0°, 4° and 8°. The Mach number is confirmed from the paper first; otherwise it is labelled "fitted 0.25" |
| E4 | Speed: is the PINN fast to use? | PINN inference (warm-up, then 100 calls), training time, panel, NeuralFoil (measured: 3.5 ms per call), FVM time quoted from the paper and labelled "quoted" |
| E5 | Where does the simple model fail? | The cylinder at Mach 0.4: panel + Prandtl-Glauert is off by RMS 0.188 (measured), and the PINN reached 0.134–0.140: better, by a modest margin |

- **Decision:** the airfoil experiments use the **original** mesh coefficient, and new shapes use the
  corrected one. **Why:** our airfoil runs stay comparable with the bundled meshes and FVM data.

### 4.12 Export and PINN Lab (Plan C6–C8)
**What:** each trained run becomes one JSON file (≤ 3 MB): surface Cp, field, FVM reference, metrics,
timing, log and per-round snapshots. The app's **PINN Lab** tab shows the PINN vs FVM vs panel Cp curve,
the error numbers, the loss curve, a **training replay** slider and the numbers dashboard.
- **Decision:** the PINN field (on a curved mesh) is resampled onto the app's grid with a triangulation
  that respects the body. **Why:** it reuses the same canvas, so the PINN flow is animated too. **Rejected:**
  `scipy.griddata` (its triangles bridge across the body).
- **Decision:** the "PINN" button on the main solve stays locked for arbitrary inputs. **Why:** a model
  only knows its one trained condition. Pretending otherwise would be fake.

### 4.13 Train a PINN on your drawn shape (Plan C10)
**What:** on the Draw tab, the button "Train a neural network on this shape" builds a 50×25 mesh (about
1 s), trains 3–5 rounds in the background (about 4–6 min), and the result appears in the PINN Lab.
- **Decision:** one job at a time, only if the research environment exists, refused if the mesh folds.
  **Why:** protects the laptop during a demo and gives a clear message when a shape is too weird.
- **Tip for the live demo:** start a training job at the beginning of your talk and show the result at the end.

### 4.14 Final package (Plan D)
The report, slides, synopsis, progress report and papers guide are written as **Markdown and generated**
into .docx and .pptx by small scripts. The numbers come in through placeholders
(`{{E2|pinn|50x25|rms_cp}}`), and a missing number is a hard error. **Why:** no stale or typed-in numbers,
and when a run finishes you regenerate the documents in seconds.

---

## 5. Risks and what the plan does about them

| Risk | Mitigation in the plan |
|---|---|
| 100×50 or 200×100 training is too slow | Measure round 1 first. 3-hour budget rule: shrink the mesh or network and say so |
| The positivity fix makes results worse | Report it honestly; Plan C lists two remedies (better output initialisation) |
| Flow conditions of the 2412 and random-airfoil files are unknown | Confirm from the paper; otherwise label "fitted"; never compare at a guess |
| A drawn shape breaks the solver or mesher | Validation with clear messages; mesh fold check before training |
| Particles are slow on the demo laptop | A 30 fps check in M2; reduced-motion and pause options |
| The demo laptop fails on the day | Offline clean-copy test, `?present` mode, and a backup video (D5) |

## 6. Things only you can do

1. **Tell Sonnet (Plan D asks you) whether the original deliverables** (report, slides, synopsis) exist
   somewhere outside the repo. The README lists a `deliverables/` folder, but it isn't there.
2. **Record the 2-minute backup video** (Plan D5 gives the shot list).
3. **Confirm the flow conditions** for the NACA 2412 and random-airfoil files from the NNfoil paper, if
   Sonnet can't read it.
4. **Ask the NNfoil authors** before making anything public (no licence).
5. **Show the mentor** after each milestone; Sonnet stops and tells you when M1, M2 and M3 are ready.

## 7. How to hand this to Sonnet

Two sessions in parallel (they touch different files):

> **Session 1:** "Read `docs/superpowers/plans/2026-11-05-aeronet-v2-roadmap.md`, then execute
> `2026-11-05-plan-A-nnfoil-verified.md` task by task with superpowers:subagent-driven-development.
> Commit after each task. Stop at Milestone 1 and report."

> **Session 2:** "Read the roadmap, then execute `2026-11-05-plan-B-demo-v2.md` the same way. Don't touch
> `NNfoil/` or `research/`. Stop at Milestone 2 and report."

When both are done: one session runs Plan C (stop at M3), then Plan D.

A separate research session is also looking for further NNfoil improvements. Its write-up will be
`docs/nnfoil_improvements.md`. Read it before Plan C: its top ideas can become extra experiments.
