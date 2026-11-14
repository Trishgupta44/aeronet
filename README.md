# AeroNet: Major Project

AeroNet is a web app and a research pack about physics-informed neural networks (PINNs) for airflow around airfoils. It
verifies NNfoil, a PINN Euler solver, runs it on a laptop CPU and measures it against a finite-volume (FVM) reference and a
classical panel method. The app lets you explore airfoils, draw any 2D shape, and open the trained networks in a PINN Lab with
every number labelled **measured**, **quoted** or **fitted**.

**Credit.** The PINN solver is **NNfoil** by Cao, Song and Zhang, "A solver for subsonic flow around airfoils based on
physics-informed neural networks and mesh transformation" (arXiv 2401.08705), vendored in `NNfoil/` with our fixes listed in
[`NNfoil/AERONET_CHANGES.md`](NNfoil/AERONET_CHANGES.md). The upstream repository (https://github.com/Cao-WenBo/NNfoil) has **no
licence file**, so keep this repository **private** and keep the credit if you share any of it.

## What is in this folder

| Path | What it is |
|---|---|
| `NNfoil/` | The vendored NNfoil code and its bundled FVM reference data (`FVM_Cp*`, `mesh/`), with the CPU fixes and the run CLI `run_case.py` |
| `research/` | Experiments, metrics, figures, table printers, the exporter to the app, and the research tests (`research/tests/`). Needs `research/.venv` (PyTorch lives only there) |
| `results/` | `summary.csv` (every result row with its source label), per-experiment CSVs, `figures/`, and `runs/<id>/` (config, log, snapshots of each finished run) |
| `app/` | The demo: React (Vite) frontend and FastAPI backend with the panel-method solver. See [`app/README.md`](app/README.md) |
| `docs/` | Design report, handoff, verification table, demo script, design brief, and the plans in `docs/superpowers/plans/` |
| `docs/milestones/` | The three milestone reports ([M1](docs/milestones/M1-nnfoil-verified.md), [M2](docs/milestones/M2-demo-v2.md), [M3](docs/milestones/M3-pinn-results.md)) and the screenshots in `img/` |
| `prototype/` | The first single-file demo (plain HTML/JS, panel method), kept as the reference the app was ported from |
| `deliverables/` | Not in the repository yet: the report, slides and other documents are generated from Markdown in `docs/deliverables_src/` by the builders in `research/docs_build/` (Plan D) |

Start with [`docs/HANDOFF.md`](docs/HANDOFF.md) for the current state, [`docs/verification.md`](docs/verification.md) for the
line-by-line check of NNfoil, and [`docs/demo_script.md`](docs/demo_script.md) for the 5-minute live demo.

## Quick start: the app

Needs Python 3.10+ (developed on 3.13) and Node.js 18+. Details, endpoints and URL options are in [`app/README.md`](app/README.md).

Windows, one command (builds the frontend and creates `app/backend/.venv` on the first run; the first run needs the internet for
`npm install` and `pip install`):

```powershell
app\start.ps1
```

Then open http://localhost:8000. Manually:

```bash
cd app/frontend && npm install && npm run build      # once
cd ../backend && python -m uvicorn aeronet.main:app --port 8000
```

`app/backend/.venv` is optional: any Python with `app/backend/requirements.txt` installed works (the system Python 3.13 was used
for the backend tests). The PINN Lab needs no PyTorch: the trained cases are already exported to `app/backend/data/`.

## Quick start: the research

From the repo root. Windows paths shown; on Linux or macOS use `bin/` instead of `Scripts/`.

```bash
python -m venv research/.venv
research/.venv/Scripts/python -m pip install -r research/requirements.txt     # CPU PyTorch, NumPy, SciPy, Matplotlib, pytest, NeuralFoil

# train one case (writes a run folder after every round; 8 threads; measured on this laptop: 334 s for a 50x25 run, 578 s for a 100x50 run)
research/.venv/Scripts/python NNfoil/run_case.py --mesh NNfoil/mesh/NACA0012_100_50.x --shape NACA0012 --mach 0.4 --alpha 4 --rounds 20 --threads 8 --out results/runs/naca0012_100x50_a4_m04

research/.venv/Scripts/python research/make_figures.py       # the m3_*.png figures in results/figures/
research/.venv/Scripts/python research/report_tables.py      # the Markdown tables of the M3 report, from results/
research/.venv/Scripts/python research/export_results.py     # finished runs -> app/backend/data (what the PINN Lab shows)
research/.venv/Scripts/python research/check_links.py        # every relative link in the Markdown files must exist
```

Optional training flags: `--no-positivity`, `--layers`, `--lbfgs-iter`, `--seed`, `--resume`. The run id and the other
conventions are in the roadmap, section 5.8. Timed runs need an idle, awake CPU.

**Tests** (counts as run on 2026-11-14):

| Suite | Command | Result |
|---|---|---|
| Frontend (Vitest) | `cd app/frontend && npm test` | 106 passed |
| Backend (pytest) | `cd app/backend && python -m pytest -q` | 153 passed, 2 skipped (NeuralFoil is optional) |
| Research (pytest) | `OMP_NUM_THREADS=1 research/.venv/Scripts/python -m pytest research/tests -q` | 228 passed (experiments, metrics, exporters, document builders, link checker) |

## Key results

NACA 0012, Mach 0.4, 4 degrees, against the FVM 800x400 solution (Cl 0.5193); Cp RMS is the nearest-point metric of
`research/metrics.py`. All rows were produced on one laptop CPU (AMD Ryzen 9 6900HS, 8 threads) and come from
`research/report_tables.py`, which reads `results/summary.csv`. Single runs unless stated; see the milestone reports for seeds.

| Result | Value | Source row | Label |
|---|---|---|---|
| FVM on its own 100x50 mesh | Cp RMS 0.0460, Cl 0.4803 | `summary.csv` E2/B0, fvm 100x50 | measured |
| PINN, cold start, positivity on, 100x50 | Cp RMS 0.0145, Cl 0.5163, 577.6 s | `summary.csv` E2, pinn 100x50 | measured |
| PINN, grid-sequenced, positivity off, 100x50 (4 rounds, after a 50x25 run) | Cp RMS 0.0148, Cl 0.5117, 143.9 s | `summary.csv` E2, pinn_gs_nopos 100x50 | measured |
| FVM 200x100 against the PINN on the same mesh | Cp RMS 0.0186 against 0.0092 (training time withheld: the laptop slept) | `summary.csv` E2, fvm and pinn_gs_nopos 200x100 | measured |
| PINN at 50x25 (final round): positivity off, positivity on | Cp RMS 0.0325 and 0.0734, against the FVM's 0.1081 | `summary.csv` E2, pinn_nopos and pinn 50x25 | measured |
| Panel method (Hess-Smith + Prandtl-Glauert), 200 panels | Cp RMS 0.0099, Cl 0.5262 | `summary.csv` B0, panel | measured |
| Cylinder, Mach 0.4: PINN against panel + Prandtl-Glauert | Cp RMS 0.1337 to 0.1403 against 0.1877 (FVM 200x100: 0.0790) | `summary.csv` E5 | measured |
| Seed check, grid sequencing, 100x50 (8 runs) | Cl 0.502 to 0.516 in all 8 | `results/seed_check.csv` | measured |
| Evaluate a trained PINN over the whole 100x50 field | 3.75 ms (8 threads); the panel solve takes 3.64 ms; the paper's FVM takes 24 s on another machine | `results/e4_speed.md` | measured; FVM quoted |

The reading, in one line: on a sparse mesh the PINN is more accurate than the FVM on the same mesh (about 3x at 100x50, 2x at
200x100), but it is not a speed win, it answers only the one condition it was trained for, and the 50x25 training drifts if it
runs too long. Full tables: [`docs/milestones/M3-pinn-results.md`](docs/milestones/M3-pinn-results.md) (printed by
`report_tables.py headline`, `e4`, `e5` and `seed_check`). The NNfoil check itself is in [`docs/verification.md`](docs/verification.md)
and [`docs/milestones/M1-nnfoil-verified.md`](docs/milestones/M1-nnfoil-verified.md).

## What does not work, and the limits

- **The stop rule failed.** A rule that stops training without the reference stopped 0 of 8 drifted 50x25 runs (0 of 5 on the
  development runs, 0 of 3 on held-out runs). What works in practice is a fixed early stop at rounds 3 to 5, and even that is not
  risk-free (`results/e6_stop_rule.md`, M3 section 9).
- **The late drift is unsolved.** At 50x25 every run is accurate by round 3, then 5 of 6 degrade while the training loss keeps
  falling (the loss is negatively correlated with the error). No fix has been tried; the candidates are a Kutta-type constraint,
  periodic input and a panel-method prior (M3 sections 3 and 10).
- **Cold starts can collapse.** One of 3 seeds collapsed without positivity at 100x50, and one NACA 2412 run with positivity
  collapsed at 4 degrees (kept in the data, with two working retries). Grid sequencing (our addition: train on 50x25, then move the
  weights to 100x50) did not collapse in 8 runs.
- **Inviscid only.** The Euler equations and the panel method have no friction, separation or wake, so there is no stall and the
  drag shown for a lifting body is numerical error. The optional NeuralFoil check is viscous and not comparable.
- **Cars are idealized.** The sedan and sports car are potential-flow outlines with a road. The pressure behind them is not
  realistic and the app never shows a car's drag.
- **A trained network answers one condition** (one shape, Mach number and angle). A new condition means retraining; the panel method
  needs no training. The PINN Lab shows trained cases and cannot sit behind the Explore sliders, so `solver=pinn` on `/api/solve` returns 409 by design.
- **Mostly single seeds, one laptop.** Replication exists only for the seed check and the 50x25 runs. Some numbers are quoted from the
  paper (the FVM times, the Mach number of the NACA 2412 and cylinder files) and are labelled so.
- **Not done:** random-airfoil runs, a 400x200 PINN, a parametric angle of attack, 3D, and training a network on a drawn shape.

## Run the prototype (reference only)

```
python -m http.server 8765 --directory prototype
node prototype/test_solver.js
```

The prototype's solver is a classical panel method (no neural network); the app in `app/` supersedes it.
