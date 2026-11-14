# Deploying AeroNet

Three ways, from least to most to run. All of them are built from this repository; none of them publishes anything by itself
(no accounts, no pushes: you do the last step on the host you choose).

| | What it is | Live solving | Needs a server | Good for |
|---|---|---|---|---|
| **A. Static demo** | The React page as plain files; every result is a saved one (`public/cache/bundle.json`) | No: presets at their starting angle and Mach, the PINN results, the charts | No (any static host) | A link to share, a portfolio, a backup if the laptop fails |
| **B. One container** | FastAPI serves the page and the API on one port (`deploy/Dockerfile`) | Yes: sliders, drawing, optimiser, NeuralFoil | Yes (any host that runs a container and gives HTTPS) | The real demo, reachable by anyone |
| **C. Laptop** | `uvicorn` plus the built page (README, `app/start.ps1`) | Yes | Your laptop | The viva; the offline fallback covers a crash |

## The licence line for anything public

NNfoil (Cao, Song and Zhang, arXiv 2401.08705) has **no licence**, so its files and data may not be redistributed. A public
deployment therefore runs with `AERONET_WITHHOLD_FVM_DATA=1`:

* **Not served:** the bundled FVM sample (`/api/reference/sample`, the "Load bundled FVM sample" button), every FVM Cp curve in the
  PINN showdown, and the overlays computed from them.
* **Still shown:** this project's own results (the panel method, PINN wall-Cp curves, seed bands, timings) and the *numbers* computed
  against the FVM data (errors, Cl). They are facts with their source labelled. Whether that is acceptable is a judgement for you
  and the NNfoil authors (this is not legal advice); asking them for permission to redistribute the FVM Cp files would lift the limit.
* **Credit stays on the page:** footer, showdown card and `docs/` name Cao, Song and Zhang.
* **A guard:** `npm run check:public` (run by the Docker build and by `build:public` / `build:static`) fails if the saved results
  contain the FVM sample, an FVM curve or an overlay, so a private bundle cannot be published by accident.

The bundle in the repository is a public-safe one. If you regenerate it from a normal backend it will contain FVM data and the
guard will refuse it: use `deploy/prepare-public.ps1` (below).

## Prepare (A and B)

```powershell
# from the repository root, after the PINN runs are collected into results/
.\deploy\prepare-public.ps1 -ResultsDir ..\Aero-Net\results
```

It (1) copies only what the app reads (`summary.csv`, `seed_check.csv`, and `config/log/snapshots.json` of the runs shown; no model
weights, no `NNfoil/`) into `deploy/data/results` (about 0.5 MB), (2) starts a backend that withholds the FVM data and uses only
that folder, (3) writes `app/frontend/public/cache/bundle.json` from it, and (4) runs the guard. Run it again whenever results change.

## A. Static demo

```powershell
cd app\frontend
npm ci
npm run build:static          # dist/ = plain files; the page never calls /api
```

Upload the contents of `dist/` to any static host (a file host, a CDN, a repository's static pages). Page addresses use `#hash`
links, so no server rewrite rules are needed. A banner says the results are saved and from when. Sliders, drawing, the optimiser and
imports show "Cannot reach the solver API" because there is no solver.

## B. One container

```powershell
docker build -f deploy/Dockerfile -t aeronet .
docker run --rm -p 8000:8000 aeronet            # http://localhost:8000
```

Push the image to a registry and run it on a container host (any of the usual ones: a small VM with Docker, or a managed
container service). What the host must do: set `PORT` if it assigns one (the image honours it), give HTTPS, and keep one instance
running. Look up each provider's current free-tier and cold-start behaviour yourself; the numbers change and are not stated here.

* The image has no NNfoil files and no FVM data; the results folder is the slim one.
* `--build-arg WITH_NEURALFOIL=0` leaves out NeuralFoil (a smaller image; the viscous card then says it is not installed).
* Starts as a non-root user, has a health check on `/api/health`, and trusts the platform's proxy headers for the client address.

### Settings (environment variables)

| Variable | Default in the image | Meaning |
|---|---|---|
| `AERONET_WITHHOLD_FVM_DATA` | `1` | Public mode (see the licence line). Unset or `0` for a private deployment with the FVM data |
| `AERONET_RESULTS_DIR` | `/app/data/results` | Where the PINN results are |
| `AERONET_MAX_HEAVY` | `2` | Optimiser and viscous requests that may run at once; more get 503 "busy" |
| `AERONET_HEAVY_PER_MIN` | `30` | Those requests per minute and client address; more get 429 |
| `AERONET_CORS_ORIGINS` | none | Extra origins that may call the API from another site (not needed for one server) |
| `PORT` | `8000` | Port to listen on |

The optimiser takes about a second of one CPU core, so the two limits are what protect a small server from a burst of clicks.

## Before going public

1. Decide about the NNfoil data (above). Keep the repository private until the authors answer.
2. `deploy/prepare-public.ps1`, then build, then open the result and click through every tab (the guard only checks the bundle).
3. Check the footer and the Results tab: scope (inviscid, subsonic, 2D), "not a certification tool", credits.
4. Nothing secret is in the image: no keys, no accounts. There is no login and no analytics; the API is read-only apart from
   computing answers. FastAPI's `/docs` page is on; remove it (`docs_url=None`) if you prefer.
5. Plan B and the other branches must be merged first; this work lives on `features-v3`.

## What was tested (2026-11-14, on the project laptop)

* **Container:** built from `deploy/Dockerfile` (with NeuralFoil; docker reported 284 MB), run on a local port. Health check "healthy"; about
  140 MiB of memory at rest and after a burst; runs as user `aeronet`; no `NNfoil` or `*_Cp.dat` file anywhere in the image.
* **Endpoints in the container:** `/api/capabilities` says public build; `/api/reference/sample` answers 404; `/api/showdown` has the
  error numbers and the PINN curves but no FVM curves; the trade-off chart, NeuralFoil check (Cl 0.426, Cd 0.0073 at Re 1 M) and the
  optimiser (critical Mach gain 0.22 in about 1.2 s) work; the page and `bundle.json` are served.
* **Limits:** five optimiser requests at once gave 200, 200, 503, 503, 503 (two slots, three told the server is busy); a bad input gives 422.
* **Static demo:** `npm run build:static` served by a plain file server: no request to `/api`, no console error, presets, showdown
  (without FVM curves, with the explanatory note) and the accuracy chart work; the FVM sample button is hidden.
* **Tests:** 122 backend tests and 60 frontend tests pass (new: public mode, limits, CORS, the bundle guard).

## Not done here

No hosting account, no domain, no push, no registry upload: those are your decisions and your credentials. No load test beyond the
five-request burst above. The container was built and run on this laptop only, with Docker Desktop (which was started for the test).
Not checked: a real container host (cold start, memory limits, HTTPS), Firefox and Safari, and how the public pages look to a first-time
visitor on a phone.
