# Deploying AeroNet

Three ways, from least to most to run. All of them are built from this repository; none of them publishes anything by itself (no
accounts, no pushes, no uploads: you do the last step on the host you choose).

| | What it is | Live solving | Needs a server | Good for |
|---|---|---|---|---|
| **A. Static demo** | The React page as plain files; every result is a saved one (`public/cache/bundle.json`) | No: presets at their starting angle and Mach, the PINN results, the charts | No (any static file host) | A link to share, a portfolio, a backup if the laptop fails |
| **B. One container** | FastAPI serves the page and the API on one port (`deploy/Dockerfile`) | Yes: sliders, drawing, optimiser, NeuralFoil (if installed in the image) | Yes (a host that runs a container and gives HTTPS) | The real demo, reachable by anyone |
| **C. Laptop** | `uvicorn` plus the built page (`app/README.md`, `app/start.ps1`) | Yes | Your laptop | The viva; the offline fallback covers a crash |

## Licence: what is allowed, and what to keep

* **NNfoil** (Cao, Song and Zhang, arXiv 2401.08705, github.com/Cao-WenBo/NNfoil): the GitHub repository has **no licence file**
  (checked through the GitHub API on 2026-11-14: `license` is `null`). Without a licence nothing may be redistributed by default, so this
  project relies on **the author's permission, given by email on [FILL IN: date]** (the reply was "no worries"). **Keep that email**:
  it is the only document that says you may show the data.
* **What that permission covers, as we read it:** using and showing NNfoil's reference data (the finite-volume Cp curves) in this project and
  its demo. The default builds therefore **include** that data (the `reference` and `fvm_same_mesh` curves of the exported PINN cases, the
  bundled `*_Cp.dat` sample, the showdown and overlay curves), and the pages that show it carry the line "Reference data and NNfoil by Cao,
  Song and Zhang (arXiv 2401.08705), shown with the author's permission." The footer, the showdown card and `docs/` credit them as well.
* **What it does not obviously cover:** republishing NNfoil's *source code*, or its data files as files, in a public repository of your own.
  Permission to show reference data in a demo is not the same thing. Before you make this repository (or any repository that contains
  `NNfoil/` or `app/backend/data/*.dat`) public, ask the author to confirm that too, in writing, and keep the answer next to the first email.
  Until then keep the repository private; a deployed demo does not publish the repository.
* **Strict mode (optional).** `AERONET_WITHHOLD_FVM_DATA=1` makes the server serve none of the FVM curves, and the strict builds below contain
  none either. Use it if the permission is ever withdrawn, or for a deployment that must be stricter. The numbers computed against the
  FVM data (errors, Cl, same-mesh errors, the Cp RMS of every training round) and this project's own results (PINN wall Cp, panel method,
  timings, seeds) stay, labelled with their source, and the page says "FVM reference curves withheld: NNfoil has no licence for
  redistribution; error numbers computed against them are shown." The PINN Lab, the Results page and the speed card all still work.
* This is not legal advice.

### What strict mode does and removes

| | default (include) | strict (`AERONET_WITHHOLD_FVM_DATA=1`) |
|---|---|---|
| `GET /api/reference/sample` | the NNfoil FVM sample | 404 |
| `GET /api/pinn/cases/<id>` | full case with `reference` (FVM 800x400) and `fvm_same_mesh` points | no `reference`, no points; `fvm_ref_name`, `fvm_same_errors` (name, Cp RMS, max, Cl), `rms_by_round`, `fvm_curves_withheld: true` |
| `GET /api/showdown` | FVM curves and PINN curves | PINN curves and all error numbers; `fvm_curves_withheld: true` |
| `GET /api/capabilities` | `fvm_sample: true`, `fvm_curves: true`, `public_build: false` | `fvm_sample: false`, `fvm_curves: false`, `public_build: true` |
| PINN Lab | FVM overlay toggles, same-mesh curve | toggles hidden, note; PINN vs panel wall Cp, replay, loss history, flow field unchanged |
| "Load bundled FVM sample" button | shown | hidden, with a note |

In strict mode the **files** carry no FVM curve either, because stripping at response time alone would leave the data inside an image:

* `python deploy/make_data.py` writes a stripped copy of `app/backend/data` to `deploy/data/public` (git-ignored, generated): every case goes through
  `aeronet.pinn_cases.strip_fvm`, the `*.dat` sample is not copied, and the result is verified (`--check <folder>` exits 1 if any `*.dat`, any `reference` /
  `fvm_same_mesh` points or any other `points` curve is found). The committed folder fails that check on purpose (it holds the FVM data).
* The strict image (`--build-arg WITHHOLD_FVM=1`) is built from such a copy made inside the build, verified there, and only that copy is copied into the image.
* The strict static bundle is made by a backend that withholds the data; `npm run check:public:strict` fails if the bundle contains any.

## Preparing the saved results (A and B)

The committed `app/frontend/public/cache/bundle.json` is an **include-mode** bundle (it carries the reference data), made from the current backend.
You only regenerate it when the results or the app's answers change:

```powershell
.\deploy\prepare-public.ps1             # include mode: starts a backend on a spare port, writes the bundle, checks it, stops the backend
.\deploy\prepare-public.ps1 -Withhold   # strict mode: stripped data copy + withholding backend -> bundle without FVM data
```

`-Withhold` overwrites `public/cache/bundle.json` with the strict one: build, then restore the committed one (`git checkout app/frontend/public/cache/bundle.json`).
`npm run check:public` (include) and `npm run check:public:strict` (or `AERONET_WITHHOLD_FVM_DATA=1 npm run check:public`) verify that the bundle
matches the mode: include mode refuses a bundle without the reference data, strict mode refuses one with any.

## A. Static demo

```powershell
cd app\frontend
npm ci
npm run build:static            # include mode (default). dist/ = plain files; the page never calls /api
npm run build:static:strict     # strict mode (after prepare-public.ps1 -Withhold)
```

Upload the contents of `dist/` to any static host. Page addresses use `#hash` links, so no server rewrite rules are needed. A banner says the
results are saved, and from when. Sliders, drawing, the optimiser and imports say that they need the live app, because there is no solver.
Preview locally with `npm run preview` (stop it afterwards).

## B. One container

```powershell
docker build -f deploy/Dockerfile -t aeronet .                                  # default: reference data included
docker run --rm -p 8000:8000 aeronet                                            # http://localhost:8000

docker build -f deploy/Dockerfile --build-arg WITHHOLD_FVM=1 -t aeronet-strict .   # strict (needs the strict bundle, see above)
```

* `--build-arg WITH_NEURALFOIL=0` leaves out NeuralFoil (a smaller image; the viscous card then says it is not installed).
* Starts as a non-root user, has a health check on `/api/health`, honours `PORT`, and trusts the platform's proxy headers for the client address.
* The build fails if the saved results do not match the mode (the same check as `npm run check:public`).

### Settings (environment variables)

| Variable | Default in the image | Meaning |
|---|---|---|
| `AERONET_WITHHOLD_FVM_DATA` | `0` (`1` in a strict image) | Strict mode (see above) |
| `AERONET_DATA_DIR` | none (the image's `/app/backend/data`) | Another folder of exported data (read once at start); the strict local run uses it for the stripped copy |
| `AERONET_MAX_HEAVY` | `2` | Optimiser and viscous requests that may run at once; more get 503 "busy" |
| `AERONET_HEAVY_PER_MIN` | `30` | Those requests per minute and client address; more get 429 |
| `AERONET_CORS_ORIGINS` | none | Extra origins that may call the API from another site (not needed when one server serves page and API) |
| `PORT` | `8000` | Port to listen on |

The optimiser takes about a second of one CPU core, so the two limits protect a small server from a burst of clicks. They live in the process, so
several workers each have their own.

## Hosting options (types, not recommendations)

Look up each provider's current terms, limits, cold-start behaviour and pricing yourself; none is stated here and none was tried.

* **A (static demo):** a static file host or object-storage bucket with a CDN, or the static pages feature of a code host. Nothing runs, nothing to patch.
* **B (container):** a managed container service, a platform that builds and runs a Dockerfile, or your own small virtual machine with Docker and a reverse
  proxy for HTTPS. Needs: HTTPS, one always-on instance (the limits are per process), the `PORT` variable if the platform assigns one.
* **C (laptop):** a tunnel to your own machine is possible but ties the demo to the laptop being on; the static demo is the safer public link.

## Before going public

1. Keep the author's email and fill in its date above. Decide about publishing NNfoil's source or data files in a repository (see "Licence").
2. Run `npm run check:public` (or the strict one), build, open the result and click through every tab (the guard only checks the saved results).
3. Check the footer and the Results tab: scope (inviscid, subsonic, 2D), "not a certification tool", the credit to Cao, Song and Zhang.
4. Nothing secret is in the image: no keys, no accounts. There is no login and no analytics; the API is read-only apart from computing answers.
   FastAPI's `/docs` page is on; remove it (`docs_url=None`) if you prefer.

## What was tested (2026-11-14, on the project laptop)

* **Both images built** from `deploy/Dockerfile` (default, and `--build-arg WITHHOLD_FVM=1`), run on local ports, health check "healthy" for both, about 140 MiB of memory at
  rest each, runs as user `aeronet`. `docker images` reports about 1.2 GB for each (with NeuralFoil, which was present and answered `/api/viscous`).
* **Default image:** `/api/capabilities` says `fvm_sample: true, fvm_curves: true`; `/api/reference/sample` answers 200; the case endpoint serves `reference` (800 points)
  and `fvm_same_mesh` (100 points); the showdown has the FVM curves; the page and `bundle.json` are served; the data folder holds the `*.dat` sample.
* **Strict image:** inside it no `*.dat` / `*_Cp*` file other than the operating system's own, no `NNfoil` folder, no case file with a `reference` or `fvm_same_mesh` curve,
  and none in the served bundle (`/app/backend/data` has `MANIFEST.json` and the stripped cases). `/api/capabilities` says `fvm_sample: false, fvm_curves: false, public_build: true`;
  `/api/reference/sample` answers 404; the case endpoint has no `reference` and no `points`, only `fvm_ref_name`, `fvm_same_errors` (Cp RMS 0.0460, max 0.261, Cl 0.480 for
  100x50) and `rms_by_round`; the showdown has the error numbers and the PINN curves but no FVM curve. A script that fetched every case, the showdown, results, trade-off and
  capabilities answers and searched the raw text for each of the 5,196 distinct FVM `[x, y, Cp]` triples of the committed data found none (the same script finds 12,022 in the default mode).
* **Local strict server** (stripped copy, `AERONET_DATA_DIR`, flag on): the same results; the PINN Lab (flow field, PINN vs panel wall Cp, replay, loss history, showdown) and
  the Results page rendered with the withheld note and no FVM overlay (screenshots `docs/milestones/img/public_pinn_lab_strict.png` and `public_results_strict.png`); the default
  mode shows the overlay toggles and the credit line (`public_pinn_lab_default_fvm.png`).
* **Static demo:** `npm run build:static` (include) passes its check and `dist/cache/bundle.json` holds the sample and the 800-point reference; `npm run build:static:strict` with
  a strict bundle passes `check:public:strict` and its `dist/` bundle holds no `reference` or `fvm_same_mesh` curve and no sample.
* **Tests:** backend 171 passed, 2 skipped; frontend 126 (including the bundle check in both modes and the PINN Lab helpers for a case without FVM points); research 228; 0 broken links.

## Not done here

No hosting account, no domain, no push, no registry upload: those are your decisions and your credentials. No load test. Not checked: a real container
host (cold start, memory limits, HTTPS), Firefox and Safari, how the public pages look on a phone, and NeuralFoil inside the new image (the laptop's Python
has no NeuralFoil installed; the earlier image test of the old layout did include it). Nobody has asked the NNfoil author about republishing source code.
