# AeroNet: One-Week Execution Plan

> Physics-Informed Neural Network (PINN) solver for airfoil aerodynamics, built on NNfoil (Cao, Song & Zhang 2024).
> Plan written 2026-09-24 from: the AeroNet proposal, Sun et al. 2023 (CMAME) and Sharpe & Hansman 2025 (NeuralFoil).

**Goal:** In 7 days, produce a verified, reproducible NNfoil run, a quantitative validation against the bundled FVM reference data, 3-4 small but real experiments, an interactive demo, a report and a presentation.

**Architecture:** Fork NNfoil, fix and verify it (Days 1-2), train and validate (Days 2-4), wrap it in a Streamlit demo (Days 4-5), write it up (Days 5-7). Training runs go in the background while other work continues.

**Tech stack:** Python 3.10+, PyTorch, NumPy/SciPy/Matplotlib, Streamlit (demo), Google Colab or Kaggle GPU (training), NeuralFoil (`pip install neuralfoil`, comparison only), Git/GitHub, LaTeX or Word for the report.

---

## 0. What the three documents say (the research core)

| Doc | What it is | What we use it for |
|---|---|---|
| **Proposal** | AeroNet = verify, reproduce, validate and present NNfoil. Inviscid, subsonic, data-free PINN. | The project scope. Its 4 pillars (Verification, Reproducibility, Validation, Presentation) become the 4 workstreams. |
| **Paper 1: Sun, Sengupta, Juniper (CMAME 2023)** | A PINN that models flow and optimizes airfoil shape in one training run. Inputs are (x, y) plus design parameters (1 angle of attack, or 11 PARSEC shape parameters). Solves **Navier-Stokes at Re = 20**, checked against FEniCS. | The "contrast" paper. It shows the PINN-plus-optimization idea, adaptive sampling (add points where the residual is high) and honest cost numbers: **18 h** training for the 1-parameter case, **75 h** for 11 parameters on an RTX 3080. Its pointwise errors are about 1e-2 for velocity and 1e-1 for pressure, and it says the biggest error is at the leading edge. That is exactly the weakness NNfoil's mesh transformation targets. |
| **Paper 2: NeuralFoil (MIT, 2025)** | A hybrid ML surrogate for XFoil, trained on **7.9 million** XFoil runs. Gives **8x-1000x** speedups at equal accuracy (1.2 ms vs 73 ms per run; 100,000 runs in 0.87 s vs 42 min). Includes an "analysis confidence" output. | The "industrial comparison" paper. It is data-driven and viscous, while NNfoil is data-free and inviscid, so this contrast is the heart of the literature discussion. Its rule that **speed claims must control for accuracy** is the standard we hold our own speed comparison to. |
| **NNfoil repo (checked online)** | `cst_geom.py`, `mesh_gen.py`, `NNfoil_Euler2D.py`, `util.py`, `Airfoil2CST/`, `FVM_Cp/`, `FVM_Cp_NACA2412/`, `FVM_Cp_random_airfoil/`, `mesh/`, `mesh_generation/`. | The code base. **Not confirmed from the page:** dependencies, GPU needs, training time, license. Day 1 settles these. |

**One-sentence positioning for the report and viva:**
Sun et al. show a PINN can be simultaneously a surrogate and an optimizer but is slow to train and weak at the leading edge; NeuralFoil shows surrogates are practical but needs millions of labelled simulations; NNfoil/AeroNet sits in between: data-free like Sun, with a mesh transformation that fixes the leading-edge problem, and we verify, reproduce and validate it.

---

## 1. Ground rules and risks (read before starting)

1. **Do not claim the three "bugs" until you reproduce them.** The proposal lists a hard-coded GPU-only check, a wrong smoothing coefficient and a missing density positivity constraint. Treat these as *hypotheses*. On Day 1-2, find each in the code, write down the exact file and line, and show before/after behaviour. If one turns out not to be a bug, say so in the report. Honest verification is worth more marks than an inflated claim.
2. **The bundled FVM data is Cp values, not an FVM solver.** So "speed vs FVM" cannot be measured by running an FVM code. Plan: (a) accuracy is measured against the bundled FVM Cp; (b) speed compares PINN inference time (ms), PINN training time (amortised), and NeuralFoil/XFoil timing that we *can* run; (c) the FVM solve time is quoted from the NNfoil paper (Ref [1]) and labelled as quoted, not measured. Check on Day 1 what angle of attack the FVM files were computed at.
3. **Inviscid flow has ideal drag of about 0** (d'Alembert's paradox). Any non-zero drag from the PINN is numerical error, so report it as an *error indicator*, not as a real drag prediction. Do not compare it against real-world drag.
4. **Compute risk.** Sun et al. needed 18-75 h on a GPU. NNfoil may be cheaper, but we do not know. Start the first training run by end of Day 2 and treat GPU time as the critical path. Use Colab/Kaggle if your laptop has no GPU (this is why the "GPU-only check" fix matters).
5. **Scope discipline.** Everything is tiered: **Must / Should / Stretch**. Cut from the bottom, never from Must.
6. **Licence and attribution.** Check the NNfoil licence on Day 1. Keep the original authors credited in the README and report, and keep your changes in clearly marked commits.

**Assumptions I made (change if wrong):** a team of 2-4, about 6-8 productive hours per person per day, and that a free GPU (Colab/Kaggle) is available.

---

## 2. Deliverables checklist

- [ ] GitHub repo: fork of NNfoil with `requirements.txt`, setup instructions, fixes as separate commits, `README.md` with a one-command run
- [ ] `docs/verification.md`: equation-by-equation check table (see Day 2)
- [ ] `results/`: validation figures and CSV tables
- [ ] Streamlit demo (`app.py`) plus a 2-minute backup screen-recording
- [ ] Report, 15-20 pages (outline in section 5)
- [ ] Slides, 12-15 (outline in section 6)
- [ ] Viva Q&A sheet (section 7)

---

## 3. Team split (if 3 people; merge roles if fewer)

| Role | Owns |
|---|---|
| **A: Solver & Verification** | Repo, environment, bug hunt, verification table, training runs |
| **B: Validation & Experiments** | Loading FVM data, metrics, plots, ablations, NeuralFoil comparison |
| **C: Demo & Documentation** | Streamlit app, literature review, report, slides |

Everyone reads the two papers on Day 1 and joins the final run-through on Day 7.

---

## 4. Day-by-day plan

### Day 1: Environment, reading, and a first run
**Outcome:** NNfoil runs (even for a few epochs) on your machine or Colab, and everyone understands the method.

- [ ] Fork `github.com/Cao-WenBo/NNfoil`; clone; create a `venv` or conda env.
- [ ] Read `Readme.md`, then skim `NNfoil_Euler2D.py`, `mesh_gen.py`, `cst_geom.py`, `util.py`. Write a 1-page "how the code is organised" note.
- [ ] Discover and record: PyTorch version needed, GPU requirement, default epochs and network size, licence, and which AoA/Mach the `FVM_Cp*` files use. Freeze the working versions into `requirements.txt`.
- [ ] Try to run the main script for a few hundred iterations (a smoke test). Note every error. The GPU-only device check will probably show up here.
- [ ] Read the NNfoil paper (Ref [1], arXiv 2401.08705) sections on mesh transformation, loss terms and boundary conditions.
- [ ] Reading split: A = NNfoil paper; B = Sun et al. (focus on Sections 2-3: PINN setup, adaptive sampling, results vs FEniCS); C = NeuralFoil (Sections III-IV: architecture, accuracy vs speed). Each writes 5 bullet points for the others.
- **Checkpoint (end of day):** "Smoke test runs" or a written list of what blocks it.

### Day 2: Verification and bug fixes, then start the real training run
**Outcome:** Verification table done, three suspected bugs each confirmed or refuted, and one full training run running in the background.

- [ ] Build `docs/verification.md` with one row per item: *equation or component → where in the code → what the theory says → what the code does → verdict (OK / bug / unclear)*. Items to cover:
  - Euler residuals (continuity, x-momentum, y-momentum, energy) in computational space
  - Coordinate transform and Jacobian terms (metrics of the mesh mapping)
  - Boundary conditions: wall (slip), far field
  - Lift/drag integration from Cp
  - Mesh smoothing step (the "incorrect coefficient" claim)
  - Density/pressure positivity (the "missing constraint" claim)
- [ ] For each of the 3 claimed bugs: reproduce first (a minimal demo of the symptom), fix in its own commit, and record before/after in the table. Suggested fixes:
  - Device check: `device = "cuda" if torch.cuda.is_available() else "cpu"`
  - Positivity: wrap predicted density (and pressure if needed) with `softplus` or `exp`, or clamp with a small epsilon; verify it does not change good solutions
  - Smoothing coefficient: derive the correct value from the paper's formula and compare
- [ ] Sanity tests (small `pytest` file): the transform maps the airfoil surface to the boundary of the uniform grid; freestream state gives zero residual; symmetric airfoil at AoA = 0 gives Cl of about 0.
- [ ] **Launch the main training run** (NACA 0012, the default resolution, the paper's settings) on Colab/Kaggle with checkpoint saving every N epochs. Log loss to a CSV. Record wall-clock time per 1,000 iterations, since this drives the schedule.
- **Checkpoint:** training is running and saving checkpoints. If per-iteration time predicts more than about 12 h for the full run, reduce the mesh resolution or iterations *today* and note it as a deviation.

### Day 3: Validation against FVM
**Outcome:** The core quantitative results.

- [ ] Write `validate.py`: load PINN Cp on the airfoil surface, load the matching `FVM_Cp*` file, interpolate to common x/c, and compute L2 and max error of Cp, plus Cl from integrated Cp.
- [ ] **Must:** NACA 0012 vs FVM at one mesh resolution. Plot Cp vs x/c (PINN vs FVM), with an error subplot.
- [ ] **Must:** Cl and "drag" (error indicator) table. Sanity-check Cl against thin-airfoil theory (Cl about 2π·α for a symmetric airfoil at small α).
- [ ] **Should:** the same for NACA 2412 and at least one of the random airfoils.
- [ ] **Should:** mesh-resolution study: PINN trained on 2-3 mesh sizes (for example 50×25, 100×50, 200×100) vs the FVM data at the same sizes. This reproduces the paper's key claim (PINN wins on sparse meshes) on a small scale, which is the strongest research result you can get in a week.
- [ ] Start the next background training runs (see Day 4 experiments).
- **Checkpoint:** at least one figure comparing PINN vs FVM that you would be comfortable putting in the report.

### Day 4: Experiments (the "decent research" part) and comparison
**Outcome:** 3-4 small, well-controlled experiments. Pick from this list in order:

| # | Experiment | Why it matters | Tier |
|---|---|---|---|
| E1 | **Before vs after fixes**: run with and without the density positivity fix (and the corrected smoothing coefficient) | Shows the value of your verification work with numbers | Must |
| E2 | **Mesh resolution vs error** (Day 3 study, cleaned up) | Reproduces the paper's headline claim | Must |
| E3 | **Angle of attack sweep**: for a parametric model, Cl vs α against thin-airfoil theory, and Cp at several α vs FVM if available | Shows the "train once, query many" benefit (the Sun et al. idea) | Should |
| E4 | **Shape generalisation**: build 3-5 airfoils from the UIUC database (via `Airfoil2CST` / `cst_geom.py`) and compare Cl/Cp against NeuralFoil at high Re as a *rough* reference | Uses the dataset in the proposal; note clearly that NeuralFoil includes viscosity so agreement is expected only approximately (Cl, attached flow) | Should |
| E5 | **Network/sampling ablation**: width, depth or number of collocation points | Connects to Sun et al.'s adaptive sampling discussion | Stretch |

- [ ] **Speed/accuracy comparison table** (follow NeuralFoil's principle of controlling for accuracy):

| Method | Setup cost | Cost per new query | Physics | Data needed |
|---|---|---|---|---|
| FVM (bundled data; time quoted from Ref [1]) | none | full re-solve | Euler | none |
| NNfoil PINN (measured) | training time | inference ms | Euler | none |
| NeuralFoil (measured with `pip install neuralfoil`) | none for the user | about 1 ms | viscous, learned | 7.9M XFoil runs |
| Sun et al. PINN (from paper) | 18-75 h | fast | Navier-Stokes, Re = 20 | none |

- [ ] Measure inference time properly: warm-up runs, average of at least 100 calls, CPU and GPU stated.
- [ ] Keep a running `results/summary.csv` so the report tables are generated, not retyped.
- **Checkpoint:** E1, E2 done; E3/E4 in progress or done.

### Day 5: Interactive demo
**Outcome:** A working Streamlit app that can be shown live.

- [ ] `app.py` (Streamlit) with sidebar: airfoil dropdown (only shapes with a trained checkpoint), angle-of-attack slider (only over the range the model was trained on), and a "compare with FVM" toggle.
- [ ] Main panel: pressure-coefficient field/contour, Cp-vs-x/c curve with FVM overlay, live Cl, inference time in ms, and a small "PINN vs FVM vs NeuralFoil" bar for time.
- [ ] Load saved checkpoints; do **not** train inside the demo. Cache with `st.cache_resource`.
- [ ] Add a visible disclaimer: *inviscid, subsonic, screening tool, not a replacement for high-fidelity CFD*.
- [ ] Record a 2-minute screen recording as a fallback for the presentation day.
- [ ] Deploy locally; optionally to Streamlit Community Cloud if the checkpoint is small.
- **Checkpoint:** demo runs from a clean clone using only the README instructions (test on a second machine or fresh Colab).

### Day 6: Report writing
**Outcome:** Full first draft of the report with every figure and table in place.

- [ ] Write in this order: Results, Methodology, Verification, Literature review, Introduction, Conclusion, Abstract last.
- [ ] Generate all figures from scripts (`make_figures.py`) so they can be re-made in minutes.
- [ ] Literature review: use the 6 references from the proposal, plus the contrast framing in section 0 above. Do not add references you have not opened; every citation must be one someone on the team actually read.
- [ ] Include a "Limitations and honest scope" section (inviscid only, subsonic only, drag not meaningful, small experiment sizes, limited training budget).
- [ ] Include a "Future work" section: parametric shape input (Cao et al. Ref [2]), optimization on top of the PINN (Sun et al. Ref [3]), viscous extension, confidence estimates like NeuralFoil's analysis confidence.
- **Checkpoint:** a complete draft that a teammate has read end-to-end.

### Day 7: Slides, rehearsal and buffer
**Outcome:** Submission-ready package.

- [ ] Slides (outline below), 12-15 slides with figures from `results/`.
- [ ] Full rehearsal with a timer, including the live demo and the fallback video. Do it twice.
- [ ] Viva drill: each person answers the questions in section 7 for the parts they did not own.
- [ ] Final repo cleanup: README quick start, `requirements.txt`, tagged release `v1.0`, no large files (host checkpoints via release assets or Drive).
- [ ] Buffer: use spare time for E5 or for fixing the weakest figure.
- **Checkpoint:** everything committed, tagged and backed up.

---

## 5. Report outline (15-20 pages)

1. **Abstract** (150-200 words)
2. **Introduction**: cost of CFD for design exploration; PINN idea; contribution list
3. **Literature review**: PINNs (Raissi et al. 2019, Ref [5]); PINN airfoil work (Ref [1], [2], [3]); industrial surrogate (Ref [4]); the comparison table (data-free vs data-driven, inviscid vs viscous)
4. **Background**: Euler equations, Cp and Cl from surface pressure, CST parameterisation, mesh transformation and the Jacobian
5. **Method**: NNfoil architecture, loss, sampling, training setup; AeroNet additions (env, fixes, validation pipeline, demo)
6. **Verification**: the table from Day 2, including the three claimed bugs, each with a before/after
7. **Results**: E1-E4, timing table
8. **Discussion**: what the results mean, comparison to Sun et al. and NeuralFoil, limitations
9. **Conclusion and future work**
10. **References** (IEEE style), **Appendix** (setup instructions, hyperparameters, full tables)

## 6. Slide outline (12-15 slides)

1. Title and team · 2. Problem: CFD is slow for design exploration · 3. What is a PINN (one picture) · 4. Why airfoils are hard for PINNs (leading edge) · 5. NNfoil's mesh transformation (before/after picture) · 6. Related work: Sun et al. vs NeuralFoil vs NNfoil (comparison table) · 7. What AeroNet adds (4 pillars) · 8. Verification and bugs found · 9. Validation vs FVM (Cp plot) · 10. Mesh-resolution result · 11. Angle-of-attack / shape results · 12. Speed and accuracy comparison · 13. **Live demo** · 14. Limitations and future work · 15. Conclusion / Q&A

## 7. Viva question bank (prepare answers)

1. Why does a PINN need no labelled data? *(The loss is the equation residual at sampled points plus boundary conditions.)*
2. Why is the leading edge hard, and how does the mesh transformation help? *(Sharp gradients in physical space; a uniform computational grid smooths them, Ref [1]. Sun et al. also report their maximum error there.)*
3. Why is drag about zero in your results? *(d'Alembert's paradox for inviscid flow; any value is numerical error.)*
4. How is this different from NeuralFoil? *(NeuralFoil is data-driven, viscous, needs 7.9M XFoil runs, and is much faster per query; NNfoil is data-free and inviscid but needs training per setup.)*
5. How is this different from Sun et al.? *(Navier-Stokes at low Re with shape parameters as inputs and built-in optimization vs Euler with mesh transformation.)*
6. Is your speed comparison fair? *(State exactly what was measured and what was quoted; accuracy is controlled by comparing at matched error, the way NeuralFoil does.)*
7. What bugs did you find and how do you know they were bugs? *(Point to the before/after evidence.)*
8. Where does this fail? *(Viscous effects, stall, transonic shocks, shapes far from anything tested.)*
9. What would you do next? *(Ref [2] parametric shapes, Ref [3] optimization, confidence estimates.)*

---

## 8. Fallbacks if things slip

| Problem | Fallback |
|---|---|
| Training is too slow | Smaller mesh/fewer iterations; state it clearly; lean on E1/E2 at small scale |
| A "bug" is not real | Report it as verified-correct, with the evidence |
| Colab session limits | Checkpoint every N epochs and resume; run on Kaggle in parallel |
| Results worse than the paper | Report them honestly with a hypothesis (fewer iterations, different seed); include the loss curve |
| NeuralFoil comparison too rough | Reduce E4 to a table of Cl at 2-3 angles and label it as indicative |
| Demo breaks on presentation day | Play the recorded video |

## 9. Daily 10-minute standup questions

1. What finished yesterday? 2. What is the one thing that blocks today? 3. Is any run in the background, and how is it doing? 4. Do we need to cut anything from Should/Stretch?
