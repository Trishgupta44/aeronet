# AeroNet: 7-Day Plan (simple version)

This replaces the earlier long plan. It is written for a team with **no GPU** and a **live demo** as the submission.
Everything marked *(checked)* was verified by reading the NNfoil code or running something. Everything marked *(unknown)* must be measured on Day 1.

---

## 1. The project in three sentences

1. A normal CFD solver has to redo all its work for every new airfoil or angle. A PINN (a neural network trained to obey the flow equations) learns once and then answers instantly.
2. NNfoil is an open-source PINN for airfoil flow (inviscid, subsonic). We **check it, fix it, run it, test it against reference data, and present it in a live demo.**
3. We are honest about scope: it is a fast *screening* tool, not a replacement for full CFD.

**How the two papers fit in:**
- *Sun et al. 2023* shows a PINN can also optimize a shape, but takes 18 to 75 hours on a GPU, and its biggest errors are at the leading edge.
- *NeuralFoil 2025* is fast (about 1.2 ms per run vs 73 ms for XFoil) but needs 7.9 million XFoil runs to train, and it handles viscous flow.
- *NNfoil (our base)* needs no training data and fixes the leading-edge problem with a mesh transformation.

---

## 2. What I found in the real NNfoil code (checked)

The repo has one main file, `NNfoil_Euler2D.py` (335 lines), plus `util.py`. No `requirements.txt`, and **no LICENSE file**.

| # | Finding | Where | What we do |
|---|---|---|---|
| 1 | **GPU-only.** `device = torch.device("cuda:0" if 1 else "cpu")` always picks the GPU. It will crash on your laptops. | main, line 269 | Use `torch.cuda.is_available()`. **Confirmed bug.** |
| 2 | **No positivity for density and pressure.** The network's output is used directly, then divided by (`1/rho`, `p/rho`). A zero or negative value breaks the physics. | `Msef`, lines 193 to 197 | Force `rho` and `p` to stay positive (softplus). **Confirmed missing.** |
| 3 | **Mesh-smoothing coefficient.** The proposal says one is wrong. Not confirmed. The smoothing code is in `mesh_generation/mesh_gen.py` lines 61 to 69, and it only matters when *making new meshes*. The ready-made meshes are not affected. | mesh_gen.py | Compare it with the standard elliptic grid equation. Report "bug" or "checked, correct". **Unconfirmed.** |
| 4 | Only **1 CPU thread** is used: `torch.set_num_threads(1)`. Your laptops have 4. | main, line 265 | Set 4. |
| 5 | `np.math.factorial` was removed in NumPy 2. It will crash. | line 110 | Use `math.factorial`. |
| 6 | `torch.load('model_temp.pth')` fails on new PyTorch (default changed) and the file name is shared between runs. | lines 247, 250 | `weights_only=False`, `map_location`, one file name per run. |
| 7 | **The main script never computes lift.** `Calculate_Force()` exists but is never called. | lines 207 to 210 | Call it after training and print Cl, Cd. |
| 8 | **One flow condition per run.** Angle 4° and Mach 0.4 are hard-coded, and the network only takes (ξ, η). The "continuous angle" idea in the paper is *not* in this file. | lines 55, 267 | Plan one trained model per case. No free slider for the PINN. |
| 9 | Default training is heavy: 5 layers of 128, L-BFGS up to 1000 iterations, 20 rounds, 16,000 points on the 200×100 mesh. | lines 235 to 237, 267, 281 | Use small meshes (see section 4). |

**Reference data (checked by running comparisons):**
- NACA0012 files match **4° angle, Mach 0.4** (my stand-in solver fits them with RMS Cp error 0.009 on the 800×400 file).
- `NACA2412_800_400_AoA{-10…10}` files exist for every 2°. The AoA0 file matched 0°.
- Cylinder files are symmetric, as expected.
- Meshes available: NACA0012 (50×25 up to 800×400), NACA2412 (200×100 up to 800×400), Airfoil1 to 5 (200×100).

**Your machines *(checked on this laptop)*:** 4 CPU threads, only Python 3.14 installed, PyTorch not installed. Install **Python 3.12** and use CPU-only PyTorch (PyTorch support for 3.14 may be shaky).

---

## 3. The rules of this plan

1. **Test on tiny meshes first.** CPU training time is *(unknown)*, so Day 1 measures it before we promise anything.
2. **Three levels for everything:** *Must* (do it), *Should* (do it if on time), *Stretch* (only if free).
3. **Checkpoints are pass/fail.** If a checkpoint fails, use its fallback and move on. Never spend two days on one problem.
4. **No fake numbers.** Anything not measured is labelled "quoted from paper" or "not measured".
5. **Credit the authors** (Cao, Song, Zhang). The repo has no license, so keep our fork **private** and ask the authors before publishing it.

---

## 4. Working on a CPU-only laptop

- **Fast meshes for testing:** NACA0012 at **50×25** (1,250 points) and **100×50** (5,000 points).
- **Main result:** 100×50 for NACA0012 at 4°. Try 200×100 *overnight* if 100×50 is fast enough.
- **Smaller network if needed:** 4 layers × 64 instead of 5 × 128, and say so in the report.
- **Run in the background** while doing other work. Save a checkpoint after every round.
- **Free GPU backup:** Google Colab or Kaggle can run the same script and you download the checkpoint. Use only if the CPU is too slow.
- **Budget rule:** if one run is predicted to take over 10 hours, shrink it *that day*.

---

## 5. Day-by-day plan

### Day 1: Set up and measure
**Goal:** NNfoil runs on a laptop, and we know how long training takes.
- [ ] Install Python 3.12. Create a virtual environment. Install CPU PyTorch, NumPy, SciPy, Matplotlib. Freeze `requirements.txt`.
- [ ] Fork the repo (private). Read the 9 findings above and find each one in the code yourself.
- [ ] Apply findings 1, 4, 5, 6 (small fixes) so the script runs.
- [ ] Run the smoke test on **NACA0012 50×25**, `train(1)`. Note the time per round.
- [ ] Everyone reads their paper and writes 5 bullet points: A = NNfoil paper (arXiv 2401.08705), B = Sun et al., C = NeuralFoil.
- [ ] Open the prototype (section 8) and agree on what the final demo should show.

**✅ Checkpoint 1:** the script prints a loss on 50×25 **and** you have written down "seconds per round" for 50×25.
**If it fails:** run on Colab for now, fix the local error tomorrow.

### Day 2: Fix, verify, start the real training run
**Goal:** the code is fixed and documented, and the main run is training.
- [ ] Apply finding 2 (positivity) and finding 7 (print Cl and Cd). Commit each fix separately with a clear message.
- [ ] Look at finding 3 in `mesh_gen.py`; decide "bug" or "correct" and write down why.
- [ ] Write the **verification table** (`docs/verification.md`): one row per part (Euler equations, mesh transformation and Jacobian, boundary conditions, force integration). Columns: what the theory says, what the code does, verdict.
- [ ] **Start the main run:** NACA0012, 100×50, 4°, Mach 0.4, saving checkpoints. Let it run all night.
- [ ] Small tests: freestream gives near-zero residual; the surface mapping is correct.

**✅ Checkpoint 2:** the verification table is done, three fixes are committed, and the main run is training.
**If the run is too slow:** switch to 50×25 for the main result and use 100×50 as a bonus.

### Day 3: Validate against the reference data
**Goal:** the first real result: PINN vs FVM.
- [ ] Write `validate.py`: load the reference `*_Cp.dat` (columns node, x, y, Cp; x runs from −0.5 to 0.5, so add 0.5), compare with the PINN surface Cp, print RMS and max error and Cl. (The prototype already has this logic in `solver.js`: `parseReference` and `compareCp`.)
- [ ] Plot Cp vs x/c: PINN, FVM, error.
- [ ] Sanity check Cl: roughly **0.5** for NACA0012 at 4°, Mach 0.4. Drag should be near 0, and any non-zero drag is numerical error.
- [ ] Start the next run in the background (50×25 and 200×100 for the mesh study, or NACA2412 at 4°).

**✅ Checkpoint 3:** one figure comparing PINN and FVM that you are happy to put in the report, with an RMS error number.
**If the PINN result is poor:** show the loss curve and report honestly. Try more rounds or a different random seed before changing anything else.

### Day 4: Experiments and comparison
**Goal:** 3 small, fair experiments. Pick in this order.
- [ ] **E1 (Must): before vs after the positivity fix.** Same run with and without it. Compare error and stability.
- [ ] **E2 (Must): mesh size vs error.** 50×25, 100×50 (and 200×100 if time) against the FVM files of the same size. The paper claims the PINN wins on sparse meshes; we check it on a small scale.
- [ ] **E3 (Should): a second airfoil.** NACA2412 at 4° against `NACA2412_800_400_AoA4_Cp.dat`.
- [ ] **E4 (Should): speed table.** Measure PINN **inference time** (average of 100 calls, CPU) and training time. Install NeuralFoil (`pip install neuralfoil`, it only needs NumPy) and time it for the same airfoil. Quote FVM time from the NNfoil paper and label it "quoted".
- [ ] **E5 (Stretch): more shapes** from the UIUC database via `Airfoil2CST`.
- [ ] Save every result to `results/summary.csv`.

**✅ Checkpoint 4:** E1 and E2 done, results saved as CSV and figures.
**If time is short:** cut E3, E4 and E5 to one line each in "future work".

### Day 5: Build the live demo
**Goal:** the prototype page shows real PINN results.
- [ ] Write `export_results.py`: for each trained case, save one JSON file with surface Cp, Cl, Cd, inference time, and the field values on the mesh.
- [ ] Add a "NNfoil PINN" mode to the prototype: a dropdown of **trained cases only**, with the FVM reference overlaid and the error shown.
- [ ] Keep the panel-method mode as the "instant what-if" explorer, clearly labelled as a classical method.
- [ ] Fill the speed table with the **measured** numbers. Fill the training-history panel from the training log.
- [ ] Test on a **second laptop with no internet** using only the README steps.

**✅ Checkpoint 5:** the demo runs offline from a clean copy in under 2 minutes of setup.
**If the field plot is hard:** show only the surface Cp curve and the numbers. That is enough.

### Day 6: Write the report
**Goal:** a complete draft.
- [ ] Write in this order: Results → Method → Verification → Literature review → Introduction → Conclusion → Abstract.
- [ ] Generate all figures from scripts so they can be redrawn in minutes.
- [ ] Include: limits (inviscid only, subsonic only, one flow condition per model, small CPU budget) and future work (parametric shapes, shape optimization, confidence scores like NeuralFoil's).
- [ ] Only cite papers a team member has actually opened.

**✅ Checkpoint 6:** a teammate has read the whole draft and marked what is unclear.

### Day 7: Rehearse and buffer
**Goal:** ready to present.
- [ ] Slides (12 to 15) and a **5-minute live demo script** (section 7).
- [ ] Rehearse twice with a timer. Everyone answers the viva questions for parts they did not build.
- [ ] Record a 2-minute backup video of the demo.
- [ ] Clean the repo, tag `v1.0`, back it up in two places.

**✅ Checkpoint 7:** demo, report, slides, backup video and repo are all saved and tested.

---

## 6. If something goes wrong

| Problem | What to do |
|---|---|
| CPU training is too slow | Smaller mesh and network; use Colab for one run; say so in the report |
| PINN error is bigger than the paper's | Report it honestly with the loss curve and a guess at why (fewer iterations, small mesh) |
| A "bug" turns out to be correct | Report "checked and correct" with evidence, which still counts as verification |
| Field plot from the PINN is hard | Show only the surface Cp curve and the numbers |
| Demo laptop fails | Play the backup video and use a second laptop |
| One person is stuck | Ask for help after 1 hour, not at the end of the day |

---

## 7. Live demo script (about 5 minutes)

1. **(30 s) Problem:** "CFD re-solves everything for each new shape. A PINN learns once."
2. **(30 s) What we did:** verified NNfoil, fixed the issues, ran it on CPU.
3. **(60 s) Demo, panel mode:** change airfoil and angle live; the pressure field and Cl update instantly. Point out that this is the classical stand-in.
4. **(90 s) Demo, PINN mode:** choose the trained case, overlay the FVM reference, read out the RMS error and Cl.
5. **(45 s) Verification:** show the table and the before/after result for the positivity fix.
6. **(45 s) Speed and honesty:** show the comparison table and say clearly what is measured and what is quoted.
7. **(30 s) Limits and next steps.**

Likely viva questions: why PINNs need no data; why the leading edge is hard and how the mesh transformation helps; why drag is about zero (d'Alembert's paradox); how NNfoil differs from Sun et al. and from NeuralFoil; which of our speed numbers are measured; what bugs we found and how we know; where the method fails (viscous effects, stall, shocks).

---

## 8. The prototype (already built)

Folder: `prototype/` (open with `python -m http.server 8765 --directory prototype`, then visit `http://localhost:8765`).

| What you see | Status |
|---|---|
| Airfoil presets, custom NACA 4-digit input with validation | Works |
| Angle-of-attack slider, Mach slider | Works |
| Pressure field (Cp) around the airfoil, with colour bar | Works (panel method, not PINN) |
| Cl, thin-airfoil Cl, drag error, solve time | Works. Tests pass (`node prototype/test_solver.js`) |
| Cp curve with FVM reference overlay and RMS error | Works with the real bundled NNfoil data |
| Speed and accuracy table | Live row for this page; other rows show "measure on Day N" |
| Training-history panel | Placeholder, shows nothing until a PINN run is logged |
| "NNfoil PINN" solver button | Disabled until Day 5 |

Checks the solver passes: Cl = 0 at 0° for a symmetric airfoil, antisymmetry in angle, Kutta condition, stagnation Cp near 1, drag error near 0, and **RMS Cp error 0.009 against the bundled NACA0012 FVM data** at 4°, Mach 0.4.

**Known limit:** inviscid theory has no stall, so lift keeps rising at large angles. The demo should stay at moderate angles.

---

## 9. Handing tasks to Claude Code

Give it one day at a time, with the finding table from section 2. Example for Day 2: *"In NNfoil_Euler2D.py apply findings 2 and 7. Keep each change as its own commit. Add a test that checks softplus keeps rho and p above 0.001. Do not change anything else."* Ask it to show you the diff before committing, and always run the checkpoint yourself.
