# AeroNet: Progress Report
Physics-informed neural network solver for airfoil aerodynamics (built on NNfoil by Cao, Song and Zhang, arXiv 2401.08705)
Students: [FILL IN: student names and enrolment numbers]
Supervisor: [FILL IN: supervisor name]
Programme and semester: [FILL IN: programme, semester and year]
Reporting period and date: [FILL IN: reporting period and date of submission]

## 1. Summary

The work is organised in three milestones, each documented as a self-contained pack in `docs/milestones/`. Dates below are the dates of the commits in the project repository. The proposal and the one-week plan date from 24 and 25 September 2026; the milestones were carried out between 5 and 14 November 2026 (the three overlap, because experiments ran in the background while the application was built). Every number is from a file in `results/` and is labelled measured (run on this laptop), quoted (copied from the paper) or fitted (tuned to data).

| Milestone | Dates | Outcome |
|---|---|---|
| M1: NNfoil verified and running on a CPU | 5 to 6 November | 11 findings; training on a laptop CPU; first validation and seed check |
| M2: Demo v2 (draw any shape, live flow) | 7 to 8 November | Web app with a panel-method solver, drawn shapes, cars on a road, polar, `?present` mode |
| M3: PINN results and PINN Lab | 6 to 14 November | Experiments E1 to E6; PINN Lab with 14 trained cases; results page |

## 2. Milestone 1: NNfoil verified and running on a CPU (5 to 6 November)

**What was done.** The upstream code was vendored unmodified, then checked line by line against the Euler equations and the mesh theory (`docs/verification.md`). It was made to run on a CPU, given a command-line runner that writes one folder per run, and measured against the bundled FVM data.

**What was measured.** Eleven findings, with a test or a CSV for each: 7 fixed in code (forced CUDA device, one CPU thread, forces never computed, flow condition hard-coded, two crashes on current NumPy and PyTorch, and a positivity safeguard for pressure and density) and 4 documented. The mesh-smoothing coefficient is confirmed 4 times too small, with low impact: the Winslow residual is {{csv:finding3_mesh_smoothing.csv|variant=upstream|max_abs_winslow_residual:e2}} with the code's value against {{csv:finding3_mesh_smoothing.csv|variant=winslow|max_abs_winslow_residual:e2}} with the correct one, and both grids are valid. Training on a CPU takes {{E2b|pinn|50x25|train_s:0}} (50x25) and {{E2b|pinn|100x50|train_s:0}} (100x50). At 100x50 the PINN has Cp RMS {{E2|pinn|100x50|rms_cp}} against {{E2|fvm|100x50|rms_cp}} for the FVM on the same mesh. Grid sequencing, our addition, trains on 50x25 and continues on 100x50.

**What failed or was weaker than hoped.** The positivity fix did not help accuracy. Training at 50x25 turned out to be less stable than expected (see M3). A cold start on 100x50 without positivity collapsed in {{count:seed_check.csv|mode=nopos|method=cold|collapsed=yes}} of {{count:seed_check.csv|mode=nopos|method=cold}} seeds, while grid sequencing collapsed in {{count:seed_check.csv|method=gs|collapsed=yes}} of {{count:seed_check.csv|method=gs}} runs.

**Next at that point.** Decide the positivity default by experiment (E1), and build the demonstration.

## 3. Milestone 2: Demo v2 (7 to 8 November)

**What was done.** A web application: a FastAPI backend and a React front end in an aerospace-corporate style (`docs/DESIGN_BRIEF.md`; no third-party logo, text or fonts copied). The live solver is a Hess-Smith panel method with the Prandtl-Glauert correction. The user can drag the angle of attack, draw any closed outline, load a shape library (circle, teardrop, bird wing, sedan and sports car with a road under them by the method of images), read a lift polar, pin and compare, export PNG or CSV, and run a self-cycling `?present` mode. Alongside it (8 November) came an offline fallback (saved results when the backend is down), a critical-Mach warning, a same-mesh showdown card, and an optional NeuralFoil check.

**What was measured.** The panel method gives Cp max 0.9989 and Cp min -2.9989 for a circle (exact 1 and -3) and Cl 0.5247 for a NACA 0012 outline given in reverse order against 0.5263 from the code path. A solve takes 6 to 18 ms per preset (`docs/milestones/M2-demo-v2.md`). The panel method agrees with the 800x400 FVM reference to Cp RMS {{E2|panel|140 panels|rms_cp}}.

**What failed or is honest-limited.** The flow is inviscid, so the app never shows a car's drag, and the rear pressure of a car is not realistic (amber note on screen). Freehand outlines are not stored in the URL.

**Next.** Show the trained network next to the solver in the same page.

## 4. Milestone 3: PINN results and PINN Lab (6 to 14 November)

**What was done.** Six experiments, run on a laptop CPU one at a time, then the PINN Lab (14 exported cases, a training replay, a numbers dashboard) and a redesigned results page. E1: positivity on or off, 3 seeds, 20 rounds. E2: mesh size. E3: NACA 2412 at four angles. E4: speed. E5: a cylinder. E6: a stop rule for training.

**What was measured.** E2: on the same mesh the PINN beats the FVM: {{ratio:E2|fvm|100x50|rms_cp ; E2|pinn|100x50|rms_cp}} lower Cp RMS at 100x50 and {{ratio:E2|fvm|200x100|rms_cp ; E2|pinn_gs_nopos|200x100|rms_cp}} at 200x100 (one seed per cell). E4: a trained network evaluates a 50x25 field in {{E4|pinn|50x25|infer_ms}} and a panel solve takes {{E4|panel|140 panels|infer_ms}} (measured); the FVM solve times {{sec:E4|fvm|50x25|infer_ms}} to {{sec:E4|fvm|400x200|infer_ms}} are quoted from the paper, so this is not a like-for-like comparison. E5: on the cylinder the PINN has Cp RMS {{E5|pinn|50x25|rms_cp}} against {{E5|panel|140 panels|rms_cp}} for the panel method with Prandtl-Glauert, a modest gain, and does not beat the FVM at 200x100 ({{E5|fvm|200x100|rms_cp}}).

**What failed.** (a) Training at 50x25 reaches its best accuracy at rounds 3 to 5 and then 5 of 6 runs degrade while the loss keeps falling; the suspected cause, not yet tested, is that the Euler residual does not pin the circulation (no Kutta condition). (b) The reference-free plateau stop rule stopped {{count:e6_replay.csv|drifted=True|stop_round!=never}} of {{count:e6_replay.csv|drifted=True}} drifting runs: a negative result; the practice is a fixed early stop at round 3 to 5. (c) One NACA 2412 run collapsed to the trivial solution (Cp RMS {{E3|pinn|100x50|rms_cp:1|NACA2412_a4}}); it is kept in the data and two retries worked. (d) The 200x100 training time is withheld because the laptop slept during the run. (e) An early conclusion that positivity hurts came from one lucky seed and was withdrawn: neither setting wins.

**Next.** Candidate fixes for the drift (a Kutta-type constraint, a periodic input, a panel-method prior), a parametric angle of attack, training on a drawn shape, and a viscous extension; none is started.

## 5. Remaining work

| Item | State |
|---|---|
| Project report in the institute's capstone template | Built by the report builder; the people-specific fields are left for the team |
| Panel presentation (17 slides with speaker notes), synopsis, related-papers guide, viva sheet | Built (this package) |
| Clean-copy offline test and a recorded backup video | Not done; the video must be recorded by a team member |
| Final checklist, tag v1.0 | Not done |
| Ablations for the drift, training on a drawn shape, a 400x200 PINN, random airfoils | Not done (future work) |

## Signatures

Students: [FILL IN: student signatures]
Supervisor: [FILL IN: supervisor remarks, signature and date]
