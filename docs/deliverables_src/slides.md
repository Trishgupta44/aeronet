# AeroNet
Layout: title
Eyebrow: Major project  /  panel presentation
- A physics-informed neural network solver for airfoil flow: verified, validated and shown live
- Team: [FILL IN: team member names and enrolment numbers]
- Course: [FILL IN: course, semester and year]
- Guide: [FILL IN: supervisor name]
Notes:
Open with one sentence: AeroNet takes a published physics-informed neural network (PINN) solver for airflow around wings, called NNfoil, by Cao, Song and Zhang (arXiv 2401.08705), and asks three honest questions: is the code correct, does it reproduce the paper's claim on a laptop CPU, and can we show it live.

Say the credit out loud: NNfoil is their work. We did not invent the method. What we added is the verification, the experiments, the grid-sequencing trick, the stop-rule test and the web app.

Say the rule of the talk: every number on a slide is labelled measured (we ran it on this laptop), quoted (copied from the paper) or fitted (tuned to data). Where a result is negative we show it.

The three fields on this slide are filled in by the team before the presentation.

---
# Predicting the air around a wing is slow
Eyebrow: 01  The problem
Tag: quoted
- Classical CFD solves the flow again for every new shape or angle
- It needs a fine mesh: the paper's finite-volume solver took {{sec:E4|fvm|400x200|infer_ms}} on its finest mesh
- A PINN learns from the equations, with no labelled data
- A trained network answers its one case in milliseconds
- Our question: how good is it, honestly?
Notes:
Wings, drone rotors and turbine blades all depend on the flow around an airfoil. A classical finite-volume (FVM) solver gives accurate answers but it must be run again for every new shape, angle or Mach number, and accuracy needs a fine mesh. The NNfoil paper reports FVM wall times of {{sec:E4|fvm|50x25|infer_ms}}, {{sec:E4|fvm|100x50|infer_ms}}, {{sec:E4|fvm|200x100|infer_ms}} and {{sec:E4|fvm|400x200|infer_ms}} for four meshes (50x25 up to 400x200). These are QUOTED from Table 1 of the paper, measured by its authors on another machine, and the paper itself says the table is not meant as a rigorous comparison. We never re-ran the FVM.

The PINN idea is to skip the labelled data: the network is trained to satisfy the Euler equations of inviscid flow directly. Once trained, evaluating it is fast, but it answers only the one condition it was trained for. That trade is the heart of the project and it comes back on the speed slide.

Source: results/summary.csv, experiment E4 (quoted rows).

---
# What a PINN is
Eyebrow: 02  The method
- A network maps a mesh position to the flow state (u, v, p, rho)
- Loss = how badly the Euler equations and boundary conditions are broken
- Training here: minutes on a laptop CPU, for example {{E2b|pinn|50x25|train_s:0}} at 50x25 (measured)
![PINN training loop](docs/deliverables_src/slides_img/pinn_diagram.png)
Notes:
A PINN is a neural network trained to obey a physical law instead of fitting examples. In NNfoil the network is five layers of 128 units. It takes a position on the computational mesh and returns velocity components u and v, pressure p and density rho. Automatic differentiation gives the derivatives, so we can form the residual of the four Euler equations (continuity, two momentum equations, energy) at the mesh points. The loss is that residual plus the wall condition (no flow through the airfoil) and the far-field condition (free stream far away). The optimiser is L-BFGS, run in rounds.

We checked the four residuals against an independent conservative form of the Euler equations in the verification table (docs/verification.md): they match term by term, so the physics in the loss is correct.

Why this needs no labelled data: the data would be the answer, and the equations already contain it. The FVM data is used only afterwards, to score the network.

Measured training time: {{E2b|pinn|50x25|train_s:0}} for the 50x25 run with positivity on and {{E2b|pinn|100x50|train_s:0}} for the 100x50 cold start (8 CPU threads, idle laptop; results/summary.csv, E2b).

---
# Why airfoils are hard: the leading edge
Eyebrow: 03  The difficulty
Tag: measured
- Pressure changes fastest in the first few percent of the chord
- The 50x25 mesh has only a few wall nodes there, so a smooth network smears the peak
- Sun et al. also report their largest errors at the leading edge (quoted)
![Cp of NACA 0012 and the wall nodes of the 50x25 mesh](docs/deliverables_src/slides_img/le_gradient.png)
Notes:
The picture shows the FVM reference data bundled with NNfoil (800x400 mesh, NACA 0012, Mach 0.4, 4 degrees): the pressure coefficient Cp falls from the stagnation value near 1 to a suction peak within about the first 2 percent of the chord. On the right, the red lines are the wall nodes of the 50x25 mesh in that region: only a few. Any solver with so few points there has to represent a very sharp change.

Two sentences of context. The NNfoil abstract says exactly this: PINNs struggle with airfoil flow because the fluid accelerates sharply near the leading edge. Sun, Sengupta and Juniper (CMAME 2023) report their largest errors at the leading edge; that statement comes from the one-week plan's summary of their paper, not from a fresh reading, so say "quoted".

Data: NNfoil/FVM_Cp/NACA0012_800_400_Cp.dat and NNfoil/mesh/NACA0012_50_25.x, drawn by research/docs_build/make_slide_figures.py.

---
# NNfoil's trick: a mesh transformation
Eyebrow: 04  The method
- The curved mesh around the airfoil becomes a plain rectangle
- The network learns on the rectangle (xi, eta)
- The wall is one straight edge; the equations are mapped back with the Jacobian
![Physical mesh and computational rectangle](docs/deliverables_src/slides_img/mesh_transformation.png)
Notes:
This is the central idea of Cao, Song and Zhang. The mesh around the airfoil is an O-grid: lines of constant eta are rings around the wall, lines of constant xi run outwards. Their abstract says the network learns the flow in a uniform computational space, which avoids the sharp local transition and the wall boundary of physical space. The picture is drawn from the bundled 50x25 mesh: left is what we see, right is what the network sees. The airfoil surface is the bottom edge, so the slip-wall condition becomes a condition on one straight line.

The price: the derivatives in the Euler equations have to be transformed back with the metric Jacobian of the mapping. We checked that chain rule against an independent derivative in physical space (verification table). Two findings live here and come up on the verification slide: the mesh generator's smoothing term, and a wrong-sign Jacobian cell at the trailing-edge seam.

Only the statement of the idea comes from the paper's abstract (which we read); the implementation details are our reading of the code.

---
# Related work: three approaches
Eyebrow: 05  Related work
Tag: quoted + measured
| | Sun et al. 2023 | NeuralFoil 2025 | NNfoil 2024 (our base) |
|---|---|---|---|
| Physics | Navier-Stokes, Re 20 | viscous, learned from XFoil | Euler, inviscid |
| Training data | none | 7.9 million XFoil runs | none |
| Cost | 18 h or 75 h on a GPU | about {{E4|neuralfoil|large|infer_ms}} per call (measured here) | one network per case: minutes on a CPU (measured here) |
| Strength | shape optimisation inside one training | fast design tool | accurate on sparse meshes |
Notes:
Three papers, three trade-offs. Read the table as quoted unless it says measured.

Sun, Sengupta and Juniper (CMAME 2023): one PINN models the flow and optimises the airfoil shape at the same time, with adaptive sampling of the points. Its abstract also says it is more expensive than adjoint-based optimisation. The Reynolds number 20 and the training times 18 h (one parameter) and 75 h (eleven parameters, RTX 3080) come from our one-week plan's summary of the paper; we did not re-check them against the full text, so they are quoted.

NeuralFoil (Sharpe and Hansman, 2025): a surrogate of XFoil trained on millions of XFoil runs, with speedups of 8x to 1000x reported at equal accuracy; the 7.9 million figure is from the plan's summary. It is viscous and data-driven, the opposite of NNfoil. We installed it and measured one call at {{E4|neuralfoil|large|infer_ms}} (mean of 100 calls, large model); its lift value is viscous at Re 6e6 so it is not comparable with our inviscid cases.

NNfoil (Cao, Song and Zhang, 2024): data-free, inviscid, solved on a transformed mesh, with an open-source solver. Per condition it trains for minutes (we measured {{E2b|pinn|50x25|train_s:0}} to {{E2b|pinn|100x50|train_s:0}}).

Positioning sentence for the panel: Sun et al. show a PINN can model and optimise but is slow and weak at the leading edge; NeuralFoil shows surrogates are practical but needs millions of labelled runs; NNfoil sits between them, and AeroNet verifies, reproduces and validates it.

---
# What AeroNet adds
Eyebrow: 06  Our contribution
- **Verify**: line-by-line check of the solver, 11 findings
- **Reproduce**: runs on a laptop CPU, no GPU
- **Validate**: against the bundled FVM data, experiments E1 to E6
- **Present**: web app to draw a shape, a PINN Lab and a results page
![AeroNet home page](docs/milestones/img/merge_tabs_home_dark_1440.png)
Notes:
The proposal had four pillars and each one has a deliverable.

Verify: every component of NNfoil is compared with the theory in docs/verification.md (equations, mesh mapping, boundary conditions, forces). Reproduce: the code forced a CUDA GPU; it now runs on a CPU with a pinned environment, one command per run. Validate: the solver is scored against the bundled FVM Cp files (NACA 0012 on five meshes, NACA 2412 at several angles, a cylinder), with one error metric for every method. Present: the web app is a FastAPI backend and a React front end, with the classical panel method as a live solver, a PINN Lab with 14 trained cases and a training replay, and a results page.

What we did not do, so nobody has to ask: training a PINN on a hand-drawn shape in the app is not done; the app's live solver is the classical panel method, and the PINN Lab shows cases we trained offline. The screenshot is the Home page in dark mode.

---
# Verification: 11 findings in the code
Eyebrow: 07  Verification
Tag: measured
- **Fixed:** CUDA forced (1), one CPU thread (4), forces never computed (7), flow condition hard-coded (8)
- **Fixed:** NumPy 2 and PyTorch 2.6 crashes (5, 6)
- **Safety fix, no accuracy gain:** p and rho unconstrained (2)
- **Confirmed, low impact:** mesh smoothing cross term 0.5 where Winslow needs 2 (3); wrong-sign Jacobian cells at the seam (10)
- **Documented:** weight init is a no-op (11), training cost (9)
Notes:
Eleven findings, each with the upstream file and line, a fix or a verdict, and a test or a CSV in results/ (table: docs/milestones/M1-nnfoil-verified.md, details: docs/verification.md). Our rule from the start was to treat the proposal's suspected bugs as hypotheses, so some are confirmed with their impact stated and some turned out to matter less than expected.

Finding 3, the mesh generator: the cross-derivative term uses a coefficient of 0.5 times beta where the Winslow elliptic equation needs 2 times beta, four times too small. CONFIRMED. Impact is low: with the code's coefficient the true Winslow residual is {{csv:finding3_mesh_smoothing.csv|variant=upstream|max_abs_winslow_residual:e2}}, with 2 it is {{csv:finding3_mesh_smoothing.csv|variant=winslow|max_abs_winslow_residual:e2}}; nodes move by at most {{csv:finding3_mesh_smoothing.csv|variant=winslow|max_dist_to_bundled:f4}} chord and both grids are valid (single-signed Jacobian, minimum angle {{csv:finding3_mesh_smoothing.csv|variant=upstream|min_grid_angle_deg:f1}} against {{csv:finding3_mesh_smoothing.csv|variant=winslow|min_grid_angle_deg:f1}} degrees). We did NOT change it, because the shipped meshes and the FVM data were made with it; our own mesh generator for the cylinder and NACA 2412 uses the correct 2. Source: results/finding3_mesh_smoothing.csv.

Finding 10, the Jacobian seam: at the trailing-edge seam the code differences one-sidedly across a periodic direction, which gives {{csv:finding10_jacobian.csv|mesh=NACA0012_50_25|upstream_wrong_cells:i}} cells of the wrong sign on the 50x25 mesh; periodic differencing gives {{csv:finding10_jacobian.csv|mesh=NACA0012_50_25|periodic_wrong_cells:i}}. The impact is low because those cells carry almost no weight in the Jacobian-squared loss weighting. An opt-in flag exists; the default is unchanged. Source: results/finding10_jacobian.csv.

Finding 2 deserves the honest sentence: we added softplus to keep pressure and density positive. It is a safety measure. Experiment E1 later showed it does not improve accuracy and does not decide whether training drifts (next slides).

The fixes: 1 device (tested), 4 threads (a runtime setting, no test), 5 and 6 are crashes on current NumPy and PyTorch (tested), 7 and 8 add Cl and Cd and the flow-condition arguments (tested). Finding 11: the intended N(0,1) weight initialisation is overwritten by weight-norm on the first forward pass; documented, not changed because it would alter every result.

---
# Validation: the PINN against the reference
Eyebrow: 08  Validation
Tag: measured
- NACA 0012, Mach 0.4, 4 degrees, 100x50 mesh
- The PINN's wall Cp lies on the 800x400 reference; the FVM on the same mesh does not
- Cp RMS: PINN {{E2|pinn|100x50|rms_cp}}, FVM {{E2|fvm|100x50|rms_cp}}
- Cl: PINN {{E2|pinn|100x50|cl}}, reference {{E2|pinn|100x50|cl_ref}}
![Wall Cp at 100x50: PINN, FVM on the same mesh and the 800x400 reference](results/figures/m1_cp_100x50.png)
Notes:
The grey dots are the reference: the FVM solution on an 800x400 mesh. The black squares are the FVM on the same 100x50 mesh as the network: its suction peak is visibly too low. The two lines are the PINN (a cold-start run and a grid-sequenced run); both sit on the reference, including the suction peak.

Numbers (measured, results/summary.csv, experiment E2): Cp error is the RMS over reference points with 0.02 <= x/c <= 0.98, nearest predicted surface point, one definition for every method. PINN cold start at 100x50: {{E2|pinn|100x50|rms_cp}}; FVM at 100x50: {{E2|fvm|100x50|rms_cp}}. Lift: the PINN gives Cl {{E2|pinn|100x50|cl}} against the reference {{E2|pinn|100x50|cl_ref}} ({{E2|pinn|100x50|cl_err_pct}}); the FVM at 100x50 gives {{E2|fvm|100x50|cl}}.

Caveats to say: the reference is itself an FVM result (800x400); a panel method with Prandtl-Glauert correction agrees with it to RMS {{E2|panel|140 panels|rms_cp}}, which is our best independent check. One seed per number. Drag from an inviscid solver is zero in theory (d'Alembert), so any drag the network reports is numerical error and we never present it as drag.

---
# Mesh-size result: PINN against FVM
Eyebrow: 09  Experiment E2
Tag: measured
- On the same sparse mesh the PINN has the lower Cp error, in every mesh we trained
- 100x50: {{ratio:E2|fvm|100x50|rms_cp ; E2|pinn|100x50|rms_cp}} lower (cold start)
- 200x100: {{ratio:E2|fvm|200x100|rms_cp ; E2|pinn_gs_nopos|200x100|rms_cp}} lower (one run)
- A 140-panel panel method is level with it: {{E2|panel|140 panels|rms_cp}}
- One seed per cell; no 400x200 PINN
![Cp RMS against mesh size, PINN and FVM](results/figures/m3_e2_mesh_study.png)
Notes:
This reproduces, on a small scale, the headline claim of the NNfoil paper: on very sparse meshes the PINN is more accurate than a second-order finite-volume solver.

Same-mesh comparison against the same 800x400 reference. 50x25: FVM {{E2|fvm|50x25|rms_cp}}, PINN {{E2|pinn|50x25|rms_cp}} (positivity on) and {{E2|pinn_nopos|50x25|rms_cp}} (positivity off). 100x50: FVM {{E2|fvm|100x50|rms_cp}}; PINN cold {{E2|pinn|100x50|rms_cp}}, grid-sequenced {{E2|pinn_gs|100x50|rms_cp}} and {{E2|pinn_gs_nopos|100x50|rms_cp}}. 200x100: FVM {{E2|fvm|200x100|rms_cp}}, PINN {{E2|pinn_gs_nopos|200x100|rms_cp}}. 400x200: FVM {{E2|fvm|400x200|rms_cp}}; we did not train a PINN there.

Three honesty points. First, the PINN is not copying the coarse FVM: its Cp differs from the same-mesh FVM by about the FVM's own error. Second, the panel method is just as accurate as the PINN on this smooth case, so accuracy is not a reason to prefer the PINN over a good panel method; what the PINN offers is that it solves the Euler equations without the Prandtl-Glauert assumption (our interpretation, not tested). Third, the training time of the 200x100 run is withheld: the laptop went to sleep during it and one logged round spans about 34,000 seconds, so its wall time is not valid. We show no number rather than a wrong one.

---
# Grid sequencing and the seed check
Eyebrow: 10  Experiment: seeds
Tag: measured
- Grid sequencing (our addition): train on 50x25, move the weights to 100x50, 4 more rounds
- Grid-sequenced: {{count:seed_check.csv|method=gs|collapsed=yes}} of {{count:seed_check.csv|method=gs}} runs collapsed
- Cold start without positivity collapsed in {{count:seed_check.csv|mode=nopos|method=cold|collapsed=yes}} of {{count:seed_check.csv|mode=nopos|method=cold}} seeds; with positivity {{count:seed_check.csv|mode=pos|method=cold|collapsed=yes}} of {{count:seed_check.csv|mode=pos|method=cold}}, but a NACA 2412 cold start collapsed
![Final Cl by seed: grid-sequenced and cold starts](results/figures/m3_seed_check.png)
Notes:
Observation from a separate research session: from a cold start on 100x50, the original method can fall into a trivial solution (the loss is exactly zero and the lift is zero). Training on the cheap 50x25 mesh first and moving the weights to 100x50, with the input normalisation rescaled, avoids it. This is not in the NNfoil paper; it is our addition.

The seed check (results/seed_check.csv, measured): seeds 0 to 3, both positivity modes, NACA 0012 at Mach 0.4 and 4 degrees. Grid-sequenced: {{count:seed_check.csv|method=gs|collapsed=no}} of {{count:seed_check.csv|method=gs}} runs completed without collapse, with final Cl between 0.502 and 0.516 against the reference 0.519 and Cp RMS about 0.015 to 0.016. The pass line was set before the check. Cold start without positivity: seed 3 collapsed, all five rounds failed. Cold start with positivity: no collapse in four seeds, but the NACA 2412 run at 4 degrees with positivity on, cold start, did collapse (next slide but one), so positivity is not a cure.

What we do NOT claim: speed. The coarse stage costs a full 50x25 run, and the 100x50 stage times in the tables exclude it. Reliability is the claim, not speed.

---
# Late drift, and a stop rule that failed
Eyebrow: 11  Experiments E1 and E6
Tag: measured
- 50x25: accuracy peaks at rounds 3 to 5, then 5 of 6 runs degrade while the loss keeps falling (run 1: Cp RMS {{csv:e1_rounds.csv|case=NACA0012_a4_s0|method=pinn|round=3|cp_rms:f3}} at round 3, {{csv:e1_rounds.csv|case=NACA0012_a4_s0|method=pinn|round=19|cp_rms:f3}} at round 19)
- Positivity on or off: neither wins
- Plateau stop rule: stopped {{count:e6_replay.csv|drifted=True|stop_round!=never}} of {{count:e6_replay.csv|drifted=True}} drifting runs; practice is to stop at round 3 to 5
![E1: Cp error and lift by training round, six runs](results/figures/m3_e1_trajectories.png)
Notes:
This is the most important negative result and the panel should hear it from us first.

E1: NACA 0012 at 50x25, three seeds, positivity on and off, up to 20 rounds. In all six runs the Cp error at round 3 is between 0.027 and 0.032, about four times better than the FVM on its own mesh. Then five of six runs degrade: the final error rises to between 0.073 and 0.222, and the lift falls to between 0.168 and 0.412 against the reference 0.519. In run 1 (seed 0, positivity on) the loss falls from {{csv:e1_rounds.csv|case=NACA0012_a4_s0|method=pinn|round=3|loss:e1}} at round 3 to {{csv:e1_rounds.csv|case=NACA0012_a4_s0|method=pinn|round=19|loss:e1}} at round 19, while the Cp error rises from {{csv:e1_rounds.csv|case=NACA0012_a4_s0|method=pinn|round=3|cp_rms:f3}} to {{csv:e1_rounds.csv|case=NACA0012_a4_s0|method=pinn|round=19|cp_rms:f3}}. A lower loss means a worse answer.

Why (suspected, not tested): the Euler residual does not pin the circulation. Nothing in the loss enforces a Kutta condition (finding N22 in docs/nnfoil_improvements.md), so many circulation values give nearly the same residual and the network can drift along that valley. We list a Kutta-type constraint, a periodic input and a panel-method prior as candidate fixes; none has been tried.

We expected positivity (the softplus fix) to matter. An early single seed made it look harmful; that came from one lucky positivity-off run and the conclusion was withdrawn. Mean final error over three seeds is 0.130 with positivity and 0.146 without: neither mode wins; positivity only changes when the jump happens.

E6: could a rule that does not use the FVM reference stop training near the good answer? Rule: stop when the lift change is below 0.002 and the wall-Cp change below 0.005 for two rounds. We set the thresholds after looking at E1, so E1 is the development set and supports no claim; the judge is the 15 held-out runs. Result: the rule stopped {{count:e6_replay.csv|drifted=True|stop_round!=never}} of {{count:e6_replay.csv|drifted=True}} drifting runs. It did fire in some non-drifting runs and never stopped badly, but that is harmless, not useful. The online confirmation run was cut off and tests nothing. This is a negative result. What works is a fixed early stop at round 3 to 5, a recipe derived on E1 and checked on six new seeds; it is not risk-free (round 4 already had one run at 0.101). Sources: results/e1_rounds.csv, results/e6_replay.csv, results/e6_stop_rule.md.

---
# NACA 2412 and the cylinder
Eyebrow: 12  Experiments E3 and E5
Tag: measured + quoted Mach
- NACA 2412 (Mach 0.3, quoted): Cl error {{E3|pinn|100x50|cl_diff|NACA2412_a-4}} at -4 deg, {{E3|pinn|100x50|cl_diff|NACA2412_a8}} at 8 deg; the 4 deg cold start collapsed (Cp RMS {{E3|pinn|100x50|rms_cp:1|NACA2412_a4}}) and is kept
- Cylinder: PINN Cp RMS {{E5|pinn|50x25|rms_cp}} and {{E5|pinn_nopos|50x25|rms_cp}} against panel + PG {{E5|panel|140 panels|rms_cp}}: modestly better; FVM at 200x100 is {{E5|fvm|200x100|rms_cp}}
![E3: lift polar of NACA 2412](results/figures/m3_e3_polar.png)
![E5: cylinder, PINN against panel method and FVM](results/figures/m3_e5_cylinder.png)
Notes:
NACA 2412 (E3). The Mach number of the bundled files is not stored with them. The paper's parametric NACA 2412 case is Mach 0.3 (quoted from section 3.4); our own panel-method fit of the files preferred about 0.25 (fitted, within what a Prandtl-Glauert panel method can resolve). Every E3 row uses 0.3 and the Mach number is unconfirmed by us. We trained 100x50 cold-start PINNs at four angles. Lift errors against the FVM: {{E3|pinn|100x50|cl_diff|NACA2412_a-4}} at -4 degrees, {{E3|pinn|100x50|cl_diff|NACA2412_a0}} at 0, {{E3|pinn|100x50|cl_diff|NACA2412_a8}} at 8. At the same four angles the panel method's lift error is between -0.009 and +0.022, so overall the PINN is not better than the panel method on this case. The first 4-degree run (positivity on, seed 0) collapsed to the trivial solution, Cl about zero and Cp RMS {{E3|pinn|100x50|rms_cp:1|NACA2412_a4}}; it stays in the data. Two retries worked: seed 1 gives Cl {{E3|pinn|100x50|cl|NACA2412_a4_s1}} and positivity off gives {{E3|pinn_nopos|100x50|cl|NACA2412_a4}}, against the reference {{E3|pinn|100x50|cl_ref|NACA2412_a4_s1}}. The retries were chosen because the first run failed, and we say so.

Cylinder (E5). Mach 0.4 and angle 0 are quoted from the paper section 3.1 and confirmed by the stagnation point. The panel method with Prandtl-Glauert has its suction peak at Cp {{E5|panel|140 panels|note:Cp_min}} against {{E5|fvm|800x400|note:Cp_min}} for the reference. The PINN (50x25, 10 rounds) has Cp RMS {{E5|pinn|50x25|rms_cp}} (positivity on) and {{E5|pinn_nopos|50x25|rms_cp}} (off), against {{E5|panel|140 panels|rms_cp}} for the panel method with Prandtl-Glauert: better, but by a modest margin. An earlier note in our docs said about 0.30 for the panel method; that was the incompressible circle (RMS {{E5|panel|140 panels (Ma 0)|rms_cp}}), and it was corrected. The PINN is not better than the FVM at 200x100 ({{E5|fvm|200x100|rms_cp}}). The exact Cl of a cylinder is zero; the PINN gives {{E5|pinn|50x25|cl}} (positivity on) and {{E5|pinn_nopos|50x25|cl}} (off), a small spurious circulation.

---
# Speed: measured against quoted
Eyebrow: 13  Experiment E4
Tag: measured + quoted
- Measured: trained PINN, full field at 50x25, {{E4|pinn|50x25|infer_ms}}; panel solve, {{E4|panel|140 panels|infer_ms}}
- Quoted (another machine): FVM solve {{sec:E4|fvm|50x25|infer_ms}} at 50x25 up to {{sec:E4|fvm|400x200|infer_ms}} at 400x200
- Not like for like: a PINN answers ONE condition; a new one means retraining
![Inference and solve times, measured and quoted](results/figures/m3_e4_speed.png)
Notes:
Read this slide with the warning first: this is not a like-for-like comparison, and the paper itself says its timing table is not meant as a rigorous comparison.

Measured here (AMD Ryzen 9 6900HS laptop, CPU only, PyTorch, 8 threads, mean of 100 calls after warm-up, nothing else running): evaluating an already trained PINN over every node of the mesh takes {{E4|pinn|50x25|infer_ms}} at 50x25, {{E4|pinn|100x50|infer_ms}} at 100x50 and {{E4|pinn_gs_nopos|200x100|infer_ms}} at 200x100; a Hess-Smith panel solve with 140 panels takes {{E4|panel|140 panels|infer_ms}}; NeuralFoil takes {{E4|neuralfoil|large|infer_ms}} per call, but it is viscous at Re 6e6 so only its time is shown.

Quoted: the FVM solve times {{sec:E4|fvm|50x25|infer_ms}}, {{sec:E4|fvm|100x50|infer_ms}}, {{sec:E4|fvm|200x100|infer_ms}} and {{sec:E4|fvm|400x200|infer_ms}} for the four meshes come from Table 1 of the NNfoil paper, run by its authors on another machine.

Why it is not comparable: the milliseconds are a lookup of ONE condition. A new shape, Mach number or angle needs retraining, at {{E2b|pinn|50x25|train_s:0}} to {{E2b|pinn|100x50|train_s:0}} per NACA 0012 run on this laptop. The panel method needs no training and costs a few milliseconds per new condition. So we claim no speed win for the PINN; the honest comparison is accuracy at equal cost, which is E2. The 200x100 training time is withheld because the laptop slept.

---
# Live demo
Eyebrow: 14  Demo
Layout: demo
- **Explore**: NACA 0012; drag the angle of attack
- **Draw**: sketch a blob; load the sedan on its road
- **PINN Lab**: the 100x50 case, then the 50x25 replay
- **Results**: headline tiles, then what did not work
- **About**: what is real, what is a stand-in
![PINN Lab, a trained case with the Cp comparison](docs/milestones/img/c8_lab_gs100_1440.png)
Notes:
Five-minute script (from docs/demo_script.md). Before the talk, with the backend running (app\start.ps1, then http://localhost:8000 in a full-screen browser at 1440 px or wider), wait for the hero flow to move once, click through the PINN Lab once to warm it up, close other programs. Spare tabs: http://localhost:8000/?present and http://localhost:8000/?pinn=naca0012_50x25_a4_m04&round=3#pinn . Say "measured" or "quoted" with every number.

0:00 (15 s) Home. The hero shows live flow around NACA 0012. Say: "AeroNet shows how air moves around a shape. This flow is computed live by a classical panel method. Beside it I will show a neural network, a PINN, that solves the same physics, and I will say plainly where it wins and where it does not."

0:15 (45 s) Explore. Header: Explore, preset NACA 0012. Drag Angle of attack from 4 to 10, back to 4, then to 0. Hover the field to read Cp; click the Forces layer for the lift arrow. Say: "Real panel-method physics, a few milliseconds per solve (the page shows the time, measured on this laptop). Suction on top, stagnation at the nose. At zero degrees a symmetric airfoil gives zero lift, as it should. This is inviscid and subsonic: no stall, no friction."

1:00 (60 s) Draw. Header: Draw. Mouse down in the drawing area, sketch a closed blob, lift the mouse: it solves. Untick Lift (Kutta condition) if the warning about the rounded rear appears. Then click Sedan in the shape library; the road and the amber note appear. Read the note aloud. Say: "Any closed outline is solved when you lift the pen. A blunt blob has Cl about zero. The sedan sits on a road (ground effect by images) and shows downforce, a negative Cl. Honest note, in amber on screen: this is idealised potential flow, there is no wake, so the pressure at the rear is not realistic and the app never shows a car's drag."

2:00 (90 s) PINN Lab. Header: PINN Lab. Pick a trained case: NACA 0012, 100x50, 4 deg, Ma 0.4, grid-seq., positivity on. Read Cp RMS and Cl in the metrics row; look at the wall Cp against FVM and panel. Then choose NACA 0012, 50x25, 4 deg, Ma 0.4, positivity on. Under Training replay press Play, stop near rounds 3 to 8, then let it run to the end. Say: "Each case is one trained network for one shape, Mach number and angle. This 100x50 one agrees with the finite-volume reference to Cp RMS {{E2|pinn_gs|100x50|rms_cp}} and Cl {{E2|pinn_gs|100x50|cl}} against {{E2|pinn_gs|100x50|cl_ref}} (measured; the FVM on the same mesh is at {{E2|fvm|100x50|rms_cp}}). Now the 50x25 one. Watch the training: around round 3 it is accurate (RMS {{csv:e1_rounds.csv|case=NACA0012_a4_s0|method=pinn|round=3|cp_rms:f3}}), then at round 8 it jumps to {{csv:e1_rounds.csv|case=NACA0012_a4_s0|method=pinn|round=8|cp_rms:f3}} and ends at {{csv:e1_rounds.csv|case=NACA0012_a4_s0|method=pinn|round=19|cp_rms:f3}} while the loss keeps falling. That late drift is unsolved. What works is stopping at rounds 3 to 5." The replay may show only a subset of rounds, so read the round on screen rather than trusting these numbers to the last digit.

3:30 (45 s) Results. Header: Results. Point at the headline tiles, then scroll to What did not work. Say: "On the same sparse mesh the PINN has {{ratio:E2|fvm|100x50|rms_cp ; E2|pinn|100x50|rms_cp}} lower Cp error than the FVM at 100x50 and {{ratio:E2|fvm|200x100|rms_cp ; E2|pinn_gs_nopos|200x100|rms_cp}} at 200x100, lift error {{E2|pinn|100x50|cl_err_pct}} (all measured, one seed per cell). {{E4|pinn|100x50|infer_ms}} is evaluating a trained network, not a new solve: a new condition means retraining. And here is what did not work: the stop rule stopped {{count:e6_replay.csv|drifted=True|stop_round!=never}} of {{count:e6_replay.csv|drifted=True}} drifted runs, a cold start collapsed in {{count:seed_check.csv|mode=nopos|method=cold|collapsed=yes}} of {{count:seed_check.csv|mode=nopos|method=cold}} seeds, 5 of 6 runs drift, and one NACA 2412 run collapsed and is kept in the data."

4:15 (30 s) About, at the bottom of Results: What is real, what is a stand-in. Say: "Limits: inviscid only; cars are idealised; one network per condition; mostly single seeds on one laptop CPU; the FVM times are quoted from the paper. Next: a Kutta-type fix for the drift, a parametric angle of attack, training on a drawn shape, a viscous extension. NNfoil is by Cao, Song and Zhang; we verified it, ran it on a CPU and measured it."

4:45 Spare 15 s, questions.

If something fails. Page slow: press Pause on the flow card. Backend dies: the page switches to saved results and shows a banner with the date; say "the live solver is down, these are saved results", restart with app\start.ps1 and press Try again (only works if the page was loaded once while the backend was up). Network fails: nothing in the demo needs it. Laptop fails: show the screenshots in docs/milestones/img/ (Explore b9_explore_canvas_1440.png, blob b8_draw_blob_1440.png, sedan b8_draw_sedan_road_1440.png, 100x50 case c8_lab_gs100_1440.png, 50x25 replay c8_replay_drift_50x25_1440.png, Results results_1440_light.png). A recorded backup video has not been recorded yet (a human step, Plan D5).

---
# Limits and what comes next
Eyebrow: 15  Honest scope
- Inviscid and subsonic: no stall, no friction; cars are idealised and show no drag
- One network per flow condition; mostly single seeds on one laptop CPU
- Open: the late drift and a stop rule; the 200x100 training time is missing
- Not done: Kutta-type fix, periodic input, panel prior; training on a drawn shape; 400x200; random airfoils
- Next: parametric angle of attack, a viscous extension, 3D
Notes:
Say the limits ourselves, before the panel does.

Physics: everything is the Euler equations, so no boundary layer, no separation and no friction drag. The drag a solver reports is numerical error. Mach numbers above the critical value have shocks, which this formulation cannot capture (the paper's Mach 0.68 random-airfoil case is quoted as not captured). The cars in the app are idealised: the panel method has no wake, so rear pressure is not realistic and the app never shows a car's drag.

Evidence: most cells are a single seed. Replication exists only in the seed check (4 seeds) and in E1 and E6 (6 seeds each at 50x25), where the spread is large. All timings are one laptop CPU. The NACA 2412 Mach number and the FVM times are quoted from the paper.

Open problems we did not solve: the circulation drift (candidate fixes untried: Kutta-type constraint, periodic input, hard far-field, panel-method prior), a reference-free stop rule (the plateau rule failed), the 200x100 training time (laptop slept), a 400x200 PINN, the random airfoils, parametric angle of attack (the paper trains one network over alpha, which we did not reproduce), and training a PINN on a hand-drawn shape in the app.

Next: the parametric angle, optimisation on top of the PINN as in Sun et al., a viscous extension with confidence estimates in the style of NeuralFoil, and 3D.

---
# Conclusion
Eyebrow: 16  Summary
- The code is verified: 11 findings, fixed or documented, runs on a CPU
- On a sparse mesh the PINN beats the FVM: {{ratio:E2|fvm|100x50|rms_cp ; E2|pinn|100x50|rms_cp}} lower Cp error at 100x50 (measured)
- Grid sequencing is reliable; late drift and the failed stop rule are reported, not hidden
- A working app: draw a shape, PINN Lab, results page
- Questions?
Notes:
Four sentences to close. One: we verified a published PINN solver line by line and made it run on a laptop CPU. Two: we reproduced its headline claim on a small scale: on sparse meshes the PINN's wall pressure is closer to the reference than the finite-volume solver's on the same mesh ({{E2|pinn|100x50|rms_cp}} against {{E2|fvm|100x50|rms_cp}} at 100x50, measured, one seed). Three: we also found where it fails, the late drift of training at 50x25 and a stop rule that does not work, and a practical recipe that does. Four: everything is in a working demo that anyone can run offline.

For questions the viva sheet (6_Viva_Sheet.docx) has fifteen prepared answers with the evidence file for each. The hard ones to be ready for: why the stop rule failed, why positivity does not decide, why the drift happens (circulation not pinned by the Euler residual), what is measured against quoted, why the 200x100 time is missing, and why the cylinder PINN beats panel plus Prandtl-Glauert only modestly.
