# AeroNet: Viva Question and Answer Sheet
Fifteen prepared questions with short answers and the file that backs each one
Prepared for: {{field:students_names}}; panel or examiner: [FILL IN: panel or examiner]
Who answers which question: [FILL IN: question owners]

## How to use this sheet

Questions 1 to 9 are the standard bank from the one-week plan (section 7), answered with the final results. Questions 10 to 15 are the hard ones that this project's own results raise. Every answer names the evidence file. Say "measured" for numbers we ran, "quoted" for numbers copied from a paper, and "fitted" for numbers tuned to data. If you do not know, say which file would tell you.

## Standard questions

### 1. Why does a PINN need no labelled data?

The loss is the residual of the Euler equations at the mesh points plus the wall and far-field conditions, so the equations themselves are the teacher. The finite-volume data is used only afterwards, to score the network. We checked that the four residuals match an independent conservative evaluation of the Euler equations. Evidence: `docs/verification.md` (rows "Euler e1 to e4" and "Loss composition").

### 2. Why is the leading edge hard, and how does the mesh transformation help?

The flow accelerates sharply there (the NNfoil abstract), so the pressure changes very fast over a tiny distance, and a smooth network on a coarse mesh smears the suction peak. NNfoil maps the curved mesh onto a plain rectangle, so the network learns in a uniform computational space and the wall becomes one straight edge. Sun et al. also report their largest errors at the leading edge (quoted from our team's summary, not re-read). Evidence: `docs/deliverables_src/slides_img/le_gradient.png`, `docs/superpowers/plans/2026-09-24-aeronet-one-week-plan.md` section 0.

### 3. Why is drag about zero in your results?

For steady inviscid subcritical flow the exact drag is zero (d'Alembert's paradox), so any drag a solver reports is numerical error, for example the FVM reports Cd {{E2|fvm|50x25|cd}} at 50x25 and {{E2|fvm|400x200|cd}} at 400x200, falling toward zero as the mesh is refined (measured). That is why the app never shows a car's drag. Evidence: `docs/verification.md` (force row), `results/summary.csv` (column `cd`).

### 4. How is this different from NeuralFoil?

NeuralFoil is a network trained on millions of XFoil runs: viscous, data-driven, and about a millisecond per call ({{E4|neuralfoil|large|infer_ms}}, measured here). NNfoil is data-free, inviscid, and needs training for each flow condition. Its lift at 4 degrees is viscous at Re 6e6, so it is not comparable with our inviscid values and we show only its time. Evidence: `results/e4_speed.md`, `docs/deliverables_src/papers_guide.md`.

### 5. How is this different from Sun et al.?

Sun et al. solve Navier-Stokes at low Reynolds number with the shape parameters as network inputs and optimisation built in; reported training is 18 to 75 hours on a GPU (quoted from our team's summary of the paper). NNfoil solves Euler flow with a mesh transformation, one network per condition, and we measured {{E2b|pinn|50x25|train_s:0}} to {{E2b|pinn|100x50|train_s:0}} per run on a CPU. We did not run their method. Evidence: `docs/superpowers/plans/2026-09-24-aeronet-one-week-plan.md` section 0, `results/summary.csv` (E2b).

### 6. Is your speed comparison fair?

No claim of a fair speed win is made. Measured here: evaluating a trained PINN ({{E4|pinn|50x25|infer_ms}} at 50x25), a panel solve ({{E4|panel|140 panels|infer_ms}}) and NeuralFoil. Quoted from the paper: the FVM times ({{sec:E4|fvm|50x25|infer_ms}} to {{sec:E4|fvm|400x200|infer_ms}}), run by its authors on another machine, and the paper says its table is not a rigorous comparison. A PINN answers one condition, so a new condition means retraining; accuracy at equal cost is experiment E2. Evidence: `results/e4_speed.md`, `docs/milestones/M3-pinn-results.md` section 7.

### 7. What bugs did you find, and how do you know they were bugs?

Eleven findings in `docs/verification.md`: 7 fixed (CUDA forced, one CPU thread, forces never computed, a flow condition hard-coded, two crashes on current NumPy and PyTorch, and a positivity safeguard) and 4 documented. Each has a test or a CSV. For example the mesh-smoothing coefficient is 0.5 where Winslow needs 2; the Winslow residual is {{csv:finding3_mesh_smoothing.csv|variant=upstream|max_abs_winslow_residual:e2}} against {{csv:finding3_mesh_smoothing.csv|variant=winslow|max_abs_winslow_residual:e2}}, with low impact. Evidence: `results/finding3_mesh_smoothing.csv`, `results/finding10_jacobian.csv`, `docs/milestones/M1-nnfoil-verified.md`.

### 8. Where does this fail?

It is inviscid and subsonic only (no stall or shocks; the paper's Mach 0.68 case is not captured, quoted). Training at 50x25 drifts after round 3 to 5, the stop rule failed, a cold start collapsed in {{count:seed_check.csv|mode=nopos|method=cold|collapsed=yes}} of {{count:seed_check.csv|mode=nopos|method=cold}} seeds without positivity and in one NACA 2412 run with it, and the PINN does not beat the FVM at 200x100 on the cylinder ({{E5|pinn|50x25|rms_cp}} against {{E5|fvm|200x100|rms_cp}}). It needs one network per condition. Evidence: `docs/milestones/M3-pinn-results.md` sections 3, 5, 6, 8, 9 and 11.

### 9. What would you do next?

Try a Kutta-type constraint, a periodic input and a panel-method prior against the drift (none tried yet); a parametric angle of attack as in the paper; training on a drawn shape; a viscous extension with confidence estimates; and 3D. Evidence: `docs/milestones/M3-pinn-results.md` section 10, `docs/nnfoil_improvements.md`.

## Questions from this project's own results

### 10. Why did your stop rule fail?

The rule stops when the lift change is below 0.002 and the wall-Cp change below 0.005 for two rounds in a row. On drifting runs those changes never stay that small, because the answer keeps moving: the rule stopped {{count:e6_replay.csv|drifted=True|stop_round!=never}} of {{count:e6_replay.csv|drifted=True}} drifting runs. It fired in some non-drifting runs and never stopped badly, but that is harmless, not useful. The thresholds were set after looking at the development runs, so only the held-out runs count. The practice is a fixed early stop at round 3 to 5, which is a recipe, not a guarantee. Evidence: `results/e6_stop_rule.md`, `results/e6_replay.csv`.

### 11. Why does positivity not decide whether training works?

Positivity (softplus on p and rho) was our safety fix. Over three seeds at 50x25 the mean final Cp RMS is 0.130 with it and 0.146 without, and 3 of 3 runs drift with it against 2 of 3 without it, so neither setting wins; it changes only when the jump happens. An early single-seed result that it hurt came from one lucky positivity-off run and was withdrawn. A cold NACA 2412 start with positivity on collapsed too. Evidence: `results/e1_rounds.csv`, `docs/milestones/M3-pinn-results.md` sections 3 and 6.

### 12. Why does the drift happen?

Our suspected cause, not yet tested: the Euler residual does not pin the circulation. Nothing in the loss enforces a Kutta condition (finding N22), so many circulation values give a similar residual and the network can wander along that valley. Evidence for it: in run 1 the loss falls from {{csv:e1_rounds.csv|case=NACA0012_a4_s0|method=pinn|round=3|loss:e1}} at round 3 to {{csv:e1_rounds.csv|case=NACA0012_a4_s0|method=pinn|round=19|loss:e1}} at round 19 while the Cp RMS rises from {{csv:e1_rounds.csv|case=NACA0012_a4_s0|method=pinn|round=3|cp_rms:f3}} to {{csv:e1_rounds.csv|case=NACA0012_a4_s0|method=pinn|round=19|cp_rms:f3}}, and the lift falls. Evidence: `docs/nnfoil_improvements.md` (N22), `results/drift_step1.csv`, `docs/milestones/M3-pinn-results.md` section 3.

### 13. What is measured, what is quoted, and what is fitted?

Measured: every training time, every error against the FVM reference, every inference time, the panel-method and NeuralFoil timings. Quoted: the FVM solve times, the NNfoil training times, the Mach numbers 0.4 (NACA 0012 and cylinder) and 0.3 (NACA 2412), all from the paper. Fitted: an earlier panel-method fit that preferred a Mach number of about 0.25 for the NACA 2412 files, which we did not use. Evidence: the `source` column of `results/summary.csv`, `docs/verification.md` ("Reference data conditions").

### 14. Why is the 200x100 training time missing?

The laptop went to sleep while that run was going, and one logged round spans about 34,000 seconds, so its wall time is not valid. We show no number rather than a wrong one. The accuracy of that run is valid (Cp RMS {{E2|pinn_gs_nopos|200x100|rms_cp}}, measured, one run). Evidence: the note in `results/summary.csv` (E2, 200x100 row) and `docs/milestones/M3-pinn-results.md` section 2.

### 15. Why is the cylinder PINN better than panel plus Prandtl-Glauert only modestly?

On the cylinder the PINN has Cp RMS {{E5|pinn|50x25|rms_cp}} and {{E5|pinn_nopos|50x25|rms_cp}} against {{E5|panel|140 panels|rms_cp}} for the panel method with Prandtl-Glauert, and the panel method's suction peak is {{E5|panel|140 panels|note:Cp_min}} against {{E5|fvm|800x400|note:Cp_min}} in the reference. The PINN is trained on a coarse 50x25 mesh for 10 rounds, and the FVM on that same mesh is far worse ({{E5|fvm|50x25|rms_cp}}), so we think the mesh limits it (not tested); it does not beat the FVM at 200x100 ({{E5|fvm|200x100|rms_cp}}). Why exactly the gain is small was not tested. Evidence: `docs/milestones/M3-pinn-results.md` section 8, `results/summary.csv` (E5).
