<!-- AeroNet project report, institute capstone template. Built by research/docs_build/build_all.py.
Numbers are double-brace placeholders (experiment, method, mesh, column, optional case) filled from results/summary.csv; a missing number is a hard error.
People-specific details are field placeholders (double braces, field: and a key) filled from docs/deliverables_src/fields.json; an empty value stays a highlighted fill-in field.
Highlighted fill-in fields are things only the team can supply: see docs/deliverables_src/FILL_IN_CHECKLIST.md. -->

@center[size=18,bold,before=36,after=8]: Jaypee Institute of Information Technology, Noida
@center[size=13,after=14]: {{field:department_line}}
@logo: docs/deliverables_src/jiit_logo.png | [FILL IN: institute logo image. Save it as docs/deliverables_src/jiit_logo.png and rebuild]
@center[size=15,before=30,after=24]: **Major Project Title:** AeroNet: A Physics-Informed Neural Network Solver for Airfoil Aerodynamics

@plaintable
| Enrol. No. | Name of Student |
|---|---|
| {{field:student_a_enrol}} | {{field:student_a_name}} |
| {{field:student_b_enrol}} | {{field:student_b_name}} |
| {{field:student_c_enrol}} | {{field:student_c_name}} |

@center[size=12,before=24,after=2]: Course Name: {{field:course}}
@center[size=12,after=2]: Program: {{field:programme}}
@center[size=12,after=2]: {{field:semester}}
@center[size=12,bold,before=18]: {{field:academic_year}}

#* Certificate

This is to certify that the project report entitled **AeroNet: A Physics-Informed Neural Network Solver for Airfoil Aerodynamics**, submitted by {{field:students_names_enrol}} in partial fulfilment of the requirements for the award of the degree of {{field:degree_programme}} at Jaypee Institute of Information Technology, Noida, during the academic session {{field:academic_year}}, is a record of the work carried out by them under my supervision. To the best of my knowledge, the work has not been submitted elsewhere for any other degree or diploma.

<br>

Signature of the Supervisor: ______________________________

Name of the Supervisor: {{field:supervisor_name_designation}}

Date: [FILL IN: date]

<br>

Signature of the Head of the Department: ______________________________

Name of the Head of the Department: {{field:hod_name}}

Date: [FILL IN: date]

#* Declaration of Originality

We hereby declare that this report entitled **AeroNet: A Physics-Informed Neural Network Solver for Airfoil Aerodynamics** is our own work, carried out under the guidance of {{field:supervisor_name}}. Wherever the work of others has been used, it is acknowledged in the text and in the list of references. The report has not been submitted, in whole or in part, for any other degree or diploma.

The physics-informed solver at the centre of this project, NNfoil, was written by W. Cao, J. Song and W. Zhang [1]; it is vendored in the project repository, which therefore stays private because the upstream code carries no licence. The fixes, the experiments, the application and this report are ours. An AI coding assistant was used to help review code and draft text; every result in this report was checked against the files in the repository by the team. [FILL IN: confirm this wording with your supervisor and the institute's rules on AI assistance.]

<br>

| Signature | Name of Student | Enrol. No. | Date |
|---|---|---|---|
| ____________ | {{field:student_a_name}} | {{field:student_a_enrol}} | [FILL IN: date] |
| ____________ | {{field:student_b_name}} | {{field:student_b_enrol}} | [FILL IN: date] |
| ____________ | {{field:student_c_name}} | {{field:student_c_enrol}} | [FILL IN: date] |

#* Acknowledgement

We would like to express our sincere gratitude to our supervisor, {{field:supervisor_name}}, for the guidance, encouragement and critical questions that shaped this project from its first synopsis to this report. We thank the Head of the Department, {{field:hod_name}}, and the faculty of the {{field:department_line}} for the environment and the facilities that made the work possible.

We are grateful to the authors of NNfoil, W. Cao, J. Song and W. Zhang, whose open-source solver and finite-volume reference data are the foundation of this study, and to the authors of NeuralFoil, PyTorch, NumPy, SciPy, Matplotlib, FastAPI and React, whose free software we used throughout. We also thank [FILL IN: friends, family, lab staff or anyone else the team wishes to thank].

{{field:students_names_enrol}}

#* Abstract

**Problem.** Predicting the pressure and the lift of an airfoil normally needs a computational fluid dynamics (CFD) solve for every shape and flow condition. Physics-informed neural networks (PINNs) learn a solution from the governing equations alone, with no labelled data. AeroNet builds on NNfoil [1], an open-source PINN for inviscid subsonic flow, and asks whether it can be verified, trained on a laptop CPU without a GPU, and shown honestly next to classical methods.

**Methodology.** We checked NNfoil line by line against the Euler equations (eleven numbered findings, each with a test or a measured file), fixed what stopped it running on a CPU, and validated every run against a bundled 800x400 finite-volume (FVM) reference with one pressure-coefficient (Cp) error metric. A FastAPI and React application lets a user draw any 2D shape and see a panel-method flow solution, and shows the trained PINN results in a PINN Lab.

**Key results** (NACA 0012, Mach 0.4, 4 degrees, one laptop CPU, all measured). On the same mesh the PINN is more accurate than the FVM: Cp RMS {{E2|pinn_gs_nopos|100x50|rms_cp}} at 100x50 against {{E2|fvm|100x50|rms_cp}} for the FVM, and {{E2|pinn_gs_nopos|200x100|rms_cp}} at 200x100 against {{E2|fvm|200x100|rms_cp}}. Training is not stable on the sparse 50x25 mesh: accuracy peaks at rounds 3 to 5 and then drifts in 5 of 6 runs while the loss keeps falling. A reference-free stop rule failed on held-out runs, one cold start collapsed, and the PINN is not faster than a panel method once training is counted.

**Societal relevance.** Cheap, transparent flow estimates help students and small teams explore designs without a GPU or a CFD licence, and the report shows how to label measured, quoted and fitted numbers so that results can be trusted.

#* Table of Contents

\toc

#* List of Figures

\listoffigures

#* List of Tables

\listoftables

#* List of Abbreviations

| Abbreviation | Meaning |
|---|---|
| CFD | Computational fluid dynamics |
| Cp | Pressure coefficient, (p - p_inf) / (0.5 rho_inf V_inf^2) |
| Cl, Cd | Lift and drag coefficients |
| CPU, GPU | Central and graphics processing unit |
| FVM | Finite-volume method (the reference solver for the bundled data) |
| NNfoil | The PINN Euler solver of Cao, Song and Zhang [1] |
| PINN | Physics-informed neural network |
| RMS | Root mean square (here of the Cp error) |
| Ma | Mach number |
| PG | Prandtl-Glauert compressibility correction |
| L-BFGS | Limited-memory Broyden-Fletcher-Goldfarb-Shanno optimiser |
| API | Application programming interface |
| WBS | Work breakdown structure |
| PO, PSO | Programme outcome, programme specific outcome |

\mainmatter

# Chapter 1: Introduction

## 1.1 Background

The lift and the drag of a wing, a drone rotor or a wind-turbine blade depend on the shape of its cross-section, the airfoil, and on the angle at which the air meets it. Engineers estimate these forces with computational fluid dynamics (CFD): the space around the airfoil is divided into a mesh and the flow equations are solved on it, most often with the finite-volume method (FVM), the method that produced the reference data bundled with NNfoil [1]. A CFD solve is accurate but must be repeated for every new shape, angle or Mach number, which makes early design exploration slow.

Physics-informed neural networks (PINNs) offer a different route [5], [6]. A network takes a point in space as input and returns the flow variables there; instead of learning from labelled examples, it is trained to make the residual of the governing equations and the boundary conditions small. No simulation data are needed, and the review literature describes PINNs as attractive for fluid problems [7], [8], although training them is known to be delicate [13] to [16]. Airfoils are a hard case: the flow accelerates sharply around the leading edge. NNfoil [1] meets this by mapping the curved mesh around the airfoil onto a uniform grid and training the network there, and its authors report a clear accuracy gain over a second-order FVM on very sparse meshes.

AeroNet takes NNfoil as its starting point. The released code assumes a GPU, fixes one flow condition per run, has no checks that its equations are implemented as intended, and offers results only as scripts. This project verifies the solver, runs it on a laptop CPU, measures it against the finite-volume data bundled with it, and presents it, next to a classical panel method, in an interactive application whose every number is labelled as measured, quoted or fitted.

## 1.2 Problem Definition and Objectives

**Problem.** There is no independently checked, easily reproducible and clearly presented version of the NNfoil solver. The technical challenge is to establish, with evidence, what the solver does correctly, what it gets wrong, how accurate and how costly it is on ordinary hardware, and where a simple classical method is as good.

**Objectives.** Five measurable goals were set. Chapter 5 (section 5.2) reports honestly which were met.

1. **O1, verification.** Check every component of NNfoil against the Euler equations and fix or document each defect; every finding needs a test or a measured data file.
2. **O2, CPU.** Make the solver train on a laptop CPU with no GPU, and measure the wall time per mesh.
3. **O3, accuracy.** Validate against the 800x400 FVM reference with one Cp error metric, and test the paper's claim that the PINN is more accurate than the FVM on the same sparse mesh (50x25, 100x50, 200x100).
4. **O4, application.** Build a demo that lets a user draw any 2D shape, see an animated flow field and the forces, and inspect the trained PINN results, with each number carrying its source label.
5. **O5, stopping.** Find a rule that stops training near the best answer without using the reference data.

## 1.3 Social and Environmental Relevance

A fast and transparent flow estimate helps students, hobbyists and small design teams who have neither a GPU nor a commercial CFD licence. The project runs entirely on a laptop CPU, offline once installed, and trains the networks it reports in minutes (Chapter 5), so the compute and energy it needs are small compared with GPU clusters (energy was not measured: see the Sustainability Statement in the annexure). The honest-labelling practice, where each number is measured, quoted from a paper or fitted, is itself a social contribution: it makes a machine-learning result checkable. The risk to avoid is false confidence: the application is inviscid and subsonic, never shows a car drag value, and states its limits on screen.

## 1.4 Project Plan

### 1.4.1 Work breakdown structure

Table: Work breakdown structure
| Work package | Content | Output |
|---|---|---|
| WP1 Literature and synopsis | Read NNfoil, Sun et al., NeuralFoil and the PINN literature; define scope | Synopsis, mid-term report |
| WP2 NNfoil verification (Milestone 1) | Line-by-line check against the Euler equations; CPU fixes; run folders; metrics; baselines | docs/verification.md, 11 findings, results/summary.csv |
| WP3 Demo v2 (Milestone 2) | Draw-any-shape panel solver, animated flow, polar, car with road | app/, docs/milestones/M2-demo-v2.md |
| WP4 PINN experiments (Milestone 3) | E1 to E6: positivity, mesh study, NACA 2412, speed, cylinder, stop rule | results/, docs/milestones/M3-pinn-results.md |
| WP5 PINN Lab in the app | Export of run folders, backend routes, PINN Lab, numbers dashboard | app/ PINN Lab, Results page |
| WP6 Final package | Report, slides, synopsis, offline test | deliverables/ |

### 1.4.2 Timeline

The dates below are the dates of the commits in the project's git history (the work before September 2026 is the synopsis and the first prototype).

Table: Timeline from the git history
| Date | Event |
|---|---|
| 2026-09-25 | First draft and the first React and Python prototype committed (panel-method stand-in) |
| 2026-11-05 | NNfoil vendored and fixed; run folders, metrics and baselines; grid sequencing; Milestone 1 runs |
| 2026-11-06 | Seed check; Milestone 1 complete |
| 2026-11-08 | Milestone 2 (demo v2) pack complete |
| 2026-11-14 | Milestone 3 research half complete (E1 to E6); features merged into the main branch; PINN Lab and Results page |

### 1.4.3 Roles and responsibilities

The roles of the team members are given in the annexure (Team Roles). {{field:roles_summary}} The study of the PINN method, the design of the experiments and the evaluation of the results were done by all three members together.

### 1.4.4 Challenges and risk management

Table: Risks and what was done about them
| Risk | What was done |
|---|---|
| Training on 100x50 or 200x100 is too slow | The first round was timed; a rule limited each run to a few hours; one run's wall time was lost when the laptop slept and is withheld |
| The positivity fix makes results worse | It was kept behind a flag and measured with 3 seeds per setting (experiment E1) |
| Flow conditions of the bundled data are unknown | The Mach numbers were read from the paper and labelled quoted, or labelled fitted; no comparison at a guessed condition |
| A drawn shape breaks the solver or the mesher | Input validation, a self-crossing check, a mesh fold check |
| A cold-start training run collapses to a trivial solution | Recorded as a result (Chapter 5), not hidden; grid sequencing used where possible |
| Timings spoilt by a busy or sleeping machine | Only idle, awake runs are quoted |
| The demo laptop fails on the day | An offline mode, a presenter mode and a backup video plan |

## 1.5 Scope and Limitations

In scope: two-dimensional, steady, inviscid, subsonic flow around a single closed body; NACA 0012, NACA 2412 and a cylinder for the PINN experiments; any drawn shape for the panel-method demo; CPU-only training and inference on one laptop.

Out of scope and stated as limits: viscous effects and separation (so no friction drag, and no car drag value); transonic flow with shocks [11]; three dimensions; training a network that covers a range of angles (each PINN here is one network per shape, Mach number and angle); training a PINN on a drawn shape from inside the application (planned, not done); a PINN at the 400x200 mesh. Most results are single seeds, and the report says where the seed spread is known.

# Chapter 2: Literature Survey

## 2.1 Critical Review

**PINN foundations.** Raissi, Perdikaris and Karniadakis [5] introduced the PINN framework for forward and inverse problems with nonlinear partial differential equations, and Karniadakis et al. [6] place it within physics-informed machine learning more broadly. Reviews of PINNs for fluid mechanics [7] and of the field in general [8] collect the successes and the open problems; software such as DeepXDE [17] and applications such as NSFnets for incompressible flow [9] and Reynolds-averaged flow [10] show the range. A common thread in the training-difficulty papers is that PINN losses are hard to optimise: the loss terms can have very different gradients [13], the terms converge at different rates [14], failure can occur even when the network is expressive enough [15], and the loss landscape is ill-conditioned in ways that favour second-order optimisers such as L-BFGS [16]. We read these as context only; we did not test their explanations (they are summarised from the abstracts, see the references note).

**NNfoil and its family.** Cao, Song and Zhang [1] solve the steady Euler equations for subsonic flow around airfoils with a PINN. Their key step is a mesh transformation: the curved O-grid around the body is mapped onto a uniform computational grid whose indices are the network inputs, which removes the steep leading-edge gradient from the learning problem. They compare with a second-order FVM on meshes from 50x25 to 400x200 and quote NNfoil training times of a few hundred seconds. A companion paper [2] treats high-dimensional parametric problems (a range of angles and shapes) for the same inviscid flows. Wassing, Langer and Bekemeyer [11] study PINNs for inviscid transonic flow around an airfoil, where shocks need extra care, and Sun, Sengupta and Juniper [3] use a PINN as a surrogate with the design variables as inputs and couple it with shape optimisation.

**Data-driven airfoil tools.** NeuralFoil [4] is a fast, physics-informed machine-learning tool for viscous airfoil analysis; it answers a different question (viscous flow at a Reynolds number) and needs no per-case training. Machine learning for aerodynamic shape optimisation is surveyed in [12].

**Classical methods.** The Hess-Smith panel method [22] solves the linear potential flow around arbitrary bodies and is covered in standard texts [23], [24]; with the Prandtl-Glauert rule it gives a quick subsonic estimate. XFOIL [25] couples a panel method with a boundary-layer model for viscous analysis. FVM produced the reference data bundled with NNfoil [1]. Mesh generation by elliptic equations goes back to Winslow [20] and is treated in [21]; the CST airfoil parametrisation is in [26]; the optimiser used to train NNfoil is L-BFGS [27].

## 2.2 Summary Table

Table: Summary of previous work and classical methods (the last column is the limitation that matters for this project)
| Work or method | Methodology | Data and cost | Limitation for this project |
|---|---|---|---|
| NNfoil, Cao et al. [1] | PINN on the steady Euler equations, mesh transformation to a uniform grid | No labelled data; training of a few hundred seconds quoted from the paper (GPU-style code) | One network per flow condition; no licence; released code assumes a GPU and has hard-coded settings |
| Cao et al. [2] | PINN for high-dimensional parametric inviscid problems | Parametric network trained over a range of conditions | Not reproduced here (no code run); heavier training |
| Sun et al. [3] | PINN surrogate with design variables as inputs, with PDE-constrained shape optimisation | Cost to be confirmed from the paper (the team's synopsis called it slow to train) | Not an Euler-on-uniform-grid method; not reproduced |
| Wassing et al. [11] | PINN for transonic inviscid flow with artificial dissipation | Per the abstract; not reproduced | Needs shock handling that NNfoil lacks |
| NeuralFoil [4] | Machine-learning surrogate for viscous airfoil analysis | No per-case training; {{E4|neuralfoil|large|infer_ms}} per call, measured here | Solves a different problem (viscous); not comparable in accuracy |
| Panel method [22], [23] | Hess-Smith linear potential flow plus Prandtl-Glauert | No training; {{E4|panel|140 panels|infer_ms}} per solve, measured here | Inviscid and linear; weaker on thick bodies (cylinder) |
| XFOIL [25] | Panel method plus boundary layer | Not run in this project | Viscous model; no time or accuracy measured here |
| FVM (reference data of [1]) | Finite-volume Euler or Navier-Stokes solver | {{E4|fvm|50x25|infer_ms:0}} to {{E4|fvm|400x200|infer_ms:0}} per solve, quoted from [1] (another machine) | Accurate only on fine meshes; cost grows with the mesh |

## 2.3 Identification of Research Gaps

1. **Independent verification.** NNfoil's code had not been checked against its own equations by a third party; AeroNet's contribution is a documented, tested component-by-component check.
2. **CPU practicality.** The solver was written for a GPU; whether it is usable on a laptop, and at what cost per mesh, was not measured.
3. **Honest accuracy at equal mesh.** The paper's headline (better than FVM on sparse meshes) had not been repeated with several seeds, with the reference metric fixed in advance, and with the failures reported.
4. **Training stability and stopping.** How many rounds to train, and how to stop without the reference solution, is not addressed; the loss turned out not to be a usable signal (Chapter 5).
5. **Presentation.** There was no interactive way to compare a PINN with a classical solver, with each number labelled by its source.

# Chapter 3: System Design and Methodology

## 3.1 Design Considerations

Table: Design aspects and how each was handled
| Aspect | Consideration | Design response |
|---|---|---|
| Economic | A student team has one laptop and no budget for GPUs or licences | CPU-only PyTorch, free open-source software, small networks (5 layers of 128 units), runs of minutes |
| Environmental | Compute energy | No GPU, 8 of 16 CPU threads, short runs; exact energy was not measured |
| Social | Users range from students to a panel of examiners | A plain web page, offline once installed, phone-width layout, a reduced-motion mode, light and dark themes |
| Political and legal | NNfoil has no licence | The repository is private, NNfoil is credited by name in the README, the app footer and this report; nothing is pushed publicly |
| Ethical | A flashy flow picture can be mistaken for a validated prediction | Every number is labelled measured, quoted or fitted; failures are reported; the app never shows a car drag value; idealised flow is marked on screen |
| Health and safety | Software only; no physical hazard. Cyber risks are small but real | Input validation, no user data collected, no path traversal (section 4.4) |
| Sustainability | Results must stay reproducible and not go stale | One results table (summary.csv) feeds the app, the figures and this report; documents are regenerated by script; a missing number is a hard error |

## 3.2 Methodology

Figure 3.1 shows the system. NNfoil, the PINN solver, is trained on a mesh (bundled with it or generated by the project's O-grid generator) and writes one run folder per case: the settings, a log of loss and lift per round, the wall Cp after every round, the flow field and the weights. An export script turns each run folder into a small JSON case (at most 3 MB) and the numbers into one results table. The web application has no PyTorch: its FastAPI backend solves drawn shapes with a panel method and serves the exported PINN cases; the React front end shows them. The same results table feeds the documents.

![System architecture: from the PINN training and the reference data to the web application and the documents](docs/deliverables_src/img/system_architecture.png)

The training procedure (Figure 3.2) is a loop of rounds. In each round the L-BFGS optimiser runs up to 1000 iterations on the loss of section 3.3; if the round fails (a NaN), the optimiser restarts; after each good round the wall Cp, the lift and the loss are written to the run folder. Training can start cold or from the weights of a coarser mesh (grid sequencing, our addition, not in [1]). The practice that the experiments support is a fixed early stop at round 3 to 5 (section 5.1.7).

![Training flowchart for one PINN case](docs/deliverables_src/img/training_flowchart.png)

The draw-a-shape path (Figure 3.3) does not use a neural network. The outline is cleaned, checked and solved with the panel method; the browser draws the result with a colour map, isobars and animated particles.

![Flowchart of the draw-a-shape path](docs/deliverables_src/img/draw_flowchart.png)

**Algorithm (training one case with grid sequencing).**

1. Choose the shape, the Mach number and the angle of attack; load or build the O-grid mesh and check that no cell is folded.
2. Train the 5 x 128 network on the coarse mesh (50x25) with L-BFGS rounds; stop at a fixed round.
3. Move the weights to the fine mesh (100x50 or 200x100), rescaling the input normalisation.
4. Continue with a few rounds on the fine mesh; after every round write Cp, Cl, Cd and the loss.
5. Compare the final round with the 800x400 FVM reference using the Cp RMS metric of section 3.3, and export the run.

## 3.3 Modelling

**Governing equations.** NNfoil solves the steady two-dimensional Euler equations for a perfect gas (γ = 1.4). With velocity (u, v), pressure p, density ρ and a² = γ p / ρ, the four residuals it drives to zero, in the non-conservative form implemented in the code and checked against the conservative form in `docs/verification.md`, are:

$$ e1 = ρ u_{x} + u ρ_{x} + ρ v_{y} + v ρ_{y} = 0 $$

$$ e2 = u u_{x} + v u_{y} + p_{x} / ρ = 0 $$

$$ e3 = u v_{x} + v v_{y} + p_{y} / ρ = 0 $$

$$ e4 = ρ a² (u_{x} + v_{y}) + u p_{x} + v p_{y} = 0 $$

Lengths are scaled by the chord, density and speed by their free-stream values, so the free-stream pressure is p∞ = 1 / (γ Ma²). The pressure coefficient and the forces are:

$$ Cp = (p − p∞) / 0.5 $$

$$ F = −(1/c) ∑ Cp n dl,   Cl = F · (−sin α, cos α),   Cd = F · (cos α, sin α) $$

The true inviscid subcritical drag is zero (d'Alembert), so any Cd printed for an Euler or a panel solution is numerical error.

**Mesh transformation.** The network input is the mesh index pair (ξ, η) of an O-grid with the wall at η = 0 and the far field (15 chords away) at the last eta row. Derivatives with respect to (x, y) come from the metric Jacobian by the chain rule. The grid itself satisfies the Winslow equation α x_{ξξ} − 2β x_{ξη} + γ x_{ηη} = 0 [20], [21]; the bundled generator uses a factor 0.5 instead of 2 on the cross term (finding 3 of Chapter 4; low impact, both grids valid).

**Loss.** NNfoil minimises

$$ L = 2×10⁴ · Msef + (mseb1 + mseb2 + mseb3) $$

where Msef is the Jacobian-weighted mean squared PDE residual (the sum of |R|² J² over the sum of J²), mseb1 is the far-field Dirichlet term, mseb2 the zero-normal-velocity wall term (penalty), and mseb3 the periodic seam term. No Kutta condition is imposed, which is the suspected reason why the loss and the accuracy can move in opposite directions (section 5.1.2).

**Positivity option (our change).** With the flag on, p and ρ are the softplus of the network outputs plus 1e-3, so the divisions by ρ and the square root in a never see a non-positive value.

**Baselines.** The panel method (Hess-Smith sources and vortices with the Kutta condition [22]) gives the incompressible pressure Cp0, and the Prandtl-Glauert rule scales it by 1/β with β = √(1 − Ma²) [23], [24]. The thin-airfoil lift with the same correction is a second baseline. The incompressible circle has the exact Cp = 1 − 4 sin²θ.

**Error metric (one for every method).** For every reference point with 0.02 <= x/c <= 0.98 take the nearest predicted surface point and compute the root mean square of the Cp difference: this is the Cp RMS. The reference is the 800x400 FVM file; the lift of that file is integrated from its Cp after re-ordering the points along the surface (the files are not stored in surface order; an unordered integral gives the wrong lift).

## 3.4 Tool and Technique Selection

Table: Why each tool was chosen
| Choice | Reason | Alternative rejected |
|---|---|---|
| PyTorch on CPU [18] | NNfoil is written in PyTorch; the CPU build installs without CUDA and runs the 5 x 128 network at the sizes used | A GPU: not available; would hide the cost a student team can pay |
| A separate research environment | The web app stays light; only the research scripts need torch | One shared environment (a 200 MB install for the demo) |
| FastAPI and pydantic | Typed requests with limits, automatic validation errors, easy tests | A hand-written server |
| React with plain Canvas 2D | No heavy libraries; works offline; thousands of animated particles from one JSON field | WebGL or charting libraries |
| Panel method as the baseline and the drawn-shape solver | Milliseconds per solve, no training, exact checks available (circle, road) | A Navier-Stokes solver: far out of scope |
| NNfoil's own FVM reference files | Independent of our code; the finest mesh (800x400) is the closest thing to truth we have | Producing our own reference: out of scope |
| python-docx and python-pptx with placeholders | A number is looked up in results/summary.csv; a missing number stops the build | Typing numbers into the report by hand |
| Vendoring NNfoil with one commit per fix | git log is the bug-fix evidence | A submodule or a copy without history |

# Chapter 4: Implementation and Testing

## 4.1 Development Details

The work followed the three milestones of Chapter 1.

**Step 1, NNfoil verified (Milestone 1).** The upstream repository was added as its own first commit and each fix became one commit. Table 4.1 lists the findings; the full verification table is `docs/verification.md`.

Table: Findings of the NNfoil review (file and line refer to the upstream code)
| # | Problem | Verdict |
|---|---|---|
| 1 | Device forced to CUDA | Fixed: device picked at run time, CPU fallback |
| 2 | p and rho unconstrained but divided by | Fixed as an option (positivity flag); the measured effect on accuracy is in Chapter 5 |
| 3 | Cross term in the mesh generator 4 times too small | Documented, not changed (shipped meshes and FVM data use it); both grids valid |
| 4 | Only one CPU thread | Fixed: up to 8 threads |
| 5 | np.math removed in NumPy 2 | Fixed |
| 6 | Whole-module torch.load fails in recent PyTorch; shared checkpoint name | Fixed: state_dict and one path per run |
| 7 | Force routine never called, so no Cl or Cd | Fixed: evaluate_wall() |
| 8 | Mach 0.4 and 4 degrees hard-coded | Fixed: constructor arguments (still one condition per model) |
| 9 | Training cost (L-BFGS, 1000 iterations per round) | Limitation; measured in Chapter 5 |
| 10 | One-sided Jacobian at the trailing-edge seam: 1 or 2 wrong-sign cells on all 9 bundled meshes | Documented, opt-in flag; default unchanged |
| 11 | Weight initialisation line is overwritten by weight normalisation | Documented, not changed |

Four further components were checked and found correct: the four Euler residuals (against the conservative form to below 1e-9 on a manufactured field), the wall mapping and outward normals, the Cp formula and the force integration. The Kutta condition, the parametric angle of attack, the far-field boundary and the loss weights are limitations of the method, not defects.

**Step 2, one run folder per case.** A command-line tool trains one case and rewrites the run folder after every round, so a run killed after two hours still has its results; fixed run names (for example `naca0012_100x50_a4_m04`) let every script find a run without a database.

**Step 3, the demo (Milestone 2).** The panel solver was generalised to any closed outline with an optional road (method of images). The front end gained a drawing pad, a live flow canvas, a polar and pin-and-compare. Figures 4.1 to 4.6 are screenshots.

![The landing page with the live flow hero (desktop)](docs/milestones/img/b9_hero_1440.png)

![Explore: pressure colour map, isobars and animated particles around an airfoil](docs/milestones/img/b9_explore_canvas_1440.png)

![Draw: a hand-drawn blob solved when the pen is lifted](docs/milestones/img/b8_draw_blob_1440.png)

![Draw: a sedan with a road under it (ground effect); the amber note says the flow is idealised and no drag is shown](docs/milestones/img/b8_draw_sedan_road_1440.png)

![Pressure coefficient curve and the lift polar card](docs/milestones/img/b9_cp_polar_1440.png)

![Pin and compare: a ghost of the earlier result and the change in lift](docs/milestones/img/b9_explore_pinned_1440.png)

**Step 4, PINN results in the app (Milestone 3).** Each trained run is exported to one case file; the application's PINN Lab shows 14 exported trained cases with the PINN, FVM and panel Cp curves, the error numbers, the training history, a replay slider that shows the network learning round by round, and a numbers dashboard. A Results page shows the headline numbers and the finding charts.

![PINN Lab: replay of the 50x25 run, showing the drift after the early rounds](docs/milestones/img/c8_replay_drift_50x25_1440.png)

![PINN Lab: the numbers dashboard, each number with its source label](docs/milestones/img/c8_numbers_dashboard_1440.png)

![PINN Lab: the NACA 2412 four-degree run that collapsed to the trivial solution, shown as such](docs/milestones/img/c8_collapsed_naca2412_1440.png)

![The speed table in the application: measured and quoted times, with the not-comparable note](docs/milestones/img/c8_speed_table_1440.png)

Not done: training a PINN on a drawn shape from inside the application, and the ablations that might repair the drift (robust loop, periodic input, hard far-field input, panel-method prior).

## 4.2 Experimental Setup

Table: Hardware and software (measured, from results/e4_speed.md)
| Item | Value |
|---|---|
| CPU | AMD Ryzen 9 6900HS with Radeon Graphics, 8 cores and 16 logical threads |
| Operating system | Windows 11 |
| Python, torch, numpy | Python 3.13.7, torch 2.15.0+cpu, numpy 2.5.4 |
| NeuralFoil (comparison only) | 0.3.3 |
| PINN | 5 layers of 128 units, L-BFGS, up to 1000 iterations per round, 8 torch threads, float32 |
| Case studied | NACA 0012, Mach 0.4, 4 degrees (the condition of the bundled FVM files, quoted from [1]) |
| Meshes | O-grids 50x25, 100x50, 200x100 (and 400x200 for the FVM only); far field at 15 chords |
| Reference | FVM 800x400 file, one Cp error metric (section 3.3) |
| Timing rule | Idle, awake CPU, one run at a time; inference timed as the mean of 100 calls after 10 warm-up calls |
| GPU | None |

## 4.3 Test Cases

The project has four test suites. Research tests: 218 passed (`OMP_NUM_THREADS=1 research/.venv/Scripts/python -m pytest research/tests -q`, run for this report; it includes the tests of the document builder). Front-end tests: 106 passed (`npm test`, run for this report). Back-end tests: 153 passed and 2 skipped (NeuralFoil is optional), as recorded in `docs/HANDOFF.md` on 2026-11-14; the back-end suite was not re-run for this report because its environment is not on the machine that built it.

Table: Input against expected output for the main checks (measured unless marked)
| Check | Input | Expected | Observed |
|---|---|---|---|
| Circle, non-lifting panel solve | Circle, 0 degrees | Cp max 1, Cp min -3 (exact potential flow) | Cp max 0.9989, Cp min -2.9989; Cl and Cd about 1e-16 (M2) |
| Reverse-ordered outline | NACA 0012 outline listed in reverse, 4 degrees, Mach 0.4 | Same lift as the forward outline | Cl 0.5247 against 0.5263 (M2) |
| Ground effect, road far away | Road at a large gap | Same lift as free air | Cl 0.4821 against 0.4822 (M2) |
| No flow through the road | Speed on the road line | Exactly 0 | Exactly 0 (M2) |
| Freestream residual | Uniform freestream network | PDE residual 0, far-field term 0, non-zero wall term | As expected (test_freestream_has_zero_residual_and_forces) |
| Cp definition | p - p_far = 0.5 | Cp = 1 (stagnation) | Cp = 1 within 1e-5 (test_cp_formula_is_pressure_coefficient) |
| Force convention | Panel Cp pushed through the force routine | Panel Cl and Cd to 1e-9 | Matches (test_panel_cp_through_forces_matches_panel_cl_cd) |
| Reference lift | FVM 800x400 Cp file, re-ordered | Cl {{E2|fvm|50x25|cl_ref:4}} | Within 5e-4 (test_fvm_forces_cl) |
| Panel against the finest FVM | NACA 0012, Mach 0.4, 4 degrees | Cp RMS about 0.0099 | {{B0|panel|-|rms_cp|NACA0012_a4}} (200 panels) |
| Euler residual identities | Manufactured smooth field | e1 to e4 equal the conservative residuals | Equal below 1e-9 (test_e1_e4_equal_conservative_euler_residuals) |
| Positivity | Any network output | p and rho at least 1e-3 | Holds (test_positivity_keeps_p_rho_above_floor) |
| Checkpoint round trip | Save then load | Identical output | Identical (test_checkpoint_roundtrip) |
| Self-crossing outline | A bow-tie outline posted to the API | Refused with a clear message | HTTP 422, "The outline crosses itself" |
| Unknown PINN case id | A path-like id | Not found | HTTP 404 (section 4.4) |

## 4.4 Safety Measures

There is no physical hazard: the project is software run on a laptop. The cyber-security and data-safety measures followed during the build are:

- **Offline-capable and no data collection.** The application has no accounts, no analytics and no third-party calls; fonts are self-hosted; a drawn shape is sent only to the local server.
- **Licence safety.** NNfoil has no licence, so the repository is private and nothing has been pushed; NNfoil is credited wherever it is used.
- **Input validation.** Requests are checked by pydantic limits (angle of attack within 20 degrees, Mach number between 0 and 0.7, at least 8 and at most 4000 outline points, bounded grid sizes); a self-crossing outline returns 422; a request for a solver that is not installed returns 409; a missing solver or case returns 404; PINN results that cannot be read return 503.
- **No path traversal.** A PINN case id must match lowercase letters, digits and underscores only and must be the name of a case file that exists, so no file path can be reached from an id.
- **Restricted cross-origin access.** Only the local development origins are allowed.
- **Safe model files.** Weights are saved as state dictionaries (not whole pickled modules), which loads under the stricter default of current PyTorch.
- **Machine safety.** Training uses 8 of 16 threads so that the laptop stays usable and cool; timed runs are repeated if the machine slept.

# Chapter 5: Results and Discussion

All results are for NACA 0012 at Mach 0.4 and 4 degrees unless a section says otherwise, on one laptop CPU (section 4.2). Every number in the tables is filled from `results/summary.csv` (or the two result files named in the builder) by the placeholder mechanism, so a missing number stops the build. Each row carries its source label: **measured** (produced by code in this repository on this laptop), **quoted** (copied from a paper, not re-run) or **fitted** (tuned to data). The reference is always the FVM 800x400 file, with Cl {{E2|fvm|50x25|cl_ref:4}}; the error metric is the Cp RMS of section 3.3. PINN rows are the final round of each run unless stated.

## 5.1 Result Analysis

### 5.1.1 Accuracy against the FVM on the same mesh (experiment E2)

Table: Headline results, NACA 0012, Mach 0.4, 4 degrees (the reference is FVM 800x400; PINN rows are the final round, seed 0)
| Method | Mesh | Cl | Cl error | Cp RMS | Cp max error | Training time | Source |
|---|---|---|---|---|---|---|---|
| FVM | 50x25 | {{E2|fvm|50x25|cl:4}} | {{E2|fvm|50x25|cl_err_pct}} | {{E2|fvm|50x25|rms_cp}} | {{E2|fvm|50x25|max_cp:3}} | - | {{E2|fvm|50x25|source}} |
| FVM | 100x50 | {{E2|fvm|100x50|cl:4}} | {{E2|fvm|100x50|cl_err_pct}} | {{E2|fvm|100x50|rms_cp}} | {{E2|fvm|100x50|max_cp:3}} | - | {{E2|fvm|100x50|source}} |
| FVM | 200x100 | {{E2|fvm|200x100|cl:4}} | {{E2|fvm|200x100|cl_err_pct}} | {{E2|fvm|200x100|rms_cp}} | {{E2|fvm|200x100|max_cp:3}} | - | {{E2|fvm|200x100|source}} |
| FVM | 400x200 | {{E2|fvm|400x200|cl:4}} | {{E2|fvm|400x200|cl_err_pct}} | {{E2|fvm|400x200|rms_cp}} | {{E2|fvm|400x200|max_cp:3}} | - | {{E2|fvm|400x200|source}} |
| Panel method (Hess-Smith with Prandtl-Glauert) | 140 panels | {{E2|panel|140 panels|cl:4}} | {{E2|panel|140 panels|cl_err_pct}} | {{E2|panel|140 panels|rms_cp}} | {{E2|panel|140 panels|max_cp:3}} | - | {{E2|panel|140 panels|source}} |
| Panel method | 200 panels | {{B0|panel|-|cl:4|NACA0012_a4}} | {{B0|panel|-|cl_err_pct|NACA0012_a4}} | {{B0|panel|-|rms_cp|NACA0012_a4}} | {{B0|panel|-|max_cp:3|NACA0012_a4}} | - | {{B0|panel|-|source|NACA0012_a4}} |
| Thin-airfoil theory (PG corrected) | - | {{B0|thin_airfoil|-|cl:4|NACA0012_a4}} | {{B0|thin_airfoil|-|cl_err_pct|NACA0012_a4}} | - | - | - | {{B0|thin_airfoil|-|source|NACA0012_a4}} |
| PINN, positivity on | 50x25 | {{E2|pinn|50x25|cl:4}} | {{E2|pinn|50x25|cl_err_pct}} | {{E2|pinn|50x25|rms_cp}} | {{E2|pinn|50x25|max_cp:3}} | {{E2|pinn|50x25|train_s}} | {{E2|pinn|50x25|source}} |
| PINN, positivity off | 50x25 | {{E2|pinn_nopos|50x25|cl:4}} | {{E2|pinn_nopos|50x25|cl_err_pct}} | {{E2|pinn_nopos|50x25|rms_cp}} | {{E2|pinn_nopos|50x25|max_cp:3}} | {{E2|pinn_nopos|50x25|train_s}} | {{E2|pinn_nopos|50x25|source}} |
| PINN cold start, positivity on | 100x50 | {{E2|pinn|100x50|cl:4}} | {{E2|pinn|100x50|cl_err_pct}} | {{E2|pinn|100x50|rms_cp}} | {{E2|pinn|100x50|max_cp:3}} | {{E2|pinn|100x50|train_s}} | {{E2|pinn|100x50|source}} |
| PINN grid-sequenced, positivity on | 100x50 | {{E2|pinn_gs|100x50|cl:4}} | {{E2|pinn_gs|100x50|cl_err_pct}} | {{E2|pinn_gs|100x50|rms_cp}} | {{E2|pinn_gs|100x50|max_cp:3}} | {{E2|pinn_gs|100x50|train_s}}, this stage only | {{E2|pinn_gs|100x50|source}} |
| PINN grid-sequenced, positivity off | 100x50 | {{E2|pinn_gs_nopos|100x50|cl:4}} | {{E2|pinn_gs_nopos|100x50|cl_err_pct}} | {{E2|pinn_gs_nopos|100x50|rms_cp}} | {{E2|pinn_gs_nopos|100x50|max_cp:3}} | {{E2|pinn_gs_nopos|100x50|train_s}}, this stage only | {{E2|pinn_gs_nopos|100x50|source}} |
| PINN grid-sequenced, positivity off | 200x100 | {{E2|pinn_gs_nopos|200x100|cl:4}} | {{E2|pinn_gs_nopos|200x100|cl_err_pct}} | {{E2|pinn_gs_nopos|200x100|rms_cp}} | {{E2|pinn_gs_nopos|200x100|max_cp:3}} | withheld (the laptop slept) | {{E2|pinn_gs_nopos|200x100|source}} |

Grid-sequenced training times count only their own stage and exclude the parent run that supplied the weights. The 200x100 run's training time is withheld: the laptop slept during it, so its wall time is not usable. There is no PINN at 400x200. The panel method appears twice because the two rows use different numbers of panels; both are measured.

![Cp RMS against the 800x400 FVM for each mesh: the FVM on its own mesh (bars) and the PINN runs (markers); log scale](results/figures/m3_e2_mesh_study.png)

![NACA 0012 surface Cp on the 100x50 mesh: the FVM 800x400 reference, the 100x50 FVM and two PINN runs](results/figures/m1_cp_100x50.png)

**Reading.** On every mesh we trained, the PINN's Cp RMS is lower than the FVM's on the same mesh: at 100x50 it is {{E2|pinn_gs_nopos|100x50|rms_cp}} for the PINN against {{E2|fvm|100x50|rms_cp}} for the FVM, about three times lower, and at 200x100 {{E2|pinn_gs_nopos|200x100|rms_cp}} against {{E2|fvm|200x100|rms_cp}}, about half. The PINN is not imitating the coarse FVM: its Cp differs from the same-mesh FVM by about the FVM's own distance to the reference. Against the panel method ({{E2|panel|140 panels|rms_cp}} with 140 panels) the PINN is about as accurate at 100x50 and better only at 200x100, so on this smooth case accuracy alone is not a reason to prefer the PINN; what it offers is that it solves the Euler equations without the Prandtl-Glauert assumption (our interpretation, not tested). The 50x25 PINN with positivity on is not better than the FVM in lift (Cl {{E2|pinn|50x25|cl:4}} against {{E2|fvm|50x25|cl:4}}), although better in Cp. Each cell is a single seed: this is a measured result, not yet a statistical one.

### 5.1.2 Training is not stable on the sparse mesh (experiment E1)

Table: E1, NACA 0012 on the 50x25 mesh, three seeds with the positivity option on and off, up to 20 rounds each
| Positivity | Seed | Good rounds (restarts) | Cp RMS round 3 | Round 4 | Round 5 | Final round | Cl final | Training time |
|---|---|---|---|---|---|---|---|---|
| on | 0 | {{E1|pinn|50x25|note:rounds|NACA0012_a4_s0}} ({{E1|pinn|50x25|note:restarts|NACA0012_a4_s0}}) | {{E1R|pinn|3|cp_rms:3|NACA0012_a4_s0}} | {{E1R|pinn|4|cp_rms:3|NACA0012_a4_s0}} | {{E1R|pinn|5|cp_rms:3|NACA0012_a4_s0}} | {{E1|pinn|50x25|rms_cp:3|NACA0012_a4_s0}} | {{E1|pinn|50x25|cl:3|NACA0012_a4_s0}} | {{E1|pinn|50x25|train_s|NACA0012_a4_s0}} |
| on | 1 | {{E1|pinn|50x25|note:rounds|NACA0012_a4_s1}} ({{E1|pinn|50x25|note:restarts|NACA0012_a4_s1}}) | {{E1R|pinn|3|cp_rms:3|NACA0012_a4_s1}} | {{E1R|pinn|4|cp_rms:3|NACA0012_a4_s1}} | {{E1R|pinn|5|cp_rms:3|NACA0012_a4_s1}} | {{E1|pinn|50x25|rms_cp:3|NACA0012_a4_s1}} | {{E1|pinn|50x25|cl:3|NACA0012_a4_s1}} | {{E1|pinn|50x25|train_s|NACA0012_a4_s1}} |
| on | 2 | {{E1|pinn|50x25|note:rounds|NACA0012_a4_s2}} ({{E1|pinn|50x25|note:restarts|NACA0012_a4_s2}}) | {{E1R|pinn|3|cp_rms:3|NACA0012_a4_s2}} | {{E1R|pinn|4|cp_rms:3|NACA0012_a4_s2}} | {{E1R|pinn|5|cp_rms:3|NACA0012_a4_s2}} | {{E1|pinn|50x25|rms_cp:3|NACA0012_a4_s2}} | {{E1|pinn|50x25|cl:3|NACA0012_a4_s2}} | {{E1|pinn|50x25|train_s|NACA0012_a4_s2}} |
| off | 0 | {{E1|pinn_nopos|50x25|note:rounds|NACA0012_a4_s0}} ({{E1|pinn_nopos|50x25|note:restarts|NACA0012_a4_s0}}) | {{E1R|pinn_nopos|3|cp_rms:3|NACA0012_a4_s0}} | {{E1R|pinn_nopos|4|cp_rms:3|NACA0012_a4_s0}} | {{E1R|pinn_nopos|5|cp_rms:3|NACA0012_a4_s0}} | {{E1|pinn_nopos|50x25|rms_cp:3|NACA0012_a4_s0}} | {{E1|pinn_nopos|50x25|cl:3|NACA0012_a4_s0}} | {{E1|pinn_nopos|50x25|train_s|NACA0012_a4_s0}} |
| off | 1 | {{E1|pinn_nopos|50x25|note:rounds|NACA0012_a4_s1}} ({{E1|pinn_nopos|50x25|note:restarts|NACA0012_a4_s1}}) | {{E1R|pinn_nopos|3|cp_rms:3|NACA0012_a4_s1}} | {{E1R|pinn_nopos|4|cp_rms:3|NACA0012_a4_s1}} | {{E1R|pinn_nopos|5|cp_rms:3|NACA0012_a4_s1}} | {{E1|pinn_nopos|50x25|rms_cp:3|NACA0012_a4_s1}} | {{E1|pinn_nopos|50x25|cl:3|NACA0012_a4_s1}} | {{E1|pinn_nopos|50x25|train_s|NACA0012_a4_s1}} |
| off | 2 | {{E1|pinn_nopos|50x25|note:rounds|NACA0012_a4_s2}} ({{E1|pinn_nopos|50x25|note:restarts|NACA0012_a4_s2}}) | {{E1R|pinn_nopos|3|cp_rms:3|NACA0012_a4_s2}} | {{E1R|pinn_nopos|4|cp_rms:3|NACA0012_a4_s2}} | {{E1R|pinn_nopos|5|cp_rms:3|NACA0012_a4_s2}} | {{E1|pinn_nopos|50x25|rms_cp:3|NACA0012_a4_s2}} | {{E1|pinn_nopos|50x25|cl:3|NACA0012_a4_s2}} | {{E1|pinn_nopos|50x25|train_s|NACA0012_a4_s2}} |

![Cp RMS and Cl after every round of the six E1 runs](results/figures/m3_e1_trajectories.png)

**Reading.** All six runs are accurate at round 3 (Cp RMS 0.027 to 0.032, about four times better than the FVM on its own 50x25 mesh, {{E2|fvm|50x25|rms_cp}}). Then five of the six degrade (the final error is more than twice the best round) while the training loss keeps falling: only one run, positivity off with seed 0, stayed good. Rounds 4 and 5 are not safe in every run (one run is at 0.101 in round 4 and 0.147 in round 5). The earlier belief that the positivity option hurts came from that one lucky seed and is withdrawn: neither setting wins (mean final Cp RMS 0.130 with it, 0.146 without it; 3 of 3 runs degrade with it and 2 of 3 without), and positivity only changes when the jump happens. Over the rounds, the rank correlation between the total loss and the Cp error is negative in the five degrading runs (-0.48 to -0.98, `results/drift_step1.csv`), so a lower loss goes with a worse answer: the Euler residual does not pin the circulation because no Kutta condition is imposed. That explanation is the suspected cause, not a tested one, and rounds inside one run are not independent samples, so the correlation describes the shape of each run, not a significance test.

### 5.1.3 Grid sequencing and the seed check (measured, `results/seed_check.csv`)

Grid sequencing trains on 50x25, moves the weights to 100x50 with rescaled input normalisation and continues for 4 rounds. It is our addition, not in [1]. Seeds 0 to 3, both positivity modes, final Cl against {{E2|fvm|50x25|cl_ref:4}} and Cp RMS against FVM 800x400:

Table: Seed check on the 100x50 mesh: grid sequencing against a cold start (Cl, with Cp RMS in brackets)
| Positivity | Method | Seed 0 | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|---|
| off | grid-sequenced | {{SEED|nopos_gs|0|final_cl}} ({{SEED|nopos_gs|0|cp_rms}}) | {{SEED|nopos_gs|1|final_cl}} ({{SEED|nopos_gs|1|cp_rms}}) | {{SEED|nopos_gs|2|final_cl}} ({{SEED|nopos_gs|2|cp_rms}}) | {{SEED|nopos_gs|3|final_cl}} ({{SEED|nopos_gs|3|cp_rms}}) |
| off | cold start | not run | {{SEED|nopos_cold|1|final_cl}} ({{SEED|nopos_cold|1|cp_rms}}), {{SEED|nopos_cold|1|failed_rounds}} failed round | {{SEED|nopos_cold|2|final_cl}} ({{SEED|nopos_cold|2|cp_rms}}), {{SEED|nopos_cold|2|failed_rounds}} failed round | **collapsed: all {{SEED|nopos_cold|3|failed_rounds}} rounds failed** |
| on | grid-sequenced | {{SEED|pos_gs|0|final_cl}} ({{SEED|pos_gs|0|cp_rms}}) | {{SEED|pos_gs|1|final_cl}} ({{SEED|pos_gs|1|cp_rms}}) | {{SEED|pos_gs|2|final_cl}} ({{SEED|pos_gs|2|cp_rms}}) | {{SEED|pos_gs|3|final_cl}} ({{SEED|pos_gs|3|cp_rms}}) |
| on | cold start | {{SEED|pos_cold|0|final_cl}} ({{SEED|pos_cold|0|cp_rms}}), {{SEED|pos_cold|0|rounds}} rounds | {{SEED|pos_cold|1|final_cl}} ({{SEED|pos_cold|1|cp_rms}}) | {{SEED|pos_cold|2|final_cl}} ({{SEED|pos_cold|2|cp_rms}}) | {{SEED|pos_cold|3|final_cl}} ({{SEED|pos_cold|3|cp_rms}}) |

![Seed check: final Cl and Cp RMS for grid-sequenced and cold-start runs](results/figures/m3_seed_check.png)

All eight grid-sequenced runs end with Cl between 0.50 and 0.52 (the pass line set in advance was Cl above 0.49). A cold 100x50 start without positivity collapsed in 1 of 3 seeds (seed 3: every round failed and no lift was produced); with positivity 0 of 4 collapsed. The collapse is real but unreliable, and the apparent protection by positivity rests on four runs; section 5.1.4 shows a cold start with positivity on can collapse too. No speed advantage is claimed for grid sequencing: its coarse stage costs a full 50x25 run, and wall times of the seed-check runs are not quoted because some were stalled by the machine.

### 5.1.4 A second airfoil: NACA 2412 (experiment E3)

The Mach number of the NACA 2412 files is not stored in them. We use 0.3, **quoted** from the paper's parametric NACA 2412 case [1]; our own panel-method fit preferred about 0.25 (**fitted**), within what a Prandtl-Glauert panel method can resolve. The PINN was trained on a 100x50 mesh built here from the bundled wall, from a cold start, for 10 attempted rounds, seed 0, positivity on, except where the row says otherwise. FVM, panel and thin-airfoil values are measured here; differences are prediction minus FVM.

Table: E3, NACA 2412 at Mach 0.3 (quoted): lift against the FVM 800x400 file
| Angle | FVM Cl | Panel Cl (diff) | Thin-airfoil Cl (diff) | PINN run | PINN Cl (diff) | PINN Cp RMS | Training time |
|---|---|---|---|---|---|---|---|
| -4 | {{E3|fvm|800x400|cl:4|NACA2412_a-4}} | {{E3|panel|140 panels|cl:4|NACA2412_a-4}} ({{E3|panel|140 panels|cl_diff|NACA2412_a-4}}) | {{E3|thin_airfoil|-|cl:4|NACA2412_a-4}} ({{E3|thin_airfoil|-|cl_diff|NACA2412_a-4}}) | seed 0, positivity on | {{E3|pinn|100x50|cl:4|NACA2412_a-4}} ({{E3|pinn|100x50|cl_diff|NACA2412_a-4}}) | {{E3|pinn|100x50|rms_cp|NACA2412_a-4}} | {{E3|pinn|100x50|train_s|NACA2412_a-4}} |
| 0 | {{E3|fvm|800x400|cl:4|NACA2412_a0}} | {{E3|panel|140 panels|cl:4|NACA2412_a0}} ({{E3|panel|140 panels|cl_diff|NACA2412_a0}}) | {{E3|thin_airfoil|-|cl:4|NACA2412_a0}} ({{E3|thin_airfoil|-|cl_diff|NACA2412_a0}}) | seed 0, positivity on | {{E3|pinn|100x50|cl:4|NACA2412_a0}} ({{E3|pinn|100x50|cl_diff|NACA2412_a0}}) | {{E3|pinn|100x50|rms_cp|NACA2412_a0}} | {{E3|pinn|100x50|train_s|NACA2412_a0}} |
| 4 | {{E3|fvm|800x400|cl:4|NACA2412_a4}} | {{E3|panel|140 panels|cl:4|NACA2412_a4}} ({{E3|panel|140 panels|cl_diff|NACA2412_a4}}) | {{E3|thin_airfoil|-|cl:4|NACA2412_a4}} ({{E3|thin_airfoil|-|cl_diff|NACA2412_a4}}) | seed 0, positivity on: **collapsed** (Cl about 0) | - | {{E3|pinn|100x50|rms_cp:1|NACA2412_a4}} (collapsed) | {{E3|pinn|100x50|train_s|NACA2412_a4}} |
| 4 | same | same | same | seed 1, positivity on (retry) | {{E3|pinn|100x50|cl:4|NACA2412_a4_s1}} ({{E3|pinn|100x50|cl_diff|NACA2412_a4_s1}}) | {{E3|pinn|100x50|rms_cp|NACA2412_a4_s1}} | {{E3|pinn|100x50|train_s|NACA2412_a4_s1}} |
| 4 | same | same | same | seed 0, positivity off (retry) | {{E3|pinn_nopos|100x50|cl:4|NACA2412_a4}} ({{E3|pinn_nopos|100x50|cl_diff|NACA2412_a4}}) | {{E3|pinn_nopos|100x50|rms_cp|NACA2412_a4}} | {{E3|pinn_nopos|100x50|train_s|NACA2412_a4}} |
| 8 | {{E3|fvm|800x400|cl:4|NACA2412_a8}} | {{E3|panel|140 panels|cl:4|NACA2412_a8}} ({{E3|panel|140 panels|cl_diff|NACA2412_a8}}) | {{E3|thin_airfoil|-|cl:4|NACA2412_a8}} ({{E3|thin_airfoil|-|cl_diff|NACA2412_a8}}) | seed 0, positivity on | {{E3|pinn|100x50|cl:4|NACA2412_a8}} ({{E3|pinn|100x50|cl_diff|NACA2412_a8}}) | {{E3|pinn|100x50|rms_cp|NACA2412_a8}} | {{E3|pinn|100x50|train_s|NACA2412_a8}} |

![NACA 2412 lift against angle of attack: FVM, panel method, thin-airfoil theory and the PINN runs](results/figures/m3_e3_polar.png)

The 4-degree seed-0 run **collapsed to the trivial solution** (lift about zero, 3 good rounds, 7 restarts, a Cp RMS of {{E3|pinn|100x50|rms_cp:1|NACA2412_a4}}). It is not an E1 verdict; it shows that a cold start with positivity on can collapse, which the NACA 0012 seed check (0 of 4) did not. The two retries were chosen because the first run collapsed, and both worked. Where the PINN worked, its lift is within 0.002 to 0.060 of the FVM (worst at 8 degrees) and the panel method is within 0.009 to 0.022 at the same angles, so the PINN is not better than the panel method on this case. Other angles, a parametric network (the paper trains one network over the angle range, 1543 s, quoted) and the random airfoils were not run, and the seed spread of E3 is unknown.

### 5.1.5 Where the simple model fails: the cylinder (experiment E5)

The cylinder is run at Mach 0.4 and 0 degrees, **quoted** from [1] and confirmed by the stagnation point at 180 degrees (Cp 1.044 in the reference). The PINN was trained for 10 rounds on a 50x25 O-grid built here, seed 0.

Table: E5, cylinder at Mach 0.4: pressure coefficient against the FVM 800x400 file
| Method | Mesh | Cp RMS | Cp minimum | Cp at stagnation | Source |
|---|---|---|---|---|---|
| PINN, positivity on | 50x25 | {{E5|pinn|50x25|rms_cp}} | {{E5|pinn|50x25|note:Cp_min:2}} | {{E5|pinn|50x25|note:Cp_stag:2}} | {{E5|pinn|50x25|source}} |
| PINN, positivity off | 50x25 | {{E5|pinn_nopos|50x25|rms_cp}} | {{E5|pinn_nopos|50x25|note:Cp_min:2}} | {{E5|pinn_nopos|50x25|note:Cp_stag:2}} | {{E5|pinn_nopos|50x25|source}} |
| Panel method with Prandtl-Glauert (Mach 0.4) | 140 panels | {{E5|panel|140 panels|rms_cp}} | {{E5|panel|140 panels|note:Cp_min:2}} | {{E5|panel|140 panels|note:Cp_stag:2}} | {{E5|panel|140 panels|source}} |
| Incompressible circle (the same panel method at Mach 0) | 140 panels | {{E5|panel|140 panels (Ma 0)|rms_cp}} | {{E5|panel|140 panels (Ma 0)|note:Cp_min:2}} | {{E5|panel|140 panels (Ma 0)|note:Cp_stag:2}} | {{E5|panel|140 panels (Ma 0)|source}} |
| FVM | 50x25 | {{E5|fvm|50x25|rms_cp}} | {{E5|fvm|50x25|note:Cp_min:2}} | {{E5|fvm|50x25|note:Cp_stag:2}} | {{E5|fvm|50x25|source}} |
| FVM | 100x50 | {{E5|fvm|100x50|rms_cp}} | {{E5|fvm|100x50|note:Cp_min:2}} | {{E5|fvm|100x50|note:Cp_stag:2}} | {{E5|fvm|100x50|source}} |
| FVM | 200x100 | {{E5|fvm|200x100|rms_cp}} | {{E5|fvm|200x100|note:Cp_min:2}} | {{E5|fvm|200x100|note:Cp_stag:2}} | {{E5|fvm|200x100|source}} |
| FVM | 400x200 | {{E5|fvm|400x200|rms_cp}} | {{E5|fvm|400x200|note:Cp_min:2}} | {{E5|fvm|400x200|note:Cp_stag:2}} | {{E5|fvm|400x200|source}} |
| FVM (the reference) | 800x400 | - | {{E5|fvm|800x400|note:Cp_min:2}} | {{E5|fvm|800x400|note:Cp_stag:2}} | {{E5|fvm|800x400|source}} |

![Cp around the cylinder: FVM reference, panel method with Prandtl-Glauert, the incompressible circle and the two PINN runs](results/figures/m3_e5_cylinder.png)

**Corrected number.** The panel method with Prandtl-Glauert has a Cp RMS of {{E5|panel|140 panels|rms_cp}} against the reference. An earlier note in the project said about 0.30; that figure is the incompressible circle (Mach 0, {{E5|panel|140 panels (Ma 0)|rms_cp}}). The PINN ({{E5|pinn_nopos|50x25|rms_cp}} and {{E5|pinn|50x25|rms_cp}}) beats the panel method by a modest margin and sits close to the FVM at the suction peak, but it does not beat the FVM at 200x100 ({{E5|fvm|200x100|rms_cp}}). The exact lift of a symmetric cylinder is zero; the PINN's Cl is {{E5|pinn|50x25|cl:3}} (positivity on) and {{E5|pinn_nopos|50x25|cl:3}} (off), so the off run carries a small spurious circulation. One seed per mode.

### 5.1.6 Speed (experiment E4): measured and quoted, not a like-for-like comparison

Table: E4, time to evaluate one case (NACA 0012, 4 degrees, Mach 0.4); the PINN time is one forward pass of an already trained network over every mesh node, 8 threads
| What is timed | Mesh | Time | Training time of that network | Source |
|---|---|---|---|---|
| PINN, positivity on | 50x25 | {{E4|pinn|50x25|infer_ms}} | {{E4|pinn|50x25|train_s}} | {{E4|pinn|50x25|source}} |
| PINN, positivity off | 50x25 | {{E4|pinn_nopos|50x25|infer_ms}} | {{E4|pinn_nopos|50x25|train_s}} | {{E4|pinn_nopos|50x25|source}} |
| PINN, positivity on | 100x50 | {{E4|pinn|100x50|infer_ms}} | {{E4|pinn|100x50|train_s}} | {{E4|pinn|100x50|source}} |
| PINN grid-sequenced, positivity on | 100x50 | {{E4|pinn_gs|100x50|infer_ms}} | {{E4|pinn_gs|100x50|train_s}} (this stage only) | {{E4|pinn_gs|100x50|source}} |
| PINN grid-sequenced, positivity off | 100x50 | {{E4|pinn_gs_nopos|100x50|infer_ms}} | {{E4|pinn_gs_nopos|100x50|train_s}} (this stage only) | {{E4|pinn_gs_nopos|100x50|source}} |
| PINN grid-sequenced, positivity off | 200x100 | {{E4|pinn_gs_nopos|200x100|infer_ms}} | withheld (the laptop slept) | {{E4|pinn_gs_nopos|200x100|source}} |
| Panel method, solve only | 140 panels | {{E4|panel|140 panels|infer_ms}} | none needed | {{E4|panel|140 panels|source}} |
| NeuralFoil (viscous, Re 6e6: not comparable) | large | {{E4|neuralfoil|large|infer_ms}} | none needed | {{E4|neuralfoil|large|source}} |
| FVM solve (another machine, from [1]) | 50x25 | {{E4|fvm|50x25|infer_ms:0}} | - | {{E4|fvm|50x25|source}} |
| FVM solve (another machine, from [1]) | 100x50 | {{E4|fvm|100x50|infer_ms:0}} | - | {{E4|fvm|100x50|source}} |
| FVM solve (another machine, from [1]) | 200x100 | {{E4|fvm|200x100|infer_ms:0}} | - | {{E4|fvm|200x100|source}} |
| FVM solve (another machine, from [1]) | 400x200 | {{E4|fvm|400x200|infer_ms:0}} | - | {{E4|fvm|400x200|source}} |

![Time to evaluate one case: measured PINN, panel and NeuralFoil times against the quoted FVM times](results/figures/m3_e4_speed.png)

Evaluating a trained network is fast (about a millisecond at 50x25), but it answers **one** shape, Mach number and angle: a new condition means retraining at the times in the table (minutes per run). The panel method needs no training and answers a new condition in milliseconds; NeuralFoil solves a different (viscous) problem and is shown only as a speed reference; the FVM times are one solve each, quoted from the paper's table, run by its authors on another machine, and the paper itself says that table is not meant as a rigorous comparison. No speed claim is made beyond what is measured. Only the evaluation of a trained network is fast; the PINN is not faster than a panel method once its training is counted.

### 5.1.7 The stop rule failed; the practice is a fixed early stop (experiment E6)

We asked whether a rule that does not use the FVM reference can stop training near the good answer. The rule: stop when the lift changes by less than 0.002 and the wall-Cp RMS change is below 0.005 for 2 rounds in a row. The thresholds were set after looking at the E1 runs (the development set, which supports no claim) and were not changed afterwards. It was judged on 15 held-out runs: six new 50x25 seeds with 20 attempts each, plus the cylinder, NACA 2412, 100x50 and 200x100 runs, replayed with the functions the training script uses (measured, `results/e6_stop_rule.md`, `results/e6_replay.csv`).

- **0 of 8 drifted 50x25 runs were stopped** (0 of 5 in the development set, 0 of 3 held-out). In the three held-out runs that drifted, the final error is 3.4 to 5.6 times the best round, and the closest the rule came to firing was 1.64 to 2.39 (it fires below 1): the changes never stay below the thresholds for two rounds in a row.
- The rule fired in 4 of 15 held-out runs, all of them non-drifting runs, with no bad stop, saving 322.9 s in total: harmless, not useful. An online run was cut off after 12 of 20 attempts without firing; it does not drift, so it tests nothing and the result is inconclusive.
- **What works as a practice is a fixed early stop.** The Cp RMS of the 12 runs at 50x25 (6 development runs from Table 5.2 and 6 held-out seeds) at a fixed round: round 3, 0.027 to 0.032 (mean 0.029) for the development runs and 0.027 to 0.033 (mean 0.030) for the held-out seeds; round 4, 0.026 to 0.101 and 0.027 to 0.059; round 5, 0.027 to 0.147 and 0.027 to 0.058; final round, 0.033 to 0.222 and 0.029 to 0.166 (these held-out figures are measured and printed in `results/e6_stop_rule.md`; the development figures are those of Table 5.2).

![E6: the plateau stop rule on the runs, and the Cp RMS at fixed early rounds](results/figures/m3_e6_stop_rule.png)

The round number comes from E1 (a recipe derived on the development set and checked on six held-out seeds). It is **not risk-free** (round 4 already had one run at 0.101 and round 5 one at 0.147) and it was not tested on other meshes or shapes. The late drift is **unsolved**: the candidate remedies (a Kutta-type constraint or panel prior, periodic input, a hard far-field input) were not tried.

## 5.2 Comparison with the Objectives of Chapter 1

Table: The five objectives and what was achieved
| Objective | Result | Status |
|---|---|---|
| O1: verify NNfoil component by component | 11 numbered findings in docs/verification.md: 7 fixed with a test or a flag, 3 documented and left unchanged on purpose, 1 limitation measured; the Euler residuals, mapping, Cp and force routines confirmed correct | Met |
| O2: train on a laptop CPU | Trained 50x25, 100x50 and 200x100 meshes on the CPU; {{E2|pinn|50x25|train_s}} at 50x25 and {{E2|pinn|100x50|train_s}} at 100x50 (cold start); the 200x100 time is withheld | Met (one wall time withheld) |
| O3: PINN more accurate than the FVM on the same mesh, against the 800x400 reference | Lower Cp RMS than the FVM on all three meshes (Table 5.1), but each cell is one seed; the 50x25 lift is not better than the FVM's; the 50x25 training drifts in 5 of 6 runs | Met for the Cp metric; **partly met** as a robust claim |
| O4: application with draw-any-shape, animated flow, PINN results and labelled numbers | Built and tested (106 front-end and 153 back-end tests); PINN Lab with 14 cases and a replay; train-a-PINN-on-your-shape not built | Met, except the train-your-shape feature (**not done**) |
| O5: a reference-free rule to stop at the best answer | The plateau rule stopped 0 of 8 drifted runs; only a fixed early stop (round 3 to 5) works, and it is not risk-free | **Not met** |

Other honest findings that are results rather than objectives: one cold start without positivity collapsed in 3 seeds, a cold NACA 2412 run with positivity collapsed at 4 degrees, the positivity option has no consistent effect, and the PINN is not better than the panel method on NACA 2412 or faster once training is counted.

## 5.3 Economic Analysis

Table: Bill of materials
| Item | Quantity | Cost | Note |
|---|---|---|---|
| Laptop, AMD Ryzen 9 6900HS, 16 logical threads, no GPU | 1 (already owned) | No additional cost [FILL IN: purchase price, only if the department wants the laptop counted] | Used for all training, timing and development |
| Python, PyTorch (CPU), NumPy, SciPy, Matplotlib, pytest | - | Free, open source | |
| FastAPI, pydantic, uvicorn, React, Vite | - | Free, open source | |
| python-docx and python-pptx | - | Free, open source | Used to build this report |
| NNfoil source and FVM reference data | - | Free to read; no licence | Credited; repository kept private |
| NeuralFoil | - | Free, open source | Comparison only |
| Cloud service, GPU | 0 | None used | |

**Resource utilisation (measured).** All training used 8 of the 16 threads, no GPU. Training times on this laptop: {{E2|pinn|50x25|train_s}} for a 50x25 run, {{E2|pinn|100x50|train_s}} for a 100x50 cold-start run, and {{E2|pinn_gs|100x50|train_s}} for a grid-sequenced 100x50 stage (excluding its 50x25 parent run). The 200x100 time is withheld.

**Cost-benefit against a cloud GPU (an ESTIMATE, with stated assumptions).** The price of a cloud GPU hour is not quoted here. Let R be an **assumed** price per cloud GPU-hour, in any currency. Assume, generously for the cloud, that the same run would take the same wall time on a rented GPU (a GPU would probably be faster, so the hours below are an upper bound), and ignore set-up, storage and data transfer. Then the cost of a run is R times its hours:

Table: Estimated cloud cost of the runs that were done on the laptop (ESTIMATE: R is an assumed price per GPU-hour)
| Run | Measured laptop time | Hours | Estimated cloud cost |
|---|---|---|---|
| 50x25 cold start, positivity on | {{E2|pinn|50x25|train_s}} | {{E2|pinn|50x25|train_h}} | {{E2|pinn|50x25|train_h}} x R |
| 100x50 cold start, positivity on | {{E2|pinn|100x50|train_s}} | {{E2|pinn|100x50|train_h}} | {{E2|pinn|100x50|train_h}} x R |
| 100x50 grid-sequenced stage, positivity off | {{E2|pinn_gs_nopos|100x50|train_s}} | {{E2|pinn_gs_nopos|100x50|train_h}} | {{E2|pinn_gs_nopos|100x50|train_h}} x R |

Every run in this study took less than one hour, so on the laptop the marginal cost is zero (the electricity was not measured) and a rented GPU would have cost less than R per run: the benefit of the laptop route is not the saving, which is small, but that no account, card or queue is needed and that the whole study is reproducible offline by a student team. This is an estimate and must be read as such; a real price should be inserted by the team if the committee asks for a figure.

## 5.4 Comparison with Existing Solutions and Standards

Table: AeroNet's PINN against the alternatives that were run or quoted
| Solution | Problem solved | Time to answer one case | Per-case training | Source |
|---|---|---|---|---|
| NNfoil as released [1] | Euler, subsonic, one condition per model | Training 384, 266, 262 and 271 s at 50x25, 100x50, 200x100 and 400x200 (paper's machine) | Yes | quoted from [1] via results/e4_speed.md; the paper says its table is not a rigorous comparison and that L-BFGS often stops early |
| AeroNet PINN on this laptop | The same | Training {{E2|pinn|50x25|train_s}} at 50x25 and {{E2|pinn|100x50|train_s}} at 100x50; evaluation {{E4|pinn|50x25|infer_ms}} | Yes | measured |
| FVM (reference solver) | Euler on a mesh | {{E4|fvm|50x25|infer_ms:0}} to {{E4|fvm|400x200|infer_ms:0}} per solve | No | quoted from [1] (other machine) |
| Panel method with Prandtl-Glauert (ours) | Linear potential flow | {{E4|panel|140 panels|infer_ms}} | No | measured |
| NeuralFoil [4] | Viscous, incompressible, Re 6e6 | {{E4|neuralfoil|large|infer_ms}} | No | measured (speed only) |
| XFOIL [25] | Viscous-inviscid coupling | Not measured | No | not run |

The quoted paper times and our laptop times are of the same order, which is all that can be said: the machines differ and the paper's table is explicitly not a rigorous comparison. **Standards.** There is no formal standard for PINN solvers. We followed the practice of verification and validation in CFD: one metric fixed in advance, a comparison against the finest available reference, a mesh study, several seeds where affordable, failures reported, and every number labelled by its origin. The references follow the IEEE style and the report follows the institute's capstone template.

# Chapter 6: Conclusion and Future Scope

## 6.1 Summary of Achievements

- NNfoil [1] was checked component by component against the Euler equations: eleven numbered findings, each with a test or a measured file, seven fixed or made optional, three documented and deliberately left unchanged. The runs now work on a CPU with 8 threads.
- Every run is judged with one Cp metric against the 800x400 FVM reference. On the same mesh the PINN beat the FVM on the Cp metric at 50x25, 100x50 and 200x100 (Table 5.1), and trained in minutes on the laptop.
- Grid sequencing, an addition of ours, gave a lift within 0.50 to 0.52 in all eight seed-check runs at 100x50.
- A web application lets a user draw any 2D shape, see an animated flow and forces, and inspect 14 trained PINN cases with a replay and a numbers dashboard, each number labelled measured, quoted or fitted.
- The report, like the application, is built from one results table: a missing number stops the build.

## 6.2 Impact on Society, Environmental Sustainability, Ethical Issues and Compliance

**Society.** The project lowers the barrier to experimenting with flow solvers: a laptop, free software and a browser are enough, and the application is usable on a phone and offline. It is a teaching and screening tool, not a certification tool.

**Environment.** Training ran for minutes on a laptop CPU rather than on a GPU cluster. The energy was not measured, so no saving is claimed in numbers.

**Ethics and compliance.** The main ethical risks are overclaiming and unattributed reuse. Against the first, results that went against the project's hopes are reported: the stop rule failed on held-out runs, training drifts, one cold start collapsed, one quoted number was corrected, one wall time is withheld, and the PINN is not better than the panel method on NACA 2412. Against the second, NNfoil is credited in the README, the application footer and this report. Because the upstream code has no licence, the repository is private and nothing is pushed publicly; the authors should be asked before any release. An AI coding assistant was used and is disclosed in the declaration.

## 6.3 Limitations

- Inviscid, steady, two-dimensional, subsonic flow only: no separation, no friction drag, no shocks.
- One network per shape, Mach number and angle; no parametric network.
- Most cells are one seed; where seeds exist (E1, E6, the seed check) the spread is large.
- The reference is itself an FVM result; the panel method agrees with it to Cp RMS {{B0|panel|-|rms_cp|NACA0012_a4}}, which is the best independent check available.
- The drift regime was studied on one case family (NACA 0012, 4 degrees, Mach 0.4, 50x25); other meshes and shapes mostly did not drift, so the stop rule could not be tested on its main job there.
- All timings are from one laptop CPU with PyTorch; one wall time is withheld; the quoted FVM and NNfoil times are from another machine.
- Mach 0.3 for NACA 2412 and the cylinder condition are quoted from the paper.

## 6.4 Future Work

- A Kutta-type constraint or a panel-method prior that supplies the right circulation, to repair the late drift, together with periodic network input and a hard far-field input.
- A parametric angle of attack (one network for a range of angles), as in [2].
- Training a PINN on a user's drawn shape from inside the application (the mesh and fold check exist; the job runner is not built), and a 400x200 PINN.
- A viscous extension (a boundary-layer coupling), three dimensions, and shape optimisation with the trained network.
- More seeds for the NACA 2412 and cylinder cases, and the random airfoils whose conditions are known from the paper.

# References

[FILL IN: before submission the team must read every reference below and confirm its details; see docs/deliverables_src/FILL_IN_CHECKLIST.md]

[1] W. Cao, J. Song, and W. Zhang, "A solver for subsonic flow around airfoils based on physics-informed neural networks and mesh transformation," *Phys. Fluids*, vol. 36, no. 2, Art. no. 027134, 2024, doi: 10.1063/5.0188665 (arXiv:2401.08705).

[2] W. Cao, J. Song, and W. Zhang, "Solving high-dimensional parametric engineering problems for inviscid flow around airfoils based on physics-informed neural networks," *J. Comput. Phys.*, vol. 516, Art. no. 113285, 2024, doi: 10.1016/j.jcp.2024.113285.

[3] Y. Sun, U. Sengupta, and M. Juniper, "Physics-informed deep learning for simultaneous surrogate modeling and PDE-constrained optimization of an airfoil geometry," *Comput. Methods Appl. Mech. Eng.*, vol. 411, Art. no. 116042, 2023, doi: 10.1016/j.cma.2023.116042.

[4] P. Sharpe and R. J. Hansman, "NeuralFoil: An airfoil aerodynamics analysis tool using physics-informed machine learning," arXiv:2503.16323, Mar. 2025.

[5] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *J. Comput. Phys.*, vol. 378, pp. 686-707, 2019, doi: 10.1016/j.jcp.2018.10.045.

[6] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, and L. Yang, "Physics-informed machine learning," *Nat. Rev. Phys.*, vol. 3, no. 6, pp. 422-440, 2021, doi: 10.1038/s42254-021-00314-5.

[7] S. Cai, Z. Mao, Z. Wang, M. Yin, and G. E. Karniadakis, "Physics-informed neural networks (PINNs) for fluid mechanics: A review," *Acta Mech. Sin.*, vol. 37, no. 12, pp. 1727-1738, 2021, doi: 10.1007/s10409-021-01148-1.

[8] S. Cuomo, V. Schiano Di Cola, F. Giampaolo, G. Rozza, M. Raissi, and F. Piccialli, "Scientific machine learning through physics-informed neural networks: Where we are and what's next," *J. Sci. Comput.*, vol. 92, no. 3, Art. no. 88, 2022, doi: 10.1007/s10915-022-01939-z.

[9] X. Jin, S. Cai, H. Li, and G. E. Karniadakis, "NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations," *J. Comput. Phys.*, vol. 426, Art. no. 109951, 2021, doi: 10.1016/j.jcp.2020.109951.

[10] H. Eivazi, M. Tahani, P. Schlatter, and R. Vinuesa, "Physics-informed neural networks for solving Reynolds-averaged Navier-Stokes equations," *Phys. Fluids*, vol. 34, no. 7, Art. no. 075117, 2022, doi: 10.1063/5.0095270.

[11] S. Wassing, S. Langer, and P. Bekemeyer, "Physics-informed neural networks for inviscid transonic flows around an airfoil," *Phys. Fluids*, vol. 37, no. 8, Art. no. 086169, 2025, doi: 10.1063/5.0276518 (arXiv:2408.17364).

[12] J. Li, X. Du, and J. R. R. A. Martins, "Machine learning in aerodynamic shape optimization," *Prog. Aerosp. Sci.*, vol. 134, Art. no. 100849, 2022, doi: 10.1016/j.paerosci.2022.100849.

[13] S. Wang, Y. Teng, and P. Perdikaris, "Understanding and mitigating gradient flow pathologies in physics-informed neural networks," *SIAM J. Sci. Comput.*, vol. 43, no. 5, pp. A3055-A3081, 2021, doi: 10.1137/20M1318043.

[14] S. Wang, X. Yu, and P. Perdikaris, "When and why PINNs fail to train: A neural tangent kernel perspective," *J. Comput. Phys.*, vol. 449, Art. no. 110768, 2022, doi: 10.1016/j.jcp.2021.110768.

[15] A. S. Krishnapriyan, A. Gholami, S. Zhe, R. M. Kirby, and M. W. Mahoney, "Characterizing possible failure modes in physics-informed neural networks," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2021, arXiv:2109.01050.

[16] P. Rathore, W. Lei, Z. Frangella, L. Lu, and M. Udell, "Challenges in training PINNs: A loss landscape perspective," arXiv:2402.01868, 2024.

[17] L. Lu, X. Meng, Z. Mao, and G. E. Karniadakis, "DeepXDE: A deep learning library for solving differential equations," *SIAM Rev.*, vol. 63, no. 1, pp. 208-228, 2021, doi: 10.1137/19M1274067.

[18] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019, arXiv:1912.01703.

[19] M. Raissi, A. Yazdani, and G. E. Karniadakis, "Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations," *Science*, vol. 367, no. 6481, pp. 1026-1030, 2020, doi: 10.1126/science.aaw4741.

[20] A. M. Winslow, "Numerical solution of the quasilinear Poisson equation in a nonuniform triangle mesh," *J. Comput. Phys.*, vol. 1, no. 2, pp. 149-172, 1966, doi: 10.1016/0021-9991(66)90001-5.

[21] J. F. Thompson, Z. U. A. Warsi, and C. W. Mastin, *Numerical Grid Generation: Foundations and Applications*. New York, NY, USA: North-Holland, 1985.

[22] J. L. Hess and A. M. O. Smith, "Calculation of potential flow about arbitrary bodies," *Prog. Aerosp. Sci.*, vol. 8, pp. 1-138, 1967, doi: 10.1016/0376-0421(67)90003-6.

[23] J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed. Cambridge, U.K.: Cambridge Univ. Press, 2001, doi: 10.1017/CBO9780511810329.

[24] J. D. Anderson, Jr., *Fundamentals of Aerodynamics*, 6th ed. New York, NY, USA: McGraw-Hill Education, 2017.

[25] M. Drela, "XFOIL: An analysis and design system for low Reynolds number airfoils," in *Low Reynolds Number Aerodynamics* (Lecture Notes in Engineering). Berlin, Germany: Springer, 1989, pp. 1-12, doi: 10.1007/978-3-642-84010-4_1.

[26] B. M. Kulfan, "Universal parametric geometry representation method," *J. Aircr.*, vol. 45, no. 1, pp. 142-158, 2008, doi: 10.2514/1.29958.

[27] D. C. Liu and J. Nocedal, "On the limited memory BFGS method for large scale optimization," *Math. Program.*, vol. 45, no. 1-3, pp. 503-528, 1989, doi: 10.1007/BF01589116.

[28] W. Cao, "NNfoil: PINN solver for subsonic flow around airfoils," GitHub repository. [Online]. Available: https://github.com/Cao-WenBo/NNfoil (vendored copy in this project; the upstream repository has no licence).

# Appendix A: Hyperparameters and Setup

Table: Training settings used for every PINN run unless a row says otherwise
| Setting | Value |
|---|---|
| Network | 5 layers of 128 units, tanh activations, weight normalisation (NNfoil default) |
| Optimiser | L-BFGS, up to 1000 iterations per round; a failed round restarts the optimiser |
| Loss | 2e4 times the weighted PDE residual plus far-field, wall and seam terms (unit weights) |
| Positivity option | softplus plus 1e-3 on p and rho (on by default in our code; off reproduces the upstream behaviour) |
| Threads | 8 of 16 logical threads; float32 |
| Rounds | 20 attempted for E1 and E2 (50x25), 10 for E3 and E5, 4 on the fine mesh for grid sequencing |
| Seeds | 0 for most cells; 0 to 2 for E1; 3 to 5 for the held-out E6 runs; 0 to 3 for the seed check |
| Mach and angle | 0.4 and 4 degrees (NACA 0012); 0.3 (quoted) and -4, 0, 4, 8 degrees (NACA 2412); 0.4 and 0 degrees (cylinder, quoted) |
| Meshes | 50x25, 100x50, 200x100 O-grids; far field 15 chords; the NACA 2412 and cylinder meshes were generated here |
| Reference | FVM 800x400 file; Cp RMS over reference points with 0.02 <= x/c <= 0.98 |
| Run names | Fixed, for example naca0012_100x50_a4_m04, so that any script can find a run |

# Appendix B: Reproducing the Results

From the repository root (Windows; the research environment holds PyTorch, NumPy and Matplotlib):

```
research/.venv/Scripts/python research/experiments.py commands e1
research/.venv/Scripts/python research/experiments.py collect e1
research/.venv/Scripts/python research/time_inference.py --e4
research/.venv/Scripts/python research/seed_check.py
research/.venv/Scripts/python research/stop_rule_eval.py replay --summary
research/.venv/Scripts/python research/make_figures.py
research/.venv/Scripts/python research/docs_build/build_all.py --check
research/.venv/Scripts/python research/docs_build/build_all.py
OMP_NUM_THREADS=1 research/.venv/Scripts/python -m pytest research/tests -q
```

Timed runs need an idle, awake machine and one training process at a time. The run folders under results/runs keep the settings, the log and the per-round snapshots; the weights and the field files are not committed.

# Appendix C: Short Code Listings

The error metric used for every method (research/metrics.py), which is the whole definition of "Cp RMS" in this report:

```
def cp_error(x, y, cp, ref) -> dict:
    keep = (ref[:, 0] >= 0.02) & (ref[:, 0] <= 0.98)
    r = ref[keep]
    d2 = (x[None, :] - r[:, 0:1]) ** 2 + (y[None, :] - r[:, 1:2]) ** 2
    err = cp[d2.argmin(axis=1)] - r[:, 2]
    return {"rms": float(np.sqrt(np.mean(err**2))),
            "max": float(np.max(np.abs(err))), "n": int(err.size)}
```

How a number enters this report: the Markdown source holds a placeholder, the builder looks it up in results/summary.csv, and a missing or empty cell stops the build:

```
{{E2|fvm|100x50|rms_cp}}   ->  the Cp RMS of the 100x50 FVM, filled from the results table
{{E9|fvm|100x50|rms_cp}}   ->  KeyError: expected exactly one row ... found 0
```

# Annexure

## A. Plagiarism Report

[FILL IN: attach the Turnitin similarity summary here (one page) and write the similarity percentage: ____ %.] The report must be submitted to the institute's plagiarism check before this page is completed; no percentage has been measured by the authors of this document.

## B. PO/PSO Mapping

The table maps the technical tasks of this project to the twelve standard graduate attributes (programme outcomes, POs). This is a draft: **confirm against your department's PO/PSO list** before submission, and fill in the programme specific outcomes.

Table: Technical tasks linked to programme outcomes (draft)
| PO | Graduate attribute | Where this project addresses it |
|---|---|---|
| PO1 | Engineering knowledge | Euler equations, panel method, mesh transformation and PINN training applied to airfoil flow (Chapter 3) |
| PO2 | Problem analysis | Component-by-component review of NNfoil; the drift and collapse analyses (Chapters 4 and 5) |
| PO3 | Design and development of solutions | The application, the run-folder pipeline, grid sequencing (Chapters 3 and 4) |
| PO4 | Conduct investigations of complex problems | Experiments E1 to E6 with fixed metric, mesh study, seeds, held-out evaluation (Chapter 5) |
| PO5 | Modern tool usage | PyTorch, FastAPI, React, git, pytest, scripted document builds |
| PO6 | The engineer and society | Accessible CPU-only tool; limits shown on screen (Sections 1.3, 6.2) |
| PO7 | Environment and sustainability | CPU-only, short runs; sustainability statement (Annexure E) |
| PO8 | Ethics | Honest reporting of failures, labelled measured, quoted or fitted numbers, credit to NNfoil, licence respected (Section 6.2) |
| PO9 | Individual and team work | See Team Roles (Annexure D) |
| PO10 | Communication | This report, the slides, the application's on-screen explanations |
| PO11 | Project management and finance | Plan, timeline, risks (Section 1.4); economic analysis (Section 5.3) |
| PO12 | Life-long learning | Learning PINNs, CFD basics, and a new software stack in one project |
| PSO | [FILL IN: the department's programme specific outcomes] | [FILL IN: how the project meets each] |

## C. Outcome of the Report

What exists today (all in the project repository, kept private; nothing has been pushed to a public site):

- a working web application (draw any shape, flow view, PINN Lab with 14 trained cases, Results page);
- the verified and CPU-ready NNfoil copy with its verification table and 11 findings;
- the experiment results and figures (results/summary.csv and results/figures);
- three milestone reports (docs/milestones) and this report.

Proof of outcome that only the team can supply:

- [FILL IN: paper, journal or conference submission or acceptance proof (title, venue, status, attach the first page), or "none"]
- [FILL IN: accepted abstract, patent filing or product link, or "none"]

## D. Team Roles

Table: Contribution of each team member
| Student | Contribution |
|---|---|
| {{field:student_a_name_enrol}} | {{field:student_a_contribution}} |
| {{field:student_b_name_enrol}} | {{field:student_b_contribution}} |
| {{field:student_c_name_enrol}} | {{field:student_c_contribution}} |

## E. Sustainability Statement

**Environment.** All training and timing ran on one laptop CPU, with no GPU and no cloud service; the longest single run measured here took {{E1|pinn|50x25|train_s|NACA0012_a4_s1}} (the 50x25 seed-1 run) and a typical 100x50 cold start {{E2|pinn|100x50|train_s}}. The electrical energy used was not measured, so no figure is given. The application itself runs offline on a laptop and needs no data centre.

**Society.** The tool makes flow estimates available to people without GPUs or CFD licences, and it teaches how much trust a number deserves by labelling it measured, quoted or fitted. It may be misused if its inviscid results are read as complete aerodynamics: it states its limits on screen and never shows a drag figure for a car.

**Long-term.** The results table, the scripts and the tests let a later team re-run and regenerate everything. The main risk to sustainability is licensing: NNfoil has no licence, so the project cannot be released publicly without the authors' permission.
