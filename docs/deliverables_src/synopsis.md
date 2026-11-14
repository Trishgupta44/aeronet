# AeroNet: A Physics-Informed Neural Network Solver for Airfoil Aerodynamics
Project synopsis (final)
Students: {{field:students_names_enrol}}
Supervisor: {{field:supervisor_name}}
Programme and semester: {{field:programme_semester_year}}
Department and institute: {{field:department_institute}}
Date: [FILL IN: date of submission]

## 1. Problem

Predicting the airflow and the pressure on an airfoil is central to the design of aircraft wings, drone rotors and turbine blades. A classical finite-volume solver (FVM) is accurate, but it has to be run again for every new shape or flow condition, and it needs a fine mesh: the paper we build on quotes {{sec:E4|fvm|50x25|infer_ms}} for one solve on its coarsest mesh and {{sec:E4|fvm|400x200|infer_ms}} on its finest (quoted from Table 1 of arXiv 2401.08705, measured by the authors on another machine, not by us). A physics-informed neural network (PINN) is trained on the governing equations instead of on labelled simulations [4]. NNfoil, by Cao, Song and Zhang [1], is an open-source PINN solver for inviscid subsonic flow around airfoils. It maps the curved mesh around the airfoil onto a plain rectangle, so that the network does not have to resolve the sharp suction peak at the leading edge in physical space, and its authors report much lower error than a second-order FVM on very sparse meshes.

The problem for this project is practical. The released NNfoil code was not independently checked, assumed a GPU, fixed one flow condition per run and shows its results only as scripts. We therefore ask: is the code correct, does the claim hold when it is run on a laptop CPU, and can the result be shown clearly and honestly?

## 2. Objectives

1. Review the NNfoil code equation by equation, and fix or document each defect with its evidence.
2. Make the solver run reproducibly on a CPU, with a pinned environment and one command per run.
3. Validate the predicted surface pressure and lift against the finite-volume reference data bundled with NNfoil, with one error metric for every method.
4. Run small, controlled experiments: the effect of the density and pressure positivity fix, the effect of mesh size, a second airfoil, a cylinder, speed, and a stopping rule for training.
5. Build an interactive demonstration in which a user can draw a shape, see the flow, and compare the trained network with the reference, with every number labelled measured, quoted or fitted.

## 3. Methodology

**Verify.** Each component of the solver (the four Euler residuals, the mesh mapping and its Jacobian, the wall and far-field conditions, the force integration, the mesh generator) is compared with the theory in a table with file and line references, and each verdict has a test or a CSV in `results/`. Suspected defects are treated as hypotheses until reproduced.

**Reproduce and validate.** Training runs use PyTorch on a laptop CPU (8 threads) with the L-BFGS optimiser, in rounds. The error metric is the root-mean-square pressure-coefficient error (Cp RMS) over the reference points, against the FVM solution on an 800x400 mesh, plus the lift coefficient (Cl) integrated from the pressure. The baselines are the FVM on the same coarse mesh, a Hess-Smith panel method with the Prandtl-Glauert correction, and thin-airfoil theory. We added one technique that is not in the paper, grid sequencing: train on a 50x25 mesh, move the weights to 100x50, and continue.

**Present.** A web application (FastAPI backend, React front end) lets the user explore a NACA 0012 airfoil, draw any closed shape (including idealised cars on a road), browse 14 trained PINN cases with a training replay, and read the results page. The live solver in the app is the classical panel method; the PINN cases are trained offline.

## 4. Expected and achieved results

| Objective | Expected | Achieved (measured unless marked) |
|---|---|---|
| 1. Code review | Confirm or refute the suspected defects | 11 findings: 7 fixed in code, 4 documented or opt-in. The mesh-smoothing coefficient is confirmed 4 times too small, with low impact (`docs/verification.md`) |
| 2. CPU run | A run that finishes on a laptop | {{E2b|pinn|50x25|train_s:0}} for a 50x25 run and {{E2b|pinn|100x50|train_s:0}} for a 100x50 run, no GPU |
| 3. Validation | PINN closer to the reference than the FVM on the same mesh | At 100x50: Cp RMS {{E2|pinn|100x50|rms_cp}} against {{E2|fvm|100x50|rms_cp}} for the FVM, {{ratio:E2|fvm|100x50|rms_cp ; E2|pinn|100x50|rms_cp}} lower. At 200x100: {{E2|pinn_gs_nopos|200x100|rms_cp}} against {{E2|fvm|200x100|rms_cp}}. One seed per cell |
| 4. Experiments | Three small experiments | Six (E1 to E6) and a seed check. Negative results: the late drift of training at 50x25, a stop rule that stopped {{count:e6_replay.csv|drifted=True|stop_round!=never}} of {{count:e6_replay.csv|drifted=True}} drifting runs, and one collapsed NACA 2412 run |
| 5. Demonstration | A live demo | Web app with Explore, Draw, PINN Lab (14 cases) and Results; a 5-minute demo script |

## 5. Scope and limitations

The flow is inviscid and subsonic (Euler equations): there is no boundary layer, stall or friction drag, and the drag a solver reports is numerical error. One network is trained per flow condition. Most results are single seeds on one laptop CPU. The FVM timings and the Mach numbers of the NACA 2412 and cylinder files are quoted from the paper. The training time of the 200x100 run is withheld because the laptop slept during it. The late drift of 50x25 training is not solved; the practice that works is to stop at round 3 to 5.

## 6. References

1. W. Cao, J. Song and W. Zhang, "A solver for subsonic flow around airfoils based on physics-informed neural networks and mesh transformation", arXiv:2401.08705 (2024); published in Physics of Fluids 36(2), 027134 (journal details as listed in the team's earlier synopsis, not re-checked).
2. L. Sun, A. Sengupta and M. P. Juniper, "Physics-informed deep learning for simultaneous surrogate modeling and PDE-constrained optimization of an airfoil geometry", Computer Methods in Applied Mechanics and Engineering 411, 116042 (2023).
3. P. Sharpe and R. J. Hansman, "NeuralFoil: An airfoil aerodynamics analysis tool using physics-informed machine learning", arXiv:2503.16323 (2025).
4. M. Raissi, P. Perdikaris and G. E. Karniadakis, "Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations", Journal of Computational Physics 378, 686-707 (2019).

## Signatures

Students: [FILL IN: student signatures]
Supervisor: [FILL IN: supervisor signature and date]
