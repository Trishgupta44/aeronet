# Related Papers Guide: four papers in plain language
AeroNet project: companion note for the panel and the viva
Prepared by: [FILL IN: team member names]

## How to read the tags

**(abstract)**: checked against the paper's abstract page only; the full text was not re-read for this guide. **(team summary)**: from the summary a team member wrote while reading the paper (`docs/superpowers/plans/2026-09-24-aeronet-one-week-plan.md`, section 0), not re-checked here. **(paper text)**: read from the NNfoil PDF by an earlier session and recorded in `docs/verification.md`. **(measured here)**: our own run, from `results/`. An untagged sentence is our explanation, not a claim about the paper.

## 1. NNfoil: Cao, Song and Zhang (arXiv 2401.08705, 2024), the paper AeroNet builds on

**The idea.** A PINN struggles with airfoil flow because the air accelerates sharply near the leading edge. The authors map the curved mesh to a uniform computational space, so the network does not meet that sharp change or the wall directly, solve inviscid flow, and release an open-source solver. On very sparse meshes they report roughly an order of magnitude lower error than second-order finite-volume methods; on fine meshes it matches them in accuracy and efficiency (abstract).

**What it costs.** NNfoil training took 384, 266, 262 and 271 s on four meshes against FVM solves of 9, 24, 53 and 285 s, in a table the paper says is "not meant as a rigorous comparison" (paper text, Table 1). One network is one flow condition. We measured {{E2b|pinn|50x25|train_s:0}} and {{E2b|pinn|100x50|train_s:0}} per run on a laptop CPU (measured here).

**How AeroNet differs.** We kept the method and added verification (11 findings), a CPU runner, validation against the bundled FVM data, grid sequencing and a stopping-rule test. At 100x50 the Cp RMS is {{E2|pinn|100x50|rms_cp}} for the PINN and {{E2|fvm|100x50|rms_cp}} for the FVM on the same mesh (measured here, one seed).

**One viva question.** *The paper says accuracy barely improves with mesh refinement; do your results agree?* The PINN's Cp RMS fell from {{E2|pinn|50x25|rms_cp}} (50x25) to {{E2|pinn_gs_nopos|200x100|rms_cp}} (200x100), but these are single runs with different settings (`docs/milestones/M3-pinn-results.md`, section 4), so we do not claim to confirm or refute it.

## 2. Sun, Sengupta and Juniper (Computer Methods in Applied Mechanics and Engineering 411, 116042, 2023)

**The idea.** One PINN models the flow around an airfoil and optimises its shape for the lift-to-drag ratio in the same training. Automatic differentiation gives the shape gradients without a separate adjoint code, and the collocation points are sampled adaptively. The authors say it costs more than adjoint-based optimisation but may become faster with better hardware (abstract). The flow is Navier-Stokes at Reynolds number 20, checked against FEniCS, with the shape parameters as network inputs (team summary).

**What it costs.** About 18 h (one parameter) and 75 h (eleven parameters) on an RTX 3080; pointwise errors about 1e-2 for velocity and 1e-1 for pressure, largest at the leading edge (team summary).

**How AeroNet differs.** Their problem is viscous with the shape inside the network; ours is inviscid Euler flow with one network per condition. We never ran their method. That their leading-edge weakness motivates NNfoil's mesh transformation is our reading, not a statement from either abstract.

**One viva question.** *Why did you not optimise the shape as they do?* The project's scope is verifying and validating a solver; PINN-based optimisation is future work (the app's one optimiser, for the critical Mach number, uses the panel method).

## 3. NeuralFoil: Sharpe and Hansman (arXiv 2503.16323, 2025)

**The idea.** An open-source Python tool for fast airfoil analysis, similar in purpose to XFoil, reporting speedups of 8 to 1000 times at equivalent accuracy. It matches XFoil closely (mean relative drag error 0.37 percent on simple cases, up to 2.0 percent with post-stall and transitional cases), has smooth differentiable outputs for gradient-based design, and adds a way to estimate surrogate uncertainty (abstract).

**What it costs.** About 7.9 million XFoil runs to train; about 1.2 ms per call against about 73 ms for XFoil (team summary). We measured {{E4|neuralfoil|large|infer_ms}} per call (measured here).

**How AeroNet differs.** NeuralFoil is data-driven and viscous; NNfoil is data-free and inviscid. We use NeuralFoil only for its time: its lift (Re 6e6) is viscous and not comparable with our inviscid values (`results/e4_speed.md`).

**One viva question.** *Your PINN evaluates in milliseconds: is it faster than NeuralFoil?* Not in a fair sense. Both take about a millisecond ({{E4|pinn|50x25|infer_ms}} against {{E4|neuralfoil|large|infer_ms}}, measured here), but the PINN answers one condition and must be retrained for another.

## 4. Raissi, Perdikaris and Karniadakis (Journal of Computational Physics 378, 686-707, 2019)

**The idea.** Neural networks are trained on supervised tasks while respecting physical laws given as nonlinear partial differential equations, for two problems: solving an equation from data and discovering one from data, with continuous-time and discrete-time models; the networks are described as data-efficient universal function approximators that encode physical laws as prior information (abstract of the arXiv preprint "Physics Informed Deep Learning (Part I)", arXiv 1711.10561; we understand the journal paper to be its published form but did not compare the texts).

**What it costs.** We did not read its cost figures. In our runs a PINN costs the training: minutes on a laptop CPU per case (measured here), repeated per condition.

**How AeroNet differs.** This is the foundation, not a competitor: NNfoil applies the idea to the Euler equations around an airfoil. We checked its loss residuals term by term against an independent conservative form (`docs/verification.md`).

**One viva question.** *Why does a PINN need no labelled data?* The loss is the equation residual at sample points plus the boundary conditions; the FVM data is used only afterwards to score the network.
