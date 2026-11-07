# Milestone 1: NNfoil verified and running on a laptop CPU

Status: **complete** (seed check done, 2026-11-06). Branch `plan-a-nnfoil-verified`. Every number below comes from
`results/summary.csv` (S), a run's `rounds.csv` / `log.json` (R), `results/finding3_mesh_smoothing.csv` and
`results/finding10_jacobian.csv` (C), `NNfoil/AERONET_CHANGES.md` or `git log`. Labels: **measured** = produced by
code in this repo; **quoted** = copied from another document, not re-run here; **fitted** = tuned to data.

## 1. What NNfoil is, and what we did

NNfoil is the physics-informed neural network (PINN) Euler solver of Cao, Song and Zhang (arXiv 2401.08705): a
network maps mesh indices (xi, eta) of an O-grid to (u, v, p, rho) and is trained on the Euler residual. The repo has
**no licence**, so it stays private and fully credited to its authors. We vendored it, checked it line by line
(`docs/verification.md`), made it run on a CPU, added a CLI that writes a run folder per case (`NNfoil/run_case.py`),
and measured it against the bundled finite-volume (FVM) data. All runs: NACA 0012, Mach 0.4, 4 deg, seed 0.

## 2. Findings

Line numbers are upstream (commit 4732498). Details and tests: [`docs/verification.md`](../verification.md),
[`NNfoil/AERONET_CHANGES.md`](../../NNfoil/AERONET_CHANGES.md).

| # | file:line | Problem | Fix or verdict | Commit | Evidence |
|---|---|---|---|---|---|
| 1 | NNfoil_Euler2D.py:269 | Device forced to CUDA | `pick_device()`; fixed | 6b10ac1 | `test_pick_device_falls_back_to_cpu` |
| 2 | NNfoil_Euler2D.py:193-197 | p, rho unconstrained but divided by | softplus + 1e-3 flag (default on). Fixed as a safety measure; **it did not help accuracy here** (4c) | 12ba8c1 | `test_positivity_keeps_p_rho_above_floor`; runs in 4c |
| 3 | mesh_gen.py:66,68 | Cross term 0.5*beta, Winslow needs 2 (4x too small): **confirmed, low impact** | Documented, not changed (shipped meshes and FVM data use it). Winslow residual 1.95e-4 vs 1.99e-6 with 2; nodes move up to 0.0113 chord; both grids valid (C) | 77eeaf4 | `results/finding3_mesh_smoothing.csv` |
| 4 | NNfoil_Euler2D.py:265 | 1 CPU thread | `min(8, cpu_count)`; fixed | 1bed181 | runtime setting, no test |
| 5 | NNfoil_Euler2D.py:110 | `np.math` removed in NumPy 2 | `math.factorial`; fixed | c77caa2 | `test_nnfoil_constructs_on_numpy2` |
| 6 | NNfoil_Euler2D.py:247,250 | Whole-module `torch.load` fails (PyTorch >= 2.6); shared file name | `state_dict` + per-run path; fixed | c475030 | `test_checkpoint_roundtrip` |
| 7 | NNfoil_Euler2D.py:207-210 | `Calculate_Force` never called, no Cl/Cd | `evaluate_wall()`; fixed | 0822cad | `test_evaluate_wall_returns_finite_floats` |
| 8 | NNfoil_Euler2D.py:55 | Mach 0.4 and 4 deg hard-coded; one condition per model | Constructor arguments; fixed (still one condition per model) | 0822cad | `test_flow_condition_arguments` |
| 9 | NNfoil_Euler2D.py:235-237,267,281 | Training cost (5x128 net, L-BFGS 1000 iterations/round) | Limitation. Measured on a laptop CPU in section 3 | (none) | R: `results/runs/*/log.json` |
| 10 | NNfoil_Euler2D.py:135 | One-sided Jacobian at the trailing-edge seam: 1-2 wrong-sign cells on all 9 bundled meshes | Documented; opt-in `periodic_jacobian` flag (0 wrong-sign cells with it), default unchanged | 77eeaf4 | `results/finding10_jacobian.csv` |
| 11 | NNfoil_Euler2D.py:30 | `init.normal_` after `weight_norm` is a no-op | Documented, not changed (would alter every result) | 77eeaf4 | `test_normal_init_after_weight_norm_is_a_noop` |
| A12a | `train()` first-round restart | A failed first round discards a warm start | `set_warm_start`; cold start unchanged | da138d8 (+ `--init-from` in 1470816) | `test_warm_started_failed_first_round_keeps_the_transferred_weights` |

## 3. CPU cost (measured)

Machine: AMD Ryzen 9 6900HS (8 cores, 16 logical threads; read with `Get-CimInstance Win32_Processor`, since `wmic`
is not installed), PyTorch 2.15.0+cpu, **8 threads**, idle CPU, float32.

- **50x25: about 53 s per full round** (rounds 1-3 of the positivity run: 52.5, 53.4, 53.0 s). **100x50: about 90 s**
  (rounds 1-3: 90.2, 91.3, 88.1 s). Source: differences of `time_s` in `log.json`.
- Most later rounds stop early (L-BFGS tolerance): 1.7-26.5 s (50x25 positivity, rounds 4-19), 0.2-46.1 s (50x25 no
  positivity), 0.8-36.6 s (100x50 plain, rounds 4-20). So a "20 round" run is far fewer iterations than the paper's.
- Totals: 334.2 s (50x25, 19 good rounds, 1 failed), 416.2 s (50x25 no positivity, 18 good, 2 failed), 577.6 s (100x50
  plain, 20 rounds), 143.9 s and 121.1 s (100x50 grid-sequenced, 4 rounds each, **not** counting the 50x25 run they
  start from). No speed advantage of grid sequencing is claimed: its coarse stage here was a full 50x25 run.
  200x100 was not run.

## 4. Results

### Headline table (S; Cl reference 0.5193 and Cp reference = FVM 800x400, nearest-point RMS over 654 points, measured)

| Method | Mesh | Rounds (failed) | Cl | Cl error | Cp RMS | Cp max | Train s |
|---|---|---|---|---|---|---|---|
| FVM | 50x25 | - | 0.4265 | -17.9% | 0.1081 | 0.722 | - |
| PINN, positivity on | 50x25 | 19 (1) | 0.4119 | -20.7% | 0.0734 | 0.213 | 334.2 |
| PINN, positivity off | 50x25 | 18 (2) | 0.4829 | -7.0% | 0.0325 | 0.146 | 416.2 |
| FVM | 100x50 | - | 0.4803 | -7.5% | 0.0460 | 0.261 | - |
| PINN plain, positivity on | 100x50 | 20 (0) | 0.5163 | -0.6% | 0.0145 | 0.083 | 577.6 |
| PINN grid-seq., positivity off | 100x50 | 4 (0) | 0.5117 | -1.5% | 0.0148 | 0.086 | 143.9 |
| PINN grid-seq., positivity on | 100x50 | 4 (0) | 0.5024 | -3.3% | 0.0162 | 0.083 | 121.1 |
| Panel method (Hess-Smith + Prandtl-Glauert) | - | - | 0.5262 | +1.3% | 0.0099 | 0.071 | - |
| Thin-airfoil theory (PG corrected) | - | - | 0.4786 | -7.8% | - | - | - |

PINN rows are the **final round** of each run. Figures: [`m1_cp_50x25.png`](../../results/figures/m1_cp_50x25.png),
[`m1_cp_100x50.png`](../../results/figures/m1_cp_100x50.png) and every round of all five runs in
[`m1_trajectory.png`](../../results/figures/m1_trajectory.png). Per-round numbers: `results/runs/<run>/rounds.csv`.

### The honest reading (all single seed, seed 0)

**(a) PINN against FVM on the same sparse mesh.** Against the 800x400 reference, the PINN's Cp RMS is lower than the FVM's
on the same mesh in every run: 0.0325 against 0.1081 at 50x25 without positivity (3.3x), 0.0734 with it (1.5x), and
0.0145 / 0.0148 / 0.0162 against 0.0460 at 100x50 (3.2x / 3.1x / 2.8x). Cl follows only partly: the 50x25 positivity
run (Cl 0.4119) is **no better than the FVM** (0.4265). The PINN is not imitating the coarse FVM: its Cp differs from
the same-mesh FVM by RMS 0.0935 to 0.1074 at 50x25 and 0.0568 to 0.0620 at 100x50 (E2b rows), about the FVM's own
distance to the reference. Caveats: the 800x400 file is itself an FVM result, though the panel method agrees with it to
RMS 0.0099; we did not time the FVM, so no speed claim is made.

**(b) Trajectory of the 50x25 positivity run (R).** Cp RMS and Cl by round, all 19 good rounds:

| round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RMS | .193 | .037 | .030 | .028 | .030 | .029 | .034 | .068 | .056 | .053 | .064 | .057 | .057 | .060 | .064 | .060 | .060 | .065 | .073 |
| Cl | .307 | .478 | .497 | .499 | .493 | .501 | .482 | .408 | .430 | .442 | .419 | .435 | .435 | .428 | .424 | .428 | .430 | .421 | .412 |

Accuracy peaks in rounds 3-6 and then drifts, while the loss keeps falling (total loss 1.3e-4 at round 3, 7.2e-5 at
round 19). Nothing in the Euler residual enforces a Kutta condition (verification table), so a lower loss does not
mean a better Cl. (The best round by RMS is round 4, 0.0275: **post-hoc, chosen after seeing the FVM error**; the
headline row uses the final round.) The no-positivity 50x25 run is steadier: RMS 0.026-0.037 from round 2, Cl 0.48-0.52.

**(c) Positivity (finding 2).** On 50x25 it made training worse (RMS 0.0734 against 0.0325, Cl 0.4119 against 0.4829)
and its first round failed. On 100x50 from scratch it converged (Cl 0.5163, RMS 0.0145). A separate research
session reported that the original method (no positivity) collapses to the trivial solution on 100x50 (quoted,
`docs/nnfoil_improvements.md` N28); this pack has no cold-start no-positivity 100x50 run. **Single seed, not a
conclusion.** Experiment E1 (Plan C) will decide the default.

**(d) Grid sequencing (our addition, not in the paper).** Train on 50x25, move the weights to 100x50 with rescaled
input normalisation, continue 4 rounds. Cl right after the transfer: 0.4832 (positivity off) and 0.4121 (on); after 4
rounds 0.5117 and 0.5024. The research session's figures were 0.504 and 0.5065 (quoted). Both land at Cp RMS 0.015-0.016
in 121-144 s of fine training, the same accuracy as the plain 100x50 run.

### Seed check (A12b, measured; `results/seed_check.csv`)

Seeds 1-3, both positivity modes, NACA 0012 at Mach 0.4 and 4 deg, 100x50. Grid-sequenced = 5 rounds on 50x25, weights
moved, 4 rounds on 100x50. Cold start = 5 rounds on 100x50 from scratch. Final Cl against 0.5193 and Cp RMS against FVM 800x400.

| Mode | Method | Seed 1 | Seed 2 | Seed 3 | Seed 0 (earlier) |
|---|---|---|---|---|---|
| positivity off | grid-sequenced | Cl 0.504 (RMS 0.0162) | 0.507 (0.0154) | 0.509 (0.0147) | 0.512 (0.0148) |
| positivity off | cold start | 0.511 (0.0148), 1 failed round | 0.518 (0.0153), 1 failed round | **collapsed: all 5 rounds failed** | not run |
| positivity on | grid-sequenced | 0.516 (0.0145) | 0.513 (0.0146) | 0.513 (0.0151) | 0.502 (0.0162) |
| positivity on | cold start | 0.507 (0.0154) | 0.501 (0.0165) | 0.510 (0.0145) | 0.516 (0.0145), 20 rounds |

Pass criteria (set by the research session): **(1) grid sequencing gives Cl > 0.49 for all 3 seeds: yes, in both modes
(0.504-0.516).** **(2) the cold start collapses in at least 1 of 3 seeds: yes without positivity (1 of 3), no with it
(0 of 3).** So the cold-start collapse is real but unreliable: the original method collapsed once in three seeds, and our
positivity change never collapsed in four seeds (0-3). That is the first evidence in positivity's favour. It is still only
3-4 seeds on one mesh, and the same change hurt on 50x25 (4c), so E1 must settle the default.

Grid sequencing is reliable, but it is not shown to be faster here: its coarse stage costs a 50x25 run. Run times of this
check are wall-clock and some runs were stalled by the machine (one first round took 1536 s), so they are not quoted.

## 5. Flow-condition findings (the bundled data is not all Mach 0.4 and 4 deg)

- **NACA 2412 sweep files** (`FVM_Cp_NACA2412/*AoA*`): the panel method fits them at **Mach about 0.25** (**fitted**, steps
  of 0.05), for example RMS 0.0121 at 4 deg (S, experiment B0), not at Mach 0.4 (RMS 0.036 there, quoted from the
  roadmap). The paper's parametric case is quoted as Mach 0.3 in `docs/nnfoil_improvements.md` (K8); this session
  could not read the PDF text, so the Mach number is **unconfirmed**. Rows are labelled "fitted".
- **Random airfoils 1-5**: the geometry matches the meshes, but the best-fit conditions differ per airfoil (Airfoil1
  about Mach 0.25 at 6 deg, Airfoil5 about Mach 0.45 at 4.5 deg; fitted, roadmap; Airfoils 2 and 3 fit poorly). No
  comparison is made until the paper's conditions are known.
- **Cylinder** (alpha 0, Mach about 0.4): **superseded by E5 (Milestone 3):** an earlier version of this note quoted an RMS of about 0.30 for the
  panel method with Prandtl-Glauert. The measured value is **0.188** (0.307 is the incompressible circle). The PINN
  measured later: RMS 0.134 to 0.140 after 10 rounds on 50x25 (`results/summary.csv`, experiment E5). Minimum Cp
  -3.70 (FVM) against -3.27 (panel + PG).

## 6. Next steps

- Settle the positivity default with E1 (2 seeds, 20 rounds): it hurt on 50x25 and protected 100x50 in the seed check.
- **Milestone 2 (Plan B):** demo v2 with drawn shapes, a live velocity field and a polar.
- **Milestone 3 (Plan C):** PINN results in the app, experiments E1-E5, the 200x100 mesh, the cylinder, a Kutta/circulation
  remedy for the drift in 4(b) (for example a panel-method prior), and the corrected mesh generator (finding 3).
