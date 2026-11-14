# NNfoil verification table

NNfoil is by Cao, Song and Zhang (arXiv 2401.08705; no licence, kept private). This page records, component by
component, what the theory says, what the vendored code does, and how that was checked. Verdicts are one of
**correct**, **bug (fixed in `<commit>`)**, **bug (documented, not changed)** or **limitation**.

## How to read this

- Code references are `file:line` in the **current** `NNfoil/` files. `u:NNN` is the line in the upstream file at
  commit 4732498 (`git show 4732498:NNfoil/NNfoil_Euler2D.py`), which is what `NNfoil/AERONET_CHANGES.md` cites.
- Every number below comes from a file in `results/` or from a named test, and is labelled **measured** (produced by
  code in this repo), **quoted** (copied from another document, not re-run here) or **fitted** (none are used).
  Anything not labelled is a formula, not a number.
- Test names are in `research/tests/`: `test_nnfoil_core.py`, `test_nnfoil_checks.py`, `test_metrics.py`,
  `test_verification_evidence.py`. Run them with `research/.venv/Scripts/python -m pytest research/tests -q`.
- Regenerate the two CSVs with `research/.venv/Scripts/python research/check_mesh_smoothing.py` and
  `research/.venv/Scripts/python research/check_jacobian.py`.
- The paper was not re-read for this page; statements about "the paper" are limited to the abstract in
  `NNfoil/Readme.md`.

## Where this is recorded

- Milestone reports: [`M1-nnfoil-verified.md`](milestones/M1-nnfoil-verified.md) (the findings, CPU cost, first results and the
  seed check) and [`M3-pinn-results.md`](milestones/M3-pinn-results.md) (experiments E1 to E6, which measure the accuracy
  effects this page leaves open). The vendored file changes are listed in [`../NNfoil/AERONET_CHANGES.md`](../NNfoil/AERONET_CHANGES.md).
- Commits (this repository, `git show <hash>`), all dated 2026-11-05:

| Commit | What it did |
|---|---|
| `4732498` | upstream NNfoil vendored unmodified (the `u:NNN` line numbers refer to this commit) |
| `6b10ac1`, `12ba8c1`, `1bed181`, `c77caa2`, `c475030`, `0822cad` | the fixes for findings 1, 2, 4, 5, 6, and 7 with 8 |
| `77eeaf4` | this table, with the evidence for findings 3 and 10 (documented, not changed) |
| `da138d8`, `1470816` | warm start that survives a failed first round, and grid sequencing (`--init-from`) |

## The table

| Component | What the theory says | What the code does (file:line) | Test or evidence | Verdict |
|---|---|---|---|---|
| Euler e1: continuity | Steady 2D continuity: div(rho V) = 0. Product rule: rho u_x + u rho_x + rho v_y + v rho_y = 0. | `e1 = rho*u_x + u*rho_x + rho*v_y + v*rho_y`, `NNfoil_Euler2D.py:210` (u:194). | Identical term by term. `test_e1_e4_equal_conservative_euler_residuals`: e1 equals the conservative residual C1 = d(rho u)/dx + d(rho v)/dy (identity asserted below 1e-9). | **correct** |
| Euler e2: x-momentum | Steady x-momentum, conservative: d(rho u^2 + p)/dx + d(rho u v)/dy = 0. Expand and subtract u times continuity: rho (u u_x + v u_y) + p_x = 0. Divide by rho. | `e2 = u*u_x + v*u_y + 1/rho*p_x`, `:211` (u:195). | Same test: Cx = u e1 + rho e2, so e2 = (Cx - u C1) / rho matches the code's e2 (asserted below 1e-9). | **correct** |
| Euler e3: y-momentum | d(rho u v)/dx + d(rho v^2 + p)/dy = 0, i.e. u v_x + v v_y + p_y / rho = 0 after the same steps. | `e3 = u*v_x + v*v_y + 1/rho*p_y`, `:212` (u:196). | Same test: Cy = v e1 + rho e3 (asserted below 1e-9). | **correct** |
| Euler e4: pressure (energy) equation | Perfect gas, e = p / ((gamma-1) rho). Energy: rho De/Dt + p div V = 0. With continuity D rho/Dt = -rho div V this gives Dp/Dt + gamma p div V = 0. Steady: **V . grad p + rho a^2 div V = 0** with a^2 = gamma p / rho. | `a2 = gamma*p/rho` (`:209`, gamma = 1.4 at `:193`); `e4 = rho*a2*u_x + rho*a2*v_y + u*p_x + v*p_y` (`:213`, u:197). Matches the equation. | `test_e1_e4_equal_conservative_euler_residuals`: with CE = div((rho E + p) V), the identity e4 = (gamma-1)(CE + 0.5 \|V\|^2 C1 - u Cx - v Cy) holds on a manufactured field (asserted below 1e-9). Whole `Msef` on an analytic polar mesh agrees with the independent conservative evaluation to a relative difference below 1e-3 (the test bound; ~5e-5 was observed when written, measured). `test_e4_needs_the_gamma_in_a2`: dropping gamma breaks the match (sensitivity check). | **correct** |
| Flow regime (smooth, subsonic) | The quasi-linear (non-conservative) form equals the conservative Euler equations only where the solution is differentiable. Shocks need Rankine-Hugoniot conditions that this form cannot enforce. | Residuals use point derivatives of a smooth tanh network (`:187-221`). No entropy fix, no shock capturing. The paper's title and `Readme.md` abstract say "subsonic". | Reading only; no transonic case run. | **limitation** |
| Mapping (xi, eta) to (x, y): wall and faces | The wall is the coordinate line eta = 0 of the mesh. xi runs counter-clockwise (from the trailing edge (1, 0) on the bundled meshes), closed by repeating row 0. Face outward normal for a counter-clockwise polygon is (dy, -dx) / dl. | Wall nodes = mesh column 0 (`:89-91`, `:136`); face centres `X_f`, lengths `dl_f`, `tan_f = -dX/dl`, `nor_f = (-tan_y, tan_x)` (`:139-142`, u:128-129). The CST deformation branch (`:131-135`) is dead: `use_cst = False` is hard-coded (`:74`) and the branch would also mis-split the N+1 closed rows. | `test_surface_mapping_wall_is_mesh_column_0` (max difference below 1e-6); `test_outward_normals_for_every_face` (all faces outward). | **correct** |
| Metric Jacobian and derivative transform (`util.jacobian_trans`) | Chain rule: [u_xi; u_eta] = Jac [u_x; u_y] with Jac = [[x_xi, y_xi], [x_eta, y_eta]], so grad u = Jac^-1 grad_xi_eta u. det Jac = x_xi y_eta - x_eta y_xi must be non-zero and single-signed. xi is periodic. | Seven-point stencil, nodes -3..3 (`:147`); `jacobian_trans(..., boundary=(1,1))` (`:151`, `util.py:51`, u:135) uses **one-sided** stencils at xi rows 0 and N. Opt-in fix: `periodic_jacobian=True` (`:148-149`, `util.py:61`). | Chain rule: `Msef` matches an independent autograd derivative in (x, y) (previous rows). Seam: `results/finding10_jacobian.csv` (measured): all 9 bundled meshes have 1 or 2 cells with the wrong sign, always at eta column 0 (the trailing-edge wall node) and xi row 0 (and N where 2); with periodic differencing 0 wrong-sign cells on all 9. Max abs of Jac^-1 falls from 9.3e4 to 2.9e3 on 50x25 (same CSV). Tests: `test_finding10_periodic_flag_removes_wrong_sign_cells`, `test_finding10_all_bundled_meshes`, `test_default_jacobian_is_the_upstream_call`. det Jac is negative on every normal cell (xi counter-clockwise, eta outward), so "wrong sign" means "differs from the majority". | **bug (documented, not changed)** |
| Jacobian-weighted residual (J^2 weights) | Any positive weights give a valid residual minimisation. The physical-domain L2 norm would weight each cell by its area, J. Because J Jac^-1 = adj(Jac) is finite, J^2 \|R\|^2 is also finite where J tends to 0. | `msef = sum(\|R\|^2 J^2) / sum(J^2)` over the batch (`:219`, u:203). | Algebra above (explains why the seam cell's large Jac^-1 does not blow up the loss). Weight spread, `results/finding10_jacobian.csv` (measured): the inner quarter of eta rows carries 3.6e-6 to 6.6e-6 of sum J^2, the outer quarter 0.980 to 0.983 on all 9 meshes. So the PDE loss is almost entirely the outer region. Whether this is what the paper intends was not checked; the weighting is upstream's choice. | **limitation** |
| Loss composition | A penalty method only enforces BCs as the weight tends to infinity; relative weights are heuristic. | `loss = 2e4 * Msef + (mseb1 + mseb2 + mseb3)` with unit weights on the BC terms (`:238-241`, u:213). 2e4 is hard-coded. | Reading. `test_freestream_has_zero_residual_and_forces`: freestream gives Msef = 0 and a zero far-field term, but a non-zero wall term, so the terms respond as expected. | **limitation** |
| Far-field BC | Subsonic far field: freestream, with a vortex correction for a lifting body (disturbance decays like Gamma / (2 pi r)). Characteristic BCs fix 3 variables at inflow and 1 at outflow. | Dirichlet on all four variables, `(u, v, p, rho) = (cos a, sin a, 1/(1.4 Ma^2), 1)`, on the whole outer eta row (`:108-112`, `:158-161`, u:95-99, u:144). Outer radius is R = 15 chords (`mesh_gen.py:19`). | `test_freestream_has_zero_residual_and_forces`: the far-field term is 0 for a freestream model. `test_seam_and_far_field_collocation_points`: the far-field points are exactly the outermost eta row, all xi. No vortex correction and no characteristic treatment, so the outflow side is over-specified; the error is controlled only by the distance R. | **limitation** |
| Wall BC (zero normal velocity) | Slip wall: V . n = 0, n the unit normal. | `vn_f = u_f*nor_x + v_f*nor_y` at the N face centres of the eta = 0 row, penalised as `mean(vn_f^2)` (`:164-167`, u:151). Normals from the straight panels between mesh nodes. | Normals outward for every face (`test_outward_normals_for_every_face`). Freestream (not tangent to the wall) gives a non-zero wall term (`test_freestream_has_zero_residual_and_forces`). The condition is soft (a penalty) and holds only at face centres. | **correct** |
| Periodic seam (interface loss `mseb3`) | The mapping is periodic: U(xi = 0, eta) = U(xi = N, eta), and derivatives must be continuous across the seam. | `bond_left` = points (0, j), `bond_right` = points (N, j) (`:103-104`, u:90-91); `mseb3 = mse(U_left, U_right)` (`:170-175`, u:159). Values only; rows 0 and N are both collocation points, so seam cells count twice in the PDE loss. | `test_seam_and_far_field_collocation_points`: xi = 0 is paired with xi = N at the same eta, and those two mesh rows coincide. Derivative continuity is not enforced, and the Jacobian at the seam is the finding 10 issue. | **limitation** |
| Non-dimensionalisation | Scales: length = chord, rho_inf, V_inf; pressure by rho_inf V_inf^2. a_inf^2 = gamma p_inf / rho_inf and Ma = V_inf / a_inf give p_inf = rho_inf V_inf^2 / (gamma Ma^2). With rho_inf = V_inf = 1: **p_far = 1/(gamma Ma^2)**. | `far['p'] = 1/(1.4*Ma^2)`, `far['rho'] = 1`, `far['u'] = cos a`, `far['v'] = sin a` (`:109-112`, u:96-99); chord = 1 because the wall x runs over [-0.5, 0.5]. gamma = 1.4 is a literal in several places (`:111`, `:168`, `:183`, `:193`, `:232`, `:234`), all consistent. | `test_cp_formula_is_pressure_coefficient`: p_far = 1/(1.4 * 0.4^2) and \|V_inf\| = 1 to 1e-6. `test_flow_condition_arguments`: Ma = 0.3, alpha = 2 deg gives the expected far values. | **correct** |
| Cp formula | Cp = (p - p_inf) / (0.5 rho_inf V_inf^2), which is (p - p_far) / 0.5 in these units. | `Cp = (p - 1/(1.4*Ma^2)) / 0.5` (`:168`, `:183`, `:232`, `:234`; u:152, u:167). | `test_cp_formula_is_pressure_coefficient`: p - p_far = 0.5 gives Cp = 1 (stagnation), -0.25 gives -0.5, 0 gives 0 (atol 1e-5). `test_cp_error_panel_vs_fvm_800`: the Ma 0.4 panel Cp and the 800x400 FVM file agree to RMS 0.0099 +/- 0.0005 over n = 654 points (asserted in the test; measured), so the file uses the same quantity. | **correct** |
| Force integration (`Calculate_Force`) | Force coefficient vector = -integral of Cp n dl / c (p_inf drops out on a closed body). Cl = F . (-sin a, cos a), Cd = F . (cos a, sin a). The true inviscid subcritical Cd is 0 (d'Alembert), so a reported Cd is numerical error. | `Force = Cp_f * dl_f * nor_f`; `Cl = -(cos a F_y - sin a F_x)`, `Cd = -(sin a F_y + cos a F_x)` (`:223-226`, u:207-210): midpoint rule on the mesh polygon. | `test_calculate_force_matches_metrics_convention` (1e-5 against `metrics.forces_from_faces`); `test_panel_cp_through_forces_matches_panel_cl_cd` (the same convention reproduces the panel method's Cl and Cd to 1e-9); `test_uniform_cp_gives_zero_forces`; freestream gives Cl = Cd = 0; `test_fvm_forces_cl`: Cl 0.5193 for FVM 800x400 (asserted within 5e-4; measured). | **correct** |
| Kutta condition and lift | A sharp trailing edge needs the Kutta condition to fix circulation, hence Cl. FVM gets it from numerical dissipation. | None is imposed. Circulation is selected implicitly by network smoothness and the value-only seam condition. | Reading (not in the plan; added in A9). | **limitation** |
| Parametric angle of attack | The abstract claims an advantage on parametric problems (continuous angle of attack). | Network input is (xi, eta) only (`:99-100`); Ma and alpha are constructor arguments, one condition per model. | `Readme.md` abstract (quoted); `test_flow_condition_arguments`. | **limitation** |
| Mesh generator (finding 3) | Winslow/Thompson: alpha x_xixi - 2 beta x_xieta + gamma x_etaeta = 0. The Jacobi update used here has the fixed point alpha X_xixi - c beta X_xieta + gamma X_etaeta = 0 with central stencils, so c must be 2. | `mesh_gen.py:66,68` use `0.5 * beta * diff2d(dXdxi, nodes, 0)`, c = 0.5. Not changed: the bundled meshes and the FVM data came from this code. | `results/finding3_mesh_smoothing.csv` (measured, 50x25): c = 0.5 takes 3287 iterations, max \|Winslow residual\| 1.95e-4, min grid angle 85.98 deg, Jacobian single-signed, max distance to the bundled mesh 3.3e-4 (so the bundled mesh came from this code; the bundled mesh itself has residual 1.95e-4). c = 2 takes 3290 iterations, residual 1.99e-6, min angle 86.85 deg, single-signed, distance 0.011. Low impact: both meshes are valid. Tests: `test_finding3_*`. Corrected generator (cross coefficient 2): `research/ogrid.py`, used for the cylinder and NACA 2412 meshes in `results/meshes/`. | **bug (documented, not changed)** |
| Finding 1: device | Code must run on a machine without CUDA. | Upstream forced `torch.device("cuda:0" if 1 else "cpu")` (u:269). Now `pick_device()` (`:21-22`, used at `:301`). | `test_pick_device_falls_back_to_cpu`. | **bug (fixed in 6b10ac1)** |
| Finding 2: positivity of p and rho | a^2 = gamma p / rho and 1/rho need p > 0 and rho > 0. | Upstream used raw outputs (u:193-197). Now `softplus(.) + 1e-3` on p and rho, flag `positivity` default on (`:45-47`). The floor and softplus change the function class; their effect on accuracy is for Plan C to measure. | `test_positivity_keeps_p_rho_above_floor`, `test_positivity_off_is_raw`. | **bug (fixed in 12ba8c1)** |
| Finding 4: CPU threads | Not a numerical issue: single-threaded runs are several times slower on a 16-thread machine. | Upstream `torch.set_num_threads(1)` (u:265). Now `min(8, os.cpu_count() or 1)` in `__main__` (`:297`). | No test (a runtime setting). | **bug (fixed in 1bed181)** |
| Finding 5: factorial | `np.math` was removed in NumPy 2. | Upstream `np.math.factorial` (u:110). Now `math.factorial` (`:123`). | `test_nnfoil_constructs_on_numpy2`. | **bug (fixed in c77caa2)** |
| Finding 6: checkpoints | Whole-module `torch.save` and `torch.load` fail under `weights_only=True` (PyTorch 2.6 default) and the fixed file name collides between runs. | Upstream u:247, u:250. Now `_save_ckpt`/`_load_ckpt` with `state_dict` and a per-run `ckpt_path` (`:245-249`). `X_mean` and `X_std` are not in the state dict but are recomputed from the mesh shape in `Net.__init__`, so a reload on the same mesh is exact. | `test_checkpoint_roundtrip`. | **bug (fixed in c475030)** |
| Finding 7: forces never computed | Cl and Cd are the outputs the project reports. | Upstream never called `Calculate_Force()` (u:207-210). Now `evaluate_wall()` recomputes Cp and calls it (`:228-236`). | `test_evaluate_wall_returns_finite_floats`; the force row above. | **bug (fixed in 0822cad)** |
| Finding 8: hard-coded flow condition | Ma and alpha must be inputs for any case other than Ma 0.4 and 4 deg. | Upstream u:55 hard-coded both. Now constructor arguments `Ma`, `alpha_deg` (`:68`). Still one condition per model (see the parametric row). | `test_flow_condition_arguments`. | **bug (fixed in 0822cad)** |
| Weight initialisation (finding 11, new in A9) | `init.normal_` on the weight is meant to give N(0, 1) weights. | `torch.nn.init.normal_(temp[l-1].weight)` after `weight_norm` (`:35-36`, u:29-30). The `weight` attribute is recomputed from `weight_g` and `weight_v` on every forward pass, so the line is overwritten and the effective init is PyTorch's default. | `test_normal_init_after_weight_norm_is_a_noop`: before the first forward the attribute has std above 0.7; afterwards it equals g * v / \|v\| and has std below 0.3. Changing it would alter every result, so it is only documented (`NNfoil/AERONET_CHANGES.md`, finding 11). | **bug (documented, not changed)** |

## Algebra for the Euler rows

With V = (u, v), D/Dt = V . grad in the steady case.

- Continuity: C1 = div(rho V) = e1.
- x-momentum: Cx = d(rho u^2 + p)/dx + d(rho u v)/dy = u C1 + rho (u u_x + v u_y + p_x / rho) = u e1 + rho e2.
- y-momentum: Cy = v e1 + rho e3, likewise.
- Energy, with total enthalpy H = E + p / rho: CE = div(rho H V) = (\|V\|^2 / 2) e1 + rho (u e2 + v e3) + e4 / (gamma - 1).
  The last term comes from rho V . grad h with h = gamma p / ((gamma - 1) rho) and V . grad rho = e1 - rho div V.
- Solving for e4: e4 = (gamma - 1)(CE + (\|V\|^2 / 2) C1 - u Cx - v Cy). This is the identity the test uses.

## Reference data conditions

Read from the paper's PDF text (arXiv 2401.08705, text extracted with `pdftotext`; the Greek symbols and degree signs
were lost in the extraction, so angles are read from the surrounding sentence). All values in this section are
**quoted** from the paper; nothing was re-run. The earlier panel-method fit of the Mach number is labelled **fitted**.

| Bundled data | Condition in the paper | Where | Check against our fit |
|---|---|---|---|
| `FVM_Cp/NACA0012_*` | Ma 0.4, angle of attack 4 degrees, meshes refined from 50x25 to 400x200 | sec 3.2 | panel method matches the 800x400 file at Ma 0.4 (measured, `results/summary.csv`, B0) |
| `FVM_Cp/Cylinder_*` | "Ma = 0.4" with angle of attack 0 (the cylinder is the thickest ellipse) | sec 3.1 | agrees with the fitted stagnation point and Cp (measured earlier, roadmap finding 4) |
| `FVM_Cp_NACA2412/*AoA*` | NACA 2412, "Ma = 0.3", angle of attack in [-10, 10] degrees | sec 3.4 | our panel fit preferred Ma 0.25 (**fitted**; Cp RMS 0.0121 at 0.25, 0.0142 at 0.3, 0.0332 at 0.4 for the 4-degree file). The paper's 0.3 is used for Plan C; the small difference is inside what a Prandtl-Glauert panel method can resolve |
| `FVM_Cp_random_airfoil/Airfoil1..5` (Table 2 cases 1 to 5) | (Ma, angle of attack): 0.28 and 6.1; 0.24 and -3.0; 0.41 and -3.5; 0.68 and 1.8; 0.34 and 4.9 | sec 3.3, Table 2 | our fits agreed only roughly (Airfoil1 about 0.25 and 6 degrees; Airfoil5 about 0.45 and 4.5 degrees); use the paper's values. Case 4 (Ma 0.68) has a shock that NNfoil does not capture, per the paper |

FVM and NNfoil wall times quoted from Table 1 (NACA 0012, 4 degrees, Ma 0.4): FVM 9 s, 24 s, 53 s, 285 s and NNfoil
384 s, 266 s, 262 s, 271 s for 50x25, 100x50, 200x100, 400x200. The paper says the table "is not meant as a rigorous
comparison" and that LBFGS often stops early. Parametric NNfoil on NACA 2412 (sec 3.4): training wall time 1543 s.

## Notes on the evidence

- `test_e1_e4_equal_conservative_euler_residuals` builds an analytic polar mesh (64 x 30, radii 0.5 to 6) and a smooth
  manufactured field that is not an Euler solution, so every residual is non-zero. It compares `Msef()` with the
  residual computed independently from autograd derivatives in (x, y) and the conservative equations.
  The remaining difference comes from the finite-difference Jacobian.
- The weight-spread numbers and the wrong-sign counts are in `results/finding10_jacobian.csv`; the mesh-smoothing
  numbers are in `results/finding3_mesh_smoothing.csv`. Timings in that CSV are machine-dependent and are not quoted.
- The accuracy effects were measured afterwards: positivity on or off is experiment E1 in [`M3-pinn-results.md`](milestones/M3-pinn-results.md)
  (neither mode wins; positivity only changes when the drift starts). No training run in M3 varies `periodic_jacobian` or the J^2 weights. Nothing on
  this page claims an accuracy result for the trained network.
