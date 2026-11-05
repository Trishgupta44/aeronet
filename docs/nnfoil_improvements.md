# NNfoil: verified issues and evidence-backed improvements (CPU-only)

NNfoil is by **Wenbo Cao, Jiahao Song and Weiwei Zhang**: "A solver for subsonic flow around airfoils based on physics-informed neural networks and mesh transformation", *Physics of Fluids* 36, 027134 (2024), arXiv:2401.08705, code at github.com/Cao-WenBo/NNfoil. The repo has **no licence**, so all of it stays credited to them. This document changes none of their code. Every fix below was tried only in a scratch copy outside the repo.

## How to read this

Each finding says how it was checked:

- **RAN**: checked by running code (a scratch copy of the script, a mesh script, or a small test).
- **READ**: checked by reading the code (and, where stated, PyTorch's own source).
- **PAPER**: quoted from the paper (arXiv v1 PDF).
- **EST**: an estimate, not measured.

Line numbers refer to the files as they are in `NNfoil/` today.

**Test setup (RAN):** Windows 11, 16 logical CPU threads, no GPU. Python 3.13.7 venv outside the repo (the registered Python 3.12 on this machine points to a missing `python.exe`). torch 2.15.0+cpu from download.pytorch.org/whl/cpu, numpy 2.5.4, scipy 1.18.1. The scratch copy (`nnfoil_bench.py`) has only these changes: CPU device, a thread-count flag, `math.factorial`, an in-memory "best model" copy in place of `torch.load`, a call to `Calculate_Force()` after every round, and extra logging. Loss, network, optimiser, sampling and seeds are unchanged.

**Caveat on timings:** from 19:45 to about 20:15 another session ran NNfoil training on the same CPU. The 50×25 10-round run (section 2b) overlapped with it, so its seconds per round are pessimistic and noisy (75 to 527 ms per iteration). The thread scan (2a) ran before that, and the 100×50 runs ran on an otherwise idle machine.

---

## 0. Implementation brief (for the coding model: read this first)

**Goal:** a CPU-runnable, fixed NNfoil that reports Cl, Cd and Cp error against FVM, writes JSON for the demo, and can mesh a user-drawn shape. Finding IDs (K1, N6, ...) point to the evidence tables in section 1.

**Ground rules**
- Do not edit `NNfoil/` (the authors' code; it has its own nested `.git`). Put the fixed copy in a new folder, for example `aeronet_solver/`: copy `NNfoil_Euler2D.py` → `aeronet_solver/nnfoil_cpu.py`, `util.py` → `aeronet_solver/util.py`, and `mesh_generation/differ.py` → `aeronet_solver/differ.py`. Keep a header comment crediting Cao, Song and Zhang (2024) and noting the source has no licence.
- Read meshes and FVM files from `NNfoil/mesh/` and `NNfoil/FVM_Cp*/` in place. Do not copy or regenerate them (the FVM reference was computed on those exact meshes, K3).
- Every number written to the report or JSON must come from a run. Label anything else "estimated" or "from paper".
- Default test case: `NACA0012_50_25` (fast), Ma 0.4, α 4°. Reference Cl = **0.519** (FVM 800×400, integrated from the shipped Cp file, section 2c).

**Task 1: runnable baseline (Must, about 1 h).** In `nnfoil_cpu.py`:
1. Line 269: `torch.device("cuda" if torch.cuda.is_available() else "cpu")` (K1).
2. Line 265: `torch.set_num_threads(args.threads)` with default 8 (K4, measured fastest).
3. Line 110: `f = math.comb(self.degree, index)` and `import math` (K5).
4. Lines 241-250: replace `torch.save`/`torch.load('model_temp.pth')` with `best = copy.deepcopy(self.model.state_dict())` / `self.model.load_state_dict(best)` (K6).
5. Line 86: add `indexing='ij'` (N20).
6. Add argparse: `--mesh --rounds --max_iter --threads --dtype --seed --out --Ma --alpha_deg`. Make `Ma` and `alpha` constructor arguments (K8, single-condition part).
7. After every round: `with torch.no_grad(): self.Mseb(); self.Calculate_Force()`. Append `Cl`, `Cd` and the round's seconds to `self.log` (K7).
- **Check:** `python aeronet_solver/nnfoil_cpu.py --mesh NACA0012_50_25 --rounds 3` runs on CPU with no warnings except the `weight_norm` FutureWarning, prints Cl per round, and Cl is positive. Compare with the section 2b numbers.

**Task 2: validation and JSON export (Must, about 1-2 h).** New `aeronet_solver/validate.py` plus a `save_result(path)` method:
1. Cp relative L1 error ‖Cp − Cp_ref‖₁ / ‖Cp_ref‖₁ at the PINN wall nodes. Cp_ref comes from `NNfoil/FVM_Cp/NACA0012_800_400_Cp.dat`, split into upper (y ≥ 0) and lower (y < 0), each sorted by x and interpolated with `np.interp`. The TE node is matched by nearest point. This is the paper's metric (N26).
2. Cl relative error against 0.519, plus the same-mesh FVM Cl (match FVM rows to wall nodes by nearest (x, y); distances are under 1e-10 for 50×25 to 400×200).
3. Write the JSON schema in N27 (meta, timing, loss log, Cl, Cd, wall Cp, full field, fvm_ref).
- **Check:** applying the Cl routine to the FVM file itself gives 0.4265 (50×25) and 0.5193 (800×400). These were measured by this research, so the routine is right only if it reproduces them.

**Task 3: robust training loop (Must, about 1 h).**
1. After `optimizer.step(closure)`, compute `loss_after = self.Loss()[0].detach()` and use it, not `self.loss`, in the NaN, 1e-15 and 4× checks. Also check `all(torch.isfinite(p).all() for p in self.model.parameters())` (N1).
2. Run the 4× rule on a **fixed** validation loss: the PDE+BC loss on the points not in `batch_ind`, or the full grid (N2).
3. `history_size=100` (N3).
4. Add a `--dtype float64` option: set `torch.set_default_dtype` and replace every `.float()` with `.to(torch.get_default_dtype())` (N8). Measured: float64 alone does not fix N28 and costs 1.3-2.3× per closure, so keep float32 as the default and use float64 only as a flag.
5. Exclude the two TE wall nodes (index 0 and M−1 at η = 0) from `batch_ind` (N4).
6. **Escape the trivial-solution collapse (N28, the most important item, measured on 100×50).** If `loss_f < 1e-12` in *any* round, or the loss has not dropped by at least 1% over 2 rounds, rebuild the network with a new seed (`torch.manual_seed(seed+k)`) instead of reloading, and cap this at 3 retries per run. Add `--adam_warmup 2000` (Adam, lr 1e-3, on the same loss) before the first L-BFGS round. Log every restart.
- **Check:** 5 rounds on 50×25 finish with no "Failed" lines, and the log shows `loss_after`. **Then the real test: 10 rounds on 100×50 must end with Cl > 0.45.** The unfixed code ends at Cl ≈ 0 (measured). If it still collapses, try `--periodic 1 --hardfar 1`, which did not collapse (measured), and report what happened.

**Task 3b: grid sequencing (Must, about 1 h; measured to fix the 100×50 collapse).** Add `--init_from <coarse_mesh> --coarse_rounds 5`:
1. Train on the coarse mesh (for example `NACA0012_50_25`) for the given rounds.
2. Build the solver on the fine mesh and `load_state_dict` the coarse weights.
3. Rescale the input normalisation so the same *relative* grid position gives the same network input: `X_mean_f = X_mean_c * s`, `X_std_f = X_std_c * s`, with `s = [(M_f−1)/(M_c−1), (N_f−1)/(N_c−1)]` (M = ξ rows including the closing row, N = η columns). This works because every shipped mesh uses the same θ-spaced wall and the same smoother, so a relative index (ξ/(M−1), η/(N−1)) is nearly the same physical point at every resolution.
4. Skip the "first round → re-initialise" branch for the fine mesh, so the warm start is not thrown away.
- **Check (measured target):** 50×25 → 100×50 gives Cl = 0.504 right after the transfer, before any fine training, and **Cl = 0.5065, Cp-L1 = 0.027 after 4 fine rounds, 302 s in total**. Then try the chain 50×25 → 100×50 → 200×100 (the paper's main mesh; not measured).

**Task 4: accuracy options (Should, about 2 h, each behind a flag so it is an ablation).**
1. `--periodic 1`: input `[cos(2πξ/(M−1)), sin(2πξ/(M−1)), (η−mean)/std]`, `layers[0]=3`, and drop `mseb3` (N6).
2. `--hardfar 1`: `out = q_inf + (1 − η/η_max)·net(x)`, with `q_inf = [cosα, sinα, 1/(1.4Ma²), 1]`, and drop `mseb1` (N10). This also covers K2's "start at freestream".
3. `--weight {J2,absJ}` for the residual weight (N5). Keep J2 as the default (the paper's method).
4. `X_mean`/`X_std` as `register_buffer` (N15) so the demo can load a `state_dict`.
- **Check:** each flag runs 5 rounds on 50×25. Compare Cl and Cp-L1 with the baseline in a small table (3 seeds if time allows, N18). Measured on 100×50 (N6+N10 together): no collapse, but Cl 0.271 against 0.519. Run `--periodic` and `--hardfar` **separately** on 50×25, where the baseline works (Cl 0.4975), and keep a flag only if it keeps Cl within 5% and lowers Cp-L1. Report the TE Cp jump (N22) for each.

**Task 5: mesh for a user-drawn shape (Should, about half a day).** New `aeronet_solver/make_mesh.py` following the 8 steps in section 4:
- `make_mesh(wall_xy, N=50, M=25, R=15) -> X, Y` and `write_x(path, X, Y)` in the `.x` layout.
- Smoother cross-term coefficient **2** (K3). Stop when both X and Y converge. Add SOR (ω about 1.5-1.8) so 50×25 takes well under the measured 8 s Jacobi time (N23).
- `check_mesh(X, Y)`: use `util.jacobian_trans(..., boundary=(1,1))` on the NNfoil layout. Pass only if J has one sign everywhere except the TE node(s), with no near-zero J elsewhere (N4).
- **Check 1:** feeding the NACA0012 CST wall (from `mesh_gen.py`) with coefficient 0.5 reproduces `NNfoil/mesh/NACA0012_50_25.x` to about 3e-4 (measured). With 2 it differs by about 1e-2 and still passes `check_mesh`.
- **Check 2:** a circle through this pipeline, trained and compared with `NNfoil/FVM_Cp/Cylinder_50_25_Cp.dat`.

**Task 6 (Stretch): parametric α (K8).** Follow PAPER §3.4: input (ξ, η, α), α sampled per collocation point in a range (for example [−6°, 10°]), far-field u, v per point, 8 hidden layers. The paper needed 1543 s on an RTX 4090. On this CPU, expect several hours on 50×25 (EST from section 2b). Only attempt it after Tasks 1-4 give good numbers.

**Do not do:** regenerate the shipped meshes; change the default J² weighting or the 2e4 PDE weight without an ablation; add features to the authors' files in `NNfoil/`.

---

## 1. Summary table of findings

Effort: S = under 1 hour, M = half a day to a day, L = several days. Impact: Acc = accuracy, Spd = speed, Cap = capability, Rob = robustness (fewer crashes and NaNs).

### 1a. The nine known issues (from `docs/AeroNet_7_Day_Plan.md`, section 2), re-checked

| # | file:line | What is wrong | Evidence | Fix | Effort | Impact |
|---|---|---|---|---|---|---|
| K1 | `NNfoil_Euler2D.py:269` | `torch.device("cuda:0" if 1 else "cpu")` always picks CUDA. It crashes with no GPU. | READ. The scratch copy runs on CPU after this fix (RAN). | `torch.device("cuda" if torch.cuda.is_available() else "cpu")`. | S | Cap (it runs at all) |
| K2 | `NNfoil_Euler2D.py:193-197` | `rho` and `p` are raw linear outputs. They are divided by `rho` (`a2=gamma*p/rho`, `1/rho*p_x`). Nothing keeps them positive. | READ. No negative ρ or p was seen in the runs, but the failure that did happen is a different one (N28). | Predict a perturbation from the freestream and squash it: `rho = 1 + net_rho`, `p = p_inf*(1 + net_p)`, with a softplus floor (for example `rho = softplus(...)+1e-3`). The "hard far-field" form (finding N10) does the first part for free. | S | Rob |
| K3 | `mesh_generation/mesh_gen.py:66-69` | **Confirmed deviation from the Winslow/Thompson equations.** The cross term uses `-0.5*beta*X_xi_eta`. Here `alpha`, `beta`, `gamma` and the mixed derivative are all built from *scaled* central differences (stencil weights ±0.5, so `diff2d(diff2d(X,0),1)` already contains the 1/4). With scaled differences the standard Jacobi update is `x = [alpha(x_E+x_W) + gamma(x_N+x_S) - 2*beta*x_xi_eta] / (2(alpha+gamma))`, so the coefficient should be **2**, not 0.5 (4 times too small). 0.5 is the textbook value only when the mixed difference is the *unscaled* four-point sum. | RAN (`meshgen_test.py`). The shipped meshes were made with 0.5: re-running the smoother with 0.5 reproduces `NACA0012_50_25.x` to 3.2e-4 and `100_50` to 1.3e-3, while 2 differs by 1.1e-2 (chord = 1). The standard Winslow residual (max over the interior, scaled by alpha+gamma) of the shipped 50×25 mesh is 4.8e-4. With 2 it falls to 2.0e-6 (the solver tolerance). Both meshes are valid: the Jacobian has one sign everywhere except the trailing-edge node (see N4), and the grid lines stay close to orthogonal (mean \|cos\| 0.009 with 0.5 vs 0.010 with 2). | Use 2 when making **new** meshes (user-drawn shapes, where beta is larger). **Do not regenerate the shipped meshes**: the FVM reference Cp files were computed on them, so changing them would break the comparison. Report it as "checked: cross-term coefficient is 4 times too small, harmless for the shipped near-orthogonal O-grids". | S | Acc (new meshes only), mostly cosmetic |
| K4 | `NNfoil_Euler2D.py:265` | `torch.set_num_threads(1)`. | RAN, see section 2. 8 threads was fastest. 16 was slower than 8. | `torch.set_num_threads(8)` or a CLI flag. Do not use 16 (hyper-threads compete). | S | Spd |
| K5 | `NNfoil_Euler2D.py:110`, `mesh_gen.py:12`, `Airfoil2CST/cst_geom.py:31` | `np.math.factorial` is gone in NumPy 2. | RAN: numpy 2.5.4 raises `AttributeError: module 'numpy' has no attribute 'math'`. | `math.comb(degree, index)` (one call replaces the three factorials). | S | Cap |
| K6 | `NNfoil_Euler2D.py:247,250` | `torch.load('model_temp.pth')` of a whole pickled module fails on PyTorch ≥ 2.6 (`weights_only=True` by default). The file name is shared by every run in the folder, so two runs at once overwrite each other's rollback point. | RAN: torch 2.15 raises `UnpicklingError: Weights only load failed`. | Keep the rollback copy in memory: `best = copy.deepcopy(model.state_dict())`, then `model.load_state_dict(best)`. If a file is needed, save `state_dict()` to a per-run path (`f"ckpt_{run_id}.pt"`). | S | Rob |
| K7 | `NNfoil_Euler2D.py:207-210` | `Calculate_Force` is never called, so no Cl or Cd is reported. It also depends on `self.wall['Cp_f']`, which only `Mseb()` sets. | READ, then RAN (the copy calls `Mseb(); Calculate_Force()` under `no_grad` after each round; Cl values in section 2). | After each round: `with torch.no_grad(): nn.Mseb(); nn.Calculate_Force()`, and log `Cl` and `Cd` (the `log` dict already has an unused `'Cl'` key, line 49). | S | Cap |
| K8 | `NNfoil_Euler2D.py:55, 96-98, 267` | Mach 0.4 and α = 4° are hard-coded. The network input is (ξ, η) only. The paper's continuous-AoA model is not in this file. | READ. PAPER §3.4: input expanded to (ξ, η, α), NACA2412, Ma = 0.3, α ∈ [-10°, 10°], 8 hidden layers, 30,000 L-BFGS iterations, 1543 s on an RTX 4090, "about 5 times" one single-angle solve. | Constructor arguments for `Ma` and `alpha` (S). Parametric model (M to L): `layers[0]=3`, add α to every collocation and boundary point, make `far['u'] = cos(α_i)` and `far['v'] = sin(α_i)` per point, and normalise α into the same z-score. For Mach as an input, change the pressure scaling first (N11), because `p_inf = 1/(1.4 Ma²)` runs from 17.9 to 2.0 over Ma = 0.2 to 0.6. | S / L | Cap |
| K9 | `NNfoil_Euler2D.py:235-237, 267, 281` | Training is heavy for a CPU: 5×128 tanh, L-BFGS `max_iter=1000`, `history_size=1000`, 20 rounds = 20,000 iterations (PAPER: 20,000 total iterations on an RTX 4090). | RAN, see section 2. | Section 2 and the top 5. | — | Spd |

### 1b. New findings (beyond the known list)

| # | file:line | What is wrong or improvable | Evidence | Fix | Effort | Impact |
|---|---|---|---|---|---|---|
| N1 | `NNfoil_Euler2D.py:236-248` | **The NaN check can accept a NaN model.** With `line_search_fn=None`, PyTorch's L-BFGS does *not* re-evaluate the closure after the step on the last iteration (`torch/optim/lbfgs.py`: "re-evaluate function only if not in last iteration"). So `self.loss` and `self.loss_f`, used at lines 241 and 245, belong to the parameters *before* the final step. If the final step makes the weights NaN or blows them up, the check passes and line 250 saves that model as the rollback point. | READ (torch source, lbfgs.py lines 471-478 in torch 2.15). RAN: the copy logs a fresh loss at the final parameters (`loss_after`). Measured: the final step does change the loss (for example 1.0664e-3 → 1.0617e-3, and 3.66e-2 → 4.15e-2 on 100×50). No NaN happened in these runs, so the unsafe path was not triggered, but it exists. | After `optimizer.step(closure)`, evaluate `loss_after = self.Loss()[0]` once (it costs one closure), and run the NaN and 4× checks on that value, plus `all(torch.isfinite(p).all() for p in model.parameters())`. | S | Rob |
| N2 | `NNfoil_Euler2D.py:235, 245` | The "loss jumped 4×, roll back" rule compares losses on **different random collocation subsets**, because `batch_ind` is redrawn every round (line 235). Some of the jump is sampling noise, not divergence. | READ. | Compare on a fixed validation set: use the 20% of points not drawn that round, or a fixed full-grid evaluation every round. | S | Rob |
| N3 | `NNfoil_Euler2D.py:236-237` | `history_size=1000` is 10 times the PyTorch default. L-BFGS's two-loop recursion is a Python loop over the history, so the cost per iteration grows through each 1000-iteration round. `tolerance_change=1e-10` is below float32 resolution (about 1e-7 relative), so in float32 that stop rule never fires. | RAN: READ, plus EST: the first 100 iterations cost 48 ms each (8 threads, 50×25), while whole 1000-iteration rounds averaged 75-150 ms per iteration. The history-100 timing run was cancelled to make room for N28, so the share caused by history length is **not measured**. | `history_size=50..100`. In float32, use `tolerance_change≈1e-9` on the absolute loss, or switch to float64 (N8). | S | Spd |
| N4 | `NNfoil_Euler2D.py:135` (metrics), mesh files | **Singular trailing-edge (TE) node.** On every shipped NACA0012 mesh the TE wall node (rows 0 and M-1, η = 0) has J ≈ 0 with the *opposite* sign to every other node: J = +2.1e-8 on 50×25, +3.4e-10 on 100×50, +5.8e-12 on 200×100, against J < 0 everywhere else. \|J⁻¹\| reaches 9.3e4 (50×25), 1.5e6 (100×50) and 2.5e7 (200×100) there. The physical-space residual at that node is meaningless. The J² weight (N5) keeps it finite (R ∝ 1/J, so R²J² stays O(1)), which is one reason the J² weighting "works". | RAN (`jsign.py`). | Drop the two TE wall nodes from the PDE collocation set (they keep the wall BC). If N5's weighting is ever changed, this exclusion becomes **required**. The negative sign everywhere else is just NNfoil's orientation: ξ runs counter-clockwise and η outward, so det < 0. For new meshes, check "one sign everywhere except the TE node", not "positive". | S | Rob, Acc near the TE |
| N5 | `NNfoil_Euler2D.py:203` | The PDE loss is weighted by J² / ΣJ² (the paper's "volume-weighted" residual, PAPER Eq. 9, citing Song et al., VW-PINNs). This is intended, but it is extremely lopsided on these O-grids: the inner 25% of η rows (the body and the leading-edge region you care about) get **4e-6 to 1.4e-5 of the total PDE weight**, and the outer 25% get **97-98%**. \|J\| varies by 2e8 (50×25) up to 7e12 (800×400). | RAN (`analyse_mesh.py`). Weighting by \|J\| would give the inner quarter 0.17-0.31%. | Do not change it silently, because it is the paper's method. Run an ablation: (a) J² (baseline), (b) \|J\|, (c) J² plus oversampling of the inner η rows. Keep N4's TE exclusion whenever J² is not used. | M | Acc (hypothesis, not measured) |
| N6 | `NNfoil_Euler2D.py:155-159` | The seam ξ = 0 / ξ = M-1 sits **at the trailing edge**, and the interface loss enforces only equal *values* across it (C0), not equal derivatives, even though ξ is periodic. | READ. | **Periodic embedding**: feed `[cos(2πξ/(M-1)), sin(2πξ/(M-1)), η_norm]` instead of the z-scored ξ. The network is then exactly periodic in value and all derivatives, and `mseb3` can be deleted. RAN together with N10 on 100×50 (one run, seed 0): 5 of 5 rounds succeeded (no collapse, unlike every other 100×50 run, N28), but Cl stalled at **0.271** (reference 0.519) and L-BFGS stopped after 17 and 6 iterations in rounds 3-4. It converged to a low-circulation solution (see N22). **Not yet a recommended default**: test N6 and N10 separately, with the Kutta diagnostic. | S | Acc, Rob |
| N7 | `NNfoil_Euler2D.py:135`; `util.py:82-102` | The ξ metrics use `boundary=1` (one-sided stencils) even though ξ is periodic (`util.diff2d` already supports `boundary=2`). | RAN: with periodic stencils (unique rows, then the closing row appended) J changes by at most 0.74% (50×25) to 0.38% (800×400), only on the 3 rows next to the seam. The rest is identical. | `jacobian_trans(x[:-1], y[:-1], nodes, boundary=(2,1))`, then append row 0. | S | Acc (small, at the TE) |
| N8 | `NNfoil_Euler2D.py:73-74, 112, 136-137` | Everything is float32 (the metrics are computed in float64 and then cast down). On a CPU, float64 costs only a small factor (unlike consumer GPUs), and L-BFGS curvature pairs are much better behaved. | RAN: RAN on 100×50: float64 did **not** stop the collapse (N28): rounds 1, 2 and 4 fell to PDE loss 0, final Cl 0.001. Cost per closure in float64 vs float32 (micro-benchmark, 8 threads): 1.3× at 1,020 points, 1.6-3.5× at 4,040 points, 2.3× at 16,080 points. Whole rounds on 100×50: 178 ms per iteration (float64) vs about 95 ms (float32). | `torch.set_default_dtype(torch.float64)` and replace `.float()` with `.to(torch.get_default_dtype())`. | S | Rob, Acc |
| N9 | `NNfoil_Euler2D.py:236` | No line search: L-BFGS takes fixed unit steps. Many PINN codes use `line_search_fn="strong_wolfe"`. | RAN: RAN on 100×50: `strong_wolfe` did **not** help. It stalled at loss 1.076 (boundary loss 1.071, the far field not met) with Cl 0.003, and rounds 1-4 stopped after 2-6 iterations. | Try `line_search_fn="strong_wolfe"`, with `max_iter` reduced so a round costs the same. | S | Rob |
| N10 | `NNfoil_Euler2D.py:142-145, 98` | Far-field BCs are soft. The outputs are unscaled: p∞ = 1/(1.4·0.4²) = 4.46 while the other variables are O(1). | READ. | **Hard far field + perturbation output**: `q = q_inf + (1 - η/η_max)·net(·)`. Then the far field holds exactly, `mseb1` can be deleted, outputs are O(Δq), and ρ, p start at freestream (helps K2). See N6: measured only together with N6. | S | Acc, Rob |
| N11 | `NNfoil_Euler2D.py:98, 152, 167` | Non-dimensionalisation check: ρ∞ = 1, \|V∞\| = 1, p∞ = 1/(γMa²) is the standard ρ∞V∞² scaling, and Cp = (p - p∞)/0.5 matches PAPER Eq. 3 and its Cp definition. **Correct.** For a Mach-parametric model, use p/p∞ (or p·γMa²) as the network output so its scale does not change with Mach. | READ. | None needed for single-Mach runs. | — | Cap (for K8) |
| N12 | `NNfoil_Euler2D.py:126-129, 207-210` | Wall-normal sign check: the wall runs TE → upper → LE → lower (counter-clockwise), `tan_f` is minus the index direction, and `nor_f = (-tan_y, tan_x)` points **out of the body into the fluid** (for example, mid-chord upper surface gives n = (0, +1)). Then F = -∑Cp·n·dl, so Cl = -(cosα·F_y - sinα·F_x) and Cd = -(sinα·F_y + cosα·F_x), exactly as coded. **Correct.** | READ, and RAN (Cl comes out positive at +4°; the same formula applied to the FVM Cp gives the expected values, section 2). | None. | — | — |
| N13 | `NNfoil_Euler2D.py:144-145` | Far-field Dirichlet condition (freestream) at R = 15 chords (PAPER: domain diameter 30). A lifting airfoil's velocity perturbation decays only like Γ/(2πr). At Cl ≈ 0.5, Γ = ½·Cl·c·V∞ ≈ 0.25, so u' ≈ 0.25/(2π·15) ≈ 0.003 V∞ at the boundary (EST). Forcing u' = 0 there slightly cuts circulation. The FVM reference uses the same radius, so the PINN-vs-FVM comparison is fair, but both carry this domain error. | READ, EST. | Point-vortex far-field correction (standard in Euler CFD): impose `u_inf + Γ/(2π)·(...)` with Γ = ½·Cl·V∞·c from the previous round's Cl. It fits naturally into N10's hard far-field function. | M | Acc (Cl, small) |
| N14 | `NNfoil_Euler2D.py:29-30` | `torch.nn.init.normal_(layer.weight)` after `weight_norm` is a **no-op**: the weight is recomputed from g and v on every forward. The effective initialisation is PyTorch's default, not N(0,1). `torch.nn.utils.weight_norm` is deprecated. | RAN: the forward output is identical before and after `normal_`, and torch 2.15 prints a `FutureWarning` recommending `torch.nn.utils.parametrizations.weight_norm`. | Delete the `normal_` line (keeps today's behaviour) or initialise the inner `Linear` before wrapping. Move to `parametrizations.weight_norm`. Note that changing the initialisation changes results, so treat that as an experiment. | S | Rob (future torch) |
| N15 | `NNfoil_Euler2D.py:23-24` | `X_mean` and `X_std` are plain attributes, not buffers, so `state_dict()` does not contain them. The demo cannot reload a model from a state dict. | READ. | `self.register_buffer('X_mean', ...)` and the same for `X_std`. Save `{'state_dict', 'layers', 'Ma', 'alpha', 'mesh', 'dtype'}`. | S | Cap (export) |
| N16 | `NNfoil_Euler2D.py:178-186`; `util.py:6-9` | Four separate reverse-mode `autograd.grad(..., create_graph=True)` calls (one per output), each a full backward pass. Input dimension is only 2, so forward mode (2 tangents) gives all 8 derivatives. | RAN (micro-benchmark): `vmap(jacfwd)` vs 4× `autograd.grad`, 5×128 MLP, 8 threads, ms per loss+backward: float32 20.2 vs 23.2 (1,020 pts), 81.0 vs 80.4 (4,040), 284.7 vs 276.7 (16,080), so **no gain in float32**. float64: 25.9 vs 27.4, 286 vs 131, 644 vs 596. The 2.2× at 4,040 looks like an outlier. **Low priority.** | `torch.func.vmap(torch.func.jacfwd(f))` on a functional model, or `torch.autograd.grad(..., is_grads_batched=True)`. Needs the `parametrizations` weight norm for `functional_call`. | M | Spd |
| N17 | `NNfoil_Euler2D.py:142-161` | `Mseb` runs the network 5 times on small point sets (far, wall faces, left seam, right seam, wall nodes), plus once in `Msef`. | READ. | Concatenate all boundary inputs into one forward pass and slice. | S | Spd (small) |
| N18 | `NNfoil_Euler2D.py:12-17` | Seeds are fixed, but results still depend on the thread count (float32 reduction order). After 100 L-BFGS iterations on 50×25, the loss differed in the 3rd digit between 1, 4, 8 and 16 threads. | RAN (thread scan). | Record `threads`, `dtype` and the torch version with every result. Treat run-to-run spread as an error bar (run 3 seeds). | S | Reproducibility |
| N19 | `NNfoil_Euler2D.py:235` | 80% of points are resampled each round (up to 20,000), and the other 20% are thrown away. | READ. | Use the unused 20% as a free held-out residual each round (fixes N2 and gives a "generalisation residual" plot for the report). Later: residual-based adaptive sampling (add points where \|R\|·J is largest, as Sun et al. 2023 did in physical space). | S / M | Acc, report quality |
| N20 | `NNfoil_Euler2D.py:86` | `torch.meshgrid` without `indexing=` (UserWarning in torch 2.15, future error). The default `'ij'` is what the code needs. | RAN. | `torch.meshgrid(..., indexing='ij')`. | S | Cap (future torch) |
| N21 | `NNfoil_Euler2D.py:5-10` | Unused imports: `scipy.io`, `lr_scheduler`, `pickle`, `F` is used only once. Not a bug. | READ. | Tidy-up. | S | — |
| N22 | — (no Kutta treatment) | Nothing enforces a Kutta condition. The circulation is picked implicitly by the network's smoothness across the seam, which sits at the TE. With N6 the TE is a single smooth point in the input space. | READ. Cl agreement in section 2 shows the implicit choice is roughly right. | Diagnostic: report the Cp jump between the last upper and first lower wall node at the TE (should go to 0). Optional loss: penalise \|V\|² at the TE node (a stagnation point for a finite-angle TE). Experiment only. | M | Acc (unknown) |
| N23 | `mesh_gen.py:58-79` | The smoother is a plain Jacobi sweep in Python/NumPy, up to 500,000 sweeps. The stop test looks at X only (`err = abs(tem - X).max()`), not Y. | RAN: 50×25 took 3,288 sweeps / 8.1 s, 100×50 took 11,279 sweeps / 37.6 s (to 1e-6). The sweep count grows about 3.4× for each doubling. | Successive over-relaxation (ω ≈ 1.8) or a sparse direct solve with frozen α, β, γ (a few Picard steps) cuts this to well under a second (EST). Test both X and Y in the stop rule. | M | Spd (shape demo latency) |
| N24 | `mesh_gen.py:40-45, 87-94` | The wall comes only from CST parameters (`x = 0.5·cos θ` fixed, upper = first N/2 points, so N must be even). The output name is hard-coded (`'NACA0012_200_100.x'`) and written to the working folder. | READ. | Turn it into `make_mesh(wall_xy, N, M, R=15) -> (X, Y)` that takes any closed wall polyline (section 4). | M | Cap |
| N25 | `Airfoil2CST/cst_geom.py:7-10, 67` | Input files (`naca2412_*.dat`) and the Tecplot point count (`ZONE I=364`) are hard-coded. The CST class has no trailing-edge-thickness term (closed TE only). | READ. | Parameters for both. Add a TE-thickness term if blunt-TE shapes matter. | S | Cap |
| N26 | `NNfoil_Euler2D.py:318-326` | The script only plots Cp against FVM. No error metric is computed. The paper's metric is the relative L1 error ‖q̂ - q_ref‖₁ / ‖q_ref‖₁ against FVM 800×400 (PAPER §3.1). | READ. | Add `validate.py` (or a `report()` method) that computes Cp relative L1 error and Cl relative error against `FVM_Cp/NACA0012_800_400_Cp.dat`, interpolated in x on the upper and lower surfaces. | S | Report quality |
| N27 | — | No machine-readable output. | — | Write `result.json`: `{"meta": {mesh, Ma, alpha_deg, layers, dtype, threads, torch, seed}, "timing_s": {per_round, total}, "loss_log": {...}, "Cl", "Cd", "wall": {"x", "y", "Cp"}, "field": {"x", "y", "u", "v", "p", "rho"} (M×N arrays), "fvm_ref": {"file", "Cl", "Cp_rel_L1"}}`. The demo reads this directly, so the PINN-in-demo step does not need PyTorch in the web server. | S | Cap |
| **N28** | `NNfoil_Euler2D.py:240-248` | **Trivial-solution collapse, and a reload that cannot escape it.** On 100×50 the network repeatedly falls into a state with PDE loss **exactly 0** and boundary loss ≈ 0.04-0.07 (the constant output of a saturated tanh network has zero derivatives, so the residual is exactly 0 in float32). The `loss_f < 1e-15` test catches it, but outside round 0 the code *reloads the previous model and runs the same L-BFGS again*, which falls back into the same basin. A second stuck state (total loss ≈ 1.076, boundary loss 1.07, Cl ≈ 0) also stops L-BFGS after 2-8 iterations. | RAN: 100×50 baseline, rounds `r o o R o R R o R R`, final Cl -0.002. float64: `o R R o R`, final Cl 0.001. strong_wolfe: stalled at 1.076, Cl 0.003. 50×25 also hit the trivial state in round 0 (restarted, then fine). | (1) When `loss_f < 1e-12`, or when 2 rounds in a row make no progress, **reinitialise with a new seed** instead of reloading, and cap the retries at 3. (2) Warm-start with Adam (for example 1-2k steps, lr 1e-3) before the first L-BFGS round. (3) Optionally add a small penalty that grows when outputs are constant, for example −λ·Var(u) early on. (2) and (3) are EST and not measured. Only the periodic + hard far-field variant avoided the collapse in our runs (N6). | S-M | **Rob, top priority** |

---

## 2. Measured CPU timings and Cl

All rows are RAN unless marked. "s/round" is the wall time of one `train()` round (one L-BFGS call, up to 1000 iterations). Cl and Cd come from NNfoil's own `Calculate_Force()` (midpoint rule on face-centre Cp) after the given round.

### 2a. Thread scan (100 L-BFGS iterations, 5×128 network, float32)

| Mesh (points, batch) | 1 thread | 4 threads | 8 threads | 16 threads |
|---|---|---|---|---|
| 50×25 (1,275 pts, 1,020 per batch) | 61 ms/iter | 48 ms/iter | 48 ms/iter | 58 ms/iter |
| 100×50 (5,050 pts, 4,040 per batch) | 177 ms/iter | 112 ms/iter | 97 ms/iter | 127 ms/iter |

These were measured before the other session's training started (see the timing caveat at the top). Early iterations are cheaper than late ones because the L-BFGS history is still short (N3).

### 2b. Full rounds (8 threads)

| Run | Mesh | Rounds done | L-BFGS iterations | Wall time | ms per iteration | Round status (r = restart, o = ok, R = reload) | Final Cl | Cl error vs 0.519 | Cp rel. L1 vs FVM 800×400 |
|---|---|---|---|---|---|---|---|---|---|
| base (float32, original method) | 50×25 | 10 | 6,516 | 1,239 s (*) | 190 (*) | r o o o o o o o o R | **0.4975** | **4.1%** | **0.047** |
| base (float32, original method) | 100×50 | 10 | 3,805 | 380 s | 100 | r o o R o R R o R R | -0.002 | collapsed | 1.00 |
| float64 | 100×50 | 5 | 3,588 | 639 s | 178 | o R R o R | 0.001 | collapsed | 1.00 |
| strong_wolfe line search | 100×50 | 5 | 801 | 91 s | 114 | o o o o o (stalled) | 0.003 | stalled | 1.00 |
| periodic ξ + hard far field (N6+N10) | 100×50 | 5 | 2,387 | 225 s | 94 | o o o o o | 0.271 | 48% | 0.368 |
| **grid sequencing: 5 rounds 50×25, then weights moved to 100×50, 4 rounds** (idea I1) | 100×50 | 5 + 4 | 1,951 on the fine mesh | **302 s total** (149 + 153) | 78 (fine) | coarse r o o o o, fine o o o o | **0.5065** | **2.4%** | **0.027** |
| *FVM, same mesh (shipped data)* | 50×25 | — | — | — | — | — | 0.4265 | 17.9% | 0.250 |
| *FVM, same mesh (shipped data)* | 100×50 | — | — | — | — | — | 0.4803 | 7.5% | 0.113 |

(*) Overlapped with another session's training (see the caveat at the top).

Per-round times for 1,000 iterations (RAN): **50×25 ≈ 50-150 s** (the low end from the clean thread scan and the uncontended rounds), **100×50 ≈ 95-100 s** in float32. Later, on a quieter machine, 50×25 full rounds took 57.7 s and 56.6 s (about 57 ms per iteration, RAN) (rounds that ran the full 1,000 iterations: 96.9, 98.6, 95.7, 96.8 s). L-BFGS often stops early, so many rounds are much shorter.

50×25 Cl by round (RAN): -0.125 (trivial, restarted), 0.099, 0.370, 0.484, 0.504, 0.468, 0.508, 0.507, 0.497, 0.497. After 4 good rounds Cl is already within 3% of the reference, and it then wanders by ±0.02.

Compared with the paper (PAPER Table 1, RTX 4090, 20,000 iterations): NNfoil 50×25 Cp error 0.029, Cl error 0.022, 384 s. **Our CPU run reached Cl error 4.1% and Cp error 0.047 in 6,516 iterations and about 21 minutes (contended).** That is worse than the paper, but already about 4 times better than same-mesh FVM, which reproduces the paper's main claim on sparse meshes. Our FVM Cp L1 errors (0.250, 0.113) are about 1.45 times the paper's (0.176, 0.077), probably because of a different interpolation; the ratios between meshes agree.

EST for a full paper-length run (20 rounds × 1,000 iterations) on this CPU: about 20-50 min on 50×25 and about 30-35 min on 100×50, because of early stops. Not measured.

### 2c. Reference Cl from the shipped FVM data (RAN)

The Cp files were integrated with the same midpoint rule as `Calculate_Force`, on the mesh each file belongs to (Ma = 0.4, α = 4°):

| FVM mesh | 50×25 | 100×50 | 200×100 | 400×200 | 800×400 |
|---|---|---|---|---|---|
| Cl | 0.4265 | 0.4803 | 0.5046 | 0.5146 | **0.5193** |
| Cd (should be 0, numerical) | 0.0203 | 0.0116 | 0.0069 | 0.0041 | 0.0025 |
| Cl error vs 800×400 | 17.9% | 7.5% | 2.8% | 0.9% | — |

PAPER Table 1 quotes FVM Cl errors of 0.184, 0.038, 0.012 and 0.004 for these meshes, and NNfoil errors of 0.022, 0.010, 0.018 and 0.011, with NNfoil wall times of 384 s, 266 s, 262 s and 271 s on an RTX 4090 (FVM: 9 s to 285 s on 4 i9-13900KS cores). Our 50×25 FVM number matches the paper (17.9% vs 18.4%), but our 100×50 to 400×200 values are about twice the paper's. The paper probably integrated forces inside the FVM solver rather than from the exported nodal Cp. **For the report, use 0.519 (FVM 800×400) as the reference Cl**, and say it was integrated from the shipped Cp file.

Sanity check (EST): thin-airfoil theory with Prandtl-Glauert gives 2π·(4π/180)/√(1-0.4²) = 0.479. Thickness raises this a little, so ≈ 0.5 is right.

---

## 3. Recommended top 5 (CPU-only student project that needs good numbers)

Ranked by "good reportable numbers per hour of student effort on a CPU". Each maps to a task in section 0.

1. **Make it run and measure itself** (K1, K4, K5, K6, K7, Task 1). Without this there are no numbers. 8 threads instead of 1 was measured at 1.3× (50×25) to 1.8× (100×50) faster per iteration.
2. **Validation and JSON export** (N26, N27, Task 2). Turns runs into the paper's metric (Cp relative L1 and Cl error against FVM 800×400) and feeds the demo without PyTorch on the server.
3. **Grid sequencing, coarse to fine (idea I1, Task 3b), plus training-loop robustness** (N1-N4, N28, Task 3). **Measured: this turned a run that always collapsed on 100×50 (Cl ≈ 0) into Cl 0.5065 (2.4% error) and Cp-L1 0.027 in 302 s on a CPU**, better than our 50×25 result (4.1%) and better than same-mesh FVM (7.5%, 0.113). Removes the silent "saved a NaN model" path and the shared checkpoint file, and makes long CPU runs safe to leave overnight. **Most important measured problem:** the 100×50 baseline collapsed to the trivial solution in every round (N28). Fixing that (reinitialise with a new seed instead of reloading, plus an Adam warm-up) comes before anything else on bigger meshes.
4. **Periodic ξ embedding plus hard far field** (N6, N10, Task 4). Small code changes that remove two of the three boundary losses and enforce the O-grid topology exactly. A clean ablation for the report. Measured on 100×50: it removed the collapse (5/5 rounds ok), but Cl came out at 0.271, so it must be checked on 50×25 first and paired with the Kutta diagnostic (N22).
5. **Mesh pipeline for user shapes with the corrected smoother** (K3, N4, N23, N24, Task 5). Gives the "draw a shape" feature a validated mesh, a Jacobian check, and a cylinder test against the shipped FVM data. It also closes the open "mesh coefficient" question with a measured answer.

Next best, if there is time: the panel-method prior (I2, section 3b), which is designed but not measured. Not in the top 5 (worth mentioning in the report as future work): the J² weighting ablation (N5), the far-field vortex correction (N13), `torch.func` derivatives (N16), parametric α (K8), and Kutta diagnostics (N22).

---

## 3b. Out-of-the-box improvement ideas

### I1. Grid sequencing for a PINN (TESTED, works)

Classical CFD codes solve on a coarse grid first and use that answer to start the fine grid ("mesh sequencing"). NNfoil makes this almost free, because its network input is the *grid index* (ξ, η), and all shipped meshes for one airfoil share the same θ-spaced wall and the same smoother. A network trained on 50×25 is therefore already a good answer on 100×50, once the input normalisation is rescaled (Task 3b).

RAN, one seed, 8 threads:

| | Before | With grid sequencing |
|---|---|---|
| 100×50 result | collapsed in every round: Cl -0.002, Cp-L1 1.00 (baseline); float64 and line-search variants also failed | **Cl 0.5065 (2.4% error), Cp-L1 0.027**, all 4 fine rounds ok |
| Cl right after the weight transfer, with no fine-mesh training | — | 0.504 |
| Total CPU time | 380 s wasted (10 rounds, no result) | 302 s (149 s coarse + 153 s fine) |

Why it matters for the project:
- It reproduces the paper's sparse-mesh claim on a CPU (PINN Cl error 2.4% vs same-mesh FVM 7.5%).
- It is the cheapest way to reach the paper's 200×100 mesh (chain 50 → 100 → 200; not measured).
- It is a clean "our own contribution" for the report: an idea from classical CFD that is not in the paper.

Limit: one seed only. Repeat with 3 seeds before quoting an error bar (N18).

### I2. Panel-method prior: learn only the compressibility correction (DESIGNED, not measured)

The demo already has a vortex-panel solver with a Kutta condition and the Prandtl-Glauert correction (`app/backend/aeronet/panel.py`, `field.py:34-75`). It computes u, v at any point in milliseconds. Use it as a *physics prior*:

`q(ξ,η) = q_panel(x(ξ,η), y(ξ,η)) + (1 − η/η_max) · net(ξ,η)`

Here `q_panel` comes from the panel velocities: u, v from the field evaluator, and p, ρ from the isentropic relations at Ma∞. The network learns only the nonlinear compressible correction, and it vanishes at the far field (as in N10).

Why it should help (EST):
1. The panel solution already has the right circulation (an explicit Kutta condition). That addresses the low-Cl failure we measured for the periodic + hard far-field variant (Cl 0.271) and the missing Kutta treatment (N22).
2. The trivial constant state (N28) is no longer a minimum, because the prior is not constant.
3. The demo gets a natural story: instant panel answer → PINN refinement → FVM reference.

How to build it (M, about 1 day):
- Add `field_uv(sol, px, py)` to `field.py`, a copy of `_cp_chunk` that returns `(u, v)` instead of Cp (no behaviour change for the app).
- Geometry: run the panel solver on the **same wall points** as the NNfoil mesh, shifted by +0.5 in x (the panel code uses x ∈ [0, 1]; NNfoil uses [-0.5, 0.5]). Do not use `naca4()`, so both see the same shape.
- Precompute `q_panel` once on the grid (a fixed tensor), add it inside `Net.forward` via a lookup of the batch indices, and train as usual.
- **Check:** at initialisation Cl should already be about 0.48 (the PG panel value), and after training it should be at least as good as Task 3b. Report both.

Risk: the PG-corrected field near the leading edge may be a poor prior at Ma 0.4 (Karman-Tsien is a better correction). The network correction absorbs that, but if training is slower than Task 3b, keep Task 3b as the default.

---

## 4. Running NNfoil on a user-drawn closed 2D shape (design for the plan)

NNfoil itself does not care how the mesh was made: with `use_cst=False` it needs only an O-grid in the `.x` layout (`N M`, then M rows of X, then M rows of Y; wall = first row, far field = last row, wall ordered TE → upper → LE → lower, counter-clockwise). Steps:

1. **Clean the curve.** Remove duplicate points. Make the polygon counter-clockwise (reverse it if the signed area is negative). Reject self-intersecting curves (segment-intersection test) and curves with fewer than about 20 points.
2. **Normalise.** LE = the point with minimum x (after a rough principal-axis rotation), TE = the point farthest from the LE. Translate, rotate and scale so the chord runs from (-0.5, 0) to (0.5, 0), the convention every NNfoil file uses. α is then measured from this chord.
3. **Resample.** Split the curve at the TE and LE into upper (TE→LE) and lower (LE→TE) parts. Resample each by arc length to N/2 points with cosine clustering at both ends, so index 0 = TE and index N/2 = LE, as in `mesh_gen.py` (N even). Smooth shapes with no sharp TE (circle, ellipse) work the same way.
4. **Far field.** Circle R = 15, with θ_k given by the wall's normalised arc-length parameter (not uniform θ), so radial lines are not badly skewed.
5. **Smooth.** Use the `mesh_gen.py` elliptic smoother with the **corrected cross-term coefficient 2** (K3), the stop test on X and Y, and SOR or a sparse solve for speed (N23; plain Jacobi took 8 s at 50×25 and 38 s at 100×50, RAN).
6. **Validate** with `util.jacobian_trans(..., boundary=(1,1))` (the exact metrics NNfoil will use): J must have **one sign everywhere except the TE node(s)** (negative in NNfoil's orientation, N4). Reject the mesh if min\|J\| over all other nodes is below about 1e-6 of the median wall-row \|J\|, or if any grid cell folds (sign change). Also report the maximum non-orthogonality at the first off-wall row (the shipped meshes have max \|cos\| 0.05-0.08).
7. **Write** the `.x` file and run NNfoil with `use_cst=False`.
8. **Test case:** `FVM_Cp/Cylinder_*_Cp.dat` exists (Ma 0.4, α 4°, as with the other files), but no cylinder mesh is shipped. Making a circle through this pipeline and comparing with `Cylinder_100_50_Cp.dat` validates the arbitrary-shape path on a non-airfoil body.

Cost: every new shape needs a fresh training (minutes on a CPU, section 2), so the demo cannot be instant for a new shape. Options: (a) run in a background job and show progress, (b) warm-start from the NACA0012 weights. Warm-starting is EST (likely faster, not measured).

---

## 5. Scratch files (outside the repo, not committed)

In `C:\Users\trish\AppData\Local\Temp\claude\3f043a2d4c7c1ba0\fe71d6ad-81fc-42f1-958d-bc18716ab325\scratchpad\bench\`:

- `nnfoil_bench.py`: minimal-fix copy used for all timings.
- `nnfoil_variants.py`: the same, plus `--periodic` (N6) and `--hardfar` (N10).
- `analyse_mesh.py`, `jsign.py`: Jacobian, weighting and FVM-Cl checks.
- `meshgen_test.py`: smoother with the cross-term coefficient as a parameter.
- `bench_grad.py`: autograd vs `torch.func` micro-benchmark.
- `res_*.json`: per-run logs (loss, Cl, Cd, s/round, closures per round).

The scratchpad is temporary. Copy anything worth keeping into the project.
