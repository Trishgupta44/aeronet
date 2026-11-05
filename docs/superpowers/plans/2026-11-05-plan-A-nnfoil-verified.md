# Plan A: NNfoil verified and running on CPU (ends at Milestone 1)

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. **Read
> `2026-11-05-aeronet-v2-roadmap.md` first**: §5 (interfaces) and §6 (rules) are binding.
> This plan is a guide: it says what to build, why, the interfaces and how to check them. You write the
> code (use TDD: write the test, see it fail, implement, see it pass, commit).

**Goal:** the upstream NNfoil code, fixed and verified, trains on this laptop's CPU. It writes standard run
folders, and its results are compared with the FVM reference data by one shared metric. The output is the
Milestone 1 pack for the mentor.

**Architecture:** NNfoil is vendored into the main repo (its original version is the first commit, and
every fix is a separate commit, so `git log NNfoil/` is the bug-fix evidence). A thin CLI
(`NNfoil/run_case.py`) wraps the fixed class. All measuring code lives in `research/` and shares
`research/metrics.py`.

**Tech stack:** Python 3.13 (this machine), `torch` CPU wheel (2.15.0+cpu was verified to install), NumPy 2,
SciPy, Matplotlib (Agg backend), pytest.

**Already verified on 2026-11-05 (in a scratch copy, not in the repo):** every fix in A2–A6, the tests in
A8, the metric functions in A7, and the finding-3 and finding-10 checks in A9. Expected numbers are given
in each task. If you get very different numbers, stop and investigate.

---

## File map

| Path | Responsibility |
|---|---|
| `NNfoil/NNfoil_Euler2D.py` | Upstream solver, fixed in place (A2–A6). The `__main__` block still works. |
| `NNfoil/util.py` | Use the Agg backend so imports work headless (A2) |
| `NNfoil/run_case.py` | CLI: train one case and write the run folder (roadmap §5.3, §5.8) (A10) |
| `NNfoil/AERONET_CHANGES.md` | One line per fix: finding #, file:line, commit (A2–A6, A9) |
| `research/requirements.txt` | torch (CPU index), numpy, scipy, matplotlib, pytest |
| `research/metrics.py` | Loaders, `cp_error`, `forces_from_faces`, `order_contour`, `fvm_forces`, `append_summary` (A7) |
| `research/check_mesh_smoothing.py` | Finding 3 evidence (A9) |
| `research/check_jacobian.py` | Finding 10 evidence (A9) |
| `research/validate.py` | Run folder vs FVM: metrics row plus Cp figure (A11) |
| `research/baselines.py` | FVM coarse-vs-fine, panel vs FVM and thin-airfoil rows (A11) |
| `research/tests/conftest.py`, `test_*.py` | Tests for all of the above |
| `docs/verification.md` | Theory vs code vs verdict table (A9) |
| `docs/milestones/M1-nnfoil-verified.md` | Mentor pack (A12) |
| `results/summary.csv`, `results/figures/`, `results/runs/` | Outputs |

---

### Task A1: Vendor NNfoil and set up the research environment

**Why:** `NNfoil/` is an untracked folder with its own `.git`, which the outer repo can't track. Vendoring
puts the upstream code in history unchanged, so each fix shows up as its own diff.

- [ ] Look first: `git -C NNfoil log --oneline -1` should show `14f12c5`, and `git -C NNfoil status` should be clean.
- [ ] Move (don't delete) the nested git folder out of the repo:
      `mv NNfoil/.git ../NNfoil-upstream.git`. Tell the user where it went.
- [ ] Delete `NNfoil/mesh_generation/__pycache__/`.
- [ ] Add to the root `.gitignore` (create it): `research/.venv/`, `**/__pycache__/`, `*.pth`,
      `results/runs/*/model.pt`, `results/runs/*/train.log`.
- [ ] Commit 1: `git add NNfoil .gitignore`, then
      `vendor: NNfoil upstream at 14f12c5 (Cao, Song, Zhang; arXiv 2401.08705; no licence, keep private)`.
      This commit must contain the **unmodified** upstream files.
- [ ] Create `research/requirements.txt` with `--extra-index-url https://download.pytorch.org/whl/cpu`,
      `torch`, `numpy>=2.1`, `scipy`, `matplotlib`, `pytest`.
- [ ] Create the venv: `python -m venv research/.venv`, then
      `research/.venv/Scripts/python -m pip install -r research/requirements.txt`.
- [ ] Check: `research/.venv/Scripts/python -c "import torch,numpy;print(torch.__version__,numpy.__version__)"`
      prints a `+cpu` torch and numpy 2.x. Record the versions in `research/requirements.txt` as a
      comment (`# verified: torch 2.x.x+cpu, numpy 2.x.x`). Pinning exact versions is optional.
- [ ] Commit 2: `chore(research): CPU research environment`.

### Task A2: Test harness, plus finding 1 (GPU-only device)

**Bug:** `NNfoil_Euler2D.py:269` is `device = torch.device("cuda:0" if 1 else "cpu")`, which always picks
CUDA and crashes on a laptop.

- [ ] Create `research/tests/conftest.py`: set `os.environ["MPLBACKEND"]="Agg"` before anything else, put
      `NNfoil/` and `research/` on `sys.path`, and provide the fixtures `mesh_50x25`
      (`metrics.load_mesh("NNfoil/mesh/NACA0012_50_25.x")`; until A7 exists, inline the loading:
      skip 1 header row, first half of the rows = X, second half = Y, transpose, append row 0) and `repo`
      (the repo root path).
- [ ] In `NNfoil/util.py`, use `matplotlib.use("Agg")` before `import matplotlib.pyplot` (headless import).
- [ ] Test `test_pick_device_falls_back_to_cpu`: monkeypatch `torch.cuda.is_available` to return False.
      Then `NNfoil_Euler2D.pick_device().type == "cpu"`. It fails first (no such function).
- [ ] Implement a module-level `pick_device()` that returns `cuda:0` if `torch.cuda.is_available()`, else
      `cpu`. Use it in `__main__` instead of line 269.
- [ ] Commit: `fix(nnfoil): finding 1 - pick CPU when CUDA is unavailable`.

### Task A3: Findings 4 and 5 (one commit each)

- [ ] **Finding 5 first** (it blocks everything): `np.math.factorial` (line 110, three uses) was removed in
      NumPy 2. Test `test_nnfoil_constructs_on_numpy2`: building `NNfoil([2,16,16,4], cpu, xx, yy)` on the
      50×25 mesh does not raise. It fails with `AttributeError` first. Fix: `import math`, then
      `math.factorial`.
      Commit: `fix(nnfoil): finding 5 - math.factorial (np.math removed in NumPy 2)`.
- [ ] **Finding 4:** `torch.set_num_threads(1)` (line 265). Change it to `min(8, os.cpu_count() or 1)`.
      This machine has 16 logical threads; 8 was used for every measurement. No test; it is a `__main__`
      setting. Commit: `fix(nnfoil): finding 4 - use up to 8 CPU threads`.

### Task A4: Finding 6 (checkpoint save and load)

**Bug:** `torch.save(self.model, 'model_temp.pth')` and `torch.load('model_temp.pth')` (lines 247 and 250)
pickle the whole module. Since PyTorch 2.6, `torch.load` defaults to `weights_only=True`, so loading
fails. The fixed file name also collides between runs.

- [ ] Add a constructor argument `ckpt_path='model_temp.pth'`. Save with `torch.save(self.model.state_dict(), self.ckpt_path)`
      and load with `self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device, weights_only=True))`.
- [ ] Test `test_checkpoint_roundtrip(tmp_path)`: construct, save, perturb the weights, then call the same
      load code (extract tiny `_save_ckpt()` and `_load_ckpt()` methods so they can be tested). The outputs
      on `XE` must equal the saved ones exactly.
- [ ] Commit: `fix(nnfoil): finding 6 - state_dict checkpoints with per-run path`.

### Task A5: Finding 2 (positivity of rho and p)

**Bug:** in `Msef` (lines 193–197) the raw network outputs p and rho are used as `gamma*p/rho` and `1/rho`.
A zero or negative value breaks the physics.

- [ ] `Net.__init__(..., positivity=True)`. In `forward`, after the last layer, when `positivity` is set,
      return `cat([x[:, 0:2], softplus(x[:, 2:4]) + 1e-3])`. The output order is (u, v, p, rho).
- [ ] `NNfoil.__init__(..., positivity=True)` passes the flag at **both** places where `Net(...)` is
      created (the constructor and the "Failed, Restart" branch in `train`).
- [ ] Tests:
      - `test_positivity_keeps_p_rho_above_floor`: scale every `weight_g` by 50 under `no_grad`. Then
        `min(p), min(rho) >= 1e-3` over `XE`. Verified value: exactly 0.001.
      - `test_positivity_off_is_raw`: the same with `positivity=False`, and `min(p) < 0`. Verified: about −47.
- [ ] Commit: `fix(nnfoil): finding 2 - softplus positivity for p and rho (flag, default on)`.
- [ ] **Known effect (measured, for E1 in Plan C):** with positivity on, round 1 printed `Failed, Restart`.
      Rounds 2 and 3 then reached Cl 0.307 and 0.478 (RMS vs FVM 800×400: 0.193 and 0.037). Without it, Cl
      was 0.516 after round 1. Do not "fix" this here. Plan C measures it properly.

### Task A6: Findings 8 and 7 (flow condition as an argument; compute Cl and Cd)

- [ ] **Finding 8:** `self.Ma = torch.tensor(0.4); self.alpha = torch.tensor(4/180*torch.pi)` (line 55) is
      hard-coded. Add constructor arguments `Ma=0.4, alpha_deg=4.0`. Keep it one condition per model: the
      network input stays (ξ, η), as in the paper's code. Document this in `AERONET_CHANGES.md` (the paper's
      continuous-AoA idea is not in this file).
- [ ] **Finding 7:** `Calculate_Force()` (lines 207–210) is never called. Add
      `evaluate_wall() -> (cl, cd)`. Under `no_grad` it computes `wall['Cp_f']` from `model(wall['XE_f'])`
      and `wall['Cp']` from `model(wall['XE'])`, with `Cp = (p − 1/(1.4 Ma²)) / 0.5`. It then calls
      `Calculate_Force()` and returns floats. `__main__` prints Cl and Cd after training.
- [ ] Tests:
      - `test_flow_condition_arguments`: with `Ma=0.3, alpha_deg=2`, `far['u']` is close to cos 2°, and
        `far['p']` is close to 1/(1.4·0.09).
      - `test_evaluate_wall_returns_finite_floats` on an untrained small net.
- [ ] Commit: `fix(nnfoil): findings 7, 8 - Ma/alpha arguments and evaluate_wall() for Cl, Cd`.

### Task A7: `research/metrics.py`

**Interface: roadmap §5.8, exactly.** The notes below are the parts that are easy to get wrong.

- `load_fvm`: `np.loadtxt(skiprows=1)`; columns node, x, y, Cp. Return `(x + 0.5, y, cp)`.
- `load_mesh`: the first line is `N M`, then the data. Rows `[:h]` are X and `[h:]` are Y, with
  `h = rows // 2`. Transpose to (N, M), then append row 0 (closed ring), giving (N+1, M). The wall is column 0.
  Verified: 50×25 wall nodes run from (1, 0) counter-clockwise, signed area +0.0819.
- `cp_error`: nearest-point rule, roadmap §5.5. Keep only reference points with 0.02 ≤ x ≤ 0.98.
- `forces_from_faces`: copy NNfoil's convention: `tan = -(ΔX)/dl`, `nor = (-tan_y, tan_x)`, `F = Cp·dl·nor`,
  `Cl = -(cos α F_y − sin α F_x)`, `Cd = -(sin α F_y + cos α F_x)`.
- `order_contour`: sort by `atan2(y − ȳ, x − x̄) mod 2π`. **The FVM files are not in contour order** (an
  unordered integration gives Cl 0.355 instead of 0.519).
- `fvm_forces(path, alpha)`: order, then face Cp = mean of neighbouring nodes (wrap around), then
  `forces_from_faces` on the closed ring.
- `append_summary`: `csv.DictWriter` with the §5.4 columns. Rewrite the file, replacing a row with the same
  (experiment, case, method, mesh) key. Reject unknown columns with `ValueError`. `None` is written as empty.

Tests (expected values were verified on 2026-11-05):
- [ ] Panel Cp from `aeronet.panel.solve("0012", 4, 0.4, 100)`, closed with node 0, through
      `forces_from_faces`, equals the panel's own `cl` and `cd` to 1e-9 (0.526233). Put `app/backend` on
      `sys.path`; it needs numpy only.
- [ ] A uniform Cp gives Cl = Cd = 0 (abs 1e-12).
- [ ] `cp_error(panel vs load_fvm(FVM_Cp/NACA0012_800_400_Cp.dat))`: RMS 0.0099 ± 0.0005, n = 654.
- [ ] `fvm_forces(NACA0012_800_400, 4)`: Cl 0.5193 ± 0.0005. 50×25: 0.4265 ± 0.0005.
      `NACA2412_800_400_AoA4`: 0.7605 ± 0.0005.
- [ ] `append_summary` twice with the same key leaves one row, holding the second value. An unknown column raises.
- [ ] Commit: `feat(research): shared metrics and loaders`.

### Task A8: Verification tests (promise P7)

These go in `research/tests/test_nnfoil_checks.py`, on the 50×25 mesh with `layers=[2,16,16,4]`, so they
run fast.
- [ ] **Surface mapping:** `wall['X']` equals mesh column 0 (max difference < 1e-6; verified 1.4e-8).
- [ ] **Outward normals:** `nor_f · (X_f − centroid) > 0` for 100% of the faces (verified 1.0).
- [ ] **Freestream gives zero residual:** replace `nn.model` with a module returning the constant
      `(cos α, sin α, p_far, 1)`. Add `0*x.sum(1, keepdim=True)` so autograd has a path, otherwise
      `fwd_gradients` gets `None`. Set `nn.batch_ind = arange(all points)`. Then `Msef() == 0` (verified 0.0).
      Also the far-field part of `Mseb` is 0, and `evaluate_wall()` gives Cl = Cd = 0.
      (The wall part of `Mseb` is not 0, because freestream is not tangent to the wall. This is expected.)
- [ ] **Force convention:** for an arbitrary `Cp_f` (for example `sin(arange)`), `Calculate_Force()`
      equals `metrics.forces_from_faces(xx[:,0], yy[:,0], Cp_f, 4)` to 1e-5 (verified to 1.7e-8).
- [ ] Commit: `test(nnfoil): mapping, normals, freestream residual, force convention`.

### Task A9: Findings 3 and 10, and `docs/verification.md`

**Finding 3 (mesh-smoothing coefficient): CONFIRMED, low impact.** In `NNfoil/mesh_generation/mesh_gen.py`
(the update lines for X and Y), the cross term is `- 0.5 * beta * X_ξη`. The Winslow/Thompson elliptic
grid equation is `α x_ξξ − 2β x_ξη + γ x_ηη = 0`, with α = x_η²+y_η², β = x_ξx_η+y_ξy_η and γ = x_ξ²+y_ξ².
`differ.diff2d` uses central stencils (first derivative `[-½, 0, ½]`, second `[1, −2, 1]`), so the
correct factor is 2, not 0.5.
- [ ] `research/check_mesh_smoothing.py`: regenerate the 50×25 NACA0012 grid with the CST parameters in
      `mesh_gen.py`, once with factor 0.5 and once with factor 2.0. Report the iterations, time, max |true
      Winslow residual| (interior), the minimum grid-line angle, whether the Jacobian is single-signed, and
      the max distance to the bundled `NACA0012_50_25.x`. Expected (verified): upstream 3287 iterations,
      residual 1.95e-4, minimum angle 86.0°, distance to bundled 3.3e-4 (so the bundled meshes came from
      this code); corrected residual 1.99e-6, minimum angle 86.8°, distance 0.011. Write the table to
      `results/finding3_mesh_smoothing.csv`.
- [ ] Do **not** change `mesh_gen.py` (the bundled meshes and FVM data depend on it). The corrected
      generator is `research/ogrid.py` in Plan C.

**Finding 10 (new, found 2026-11-05): wrong-sign Jacobian at the trailing-edge seam. Low impact.**
`NNfoil.Mesh()` calls `jacobian_trans(..., boundary=(1,1))`, which uses one-sided stencils in ξ. But ξ is
periodic: row N is row 0 repeated. At the trailing-edge node this gives det J of the wrong sign, close to
0, on every bundled mesh (verified: 2 cells on NACA0012 meshes, 1 on 2412 and Airfoil2). With periodic
differencing (`boundary=(2,1)` on the open ring `X[:-1]`, then re-closed) there are 0 wrong-sign cells.
The loss weights residuals by J², so the effect on training is small.
- [ ] `research/check_jacobian.py`: for each bundled 200×100 mesh plus NACA0012 50×25 and 100×50, print
      the wrong-sign cell count both ways and write `results/finding10_jacobian.csv`.
- [ ] Do not change the training default. Add an opt-in `periodic_jacobian=False` constructor flag that
      uses the periodic version. A test checks that the flag gives zero wrong-sign cells on 50×25.
      Plan C may ablate it.

**`docs/verification.md`** (promise P6): one table row per component, with the columns
*Component · What the theory says · What the code does (file:line) · Test or evidence · Verdict*.
- [ ] Cover the rows Euler equations in conservative/quasi-linear form (e1–e4 in `Msef`); the mapping
      (ξ,η)→(x,y) and its Jacobian (`util.jacobian_trans`, plus finding 10); the Jacobian-weighted residual
      (J² weights); far-field BC; wall BC (zero normal velocity); periodic seam (interface loss `mseb3`);
      non-dimensionalisation (`p_far = 1/(γ Ma²)`, `rho_far = 1`, `|V∞| = 1`); the Cp formula; force
      integration (`Calculate_Force`); the mesh generator (finding 3); and findings 1, 2 and 4–8.
- [ ] Each verdict is one of: **correct**, **bug (fixed in <commit>)**, **bug (documented, not changed)**
      or **limitation**.
- [ ] Commit: `docs: verification table; findings 3 and 10 evidence`.

### Task A10: `NNfoil/run_case.py` (the training CLI)

**Interface:** roadmap §5.3 (run folder) and §5.8 (arguments and run id). Write it in `NNfoil/` so
`from util import ...` works.

- [ ] Parse the arguments, set the seeds (numpy and torch) after importing the module (upstream seeds 0 at
      import), `torch.set_num_threads(threads)`, then `load_mesh`. Construct
      `NNfoil(layers, pick_device(), xx, yy, Ma, alpha_deg, positivity, ckpt_path=out/"model.pt")`.
- [ ] Loop over the rounds: `nn.train(1)`. A round "failed" if `len(nn.log['losses'])` did not grow (upstream
      restarts or reloads silently); count those in `log.json["failed_rounds"]`. Then call
      `evaluate_wall()` and append `round, time_s (nn.log['time'][-1]), loss, loss_f, loss_b, cl, cd`.
      **Rewrite every output file after each round**, so a killed run keeps its results.
- [ ] `snapshots.json`: wall face coordinates `wall['X_f']` in the app frame (x+0.5) once, plus each
      round's face Cp.
- [ ] `field.npz`: `model(XE)` reshaped to `(M, N, 4)` with `M = xx.shape[0]` and `N = xx.shape[1]`.
      Store `X = xx0 + 0.5`, `Y = yy`, and `u, v, p, rho, cp`.
- [ ] `--resume`: load `model.pt` and `log.json`, restore `nn.log` (so the cumulative `time` continues) and
      continue the round numbers.
- [ ] Tests (CPU, tiny): `layers 2,8,8,4`, `--lbfgs-iter 5`, `--rounds 2` on 50×25 into `tmp_path`. All 5
      files exist, `log.json` has 2 rounds, `snapshots.json` has 2 Cp lists of length N, and the `field.npz`
      shapes are (51, 25). `--resume --rounds 1` takes it to 3 rounds. Add `--lbfgs-iter` by making
      `max_iter` a constructor argument (default 1000).
- [ ] Commit: `feat(nnfoil): run_case CLI writing standard run folders`.

### Task A11: `research/validate.py` and `research/baselines.py`

- [ ] `validate.py <run_dir> --ref <fvm file> --experiment <id>`: read the config, last snapshot and log;
      compute `cp_error` against the reference, `cl_ref = fvm_forces(ref, alpha)` and
      `train_s = log.time_s[-1]`. Plot Cp vs x/c (inverted y): PINN line, FVM dots, panel line from
      `aeronet.panel.solve` for NACA shapes only. Save `<run_dir>/cp_vs_fvm.png`, then `append_summary`
      with `case = <Shape>_a<alpha>[_s<seed>]` (roadmap §5.4), `method = "pinn"` or `"pinn_nopos"` and
      `source = "measured"`.
- [ ] `baselines.py` writes these rows (experiment `B0`):
      - FVM NACA0012 at 50×25, 100×50, 200×100 and 400×200 vs 800×400: `rms_cp`, `max_cp`, `cl`
        (`fvm_forces`) and `cl_ref` (800×400). Expected RMS 0.108, 0.046, 0.019, 0.006; Cl 0.4265, 0.4803,
        0.5046, 0.5146; Cl_ref 0.5193.
      - Panel NACA0012 at Mach 0.4, 4°, vs 800×400: RMS 0.0099, Cl 0.526.
      - Panel NACA2412 at each AoA file at **Mach 0.25** (`source = fitted`, note "Mach fitted, confirm from
        paper"), plus `cl_ref` from `fvm_forces` per AoA.
      - Thin-airfoil Cl for both shapes (`aeronet.compare.thin_airfoil_cl`).
- [ ] Tests: `baselines.py --out tmp` produces at least 14 rows, and the NACA0012 FVM 50×25 row has
      RMS 0.108 ± 0.002. `validate.py` on the tiny run from A10's test writes a figure and a row.
- [ ] Commit: `feat(research): validate.py and FVM/panel baselines`.

### Task A12a: Grid sequencing in `run_case.py` (found by the improvements research; needed for 100x50)

**Why:** a separate research session measured that plain NNfoil on the 100x50 mesh collapses to the trivial
solution (PDE loss exactly 0, Cl about 0) in every variant it tried. **Grid sequencing** fixed it: train on
50x25, move the weights to 100x50, continue. Measured there (one seed, `docs/nnfoil_improvements.md` §I1 and
Task 3b): Cl **0.5065** (2.4% error) and Cp relative-L1 0.027 after 4 fine rounds, with Cl 0.504 right after
the transfer. This is not in the NNfoil paper. Our own contribution for the report: label it so.

- [ ] Add `--init-from <run_dir>` to `NNfoil/run_case.py`. Read the coarse run's `config.json` (mesh size) and
      `model.pt`. Build the fine model, then `load_state_dict` the coarse weights.
- [ ] Rescale the input normalisation so the same *relative* grid position gives the same network input:
      `X_mean_fine = X_mean_coarse * s`, `X_std_fine = X_std_coarse * s`, with
      `s = [(M_f - 1) / (M_c - 1), (N_f - 1) / (N_c - 1)]` (M = xi rows including the closing row, N = eta
      columns). `X_mean` and `X_std` are plain attributes on `Net`, not buffers (finding N15 in the research
      doc), so set them explicitly after loading. Also check that `model.pt` of a run can be reloaded by
      `--resume` and `--init-from` alone: if the coarse run's mean/std are needed, recompute them from the
      coarse mesh (they are `XE.mean(0)` and `XE.std(0)`), so nothing extra needs to be stored.
- [ ] Skip the "first round -> re-initialise" branch of upstream `train()` for a warm-started model (it keys on
      `len(log['losses']) == 0`, which would throw the warm start away). Add the minimal change in
      `NNfoil_Euler2D.py` (a constructor flag or setting `nn.warm_started = True`), as its own commit.
- [ ] Record `"init_from": "<coarse run id>"` in `config.json`, and a `grid_sequencing` marker in the run id
      (`..._gs`) so the summary row is distinguishable.
- [ ] Tests (tiny nets, 1 thread): (1) the transfer keeps the network output at the same relative grid points:
      build a tiny 50x25 model, transfer to 100x50, and compare the output at ((xi, eta) = (25, 12)) vs
      ((50, 24)): equal to 1e-5; (2) `--init-from` run writes a folder with `init_from` in `config.json`.
- [ ] Commit: `feat(nnfoil): grid sequencing (--init-from) with rescaled input normalisation`.
- [ ] Acceptance (run for real in A12): 50x25 -> 100x50, Cl after the transfer (before fine training) near 0.50
      and after 4 fine rounds near 0.5065 (the measured target; one seed, so do not expect the third digit).

### Task A12: Real training runs, then the Milestone 1 pack

- [ ] Start in the background (roadmap rule 3), logging to `train.log`:
      1. `naca0012_50x25_a4_m04`: 20 rounds. Measured about 72 s per round, so about 25 min.
      2. `naca0012_100x50_a4_m04` (plain, from scratch). **Expected to collapse** (Cl about 0, per the
         research doc). Keep it as the baseline evidence and report it honestly. 20 rounds at about 100 s.
      3. `naca0012_50x25_a4_m04_nopos` with `--no-positivity` (needed for E1 later, and cheap).
      4. `naca0012_100x50_a4_m04_gs`: `--init-from results/runs/naca0012_50x25_a4_m04` (task A12a), 4-10
         fine rounds. This is the PINN 100x50 number for the headline table.
      Run them one after another, not in parallel (parallel runs ruin the timing numbers).
- [ ] While they run, start Plan B, or write the M1 doc skeleton.
- [ ] When each run finishes: `validate.py` against `FVM_Cp/NACA0012_800_400_Cp.dat` (experiment `E2`).
      Also validate the 50×25 run against `NACA0012_50_25_Cp.dat` and note "same-mesh FVM" in `notes`.
- [ ] Copy the figures to `results/figures/m1_*.png`.
- [ ] Write `docs/milestones/M1-nnfoil-verified.md` (at most 2 pages; every number from `summary.csv` or
      a check CSV):
      1. What NNfoil is (3 lines) and what we did.
      2. The findings table: #, file:line, problem, fix, commit hash, evidence (link the test or CSV).
         Findings 1–8 come from 7DP §2; 9 is the training cost; 10 is the Jacobian seam.
      3. CPU cost: s per round for 50×25 and 100×50, threads, CPU name.
      4. **The headline table: Cl and Cp RMS vs FVM 800×400 for FVM 50×25, PINN 50×25, FVM 100×50, PINN 100×50
         (plain, collapsed) and PINN 100×50 with grid sequencing, plus panel.** One sentence: does the PINN on a sparse mesh beat the FVM on the same
         mesh? (Measured early evidence: PINN 50×25 Cl 0.514 vs FVM 50×25 0.426, against a reference of 0.519.)
      5. The flow-condition findings (roadmap §1a items 2–3).
      6. Next steps (Milestones 2 and 3).
- [ ] Milestone check (roadmap §4): `research/.venv/Scripts/python -m pytest research/tests -q` passes,
      `summary.csv` has at least 8 rows and the figures exist.
- [ ] Commit: `docs: Milestone 1 - NNfoil verified on CPU`.
- [ ] **Stop and tell the user that M1 is ready to show the mentor.**
