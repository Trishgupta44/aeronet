# AeroNet changes to vendored NNfoil

Upstream: Cao, Song, Zhang (arXiv 2401.08705), vendored unmodified in commit 4732498.
Line numbers refer to the upstream files in that commit. `git log NNfoil/` shows one commit per finding.

| Finding | File:line (upstream) | Change | Commit |
|---|---|---|---|
| 1 | NNfoil_Euler2D.py:269 | `pick_device()` returns CPU when CUDA is unavailable (also `matplotlib.use("Agg")` in util.py so imports work headless) | 6b10ac1 |
| 5 | NNfoil_Euler2D.py:110 | `np.math.factorial` (removed in NumPy 2) replaced by `math.factorial` | c77caa2 |
| 4 | NNfoil_Euler2D.py:265 | `torch.set_num_threads(1)` changed to `min(8, os.cpu_count() or 1)` (`__main__` setting, no test) | 1bed181 |
| 6 | NNfoil_Euler2D.py:247,250 | Whole-module `torch.save`/`torch.load` (fails with `weights_only=True` default since PyTorch 2.6) replaced by `_save_ckpt`/`_load_ckpt` using `state_dict`, with a per-run `ckpt_path` argument | c475030 |
| 2 | NNfoil_Euler2D.py:193-197 (`Msef`) | `Net(..., positivity=True)`: softplus + 1e-3 on the p and rho outputs so `gamma*p/rho` and `1/rho` are defined. Flag `positivity` (default on) on `Net` and `NNfoil`, passed at both `Net(...)` construction sites | 12ba8c1 |
| 8 | NNfoil_Euler2D.py:55 | Hard-coded Ma 0.4 / alpha 4 deg became constructor arguments `Ma=0.4, alpha_deg=4.0`. Still one flow condition per model: the network input stays (xi, eta), as in the paper's code. The paper's continuous-AoA idea is not in this file. | 0822cad |
| 7 | NNfoil_Euler2D.py:207-210 | `Calculate_Force()` was never called. Added `evaluate_wall() -> (cl, cd)` (no_grad; recomputes `Cp_f` and `Cp` from the model, then calls `Calculate_Force()`); `__main__` prints Cl and Cd after training. | 0822cad |
