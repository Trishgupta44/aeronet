# AeroNet changes to vendored NNfoil

Upstream: Cao, Song, Zhang (arXiv 2401.08705), vendored unmodified in commit 4732498.
Line numbers refer to the upstream files in that commit. `git log NNfoil/` shows one commit per finding.

| Finding | File:line (upstream) | Change | Commit |
|---|---|---|---|
| 1 | NNfoil_Euler2D.py:269 | `pick_device()` returns CPU when CUDA is unavailable (also `matplotlib.use("Agg")` in util.py so imports work headless) | <hash-1> |
| 5 | NNfoil_Euler2D.py:110 | `np.math.factorial` (removed in NumPy 2) replaced by `math.factorial` | <hash-5> |
| 4 | NNfoil_Euler2D.py:265 | `torch.set_num_threads(1)` changed to `min(8, os.cpu_count() or 1)` (`__main__` setting, no test) | <hash-4> |
| 6 | NNfoil_Euler2D.py:247,250 | Whole-module `torch.save`/`torch.load` (fails with `weights_only=True` default since PyTorch 2.6) replaced by `_save_ckpt`/`_load_ckpt` using `state_dict`, with a per-run `ckpt_path` argument | <hash-6> |
| 2 | NNfoil_Euler2D.py:193-197 (`Msef`) | `Net(..., positivity=True)`: softplus + 1e-3 on the p and rho outputs so `gamma*p/rho` and `1/rho` are defined. Flag `positivity` (default on) on `Net` and `NNfoil`, passed at both `Net(...)` construction sites | <hash-2> |
