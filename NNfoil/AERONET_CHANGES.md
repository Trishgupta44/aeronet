# AeroNet changes to vendored NNfoil

Upstream: Cao, Song, Zhang (arXiv 2401.08705), vendored unmodified in commit 4732498.
Line numbers refer to the upstream files in that commit. `git log NNfoil/` shows one commit per finding.

| Finding | File:line (upstream) | Change | Commit |
|---|---|---|---|
| 1 | NNfoil_Euler2D.py:269 | `pick_device()` returns CPU when CUDA is unavailable (also `matplotlib.use("Agg")` in util.py so imports work headless) | <hash-1> |
