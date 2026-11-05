"""Helpers for run folders (roadmap 5.3) and the training commands of the experiments (Plan C).

No torch at import time: only :func:`build_model` imports torch and NNfoil.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
RUN_CASE = ROOT / "NNfoil" / "run_case.py"
FVM = {
    "NACA0012": ROOT / "NNfoil" / "FVM_Cp" / "NACA0012_800_400_Cp.dat",
    "Cylinder": ROOT / "NNfoil" / "FVM_Cp" / "Cylinder_800_400_Cp.dat",
}


def fvm_2412(alpha) -> Path:
    """The 800x400 FVM Cp file of NACA 2412 at an integer angle of attack (-10..10 in steps of 2)."""
    return ROOT / "NNfoil" / "FVM_Cp_NACA2412" / f"NACA2412_800_400_AoA{int(alpha)}_Cp.dat"


# ---------------------------------------------------------------- naming (same rules as NNfoil/run_case.py)
def _alpha_token(alpha: float) -> str:
    s = f"{abs(float(alpha)):g}".replace(".", "p")
    return ("m" if float(alpha) < 0 else "") + s


def _mach_token(mach: float) -> str:
    m10 = round(float(mach) * 10, 6)
    if abs(m10 - round(m10)) < 1e-6:
        return f"{int(round(m10)):02d}"
    return f"{int(round(float(mach) * 100)):03d}"


def run_id(shape, mesh, alpha, mach, positivity=True, seed=None, grid_seq=False) -> str:
    """``<shape lowercase>_<NxM>_a<alpha>_m<mach>[_gs][_nopos][_s<seed>]`` (roadmap 5.8; seed 0 has no suffix)."""
    rid = f"{shape.lower()}_{mesh}_a{_alpha_token(alpha)}_m{_mach_token(mach)}"
    if grid_seq:
        rid += "_gs"
    if not positivity:
        rid += "_nopos"
    if seed:
        rid += f"_s{int(seed)}"
    return rid


def train_cmd(mesh_file, shape, mach, alpha, rounds, out, positivity=True, seed=0, threads=8, layers=None,
              lbfgs_iter=None, init_from=None, resume=False) -> list[str]:
    """The ``NNfoil/run_case.py`` command (roadmap 5.8) as an argument list, run with the current interpreter."""
    cmd = [sys.executable, str(RUN_CASE), "--mesh", str(mesh_file), "--shape", shape, "--mach", f"{float(mach):g}",
           "--alpha", f"{float(alpha):g}", "--rounds", str(int(rounds)), "--threads", str(int(threads)),
           "--out", str(out), "--seed", str(int(seed))]
    if not positivity:
        cmd.append("--no-positivity")
    if layers is not None:
        cmd += ["--layers", ",".join(str(int(v)) for v in layers)]
    if lbfgs_iter is not None:
        cmd += ["--lbfgs-iter", str(int(lbfgs_iter))]
    if init_from is not None:
        cmd += ["--init-from", str(init_from)]
    if resume:
        cmd.append("--resume")
    return cmd


# ---------------------------------------------------------------- reading a run folder
def load_run(run_dir) -> dict:
    """``{"dir","config","log","snapshots","field"}``; ``field`` is ``dict(np.load(field.npz))`` or None."""
    d = Path(run_dir)

    def js(name):
        return json.loads((d / name).read_text(encoding="utf8"))

    field = None
    if (d / "field.npz").exists():
        with np.load(d / "field.npz") as z:
            field = dict(z)
    return {"dir": d, "config": js("config.json"), "log": js("log.json"), "snapshots": js("snapshots.json"),
            "field": field}


def attempts(run) -> int:
    """Training rounds attempted so far: finished rounds plus failed ones (what ``--rounds`` counts)."""
    log = run["log"]
    return len(log["round"]) + int(log.get("failed_rounds", 0))


def wall_faces(snapshots):
    """``(x, y)`` of the wall points that the Cp values belong to.

    Plan A stores face centres (same length as ``cp``): returned as they are. If ``x`` has one more entry than ``cp``
    (closed node list), the midpoints of consecutive nodes are returned instead.
    """
    x, y = np.asarray(snapshots["x"], dtype=float), np.asarray(snapshots["y"], dtype=float)
    n_cp = len(snapshots["rounds"][-1]["cp"])
    if len(x) == n_cp + 1:
        return 0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:])
    return x, y


def final_surface(run):
    """``(x, y, cp)`` of the last snapshot (wall faces, app frame)."""
    x, y = wall_faces(run["snapshots"])
    return x, y, np.asarray(run["snapshots"]["rounds"][-1]["cp"], dtype=float)


def count_restarts(run_dir) -> int:
    """Lines starting with ``Failed`` in ``train.log`` (``Failed, Restart`` and ``Failed, Load previous model``)."""
    p = Path(run_dir) / "train.log"
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text(encoding="utf8", errors="replace").splitlines() if line.startswith("Failed"))


def build_model(run_dir, threads=8):
    """Rebuild the trained ``NNfoil`` of a run (weights from ``model.pt``, eval mode) for inference timing and fields."""
    sys.path.insert(0, str(ROOT / "NNfoil"))
    import torch

    import metrics
    import NNfoil_Euler2D as nf

    torch.set_num_threads(int(threads))
    d = Path(run_dir)
    cfg = json.loads((d / "config.json").read_text(encoding="utf8"))
    xx, yy = metrics.load_mesh(ROOT / cfg["mesh_file"] if not Path(cfg["mesh_file"]).is_absolute() else cfg["mesh_file"])
    nn = nf.NNfoil(cfg["layers"], torch.device("cpu"), xx, yy, Ma=cfg["mach"], alpha_deg=cfg["alpha_deg"],
                   positivity=cfg.get("positivity", True), ckpt_path=str(d / "_tmp.pth"))
    if cfg.get("init_from_mesh"):  # a grid-sequenced run was trained with the rescaled input normalisation
        import run_case

        run_case.apply_grid_seq_norm(nn, cfg["init_from_mesh"])
    nn.model.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True))
    nn.model.eval()
    return nn
