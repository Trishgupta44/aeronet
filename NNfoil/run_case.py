"""Train one NNfoil case and write a standard run folder (roadmap 5.3), rewritten after every round.

    python NNfoil/run_case.py --mesh NNfoil/mesh/NACA0012_100_50.x --shape NACA0012 --mach 0.4 --alpha 4 \
        --rounds 20 --threads 8 --out results/runs/naca0012_100x50_a4_m04

Run folder: config.json, log.json, snapshots.json, field.npz, model.pt. A killed run keeps everything up to the
last finished round; ``--resume`` continues from it (cumulative training time and round numbers continue).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

import NNfoil_Euler2D as nf

DEFAULT_LAYERS = [2, 128, 128, 128, 128, 128, 4]


# ---------------------------------------------------------------- naming
def _alpha_token(alpha: float) -> str:
    a = float(alpha)
    s = f"{abs(a):g}".replace(".", "p")
    return ("m" if a < 0 else "") + s


def _mach_token(mach: float) -> str:
    m10 = round(float(mach) * 10, 6)
    if abs(m10 - round(m10)) < 1e-6:
        return f"{int(round(m10)):02d}"
    return f"{int(round(float(mach) * 100)):03d}"


def make_run_id(shape: str, mesh: str, alpha: float, mach: float, positivity: bool = True, seed: int | None = None) -> str:
    """``<shape lowercase>_<NxM>_a<alpha>_m<mach>[_nopos][_s<seed>]`` (roadmap 5.8). Seed 0 (the default) has no suffix."""
    rid = f"{shape.lower()}_{mesh}_a{_alpha_token(alpha)}_m{_mach_token(mach)}"
    if not positivity:
        rid += "_nopos"
    if seed:
        rid += f"_s{int(seed)}"
    return rid


# ---------------------------------------------------------------- file helpers
def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf8")
    os.replace(tmp, path)


def _atomic_json(path: Path, obj) -> None:
    _atomic_write_text(path, json.dumps(obj))


def _atomic_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _atomic_state_dict(path: Path, model) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------- mesh
def load_mesh(path):
    """Closed (xx, yy) of shape (N_xi+1, N_eta) in the NNfoil frame, exactly as the upstream main script."""
    d = np.loadtxt(path, skiprows=1)
    h = round(d.shape[0] / 2)
    xx, yy = d[:h].T, d[h:].T
    return np.r_[xx, xx[0:1]], np.r_[yy, yy[0:1]]


# ---------------------------------------------------------------- run
def _field(nn) -> dict:
    with torch.no_grad():
        U = nn.model(nn.XE).cpu().numpy().reshape(nn.M, nn.N, 4)
    ma = float(nn.Ma)
    p = U[:, :, 2]
    return {
        "X": nn.xx0.cpu().numpy() + 0.5,
        "Y": nn.yy.cpu().numpy(),
        "u": U[:, :, 0],
        "v": U[:, :, 1],
        "p": p,
        "rho": U[:, :, 3],
        "cp": (p - 1.0 / (1.4 * ma**2)) / 0.5,
    }


def _write_outputs(out: Path, nn, log: dict, snaps: dict) -> None:
    _atomic_json(out / "log.json", log)
    _atomic_json(out / "snapshots.json", snaps)
    _atomic_npz(out / "field.npz", **_field(nn))
    _atomic_state_dict(out / "model.pt", nn.model)


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Train one NNfoil case and write a run folder.")
    ap.add_argument("--mesh", required=True, help="NNfoil .x mesh file")
    ap.add_argument("--shape", help="shape label (default: mesh file name up to the first underscore)")
    ap.add_argument("--mach", type=float, default=0.4)
    ap.add_argument("--alpha", type=float, default=4.0, help="angle of attack in degrees")
    ap.add_argument("--rounds", type=int, default=20, help="training rounds to run (attempts, failed ones included)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", help="run folder (default: results/runs/<run id>)")
    ap.add_argument("--no-positivity", action="store_true", help="disable the softplus positivity of p and rho")
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    ap.add_argument("--lbfgs-iter", type=int, default=1000, help="L-BFGS max_iter per round")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue the run in --out")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    layers = [int(v) for v in a.layers.split(",")]
    if layers[0] != 2 or layers[-1] != 4:
        raise SystemExit("--layers must start with 2 (xi, eta) and end with 4 (u, v, p, rho)")
    positivity = not a.no_positivity
    shape = a.shape or Path(a.mesh).stem.split("_")[0]

    xx, yy = load_mesh(a.mesh)
    mesh_label = f"{xx.shape[0] - 1}x{xx.shape[1]}"
    run_id = make_run_id(shape, mesh_label, a.alpha, a.mach, positivity, a.seed)
    out = Path(a.out) if a.out else Path("results") / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)

    cfg_path, log_path, snap_path = out / "config.json", out / "log.json", out / "snapshots.json"
    if a.resume:
        if not (cfg_path.exists() and log_path.exists() and snap_path.exists() and (out / "model.pt").exists()):
            raise SystemExit(f"--resume: {out} does not hold a finished round (config/log/snapshots/model.pt)")
        cfg = json.loads(cfg_path.read_text(encoding="utf8"))
        for key, val in [("mesh", mesh_label), ("mach", a.mach), ("alpha_deg", a.alpha), ("layers", layers),
                         ("positivity", positivity)]:
            if cfg[key] != val:
                raise SystemExit(f"--resume: {key} differs from the stored run ({cfg[key]!r} vs {val!r})")
        log = json.loads(log_path.read_text(encoding="utf8"))
        snaps = json.loads(snap_path.read_text(encoding="utf8"))
        cfg["rounds"] = cfg["rounds"] + a.rounds
        cfg["threads"] = a.threads
        cfg["lbfgs_max_iter"] = a.lbfgs_iter
    else:
        if log_path.exists():
            raise SystemExit(f"{out} already holds a run; pass --resume to continue it or choose another --out")
        cfg = {
            "run_id": out.name, "mesh_file": str(a.mesh).replace("\\", "/"), "mesh": mesh_label, "shape": shape,
            "mach": a.mach, "alpha_deg": a.alpha, "layers": layers, "rounds": a.rounds,
            "lbfgs_max_iter": a.lbfgs_iter, "threads": a.threads, "positivity": positivity, "seed": a.seed,
            "torch_version": torch.__version__,
            "started_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        log = {"round": [], "time_s": [], "loss": [], "loss_f": [], "loss_b": [], "cl": [], "cd": [],
               "failed_rounds": 0}
        snaps = None
    _atomic_json(cfg_path, cfg)

    done = len(log["round"])
    torch.set_num_threads(a.threads)
    # upstream seeds 0 at import; a resumed run offsets the seed so it draws different collocation batches
    np.random.seed(a.seed + done)
    torch.manual_seed(a.seed + done)

    nn = nf.NNfoil(layers, nf.pick_device(), xx, yy, a.mach, a.alpha, positivity, ckpt_path=str(out / "model.pt"),
                   lbfgs_max_iter=a.lbfgs_iter)
    if snaps is None:
        snaps = {"x": (nn.wall["X_f"][:, 0] + 0.5).cpu().tolist(), "y": nn.wall["X_f"][:, 1].cpu().tolist(),
                 "rounds": []}
    if a.resume:
        nn._load_ckpt()
        nn.log["losses"] = list(log["loss"])
        nn.log["losses_f"] = list(log["loss_f"])
        nn.log["losses_b"] = list(log["loss_b"])
        nn.log["time"] = list(log["time_s"])  # cumulative: train() continues from the last value

    for _ in range(a.rounds):
        before = len(nn.log["losses"])
        nn.train(1)
        if len(nn.log["losses"]) == before:  # upstream restarted / reloaded silently: a failed round
            log["failed_rounds"] += 1
            print(f"round failed (total failed: {log['failed_rounds']})", flush=True)
            _atomic_json(log_path, log)
            continue
        cl, cd = nn.evaluate_wall()
        r = len(log["round"]) + 1
        log["round"].append(r)
        log["time_s"].append(nn.log["time"][-1])
        log["loss"].append(nn.log["losses"][-1])
        log["loss_f"].append(nn.log["losses_f"][-1])
        log["loss_b"].append(nn.log["losses_b"][-1])
        log["cl"].append(cl)
        log["cd"].append(cd)
        snaps["rounds"].append({"round": r, "cp": nn.wall["Cp_f"][:, 0].cpu().tolist()})
        _write_outputs(out, nn, log, snaps)
        print(f"round {r}: loss={log['loss'][-1]:.4g} Cl={cl:.4f} Cd={cd:.4f} t={log['time_s'][-1]:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
