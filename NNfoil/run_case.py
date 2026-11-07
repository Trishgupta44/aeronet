"""Train one NNfoil case and write a standard run folder (roadmap 5.3), rewritten after every round.

    python NNfoil/run_case.py --mesh NNfoil/mesh/NACA0012_100_50.x --shape NACA0012 --mach 0.4 --alpha 4 \
        --rounds 20 --threads 8 --out results/runs/naca0012_100x50_a4_m04

Run folder: config.json, log.json, snapshots.json, field.npz, model.pt. A killed run keeps everything up to the
last finished round; ``--resume`` continues from it (cumulative training time and round numbers continue).

Grid sequencing (AeroNet's own addition, not in the NNfoil paper): ``--init-from <coarse run dir>`` starts the
network from a run on a coarser mesh of the same airfoil, with the input normalisation rescaled so that the same
relative grid position gives the same network input (``transfer_from_coarse``)::

    python NNfoil/run_case.py --mesh NNfoil/mesh/NACA0012_100_50.x --shape NACA0012 --mach 0.4 --alpha 4 \
        --rounds 4 --threads 8 --init-from results/runs/naca0012_50x25_a4_m04
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


def make_run_id(shape: str, mesh: str, alpha: float, mach: float, positivity: bool = True, seed: int | None = None,
                grid_seq: bool = False) -> str:
    """``<shape lowercase>_<NxM>_a<alpha>_m<mach>[_gs][_nopos][_s<seed>]`` (roadmap 5.8).

    The marker order is the order in which a run is described: the method (``_gs`` = grid-sequenced, started from a
    coarser run) comes first, then the ablation flag (``_nopos``), then the repeat index (``_s<seed>``), so a seed
    suffix is always last and ``..._gs_s1`` / ``..._gs_nopos_s1`` sort next to their ``..._gs`` base run.
    Seed 0 (the default) has no suffix.
    """
    rid = f"{shape.lower()}_{mesh}_a{_alpha_token(alpha)}_m{_mach_token(mach)}"
    if grid_seq:
        rid += "_gs"
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


# ---------------------------------------------------------------- grid sequencing
def _mesh_dims(label: str) -> tuple[int, int]:
    """``"50x25"`` -> (N_xi, N_eta) = (50, 25). The network sees M = N_xi + 1 xi rows (closing row) and N = N_eta columns."""
    n_xi, n_eta = (int(v) for v in label.split("x"))
    return n_xi, n_eta


def _index_grid_stats(label: str, like: torch.Tensor):
    """``XE.mean(0)`` and ``XE.std(0)`` of a mesh, exactly as ``Net.__init__`` computes them (shape (1, 2))."""
    n_xi, n_eta = _mesh_dims(label)
    xxi, eeta = torch.meshgrid(torch.arange(n_xi + 1), torch.arange(n_eta), indexing="ij")
    xe = torch.cat([xxi.reshape(-1, 1), eeta.reshape(-1, 1)], dim=1).to(like.device).to(like.dtype)
    return xe.mean(0, keepdim=True), xe.std(0, keepdim=True)


def apply_grid_seq_norm(nn, coarse_label: str) -> None:
    """Set ``nn.model.X_mean/X_std`` to the coarse mesh's values times ``s = [(M_f-1)/(M_c-1), (N_f-1)/(N_c-1)]``.

    ``X_mean`` / ``X_std`` are plain tensors on ``Net`` (not buffers, so not in ``state_dict``): ``Net.__init__``
    derives them from the fine mesh, which would shift every input of a transferred network. The network input is the
    grid index (xi, eta), so with this scaling a relative grid position gives the same normalised input on both
    meshes. The coarse statistics depend only on the coarse index grid, so the mesh label is enough.
    """
    n_xi_c, n_eta_c = _mesh_dims(coarse_label)
    s = torch.tensor([(nn.M - 1) / n_xi_c, (nn.N - 1) / (n_eta_c - 1)]).to(nn.XE.device).to(nn.XE.dtype)
    mean_c, std_c = _index_grid_stats(coarse_label, nn.XE)
    nn.model.X_mean = mean_c * s
    nn.model.X_std = std_c * s


def transfer_from_coarse(nn, coarse_dir) -> dict:
    """Warm-start ``nn`` (fine mesh) from the run folder ``coarse_dir`` and rescale the input normalisation.

    Returns the coarse run's ``config.json``. Weights are loaded through ``NNfoil.set_warm_start`` so that a failed
    first round reloads them instead of re-initialising the network.
    """
    coarse_dir = Path(coarse_dir)
    cfg_c = json.loads((coarse_dir / "config.json").read_text(encoding="utf8"))
    if cfg_c["layers"] != list(nn.layers):
        raise SystemExit(f"--init-from: the coarse run has layers {cfg_c['layers']}, this run {list(nn.layers)}")
    state = torch.load(coarse_dir / "model.pt", map_location=nn.device, weights_only=True)
    nn.set_warm_start(state)
    apply_grid_seq_norm(nn, cfg_c["mesh"])
    return cfg_c


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


# ---------------------------------------------------------------- reference-free diagnostics and stop rules
# AeroNet's own addition (not in the NNfoil paper). Everything here is opt-in (``--diagnostics``, ``--stop-rule``) and
# uses no reference solution, so it can be used to decide when to stop without looking at the FVM answer.
GAMMA = 1.4
INNER_FRACTION = 0.25  # "inner" = the eta rows closest to the wall (first quarter of the rows)
SEAM_CELLS = 3         # unweighted diagnostics leave out this many xi-cells at each end (the trailing-edge seam)


def cell_centre_metrics(xx, yy) -> dict:
    """Mesh metrics at the centres of the (M-1) x (N-1) cells of a closed mesh, i.e. at (xi + 0.5, eta + 0.5).

    ``xx``, ``yy`` are the (M, N) node arrays (the last xi row closes the loop). The centre metrics are the average of
    the two parallel cell edges, which is second-order accurate. They have the same layout as ``Mesh()``:
    ``jac = [[x_xi, y_xi], [x_eta, y_eta]]`` so that ``[q_x, q_y] = jac_inv @ [q_xi, q_eta]``, and ``J = det(jac)``.
    Returns numpy arrays ``XE`` (n, 2) (the (xi, eta) of each centre), ``jac_inv`` (n, 2, 2), ``J`` (n,) and the
    integer row index ``j`` (n,) of each centre.
    """
    x, y = np.asarray(xx, dtype=float), np.asarray(yy, dtype=float)

    def d_xi(a):
        return 0.5 * ((a[1:, :-1] - a[:-1, :-1]) + (a[1:, 1:] - a[:-1, 1:]))

    def d_eta(a):
        return 0.5 * ((a[:-1, 1:] - a[:-1, :-1]) + (a[1:, 1:] - a[1:, :-1]))

    x_xi, y_xi, x_eta, y_eta = d_xi(x), d_xi(y), d_eta(x), d_eta(y)
    det = x_xi * y_eta - y_xi * x_eta
    inv = np.stack([y_eta, -y_xi, -x_eta, x_xi], axis=-1) / det[..., None]
    m, n = det.shape
    ii, jj = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    return {"XE": np.stack([ii.ravel() + 0.5, jj.ravel() + 0.5], axis=1), "jac_inv": inv.reshape(-1, 2, 2),
            "J": det.ravel(), "j": jj.ravel()}


def euler_residual_sq(model, XE, jac_inv, gamma: float = GAMMA):
    """Squared Euler residual ``e1^2 + e2^2 + e3^2 + e4^2`` at the points ``XE`` (n, 2), one value per point.

    Same four equations as ``NNfoil.Msef`` (continuity, two momentum, energy); ``jac_inv`` (n, 2, 2) maps the
    (xi, eta) derivatives to (x, y). Not weighted, not scaled by the 2e4 loss factor.
    """
    with torch.enable_grad():
        XE = XE.clone().requires_grad_(True)
        pred = model(XE)
        grads = torch.stack([nf.fwd_gradients(pred[:, k:k + 1], XE) for k in range(4)], dim=1)  # (n, 4, 2)
        xy = torch.einsum("nij,nkj->nki", jac_inv.to(grads.dtype), grads)  # (n, 4, 2): d/dx, d/dy of u, v, p, rho
        u, v, p, rho = (pred[:, k] for k in range(4))
        u_x, u_y, v_x, v_y = xy[:, 0, 0], xy[:, 0, 1], xy[:, 1, 0], xy[:, 1, 1]
        p_x, p_y, rho_x, rho_y = xy[:, 2, 0], xy[:, 2, 1], xy[:, 3, 0], xy[:, 3, 1]
        a2 = gamma * p / rho
        e1 = rho * u_x + u * rho_x + rho * v_y + v * rho_y
        e2 = u * u_x + v * u_y + p_x / rho
        e3 = u * v_x + v * v_y + p_y / rho
        e4 = rho * a2 * u_x + rho * a2 * v_y + u * p_x + v * p_y
        return (e1 ** 2 + e2 ** 2 + e3 ** 2 + e4 ** 2).detach()


def _inner_rows(n_eta: int) -> int:
    """Number of wall-nearest eta rows counted as "inner" (a quarter of the rows, at least one)."""
    return max(1, int(round(INNER_FRACTION * (n_eta - 1))))


def diagnostics_context(nn) -> dict:
    """Precompute the points, weights and masks the per-round diagnostics need (once per run)."""
    ref = nn.XE
    dev, dt = ref.device, ref.dtype
    metr = cell_centre_metrics(nn.xx0.detach().cpu().numpy(), nn.yy.detach().cpu().numpy())
    n_inner = _inner_rows(nn.N)
    m_last = nn.M - 1  # xi index of the closing row; the trailing-edge seam is at xi = 0 and xi = m_last
    xi_n = nn.xxi.reshape(-1).cpu().numpy()
    eta_n = nn.eeta.reshape(-1).cpu().numpy()
    # unweighted means leave out the cells / nodes next to the seam: the mesh metrics blow up there (J -> 0)
    cell_ok = (metr["XE"][:, 0] > SEAM_CELLS) & (metr["XE"][:, 0] < m_last - SEAM_CELLS)
    node_ok = (xi_n >= SEAM_CELLS) & (xi_n <= m_last - SEAM_CELLS)
    t = lambda arr, **kw: torch.as_tensor(arr, device=dev, **kw)  # noqa: E731
    return {
        "cell_XE": t(metr["XE"], dtype=dt), "cell_jac_inv": t(metr["jac_inv"], dtype=dt),
        "cell_J2": t(metr["J"] ** 2, dtype=dt),
        "cell_ok": t(cell_ok), "cell_inner_ok": t(cell_ok & (metr["j"] < n_inner)),
        "node_XE": ref, "node_jac_inv": nn.Jac_inv, "node_J2": nn.J[:, 0] ** 2,
        "node_ok": t(node_ok), "node_inner_ok": t(node_ok & (eta_n < n_inner)),
        "n_inner": n_inner,
    }


def round_diagnostics(nn, ctx: dict) -> dict:
    """Reference-free numbers for the current model (plain floats; raw residuals without the 2e4 factor).

    ``*_w`` use the loss's own J^2 weighting (which switches off the near-singular cells), the others are plain means
    over the points away from the trailing-edge seam (``SEAM_CELLS`` xi-cells at each end left out).
    "node" = the mesh nodes the network is trained on; "off" = the cell centres (xi + 0.5, eta + 0.5), which the
    network never sees as training points but where the wall Cp is read.

    * ``res_node_w``, ``res_off_w``       weighted mean squared residual (``res_node_w`` * 2e4 is the PDE loss on all nodes)
    * ``res_node``, ``res_off``           unweighted mean away from the seam
    * ``res_node_inner``, ``res_off_inner`` the same over the inner quarter of the eta rows only
    * ``wall_vn_rms``                      RMS normal velocity at the wall face centres (wall slip; V_inf = 1)
    """
    r_c = euler_residual_sq(nn.model, ctx["cell_XE"], ctx["cell_jac_inv"])
    r_n = euler_residual_sq(nn.model, ctx["node_XE"], ctx["node_jac_inv"])
    wc, wn = ctx["cell_J2"], ctx["node_J2"]
    with torch.no_grad():
        pred = nn.model(nn.wall["XE_f"])
        vn = pred[:, 0:1] * nn.wall["nor_f"][:, 0:1] + pred[:, 1:2] * nn.wall["nor_f"][:, 1:2]
    return {
        "res_node_w": float((r_n * wn).sum() / wn.sum()), "res_off_w": float((r_c * wc).sum() / wc.sum()),
        "res_node": float(r_n[ctx["node_ok"]].mean()), "res_off": float(r_c[ctx["cell_ok"]].mean()),
        "res_node_inner": float(r_n[ctx["node_inner_ok"]].mean()),
        "res_off_inner": float(r_c[ctx["cell_inner_ok"]].mean()),
        "wall_vn_rms": float(torch.sqrt((vn ** 2).mean())),
    }


def round_changes(cl, cp, prev_cl, prev_cp):
    """(|Cl - Cl_prev|, RMS(Cp - Cp_prev) over the wall faces), or (None, None) for the first round."""
    if prev_cl is None or prev_cp is None:
        return None, None
    return abs(float(cl) - float(prev_cl)), float(np.sqrt(np.mean((np.asarray(cp) - np.asarray(prev_cp)) ** 2)))


def record_diag(log: dict, rec: dict) -> None:
    """Append one round's diagnostics to ``log["diag"]`` so every list has one entry per logged round (None = n/a)."""
    n = len(log["round"])  # the round being recorded is already in log["round"]
    diag = log.setdefault("diag", {})
    for key in set(diag) | set(rec):
        lst = diag.setdefault(key, [])
        lst.extend([None] * (n - 1 - len(lst)))
        lst.append(rec.get(key))


def parse_stop_rule(text):
    """``None``/``"none"`` -> ("none", None); ``"plateau"`` -> ("plateau", None); ``"fixed:N"`` -> ("fixed", N)."""
    if text is None or text == "none":
        return "none", None
    if text == "plateau":
        return "plateau", None
    if text.startswith("fixed:"):
        try:
            n = int(text[len("fixed:"):])
        except ValueError:
            n = 0
        if n >= 1:
            return "fixed", n
    raise SystemExit(f"--stop-rule must be none, plateau or fixed:N (N >= 1), got {text!r}")


def plateau_reached(d_cl, d_cp, dcl_tol: float = 0.002, dcp_tol: float = 0.005, window: int = 2) -> bool:
    """True when the last ``window`` rounds all have |dCl| < dcl_tol and a wall-Cp RMS change < dcp_tol."""
    if window < 1 or len(d_cl) < window or len(d_cp) < window:
        return False
    pairs = list(zip(d_cl[-window:], d_cp[-window:]))
    return all(a is not None and b is not None and a < dcl_tol and b < dcp_tol for a, b in pairs)


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
    ap.add_argument("--init-from", help="grid sequencing: coarse run folder (config.json + model.pt) to start from")
    ap.add_argument("--diagnostics", action="store_true",
                    help="after every round also log reference-free diagnostics (off-grid and inner-row residual, wall "
                         "slip, round-to-round change) to log.json under 'diag'; costs one extra residual evaluation per round")
    ap.add_argument("--stop-rule", default="none",
                    help="none (default) | fixed:N (stop after N logged rounds) | plateau (stop when |dCl| and the wall-Cp "
                         "change stay below the --plateau-* limits for --plateau-window rounds); --rounds stays the maximum")
    ap.add_argument("--plateau-dcl", type=float, default=0.002)
    ap.add_argument("--plateau-dcp", type=float, default=0.005)
    ap.add_argument("--plateau-window", type=int, default=2)
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
    if a.init_from and a.resume:
        raise SystemExit("--init-from applies to a new run; a resumed grid-sequenced run remembers its origin")
    if a.init_from and not (Path(a.init_from) / "config.json").exists():
        raise SystemExit(f"--init-from: {a.init_from} has no config.json")
    run_id = make_run_id(shape, mesh_label, a.alpha, a.mach, positivity, a.seed, grid_seq=bool(a.init_from))
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
        init_mesh = cfg.get("init_from_mesh")  # a grid-sequenced run: its weights expect the scaled normalisation
    else:
        init_mesh = None
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
    coarse_cfg = None
    if a.init_from:
        coarse_cfg = json.loads((Path(a.init_from) / "config.json").read_text(encoding="utf8"))
        cfg["init_from"] = coarse_cfg.get("run_id", Path(a.init_from).name)
        cfg["init_from_mesh"] = coarse_cfg["mesh"]  # lets --resume rebuild the scaled normalisation without the coarse run
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
    if coarse_cfg is not None:
        transfer_from_coarse(nn, a.init_from)
        log["cl_after_transfer"] = nn.evaluate_wall()[0]  # the coarse answer on the fine grid, before fine training
        print(f"transferred {cfg['init_from']} ({coarse_cfg['mesh']}) -> {mesh_label}: "
              f"Cl after transfer = {log['cl_after_transfer']:.4f}", flush=True)
    if a.resume:
        if init_mesh:
            apply_grid_seq_norm(nn, init_mesh)
        nn._load_ckpt()
        nn.log["losses"] = list(log["loss"])
        nn.log["losses_f"] = list(log["loss_f"])
        nn.log["losses_b"] = list(log["loss_b"])
        nn.log["time"] = list(log["time_s"])  # cumulative: train() continues from the last value

    stop_kind, stop_n = parse_stop_rule(a.stop_rule)
    want_changes = a.diagnostics or stop_kind == "plateau"  # the cheap round-to-round numbers
    if stop_kind != "none" or a.diagnostics:  # recorded in config.json only when used, so old run folders look unchanged
        cfg["stop_rule"] = a.stop_rule
        cfg["diagnostics"] = bool(a.diagnostics)
        _atomic_json(cfg_path, cfg)
    ctx = diagnostics_context(nn) if a.diagnostics else None
    if stop_kind == "fixed" and len(log["round"]) >= stop_n:
        print(f"stop rule {a.stop_rule}: {len(log['round'])} rounds already logged, nothing to do", flush=True)
        return 0

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
        if want_changes:
            prev_cl = log["cl"][-1] if log["cl"] else None
            prev_cp = snaps["rounds"][-1]["cp"] if snaps["rounds"] else None
        log["round"].append(r)
        log["time_s"].append(nn.log["time"][-1])
        log["loss"].append(nn.log["losses"][-1])
        log["loss_f"].append(nn.log["losses_f"][-1])
        log["loss_b"].append(nn.log["losses_b"][-1])
        log["cl"].append(cl)
        log["cd"].append(cd)
        snaps["rounds"].append({"round": r, "cp": nn.wall["Cp_f"][:, 0].cpu().tolist()})
        if want_changes:
            d_cl, d_cp = round_changes(cl, snaps["rounds"][-1]["cp"], prev_cl, prev_cp)
            rec = {"d_cl": d_cl, "d_cp": d_cp}
            if a.diagnostics:
                rec.update(round_diagnostics(nn, ctx))
            record_diag(log, rec)
        stop = None
        if stop_kind == "fixed" and r >= stop_n:
            stop = a.stop_rule
        elif stop_kind == "plateau" and plateau_reached(log["diag"]["d_cl"], log["diag"]["d_cp"], a.plateau_dcl,
                                                         a.plateau_dcp, a.plateau_window):
            stop = f"plateau (dCl<{a.plateau_dcl}, dCp<{a.plateau_dcp}, {a.plateau_window} rounds)"
        if stop:
            log["stop"] = {"rule": stop, "round": r}
        _write_outputs(out, nn, log, snaps)
        extra = "" if not want_changes or d_cp is None else f" dCl={d_cl:.4f} dCp={d_cp:.4f}"
        print(f"round {r}: loss={log['loss'][-1]:.4g} Cl={cl:.4f} Cd={cd:.4f} t={log['time_s'][-1]:.1f}s{extra}", flush=True)
        if stop:
            print(f"stopped by rule {stop} after round {r}", flush=True)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
