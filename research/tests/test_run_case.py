import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

import NNfoil_Euler2D as nf
import run_case

FILES = ["config.json", "log.json", "snapshots.json", "model.pt", "field.npz"]


@pytest.mark.parametrize(
    "kw, expected",
    [
        (dict(shape="NACA0012", mesh="100x50", alpha=4, mach=0.4), "naca0012_100x50_a4_m04"),
        (dict(shape="NACA2412", mesh="100x50", alpha=-4, mach=0.25, positivity=False), "naca2412_100x50_am4_m025_nopos"),
        (dict(shape="NACA0012", mesh="50x25", alpha=2.5, mach=0.4, seed=1), "naca0012_50x25_a2p5_m04_s1"),
        (dict(shape="NACA0012", mesh="50x25", alpha=-2.5, mach=0.4, positivity=False, seed=2),
         "naca0012_50x25_am2p5_m04_nopos_s2"),
        (dict(shape="Cylinder", mesh="200x100", alpha=0, mach=0.4, seed=0), "cylinder_200x100_a0_m04"),
    ],
)
def test_make_run_id(kw, expected):
    assert run_case.make_run_id(**kw) == expected


def test_lbfgs_max_iter_is_a_constructor_argument(mesh_50x25):
    xx, yy = mesh_50x25
    dev = torch.device("cpu")
    assert nf.NNfoil([2, 8, 4], dev, xx, yy).lbfgs_max_iter == 1000
    assert nf.NNfoil([2, 8, 4], dev, xx, yy, lbfgs_max_iter=7).lbfgs_max_iter == 7


def test_run_folder_files_and_shapes(tiny_run):
    for f in FILES:
        assert (tiny_run / f).exists(), f
    cfg = json.loads((tiny_run / "config.json").read_text())
    for key in ["run_id", "mesh_file", "mesh", "shape", "mach", "alpha_deg", "layers", "rounds", "lbfgs_max_iter",
                "threads", "positivity", "seed", "torch_version", "started_utc"]:
        assert key in cfg, key
    assert cfg["mesh"] == "50x25" and cfg["layers"] == [2, 8, 8, 4] and cfg["lbfgs_max_iter"] == 5
    assert cfg["run_id"] == "tiny" or cfg["run_id"] == tiny_run.name

    log = json.loads((tiny_run / "log.json").read_text())
    assert log["round"] == [1, 2]
    for k in ["time_s", "loss", "loss_f", "loss_b", "cl", "cd"]:
        assert len(log[k]) == 2 and all(np.isfinite(log[k])), k
    assert log["time_s"][1] >= log["time_s"][0] > 0
    assert log["failed_rounds"] == 0

    snap = json.loads((tiny_run / "snapshots.json").read_text())
    assert len(snap["x"]) == len(snap["y"]) == 50
    assert [r["round"] for r in snap["rounds"]] == [1, 2]
    assert all(len(r["cp"]) == 50 for r in snap["rounds"])
    assert 0.0 <= min(snap["x"]) and max(snap["x"]) <= 1.0  # app frame

    f = np.load(tiny_run / "field.npz")
    for k in ["X", "Y", "u", "v", "p", "rho", "cp"]:
        assert f[k].shape == (51, 25), k
    assert f["X"][:, 0].min() >= 0.0 and f["X"][:, 0].max() <= 1.0  # wall row, app frame
    assert np.allclose(f["cp"], (f["p"] - 1 / (1.4 * 0.4**2)) / 0.5, atol=1e-4)

    sd = torch.load(tiny_run / "model.pt", weights_only=True)
    assert isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values())


def test_resume_continues_rounds_and_time(tiny_run, tmp_path, repo):
    run = tmp_path / "resumed"
    shutil.copytree(tiny_run, run)
    before = json.loads((run / "log.json").read_text())
    rc = run_case.main(["--mesh", str(repo / "NNfoil" / "mesh" / "NACA0012_50_25.x"), "--shape", "NACA0012",
                        "--layers", "2,8,8,4", "--lbfgs-iter", "5", "--rounds", "1", "--threads", "2",
                        "--out", str(run), "--resume"])
    assert rc == 0
    log = json.loads((run / "log.json").read_text())
    assert log["round"] == [1, 2, 3]
    assert log["time_s"][:2] == before["time_s"]
    assert log["time_s"][2] > log["time_s"][1]  # cumulative time continues
    snap = json.loads((run / "snapshots.json").read_text())
    assert [r["round"] for r in snap["rounds"]] == [1, 2, 3]
    assert json.loads((run / "config.json").read_text())["rounds"] == 3


def test_existing_run_without_resume_is_refused(tiny_run, repo):
    with pytest.raises(SystemExit):
        run_case.main(["--mesh", str(repo / "NNfoil" / "mesh" / "NACA0012_50_25.x"), "--shape", "NACA0012",
                       "--layers", "2,8,8,4", "--lbfgs-iter", "5", "--rounds", "1", "--out", str(tiny_run)])


def test_failed_round_is_counted_not_logged(tmp_path, repo, monkeypatch):
    real = nf.NNfoil.train
    calls = {"n": 0}

    def flaky(self, epoch):
        calls["n"] += 1
        if calls["n"] == 2:  # simulate upstream's silent "Failed, Load previous model"
            return
        return real(self, epoch)

    monkeypatch.setattr(nf.NNfoil, "train", flaky)
    out = tmp_path / "flaky"
    run_case.main(["--mesh", str(repo / "NNfoil" / "mesh" / "NACA0012_50_25.x"), "--shape", "NACA0012",
                   "--layers", "2,8,8,4", "--lbfgs-iter", "5", "--rounds", "3", "--threads", "2", "--out", str(out)])
    log = json.loads((out / "log.json").read_text())
    assert log["failed_rounds"] == 1
    assert log["round"] == [1, 2]
    assert len(json.loads((out / "snapshots.json").read_text())["rounds"]) == 2


def test_cli_smoke_subprocess(tmp_path, repo):
    out = tmp_path / "cli"
    r = subprocess.run([sys.executable, str(repo / "NNfoil" / "run_case.py"), "--mesh",
                        str(repo / "NNfoil" / "mesh" / "NACA0012_50_25.x"), "--shape", "NACA0012", "--layers",
                        "2,8,8,4", "--lbfgs-iter", "3", "--rounds", "1", "--threads", "2", "--out", str(out)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert all((out / f).exists() for f in FILES)


# ---------------------------------------------------------------- A12a: warm start guard
def _tiny_nn(mesh, seed=0):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    xx, yy = mesh
    return nf.NNfoil([2, 8, 8, 4], torch.device("cpu"), xx, yy, lbfgs_max_iter=2)


def _force_nan_loss(nn, monkeypatch):
    def nan_loss():
        bad = sum(p.sum() for p in nn.model.parameters()) * float("nan")
        return bad, bad, bad

    monkeypatch.setattr(nn, "Loss", nan_loss)


def test_warm_started_failed_first_round_keeps_the_transferred_weights(mesh_50x25, monkeypatch, tmp_path):
    nn = _tiny_nn(mesh_50x25)
    assert nn.warm_started is False
    other = _tiny_nn(mesh_50x25, seed=5)  # the "transferred" weights come from a different network
    nn.ckpt_path = str(tmp_path / "ckpt.pt")
    nn.set_warm_start(other.model.state_dict())
    assert nn.warm_started is True
    model_before = nn.model
    _force_nan_loss(nn, monkeypatch)
    nn.train(1)
    assert nn.model is model_before  # not re-initialised
    assert len(nn.log["losses"]) == 0  # still a failed round
    for k, v in other.model.state_dict().items():
        assert torch.equal(nn.model.state_dict()[k], v), k


def test_cold_start_failed_first_round_still_reinitialises(mesh_50x25, monkeypatch, tmp_path):
    nn = _tiny_nn(mesh_50x25)
    nn.ckpt_path = str(tmp_path / "ckpt.pt")
    model_before = nn.model
    _force_nan_loss(nn, monkeypatch)
    nn.train(1)
    assert nn.model is not model_before  # upstream behaviour unchanged


# ---------------------------------------------------------------- A12a: grid sequencing (--init-from)
def _mesh_file(repo, name):
    return repo / "NNfoil" / "mesh" / name


def _fine_nn(repo):
    xx, yy = run_case.load_mesh(_mesh_file(repo, "NACA0012_100_50.x"))
    torch.set_num_threads(1)
    return nf.NNfoil([2, 8, 8, 4], torch.device("cpu"), xx, yy, lbfgs_max_iter=5)


def _coarse_nn(tiny_run, repo):
    xx, yy = run_case.load_mesh(_mesh_file(repo, "NACA0012_50_25.x"))
    nn = nf.NNfoil([2, 8, 8, 4], torch.device("cpu"), xx, yy, lbfgs_max_iter=5)
    nn.model.load_state_dict(torch.load(tiny_run / "model.pt", weights_only=True))
    return nn


def test_make_run_id_grid_sequenced_marker_order():
    assert run_case.make_run_id("NACA0012", "100x50", 4, 0.4, grid_seq=True) == "naca0012_100x50_a4_m04_gs"
    assert (run_case.make_run_id("NACA0012", "100x50", 4, 0.4, positivity=False, seed=2, grid_seq=True)
            == "naca0012_100x50_a4_m04_gs_nopos_s2")


def test_transfer_keeps_output_at_the_same_relative_grid_point(tiny_run, repo):
    coarse = _coarse_nn(tiny_run, repo)
    fine = _fine_nn(repo)
    cfg = run_case.transfer_from_coarse(fine, tiny_run)
    assert cfg["mesh"] == "50x25"
    s = torch.tensor([100 / 50, 49 / 24])  # (N_xi_f/N_xi_c, (N_eta_f-1)/(N_eta_c-1)), M includes the closing row
    assert torch.allclose(fine.model.X_mean, coarse.model.X_mean * s, rtol=1e-6)
    assert torch.allclose(fine.model.X_std, coarse.model.X_std * s, rtol=1e-6)
    # the same relative position (xi/(M-1), eta/(N-1)): xi 25 -> 50 (middle of the airfoil), eta 0 and last
    pts = [((25.0, 0.0), (50.0, 0.0)), ((25.0, 24.0), (50.0, 49.0)), ((0.0, 24.0), (0.0, 49.0)),
           ((50.0, 0.0), (100.0, 0.0))]
    with torch.no_grad():
        for pc, pf in pts:
            uc = coarse.model(torch.tensor([pc]))
            uf = fine.model(torch.tensor([pf]))
            assert torch.allclose(uc, uf, atol=1e-5), (pc, pf, uc, uf)


def test_transfer_rejects_a_different_network(tiny_run, repo):
    xx, yy = run_case.load_mesh(_mesh_file(repo, "NACA0012_100_50.x"))
    fine = nf.NNfoil([2, 16, 4], torch.device("cpu"), xx, yy)
    with pytest.raises(SystemExit):
        run_case.transfer_from_coarse(fine, tiny_run)


def test_init_from_run_folder(tiny_run, repo, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the default --out is results/runs/<run id>
    rc = run_case.main(["--mesh", str(_mesh_file(repo, "NACA0012_100_50.x")), "--shape", "NACA0012",
                        "--layers", "2,8,8,4", "--lbfgs-iter", "5", "--rounds", "1", "--threads", "1",
                        "--init-from", str(tiny_run)])
    assert rc == 0
    out = tmp_path / "results" / "runs" / "naca0012_100x50_a4_m04_gs"
    assert out.is_dir() and "_gs" in out.name
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["init_from"] == "tiny" or cfg["init_from"] == tiny_run.name
    assert cfg["run_id"] == out.name and cfg["mesh"] == "100x50"
    log = json.loads((out / "log.json").read_text())
    assert isinstance(log["cl_after_transfer"], float) and np.isfinite(log["cl_after_transfer"])
    assert log["round"] == [1] and all(len(log[k]) == 1 for k in ["loss", "cl", "cd"])
    assert all((out / f).exists() for f in FILES)
    # the number is exactly what the coarse network gives on the fine grid: no training happened before it
    fine = _fine_nn(repo)
    run_case.transfer_from_coarse(fine, tiny_run)
    cl, _ = fine.evaluate_wall()
    assert log["cl_after_transfer"] == pytest.approx(cl, abs=1e-6)


def test_resume_of_a_grid_sequenced_run_restores_the_scaled_normalisation(tiny_run, repo, tmp_path, monkeypatch):
    out = tmp_path / "gs"
    base = ["--mesh", str(_mesh_file(repo, "NACA0012_100_50.x")), "--shape", "NACA0012", "--layers", "2,8,8,4",
            "--lbfgs-iter", "5", "--threads", "1", "--out", str(out)]
    assert run_case.main(base + ["--rounds", "1", "--init-from", str(tiny_run)]) == 0
    seen = []
    real = run_case.apply_grid_seq_norm
    monkeypatch.setattr(run_case, "apply_grid_seq_norm", lambda nn, label: (seen.append(label), real(nn, label))[1])
    assert run_case.main(base + ["--rounds", "1", "--resume"]) == 0
    assert seen == ["50x25"]
    log = json.loads((out / "log.json").read_text())
    assert log["round"] == [1, 2] and "cl_after_transfer" in log


# ---------------------------------------------------------------- reference-free diagnostics and stop rules
def test_default_run_has_no_diagnostics_or_stop_keys(tiny_run):
    """Without --diagnostics / --stop-rule the run folder is exactly what it was before these options existed."""
    log = json.loads((tiny_run / "log.json").read_text())
    cfg = json.loads((tiny_run / "config.json").read_text())
    assert "diag" not in log and "stop" not in log
    assert "stop_rule" not in cfg and "diagnostics" not in cfg


def test_cell_centre_metrics_exact_on_an_affine_mesh():
    i, j = np.meshgrid(np.arange(6), np.arange(4), indexing="ij")
    a, b, c, d = 2.0, 0.5, -0.3, 3.0
    m = run_case.cell_centre_metrics(a * i + b * j, c * i + d * j)  # x = a i + b j, y = c i + d j
    assert m["XE"].shape == (5 * 3, 2) and m["J"].shape == (15,)
    assert m["XE"][0].tolist() == [0.5, 0.5] and m["XE"][-1].tolist() == [4.5, 2.5]
    assert np.allclose(m["J"], a * d - c * b)
    jac = np.array([[a, c], [b, d]])  # [[x_xi, y_xi], [x_eta, y_eta]]
    assert np.allclose(m["jac_inv"], np.linalg.inv(jac))
    assert m["j"].tolist() == [0, 1, 2] * 5


def test_euler_residual_matches_msef_on_the_nodes(mesh_50x25):
    xx, yy = mesh_50x25
    torch.manual_seed(0)
    nn = nf.NNfoil([2, 8, 8, 4], torch.device("cpu"), xx, yy)
    nn.batch_ind = np.arange(nn.XE.shape[0])
    ref = float(nn.Msef())
    r = run_case.euler_residual_sq(nn.model, nn.XE, nn.Jac_inv)
    w = nn.J[:, 0] ** 2
    assert r.shape == (nn.XE.shape[0],)
    assert float((r * w).sum() / w.sum()) == pytest.approx(ref, rel=1e-3)


def test_uniform_flow_has_zero_residual():
    class Uniform(torch.nn.Module):
        def forward(self, x):  # freestream at Ma 0.4, 4 deg, independent of the point
            base = torch.tensor([[np.cos(np.radians(4)), np.sin(np.radians(4)), 1 / (1.4 * 0.16), 1.0]])
            return base + 0.0 * x[:, :1]

    XE = torch.rand(7, 2) * 10
    inv = torch.eye(2).repeat(7, 1, 1)
    assert float(run_case.euler_residual_sq(Uniform(), XE, inv).max()) == 0.0


def test_inner_rows_is_a_quarter_of_the_rows():
    assert run_case._inner_rows(25) == 6 and run_case._inner_rows(100) == 25 and run_case._inner_rows(2) == 1


def test_round_changes():
    assert run_case.round_changes(0.5, [1, 2], None, None) == (None, None)
    d_cl, d_cp = run_case.round_changes(0.51, [1.0, 3.0], 0.50, [1.0, 1.0])
    assert d_cl == pytest.approx(0.01) and d_cp == pytest.approx(np.sqrt(2.0))


@pytest.mark.parametrize("text, expected", [(None, ("none", None)), ("none", ("none", None)),
                                            ("plateau", ("plateau", None)), ("fixed:5", ("fixed", 5))])
def test_parse_stop_rule(text, expected):
    assert run_case.parse_stop_rule(text) == expected


@pytest.mark.parametrize("bad", ["fixed:0", "fixed:x", "fixed:", "stop", "plateau:2"])
def test_parse_stop_rule_rejects(bad):
    with pytest.raises(SystemExit):
        run_case.parse_stop_rule(bad)


def test_plateau_reached():
    f = run_case.plateau_reached
    assert f([None, 0.001, 0.0015], [None, 0.004, 0.003]) is True            # last 2 rounds both below the limits
    assert f([None, 0.001, 0.0035], [None, 0.004, 0.003]) is False          # |dCl| too large in the last round
    assert f([None, 0.001, 0.0015], [None, 0.006, 0.003]) is False          # wall-Cp change too large two rounds ago
    assert f([None, 0.001], [None, 0.001]) is False                          # the first round has no change (None)
    assert f([0.001], [0.001], window=2) is False                            # fewer rounds than the window
    assert f([0.001], [0.001], window=1) is True
    assert f([0.002], [0.001], window=1) is False                            # strictly below the limit


def test_record_diag_keeps_one_entry_per_round():
    log = {"round": [1]}
    run_case.record_diag(log, {"d_cl": None, "d_cp": None})
    log["round"].append(2)
    run_case.record_diag(log, {"d_cl": 0.1, "d_cp": 0.2, "res_off": 3.0})  # a key that first appears in round 2
    log["round"].append(3)
    run_case.record_diag(log, {"d_cl": 0.0, "d_cp": 0.0})                   # a key missing in round 3
    assert log["diag"]["d_cl"] == [None, 0.1, 0.0]
    assert log["diag"]["res_off"] == [None, 3.0, None]
    assert all(len(v) == 3 for v in log["diag"].values())


def _args(repo, out, *extra):
    return ["--mesh", str(repo / "NNfoil" / "mesh" / "NACA0012_50_25.x"), "--shape", "NACA0012",
            "--layers", "2,8,8,4", "--lbfgs-iter", "5", "--threads", "1", "--out", str(out), *extra]


def test_diagnostics_are_logged_per_round_and_fixed_rule_stops(tmp_path, repo):
    out = tmp_path / "diag"
    assert run_case.main(_args(repo, out, "--rounds", "5", "--diagnostics", "--stop-rule", "fixed:2")) == 0
    log = json.loads((out / "log.json").read_text())
    assert log["round"] == [1, 2] and log["stop"] == {"rule": "fixed:2", "round": 2}
    keys = {"d_cl", "d_cp", "res_node_w", "res_off_w", "res_node", "res_off", "res_node_inner", "res_off_inner",
            "wall_vn_rms"}
    assert keys <= set(log["diag"])
    assert all(len(v) == 2 for v in log["diag"].values())
    assert log["diag"]["d_cl"][0] is None and log["diag"]["d_cl"][1] is not None
    for k in keys - {"d_cl", "d_cp"}:
        assert all(np.isfinite(v) and v >= 0 for v in log["diag"][k]), k
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["stop_rule"] == "fixed:2" and cfg["diagnostics"] is True


def test_plateau_rule_stops_when_limits_are_loose(tmp_path, repo):
    out = tmp_path / "plateau"
    run_case.main(_args(repo, out, "--rounds", "6", "--stop-rule", "plateau", "--plateau-dcl", "1e9",
                        "--plateau-dcp", "1e9"))
    log = json.loads((out / "log.json").read_text())
    assert log["round"] == [1, 2, 3]  # round 1 has no change, so the 2-round window is first full after round 3
    assert log["stop"]["round"] == 3 and log["stop"]["rule"].startswith("plateau")
    assert "res_off" not in log["diag"]  # --diagnostics was not asked for: only the cheap changes are stored


def test_plateau_rule_does_not_stop_with_tight_limits(tmp_path, repo):
    out = tmp_path / "tight"
    run_case.main(_args(repo, out, "--rounds", "3", "--stop-rule", "plateau", "--plateau-dcl", "0",
                        "--plateau-dcp", "0"))
    log = json.loads((out / "log.json").read_text())
    assert log["round"] == [1, 2, 3] and "stop" not in log


def test_fixed_rule_with_enough_rounds_logged_does_nothing_on_resume(tiny_run, tmp_path, repo):
    run = tmp_path / "done"
    shutil.copytree(tiny_run, run)
    rc = run_case.main(_args(repo, run, "--rounds", "2", "--resume", "--stop-rule", "fixed:2"))
    assert rc == 0
    assert json.loads((run / "log.json").read_text())["round"] == [1, 2]  # no extra round was trained


def test_round_diagnostics_weighted_node_residual_is_the_pde_loss(mesh_50x25):
    """res_node_w * 2e4 is the training PDE loss evaluated on all nodes; the masks leave out the seam cells only."""
    xx, yy = mesh_50x25
    torch.manual_seed(0)
    nn = nf.NNfoil([2, 8, 8, 4], torch.device("cpu"), xx, yy)
    ctx = run_case.diagnostics_context(nn)
    d = run_case.round_diagnostics(nn, ctx)
    nn.batch_ind = np.arange(nn.XE.shape[0])
    assert d["res_node_w"] * 2e4 == pytest.approx(float(nn.Msef()) * 2e4, rel=1e-3)
    n_cells = (nn.M - 1) * (nn.N - 1)
    assert ctx["cell_XE"].shape[0] == n_cells
    assert int(ctx["cell_ok"].sum()) == (nn.M - 1 - 2 * run_case.SEAM_CELLS) * (nn.N - 1)
    assert int(ctx["cell_inner_ok"].sum()) == (nn.M - 1 - 2 * run_case.SEAM_CELLS) * ctx["n_inner"]
    assert int(ctx["node_ok"].sum()) == (nn.M - 2 * run_case.SEAM_CELLS) * nn.N  # xi = 3 .. M-1-3
    assert all(np.isfinite(v) and v >= 0 for v in d.values())
