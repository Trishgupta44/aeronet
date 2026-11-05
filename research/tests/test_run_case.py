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
