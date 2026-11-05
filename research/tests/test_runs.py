import json
import sys

import numpy as np
import pytest

import runs


@pytest.mark.parametrize(
    "kw, expected",
    [
        (dict(shape="NACA0012", mesh="100x50", alpha=4, mach=0.4), "naca0012_100x50_a4_m04"),
        (dict(shape="NACA2412", mesh="100x50", alpha=-4, mach=0.25, positivity=False), "naca2412_100x50_am4_m025_nopos"),
        (dict(shape="NACA2412", mesh="100x50", alpha=8, mach=0.3), "naca2412_100x50_a8_m03"),
        (dict(shape="NACA0012", mesh="50x25", alpha=2.5, mach=0.4, seed=1), "naca0012_50x25_a2p5_m04_s1"),
        (dict(shape="NACA0012", mesh="50x25", alpha=4, mach=0.4, positivity=False, seed=2), "naca0012_50x25_a4_m04_nopos_s2"),
        (dict(shape="NACA0012", mesh="50x25", alpha=4, mach=0.4, seed=0), "naca0012_50x25_a4_m04"),
    ],
)
def test_run_id(kw, expected):
    assert runs.run_id(**kw) == expected


@pytest.mark.parametrize("kw", [
    dict(shape="NACA0012", mesh="200x100", alpha=4, mach=0.4, grid_seq=True),
    dict(shape="NACA0012", mesh="100x50", alpha=4, mach=0.4, positivity=False, seed=3, grid_seq=True),
    dict(shape="NACA2412", mesh="100x50", alpha=-4, mach=0.25, positivity=False),
])
def test_run_id_matches_the_run_case_naming(kw):
    import run_case  # needs torch; this is the one place the two namers are compared

    assert runs.run_id(**kw) == run_case.make_run_id(**kw)


def test_wall_faces_midpoints_for_nodes_and_passthrough_for_faces():
    x = np.linspace(0, 1, 11)
    y = x**2
    snap = {"x": x.tolist(), "y": y.tolist(), "rounds": [{"round": 1, "cp": [0.0] * 10}]}
    fx, fy = runs.wall_faces(snap)
    assert len(fx) == len(fy) == 10
    assert np.allclose(fx, 0.5 * (x[:-1] + x[1:])) and np.allclose(fy, 0.5 * (y[:-1] + y[1:]))
    snap2 = {"x": x[:10].tolist(), "y": y[:10].tolist(), "rounds": [{"round": 1, "cp": [0.0] * 10}]}
    px, py = runs.wall_faces(snap2)
    assert np.array_equal(px, x[:10]) and np.array_equal(py, y[:10])


def test_count_restarts_counts_failed_lines(tmp_path):
    (tmp_path / "train.log").write_text(
        "warning line\nFailed, Restart\nround failed (total failed: 1)\n0|1 loss=1\nFailed, Load previous model\nround 2\n")
    assert runs.count_restarts(tmp_path) == 2
    assert runs.count_restarts(tmp_path / "nowhere") == 0  # no train.log: nothing to count


def test_train_cmd_positivity_flag_and_options():
    on = runs.train_cmd("m.x", "NACA0012", 0.4, 4, 20, "results/runs/a")
    off = runs.train_cmd("m.x", "NACA0012", 0.4, 4, 20, "results/runs/a", positivity=False)
    assert "--no-positivity" not in on and "--no-positivity" in off
    assert on[0] == sys.executable and on[1].replace("\\", "/").endswith("NNfoil/run_case.py")
    for flag, val in [("--mesh", "m.x"), ("--shape", "NACA0012"), ("--mach", "0.4"), ("--alpha", "4"),
                      ("--rounds", "20"), ("--threads", "8"), ("--out", "results/runs/a"), ("--seed", "0")]:
        assert on[on.index(flag) + 1] == val
    assert "--layers" not in on and "--lbfgs-iter" not in on
    c = runs.train_cmd("m.x", "X", 0.4, 4, 5, "o", seed=2, threads=1, layers=[2, 8, 4], lbfgs_iter=5,
                       init_from="results/runs/c", resume=True)
    assert c[c.index("--layers") + 1] == "2,8,4" and c[c.index("--lbfgs-iter") + 1] == "5"
    assert c[c.index("--seed") + 1] == "2" and c[c.index("--threads") + 1] == "1"
    assert c[c.index("--init-from") + 1] == "results/runs/c" and "--resume" in c


def test_fvm_paths_exist():
    assert runs.FVM["NACA0012"].exists() and runs.FVM["Cylinder"].exists()
    assert runs.fvm_2412(4).name == "NACA2412_800_400_AoA4_Cp.dat" and runs.fvm_2412(-10).exists()


def _fake_run(d, n_cp=4, rounds=(1, 2), failed=1, restarts=2):
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"mesh": "50x25", "shape": "NACA0012"}))
    (d / "log.json").write_text(json.dumps({"round": list(rounds), "time_s": [1.0, 2.0], "loss": [1, 1], "loss_f": [1, 1],
                                           "loss_b": [1, 1], "cl": [0.4, 0.5], "cd": [0, 0], "failed_rounds": failed}))
    x = np.linspace(0, 1, n_cp + 1)
    (d / "snapshots.json").write_text(json.dumps({"x": x.tolist(), "y": [0.0] * (n_cp + 1),
                                                  "rounds": [{"round": r, "cp": [float(r)] * n_cp} for r in rounds]}))
    (d / "train.log").write_text("Failed, Restart\n" * restarts)
    return d


def test_load_run_attempts_and_final_surface(tmp_path):
    d = _fake_run(tmp_path / "r")
    run = runs.load_run(d)
    assert set(run) == {"dir", "config", "log", "snapshots", "field"} and run["field"] is None
    assert runs.attempts(run) == 3  # 2 good rounds + 1 failed one
    x, y, cp = runs.final_surface(run)
    assert len(x) == len(y) == len(cp) == 4 and np.all(cp == 2.0)  # last snapshot, node midpoints
    np.savez(d / "field.npz", cp=np.zeros((2, 2)))
    assert runs.load_run(d)["field"]["cp"].shape == (2, 2)


def test_build_model_rebuilds_the_trained_network(tiny_run):
    run = runs.load_run(tiny_run)
    nn = runs.build_model(tiny_run, threads=1)
    assert not nn.model.training
    cl, cd = nn.evaluate_wall()
    assert cl == pytest.approx(run["log"]["cl"][-1], abs=1e-4)  # the saved weights reproduce the logged final Cl
    import torch

    with torch.no_grad():
        out = nn.model(nn.XE)
    assert tuple(out.shape) == (51 * 25, 4)
