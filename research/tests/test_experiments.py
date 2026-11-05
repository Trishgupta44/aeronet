import csv
import json
import math

import numpy as np
import pytest

import experiments as ex
import metrics
import runs


# ---------------------------------------------------------------- fake run folders (formats written by run_case.py)
def fake_run(root, name, *, spec_like=None, mesh="50x25", positivity=True, seed=0, rounds=4, failed=0, attempts_asked=None,
             init_from=None, ref=None, offset=0.01, shape="NACA0012", alpha=4.0, mach=0.4, time_per_round=10.0,
             model=True, train_log=""):
    d = root / name
    d.mkdir(parents=True)
    cfg = {"run_id": name, "mesh_file": "x", "mesh": mesh, "shape": shape, "mach": mach, "alpha_deg": alpha,
           "layers": [2, 8, 4], "rounds": attempts_asked or rounds + failed, "lbfgs_max_iter": 5, "threads": 1,
           "positivity": positivity, "seed": seed, "torch_version": "x", "started_utc": "2026-11-05T00:00:00Z"}
    if init_from:
        cfg.update({"init_from": init_from, "init_from_mesh": "50x25"})
    log = {"round": list(range(1, rounds + 1)), "time_s": [time_per_round * (i + 1) for i in range(rounds)],
           "loss": [1e-4] * rounds, "loss_f": [1e-5] * rounds, "loss_b": [9e-5] * rounds,
           "cl": [0.4 + 0.01 * i for i in range(rounds)], "cd": [0.001] * rounds, "failed_rounds": failed}
    if ref is not None:
        snap = {"x": ref[:, 0].tolist(), "y": ref[:, 1].tolist(),
                "rounds": [{"round": r, "cp": (ref[:, 2] + offset * (1 + (r == 2))).tolist()} for r in log["round"]]}
    else:
        snap = {"x": [0.0, 1.0], "y": [0.0, 0.0], "rounds": [{"round": r, "cp": [0.0, 0.0]} for r in log["round"]]}
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "log.json").write_text(json.dumps(log))
    (d / "snapshots.json").write_text(json.dumps(snap))
    if model:
        (d / "model.pt").write_bytes(b"x")
    if train_log:
        (d / "train.log").write_text(train_log)
    return d


@pytest.fixture(scope="module")
def ref0012():
    return metrics.load_fvm(runs.FVM["NACA0012"])


# ---------------------------------------------------------------- shared helpers
def test_shell_line_makes_the_folder_redirects_the_log_and_uses_repo_relative_paths():
    cmd = runs.train_cmd("NNfoil/mesh/NACA0012_50_25.x", "NACA0012", 0.4, 4, 20, "results/runs/a", seed=1)
    line = ex.shell_line(cmd, "results/runs/a")
    assert line.startswith("mkdir -p results/runs/a && ")
    assert line.endswith("> results/runs/a/train.log 2>&1")
    assert "NNfoil/run_case.py" in line and "\\" not in line and "C:" not in line
    assert ">> results/runs/a/train.log" in ex.shell_line(cmd, "results/runs/a", append=True)


def test_rounds_for_budget_is_the_3_hour_rule():
    assert ex.rounds_for_budget(600.0) == 18
    assert ex.rounds_for_budget(100.0) == 20  # capped at 20
    assert ex.rounds_for_budget(5000.0) == 2
    assert ex.rounds_for_budget(0.0) == 20


def test_trajectory_stats_labels_the_best_round_as_post_hoc():
    rows = [{"round": i + 1, "cp_rms": v, "cl": 0.5} for i, v in enumerate([0.2, 0.04, 0.03, 0.05, 0.06, 0.07, 0.08])]
    s = ex.trajectory_notes(rows)
    assert "rms_round3=0.03" in s and "best_round_rms=0.03@round3" in s and "post-hoc" in s
    assert "mean_rms_last5=" in s
    assert "n/a" in ex.trajectory_notes(rows[:2])  # fewer than 3 rounds: round 3 is not available


# ---------------------------------------------------------------- E1
def test_e1_specs_and_run_dirs(tmp_path):
    specs = ex.e1_specs(tmp_path)
    assert len(specs) == 6
    names = {s["dir"].name for s in specs}
    assert {"naca0012_50x25_a4_m04", "naca0012_50x25_a4_m04_nopos"} <= names  # seed 0 reuses the Plan A runs
    assert {"naca0012_50x25_a4_m04_s1_r20", "naca0012_50x25_a4_m04_nopos_s2_r20"} <= names  # new folders, 20 rounds
    assert all(s["rounds"] == 20 and s["mesh"] == "50x25" for s in specs)
    assert {(s["positivity"], s["seed"]) for s in specs} == {(p, s) for p in (True, False) for s in (0, 1, 2)}
    # the short Plan A seed-check runs (5 rounds, used as grid-sequencing parents) must never be reused or modified
    assert not any(s["dir"].name.endswith(("_s1", "_s2")) for s in specs)


def test_commands_e1_lines(tmp_path):
    fake_run(tmp_path, "naca0012_50x25_a4_m04", rounds=19, failed=1, attempts_asked=20)
    fake_run(tmp_path, "naca0012_50x25_a4_m04_nopos", positivity=False, rounds=18, failed=2, attempts_asked=20)
    fake_run(tmp_path, "naca0012_50x25_a4_m04_s1_r20", seed=1, rounds=5, failed=0, attempts_asked=5)
    lines, notes = ex.plan_commands("e1", tmp_path)
    assert len(lines) == 4  # 1 resume + 3 new; the two seed-0 runs are complete and skipped
    resume = [ln for ln in lines if "--resume" in ln]
    assert len(resume) == 1 and "--rounds 15" in resume[0] and "_s1_r20" in resume[0] and ">>" in resume[0]
    new = [ln for ln in lines if "--resume" not in ln]
    assert len(new) == 3 and all("--rounds 20" in ln and "--lbfgs-iter" not in ln for ln in new)
    assert sum("--no-positivity" in ln for ln in new) == 2 and all("NACA0012_50_25.x" in ln for ln in new)
    assert any("--seed 2" in ln for ln in new) and not any("--seed 0" in ln for ln in lines)
    assert all("--threads 8" in ln and "--init-from" not in ln for ln in lines)


def test_collect_e1_rows_and_trajectory_csv(tmp_path, ref0012):
    root = tmp_path / "runs"
    root.mkdir()
    for s in ex.e1_specs(root):
        fake_run(root, s["dir"].name, positivity=s["positivity"], seed=s["seed"], rounds=5, failed=1, attempts_asked=6,
                 ref=ref0012, offset=0.01 if s["positivity"] else 0.02, train_log="Failed, Restart\n")
    summary = tmp_path / "summary.csv"
    out = ex.collect("e1", runs_root=root, summary=summary, rounds_dir=tmp_path)
    assert len(out) == 6
    rows = list(csv.DictReader(summary.open()))
    assert len(rows) == 6 and {r["experiment"] for r in rows} == {"E1"}
    assert {r["method"] for r in rows} == {"pinn", "pinn_nopos"}
    assert {r["case"] for r in rows} == {"NACA0012_a4_s0", "NACA0012_a4_s1", "NACA0012_a4_s2"}
    assert len({(r["case"], r["method"]) for r in rows}) == 6
    pos = next(r for r in rows if r["method"] == "pinn" and r["case"].endswith("s1"))
    assert float(pos["rms_cp"]) == pytest.approx(0.01, abs=1e-4)  # final round: offset 0.01
    assert float(pos["cl_ref"]) == pytest.approx(0.5193, abs=5e-4) and pos["source"] == "measured"
    for key in ("rounds=5/6", "seed=1", "restarts=1", "nan_rounds=0", "best_round_rms=", "post-hoc", "rms_round3="):
        assert key in pos["notes"], key
    traj = list(csv.DictReader((tmp_path / "e1_rounds.csv").open()))
    assert len(traj) == 6 * 5  # every round of every run is kept, not only the best one
    assert {"run_id", "method", "case", "round", "cp_rms", "cl"} <= set(traj[0])


# ---------------------------------------------------------------- E2
def test_e2_fvm_rows():
    rows = ex.fvm_mesh_rows()
    assert len(rows) == 4 and [r["mesh"] for r in rows] == ["50x25", "100x50", "200x100", "400x200"]
    assert 0.10 <= rows[0]["rms_cp"] <= 0.12 and rows[3]["rms_cp"] < 0.01
    assert all(r["experiment"] == "E2" and r["method"] == "fvm" and r["case"] == "NACA0012_a4" for r in rows)
    assert [r["rms_cp"] for r in rows] == sorted((r["rms_cp"] for r in rows), reverse=True)  # error falls with the mesh
    assert rows[0]["cl_ref"] == pytest.approx(0.5193, abs=5e-4) and rows[0]["source"] == "measured"


def test_e2_panel_row():
    r = ex.panel_row()
    assert r["method"] == "panel" and r["mesh"].endswith("panels") and r["experiment"] == "E2"
    assert r["rms_cp"] < 0.02 and r["cl"] == pytest.approx(0.525, abs=0.01)  # about 0.0099 in the baseline rows


def test_commands_e2_probe_then_resume_by_the_budget(tmp_path):
    parent = tmp_path / "naca0012_100x50_a4_m04_gs_nopos"
    parent.mkdir()
    (parent / "config.json").write_text("{}")
    (parent / "model.pt").write_bytes(b"x")
    lines, notes = ex.plan_commands("e2", tmp_path)  # no 200x100 run yet: a 1-round probe, chained from the 100x50 run
    assert len(lines) == 1
    ln = lines[0]
    assert "NACA0012_200_100.x" in ln and "--rounds 1 " in ln and "--no-positivity" in ln
    assert "--init-from results/runs/naca0012_100x50_a4_m04_gs_nopos" in ln or str(parent).replace("\\", "/") in ln
    assert "naca0012_200x100_a4_m04_gs_nopos" in ln and "--resume" not in ln

    probe = fake_run(tmp_path, "naca0012_200x100_a4_m04_gs_nopos", mesh="200x100", positivity=False, rounds=1,
                     time_per_round=600.0, init_from="x")
    lines, notes = ex.plan_commands("e2", tmp_path)  # 600 s per round -> min(20, 18) rounds in total, 17 left
    assert len(lines) == 1 and "--resume" in lines[0] and "--rounds 17" in lines[0]

    (probe / "log.json").write_text(json.dumps({**json.loads((probe / "log.json").read_text()), "time_s": [5000.0]}))
    lines, notes = ex.plan_commands("e2", tmp_path)  # 5000 s per round -> only 2 fit in 3 h: skipped, with the reason
    assert lines == [] and any("fewer than 3" in n for n in notes)


def test_collect_e2_adds_fvm_panel_and_pinn_rows(tmp_path, ref0012):
    root = tmp_path / "runs"
    root.mkdir()
    fake_run(root, "naca0012_50x25_a4_m04", rounds=5, failed=0, ref=ref0012, offset=0.01)
    fake_run(root, "naca0012_100x50_a4_m04_gs_nopos", mesh="100x50", positivity=False, rounds=4, init_from="a",
             ref=ref0012, offset=0.2)
    summary = tmp_path / "summary.csv"
    ex.collect("e2", runs_root=root, summary=summary, rounds_dir=tmp_path)
    rows = list(csv.DictReader(summary.open()))
    assert len(rows) >= 7 and {r["experiment"] for r in rows} == {"E2"}  # 4 fvm + panel + 2 pinn
    assert [r["method"] for r in rows].count("fvm") == 4 and [r["method"] for r in rows].count("panel") == 1
    p50 = next(r for r in rows if r["method"] == "pinn" and r["mesh"] == "50x25")
    assert "PINN RMS < FVM RMS at 50x25: yes" in p50["notes"]
    p100 = next(r for r in rows if r["method"] == "pinn_gs_nopos")
    assert "PINN RMS < FVM RMS at 100x50: no" in p100["notes"]  # offset 0.2 is worse than the FVM's 0.046


# ---------------------------------------------------------------- E3
def test_e3_mach_is_the_paper_value():
    assert ex.E3_MACH == 0.3 and "2401.08705" in ex.E3_MACH_NOTE


def test_e3_mesh_is_built_with_both_coefficients(tmp_path):
    paths = ex.ensure_e3_meshes(tmp_path)
    assert set(paths) == {"upstream", "winslow2"}
    for p in paths.values():
        assert p.read_text().splitlines()[0] == "100\t50"
        xx, yy = metrics.load_mesh(p)
        assert xx.shape == (101, 50)
        assert xx[0, 0] == pytest.approx(0.5) and abs(yy[0, 0]) < 1e-9  # trailing edge on the right
    a, b = (metrics.load_mesh(p)[0] for p in (paths["upstream"], paths["winslow2"]))
    assert 1e-4 < np.abs(a - b).max() < 0.05  # the coefficients differ a little


def test_e3_polar_rows():
    fvm = ex.e3_fvm_rows()
    assert len(fvm) == 11 and [r["alpha_deg"] for r in fvm] == list(range(-10, 11, 2))
    assert next(r for r in fvm if r["alpha_deg"] == 4)["cl"] == pytest.approx(0.7605, abs=5e-4)
    assert all(r["method"] == "fvm" and r["mesh"] == "800x400" and r["mach"] == 0.3 for r in fvm)
    panel = ex.e3_panel_rows()
    assert len(panel) == 11 and all(r["method"] == "panel" and r["rms_cp"] < 0.06 for r in panel)
    thin = ex.e3_thin_rows()
    assert len(thin) == 11 and all(math.isfinite(r["cl"]) and r["method"] == "thin_airfoil" for r in thin)


def test_commands_e3_four_angles_on_the_100x50_mesh(tmp_path):
    lines, notes = ex.plan_commands("e3", tmp_path, mesh_dir=tmp_path / "meshes")
    assert len(lines) == 4
    for ln, a in zip(lines, ["-4", "0", "4", "8"]):
        assert f"--alpha {a} " in ln and "--mach 0.3 " in ln and "--rounds 10 " in ln
        assert "NACA2412_100_50.x" in ln and "--shape NACA2412" in ln and "--init-from" not in ln
    assert "naca2412_100x50_am4_m03" in lines[0] and "naca2412_100x50_a8_m03" in lines[3]
    assert (tmp_path / "meshes" / "NACA2412_100_50.x").exists()  # the mesh is made on demand


def test_collect_e3_rows(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    refp = {a: metrics.load_fvm(runs.fvm_2412(a)) for a in (-4, 0, 4, 8)}
    for a in (-4, 0, 4, 8):
        fake_run(root, runs.run_id("NACA2412", "100x50", a, 0.3), mesh="100x50", shape="NACA2412", alpha=float(a),
                 mach=0.3, rounds=3, ref=refp[a], offset=0.02)
    summary = tmp_path / "summary.csv"
    ex.collect("e3", runs_root=root, summary=summary, rounds_dir=tmp_path)
    rows = list(csv.DictReader(summary.open()))
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r)
    assert len(by["fvm"]) == 11 and len(by["pinn"]) == 4 and len(by["panel"]) == 11 and len(by["thin_airfoil"]) == 11
    assert {r["case"] for r in by["pinn"]} == {"NACA2412_a-4", "NACA2412_a0", "NACA2412_a4", "NACA2412_a8"}
    p4 = next(r for r in by["pinn"] if r["case"] == "NACA2412_a4")
    assert "0.5" in p4["notes"] and "quoted" in p4["notes"] and "2401.08705" in p4["notes"]
    assert float(p4["cl_ref"]) == pytest.approx(0.7605, abs=5e-4)
