"""Tests for research/stop_rule_eval.py: synthetic trajectories only (no training, tiny and fast)."""

import csv
import json

import numpy as np
import pytest

import metrics
import run_case
import runs
import stop_rule_eval as sre


# ---------------------------------------------------------------- synthetic runs
def make_ref(tmp_path):
    """A closed ellipse contour with Cp = 0.3 cos t, in the NNfoil Cp.dat format (header, then node x y cp)."""
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    x, y, cp = 0.5 * np.cos(t), 0.06 * np.sin(t), 0.3 * np.cos(t)
    p = tmp_path / "ref_Cp.dat"
    with p.open("w") as f:
        f.write("node x y cp\n")
        for i in range(t.size):
            f.write(f"{i} {x[i]} {y[i]} {cp[i]}\n")
    return p


def make_run(root, name, offsets, cls, *, times=None, ref=None, shape="NACA0012", stop_rule=None, positivity=True,
             seed=0, alpha=4.0, mach=0.4, mesh="50x25"):
    """Run folder whose wall Cp at round k is the reference Cp plus ``offsets[k]``, so the Cp RMS error equals |offsets[k]|."""
    d = root / name
    d.mkdir(parents=True)
    pts = metrics.load_fvm(ref)
    n = len(offsets)
    cfg = {"run_id": name, "mesh_file": "x", "mesh": mesh, "shape": shape, "mach": mach, "alpha_deg": alpha,
           "layers": [2, 8, 4], "rounds": n, "lbfgs_max_iter": 5, "threads": 1, "positivity": positivity, "seed": seed}
    if stop_rule:
        cfg["stop_rule"] = stop_rule
    times = times or [10.0 * (i + 1) for i in range(n)]
    log = {"round": list(range(1, n + 1)), "time_s": times, "loss": [1e-4] * n, "loss_f": [1e-5] * n,
           "loss_b": [9e-5] * n, "cl": list(cls), "cd": [0.001] * n, "failed_rounds": 0}
    snap = {"x": pts[:, 0].tolist(), "y": pts[:, 1].tolist(),
            "rounds": [{"round": k + 1, "cp": (pts[:, 2] + offsets[k]).tolist()} for k in range(n)]}
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "log.json").write_text(json.dumps(log))
    (d / "snapshots.json").write_text(json.dumps(snap))
    return d


# settles at round 5 (rounds 4 and 5 both change little), then jumps away
SETTLE_OFF = [0.20, 0.10, 0.03, 0.028, 0.0285, 0.20, 0.25, 0.30]
SETTLE_CL = [0.30, 0.45, 0.50, 0.501, 0.5015, 0.30, 0.20, 0.10]


# ---------------------------------------------------------------- the replay itself
def test_replay_stop_uses_the_run_case_rule_and_stops_at_the_first_round_that_satisfies_it():
    d_cl = [None, 0.1, 0.0005, 0.0005, 0.0005]
    d_cp = [None, 0.1, 0.001, 0.001, 0.001]
    assert sre.replay_stop(d_cl, d_cp) == 3          # rounds 3 and 4 (indices 2 and 3) are both small
    for k in range(len(d_cl)):                        # identical to calling plateau_reached on each prefix
        expect = run_case.plateau_reached(d_cl[:k + 1], d_cp[:k + 1], 0.002, 0.005, 2)
        assert (sre.replay_stop(d_cl, d_cp) <= k) == expect


def test_replay_stop_never_when_the_changes_stay_large_or_one_round_is_not_enough():
    assert sre.replay_stop([None, 0.01, 0.01, 0.01], [None, 0.02, 0.02, 0.02]) is None
    assert sre.replay_stop([None, 0.0001, 0.5, 0.0001], [None, 0.0001, 0.0001, 0.0001]) is None  # never 2 in a row
    assert sre.replay_stop([None, 0.0001], [None, 0.0001]) is None  # the first round has no change: only 1 usable
    assert sre.replay_stop([None, 0.0001, 0.0001], [None, 0.0001, 0.0001]) == 2


def test_replay_stop_respects_other_thresholds_and_window():
    d_cl = [None, 0.004, 0.004, 0.004]
    d_cp = [None, 0.008, 0.008, 0.008]
    assert sre.replay_stop(d_cl, d_cp) is None
    assert sre.replay_stop(d_cl, d_cp, dcl=0.005, dcp=0.01) == 2
    assert sre.replay_stop(d_cl, d_cp, dcl=0.005, dcp=0.01, window=3) == 3


def test_plateau_margin_is_below_one_exactly_when_the_rule_fires():
    ok = [{"d_cl": None, "d_cp": None}, {"d_cl": 0.001, "d_cp": 0.0025}, {"d_cl": 0.0005, "d_cp": 0.001}]
    bad = [{"d_cl": None, "d_cp": None}, {"d_cl": 0.004, "d_cp": 0.001}, {"d_cl": 0.001, "d_cp": 0.001}]
    assert sre.plateau_margin(ok) == pytest.approx(0.5)
    assert sre.plateau_margin(bad) == pytest.approx(2.0)


# ---------------------------------------------------------------- one run folder
def test_evaluate_run_settle_then_jump(tmp_path):
    ref = make_ref(tmp_path)
    d = make_run(tmp_path, "settle", SETTLE_OFF, SETTLE_CL, ref=ref)
    r = sre.evaluate_run(d, ref_path=ref)
    assert r["stop_round"] == 5 and r["n_rounds"] == 8
    assert r["rms_stop"] == pytest.approx(0.0285, abs=1e-6)
    assert r["rms_final"] == pytest.approx(0.30, abs=1e-6) and r["final_round"] == 8
    assert r["best_round"] == 4 and r["rms_best"] == pytest.approx(0.028, abs=1e-6)
    assert r["drifted"] and r["jump_round"] == 6 and r["stopped_before_jump"] is True
    assert r["bad_stop"] is False and r["bad_stop_kind"] == ""
    assert r["train_s_stop"] == 50.0 and r["train_s_final"] == 80.0 and r["saved_s"] == 30.0
    assert r["cl_stop"] == pytest.approx(0.5015) and r["cl_final"] == pytest.approx(0.10)
    assert r["plateau_margin"] < 1.0


def test_evaluate_run_never_triggers_and_is_judged_at_the_final_round(tmp_path):
    ref = make_ref(tmp_path)
    off = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24]
    d = make_run(tmp_path, "wander", off, [0.5 - 0.05 * i for i in range(8)], ref=ref)
    r = sre.evaluate_run(d, ref_path=ref)
    assert r["stop_round"] is None and r["rms_stop"] == r["rms_final"] == pytest.approx(0.24, abs=1e-6)
    assert r["saved_s"] is None and r["stopped_before_jump"] is None and r["bad_stop"] is False
    assert r["drifted"] and r["plateau_margin"] > 1.0


def test_evaluate_run_flags_a_too_early_stop(tmp_path):
    ref = make_ref(tmp_path)
    off = [0.30, 0.3002, 0.3004, 0.15, 0.03, 0.02, 0.02, 0.02]
    r = sre.evaluate_run(make_run(tmp_path, "early", off, [0.1, 0.1002, 0.1004, 0.3, 0.45, 0.5, 0.5, 0.5], ref=ref), ref_path=ref)
    assert r["stop_round"] == 3 and r["bad_stop"] is True and r["bad_stop_kind"] == "early"
    assert r["best_round"] == 6 and not r["drifted"]


def test_evaluate_run_flags_a_late_bad_stop(tmp_path):
    ref = make_ref(tmp_path)
    off = [0.2, 0.1, 0.03, 0.2, 0.2001, 0.2002, 0.2002, 0.2002]
    r = sre.evaluate_run(make_run(tmp_path, "late", off, [0.3, 0.4, 0.5, 0.2, 0.2002, 0.2004, 0.2004, 0.2004], ref=ref), ref_path=ref)
    assert r["stop_round"] == 6 and r["best_round"] == 3 and r["bad_stop_kind"] == "late"
    assert r["stopped_before_jump"] is False


def test_evaluate_run_skips_short_runs_online_runs_and_incomplete_folders(tmp_path):
    ref = make_ref(tmp_path)
    assert sre.evaluate_run(make_run(tmp_path, "short", SETTLE_OFF[:5], SETTLE_CL[:5], ref=ref), ref_path=ref) is None
    assert sre.evaluate_run(make_run(tmp_path, "online", SETTLE_OFF, SETTLE_CL, ref=ref, stop_rule="plateau"), ref_path=ref) is None
    assert sre.evaluate_run(make_run(tmp_path, "sixrounds", SETTLE_OFF[:6], SETTLE_CL[:6], ref=ref), ref_path=ref) is not None
    (tmp_path / "empty").mkdir()
    assert sre.evaluate_run(tmp_path / "empty", ref_path=ref) is None
    d = make_run(tmp_path, "nosnap", SETTLE_OFF, SETTLE_CL, ref=ref)
    (d / "snapshots.json").unlink()
    assert sre.evaluate_run(d, ref_path=ref) is None


def test_laptop_sleep_in_the_timing_is_flagged_and_the_train_time_is_dropped_from_the_summary_row(tmp_path):
    ref = make_ref(tmp_path)
    times = [10.0, 20.0, 30.0, 40.0, 5040.0, 5050.0, 5060.0, 5070.0]   # round 5 took 5000 s
    r = sre.evaluate_run(make_run(tmp_path, "slept", SETTLE_OFF, SETTLE_CL, ref=ref, times=times), ref_path=ref)
    assert r["timing_suspect"] is True
    row = sre.replay_summary_row(r)
    assert row["train_s"] is None and "DISCARDED" in row["notes"]
    ok = sre.evaluate_run(make_run(tmp_path, "awake", SETTLE_OFF, SETTLE_CL, ref=ref), ref_path=ref)
    assert ok["timing_suspect"] is False and sre.replay_summary_row(ok)["train_s"] == 50.0
    assert not sre.timing_suspect([50, 100, 150, 200])
    assert sre.timing_suspect([50, 100, 150, 2000])


# ---------------------------------------------------------------- groups, references, aggregate, summary rows
def test_group_of_and_reference_for():
    assert sre.group_of("naca0012_50x25_a4_m04") == "development"
    assert sre.group_of("naca0012_50x25_a4_m04_nopos_s2_r20") == "development"
    assert sre.group_of("naca0012_50x25_a4_m04_s1") == "held-out"           # the 5-round seed check is not E1
    assert sre.group_of("naca2412_100x50_a4_m03") == "held-out"
    assert sre.group_of("naca0012_50x25_a4_m04_s3_e6") == "held-out-new"
    assert sre.group_of("naca0012_50x25_a4_m04_nopos_s5_e6") == "held-out-new"
    assert sre.group_of("naca0012_50x25_a4_m04_s3_e6_online") == "held-out"   # never replayed: it has a stop rule
    assert sre.reference_for({"shape": "NACA0012", "alpha_deg": 4.0}).name == "NACA0012_800_400_Cp.dat"
    assert sre.reference_for({"shape": "Cylinder", "alpha_deg": 0.0}).name == "Cylinder_800_400_Cp.dat"
    p = sre.reference_for({"shape": "NACA2412", "alpha_deg": -4.0})
    assert p.name == "NACA2412_800_400_AoA-4_Cp.dat" and p.exists()
    assert sre.reference_for({"shape": "Banana", "alpha_deg": 0.0}) is None


def test_evaluate_all_groups_and_aggregate(tmp_path):
    ref = make_ref(tmp_path)
    root = tmp_path / "runs"
    root.mkdir()
    make_run(root, "naca0012_50x25_a4_m04", SETTLE_OFF, SETTLE_CL, ref=ref)           # development
    make_run(root, "naca0012_50x25_a4_m04_s3_e6", SETTLE_OFF, SETTLE_CL, ref=ref)    # new held-out
    # explicit evaluate_run calls: the ellipse reference is not the one reference_for would pick for NACA0012
    rows = [sre.evaluate_run(root / n, ref_path=ref) for n in ("naca0012_50x25_a4_m04", "naca0012_50x25_a4_m04_s3_e6")]
    assert [r["group"] for r in rows] == ["development", "held-out-new"]
    a = sre.aggregate(rows)
    assert a["n"] == 2 and a["n_stopped"] == 2 and a["n_never"] == 0 and a["n_drifted"] == 2
    assert a["n_drifted_stopped_before_jump"] == 2 and a["n_bad_stop"] == 0
    assert a["mean_stop_round"] == 5 and a["mean_rms_stop_stopped"] == pytest.approx(0.0285, abs=1e-6)
    assert a["mean_rms_final_stopped"] == pytest.approx(0.30, abs=1e-6)
    assert a["saved_s_total"] == 60.0 and a["n_saved_s"] == 2
    assert sre.aggregate([])["n"] == 0


def test_summary_rows_for_the_replay_and_the_new_runs(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    # a stopped run (settles) and a never-stopping run, both on the real NACA0012 reference
    make_run(root, "naca0012_50x25_a4_m04_s3_e6", SETTLE_OFF, SETTLE_CL, ref=runs.FVM["NACA0012"], seed=3)
    make_run(root, "naca0012_50x25_a4_m04_nopos_s4_e6", [0.03 * (i + 1) for i in range(8)],
             [0.5 - 0.05 * i for i in range(8)], ref=runs.FVM["NACA0012"], positivity=False, seed=4)
    make_run(root, "naca0012_50x25_a4_m04_s1_r20", SETTLE_OFF, SETTLE_CL, ref=runs.FVM["NACA0012"], seed=1)
    rows = sre.evaluate_all(root)
    assert [r["group"] for r in rows] == ["development", "held-out-new", "held-out-new"]
    assert [r["run_id"] for r in rows][1:] == ["naca0012_50x25_a4_m04_nopos_s4_e6", "naca0012_50x25_a4_m04_s3_e6"]
    dev = sre.replay_summary_row(rows[0])
    assert dev["experiment"] == "E6" and dev["method"] == "pinn_stoprule_replay" and dev["source"] == "measured"
    assert dev["case"] == "naca0012_50x25_a4_m04_s1_r20" and dev["train_s"] == 50.0
    assert "set after looking at E1" in dev["notes"] and "dCl 0.002, dCp 0.005, window 2" in dev["notes"]
    held = sre.replay_summary_row({**rows[1], "group": "held-out"})
    assert "held-out" in held["notes"] and "set after looking at E1" not in held["notes"]
    # the new-seed rows: full run (pinn / pinn_nopos) next to pinn_stoprule / pinn_nopos_stoprule
    out = sre.new_run_summary_rows(root, rows)
    assert [(o["case"], o["method"]) for o in out] == [
        ("NACA0012_a4_s4", "pinn_nopos"), ("NACA0012_a4_s4", "pinn_nopos_stoprule"),
        ("NACA0012_a4_s3", "pinn"), ("NACA0012_a4_s3", "pinn_stoprule")]
    full, stop = out[2], out[3]
    assert full["experiment"] == stop["experiment"] == "E6"
    assert full["rms_cp"] == pytest.approx(0.30, abs=1e-4) and stop["rms_cp"] == pytest.approx(0.0285, abs=1e-4)
    assert stop["train_s"] == 50.0 and full["train_s"] == 80.0
    assert "stopped at round 5 of 8" in stop["notes"]
    never = out[1]
    assert "never triggered" in never["notes"] and never["rms_cp"] == pytest.approx(0.24, abs=1e-4)
    # rows go through append_summary and replace on a repeat
    path = tmp_path / "summary.csv"
    for o in out + out:
        metrics.append_summary(o, path)
    assert len(list(csv.DictReader(path.open(encoding="utf8")))) == 4


def test_write_replay_csv_and_markdown_tables(tmp_path):
    ref = make_ref(tmp_path)
    rows = [sre.evaluate_run(make_run(tmp_path, "settle", SETTLE_OFF, SETTLE_CL, ref=ref), ref_path=ref),
            sre.evaluate_run(make_run(tmp_path, "wander", [0.03 * (i + 1) for i in range(8)], [0.5] * 8, ref=ref), ref_path=ref)]
    p = tmp_path / "e6.csv"
    sre.write_replay_csv(rows, p)
    got = list(csv.DictReader(p.open(encoding="utf8")))
    assert [g["stop_round"] for g in got] == ["5", "never"] and set(got[0]) == set(sre.COLUMNS)
    md = sre.run_table_md(rows)
    assert "good stop before the jump" in md and "never stops (drifted)" in md
    assert "| DEVELOPMENT" in sre.aggregate_table_md({"DEVELOPMENT": rows})
    s = sre.sensitivity(rows)
    assert s[0]["dcl"] == 0.002 and s[0]["held_n"] == 2 and s[0]["held_stopped"] == 1 and s[0]["dev_n"] == 0
    assert "| 0.002 | 0.005 | 2 |" in sre.sensitivity_md(rows)


def test_online_summary_row_compares_with_the_replay_prediction(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    ref = runs.FVM["NACA0012"]
    make_run(root, "naca0012_50x25_a4_m04_s3_e6", SETTLE_OFF, SETTLE_CL, ref=ref, seed=3)
    d = make_run(root, "naca0012_50x25_a4_m04_s3_e6_online", SETTLE_OFF[:5], SETTLE_CL[:5], ref=ref, seed=3, stop_rule="plateau")
    log = json.loads((d / "log.json").read_text())
    log["stop"] = {"rule": "plateau (dCl<0.002, dCp<0.005, 2 rounds)", "round": 5}
    (d / "log.json").write_text(json.dumps(log))
    rows = sre.evaluate_all(root)
    assert [r["run_id"] for r in rows] == ["naca0012_50x25_a4_m04_s3_e6"]  # the online run has a stop rule: never replayed
    row = sre.online_summary_row(root, "naca0012_50x25_a4_m04_s3_e6_online", rows)
    assert row["method"] == "pinn_stoprule_online" and row["case"] == "NACA0012_a4_s3" and row["experiment"] == "E6"
    assert "stopped by the rule at round 5" in row["notes"] and "predicted round 5" in row["notes"]
    assert row["rms_cp"] == pytest.approx(0.0285, abs=1e-4)
