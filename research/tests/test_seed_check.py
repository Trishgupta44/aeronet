import csv
import json

import pytest

import metrics
import seed_check


def _make_run(root, name, *, mesh="100x50", positivity=True, seed=0, init_from=None, cl=0.5, offset=0.01,
              rounds=4, failed=0, cl_after_transfer=None, requested=None, ref=None):
    """A run folder in the formats run_case.py writes; Cp = reference Cp + offset, so the RMS error is |offset|."""
    d = root / name
    d.mkdir(parents=True)
    cfg = {"run_id": name, "mesh_file": "x", "mesh": mesh, "shape": "NACA0012", "mach": 0.4, "alpha_deg": 4.0,
           "layers": [2, 8, 8, 4], "rounds": requested or rounds + failed, "lbfgs_max_iter": 5, "threads": 1,
           "positivity": positivity, "seed": seed, "torch_version": "x", "started_utc": "2026-11-05T00:00:00Z"}
    if init_from:
        cfg.update({"init_from": init_from, "init_from_mesh": "50x25"})
    log = {"round": list(range(1, rounds + 1)), "time_s": [10.0 * (i + 1) for i in range(rounds)],
           "loss": [1e-4] * rounds, "loss_f": [1e-5] * rounds, "loss_b": [9e-5] * rounds,
           "cl": [cl - 0.01 * (rounds - 1 - i) for i in range(rounds)], "cd": [0.001] * rounds,
           "failed_rounds": failed}
    if cl_after_transfer is not None:
        log["cl_after_transfer"] = cl_after_transfer
    snap = {"x": ref[:, 0].tolist(), "y": ref[:, 1].tolist(),
            "rounds": [{"round": r, "cp": (ref[:, 2] + offset).tolist()} for r in log["round"]]}
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "log.json").write_text(json.dumps(log))
    (d / "snapshots.json").write_text(json.dumps(snap))
    return d


@pytest.fixture(scope="module")
def ref_file(repo):
    return repo / "NNfoil" / "FVM_Cp" / "NACA0012_800_400_Cp.dat"


@pytest.fixture()
def runs(tmp_path, ref_file):
    ref = metrics.load_fvm(ref_file)
    root = tmp_path / "runs"
    root.mkdir()
    for pos in (False, True):
        t = "" if pos else "_nopos"
        for s in (1, 2, 3):
            # the 50x25 coarse stage of each chain must be ignored
            _make_run(root, f"naca0012_50x25_a4_m04{t}_s{s}", mesh="50x25", positivity=pos, seed=s, rounds=5, ref=ref)
            gs_cl = 0.45 if (pos and s == 3) else 0.5 + 0.005 * s
            _make_run(root, f"naca0012_100x50_a4_m04_gs{t}_s{s}", positivity=pos, seed=s, init_from="coarse",
                      cl=gs_cl, cl_after_transfer=0.48 + 0.001 * s, ref=ref)
            collapsed = (not pos) and s == 2
            _make_run(root, f"naca0012_100x50_a4_m04{t}_s{s}", positivity=pos, seed=s,
                      cl=-0.002 if collapsed else 0.51, rounds=5, failed=1 if collapsed else 0,
                      offset=0.4 if collapsed else 0.015, ref=ref)
    # the seed-0 run (no suffix) is reported but is not part of the verdicts
    _make_run(root, "naca0012_100x50_a4_m04", positivity=True, seed=0, cl=0.516, rounds=20, ref=ref)
    return root


def test_collect_rows_and_collapsed_flag(runs, ref_file):
    rows = seed_check.collect(runs, ref_file)
    assert len(rows) == 12 + 1  # 2 modes x 2 methods x 3 seeds, plus seed 0; no 50x25 rows
    by = {(r["mode"], r["method"], r["seed"]): r for r in rows}
    collapsed = by[("nopos", "cold", 2)]
    assert collapsed["collapsed"] == "yes" and collapsed["failed_rounds"] == 1 and collapsed["rounds"] == 5
    assert collapsed["cp_rms"] == pytest.approx(0.4, abs=1e-6)
    ok = by[("nopos", "cold", 1)]
    assert ok["collapsed"] == "no" and ok["final_cl"] == pytest.approx(0.51)
    assert ok["cp_rms"] == pytest.approx(0.015, abs=1e-6) and ok["seconds"] == 50.0 and ok["complete"] == "yes"
    gs = by[("nopos", "gs", 2)]
    assert gs["cl_after_transfer"] == pytest.approx(0.482) and gs["rounds"] == 4 and gs["seconds"] == 40.0
    assert by[("nopos", "cold", 1)]["cl_after_transfer"] is None
    seed0 = by[("pos", "cold", 0)]
    assert seed0["rounds"] == 20 and seed0["final_cl"] == pytest.approx(0.516)


def test_verdicts_follow_the_pass_criteria(runs, ref_file):
    lines = seed_check.verdicts(seed_check.collect(runs, ref_file))
    assert lines == [
        "[nopos] grid sequencing gives Cl > 0.49 for all 3 seeds: yes",
        "[nopos] cold start collapses in at least 1 of 3 seeds: yes (1 of 3)",
        "[pos] grid sequencing gives Cl > 0.49 for all 3 seeds: no",
        "[pos] cold start collapses in at least 1 of 3 seeds: no (0 of 3)",
    ]


def test_running_run_is_pending_not_counted(tmp_path, ref_file):
    ref = metrics.load_fvm(ref_file)
    root = tmp_path / "runs"
    root.mkdir()
    _make_run(root, "naca0012_100x50_a4_m04_gs_nopos_s1", positivity=False, seed=1, init_from="c", cl=0.5, ref=ref)
    # still training: 2 of 5 requested rounds logged
    _make_run(root, "naca0012_100x50_a4_m04_nopos_s1", positivity=False, seed=1, cl=0.01, rounds=2, requested=5,
              ref=ref)
    (root / "naca0012_100x50_a4_m04_nopos_s2").mkdir()  # started, no files yet
    rows = seed_check.collect(root, ref_file)
    assert [(r["method"], r["complete"]) for r in rows] == [("gs", "yes"), ("cold", "no")]
    lines = seed_check.verdicts(rows)
    assert lines[0].endswith("pending (1 of 3 seeds complete so far)")
    assert lines[1].endswith("pending (0 of 3 seeds complete so far)")


def test_main_writes_csv_and_prints_verdicts(runs, ref_file, tmp_path, capsys):
    out = tmp_path / "seed_check.csv"
    assert seed_check.main(["--runs", str(runs), "--ref", str(ref_file), "--out", str(out)]) == 0
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == seed_check.CSV_COLUMNS
    assert len(rows) == 13
    assert rows[0]["mode"] == "nopos" and rows[0]["method"] == "gs" and rows[0]["seed"] == "1"
    cold_pos0 = [r for r in rows if (r["mode"], r["method"], r["seed"]) == ("pos", "cold", "0")][0]
    assert cold_pos0["cl_after_transfer"] == ""  # empty cells stay empty
    text = capsys.readouterr().out
    assert "[nopos] cold start collapses in at least 1 of 3 seeds: yes (1 of 3)" in text
