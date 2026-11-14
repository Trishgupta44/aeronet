"""export_results.py (C6): run folder -> case JSON (roadmap 5.7). No torch: the run folder is synthetic, built from the bundled
100x50 mesh and analytic fields."""

import csv
import json

import numpy as np
import pytest
from matplotlib.path import Path as MplPath

import export_results as ex
import metrics
import runs


def _cp(x, y):
    return np.sin(3 * x) * np.exp(-np.hypot(x - 0.5, y))


def _uv(x, y):
    return np.cos(2 * x) * np.exp(-0.5 * np.hypot(x - 0.5, y)), 0.3 * np.sin(2 * y)


@pytest.fixture(scope="module")
def synth(repo, tmp_path_factory):
    """A finished-looking run folder: 3 rounds on the bundled NACA 0012 100x50 mesh, analytic field, and an FVM-style file."""
    root = tmp_path_factory.mktemp("synth")
    d = root / "syn_100x50_a4_m04"
    d.mkdir()
    mesh_file = repo / "NNfoil" / "mesh" / "NACA0012_100_50.x"
    xx, yy = metrics.load_mesh(mesh_file)
    X, Y = xx + 0.5, yy  # app frame, closed (51, 25)
    wx, wy = X[:-1, 0], Y[:-1, 0]
    fx, fy = 0.5 * (wx + np.roll(wx, -1)), 0.5 * (wy + np.roll(wy, -1))  # face centres
    cfg = {"run_id": d.name, "mesh_file": str(mesh_file), "mesh": "100x50", "shape": "NACA0012", "mach": 0.4, "alpha_deg": 4.0,
           "layers": [2, 8, 8, 4], "rounds": 3, "lbfgs_max_iter": 5, "threads": 8, "positivity": True, "seed": 0}
    n = 3
    log = {"round": [1, 2, 3], "time_s": [10.0, 20.0, 30.0], "loss": [3e-3, 2e-3, 1e-3], "loss_f": [2e-3, 1e-3, 5e-4],
           "loss_b": [1e-3, 1e-3, 5e-4], "cl": [0.1, 0.3, 0.45], "cd": [0.01, 0.005, 0.001], "failed_rounds": 1}
    snaps = {"x": fx.tolist(), "y": fy.tolist(),
             "rounds": [{"round": k + 1, "cp": (_cp(fx, fy) * (0.5 + 0.25 * k)).tolist()} for k in range(n)]}
    for name, obj in (("config.json", cfg), ("log.json", log), ("snapshots.json", snaps)):
        (d / name).write_text(json.dumps(obj))
    u, v = _uv(X, Y)
    np.savez(d / "field.npz", X=X, Y=Y, u=u, v=v, p=np.ones_like(X), rho=np.ones_like(X), cp=_cp(X, Y))
    # an FVM-style reference file (NNfoil frame: x - 0.5), Cp = the analytic final-round Cp at those points
    ang = np.linspace(0.05, 2 * np.pi - 0.05, 80)
    rx, ry = 0.5 + 0.5 * np.cos(ang), 0.06 * np.sin(ang)
    ref = root / "ref_Cp.dat"
    ref.write_text("nodenumber x-coordinate y-coordinate pressure-coefficient\n" + "\n".join(
        f"{i + 1} {rx[i] - 0.5:.6e} {ry[i]:.6e} {_cp(rx[i], ry[i]):.6e}" for i in range(len(ang))) + "\n")
    return {"dir": d, "ref": ref, "wall": (wx, wy), "mesh_file": mesh_file, "X": X, "Y": Y}


def test_case_has_every_section_5_7_key_and_the_reference_may_be_null(synth):
    case = ex.build_case(synth["dir"], ref_path=synth["ref"], width=120)
    for k in ("id", "label", "shape", "mesh", "mach", "alpha", "x", "y", "surface", "reference", "panel", "metrics", "timing",
              "log", "snapshots", "field", "source"):
        assert k in case
    assert case["id"] == "syn_100x50_a4_m04" and case["shape"] == "NACA0012" and case["mesh"] == "100x50"
    assert set(case["surface"]) == {"x", "y", "cp"} and len(case["x"]) == len(case["y"]) == 100
    assert len(case["surface"]["cp"]) == 100 and len(case["snapshots"]) == 3
    assert set(case["metrics"]) >= {"rms", "max", "n", "cl", "cd", "cl_ref", "collapsed"}
    assert set(case["timing"]) >= {"train_s", "infer_ms", "threads", "cpu"}
    assert set(case["log"]) >= {"round", "time_s", "loss", "loss_f", "loss_b", "cl"}
    assert case["reference"]["name"].startswith("FVM") and len(case["reference"]["points"]) == 80
    assert case["panel"]["cl"] > 0 and len(case["panel"]["surface"]["cp"]) == 140
    json.dumps(case, allow_nan=False)  # valid JSON: no NaN, no Infinity

    none = ex.build_case(synth["dir"], ref_path=None, width=120)
    assert none["reference"] is None and none["metrics"]["rms"] is None and none["metrics"]["cl_ref"] is None
    assert none["metrics"]["collapsed"] is False


def test_surface_is_the_final_round_and_metrics_use_the_nearest_point_rule(synth):
    case = ex.build_case(synth["dir"], ref_path=synth["ref"], width=120)
    final = _cp(*[np.asarray(a) for a in (case["surface"]["x"], case["surface"]["y"])]) * 1.0  # round 3 factor is 1.0
    assert np.allclose(case["surface"]["cp"], final, atol=1e-4)
    assert [s["round"] for s in case["snapshots"]] == [1, 2, 3]
    ref = metrics.load_fvm(synth["ref"])
    want = metrics.cp_error(case["surface"]["x"], case["surface"]["y"], case["surface"]["cp"], ref)
    assert case["metrics"]["rms"] == pytest.approx(want["rms"], abs=2e-5) and case["metrics"]["n"] == want["n"]
    assert case["metrics"]["cl"] == 0.45 and case["metrics"]["cd"] == 0.001  # final entries of the log
    assert case["metrics"]["cl_ref"] == pytest.approx(metrics.fvm_forces(synth["ref"], 4.0)[0], abs=1e-5)


def test_field_has_w_times_h_cells_and_null_exactly_inside_the_body(synth):
    case = ex.build_case(synth["dir"], width=170)
    f = case["field"]
    assert (f["width"], f["height"]) == (170, round(170 / 1.8))
    assert (f["xmin"], f["xmax"], f["ymin"], f["ymax"]) == (-0.4, 1.4, -0.5, 0.5)
    for k in ("cp", "u", "v"):
        assert len(f[k]) == f["width"] * f["height"]
    gx, gy, _ = ex.grid_points(170)
    wx, wy = synth["wall"]
    inside = MplPath(np.c_[wx, wy]).contains_points(np.c_[gx.ravel(), gy.ravel()])
    null = np.array([v is None for v in f["cp"]])
    assert null.sum() > 100  # there is a body
    assert abs(int(null.sum()) - int(inside.sum())) <= 2 and int((null != inside).sum()) <= 2  # only cells on the wall may differ
    assert (np.array([v is None for v in f["u"]]) == null).all() and (np.array([v is None for v in f["v"]]) == null).all()
    outside = np.array([v for v, n in zip(f["cp"], null) if not n], dtype=float)
    assert np.isfinite(outside).all()


def test_field_row_order_is_row_major_top_row_first(synth):
    f = ex.resample_field(synth["X"], synth["Y"], {"cp": _cp(synth["X"], synth["Y"])}, 90)
    W, H = f["width"], f["height"]
    gx, gy, _ = ex.grid_points(90)
    cp = np.array([np.nan if v is None else v for v in f["cp"]]).reshape(H, W)
    ok = ~np.isnan(cp)
    assert np.abs(cp[ok] - _cp(gx, gy)[ok]).max() < 0.01
    assert gy[0, 0] > gy[-1, 0] and gx[0, 0] < gx[0, -1]  # first row is the top (largest y), columns run left to right


def test_interpolation_error_is_small_for_a_smooth_field(synth):
    X, Y = synth["X"], synth["Y"]
    f = ex.resample_field(X, Y, {"cp": _cp(X, Y)}, 340)
    gx, gy, _ = ex.grid_points(340)
    got = np.array([np.nan if v is None else v for v in f["cp"]])
    ok = ~np.isnan(got)
    assert np.abs(got[ok] - _cp(gx.ravel(), gy.ravel())[ok]).max() < 0.01


def test_file_is_under_3_mb_and_the_size_guard_steps_down(synth, tmp_path):
    p = ex.export_case(synth["dir"], None, synth["ref"], True, 340, tmp_path / "out")
    assert p.name == "syn_100x50_a4_m04.json" and p.stat().st_size < 3_000_000
    full = json.loads(p.read_text(encoding="utf8"))
    assert full["field"]["width"] == 340 and len(full["snapshots"]) == 3

    small = ex.export_case(synth["dir"], None, synth["ref"], True, 340, tmp_path / "small", max_bytes=int(p.stat().st_size * 0.5))
    c = json.loads(small.read_text(encoding="utf8"))
    assert small.stat().st_size <= p.stat().st_size * 0.5 and c["field"]["width"] < 340

    with pytest.raises(ValueError):
        ex.export_case(synth["dir"], None, synth["ref"], True, 340, tmp_path / "none", max_bytes=2000)


def test_snapshots_are_thinned_to_first_middle_last_when_widths_are_not_enough(synth):
    case = ex.build_case(synth["dir"], width=80, keep_snapshots=2)
    assert [s["round"] for s in case["snapshots"]] == [1, 3]


def test_withheld_training_time_is_null_with_a_note_and_the_log_time_is_nulled_too(synth):
    row = {"train_s": None, "notes": "x; train_s withheld: the laptop slept during this run; y"}
    case = ex.build_case(synth["dir"], width=80, row=row)
    assert case["timing"]["train_s"] is None and "laptop slept" in case["timing"]["note"]
    assert case["log"]["time_s"] == [None, None, None]
    ok = ex.build_case(synth["dir"], width=80, row={"train_s": 321.5, "notes": ""})
    assert ok["timing"]["train_s"] == 321.5 and ok["timing"]["note"] is None and ok["log"]["time_s"] == [10.0, 20.0, 30.0]
    no_row = ex.build_case(synth["dir"], width=80)
    assert no_row["timing"]["train_s"] == 30.0  # no summary row: the log's cumulative time


def test_collapse_detection():
    assert ex.detect_collapse(15.6, 0.0, 0.76)[0] is True            # the real NACA 2412 4-degree seed-0 run
    assert ex.detect_collapse(0.02, 1e-8, 0.76)[0] is True            # Cl about 0 where the reference has lift
    assert ex.detect_collapse(0.14, -0.004, 2.8e-5)[0] is False       # a cylinder's Cl is 0: not a collapse
    assert ex.detect_collapse(0.07, 0.41, 0.52)[0] is False           # drifted but alive
    assert ex.detect_collapse(None, 0.3, None)[0] is False


def test_labels():
    assert ex.make_label("NACA0012", "100x50", 4, 0.4) == "NACA 0012 · 100×50 · 4° · Ma 0.4"
    assert ex.make_label("Cylinder", "50x25", 0.0, 0.4, "positivity on").startswith("Cylinder · 50×25 · 0°")
    assert ex.make_label("NACA2412", "100x50", -4, 0.3, collapsed=True).endswith("collapsed")


def test_summary_json_converts_numbers_and_empty_cells(tmp_path):
    src = tmp_path / "summary.csv"
    with src.open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(metrics.SUMMARY_COLUMNS)
        w.writerow(["E2", "NACA0012_a4", "pinn", "50x25", "0.4", "4", "0.0734", "0.2", "0.41", "0.52", "0.02", "334.2", "",
                    "measured", "rounds=19; note, with comma"])
        w.writerow(["E2", "NACA0012_a4", "fvm", "800x400", "0.4", "4", "", "", "0.5", "0.5", "", "", "", "quoted", ""])
    out = ex.write_summary_json(src, tmp_path / "out" / "results_summary.json")
    rows = json.loads(out.read_text(encoding="utf8"))
    assert len(rows) == 2 and list(rows[0]) == metrics.SUMMARY_COLUMNS
    assert rows[0]["rms_cp"] == 0.0734 and rows[0]["train_s"] == 334.2 and rows[0]["infer_ms"] is None
    assert rows[0]["mesh"] == "50x25" and rows[0]["notes"] == "rounds=19; note, with comma"
    assert rows[1]["rms_cp"] is None and rows[1]["notes"] is None and rows[1]["source"] == "quoted"


def test_seed_groups_keep_curves_only_for_runs_with_a_result(synth, tmp_path):
    rows = ex.read_summary_rows(tmp_path / "missing.csv")
    assert rows == []
    sc = tmp_path / "seed_check.csv"
    sc.write_text("mode,method,seed,run_id,cl_after_transfer,final_cl,cp_rms,rounds,failed_rounds,seconds,collapsed,complete\n"
                  "pos,cold,0,syn_100x50_a4_m04,,0.5,0.0145,20,0,577.6,no,yes\n"
                  "pos,cold,1,not_a_run,,,,0,5,,yes,yes\n")
    root = synth["dir"].parent
    g = ex.seed_groups(rows, runs_root=root, seed_check=sc)["100x50:pinn"]
    assert g["source"] == "results/seed_check.csv" and [r["seed"] for r in g["runs"]] == [0, 1]
    assert g["runs"][0]["curve"] is not None and len(g["runs"][0]["curve"]["x"]) == len(g["runs"][0]["curve"]["cp"]) == 100
    assert g["runs"][1]["collapsed"] is True and g["runs"][1]["curve"] is None and g["runs"][1]["rms"] is None


def test_cases_are_unique_and_cover_the_showdown_runs():
    ids = [c["run"] for c in ex.CASES]
    assert len(ids) == len(set(ids))
    from aeronet import pinn_data

    assert set(pinn_data.E2_RUNS.values()) <= set(ids)  # the showdown card needs a case for each of its runs


def test_cylinder_panel_is_the_non_lifting_prandtl_glauert_circle(repo):
    p = ex.panel_solution("Cylinder", 0.0, 0.4, lifting=False)
    ref = metrics.load_fvm(runs.FVM["Cylinder"])
    e = metrics.cp_error(p["surface"]["x"], p["surface"]["y"], p["surface"]["cp"], ref)
    assert abs(p["cl"]) < 1e-6 and e["rms"] == pytest.approx(0.1877, abs=1e-3)  # results/summary.csv, experiment E5
    assert ex.panel_solution("Blob", 0.0, 0.4) is None


def test_real_collapsed_run_is_exported_and_flagged(repo):
    d = repo / "results" / "runs" / "naca2412_100x50_a4_m03"
    if not (d / "snapshots.json").exists():
        pytest.skip("no results/ here")
    case = ex.build_case(d, ref_path=runs.fvm_2412(4), with_field=False)
    m = case["metrics"]
    assert m["collapsed"] is True and abs(m["cl"]) < 1e-3 and m["cl_ref"] > 0.7 and m["rms"] > 10
    assert case["config"]["failed_rounds"] == 7 and "collapsed" in case["label"]


def test_real_export_matches_the_summary_row(repo):
    d = repo / "results" / "runs" / "naca0012_50x25_a4_m04"
    rows = ex.read_summary_rows()
    spec = next(c for c in ex.CASES if c["run"] == d.name)
    row = ex.find_row(rows, spec["row"])
    if not (d / "snapshots.json").exists() or row is None:
        pytest.skip("no results/ here")
    case = ex.build_case(d, ref_path=runs.FVM["NACA0012"], row=row, with_field=False)
    assert case["metrics"]["rms"] == pytest.approx(row["rms_cp"], abs=1e-6)
    assert case["metrics"]["cl"] == pytest.approx(row["cl"], abs=1e-5)
    assert case["timing"]["train_s"] == row["train_s"]
