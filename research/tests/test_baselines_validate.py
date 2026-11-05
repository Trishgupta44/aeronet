import csv

import pytest

import baselines
import validate


def _rows(path):
    with open(path, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def baseline_rows(tmp_path_factory):
    out = tmp_path_factory.mktemp("b0") / "summary.csv"
    assert baselines.main(["--out", str(out)]) == 0
    return _rows(out)


def test_baselines_row_count_and_columns(baseline_rows):
    from metrics import SUMMARY_COLUMNS

    assert len(baseline_rows) >= 14
    assert list(baseline_rows[0].keys()) == SUMMARY_COLUMNS
    assert {r["experiment"] for r in baseline_rows} == {"B0"}
    assert {r["method"] for r in baseline_rows} == {"fvm", "panel", "thin_airfoil"}
    assert {r["source"] for r in baseline_rows} <= {"measured", "fitted"}
    keys = [(r["case"], r["method"], r["mesh"]) for r in baseline_rows]
    assert len(keys) == len(set(keys))


def test_fvm_coarse_rows_match_roadmap(baseline_rows):
    fvm = {r["mesh"]: r for r in baseline_rows if r["method"] == "fvm" and r["case"] == "NACA0012_a4"}
    assert float(fvm["50x25"]["rms_cp"]) == pytest.approx(0.108, abs=0.002)
    assert float(fvm["100x50"]["rms_cp"]) == pytest.approx(0.046, abs=0.002)
    assert float(fvm["200x100"]["rms_cp"]) == pytest.approx(0.019, abs=0.002)
    assert float(fvm["400x200"]["rms_cp"]) == pytest.approx(0.006, abs=0.002)
    assert float(fvm["50x25"]["cl"]) == pytest.approx(0.4265, abs=0.002)
    assert float(fvm["50x25"]["cl_ref"]) == pytest.approx(0.5193, abs=0.002)
    assert fvm["50x25"]["mach"] == "0.4" and fvm["50x25"]["alpha_deg"] == "4"


def test_panel_and_thin_rows(baseline_rows):
    p0012 = [r for r in baseline_rows if r["method"] == "panel" and r["case"] == "NACA0012_a4"]
    assert len(p0012) == 1
    assert float(p0012[0]["rms_cp"]) == pytest.approx(0.0099, abs=0.0015)
    assert float(p0012[0]["cl"]) == pytest.approx(0.526, abs=0.003)
    p2412 = [r for r in baseline_rows if r["method"] == "panel" and r["case"].startswith("NACA2412")]
    assert len(p2412) == 11
    assert all(r["source"] == "fitted" and r["mach"] == "0.25" and "Mach fitted" in r["notes"] for r in p2412)
    assert all(r["cl_ref"] != "" and r["rms_cp"] != "" for r in p2412)
    thin = {r["case"]: r for r in baseline_rows if r["method"] == "thin_airfoil"}
    assert set(thin) == {"NACA0012_a4", "NACA2412_a4"}
    assert float(thin["NACA0012_a4"]["cl"]) == pytest.approx(0.479, abs=0.002)
    assert all(r["rms_cp"] == "" for r in thin.values())  # empty cells stay empty, never 0


def test_validate_writes_figure_and_row(tiny_run, repo, tmp_path):
    summary = tmp_path / "summary.csv"
    ref = repo / "NNfoil" / "FVM_Cp" / "NACA0012_800_400_Cp.dat"
    rc = validate.main([str(tiny_run), "--ref", str(ref), "--experiment", "E2", "--summary", str(summary)])
    assert rc == 0
    png = tiny_run / "cp_vs_fvm.png"
    assert png.exists() and png.stat().st_size > 5000
    (row,) = _rows(summary)
    assert row["experiment"] == "E2" and row["case"] == "NACA0012_a4" and row["method"] == "pinn"
    assert row["mesh"] == "50x25" and row["source"] == "measured" and row["mach"] == "0.4"
    assert float(row["rms_cp"]) > 0 and float(row["max_cp"]) >= float(row["rms_cp"])
    assert float(row["cl_ref"]) == pytest.approx(0.5193, abs=0.002)
    assert float(row["train_s"]) > 0 and row["infer_ms"] == ""
    assert float(row["cl"]) == pytest.approx(
        __import__("json").loads((tiny_run / "log.json").read_text())["cl"][-1], abs=1e-5)


def test_validate_case_name_and_method_for_nopos_and_seed():
    cfg = {"shape": "NACA0012", "alpha_deg": 4.0, "seed": 1, "positivity": False}
    assert validate.case_name(cfg) == "NACA0012_a4_s1"
    assert validate.method_name(cfg) == "pinn_nopos"
    assert validate.case_name({"shape": "NACA2412", "alpha_deg": -2.5, "seed": 0, "positivity": True}) == "NACA2412_a-2.5"
    assert validate.method_name({"positivity": True}) == "pinn"


def test_validate_method_name_combinations():
    assert validate.method_name({"positivity": True}) == "pinn"
    assert validate.method_name({"positivity": False}) == "pinn_nopos"
    assert validate.method_name({"positivity": True, "init_from": "naca0012_50x25_a4_m04"}) == "pinn_gs"
    assert validate.method_name({"positivity": False, "init_from": "x"}) == "pinn_gs_nopos"
    assert validate.method_name({"positivity": True, "init_from": None}) == "pinn"
    assert validate.case_name({"shape": "NACA0012", "alpha_deg": 4.0, "seed": 0}) == "NACA0012_a4"


def test_validate_writes_rounds_csv(tiny_run, repo, tmp_path):
    import json

    ref = repo / "NNfoil" / "FVM_Cp" / "NACA0012_800_400_Cp.dat"
    validate.main([str(tiny_run), "--ref", str(ref), "--experiment", "E2", "--summary", str(tmp_path / "s.csv")])
    rows = _rows(tiny_run / "rounds.csv")
    log = json.loads((tiny_run / "log.json").read_text())
    assert list(rows[0].keys()) == ["round", "cp_rms", "cp_max", "cl", "cd", "loss", "loss_f", "loss_b", "time_s"]
    assert [r["round"] for r in rows] == ["1", "2"] == [str(k) for k in log["round"]]
    assert float(rows[1]["cl"]) == pytest.approx(log["cl"][1], rel=1e-5)
    assert float(rows[1]["time_s"]) == pytest.approx(log["time_s"][1], rel=1e-5)
    assert all(float(r["cp_rms"]) > 0 and float(r["cp_max"]) >= float(r["cp_rms"]) for r in rows)
    # the last round's row equals the summary row's metrics (same reference, same metric)
    (srow,) = _rows(tmp_path / "s.csv")
    assert float(rows[-1]["cp_rms"]) == pytest.approx(float(srow["rms_cp"]), abs=1e-5)
    assert float(rows[-1]["cp_max"]) == pytest.approx(float(srow["max_cp"]), abs=1e-5)


def test_validate_notes_convention_plain_and_grid_sequenced(tiny_run, repo, tmp_path):
    import json
    import shutil

    ref = repo / "NNfoil" / "FVM_Cp" / "NACA0012_800_400_Cp.dat"
    summary = tmp_path / "s.csv"
    validate.main([str(tiny_run), "--ref", str(ref), "--experiment", "E2", "--summary", str(summary)])
    (plain,) = _rows(summary)
    assert plain["notes"] == "final round; rounds=2; failed_rounds=0"
    assert plain["method"] == "pinn"

    gs = tmp_path / "gs"
    shutil.copytree(tiny_run, gs)
    cfg = json.loads((gs / "config.json").read_text())
    cfg.update({"init_from": "coarse_run", "positivity": False})
    (gs / "config.json").write_text(json.dumps(cfg))
    log = json.loads((gs / "log.json").read_text())
    log.update({"cl_after_transfer": 0.48315, "failed_rounds": 1})
    (gs / "log.json").write_text(json.dumps(log))
    validate.main([str(gs), "--ref", str(ref), "--experiment", "E2b", "--summary", str(summary),
                   "--ref-label", "FVM 50x25 (same mesh)"])
    gsrow = [r for r in _rows(summary) if r["experiment"] == "E2b"][0]
    assert gsrow["method"] == "pinn_gs_nopos"
    assert gsrow["notes"] == ("ref: FVM 50x25 (same mesh); final round; rounds=2; failed_rounds=1; "
                              "init_from=coarse_run; cl_after_transfer=0.4832")
