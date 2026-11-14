"""make_figures.py runs on the real committed results and writes every Milestone 3 figure (no skips)."""

import pytest

import make_figures
import results_io as rio

EXPECTED = ["m3_e1_trajectories.png", "m3_e2_mesh_study.png", "m3_e3_polar.png", "m3_e4_speed.png",
            "m3_e5_cylinder.png", "m3_e6_stop_rule.png", "m3_seed_check.png"]


def test_all_seven_figures_are_written_from_the_real_results(tmp_path):
    out = tmp_path / "figures"
    assert make_figures.main(["--out", str(out)]) == 0
    assert sorted(p.name for p in out.glob("m3_*.png")) == sorted(EXPECTED)
    for name in EXPECTED:
        assert (out / name).stat().st_size > 30000, name


def test_a_missing_input_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(rio, "RESULTS", tmp_path)  # an empty results folder
    with pytest.raises(FileNotFoundError, match="missing input"):
        make_figures.make_figures(tmp_path / "out", summary=tmp_path / "summary.csv", only=["e1"])


def test_every_figure_name_is_registered():
    assert set(make_figures.FIGS) == set(make_figures.NAMES) == set(make_figures.FILES)


def test_round_series_matches_the_e1_trajectory_file(repo):
    """Cp RMS recomputed from a run folder equals the number the E1 collector stored (one definition of the metric)."""
    rows = [r for r in rio.read_csv(rio.RESULTS / "e1_rounds.csv") if r["run_id"] == "naca0012_50x25_a4_m04_s1_r20"]
    ser = rio.round_series(rio.RESULTS / "runs" / "naca0012_50x25_a4_m04_s1_r20", rio.REF_0012)
    assert ser["round"] == [int(r["round"]) for r in rows]
    for a, r in zip(ser["cp_rms"], rows):
        assert a == pytest.approx(float(r["cp_rms"]), abs=1e-5)
