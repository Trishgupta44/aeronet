"""report_tables.py: every table is built from the files in results/ and agrees with them."""

import results_io as rio
import report_tables as rt


def _lines(md):
    return md.splitlines()


def test_headline_table_has_every_e2_row_and_the_withheld_time():
    md = rt.headline_table()
    s = rio.read_summary()
    n_e2 = len(rio.select(s, experiment="E2"))
    # E2 rows + the B0 panel row + the thin-airfoil row; header and rule lines are two more
    assert len(_lines(md)) == 2 + n_e2 + 2
    assert "withheld" in md
    pin = rio.pick(rio.select(s, experiment="E2"), method="pinn_nopos", mesh="50x25")
    assert f"{float(pin['rms_cp']):.4f}" in md


def test_e3_table_flags_the_collapse_and_lists_both_retries():
    md = rt.e3_table()
    assert md.count("collapsed**") == 1
    assert "seed 1, positivity on" in md and "seed 0, positivity off" in md


def test_e5_table_reports_the_panel_pg_value_not_the_incompressible_one():
    md = rt.e5_table()
    s = rio.read_summary()
    pg = rio.pick(rio.select(s, experiment="E5", method="panel"), mach=0.4)
    assert f"{float(pg['rms_cp']):.4f}" in md
    assert "0.1877" in md and "Incompressible circle" in md


def test_e4_table_keeps_quoted_and_measured_apart():
    md = rt.e4_table()
    assert md.count("| quoted |") == 4
    assert "NeuralFoil" in md and "not comparable" in md


def test_e6_table_counts_match_the_replay_file():
    d = rio.read_csv(rio.RESULTS / "e6_replay.csv")
    drifted_stopped = [r for r in d if r["drifted"] == "True" and r["stop_round"] != "never"]
    assert drifted_stopped == []  # the negative result: the rule never stopped a drifted run
    md = rt.e6_table()
    assert "HELD-OUT, all" in md and "0 of 3" in md


def test_e6_fixed_table_and_e1_table_agree_with_e1_rounds():
    assert "E1 runs (development)" in rt.e6_fixed_table()
    assert len(_lines(rt.e1_table())) == 2 + 6


def test_seed_check_table_marks_the_collapsed_run():
    md = rt.seed_check_table()
    assert md.count("collapsed") == 1 and "not run" in md


def test_command_line_prints_a_named_table(capsys):
    assert rt.main(["e5"]) == 0
    assert "### e5" in capsys.readouterr().out
