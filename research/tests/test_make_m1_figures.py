import make_m1_figures


def test_m1_figures_are_written_from_the_real_runs(repo, tmp_path):
    out = tmp_path / "figures"
    rc = make_m1_figures.main(["--runs", str(repo / "results" / "runs"), "--out", str(out),
                               "--summary", str(repo / "results" / "summary.csv")])
    assert rc == 0
    for name in ("m1_cp_50x25.png", "m1_cp_100x50.png", "m1_trajectory.png"):
        p = out / name
        assert p.exists() and p.stat().st_size > 20000, name
