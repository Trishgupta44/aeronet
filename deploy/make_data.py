"""Build the small results folder a public deployment needs (no FVM data, no model weights, no training logs).

    python deploy/make_data.py --results ../Aero-Net/results --out deploy/data/results

Copies results/summary.csv, results/seed_check.csv and, for every run the app shows (the PINN showdown, the seed bands), only
config.json, log.json and snapshots.json. It never copies model.pt, field.npz, NNfoil files or FVM Cp files. The backend then runs
with AERONET_RESULTS_DIR pointing at the output folder and AERONET_WITHHOLD_FVM_DATA=1.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app" / "backend"))

from aeronet import pinn_data  # noqa: E402

RUN_FILES = ("config.json", "log.json", "snapshots.json")


def needed_runs(seed_check: Path) -> list[str]:
    runs = set(pinn_data.E2_RUNS.values())
    for by_seed in pinn_data.E1_RUNS.values():
        runs.update(by_seed.values())
    if seed_check.exists():
        with seed_check.open(newline="", encoding="utf8") as f:
            for r in csv.DictReader(f):
                runs.add(r["run_id"])
    return sorted(runs)


def build(results: Path, out: Path) -> dict:
    if not (results / "summary.csv").exists():
        raise SystemExit(f"{results / 'summary.csv'} not found: pass --results <folder with summary.csv and runs/>")
    if out.exists():
        shutil.rmtree(out)
    (out / "runs").mkdir(parents=True)
    files, missing = [], []
    for name in ("summary.csv", "seed_check.csv"):
        if (results / name).exists():
            shutil.copy2(results / name, out / name)
            files.append(name)
    for run in needed_runs(results / "seed_check.csv"):
        src = results / "runs" / run
        got = [f for f in RUN_FILES if (src / f).exists()]
        if not got:
            missing.append(run)
            continue
        (out / "runs" / run).mkdir()
        for f in got:
            shutil.copy2(src / f, out / "runs" / run / f)
            files.append(f"runs/{run}/{f}")
    manifest = {"files": files, "runs_without_files": missing, "bytes": sum((out / f).stat().st_size for f in files)}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True, help="source results folder (with summary.csv and runs/)")
    ap.add_argument("--out", default=str(REPO / "deploy" / "data" / "results"))
    a = ap.parse_args(argv)
    m = build(Path(a.results), Path(a.out))
    print(f"{len(m['files'])} files, {m['bytes'] / 1e6:.2f} MB -> {a.out}")
    if m["runs_without_files"]:
        print(f"runs with no files (not shown by the app): {', '.join(m['runs_without_files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
