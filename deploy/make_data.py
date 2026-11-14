"""Build the data folder of a PUBLIC deployment: a copy of app/backend/data with every NNfoil finite-volume (FVM) Cp curve taken out.

    python deploy/make_data.py                       # app/backend/data  ->  deploy/data/public
    python deploy/make_data.py --src X --out Y       # other folders
    python deploy/make_data.py --check <folder>      # only verify a folder: exit 1 if it holds FVM data

NNfoil has no licence, so a public image or static bundle must not contain its FVM data. The committed data does contain it (the
exported PINN cases carry the FVM 800x400 ``reference`` points and the ``fvm_same_mesh`` points, and the bundled ``*_Cp.dat`` sample is
there too), so the public build is made from this generated copy, never from the committed folder:

  * every ``pinn_cases/<id>.json`` goes through ``aeronet.pinn_cases.strip_fvm`` (the same function the server applies in public mode): the
    reference and same-mesh points are removed; the numbers computed against them (metrics, same-mesh Cp RMS/max/Cl, the Cp RMS of every
    training round) and this project's own curves (PINN wall Cp, panel method, flow field, log, snapshots) stay;
  * ``results_summary.json``, ``seed_runs.json``, ``rounds_e1.json`` are copied as they are (numbers and the PINN's own curves);
  * the ``*.dat`` files (the bundled FVM sample) are NOT copied.

The output is verified before this script finishes. ``deploy/data/`` is git-ignored: it is generated, not committed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO / "app" / "backend" / "data"
DEFAULT_OUT = REPO / "deploy" / "data" / "public"

COPIED = ("results_summary.json", "seed_runs.json", "rounds_e1.json")
KEYS_WITH_POINTS = ("reference", "fvm_same_mesh")


def _has_points(node) -> bool:
    return isinstance(node, dict) and bool(node.get("points"))


def find_fvm_data(folder: Path) -> list[str]:
    """What in ``folder`` is FVM data (empty list = clean). Looks for: any ``*.dat`` or ``*_Cp*`` file, a case with a ``reference`` or
    ``fvm_same_mesh`` that has points, and any non-empty ``points`` list anywhere in a case (the format of an FVM curve)."""
    problems: list[str] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and (p.suffix.lower() == ".dat" or "_cp" in p.name.lower()):
            problems.append(f"{p.relative_to(folder)}: a Cp data file")
    for p in sorted(folder.rglob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf8"))
        except (OSError, ValueError):
            problems.append(f"{p.relative_to(folder)}: not readable JSON")
            continue
        stack = [("", data)]
        while stack:
            path, node = stack.pop()
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in KEYS_WITH_POINTS and _has_points(v):
                        problems.append(f"{p.relative_to(folder)}: {path}{k} has FVM points")
                    elif k == "points" and v:
                        problems.append(f"{p.relative_to(folder)}: {path}points is a non-empty curve")
                    elif isinstance(v, (dict, list)):
                        stack.append((f"{path}{k}.", v))
            elif isinstance(node, list):
                stack.extend((path, v) for v in node if isinstance(v, (dict, list)))
    return problems


def build(src: Path, out: Path) -> dict:
    sys.path.insert(0, str(REPO / "app" / "backend"))
    from aeronet.pinn_cases import strip_fvm  # imported late: --check needs no aeronet

    if not (src / "pinn_cases").is_dir():
        raise SystemExit(f"{src / 'pinn_cases'} not found: pass --src <app/backend/data>")
    if out.exists():
        shutil.rmtree(out)
    (out / "pinn_cases").mkdir(parents=True)
    files = []
    for name in COPIED:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
            files.append(name)
    stripped = 0
    for case_file in sorted((src / "pinn_cases").glob("*.json")):
        case = json.loads(case_file.read_text(encoding="utf8"))
        had = bool(case.get("reference") or case.get("fvm_same_mesh"))
        (out / "pinn_cases" / case_file.name).write_text(json.dumps(strip_fvm(case), separators=(",", ":")), encoding="utf8")
        stripped += had
        files.append(f"pinn_cases/{case_file.name}")
    problems = find_fvm_data(out)
    if problems:
        shutil.rmtree(out)
        raise SystemExit("the stripped copy still holds FVM data (removed again):\n  " + "\n  ".join(problems))
    manifest = {
        "what": "app/backend/data without any NNfoil FVM Cp curve (see deploy/make_data.py)",
        "cases": len(files) - sum(1 for f in files if f in COPIED),
        "cases_stripped": stripped,
        "files": files,
        "bytes": sum((out / f).stat().st_size for f in files),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="the committed data folder (default: app/backend/data)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="the stripped copy to write (default: deploy/data/public)")
    ap.add_argument("--check", metavar="FOLDER", help="only verify FOLDER: exit 1 if it contains FVM data")
    a = ap.parse_args(argv)
    if a.check:
        problems = find_fvm_data(Path(a.check))
        if problems:
            print("FVM data found:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 1
        print(f"ok: {a.check} holds no FVM data")
        return 0
    m = build(Path(a.src), Path(a.out))
    print(f"{m['cases']} cases ({m['cases_stripped']} stripped), {m['bytes'] / 1e6:.1f} MB -> {a.out}; verified: no FVM data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
