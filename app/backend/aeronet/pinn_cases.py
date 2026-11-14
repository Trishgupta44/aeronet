"""The ONE loader for trained-PINN results: exported case files, the results table and the seed-spread runs.

Everything is read from ``app/backend/data`` (written by ``research/export_results.py``), so the app works from a clean
clone: no ``NNfoil/``, no ``results/`` and no torch are needed at run time.

* ``data/pinn_cases/<case_id>.json``  one trained case each (roadmap 5.7: surface Cp, reference, panel solution, metrics,
  timing, log, per-round snapshots, flow field),
* ``data/results_summary.json``       the rows of ``results/summary.csv``,
* ``data/seed_runs.json``             the seed-to-seed spread groups behind the same-mesh showdown.

The module attributes ``CASES_DIR``, ``RESULTS_FILE`` and ``SEED_RUNS_FILE`` can be monkeypatched (the tests do).
Small files are parsed once and cached by modification time; a case is cached WITHOUT its flow field (the field is
about 1 MB of numbers per case and only the Lab needs it: the endpoint streams the file itself). Nothing here is computed:
a number is whatever the export wrote.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CASES_DIR = DATA_DIR / "pinn_cases"
RESULTS_FILE = DATA_DIR / "results_summary.json"
SEED_RUNS_FILE = DATA_DIR / "seed_runs.json"

_ID = re.compile(r"[a-z0-9_]+")  # a case id is a run id: lowercase letters, digits and underscores only
_cache: dict[Path, tuple[tuple[int, int], object]] = {}


def _read_json(path: Path):
    """Parse a small data file, cached by (mtime, size); a changed file is read again."""
    st = path.stat()
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _cache.get(path)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    with path.open(encoding="utf8") as f:
        data = json.load(f)
    _cache[path] = (stamp, data)
    return data


def case_ids() -> list[str]:
    """Ids of the case files present (sorted). Files whose name is not a valid id are ignored."""
    if not CASES_DIR.is_dir():
        return []
    return sorted(p.stem for p in CASES_DIR.glob("*.json") if _ID.fullmatch(p.stem))


def has_cases() -> bool:
    return bool(case_ids())


def case_path(case_id: str) -> Path:
    """The file of a case. ``KeyError`` for an invalid or unknown id (the id is checked against the list of files, so
    no input can reach any other path)."""
    if not isinstance(case_id, str) or not _ID.fullmatch(case_id) or case_id not in case_ids():
        raise KeyError(f"Unknown PINN case '{case_id}'.")
    return CASES_DIR / f"{case_id}.json"


def get_case(case_id: str) -> dict:
    """The full case JSON as a dict, flow field included (read from the file each time). ``KeyError`` for an unknown id."""
    with case_path(case_id).open(encoding="utf8") as f:
        return json.load(f)


def get_case_meta(case_id: str) -> dict:
    """The case without its ``field`` (cached by modification time): everything the lists and the showdown need."""
    path = case_path(case_id)
    st = path.stat()
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _cache.get(path)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    with path.open(encoding="utf8") as f:
        case = json.load(f)
    case.pop("field", None)
    _cache[path] = (stamp, case)
    return case


def _summary(case: dict) -> dict:
    m = case.get("metrics") or {}
    return {
        "id": case["id"],
        "label": case.get("label", case["id"]),
        "shape": case.get("shape"),
        "mesh": case.get("mesh"),
        "mach": case.get("mach"),
        "alpha": case.get("alpha"),
        "rms": m.get("rms"),
        "cl": m.get("cl"),
        "collapsed": bool(m.get("collapsed", False)),
        "source": case.get("source"),
    }


def list_cases() -> list[dict]:
    """One short summary per case, sorted by label: ``id, label, shape, mesh, mach, alpha, rms, cl, collapsed, source``."""
    out = [_summary(get_case_meta(i)) for i in case_ids()]
    return sorted(out, key=lambda d: (d["label"], d["id"]))


def load_results() -> list[dict]:
    """The rows of ``results_summary.json`` (numeric columns as numbers, empty cells ``None``); ``[]`` if the file is missing."""
    if not RESULTS_FILE.is_file():
        return []
    return _read_json(RESULTS_FILE)


def load_seed_runs() -> dict:
    """``{"<mesh>:<method>": {"source", "runs": [...]}}`` for the showdown's seed spread; ``{}`` if the file is missing."""
    if not SEED_RUNS_FILE.is_file():
        return {}
    return _read_json(SEED_RUNS_FILE).get("groups", {})
