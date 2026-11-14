"""The ONE loader for trained-PINN results: exported case files, the results table and the seed-spread runs.

Everything is read from ``app/backend/data`` (written by ``research/export_results.py``), so the app works from a clean
clone: no ``NNfoil/``, no ``results/`` and no torch are needed at run time.

* ``data/pinn_cases/<case_id>.json``  one trained case each (roadmap 5.7: surface Cp, reference, panel solution, metrics,
  timing, log, per-round snapshots, flow field),
* ``data/results_summary.json``       the rows of ``results/summary.csv``,
* ``data/seed_runs.json``             the seed-to-seed spread groups behind the same-mesh showdown,
* ``data/rounds_e1.json``             the round-by-round Cp RMS of the six E1 runs (the training-trajectory chart of the Results page).

The module attributes ``CASES_DIR``, ``RESULTS_FILE``, ``SEED_RUNS_FILE`` and ``ROUNDS_E1_FILE`` can be monkeypatched (the tests do).
Small files are parsed once and cached by modification time; a case is cached WITHOUT its flow field (the field is
about 1 MB of numbers per case and only the Lab needs it: the endpoint streams the file itself). Nothing here is computed:
a number is whatever the export wrote.

PUBLIC MODE (``AERONET_WITHHOLD_FVM_DATA=1``, see ``config.py``): NNfoil has no licence, so its finite-volume (FVM) Cp points must not be
served. :func:`strip_fvm` removes them from a case: the ``reference`` points (FVM 800x400) and the ``fvm_same_mesh`` points (FVM on the run's own
mesh). What stays are numbers computed against them (the metrics, the same-mesh Cp RMS/max/Cl, and ``rms_by_round``, the Cp RMS of every
replay round) and this project's own curves (PINN wall Cp, panel method, field, log). Every function here that hands a case to a caller
applies it in public mode; ``deploy/make_data.py`` applies the same function to write the files of a public image, so that image holds no FVM
curve at all (stripping at response time alone would leave the data in the image).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import config

DATA_DIR = config.data_dir(Path(__file__).resolve().parent.parent / "data")  # AERONET_DATA_DIR (read at start) or app/backend/data
CASES_DIR = DATA_DIR / "pinn_cases"
RESULTS_FILE = DATA_DIR / "results_summary.json"
SEED_RUNS_FILE = DATA_DIR / "seed_runs.json"
ROUNDS_E1_FILE = DATA_DIR / "rounds_e1.json"

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


def round_rms(case: dict) -> list[float | None]:
    """Cp RMS of every snapshot against the case's FVM reference points (research/metrics.py::cp_error: reference points with
    0.02 <= x <= 0.98, each compared with the nearest wall face). ``None`` each when the case has no usable reference."""
    snaps = case.get("snapshots") or []
    ref = (case.get("reference") or {}).get("points")
    surf = case.get("surface") or {}
    if not snaps or not ref or not surf.get("x"):
        return [None] * len(snaps)
    r = np.asarray(ref, dtype=float)
    r = r[(r[:, 0] >= 0.02) & (r[:, 0] <= 0.98)]
    if r.size == 0:
        return [None] * len(snaps)
    x, y = np.asarray(surf["x"], dtype=float), np.asarray(surf["y"], dtype=float)
    nearest = ((x[None, :] - r[:, 0:1]) ** 2 + (y[None, :] - r[:, 1:2]) ** 2).argmin(axis=1)  # geometry only: the same for every round
    return [round(float(np.sqrt(np.mean((np.asarray(s["cp"], dtype=float)[nearest] - r[:, 2]) ** 2))), 6) for s in snaps]


def strip_fvm(case: dict) -> dict:
    """A copy of a case without any FVM Cp curve (public mode). Removed: ``reference`` (FVM 800x400 points) and the ``points`` of
    ``fvm_same_mesh``. Kept, labelled with their source: the name of the reference (``fvm_ref_name``), the numeric same-mesh result
    (``fvm_same_errors``: name, Cp RMS, max, Cl), the Cp RMS of every training round (``rms_by_round``) and all of this project's own data.
    The result says so itself (``fvm_curves_withheld: true``)."""
    out = {k: v for k, v in case.items() if k not in ("reference", "fvm_same_mesh")}
    ref = case.get("reference")
    same = case.get("fvm_same_mesh")
    if ref:
        out["fvm_ref_name"] = ref.get("name")
        out["rms_by_round"] = case.get("rms_by_round") or round_rms(case)
    if same:
        out["fvm_same_errors"] = {k: v for k, v in same.items() if k != "points"}
    out["fvm_curves_withheld"] = True
    return out


def _served(case: dict) -> dict:
    return strip_fvm(case) if config.withhold_fvm_data() else case


def get_case(case_id: str) -> dict:
    """The full case JSON as a dict, flow field included (read from the file each time; in public mode without any FVM curve).
    ``KeyError`` for an unknown id."""
    with case_path(case_id).open(encoding="utf8") as f:
        return _served(json.load(f))


def _raw_meta(case_id: str) -> dict:
    """The case without its ``field``, exactly as exported (cached by modification time). Internal: it may hold FVM curves."""
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


_stripped: dict[Path, tuple[tuple[int, int], dict]] = {}


def get_case_meta(case_id: str) -> dict:
    """The case without its ``field`` (cached by modification time): everything the lists and the showdown need. In public mode
    without any FVM curve (the stripped copy is cached too)."""
    case = _raw_meta(case_id)
    if not config.withhold_fvm_data():
        return case
    path = case_path(case_id)
    st = path.stat()
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _stripped.get(path)
    if hit is None or hit[0] != stamp:
        hit = (stamp, strip_fvm(case))
        _stripped[path] = hit
    return hit[1]


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
    out = [_summary(_raw_meta(i)) for i in case_ids()]  # a summary holds no curve, so the raw case is fine here
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


def load_rounds_e1() -> dict:
    """``{"source", "runs": [{run_id, method, positivity, seed, mesh, rounds, cp_rms, cl}]}``: the E1 trajectories; ``{}`` if the file is missing."""
    if not ROUNDS_E1_FILE.is_file():
        return {}
    return _read_json(ROUNDS_E1_FILE)
