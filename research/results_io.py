"""Readers for the files in ``results/`` shared by ``make_figures.py`` and ``report_tables.py``.

Everything here fails loudly: a missing file or a row that cannot be found raises an error that names it (the report must
never be built from a silently empty input). Nothing in this module holds a result number.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import metrics  # noqa: E402

FVM_DIR = REPO / "NNfoil" / "FVM_Cp"
REF_0012 = FVM_DIR / "NACA0012_800_400_Cp.dat"
REF_CYL = FVM_DIR / "Cylinder_800_400_Cp.dat"


def require(path) -> Path:
    """The path as a ``Path``, or ``FileNotFoundError`` naming it."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing input: {p}")
    return p


def read_csv(path) -> list[dict]:
    with require(path).open(newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def read_summary(path=None) -> list[dict]:
    return read_csv(path if path is not None else RESULTS / "summary.csv")


def read_json(path):
    return json.loads(require(path).read_text(encoding="utf8"))


def num(v):
    """``float`` of a CSV cell, or None for an empty or non-numeric one."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return None if math.isnan(f) else f


def select(rows, **crit) -> list[dict]:
    """Rows whose columns equal the criteria (numbers compare as numbers, everything else as text)."""
    out = []
    for r in rows:
        ok = True
        for k, want in crit.items():
            have = r.get(k)
            if isinstance(want, (int, float)) and not isinstance(want, bool):
                have = num(have)
                ok = have is not None and abs(have - want) < 1e-9
            else:
                ok = str(have) == str(want)
            if not ok:
                break
        if ok:
            out.append(r)
    return out


def pick(rows, **crit) -> dict:
    """The one row that matches; raises ``LookupError`` (naming the criteria) for none or several."""
    got = select(rows, **crit)
    if len(got) != 1:
        raise LookupError(f"expected exactly one summary row for {crit}, found {len(got)}")
    return got[0]


def note_value(notes: str, key: str):
    """Number after ``key=`` in a notes cell (``rounds=19/20`` is handled by :func:`rounds_of`), or None."""
    m = re.search(rf"{re.escape(key)}=(-?\d+(?:\.\d+)?(?:e-?\d+)?)", notes or "")
    return float(m.group(1)) if m else None


def rounds_of(notes: str):
    """``(good, attempted)`` from ``rounds=19/20`` in a notes cell, or None."""
    m = re.search(r"rounds=(\d+)/(\d+)", notes or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def round_series(run_dir, ref_path) -> dict:
    """Per logged round of a run folder: ``{"round","cp_rms","cl"}`` (lists), Cp RMS against ``ref_path`` (section 5.5).

    Uses only the committed ``log.json`` and ``snapshots.json`` of the run (no ``field.npz``, no torch).
    """
    d = Path(run_dir)
    log = read_json(d / "log.json")
    snap = read_json(d / "snapshots.json")
    by_round = {int(r["round"]): r["cp"] for r in snap["rounds"]}
    ref = metrics.load_fvm(require(ref_path))
    out = {"round": [], "cp_rms": [], "cl": []}
    for i, r in enumerate(log["round"]):
        r = int(r)
        err = metrics.cp_error(snap["x"], snap["y"], by_round[r], ref)
        out["round"].append(r)
        out["cp_rms"].append(err["rms"])
        out["cl"].append(float(log["cl"][i]))
    return out
