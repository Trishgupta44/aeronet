"""Numbers for the generated documents: ``{{experiment|method|mesh|column[|case]}}`` placeholders.

Every number in a report comes from a file in ``results/``. A placeholder names one row; it must match exactly one row
and the cell must be non-empty, otherwise ``KeyError`` (the message names the placeholder). Nothing is ever left blank.

Rows come from ``results/summary.csv`` plus two views of other result files, so that every table in the report is
filled the same way:

* ``SEED``: ``results/seed_check.csv`` (experiment ``SEED``, method ``<mode>_<method>`` such as ``nopos_cold``,
  mesh = the seed number, columns as in that file).
* ``E1R``: ``results/e1_rounds.csv`` (experiment ``E1R``, method as in the file, mesh = the round number, case as in
  the file, columns as in that file).

Columns: any column of the row, plus the derived ``train_h`` (train_s / 3600), ``cl_err_pct`` (100 (cl - cl_ref) / cl_ref), ``cl_diff`` (cl - cl_ref)
and ``note:<key>`` (the number after ``key=`` in the notes cell). Append ``:N`` to force N decimals, for example
``rms_cp:3``. A mesh of ``-`` selects a row with an empty mesh.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = REPO / "results"

PLACEHOLDER = re.compile(r"\{\{(.*?)\}\}")


# ----------------------------------------------------------------------------- loading
def _read(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing input: {p}")
    with p.open(newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty input: {p}")
    return rows


def load_rows(summary=None, results_dir=None) -> list[dict]:
    """``summary.csv`` rows, plus the ``SEED`` and ``E1R`` rows when the files sit next to it."""
    summary = Path(summary) if summary is not None else (Path(results_dir) if results_dir else RESULTS) / "summary.csv"
    rows = _read(summary)
    d = summary.parent
    seed = d / "seed_check.csv"
    if seed.exists():
        for r in _read(seed):
            row = dict(r)
            row.update({"experiment": "SEED", "method": f"{r['mode']}_{r['method']}", "mesh": str(r["seed"]),
                        "case": r["run_id"], "cl": r.get("final_cl", ""), "rms_cp": r.get("cp_rms", ""),
                        "source": "measured", "notes": ""})
            rows.append(row)
    e1 = d / "e1_rounds.csv"
    if e1.exists():
        for r in _read(e1):
            row = dict(r)
            row.update({"experiment": "E1R", "mesh": str(r["round"]), "rms_cp": r["cp_rms"], "source": "measured",
                        "notes": ""})
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------- formatting
def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"not a number: {v!r}")


def split_col(column: str):
    """``rms_cp:3`` -> (``rms_cp``, 3); ``note:rounds`` -> (``note:rounds``, None)."""
    m = re.fullmatch(r"(.*):(\d+)", column)
    return (m.group(1), int(m.group(2))) if m else (column, None)


def fmt(value, column: str, decimals: int | None = None) -> str:
    """Text for a number. rms/max/cd 4 decimals, cl 3, train_s ``N s`` below 120 else ``N.N min``, infer_ms ``N.NN ms``."""
    col, d = split_col(column)
    if col in ("source", "method", "mesh", "case", "notes", "experiment", "mode", "run_id", "collapsed", "complete"):
        return str(value).strip()
    v = _f(value)
    if decimals is None:
        decimals = d
    if col == "train_h":
        return f"{v:.{2 if decimals is None else decimals}f} h"
    if col == "train_s" or col == "seconds":
        if decimals is not None:
            return f"{v:.{decimals}f} s"
        return f"{v:.0f} s" if v < 120 else f"{v / 60.0:.1f} min"
    if col == "infer_ms":
        return f"{v:.{2 if decimals is None else decimals}f} ms"
    if col == "cl_err_pct":
        return f"{v:+.{1 if decimals is None else decimals}f}%"
    if col == "cl_diff":
        return f"{v:+.{3 if decimals is None else decimals}f}"
    if decimals is None:
        if col in ("rms_cp", "max_cp", "cd", "cp_rms", "cp_max"):
            decimals = 4
        elif col in ("cl", "cl_ref", "final_cl", "cl_after_transfer"):
            decimals = 3
        elif col in ("loss",):
            return f"{v:.2e}"
        else:
            return f"{v:g}"
    return f"{v:.{decimals}f}"


# ----------------------------------------------------------------------------- lookup
def _note(notes: str, key: str):
    m = re.search(rf"{re.escape(key)}=(-?\d+(?:\.\d+)?(?:e-?\d+)?)", notes or "")
    return m.group(1) if m else None


def lookup(rows, exp, method, mesh, column, case=None, _ph=None):
    """The raw value of one cell (a string). ``KeyError`` naming the placeholder if no row, several rows, or empty."""
    ph = _ph or "{{" + "|".join(x for x in (exp, method, mesh, column, case) if x is not None) + "}}"
    want_mesh = "" if mesh in ("-", None) else mesh
    got = [r for r in rows if r.get("experiment") == exp and r.get("method") == method
           and (r.get("mesh") or "") == want_mesh and (case is None or r.get("case") == case)]
    if len(got) != 1:
        raise KeyError(f"{ph}: expected exactly one row for experiment={exp} method={method} mesh={mesh}"
                       f"{'' if case is None else ' case=' + case}, found {len(got)}")
    r = got[0]
    base, _ = split_col(column)
    if base.startswith("note:"):
        v = _note(r.get("notes", ""), base[5:])
    elif base == "train_h":
        t = (r.get("train_s") or "").strip()
        if not t:
            raise KeyError(f"{ph}: train_s is empty in the matching row")
        v = _f(t) / 3600.0
    elif base in ("cl_err_pct", "cl_diff"):
        cl, ref = (r.get("cl") or "").strip(), (r.get("cl_ref") or "").strip()
        if not cl or not ref:
            raise KeyError(f"{ph}: cl or cl_ref is empty in the matching row")
        v = (100.0 * (_f(cl) - _f(ref)) / _f(ref)) if base == "cl_err_pct" else (_f(cl) - _f(ref))
    else:
        if base not in r:
            raise KeyError(f"{ph}: column {base!r} does not exist")
        v = r[base]
    if v is None or str(v).strip() == "":
        raise KeyError(f"{ph}: the cell {base!r} is empty in the matching row")
    return v


def resolve(ph_body: str, rows) -> str:
    parts = [p.strip() for p in ph_body.split("|")]
    ph = "{{" + ph_body + "}}"
    if len(parts) not in (4, 5):
        raise KeyError(f"{ph}: a placeholder needs experiment|method|mesh|column or experiment|method|mesh|column|case")
    exp, method, mesh, column = parts[:4]
    case = parts[4] if len(parts) == 5 else None
    v = lookup(rows, exp, method, mesh, column, case, _ph=ph)
    return fmt(v, column)


# fenced code blocks and HTML comments are shown as written: a placeholder inside them is an example, not a number
_LITERAL = re.compile(r"(```.*?```|<!--.*?-->)", re.S)


def _apply(text: str, fn) -> str:
    parts = _LITERAL.split(text)
    return "".join(p if i % 2 else PLACEHOLDER.sub(fn, p) for i, p in enumerate(parts))


def fill(text: str, rows) -> str:
    """Replace every placeholder (outside code blocks and comments); the first problem raises ``KeyError``."""
    return _apply(text, lambda m: resolve(m.group(1), rows))


def check(text: str, rows) -> list[str]:
    """Every problem in ``text`` as a list of messages (empty when every placeholder resolves)."""
    errs = []

    def one(m):
        try:
            resolve(m.group(1), rows)
        except (KeyError, ValueError) as e:
            errs.append(str(e.args[0]) if e.args else str(e))
        return ""

    _apply(text, one)
    return errs
