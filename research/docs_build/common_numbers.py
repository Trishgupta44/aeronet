"""Number filling shared by the slide and the small-document builders (D3 and D4).

It reuses ``numbers_fill`` (the ``{{experiment|method|mesh|column[|case]}}`` placeholders with a hard ``KeyError``
for a missing number) and adds one thing: a ratio placeholder

    {{ratio: E2|fvm|100x50|rms_cp ; E2|pinn|100x50|rms_cp}}   ->  "3.2x"   (first divided by second, one decimal)

Both halves are looked up with the same exact-one-row rule, so a missing number is still an error.

Other forms (all hard errors when the file, row or cell is missing):

    {{sec:E4|fvm|400x200|infer_ms}}                      the cell (milliseconds) shown in seconds, "285 s"
    {{csv:e1_rounds.csv|case=NACA0012_a4_s0|round=3|cp_rms:f3}}
                                                         one cell of any CSV in results/: key=value filters must select
                                                         exactly one row; the last part is the column, with an optional
                                                         format (f3 = 3 decimals, e2 = 1.23e-04, g3, i = integer)
    {{count:seed_check.csv|mode=nopos|method=cold}}      how many rows match (key!=value is allowed; zero is a valid count)
Also here: the ``[FILL IN: ...]`` field finder.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numbers_fill  # noqa: E402

FILL_IN = re.compile(r"\[FILL IN[^\]]*\]")
RATIO = re.compile(r"\{\{ratio:(.*?)\}\}")
PLACEHOLDER = numbers_fill.PLACEHOLDER


def load_rows():
    return numbers_fill.load_rows()


def _ratio(body: str, rows) -> str:
    ph = "{{ratio:" + body + "}}"
    halves = [h.strip() for h in body.split(";")]
    if len(halves) != 2:
        raise KeyError(f"{ph}: a ratio needs two placeholder bodies separated by ';'")
    vals = []
    for h in halves:
        parts = [p.strip() for p in h.split("|")]
        if len(parts) not in (4, 5):
            raise KeyError(f"{ph}: each half needs experiment|method|mesh|column[|case]")
        case = parts[4] if len(parts) == 5 else None
        v = numbers_fill.lookup(rows, parts[0], parts[1], parts[2], parts[3], case, _ph=ph)
        vals.append(numbers_fill._f(v))
    if vals[1] == 0:
        raise KeyError(f"{ph}: division by zero")
    return f"{vals[0] / vals[1]:.1f}x"


SEC = re.compile(r"\{\{sec:(.*?)\}\}")
CSV = re.compile(r"\{\{csv:(.*?)\}\}")
COUNT = re.compile(r"\{\{count:(.*?)\}\}")


def _read_csv(name: str, ph: str):
    path = numbers_fill.RESULTS / name
    if not path.exists():
        raise KeyError(f"{ph}: results/{name} does not exist")
    try:
        return numbers_fill._read(path)
    except ValueError as e:
        raise KeyError(f"{ph}: {e}")


def _match(rows, filters, ph):
    out = rows
    for f in filters:
        m = re.fullmatch(r"([^=!]+)(!?=)(.*)", f)
        if not m:
            raise KeyError(f"{ph}: '{f}' is not a key=value or key!=value filter")
        k, op, v = m.group(1).strip(), m.group(2), m.group(3).strip()
        if rows and k not in rows[0]:
            raise KeyError(f"{ph}: column {k!r} does not exist")
        out = [r for r in out if ((r.get(k, "").strip() == v) == (op == "="))]
    return out


def _sec(body: str, rows) -> str:
    ph = "{{sec:" + body + "}}"
    parts = [p.strip() for p in body.split("|")]
    if len(parts) not in (4, 5):
        raise KeyError(f"{ph}: needs experiment|method|mesh|column[|case]")
    v = numbers_fill._f(numbers_fill.lookup(rows, parts[0], parts[1], parts[2], parts[3],
                                             parts[4] if len(parts) == 5 else None, _ph=ph)) / 1000.0
    return f"{v:.0f} s" if v >= 10 else f"{v:.1f} s"


def _csv(body: str) -> str:
    ph = "{{csv:" + body + "}}"
    parts = [p.strip() for p in body.split("|")]
    if len(parts) < 3:
        raise KeyError(f"{ph}: needs file|filter...|column")
    name, filters, col = parts[0], parts[1:-1], parts[-1]
    col, _, fmt = col.partition(":")
    rows = _match(_read_csv(name, ph), filters, ph)
    if len(rows) != 1:
        raise KeyError(f"{ph}: expected exactly one row, found {len(rows)}")
    if col not in rows[0]:
        raise KeyError(f"{ph}: column {col!r} does not exist")
    cell = (rows[0][col] or "").strip()
    if cell == "":
        raise KeyError(f"{ph}: the cell {col!r} is empty")
    try:
        v = float(cell)
    except ValueError:
        return cell
    if fmt == "i":
        return f"{v:.0f}"
    if fmt and fmt[0] in "feg":
        return f"{v:.{int(fmt[1:])}{fmt[0]}}"
    return f"{v:.4g}"


def _count(body: str) -> str:
    ph = "{{count:" + body + "}}"
    parts = [p.strip() for p in body.split("|")]
    return str(len(_match(_read_csv(parts[0], ph), parts[1:], ph)))


def fill_text(text: str, rows=None) -> str:
    """Fill every placeholder in ``text``. The first missing number raises ``KeyError`` naming it."""
    rows = load_rows() if rows is None else rows
    text = RATIO.sub(lambda m: _ratio(m.group(1), rows), text)
    text = SEC.sub(lambda m: _sec(m.group(1), rows), text)
    text = CSV.sub(lambda m: _csv(m.group(1)), text)
    text = COUNT.sub(lambda m: _count(m.group(1)), text)
    return numbers_fill.fill(text, rows)


def check_text(text: str, rows=None) -> list[str]:
    """Every missing-number message in ``text`` (empty list when all placeholders resolve)."""
    rows = load_rows() if rows is None else rows
    errs = []
    for rx, fn in ((RATIO, lambda b: _ratio(b, rows)), (SEC, lambda b: _sec(b, rows)), (CSV, _csv), (COUNT, _count)):
        for m in rx.finditer(text):
            try:
                fn(m.group(1))
            except (KeyError, ValueError) as e:
                errs.append(str(e.args[0]) if e.args else str(e))
    rest = COUNT.sub("", CSV.sub("", SEC.sub("", RATIO.sub("", text))))
    errs += numbers_fill.check(rest, rows)
    return errs


def fill_ins(text: str) -> list[str]:
    return FILL_IN.findall(text)
