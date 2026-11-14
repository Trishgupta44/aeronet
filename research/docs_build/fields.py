"""People-specific details of the deliverables: one source of truth, ``docs/deliverables_src/fields.json``.

The document sources (``report.md``, ``slides.md``, ``synopsis.md`` ...) name a detail with ``{{field:key}}``. Every
builder replaces it from ``fields.json`` before it fills the numbers, so a name or enrolment number is typed once.

Rules (the same in every builder):

* a key with a non-empty value is replaced by the value;
* a key that is empty or absent in ``fields.json`` becomes a visible ``[FILL IN: <description>]`` field, so it is still
  highlighted in the document and listed by every ``--check``;
* a key that this module does not know (a typo in a source, or an unknown key in ``fields.json``) is a hard error
  (``KeyError`` / ``ValueError`` naming it), never a silent blank.

Base keys come from ``fields.json``; the derived keys (``students_names_enrol`` ...) are assembled from them and are
empty, hence a visible field, as soon as one of their parts is empty. Placeholders inside fenced code blocks and HTML
comments are examples and are left as written.

A key starting with ``_`` in ``fields.json`` is a note (for example ``_source``) and is ignored.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIELDS_PATH = REPO / "docs" / "deliverables_src" / "fields.json"

FIELD = re.compile(r"\{\{field:([^}]*)\}\}")

# base key -> description shown in the visible field when the value is missing
BASE = {
    "student_a_name": "name, student A",
    "student_a_enrol": "enrolment number, student A",
    "student_b_name": "name, student B",
    "student_b_enrol": "enrolment number, student B",
    "student_c_name": "name, student C",
    "student_c_enrol": "enrolment number, student C",
    "student_a_lead_area": "lead area, student A",
    "student_b_lead_area": "lead area, student B",
    "student_c_lead_area": "lead area, student C",
    "student_a_contribution": "what student A specifically contributed",
    "student_b_contribution": "what student B specifically contributed",
    "student_c_contribution": "what student C specifically contributed",
    "supervisor_name": "supervisor name",
    "supervisor_designation": "supervisor designation",
    "hod_name": "HOD name",
    "institute": "institute",
    "department_line": "department line",
    "course": "course name",
    "programme": "programme",
    "degree_programme": "degree and programme",
    "semester": "semester",
    "academic_year": "academic year",
}

DERIVED = {
    "students_names": "student names",
    "students_names_enrol": "student names and enrolment numbers",
    "student_a_name_enrol": "name and enrolment number, student A",
    "student_b_name_enrol": "name and enrolment number, student B",
    "student_c_name_enrol": "name and enrolment number, student C",
    "supervisor_name_designation": "supervisor name and designation",
    "department_institute": "department and institute",
    "programme_semester_year": "programme, semester and year",
    "course_semester_year": "course, semester and year",
    "roles_summary": "one sentence per student on who led which work package",
}

KNOWN = {**BASE, **DERIVED}


# ----------------------------------------------------------------------------- loading
def load_fields(path=None) -> dict[str, str]:
    """The base values of ``fields.json`` (strings, stripped). Unknown keys and bad values raise ``ValueError``."""
    p = Path(path) if path is not None else FIELDS_PATH
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf8"))
    if not isinstance(data, dict):
        raise ValueError(f"{p}: must be a JSON object")
    out = {}
    for k, v in data.items():
        if k.startswith("_"):
            continue
        if k not in BASE:
            raise ValueError(f"{p}: unknown field {k!r} (known: {', '.join(sorted(BASE))})")
        if not isinstance(v, str):
            raise ValueError(f"{p}: field {k!r} must be a string")
        v = v.strip()
        if re.search(r"[|\n\r]", v):
            raise ValueError(f"{p}: field {k!r} may not contain '|' or a line break")
        out[k] = v
    return out


def _join(items: list[str]) -> str:
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else (items[0] if items else "")


def _lower_first(s: str) -> str:
    return s if s[:2].isupper() else s[:1].lower() + s[1:]


def values(base: dict[str, str] | None = None) -> dict[str, str]:
    """Every key (base and derived) with its value; an empty string means "not given"."""
    b = {k: "" for k in BASE}
    b.update(load_fields() if base is None else base)
    v = dict(b)
    for x in "abc":
        n, e = b[f"student_{x}_name"], b[f"student_{x}_enrol"]
        v[f"student_{x}_name_enrol"] = f"{n} ({e})" if n and e else ""
    names = [b[f"student_{x}_name"] for x in "abc"]
    both = [v[f"student_{x}_name_enrol"] for x in "abc"]
    v["students_names"] = _join(names) if all(names) else ""
    v["students_names_enrol"] = _join(both) if all(both) else ""
    sn, sd = b["supervisor_name"], b["supervisor_designation"]
    v["supervisor_name_designation"] = f"{sn}, {sd}" if sn and sd else ""
    d, i = b["department_line"], b["institute"]
    v["department_institute"] = f"{d}, {i}" if d and i else ""
    pr, se, ay, co = b["programme"], b["semester"], b["academic_year"], b["course"]
    v["programme_semester_year"] = f"{pr}, {se}, {ay}" if pr and se and ay else ""
    v["course_semester_year"] = f"{co}, {se}, {ay}" if co and se and ay else ""
    leads = [(b[f"student_{x}_name"], b[f"student_{x}_lead_area"]) for x in "abc"]
    v["roles_summary"] = (_join([f"{n} led the {_lower_first(a)}" for n, a in leads]) + "."
                          if all(n and a for n, a in leads) else "")
    return v


def visible(key: str) -> str:
    """The visible field that stands in for a key without a value."""
    return f"[FILL IN: {KNOWN[key]}]"


def resolve(key: str, vals: dict[str, str]) -> str:
    key = key.strip()
    if key not in KNOWN:
        raise KeyError(f"{{{{field:{key}}}}}: unknown field (known: {', '.join(sorted(KNOWN))})")
    return vals.get(key, "") or visible(key)


# ----------------------------------------------------------------------------- applying
_LITERAL = re.compile(r"(```.*?```|<!--.*?-->)", re.S)


def apply(text: str, base: dict[str, str] | None = None) -> str:
    """Replace every ``{{field:key}}`` (outside code blocks and comments); an unknown key raises ``KeyError``."""
    vals = values(base)
    parts = _LITERAL.split(text)
    return "".join(p if i % 2 else FIELD.sub(lambda m: resolve(m.group(1), vals), p) for i, p in enumerate(parts))


def used_keys(text: str) -> list[str]:
    """The keys named in ``text`` (outside code blocks and comments), in order of appearance."""
    parts = _LITERAL.split(text)
    return [m.group(1).strip() for i, p in enumerate(parts) if not i % 2 for m in FIELD.finditer(p)]
