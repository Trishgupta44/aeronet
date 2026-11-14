"""Tests for research/docs_build/fields.py: the people-specific details of the deliverables (fields.json)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from docx import Document

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "research" / "docs_build"))

import build_all  # noqa: E402
import common_numbers as cn  # noqa: E402
import fields as pf  # noqa: E402

BASE = {"student_a_name": "Ann Example", "student_a_enrol": "111", "student_b_name": "Bo Example",
        "student_b_enrol": "222", "student_c_name": "Cy Example", "student_c_enrol": "333",
        "supervisor_name": "Dr Test", "supervisor_designation": "Professor"}


def test_a_filled_field_is_replaced():
    assert pf.apply("Hi {{field:student_a_name}} ({{field:student_a_enrol}}).", BASE) == "Hi Ann Example (111)."
    assert pf.apply("{{field:students_names_enrol}}", BASE) == "Ann Example (111), Bo Example (222) and Cy Example (333)"
    assert pf.apply("{{field:supervisor_name_designation}}", BASE) == "Dr Test, Professor"
    assert pf.apply("{{field:student_b_name}}", BASE) == "Bo Example"


def test_an_empty_or_absent_value_stays_a_visible_fill_in_field():
    assert pf.apply("HOD: {{field:hod_name}}", {**BASE, "hod_name": ""}) == "HOD: [FILL IN: HOD name]"
    assert pf.apply("HOD: {{field:hod_name}}", BASE) == "HOD: [FILL IN: HOD name]"
    # a derived key with a missing part is a visible field, not a half-filled sentence
    part = {k: v for k, v in BASE.items() if k != "student_c_enrol"}
    assert pf.apply("{{field:students_names_enrol}}", part) == "[FILL IN: student names and enrolment numbers]"
    # and the builders' check lists it
    assert cn.FILL_IN.findall(pf.apply("{{field:semester}}", {})) == ["[FILL IN: semester]"]


def test_an_unknown_key_fails_loudly():
    with pytest.raises(KeyError, match="student_z_name"):
        pf.apply("{{field:student_z_name}}", BASE)


def test_unknown_or_bad_entries_in_fields_json_fail_loudly(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"student_a_nmae": "typo"}), encoding="utf8")
    with pytest.raises(ValueError, match="student_a_nmae"):
        pf.load_fields(p)
    p.write_text(json.dumps({"student_a_name": "A | B"}), encoding="utf8")
    with pytest.raises(ValueError, match="student_a_name"):
        pf.load_fields(p)
    p.write_text(json.dumps({"_note": "ignored", "student_a_name": " Ann "}), encoding="utf8")
    assert pf.load_fields(p) == {"student_a_name": "Ann"}


def test_examples_in_code_blocks_and_comments_are_left_alone():
    text = "<!-- {{field:nope}} -->\n```\n{{field:nope}}\n```\n{{field:semester}}"
    assert pf.apply(text, {"semester": "7th"}) == "<!-- {{field:nope}} -->\n```\n{{field:nope}}\n```\n7th"


def test_every_builder_applies_the_fields_before_the_numbers():
    text = "Team {{field:students_names}}; number {{E2|fvm|50x25|rms_cp}}"
    rows = [{"experiment": "E2", "case": "", "method": "fvm", "mesh": "50x25", "rms_cp": "0.1", "cl": "", "cl_ref": "",
             "train_s": "", "infer_ms": "", "notes": ""}]
    out = cn.fill_text(text, rows)
    assert "0.1000" in out and "{{" not in out
    assert cn.check_text(text, rows) == []
    with pytest.raises(KeyError, match="nope"):
        cn.fill_text("{{field:nope}}", rows)


def test_the_real_fields_json_is_valid_and_the_sources_use_only_known_keys():
    base = pf.load_fields()  # raises on an unknown key
    for name in ("report.md", "slides.md", "synopsis.md", "progress_report.md", "papers_guide.md", "viva_sheet.md"):
        for key in pf.used_keys((REPO / "docs" / "deliverables_src" / name).read_text(encoding="utf8")):
            assert key in pf.KNOWN, f"{name}: {key}"
    for x in "abc":  # the three students are given completely, in the order of the cover table
        assert base[f"student_{x}_name"] and base[f"student_{x}_enrol"].isdigit()
    assert [base[f"student_{x}_enrol"] for x in "abc"] == sorted(base[f"student_{x}_enrol"] for x in "abc")


def test_the_cover_table_of_the_report_lists_the_three_students_in_order(tmp_path):
    base = pf.load_fields()
    text = pf.apply((REPO / "docs" / "deliverables_src" / "report.md").read_text(encoding="utf8"))
    import numbers_fill as nf
    import build_docx
    out = tmp_path / "r.docx"
    build_docx.build(nf.fill(text, nf.load_rows()), out, REPO)
    doc = Document(str(out))
    tables = [[[c.text for c in r.cells] for r in t.rows] for t in doc.tables]
    rows = next(t for t in tables if t[0] == ["Enrol. No.", "Name of Student"])
    assert rows[0] == ["Enrol. No.", "Name of Student"]
    assert rows[1:4] == [[base[f"student_{x}_enrol"], base[f"student_{x}_name"]] for x in "abc"]
    # a field without a value is still listed by the report builder's check
    left = build_all.process("report", True, False)["fields"]
    assert ("[FILL IN: HOD name]" in left) == (not base["hod_name"])
    assert not set(left) & {pf.visible(k) for k in pf.KNOWN if k != "hod_name"}
