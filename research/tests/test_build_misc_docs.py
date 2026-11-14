"""Tests for research/docs_build/build_misc_docs.py (Task D4: synopsis, progress report, papers guide, viva sheet)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from docx import Document
from docx.shared import Cm, Pt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "research" / "docs_build"))

import build_misc_docs as bm  # noqa: E402
import common_numbers as cn  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("docs")
    res = {}
    for name, (src, outname, _) in bm.DOCS.items():
        md = (bm.SRC / src).read_text(encoding="utf8")
        res[name] = (bm.build_doc(md, out / outname), Document(str(out / outname)), md)
    return res


def _text(doc):
    parts = [p.text for p in doc.paragraphs]
    for t in doc.tables:
        parts += [c.text for r in t.rows for c in r.cells]
    return "\n".join(parts)


def test_all_four_documents_build_with_the_required_headings(built):
    assert set(built) == {"progress", "synopsis", "papers", "viva"}
    for name, (info, doc, _) in built.items():
        heads = [h for _, h in info["headings"]]
        for want in bm.DOCS[name][2]:
            assert any(h.startswith(want) for h in heads), f"{name}: missing heading {want!r}"


def test_no_unfilled_placeholder_remains(built):
    for name, (_, doc, _) in built.items():
        text = _text(doc)
        assert "{{" not in text and "}}" not in text, name


def test_page_format_is_a4_times_12_with_2cm_margins_and_1_5_spacing(built):
    for name, (_, doc, _) in built.items():
        sec = doc.sections[0]
        assert abs(sec.page_width - Cm(21.0)) < Cm(0.01) and abs(sec.page_height - Cm(29.7)) < Cm(0.01)
        for m in (sec.left_margin, sec.right_margin, sec.top_margin, sec.bottom_margin):
            assert abs(m - Cm(2.0)) < Cm(0.01), name
        normal = doc.styles["Normal"]
        assert normal.font.name == "Times New Roman" and normal.font.size == Pt(12)
        assert normal.paragraph_format.line_spacing == 1.5


def test_a_missing_number_stops_the_build(tmp_path):
    md = "# T\ncover line\n## 1. A\nNumber {{E2|fvm|9999x1|rms_cp}}.\n"
    with pytest.raises(KeyError, match="9999x1"):
        bm.build_doc(md, tmp_path / "x.docx")


def test_fill_in_fields_only_in_the_cover_block_or_signature_section(built, tmp_path):
    for name, (info, doc, md) in built.items():
        assert not bm.fill_in_violations(md), name
    bad = "# T\nStudents: [FILL IN: names]\n## 1. Body\nSigned by [FILL IN: someone]\n"
    assert bm.fill_in_violations(bad) == ["Signed by [FILL IN: someone]"]
    with pytest.raises(ValueError):
        bm.build_doc(bad, tmp_path / "bad.docx")
    ok = "# T\nStudents: [FILL IN: names]\n## 1. Body\ntext\n## Signatures\nStudents: [FILL IN: signatures]\n"
    assert not bm.fill_in_violations(ok)


def test_fill_ins_are_highlighted_and_nothing_is_invented(built):
    for name, (_, doc, md) in built.items():
        for f in cn.fill_ins(md):
            assert f in _text(doc), f"{name}: {f} not in the document"
        # No student or supervisor name from the repository's earlier drafts has been copied in.
        for banned in ("Awasthi", "Krishnna", "Riya", "9923103"):
            assert banned not in _text(doc), f"{name}: contains {banned!r}"


def test_viva_sheet_has_fifteen_questions_each_citing_evidence(built):
    _, doc, md = built["viva"]
    qs = re.findall(r"^### (\d+)\. ", md, flags=re.M)
    assert [int(q) for q in qs] == list(range(1, 16))
    blocks = re.split(r"^### \d+\. ", md, flags=re.M)[1:]
    assert len(blocks) == 15
    for i, b in enumerate(blocks, 1):
        assert "Evidence:" in b and "`" in b, f"question {i} has no evidence file"
        body = b.split("\n", 1)[1].strip()
        assert 40 <= len(body.split()) <= 140, f"question {i}: {len(body.split())} words"
    hard = " ".join(blocks[9:])
    for kw in ("stop rule", "Positivity", "circulation", "quoted", "200x100", "Prandtl-Glauert"):
        assert kw in hard


def test_papers_guide_tags_every_paper_and_has_a_viva_question_each(built):
    _, doc, md = built["papers"]
    for h in ("1. NNfoil", "2. Sun, Sengupta and Juniper", "3. NeuralFoil", "4. Raissi"):
        assert any(h in x for x in md.splitlines())
    assert md.count("**One viva question.**") == 4
    assert md.count("**The idea.**") == 4 and md.count("**What it costs.**") == 4 and md.count("**How AeroNet differs.**") == 4
    for tag in ("(abstract", "(team summary)", "(measured here)", "(paper text"):
        assert tag in md


def test_progress_report_lists_the_three_milestones_with_dates(built):
    _, doc, md = built["progress"]
    for m in ("M1", "M2", "M3"):
        assert f"Milestone {m[1]}" in md
    assert "5 to 6 November" in md and "7 to 8 November" in md and "6 to 14 November" in md
