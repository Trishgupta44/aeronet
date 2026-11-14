"""Tests for research/docs_build: placeholders with hard errors, number formats, and the template's page rules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "docs_build"))

import build_docx  # noqa: E402
import numbers_fill as nf  # noqa: E402
from docx import Document  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt  # noqa: E402

ROWS = [
    {"experiment": "E2", "case": "NACA0012_a4", "method": "pinn", "mesh": "50x25", "rms_cp": "0.0123", "cl": "0.4",
     "cl_ref": "0.5", "train_s": "1450", "infer_ms": "", "notes": "rounds=19/20; seed=0"},
    {"experiment": "E2", "case": "NACA0012_a4", "method": "fvm", "mesh": "50x25", "rms_cp": "0.1", "cl": "0.4",
     "cl_ref": "0.5", "train_s": "", "infer_ms": "", "notes": ""},
    {"experiment": "E2", "case": "NACA0012_a4", "method": "fvm", "mesh": "50x25", "rms_cp": "0.2", "cl": "", "cl_ref": "",
     "train_s": "", "infer_ms": "", "notes": ""},
]


def test_fill_replaces_a_known_placeholder():
    assert nf.fill("RMS {{E2|pinn|50x25|rms_cp}}.", ROWS) == "RMS 0.0123."
    assert nf.fill("{{E2|pinn|50x25|cl_err_pct}} {{E2|pinn|50x25|rms_cp:2}} {{E2|pinn|50x25|note:rounds}}", ROWS) \
        == "-20.0% 0.01 19"


def test_missing_row_and_empty_cell_raise_and_name_the_placeholder():
    with pytest.raises(KeyError) as e1:
        nf.fill("{{E9|pinn|50x25|rms_cp}}", ROWS)
    assert "{{E9|pinn|50x25|rms_cp}}" in str(e1.value)
    with pytest.raises(KeyError) as e2:
        nf.fill("{{E2|pinn|50x25|infer_ms}}", ROWS)
    assert "{{E2|pinn|50x25|infer_ms}}" in str(e2.value) and "empty" in str(e2.value)
    with pytest.raises(KeyError) as e3:  # two rows match: ambiguous
        nf.fill("{{E2|fvm|50x25|rms_cp}}", ROWS)
    assert "found 2" in str(e3.value)
    assert len(nf.check("{{E9|a|b|c}} {{E2|pinn|50x25|rms_cp}} {{E2|pinn|50x25|infer_ms}}", ROWS)) == 2


def test_fmt():
    assert nf.fmt(1450, "train_s") == "24.2 min"
    assert nf.fmt(90, "train_s") == "90 s"
    assert nf.fmt(0.01234, "rms_cp") == "0.0123"
    assert nf.fmt(0.5193, "cl") == "0.519"
    assert nf.fmt(1.2651, "infer_ms") == "1.27 ms"
    assert nf.fmt(-17.94, "cl_err_pct") == "-17.9%"


def test_real_summary_placeholders_resolve():
    rows = nf.load_rows()
    assert nf.fill("{{E2|fvm|50x25|rms_cp}}", rows) == "0.1081"
    assert nf.fill("{{SEED|nopos_cold|3|failed_rounds}}", rows) == "5"
    assert nf.fill("{{E1R|pinn|3|cp_rms:3|NACA0012_a4_s0}}", rows) == "0.030"


SAMPLE = """# Chapter 1: Intro

## 1.1 Background

A paragraph with **bold** text and a number 0.0123.

- first
- second

1. one
2. two

Table: Sample table

| Name | Value |
|---|---|
| RMS | 0.0123 |

![A picture](pic.png)

$$ a = b + c $$

\\pagebreak

\\mainmatter

# Chapter 2: More

## 2.1 Text

Some text. [FILL IN: a name]
"""


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("docs")
    from PIL import Image
    Image.new("RGB", (800, 400), "white").save(d / "pic.png")
    out = d / "t.docx"
    info = build_docx.build(SAMPLE, out, d)
    return Document(str(out)), info


def test_sample_has_one_table_and_one_picture_and_the_cell_text(built):
    doc, info = built
    assert len(doc.tables) == 1
    assert len(doc.inline_shapes) == 1
    assert doc.tables[0].rows[1].cells[1].text == "0.0123"
    assert info["figures"] == [("Figure 1.1", "A picture")]
    assert info["tables"] == [("Table 1.1", "Sample table")]
    assert info["fill_ins"] == ["[FILL IN: a name]"]
    styles = {p.style.name for p in doc.paragraphs}
    assert {"List Bullet", "List Number", "Heading 1", "Heading 2", "Caption"} <= styles
    texts = [p.text for p in doc.paragraphs]
    assert "Figure 1.1: A picture" in texts and "Table 1.1: Sample table" in texts


def test_page_and_font_rules(built):
    doc, _ = built
    for s in doc.sections:
        assert abs(s.page_width - Cm(21.0)) < Cm(0.01) and abs(s.page_height - Cm(29.7)) < Cm(0.01)
        for m in (s.left_margin, s.right_margin, s.top_margin, s.bottom_margin):
            assert abs(m - Cm(2.0)) < Cm(0.01)
    normal = doc.styles["Normal"]
    assert normal.font.name == "Times New Roman" and normal.font.size == Pt(12)
    assert normal.paragraph_format.line_spacing == 1.5
    assert doc.styles["Heading 1"].font.size == Pt(16)  # chapter titles
    assert doc.styles["Heading 2"].font.size == Pt(14)  # sub-headings
    assert doc.styles["Heading 1"].font.name == "Times New Roman"


def test_roman_section_then_decimal_section_with_a_page_field(built):
    doc, _ = built
    assert len(doc.sections) == 2
    fmts = []
    for s in doc.sections:
        el = s._sectPr.find(qn("w:pgNumType"))
        fmts.append((el.get(qn("w:fmt")), el.get(qn("w:start"))))
    assert fmts == [("lowerRoman", "1"), ("decimal", "1")]
    footer_xml = doc.sections[0].footer._element.xml
    assert "PAGE" in footer_xml and 'w:jc w:val="center"' in footer_xml
    assert doc.sections[0].different_first_page_header_footer  # the cover page shows no number


def test_numbered_lists_restart_at_one(tmp_path):
    out = tmp_path / "l.docx"
    build_docx.build("1. a\n2. b\n\ntext\n\n1. c\n", out, tmp_path)
    doc = Document(str(out))
    nums = [p._p.pPr.numPr.numId.val for p in doc.paragraphs if p.style.name == "List Number"]
    assert nums[0] == nums[1] != nums[2]


def test_missing_image_is_a_hard_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_docx.build("# Chapter 1: A\n\n![x](nope.png)\n", tmp_path / "x.docx", tmp_path)


def test_toc_field_and_static_lists(tmp_path):
    out = tmp_path / "t.docx"
    md = "\\toc\n\n# Chapter 1: A\n\n## 1.1 B\n\ntext\n"
    build_docx.build(md, out, tmp_path, pages={"Chapter 1: A": "1"})
    xml = Document(str(out)).element.xml
    assert "TOC " in xml and "Chapter 1: A" in xml and "1.1 B" in xml


def test_report_source_has_no_missing_numbers_and_builds(tmp_path):
    """The committed report source: every placeholder resolves, and the template's chapters are all there."""
    src = nf.REPO / "docs" / "deliverables_src" / "report.md"
    text = src.read_text(encoding="utf8")
    rows = nf.load_rows()
    assert nf.check(text, rows) == []
    filled = nf.fill(text, rows)
    assert "{{E2" not in filled.split("# Appendix C")[0]
    info = build_docx.build(filled, tmp_path / "report.docx", nf.REPO)
    titles = [t for lv, t in info["headings"] if lv in (0, 1)]
    for need in ("Certificate", "Declaration of Originality", "Acknowledgement", "Abstract", "Table of Contents",
                 "List of Figures", "List of Tables", "List of Abbreviations", "Chapter 1: Introduction",
                 "Chapter 2: Literature Survey", "Chapter 3: System Design and Methodology",
                 "Chapter 4: Implementation and Testing", "Chapter 5: Results and Discussion",
                 "Chapter 6: Conclusion and Future Scope", "References", "Annexure"):
        assert need in titles, need
    assert len(info["figures"]) >= 20 and len(info["tables"]) >= 15
    doc = Document(str(tmp_path / "report.docx"))
    assert len(doc.sections) == 2
