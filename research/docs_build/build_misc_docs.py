"""Build the four short documents of Task D4 from ``docs/deliverables_src/``.

    research/.venv/Scripts/python research/docs_build/build_misc_docs.py            # build all four into deliverables/
    research/.venv/Scripts/python research/docs_build/build_misc_docs.py --check    # only fill: missing numbers and fields

    synopsis.md         -> deliverables/4_Synopsis.docx
    progress_report.md  -> deliverables/1_Progress_Report.docx
    papers_guide.md     -> deliverables/5_Related_Papers_Guide.docx
    viva_sheet.md       -> deliverables/6_Viva_Sheet.docx

The department gave no template for these, so they are plain: A4, Times New Roman 12 pt, 1.5 line spacing, 2 cm margins,
page numbers centred in the footer. If your department has a template, paste the text into it.

Markdown subset: ``# title`` (once, first), lines before the first ``## `` are the cover block (``Key: value``),
``## heading``, ``### subheading``, paragraphs, ``- `` bullets, ``1. `` numbered items, ``| tables |``, and inline
``**bold**``, ``*italic*``, ```code``` and ``[FILL IN: ...]`` (highlighted yellow). Placeholders ``{{...}}`` are filled from
``results/`` first (see common_numbers.py); a missing number raises ``KeyError`` and the document is not built.
``[FILL IN: ...]`` fields may only appear in the cover block or under a heading that contains "Signature".
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import common_numbers  # noqa: E402

SRC = REPO / "docs" / "deliverables_src"
OUT = REPO / "deliverables"

# name -> (source, output, headings every build must contain)
DOCS = {
    "progress": ("progress_report.md", "1_Progress_Report.docx",
                 ["1. Summary", "2. Milestone 1", "3. Milestone 2", "4. Milestone 3", "5. Remaining work"]),
    "synopsis": ("synopsis.md", "4_Synopsis.docx",
                 ["1. Problem", "2. Objectives", "3. Methodology", "4. Expected and achieved results",
                  "5. Scope and limitations", "6. References"]),
    "papers": ("papers_guide.md", "5_Related_Papers_Guide.docx",
               ["How to read the tags", "1. NNfoil", "2. Sun, Sengupta and Juniper", "3. NeuralFoil", "4. Raissi"]),
    "viva": ("viva_sheet.md", "6_Viva_Sheet.docx",
             ["How to use this sheet", "Standard questions", "Questions from this project's own results"]
             + [f"{n}. " for n in range(1, 16)]),
}

FONT = "Times New Roman"
TOKEN = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\[FILL IN[^\]]*\]|(?<![\w*])\*[^*\s][^*]*?\*(?![\w*]))")


# ----------------------------------------------------------------------------- fields
def fill_in_violations(md: str) -> list[str]:
    """Lines with a ``[FILL IN`` field outside the cover block and the signature section."""
    bad, in_cover, allowed = [], True, True
    for line in md.splitlines():
        if line.startswith("## "):
            in_cover = False
            allowed = "signature" in line.lower()
        elif line.startswith("# ") and not line.startswith("## "):
            continue
        if "[FILL IN" in line and not (in_cover or allowed):
            bad.append(line.strip())
    return bad


# ----------------------------------------------------------------------------- docx helpers
def _set_font(run, size=12, bold=None, italic=None, name=FONT, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color


def add_inline(par, text, size=12, bold=False, italic=False):
    for piece in TOKEN.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            r = par.add_run(piece[2:-2])
            _set_font(r, size, bold=True, italic=italic)
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            r = par.add_run(piece[1:-1])
            _set_font(r, size - 1.5, bold=bold, italic=italic, name="Courier New")
        elif piece.startswith("[FILL IN"):
            r = par.add_run(piece)
            _set_font(r, size, bold=bold, italic=italic)
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            r = par.add_run(piece[1:-1])
            _set_font(r, size, bold=bold, italic=True)
        else:
            r = par.add_run(piece)
            _set_font(r, size, bold=bold, italic=italic)


def _spacing(par, line=1.5, before=0, after=6):
    pf = par.paragraph_format
    pf.line_spacing = line
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)


def _shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def _page_number_footer(doc):
    sec = doc.sections[0]
    p = sec.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    _set_font(run, 11)
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)


def new_document(title: str) -> Document:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2.0)
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    st.font.size = Pt(12)
    st.paragraph_format.line_spacing = 1.5
    doc.core_properties.title = title
    doc.core_properties.author = ""
    _page_number_footer(doc)
    return doc


def add_table(doc, rows):
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    tbl = doc.add_table(rows=len(rows), cols=ncol)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    total = 17.0  # usable width in cm
    weights = [max(16, max(len(re.sub(r"[*`]", "", r[c])) for r in rows)) ** 0.6 for c in range(ncol)]
    for c in range(ncol):
        w = Cm(total * weights[c] / sum(weights))
        for r in range(len(rows)):
            tbl.cell(r, c).width = w
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            cell = tbl.cell(r, c)
            par = cell.paragraphs[0]
            _spacing(par, line=1.0, before=2, after=2)
            add_inline(par, text, size=10.5, bold=(r == 0))
            if r == 0:
                _shade(cell, "D9E2F3")
    sp = doc.add_paragraph()
    _spacing(sp, line=1.0, after=4)
    return tbl


def parse_md(md: str) -> dict:
    """Split into title, cover lines and body blocks."""
    lines = md.splitlines()
    title = ""
    cover, body = [], []
    seen_h2 = False
    for ln in lines:
        if ln.startswith("# ") and not title:
            title = ln[2:].strip()
        elif ln.startswith("## "):
            seen_h2 = True
            body.append(ln)
        elif seen_h2:
            body.append(ln)
        elif ln.strip():
            cover.append(ln.strip())
    return {"title": title, "cover": cover, "body": body}


def build_doc(md: str, out_path, rows=None) -> dict:
    """Fill the placeholders (hard error on a missing number) and write the .docx. Returns a summary dict."""
    md = common_numbers.apply_fields(md)
    bad = fill_in_violations(md)
    if bad:
        raise ValueError("[FILL IN] field outside the cover block or a Signature section: " + bad[0])
    filled = common_numbers.fill_text(md, rows)
    parsed = parse_md(filled)
    doc = new_document(parsed["title"])
    headings = []

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _spacing(p, line=1.15, before=0, after=6)
    add_inline(p, parsed["title"], size=16, bold=True)
    for i, ln in enumerate(parsed["cover"]):
        p = doc.add_paragraph()
        _spacing(p, line=1.15, after=3)
        if i == 0:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(p, ln, size=12, italic=True)
        else:
            key, sep, val = ln.partition(":")
            if sep and not key.startswith("["):
                add_inline(p, key + ":", size=12, bold=True)
                add_inline(p, val, size=12)
            else:
                add_inline(p, ln, size=12)

    i, body = 0, parsed["body"]
    while i < len(body):
        ln = body[i]
        s = ln.strip()
        if not s:
            i += 1
        elif s.startswith("## "):
            p = doc.add_paragraph()
            _spacing(p, line=1.15, before=10, after=4)
            p.paragraph_format.keep_with_next = True
            add_inline(p, s[3:], size=14, bold=True)
            headings.append((1, s[3:]))
            i += 1
        elif s.startswith("### "):
            p = doc.add_paragraph()
            _spacing(p, line=1.15, before=8, after=2)
            p.paragraph_format.keep_with_next = True
            add_inline(p, s[4:], size=12, bold=True)
            headings.append((2, s[4:]))
            i += 1
        elif s.startswith("|"):
            tbl_rows = []
            while i < len(body) and body[i].strip().startswith("|"):
                cells = [c.strip() for c in body[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    tbl_rows.append(cells)
                i += 1
            add_table(doc, tbl_rows)
        elif s.startswith("- "):
            p = doc.add_paragraph()
            _spacing(p, after=3)
            p.paragraph_format.left_indent = Cm(1.0)
            p.paragraph_format.first_line_indent = Cm(-0.5)
            add_inline(p, "•  " + s[2:])
            i += 1
        elif re.match(r"\d+\. ", s):
            m = re.match(r"(\d+)\. (.*)", s)
            p = doc.add_paragraph()
            _spacing(p, after=3)
            p.paragraph_format.left_indent = Cm(1.0)
            p.paragraph_format.first_line_indent = Cm(-0.7)
            add_inline(p, f"{m.group(1)}.  " + m.group(2))
            i += 1
        else:
            p = doc.add_paragraph()
            _spacing(p, after=6)
            add_inline(p, s)
            i += 1
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    return {"title": parsed["title"], "headings": headings, "fields": common_numbers.fill_ins(md), "path": out_path}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fill only: list missing numbers and [FILL IN] fields")
    ap.add_argument("--only", choices=sorted(DOCS))
    a = ap.parse_args(argv)
    rows = common_numbers.load_rows()
    total_missing, fields = 0, {}
    for name in ([a.only] if a.only else list(DOCS)):
        src, outname, _ = DOCS[name]
        path = SRC / src
        if not path.exists():
            print(f"{name}: no source yet")
            continue
        text = common_numbers.apply_fields(path.read_text(encoding="utf8"))
        errs = common_numbers.check_text(text, rows)
        for e in errs:
            print(f"MISSING in {name}: {e}")
        total_missing += len(errs)
        for f in fill_in_violations(text):
            print(f"FIELD in the wrong place in {name}: {f}")
        for f in common_numbers.fill_ins(text):
            fields.setdefault(f, []).append(name)
        if not errs and not a.check:
            info = build_doc(text, OUT / outname, rows)
            print(f"built {info['path']}")
    print(f"{total_missing} missing numbers")
    print(f"{len(fields)} distinct [FILL IN] fields:")
    for f, where in sorted(fields.items()):
        print(f"  {f}  ({', '.join(where)})")
    return 1 if total_missing else 0


if __name__ == "__main__":
    sys.exit(main())
