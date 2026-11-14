"""Markdown to .docx in the institute's capstone-report format (python-docx).

``build(md_text, out_path, base_dir, pages=None)`` returns a dict (headings, figures, tables, fill-in fields).

Page: A4, margins 2 cm, Times New Roman, 12 pt body (1.5 line spacing, justified), 14 pt sub-headings (``##``),
16 pt chapter titles (``#``), page numbers bottom-centre: the first Word section is numbered in lower-case Roman, the
section that starts at ``\\mainmatter`` is numbered in Arabic from 1. Tables and code listings use smaller type (10 pt
and 9 pt, single spacing) so that they fit the page; everything else follows the template.

Markdown subset
    #  chapter title (new page, 16 pt)         #* unnumbered front-page title (new page, 16 pt, centred)
    ## sub-heading (14 pt)                     ### sub-sub-heading (12 pt bold)
    paragraphs (consecutive lines join), ``- `` bullets, ``1. `` numbered lists (each list restarts at 1)
    | pipe | tables |  (``---`` row skipped, first row bold); a line ``Table: caption`` before it adds "Table n.m: caption"
    @plaintable before a pipe table: borderless, centred, no caption (the cover page)
    ![caption](path)  picture (at most 15 cm wide, centred) with "Figure n.m: caption"
    $$ expression $$  centred equation, numbered "(n.m)"
    ``` fenced code ``` (Courier New 9 pt)
    @center[size=18,bold,before=24]: text      centred line      @logo: path | fallback text
    \\pagebreak  \\mainmatter  \\toc  \\listoffigures  \\listoftables
    **bold**  *italic*  `code`  ^{sup}  _{sub}  <br>  [FILL IN: ...] (highlighted yellow)
Chapter numbers for captions come from the last ``# Chapter N`` (or ``# Appendix A``) heading.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor, Twips

FONT = "Times New Roman"
BODY_PT, SUB_PT, CHAPTER_PT = 12, 14, 16
TEXT_WIDTH_CM = 17.0  # A4 21 cm minus two 2 cm margins
FILL_IN = re.compile(r"\[FILL IN[^\]]*\]")

INLINE = re.compile(
    r"(\*\*.+?\*\*|(?<![\w*])\*(?=[^\s*])[^*]+?(?<=[^\s*])\*(?![\w*])|`[^`]+`|\[FILL IN[^\]]*\]|<br>|\^\{[^}]*\}|_\{[^}]*\})")


# ----------------------------------------------------------------------------- low-level helpers
def _set_rfonts(rpr, name=FONT):
    rfonts = rpr.get_or_add_rFonts()
    for a in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        k = qn("w:" + a)
        if k in rfonts.attrib:
            del rfonts.attrib[k]
    for a in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn("w:" + a), name)


def _style_font(style, size=None, bold=None, italic=None, color_black=False):
    _set_rfonts(style.element.get_or_add_rPr())
    style.font.name = FONT
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    if color_black:
        style.font.color.rgb = RGBColor(0, 0, 0)


def _add_field_runs(paragraph, instr):
    """Begin / instruction / separate runs of a field; returns nothing (the end run is added by :func:`_field_end`)."""
    for kind in ("begin", None, "separate"):
        r = paragraph.add_run()
        if kind:
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), kind)
            r._r.append(e)
        else:
            t = OxmlElement("w:instrText")
            t.set(qn("xml:space"), "preserve")
            t.text = f" {instr} "
            r._r.append(t)


def _field_end(paragraph):
    r = paragraph.add_run()
    e = OxmlElement("w:fldChar")
    e.set(qn("w:fldCharType"), "end")
    r._r.append(e)


def _add_page_field(paragraph):
    _add_field_runs(paragraph, "PAGE")
    paragraph.add_run("1")
    _field_end(paragraph)


def _page_setup(section):
    section.page_width, section.page_height = Twips(11906), Twips(16838)  # A4 exactly (210 x 297 mm)
    section.left_margin = section.right_margin = section.top_margin = section.bottom_margin = Twips(1134)  # 2 cm
    section.header_distance = section.footer_distance = Cm(1.0)


def _pg_num(section, fmt, start=None):
    sectPr = section._sectPr
    for old in sectPr.findall(qn("w:pgNumType")):
        sectPr.remove(old)
    el = OxmlElement("w:pgNumType")
    el.set(qn("w:fmt"), fmt)
    if start is not None:
        el.set(qn("w:start"), str(start))
    # schema order: pgNumType comes before cols/titlePg/docGrid; insert before the first of those if present
    anchor = None
    for tag in ("w:cols", "w:formProt", "w:vAlign", "w:noEndnote", "w:titlePg", "w:textDirection", "w:bidi",
                "w:rtlGutter", "w:docGrid"):
        anchor = sectPr.find(qn(tag))
        if anchor is not None:
            break
    if anchor is not None:
        anchor.addprevious(el)
    else:
        sectPr.append(el)


def _shade(cell, fill="E7E6E6"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def _row_props(row, header=False):
    trPr = row._tr.get_or_add_trPr()
    cs = OxmlElement("w:cantSplit")
    trPr.append(cs)
    if header:
        h = OxmlElement("w:tblHeader")
        trPr.append(h)


def _restart_numbering(doc, paragraph, style_name="List Number"):
    """Give ``paragraph`` a fresh numbering instance (so each numbered list starts at 1)."""
    st = doc.styles[style_name]
    base = st.element.pPr.numPr.numId.val
    numbering = doc.part.numbering_part.numbering_definitions._numbering
    abstract = numbering.num_having_numId(base).abstractNumId.val
    num = numbering.add_num(abstract)
    ov = num.add_lvlOverride(ilvl=0)
    ov.add_startOverride(1)
    return num.numId


def _set_num(paragraph, num_id):
    pPr = paragraph._p.get_or_add_pPr()
    numPr = pPr.get_or_add_numPr()
    numPr.get_or_add_ilvl().val = 0
    numPr.get_or_add_numId().val = num_id


# ----------------------------------------------------------------------------- inline text
def add_inline(par, text, size=None, bold=False, italic=False):
    """Add ``text`` (inline Markdown subset) to a paragraph."""
    pos = 0
    for m in INLINE.finditer(text):
        if m.start() > pos:
            _run(par, text[pos:m.start()], size, bold, italic)
        tok = m.group(0)
        if tok.startswith("**"):
            _run(par, tok[2:-2], size, True, italic)
        elif tok.startswith("`"):
            r = _run(par, tok[1:-1], size, bold, italic)
            r.font.name = "Courier New"
            _set_rfonts(r._r.get_or_add_rPr(), "Courier New")
            r.font.size = Pt((size or BODY_PT) - 1.5)
        elif tok.startswith("[FILL IN"):
            r = _run(par, tok, size, bold, italic)
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        elif tok == "<br>":
            par.add_run().add_break()
        elif tok.startswith("^{"):
            _run(par, tok[2:-1], size, bold, italic).font.superscript = True
        elif tok.startswith("_{"):
            _run(par, tok[2:-1], size, bold, italic).font.subscript = True
        else:
            _run(par, tok[1:-1], size, bold, True)
        pos = m.end()
    if pos < len(text):
        _run(par, text[pos:], size, bold, italic)


def _run(par, text, size, bold, italic):
    r = par.add_run(text)
    if size:
        r.font.size = Pt(size)
    if bold:
        r.font.bold = True
    if italic:
        r.font.italic = True
    return r


# ----------------------------------------------------------------------------- parsing
def _strip_md(s):
    return re.sub(r"\*\*|`|\^\{|_\{|\}", "", s).replace("*", "") if s else s


def parse(md_text: str) -> list[dict]:
    md_text = re.sub(r"<!--.*?-->", "", md_text, flags=re.S)
    lines = md_text.replace("\r\n", "\n").split("\n")
    blocks, para, i = [], [], 0

    def flush():
        if para:
            blocks.append({"t": "p", "text": " ".join(x.strip() for x in para)})
            para.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("<!--") and s.endswith("-->"):
            i += 1
            continue
        if not s:
            flush()
            i += 1
            continue
        if s.startswith("```"):
            flush()
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i].rstrip())
                i += 1
            i += 1
            blocks.append({"t": "code", "lines": code})
            continue
        if s in ("\\pagebreak", "\\mainmatter", "\\toc", "\\listoffigures", "\\listoftables", "@plaintable"):
            flush()
            blocks.append({"t": s.lstrip("\\@")})
            i += 1
            continue
        m = re.match(r"^(#\*|###|##|#)\s+(.*)$", s)
        if m:
            flush()
            lvl = {"#*": 0, "#": 1, "##": 2, "###": 3}[m.group(1)]
            blocks.append({"t": "h", "level": lvl, "text": m.group(2).strip()})
            i += 1
            continue
        m = re.match(r"^@center(?:\[(.*?)\])?:\s*(.*)$", s)
        if m:
            flush()
            blocks.append({"t": "center", "opts": m.group(1) or "", "text": m.group(2)})
            i += 1
            continue
        m = re.match(r"^@logo:\s*(.*?)\s*\|\s*(.*)$", s)
        if m:
            flush()
            blocks.append({"t": "logo", "path": m.group(1), "fallback": m.group(2)})
            i += 1
            continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)$", s)
        if m:
            flush()
            blocks.append({"t": "img", "caption": m.group(1), "path": m.group(2)})
            i += 1
            continue
        m = re.match(r"^\$\$\s*(.*?)\s*\$\$$", s)
        if m:
            flush()
            blocks.append({"t": "eq", "text": m.group(1)})
            i += 1
            continue
        m = re.match(r"^Table:\s*(.*)$", s)
        if m:
            flush()
            blocks.append({"t": "tcap", "text": m.group(1)})
            i += 1
            continue
        if s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append({"t": "table", "rows": rows})
            continue
        m = re.match(r"^(-|\d+\.)\s+(.*)$", s)
        if m:
            flush()
            kind = "bullet" if m.group(1) == "-" else "number"
            text = [m.group(2)]
            i += 1
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() \
                    and not re.match(r"^\s*(-|\d+\.)\s", lines[i]):
                text.append(lines[i].strip())
                i += 1
            blocks.append({"t": "li", "kind": kind, "text": " ".join(text)})
            continue
        para.append(ln)
        i += 1
    flush()
    return blocks


# ----------------------------------------------------------------------------- the document
def _new_document():
    doc = Document()
    _page_setup(doc.sections[0])
    cp = doc.core_properties
    cp.author, cp.last_modified_by, cp.comments = "", "", ""
    # docDefaults and Normal
    dd = doc.styles.element.find(qn("w:docDefaults"))
    if dd is not None:
        rpr = dd.find(qn("w:rPrDefault")).find(qn("w:rPr"))
        _set_rfonts(rpr)
    normal = doc.styles["Normal"]
    _style_font(normal, BODY_PT)
    pf = normal.paragraph_format
    pf.line_spacing = 1.5
    pf.space_after = Pt(6)
    pf.space_before = Pt(0)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for name, size in (("Heading 1", CHAPTER_PT), ("Heading 2", SUB_PT), ("Heading 3", BODY_PT)):
        st = doc.styles[name]
        _style_font(st, size, bold=True, italic=False, color_black=True)
        st.paragraph_format.line_spacing = 1.5
        st.paragraph_format.space_before = Pt(12 if name != "Heading 1" else 0)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.keep_with_next = True
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    doc.styles["Heading 1"].paragraph_format.page_break_before = True
    from docx.enum.style import WD_STYLE_TYPE
    ft = doc.styles.add_style("Front Title", WD_STYLE_TYPE.PARAGRAPH)
    ft.base_style = doc.styles["Normal"]
    _style_font(ft, CHAPTER_PT, bold=True, color_black=True)
    ft.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ft.paragraph_format.page_break_before = True
    ft.paragraph_format.space_after = Pt(12)
    ft.paragraph_format.keep_with_next = True
    ol = OxmlElement("w:outlineLvl")
    ol.set(qn("w:val"), "0")
    ft.element.get_or_add_pPr().append(ol)
    for name in ("List Bullet", "List Number", "Caption", "Table Grid"):
        st = doc.styles[name]
        if st.type == WD_STYLE_TYPE.PARAGRAPH:
            _style_font(st, BODY_PT, bold=False, italic=False, color_black=True)
    cap = doc.styles["Caption"]
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.line_spacing = 1.15
    cap.paragraph_format.space_after = Pt(10)
    for name in ("List Bullet", "List Number"):
        doc.styles[name].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        doc.styles[name].paragraph_format.space_after = Pt(3)
    return doc


def _footer(section):
    f = section.footer
    f.is_linked_to_previous = False
    p = f.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0
    _add_page_field(p)


def _label_of(heading_text, current):
    m = re.match(r"^Chapter\s+(\d+)", heading_text)
    if m:
        return m.group(1)
    m = re.match(r"^Appendix\s+([A-Z])\b", heading_text)
    if m:
        return m.group(1)
    if heading_text.startswith("Annexure"):
        return "AN"
    return current


def _widths(rows):
    n = max(len(r) for r in rows)
    w = []
    for c in range(n):
        longest = max((len(_strip_md(r[c])) if c < len(r) else 0) for r in rows)
        w.append(max(5, min(42, longest)) ** 0.8)
    tot = sum(w)
    return [Cm(TEXT_WIDTH_CM * x / tot) for x in w]


def _toc_paragraph(doc, level, text, page, first_instr=None, last=False, bold_top=True):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.15
    pf.space_after = Pt(2)
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.left_indent = Cm(0.6 * max(level - 1, 0))
    pf.tab_stops.add_tab_stop(Cm(TEXT_WIDTH_CM), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    if first_instr:
        _add_field_runs(p, first_instr)
    r = p.add_run(text + ("\t" + page if page else ""))
    if level == 1 and bold_top:
        r.font.bold = True
    if last:
        _field_end(p)
    return p


def build(md_text: str, out_path, base_dir=".", pages: dict | None = None) -> dict:
    """Build the document. ``pages`` maps heading/caption text to a page label for the static lists (optional)."""
    base = Path(base_dir)
    pages = pages or {}
    blocks = parse(md_text)
    doc = _new_document()
    sec0 = doc.sections[0]
    sec0.different_first_page_header_footer = True  # the cover page has no number
    _pg_num(sec0, "lowerRoman", 1)
    _footer(sec0)

    # first pass: headings, captions, counters
    label, nfig, ntab, neq = "0", 0, 0, 0
    heads, figs, tabs = [], [], []
    pending_tcap = None
    plan = []
    for b in blocks:
        if b["t"] == "h":
            if b["level"] in (0, 1):
                label = _label_of(b["text"], label)
                nfig = ntab = neq = 0
            heads.append((b["level"], b["text"]))
        elif b["t"] == "img":
            nfig += 1
            b["label"] = f"Figure {label}.{nfig}"
            figs.append((b["label"], b["caption"]))
        elif b["t"] == "tcap":
            pending_tcap = b["text"]
        elif b["t"] == "table":
            if pending_tcap is not None:
                ntab += 1
                b["label"] = f"Table {label}.{ntab}"
                b["caption"] = pending_tcap
                tabs.append((b["label"], pending_tcap))
                pending_tcap = None
        elif b["t"] == "eq":
            neq += 1
            b["label"] = f"({label}.{neq})"
    plain_next = False
    info = {"headings": heads, "figures": figs, "tables": tabs,
            "fill_ins": sorted(set(FILL_IN.findall(md_text)))}

    num_id = None
    prev = None
    for b in blocks:
        t = b["t"]
        if t == "tcap":
            continue
        if t == "plaintable":
            plain_next = True
            continue
        if t == "pagebreak":
            doc.add_page_break()
        elif t == "mainmatter":
            sec = doc.add_section(WD_SECTION.NEW_PAGE)
            _page_setup(sec)
            sec.different_first_page_header_footer = False
            _pg_num(sec, "decimal", 1)
            sec.footer.is_linked_to_previous = True
        elif t == "h":
            lvl = b["level"]
            if lvl == 0:
                p = doc.add_paragraph(style="Front Title")
            else:
                p = doc.add_paragraph(style=f"Heading {lvl}")
            add_inline(p, b["text"])
        elif t == "p":
            p = doc.add_paragraph()
            add_inline(p, b["text"])
        elif t == "li":
            style = "List Bullet" if b["kind"] == "bullet" else "List Number"
            p = doc.add_paragraph(style=style)
            if b["kind"] == "number":
                if prev is None or prev["t"] != "li" or prev["kind"] != "number":
                    num_id = _restart_numbering(doc, p)
                _set_num(p, num_id)
            add_inline(p, b["text"])
        elif t == "center":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            opts = {k.strip(): v for k, _, v in (o.partition("=") for o in b["opts"].split(",") if o.strip())}
            size = float(opts["size"]) if "size" in opts else None
            if "before" in opts:
                p.paragraph_format.space_before = Pt(float(opts["before"]))
            p.paragraph_format.space_after = Pt(float(opts.get("after", 6)))
            p.paragraph_format.line_spacing = 1.15
            add_inline(p, b["text"], size=size, bold="bold" in opts)
        elif t == "logo":
            path = base / b["path"]
            if path.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(str(path), height=Cm(5.0))
            else:
                tbl = doc.add_table(rows=1, cols=1)
                tbl.style = "Table Grid"
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                c = tbl.rows[0].cells[0]
                c.width = Cm(6)
                cp_ = c.paragraphs[0]
                cp_.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cp_.paragraph_format.line_spacing = 1.0
                add_inline(cp_, b["fallback"], size=11)
                doc.add_paragraph()
        elif t == "img":
            path = base / b["path"]
            if not path.exists():
                raise FileNotFoundError(f"image for '{b['caption']}' not found: {path}")
            from PIL import Image
            with Image.open(path) as im:
                w, h = im.size
            width_cm = 15.0 if w >= 600 else min(15.0, max(8.0, w / 40))  # small images are not blown up
            if width_cm * h / w > 12.0:
                width_cm = 12.0 * w / h
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.line_spacing = 1.0
            src = str(path)
            if w > 1800:  # keep the .docx small: 1800 px is 300 dpi at 15 cm
                import io
                with Image.open(path) as im:
                    im = im.convert("RGB").resize((1800, round(h * 1800 / w)), Image.LANCZOS)
                    buf = io.BytesIO()
                    im.save(buf, "PNG", optimize=True)
                    buf.seek(0)
                    src = buf
            p.add_run().add_picture(src, width=Cm(width_cm))
            cp_ = doc.add_paragraph(style="Caption")
            add_inline(cp_, f"{b['label']}: {b['caption']}")
        elif t == "eq":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_WIDTH_CM / 2), WD_TAB_ALIGNMENT.CENTER)
            p.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_WIDTH_CM), WD_TAB_ALIGNMENT.RIGHT)
            p.paragraph_format.line_spacing = 1.15
            p.add_run("\t")
            add_inline(p, b["text"])
            p.add_run("\t" + b["label"])
        elif t == "code":
            for ln in b["lines"] or [""]:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                pf = p.paragraph_format
                pf.line_spacing, pf.space_after, pf.left_indent = 1.0, Pt(0), Cm(0.5)
                r = p.add_run(ln)
                r.font.name = "Courier New"
                _set_rfonts(r._r.get_or_add_rPr(), "Courier New")
                r.font.size = Pt(9)
            doc.add_paragraph().paragraph_format.space_after = Pt(0)
        elif t == "table":
            rows = b["rows"]
            plain = plain_next
            plain_next = False
            if not plain and b.get("caption"):
                cp_ = doc.add_paragraph(style="Caption")
                cp_.paragraph_format.keep_with_next = True
                cp_.paragraph_format.space_after = Pt(4)
                add_inline(cp_, f"{b['label']}: {b['caption']}")
            ncol = max(len(r) for r in rows)
            tbl = doc.add_table(rows=len(rows), cols=ncol)
            tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            if not plain:
                tbl.style = "Table Grid"
            widths = _widths(rows) if not plain else None
            for ri, row in enumerate(rows):
                _row_props(tbl.rows[ri], header=(ri == 0 and not plain))
                for ci in range(ncol):
                    cell = tbl.rows[ri].cells[ci]
                    if widths:
                        cell.width = widths[ci]
                    par = cell.paragraphs[0]
                    par.paragraph_format.line_spacing = 1.0
                    par.paragraph_format.space_after = Pt(2)
                    par.alignment = WD_ALIGN_PARAGRAPH.CENTER if plain else WD_ALIGN_PARAGRAPH.LEFT
                    add_inline(par, row[ci] if ci < len(row) else "", size=12 if plain else 10, bold=(ri == 0))
                    if ri == 0 and not plain:
                        _shade(cell)
            if not plain:
                doc.add_paragraph().paragraph_format.space_after = Pt(4)
        elif t == "toc":
            entries = [(lv, tx) for lv, tx in heads]
            for k, (lv, tx) in enumerate(entries):
                _toc_paragraph(doc, max(lv, 1), _strip_md(tx), pages.get(_strip_md(tx), ""),
                               first_instr='TOC \\o "1-3" \\h \\z \\u' if k == 0 else None,
                               last=(k == len(entries) - 1))
        elif t in ("listoffigures", "listoftables"):
            entries = figs if t == "listoffigures" else tabs
            if not entries:
                p = doc.add_paragraph("(none)")
            for k, (lab, cap) in enumerate(entries):
                _toc_paragraph(doc, 1, f"{lab}: {_strip_md(cap)}", pages.get(lab, ""), bold_top=False)
        prev = b
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    return info
