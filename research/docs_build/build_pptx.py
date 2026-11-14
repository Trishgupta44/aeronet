"""Build ``deliverables/3_Panel_Presentation.pptx`` from ``docs/deliverables_src/slides.md``.

    research/.venv/Scripts/python research/docs_build/build_pptx.py            # build
    research/.venv/Scripts/python research/docs_build/build_pptx.py --check    # fill only: list missing numbers

Source format (slides are separated by a line ``---``):

    # Title text                   (required)
    Eyebrow: 03 Verification       (small caps line above the title)
    Tag: measured                  (pill at the top right: measured / quoted / fitted)
    Layout: title                  (title slide) or demo (default is a normal slide)
    - a bullet with **bold**       (at most 6 lines; 20-22 pt)
    | a | table | row |            (markdown table; the separator row is skipped)
    ![caption](path/from/repo)     (one or two images; the text in [] is the caption and the alt text)
    Notes:                         (everything after this line is the speaker notes)
    Placeholders ``{{E2|fvm|100x50|rms_cp}}`` are filled from ``results/`` first (see common_numbers.py); a missing
    number raises ``KeyError`` and nothing is built.

16:9, built on the default "Title Only" layout, fonts Arial Narrow (titles) and Calibri (text), not embedded.
No third-party logos. Footer on every slide: the NNfoil credit.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import common_numbers  # noqa: E402

SRC = REPO / "docs" / "deliverables_src" / "slides.md"
OUT = REPO / "deliverables" / "3_Panel_Presentation.pptx"

NAVY, INDIGO, SKY = RGBColor(0x0A, 0x1A, 0x5C), RGBColor(0x1A, 0x0F, 0x9E), RGBColor(0xEF, 0xF4, 0xF7)
INK, MUTED, LINE, WHITE = RGBColor(0x0B, 0x10, 0x20), RGBColor(0x56, 0x60, 0x70), RGBColor(0xDF, 0xE5, 0xEC), RGBColor(255, 255, 255)
TITLE_FONT, BODY_FONT, MONO_FONT = "Arial Narrow", "Calibri", "Consolas"
CREDIT = "NNfoil by Cao, Song and Zhang (arXiv 2401.08705)"
SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.6


# ----------------------------------------------------------------------------- parsing
def parse_slides(md: str) -> list[dict]:
    """Split the (already filled) markdown into slide dicts."""
    blocks, cur = [], []
    for line in md.splitlines():
        if line.strip() == "---":
            blocks.append(cur)
            cur = []
        else:
            cur.append(line)
    blocks.append(cur)
    slides = []
    for b in blocks:
        if not any(x.strip() for x in b):
            continue
        s = {"title": "", "eyebrow": "", "tag": "", "layout": "normal", "bullets": [], "table": [], "images": [],
             "notes": ""}
        notes, in_notes = [], False
        for line in b:
            if in_notes:
                notes.append(line)
                continue
            t = line.strip()
            if not t:
                continue
            if t.startswith("Notes:"):
                in_notes = True
                rest = t[len("Notes:"):].strip()
                if rest:
                    notes.append(rest)
            elif t.startswith("# "):
                s["title"] = t[2:].strip()
            elif t.startswith("Eyebrow:"):
                s["eyebrow"] = t[8:].strip()
            elif t.startswith("Tag:"):
                s["tag"] = t[4:].strip()
            elif t.startswith("Layout:"):
                s["layout"] = t[7:].strip().lower()
            elif t.startswith("- "):
                s["bullets"].append(t[2:].strip())
            elif t.startswith("|"):
                cells = [c.strip() for c in t.strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    s["table"].append(cells)
            else:
                m = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", t)
                if m:
                    s["images"].append((m.group(1).strip(), m.group(2).strip()))
        s["notes"] = "\n".join(notes).strip()
        slides.append(s)
    return slides


# ----------------------------------------------------------------------------- drawing helpers
def _runs(par, text, size, color, font=BODY_FONT, bold=False, italic=False):
    """Add runs to a paragraph; ``**x**`` is bold."""
    for i, piece in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if not piece:
            continue
        r = par.add_run()
        r.text = piece
        r.font.size = Pt(size)
        r.font.name = font
        r.font.bold = bold or (i % 2 == 1)
        r.font.italic = italic
        r.font.color.rgb = color


def _textbox(slide, x, y, w, h, wrap=True, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    return tb, tf


def _bullet(par, indent_in=0.28):
    pPr = par._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(indent_in))))
    pPr.set("indent", str(-int(Inches(indent_in))))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    clr = pPr.makeelement(qn("a:buClr"), {})
    srgb = clr.makeelement(qn("a:srgbClr"), {"val": "1A0F9E"})
    clr.append(srgb)
    pPr.append(clr)
    bu = pPr.makeelement(qn("a:buChar"), {"char": "■"})
    pPr.append(bu)


def estimate_height(bullets, width_in, pt):
    """Rough height in inches of the bullets at ``pt`` (Calibri averages about 0.5 em per character)."""
    cw = 0.47 * pt / 72.0
    per_line = max(8, int((width_in - 0.4) / cw))
    lines = 0
    for b in bullets:
        plain = re.sub(r"\*\*", "", b)
        lines += max(1, math.ceil(len(plain) / per_line))
    return lines * pt * 1.22 / 72.0 + len(bullets) * 0.14


def fit_size(bullets, width_in, height_in, start=22, minimum=14):
    pt = start
    while pt > minimum and estimate_height(bullets, width_in, pt) > height_in:
        pt -= 1
    return pt


def add_bullets(slide, bullets, x, y, w, h, start=22):
    pt = fit_size(bullets, w, h, start)
    _, tf = _textbox(slide, x, y, w, h)
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        _bullet(p)
        p.space_after = Pt(9)
        _runs(p, b, pt, INK)
    return pt


def add_picture(slide, path, box, caption, alt):
    x, y, w, h = box
    full = REPO / path
    if not full.exists():
        raise FileNotFoundError(f"slide image missing: {path}")
    iw, ih = Image.open(full).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    px, py = x + (w - pw) / 2, y + (h - ph) / 2
    pic = slide.shapes.add_picture(str(full), Inches(px), Inches(py), Inches(pw), Inches(ph))
    pic._element.nvPicPr.cNvPr.set("descr", alt or caption or Path(path).stem)
    if caption:
        _, tf = _textbox(slide, x, py + ph + 0.05, w, 0.4)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        _runs(p, caption, 11, MUTED, italic=True)
    return pic


def add_table(slide, rows, x, y, w, pt=15):
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    weights = [max(8, max(len(re.sub(r"\*\*", "", r[c])) for r in rows)) ** 0.7 for c in range(ncol)]
    total = sum(weights)
    widths = [w * wt / total for wt in weights]
    row_h = 0.62
    gf = slide.shapes.add_table(len(rows), ncol, Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    for attr in ("firstRow", "bandRow"):
        tblPr.set(attr, "0")
    for c, cw in enumerate(widths):
        tbl.columns[c].width = Inches(cw)
    for r, row in enumerate(rows):
        tbl.rows[r].height = Inches(row_h)
        for c, text in enumerate(row):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if r == 0 else (SKY if r % 2 == 0 else WHITE)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            _runs(tf.paragraphs[0], text, pt, WHITE if r == 0 else INK, bold=(r == 0 or c == 0))
    return gf


def add_footer(slide, n, total, dark=False):
    col = RGBColor(0xB8, 0xC4, 0xE8) if dark else MUTED
    _, tf = _textbox(slide, MARGIN, 7.02, 8.5, 0.3)
    _runs(tf.paragraphs[0], CREDIT, 10, col)
    _, tf = _textbox(slide, SLIDE_W - MARGIN - 3.0, 7.02, 3.0, 0.3)
    tf.paragraphs[0].alignment = PP_ALIGN.RIGHT
    _runs(tf.paragraphs[0], f"AeroNet  ·  {n} / {total}", 10, col)


def add_tag(slide, text):
    w = max(1.6, 0.13 * len(text) + 0.5)
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(SLIDE_W - MARGIN - w), Inches(0.42), Inches(w), Inches(0.34))
    shp.fill.solid()
    shp.fill.fore_color.rgb = SKY
    shp.line.color.rgb = LINE
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _runs(p, text.upper(), 11, INDIGO, font=MONO_FONT, bold=True)


def set_title(slide, text, color, size=34, x=MARGIN, y=0.62, w=None, h=0.95):
    t = slide.shapes.title
    t.left, t.top = Inches(x), Inches(y)
    t.width, t.height = Inches(w or (SLIDE_W - 2 * MARGIN)), Inches(h)
    tf = t.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = Inches(0.04)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    p.line_spacing = 0.9
    pt = size if len(text) <= 48 else size - 4 if len(text) <= 62 else size - 8
    _runs(p, text.upper(), pt, color, font=TITLE_FONT, bold=True)


# ----------------------------------------------------------------------------- slide types
def build_title_slide(prs, s, n, total):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    bg.shadow.inherit = False
    slide.shapes._spTree.remove(bg._element)
    slide.shapes._spTree.insert(2, bg._element)  # behind everything
    if s["eyebrow"]:
        _, tf = _textbox(slide, 0.9, 1.1, 10, 0.4)
        _runs(tf.paragraphs[0], s["eyebrow"].upper(), 13, RGBColor(0xB8, 0xC4, 0xE8), font=MONO_FONT, bold=True)
    set_title(slide, s["title"], WHITE, size=72, x=0.9, y=1.5, w=11, h=1.3)
    _, tf = _textbox(slide, 0.9, 3.0, 11.2, 3.6)
    for i, b in enumerate(s["bullets"]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(12)
        fill_in = "[FILL IN" in b
        _runs(p, b, 24 if i == 0 else 20, RGBColor(0xFF, 0xD9, 0x7A) if fill_in else WHITE)
    add_footer(slide, n, total, dark=True)
    return slide


def build_normal_slide(prs, s, n, total):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    if s["eyebrow"]:
        _, tf = _textbox(slide, MARGIN, 0.3, 8, 0.32)
        _runs(tf.paragraphs[0], s["eyebrow"].upper(), 12, INDIGO, font=MONO_FONT, bold=True)
    set_title(slide, s["title"], NAVY, w=SLIDE_W - 2 * MARGIN - (2.4 if s["tag"] else 0))
    if s["tag"]:
        add_tag(slide, s["tag"])
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(MARGIN), Inches(1.6), Inches(SLIDE_W - 2 * MARGIN), Inches(0.03))
    rule.fill.solid()
    rule.fill.fore_color.rgb = LINE
    rule.line.fill.background()
    rule.shadow.inherit = False

    top, bottom = 1.8, 6.85
    full_w = SLIDE_W - 2 * MARGIN
    imgs = s["images"]
    bullets = s["bullets"]
    if s["table"]:
        y = top
        if bullets:
            h = min(1.3, 0.5 * len(bullets) + 0.2)
            add_bullets(slide, bullets, MARGIN, y, full_w, h, start=20)
            y += h + 0.1
        add_table(slide, s["table"], MARGIN, y, full_w, pt=15 if len(s["table"]) <= 5 else 13)
    elif len(imgs) == 2:
        y = top
        if bullets:
            h = min(1.7, 0.5 * len(bullets) + 0.15)
            add_bullets(slide, bullets, MARGIN, y, full_w, h, start=20)
            y += h + 0.05
        bw = (full_w - 0.3) / 2
        for k, (cap, path) in enumerate(imgs):
            add_picture(slide, path, (MARGIN + k * (bw + 0.3), y, bw, bottom - y - 0.45), cap, cap)
    elif len(imgs) == 1:
        cap, path = imgs[0]
        iw, ih = Image.open(REPO / path).size
        aspect = iw / ih
        if not bullets:
            add_picture(slide, path, (1.0, top, SLIDE_W - 2.0, bottom - top - 0.45), cap, cap)
        elif aspect >= 1.75:  # wide figure: bullets on top, picture below
            h = min(1.9, 0.5 * len(bullets) + 0.2)
            add_bullets(slide, bullets, MARGIN, top, full_w, h, start=20)
            y = top + h + 0.05
            add_picture(slide, path, (MARGIN, y, full_w, bottom - y - 0.35), cap, cap)
        else:
            bw = 5.5
            add_bullets(slide, bullets, MARGIN, top, bw, bottom - top, start=22)
            add_picture(slide, path, (MARGIN + bw + 0.3, top, full_w - bw - 0.3, bottom - top - 0.45), cap, cap)
    else:
        add_bullets(slide, bullets, MARGIN, top + 0.1, full_w - 1.0, bottom - top - 0.2, start=26)
    add_footer(slide, n, total)
    return slide


def build(md: str, out_path, rows=None) -> list[dict]:
    """Fill the placeholders (hard error on a missing number), build the deck, return the parsed slides."""
    filled = common_numbers.fill_text(md, rows)
    slides = parse_slides(filled)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SLIDE_W), Inches(SLIDE_H)
    total = len(slides)
    for n, s in enumerate(slides, 1):
        if not s["title"]:
            raise ValueError(f"slide {n} has no '# Title'")
        slide = (build_title_slide if s["layout"] == "title" else build_normal_slide)(prs, s, n, total)
        slide.notes_slide.notes_text_frame.text = s["notes"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    return slides


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fill only: list missing numbers and [FILL IN] fields")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    text = Path(a.src).read_text(encoding="utf8")
    errs = common_numbers.check_text(text)
    for e in errs:
        print("MISSING:", e)
    print(f"{len(errs)} missing numbers")
    for f in sorted(set(common_numbers.fill_ins(text))):
        print("  field:", f)
    if errs or a.check:
        return 1 if errs else 0
    slides = build(text, a.out)
    print(f"built {a.out}: {len(slides)} slides")
    return 0


if __name__ == "__main__":
    sys.exit(main())
