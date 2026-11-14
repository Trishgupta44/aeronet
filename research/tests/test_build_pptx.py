"""Tests for research/docs_build/build_pptx.py and common_numbers.py (Task D3)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
from pptx import Presentation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "research" / "docs_build"))

import build_pptx  # noqa: E402
import common_numbers as cn  # noqa: E402

SLIDES_MD = REPO / "docs" / "deliverables_src" / "slides.md"

SAMPLE = """# First slide
- one
- two

Notes:
These are the notes of slide one.

Second paragraph.

---
# Second slide
- three
Notes: inline notes
"""


@pytest.fixture(scope="module")
def deck(tmp_path_factory):
    out = tmp_path_factory.mktemp("deck") / "deck.pptx"
    slides = build_pptx.build(SLIDES_MD.read_text(encoding="utf8"), out)
    return Presentation(str(out)), slides


def test_two_sample_slides_and_notes(tmp_path):
    out = tmp_path / "s.pptx"
    build_pptx.build(SAMPLE, out)
    prs = Presentation(str(out))
    assert len(prs.slides) == 2
    assert prs.slides[0].notes_slide.notes_text_frame.text == "These are the notes of slide one.\n\nSecond paragraph."
    assert prs.slides[1].notes_slide.notes_text_frame.text == "inline notes"


def test_deck_has_17_slides_16x9_and_every_slide_has_notes(deck):
    prs, _ = deck
    assert len(prs.slides) == 17
    assert abs(prs.slide_width / prs.slide_height - 16 / 9) < 0.01
    for i, s in enumerate(prs.slides, 1):
        assert len(s.notes_slide.notes_text_frame.text.split()) >= 30, f"slide {i} has thin or empty notes"


def test_no_unfilled_placeholder_in_text_or_notes(deck):
    prs, _ = deck
    for i, s in enumerate(prs.slides, 1):
        texts = [s.notes_slide.notes_text_frame.text]
        for sh in s.shapes:
            if sh.has_text_frame:
                texts.append(sh.text_frame.text)
            if sh.has_table:
                texts += [c.text for r in sh.table.rows for c in r.cells]
        for t in texts:
            assert "{{" not in t and "}}" not in t, f"slide {i}: unfilled placeholder in {t[:80]!r}"


def test_a_missing_number_raises(tmp_path):
    with pytest.raises(KeyError, match="E2"):
        build_pptx.build("# T\n- {{E2|fvm|9999x1|rms_cp}}\nNotes: x", tmp_path / "x.pptx")
    with pytest.raises(KeyError):
        cn.fill_text("{{csv:seed_check.csv|mode=nope|method=cold|final_cl}}")
    with pytest.raises(KeyError):
        cn.fill_text("{{ratio:E2|fvm|100x50|rms_cp ; E2|fvm|9999x1|rms_cp}}")
    with pytest.raises(KeyError):
        cn.fill_text("{{csv:no_such_file.csv|a=1|b}}")
    assert any("9999x1" in e for e in cn.check_text("{{E2|fvm|9999x1|rms_cp}}"))


def test_fill_in_fields_only_on_the_title_slide(deck):
    prs, slides = deck
    for i, s in enumerate(slides, 1):
        body = "\n".join([s["title"], s["eyebrow"], *s["bullets"], s["notes"]] + [" ".join(r) for r in s["table"]])
        found = cn.fill_ins(body)
        if i == 1:
            assert len(found) == 3 and all(f.startswith("[FILL IN:") for f in found)
        else:
            assert not found, f"slide {i} carries a [FILL IN] field"


def test_images_exist_and_are_on_the_slides(deck):
    prs, slides = deck
    want = [len(s["images"]) for s in slides]
    assert sum(want) >= 12
    for i, (s, n) in enumerate(zip(prs.slides, want), 1):
        pics = [sh for sh in s.shapes if sh.shape_type == 13]
        assert len(pics) == n, f"slide {i}: {len(pics)} pictures, {n} expected"
        for sh in pics:
            assert sh.image.blob  # the image is embedded
            assert sh._element.nvPicPr.cNvPr.get("descr")  # alt text


def test_footer_credit_on_every_slide(deck):
    prs, _ = deck
    for i, s in enumerate(prs.slides, 1):
        assert any(sh.has_text_frame and "arXiv 2401.08705" in sh.text_frame.text for sh in s.shapes), f"slide {i}"


def test_every_content_slide_has_at_most_six_bullet_lines(deck):
    _, slides = deck
    for i, s in enumerate(slides, 1):
        assert len(s["bullets"]) <= 6 or s["layout"] == "title", f"slide {i}"


def test_extra_placeholder_forms():
    assert cn.fill_text("{{ratio:E2|fvm|100x50|rms_cp ; E2|pinn|100x50|rms_cp}}") == "3.2x"
    assert cn.fill_text("{{sec:E4|fvm|400x200|infer_ms}}") == "285 s"
    assert cn.fill_text("{{count:seed_check.csv|mode=nopos|method=cold|collapsed=yes}}") == "1"
    assert cn.fill_text("{{count:e6_replay.csv|drifted=True|stop_round!=never}}") == "0"
    assert cn.fill_text("{{csv:finding3_mesh_smoothing.csv|variant=upstream|max_abs_winslow_residual:e2}}") == "1.95e-04"


def _csv(name):
    with (REPO / "results" / name).open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def test_claims_typed_in_the_slide_notes_match_the_result_files():
    """The few numbers written by hand in slides.md (ranges and counts) are checked against the files here."""
    e1 = [r for r in _csv("e1_rounds.csv")]
    runs = {}
    for r in e1:
        runs.setdefault((r["method"], r["case"]), []).append(r)
    assert len(runs) == 6
    r3 = [float(next(x for x in v if x["round"] == "3")["cp_rms"]) for v in runs.values()]
    assert 0.027 <= min(r3) and max(r3) <= 0.032 + 1e-4  # "between 0.027 and 0.032"
    finals = {k: float(max(v, key=lambda x: int(x["round"]))["cp_rms"]) for k, v in runs.items()}
    drift = [v for v in finals.values() if v > 0.06]
    assert len(drift) == 5 and round(min(drift), 3) == 0.073 and round(max(drift), 3) == 0.222  # "5 of 6", "0.073 to 0.222"
    cl_final = [float(max(v, key=lambda x: int(x["round"]))["cl"]) for k, v in runs.items() if finals[k] > 0.06]
    assert round(min(cl_final), 3) == 0.168 and round(max(cl_final), 3) == 0.412  # "0.168 to 0.412"
    pos = [finals[k] for k in finals if k[0] == "pinn"]
    neg = [finals[k] for k in finals if k[0] == "pinn_nopos"]
    assert round(sum(pos) / 3, 3) == 0.130 and round(sum(neg) / 3, 3) == 0.146
    gs = [float(r["final_cl"]) for r in _csv("seed_check.csv") if r["method"] == "gs"]
    assert len(gs) == 8 and round(min(gs), 3) == 0.502 and round(max(gs), 3) == 0.516  # "0.502 and 0.516"
    e6 = _csv("e6_replay.csv")
    assert sum(r["drifted"] == "True" for r in e6) == 8
    assert len(e6) == 21
