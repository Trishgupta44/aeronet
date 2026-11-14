"""Fill the placeholders and build every deliverable that has a source in ``docs/deliverables_src/``.

    research/.venv/Scripts/python research/docs_build/build_all.py --check     # only fill: list missing numbers and fields
    research/.venv/Scripts/python research/docs_build/build_all.py             # build into deliverables/
    research/.venv/Scripts/python research/docs_build/build_all.py --pdf       # also convert with LibreOffice (page numbers
                                                                                 # of the contents lists come from the PDF)

``--check`` prints ``N missing numbers`` (it must be 0) and the ``[FILL IN: ...]`` fields still in the sources.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import build_docx  # noqa: E402
import numbers_fill  # noqa: E402

SRC = REPO / "docs" / "deliverables_src"
OUT = REPO / "deliverables"
# name -> (source, output, kind). Other tasks add their documents here.
DOCS = {
    "report": ("report.md", "2_Project_Report_AeroNet.docx", "docx"),
}


def soffice() -> str | None:
    for c in (shutil.which("soffice"), r"C:\Program Files\LibreOffice\program\soffice.exe",
              r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"):
        if c and Path(c).exists():
            return c
    return None


def to_pdf(docx: Path, outdir: Path) -> Path | None:
    exe = soffice()
    if exe is None:
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", str(outdir), str(docx)],
                   check=True, capture_output=True, timeout=300)
    pdf = outdir / (docx.stem + ".pdf")
    return pdf if pdf.exists() else None


def roman(n: int) -> str:
    out = ""
    for v, s in ((10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= v:
            out += s
            n -= v
    return out


def page_labels(pdf: Path, info: dict) -> dict:
    """Map heading text and caption labels to the page label printed in the footer (from the PDF text)."""
    from pypdf import PdfReader
    pages = [[ln.strip() for ln in (p.extract_text() or "").splitlines()] for p in PdfReader(str(pdf)).pages]
    first_chapter = next((t for lv, t in info["headings"] if lv == 1), None)
    main = next((i for i, ls in enumerate(pages) if ls and first_chapter and ls[0].startswith(first_chapter[:30])), None)
    if main is None:
        return {}

    def label(i):
        return roman(i + 1) if i < main else str(i - main + 1)

    out = {}

    def find(text, rng, exact):
        key = re.sub(r"\s+", " ", text)
        for i in rng:
            for ln in pages[i]:
                ln = re.sub(r"\s+", " ", ln)
                if (ln == key) if exact else ln.startswith(key[:38]):
                    return label(i)
        return None

    for lv, t in info["headings"]:
        k = build_docx._strip_md(t)
        got = find(k, range(0, main) if lv == 0 else range(main, len(pages)), exact=(lv == 0))
        if got:
            out[k] = got
    for lab, _ in info["figures"] + info["tables"]:
        got = find(lab + ":", range(main, len(pages)), exact=False)
        if got:
            out[lab] = got
    return out


def process(name, check_only, want_pdf):
    src, outname, kind = DOCS[name]
    path = SRC / src
    if not path.exists():
        return None
    text = path.read_text(encoding="utf8")
    rows = numbers_fill.load_rows()
    errs = numbers_fill.check(text, rows)
    fields = sorted(set(build_docx.FILL_IN.findall(text)))
    if check_only or errs:
        return {"name": name, "errors": errs, "fields": fields, "built": None}
    filled = numbers_fill.fill(text, rows)
    out = OUT / outname
    info = build_docx.build(filled, out, REPO)
    pdf = None
    if want_pdf:
        pdf = to_pdf(out, OUT / "pdf")
        if pdf is not None:
            labels = page_labels(pdf, info)
            info = build_docx.build(filled, out, REPO, pages=labels)
            pdf = to_pdf(out, OUT / "pdf")
    return {"name": name, "errors": errs, "fields": fields, "built": out, "pdf": pdf, "info": info}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fill only; report missing numbers and [FILL IN] fields")
    ap.add_argument("--pdf", action="store_true", help="also convert to PDF with LibreOffice (if installed)")
    ap.add_argument("--only", choices=sorted(DOCS))
    a = ap.parse_args(argv)
    total_missing, all_fields = 0, {}
    for name in ([a.only] if a.only else list(DOCS)):
        r = process(name, a.check, a.pdf)
        if r is None:
            print(f"{name}: no source yet")
            continue
        for e in r["errors"]:
            print(f"MISSING in {name}: {e}")
        total_missing += len(r["errors"])
        for f in r["fields"]:
            all_fields.setdefault(f, []).append(name)
        if r["built"]:
            print(f"built {r['built']}" + (f" and {r['pdf']}" if r.get("pdf") else ""))
    print(f"{total_missing} missing numbers")
    print(f"{len(all_fields)} distinct [FILL IN] fields remain:")
    for f, where in sorted(all_fields.items()):
        print(f"  {f}  ({', '.join(where)})")
    return 1 if total_missing else 0


if __name__ == "__main__":
    sys.exit(main())
