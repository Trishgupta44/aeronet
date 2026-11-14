"""Check that every relative link and image target in the repo's Markdown files exists.

    research/.venv/Scripts/python research/check_links.py [--root <repo folder>]

Files checked: ``README.md``, ``app/README.md`` and every ``docs/**/*.md`` except ``docs/DOCS NEEDED FOR MAJOR/``
(institute documents, not committed). Links that start with ``http://``, ``https://``, ``mailto:`` or ``#`` are
ignored, and so is anything inside a code fence or an inline code span. A ``#anchor`` or ``?query`` after a path is
dropped before the file is looked up (the anchor itself is not checked). Each broken link is printed as
``file:line: target``, and the last line is ``N broken links``. Exit code 0 when there are none, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

EXCLUDED_DIRS = ("DOCS NEEDED FOR MAJOR",)
# [text](target) and ![alt](target); the target may be <wrapped> and may carry a "title"
LINK = re.compile(r"!?\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)")
INLINE_CODE = re.compile(r"`[^`]*`")
FENCE = re.compile(r"^\s*(```|~~~)")
EXTERNAL = ("http://", "https://", "mailto:", "ftp://", "tel:", "data:")


def markdown_files(root: Path) -> list[Path]:
    """The Markdown files to check, sorted."""
    files = [root / "README.md", root / "app" / "README.md"]
    docs = root / "docs"
    if docs.is_dir():
        files += [p for p in docs.rglob("*.md") if not any(d in p.relative_to(root).parts for d in EXCLUDED_DIRS)]
    return sorted(p for p in set(files) if p.is_file())


def links_in(text: str):
    """Yield ``(line_number, target)`` for every link and image in a Markdown text, skipping code."""
    in_fence = False
    for n, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in LINK.finditer(INLINE_CODE.sub("", line)):
            yield n, m.group(1).strip("<>")


# The document builders (research/docs_build) read these sources and resolve image paths from the repository root,
# so a path such as ``docs/milestones/img/x.png`` is correct there even though it is not relative to the file.
ROOT_RELATIVE_DIRS = ("docs/deliverables_src",)


def target_exists(md_file: Path, target: str, root: Path | None = None) -> bool | None:
    """True or False for a relative target, None when it is not a file link (external or an anchor on the same page).

    Files under ``ROOT_RELATIVE_DIRS`` may also name their targets relative to the repository ``root``.
    """
    if not target or target.startswith("#") or target.lower().startswith(EXTERNAL):
        return None
    path = unquote(re.split(r"[#?]", target, maxsplit=1)[0])
    if not path:
        return None
    if (md_file.parent / path).exists():
        return True
    if root is not None:
        rel = md_file.resolve().relative_to(root.resolve()).as_posix() if md_file.resolve().is_relative_to(root.resolve()) else ""
        if any(rel.startswith(d + "/") for d in ROOT_RELATIVE_DIRS):
            return (root / path).exists()
    return False


def find_broken(root: Path) -> list[tuple[Path, int, str]]:
    """``[(markdown file, line, target)]`` for every relative link whose target does not exist."""
    broken = []
    for md in markdown_files(root):
        for line, target in links_in(md.read_text(encoding="utf8")):
            if target_exists(md, target, root) is False:
                broken.append((md, line, target))
    return broken


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent, help="repo folder (default: this repo)")
    root = ap.parse_args(argv).root.resolve()
    broken = find_broken(root)
    for md, line, target in broken:
        print(f"{md.relative_to(root).as_posix()}:{line}: {target}")
    print(f"{len(broken)} broken links")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
