"""research/check_links.py: relative Markdown links and images must point at files that exist."""

from pathlib import Path

import check_links


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf8")
    return path


def test_one_good_and_one_broken_link(tmp_path):
    _write(tmp_path / "docs" / "other.md", "# other\n")
    _write(tmp_path / "README.md", "See [other](docs/other.md) and [gone](docs/missing.md).\n")
    broken = check_links.find_broken(tmp_path)
    assert [(p.name, line, t) for p, line, t in broken] == [("README.md", 1, "docs/missing.md")]


def test_images_anchors_external_and_code_are_handled(tmp_path):
    _write(tmp_path / "docs" / "img" / "a.png", "x")
    _write(tmp_path / "docs" / "other.md", "# other\n")
    text = (
        "![ok](img/a.png) ![bad](img/b.png)\n"
        "[anchor only](#top) [web](https://example.com/x.md) [mail](mailto:a@b.c)\n"
        "[file with anchor](other.md#section) [spaces](<other.md>)\n"
        "`[in code](nothing.md)`\n"
        "```\n[in fence](nothing2.md)\n```\n"
    )
    _write(tmp_path / "docs" / "page.md", text)
    broken = check_links.find_broken(tmp_path)
    assert [t for _, _, t in broken] == ["img/b.png"]


def test_excluded_folder_is_not_checked(tmp_path):
    _write(tmp_path / "docs" / "DOCS NEEDED FOR MAJOR" / "x.md", "[bad](nope.md)\n")
    assert check_links.find_broken(tmp_path) == []


def test_main_prints_count_and_exit_code(tmp_path, capsys):
    _write(tmp_path / "README.md", "[gone](missing.md)\n")
    assert check_links.main(["--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "README.md:1: missing.md" in out
    assert out.strip().endswith("1 broken links")
    _write(tmp_path / "README.md", "fine\n")
    assert check_links.main(["--root", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "0 broken links"
