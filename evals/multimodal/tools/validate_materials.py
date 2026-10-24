"""Read-only structural checks for the delivered teaching artifacts."""
from __future__ import annotations

import ast
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlsplit

import nbformat
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def local_links(text: str, parent: Path, label: str, errors: list[str]) -> int:
    # Code snippets can contain illustrative paths: only prose links count.
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    checked = 0
    for destination in links:
        destination = destination.strip().split(' "')[0].strip("<>")
        url = urlsplit(destination)
        if url.scheme or not url.path:
            continue
        path = parent / unquote(url.path)
        if not path.exists():
            errors.append(f"{label}: broken link {destination}")
        checked += 1
    return checked


def main() -> None:
    errors: list[str] = []
    counts = {"notebooks": 0, "code_cells": 0, "markdown_files": 0,
              "svg_files": 0, "slide_files": 0, "local_links": 0}
    for path in sorted(ROOT.glob("[0-9][0-9]_*.ipynb")):
        counts["notebooks"] += 1
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for index, cell in enumerate(nb.cells):
            label = f"{path.name} cell {index}"
            if cell.cell_type == "code":
                counts["code_cells"] += 1
                ast.parse(cell.source)
                if cell.source.strip() and cell.execution_count is None:
                    errors.append(f"{label}: missing saved execution")
                if any(o.output_type == "error" for o in cell.get("outputs", [])):
                    errors.append(f"{label}: saved exception")
            else:
                counts["local_links"] += local_links(cell.source, path.parent, label, errors)
    for path in ROOT.rglob("*.md"):
        if any(part.startswith(".") or part == "artifacts" for part in path.relative_to(ROOT).parts):
            continue
        counts["markdown_files"] += 1
        counts["local_links"] += local_links(path.read_text(), path.parent, str(path.relative_to(ROOT)), errors)
    ns = {"s": "http://www.w3.org/2000/svg"}
    for path in (ROOT / "assets").glob("*.svg"):
        counts["svg_files"] += 1
        root = ET.parse(path).getroot()
        if not root.get("viewBox"):
            errors.append(f"{path.name}: missing viewBox")
        for tag in ["title", "desc"]:
            node = root.find(f"s:{tag}", ns)
            if node is None or not node.text:
                errors.append(f"{path.name}: missing accessible {tag}")
    for path in (ROOT / "slides" / "outputs" / "html").glob("slide_*.html"):
        counts["slide_files"] += 1
        soup = BeautifulSoup(path.read_text(), "html.parser")
        if "© mui-group" not in soup.get_text():
            errors.append(f"{path.name}: missing attribution")
        if soup.find("script"):
            errors.append(f"{path.name}: standalone slides must be static")
    if counts["notebooks"] != 5:
        errors.append(f"Expected 5 notebooks, found {counts['notebooks']}")
    if counts["svg_files"] < 12 or counts["slide_files"] < 35:
        errors.append("Missing expected illustration or slide set")
    print(json.dumps({"counts": counts, "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
