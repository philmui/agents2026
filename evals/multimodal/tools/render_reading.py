"""Create ignored reading previews and check all displayed images in a browser."""
from __future__ import annotations

import html
import json
from pathlib import Path

import mistune
import nbformat
from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright
from browser_support import launch_browser

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "reading_review"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    link = OUT / "assets"
    if not link.exists():
        link.symlink_to(ROOT / "assets", target_is_directory=True)
    paths = []
    markdown = mistune.create_markdown(plugins=["table"])
    for path in [ROOT/"README.md", ROOT/"THEORY.md", ROOT/"FEATURE_STORES.md", ROOT/"SOURCE_MAP.md"]:
        body = markdown(path.read_text())
        output = OUT / f"{path.stem}.html"
        output.write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>{html.escape(path.stem)}</title><style>
body{{font:18px/1.65 Inter,Arial,sans-serif;color:#202727;max-width:1050px;margin:50px auto;padding:0 32px}}
img{{max-width:100%;height:auto}}h1{{line-height:1.2;font-size:42px}}h2{{margin-top:2.5em;line-height:1.3}}
pre{{background:#f3f6f4;padding:20px;overflow:auto;font-size:15px;line-height:1.5}}
table{{border-collapse:collapse;width:100%;font-size:16px}}th,td{{border:1px solid #d5ded9;padding:12px;text-align:left}}
a{{color:#08796f}}blockquote{{border-left:4px solid #08796f;padding-left:20px}}
</style><main>{body}</main></html>""")
        paths.append(output)
    exporter = HTMLExporter(template_name="lab")
    for path in sorted(ROOT.glob("[0-9][0-9]_*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        body, _ = exporter.from_notebook_node(notebook, resources={"metadata":{"path":str(ROOT)}})
        output = OUT/f"{path.stem}.html"
        output.write_text(body)
        paths.append(output)
    reports = []
    with sync_playwright() as p:
        browser = launch_browser(p)
        print("Review browser:", browser.version)
        page = browser.new_page(viewport={"width":1440,"height":1000},device_scale_factor=1)
        # Preview local content without waiting for CDN assets such as MathJax.
        # JupyterLab itself supplies its normal math renderer when studying.
        page.route("https://**/*", lambda route: route.abort())
        page.route("http://**/*", lambda route: route.abort())
        for path in paths:
            page.goto(path.as_uri(),wait_until="load")
            info = page.evaluate("""() => ({
                images:document.images.length,
                broken:[...document.images].filter(x=>!x.complete||x.naturalWidth===0).map(x=>x.src),
                pageOverflow:document.documentElement.scrollWidth>innerWidth+2,
                height:document.documentElement.scrollHeight
            })""")
            page.screenshot(path=str(OUT/f"{path.stem}_top.png"))
            page.evaluate("window.scrollTo(0,document.documentElement.scrollHeight/2)")
            page.screenshot(path=str(OUT/f"{path.stem}_middle.png"))
            reports.append({"document":path.name,**info})
        browser.close()
    (OUT/"report.json").write_text(json.dumps(reports,indent=2)+"\n")
    print(json.dumps(reports,indent=2))
    if any(r["broken"] or r["pageOverflow"] for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
