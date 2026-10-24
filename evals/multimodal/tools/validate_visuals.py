"""Render every standalone slide and SVG; check text bounds and save review sheets.

This is a geometric check, not a claim that a browser can judge pedagogy.
The contact sheets support the separate human/agent visual review.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from browser_support import launch_browser

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "visual_review"

BOUNDS = """() => {
  const svg = document.documentElement.tagName.toLowerCase() === 'svg';
  const main = svg ? document.documentElement : document.querySelector('.main-frame');
  if (!main) return {issues:['Missing main-frame'],textBoxes:0};
  const bounds = main.getBoundingClientRect();
  const issues = [];
  const texts = [];
  const walker = document.createTreeWalker(main,NodeFilter.SHOW_TEXT);
  let node;
  while (node=walker.nextNode()) {
    if (!node.textContent.trim()) continue;
    const parent = node.parentElement;
    if (['STYLE','SCRIPT','TITLE','DESC'].includes(parent.tagName.toUpperCase())) continue;
    const style = getComputedStyle(parent);
    if (style.display==='none' || style.visibility==='hidden') continue;
    const range=document.createRange();range.selectNodeContents(node);
    const rects=Array.from(range.getClientRects());
    for (const r of rects) {
      if (!r.width || !r.height) continue;
      const text=node.textContent.trim().slice(0,95);
      if (r.left<bounds.left-2 || r.right>bounds.right+2 || r.top<bounds.top-2 || r.bottom>bounds.bottom+2)
        issues.push('Text outside content frame: '+text);
      texts.push({text,left:r.left,top:r.top,right:r.right,bottom:r.bottom});
    }
  }
  if (svg) {
    // SVG <text> gives one semantic label, unlike wrapped HTML ranges.
    const labels=Array.from(document.querySelectorAll('text')).map(e=>({e,r:e.getBoundingClientRect()}));
    for (let i=0;i<labels.length;i++) for(let j=i+1;j<labels.length;j++) {
      const a=labels[i],b=labels[j];
      if(a.e.contains(b.e)||b.e.contains(a.e))continue;
      const w=Math.min(a.r.right,b.r.right)-Math.max(a.r.left,b.r.left);
      const h=Math.min(a.r.bottom,b.r.bottom)-Math.max(a.r.top,b.r.top);
      if(w>3&&h>3)issues.push('Overlapping labels: '+a.e.textContent+' / '+b.e.textContent);
    }
    // Lines can cross letters even when no two labels overlap. Sample only
    // visible open paths, excluding marker definitions and filled artwork.
    for (const path of document.querySelectorAll('path[stroke]')) {
      if(path.closest('defs') || getComputedStyle(path).fill!=='none')continue;
      const length=path.getTotalLength(),matrix=path.getScreenCTM();
      for (const label of labels) {
        let crosses=false;
        for(let d=0;d<=length;d+=3) {
          const point=path.getPointAtLength(d);
          const q=new DOMPoint(point.x,point.y).matrixTransform(matrix);
          const r=label.r;
          if(q.x>r.left+1&&q.x<r.right-1&&q.y>r.top+1&&q.y<r.bottom-1){crosses=true;break;}
        }
        if(crosses)issues.push('Line crosses label: '+label.e.textContent);
      }
    }
  }
  return {issues:[...new Set(issues)],textBoxes:texts.length};
}"""


def sheet(files: list[Path], name: str, columns=4, thumb_width=400):
    if not files:
        return
    height = 258
    result = Image.new("RGB", (columns * thumb_width, math.ceil(len(files)/columns)*height), "#e7e7e3")
    draw = ImageDraw.Draw(result)
    for index, path in enumerate(files):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_width-12, height-30))
        x, y = index % columns*thumb_width+6, index//columns*height+6
        result.paste(image, (x,y))
        draw.text((x,y+height-24), path.stem, fill="#222222")
    result.save(OUT / name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report, screenshots = [], {"slides": [], "figures": []}
    with sync_playwright() as p:
        browser = launch_browser(p)
        print("Review browser:", browser.version)
        page = browser.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1)
        paths = [(path,"slides") for path in sorted((ROOT/"slides"/"outputs"/"html").glob("slide_*.html"))]
        paths += [(path,"figures") for path in sorted((ROOT/"assets").glob("*.svg"))]
        if not paths:
            raise SystemExit("No visual artifacts found")
        for path, kind in paths:
            page.set_viewport_size({"width":1600,"height":900} if kind=="slides" else {"width":1200,"height":620})
            page.goto(path.as_uri(), wait_until="load")
            page.evaluate("document.fonts.ready")
            check = page.evaluate(BOUNDS)
            png = OUT / (path.stem+".png")
            page.screenshot(path=str(png))
            screenshots[kind].append(png)
            report.append({"path":str(path.relative_to(ROOT)), **check})
        browser.close()
    sheet(screenshots["slides"], "slides_contact_sheet.jpg")
    sheet(screenshots["figures"], "figures_contact_sheet.jpg", columns=3)
    errors = [item for item in report if item["issues"]]
    (OUT/"geometry_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"artifacts_rendered":len(report),"issues":errors,"output":str(OUT)},indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
