"""Render notebook figures and the exported HTML for local visual inspection."""
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'artifacts'/'notebook-visual'; out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    chrome=Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    browser=p.chromium.launch(**({'executable_path': str(chrome)} if chrome.exists() else {}))
    page=browser.new_page(viewport={'width':1200,'height':900},device_scale_factor=1)
    for name in ['bridge','evaluation','graph']:
        page.goto((ROOT/'assets'/f'{name}.svg').as_uri())
        page.screenshot(path=str(out/f'{name}.png'))
    page.goto((ROOT/'artifacts'/'notebook-preview.html').as_uri())
    page.screenshot(path=str(out/'notebook-top.png'))
    assert page.locator('h1').inner_text().startswith('Evaluating a LangGraph')
    assert page.locator('.jp-OutputArea-output').count() >= 20
    browser.close()
print(out)
