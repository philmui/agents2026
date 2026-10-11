"""Check the local slide deck in a real browser; save review images to temp.

Requires Playwright plus local Chrome, or Playwright Chromium.
"""
from pathlib import Path
import json
import tempfile
import shutil
import subprocess
import re
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
CHROME = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
OUT = Path(tempfile.mkdtemp(prefix='hotpotqa-slides-'))

def main():
    errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=str(CHROME) if CHROME.exists() else None)
        page=browser.new_page(viewport={'width':1500,'height':1000},device_scale_factor=1)
        page.on('pageerror',lambda exc:errors.append(str(exc)))
        page.goto((ROOT/'index.html').as_uri())
        ids=page.locator('.slide').evaluate_all('(els)=>els.map(e=>e.id)')
        assert len(ids)==34
        for width,height in [(1500,1000),(1024,900),(390,844)]:
            page.set_viewport_size({'width':width,'height':height})
            for i,slide_id in enumerate(ids):
                page.select_option('#contents',str(i))
                page.wait_for_function('Array.from(document.images).every(im=>im.complete)')
                checks=page.evaluate('''() => {
                    const slide=document.querySelector('.slide.active');
                    const box=slide.getBoundingClientRect();
                    const bad=[];
                    for(const e of slide.querySelectorAll('h1,h2,p,pre,table,img,li')){
                        if(e.closest('details')) continue;
                        const r=e.getBoundingClientRect();
                        if(r.left<box.left-2 || r.right>box.right+2) bad.push(e.tagName+':'+e.textContent.slice(0,80));
                    }
                    return {bad,docWidth:document.documentElement.scrollWidth,viewport:innerWidth,
                      images:[...slide.querySelectorAll('img')].every(im=>im.naturalWidth>0),
                      body:slide.scrollHeight,height:slide.clientHeight};
                }''')
                assert not checks['bad'],(width,slide_id,checks)
                assert checks['docWidth']<=width,(width,slide_id,checks)
                assert checks['images'],(width,slide_id,'missing image')
                if width==1500:
                    page.locator('.slide.active').screenshot(path=str(OUT/f'{i+1:02d}-{slide_id}.png'))
            if width==390:
                page.select_option('#contents','6')
                page.screenshot(path=str(OUT/'mobile-json.png'),full_page=True)
        page.set_viewport_size({'width':1500,'height':1000})
        page.select_option('#contents','0')
        page.locator('h1').click()
        page.keyboard.press('ArrowRight')
        assert page.locator('.slide.active').get_attribute('id')==ids[1]
        page.locator('#explain').click()
        assert page.locator('.slide.active details').get_attribute('open') is not None
        page.locator('#read').click()
        assert page.locator('.slide:visible').count()==34
        page.locator('#read').click()
        assert page.locator('.slide:visible').count()==1
        page.goto((ROOT/'index.html').as_uri()+'#actual-json')
        assert page.locator('.slide.active').get_attribute('id')=='actual-json'
        page.pdf(path=str(OUT/'deck.pdf'),prefer_css_page_size=True,print_background=True)
        if shutil.which('pdfinfo'):
            info=subprocess.check_output(['pdfinfo',str(OUT/'deck.pdf')],text=True)
            assert int(re.search(r'Pages:\s+(\d+)',info).group(1)) == len(ids),info
        nojs=browser.new_context(java_script_enabled=False,viewport={'width':1024,'height':900}).new_page()
        nojs.goto((ROOT/'index.html').as_uri())
        assert nojs.locator('.slide:visible').count()==34
        assert nojs.locator('details').count()==34
        assert not errors,errors
        browser.close()
    print(json.dumps({'slides':34,'viewport_checks':102,'javascript_errors':errors,'artifacts':str(OUT)},indent=2))

if __name__=='__main__':main()
