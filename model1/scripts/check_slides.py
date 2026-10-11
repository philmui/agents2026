"""Browser regression checks for the authored deck (never calls a model API).

Install scripts/requirements-review.txt, then run this script. Use --browser
with an installed Chrome binary, or install Chromium with Playwright. Optional
--screenshots writes 52 slides and responsive/interface previews for human review.
--pdf regenerates and verifies the distributed PDF after a slide revision.
"""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / 'docs/slides/index.html'

GEOMETRY = r'''() => {
  const slide=deck.slides[deck.currentSlide], bad=[];
  const stage=deck.stage.getBoundingClientRect(), controls=document.querySelector('.deck-controls').getBoundingClientRect();
  if(Math.abs(stage.width/stage.height-16/9)>.001)bad.push('stage is not 16:9');
  if(stage.left < -.5 || stage.top < -.5 || stage.right>innerWidth+.5 || stage.bottom>controls.top+.5)bad.push('stage intrudes into viewport or navigation');
  if(document.querySelectorAll('#deckStage > .slide.active').length!==1)bad.push('active slide count');
  const heading=slide.querySelector('h1,h2').getBoundingClientRect(),lead=slide.querySelector('.lead').getBoundingClientRect();
  if(heading.bottom>lead.top+1)bad.push('heading overlaps explanation');
  if(lead.bottom>slide.querySelector('.visual').getBoundingClientRect().top+1)bad.push('explanation overlaps diagram');
  const figure=slide.querySelector('.visual').getBoundingClientRect(),takeaway=slide.querySelector('.takeaway').getBoundingClientRect();
  if(figure.bottom>takeaway.top+1)bad.push('diagram overlaps takeaway');
  if(takeaway.bottom>slide.querySelector('.credit').getBoundingClientRect().top+1)bad.push('takeaway overlaps attribution');
  const svg=slide.querySelector('svg'),v=svg.viewBox.baseVal;
  const texts=[...svg.querySelectorAll('text')].map(e=>({text:e.textContent,b:e.getBBox()}));
  for(const t of texts){const b=t.b;if(b.x< -1||b.y< -1||b.x+b.width>v.width+1||b.y+b.height>v.height+1)bad.push('SVG text outside viewBox: '+t.text);}
  for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){
    const a=texts[i].b,b=texts[j].b;
    if(Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x)>2&&Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y)>2)bad.push('SVG text collision: '+texts[i].text+' / '+texts[j].text);
  }
  for(const node of svg.querySelectorAll('.diagram-node')){
    const r=node.querySelector('rect').getBBox();
    for(const text of node.querySelectorAll('text')){const b=text.getBBox();if(b.x<r.x-2||b.x+b.width>r.x+r.width+2||b.y<r.y-2||b.y+b.height>r.y+r.height+2)bad.push('Text outside node: '+text.textContent);}
  }
  for(const el of document.querySelectorAll('.deck-controls button,.deck-controls select')){
    const b=el.getBoundingClientRect();if(!b.width||!b.height)continue;
    if(b.width<44||b.height<44)bad.push('navigation target smaller than 44 pixels');
    if(b.left<0||b.right>innerWidth||b.bottom>innerHeight)bad.push('navigation outside viewport');
  }
  if(slide.querySelector('.credit').textContent!=='@mui-group')bad.push('missing attribution');
  return bad;
}'''


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--browser',help='Path to an installed Chrome/Chromium executable')
    ap.add_argument('--screenshots',type=Path)
    ap.add_argument('--pdf',action='store_true')
    args=ap.parse_args()
    screenshots=args.screenshots
    if screenshots:screenshots.mkdir(parents=True,exist_ok=True)
    failures=[]; errors=[]; network=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,**({'executable_path':args.browser} if args.browser else {}))
        browser_version=browser.version
        context=browser.new_context(viewport={'width':1280,'height':720},reduced_motion='reduce',accept_downloads=True)
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:network.append(r.url) if r.url.startswith(('http:','https:')) else None)
        page.goto(DECK.as_uri());page.evaluate('document.fonts.ready');page.wait_for_timeout(300)
        count=page.evaluate('deck.slides.length');assert count==52
        assert page.evaluate("new Set([...document.querySelectorAll('[id]')].map(e=>e.id)).size === document.querySelectorAll('[id]').length"),'duplicate DOM IDs'
        for i in range(count):
            page.evaluate('(i)=>deck.showSlide(i)',i);page.wait_for_timeout(30)
            problems=page.evaluate(GEOMETRY)
            if problems:failures.append({'viewport':'1280x720','slide':i+1,'problems':problems})
            if screenshots:page.screenshot(path=str(screenshots/f'slide-{i+1:02d}.png'))
        for width,height in [(390,844),(844,390),(1920,1080)]:
            page.set_viewport_size({'width':width,'height':height})
            for i in range(count):
                page.evaluate('(i)=>deck.showSlide(i)',i)
                problems=page.evaluate(GEOMETRY)
                if problems:failures.append({'viewport':f'{width}x{height}','slide':i+1,'problems':problems})
            page.evaluate('deck.showSlide(23)')
            if screenshots:page.screenshot(path=str(screenshots/f'viewport-{width}x{height}.png'))
        if screenshots:(screenshots/'geometry.json').write_text(json.dumps(failures,indent=2)+'\n')
        page.set_viewport_size({'width':1280,'height':720})
        # Corrupt/deep hashes used to make every slide disappear.
        for fragment,expected in [('invalid',0),('0',0),('-3',0),('999',51),('24',23)]:
            page.goto(DECK.as_uri()+'#'+fragment);page.wait_for_timeout(60)
            assert page.evaluate('deck.currentSlide')==expected,(fragment,expected)
        page.evaluate('deck.showSlide(0)');page.locator('body').click(position={'x':10,'y':300})
        page.keyboard.press('ArrowRight');assert page.evaluate('deck.currentSlide')==1
        page.keyboard.press('End');assert page.evaluate('deck.currentSlide')==51
        page.keyboard.press('Home');assert page.evaluate('deck.currentSlide')==0
        page.select_option('#chapters',label='03 · Use context');assert page.evaluate('deck.currentSlide')==22
        page.locator('#counter').click();page.fill('#slideSearch','confidence')
        assert page.locator('.slide-card:visible').count()>=1
        page.locator('.slide-card:visible').first.click();assert page.evaluate('deck.currentSlide')==35
        # Native dialog focus containment and empty-search feedback.
        page.locator('#counter').click();page.fill('#slideSearch','no-such-topic')
        assert page.locator('.slide-card:visible').count()==0
        assert 'No matching slides' in page.locator('#overviewCount').inner_text()
        page.keyboard.press('Escape');page.locator('#overview').wait_for(state='hidden')
        page.locator('#notesToggle').focus();page.keyboard.press('Space')
        assert page.locator('#notes').is_visible();assert page.evaluate(GEOMETRY)==[]
        if screenshots:page.screenshot(path=str(screenshots/'teaching-notes.png'))
        page.locator('#closeNotes').click();assert not page.locator('#notes').is_visible()
        # Edits persist, export as a standalone file, and can be reset.
        page.evaluate('deck.toggleEditing(true)');heading=page.locator('#deckStage .active [data-edit=title]')
        original=heading.inner_text();heading.fill('A saved classroom edit')
        page.keyboard.press('Escape');page.reload();assert page.locator('#deckStage .active [data-edit=title]').inner_text()=='A saved classroom edit'
        with page.expect_download() as d:page.evaluate('deck.export()')
        exported=Path('/tmp/system-one-export-check.html');d.value.save_as(exported)
        copy=context.new_page();copy.goto(exported.as_uri());copy.wait_for_timeout(100)
        copy.evaluate('deck.showSlide(35)');assert copy.locator('#deckStage .active [data-edit=title]').inner_text()=='A saved classroom edit';copy.close();exported.unlink()
        page.locator('#more').click();page.locator('#resetEdits').click();page.keyboard.press('Escape')
        assert page.locator('#deckStage .active [data-edit=title]').inner_text()==original
        # Current and next previews preserve aspect; messages synchronize both directions.
        with page.expect_popup() as popup:page.evaluate('deck.openPresenter()')
        presenter=popup.value;presenter.on('pageerror',lambda e:errors.append(str(e)));presenter.wait_for_timeout(400)
        assert presenter.locator('.presenter-shell').is_visible()
        presenter.evaluate('deck.showSlide(17)');page.wait_for_timeout(100);assert page.evaluate('deck.currentSlide')==17
        page.evaluate('deck.showSlide(23)');presenter.wait_for_timeout(100);assert presenter.evaluate('deck.currentSlide')==23
        ratio=presenter.locator('.deck-viewport').bounding_box();assert abs(ratio['width']/ratio['height']-16/9)<.03
        for size in [(1440,900),(1280,720)]:
            presenter.set_viewport_size({'width':size[0],'height':size[1]});presenter.wait_for_timeout(60)
            current=presenter.locator('.deck-viewport').bounding_box();upcoming=presenter.locator('.next-preview').bounding_box()
            toolbar=presenter.locator('.deck-controls').bounding_box()
            assert abs(current['width']/current['height']-16/9)<.03
            assert abs(upcoming['width']/upcoming['height']-16/9)<.03
            assert upcoming['y']+upcoming['height']<toolbar['y'],'next preview clipped by navigation'
        presenter.set_viewport_size({'width':1440,'height':900})
        if screenshots:presenter.screenshot(path=str(screenshots/'presenter.png'))
        presenter.evaluate('deck.showSlide(51)');assert 'End of lesson' in presenter.locator('.next-title').inner_text()
        presenter.close()
        page.locator('#counter').click()
        if screenshots:page.screenshot(path=str(screenshots/'overview.png'))
        page.keyboard.press('Escape')
        # Default motion must reach the same readable final state.
        page.emulate_media(reduced_motion='no-preference');page.evaluate('deck.showSlide(17)');page.wait_for_timeout(1100)
        assert page.evaluate("getComputedStyle(deck.slides[17].querySelector('.visual')).opacity")=='1'
        if args.pdf:
            page.locator('#notes').evaluate('(e)=>e.hidden=true');page.evaluate('deck.fit()')
            target=ROOT/'docs/slides/from-bag-of-words-to-jev.pdf'
            page.pdf(path=str(target),print_background=True,prefer_css_page_size=True)
            from pypdf import PdfReader
            pdf=PdfReader(target);assert len(pdf.pages)==count
            assert all('@mui-group' in p.extract_text() for p in pdf.pages)
        browser.close()
    result={'browser':browser_version,'slides':count,'viewport_checks':count*4,'geometry_failures':failures,'browser_errors':errors,'remote_requests':network,'interaction_checks':'passed'}
    if screenshots:(screenshots/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert not failures and not errors and not network,'Slide review found failures; inspect the report.'


if __name__=='__main__':main()
