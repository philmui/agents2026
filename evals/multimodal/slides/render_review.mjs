// Optional rendering review: node slides/render_review.mjs [path/to/playwright]
// The lesson itself needs neither Node nor Playwright.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {chromium}=require(process.argv[2] || 'playwright');
const base=path.dirname(fileURLToPath(import.meta.url));
const out=path.join(base,'..','assets','slide_previews');
fs.mkdirSync(out,{recursive:true});
const browser=await chromium.launch({headless:true,...(process.argv[3]?{executablePath:process.argv[3]}:{})});
const page=await browser.newPage({viewport:{width:1600,height:900},deviceScaleFactor:1});
const report=[];
for(const name of fs.readdirSync(path.join(base,'outputs','html')).filter(n=>n.endsWith('.html')).sort()){
  await page.goto(pathToFileURL(path.join(base,'outputs','html',name)).href);
  await page.screenshot({path:path.join(out,name.replace('.html','.png'))});
  const inspection=await page.evaluate(()=>{
    const frame=document.querySelector('.main-frame').getBoundingClientRect();
    const out=[];
    const textNodes=[...document.querySelectorAll('.main-frame h1,.main-frame p,.main-frame pre,.main-frame svg text')];
    const rects=textNodes.map(e=>({e,r:e.getBoundingClientRect(),t:e.textContent.trim()}));
    for(const {e,r,t} of rects){
      if(r.left<frame.left-1||r.top<frame.top-1||r.right>frame.right+1||r.bottom>frame.bottom+1)out.push({kind:'outside-frame',text:t,rect:{x:r.x,y:r.y,width:r.width,height:r.height}});
      if(e.tagName==='PRE'&&e.scrollWidth>e.clientWidth+1)out.push({kind:'code-overflow',text:t});
      if(e.tagName==='text'){
        const size=parseFloat(getComputedStyle(e).fontSize), scale=e.getScreenCTM().a;
        if(size*scale<16)out.push({kind:'small-svg-label',text:t,pixels:size*scale});
      }
    }
    for(let i=0;i<rects.length;i++)for(let j=i+1;j<rects.length;j++){
      const a=rects[i],b=rects[j];
      if(a.e.contains(b.e)||b.e.contains(a.e))continue;
      const dx=Math.min(a.r.right,b.r.right)-Math.max(a.r.left,b.r.left);
      const dy=Math.min(a.r.bottom,b.r.bottom)-Math.max(a.r.top,b.r.top);
      if(dx>2&&dy>2)out.push({kind:'text-overlap',a:a.t,b:b.t,width:dx,height:dy});
    }
    return out;
  });
  report.push({slide:name,issues:inspection});
}
await page.goto(pathToFileURL(path.join(base,'..','slides.html')).href);
await page.getByRole('button',{name:'Next slide',exact:true}).click();
if(await page.locator('#counter').textContent()!=='2 / 42')throw Error('Next control failed');
await page.keyboard.press('End');
if(await page.locator('#counter').textContent()!=='42 / 42')throw Error('End shortcut failed');
await page.keyboard.press('Home');
await page.locator('#stage').click({position:{x:500,y:250}});
await page.keyboard.press('ArrowRight');
if(await page.locator('#counter').textContent()!=='2 / 42')throw Error('Navigation after clicking slide failed');
await page.keyboard.press('ArrowLeft');
await page.getByRole('button',{name:'Speaker notes',exact:true}).click();
if(!await page.locator('#notes').isVisible())throw Error('Notes control failed');
await page.screenshot({path:path.join(out,'shell-notes.png')});
await page.setViewportSize({width:390,height:844});
if(await page.evaluate(()=>document.documentElement.scrollWidth>390))throw Error('Mobile shell overflows horizontally');
const nojs=await browser.newPage({javaScriptEnabled:false});
await nojs.goto(pathToFileURL(path.join(base,'..','slides.html')).href);
if(!await nojs.getByRole('link',{name:'Read all slides and speaker notes in the static index.'}).isVisible())throw Error('No-JavaScript fallback missing');
await nojs.close();
await page.setViewportSize({width:1600,height:1900});
for(let batch=0;batch<3;batch++){
  const cards=Array.from({length:14},(_,j)=>{
    const n=String(batch*14+j+1).padStart(2,'0');
    return `<article><p>${n}</p><img src="${pathToFileURL(path.join(out,`slide_${n}.png`)).href}"></article>`;
  });
  await page.setContent(`<style>body{margin:10px;display:grid;grid-template-columns:1fr 1fr;gap:10px;background:#edf2ef;font:16px Arial}p{margin:0}article{height:255px}img{display:block;height:235px;width:100%;object-fit:contain;object-position:left top}</style>${cards.join('')}`);
  await page.locator('img').last().evaluate(img=>img.decode());
  await page.screenshot({path:path.join(out,`contact_${batch+1}.png`)});
}
await browser.close();
fs.writeFileSync(path.join(base,'render_review.json'),JSON.stringify({slides:report,shell:'Navigation, Home/End, post-click arrows, notes controls, mobile width, and no-JavaScript fallback passed'},null,2)+'\n');
const issues=report.flatMap(s=>s.issues.map(i=>({slide:s.slide,...i})));
console.log(JSON.stringify({rendered:report.length,issues},null,2));
if(issues.length)process.exitCode=1;
