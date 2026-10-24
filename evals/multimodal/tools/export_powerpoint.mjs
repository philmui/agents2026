/**
 * Export the existing HTML lesson as editable PowerPoint objects.
 * Usage: node tools/export_powerpoint.mjs /path/to/node_modules [output.pptx]
 * Dependencies: playwright and pptxgenjs. See slides/README.md.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';

const require = createRequire(import.meta.url);
const packages = process.argv[2];
const {chromium} = require(packages ? path.resolve(packages, 'playwright') : 'playwright');
const PptxGenJS = require(packages ? path.resolve(packages, 'pptxgenjs') : 'pptxgenjs');
const JSZip = require(packages ? path.resolve(packages, 'jszip') : 'jszip');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.resolve(process.argv[3] || path.join(root, 'slides/outputs/pptx/multimodal-retrieval.pptx'));
const privateDir = path.join(root, '.pptx-build');
const manifest = JSON.parse(await fs.readFile(path.join(root, 'slides/manifest.json'), 'utf8'));
const pxPerInch = 120;
const inch = n => n / pxPerInch;
const pt = n => n * 72 / pxPerInch;
const color = s => {
  if (!s || s === 'none' || s === 'transparent') return null;
  if (s.startsWith('#')) return s.slice(1);
  const v = s.match(/[\d.]+/g);
  if (!v || (v.length === 4 && +v[3] === 0)) return null;
  return v.slice(0, 3).map(n => (+n).toString(16).padStart(2, '0')).join('').toUpperCase();
};

await fs.mkdir(path.dirname(output), {recursive:true});
await fs.mkdir(privateDir, {recursive:true});
const browser = await chromium.launch({headless:true,
  ...(process.env.CHROMIUM_EXECUTABLE ? {executablePath:process.env.CHROMIUM_EXECUTABLE} : {})});
const page = await browser.newPage({viewport:{width:1600,height:900},deviceScaleFactor:1});
const scenes = [];
try {
  for (const entry of manifest) {
    await page.goto(pathToFileURL(path.join(root, 'slides', entry.file)).href);
    await page.evaluate(() => document.fonts.ready);
    const objects = await page.evaluate(() => {
      const result = [];
      function shape(e) {
        const r = e.getBoundingClientRect(), s = getComputedStyle(e);
        const m = e.getScreenCTM();
        return {kind:e.tagName, x:r.x, y:r.y, w:r.width, h:r.height,
          fill:s.fill, stroke:s.stroke, strokeWidth:parseFloat(s.strokeWidth) * m.a,
          radius:(+e.getAttribute('rx') || 0) * m.a,
          d:e.getAttribute('d'), marker:s.markerEnd !== 'none',
          matrix:{a:m.a,b:m.b,c:m.c,d:m.d,e:m.e,f:m.f}};
      }
      function text(e, svg=false) {
        const s = getComputedStyle(e);
        const scale = svg ? e.getScreenCTM().a : 1;
        const fontSize = parseFloat(s.fontSize) * scale;
        const common = {kind:'text',fontSize,bold:+s.fontWeight>=600,
          fontFace:e.closest('pre') ? 'Courier New' : 'Arial',
          color:svg ? s.fill : s.color,
          charSpacing:(parseFloat(s.letterSpacing)||0)*scale,
          role:e.tagName, svg};
        if (svg) {
          const r=e.getBoundingClientRect();
          result.push({...common,text:e.textContent,x:r.x,y:r.y,w:r.width,h:r.height});
          return;
        }
        const walker=document.createTreeWalker(e,NodeFilter.SHOW_TEXT), lines=[];
        let n;
        while(n=walker.nextNode()) {
          for(let i=0;i<n.textContent.length;i++) {
            const range=document.createRange();range.setStart(n,i);range.setEnd(n,i+1);
            const r=range.getBoundingClientRect(), char=n.textContent[i];
            if(!r.height || char==='\n') continue;
            let line=lines.find(l=>Math.abs(l.y-r.y)<1);
            if(!line) {line={text:'',x:r.x,y:r.y,w:0,h:r.height};lines.push(line);}
            line.text+=char;line.w=Math.max(line.w,r.right-line.x);
          }
        }
        for (const line of lines) {
          if (!line.text.trim()) continue;
          if (s.whiteSpace==='normal') line.text=line.text.replace(/\s+/g,' ').trim();
          if (s.textTransform==='uppercase') line.text=line.text.toUpperCase();
          result.push({...common,...line});
        }
      }
      for(const e of document.querySelectorAll('h1,p,pre,svg,footer span')) {
        if(e.closest('svg') && e.tagName!=='svg') continue;
        if(e.tagName==='svg') {
          for(const child of e.children) {
            if(child.tagName==='text') text(child,true);
            else if(['rect','path','circle'].includes(child.tagName)) result.push(shape(child));
          }
        } else {
          if(e.tagName==='PRE') {
            const r=e.getBoundingClientRect(),s=getComputedStyle(e);
            result.push({kind:'rect',x:r.x,y:r.y,w:r.width,h:r.height,
              radius:parseFloat(s.borderRadius),fill:s.backgroundColor,stroke:'none'});
          }
          text(e);
        }
      }
      return result;
    });
    scenes.push({entry,objects});
  }
} finally { await browser.close(); }
await fs.writeFile(path.join(privateDir,'scenes.json'),JSON.stringify(scenes,null,2));

const pptx = new PptxGenJS();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'mui-group';
pptx.company = 'mui-group';
pptx.subject = 'Multimodal retrieval workshop';
pptx.title = 'Search beyond words — Multimodal retrieval';
pptx.lang = 'en-US';
pptx.theme = {headFontFace:'Arial',bodyFontFace:'Arial',lang:'en-US'};

function addText(slide,o) {
  if (o.svg && ['Versioned features','Change one thing'].includes(o.text)) {
    const words = o.text === 'Versioned features' ? ['Versioned','features'] : ['Change one','thing'];
    words.forEach((text,i)=>addText(slide,{...o,text,y:o.y-15+i*32,w:210,fontSize:28}));
    return;
  }
  // Each browser-rendered line remains one native text box. A small right-side
  // allowance prevents Office font metrics from unexpectedly wrapping it.
  slide.addText(o.text, {
    x:inch(o.x),y:inch(o.y),w:inch(Math.min(o.w+16,1590-o.x)),h:inch(o.h+8),
    fontFace:o.fontFace,fontSize:pt(o.fontSize),bold:o.bold,color:color(o.color),
    charSpacing:pt(o.charSpacing),margin:0,breakLine:false,
    valign:'top',paraSpaceAfterPt:0,lineSpacingMultiple:1,
    isTextBox:true,objectName:`${o.role}: ${o.text.slice(0,70)}`,
    lang:'en-US',
  });
}
function addPath(slide,o) {
  const tokens=o.d.match(/[MLHVZmlhvz]|-?\d*\.?\d+(?:e[+-]?\d+)?/g);
  let i=0,x=0,y=0,command;
  const segments=[];
  const transform=(x,y)=>({x:o.matrix.a*x+o.matrix.c*y+o.matrix.e,
    y:o.matrix.b*x+o.matrix.d*y+o.matrix.f});
  while(i<tokens.length) {
    if(/[a-z]/i.test(tokens[i])) command=tokens[i++];
    let nx=x,ny=y;
    if(command==='M'||command==='L') {nx=+tokens[i++];ny=+tokens[i++];}
    else if(command==='H') nx=+tokens[i++];
    else if(command==='V') ny=+tokens[i++];
    else throw Error(`Unsupported SVG path command ${command}`);
    if(command==='M') command='L';
    else segments.push([transform(x,y),transform(nx,ny)]);
    x=nx;y=ny;
  }
  segments.forEach(([a,b],j)=>{
    slide.addShape(pptx.ShapeType.line,{
      x:inch(Math.min(a.x,b.x)),y:inch(Math.min(a.y,b.y)),
      w:inch(Math.abs(b.x-a.x)),h:inch(Math.abs(b.y-a.y)),
      flipH:b.x<a.x,flipV:b.y<a.y,
      line:{color:color(o.stroke)||'08796F',width:pt(o.strokeWidth),
        ...(o.marker&&j===segments.length-1?{endArrowType:'arrow'}:{})},
      objectName:'Diagram connector',
    });
  });
}
function addShape(slide,o) {
  if(o.kind==='text') return addText(slide,o);
  if(o.kind==='path') return addPath(slide,o);
  const fill=color(o.fill),stroke=color(o.stroke);
  slide.addShape(o.kind==='circle'?pptx.ShapeType.ellipse:
    o.radius?pptx.ShapeType.roundRect:pptx.ShapeType.rect,{
    x:inch(o.x),y:inch(o.y),w:inch(o.w),h:inch(o.h),
    rectRadius:inch(o.radius||0),
    fill:fill?{color:fill}:{color:'FFFFFF',transparency:100},
    line:stroke?{color:stroke,width:pt(o.strokeWidth)}:{color:'FFFFFF',transparency:100},
    objectName:o.kind==='circle'?'Diagram point':'Diagram shape',
  });
}
function addChart(slide,chartObjects) {
  const bars=chartObjects.filter(o=>o.kind==='rect');
  if(bars.length!==3) throw Error('Expected the three solar-energy bars');
  const label=chartObjects.find(o=>o.text==='April');
  const value=chartObjects.find(o=>o.text==='57');
  const step=bars[1].x-bars[0].x, plotW=step*3;
  const plotX=bars[0].x-(step-bars[0].w)/2;
  const bottom=bars[0].y+bars[0].h, top=bars[1].y;
  const labelH=55;
  const pad=pt(value.fontSize)*2;
  const height=bottom-top+pad+labelH;
  slide.addChart(pptx.ChartType.bar,[{
    name:'Solar energy (kWh)',labels:['April','May','June'],values:[42,57,51],
  }],{
    x:inch(plotX),y:inch(top-pad),w:inch(plotW),h:inch(height),
    layout:{x:0,y:pad/height,w:1,h:(bottom-top)/height},
    barDir:'col',barGrouping:'clustered',
    barGapWidthPct:Math.round(100*(step/bars[0].w-1)),
    catAxisHidden:false,catAxisLabelFontFace:'Arial',catAxisLabelFontSize:pt(label.fontSize),
    catAxisLabelColor:'536164',catAxisLineShow:false,
    catAxisMajorTickMark:'none',catAxisMinorTickMark:'none',
    valAxisHidden:true,valAxisMinVal:0,valAxisMaxVal:57,
    valAxisLineShow:false,showLegend:false,showTitle:false,
    showValue:true,dataLabelPosition:'outEnd',dataLabelColor:'202729',
    dataLabelFormatCode:'0',
    chartColors:['08796F','BF4938','08796F'],
    showCatName:false,showSerName:false,
    dataLabelFontBold:true,dataLabelFontFace:'Arial',dataLabelFontSize:pt(value.fontSize),
    catGridLine:{style:'none'},valGridLine:{style:'none'},
    chartArea:{fill:{color:'FFFFFF',transparency:100},border:{color:'FFFFFF',transparency:100}},
    plotArea:{fill:{color:'FFFFFF',transparency:100},border:{color:'FFFFFF',transparency:100}},
    objectName:'Solar energy — editable data chart',
    altText:'Solar energy in kilowatt-hours: April 42, May 57, June 51. Fictional teaching data.',
  });
}

for (const {entry,objects} of scenes) {
  const slide=pptx.addSlide();
  slide.background={color:'FFFFFF'};
  slide.name=entry.title.slice(0,31);
  let chartStart=-1,chartEnd=-1;
  if([1,2,11,19].includes(entry.number)) {
    chartStart=objects.findIndex(o=>o.text==='Solar energy');
    chartEnd=objects.findIndex((o,i)=>i>chartStart&&o.text==='kWh');
    if(chartStart<0||chartEnd<0) throw Error(`Missing chart on ${entry.number}`);
  }
  for(let i=0;i<objects.length;i++) {
    const o=objects[i];
    if(i===chartStart) {
      addShape(slide,o);
      const chartObjects=objects.slice(chartStart+1,chartEnd);
      // Keep the original axes as editable lines, with the bars and their
      // labels supplied by a real PowerPoint chart and embedded workbook.
      addChart(slide,chartObjects);
      chartObjects.filter(x=>x.kind==='path').forEach(x=>addShape(slide,x));
      i=chartEnd-1;
    } else addShape(slide,o);
  }
  slide.addNotes([
    entry.notes,
    `Notebook: ${entry.notebook}`,
    ...(entry.reference?[`Reference notebook cells ${entry.reference} (zero-based). See SOURCE_MAP.md and SOURCES.md in the tutorial module.`]:[]),
    '© mui-group',
  ].join('\n\n'));
}
const bytes=await pptx.write({outputType:'nodebuffer',compression:true});
const zip=await JSZip.loadAsync(bytes);
// PptxGenJS 4.x declares one slide master per slide in Content_Types but
// creates a single shared master. Remove only declarations for absent parts.
const contentTypes=await zip.file('[Content_Types].xml').async('string');
zip.file('[Content_Types].xml',contentTypes.replace(/<Override\b[^>]*PartName="([^"]+)"[^>]*\/>/g,
  (tag,name)=>zip.file(name.replace(/^\//,''))?tag:''));
await fs.writeFile(output,await zip.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
console.log(JSON.stringify({slides:scenes.length,nativeCharts:4,output},null,2));
