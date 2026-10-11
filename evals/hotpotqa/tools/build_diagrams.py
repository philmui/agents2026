"""Generate simple, local SVG teaching diagrams. No external rendering services."""
from pathlib import Path
from html import escape
ROOT = Path(__file__).resolve().parents[1] / 'assets'
ROOT.mkdir(exist_ok=True)

def draw(name, title, description, boxes, arrows, labels=()):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 430" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
    '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="none" stroke="#486581" stroke-width="1.5"/></marker></defs>',
    '<rect width="1040" height="430" fill="#f7fafc"/>',
    f'<text x="32" y="44" font-family="Arial,sans-serif" font-size="26" font-weight="bold" fill="#183b56">{escape(title)}</text>']
    for x1,y1,x2,y2 in arrows:
        parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="#486581" stroke-width="2.5" fill="none" marker-end="url(#arrow)"/>')
    for x,y,w,h,heading,lines,color in boxes:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{color}" stroke="#627d98"/>')
        parts.append(f'<text x="{x+16}" y="{y+31}" font-family="Arial,sans-serif" font-size="21" font-weight="bold" fill="#102a43">{escape(heading)}</text>')
        for i,line in enumerate(lines):
            parts.append(f'<text x="{x+16}" y="{y+64+i*27}" font-family="Arial,sans-serif" font-size="19" fill="#243b53">{escape(line)}</text>')
    for x,y,text in labels:
        parts.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="18" fill="#334e68">{escape(text)}</text>')
    (ROOT / f'{name}.svg').write_text('\n'.join(parts)+ '</svg>')

draw('bridge','A question needs two pieces of evidence','Invented story. The book paragraph identifies Mira Vale. The Mira Vale paragraph identifies Harbor City as birthplace.',[
(30,115,280,140,'Lantern Atlas', ['Sentence 0 identifies','its creator: Mira Vale.'],'#e3f8f5'),
(380,115,280,140,'Mira Vale',['Sentence 0 gives','birthplace: Harbor City.'],'#e3f8f5'),
(730,115,280,140,'Answer',['Harbor City','Cite both sentence 0s.'],'#fff3c4')],
[(312,183,375,183),(662,183,725,183)],[(35,315,'Bridge entity: Mira Vale links the book to the birthplace.'),(35,351,'A city description alone cannot establish where the creator was born.')])
draw('evaluation','Inference and scoring use different inputs','The agent gets question and candidate text only. The evaluator gets predictions plus reference answers and supporting facts.',[
(30,100,270,125,'Agent inputs',['Question','Candidate paragraphs'],'#e3f8f5'),
(385,100,255,125,'LangGraph agent',['Retrieve and read','Answer with citations'],'#e3f8f5'),
(725,100,280,125,'Evaluator',['Compare predictions','with reference labels'],'#fff3c4'),
(385,285,300,105,'Reference labels',['Answer + supporting facts'],'#fce4ec')],
[(302,162,380,162),(642,162,720,162),(687,334,864,228)],[(40,365,'Keep gold labels outside inference.')])
draw('graph','A bounded LangGraph RAG workflow','Retrieve then grade. If evidence is sufficient or two rounds have run, generate. Otherwise rewrite and retrieve again. Errors terminate with a recorded failure.',[
(30,115,205,110,'Retrieve',['Search current query'],'#e3f8f5'),
(300,115,205,110,'Grade',['Enough evidence?'],'#e3f8f5'),
(735,115,265,110,'Generate',['Short answer + citations'],'#fff3c4'),
(300,295,265,100,'Rewrite',['Target missing evidence'],'#e3f8f5')],
[(237,168,295,168),(507,168,730,168),(402,228,402,290),(298,345,132,345),(132,343,132,230)],
[(531,144,'Yes, or at round cap'),(426,265,'No, with budget left'),(610,329,'At most two retrieval rounds'),(610,364,'Errors end with a failure record')])
