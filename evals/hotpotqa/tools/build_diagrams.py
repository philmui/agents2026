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

(ROOT / 'stategraph.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 700" role="img" aria-labelledby="title desc">
<title id="title">The RAG StateGraph: retrieve, grade, rewrite, generate</title>
<desc id="desc">START goes to retrieve. Successful retrieval goes to grade. A successful grade goes to generate if evidence is sufficient or rounds reach max_rounds; otherwise it goes to rewrite. Successful rewriting loops to retrieve. Errors in retrieve, grade, or rewrite go to END. Generate always goes to END, including when it records an error. The two END boxes denote the same terminal state.</desc>
<defs>
  <marker id="normal-arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#315a78"/></marker>
  <marker id="error-arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#b42338"/></marker>
</defs>
<rect width="1200" height="700" rx="16" fill="#f7fafc"/>
<g font-family="Arial, sans-serif" fill="#183b56">
  <text x="36" y="48" font-size="28" font-weight="bold">The RAG StateGraph</text>
  <text x="36" y="80" font-size="18">One shared state · conditional routing · a bounded retrieval loop</text>
  <g fill="none" stroke="#315a78" stroke-width="2.5" marker-end="url(#normal-arrow)">
    <path d="M126 220 H180"/>
    <path d="M400 220 H480"/>
    <path d="M700 220 H850"/>
    <path d="M1080 220 H1110"/>
    <path d="M590 270 V370"/>
    <path d="M480 420 H304 Q290 400 276 420 H140 V220 H180"/>
  </g>
  <g fill="none" stroke="#b42338" stroke-width="2" stroke-dasharray="7 6" marker-end="url(#error-arrow)">
    <path d="M290 270 V555 H1000"/>
    <path d="M700 250 H780 V555 H1000"/>
    <path d="M700 420 H780 V555 H1000"/>
  </g>
  <g stroke="#52758e" stroke-width="1.5">
    <rect x="30" y="190" width="96" height="60" rx="30" fill="#e7eef5"/>
    <rect x="180" y="170" width="220" height="100" rx="12" fill="#e3f8f5"/>
    <rect x="480" y="170" width="220" height="100" rx="12" fill="#e3efff"/>
    <rect x="480" y="370" width="220" height="100" rx="12" fill="#e3efff"/>
    <rect x="850" y="170" width="230" height="100" rx="12" fill="#fff3c4"/>
    <rect x="1110" y="190" width="74" height="60" rx="30" fill="#e7eef5"/>
    <rect x="1000" y="520" width="150" height="70" rx="30" fill="#fff0f1" stroke="#b42338"/>
  </g>
  <g font-size="23" font-weight="bold" text-anchor="middle">
    <text x="78" y="228" font-size="19">START</text>
    <text x="290" y="205">retrieve</text>
    <text x="590" y="205">grade</text>
    <text x="590" y="405">rewrite</text>
    <text x="965" y="205">generate</text>
    <text x="1147" y="228" font-size="19">END</text>
    <text x="1075" y="551" font-size="21" fill="#b42338">END</text>
  </g>
  <g font-size="17" text-anchor="middle">
    <text x="290" y="235">Search current query</text>
    <text x="290" y="257">Accumulate documents</text>
    <text x="590" y="235">Is evidence sufficient?</text>
    <text x="590" y="257">Backend call</text>
    <text x="590" y="435">Target missing evidence</text>
    <text x="590" y="457">Backend call</text>
    <text x="965" y="235">Answer + citations</text>
    <text x="965" y="257">Backend call</text>
    <text x="1075" y="575" font-size="15" fill="#b42338">Recorded failure</text>
  </g>
  <g font-size="16">
    <text x="417" y="202">OK</text>
    <text x="720" y="123">sufficient OR</text>
    <text x="720" y="146">rounds ≥ max_rounds</text>
    <text x="607" y="310">not sufficient AND</text>
    <text x="607" y="333">rounds &lt; max_rounds</text>
    <text x="312" y="394">OK: search rewritten query</text>
    <text x="889" y="302">Always → END</text>
    <text x="889" y="325">Even on generation error</text>
    <text x="306" y="515" fill="#b42338">error</text>
    <text x="797" y="382" fill="#b42338">error</text>
    <text x="797" y="486" fill="#b42338">error</text>
    <text x="36" y="625" font-weight="bold">Solid = normal route · Dashed red = error exit · Both END boxes are the same terminal</text>
    <text x="36" y="656">MAX_ROUNDS = 2: at most two retrieval rounds. The round cap does not guarantee enough evidence.</text>
    <text x="36" y="681">record_node wraps every node: trace + timing + errors; backend_calls increases for grade, rewrite, generate.</text>
  </g>
</g>
</svg>''')
