"""Pastel classroom design, explanatory diagrams, and accessible deck controls.

The content and drawing primitives live in build_slides.py. This module owns
the visual refinements and browser UI so the authored lesson stays easy to read.
"""
import math


def refine_drawings(Drawing, palette):
    ink, blue, teal, orange, paper, pale, mint, peach = palette
    result = {}

    def save(d):
        result[d.name] = d.svg()

    d = Drawing("title-flow", "A laptop message becomes numerical features and then the category technical help.")
    d.node(30, 25, 530, 230, "My laptop will not start.", "Start with a message", pale)
    d.arrow(580, 140, 685, 140)
    d.node(710, 25, 365, 230, "Words → numbers", "Make it computable", mint)
    d.arrow(1095, 140, 1200, 140)
    d.node(1225, 25, 345, 230, "Technical help", "Choose a destination", peach)
    result[d.name] = d.svg().replace('1600 550','1600 280')

    d=Drawing('counts','The message screen screen frozen gives vocabulary counts: fee 0, frozen 1, meeting 0, screen 2, twice 0. The order of the five vocabulary positions stays fixed.')
    d.text(800,70,'MESSAGE: “screen screen frozen”',39,blue,'middle',600)
    for i,(word,count) in enumerate(zip(['fee','frozen','meeting','screen','twice'],[0,1,0,2,0])):
        x=70+i*300;d.rect(x,140,260,270,mint if count else pale)
        d.text(x+130,195,f'POSITION {i}',24,blue,'middle',600)
        d.text(x+130,300,count,83,ink,'middle',600)
        d.text(x+130,385,word,31,ink,'middle')
    d.text(800,480,'The vector is [0, 1, 0, 2, 0]. Its length stays fixed when messages get longer.',31,ink,'middle')
    save(d)

    d = Drawing("timeline", "Selected milestones, with equal spacing rather than a time scale. Word counts; embeddings and CNNs in 2013–14; Transformer in 2017; BERT and T5 in 2018–19; ModernBERT in 2024; Jev in 2026.")
    d.text(65, 45, "SELECTED MILESTONES · SPACING IS NOT A TIME SCALE", 27, blue, weight=600)
    d.line(155, 270, 1455, 270, blue, 4)
    entries = [("Earlier", "Word counts"), ("2013–14", "Embeddings + CNNs"), ("2017", "Transformer"), ("2018–19", "BERT + T5"), ("2024", "ModernBERT"), ("2026", "Jev")]
    for i, (date, title) in enumerate(entries):
        x = 155 + i*260
        d.circle(x, 270, 13, blue)
        d.text(x, 205, date, 35, blue, "middle", 600)
        d.paragraph(x, 330, title, 17, 29, ink, "middle")
    d.text(800, 480, "RNNs began earlier; ULMFiT demonstrated language-model transfer for classification in 2018.", 28, ink, "middle")
    save(d)

    d = Drawing("naive-bayes", "Invented numbers for screen frozen: technical score 0.6 times 0.2 times 0.15 equals 0.018; billing score 0.4 times 0.02 times 0.01 equals 0.00008. These are unnormalized scores, not final probabilities.")
    d.text(65, 55, 'MESSAGE: “screen frozen”', 36, blue, weight=600)
    for x, title in [(65, "CLASS"), (360, "STARTING SHARE"), (750, "WORD CLUES"), (1295, "RESULT")]:
        d.text(x, 140, title, 26, blue, weight=600)
    for y, label, prior, clues, score, fill in [(175, "technical", "0.60", "0.20 × 0.15", "0.018", mint), (310, "billing", "0.40", "0.02 × 0.01", "0.00008", peach)]:
        d.rect(45, y, 1510, 110, fill)
        for x, text in [(65, label), (390, prior), (750, clues), (1295, score)]:
            d.text(x, y+66, text, 36, ink, weight=600)
        d.text(655, y+66, "×", 36)
        d.text(1180, y+66, "=", 36)
    d.text(800, 500, "Technical has the larger score. Normalize the scores if you need probabilities.", 30, ink, "middle")
    save(d)

    d = Drawing("weighted-clues", "Invented logistic regression calculation: frozen contributes 2, screen contributes 2, fee contributes zero, bias contributes minus 1. Score 3 maps to about 0.95 through the sigmoid curve; it is not a measured confidence.")
    d.text(65, 55, "ADD THE LEARNED CONTRIBUTIONS", 27, blue, weight=600)
    for y, label, points in [(145, "frozen: 1 × (+2)", "+2"), (220, "screen: 2 × (+1)", "+2"), (295, "fee: 0 × (−2)", "0"), (370, "starting offset (bias)", "−1")]:
        d.text(65, y, label, 35)
        d.text(650, y, points, 38, blue, weight=600)
    d.line(65, 400, 750, 400, blue, 2)
    d.text(65, 465, "Evidence score", 35)
    d.text(650, 465, "3", 48, blue, weight=600)
    d.text(1115, 55, "CONVERT TO A PROBABILITY", 27, blue, "middle", 600)
    d.line(930, 420, 1515, 420); d.line(930, 420, 930, 115)
    for val in [0, .5, 1]:
        y = 420-val*290
        d.line(930, y, 1515, y, "#d8ddd9", 2)
        d.text(910, y+9, f"{val:g}", 27, ink, "end")
    pts = [(960+i*5.1, 420-290/(1+math.exp(-(-6+i*.12)))) for i in range(101)]
    d.parts.append('<polyline points="'+' '.join(f'{x:.1f},{y:.1f}' for x,y in pts)+f'" fill="none" stroke="{teal}" stroke-width="5"/>')
    x, y = 960+9/12*510, 420-290/(1+math.exp(-3))
    d.circle(x,y,9,blue); d.line(x,y,x,420,blue,2,"6 6")
    d.text(x,459,"score 3",28,blue,"middle")
    d.text(1200,515,"Probability ≈ 0.95 in this example",29,ink,"middle")
    save(d)

    d = Drawing("tfidf", "In six invented training documents the word the appears in all six and refund in two. IDF lowers the influence of ubiquitous words; rarity alone does not prove relevance.")
    d.text(800, 50, "SIX TRAINING DOCUMENTS", 29, blue, "middle", 600)
    for i in range(6):
        x = 475+i*175
        d.rect(x,85,145,310,pale)
        d.text(x+72,130,f"Doc {i+1}",27,ink,"middle",600)
        d.circle(x+72,215,21,teal); d.text(x+72,225,"✓",28,"white","middle")
        d.circle(x+72,330,21,teal if i<2 else "#f9f7f2")
        d.text(x+72,340,"✓" if i<2 else "—",28,"white" if i<2 else blue,"middle")
    d.text(65,225,'“the”',40,ink,weight=600)
    d.text(65,340,'“refund”',40,ink,weight=600)
    d.text(800,455,"Present in 6 of 6 → less distinctive      Present in 2 of 6 → more distinctive",30,ink,"middle")
    d.text(800,515,"TF counts within one document; IDF uses document frequency across training data.",28,ink,"middle")
    save(d)

    d = Drawing("rnn", "An unrolled RNN applies the same update cell three times. Initial state h0 and token not produce h1; h1 and very produce h2; h2 and helpful produce h3; the final state feeds a classification head.")
    d.text(800,50,"ONE UPDATE RULE, REUSED AT EACH POSITION",29,blue,"middle",600)
    d.text(40,260,"h₀",38,blue,weight=600)
    for i, word in enumerate(["not","very","helpful"]):
        x=160+i*375
        d.node(x,145,290,195,"Update cell","Earlier state + this token",mint)
        d.arrow(x-70,242,x-15,242)
        if i<2:d.text(x+325,220,f"h{i+1}",34,blue,"middle",600)
        d.node(x,405,290,90,word,fill=pale);d.arrow(x+145,390,x+145,355)
    d.arrow(1210,242,1280,242);d.text(1250,220,"h₃",34,blue,"middle",600)
    d.node(1300,145,250,195,"Classify","Use the final state",peach)
    d.text(800,535,"h is a numerical memory of what the model has read so far.",28,ink,"middle")
    save(d)

    d = Drawing("cnn", "A three-token filter slides over four windows of the same six-token sentence. Invented response scores are 0.1, 0.2, 0.9, 0.8. Outlines show each exact window; the same filter weights are used at every position.")
    words=["the","movie","was","surprisingly","good","today"]
    for i,w in enumerate(words):
        d.rect(50+i*215,30,195,82,pale)
        d.text(147+i*215,81,w,28,ink,"middle",600)
    for row,score in enumerate([.1,.2,.9,.8]):
        y=160+row*83
        for col,w in enumerate(words):
            d.rect(50+col*215,y,195,65,mint if row<=col<row+3 else "#f1f0ed")
            d.text(147+col*215,y+42,w,28,ink if row<=col<row+3 else "#66717b","middle")
        d.rect(45+row*215,y-5,635,75,"none",teal,8)
        d.text(1390,y+44,f"→ {score:.1f}",35,teal,weight=600)
    d.text(800,535,"Outlined triplet = one window. Every window uses the same learned filter.",29,ink,"middle")
    save(d)

    d = Drawing("attention", "For the token bank, invented attention weights are The 0.05, river 0.45, bank 0.10, was 0.05, muddy 0.35, summing to 1. This illustrates learned mixing rather than measured attention or a proof of reasoning.")
    d.node(555,15,490,200,"A new bank representation","Weighted mixture of token information",mint)
    words=["The","river","bank","was","muddy"]
    weights=[.05,.45,.10,.05,.35]
    for i,(w,p) in enumerate(zip(words,weights)):
        x=50+i*310
        d.line(x+130,340,800,230,teal,2+p*15)
        d.rect(x+75,250,110,55,paper)
        d.text(x+130,288,f"{p:.2f}",30,blue,"middle",600)
        d.node(x,365,260,100,w,fill=mint if w=="bank" else pale)
    d.text(800,535,"Invented weights sum to 1. Position information also helps distinguish word order.",28,ink,"middle")
    save(d)

    d = Drawing("attention-masks", "Rows are the token being updated, columns are the token it may read. Five tokens The river bank was muddy label both axes. An encoder permits all positions; a causal decoder permits only current and earlier columns. Check marks mean allowed; crosses mean blocked. For the bank row, the decoder can read The river bank but not was muddy.")
    words=["The","river","bank","was","muddy"]
    for panel,title in enumerate(["ENCODER: BOTH DIRECTIONS","CAUSAL DECODER: CURRENT + EARLIER"]):
        x=155+panel*800
        d.text(x+165,40,title,27,blue,"middle",600)
        d.text(x+170,90,"Read information from →",27,ink,"middle")
        for j,w in enumerate(words):d.text(x+j*72+30,138,w,25,ink,"middle")
        for row,w in enumerate(words):
            d.text(x-16,189+row*65,w,26,blue if row==2 else ink,"end",600 if row==2 else 400)
            for col in range(5):
                allowed=panel==0 or col<=row
                d.rect(x+col*72,153+row*65,61,55,mint if allowed else "#eeece7",radius=7)
                d.text(x+col*72+30,190+row*65,"✓" if allowed else "×",28,teal if allowed else "#696b72","middle",600)
        d.rect(x-8,153+2*65-5,369,65,"none",blue,7)
    d.text(800,535,"Rows = token being updated · columns = source tokens · ✓ allowed · × blocked",28,ink,"middle")
    save(d)

    d = Drawing("design-menu", "Four approaches compared by output and task setup: count classifiers need labels; encoders need task adaptation; generators can follow instructions; Jev uses described choices and must be evaluated on the application.")
    for x,t in [(80,"APPROACH"),(635,"USEFUL OUTPUT"),(1090,"TASK SETUP")]:d.text(x,50,t,27,blue,weight=600)
    rows=[("Word-count classifier","One fixed label set","Train on labeled examples"),("Fine-tuned encoder","Contextual classification","Adapt to your label set"),("Generative model","Labels and open text","Prompt; sometimes fine-tune"),("Jev decision API","Typed bounded decisions","Describe; test on your task")]
    for i,row in enumerate(rows):
        y=85+i*112;d.rect(50,y,1500,95,[pale,mint,peach,pale][i])
        for x,t in zip([80,635,1090],row):d.paragraph(x,y+38,t,30,28)
    save(d)

    d = Drawing("choice", "Invented Choice distribution for a laptop message: technical 80 percent, billing 15 percent, information 5 percent. The values sum to 100 percent, and Choice selects technical. These values are not a Jev measurement.")
    d.text(80,50,"ONE QUESTION · ONE SELECTED OPTION",29,blue,weight=600)
    for i,(label,p) in enumerate(zip(["technical","billing","information"],[.8,.15,.05])):
        y=120+i*105
        d.text(80,y+40,label,35,ink,weight=600 if i==0 else 400)
        d.rect(420,y,850,60,"#e8e8e1",radius=7)
        d.rect(420,y,850*p,60,blue if i==0 else teal,radius=7)
        d.text(1305,y+42,f"{p:.0%}"+ ("  ← selected" if i==0 else ""),32,blue if i==0 else ink,weight=600)
    for tick in [0,.25,.5,.75,1]:d.text(420+850*tick,465,f"{tick:.0%}",27,ink,"middle")
    d.text(800,535,"Illustrative probabilities total 100%. The selected label is technical.",29,ink,"middle")
    save(d)

    d = Drawing("noul", "Noul is a probability that one yes/no statement is true: 0.1 favors no, 0.5 is uncertain, 0.9 favors yes. It is not a second confidence metric.")
    d.text(800,50,'QUESTION: “Does the message describe a blocked device?”',34,blue,"middle",600)
    d.line(150,270,1450,270,ink,4)
    d.text(150,335,"0 · no",32,ink,"middle");d.text(1450,335,"1 · yes",32,ink,"middle")
    for p,label,fill in [(.1,"No is favored",pale),(.5,"Uncertain",peach),(.9,"Yes is favored",mint)]:
        x=150+p*1300;d.circle(x,270,13,blue)
        d.node(x-160,100,320,110,f"{p:.1f}",fill=fill)
        d.line(x,215,x,253,blue,2)
        d.text(x,420,label,33,ink,"middle",600)
    d.text(800,515,"For this question, P(no) = 1 − P(yes). Separate questions need not sum to 1.",29,ink,"middle")
    save(d)

    d = Drawing("confidence", "Choice confidence rescales the top probability relative to an even split. For 3 options the baseline is one-third; probability 0.80 maps to confidence 0.70. Confidence is not a measured correctness rate.")
    d.text(800,45,"SAME ANSWER, TWO DIFFERENT NUMBERS",28,blue,"middle",600)
    d.node(65,85,640,190,"Top probability: 0.80","Estimated probability of the selected option",pale)
    d.node(895,85,640,190,"Choice confidence: 0.70","Distance above an even split, rescaled to 0–1",mint)
    d.arrow(730,180,865,180)
    d.line(120,385,1480,385,ink,4)
    for p,text in [(1/3,"⅓: even split"),(.8,"0.80: this answer"),(1,"1.00")]:
        x=120+1360*p;d.circle(x,385,10,blue)
        d.text(x,345,text,28,ink,"end" if p==1 else "middle")
    d.text(800,455,"confidence = (0.80 − ⅓) / (1 − ⅓) = 0.70",40,blue,"middle",600)
    d.text(800,525,"Compare with observed outcomes before trusting the number as a routing signal.",29,ink,"middle")
    save(d)

    d = Drawing("structured-state", "Structured state contains the message and relevant verified facts. An example JSON object holds message, policy, and facts; a separate question asks which team should help. JSON structure does not verify truth.")
    d.rect(55,50,925,455,pale)
    d.text(85,100,"STATE · RELEVANT EVIDENCE",27,blue,weight=600)
    lines=['{', '  "message": "The device will not start.",', '  "policy": "Blocked devices → technical",', '  "facts": {"loan_device_available": true}', '}']
    for i,line in enumerate(lines):d.text(85,175+i*62,line,30,ink)
    d.arrow(1005,280,1095,280)
    d.node(1120,130,430,310,"QUESTION","Which team should handle this request?",mint)
    save(d)

    d = Drawing("routing", "Read Choice answer technical with top probability 0.80 and confidence 0.70. Explicit code checks confidence against an illustrative threshold 0.60. If confidence is at least 0.60, route to selected handler; otherwise review. The threshold needs development-set evidence.")
    d.node(45,160,400,230,"Read the Choice answer","technical · confidence 0.70",pale)
    d.arrow(465,275,605,275)
    d.parts.append(f'<path d="M845,130 L1065,275 L845,420 L625,275 Z" fill="{peach}" stroke="{orange}" stroke-width="2"/>')
    d.text(845,243,"confidence",33,ink,"middle",600);d.text(845,309,"≥ 0.60?",38,ink,"middle",600)
    d.arrow(1085,255,1200,115);d.arrow(1085,295,1200,430)
    d.node(1230,45,325,160,"Selected handler","Here: technical",mint)
    d.node(1230,350,325,160,"Review","Ask or escalate",peach)
    d.text(1140,168,"yes",29,teal,"middle",600);d.text(1140,415,"no",29,orange,"middle",600)
    d.text(800,535,"0.60 is an illustrative policy threshold. Choose yours on development data.",28,ink,"middle")
    save(d)

    # Merge on a bus, then use one arrow into the combining step. Three
    # coincident arrowheads obscure the direction and imply an extra node.
    d=Drawing('parallel','The same state feeds separate Choice, Noul, and Score questions. The application receives all three answers and combines them in code. The questions need not be statistically independent.')
    d.node(65,165,390,180,'One shared state')
    for i,title in enumerate(['Choice: topic','Noul: urgent?','Score: severity']):
        y=20+i*175;d.arrow(475,250,600,y+65);d.node(625,y,410,135,title,fill=[pale,mint,peach][i]);d.line(1055,y+65,1175,y+65)
    d.line(1175,85,1175,435);d.arrow(1175,250,1220,250)
    d.node(1240,165,290,180,'Code combines answers',fill=mint)
    save(d)

    d=Drawing('multi-agent','A coordinator routes tasks to search, coding, and explanation specialists. Their returned evidence joins at a review step before the workflow accepts results. This is a research proposal.')
    d.node(60,165,370,180,'Coordinator')
    for i,label in enumerate(['Search specialist','Coding specialist','Explanation specialist']):
        y=20+i*175;d.arrow(455,250,595,y+60);d.node(620,y,470,130,label,fill=[pale,mint,peach][i]);d.line(1110,y+60,1180,y+60)
    d.line(1180,80,1180,430);d.arrow(1180,250,1230,250)
    d.node(1250,165,280,180,'Check evidence',fill=mint)
    save(d)

    d = Drawing("coverage", "Invented coverage at thresholds 0.2, 0.5, 0.8 is 8, 6, 3 automatic cases out of 10. Remaining cases go to review. Letter A marks automatic; R marks review so color is not the only cue.")
    d.text(75,45,"CONFIDENCE GATE",27,blue,weight=600)
    d.text(870,45,"SAME 10 CASES AT EACH THRESHOLD",27,blue,"middle",600)
    for i,(threshold,auto) in enumerate([(.2,8),(.5,6),(.8,3)]):
        y=90+i*120;d.text(75,y+46,f"Threshold {threshold:.1f}",33)
        for j in range(10):
            x=440+j*95;d.rect(x,y,77,70,mint if j<auto else peach,radius=8)
            d.text(x+38,y+46,"A" if j<auto else "R",32,teal if j<auto else orange,"middle",600)
        d.text(1430,y+46,f"{auto}/10",33,blue,weight=600)
    d.text(800,480,"A = automatic route · R = review · Fraction = automatic coverage",29,ink,"middle")
    d.text(800,535,"Higher thresholds reduce coverage here; measure error rates separately.",29,ink,"middle")
    save(d)

    d = Drawing("agent-loop", "Plan a step, choose an allowed action, run a tool, read its result, then feed the result into the next step. The loop ends when completion or a budget limit is reached. Jev can provide bounded choices while code controls the workflow.")
    labels=[("Plan a step","A generator can plan"),("Choose an action","Jev can judge a bounded choice"),("Run a tool","Code checks permissions"),("Read the result","Update the shared state")]
    for i,(title,sub) in enumerate(labels):
        x=60+i*390;d.node(x,100,330,235,title,sub,[pale,mint,peach,pale][i])
        if i<3:d.arrow(x+343,217,x+375,217)
    d.parts.append(f'<path d="M1395,355 L1395,445 L225,445 L225,355" fill="none" stroke="{teal}" stroke-width="4" marker-end="url(#agent-loop-arrow)"/>')
    d.rect(555,416,530,57,paper)
    d.text(820,455,"Continue with the new evidence",29,teal,"middle",600)
    d.text(800,535,"Stop when the task is complete or the workflow reaches a budget limit.",29,ink,"middle")
    save(d)

    d = Drawing("trace-eval", "An ordered trace records user request, tool call, tool evidence, and final answer. The judge reads the recorded trace and returns focused typed judgments; evaluating against evidence is different from writing a plausible explanation.")
    labels=["User request","Tool call","Tool evidence","Final answer"]
    for i,label in enumerate(labels):
        y=20+i*125;d.circle(85,y+47,25,blue);d.text(85,y+57,i+1,29,"white","middle",600)
        d.node(145,y,550,95,label,fill=mint if i==2 else pale)
        if i<3:d.arrow(85,y+75,85,y+109)
    d.arrow(725,255,850,255)
    d.node(885,85,645,350,"Judge the whole recorded run","Did the final answer use the evidence? Did it satisfy the request?",mint)
    save(d)
    return result


CSS = r'''
/* === PASTEL CLASSROOM THEME === */
:root{--stage-bg:#e9e7e2;--slide-bg:#faf8f3;--ink:#26364a;--blue:#3b587c;--teal:#326657;--peach:#f6e4da;--mint:#e0eee6;--pale:#e6edf8;--line:#cbd1d4;--toolbar-height:68px}
*{box-sizing:border-box}[hidden]{display:none!important}body{font-family:'Source Sans 3',sans-serif;color:var(--ink)}
/* === FIXED SLIDE COMPOSITION === */
.slide{padding:68px 100px;background:var(--slide-bg)}
.slide:before{content:'';position:absolute;left:0;top:0;width:18px;height:100%;background:var(--chapter-color,var(--pale))}
.chapter-rail{display:flex;gap:8px;position:absolute;left:100px;right:100px;top:25px;height:5px}.chapter-rail span{flex:1;background:#dcded8}.chapter-rail .past{background:#93a8b8}.chapter-rail .current{background:var(--blue)}
.eyebrow{font-size:27px;letter-spacing:2px;text-transform:uppercase;color:var(--blue);font-weight:600;margin:0 0 24px}
.lesson-tag{position:absolute;right:100px;top:68px;font-size:24px;color:#526174;letter-spacing:1px;text-transform:uppercase}
h1,h2{font-family:'Bricolage Grotesque',sans-serif;font-size:77px;font-weight:600;line-height:1.08;letter-spacing:-2.3px;margin:0;width:1720px}
.lead{font-size:35px;line-height:1.36;margin:24px 0 0;width:1690px;max-height:145px}
.visual{position:absolute;left:100px;top:362px;width:1720px;height:591px;margin:0}.visual svg{width:100%;height:100%}
.takeaway{position:absolute;left:100px;right:100px;bottom:62px;min-height:47px;font-size:30px;line-height:1.25;padding:16px 0 0 24px;border-top:2px solid var(--line);margin:0}.takeaway:before{content:'';position:absolute;left:0;top:25px;width:9px;height:9px;background:var(--blue);border-radius:50%}
.source{position:absolute;right:100px;bottom:21px;font-size:22px;color:#465970;text-decoration-thickness:1px;text-underline-offset:3px}.page{position:absolute;left:310px;bottom:21px;font-size:22px;color:#465970;font-variant-numeric:tabular-nums}.credit{position:absolute;left:100px;bottom:19px;font-size:25px;font-weight:600;color:var(--blue)}
.chapter-start{background:linear-gradient(var(--chapter-color) 0 344px,var(--slide-bg) 344px)}
.chapter-start .lesson-tag{color:var(--blue)}
.opening{background:#e0eee6}.opening:before{background:#8daf9d;width:18px}.opening h1{font-size:130px;line-height:1.03;max-width:1500px;letter-spacing:-4px;margin-top:24px}.opening .lead{max-width:1570px;font-size:42px;margin-top:30px}.opening .visual{top:590px;height:300px;background:var(--slide-bg);border-radius:22px;padding:12px 24px}.opening .takeaway{bottom:67px;border-color:#9fb6aa;font-size:32px}
.closing{background:#e6edf8}.closing:before{background:#8ea6c1}.closing .visual{background:var(--slide-bg);border-radius:22px}
/* === CONTENT-APPROPRIATE MOTION === */
.reveal{opacity:0;transform:translateY(10px);transition:opacity .32s ease,transform .32s ease}.visible .reveal{opacity:1;transform:none}.lead.reveal{transition-delay:.04s}.visual.reveal{transition-delay:.08s}.layout-matrix .visual.reveal,.layout-comparison .visual.reveal{transform:none}.layout-flow .visual.reveal{transform:translateX(-12px)}.layout-flow.visible .visual.reveal{transform:none}
.static .reveal{opacity:1;transform:none;transition:none}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{transition:none!important;animation:none!important}.reveal{transform:none!important}}
/* === LARGE, QUIET NAVIGATION OUTSIDE THE SLIDE === */
.deck-controls{height:52px;bottom:8px;padding:3px 5px;gap:5px;display:flex;align-items:center;border:1px solid #b9c3c8;border-radius:14px;background:#faf8f3;color:var(--ink);font-size:15px;white-space:nowrap}
button,select,input{font:inherit;color:var(--ink)}button,select{cursor:pointer;min-height:44px;min-width:44px;border:1px solid transparent;background:transparent;border-radius:9px;padding:8px 12px}button:hover,select:hover{background:var(--pale)}button:disabled{opacity:.4;cursor:default}button:focus-visible,select:focus-visible,input:focus-visible,a:focus-visible,[contenteditable=true]:focus-visible{outline:3px solid #496c94;outline-offset:3px}
#prev,#next{font-size:24px}#counter{min-width:80px;font-variant-numeric:tabular-nums;border-color:var(--line)}#chapters{max-width:175px;border-left:1px solid var(--line);border-radius:0}#notesToggle[aria-expanded=true]{background:var(--mint)}
@media(max-width:650px){.deck-controls{max-width:calc(100vw - 16px);gap:2px;font-size:14px}#chapters{display:none}button{padding:8px 10px}.shortcut-hint{display:none}#counter{min-width:70px}}
/* === READING PANEL: THE STAGE SHRINKS BESIDE IT ON DESKTOP === */
#notes{position:fixed;right:12px;top:12px;bottom:80px;width:360px;overflow:auto;border:1px solid #b9c3c8;background:var(--slide-bg);border-radius:15px;z-index:1100;padding:24px;font-size:19px;line-height:1.55}#notes[hidden]{display:none}.panel-header{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:20px}.panel-label{font-size:14px;letter-spacing:1.4px;text-transform:uppercase;color:var(--blue);font-weight:600}#notes h3{font-family:'Bricolage Grotesque',sans-serif;font-size:27px;line-height:1.2;margin:0 0 18px}#notes p{margin:0 0 22px}#notes a{color:var(--blue)}.panel-meta{font-size:15px;color:#526174}.panel-header button{border:1px solid var(--line);flex-shrink:0}
@media(max-width:899px){#notes{width:min(420px,calc(100vw - 24px));box-shadow:0 8px 40px #26364a22}}
/* === NATIVE DIALOGS PROVIDE FOCUS CONTAINMENT AND ESCAPE === */
dialog{border:1px solid #abb8c2;border-radius:18px;background:var(--slide-bg);color:var(--ink);padding:24px;max-width:calc(100vw - 24px);max-height:calc(100dvh - 28px);font-family:'Source Sans 3',sans-serif}dialog::backdrop{background:#26364a88;backdrop-filter:blur(4px)}dialog h2{font-family:'Bricolage Grotesque',sans-serif;font-size:28px;line-height:1.15;letter-spacing:-.5px;width:auto}dialog p{font-size:18px;line-height:1.45}.dialog-head{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:16px}.dialog-head button{border:1px solid var(--line)}
#overview{width:1100px}#slideSearch{width:100%;min-height:46px;border:1px solid #a6b3be;background:white;border-radius:8px;padding:10px 14px;font-size:18px;margin-bottom:12px}#overviewCount{font-size:15px;color:#526174;margin:0 0 18px}.slide-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(225px,1fr));gap:14px;overflow:auto;max-height:calc(100dvh - 235px);padding:4px}.slide-card{border:1px solid #ccd1d3;text-align:left;border-radius:12px;padding:12px;white-space:normal;display:block;min-height:220px}.slide-card[aria-current=true]{border:3px solid var(--blue);padding:10px}.slide-card .thumbnail{width:100%;height:95px;display:flex;align-items:center;justify-content:center;background:var(--chapter-color);border-radius:7px;overflow:hidden;margin-bottom:12px}.thumbnail svg{width:100%;height:100%}.slide-card .card-meta{display:block;font-size:13px;color:#526174;margin-bottom:5px}.slide-card strong{display:block;font-size:18px;line-height:1.25;font-weight:600}
#tools{width:390px}#tools .tool-actions{display:grid;gap:5px}#tools .tool-actions button{text-align:left;border:1px solid var(--line);display:flex;justify-content:space-between}kbd{font:inherit;font-size:14px;color:#526174}#help{width:640px}#help dl{display:grid;grid-template-columns:150px 1fr;gap:13px;font-size:18px;line-height:1.35}#help dt{font-weight:600}#help dd{margin:0}#help a{color:var(--blue)}
/* === EDITING WITH VISIBLE SAVE STATUS AND RESET === */
.edit-hotzone{position:fixed;top:0;left:0;width:80px;height:65px;z-index:1100}.edit-toggle{position:fixed;top:12px;left:12px;opacity:0;pointer-events:none;z-index:1200;transition:opacity .2s;border-color:#a6b3be;background:var(--slide-bg)}.edit-toggle.show,.edit-toggle.active{opacity:1;pointer-events:auto}.editing .slide [contenteditable=true]{outline:2px dashed var(--blue);outline-offset:5px;background:#ffffff77}.editing .lesson-tag{visibility:hidden}#editStatus{position:fixed;top:15px;left:180px;max-width:calc(100vw - 200px);padding:8px 12px;background:var(--slide-bg);border:1px solid var(--line);border-radius:9px;z-index:1200;font-size:15px}#editStatus[hidden]{display:none}.editing .deck-controls{background:#e0eee6}
/* === TWO-SCREEN PRESENTER VIEW === */
.presenter-shell{position:fixed;inset:0 0 74px;display:grid;grid-template-columns:minmax(0,1fr) 365px;gap:22px;padding:24px;overflow:auto}.preview-column{min-width:0;min-height:0;height:100%}.presenter-label{font-size:15px;margin:0 0 12px;color:#465970;display:flex;justify-content:space-between}.presenter-shell .deck-viewport{position:relative;inset:auto;width:100%;aspect-ratio:16/9;background:var(--slide-bg);border:1px solid #b9c3c8;border-radius:8px}.next-preview{position:relative;width:min(60%,470px);aspect-ratio:16/9;overflow:hidden;background:var(--slide-bg);border:1px solid #b9c3c8;border-radius:8px;margin-top:12px}.next-stage{position:absolute;left:0;top:0;width:1920px;height:1080px;transform-origin:0 0;pointer-events:none}.next-stage .slide{visibility:visible;opacity:1}.next-stage .reveal{opacity:1;transform:none}.next-title{margin:18px 0 0;font-size:16px;color:#465970}.presenter #notes{position:static;width:auto;min-height:0;max-height:100%;bottom:auto;right:auto;top:auto;z-index:auto;margin:0;border-radius:10px}.presenter #notes .panel-header button{display:none}.presenter .edit-hotzone,.presenter .edit-toggle{display:none}.presenter .preview-column .slide{pointer-events:none}.presenter .source{pointer-events:none}
@media(max-width:900px){.presenter-shell{grid-template-columns:minmax(0,1fr);padding:12px;gap:12px}.presenter #notes{max-height:45vh}.next-preview{width:45%}}
@media print{@page{size:1920px 1080px;margin:0}.edit-hotzone,.edit-toggle,#editStatus,#notes,dialog,.presenter-shell>.panel-meta{display:none!important}.reveal{opacity:1!important;transform:none!important}.slide{print-color-adjust:exact;-webkit-print-color-adjust:exact}.presenter-shell{display:block;position:static;padding:0}.presenter-shell .deck-viewport{width:auto;aspect-ratio:auto;border:none}.next-preview,.presenter-label,.next-title{display:none!important}.editing [contenteditable=true]{outline:none!important;background:none!important}}
'''


JS = r'''
/* === OFFLINE CONTROLLER; NO THIRD-PARTY SCRIPT OR NETWORK REQUEST === */
class SlidePresentation {
  constructor() {
    this.slides = [...document.querySelectorAll('#deckStage > .slide')];
    this.stage = document.querySelector('#deckStage');
    this.viewport = document.querySelector('.deck-viewport');
    this.notes = document.querySelector('#notes');
    this.currentSlide = 0; this.editing = false; this.lastWheel = 0;
    this.presenter = new URLSearchParams(location.search).has('presenter');
    this.storageKey = 'mui-system-one-pastel-v2:'+location.pathname;
    this.originals = new Map([...document.querySelectorAll('[data-edit]')].map(el=>[this.fieldKey(el),el.textContent]));
    this.restore(); this.setupNavigation(); this.setupEditing(); this.buildOverview();
    if(this.presenter) this.setupPresenter();
    this.showSlide(this.hashIndex(), false);
    addEventListener('resize',()=>this.fit());
    addEventListener('hashchange',()=>this.showSlide(this.hashIndex()));
    addEventListener('message',e=>{
      if(e.source!==this.presenterWindow && e.source!==window.opener)return;
      if(e.data?.type==='mui-slide' && Number.isInteger(e.data.index))this.showSlide(e.data.index,false);
      if(e.data?.type==='mui-ready' && e.source===this.presenterWindow)this.sendSlide();
    });
    if(this.presenter && window.opener)window.opener.postMessage({type:'mui-ready'},'*');
    this.fit();
  }
  hashIndex(){ const n=Number(location.hash.slice(1));return Number.isFinite(n)&&n>=1?Math.floor(n)-1:0; }
  fieldKey(el){return el.closest('.slide').dataset.slideId+':'+el.dataset.edit;}
  fit(){
    let width=innerWidth,height=innerHeight-document.querySelector('.deck-controls').offsetHeight-16,left=0,top=0;
    if(this.presenter){
      const availableWidth=this.viewport.parentElement.clientWidth;
      const availableHeight=Math.max(100,this.viewport.parentElement.clientHeight-100);
      const currentHeight=Math.min(availableHeight*.74,availableWidth*9/16);
      this.viewport.style.width=`${currentHeight*16/9}px`;this.viewport.style.height=`${currentHeight}px`;
      const nextHeight=Math.min(availableHeight*.26,470*9/16,availableWidth*9/16);
      this.nextStage.parentElement.style.width=`${nextHeight*16/9}px`;this.nextStage.parentElement.style.height=`${nextHeight}px`;
      width=this.viewport.clientWidth;height=this.viewport.clientHeight;
    }
    else if(!this.notes.hidden && innerWidth>=900)width-=390;
    const s=Math.max(.01,Math.min(width/1920,height/1080));
    this.stage.style.transform=`translate(${left+(width-1920*s)/2}px,${top+(height-1080*s)/2}px) scale(${s})`;
    if(this.presenter)this.fitNext();
  }
  showSlide(n, sync=true){
    n=Number.isFinite(n)?Math.trunc(n):0;
    this.currentSlide=Math.max(0,Math.min(n,this.slides.length-1));
    this.slides.forEach((s,i)=>{const active=i===this.currentSlide;s.classList.toggle('active',active);s.classList.toggle('visible',active);s.setAttribute('aria-hidden',String(!active));s.inert=!active;});
    document.querySelector('#counter').textContent=`${this.currentSlide+1} / ${this.slides.length}`;
    document.querySelector('#counter').setAttribute('aria-label',`Slide ${this.currentSlide+1} of ${this.slides.length}. Open slide overview.`);
    document.querySelector('#prev').disabled=this.currentSlide===0;
    document.querySelector('#next').disabled=this.currentSlide===this.slides.length-1;
    document.querySelector('#chapters').value=this.slides[this.currentSlide].dataset.chapter;
    try{history.replaceState(null,'',`#${this.currentSlide+1}`);}catch{}
    this.updateNotes(); this.updateOverview();
    if(this.presenter)this.updateNext();
    if(sync)this.sendSlide();
  }
  sendSlide(){
    const target=this.presenter?window.opener:this.presenterWindow;
    if(target&&!target.closed)target.postMessage({type:'mui-slide',index:this.currentSlide},'*');
  }
  updateNotes(){
    const data=slideData[this.currentSlide];
    document.querySelector('#noteTitle').textContent=this.slides[this.currentSlide].querySelector('h1,h2').textContent;
    document.querySelector('#noteBody').textContent=data.notes;
    const source=document.querySelector('#noteSource');source.hidden=!data.source;source.href=data.source||'#';source.textContent=data.source_title;
    document.querySelector('#notePosition').textContent=`Slide ${this.currentSlide+1} · ${data.chapter}`;
    this.notes.scrollTop=0;
  }
  toggleNotes(open=!this.notes.hidden){
    if(this.presenter)return;
    this.notes.hidden=!open;document.querySelector('#notesToggle').setAttribute('aria-expanded',String(open));this.fit();
    if(open)document.querySelector('#closeNotes').focus();else document.querySelector('#notesToggle').focus();
  }
  dialog(id){
    document.querySelectorAll('dialog[open]').forEach(el=>el.close());
    const d=document.getElementById(id);d.showModal();
    if(id==='overview'){document.querySelector('#slideSearch').value='';this.filterOverview('');this.updateOverview();document.querySelector('#slideSearch').focus();}
  }
  setupNavigation(){
    const bind=(id,fn)=>document.getElementById(id).addEventListener('click',fn);
    bind('next',()=>this.showSlide(this.currentSlide+1));bind('prev',()=>this.showSlide(this.currentSlide-1));
    bind('counter',()=>this.dialog('overview'));bind('notesToggle',()=>this.toggleNotes(this.notes.hidden));bind('closeNotes',()=>this.toggleNotes(false));
    bind('more',()=>this.dialog('tools'));bind('allSlides',()=>this.dialog('overview'));bind('helpToggle',()=>this.dialog('help'));
    document.querySelectorAll('[data-close]').forEach(b=>b.onclick=()=>b.closest('dialog').close());
    document.querySelectorAll('dialog').forEach(d=>{
      d.addEventListener('keydown',e=>{if(e.key==='Escape'){e.preventDefault();e.stopPropagation();d.close();}});
      d.addEventListener('click',e=>{const r=d.getBoundingClientRect();if(e.target===d&&(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom))d.close();});
    });
    document.querySelector('#chapters').onchange=e=>this.showSlide(this.slides.findIndex(s=>s.dataset.chapter===e.target.value));
    bind('full',async()=>{document.querySelector('#tools').close();try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{document.querySelector('#uiStatus').textContent='Use your browser’s full-screen command.';}});
    bind('print',()=>{document.querySelector('#tools').close();window.print();});
    bind('present',()=>{document.querySelector('#tools').close();this.openPresenter();});
    bind('motionToggle',()=>{document.body.classList.toggle('static');document.querySelector('#motionToggle').setAttribute('aria-pressed',String(document.body.classList.contains('static')));});
    addEventListener('keydown',e=>{
      if(document.querySelector('dialog[open]'))return;
      const textTarget=e.target.closest('[contenteditable=true],input,select,textarea');
      if(e.key==='Escape'){e.preventDefault();if(this.editing)this.toggleEditing(false);else if(!this.notes.hidden&&!this.presenter)this.toggleNotes(false);else this.dialog('overview');return;}
      if(textTarget)return;
      // A focused button keeps Space/Enter as its activation key.
      if(e.target.closest('button,a')&&[' ','Enter'].includes(e.key))return;
      if(e.ctrlKey||e.metaKey||e.altKey)return;
      if(['ArrowRight','ArrowDown',' ','PageDown'].includes(e.key)){e.preventDefault();this.showSlide(this.currentSlide+1);}
      else if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){e.preventDefault();this.showSlide(this.currentSlide-1);}
      else if(e.key==='Home'){e.preventDefault();this.showSlide(0);}
      else if(e.key==='End'){e.preventDefault();this.showSlide(this.slides.length-1);}
      else if(e.key.toLowerCase()==='n')this.toggleNotes(this.notes.hidden);
      else if(e.key.toLowerCase()==='e')this.toggleEditing();
      else if(e.key.toLowerCase()==='p')this.openPresenter();
      else if(e.key.toLowerCase()==='b')document.querySelector('#motionToggle').click();
      else if(e.key==='?')this.dialog('help');
    });
    let sx=0,sy=0,touchCount=0;
    this.stage.addEventListener('touchstart',e=>{touchCount=e.touches.length;sx=e.changedTouches[0].clientX;sy=e.changedTouches[0].clientY;},{passive:true});
    this.stage.addEventListener('touchend',e=>{if(this.editing||touchCount!==1)return;const dx=e.changedTouches[0].clientX-sx,dy=e.changedTouches[0].clientY-sy;if(Math.abs(dx)>45&&Math.abs(dx)>Math.abs(dy))this.showSlide(this.currentSlide+(dx<0?1:-1));},{passive:true});
    this.stage.addEventListener('wheel',e=>{if(this.editing||e.ctrlKey||e.target.closest('a,button,input')||Math.abs(e.deltaY)<30||Date.now()-this.lastWheel<700)return;this.lastWheel=Date.now();this.showSlide(this.currentSlide+(e.deltaY>0?1:-1));},{passive:true});
  }
  buildOverview(){
    this.cards=this.slides.map((slide,i)=>{
      const b=document.createElement('button');b.className='slide-card';b.dataset.index=i;b.setAttribute('aria-label',`Slide ${i+1}: ${slideData[i].title}`);
      b.style.setProperty('--chapter-color',slide.style.getPropertyValue('--chapter-color'));
      const thumb=document.createElement('span');thumb.className='thumbnail';thumb.setAttribute('aria-hidden','true');
      const svg=slide.querySelector('svg').cloneNode(true);this.uniqueSvg(svg,`overview-${i}`);thumb.append(svg);
      const meta=document.createElement('span');meta.className='card-meta';meta.textContent=`${String(i+1).padStart(2,'0')} · ${slideData[i].chapter}`;
      const title=document.createElement('strong');title.textContent=slide.querySelector('h1,h2').textContent;
      b.append(thumb,meta,title);b.onclick=()=>{this.showSlide(i);document.querySelector('#overview').close();document.querySelector('#counter').focus();};return b;
    });
    document.querySelector('#slideGrid').replaceChildren(...this.cards);
    document.querySelector('#slideSearch').addEventListener('input',e=>this.filterOverview(e.target.value));
    this.filterOverview('');
  }
  uniqueSvg(svg,prefix){
    const ids=[...svg.querySelectorAll('[id]')].map(el=>el.id);
    for(const id of ids)for(const el of [svg,...svg.querySelectorAll('*')])for(const a of [...el.attributes]){
      if(a.name==='id'&&a.value===id)el.setAttribute(a.name,prefix+'-'+id);
      else if(a.value.includes(id))el.setAttribute(a.name,a.value.replaceAll(id,prefix+'-'+id));
    }
  }
  filterOverview(query){
    const q=query.trim().toLowerCase();let count=0;
    this.cards.forEach((card,i)=>{card.hidden=!`${i+1} ${slideData[i].chapter} ${card.querySelector('strong').textContent}`.toLowerCase().includes(q);if(!card.hidden)count++;});
    document.querySelector('#overviewCount').textContent=count?`${count} slide${count===1?'':'s'} · Select a card to jump`:'No matching slides. Try a topic such as confidence.';
  }
  updateOverview(){if(this.cards)this.cards.forEach((card,i)=>{card.setAttribute('aria-current',String(i===this.currentSlide));card.querySelector('strong').textContent=this.slides[i].querySelector('h1,h2').textContent;});}
  setupEditing(){
    const hot=document.querySelector('.edit-hotzone'),btn=document.querySelector('#editToggle');let timer;
    for(const el of [hot,btn]){el.addEventListener('mouseenter',()=>{clearTimeout(timer);btn.classList.add('show');});el.addEventListener('mouseleave',()=>{timer=setTimeout(()=>{if(!this.editing)btn.classList.remove('show');},400);});}
    btn.onclick=()=>this.toggleEditing();hot.onclick=()=>this.toggleEditing();
    document.querySelector('#editFromMenu').onclick=()=>{document.querySelector('#tools').close();this.toggleEditing();};
    document.querySelectorAll('[data-edit]').forEach(el=>{
      el.addEventListener('input',()=>{this.persist();this.updateNotes();this.updateOverview();});
      el.addEventListener('paste',e=>{if(!this.editing)return;e.preventDefault();const text=e.clipboardData.getData('text/plain');document.execCommand('insertText',false,text);});
    });
    document.querySelector('#save').onclick=()=>{document.querySelector('#tools').close();this.export();};
    document.querySelector('#resetEdits').onclick=()=>{this.originals.forEach((text,key)=>{const el=[...document.querySelectorAll('[data-edit]')].find(el=>this.fieldKey(el)===key);if(el)el.textContent=text;});try{localStorage.removeItem(this.storageKey);}catch{}this.updateNotes();this.updateOverview();document.querySelector('#uiStatus').textContent='Browser edits reset to this file’s saved text.';};
    addEventListener('keydown',e=>{if(this.editing&&(e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();this.export();}});
  }
  toggleEditing(enabled=!this.editing){
    if(this.presenter)return;
    this.editing=enabled;document.body.classList.toggle('editing',enabled);const button=document.querySelector('#editToggle');button.classList.toggle('active',enabled);button.textContent=enabled?'Finish editing':'Edit (E)';button.setAttribute('aria-pressed',String(enabled));
    document.querySelectorAll('[data-edit]').forEach(el=>{el.contentEditable=String(enabled);});
    document.querySelector('#editStatus').hidden=!enabled;document.querySelector('#editStatus').textContent='Edits save in this browser · More → Save HTML';
    if(enabled)this.slides[this.currentSlide].querySelector('h1,h2').focus();else{this.persist();document.querySelector('#counter').focus();}
  }
  persist(){
    const values={};document.querySelectorAll('[data-edit]').forEach(el=>values[this.fieldKey(el)]=el.textContent);
    try{localStorage.setItem(this.storageKey,JSON.stringify(values));document.querySelector('#editStatus').textContent='Saved in this browser · More → Save HTML';}catch{document.querySelector('#editStatus').textContent='Browser storage unavailable · use More → Save HTML';}
  }
  restore(){try{const values=JSON.parse(localStorage.getItem(this.storageKey)||'null');if(values&&typeof values==='object')document.querySelectorAll('[data-edit]').forEach(el=>{const text=values[this.fieldKey(el)];if(typeof text==='string')el.textContent=text;});}catch{}}
  export(){
    const clone=document.documentElement.cloneNode(true);
    clone.querySelector('body').classList.remove('editing','presenter');clone.querySelectorAll('[data-edit]').forEach(el=>el.contentEditable='false');clone.querySelectorAll('dialog').forEach(el=>el.removeAttribute('open'));clone.querySelector('#notes').hidden=true;clone.querySelector('#editStatus').hidden=true;clone.querySelector('#editToggle').classList.remove('active','show');clone.querySelector('#editToggle').textContent='Edit (E)';clone.querySelector('#editToggle').setAttribute('aria-pressed','false');clone.querySelector('#slideGrid').replaceChildren();
    const url=URL.createObjectURL(new Blob(['<!doctype html>\n'+clone.outerHTML],{type:'text/html'}));const a=document.createElement('a');a.href=url;a.download='from-bag-of-words-to-jev-edited.html';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  openPresenter(){
    if(this.presenter)return;
    const url=new URL(location.href);url.searchParams.set('presenter','1');
    this.presenterWindow=window.open(url.href,'mui-system-one-presenter','width=1440,height=900');
    if(!this.presenterWindow){document.querySelector('#uiStatus').textContent='Presenter window was blocked. Allow pop-ups for this file and try again.';this.dialog('tools');}else this.presenterWindow.focus();
  }
  setupPresenter(){
    document.body.classList.add('presenter');this.notes.hidden=false;
    const shell=document.createElement('div');shell.className='presenter-shell';
    const previews=document.createElement('div');previews.className='preview-column';
    const label=document.createElement('p');label.className='presenter-label';label.innerHTML='<span>Audience slide</span><span id="audienceStatus">Connecting…</span>';
    const nextLabel=document.createElement('p');nextLabel.className='next-title';nextLabel.textContent='Coming next';
    const next=document.createElement('div');next.className='next-preview';next.innerHTML='<div class="next-stage"></div>';
    previews.append(label,this.viewport,nextLabel,next);shell.append(previews,this.notes);document.body.prepend(shell);
    this.nextStage=next.querySelector('.next-stage');
    document.querySelector('#notesToggle').hidden=true;document.querySelector('#present').hidden=true;document.querySelector('#editFromMenu').hidden=true;document.querySelector('#save').hidden=true;document.querySelector('#resetEdits').hidden=true;
    const status=()=>document.querySelector('#audienceStatus').textContent=window.opener&&!window.opener.closed?'Audience tab open':'Audience tab unavailable';status();setInterval(status,1200);
  }
  updateNext(){
    if(this.currentSlide===this.slides.length-1){this.nextStage.replaceChildren();document.querySelector('.next-title').textContent='End of lesson · Home returns to the start';}
    else{const clone=this.slides[this.currentSlide+1].cloneNode(true);clone.removeAttribute('inert');clone.setAttribute('aria-hidden','false');clone.classList.add('active','visible');this.uniqueSvg(clone.querySelector('svg'),'next');clone.querySelectorAll('[data-edit]').forEach(el=>{el.removeAttribute('data-edit');el.contentEditable='false';});this.nextStage.replaceChildren(clone);document.querySelector('.next-title').textContent=`Coming next · ${slideData[this.currentSlide+1].title}`;}
    this.fitNext();
  }
  fitNext(){if(this.nextStage)this.nextStage.style.transform=`scale(${this.nextStage.parentElement.clientWidth/1920})`;}
}
window.deck=new SlidePresentation();
'''
