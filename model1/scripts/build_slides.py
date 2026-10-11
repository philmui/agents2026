"""Build the self-contained classroom deck and its original SVG illustrations.

Run from the repository root with Python 3.10+. No third-party packages needed.
The JSON slide source and SVG files are also useful independently of the deck.
"""
from pathlib import Path
import base64
import html
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/slides"
ASSETS = OUT / "assets"
INK, BLUE, TEAL, ORANGE = "#26364a", "#3b587c", "#326657", "#8b5137"
PAPER, PALE, MINT, PEACH = "#faf8f3", "#e6edf8", "#e0eee6", "#f6e4da"


class Drawing:
    def __init__(self, name, description):
        self.name, self.description, self.parts = name, description, []

    def rect(self, x, y, w, h, fill=PALE, stroke="none", radius=12):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

    def text(self, x, y, text, size=30, color=INK, anchor="start", weight=400):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{html.escape(str(text))}</text>')

    def paragraph(self, x, y, text, width=34, size=30, color=INK, anchor="start"):
        for i, line in enumerate(textwrap.wrap(text, width=width)):
            self.text(x, y + i * (size * 1.5), line, size, color, anchor)

    def node(self, x, y, w, h, title, subtitle="", fill=PALE, color=INK):
        self.parts.append('<g class="diagram-node">')
        self.rect(x, y, w, h, fill)
        title_lines = textwrap.wrap(title, max(10, int((w-36)/16)))
        title_height = len(title_lines)*50
        if subtitle:
            subtitle_width = max(10, int((w-40)/13.5))
            line_count = len(textwrap.wrap(subtitle, width=subtitle_width))
            total = title_height + 24 + line_count*42
            start = y+(h-total)/2+37
        else:
            start = y+(h-title_height)/2+37
        for i, line in enumerate(title_lines):
            self.text(x+w/2, start+i*50, line, 33, color, "middle", 600)
        if subtitle:
            for i, line in enumerate(textwrap.wrap(subtitle, width=subtitle_width)):
                self.text(x+w/2, start+title_height+24+i*42, line, 28, color, "middle")
        self.parts.append('</g>')

    def line(self, x1, y1, x2, y2, color=INK, width=3, dash=""):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>')

    def arrow(self, x1, y1, x2, y2, color=INK):
        self.parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="4" marker-end="url(#{self.name}-arrow)"/>')

    def circle(self, x, y, r, fill=BLUE):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>')

    def svg(self):
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 550" role="img" aria-labelledby="{self.name}-title {self.name}-desc">
<title id="{self.name}-title">{html.escape(self.name.replace('-', ' '))}</title><desc id="{self.name}-desc">{html.escape(self.description)}</desc>
<defs><marker id="{self.name}-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="context-stroke"/></marker></defs>
<style>text{{font-family:'Source Sans 3',sans-serif}} .mono{{font-family:monospace}}</style>
{''.join(self.parts)}</svg>'''


def flow(name, labels, descriptions=None):
    d = Drawing(name, " → ".join(labels))
    n = len(labels)
    w = (1460 - 70*(n-1))/n
    for i, title in enumerate(labels):
        x = 70+i*(w+70)
        d.text(x, 85, f"STEP {i+1:02d}", 26, BLUE, weight=600)
        d.node(x, 115, w, 280, title, (descriptions or [""]*n)[i], [PALE, MINT, PEACH, PALE][i%4])
        if i < n-1:
            d.arrow(x+w+8, 250, x+w+58, 250)
    return d


def bars(name, labels, values, heading="Illustrative probabilities"):
    d = Drawing(name, f"{heading}: " + ", ".join(f"{l}: {v:.0%}" for l,v in zip(labels, values)))
    d.text(90, 55, heading, 30, BLUE, weight=600)
    for i,(label,value) in enumerate(zip(labels,values)):
        y = 120+i*105
        d.text(90, y+38, label, 32)
        d.rect(425, y, 870, 55, "#e4e1d9", radius=8)
        if value:
            d.rect(425, y, 870*value, 55, [BLUE,TEAL,ORANGE][i%3], radius=8)
        d.text(1340, y+39, f"{value:.0%}", 34, weight=600)
    return d


def make_drawings():
    ds = {}
    def keep(d): ds[d.name] = d.svg()
    keep(flow("title-flow", ["A message", "A representation", "A decision"], ["The laptop will not start.", "Numbers a model can use", "Technical help"]))
    d=Drawing("inbox", "Three school help messages route to different teams.")
    for i,(message,team) in enumerate([("My laptop will not start.","Technical help"),("I paid the club fee twice.","Billing help"),("When does robotics meet?","Club information")]):
        y=50+i*155;d.node(70,y,820,125,message,fill=PALE);d.arrow(915,y+62,1065,y+62);d.node(1100,y,420,125,team,fill=MINT)
    keep(d)
    d=Drawing("classify-generate", "Classification chooses a label; generation produces a sequence of tokens.")
    d.node(50,80,500,340,"Same message","My laptop will not start.")
    d.arrow(570,165,760,165);d.arrow(570,340,760,340)
    d.node(800,70,730,160,"Classification → technical",fill=MINT)
    d.node(800,295,730,160,"Generation → Try checking the battery…",fill=PEACH)
    keep(d)
    d=Drawing("examples-labels", "Training pairs connect example messages to teacher-provided labels.")
    d.text(100,50,"EXAMPLE TEXT",28,BLUE);d.text(1110,50,"LABEL",28,TEAL)
    for i,(a,b) in enumerate([("The screen is frozen.","technical"),("The fee was charged twice.","billing"),("Where is the club meeting?","information")]):
        d.node(80,85+135*i,910,110,a);d.arrow(1015,140+135*i,1090,140+135*i);d.node(1120,85+135*i,390,110,b,fill=MINT)
    keep(d)
    keep(flow("pipeline", ["Collect examples", "Turn text into numbers", "Learn patterns", "Check new examples"]))
    d=Drawing("timeline", "Selected milestones, not a ranking: word counts, word embeddings, CNNs and RNNs, Transformers in 2017, BERT in 2018, T5 in 2019, ModernBERT in 2024, Jev in 2026.")
    d.line(90,265,1510,265,BLUE,5)
    entries=[("Earlier","Word counts"),("2013–14","Embeddings + CNNs"),("2017","Transformer"),("2018–19","BERT + T5"),("2024","ModernBERT"),("2026","Jev")]
    for i,(year,name) in enumerate(entries):
        x=100+i*280;d.circle(x,265,12,BLUE)
        d.text(x,210,year,29,BLUE,"middle",600)
        d.paragraph(x,320,name,17,27,INK,"middle")
    d.text(800,460,"RNNs began earlier; ULMFiT brought language-model transfer to classification in 2018.",27,INK,"middle")
    keep(d)
    keep(flow("tokens", ["Screen, screen frozen!", "screen | screen | frozen"], ["Original text", "One simple tokenizer: lowercase and split words"]))
    d=Drawing("vocabulary", "A training vocabulary fixes one position for each known word.")
    words=["fee","frozen","meeting","screen","twice"]
    for i,word in enumerate(words):
        x=90+285*i;d.node(x,145,260,185,word,f"Position {i}")
    d.text(800,425,"Keep these positions fixed when a new message arrives.",35,TEAL,"middle")
    keep(d)
    d=Drawing("counts", "Screen screen frozen gives counts fee 0, frozen 1, meeting 0, screen 2, twice 0.")
    d.text(80,60,'“screen screen frozen”',43,BLUE)
    for i,(word,count) in enumerate(zip(["fee","frozen","meeting","screen","twice"],[0,1,0,2,0])):
        x=85+i*290;d.node(x,155,260,210,str(count),word,MINT if count else PALE)
    d.text(800,465,"Five entries, even when the message gets much longer.",33,INK,"middle")
    keep(d)
    d=Drawing("naive-bayes", "Naive Bayes combines label frequency and word evidence; illustrated numbers are invented.")
    d.node(60,80,450,280,"Message: screen frozen","Count the words in each class.")
    d.arrow(540,220,660,220)
    d.node(695,65,830,165,"Technical: prior × P(screen | technical) × P(frozen | technical)",fill=MINT)
    d.node(695,270,830,165,"Billing: prior × P(screen | billing) × P(frozen | billing)",fill=PEACH)
    d.text(800,505,"Choose the largest result; the independence assumption is a useful simplification.",29,INK,"middle")
    keep(d)
    d=Drawing("weighted-clues", "Invented logistic regression weights: frozen +2, screen +1, fee -2, with bias -1.")
    for i,(title,sub) in enumerate([("frozen × 1","+2 points"),("screen × 2","+2 points"),("fee × 0","0 points"),("bias","−1 point")]):
        d.node(65+i*380,90,350,230,title,sub,[MINT,PALE,PEACH,PALE][i])
    d.text(800,415,"Evidence score = 2 + 2 + 0 − 1 = 3",39,BLUE,"middle",600)
    d.text(800,485,"A logistic curve turns this score into a probability estimate.",30,INK,"middle")
    keep(d)
    d=Drawing("tfidf", "TF-IDF gives common words less influence and unusual words more influence.")
    d.node(70,80,680,300,"“the” in almost every message","Common words help less with distinguishing documents.",PALE)
    d.node(850,80,680,300,"“refund” in a few messages","Unusual words can be useful clues.",MINT)
    d.text(800,480,"TF: count in this document. IDF: rarity across the training documents.",30,INK,"middle")
    keep(d)
    d=Drawing("order-loss", "The student helps the teacher and the teacher helps the student have identical word counts.")
    for i,sentence in enumerate(["The student helps the teacher.","The teacher helps the student."]):
        d.node(75,65+230*i,840,155,sentence);d.arrow(950,140+230*i,1090,140+230*i)
        d.node(1130,65+230*i,400,155,"Same count vector",fill=MINT)
    keep(d)
    d=Drawing("bigrams", "Neighboring pairs preserve a little word order, but require more features.")
    for i,phrase in enumerate(["student helps","helps the","the teacher"]):d.node(80+i*505,130,465,220,phrase,"A two-word feature",[PALE,MINT,PEACH][i])
    d.text(800,445,"Pairs preserve local order; they still do not capture every long-distance relationship.",30,INK,"middle")
    keep(d)
    keep(flow("data-splits", ["Training set", "Development set", "Test set"], ["Learn the model and vocabulary", "Choose settings and thresholds", "Evaluate once after choices are fixed"]))
    d=Drawing("embeddings", "An embedding is a learned list of numbers; a two-dimensional drawing is a simplified view.")
    d.line(130,420,650,420);d.line(130,420,130,70)
    for x,y,label,c in [(300,160,"screen",BLUE),(380,200,"display",BLUE),(510,325,"refund",ORANGE),(550,280,"payment",ORANGE)]:
        d.circle(x,y,11,c);d.text(x+18,y+9,label,29,c)
    d.node(860,90,650,270,"screen → [0.2, −0.7, 0.5, …]","Values are learned; axes are usually not named concepts.",MINT)
    d.text(800,495,"This invented map illustrates similarity; it is not a measured embedding plot.",28,INK,"middle")
    keep(d)
    d=Drawing("context-bank", "A static word embedding gives bank one vector; contextual models can represent its two uses differently.")
    d.node(75,70,690,170,"We sat by the river bank.","Nearby words suggest a river edge.",MINT)
    d.node(835,70,690,170,"We opened a bank account.","Nearby words suggest a financial institution.",PALE)
    d.text(800,370,"Static lookup: one bank vector",40,ORANGE,"middle")
    d.text(800,450,"Contextual model: a representation shaped by the sentence",35,BLUE,"middle")
    keep(d)
    keep(flow("rnn", ["not", "very", "helpful", "Class label"], ["Update a running state", "Carry earlier information", "Update the same state again", "Read the final summary"]))
    d=Drawing("lstm", "Learned gates control what information is kept and what new information is added.")
    d.arrow(80,245,300,245);d.text(85,205,"Earlier state",30)
    d.node(330,110,420,285,"Keep / forget gate","Which earlier information stays?",MINT)
    d.arrow(775,245,870,245)
    d.node(900,110,420,285,"Add new information","What should this word contribute?",PALE)
    d.arrow(1340,245,1510,245)
    d.text(800,480,"Gates are learned numerical operations, not hand-written language rules.",30,INK,"middle")
    keep(d)
    d=Drawing("cnn", "A three-word CNN filter uses the same weights across overlapping windows.")
    words=["the","movie","was","surprisingly","good","today"]
    for i,w in enumerate(words):d.node(65+i*250,30,225,95,w)
    for i,(phrase,score) in enumerate([("the movie was",0.1),("movie was surprisingly",0.2),("was surprisingly good",0.9),("surprisingly good today",0.8)]):
        y=175+i*82;d.rect(90+i*130,y,740,60,[PALE,PALE,MINT,MINT][i]);d.text(120+i*130,y+40,phrase,28);d.text(1150,y+40,f"Filter response: {score:.1f}",29,TEAL)
    keep(d)
    keep(flow("pooling", ["0.1, 0.2, 0.9, 0.8", "Maximum = 0.9", "Classification head"], ["Responses at different positions", "One strongest response per filter", "Combine the filter clues"]))
    keep(flow("transfer", ["Pretrain on much text", "Adapt with labeled examples", "Classify your messages"], ["Learn reusable language patterns", "Fine-tune for a specific task", "Test on held-out messages"]))
    d=Drawing("attention", "Attention mixes information from other tokens using learned weights; shown links are illustrative.")
    words=["The","river","bank","was","muddy"]
    for i,w in enumerate(words):d.node(70+i*310,330,260,110,w,fill=MINT if w=="bank" else PALE)
    for i in range(5):d.line(820,235,200+i*310,315,TEAL if i in [1,4] else BLUE,9 if i in [1,4] else 2)
    d.node(570,45,500,175,"A context-aware bank vector","Combine relevant token information.",MINT)
    d.text(800,510,"Positions also matter. Multiple layers refine these contextual representations.",29,INK,"middle")
    keep(d)
    d=Drawing("attention-masks", "An encoder can attend to both sides; a causal decoder can attend only to the present and earlier positions.")
    for panel,label in enumerate(["Encoder: both directions","Decoder: earlier positions"]):
        x=160+panel*790;d.text(x,55,label,33,BLUE)
        for row in range(5):
            for col in range(5):d.rect(x+col*72,110+row*72,63,63,TEAL if panel==0 or col<=row else "#dfddd6",radius=5)
        d.text(x,505,"Colored = allowed attention",27)
    keep(d)
    keep(flow("bert", ["[CLS] + message tokens", "Bidirectional encoder", "Classification head"], ["Include a summary position", "Read left and right context", "Predict your task labels"]))
    keep(flow("decoder", ["Text + instruction", "Causal decoder", "positive → end"], ["Read the supplied context", "Predict the next token", "Generate a label as text"]))
    keep(flow("decoder-head", ["Text tokens", "Final non-padding token", "Small classification head"], ["Use a causal model", "This position can see the full input", "Predict labels directly after task training"]))
    keep(flow("t5", ["Input text", "Encoder", "Decoder", "Label text"], ["Classify this review…", "Build contextual representations", "Generate an output sequence", "positive"]))
    d=Drawing("design-menu", "Word-count classifiers, fine-tuned encoders, generative models, and decision models offer different capabilities.")
    rows=[("Word-count classifier","Small, fixed task","Needs labeled examples"),("Fine-tuned encoder","Contextual, fixed task","Needs task adaptation"),("Generative model","Labels plus open text","Can follow task instructions"),("Jev decision API","Typed, described choices","Test its fit on your task")]
    for i,row in enumerate(rows):
        y=35+i*120;d.rect(55,y,1490,100,[PALE,MINT,PEACH,PALE][i])
        for x,t in zip([85,610,1090],row):d.paragraph(x,y+42,t,27,27)
    keep(d)
    keep(flow("system-one", ["State", "Typed questions", "Jev", "Answers + probabilities"], ["The relevant text or JSON", "Ask narrow decisions", "Decision model", "Your code chooses the next step"]))
    d=Drawing("known-unknown", "Known: API contract and declared training goal. Undisclosed: complete architecture, data, and RLCD algorithm.")
    d.node(65,70,690,350,"Publicly documented","State + questions; typed answers. TypeSafe names its training approach RLCD.",MINT)
    d.node(845,70,690,350,"Not fully disclosed","Architecture, training recipe, reward design, and full training data.",PEACH)
    keep(d)
    keep(bars("choice",["technical","billing","information"],[.8,.15,.05]))
    d=Drawing("noul", "A yes/no probability of 0.9 means yes is favored; 0.5 means ambiguity; 0.1 favors no.")
    for i,(p,label) in enumerate([(0.1,"No is favored"),(0.5,"Ambiguous"),(0.9,"Yes is favored")]):
        x=90+i*510;d.node(x,110,450,250,f"noul = {p:.1f}",label,[PALE,PEACH,MINT][i])
    d.text(800,460,"P(yes) and P(no) sum to 1 for this one question.",33,INK,"middle")
    keep(d)
    d=Drawing("multi-label", "One Choice selects one category; separate Noul questions can identify several properties at once.")
    d.node(60,90,550,310,"A payment problem that is urgent","One message can have several properties.",PALE)
    for i,(title,sub) in enumerate([("Topic: billing","Choice selects one topic"),("Urgent? 0.9","Noul for urgency"),("Frustrated? 0.7","Separate Noul for frustration")]):
        y=40+i*160;d.arrow(640,240,830,y+63);d.node(865,y,655,130,title,sub,MINT)
    keep(d)
    d=Drawing("score", "An invented Score distribution of 0.1, 0.3, 0.6 over levels 0, 1, 2 gives score 1.5.")
    for i,(level,prob) in enumerate(zip(["Cosmetic issue","Workaround exists","Blocked; no workaround"],[.1,.3,.6])):
        d.node(70+i*505,65,465,240,f"Level {i}: {prob:.0%}",level,[PALE,MINT,PEACH][i])
    d.text(800,395,"Score = 0 × 0.1 + 1 × 0.3 + 2 × 0.6 = 1.5",40,BLUE,"middle",600)
    d.text(800,475,"A position on your rubric, not a physical measurement of damage.",31,INK,"middle")
    keep(d)
    d=Drawing("confidence", "Choice probabilities 0.8, 0.15, 0.05 imply confidence 0.7 with three options.")
    d.node(80,80,670,300,"Largest probability = 0.80","Estimated probability of the chosen option.",PALE)
    d.node(850,80,670,300,"Choice confidence = 0.70","How far the top probability is above an even split.",MINT)
    d.text(800,470,"Three options: confidence = (3 × 0.80 − 1) / 2 = 0.70",36,BLUE,"middle")
    keep(d)
    d=Drawing("calibration", "Among ten illustrative predictions with probability 0.8, eight are correct; calibration concerns groups.")
    for i in range(10):
        x=120+i*150;d.circle(x,210,52,TEAL if i<8 else ORANGE);d.text(x,222,"✓" if i<8 else "×",37,"white","middle",600)
    d.text(800,90,"Ten predictions near probability 0.8",37,BLUE,"middle")
    d.text(800,365,"Eight correct would be compatible with calibration in this tiny illustration.",31,INK,"middle")
    d.text(800,455,"A real check needs many representative labeled examples.",33,INK,"middle")
    keep(d)
    d=Drawing("training-goal", "Illustrative feedback encourages correct decisions with probabilities that match observed outcomes; not a reconstruction of RLCD.")
    d.node(65,95,400,245,"Predict a decision","Include probability estimates.")
    d.arrow(485,215,565,215);d.node(590,95,420,245,"Compare with outcomes","Check correctness and probability quality.",MINT)
    d.arrow(1030,215,1110,215);d.node(1135,95,400,245,"Improve with feedback","RLCD's exact procedure is proprietary.",PEACH)
    d.text(800,465,"This is a teaching diagram of a training goal, not Jev's internal architecture.",29,INK,"middle")
    keep(d)
    d=Drawing("parallel", "Questions about the same state are evaluated separately; the application combines answers.")
    d.node(65,165,390,180,"One shared state")
    for i,title in enumerate(["Choice: topic","Noul: urgent?","Score: severity"]):
        y=20+i*175;d.arrow(475,250,600,y+65);d.node(625,y,410,135,title,fill=[PALE,MINT,PEACH][i]);d.arrow(1060,y+65,1210,250)
    d.node(1240,165,290,180,"Code combines answers",fill=MINT)
    keep(d)
    d=Drawing("structured-state", "A structured state provides message, verified policy, and relevant facts; instructions stay in questions.")
    fields=[("message.text","The device will not start."),("policy","Blocked devices go to technical help."),("facts","Loan device available: yes")]
    for i,(key,value) in enumerate(fields):
        y=35+i*140;d.rect(90,y,1420,115,[PALE,MINT,PEACH][i]);d.text(125,y+48,key,31,BLUE,weight=600);d.text(520,y+48,value,30)
    keep(d)
    keep(flow("good-questions", ["Define one decision", "Describe the boundaries", "Add missing context", "Test known cases"], ["Which help team?", "What belongs in each category?", "Include an other category", "Inspect errors and ambiguity"]))
    d=Drawing("request-response", "An API request specifies model, state, questions; the response reports model, answers, and usage.")
    d.node(60,65,680,350,"REQUEST","model, state, questions: instructions + criteria",PALE)
    d.arrow(770,240,835,240)
    d.node(865,65,680,350,"RESPONSE","model, answers: choice / noul / score; usage",MINT)
    keep(d)
    d=Drawing("routing", "Confidence-gated intent routing sends uncertain cases to review and certain cases to the chosen handler.")
    d.node(65,170,420,175,"Read the model answer")
    d.arrow(510,255,650,255);d.node(680,170,360,175,"Clear enough?",fill=PEACH)
    d.arrow(1065,245,1190,105);d.arrow(1065,265,1190,415)
    d.node(1215,35,325,145,"Chosen handler",fill=MINT);d.node(1215,350,325,145,"Human review",fill=PEACH)
    d.text(1090,130,"yes",27,TEAL);d.text(1080,405,"no",27,ORANGE)
    keep(d)
    d=Drawing("coverage", "Invented threshold examples show that more review can reduce automatic coverage.")
    for i,(threshold,auto,review) in enumerate([(.2,8,2),(.5,6,4),(.8,3,7)]):
        y=65+i*145;d.text(75,y+45,f"Threshold {threshold:.1f}",33)
        for j in range(10):d.rect(450+j*95,y,78,70,TEAL if j<auto else PEACH,radius=8)
        d.text(1440,y+45,f"{auto}/10",33,TEAL)
    d.text(800,525,"Colored blocks = automatic decisions; pale blocks = review. Numbers are invented.",28,INK,"middle")
    keep(d)
    keep(flow("agent-loop", ["Plan a step", "Choose an action", "Run a tool", "Read the result"], ["A generative model can plan", "Jev can help with bounded choices", "Code enforces allowed operations", "Continue or stop"]))
    d=Drawing("langgraph", "A graph has state, nodes doing work, and edges controlling the route.")
    d.node(60,160,360,175,"Shared state","Message + answers")
    d.arrow(445,245,540,245);d.node(570,160,350,175,"Classify node",fill=PALE)
    for i,(label,c) in enumerate([("Technical node",MINT),("Billing node",PALE),("Review node",PEACH)]):
        y=15+i*175;d.arrow(945,245,1100,y+65);d.node(1130,y,400,135,label,fill=c)
    keep(d)
    d=Drawing("trace-eval", "An agent trace records request, tool call, evidence, and final answer; a judge evaluates focused questions.")
    for i,label in enumerate(["User request","Tool call","Tool evidence","Final answer"]):d.node(70,15+i*130,660,105,label,fill=PALE)
    d.arrow(755,260,850,260)
    d.node(885,70,650,340,"Judge the recorded run","Did it use evidence? Did it meet the request? Did it stay within permissions?",MINT)
    keep(d)
    d=Drawing("judge-study", "LangChain's early study had five distinct captured weather runs, each repeatedly judged; repeats do not create new tasks.")
    for i in range(5):d.node(80+i*305,80,270,175,f"Fixed run {i+1}","100 repeated judgments",[PALE,MINT,PEACH,PALE,MINT][i])
    d.text(800,375,"500 judgments, but only 5 distinct agent runs",46,BLUE,"middle",600)
    d.text(800,460,"Repeatability and agreement on new tasks answer different questions.",33,INK,"middle")
    keep(d)
    d=Drawing("candidate-scoring", "An illustrative open-model design scores state, question, and each option with one shared scorer; it is not Jev's disclosed design.")
    for i,opt in enumerate(["technical description","billing description","information description"]):
        y=30+i*150;d.node(60,y,770,115,f"State + question + {opt}",fill=[PALE,MINT,PEACH][i]);d.arrow(850,y+55,965,y+55);d.node(990,y,300,115,"Shared scorer");d.text(1340,y+69,["2.0","0.5","−1.0"][i],34,BLUE)
    d.text(800,535,"Combine scores into probabilities. Matching the API does not establish matching quality.",28,INK,"middle")
    keep(d)
    d=Drawing("multi-agent", "A coordinator assigns tasks to specialist agents, then evaluates evidence before accepting results.")
    d.node(60,165,370,180,"Coordinator")
    for i,label in enumerate(["Search specialist","Coding specialist","Explanation specialist"]):
        y=20+i*175;d.arrow(455,250,595,y+60);d.node(620,y,470,130,label,fill=[PALE,MINT,PEACH][i]);d.arrow(1115,y+60,1220,250)
    d.node(1250,165,280,180,"Check evidence",fill=MINT)
    keep(d)
    keep(flow("research-loop", ["Ask a testable question", "Keep a held-out test set", "Compare alternatives", "Inspect the failures"], ["What could improve?", "Prevent tuning on the answer key", "Measure quality, latency, cost", "Learn where the idea breaks"]))
    keep(flow("lesson-path", ["Understand representations", "Read typed decisions", "Build a routed workflow", "Design a fair experiment"], ["Notebook: Steps 1–3", "Notebook: Steps 4–7", "Notebook: Steps 8–11", "Notebook: Steps 12–14"]))
    from slide_design import refine_drawings
    ds.update(refine_drawings(Drawing, (INK, BLUE, TEAL, ORANGE, PAPER, PALE, MINT, PEACH)))
    return ds


HISTORY="https://magazine.sebastianraschka.com/p/classifier-history-and-jev"
DOC="https://docs.typesafe.ai/"
PAPERS={"transformer":"https://arxiv.org/abs/1706.03762","bert":"https://arxiv.org/abs/1810.04805","cnn":"https://arxiv.org/abs/1408.5882","ulmfit":"https://arxiv.org/abs/1801.06146","t5":"https://arxiv.org/abs/1910.10683","modernbert":"https://arxiv.org/abs/2412.13663"}


def content():
    slides=[]
    def add(chapter,title,lead,diagram,takeaway,notes,source="",source_title="Source"):
        slides.append(dict(chapter=chapter,title=title,lead=lead,diagram=diagram,takeaway=takeaway,notes=notes,source=source,source_title=source_title))
    add("Start here","From Bag-of-Words to Jev","How can software turn a message into a useful decision? We will build the answer one step at a time.","title-flow","Language models for text classification · A visual tutorial","Begin with an everyday message, not a definition of intelligence. Ask students to name the team they would contact if a laptop would not start, then ask what information they used. The rest of the lesson turns that familiar judgment into a software workflow.")
    add("Start here","An inbox needs a destination","A school help desk receives different requests. Classification assigns each message to a category the software can use.","inbox","A category is also called a class; its name is a label.","Have students route the three messages before introducing a model. These categories were chosen by the application designer. If a message fits none of them, the application needs an explicit fallback rather than silently assuming the closest category is correct.")
    add("Start here","Choosing a label and writing a reply","A classifier can choose “technical.” A generator can write troubleshooting advice. An application may need both capabilities.","classify-generate","Define the output you need before choosing a model.","The distinction is about the task and output interface. A generative language model can also perform classification, including with constrained structured output. It may still generate output tokens to do so. A classifier does not automatically know how to write a useful explanation.")
    add("Start here","Examples provide an answer key","In supervised learning, people supply example texts and their labels. The model learns patterns that help predict labels for new texts.","examples-labels","A training example pairs an input with a known answer.","Introduce supervised as learning with an answer key. Labels should match a written definition, and people can disagree about ambiguous examples. Training does not store a perfect rule for every possible message; we must inspect behavior on unseen examples.")
    add("Start here","The same four steps keep returning","We collect examples, represent the text, learn a decision rule, and check it on new examples. The representation changes across generations of models.","pipeline","A representation is a form of the input that the model can compute with.","Use this slide as a map. Bag-of-words makes counts, embeddings make learned vectors, and contextual models use neighboring text. In Jev's hosted API, the user supplies the task descriptions instead of training a new model in this notebook.")
    add("Start here","A timeline of useful ideas","These milestones changed how models represent language and return answers. Earlier methods remain useful when their strengths match the task.","timeline","A newer architecture does not guarantee a better result on your dataset.","The timeline groups selected developments rather than claiming each idea appeared only once or replaced everything before it. Transformers arrived in 2017, BERT in 2018, T5 in 2019, ModernBERT in 2024, and Jev's launch and these tutorials belong to 2026. ULMFiT demonstrates that transfer learning was useful even with recurrent architectures.",HISTORY,"Historical overview")
    add("01 · Count words","First, decide what counts as a token","For our first model, we lowercase the text and split it into words. Real tokenizers may also split words into smaller pieces.","tokens","A token is one unit produced by the tokenizer.","Tokenization is a design choice. Punctuation, spelling, and contractions affect the units we produce. Our deliberately simple tokenizer makes the first experiment inspectable, but its limitations are part of the lesson.","https://scikit-learn.org/stable/modules/feature_extraction.html","Text feature extraction")
    add("01 · Count words","Give every known word a position","A vocabulary is the list of words found in the training examples. Each word gets a fixed position in a list of numbers.","vocabulary","Fit the vocabulary using training examples only.","Call the numerical list a vector after showing the positions. A future message uses the same positions. Words outside this vocabulary need a handling rule; our small model ignores them, which can erase useful information.")
    add("01 · Count words","Count into those positions","The message “screen screen frozen” becomes a vector. Most positions are zero because the message uses only a few vocabulary words.","counts","Bag-of-words represents a whole document by its word counts.","Ask a student to verify each entry. The vector stays the same length regardless of message length. A real vocabulary could contain many thousands of entries, so implementations often store only the nonzero counts.")
    add("01 · Count words","Naive Bayes combines word evidence","It estimates how common each word is within each class, then combines those clues with how common the class itself is.","naive-bayes","The notebook builds this small classifier with ordinary Python.","Prior means the class frequency before reading the message. Conditional probability asks how often a word appears within a class. Multiplying word evidence assumes the words are conditionally independent given the class; language violates that assumption, but the model can still be a useful baseline. Our code uses logarithms and smoothing.","https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html","Naive Bayes reference")
    add("01 · Count words","Another option is weighted evidence","Logistic regression learns a weight for each word count. Adding those contributions gives an evidence score, which becomes a probability estimate.","weighted-clues","The shown weights are invented to illustrate the calculation.","Weights are learned from labeled examples, not supplied by a person in ordinary training. Positive and negative weights push toward different labels. The bias is a starting offset. The logistic curve is a smooth conversion from any real score to a value between zero and one; that value still needs calibration checks.")
    add("01 · Count words","Some words are more informative","TF-IDF adjusts counts so words that appear almost everywhere usually receive less influence than words that distinguish documents.","tfidf","Frequency and usefulness are different ideas.","TF stands for term frequency; IDF stands for inverse document frequency. This changes the features, not the classifier architecture. The vocabulary and document-frequency statistics must come from training data. A rare word is not automatically relevant, and negation words can be important even if common.","https://scikit-learn.org/stable/modules/feature_extraction.html","TF-IDF documentation")
    add("01 · Count words","Counting loses the sentence order","These sentences describe different situations, yet they contain the same words with the same counts. A count-only classifier cannot distinguish them.","order-loss","If the representation loses a distinction, the next model cannot recover it.","Ask who is helping whom in each sentence. The limitation lies in the features before classification begins. This is an information-loss problem, not simply a matter of training longer or using a bigger classifier.")
    add("01 · Count words","Word pairs restore a little order","A bigram is a pair of neighboring tokens. Adding these pairs can help distinguish phrases such as “not helpful” from “very helpful.”","bigrams","Useful features can improve a simple model without replacing it.","Bigrams expand the vocabulary, so they can create more rare features and require more examples. They preserve short local relationships but do not capture all sentence structure. Students can add bigrams to the notebook as an exercise.")
    add("01 · Count words","Keep an honest test set","Learn from training data, choose settings on development data, and evaluate the final choices on test data that stayed separate.","data-splits","A test set loses its purpose if you repeatedly tune to its answers.","Development and validation are often used for similar roles. Explain this as studying with practice questions before opening the final exam. Repeated messages and messages from one conversation should stay in the same split when that could leak information.")
    add("02 · Learn representations","Embeddings replace isolated counts","An embedding represents a word or token with a learned vector. Words used in similar contexts can acquire similar representations.","embeddings","A two-dimensional drawing simplifies a vector with many coordinates.","The points here are invented, not extracted from a trained model. The coordinates are numerical features learned through an objective, and usually do not have simple human names. A word embedding represents a token, while a bag-of-words vector represents a whole document.")
    add("02 · Learn representations","Context changes what a word means","A static embedding gives “bank” one lookup vector. A contextual model can build different representations from the words around it.","context-bank","Meaning depends on the sentence, not only the word.","Ask which clues identify a river edge and which identify a financial institution. Static embeddings can feed a later sequence model that adds context; their lookup vectors alone are context independent. Avoid claiming each learned coordinate corresponds to a dictionary meaning.",PAPERS["bert"],"BERT paper")
    add("02 · Learn representations","An RNN carries a running state","A recurrent neural network reads tokens in order. Each step updates a numerical state using the new token and the state from the previous step.","rnn","The same update rule is reused at every position.","The diagram is unrolled: four boxes do not mean four separately learned networks. The state is a fixed-size numerical summary. Reordering words changes the sequence of updates, although important earlier information can still become hard to retain.")
    add("02 · Learn representations","Gates help retain useful information","An LSTM uses learned gates to control which earlier information stays and which new information enters the state.","lstm","LSTM means long short-term memory; it is a kind of RNN.","A gate is a numerical operation that scales information. It does not consciously decide what to remember. LSTMs made sequence learning more practical, but finite state capacity and sequential processing still matter.")
    add("02 · Learn representations","A CNN looks for local patterns","A convolutional filter reads nearby tokens with the same learned weights at each position. Different filters can learn different clues.","cnn","The responses shown are invented, not trained measurements.","This is a one-dimensional text convolution over token embeddings. A three-token window moves one position at a time. Filters can recognize local phrase patterns, and several positions can be processed in parallel. More layers can expand the receptive field.",PAPERS["cnn"],"Text CNN paper")
    add("02 · Learn representations","Pooling keeps a compact summary","Global max pooling keeps the strongest response from each filter. That gives the next layer a fixed-size input even when texts have different lengths.","pooling","A classification head is the small final component that predicts the labels.","The maximum is a summary that loses exact position and some weaker signals. With multiple filters, keep one maximum per filter, then pass that vector to the head. Other pooling choices are possible; max pooling is one inspectable example.")
    add("02 · Learn representations","Learning can transfer between tasks","Pretraining learns reusable language patterns from much text. Fine-tuning then adapts the model to a smaller labeled classification dataset.","transfer","ULMFiT showed this approach with an RNN in 2018.","Pretraining and fine-tuning are different phases, often with different objectives. Transfer learning does not require transformers. Good pretraining can help when labels are scarce, but the adapted model still needs evaluation on the target task.",PAPERS["ulmfit"],"ULMFiT paper")
    add("03 · Use context","Attention connects relevant tokens","Attention gives each token a way to mix information from other positions. The resulting representation can reflect clues elsewhere in the text.","attention","The links illustrate learned mixing; they are not measured attention weights.","A transformer combines attention with other learned operations and position information. Multiple layers progressively change representations. Drawing thick lines is a teaching simplification: attention weights do not automatically explain a model's full decision or establish human-like understanding.",PAPERS["transformer"],"Transformer paper")
    add("03 · Use context","Two useful ways to read context","An encoder usually allows a token to use context on both sides. A causal decoder restricts attention to the current and earlier positions.","attention-masks","The attention mask controls which token positions can exchange information.","Rows represent the token doing the reading; columns represent tokens it can attend to. The triangular mask supports predicting the next token without seeing future tokens. An encoder's unrestricted mask is useful for representing complete input text.",PAPERS["transformer"],"Transformer paper")
    add("03 · Use context","BERT reads before classifying","A BERT-style encoder builds contextual representations. After task training, a classification head can turn a summary representation into label probabilities.","bert","ModernBERT updates the encoder approach rather than abandoning it.","The special CLS position can summarize the input through attention. A head trained for your label set predicts task categories. ModernBERT modernizes bidirectional encoders for efficiency and longer context; we are not benchmarking it in the notebook.",PAPERS["modernbert"],"ModernBERT paper")
    add("03 · Use context","A decoder can generate a label","A prompted generative model can read a review and produce “positive.” It reaches that output by generating tokens, even when the answer is very short.","decoder","Constrained output can enforce structure, while the underlying model still generates.","Zero-shot means no labeled demonstrations in the request, not no training in the model's history. Few-shot means including a few task examples. A generated label's token probability is not automatically a calibrated probability of a semantic class.")
    add("03 · Use context","A decoder can also use a small head","A causal language model can be adapted for classification. Use a position that has seen the full input, then train the head and appropriate model parameters.","decoder-head","A classification head offers another route besides prompting for label text.","The first token in a causal model cannot see later words, so it is an unsuitable full-text summary without other changes. The final non-padding token can attend to earlier input. This is a standard architectural adaptation, not a statement about Jev's proprietary internals.")
    add("03 · Use context","T5 treats tasks as text in, text out","An encoder reads the input, and a decoder writes the answer. A classification answer can be a short label expressed as text.","t5","T5 is an encoder–decoder model, introduced in 2019.","The original transformer also used an encoder and decoder for translation. T5 combines that style with text-to-text task formatting and transfer learning. Its decoder can produce a class label, or a separate classifier can use representations.",PAPERS["t5"],"T5 paper")
    add("03 · Use context","Choose the approach for the job","A small classifier, a fine-tuned encoder, a generative model, and a hosted decision model solve overlapping tasks with different practical tradeoffs.","design-menu","Compare quality on your data along with setup effort, latency, and cost.","Avoid a universal ranking. A narrow trained model can be excellent for a stable task at high volume. A generator is needed for open writing. A decision API can reduce per-task training work, but it must be checked for task fit, ambiguous inputs, and failure cases.")
    add("04 · Meet Jev","System One is a decision interface","TypeSafe uses “System One” for models that evaluate a state and return typed decisions. Jev is its flagship model in this category.","system-one","Here, “System One” is a product category, not a claim that the model has a human mind.","The state is the relevant evidence supplied to the model; questions define the decisions. The application retains control over what actions follow. The name borrows the idea of fast judgment, but does not establish a cognitive architecture.",DOC+"concepts/system-one","System One documentation")
    add("04 · Meet Jev","Separate the interface from the internals","We can inspect the API and TypeSafe's stated training goal. The complete architecture, data, and training procedure remain proprietary.","known-unknown","An encoder-like diagram would be a hypothesis about Jev, not a verified description.","Raschka discusses possible implementations but explicitly marks them as guesses. TypeSafe calls its method reinforcement learning for calibrated decisions, or RLCD. A public method called RLCR is related by topic, but no published connection establishes that it is Jev's method.",HISTORY,"Architecture discussion")
    add("04 · Meet Jev","Choice selects one described option","Give each option a clear description. The answer includes the selected option, a probability for every option, and a confidence value.","choice","These three probabilities are invented and sum to 1.","The option descriptions matter more than their short programmatic labels. Choice is single-label even when the message has several properties. Include an other or insufficient-context option when your task needs one, then test its behavior.",DOC+"primitives/choice","Choice documentation")
    add("04 · Meet Jev","Noul asks a yes-or-no question","A Noul returns the model's estimated probability that the answer is yes. Values near 0.5 suggest that the two answers are hard to separate.","noul","Noul has no separate confidence field in the documented response.","A value near zero is a strong no, not simply low confidence. The application can use a low boundary and a high boundary, reviewing values between them. Several different Noul questions can all have large yes probabilities because they describe different events.",DOC+"primitives/noul","Noul documentation")
    add("04 · Meet Jev","Several properties can be true together","Use one Choice for a single topic, and separate Noul questions for properties such as urgency and frustration. Their probabilities do not need to sum across questions.","multi-label","Single-label categories and multi-label properties answer different questions.","The probability distribution belongs to each individual question. Topic billing and property urgent are compatible. Separately evaluated questions can still be statistically correlated because they share an input; independent evaluation does not prove independence between events.",DOC+"primitives","Question types")
    add("04 · Meet Jev","Score places an input on a rubric","Define ordered levels with concrete descriptions. The returned score is the average level number weighted by the probabilities, so it can be fractional.","score","The rubric positions are 0, 1, and 2; their descriptions define the meaning.","A score of 1.5 is a position between rubric levels. It is not a precise count, percentage, or physical quantity. Two different distributions can have the same average, so inspect the distribution as well as the scalar score. Score confidence accounts for distance between levels.",DOC+"primitives/score","Score documentation")
    add("04 · Meet Jev","Confidence differs from probability","For Choice, confidence summarizes how far the largest probability sits above an even split. It is not another estimate that the prediction is correct.","confidence","A concentrated distribution can still be confidently wrong.","With three options and top probability 0.8, the documented formula gives 0.7 confidence. This is a computed statistic, not an independently verified correctness probability. Score uses a different statistic because its levels have an order. Calibration must be evaluated with actual outcomes.",DOC+"confidence","Confidence formulas")
    add("04 · Meet Jev","Calibration is checked over many cases","If events given probability 0.8 occur about 80% of the time on representative data, those estimates are calibrated around that probability.","calibration","Accuracy asks which label wins; calibration asks whether the probabilities match outcomes.","The ten circles are only a teaching illustration. Small groups have substantial sampling noise, so real checks need much more data. Calibration can change under distribution shift. In the notebook we use synthetic probabilities to explain the calculation without claiming a result for Jev.","https://arxiv.org/abs/1706.04599","Calibration research")
    add("04 · Meet Jev","RLCD names the training goal","TypeSafe says RLCD trains decision outputs and calibrated probabilities. The exact reward design and optimization recipe are not public.","training-goal","This diagram explains the goal; it does not reconstruct RLCD.","Reinforcement learning uses feedback to improve behavior according to an objective. Correct labels and reliable uncertainty are related but distinct targets. We can study public calibration methods as research ideas without calling them Jev's training procedure.",DOC+"introduction/machine-learning-primer","TypeSafe AI primer")
    add("05 · Build a workflow","Ask narrow questions together","Jev evaluates multiple questions about a shared state in one request. Your code can combine the returned answers into a visible policy.","parallel","A question that needs an earlier answer requires a later workflow step.","TypeSafe documents parallel, separate question evaluation. Shared-state batching can avoid repeatedly sending the same context, but it does not promise unlimited throughput or zero extra latency. Do not write one question that depends on a sibling question's answer. Combine returned values in code or make another call with updated state.",DOC+"patterns/fan-out","Batching pattern")
    add("05 · Build a workflow","Supply relevant facts in the state","Text and JSON can provide a message, current policy, and verified facts. Keep instructions in the questions and point them at the fields they should judge.","structured-state","Structured inputs help clarify what evidence belongs to each decision.","The model only receives the context supplied in the request, plus whatever prior knowledge its training produced. JSON does not prove that a factual field is true. Include trusted current facts when needed and reduce irrelevant content. Images and audio require preprocessing into text for this documented Jev version.",DOC+"primitives/advanced","Structured questions")
    add("05 · Build a workflow","Write questions with clear boundaries","A useful question defines one decision and gives concrete criteria. Missing context and overlapping categories often need review or clarification.","good-questions","Evaluate changes on labeled examples rather than judging them by higher confidence alone.","Have students compare which team with is this good. The latter hides many possible dimensions. Test negation, quoted instructions, mixed requests, unseen categories, and option reordering. TypeSafe's version-specific limitations also recommend keeping arithmetic in code.",DOC+"concepts/how-to-build-with-system-one","Workflow design guide")
    add("05 · Build a workflow","A request becomes ordinary data","Send model, state, and questions to the API. Read the typed answers, then let explicit code choose the handler.","request-response","The notebook includes HTTP, Python SDK, and LangChain examples.","An API is a software interface. JSON packages fields as data; a POST sends the request. The current direct Python package is typesafe-sdk and its import is typesafe_sdk. LangChain's adapter has its own response convenience fields, so distinguish those from raw HTTP JSON. An API key is needed only for live calls.",DOC+"introduction/quickstart","TypeSafe quickstart")
    add("05 · Build a workflow","Give uncertainty a route","If the decision is clear enough for the task, send it to the chosen handler. Otherwise ask for more information or send it for review.","routing","The threshold is a policy choice that needs development-set evidence.","A confidence threshold cannot certify correctness. Consequences, calibration, and fallback quality all influence the policy. The notebook routes messages without performing real external actions. Exact permissions and numeric calculations belong in deterministic code.",DOC+"patterns/confidence-routing","Confidence routing")
    add("05 · Build a workflow","More review changes the workload","Raising a threshold often reduces automatic coverage. Measure how many cases remain automatic and how often those decisions are correct.","coverage","A high threshold does not guarantee that accuracy improves.","Coverage is the fraction automatically handled. Selective accuracy measures correctness among those accepted cases. A classifier that sends everything to review has zero coverage and no automatic accuracy to report. The graphic shows invented counts; the notebook measures this on its synthetic examples.",DOC+"patterns/intent-routing","Intent routing")
    add("06 · Connect to agents","An agent chooses steps toward a goal","An agent can plan, call tools, and inspect results. Jev can help with bounded decisions inside that loop, while generative models handle open-ended work.","agent-loop","A fast decision component still needs a reliable application around it.","A harness is the surrounding code that controls the agent's tools, state, and permissions. Model routing and tool-risk classification are examples in LangChain's article. A semantic risk classifier is one signal; code must still enforce the actions the user actually authorized.","https://www.langchain.com/blog/building-a-harness-with-jev","Jev in an agent harness")
    add("06 · Connect to agents","A graph makes the workflow visible","LangGraph represents work as nodes connected by edges. State carries the message and results, and conditional edges select the next node.","langgraph","A node may call Jev, run ordinary code, or invoke another model.","The graph can make a workflow easier to inspect and resume. The diagram's edges specify routing; they do not mean the model independently controls the graph. The notebook builds a small graph with inert handlers that return messages rather than contacting a service.","https://www.langchain.com/blog/building-prod-with-jev-and-langgraph","Production workflow example")
    add("06 · Connect to agents","Judge the recorded behavior","A trace records what an agent was asked, what it did, and what evidence it received. Focused questions can evaluate whether the final answer used that evidence.","trace-eval","A judge's score is a model prediction that also needs validation.","Code checks can verify exact fields or tool names. A semantic judge can assess whether the answer matches the retrieved evidence. Jev returns typed judgments without a written rationale. LangSmith can store these as feedback keys for inspection and monitoring, but the observer still needs human-labeled references.","https://www.langchain.com/blog/jev-is-now-available-in-langsmith-evals","LangSmith evaluation setup")
    add("06 · Connect to agents","Read early evidence at its real scale","LangChain reported strong repeatability and agreement in a small weather-agent judge study. Repeating a fixed run tests stability, not breadth across new tasks.","judge-study","The study's observations are useful evidence, not a universal ranking of judges.","The study captured five weather-agent runs and repeated judgment 100 times per run. Jev matched the human binary labels in those repetitions, with low observed score variance. These are not 500 independently sampled tasks. The article notes observational results, provider defaults, and missing Jev service-version metadata.","https://www.langchain.com/blog/jev-agent-evals-langsmith","Early judge study")
    add("07 · Investigate","A similar API is easier than a similar model","An open-model experiment could score each candidate description with a shared scorer. That illustrates one design, but says nothing definite about Jev's internals.","candidate-scoring","Matching an interface does not demonstrate generalization or calibration.","The shared scorer receives state, task description, and one candidate. A normalization step can turn scores into a distribution over candidates. Novel task evaluation and calibration require training data, held-out tasks, and careful testing. The shown numbers are invented, and the diagram is an illustrative research design.",HISTORY,"Open-model design discussion")
    add("07 · Investigate","Could a coordinator route specialist agents?","A decision model might assign work and check handoffs between agents. Research should measure the whole team's success, not only the coordinator's label accuracy.","multi-agent","Shared mistakes can make several agreeing agents confidently wrong.","This is a research proposal, not a validated result. Compare the same specialists under fixed routing, model-driven routing, and uncertainty-based escalation. Hold the tasks, tool budgets, and retry limits constant. Measure final success, cost, latency, and handoff failures, and check whether agents share failure modes.")
    add("07 · Investigate","Turn an idea into a fair experiment","Choose a hypothesis, separate development from testing, compare alternatives, and inspect the cases where the approach fails.","research-loop","The notebook and research roadmap give six concrete projects.","Possible projects study confidence under shift, multi-agent routing, judge alignment, candidate-set completeness, learned composition of atomic signals, and open-model decision training. Each needs a baseline and a useful failure analysis. Keep test tasks untouched while designing rubrics or an automated research loop.")
    add("Try it yourself","Now build the decision path","The companion notebook starts with a small word-count classifier, then teaches typed decisions, routing, agent evaluation, and research design.","lesson-path","Open notebooks/system1_tutorial.ipynb and run the offline path first.","Ask students to explain one representation choice, one API distinction, and one fair-test requirement. They should now be able to distinguish what a model predicts from what application code permits. The notebook contains exercises, worked answers, optional live calls, and source links.")
    return slides


def refine_content(slides):
    """Keep each slide understandable before consulting its teaching notes."""
    revisions={
        'title-flow':dict(lead='How does a computer turn a message into a useful decision? Follow one help-desk example from simple word counts to contextual models and Jev.'),
        'naive-bayes':dict(lead='Begin with how common each class is, then multiply by the word evidence within that class. These invented numbers show how “screen frozen” favors technical help.',takeaway='The result is a comparison score; divide by the total to obtain probabilities.'),
        'weighted-clues':dict(lead='For a two-class decision, logistic regression adds weighted word counts and a starting offset. A smooth curve converts that evidence score into a probability.',takeaway='Training learns the weights. The numbers here illustrate the calculation.'),
        'tfidf':dict(lead='A word appearing in every training document tells us little about which document is different. TF-IDF combines its frequency here with its rarity across documents.',takeaway='Rarity can make a useful clue stand out, but it does not guarantee relevance.'),
        'rnn':dict(lead='A recurrent neural network, or RNN, reads one token at a time. Each update combines that token with a numerical memory of the earlier tokens.',takeaway='The boxes show repeated uses of the same learned update rule.'),
        'lstm':dict(lead='An LSTM is an RNN with learned gates: numerical controls that keep some old information and add new information. This helps useful clues survive across many steps.'),
        'cnn':dict(lead='A convolutional neural network, or CNN, slides a learned pattern checker across small token windows. The same filter produces a response at each position.',takeaway='The responses are invented. The shared filter can learn useful local phrases.'),
        'pooling':dict(lead='A filter produces several responses, even when messages have different lengths. Max pooling keeps its strongest response; other filters contribute their own summaries.',takeaway='A classification head is the final small layer that converts these clues into label scores.'),
        'attention':dict(lead='To update “bank,” attention combines information from token representations using learned weights. Context such as “river” and “muddy” can help shape the result.',takeaway='These weights illustrate mixing; they are not measurements or a proof of reasoning.'),
        'attention-masks':dict(lead='Read a row to see which positions a token may use. An encoder can read both sides; a causal decoder can read the current position and earlier ones.',takeaway='For “bank,” the causal mask blocks “was” and “muddy,” which come later.'),
        'bert':dict(lead='An encoder builds representations using context on both sides. In a BERT-style classifier, a summary position called [CLS] feeds a small layer trained to predict your labels.'),
        'decoder':dict(lead='A causal decoder predicts the next token from the earlier context. Given “classify this review,” a generative model can write a label such as “positive.”',takeaway='A label written as text is still produced through token generation.'),
        'decoder-head':dict(lead='A causal model can also feed a small trained classifier. Its final input position has access to the full message, so that representation can support a label prediction.'),
        'system-one':dict(lead='TypeSafe calls this a System One interface: give a model the relevant evidence and a few narrow questions, then receive typed decisions. Jev is its flagship model.',takeaway='The model predicts an answer. Your code decides what happens next.'),
        'known-unknown':dict(lead='We can explain Jev’s public inputs, outputs, and stated training goal. The full architecture and training procedure have not been disclosed.',takeaway='A useful interface description is different from a verified diagram of the model’s internals.'),
        'noul':dict(lead='Noul estimates whether one statement is true. A value near 1 favors yes, near 0 favors no, and near 0.5 leaves the answer uncertain.',takeaway='Noul is P(yes) for one question; it does not include a separate confidence field.'),
        'score':dict(lead='Score uses ordered levels that you describe in a rubric. It averages their positions using the predicted probabilities, so the returned value can fall between levels.',takeaway='A score of 1.5 is an expected rubric position, rather than a new category.'),
        'confidence':dict(lead='For three Choice options, an even split gives each one ⅓. Jev’s confidence rescales the top probability relative to that baseline; it is a different number from P(top option).',takeaway='Choice confidence depends on how many options you offer and still needs validation.'),
        'training-goal':dict(lead='TypeSafe describes training Jev for correct decisions and probabilities that reflect uncertainty. It names the approach RLCD; its exact training procedure remains proprietary.'),
        'routing':dict(lead='Application code compares Choice confidence with a chosen threshold. Here 0.70 clears an illustrative 0.60 gate, so the message reaches its selected handler.',takeaway='A threshold is a policy setting. Choose it using development-set errors and review capacity.'),
        'coverage':dict(lead='Raising a confidence threshold sends more cases to review and fewer to automatic handlers. Compare those fractions with error rates to decide whether the tradeoff helps.',takeaway='Coverage measures how often you act automatically; it does not measure correctness.'),
        'agent-loop':dict(lead='An agent takes steps, runs tools, and uses the results to choose what to do next. Jev can judge bounded choices while application code controls the loop and permissions.',takeaway='Tool results become evidence for the next step. Completion and budgets stop the loop.'),
    }
    for slide in slides:
        slide.update(revisions.get(slide['diagram'],{}))
    additions={
        'naive-bayes':'The two rows are a two-class toy calculation; the numerical scores are deliberately invented and have not yet been normalized.',
        'weighted-clues':'This is a binary example, such as technical versus other. Ask students to check the two occurrences of screen before adding the contributions.',
        'attention-masks':'Have students read across the outlined bank row. In the decoder panel, The, river, and bank are available; was and muddy are blocked. The check marks indicate permission, not actual attention strength.',
        'confidence':'Have students compare the even split with the top probability before evaluating the rescaling formula. Changing the candidate set changes this baseline, so compare configurations carefully.',
        'coverage':'The letter A marks an automatic route and R marks review. The diagram shows the same ten invented cases at each threshold, so the accepted set shrinks as the gate rises. It contains no correctness labels.',
    }
    for slide in slides:
        if slide['diagram'] in additions:slide['notes']+=' '+additions[slide['diagram']]


def build():
    from slide_design import CSS, JS
    ASSETS.mkdir(parents=True,exist_ok=True)
    drawings=make_drawings()
    for name,svg in drawings.items():(ASSETS/f"{name}.svg").write_text(svg)
    slides=content()
    refine_content(slides)
    (OUT/"deck.json").write_text(json.dumps(slides,ensure_ascii=False,indent=2)+"\n")
    fontcss=""
    for name,filename,weight in [("Bricolage Grotesque","bricolage-latin.woff2","400 800"),("Source Sans 3","source-sans-latin.woff2","200 900")]:
        p=ASSETS/filename
        if p.exists():fontcss+=f"@font-face{{font-family:'{name}';font-style:normal;font-weight:{weight};font-display:swap;src:url(data:font/woff2;base64,{base64.b64encode(p.read_bytes()).decode()}) format('woff2')}}\n"
    base=(ROOT/"docs/slides/viewport-base.css").read_text()
    sections=[]
    chapters=list(dict.fromkeys(s['chapter'] for s in slides))
    chapter_colors=[MINT,PEACH,PALE,'#eee9f3',MINT,PEACH,PALE,'#eee9f3',PALE]
    layouts={
        'timeline':'timeline','attention-masks':'matrix','tfidf':'matrix',
        'coverage':'matrix','counts':'vector','vocabulary':'vector',
        'choice':'chart','calibration':'chart','weighted-clues':'calculation',
        'score':'calculation','confidence':'calculation','noul':'scale',
        'embeddings':'map','cnn':'matrix','rnn':'sequence','attention':'network',
        'parallel':'network','multi-agent':'network','langgraph':'network',
        'routing':'branch','trace-eval':'sequence','request-response':'comparison',
        'classify-generate':'comparison','known-unknown':'comparison',
        'context-bank':'comparison','order-loss':'comparison','design-menu':'comparison',
        'structured-state':'code','agent-loop':'cycle'}
    for i,s in enumerate(slides):
        ci=chapters.index(s['chapter'])
        kickoff=i>0 and slides[i-1]['chapter']!=s['chapter']
        layout=layouts.get(s['diagram'],'flow')
        classes='opening' if i==0 else 'closing' if i==len(slides)-1 else 'chapter-start' if kickoff else ''
        rail=''.join(f'<span class="{"past" if j<ci else "current" if j==ci else ""}"></span>' for j in range(len(chapters)))
        source=f'<a class="source" href="{html.escape(s["source"])}" target="_blank" rel="noopener">{html.escape(s["source_title"])}</a>' if s['source'] else ''
        sections.append(f'''<!-- === SLIDE {i+1}: {html.escape(s['title'])} === -->
<section class="slide {classes} layout-{layout} {'active visible' if i==0 else ''}" data-layout="{layout}" data-slide-id="{s['diagram']}" data-chapter="{html.escape(s['chapter'])}" style="--chapter-color:{chapter_colors[ci]}" aria-label="Slide {i+1}: {html.escape(s['title'])}">
<div class="chapter-rail" aria-hidden="true">{rail}</div><span class="lesson-tag">Language → decisions</span>
<p class="eyebrow">{html.escape(s['chapter'])}</p><{'h1' if i==0 else 'h2'} class="reveal" data-edit="title">{html.escape(s['title'])}</{'h1' if i==0 else 'h2'}>
<p class="lead reveal" data-edit="lead">{html.escape(s['lead'])}</p><figure class="visual reveal">{drawings[s['diagram']]}</figure>
<p class="takeaway" data-edit="takeaway">{html.escape(s['takeaway'])}</p><span class="credit" title="Mui Research Group at ASDRP">@mui-group</span><span class="page">{i+1:02d} / {len(slides)}</span>{source}</section>''')
    options=''.join(f'<option value="{html.escape(c)}">{html.escape(c)}</option>' for c in dict.fromkeys(s['chapter'] for s in slides))
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="A visual high school tutorial on text classification, from word counts to Jev and System One workflows."><title>From Bag-of-Words to Jev</title><style>/* === EMBEDDED GOOGLE FONTS === */\n{fontcss}\n{base}\n{CSS}</style></head><body>
<div class="deck-viewport"><main class="deck-stage" id="deckStage">{''.join(sections)}</main></div>
<nav class="deck-controls" aria-label="Presentation controls"><button id="prev" aria-label="Previous slide">←</button><button id="counter" aria-live="polite" aria-haspopup="dialog"></button><button id="next" aria-label="Next slide">→</button><select id="chapters" aria-label="Jump to chapter">{options}</select><button id="notesToggle" aria-expanded="false" aria-controls="notes">Notes <span class="shortcut-hint">(N)</span></button><button id="more" aria-haspopup="dialog">More</button></nav>
<aside id="notes" hidden aria-label="Teaching notes"><div class="panel-header"><span class="panel-label">Teaching notes</span><button id="closeNotes" aria-label="Close teaching notes">✕</button></div><p id="notePosition" class="panel-meta"></p><h3 id="noteTitle"></h3><p id="noteBody"></p><a id="noteSource" target="_blank" rel="noopener"></a></aside>
<dialog id="overview" aria-labelledby="overviewTitle"><div class="dialog-head"><h2 id="overviewTitle">Find your place in the lesson</h2><button data-close aria-label="Close slide overview">✕</button></div><input id="slideSearch" type="search" aria-label="Search slides by title, chapter, or number" placeholder="Search a topic, chapter, or slide number…"><p id="overviewCount" aria-live="polite"></p><div id="slideGrid" class="slide-grid"></div></dialog>
<dialog id="tools" aria-labelledby="toolsTitle"><div class="dialog-head"><h2 id="toolsTitle">Presentation tools</h2><button data-close aria-label="Close presentation tools">✕</button></div><div class="tool-actions"><button id="allSlides">Slide overview <kbd>Esc</kbd></button><button id="present">Open presenter window <kbd>P</kbd></button><button id="full">Full screen</button><button id="motionToggle" aria-pressed="false">Static display <kbd>B</kbd></button><button id="editFromMenu">Edit this slide <kbd>E</kbd></button><button id="save">Save HTML copy</button><button id="resetEdits">Reset browser edits</button><button id="print">Print / save PDF</button><button id="helpToggle">Help and shortcuts <kbd>?</kbd></button></div><p id="uiStatus" class="panel-meta" role="status">Works offline. On a phone, landscape gives the slide more room.</p></dialog>
<dialog id="help" aria-labelledby="helpTitle"><div class="dialog-head"><h2 id="helpTitle">Read, teach, or present</h2><button data-close aria-label="Close help">✕</button></div><p>Each slide introduces one idea. Teaching notes add examples and source links; the notebook turns those ideas into code.</p><dl><dt>Arrows / Space</dt><dd>Go forward or back. Horizontal swipes also navigate.</dd><dt>Home / End</dt><dd>First or last slide.</dd><dt>Esc / slide count</dt><dd>Open the searchable overview. Esc also closes a panel or finishes editing.</dd><dt>N / Notes</dt><dd>Read the explanation beside the slide on a wide screen.</dd><dt>P</dt><dd>Open presenter notes with current and next previews. Move the original audience tab to the projector.</dd><dt>E</dt><dd>Edit headings, explanations, and takeaways. More → Save HTML preserves a copy.</dd><dt>B</dt><dd>Toggle a static display. Reduced-motion preferences are also respected.</dd></dl><p>On a phone, rotate to landscape for larger text. The slide itself always keeps its 16:9 layout. The PDF is convenient for reading with zoom.</p></dialog>
<div class="edit-hotzone" aria-hidden="true"></div><button id="editToggle" class="edit-toggle" aria-pressed="false">Edit (E)</button><span id="editStatus" hidden role="status"></span>
<script>const slideData={json.dumps(slides,ensure_ascii=False).replace('</','<\\/')} ;\n{JS}</script></body></html>'''
    (OUT/"index.html").write_text(page)
    print(f"Built {len(slides)} slides and {len(drawings)} original SVG diagrams.")


if __name__=="__main__":build()
