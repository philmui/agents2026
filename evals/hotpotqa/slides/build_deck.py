"""Build the self-contained teaching deck and its original SVG diagrams.

Run from any directory: python3 evals/hotpotqa/slides/build_deck.py
The small hand-authored diagrams are intentionally editable SVG, as requested.
"""
from pathlib import Path
from html import escape
import json

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)
TEAL, ORANGE, INK, PALE = '#006d70', '#9a480f', '#172b34', '#e6f1ed'

def svg(name, title, description, body, height=370):
    value = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 {height}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="{TEAL}"/></marker></defs><style>text{{font-family:Arial,sans-serif;fill:{INK};font-size:23px}}.small{{font-size:19px}}.title{{font-size:27px;font-weight:bold}}.accent{{fill:{TEAL};font-weight:bold}}</style>{body}</svg>'''
    (ASSETS / (name+'.svg')).write_text(value)

def box(x,y,w,h,label,sub='',fill=PALE):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{TEAL}" stroke-width="2"/><text x="{x+w/2}" y="{y+39}" text-anchor="middle" class="title">{escape(label)}</text>' + (f'<text x="{x+w/2}" y="{y+72}" text-anchor="middle" class="small">{escape(sub)}</text>' if sub else '')
def line(x1,y1,x2,y2):
    return f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{TEAL}" stroke-width="3" marker-end="url(#arrow)"/>'
def text(x,y,value,cls=''):
    return f'<text x="{x}" y="{y}" class="{cls}">{escape(value)}</text>'

svg('rag','Retrieval augmented generation','A question retrieves source paragraphs. A generator reads those paragraphs to produce an answer and citations. The evaluator checks retrieval and final output separately.',
    box(20,70,220,100,'Question','What is being asked?')+line(245,120,320,120)+box(325,70,245,100,'Retriever','Find source paragraphs')+line(575,120,650,120)+box(655,70,225,100,'Generator','Answer with citations')+text(335,230,'Evidence available')+text(660,270,'Evidence selected')+text(95,330,'Evaluation checks both the information found and the answer produced.'))
svg('bridge','A bridge question','The invented book Lantern Atlas identifies its creator Mira Vale. Mira Vale’s sentence identifies Harbor City as her birthplace. The bridge entity is Mira Vale.',
    box(20,70,250,105,'Lantern Atlas','creator is Mira Vale')+line(275,123,325,123)+box(330,70,240,105,'Mira Vale','born in Harbor City')+line(575,123,625,123)+box(630,70,250,105,'Harbor City','answer')+text(335,240,'Bridge entity', 'accent')+text(80,315,'Each hop supplies a fact needed by the next part of the question.'))
svg('comparison','A comparison question','Two invented records say River Theater opened in 1902 and Hill Theater in 1921. Compare dates to answer River Theater.',
    box(30,30,315,105,'River Theater','opened in 1902')+box(30,205,315,105,'Hill Theater','opened in 1921')+line(350,83,470,83)+line(470,83,470,165)+line(350,258,470,258)+line(470,258,470,200)+box(500,125,340,105,'Compare dates','1902 comes before 1921'))
svg('settings','Distractor and fullwiki retrieval','Distractor retrieval chooses from ten supplied paragraphs for each question, including two gold paragraphs. Fullwiki retrieval searches the Wikipedia corpus. The top ten in the fullwiki JSON are baseline retrieval results.',
    text(25,40,'Distractor', 'title')+text(475,40,'Fullwiki', 'title')+''.join(f'<rect x="{25+i%5*77}" y="{75+i//5*80}" width="62" height="58" rx="4" fill="{TEAL if i in (1,7) else PALE}" stroke="{TEAL}"/><text x="{56+i%5*77}" y="{112+i//5*80}" text-anchor="middle" style="fill:{"white" if i in (1,7) else INK}">{i+1}</text>' for i in range(10))+box(475,80,380,110,'Wikipedia corpus','Retrieve your own candidates')+text(25,290,'10 supplied paragraphs')+text(25,327,'2 gold + 8 distractors')+text(475,290,'Gold paragraphs can be missed')+text(475,327,'Top-10 JSON ≠ whole corpus'))
svg('coordinates','Original sentence coordinates','A paragraph title names the record. Sentence indices start at zero and stay attached to original sentences. The supporting fact is the exact title and index pair.',
    text(35,45,'Paragraph title: Mira Vale','title')+box(40,85,100,65,'0')+text(165,127,'Mira Vale was born in Harbor City.')+box(40,180,100,65,'1',fill='#fff0d7')+text(165,222,'A later sentence, if present, keeps index 1.')+text(55,325,'Citation coordinate: ["Mira Vale", 0]', 'accent'))
svg('boundary','The label boundary','Question and context flow into the agent. Answer and supporting fact labels flow into the evaluator. Only predictions flow from the agent to the evaluator.',
    box(20,30,280,100,'Agent input','question + context')+line(305,80,380,80)+box(385,30,210,100,'RAG agent','no gold labels')+line(490,135,490,235)+box(385,240,260,100,'Evaluator','compare predictions')+box(20,240,280,100,'Gold labels','answer + supporting_facts',fill='#fff0d7')+line(305,290,380,290)+text(660,285,'type / level:')+text(660,319,'analysis only'))
svg('paragraphs','Paragraph retrieval preserves sentences','Paragraphs are the ranking unit. Every paragraph preserves a list of numbered sentences. Selected paragraphs are passed to the model with these original coordinates.',
    box(20,70,260,105,'Paragraph A','title + all sentences')+box(330,70,260,105,'Paragraph B','title + all sentences')+box(640,70,240,105,'Paragraph C','title + all sentences')+text(45,240,'[A, 0]   [A, 1]')+text(355,240,'[B, 0]   [B, 1]')+text(665,240,'[C, 0]')+text(65,325,'Rank paragraphs first. Ask the generator to cite individual sentences.'))
svg('tfidf','Sparse TF-IDF similarity','Question terms and paragraph terms share a sparse vocabulary. Cosine similarity ranks the paragraph vectors. The example weights are illustrative and are not measured results.',
    text(25,45,'Shared vocabulary: lantern, atlas, creator, mira, vale','title')+text(30,110,'Question')+text(275,110,'[0.5, 0.5, 0.7, 0.0, 0.0]')+text(30,170,'Paragraph A')+text(275,170,'[0.4, 0.4, 0.0, 0.5, 0.5]')+text(30,230,'Paragraph B')+text(275,230,'[0.0, 0.0, 0.0, 0.7, 0.7]')+text(35,320,'Words absent from the question can make the second hop hard to rank.','accent'))
svg('graph','Bounded LangGraph RAG workflow','Retrieve then grade. Sufficient evidence goes directly to generation. Insufficient evidence rewrites and retrieves again only while rounds remain. At the round limit, grade routes directly to generation without a rewrite.',
    box(25,40,215,95,'retrieve','round += 1')+line(245,87,310,87)+box(315,40,245,95,'grade','Enough evidence?')+line(565,87,660,87)+box(665,40,210,95,'generate','answer + citations')+text(577,57,'yes','small')+line(435,140,435,225)+box(315,230,245,95,'rewrite','if rounds remain')+f'<path d="M310 277 H130 V140" fill="none" stroke="{TEAL}" stroke-width="3" marker-end="url(#arrow)"/>'+text(265,196,'no: retry','small')+f'<path d="M535 140 V210 H772 V140" fill="none" stroke="{ORANGE}" stroke-width="3" marker-end="url(#arrow)"/>'+text(588,193,'no: limit','small'),height=370)
svg('accumulate','Evidence across retrieval rounds','Round one retrieves paragraphs A and C. Round two retrieves paragraphs B and A. The union passed to generation is A, C and B with duplicates removed.',
    box(25,40,340,105,'Round 1','Lantern Atlas + Cloud Almanac')+box(25,210,340,105,'Round 2','Mira Vale + Lantern Atlas')+line(370,92,485,92)+line(485,92,485,155)+line(370,262,485,262)+line(485,262,485,205)+box(520,125,355,105,'Evidence union','Lantern Atlas + Cloud + Mira'))
svg('metrics','Three evaluation targets','Retrieval asks whether gold evidence was available. Supporting fact scoring asks whether the output selected the gold sentences. Answer scoring asks whether the short answer matches.',
    box(20,70,265,125,'Retrieval','What reached the generator?')+box(318,70,265,125,'Supporting facts','Which sentences were cited?')+box(615,70,265,125,'Answer','What did the system say?')+text(45,280,'Custom diagnostics')+text(345,280,'Official SP scores')+text(642,280,'Official answer scores')+text(325,340,'Joint scores combine answer and SP performance.'))
svg('recall','Gold paragraph recall example','The gold title set contains A and B. The retrieval contains A and C. One of the two gold titles was found, so recall is one half. All-gold success is zero.',
    text(30,50,'Gold titles','title')+box(30,75,230,95,'A')+box(290,75,230,95,'B')+text(30,230,'Retrieved titles','title')+box(30,255,230,95,'A')+box(290,255,230,95,'C',fill='#fff0d7')+text(595,125,'Recall = 1 / 2', 'accent')+text(595,280,'All-gold = 0', 'accent'))
svg('sp','Supporting fact set comparison','Gold supporting facts are A0 and B0. Predicted facts are A0 and C1. A0 is the true positive, C1 is an extra citation and B0 is missing.',
    text(35,55,'Gold facts','title')+text(490,55,'Predicted facts','title')+box(35,90,330,85,'[A, 0]','',PALE)+box(490,90,330,85,'[A, 0]','',PALE)+line(370,133,480,133)+box(35,230,330,85,'[B, 0]','', '#fff0d7')+box(490,230,330,85,'[C, 1]','', '#fff0d7')+text(35,355,'Missing: FN = 1')+text(490,355,'Extra: FP = 1'))
svg('diagnosis','A failure diagnosis sequence','Check retrieval coverage first. If gold evidence was available, check sentence citations. If citations are correct, inspect answer reasoning or answer formatting. The path gives likely failure categories, not causal proof.',
    box(25,70,250,105,'Evidence found?','Inspect retrieved facts')+line(280,122,325,122)+box(330,70,250,105,'Gold facts cited?','Inspect predicted SP')+line(585,122,630,122)+box(635,70,240,105,'Answer correct?','Inspect short answer')+text(50,250,'No: retrieval gap')+text(350,285,'No: evidence selection')+text(655,320,'No: reasoning / format'))
svg('paired','Paired experiment design','The same frozen question IDs feed configurations A and B. Both predictions use the same scorer, and per-question differences drive comparison.',
    box(25,125,250,110,'Frozen question IDs','same candidate contexts')+line(280,180,360,180)+line(360,180,360,80)+line(360,180,360,285)+box(400,30,210,105,'System A','k = 2')+box(400,235,210,105,'System B','k = 4')+line(615,80,720,80)+line(720,80,720,180)+line(615,285,720,285)+line(720,285,720,210)+text(750,170,'Paired')+text(750,205,'scores'))

SOURCES = {
 'hotpot':('HotpotQA project','https://hotpotqa.github.io/'),
 'paper':('Yang et al., 2018','https://aclanthology.org/D18-1259/'),
 'format':('Official JSON format','https://github.com/hotpotqa/hotpot/blob/master/README.md#json-format'),
 'eval':('Official evaluator','https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py'),
 'langgraph':('LangGraph agentic RAG','https://docs.langchain.com/oss/python/langgraph/agentic-rag'),
 'tfidf':('scikit-learn TF-IDF','https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting'),
}
slides=[]
def diagram(name,alt,large=False):
    return f'<a class="diagram-open" href="assets/{name}.svg" target="_blank" rel="noopener" aria-label="Open full-size diagram"><img class="diagram{" large" if large else ""}" src="assets/{name}.svg" alt="{escape(alt)}" width="900" height="370"></a><p class="diagram-caption">Open diagram for a larger view</p>'
def code(value):return '<pre><code>'+escape(value)+'</code></pre>'
def split(a,b,kind=''):return f'<div class="split {kind}"><div>{a}</div><div>{b}</div></div>'
def add(id,title,chapter,subtitle,body,takeaway,notes,sources=(),cls=''):
    slides.append(dict(id=id,title=title,chapter=chapter,subtitle=subtitle,body=body,takeaway=takeaway,notes=notes,sources=sources,cls=cls))

a='1. Understand the task'
add('start','Evaluating RAG with HotpotQA',a,'A step-by-step lab with LangGraph, sentence evidence, and scores you can inspect.',
    '<div class="hero-line"></div><p>Find the evidence. Answer the question. Measure each step.</p><p class="small">Module 22 · Basic Python and JSON · 60–90 minutes</p><p class="small">Offline teaching path + optional live language-model experiment</p>',
    'Open the companion notebook while following the lesson.',
    '<p>This lesson assumes you can read Python lists and dictionaries. You do not need prior experience with retrieval augmented generation. Spend about 20 minutes on the dataset, 25 minutes on retrieval and graph execution, and 25 minutes on scoring. Use the remaining time for the exercises.</p><p>Arrow keys move between slides. Explain opens detailed teaching notes. Read all shows a continuous lesson and Print produces one slide per page. Every diagram is a local, editable SVG. The offline path tests software and explains measurement. Its scripted answer cannot establish language-model quality.</p>',cls='cover')
add('rag','Retrieval augmented generation',a,'RAG gives an answer generator external text to use as evidence.',
    diagram('rag','Question goes to a retriever, then retrieved source paragraphs go to a generator. Evaluate retrieval separately from the final answer.')+'<p>A <strong>retriever</strong> selects passages. A <strong>generator</strong> answers using those passages.</p>',
    'A plausible answer alone cannot tell us whether retrieval worked.',
    '<p>Imagine a reference librarian finding useful pages before you write an answer. The retrieval stage changes what information the answer writer can see. A language model may also recall facts from training, which is why a correct answer by itself does not prove that retrieved evidence caused the answer.</p><p>In this tutorial, the retriever chooses paragraphs and the generator returns a short answer plus specific sentence coordinates. These create separate observations that an evaluator can check. We preserve the retrieved evidence as well as the final output.</p>',('langgraph',))
add('dataset','What HotpotQA contains',a,'A 2018 question-answering dataset built from Wikipedia with sentence-level evidence labels.',
    split('<p class="big-number">113k</p><p>Question-answer pairs in the original dataset, approximately.</p><p>Questions often require combining information from two supporting paragraphs.</p>',
          '<h3>One labeled example includes</h3><ul><li>A question and short reference answer.</li><li>Candidate context paragraphs.</li><li>Supporting facts: the sentence coordinates needed for the answer.</li><li>Question type and difficulty metadata.</li></ul>'),
    'HotpotQA provides both answer targets and evidence targets for evaluating RAG.',
    '<p>Yang and colleagues introduced HotpotQA at EMNLP 2018. The dataset contains human-written questions based on Wikipedia. Its important feature for this lesson is annotation at the sentence level: we can ask whether a system located and selected the evidence used by the dataset annotators.</p><p>The benchmark was designed around multi-hop question answering, but a high score does not demonstrate that every prediction used the intended reasoning process. Systems can exploit shortcuts or memorized information. We use this dataset to evaluate observable outputs and retrieval coverage, and keep stronger causal claims separate.</p>',('paper','hotpot'))
add('bridge','Bridge questions',a,'One fact identifies an entity that helps locate the next fact.',
    '<p><strong>Question:</strong> In which city was the creator of Lantern Atlas born?</p>'+diagram('bridge','Lantern Atlas identifies creator Mira Vale, whose birthplace is Harbor City.')+'<p class="label">Invented teaching example. These names and facts are synthetic.</p>',
    'The first paragraph supplies the bridge entity “Mira Vale.”',
    '<p>Read the question before looking at the answer. The question names the book but does not name its creator. The first supporting sentence links Lantern Atlas to Mira Vale. The second connects Mira Vale to Harbor City. The word “bridge” describes the entity that links those pieces of evidence.</p><p>Our short synthetic record makes the data format and graph behavior easy to inspect. It has four context paragraphs, unlike the standard ten-paragraph distractor setting. It is never included in a reported HotpotQA benchmark score.</p>')
add('comparison','Comparison questions',a,'Two sources provide values that the system must compare.',
    '<p><strong>Question:</strong> Which theater opened first, River Theater or Hill Theater?</p>'+diagram('comparison','River Theater opened in 1902 and Hill Theater in 1921. Comparing the dates gives River Theater.')+'<p class="label">Invented dates and theaters, used only to explain comparison.</p>',
    'Both relevant entities may already appear in the question.',
    '<p>Comparison questions can ask whether two entities share a property, or which entity has an earlier date or larger value. Our invented example requires extracting each opening date and comparing them. Unlike the previous bridge example, both names are directly available to the first retrieval query.</p><p>HotpotQA distinguishes bridge and comparison questions. The type is useful for later error analysis, but the agent in this tutorial does not receive that metadata. We want the system to determine what to retrieve from the question and evidence.</p>',('paper','format'))
add('settings','Distractor and fullwiki settings',a,'The candidate collection changes what the experiment measures.',
    diagram('settings','Distractor has ten supplied paragraphs including two gold paragraphs. Fullwiki searches Wikipedia and can miss the gold paragraphs.')+'<p class="small">This lab ranks each example’s supplied distractor paragraphs. It does not build a full Wikipedia index.</p>',
    'Results depend on the search space. Name that search space in every report.',
    '<p>In the distractor setting, each question arrives with ten candidate paragraphs: two supporting paragraphs and eight distractors. This evaluates evidence selection within a small, favorable candidate pool.</p><p>The fullwiki task requires finding evidence across the Wikipedia corpus. The released fullwiki JSON contains ten paragraphs returned by the original baseline retriever. It does not contain the entire corpus, and those ten may omit the supporting paragraphs. Pooling a few development examples into one small index also does not reproduce fullwiki evaluation. Use the released processed Wikipedia corpus when building a comparable corpus retrieval experiment.</p>',('hotpot','format'))

b='2. Read the JSON'
toy=json.loads((ROOT.parent/'data/toy_example.json').read_text())
toy_compact='''{
  "_id": "synthetic-lantern-atlas-001",
  "question": "In which city was the creator of Lantern Atlas born?",
  "answer": "Harbor City", "type": "bridge", "level": "toy",
  "supporting_facts": [["Lantern Atlas", 0], ["Mira Vale", 0]],
  "context": [
    ["Lantern Atlas", ["Lantern Atlas is an invented book whose creator is Mira Vale."]],
    ["Mira Vale", ["Mira Vale was born in Harbor City."]],
    ["Harbor City", ["Harbor City is an invented coastal city."]],
    ["Cloud Almanac", ["Cloud Almanac is an invented book created by Niko Fern."]]
  ]
}'''
assert json.loads(toy_compact)==toy
add('raw-json','A complete, small raw JSON record',b,'The same synthetic record appears in the notebook and in the linked raw file.',code(toy_compact),
    'The real dataset file is a JSON list of records with this structure.',
    '<p>This is valid JSON, with double-quoted strings, arrays in square brackets, and an object in braces. Open <a href="../data/toy_example.json">the raw teaching JSON</a> to inspect the exact file. Here we print every field and every sentence, with whitespace compressed to fit the slide.</p><p>The four paragraphs and the level value “toy” belong to our teaching fixture. Official distractor examples have ten paragraphs and official difficulty labels use easy, medium, or hard. The top-level object here makes a single record easy to teach. Official split files wrap records in a top-level list.</p>',('format',),cls='compact')
add('json-fields','The record has two kinds of information',b,'The agent needs the task and candidate evidence. The evaluator needs the answers.',
    '<table><thead><tr><th>Field</th><th>Meaning</th><th>Use in this lab</th></tr></thead><tbody><tr><td><code>question</code></td><td>The question to answer</td><td>Agent input</td></tr><tr><td><code>context</code></td><td>Title and sentence list per paragraph</td><td>Agent input</td></tr><tr><td><code>answer</code></td><td>The reference short answer</td><td>Scoring only</td></tr><tr><td><code>supporting_facts</code></td><td>Reference sentence coordinates</td><td>Scoring only</td></tr><tr><td><code>_id, type, level</code></td><td>Identity and descriptive metadata</td><td>Join results and analyze slices</td></tr></tbody></table>',
    'Do not give the agent the entire labeled record.',
    '<p>The easiest leakage bug is sending json.dumps(example) directly to a model. That includes the reference answer and the reference supporting facts. A high score from this prompt measures access to the answer key.</p><p>Construct the graph input deliberately from the question and context. The experiment runner can retain the example ID outside the agent, then join the result to the labels afterward. The type and level fields support analysis but are not prediction inputs here. The official test release does not supply the answer and supporting-fact labels.</p>',('format',))
add('sentence-ids','A supporting fact is an address',b,'The pair [title, sentence_id] locates one original sentence.',
    diagram('coordinates','Sentence zero of paragraph Mira Vale states the birthplace. A citation uses [Mira Vale, 0].')+'<p><code>["Mira Vale", 0]</code> selects the first sentence, because indexing starts at zero.</p>',
    'Keep original titles and indices stable through ranking, chunking, and generation.',
    '<p>HotpotQA supporting facts do not store copied sentence text. They store a title string and a zero-based sentence index. The evaluator compares these pairs exactly. A different title capitalization or a changed sentence index can turn the same intended evidence into an incorrect prediction.</p><p>The second row in the diagram is illustrative: the toy Mira Vale paragraph has only sentence zero. Real paragraphs often have multiple sentences. If your chunker combines or splits sentences, retain a mapping back to the original title and index. Do not renumber a sentence merely because it is now the first sentence of a retrieved chunk.</p>',('format','eval'))
add('actual-json','An actual HotpotQA record',b,'Selected fields from data/example.json, with original strings and coordinates.',
    code('''{
  "_id": "5abd94525542992ac4f382d2",
  "question": "2014 S/S is the debut album of a South Korean boy group that was formed by who?",
  "answer": "YG Entertainment",
  "type": "bridge",
  "supporting_facts": [["2014 S/S", 0], ["Winner (band)", 0]]
}''')+
    '<p class="small"><strong>["2014 S/S", 0]:</strong> 2014 S/S is the debut album of South Korean group WINNER.</p><p class="small"><strong>["Winner (band)", 0]:</strong> Winner (Hangul: 위너), often stylized as WINNER, is a South Korean boy group formed in 2013 by YG Entertainment and debuted in 2014.</p>',
    'The full raw record retains all ten context paragraphs and every sentence.',
    '<p>Open <a href="../data/example.json">the actual raw JSON record</a>. This slide omits context and level from the JSON display to make the question and annotation legible. It then prints the two referenced context sentences separately. The full file preserves the original HotpotQA schema and the complete distractor context.</p><p>The annotated path connects the album to WINNER and WINNER to its founding company. Notice that album sentence 1 also names YG Entertainment as a label. A system might use this shortcut or prior knowledge. Matching the answer and supporting facts does not prove that the annotated reasoning path caused the prediction.</p><p>The module reconstructs the original JSON representation from the official HotpotQA Hugging Face mirror at a pinned revision. See <a href="../data/provenance.json">provenance.json</a> for source hashes and transformation details. HotpotQA data is CC BY-SA 4.0. Credit Yang et al., 2018.</p>',('format','paper'),cls='compact')
add('real-record','The real sample uses the same schema',b,'The notebook loads an attributed development sample and inspects its original records.',
    code('''examples = json.loads(Path("data/dev_sample.json").read_text())
example = examples[0]
print(example["_id"], example["type"])
print(example["question"])
print(example["supporting_facts"])

for title, sentences in example["context"]:
    for sentence_id, sentence in enumerate(sentences):
        print(title, sentence_id, sentence)'''),
    'Inspect a complete record before changing preprocessing or retrieval.',
    '<p>Run the matching inspection cell in the notebook. The real sample preserves every original context sentence and annotation for each included development record. Its provenance file identifies the upstream source and sampling procedure. The synthetic example and the real sample are separate files so their roles remain visible.</p><p>Choose one supporting-fact coordinate and manually locate that sentence in context. Count the paragraphs. Check whether two supporting sentences can share a title. These small checks establish the shape of the data before you rely on aggregate metrics. The code above uses the bundled dev_sample.json file. The notebook also opens example.json to inspect the album question shown on the previous slide.</p>',('format',))
add('label-boundary','The agent cannot see the answer key',b,'Gold labels enter the evaluator after prediction finishes.',
    diagram('boundary','Question and context enter RAG agent. Gold answer and supporting facts enter evaluator separately. Predictions connect agent to evaluator.'),
    'The graph state and model prompts contain question, retrieved text, and run metadata.',
    '<p>Separating the objects in code makes accidental leakage harder. The retriever sees the candidate texts because the distractor protocol explicitly supplies them. It must not use supporting_facts to choose which candidates enter the index or to boost their scores.</p><p>A diagnostic oracle is an intentional exception: an experiment may supply gold paragraphs to isolate the generator’s ability. Label that result as an oracle diagnostic and keep it separate from ordinary retrieval results. The default pipeline in this lesson does not use a gold oracle.</p>')

c='3. Build and run the RAG graph'
add('protocol','A concrete experiment protocol',c,'The default experiment evaluates retrieval within each real example’s distractor context.',
    '<ol><li>Freeze the included example IDs and source checksum.</li><li>Build one paragraph index for each example.</li><li>Run the same question with <code>k = 1, 2, 4, 10</code>.</li><li>Score retrieval coverage against hidden gold labels.</li><li>Inspect failures before trying the optional live agent.</li></ol>',
    'The bundled sample teaches the method. It cannot support a leaderboard claim.',
    '<p>The k values define how many paragraphs one retrieval call returns. The ten-paragraph condition is a coverage control: when all candidates enter the evidence, reference evidence should be available under the distractor setup. This does not mean the generator will use it correctly.</p><p>The real-data retrieval ablation can run without a hosted model. A separate scripted graph demonstration checks routing and output handling on a synthetic example. Only the optional live run produces model-generated answers for an answer-quality experiment. Keeping these experiments distinct makes the interpretation of each result clear.</p>')
add('setup','Notebook setup',c,'The required path runs locally after Python dependencies are installed.',
    code('''cd evals/hotpotqa
uv sync --locked
uv run jupyter lab

# Open the module notebook and run cells in order.
# Keep live model calls disabled for the offline lesson.''')+'<p>Use the module README for the notebook filename and optional model setup.</p><p class="small">A fresh kernel avoids silently reusing old data, objects, or outputs.</p>',
    'Run the offline path first, then decide whether to enable live calls.',
    '<p>The environment lock file records tested dependencies. The notebook uses LangGraph even in the offline graph demonstration, so this is not a substitute graph drawn only for presentation. TF-IDF runs locally and needs no embedding service.</p><p>Dependency installation may require network access. Once dependencies and the bundled data are present, the required lesson runs without API credentials or model calls. The optional live section explains the model name, environment variable, data sent to the provider, and call limits. Do not place API keys in the notebook or its saved outputs.</p>')
add('paragraph-index','Paragraphs are the retrieval unit',c,'Keep sentence coordinates inside each paragraph.',
    diagram('paragraphs','Paragraphs are ranked as complete title and sentence lists. Original sentence IDs remain available for citations.')+code('index = make_index(example["context"])'),
    'A paragraph can contain useful evidence and irrelevant sentences together.',
    '<p>Each context entry is already a titled paragraph represented as a list of sentences. Our baseline indexes the title and paragraph text. It does not require a general-purpose chunker or vector database, which keeps the unit of retrieval aligned with the data.</p><p>After retrieving paragraphs, render their original sentence numbers into the prompt. The generator can then select particular sentences. If you replace this baseline with smaller chunks later, evaluate whether chunking loses a bridge fact or breaks the citation mapping.</p>')
add('tfidf','A transparent TF-IDF baseline',c,'TF-IDF represents a paragraph using weighted vocabulary terms.',
    diagram('tfidf','Illustrative sparse weights show that Lantern Atlas words match the question while Mira Vale words may not.')+'<p class="small">The displayed weights illustrate the idea. They are not values measured by the notebook.</p>',
    'This is sparse lexical retrieval. Dense embedding retrieval is a later experiment.',
    '<p>Term frequency reflects how often a term appears within a document. Inverse document frequency gives more weight to terms that occur in fewer documents in the candidate collection. Cosine similarity compares a query vector with paragraph vectors. scikit-learn provides the implementation used by the tutorial.</p><p>TF-IDF is inexpensive and interpretable, but it depends on word overlap. The second paragraph in a bridge question may contain the creator’s name without containing the book’s name. That makes iterative query rewriting worth investigating. No dense embedding model or approximate nearest-neighbor database is required for this baseline.</p>',('tfidf',))
add('graph','The LangGraph workflow',c,'Nodes read shared state, return updates, and route to the next step.',
    diagram('graph','Retrieve, grade, then generate if sufficient. Otherwise rewrite and retrieve if below the round bound. Generate at the bound with available evidence.',True),
    'A fixed retrieval-round limit makes the loop finite.',
    '<p>The official LangGraph tutorial demonstrates retrieval, evidence grading, question rewriting, and answer generation. We adapt that pattern to HotpotQA’s paragraph inputs and sentence citations. The node that grades evidence asks whether the retrieved material is sufficient for the original question.</p><p>The graph retrieves before grading. A positive grade routes to generation. A negative grade routes to rewriting only if another retrieval round remains. At the bound, generation receives the evidence available and may abstain. The trace records the final sufficiency grade and round count. A run that reaches the round limit can still have status ok, so inspect the trace when diagnosing evidence insufficiency.</p>',('langgraph',))
add('state','What the graph remembers',c,'State connects retrieval decisions to observable evaluation records.',
    split('<h3>Inputs and evolving evidence</h3><ul><li>Original question and current query.</li><li>Retrieved paragraph union.</li><li>Round count and sufficiency decision.</li><li>Execution trace.</li></ul>',
          '<h3>Output for the evaluator</h3><ul><li>Short <code>answer</code> and <code>sp</code>.</li><li><code>retrieved_titles</code> and <code>retrieved_facts</code>.</li><li><code>invalid_citations</code> and status.</li><li>Latency and backend call count.</li></ul>'),
    'Save the trace so a score can be connected to the decisions that produced it.',
    '<p>The original question stays fixed while the query may change. This matters because a rewrite is an intermediate search request, not a replacement task. Accumulated evidence preserves useful paragraphs from earlier rounds.</p><p>The final result includes answer and citation outputs for official scoring, plus custom diagnostics about what the generator could see. A citation can be invalid because its title or index does not exist in the evidence. Record those errors rather than inventing a valid coordinate. The trace lets you inspect how a rewrite changed retrieval and whether generation proceeded at the round limit.</p>')
add('grader','Sufficiency is a routing decision',c,'The grader checks whether the retrieved text can answer the whole question.',
    '<div class="exercise"><p><strong>Question:</strong> In which city was the creator of Lantern Atlas born?</p><p><strong>Evidence:</strong> “Lantern Atlas is an invented book whose creator is Mira Vale.”</p><p><strong>Decision:</strong> Insufficient. The creator is identified, but the birthplace is missing.</p></div><p>The grader receives the question and retrieved text, with no reference answer or supporting-fact labels.</p>',
    'A model grader can make mistakes. Its decision is not an official HotpotQA metric.',
    '<p>Relevant evidence and sufficient evidence are different. The sentence about the book is relevant, but it does not answer the birthplace question. A sufficiency decision should consider all facts needed to support the answer, including a missing second hop.</p><p>The offline backend returns scripted decisions so both graph paths are easy to demonstrate. The live backend asks a model for a structured decision and explanation. A false-positive sufficiency decision can stop retrieval too early. A false negative can add cost or distractors. Inspect these routing errors alongside answer metrics.</p>')
add('rewrite','A rewrite can reveal the second hop',c,'Use only facts already available to the agent when forming the next query.',
    split('<h3>Initial query</h3><p>In which city was the creator of Lantern Atlas born?</p><h3>Rewritten query</h3><p>Mira Vale birthplace</p><p class="small">The name comes from retrieved text.</p>',diagram('accumulate','Round one and round two retrieval results are combined with duplicate paragraphs removed.')),
    'Two rounds with k = 2 can expose up to four distinct paragraphs.',
    '<p>After the first retrieval finds the book paragraph, the graph knows the creator’s name. Rewriting the query around that name can help the lexical retriever reach a paragraph that had little overlap with the original question.</p><p>The implementation keeps the union of paragraphs across rounds. Therefore the total evidence size may exceed k. A fair comparison with a single-call retriever should report the number of calls, total distinct evidence paragraphs, context tokens, and latency. Adding a rewrite changes both the search process and the potential evidence budget.</p>')
add('generation','The generator returns answer and citations',c,'A short answer string keeps benchmark scoring separate from explanatory prose.',
    code('''{
  "answer": "Harbor City",
  "citations": [
    {"title": "Lantern Atlas", "sentence_id": 0},
    {"title": "Mira Vale", "sentence_id": 0}
  ]
}''')+'<p>The adapter validates coordinates and converts citations into HotpotQA’s <code>sp</code> pairs.</p>',
    'A citation must identify evidence the generator actually received.',
    '<p>The internal structured response names each citation’s title and sentence ID. The adapter checks that both appear in the retrieved evidence and produces pairs for the official output format. Preserve invalid-citation diagnostics so an apparent format repair does not conceal model errors.</p><p>The live prompt instructs the model to return an empty answer string and no citations if evidence is insufficient. HotpotQA primarily measures answering its answerable benchmark questions, so abstaining generally lowers the answer score. You can still track abstention separately for a real application. A well-formed citation proves the coordinate exists. It does not by itself prove the cited sentence entails the answer.</p>')
add('offline-live','Scripted demonstration and live experiment',c,'The same graph can run with different backends.',
    '<table><thead><tr><th></th><th>Offline scripted backend</th><th>Optional live backend</th></tr></thead><tbody><tr><td>Decisions and answer</td><td>Provided by tutorial code</td><td>Generated by a model</td></tr><tr><td>What it demonstrates</td><td>Routing, retries, citation checks</td><td>Observed model behavior</td></tr><tr><td>API calls</td><td>None</td><td>Up to 4 per example at 2 rounds</td></tr><tr><td>Valid conclusion</td><td>The pipeline behaves as tested</td><td>Performance on this measured sample</td></tr></tbody></table>',
    'A scripted correct answer is not evidence of RAG answer quality.',
    '<p>ScriptedBackend supplies predefined grades, rewrites, and an answer. It is a test double: a controllable component used to verify the graph and evaluator. The known answer is deliberately part of the demonstration fixture. Do not aggregate that fixture into model benchmark results.</p><p>With max_rounds=2, LiveBackend uses at most two grader requests, one rewrite request, and one generation request at the graph level. The notebook disables client retries with max_retries=0. Other deployments may enable retries, which can add underlying HTTP attempts. Record the provider’s token usage and billing separately when available. The optional model run remains a small local experiment unless expanded to a documented benchmark protocol.</p>')

d='4. Score retrieval, evidence, and answers'
add('metric-map','Three measurements, three meanings',d,'Each metric answers a different question about one run.',diagram('metrics','Retrieval measures available evidence, SP measures selected citations, and answer scores measure the output string.'),
    'Official HotpotQA scores cover answers and supporting facts. Retrieval diagnostics are additional.',
    '<p>Retrieval coverage uses the set of sentences available to the generator. Supporting-fact scoring uses only the sentence coordinates that the system chooses as evidence in its output. These sets can be very different even when the system reads complete paragraphs.</p><p>Answer exact match and token F1 compare the short answer with its reference. Joint scores combine answer and supporting-fact precision and recall. Keep these names distinct in tables. Calling retrieved-sentence recall “supporting-fact F1” would merge two different behaviors and hide evidence-selection failures.</p>',('eval',))
add('retrieval-recall','Retrieval recall and complete coverage',d,'Use the hidden gold facts only after retrieval to measure what was found.',
    diagram('recall','Gold titles A and B. Retrieved titles A and C. Gold paragraph recall equals one half and all-gold success equals zero.')+'<p class="small"><strong>Sentence coverage:</strong> gold sentence coordinates present in retrieved evidence ÷ all gold coordinates.</p>',
    'Finding one of two supporting paragraphs can still leave a question unanswerable from evidence.',
    '<p>Gold paragraph recall is the number of distinct supporting titles retrieved divided by the number of distinct supporting titles. All-gold success is one only if every gold title is present. Average each per-example quantity across the evaluated examples.</p><p>Gold sentence coverage uses original sentence coordinates rather than title sets. Retrieving complete paragraphs often makes paragraph recall and sentence coverage move together, but they can differ when paragraphs have unequal numbers of gold sentences or when you later truncate or chunk text. Score the actual evidence visible to generation, not a larger pre-truncation retrieval list.</p>')
add('answer-metrics','Answer exact match and token F1',d,'The official scorer normalizes text before comparing answers.',
    split('<p><strong>Gold:</strong> Harbor City</p><p><strong>Prediction:</strong> Harbor</p><p class="equation">Precision = 1/1 = 1<br>Recall = 1/2 = 0.5<br>F1 = 2/3 ≈ 0.667<br>Exact match = 0</p>',
          '<h3>Normalization</h3><p>Lowercase, remove ASCII punctuation, remove “a”, “an”, “the”, and collapse whitespace.</p><p class="small">Token overlap counts repeated words. It does not measure semantic equivalence.</p>'),
    'Special answers yes, no, and noanswer get zero token F1 when they mismatch.',
    '<p>Precision divides the matched token count by the predicted token count. Recall divides that count by the reference token count. F1 is their harmonic mean. Counter-based overlap preserves token multiplicity, so repeated words are not treated as a simple vocabulary set.</p><p>Normalization lets harmless differences such as “The Harbor City!” match “Harbor City.” It does not recognize every alias or paraphrase. The yes, no, and noanswer rule prevents partial token overlap from rewarding a mismatched categorical answer. Use the official evaluator as the compatibility target and test edge cases rather than inventing an approximately similar scorer.</p>',('eval',))
add('sp-metrics','Supporting-fact precision and recall',d,'Compare sets of exact (title, sentence_id) pairs.',
    split(diagram('sp','Gold A0 and B0 versus predicted A0 and C1 gives one true positive, one false positive and one false negative.'),
          '<p class="equation">TP = 1, FP = 1, FN = 1<br>Precision = 1/2<br>Recall = 1/2<br>SP F1 = 0.5<br>SP exact match = 0</p>'),
    'Citing extra sentences can raise coverage while reducing evidence precision.',
    '<p>The gold set has two coordinates. The prediction shares one, misses one, and introduces one extra coordinate. A true positive is an exact coordinate intersection. A false positive is a predicted coordinate absent from the reference set. A false negative is a missing reference coordinate.</p><p>The official scorer converts both lists to sets, so duplicates do not improve the score. SP exact match requires the entire predicted set to equal the gold set. Titles are not answer-normalized. This metric measures agreement with the annotated evidence, not a semantic judgment that every alternative evidence sentence is false.</p>',('eval',))
add('joint-metrics','Joint F1 combines precision and recall',d,'Continue the previous worked examples: answer P = 1, R = 0.5; SP P = 0.5, R = 0.5.',
    '<p class="equation">Joint precision = answer P × SP P = 1 × 0.5 = 0.5</p><p class="equation">Joint recall = answer R × SP R = 0.5 × 0.5 = 0.25</p><p class="equation">Joint F1 = 2 × 0.5 × 0.25 / (0.5 + 0.25) = 1/3</p><p>Joint exact match = answer EM × SP EM = 0.</p><p class="small">Compute per question, then take the mean. Use the formula above, even when a particular example happens to equal the product of the two F1 scores.</p>',
    'Joint F1 is generally not answer F1 multiplied by SP F1.',
    '<p>The official implementation multiplies precision values and recall values first, then computes their harmonic mean. In this worked case the result happens to equal the product of answer F1 and SP F1, but that equality is accidental.</p><p>For a counterexample, let answer precision = 1 and recall = 0.5, and SP precision = 0.5 and recall = 1. Each component F1 is 2/3, so their product is 4/9. Official joint precision and recall are both 0.5, giving joint F1 = 0.5. The scorer averages per-example metrics, rather than pooling tokens across all questions.</p>',('eval',))
add('official-export','The official prediction file',d,'Two dictionaries use the original example ID as their key.',
    code('''{
  "answer": {
    "synthetic-lantern-atlas-001": "Harbor City"
  },
  "sp": {
    "synthetic-lantern-atlas-001": [
      ["Lantern Atlas", 0], ["Mira Vale", 0]
    ]
  }
}''')+'<p class="small">Synthetic format example. Real exports use the selected real HotpotQA IDs.</p>',
    'Score against the matching gold subset and keep its IDs with the predictions.',
    '<p>The official prediction schema differs from the labeled input schema. It uses singular answer and abbreviated sp keys whose values are dictionaries. The example above demonstrates the file shape and does not represent a benchmark submission.</p><p>The official evaluator command is python hotpot_evaluate_v1.py predictions.json gold_subset.json. The gold subset is a list of the exact examples included in the experiment. Missing answer or SP IDs still count in the denominator. Do not drop failed requests when preparing the gold subset, and do not accidentally score a small prediction file against the full development split unless that is your intended denominator.</p>',('eval','format'))
add('paired-comparison','A fair configuration comparison',d,'Change one setting and retain per-question results.',
    diagram('paired','The same question IDs and contexts feed systems with k 2 and k 4. Compare paired scores.')+'<p class="small">Report retrieval coverage, answer and SP scores where applicable, failures, latency, and model-call cost.</p>',
    'More retrieved paragraphs can improve coverage and also add distracting text.',
    '<p>The offline k ablation changes the paragraph count while keeping the candidate pool and question fixed. For a live experiment, compare systems on the same IDs, with the same prompts and model version unless the prompt or model is the factor being tested.</p><p>Save per-example differences so you can inspect what improved and what regressed. Report the exact subset size and selection method. Small selected samples have high uncertainty and may overrepresent easy questions. A larger held-out evaluation with paired bootstrap confidence intervals is a suitable next experiment, provided the questions are sampled in a way that supports the intended claim.</p>')

e='5. Interpret results and improve the experiment'
add('diagnose','Failure analysis follows the evidence',e,'Use the saved trace to locate where the run first lost the needed information.',
    diagram('diagnosis','First check gold evidence availability, then selected supporting facts, then answer correctness.')+'<p class="small">Also inspect invalid citations, malformed outputs, API errors, and runs that reached the round limit.</p>',
    'A final score summarizes a failure. The trace helps explain it.',
    '<p>If gold evidence never reached the generator, retrieval is a plausible bottleneck. If evidence was present but the citations miss it, inspect the evidence-selection prompt and output. If citations match but the answer is wrong, inspect comparison reasoning, answer extraction, or formatting.</p><p>These are useful diagnostics, not automatic causal proofs. A model can answer correctly from memorized knowledge even when evidence is missing. A model can also cite a correct sentence after guessing. An oracle-evidence experiment or controlled removal of a supporting paragraph can test a more specific hypothesis, provided it is clearly labeled as a diagnostic intervention.</p>')
add('reproducibility','A run needs an experiment record',e,'Save enough information to reproduce the input and interpret the output.',
    '<table><thead><tr><th>Record</th><th>Why it matters</th></tr></thead><tbody><tr><td>Source URL, checksum, IDs, split</td><td>Identifies exactly which data was scored</td></tr><tr><td>Candidate scope, k, round bound</td><td>Defines the retrieval task and evidence budget</td></tr><tr><td>Package lock, code, prompts, model</td><td>Identifies the implementation and model settings</td></tr><tr><td>Predictions, traces, errors, timing</td><td>Makes every scored outcome inspectable</td></tr><tr><td>Scorer version and metric definitions</td><td>Prevents silent changes in score meaning</td></tr></tbody></table>',
    'A seed does not make hosted model outputs perfectly reproducible.',
    '<p>Use a manifest to record model identifiers and generation settings, including temperature, along with dataset and package versions. Store a content hash for any downloaded file and retain source attribution. The notebook’s module artifacts provide the starting pattern for these records.</p><p>Even with a fixed seed or zero temperature, hosted model behavior can change across backend versions or be nondeterministic. Save actual outputs and report when you ran the experiment. For expensive comparisons, agree on the evaluation subset and failure policy before making calls so selection cannot drift toward favorable results.</p>')
add('limitations','What a HotpotQA result does not establish',e,'Benchmark accuracy is one part of an application evaluation.',
    '<ul><li>Wikipedia questions do not represent every production domain.</li><li>Public benchmark material may appear in a model’s training data.</li><li>Exact-string metrics can penalize valid aliases or alternative evidence.</li><li>Supporting-fact agreement does not prove faithful model reasoning.</li><li>A small distractor sample does not measure fullwiki retrieval.</li></ul>',
    'Combine benchmark scores with application-specific tasks and evidence checks.',
    '<p>HotpotQA is valuable because it provides structured evidence labels. Its evaluation design still has boundaries. Answer token overlap is a narrow measure, and the annotated evidence set is not guaranteed to list every valid way to support an answer.</p><p>For a deployed RAG application, add questions drawn from the actual corpus and user tasks. Measure citation entailment, unanswerable questions, source freshness, latency, access controls, and cost when those properties matter. Keep these custom metrics separate from official HotpotQA metrics so readers can tell what each score means.</p>',('paper','eval'))
add('exercise','Three short experiments',e,'Predict the outcome before running each notebook exercise.',
    '<ol><li><strong>Coverage:</strong> Change k from 1 to 2. Which questions gain complete gold-paragraph coverage?</li><li><strong>Citations:</strong> Keep “Harbor City” as the answer but remove one citation. Which official scores change?</li><li><strong>Loop:</strong> Compare a scripted positive grade with a negative-then-positive sequence. Which nodes run, and how many paragraphs can accumulate?</li></ol>',
    'Explain each change using saved evidence, not only the final mean.',
    '<p>For exercise one, answer metrics do not exist unless a generator actually runs. Examine retrieved-title sets and the hidden gold-title sets after the retrieval call. Some questions may improve with k=2 and others may still miss evidence.</p><p>For exercise two, a correct unchanged answer retains its answer score. Removing a necessary citation lowers SP recall and usually SP F1, which also reduces joint scores. For exercise three, compare the trace and round count. The scripted demonstration verifies that the control flow matches the specified decisions. It does not show that a model makes those decisions reliably.</p>')
add('next-steps','A complete evaluation report',e,'State the task, show the measurements, and explain their limits.',
    '<p><strong>Task:</strong> Per-example HotpotQA distractor retrieval on a documented development subset.</p><p><strong>Method:</strong> TF-IDF paragraphs, bounded LangGraph retrieval loop, backend clearly identified.</p><p><strong>Evidence:</strong> Frozen IDs, prediction file, traces, coverage diagnostics, and official scores for genuine model outputs.</p><p><strong>Interpretation:</strong> Failure categories, uncertainty, cost, and the next controlled experiment.</p>',
    'The notebook turns this report outline into executable steps.',
    '<p>A useful report identifies the evaluation population and candidate corpus before presenting a score. Readers should be able to tell whether a result came from a synthetic fixture, a real-data retrieval experiment, or a live model generation run.</p><p>Next, try dense retrieval, a reranker, or a different rewrite policy while preserving the same evaluation protocol. Use the official HotpotQA project, JSON documentation, and evaluator as references, and the LangGraph tutorial for the graph concepts. Cite Yang et al. when using the dataset and preserve the dataset’s CC BY-SA 4.0 attribution and license information when redistributing samples.</p>',('hotpot','paper','format','eval','langgraph'))

parts=['''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="An illustrated step-by-step HotpotQA RAG evaluation lesson with raw JSON, LangGraph, retrieval diagnostics and official metrics."><title>HotpotQA RAG evaluation · Module 22</title><link rel="icon" href="data:,"><link rel="stylesheet" href="slides.css"><script src="slides.js" defer></script></head><body><a class="skip" href="#deck">Skip to lesson</a><noscript><p style="padding:20px">All slides appear below. JavaScript adds navigation. Explanations remain available under each slide.</p></noscript><main id="deck">''']
for i,s in enumerate(slides,1):
    tag='h1' if i==1 else 'h2'
    links=''.join(f'<a href="{SOURCES[key][1]}">{SOURCES[key][0]}</a>' for key in s['sources'])
    parts.append(f'''<section class="slide {s['cls']}" id="{s['id']}" aria-labelledby="heading-{s['id']}" data-title="{escape(s['title'],quote=True)}"><header><p class="chapter">{s['chapter']}</p><{tag} id="heading-{s['id']}" tabindex="-1">{s['title']}</{tag}><p class="subtitle">{s['subtitle']}</p></header><div class="content">{s['body']}</div><p class="takeaway">{s['takeaway']}</p><div class="sources">{links}</div><footer class="footer"><span>HotpotQA RAG evaluation · Module 22</span><span>{i} / {len(slides)}</span></footer><details><summary>Read the explanation</summary><div class="notes">{s['notes']}</div></details></section>''')
parts.append('''</main><nav class="controls" aria-label="Slide controls"><label class="sr-only" for="contents">Jump to a slide</label><select id="contents"></select><button id="previous" type="button" aria-label="Previous slide">Previous</button><span class="progress" id="progress"></span><button id="next" type="button" aria-label="Next slide">Next</button><button id="explain" type="button" aria-pressed="false">Explain</button><button id="read" type="button" aria-pressed="false">Read all</button><button id="print" type="button">Print</button></nav><div id="announce" class="sr-only" aria-live="polite" aria-atomic="true"></div></body></html>''')
(ROOT/'index.html').write_text('\n'.join(parts))
print(f'Built {len(slides)} slides and {len(list(ASSETS.glob("*.svg")))} SVG diagrams.')
