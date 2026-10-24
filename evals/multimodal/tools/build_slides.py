"""Build the offline HTML lesson; no browser, network, or third-party package needed.

Run from any directory: python evals/multimodal/tools/build_slides.py
Edit the structured lesson below, then regenerate. The standalone slides are
static 1600 x 900 HTML with inline SVG. The presentation shell adds navigation.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "slides" / "outputs" / "html"
INK, MUTED, CORAL, TEAL = "#202729", "#536164", "#bf4938", "#08796f"
PALE, MINT, LINE = "#fff0eb", "#e9f6f2", "#c9d5d2"


def esc(value):
    return html.escape(str(value))


def txt(x, y, words, size=28, color=INK, weight=500, anchor="start"):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{esc(words)}</text>'


def box(x, y, w, h, label, fill=MINT, sub=None):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}"/>'
    s += txt(x + 24, y + (h / 2 + 10 if not sub else 42), label, 30, weight=700)
    if sub:
        s += txt(x + 24, y + 80, sub, 23, MUTED)
    return s


def arrow(x1, y1, x2, y2, color=TEAL):
    return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="3" marker-end="url(#arrow)"/>'


def svg(body, title, width=1300, height=450):
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}" xmlns="http://www.w3.org/2000/svg">
<title>{esc(title)}</title><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="none" stroke="{TEAL}" stroke-width="1.5"/></marker></defs>{body}</svg>'''


def flow(labels, subtitle="", fills=None):
    n = len(labels)
    gap = 58
    w = (1260 - gap * (n - 1)) / n
    s = ""
    for i, label in enumerate(labels):
        x = 20 + i * (w + gap)
        s += box(x, 145, w, 125, label, (fills or [MINT] * n)[i])
        if i < n - 1:
            s += arrow(x + w + 8, 207, x + w + gap - 12, 207)
    if subtitle:
        s += txt(20, 360, subtitle, 29, MUTED)
    return svg(s, " → ".join(labels) + ". " + subtitle)


def chart(x=20, y=15, w=650, h=400, values=(42, 57, 51), labels=("April", "May", "June"), unit="kWh", title="Solar energy"):
    s = txt(x + 30, y + 40, title, 28, weight=700)
    baseline = y + h - 60
    s += f'<path d="M{x+65},{y+75}V{baseline}H{x+w-20}" stroke="{LINE}" stroke-width="2" fill="none"/>'
    for i, (v, label) in enumerate(zip(values, labels)):
        bx = x + 100 + i * ((w - 160) / 3)
        bh = (h - 160) * v / max(values)
        s += f'<rect x="{bx}" y="{baseline-bh}" width="78" height="{bh}" rx="4" fill="{CORAL if i==1 else TEAL}"/>'
        s += txt(bx + 39, baseline - bh - 14, str(v), 27, anchor="middle", weight=700)
        s += txt(bx + 39, baseline + 36, label, 22, MUTED, anchor="middle")
    s += txt(x + 4, y + 83, unit, 20, MUTED)
    return s


def document(x, y, w=240, h=280, label="Memo", fill=MINT):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" rx="9"/>'
    s += txt(x + 24, y + 45, label, 28, weight=700)
    for j in range(4):
        s += f'<rect x="{x+24}" y="{y+80+j*32}" width="{w-48-(j%2)*38}" height="7" rx="3" fill="{LINE}"/>'
    return s


def scatter(shared=True):
    s = f'<path d="M65,350H650 M65,350V45" stroke="{LINE}" stroke-width="3" fill="none"/>'
    s += f'<rect x="90" y="24" width="18" height="18" fill="{TEAL}"/><circle cx="254" cy="33" r="9" fill="{CORAL}"/>'
    s += txt(122, 42, "image", 24, MUTED) + txt(275, 42, "text", 24, MUTED)
    points = [(160, 270, "image"), (230, 225, "text"), (480, 120, "image"), (525, 170, "text")]
    for x, y, kind in points:
        if kind == "image":
            s += f'<rect x="{x-12}" y="{y-12}" width="24" height="24" rx="3" fill="{TEAL}"/>'
        else:
            s += f'<circle cx="{x}" cy="{y}" r="12" fill="{CORAL}"/>'
    s += txt(85, 403, "Illustrative projection; not model output", 22, MUTED)
    s += txt(135, 322, "solar panels", 25)
    s += txt(405, 74, "controller screen", 25)
    return svg(s, "Text and image points group by meaning in an illustrative two-dimensional projection", 700, 450)


def cards(items):
    n = len(items)
    gap = 24
    w = (1260 - gap * (n - 1)) / n
    s = ""
    for i, (heading, value, caption) in enumerate(items):
        x = 20 + i * (w + gap)
        s += f'<rect x="{x}" y="50" width="{w}" height="330" rx="12" fill="{MINT if i%2==0 else PALE}"/>'
        s += txt(x + 28, 103, heading, 28, weight=700)
        s += txt(x + 28, 215, value, 60, TEAL if i % 2 == 0 else CORAL, 800)
        s += txt(x + 28, 320, caption, 24, MUTED)
    return svg(s, "; ".join(". ".join(x) for x in items))


def code_block(code):
    return f'<pre><code>{esc(code)}</code></pre>'


SLIDES = []


def add(title, section, layout, figure, notes, *, lead="", code=None, callout="", notebook=None, reference=None):
    SLIDES.append(dict(title=title, section=section, layout=layout, figure=figure,
                       notes=notes, lead=lead, code=code, callout=callout,
                       notebook=notebook, reference=reference))


add("Search beyond words", "Multimodal retrieval", "cover-swiss-rail",
    svg(chart(10, 40, 650, 370) + txt(40, 470, "Find the evidence. Then read it.", 32, weight=700), "Fictional school solar chart: April 42, May 57, June 51 kilowatt-hours", 700, 520),
    "Imagine your school is studying a solar panel. The project folder contains lab notes, energy charts, controller screenshots, a PDF report, and a short video. A question such as ‘Which month produced the most energy?’ needs the right chart and a careful reading of its labels. Multimodal means working across more than one kind of information. Retrieval means finding useful evidence. We will build the search stage, inspect where it fails, and then connect it to an optional vision reader. The school, measurements, and media in this lesson are fictional teaching data. The chart shows 42, 57, and 51 kWh; its answer is May, 57 kWh. Ask learners what evidence they would need before answering a different question: why did May produce more? The chart alone does not explain causes.",
    lead="A research workshop for curious students", notebook="01_foundations.ipynb")

add("Evidence has several shapes", "01 · Foundations", "card-constellation",
    svg(document(30, 90, 290, 285, "Lab memo") + chart(365, 95, 530, 280) + box(960, 90, 300, 285, "Video frame", PALE, "Time matters"), "Text memo, solar chart, and video frame provide different kinds of evidence"),
    "A modality is a type of input, such as text, an image, or audio. A PDF is a container: one page might hold selectable text, while another contains a scanned chart. A video combines visual frames with time and may also contain audio. Our corpus includes text memos, three charts, a controller screenshot, a mixed PDF, and a silent twelve-second video with a hand-authored companion transcript. This variety is deliberate: a cleaning policy belongs in a memo, a controller code belongs on a screenshot, and an event's location in a video needs timestamps. Counterexample: a JPEG file full of printed words is technically an image, but ordinary OCR followed by text search may solve the task better than a general vision model.",
    callout="Choose evidence by the question.", notebook="01_foundations.ipynb", reference="4, 10–11")

add("Search compares representations", "01 · Foundations", "process-ribbon",
    flow(["Source", "Representation", "Nearest matches"], "A database compares vectors; it does not see."),
    "A search engine does not compare your sentence directly with the pixels in a file. It first creates a representation: searchable words, a numerical vector, or both. A vector database stores those vectors and finds nearby ones. It can store image embeddings perfectly well; the limitation belongs to a text-only encoder, which cannot accept image pixels. Likewise, a generative vision model is not automatically a retrieval index. Its captions or supported embedding outputs must be turned into records with stable identities. A counterexample helps: merely saving a PNG path in a database payload does not make that image semantically searchable. The payload points to the evidence; an appropriate representation makes it findable.",
    notebook="01_foundations.ipynb", reference="4")

add("Direction can measure similarity", "01 · Foundations", "editorial-thesis",
    svg(f'<path d="M95,355H630 M95,355V40" stroke="{LINE}" stroke-width="3" fill="none"/>' + arrow(95, 355, 530, 110) + arrow(95, 355, 505, 165) + arrow(95, 355, 180, 60) + txt(480, 75, "query", 27) + txt(505, 205, "close", 27, TEAL) + txt(130, 35, "less aligned", 26, CORAL) + txt(85, 420, "Smaller angle → greater cosine similarity", 25, MUTED), "Vectors with smaller angles have greater cosine similarity", 700, 450),
    "An embedding is a list of numbers that a model produces from an input. The individual coordinates are usually learned features, not human labels such as ‘amount of solar energy.’ Think of an arrow from the origin. Cosine similarity compares the directions of two arrows rather than their lengths. Parallel directions have similarity 1; perpendicular directions have 0; opposite directions have -1. In the notebook, small hand-built vectors let you compute every step before any model download. Caution: a high cosine score means the representation considers two items similar. It does not measure truth, entailment, or the probability that an answer is correct. The two-dimensional drawing is an explanation, not a visualization of actual model embeddings.",
    lead="An embedding is a numerical description.", callout="Similarity is useful; certainty is different.", notebook="01_foundations.ipynb")

add("Dimensions do not define compatibility", "01 · Foundations", "comparison-bands",
    svg(box(30, 70, 550, 230, "Encoder A", MINT, "[0.2, 0.9, 0.1]") + box(720, 70, 550, 230, "Encoder B", PALE, "[0.2, 0.9, 0.1]") + txt(40, 375, "Equal length", 30, TEAL, 700) + txt(735, 375, "Different coordinate meanings", 30, CORAL, 700), "Identically sized vectors from unrelated encoders do not necessarily share coordinate meanings"),
    "Three numbers written in one coordinate system cannot safely be compared with three numbers from an unrelated one. A position measured from your classroom door differs from a position measured from the city center, even when both contain two numbers. Similarly, equal embedding dimension is necessary for a direct dot product but does not establish semantic compatibility. Query and document vectors must use the intended compatible encoders, model versions, and preprocessing. CLIP is special because its text and image encoders were jointly trained to align their outputs. Counterexample: a 512-dimensional vector from a random text model is not interchangeable with a 512-dimensional CLIP image vector. Record model identity as well as shape.",
    notebook="01_foundations.ipynb")

add("One model, two different jobs", "01 · Foundations", "ledger-columns",
    svg(box(30, 65, 570, 270, "Parse at ingestion", MINT, "Image → searchable description") + box(710, 65, 570, 270, "Read at question time", PALE, "Selected pixels → specific answer") + txt(55, 290, "Reuse the stored description", 26, TEAL) + txt(735, 290, "Check what the user asks", 26, CORAL), "Parsing happens during ingestion and reading happens after retrieval at query time"),
    "Ingestion means adding material to the collection. At ingestion, a vision model can describe a screenshot, extract text, or turn a chart into a table. You store the result and reuse it across later queries. At question time, the reader receives only a few retrieved images and answers the user's specific question from those pixels. These are separate jobs even when the same model performs both. The caption may have forgotten a small axis label; the reader can inspect it if retrieval found the source. The reader cannot rescue an image that was never retrieved. Reparse when the source or extraction process changes, rather than assuming ‘once’ means forever. On our fixture path, descriptions are authored teaching records, not claimed vision-model outputs.",
    notebook="02_describe_and_search.ipynb", reference="4, 14, 33–36")

add("Unpack the framework shortcut", "01 · Foundations", "code-split",
    svg(box(210, 25, 280, 90, "Question", MINT) + box(20, 235, 300, 105, "Text store", MINT) + box(380, 235, 300, 105, "Image store", PALE) + f'<path d="M350,125V175H170V225 M350,175H530V225" stroke="{TEAL}" stroke-width="3" fill="none"/>' + txt(25, 420, "Inspect both result lists.", 28, MUTED), "A question searches both the text store and image store", 700, 450),
    "The reference notebook opens with LlamaIndex's MultiModalVectorStoreIndex. It is a useful first picture of the system: provide documents, a text embedder, and a compatible multimodal embedder; the framework maintains representations and retrieval paths. The code here is an architectural sketch; the notebook supplies the actual adapters and configuration. In particular, image search needs the text tower of the same multimodal model, rather than the unrelated text-only embedder used for memo search. A framework saves plumbing but does not decide whether a chart is accurate, whether two scores are comparable, or whether a result contains the answer. The next lessons make those decisions visible. Our offline path does not silently download models or call a hosted service.",
    code="index = MultiModalVectorStoreIndex(\n    nodes,\n    embed_model=text_encoder,\n    image_embed_model=clip_encoder,\n)\nhits = index.as_retriever().retrieve(q)",
    callout="Architecture sketch; adapters live in notebooks.", notebook="01_foundations.ipynb", reference="8")

add("Make the workshop reproducible", "01 · Foundations", "story-rail",
    cards([("Environment", "uv", "One external project file"), ("Credentials", ".env", "Load only when needed"), ("First run", "Local", "Fixture mode needs no key")]),
    "Open a terminal in evals/multimodal and run uv sync. This installs the dependencies from the external pyproject.toml and lockfile, rather than modifying the environment in hidden notebook cells. Start Jupyter with uv run jupyter lab. Copy .env.example to .env only when configuring optional models, and keep secrets out of notebooks and version control. Python uses load_dotenv(find_dotenv()) to find configuration. The default lessons run with deterministic teaching data and deliberately simple baseline vectors; they are not benchmark results for a pretrained CLIP model or a hosted vision model. The optional live path downloads local ONNX models and can call your configured vision endpoint. A fully reproducible comparison also records model files and preprocessing, not just Python package versions.",
    callout="uv sync   →   uv run jupyter lab", notebook="01_foundations.ipynb")

add("Keep the route back to evidence", "01 · Foundations", "editorial-thesis",
    svg(f'<rect x="30" y="35" width="640" height="320" rx="10" fill="{MINT}"/>' + txt(60, 92, "A searchable record", 30, weight=700) + txt(60, 175, "source_id: chart_energy", 27) + txt(60, 225, "representation: caption", 27) + txt(60, 275, "source_path: original pixels", 27) + txt(60, 325, "model + version + metadata", 27) + txt(40, 420, "Representation ID ≠ source ID", 29, CORAL, 700), "A searchable record preserves the original source identity and links to its representation and metadata", 700, 450),
    "The original chart and a text caption of that chart are different records describing the same underlying evidence. Keep a source_id such as chart_energy, a representation identifier such as caption-v1, the source path, and useful provenance. A PDF record also needs a page number; a frame record needs a timestamp or time window. Stable source identity lets us fuse rankings without counting the same chart twice and lets an answer cite what the user can inspect. The database may use its own point ID, which is separate from the domain-level source ID. Counterexample: treating every image crop as unrelated evidence can fill the top three results with three copies of one chart and leave no room for a useful memo.",
    lead="Every vector needs an identifiable source.", notebook="01_foundations.ipynb", reference="11, 20, 23")

add("A · Describe once, search words", "02 · Descriptions", "process-ribbon",
    flow(["Image", "Caption", "Text vectors", "Search"], "The question uses the same text encoder."),
    "Strategy A converts each image into searchable language. The ingestion parser might write ‘Monthly solar energy: April 42 kWh, May 57 kWh, June 51 kWh.’ Those words can share an index with lab memos, using the same compatible text encoder for records and queries. Store the link to the image so a later reader can inspect the source. This approach is practical when important visual content can be described in words. Its weakness is information loss: retrieval sees only what the parser included. A vague caption such as ‘a bar chart’ may omit both solar energy and months. The default notebook uses transparent, authored descriptions to isolate search behavior; optional model parsing is labeled separately.",
    notebook="02_describe_and_search.ipynb", reference="14–24")

add("A caption decides what survives", "02 · Descriptions", "evidence-quote",
    svg(chart(10, 20, 660, 400), "Solar energy chart with April 42, May 57, and June 51 kWh", 700, 450),
    "Read the chart, then hide it and inspect the caption ‘A monthly solar chart.’ That caption helps find the topic, but it cannot answer which month was highest or what the units mean. Adding every visible value preserves more information, but creates more opportunities for extraction errors. Captions are lossy representations: they choose some details and discard others. OCR aims to recover visible text; a description also summarizes visual relationships; a chart-to-table extraction aims to preserve structured values. Those outputs have different uses. Do not infer that a rich caption is automatically correct. If a label is ambiguous, preserve the uncertainty and keep the source image available for inspection.",
    lead="“A monthly solar chart”", callout="Topic preserved. Values omitted.", notebook="02_describe_and_search.ipynb", reference="14–18")

add("Valid JSON can still be wrong", "02 · Descriptions", "card-constellation",
    cards([("Shape", "Schema", "Are required fields present?"), ("Grounding", "Pixels", "Do values match the source?"), ("Uncertainty", "Review", "Can we verify the claim?")]),
    "A parser returning {\"month\": \"May\", \"value\": 75, \"unit\": \"kWh\"} might satisfy the expected JSON schema while misreading the chart, whose value is 57. Schema validation checks structure and types. It cannot establish visual truth. Our generated corpus has an answer key, so a lesson can compare extracted numbers with known values. Real uploaded documents often do not have an answer key: use source inspection, cross-checks, trusted tables, or human review. A range check catches a negative energy value but cannot prove that 75 should be 57. Keep statuses such as verified, unverified, and invalid distinct. A second model agreeing is supporting evidence, not a guarantee of correctness.",
    notebook="02_describe_and_search.ipynb", reference="15–18")

add("Store vectors; keep source metadata", "02 · Descriptions", "code-split",
    svg(box(30, 70, 620, 135, "Vector", MINT, "Find nearby representations") + box(30, 245, 620, 135, "Payload", PALE, "Find the original evidence"), "A Qdrant record contains both a vector and a payload with source metadata", 700, 450),
    "Qdrant stores a vector plus a payload. The vector supports similarity search; the payload holds identifiers, modality, source paths, and provenance. A collection fixes its vector dimension and similarity metric. Query vectors must match the collection's intended encoder and dimension. The code sketch shows the query_points method used by current qdrant-client: its response contains points rather than being a raw list. In-memory local mode is ideal for a small exercise, but its records disappear when the process ends. Rebuilding after a notebook restart is expected. A persistent deployment needs explicit storage and lifecycle decisions. Metadata can also support filters; those filters should restrict eligible evidence before the result reaches a reader.",
    code="result = client.query_points(\n    collection_name=collection,\n    query=query_vector.tolist(),\n    limit=3,\n    with_payload=True,\n).points",
    notebook="02_describe_and_search.ipynb", reference="23")

add("Predict where descriptions will fail", "02 · Descriptions", "comparison-bands",
    svg(box(30, 75, 575, 265, "Likely useful", MINT, "Find the ventilation warning") + box(705, 75, 575, 265, "Needs more evidence", PALE, "Read a tiny controller code") + txt(55, 292, "Meaning appears in the caption", 26, TEAL) + txt(730, 292, "Check E17 against the pixels", 26, CORAL), "Descriptions help semantic topic search; exact codes require careful extraction and visual checking"),
    "The controller screenshot contains the exact code E17 and the warning that ventilation is blocked. A description mentioning blocked ventilation may retrieve the screenshot for ‘Why should we clear the vent?’ Exact code queries are more fragile: the parser may omit E17, a text embedding may blur it with E71, and the reader may mistake a small character. Add lexical matching or exact fields when identifiers matter. Another failure is unmentioned visual appearance: ‘the green chart’ cannot match a caption that never names color. The exercise is to name one caption omission, predict a query it will hurt, change only that caption, and rerun the query. That is a small controlled experiment, not evidence that A is universally best or worst.",
    notebook="02_describe_and_search.ipynb")

add("B · Put words beside pictures", "03 · Shared space", "editorial-thesis",
    scatter(),
    "Strategy B embeds images and text into a compatible shared space. A text question passes through the model's text tower; an image passes through its image tower. Similarity can then rank images directly without requiring a caption first. The shared space is learned rather than manually specified. The notebook begins with an intentionally weak, color-only pixel feature exercise and manually mapped color queries, so everyone can run it offline and see its limits. It is not CLIP, not a semantic substitute, and not a performance benchmark. The optional FastEmbed path loads the actual paired CLIP encoders. The plotted points are a conceptual two-dimensional drawing; real embeddings have many dimensions and their geometry is not fully visible in this sketch.",
    lead="Paired encoders learn compatible coordinates.", callout="Offline color exercise ≠ pretrained CLIP.", notebook="03_shared_space.ipynb", reference="25–29")

add("Training pulls matching pairs closer", "03 · Shared space", "ledger-columns",
    svg(box(35, 60, 400, 120, "Solar panel photo", MINT) + box(845, 60, 420, 120, "“solar panel”", MINT) + arrow(445, 120, 830, 120) + box(35, 270, 400, 120, "Controller photo", PALE) + box(845, 270, 420, 120, "“controller”", PALE) + arrow(445, 330, 830, 330) + txt(470, 235, "Matched pairs", 29, TEAL, 700), "Contrastive training encourages matching image-text pairs to be close relative to mismatched pairs"),
    "CLIP was trained with image-text pairs. In a training batch, the model learns to give matching pairs higher compatibility than mismatched pairs. The text and image towers become translators into a related coordinate system. We use a pretrained model; we do not train CLIP in these notebooks. Shared space does not mean every capability is shared equally. Natural-image concepts can work well while small text, exact counts, fine spatial relations, and chart numerals remain difficult. It also does not guarantee that text records and image records have identical score distributions. Counterexample: recognizing that a picture is an energy bar chart is a different task from reading that May is 57 kWh. Retrieval can be useful without solving the second task.",
    notebook="03_shared_space.ipynb", reference="25–27")

add("FastEmbed runs the paired towers", "03 · Shared space", "code-split",
    svg(box(35, 60, 615, 125, "Question → text tower", MINT) + box(35, 245, 615, 125, "Pixels → vision tower", PALE), "Use the matching CLIP text and vision tower models", 700, 450),
    "FastEmbed uses ONNX model files for local inference, allowing these embedding examples to avoid a large PyTorch setup. The two Qdrant CLIP model identifiers in the sketch select the intended text and vision towers. The first live run still downloads weights; ‘local inference’ does not mean ‘nothing is ever downloaded.’ After caching the weights, inference processes the inputs locally. Pin model versions or artifacts when reproducing results, and preserve preprocessing such as image resizing. TextEmbedding also supports text-only models; naming an arbitrary one here would break the shared-space design. The complete notebook guards the optional live path and reports missing models or blocked downloads instead of presenting fallback color features as CLIP results.",
    code='from fastembed import TextEmbedding, ImageEmbedding\ntext = TextEmbedding(\n    "Qdrant/clip-ViT-B-32-text")\nvision = ImageEmbedding(\n    "Qdrant/clip-ViT-B-32-vision")\nqv = next(text.embed([question]))',
    notebook="03_shared_space.ipynb", reference="27")

add("Normalize before a dot product", "03 · Shared space", "formula-panel",
    svg(txt(35, 95, "Length: √(3² + 4²) = 5", 35) + txt(35, 200, "[3, 4] / 5 = [0.6, 0.8]", 35, TEAL, 700) + txt(35, 310, "Unit vectors: cosine = dot product", 29, MUTED) + txt(35, 405, "Zero vectors need an explicit policy.", 26, CORAL), "Normalize a nonzero vector to unit length before using a dot product as cosine similarity", 700, 450),
    "The length of [3, 4] is 5 by the Pythagorean theorem. Dividing by 5 gives [0.6, 0.8], an arrow with length 1 pointing in the same direction. Once both vectors have length 1, their dot product equals cosine similarity. In numpy, axis=-1 computes one length per vector and keepdims=True preserves the dimensions needed for division. A zero vector has no direction, so reject it or define a clear fallback rather than quietly calling its cosine meaningful. Qdrant's cosine collection applies its own normalization behavior, while our numpy exercise makes the math explicit. Caution: normalizing does not make unrelated encoders compatible, and it does not calibrate scores into probabilities.",
    code="norm = np.linalg.norm(v)\nif norm == 0:\n    raise ValueError('Zero vector')\nu = v / norm\nscore = float(u @ query_unit)",
    notebook="03_shared_space.ipynb", reference="22, 27")

add("Finding a chart is not reading it", "03 · Shared space", "evidence-quote",
    svg(chart(10, 20, 660, 400), "Reading a chart requires verifying the bar values and labels, not simply retrieving its embedding", 700, 450),
    "Ask two different questions: ‘Find the monthly solar-energy chart’ and ‘How much more did May produce than April?’ CLIP may help with the first. The second requires reading 57 and 42, checking the kWh unit, and subtracting to obtain 15 kWh. A high retrieval similarity does not perform that arithmetic or certify the extracted values. Even a vision reader should cite the chart and qualify any illegible label. In the fixture lesson, known chart data makes the expected answer checkable. In a real report, you may prefer a trusted data table or human verification for precise numbers. A useful error analysis labels separately whether the image was missed, the values were misread, or the calculation was wrong.",
    lead="Find the chart. Read 57 and 42.", callout="Then calculate: 57 − 42 = 15 kWh.", notebook="03_shared_space.ipynb")

add("One space can favor one modality", "03 · Shared space", "comparison-bands",
    svg(box(30, 65, 570, 275, "One mixed ranking", PALE, "Text may crowd out images") + box(710, 65, 570, 275, "Measure by evidence type", MINT, "Memo · chart · screenshot") + txt(55, 295, "Compatible does not mean calibrated", 25, CORAL) + txt(735, 295, "Inspect who loses retrieval slots", 25, TEAL), "A shared embedding space can still show modality imbalance; inspect results by modality"),
    "The two towers can share compatible coordinates while text-text and text-image similarities follow different distributions. If all records compete in one mixed ranking, one modality may dominate the top results. Do not assume this must happen in your corpus; measure it. Examine representative queries whose gold evidence is a memo, an image, or both. A modality filter can reveal whether an image was useful but crowded out. Separate retrieval quotas or Strategy C are possible responses, each with tradeoffs. Counterexample: always reserving two image slots may hurt questions answered entirely by a safety memo. The notebook's simple color baseline cannot establish how a real CLIP model behaves; live results must be reported separately.",
    notebook="03_shared_space.ipynb", reference="28–29")

add("C · Let each specialist search", "04 · Fusion", "process-ribbon",
    svg(box(15, 150, 230, 125, "Question", MINT) + box(370, 35, 380, 125, "Text retrieval", MINT) + box(370, 280, 380, 125, "Image retrieval", PALE) + box(960, 150, 320, 125, "Rank fusion", MINT) + f'<path d="M250,212H300V97H355 M300,212V342H355 M765,97H860V212H945 M765,342H860V212" stroke="{TEAL}" stroke-width="3" fill="none"/>', "A query searches independent text and image indexes, then merges rankings"),
    "Strategy C keeps a text-specialist collection and an image collection with its paired text query encoder. The two paths search independently. Each returns a ranked list of source IDs, and a fusion rule combines those lists. This keeps unrelated vector spaces separate. The image path may use CLIP, while the text path uses a model optimized for prose. In the offline teaching path the image baseline is intentionally weaker and explicitly labeled. Fusion can combine complementary evidence but cannot recover an item absent from every candidate list. The candidate pool size and final answer context size are different parameters: retrieve enough candidates for fusion, then send only the most useful sources to the reader.",
    notebook="04_fusion_and_evaluation.ipynb", reference="29–31")

add("Raw scores use different rulers", "04 · Fusion", "story-rail",
    cards([("Text result", "0.82", "Text encoder's similarity"), ("Image result", "0.31", "Image encoder's similarity"), ("Conclusion", "?", "No common calibration")]),
    "Suppose the top text result has similarity 0.82 and the top image result has 0.31. It is tempting to declare the text result better. But these numbers come from different models and may have different distributions. They are not probabilities on a shared scale. One model can produce narrow score ranges and another broad ranges. Reciprocal rank fusion avoids that comparison by using order within each list. Proper score calibration is another approach, but it needs suitable labeled data and validation. Counterexample: multiplying image scores by three because one example looked too low is an arbitrary adjustment that may damage another query. These displayed numbers are a hypothetical demonstration, not measured scores from our notebooks.",
    notebook="04_fusion_and_evaluation.ipynb", reference="29–31")

add("RRF adds rewards for good ranks", "04 · Fusion", "formula-panel",
    svg(txt(30, 100, "score(d) = Σ 1 / (c + rank)", 40, TEAL, 700) + txt(30, 200, "First place → 1 / (60 + 1)", 31) + txt(30, 280, "Second place → 1 / (60 + 2)", 31) + txt(30, 390, "Missing from a list → no contribution", 26, MUTED), "Reciprocal rank fusion adds one over the smoothing constant plus the one-based rank in each list", 760, 450),
    "Reciprocal rank fusion, or RRF, rewards sources that appear near the top of one or more ranked lists. For every list containing source d, add 1 divided by c plus its rank. Start ranks at 1. A source missing from a list contributes zero for that list, not an invented last rank. The smoothing constant c controls how strongly first place beats lower places; 60 is a common starting value, not a universal optimum. We use c rather than k in this explanation so it is not confused with the number of results returned. Rank fusion deliberately discards similarity magnitudes. It can be robust across incompatible score scales, but does not prove relevance or require the lists to be statistically independent.",
    lead="Use position when scores are incompatible.", notebook="04_fusion_and_evaluation.ipynb", reference="30")

add("Count each source once per list", "04 · Fusion", "code-split",
    svg(box(35, 45, 615, 125, "chart_energy", MINT, "Caption + original image") + arrow(340, 180, 340, 255) + box(35, 275, 615, 115, "One source contribution", PALE), "Deduplicate representations of a source within each retrieval list before assigning source ranks", 700, 450),
    "A chart can be returned through several crops or captions. If one retrieval list counts every copy as separate support, a duplicated source unfairly gains score. Deduplicate each list by the chosen canonical source ID while preserving first occurrence, then assign one-based source ranks. The same source may legitimately contribute once from the text list and once from the image list. Decide the unit of relevance before evaluation: a PDF page, whole document, or video window may be the right unit for different tasks. Overly broad grouping can also be harmful; collapsing all video frames into a single video ID destroys temporal distinctions. The notebook helper makes the policy explicit and keeps provenance for the selected evidence.",
    code="for hits in ranked_lists:\n    ids = dedupe_sources(hits)\n    for rank, source_id in enumerate(ids, 1):\n        scores[source_id] += 1 / (c + rank)\nranked = sorted(scores,\n                key=scores.get, reverse=True)",
    notebook="04_fusion_and_evaluation.ipynb", reference="30")

add("Work one fusion example by hand", "04 · Fusion", "ledger-columns",
    svg(box(25, 50, 380, 300, "Text ranking", MINT, "1 memo · 2 chart") + txt(50, 195, "3 screenshot", 26) + box(460, 50, 380, 300, "Image ranking", PALE, "1 chart · 2 frame") + txt(485, 195, "3 screenshot", 26) + box(895, 50, 380, 300, "Chart wins", MINT, "1/62 + 1/61") + txt(920, 220, "≈ 0.03252", 38, TEAL, 800) + txt(35, 420, "Caption variant; c = 60", 25, MUTED), "The chart ranks second in text and first in images, giving RRF approximately 0.03252"),
    "Use c=60 and the two displayed lists. This example assumes captions let the chart and screenshot appear in the text list as well as the image list. If a text store contains only memos and its IDs never overlap image IDs, no source receives two contributions. The chart receives 1/62 from text and 1/61 from images, totaling about 0.032522. The screenshot receives 1/63 twice, totaling about 0.031746. The memo gets 1/61, about 0.016393, while the frame gets 1/62, about 0.016129. The fused order is chart, screenshot, memo, frame. Ask students to calculate the screenshot score before revealing it. Agreement can help, but can also reward a shared mistake. A fusion score is not an answer probability. These are illustrative rankings, not measured benchmark results.",
    notebook="04_fusion_and_evaluation.ipynb")

add("Narrow the evidence, then read", "04 · Reading", "process-ribbon",
    flow(["Question", "Retrieve", "Selected pixels", "Answer"], "Citations reconnect the answer to its sources."),
    "Retrieval is the inexpensive narrowing stage. Reading is the question-specific stage that inspects a small set of selected sources. Keep the retrieved source identity, load the corresponding original image or PDF crop, attach text evidence where relevant, and ask the vision-language model to answer only from that context. Limiting the candidate images controls latency, image-token cost, and irrelevant distraction. A larger context is not automatically more trustworthy. If no retrieved source supports the question, the correct result may be an explicit evidence gap. A model's general knowledge should not quietly replace missing evidence in a source-grounded assignment. The offline path demonstrates context construction and expected source handling without claiming a live model answer.",
    notebook="04_fusion_and_evaluation.ipynb", reference="32–36")

add("Send the source pixels explicitly", "04 · Reading", "code-split",
    svg(box(35, 65, 615, 125, "Text part", MINT, "Question + source label") + box(35, 250, 615, 125, "Image part", PALE, "Actual image bytes or URL"), "A vision request contains both question text and explicit image content", 700, 450),
    "A source path embedded in a text prompt does not transmit an image to a remote model. The request must include supported image content, for example a data URL containing base64-encoded bytes. The helper image_part handles that conversion and a suitable MIME type. LiteLLM standardizes the provider call, but each endpoint still needs an appropriate vision-capable model and configuration. The name gpt-5.6-sol requested for this course is a configurable gateway alias, not a promised public model identifier. Set VLM_MODEL for your actual gateway; text embeddings use their own embedding-model configuration. Secret values come from .env through load_dotenv(find_dotenv()). Do not print or commit them. Attach a clear source label beside each image so the response can cite the right evidence.",
    code='parts = [text_part(question)]\nparts += [text_part("[chart_energy]")]\nparts += [image_part(source_path)]\nreply = completion(model=VLM_MODEL,\n    messages=[{"role": "user",\n               "content": parts}])',
    notebook="04_fusion_and_evaluation.ipynb", reference="14, 33")

add("Evidence sets the answer boundary", "04 · Reading", "comparison-bands",
    svg(box(30, 70, 570, 275, "Supported", MINT, "May produced 57 kWh") + box(710, 70, 570, 275, "Unsupported", PALE, "May had the most sunshine") + txt(55, 290, "Visible bar and label", 26, TEAL) + txt(735, 290, "Cause is not in this chart", 26, CORAL), "The chart supports the energy value but not a causal explanation about sunshine"),
    "An answer can be fluent, plausible, and unsupported. The energy chart supports ‘May produced 57 kWh.’ It does not establish that May had the most sunshine, that the panel was more efficient, or that the school used less electricity. Distinguish observation, calculation, and explanation. A grounded answer may say ‘May was highest at 57 kWh [chart_energy]; this chart does not explain why.’ Check that every citation names a supplied source, then verify that the source actually supports the associated claim. A syntactically valid citation can point to irrelevant evidence. Also treat instructions found inside documents or screenshots as data: source content should not take control of the reader's task.",
    notebook="04_fusion_and_evaluation.ipynb", reference="33–36")

add("Define success before changing the system", "04 · Evaluation", "card-constellation",
    cards([("Question", "Query", "What should students ask?"), ("Evidence", "Gold IDs", "Which sources are relevant?"), ("Budget", "Top k", "How many sources fit?")]),
    "Create a small evaluation set before tuning the system. Each item contains a question and a set of relevant canonical source IDs. Include policy questions answered by memos, exact controller codes, visual questions, numeric chart questions, mixed evidence, and questions the corpus cannot answer. The gold set is about needed evidence, not just answer wording. For a multi-part question, several sources may be required. Decide whether duplicate representations count as one source before computing metrics. Keep held-out questions for the final comparison; repeatedly editing captions to fit every evaluation question leaks test information into the system. A small fictional corpus supports learning and debugging, not a broad claim that a strategy wins on all real-world data.",
    notebook="04_fusion_and_evaluation.ipynb", reference="37–40")

add("Hit rate and recall ask different things", "04 · Evaluation", "formula-panel",
    svg(txt(30, 80, "Gold: {memo, chart}", 32, weight=700) + txt(30, 170, "Retrieved: {chart, frame}", 32) + txt(30, 285, "Hit = 1", 48, TEAL, 800) + txt(350, 285, "Recall = 1/2", 42, CORAL, 800) + txt(30, 400, "Average per-query values to summarize.", 26, MUTED), "With two relevant sources and only one retrieved, the query hit is one while recall is one half", 760, 450),
    "For one query, hit@k is 1 if at least one relevant source appears in the first k results, otherwise 0. Recall@k is the number of relevant sources retrieved divided by the number of relevant sources in the gold set. If gold is {memo, chart} and the retrieved set is {chart, frame}, hit is 1 and recall is 1/2. Average per-query hit values to get hit rate, and per-query recall values to get macro recall. With exactly one relevant source per question, these values coincide, which can hide an incorrect metric name. The reference notebook labels an any-hit calculation as recall; this lesson separates them. For questions with no relevant source, recall has a zero denominator: evaluate abstention separately rather than silently assigning an arbitrary recall.",
    lead="One found source may be insufficient.", notebook="04_fusion_and_evaluation.ipynb", reference="37")

add("Compare like with like", "04 · Evaluation", "ledger-columns",
    svg(box(30, 60, 385, 295, "Hold constant", MINT, "Corpus · questions · k") + box(455, 60, 385, 295, "Report separately", PALE, "Fixture · live models") + box(880, 60, 385, 295, "Measure both", MINT, "Retrieval · answer quality"), "Fair evaluation holds corpus and queries fixed, separates fixture and live modes, and measures retrieval and answer quality independently"),
    "A fair A/B/C comparison uses the same corpus, gold questions, source identity policy, and final k. Record candidate-pool sizes and model choices because they can also affect results. Fixture results demonstrate pipeline behavior with authored captions and simple vectors. Live results use real embedding or vision models and must be labeled as such. Never replace an unavailable model with a weaker fallback and keep the original model's name in the score table. Retrieval quality and answer quality are separate: a correct source can be misread, while a model can guess a correct answer without retrieving evidence. Also report costs relevant to the decision, such as ingestion calls, query latency, and images sent per answer. This deck intentionally shows no invented benchmark comparison.",
    notebook="04_fusion_and_evaluation.ipynb", reference="37–40")

add("Turn a failure into an experiment", "04 · Evaluation", "offset-timeline",
    flow(["Observe", "Hypothesize", "Change one thing", "Retest"], "Keep the failed example in your regression set."),
    "Suppose E17 queries miss the controller screenshot. First inspect whether the caption contains E17. If it does not, the likely cause is extraction or representation. If it does, inspect the candidate ranking; exact lexical matching may help more than another vision call. If the screenshot is retrieved but the answer says E71, the problem is reading. Change one component at a time and rerun the same question plus neighboring cases. A single successful retest is not enough if the change harms other queries. Useful controlled experiments include adding exact-code fields, increasing candidate depth before fusion, comparing transcript-only with transcript-plus-frames, or changing the reader's image resolution. Preserve observations and uncertainty instead of explaining every failure with ‘the model is bad.’",
    notebook="04_fusion_and_evaluation.ipynb")

add("A PDF may need two routes", "05 · PDFs and video", "comparison-bands",
    svg(box(30, 60, 570, 280, "Selectable text", MINT, "Extract text → text search") + box(710, 60, 570, 280, "Scanned or visual page", PALE, "Render → OCR or image search") + txt(55, 292, "Check reading order", 26, TEAL) + txt(735, 292, "Keep the page pixels", 26, CORAL), "PDF pages containing selectable text use text extraction, while scanned and visual pages need rendering and OCR or visual representations"),
    "PDF is a format, not a single modality. Native text can often be extracted directly; scanned pages contain pixels and need OCR or visual processing. A chart may contain selectable axis labels but still require its visual arrangement to interpret it. Our tiny PDF has a native-text first page and a raster chart on page two so you can test both routes. Render pages at a readable resolution and use Pillow to inspect or crop images. Retain document ID and page number. A common counterexample is a two-column PDF whose extracted text interleaves unrelated sentences: text exists, but its reading order is broken. Inspect extracted output before treating it as reliable prose.",
    notebook="05_pdf_and_video.ipynb", reference="1, 4, 49")

add("A useful PDF citation is inspectable", "05 · PDFs and video", "editorial-thesis",
    svg(document(65, 35, 400, 355, "Report · page 2") + f'<rect x="90" y="210" width="350" height="135" rx="5" fill="{PALE}" stroke="{CORAL}" stroke-width="3"/>' + txt(112, 280, "Chart region", 29, CORAL, 700) + txt(50, 435, "Document + page + optional region", 27, MUTED), "Preserve the PDF document, one-based page number, and optional region coordinates", 700, 470),
    "A citation such as ‘the report’ is hard to check when the document is long. Preserve a document identifier, a user-facing page number, and optionally a crop or bounding box. Be explicit about indexing: PDF libraries often use zero-based page indices, while readers expect page 1, page 2, and so on. Region coordinates must record their coordinate system and page dimensions, especially if images are resized or rotated. A crop helps focus a reader but can remove the title, legend, or unit that gives the chart meaning. Keep enough surrounding context and a link to the full page. In this corpus, the energy chart is on user-facing page 2.",
    lead="Find the page. Preserve its context.", notebook="05_pdf_and_video.ipynb")

add("Video aligns two evidence streams", "05 · PDFs and video", "process-ribbon",
    svg(txt(25, 55, "Companion text", 28, weight=700) + txt(25, 250, "Sampled frames", 28, weight=700) + box(25, 80, 380, 105, "Energy · 0–4 s", MINT) + box(455, 80, 380, 105, "Controller · 4–8 s", MINT) + box(885, 80, 380, 105, "Temperature · 8–12 s", MINT) + box(25, 280, 380, 105, "Energy frame", PALE) + box(455, 280, 380, 105, "Controller frame", PALE) + box(885, 280, 380, 105, "Temperature frame", PALE), "The twelve-second teaching video aligns three visual scenes with three hand-authored text windows"),
    "A video search system can index speech transcripts as text and sampled frames as images, preserving their time alignment. Our twelve-second teaching video is silent. The accompanying text is hand-authored, not speech-recognition output, so it demonstrates alignment without pretending to transcribe absent audio. It shows energy from 0 to 4 seconds, the controller from 4 to 8, and temperature from 8 to 12, using half-open intervals in code. A real recording could use automatic speech recognition, but speech and visuals may disagree or appear at different times. Search both streams when they add value; first measure whether a transcript alone answers the target questions. Neither a transcript nor a single frame captures everything in a video.",
    notebook="05_pdf_and_video.ipynb", reference="41–45")

add("Sampling saves work and loses moments", "05 · PDFs and video", "code-split",
    svg(f'<path d="M40,240H650" stroke="{LINE}" stroke-width="4"/>' + ''.join(f'<circle cx="{x}" cy="240" r="15" fill="{TEAL if t < 12 else "white"}" stroke="{TEAL}" stroke-width="3"/>' + txt(x, 295, str(t)+(" s" if t < 12 else " s end"), 25, anchor="middle") for x,t in [(70,0),(250,4),(430,8),(610,12)]) + f'<rect x="320" y="130" width="42" height="110" fill="{PALE}" stroke="{CORAL}" stroke-width="2"/>' + txt(235, 85, "Brief event between samples", 26, CORAL), "A brief event between sampled frames can be missed even when nearby sampled frames are available", 700, 450),
    "This is an alternative seeking sketch inspired by the reference. The runnable notebook uses sequential decoding and frame_number/fps timestamps for the known constant-frame-rate teaching clip. OpenCV can also seek to a target time and read a frame, as shown here. Always check the success flag: the read may fail or the target may be beyond the last frame. Seeking and reported timing are approximate; use actual presentation timestamps for variable-frame-rate media. Sampling every few seconds reduces work but can miss a brief warning. Scene-change detection and near-duplicate removal are alternatives, each with errors. The dot at 12 seconds marks the video boundary, not a guaranteed decodable frame. Convert OpenCV's BGR channel order before displaying through Pillow or another RGB-based viewer.",
    code="cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)\nok, frame = cap.read()\nif not ok:\n    raise RuntimeError('No frame')\nrgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\nimage = Image.fromarray(rgb)",
    callout="Alternative seek sketch; notebook samples sequentially.", notebook="05_pdf_and_video.ipynb", reference="42–44")

add("A time window is not event proof", "05 · PDFs and video", "evidence-quote",
    svg(box(35, 60, 620, 290, "Retrieved window: 4–8 s", MINT, "A frame shows the controller") + txt(60, 270, "Inspect nearby frames for onset.", 29, CORAL) + txt(40, 420, "Cite the window you actually observed.", 26, MUTED), "Retrieving a four-to-eight-second window does not prove the exact onset time of an event", 700, 450),
    "Retrieving the controller window supports saying that the relevant scene is in the 4–8 second interval. It does not automatically prove the warning first appeared at exactly 4.000 seconds. A sampled frame shows what was visible at its decoded time; a transcript segment shows words assigned to a time range. They are different observations. For an onset question, inspect adjacent frames at a finer interval and report uncertainty. Formatting a timestamp as minutes and seconds is useful but does not improve temporal accuracy. Keep both t_start and t_end in metadata, and cite the supported window. Counterexample: a narrator saying ‘the warning appeared earlier’ during the current scene does not place the warning's onset at that sentence's timestamp.",
    lead="“Shown within 4–8 seconds”", notebook="05_pdf_and_video.ipynb", reference="44–47")

add("Features need a lifecycle too", "05 · Feature stores", "process-ribbon",
    flow(["Source", "Versioned features", "Search index", "Evidence"], "A feature store and vector index have different jobs."),
    "A feature is a derived property of an entity. In this project, features include OCR text, image dimensions, captions, and embeddings. A feature store manages reusable feature definitions and values, often with versioning, freshness, offline computation, and online serving. A vector database specializes in searching vectors and returning associated metadata. The concepts can connect, but they are not interchangeable, and this tutorial does not require installing a production feature platform. The local manifest is a small feature-registry teaching pattern. Keep the raw source separately; record how its features were produced; publish compatible vectors into the search index. Counterexample: a folder of arrays without source IDs or encoder versions is storage, but not a reliable feature lifecycle.",
    notebook="05_pdf_and_video.ipynb", reference="11, 20, 23, 27")

add("Reproduce what was known then", "05 · Feature stores", "offset-timeline",
    svg(box(25, 60, 385, 260, "Source event", MINT, "Observation time") + box(455, 100, 385, 260, "Feature available", PALE, "Processing finished") + box(885, 60, 385, 260, "Historical query", MINT, "Use available evidence") + txt(30, 425, "Future features must not leak into historical tests.", 27, CORAL), "Point-in-time evaluation respects both the event time and the time a feature became available"),
    "Suppose a lab observation happened at 10:00, but its verified chart caption became available at 10:20. A simulated query at 10:10 must not use that caption, even though the underlying event happened earlier. Track event time and availability time when historical correctness matters. This is the point-in-time problem familiar from feature stores: a later correction or reprocessing result should not leak into an earlier evaluation. Our small notebook records versions and demonstrates the principle; it is not a production point-in-time join engine. Also freeze the corpus and gold labels when comparing releases. Otherwise a model change may receive credit for a newer, easier corpus rather than better retrieval.",
    notebook="05_pdf_and_video.ipynb")

add("Changing an encoder changes the index", "05 · Feature stores", "comparison-bands",
    svg(box(30, 70, 570, 275, "Index version 1", MINT, "Encoder v1 + preprocessing v1") + box(710, 70, 570, 275, "Index version 2", PALE, "Recompute compatible features") + txt(55, 290, "Queries use the matching encoder", 25, TEAL) + txt(735, 290, "Evaluate before switching traffic", 25, CORAL), "An encoder upgrade requires recomputing compatible vectors and evaluating a new index before use"),
    "A new embedding model can move every item to a different coordinate system. Reusing old document vectors with new query vectors is unsafe even when their dimensions match. Version the encoder identity, model artifact or revision, preprocessing, parser prompt, source checksum, and output schema. Build and test a new index, then switch the query encoder and index together. Reparse only when that stage's inputs or logic changed; re-embed when the embedding representation changed. A source deletion should remove its searchable derived records and prevent stale citations. A lockfile fixes package versions, but model weights and prompts are separate artifacts. Counterexample: changing image cropping can alter the feature representation without changing the model name at all.",
    notebook="05_pdf_and_video.ipynb", reference="48")

add("Design your own controlled comparison", "Capstone", "card-constellation",
    cards([("Choose", "A / B / C", "Match a research question"), ("Measure", "Evidence", "Recall, grounding, and cost"), ("Defend", "Tradeoffs", "Include one failure case")]),
    "Choose a question family for the school corpus, such as exact controller codes, chart comparisons, or video scene location. Write your expected relevant sources before changing the system. Compare two retrieval strategies with fixed corpus, query set, and final k. Keep the fixture and live-model reports separate. Inspect one success and one failure in detail, then change only one component and rerun the evaluation. Your report should include the hypothesis, model and representation versions, source identity policy, retrieval metrics, answer-support checks, and the cost that matters for your setting. Add an unanswerable question and check abstention. A strong conclusion can be that OCR plus text retrieval is enough; complexity is justified only when it helps the actual task.",
    notebook="05_pdf_and_video.ipynb", reference="49–51")

add("Find evidence. Read carefully. Measure.", "Your next experiment", "manifesto-wall",
    svg(box(25, 65, 385, 300, "Represent", MINT, "Make evidence searchable") + box(455, 65, 385, 300, "Ground", PALE, "Inspect selected sources") + box(885, 65, 385, 300, "Evaluate", MINT, "Separate failure stages"), "Three principles: represent the evidence, ground answers in selected sources, and evaluate the separate stages"),
    "You now have a map of multimodal retrieval. Strategy A turns images into searchable words. Strategy B uses paired encoders with a shared space. Strategy C lets specialized paths retrieve independently and fuses their ranks. Parsing creates reusable representations; reading examines selected evidence for a specific question. PDF pages need provenance, video needs honest timestamps, and derived features need versioning. The next useful step is not automatically a larger model. It is a carefully chosen question, a clearly identified source, and a test that reveals where your pipeline succeeds or fails. Open the notebooks in order and record observations as you go. The theory guide explains the concepts in full, while FEATURE_STORES.md develops the lifecycle and historical-evaluation connections.",
    callout="Open the notebooks. Keep an experiment log.", notebook="01_foundations.ipynb")


CSS = '''
*{box-sizing:border-box}html,body{margin:0;width:1600px;height:900px;overflow:hidden}
body{font-family:Inter,Arial,Helvetica,sans-serif;color:#202729;background:#fff}
.slide{position:relative;width:1600px;height:900px;background:#fff;overflow:hidden}
.main-frame{position:absolute;left:96px;right:96px;top:108px;bottom:96px}
.section{font-size:19px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#08796f;margin:0 0 24px}
h1{font-size:57px;line-height:1.11;letter-spacing:-1.8px;font-weight:800;margin:0;max-width:1380px}
.content{height:474px;margin-top:32px;display:flex;align-items:center}
.content>svg{display:block;width:100%;max-height:450px}
.lead{font-size:31px;line-height:1.35;margin:0 0 28px;max-width:560px}
.callout{font-size:25px;line-height:1.35;color:#536164;margin:16px 0 0;font-weight:500}
.chrome-bottom{position:absolute;left:96px;right:96px;bottom:30px;display:flex;justify-content:space-between;font-size:18px;color:#536164}
.split-content{display:grid;grid-template-columns:.85fr 1.4fr;gap:56px;height:474px;margin-top:32px;align-items:center}
.split-content svg{width:100%;max-height:450px}
.code-content{display:grid;grid-template-columns:1.1fr 1fr;gap:38px;height:474px;margin-top:32px;align-items:center}
.code-content svg{width:100%;max-height:420px}
pre{font-family:'SFMono-Regular',Consolas,'Liberation Mono',monospace;font-size:23px;line-height:1.65;white-space:pre;margin:0;background:#f2f6f4;padding:32px 28px;border-radius:10px;color:#202729;overflow:visible}
.cover-swiss-rail{display:grid;grid-template-columns:.94fr 1.06fr;gap:52px;align-items:center;height:100%}
.cover-swiss-rail h1{font-size:94px;letter-spacing:-3.5px;line-height:1.03;margin-bottom:35px}
.cover-swiss-rail .lead{font-size:30px}.cover-swiss-rail svg{width:100%;max-height:610px}
.cover-swiss-rail .section{margin-bottom:42px}.cover-swiss-rail .callout{margin-top:65px}
.formula-panel .code-content{grid-template-columns:.85fr 1.25fr}
.formula-panel .split-content{grid-template-columns:.72fr 1.35fr}
.story-rail .content{padding-top:15px}.offset-timeline .content{padding-top:20px}
.evidence-quote .lead{font-size:43px;line-height:1.22;font-weight:650}
svg text{font-family:Inter,Arial,Helvetica,sans-serif}
'''


def slide_markup(s, i):
    heading = f'<p class="section">{esc(s["section"])}</p><h1>{esc(s["title"])}</h1>'
    caption = f'<p class="callout">{esc(s["callout"])}</p>' if s["callout"] else ""
    lead = f'<p class="lead">{esc(s["lead"])}</p>' if s["lead"] else ""
    if i == 0:
        inside = f'<div class="cover-swiss-rail"><div>{heading}{lead}<p class="callout">Cedar School solar lab<br>Five notebooks · one complete pipeline</p></div>{s["figure"]}</div>'
    elif s["code"]:
        inside = heading + f'<div class="code-content">{code_block(s["code"])}{s["figure"]}</div>' + caption
    elif s["lead"]:
        inside = heading + f'<div class="split-content"><div>{lead}{caption}</div>{s["figure"]}</div>'
    else:
        inside = heading + f'<div class="content">{s["figure"]}</div>' + caption
    return f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=1600"><title>{esc(s["title"])} · Multimodal retrieval</title><style>{CSS}</style></head><body><article class="slide {s["layout"]}" aria-label="Slide {i+1}: {esc(s["title"])}"><main class="main-frame">{inside}</main><footer class="chrome-bottom"><span>© mui-group</span><span>{i+1:02d} / {len(SLIDES):02d}</span></footer></article></body></html>'


SHELL = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Multimodal retrieval · Tutorial slides</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#edf2ef;color:#202729;font-family:Inter,Arial,Helvetica,sans-serif}button,select,a{font:inherit}button,select{color:#202729;background:#fff;border:1px solid #a4b5ae;border-radius:5px;padding:9px 12px}button{cursor:pointer}button:hover{background:#e9f6f2}button:focus-visible,select:focus-visible,a:focus-visible,#stage:focus-visible{outline:3px solid #bf4938;outline-offset:3px}button:disabled{opacity:.5;cursor:default}nav{min-height:70px;padding:12px 24px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;background:#fff}nav a{color:#08796f;margin-left:auto}.brand{font-weight:800;margin-right:12px}#counter{font-variant-numeric:tabular-nums;min-width:65px}#jump{max-width:360px}#stage{position:relative;margin:18px auto;background:#fff;overflow:hidden;width:calc(100vw - 36px);height:calc(100vh - 110px)}#slide{position:absolute;border:0;width:1600px;height:900px;transform-origin:top left;left:0;top:0;background:#fff;pointer-events:none}#notes{margin:0 auto 30px;padding:24px 30px;max-width:1080px;background:#fff;border-radius:8px;font-size:19px;line-height:1.65}#notes h2{font-size:25px;margin:0 0 14px}#notes p{margin:0 0 16px}#notes a{color:#08796f}#help{font-size:14px;color:#536164;margin:0 auto 16px;text-align:center}body.notes-open #stage{height:calc(70vh - 110px);min-height:260px}@media(max-width:700px){nav{padding:10px;gap:7px}.brand{width:100%}#jump{max-width:220px}nav a{margin-left:0}#stage{height:55vh}#help{padding:0 15px}#notes{margin:10px;font-size:17px}}@media print{nav,#help{display:none}#stage{width:100%;height:auto;margin:0}#slide{position:relative;transform:none!important}#notes{display:block!important}}
</style></head><body>
<nav aria-label="Presentation controls"><span class="brand">Multimodal retrieval</span><button id="prev" aria-label="Previous slide">←</button><span id="counter" aria-live="polite"></span><button id="next" aria-label="Next slide">→</button><select id="jump" aria-label="Jump to slide"></select><button id="toggle" aria-expanded="false" aria-controls="notes">Speaker notes</button><a href="slides/outputs/pptx/multimodal-retrieval.pptx" download>Download PowerPoint</a><a href="README.md">Module guide</a><a id="standalone" target="_blank" rel="noopener">Open static slide</a></nav>
<noscript><style>#stage,#help,#prev,#next,#counter,#jump,#toggle{display:none}</style><p style="padding:24px;font-size:22px">JavaScript is disabled. <a href="slides/index.html">Read all slides and speaker notes in the static index.</a></p></noscript>
<main id="stage" tabindex="0" aria-label="Current slide. Use left and right arrows to navigate."><iframe id="slide" title="Current tutorial slide" tabindex="-1"></iframe></main>
<p id="help">← → navigate · Home / End · N toggles notes · Standalone slides: 1600 × 900</p>
<aside id="notes" hidden aria-label="Speaker notes"><h2 id="notes-title"></h2><p id="notes-text"></p><p id="notes-links"></p></aside>
<script>
const slides = __DATA__;
const el = id => document.getElementById(id);
let current = 0;
slides.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${s.title}`;el('jump').append(o)});
function resize(){const stage=el('stage');const w=stage.clientWidth,h=stage.clientHeight;const z=Math.min(w/1600,h/900);el('slide').style.transform=`translate(${(w-1600*z)/2}px,${(h-900*z)/2}px) scale(${z})`;}
function show(i){current=Math.max(0,Math.min(slides.length-1,i));const s=slides[current];const path=`slides/outputs/html/slide_${String(current+1).padStart(2,'0')}.html`;el('slide').src=path;el('slide').title=`Slide ${current+1}: ${s.title}`;el('standalone').href=path;el('counter').textContent=`${current+1} / ${slides.length}`;el('jump').value=current;el('prev').disabled=current===0;el('next').disabled=current===slides.length-1;el('notes-title').textContent=s.title;el('notes-text').textContent=s.notes;el('notes-links').replaceChildren();if(s.notebook){const a=document.createElement('a');a.href=s.notebook;a.textContent=`Notebook: ${s.notebook}`;el('notes-links').append(a);}if(s.reference){el('notes-links').append(document.createTextNode(` · Reference notebook cells ${s.reference} (zero-based).`));}history.replaceState(null,'',`#slide-${current+1}`);resize();}
function toggleNotes(){const open=el('notes').hidden;el('notes').hidden=!open;el('toggle').setAttribute('aria-expanded',String(open));document.body.classList.toggle('notes-open',open);resize();}
el('prev').onclick=()=>show(current-1);el('next').onclick=()=>show(current+1);el('jump').onchange=e=>show(Number(e.target.value));el('toggle').onclick=toggleNotes;
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowRight'||e.key==='PageDown'){e.preventDefault();show(current+1)}if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();show(current-1)}if(e.key==='Home'){e.preventDefault();show(0)}if(e.key==='End'){e.preventDefault();show(slides.length-1)}if(e.key.toLowerCase()==='n'){e.preventDefault();toggleNotes()}});
window.addEventListener('resize',resize);el('slide').addEventListener('load',()=>{resize();try{el('slide').contentWindow.addEventListener('keydown',e=>{if(['ArrowRight','ArrowLeft','PageDown','PageUp','Home','End','n','N'].includes(e.key)){e.preventDefault();document.dispatchEvent(new KeyboardEvent('keydown',{key:e.key}))}})}catch(_){}});const match=location.hash.match(/^#slide-(\d+)$/);show(match?Number(match[1])-1:0);
</script></body></html>'''


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(SLIDES):
        (OUT / f"slide_{i+1:02d}.html").write_text(slide_markup(s, i), encoding="utf-8")
    manifest = []
    roles = {"cover-swiss-rail": "cover", "card-constellation": "synthesis", "process-ribbon": "chronology", "editorial-thesis": "thesis", "comparison-bands": "comparison", "ledger-columns": "comparison", "code-split": "synthesis", "story-rail": "metric", "evidence-quote": "thesis", "formula-panel": "metric", "offset-timeline": "chronology", "manifesto-wall": "closing"}
    for i, s in enumerate(SLIDES):
        manifest.append({k: v for k, v in s.items() if k != "figure"} | {"number": i + 1, "role": roles[s["layout"]], "file": f"outputs/html/slide_{i+1:02d}.html"})
    (ROOT / "slides" / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "slides.html").write_text(SHELL.replace("__DATA__", json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8")
    notes = ["# Multimodal retrieval: speaker notes", "", "These notes expand the intentionally concise slides. Reference cell numbers are zero-based indices into `S6_Multimodal_Retrieval.ipynb`. Code snippets on slides are shortened architecture excerpts; run the complete notebook cells.", ""]
    for i, s in enumerate(SLIDES):
        notes += [f"## {i+1:02d}. {s['title']}", "", s["notes"], "", f"Notebook: [{s['notebook']}](../{s['notebook']})", ""]
        if s["reference"]:
            notes += [f"Reference notebook cells: {s['reference']} (zero-based).", ""]
    (ROOT / "slides" / "speaker_notes.md").write_text("\n".join(notes), encoding="utf-8")
    index = ['<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Multimodal retrieval · Static slide index</title><style>body{font:20px/1.6 Inter,Arial,sans-serif;color:#202729;max-width:1000px;padding:30px;margin:auto}a{color:#08796f}h1{line-height:1.15}li{padding:10px 0}summary{cursor:pointer}details{margin:8px 0 20px}details p{color:#536164}footer{margin-top:40px}</style></head><body><h1>Multimodal retrieval</h1><p>42 slides with full explanations. This page works without JavaScript.</p><p><a href="../slides.html">Open presentation</a> · <a href="outputs/pptx/multimodal-retrieval.pptx" download>Download PowerPoint</a> · <a href="../README.md">Module guide</a></p><ol>']
    for i, s in enumerate(SLIDES):
        index.append(f'<li><a href="outputs/html/slide_{i+1:02d}.html">{esc(s["title"])}</a><details><summary>Speaker notes</summary><p>{esc(s["notes"])}</p></details></li>')
    index.append('</ol><footer>© mui-group</footer></body></html>')
    (ROOT / "slides" / "index.html").write_text("\n".join(index), encoding="utf-8")
    print(f"Built {len(SLIDES)} static slides, shell, manifest, and presenter notes.")


if __name__ == "__main__":
    build()
