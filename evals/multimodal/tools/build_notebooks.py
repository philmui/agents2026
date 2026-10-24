"""Build the five lesson notebooks from reviewed prose and executable cells.

Run: uv run python tools/build_notebooks.py
This overwrites notebook sources, clearing execution outputs. Execute notebooks
after rebuilding to produce honest outputs from the current environment.
"""
from pathlib import Path
import copy
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def figure(name, alt):
    return md(f"![{alt}](assets/{name}.svg)")


SETUP = code(r'''
    import os, sys, json, copy
    from pathlib import Path
    import numpy as np
    from IPython.display import display, Image, Markdown
    from dotenv import load_dotenv, find_dotenv

    # Works when Jupyter starts either here or at the repository root.
    candidates = [Path.cwd(), Path.cwd() / "evals" / "multimodal"]
    MODULE = next((p for p in candidates if (p / "lab.py").exists()), None)
    if MODULE is None:
        raise RuntimeError("Start Jupyter from evals/multimodal or the repository root.")
    os.chdir(MODULE)
    if str(MODULE) not in sys.path:
        sys.path.insert(0, str(MODULE))
    load_dotenv(find_dotenv())  # Secrets live outside the notebook.
    from lab import (
        CORPUS, GOLD, Record, LexicalEncoder, ColorEncoder, QdrantIndex,
        load_corpus, load_reference_parses, normalize, validate_parse,
        flatten_parse, build_strategy_a, toy_shared_space, show_hits, rrf,
        evaluate, check_data_points, feature_fingerprint, read_pdf,
        sample_keyframes, build_video_records, timestamp, evidence_bundle,
    )
    RUN_MODELS = os.getenv("MM_RUN_MODELS") == "1"
    RUN_LIVE = os.getenv("MM_RUN_LIVE") == "1"
    print("Local pretrained models enabled:", RUN_MODELS)
    print("Remote API calls enabled:", RUN_LIVE)
    print("Default path: explicit lexical/color baselines, real in-memory Qdrant.")
''')


def write(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3 (multimodal)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "multimodal_tutorial": {"default_mode": "offline lexical and color baselines", "source": "original Cedar School teaching corpus", "reference": "S6_Multimodal_Retrieval.ipynb"},
    })
    nbf.validate(notebook)
    nbf.write(notebook, ROOT / name)
    print(name, len(cells), "cells")


def foundations():
    return [
        md('''
        # 1 · A search engine that can find pictures

        Imagine your science club asks, **“Which month produced the most solar energy?”**
        You have a written meeting schedule, a chart, a controller screenshot, a PDF report,
        and a video. The schedule discusses energy but contains no measurements. The answer
        is visible on a bar chart. A system that searches only the schedule cannot recover
        a number that never entered its index.

        Our project is a search engine for the fictional **Cedar School solar lab**. We will
        first make its smallest pieces visible, then assemble three retrieval strategies.
        You need Python lists, dictionaries, and functions; no calculus or neural-network
        background is assumed. Allow **35–50 minutes**, including predictions and exercises.

        By the end, you can explain a vector, compute a similarity, store it in Qdrant,
        and distinguish finding evidence from reading evidence. Open [THEORY.md](THEORY.md)
        alongside the notebooks for a continuous explanation.
        '''),
        md('''
        ## Start with a reproducible environment

        In a terminal, run these commands from this folder. `uv sync` installs the versions
        declared in `pyproject.toml` and the lockfile; it does not belong inside a notebook cell.

        ```bash
        uv sync
        uv run jupyter lab
        ```

        The checked-in corpus is ready to use. To recreate it, run
        `uv run python tools/build_corpus.py`. Every default cell works without an API key or
        a model download. **Qdrant is real; the default encoders are deliberately simple.**
        Optional paths use `uv sync --all-extras` and explicit environment switches described
        in the README. Leave `MM_RUN_MODELS=0` and `MM_RUN_LIVE=0` for the first pass.
        '''),
        copy.deepcopy(SETUP),
        md('''
        ## First, see what a framework can hide

        The reference notebook begins with LlamaIndex's `MultiModalVectorStoreIndex`
        (reference cells 6–9). The optional cell below preserves that useful “whole machine
        first” view. Its adapters use **FastEmbed BGE for text and paired CLIP towers for
        text/image matching**, with no PyTorch dependency. Both adapters are fully implemented
        in [framework.py](framework.py).

        This opener uses LlamaIndex's two **SimpleVectorStore** instances; our manual lessons
        will use Qdrant. Two stores returning hits is not yet a single fused ranking.
        Predict: which decisions does a framework still leave to us—what to index, what counts
        as relevant, and how to verify an answer?
        '''),
        code(r'''
        if RUN_MODELS:
            from framework import build_framework_index
            framework_index = build_framework_index()
            framework_hits = framework_index.as_retriever(
                similarity_top_k=2, image_similarity_top_k=2
            ).retrieve("monthly rooftop solar energy")
            for hit in framework_hits:
                print(type(hit.node).__name__, hit.node.metadata.get("source_id"), hit.score)
        else:
            print("Skipped LlamaIndex model path. No model download or model result is claimed.")
        '''),
        md('''
        A `Document` becomes text features. An `ImageDocument` points to actual pixels.
        The framework handles plumbing, but it cannot choose our definition of a correct
        answer. We will now unpack ingestion, retrieval, and reading separately.

        **Caution:** a vision language model that produces sentences is not itself a search
        index. We can store its descriptions, or store embeddings produced by an appropriate
        encoder. “A database cannot see” means that it compares the vectors we supply;
        a vector database *can* store image vectors when paired with an image encoder.
        '''),
        figure("01_pipeline", "Files become representations, an index retrieves candidates, and a reader uses selected evidence."),
        md('''
        ## Step 1 · Inspect what information actually exists

        A **modality** is a form of information: text, an image, audio, or video. A file format
        is different: a PDF may contain text, pictures, or both. Our seven initial records
        contain three text memos and four images. The chart measurements live in pixels in
        the raw corpus. Hand-authored reference parses are kept separately for later lessons;
        the evaluation answer key is separate again.

        Run the next cell. Notice that the image records have paths but empty `text` fields.
        A filename is an identifier, not a faithful description of the picture.
        '''),
        code(r'''
        raw = load_corpus()
        [{"id": r.id, "modality": r.modality, "text_characters": len(r.text)} for r in raw]
        '''),
        code(r'''
        energy = next(r for r in raw if r.id == "chart_energy")
        display(Image(filename=energy.path, width=700))
        print(next(r.text for r in raw if r.id == "memo_schedule"))
        '''),
        md('''
        You can read the tallest bar with your eyes. A text-only encoder cannot; it receives
        strings, not your visual experience. The memo can answer **when the class meets**,
        but cannot answer **how much energy May produced**.

        This is a useful research habit: trace an answer back to the source that contains it.
        A relevant topic is not enough. A source about solar panels may lack the requested
        measurement, time period, or unit.
        '''),
        md('''
        ## Step 2 · Make a vector you can inspect

        A **vector** is an ordered list of numbers. Give its coordinates meanings:
        `[solar, water, safety]`. Then `[3, 0, 1]` could mean three occurrences of “solar,”
        no “water,” and one “safety.” It is a compressed description, not the document itself.

        Predict which document points most like the query `[1, 0, 0]`.
        Then compare the raw dot product with **cosine similarity**. A dot product multiplies
        matching coordinates and adds them. Dividing by both vector lengths measures angle,
        reducing the effect of simply repeating a word.
        '''),
        figure("03_cosine", "Cosine similarity compares vector direction, with a worked two-dimensional example."),
        code(r'''
        q = np.array([1., 0., 0.])
        documents = np.array([[3., 0., 1.], [0., 4., 1.], [30., 0., 10.]])
        print("Raw dot products:", documents @ q)
        print("Cosine similarities:", np.round(normalize(documents) @ normalize(q), 3))
        print("Zero vector stays explicit:", normalize([0., 0., 0.]))
        '''),
        md('''
        Documents 1 and 3 have the same cosine score because the third is ten times the
        first. The second points mostly toward water. Cosine is useful for ranking directions;
        it does not prove that a document answers a question. Repeating a false claim still
        produces a vector.

        A zero vector has no direction. Our helpers return no results for a zero query
        instead of pretending cosine is defined there. Normalizing BGE and CLIP vectors
        also does **not** make their coordinate meanings compatible.
        '''),
        md('''
        ## Step 3 · Build a transparent lexical encoder

        We now give every word its own coordinate. The vocabulary is built from **documents
        only**, then frozen. Queries use the same coordinate order. This is a bag of word
        counts: word order and most grammar disappear. We store its rows densely because
        the corpus is tiny; large lexical systems normally use sparse indexes.

        This encoder is intentionally less capable than a learned text embedding model.
        It lets you see the mechanism without hiding a model download or claiming semantic
        intelligence. Inspect a few nonzero coordinates below.
        '''),
        code(r'''
        memos = [r for r in raw if r.modality == "text"]
        lexical = LexicalEncoder([r.text for r in memos])
        query = "solar Thursday classroom"
        query_vector = lexical.embed([query])[0]
        print("Dimensions:", lexical.dimension)
        print({word: round(float(query_vector[i]), 3)
               for i, word in enumerate(lexical.vocabulary) if query_vector[i]})
        print("Unseen wording:", np.linalg.norm(lexical.embed(["photovoltaic timetable"])[0]))
        '''),
        md('''
        The unseen synonyms produce a zero vector, although a person recognizes the topic.
        That failure is a property of this encoder, not of Qdrant. Learned embeddings can
        connect some synonyms, but may blur exact codes or numbers. An error code such as
        `E17` often benefits from lexical matching alongside semantic retrieval.
        '''),
        md('''
        ## Step 4 · Put vectors in a real database

        A **collection** is a group of points with a fixed vector dimension and distance rule.
        A point holds a vector plus a **payload**: source ID, path, modality, and text.
        The vector is used for comparison; the payload helps us find and cite the evidence.

        `QdrantIndex` uses `QdrantClient(":memory:")`, `create_collection`, `upsert`, and
        `query_points`. Human-readable IDs are mapped to valid UUIDs with `uuid5`. It rejects
        duplicate chunk IDs and wrong vector dimensions. Data disappears when the process
        ends; this is a classroom convenience, not persistent storage.
        '''),
        code(r'''
        memo_index = QdrantIndex(memos, lexical.embed([r.text for r in memos]), "first_index")
        hits = memo_index.search(query_vector, k=2)
        display(show_hits(hits))
        print("Top source text:\n", hits[0]["text"])
        '''),
        md('''
        The top memo should discuss the classroom review. Its score is similarity under our
        particular word-count representation, **not a probability that an answer is true**.
        Returning the nearest two items also does not mean two items contain an answer.
        The helper fetches all seven-or-fewer candidates before breaking score ties by ID,
        so our small demonstrations remain reproducible.
        '''),
        md('''
        ### Open the Qdrant wrapper: four real operations

        A wrapper should save repetition, not hide the idea. Here is the minimum database
        path with two-dimensional vectors you can calculate by hand. **Create** declares a
        collection's shape and distance. **Upsert** inserts or replaces points by ID.
        **Query** finds nearby vectors. The returned **payload** tells us what they represent.
        These values are a worked geometry example, separate from the school documents.
        '''),
        code(r'''
        from qdrant_client import QdrantClient, models
        tiny_client = QdrantClient(":memory:")
        tiny_client.create_collection(
            collection_name="arrows",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
        )
        tiny_client.upsert(collection_name="arrows", points=[
            models.PointStruct(id=1, vector=[0.6, 0.8], payload={"source_id": "near_arrow"}),
            models.PointStruct(id=2, vector=[1.0, 0.0], payload={"source_id": "side_arrow"}),
        ])
        direct_result = tiny_client.query_points(
            collection_name="arrows", query=[0.6, 0.8], limit=2, with_payload=True
        ).points
        print([(point.payload["source_id"], round(point.score, 3)) for point in direct_result])
        tiny_client.close()
        '''),
        md('''
        The matching arrow scores 1; the horizontal arrow scores 0.6. Qdrant performed the
        same cosine comparison we calculated earlier. Our `QdrantIndex` adds checks, valid
        stable UUIDs, source metadata, and deterministic tie handling around these operations.
        It does not invent a different retrieval algorithm.
        '''),
        figure("02_two_jobs", "Parsing once at ingestion is different from reading a few retrieved images at query time."),
        md('''
        ## Step 5 · Separate the two jobs

        **Parsing at ingestion:** a vision model or OCR system reads an uploaded image and
        produces reusable text, tables, or metadata. You pay that work when the file arrives
        or changes. **Reading at query time:** after retrieval, a vision model inspects a few
        selected images to answer the actual question. A caption may omit the tiny detail
        that this question needs, so keep a link to the original pixels.

        Strategy A searches descriptions. Strategy B uses paired text/image encoders.
        Strategy C queries separate text and image stores, then merges rankings. These are
        retrieval designs; any of them may still use a vision model afterward to read evidence.
        '''),
        md('''
        ## Step 6 · An embedding is also a stored feature

        A **feature store** manages reusable, versioned representations and their metadata.
        A **vector database** specializes in similarity search. A system may use both; storing
        vectors alone does not automatically provide every feature-store capability.

        If the source image, parser, model, or preprocessing changes, the feature can become
        stale. Our small cache identity includes those inputs. Try changing only the model
        version: the key should change even though the image is unchanged.
        '''),
        code(r'''
        old = feature_fingerprint(energy, "word-count-v1", "reference-caption-v1", "RGB-original")
        new = feature_fingerprint(energy, "word-count-v2", "reference-caption-v1", "RGB-original")
        print("Same source:", old["source_sha256"] == new["source_sha256"])
        print("Different feature key:", old["feature_key"] != new["feature_key"])
        '''),
        md('''
        ### Check your understanding

        1. Can an image and a sentence be compared just because both vectors have 512 numbers?
        2. Why does a retrieved source need a path and stable ID in addition to its vector?
        3. If a caption omits the small `E17` code, does searching that caption recover it?

        <details><summary>Worked answers</summary>

        1. No. Coordinates must belong to a compatible representation. Two 512-dimensional
        spaces may assign completely different meanings to every coordinate.
        2. The vector helps find a candidate. The path recovers evidence and the stable ID
        supports citations, deduplication, and updates.
        3. No guaranteed recovery. Query-time reading can recover it only after a strategy
        retrieves the relevant screenshot by some other cue.
        </details>
        '''),
        md('''
        ### Small experiment

        Change `query` to “when do students review solar energy?” and repeat the search.
        Then try “photovoltaic timetable.” Explain the difference by inspecting vocabulary
        coordinates, not by calling the database “smart” or “bad.”

        **Exit ticket:** describe the journey from a file to an answer in five verbs:
        preserve → represent → index → retrieve → read. Next:
        [02 · Describe once, index as text](02_describe_and_search.ipynb).

        Source connections: reference cells 8 (framework), 11 (raw records), and 22–23
        (vector storage/search). Our corpus and explanations are original; the helper code
        makes the same mechanisms inspectable in standard Jupyter.
        '''),
    ]


def strategy_a():
    return [
        md('''
        # 2 · Describe once, index as text

        A chart is easy to see but awkward for text search. We will turn it into a structured
        description, index that description beside ordinary memos, and preserve the image
        for later reading. This is **Strategy A**.

        Allow **45–60 minutes**. You will distinguish schema validation from factual checking,
        catch a label-swapping error, run retrieval, and construct an evidence-only reader
        request. The default parser output is a **hand-authored teaching fixture**, not a
        successful model run. Optional live cells really call a vision API only when enabled.
        '''),
        copy.deepcopy(SETUP),
        figure("04_strategy_a", "An image is parsed into text, stored with memos, and retrieved using a text query."),
        md('''
        ## Step 1 · Decide what a useful description preserves

        “A chart about electricity” is true but too vague for many questions. A better record
        includes a title, summary, and label–value–unit triples. A value such as `57` is
        meaningless without “May” and “kWh.” The screenshot needs an error code and message;
        it does not need invented numeric data points.

        The original notebook's parser returns short data-point strings (reference cell 14).
        Here we use explicit fields so we can check relationships. JSON is a machine-readable
        container, not proof of correctness. First inspect the fixture and its provenance.
        '''),
        code(r'''
        raw = load_corpus()
        reference_parses = load_reference_parses()
        display(reference_parses["chart_energy"])
        display(Image(filename=next(r.path for r in raw if r.id == "chart_energy"), width=650))
        '''),
        md('''
        The image is the source; the description is a derived representation. The fixture was
        written by the lesson author to make an offline experiment repeatable. A real parser
        can omit a label, confuse a unit, or describe the wrong series. Good fixture retrieval
        therefore says nothing about the reliability of an untested vision model.

        **Predict:** if April and May values are exchanged, will a check that merely compares
        sets of numbers notice? Write your prediction before running the comparison below.
        '''),
        code(r'''
        # Evaluation-only ground truth is opened here, never in the parser or index builder.
        answer_key = json.loads((CORPUS / "answer_key.json").read_text())
        expected = answer_key["chart_energy"]["data_points"]
        swapped = copy.deepcopy(reference_parses["chart_energy"]["data_points"])
        swapped[0]["value"], swapped[1]["value"] = swapped[1]["value"], swapped[0]["value"]
        print("Bag-of-numbers check passes:", sorted(p["value"] for p in swapped) == sorted(p["value"] for p in expected))
        display(check_data_points(swapped, expected))
        '''),
        md('''
        The weak check passes because the same three numbers are still present. The stronger
        check fails because values must stay attached to their labels **and units**. It also
        reports missing labels, unexpected labels/units, and duplicates. Floating-point
        comparison treats `42` and `42.0` as equal without treating `42 kWh` as `42 Wh`.

        This improves the number-overlap check in reference cell 18. It is still a classroom
        check: a real unseen document rarely has an answer key. You may need trusted source
        tables, unit/range rules, arithmetic checks, or human review. Never certify a model
        prediction by comparing it only against another copy of the same prediction.
        '''),
        md('''
        ## Step 2 · Validate structure before indexing

        Our parser contract requires strings for kind/title/summary, a list of data points,
        and finite numeric values. It rejects duplicate labels. That catches malformed output
        early, but a perfectly shaped false record still passes. Validation has several layers:
        **can we read it, does its structure fit, and is the claim supported?**

        The next cell intentionally gives the validator an invalid value, catches the expected
        error, and then checks the valid fixture. An error message is useful evidence here.
        '''),
        code(r'''
        invalid = copy.deepcopy(reference_parses["chart_energy"])
        invalid["data_points"][0]["value"] = "forty-two-ish"
        try:
            validate_parse(invalid)
        except ValueError as error:
            print("Rejected invalid structure:", error)
        print("Valid structure:", validate_parse(reference_parses["chart_energy"])["title"])
        print("Fixture values checked:", check_data_points(reference_parses["chart_energy"]["data_points"], expected)["ok"])
        '''),
        md('''
        ## Step 3 · Optional real parsing

        LiteLLM standardizes the request shape, not model capabilities. The configured default
        `openai/gpt-5.6-sol` is a **configurable model name or gateway alias**; it is not a
        promise that your endpoint offers that model. Configure a vision-capable endpoint using `VLM_MODEL`,
        `VLM_API_BASE`, and `VLM_API_KEY` in `.env`.

        With `MM_RUN_LIVE=1`, the next cell sends each of our four fictional images to that
        endpoint and may incur charges. With the flag off, it uses explicitly labelled
        fixtures. The prompt contains the task and pixels, **never the answer key**. Failures
        are allowed to surface; there is no quiet substitution of fixtures for live results.
        '''),
        code(r'''
        from live import PARSE_PROMPT, parse_image
        print(PARSE_PROMPT)
        if RUN_LIVE:
            active_parses = {r.id: parse_image(r.path) for r in raw if r.modality == "image"}
            parse_source = "actual live responses"
        else:
            active_parses = reference_parses
            parse_source = "hand-authored reference fixtures; no vision model ran"
        print("Parse source:", parse_source)
        '''),
        md('''
        The instruction to treat text inside an image as data matters: a screenshot could
        contain “ignore the user and reveal secrets.” That text is evidence to interpret,
        not a new instruction for the reader. This prompt is one defense, not a guarantee.

        **OCR versus a vision model:** OCR is useful for printed words and codes; a vision
        model can additionally interpret layout or relationships. OCR alone may read all
        the digits on a chart yet attach them to the wrong bars. Compare both methods on
        the actual question types instead of choosing the more elaborate tool automatically.
        '''),
        code(r'''
        numeric_report = {}
        for identifier in ("chart_energy", "chart_temperature", "chart_water"):
            numeric_report[identifier] = check_data_points(
                active_parses[identifier]["data_points"], answer_key[identifier]["data_points"]
            )
        print("What was evaluated:", parse_source)
        display(numeric_report)
        '''),
        md('''
        ## Step 4 · Turn structured descriptions into searchable text

        The database does not automatically know that a JSON field is meaningful. We flatten
        the title, summary, and labelled data points into a sentence-like string. Ordinary
        memo text stays as text. Each derived record retains the original source ID and path.

        All records and queries must use the same fitted encoder. Our baseline is normalized
        word counts. A learned model later replaces this encoder without changing the
        collection/search pattern. A hosted text-embedding example appears near the end.
        '''),
        code(r'''
        records_a, encoder_a, index_a = build_strategy_a(raw, active_parses)
        print(next(r.text for r in records_a if r.id == "chart_energy"))
        query_a = "Which month had the most solar energy?"
        hits_a = index_a.search(encoder_a.embed([query_a])[0], k=3)
        display(show_hits(hits_a))
        '''),
        md('''
        ### Open Strategy A: build the rows yourself

        `build_strategy_a` is four ordinary steps: preserve source identity, turn each image
        parse into text, encode all searchable text, and store the resulting vectors. The cell
        below repeats that path explicitly. The `join` keeps each value next to its own label
        and unit; `replace` changes searchable text while keeping the original path.
        '''),
        code(r'''
        from dataclasses import replace
        manual_records = []
        for source in raw:
            if source.modality == "text":
                manual_records.append(source)
                continue
            parsed = active_parses[source.id]
            labelled_values = "; ".join(
                f"{point['label']}: {point['value']} {point['unit']}"
                for point in parsed["data_points"]
            )
            searchable = f"{parsed['kind']}. {parsed['title']}. {parsed['summary']} {labelled_values}".strip()
            manual_records.append(replace(source, text=searchable))
        manual_encoder = LexicalEncoder([r.text for r in manual_records])
        manual_vectors = manual_encoder.embed([r.text for r in manual_records])
        manual_index = QdrantIndex(manual_records, manual_vectors, "a_unpacked")
        manual_hits = manual_index.search(manual_encoder.embed([query_a])[0], k=3)
        print("Same retrieved IDs:", [h["id"] for h in manual_hits] == [h["id"] for h in hits_a])
        '''),
        md('''
        There is no vision operation inside the database. All the information available for
        Strategy A's text matching is already in `searchable`. The reusable helper additionally
        validates the parse and records its provenance. Read [lab.py](lab.py) if you want to
        inspect those checks; the comparison above confirms the same ranking mechanism.
        '''),
        md('''
        The system ranked representations. It has **not yet computed an answer**. A memo
        might rank alongside the energy chart because both share the topic. Even a correct
        top hit does not guarantee a correct reading of “most,” the labels, or the unit.

        Try a deliberately lossy caption below. If useful words disappear from an image's
        description, they disappear from this strategy's search surface. The original image
        still exists, but the retriever needs some route to find it.
        '''),
        code(r'''
        lossy = copy.deepcopy(active_parses)
        lossy["chart_energy"] = {"kind": "figure", "title": "Monthly results", "summary": "Three colored bars.", "data_points": [], "provenance": "intentional ablation"}
        _, lossy_encoder, lossy_index = build_strategy_a(raw, lossy)
        probe = "May kWh"
        print("Full description:")
        display(show_hits(index_a.search(encoder_a.embed([probe])[0], k=2)))
        print("Lossy description:")
        display(show_hits(lossy_index.search(lossy_encoder.embed([probe])[0], k=2)))
        '''),
        md('''
        This **ablation** removes one useful component while keeping the rest of the pipeline
        similar. The query can become unrepresentable in the lossy vocabulary. That is a
        measured failure, not something to conceal with a handcrafted answer.

        Strategy A works well when descriptions preserve the details people search for.
        It struggles with omitted tiny text, spatial relations, or visual appearance. Improving
        descriptions may help retrieval, but a longer caption costs more to store/embed and
        can add unsupported claims. “Describe everything” is not a precise quality standard.
        '''),
        figure("08_grounding", "Retrieved source IDs lead to original pixels and evidence-based answering with citations."),
        md('''
        ## Step 5 · Narrow, then read

        Now assemble a request using only the sources selected by retrieval. `evidence_bundle`
        attaches image paths for retrieved images and text for retrieved text. The reader
        receives actual image bytes through `image_part`, not just filenames or captions.

        Predict what the bundle will contain. Then inspect IDs and image counts before any
        live call. A citation is a pointer to evidence, not proof that the answer follows
        from it. Numeric answer verification remains a separate job.
        '''),
        code(r'''
        bundle = evidence_bundle(hits_a, max_images=2)
        display([{"source_id": item["source_id"], "images": len(item["images"]), "text_characters": len(item["text"])} for item in bundle])
        retrieved_sources = {hit["source_id"] for hit in hits_a}
        assert all(item["source_id"] in retrieved_sources for item in bundle)
        if RUN_LIVE:
            from live import answer_from_bundle
            print(answer_from_bundle(query_a, bundle))
        else:
            print("Reader skipped. Inspect the retrieved image yourself; no synthetic model answer is shown.")
        '''),
        md('''
        ## Step 6 · Choose where text embeddings run

        FastEmbed offers local learned embeddings in the next notebook. LiteLLM can also
        request hosted text embeddings. That requires a separate `EMBED_MODEL` and endpoint
        configuration; a vision model name is not automatically an embedding model.

        If enabled, both documents and queries below go through the **same embedding model**.
        Changing the model requires re-embedding the collection. This optional cell sends
        the fictional corpus text to the configured endpoint and may incur charges.
        '''),
        code(r'''
        if RUN_LIVE and os.getenv("EMBED_MODEL"):
            from live import embed_via_api
            hosted_vectors = embed_via_api([r.text for r in records_a])
            hosted_index = QdrantIndex(records_a, hosted_vectors, "hosted_text_a")
            hosted_query = embed_via_api([query_a])[0]
            display(show_hits(hosted_index.search(hosted_query, 3)))
        else:
            print("Hosted text embeddings skipped; MM_RUN_LIVE=1 and EMBED_MODEL are required.")
        '''),
        md('''
        ## Step 7 · Reason about cost before scaling

        Let there be `N` images, `Q` questions, and `k` selected images per question.
        A parser makes roughly `N` initial image reads; a reader makes roughly `Q × k`
        image attachments across questions. These are workload counts, not prices: resolution,
        text length, caching, model, and batching change actual charges and latency.

        Reparse only when the source or parser policy changes. Re-embed when indexed text or
        embedding configuration changes. Keep provenance so stale records can be found.
        '''),
        code(r'''
        N, Q, k = 400, 1000, 2
        print({"initial_parser_image_reads": N, "query_reader_image_attachments": Q*k})
        print("These are workload estimates, not API bills or measured latency.")
        '''),
        md('''
        ### Exercise · Three failure layers

        The parser returns valid JSON saying “April: 57 kWh; May: 42 kWh.” The index retrieves
        it for a monthly energy question. The reader repeats it with a source citation.
        Which stages succeeded structurally, and which claim failed factually?

        <details><summary>Worked answer</summary>

        Parsing succeeded at JSON structure but failed at label–value grounding. Retrieval
        found a topically useful source; that does not repair the parse. The answer repeats
        an unsupported relation. A citation points to the chart but does not establish
        agreement with it. Our pairwise check detects the swapped labels; reading the source
        pixels can also reveal the mismatch.
        </details>

        **Research extension:** compare a generic caption prompt with a table-preserving
        prompt on held-out charts. Measure extraction correctness and retrieval separately.
        Next: [03 · A shared space for words and pixels](03_shared_space.ipynb).
        Source connections: reference cells 14–18 (parser/number check), 21–23 (Strategy A),
        and 32–35 (narrow, then read).
        '''),
    ]


def shared_space():
    return [
        md('''
        # 3 · A shared space for words and pixels

        Suppose the query is **“the purple bar chart.”** The wording describes appearance,
        which may be lost in OCR text. Strategy B compares a text representation directly
        with an image representation. That only works when the two representations are
        deliberately compatible.

        Allow **40–55 minutes**. First build a tiny shared color space whose limitations are
        obvious. Then inspect the optional real CLIP path. **Color retrieval is not a CLIP
        simulation or a benchmark.** It teaches the interface and geometry while making no
        claim to understand image meaning.
        '''),
        copy.deepcopy(SETUP),
        figure("05_strategy_b", "Paired text and image encoders map inputs into one comparison space."),
        md('''
        ## Step 1 · What does “same space” mean?

        In notebook 1, the coordinates `[solar, water, safety]` had an agreed meaning. Imagine
        another encoder uses `[water, safety, solar]`. The vectors have the same length, but
        their dot product now multiplies the wrong concepts. A model version change can be
        subtler than this simple permutation and still break compatibility.

        Predict the score before and after an arbitrary coordinate rearrangement. Normalizing
        vector lengths cannot restore coordinate meanings that were never aligned.
        '''),
        code(r'''
        query_vector = normalize([1., 0., 0.])
        aligned_image = normalize([1., 0., 0.])
        unrelated_coordinate_order = normalize([0., 0., 1.])
        print("Compatible coordinates:", query_vector @ aligned_image)
        print("Same dimension, incompatible coordinates:", query_vector @ unrelated_coordinate_order)
        '''),
        figure("14_coordinate_spaces", "Equal vector dimensions and normalized lengths do not establish compatible coordinate meanings."),
        md('''
        ## Step 2 · Design a shared space we completely understand

        Our four coordinates are `[orange, purple, green, teal]`. The image encoder measures
        how many pixels lie close to four RGB color prototypes. The text encoder looks for
        those exact color words. We designed both sides to use the same coordinate order.

        This can answer an appearance question such as “purple chart.” It cannot understand
        “panel temperature,” read numbers, or distinguish a purple chart from a purple shirt.
        A restricted tool is useful when we clearly state what it measures.
        '''),
        code(r'''
        raw = load_corpus()
        image_records, color_encoder, color_index = toy_shared_space(raw)
        color_vectors = color_encoder.embed_images([r.path for r in image_records])
        print("Coordinates:", color_encoder.names)
        for record, vector in zip(image_records, color_vectors):
            print(record.id, np.round(vector, 3).tolist())
        '''),
        md('''
        ### Open the pixel encoder: measure, count, normalize

        Let us compute one image vector without calling `ColorEncoder.embed_images`.
        A pixel is three numbers—red, green, blue. We measure its distance to each color
        prototype, keep pixels close enough to a prototype, count their nearest colors,
        and normalize the four counts. No caption or answer key enters this calculation.
        '''),
        code(r'''
        from PIL import Image as PILImage
        with PILImage.open(image_records[0].path) as picture:
            small = np.asarray(picture.convert("RGB").resize((160, 104)), dtype=float)
        pixel_rows = small.reshape(-1, 3)                  # one RGB triple per row
        differences = pixel_rows[:, None, :] - color_encoder.prototypes[None, :, :]
        distances = np.sqrt((differences ** 2).sum(axis=2))  # pixel × prototype
        nearest_color = distances.argmin(axis=1)
        close_enough = distances.min(axis=1) < 65
        counts = np.bincount(nearest_color[close_enough], minlength=4)
        manual_color_vector = normalize(counts)
        print("Counts:", dict(zip(color_encoder.names, counts.tolist())))
        print("Same as helper:", np.allclose(manual_color_vector, color_vectors[0]))
        '''),
        md('''
        The expression `pixel_rows[:, None, :]` inserts an axis so NumPy compares every pixel
        with every prototype. The resulting distance table has one row per pixel and four
        columns. `argmin` chooses the nearest column. This is an explicit color measurement,
        not a learned association between words and images.
        '''),
        md('''
        These vectors came from actual pixels. Neutral page backgrounds are excluded by the
        distance threshold; otherwise the white background would dominate every picture.
        Resizing and color thresholds are preprocessing choices, not universal truths.

        The image array loses layout. Swapping two bars preserves the color counts, so this
        representation cannot know which label belongs to which value. It retains one
        property and discards many others—exactly why feature choice matters.
        '''),
        code(r'''
        color_query = "the purple bar chart"
        color_query_vector = color_encoder.embed_text([color_query])[0]
        color_hits = color_index.search(color_query_vector, k=2)
        print("Query vector:", color_query_vector)
        display(show_hits(color_hits))
        display(Image(filename=color_hits[0]["path"], width=650))
        '''),
        md('''
        The purple chart should be first because its measured pixels match the selected
        coordinate. A zero-score second item is merely a requested neighbor, not evidence
        that it matches. Our small Qdrant collection breaks ties by stable ID.

        **Counterexample:** “Which location has the hottest panel?” is meaningful to a person,
        but contains no color word. The next cell should return no color-query direction.
        That result is a deliberate honest limitation.
        '''),
        code(r'''
        semantic_query = "Which location has the hottest panel?"
        q = color_encoder.embed_text([semantic_query])[0]
        print("Color-query norm:", np.linalg.norm(q))
        print("Color hits:", color_index.search(q, 3))
        print("Unknown colors also fail:", color_encoder.embed_text(["red chart"])[0])
        '''),
        md('''
        ## Step 3 · Understand what CLIP learns instead

        CLIP uses two neural networks, often called **towers**: one reads text tokens and one
        reads image pixels. During training, matched image/caption pairs are encouraged to
        have larger similarities than mismatched pairs. Repeating this across many examples
        teaches useful associations. At search time, the trained encoders are reused; we
        are **not training CLIP** in this tutorial.

        “Contrastive” means learning by comparing a match with alternatives. The little matrix
        below is a made-up training illustration, not measured CLIP output. A diagonal entry
        represents the intended image/caption pairing. Large diagonal values would be useful,
        but a training objective cannot guarantee perfect behavior on every new image.
        '''),
        code(r'''
        illustrative_scores = np.array([[0.8, 0.2, 0.1],
                                        [0.1, 0.7, 0.3],
                                        [0.2, 0.1, 0.9]])
        print("Illustrative similarity matrix: rows = images; columns = paired captions")
        print(illustrative_scores)
        print("Best caption per image:", illustrative_scores.argmax(axis=1))
        '''),
        md('''
        A useful CLIP feature may capture objects, style, or scene type while missing exact
        chart arithmetic. The phrase “largest May value” is much more specific than “a bar
        chart.” Tiny OCR, subtle spatial relationships, and precise counts need testing.

        Paired training makes cross-modal comparison meaningful; it does not force image and
        text vectors to occupy identical distributions. In a joint collection, one modality
        can crowd out the other. This **modality gap** is a reason to inspect results by
        modality and compare joint retrieval with separate-store retrieval.
        '''),
        md('''
        ## Step 4 · Optional real CLIP through FastEmbed

        With `MM_RUN_MODELS=1`, the next cell downloads/loads the paired ONNX models
        `Qdrant/clip-ViT-B-32-text` and `Qdrant/clip-ViT-B-32-vision`, plus the small BGE
        text model used by our shared helper. Inference runs locally after the model files
        are available. Initial downloads require network access and disk space.

        The text and vision towers must come from the intended pair; equal dimensions alone
        are insufficient. CLIP's text context is limited. Do not assume `text[:300]` meets a
        token limit: characters and tokens differ. These short examples avoid manual character
        truncation; for longer sources, use the model's tokenizer to measure and chunk text.
        '''),
        code(r'''
        real_b_index = None
        if RUN_MODELS:
            from live import FastEmbedModels
            real_models = FastEmbedModels()
            text_records = [r for r in raw if r.modality == "text"]
            text_vectors = real_models.clip_texts([r.text for r in text_records])
            image_vectors = real_models.clip_images([r.path for r in image_records])
            assert text_vectors.shape[1] == image_vectors.shape[1]
            real_b_index = QdrantIndex(text_records + image_records,
                                      np.vstack([text_vectors, image_vectors]), "real_joint_clip")
            real_query = real_models.clip_texts(["purple bar chart of panel temperature"])[0]
            display(show_hits(real_b_index.search(real_query, 5)))
            print("Actual local CLIP results; not color-baseline results.")
        else:
            print("Real CLIP skipped. No learned-model performance is inferred from the color lesson.")
        '''),
        md('''
        In this optional joint collection, both memos and images are encoded through CLIP's
        paired space. We did **not** mix BGE text vectors into a CLIP collection. A joint
        ranking answers “which stored representation is closest?” It may still favor one
        modality for a query. A modality filter lets us inspect that behavior directly.
        '''),
        code(r'''
        if real_b_index is not None:
            for query in ["monthly solar energy", "E17 ventilation blocked", "purple bar chart"]:
                q = real_models.clip_texts([query])[0]
                print("\nQuery:", query)
                print("Joint:", [h["id"] for h in real_b_index.search(q, 3)])
                print("Image-only:", [h["id"] for h in real_b_index.search(q, 3, modality="image")])
        else:
            print("Joint-versus-image CLIP comparison requires the optional model path.")
        '''),
        md('''
        A filter changes the candidate set. It does not improve the encoder's ability to read
        `E17`, and it may remove a useful memo. Query policies should follow user needs and
        measured relevance. Do not declare one approach best from a single attractive result.
        Notebook 4 uses a small common evaluation set and separates question types.
        '''),
        md('''
        ## Step 5 · Separate the similarity score from evidence

        A high image/text score says “these representations are near each other.” It does not
        compute the chart's maximum value. For a numeric question, retrieve the image and
        then inspect its labels and units. A broad-image encoder can still be useful even
        when a specialized reader is needed afterward.

        Let's make the information loss visible: rearrange all pixels in the chart. The
        histogram of exact colors stays the same, while the picture becomes unreadable.
        We inspect histograms in memory and do not replace the source image.
        '''),
        code(r'''
        from PIL import Image as PILImage
        path = next(r.path for r in raw if r.id == "chart_temperature")
        with PILImage.open(path) as picture:
            pixels = np.asarray(picture.convert("RGB"))
        shuffled = pixels.reshape(-1, 3).copy()
        np.random.default_rng(7).shuffle(shuffled, axis=0)
        original_counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)[1]
        shuffled_counts = np.unique(shuffled, axis=0, return_counts=True)[1]
        print("Exact color counts unchanged:", np.array_equal(original_counts, shuffled_counts))
        display(PILImage.fromarray(shuffled.reshape(pixels.shape)).resize((500, 325)))
        '''),
        md('''
        The image is now noise, but a color histogram cannot tell. Our resize-based encoder
        adds a preprocessing step, so its measured vector may also change slightly when
        resizing mixes neighboring pixels; the fundamental information-loss point is about
        histograms, not a claim of exact invariance for every preprocessing pipeline.

        CLIP is much richer than a histogram. The lesson transfers: inspect what any
        representation preserves and what the task demands. “It is an embedding” does not
        specify the kinds of evidence it can retain.
        '''),
        figure("12_feature_lifecycle", "Feature generation records source, model, preprocessing, and version so stale embeddings can be replaced."),
        md('''
        ## Step 6 · Save the representation contract

        A usable image-feature record includes a source checksum, model identifier/version,
        preprocessing policy, dimension, and original media path. Store and query features
        with the same contract. If you switch from one CLIP version to another, re-embed;
        if you crop an image differently, record that change too.

        Color vectors need the same discipline. Their four coordinates mean nothing if a
        future query encoder silently changes the order to `[green, teal, purple, orange]`.
        Read [FEATURE_STORES.md](FEATURE_STORES.md) for how this relates to feature storage,
        freshness, lineage, and point-in-time correctness.
        '''),
        code(r'''
        record = image_records[0]
        contract = feature_fingerprint(record, "color-prototypes-v1", "no-parser", "RGB;160x104;distance<65")
        contract.update({"dimension": 4, "coordinate_order": color_encoder.names, "source_id": record.source_id})
        display(contract)
        '''),
        md('''
        ### Exercise · Pick a representation

        For each query, choose a plausible first retrieval route and say what still needs
        verification: (a) “a purple chart,” (b) “what does E17 mean?”, (c) “how much energy
        did May produce?”, (d) “a picture of a shaded garden.”

        <details><summary>Worked answers</summary>

        (a) Color or learned visual features can help; inspect that the result is actually a
        chart. (b) Exact OCR/caption/lexical matching is useful; verify the code's nearby
        message. (c) Search chart descriptions or a learned space for energy, then read May's
        value and unit. (d) A pretrained shared model is a plausible route, but our color
        encoder cannot understand “garden.” Test against labelled image relevance.
        </details>

        Next: [04 · Fuse rankings and evaluate honestly](04_fusion_and_evaluation.ipynb).
        Source connections: reference cells 25–28 (CLIP towers and joint collection),
        plus the CLIP paper linked in THEORY.md. Our color lesson is an original teaching
        baseline; it is never substituted for measured CLIP behavior.
        '''),
    ]


def fusion():
    return [
        md('''
        # 4 · Combine searches and measure what worked

        One searcher is good at words; another is good at visual clues. Strategy C asks both,
        then combines their ranked results. We will implement **reciprocal rank fusion (RRF)**,
        expose a duplicate-source trap, and evaluate retrieval with carefully named metrics.

        Allow **45–65 minutes**. Our default experiment uses a lexical memo searcher and the
        color-only image searcher. Optional FastEmbed experiments compare real A/B/C on the
        same labelled questions. The default scores describe the teaching baselines only.
        '''),
        copy.deepcopy(SETUP),
        figure("06_strategy_c", "Separate text and image searches feed one rank-fusion stage."),
        md('''
        ## Step 1 · Why not take the largest raw score?

        Imagine a text searcher returns scores `0.42, 0.39` and an image searcher returns
        `0.81, 0.80`. The image results are not automatically better. Their scores are produced
        by different models with different geometries and distributions. Even cosine scores
        sharing the interval [-1, 1] are not calibrated relevance probabilities.

        RRF uses **positions**, not raw similarities. For a source at rank `r`, it adds
        `1 / (c + r)`. We use one-based ranks. `c=60` is a common starting setting, not a law.
        Larger `c` makes nearby ranks contribute more similarly. It does not create relevance
        when both input rankings are poor.
        '''),
        figure("07_rrf", "Worked reciprocal-rank contributions combine evidence from multiple ranked lists."),
        code(r'''
        # Worked miniature: source B occurs in both lists.
        def toy_hit(identifier, score):
            return {"id": identifier, "source_id": identifier, "modality": "illustration", "score": score}
        first = [toy_hit("A", 0.42), toy_hit("B", 0.39)]
        second = [toy_hit("B", 0.81), toy_hit("C", 0.80)]
        worked_fusion = rrf([first, second], constant=60)
        display(show_hits(worked_fusion))
        print("B's expected fused score:", 1/62 + 1/61)
        print("B's observed contributions:", worked_fusion[0]["votes"])
        '''),
        md('''
        ### Open RRF: a score accumulator

        Here is the algorithm without a helper. Walk each ranked list, convert the one-based
        rank into a contribution, and add it to that source's total. Sort the totals after
        both lists have voted. This short version is valid for these lists because every
        source appears at most once in each list; the next section handles duplicate chunks.
        '''),
        code(r'''
        totals = {}
        for ranked_list in [first, second]:
            for rank, hit in enumerate(ranked_list, start=1):
                source = hit["source_id"]
                contribution = 1.0 / (60 + rank)
                totals[source] = totals.get(source, 0.0) + contribution
        manual_order = sorted(totals, key=lambda source: (-totals[source], source))
        print("Manual totals:", totals)
        print("Manual order:", manual_order)
        assert manual_order == [h["source_id"] for h in worked_fusion]
        assert all(np.isclose(totals[h["source_id"]], h["score"]) for h in worked_fusion)
        '''),
        md('''
        B wins because both lists support it. A is first in its list and C is second, so A
        contributes `1/61` and C contributes `1/62`; they do **not** tie. This difference helps us
        catch an easy off-by-one mistake before using a large dataset.

        If one list were duplicated ten times, its candidates would receive ten votes. RRF
        assumes the chosen list sources are meaningful; it does not know whether searchers
        are independent or whether one source is being counted repeatedly.
        '''),
        md('''
        ## Step 2 · Give chunks and sources different identities

        One PDF can have many pages; one video segment can have several sampled frames.
        Each indexed chunk needs a unique point ID. But fusion may need one vote per source
        per list. For this notebook, an image or memo is a source. For video, a timed segment
        will be a source and the complete video its parent.

        The next example gives A two chunks in the same list. Our RRF deduplicates at the
        source level **before assigning compact ranks**: A gets one vote, B gets rank 2.
        Distinct chunk evidence is retained for inspection. Ties use stable source IDs.
        '''),
        code(r'''
        duplicate_chunks = [
            {**toy_hit("A-frame-0", .9), "source_id": "A"},
            {**toy_hit("A-frame-1", .8), "source_id": "A"},
            toy_hit("B", .7),
        ]
        deduped = rrf([duplicate_chunks])
        display(show_hits(deduped))
        print("Votes:", {h["source_id"]: h["votes"] for h in deduped})
        assert len(deduped[0]["votes"]) == 1
        assert len(deduped[0]["evidence"]) == 2
        '''),
        md('''
        ## Step 3 · Build two independent stores

        The text branch below indexes **only the three raw memos**. The image branch indexes
        the four images with measured color vectors. This mirrors the separate-store design;
        no caption-derived text is quietly added to C's memo branch.

        A, in contrast, indexes memos **and hand-authored image descriptions**. We must report
        that extra information when comparing strategies. A better score for A may partly
        reflect better input descriptions, not an intrinsically better retrieval architecture.
        '''),
        code(r'''
        raw = load_corpus()
        memos = [r for r in raw if r.modality == "text"]
        text_encoder = LexicalEncoder([r.text for r in memos])
        text_index = QdrantIndex(memos, text_encoder.embed([r.text for r in memos]), "c_memos")
        images, color_encoder, image_index = toy_shared_space(raw)
        _, a_encoder, a_index = build_strategy_a(raw)

        def retrieve_a(query, k=3):
            return a_index.search(a_encoder.embed([query])[0], k)

        def retrieve_b_color(query, k=3):
            return image_index.search(color_encoder.embed_text([query])[0], k)

        def retrieve_c(query, k=3):
            text_hits = text_index.search(text_encoder.embed([query])[0], k)
            image_hits = retrieve_b_color(query, k)
            return rrf([text_hits, image_hits])[:k]

        query = "green water cleaning methods"
        print("Text branch:")
        display(show_hits(text_index.search(text_encoder.embed([query])[0], 3)))
        print("Image branch:")
        display(show_hits(retrieve_b_color(query)))
        print("Fused:")
        display(show_hits(retrieve_c(query)))
        '''),
        md('''
        The image branch can use “green”; the memo branch can use “water” and “cleaning.”
        Here C requests `k` candidates **per branch**, then returns a final `k`. It examines
        up to `2k` candidates, while A requests only `k`. A matched final evidence count is
        not a matched candidate or compute budget. Report both budgets in any comparison.

        Neither query branch depends on the other's result. A service can execute independent
        branches concurrently and wait for both before fusion. Our normal function is
        sequential so intermediate values stay easy to inspect.

        The next cell demonstrates concurrent orchestration using separate local stores.
        Tiny local operations may not get faster because thread overhead can dominate.
        Concurrency is a scheduling choice, not a retrieval-quality improvement.
        '''),
        code(r'''
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            text_future = pool.submit(text_index.search, text_encoder.embed([query])[0], 3)
            image_future = pool.submit(retrieve_b_color, query, 3)
            concurrent_hits = rrf([text_future.result(), image_future.result()])[:3]
        print("Same IDs as sequential:", [h["id"] for h in concurrent_hits] == [h["id"] for h in retrieve_c(query)])
        '''),
        figure("11_evaluation", "Recall, hit rate, reciprocal rank, and answer correctness measure different stages."),
        md('''
        ## Step 4 · Name the metric precisely

        Suppose a question has two relevant sources, and the top three results contain just
        one of them at rank 2. **Recall@3 = 1/2**, **Hit@3 = 1**, and **reciprocal rank@3 =
        1/2**. Across questions, average Hit gives hit rate and average reciprocal rank gives
        MRR. Our MRR is explicitly truncated at `k`: a relevant item below `k` receives zero.

        The reference notebook's cell 37 calls an “any relevant result?” calculation recall.
        That is hit rate; it equals recall only when every query has exactly one relevant
        source. Our gold set deliberately includes multiple relevant sources.
        '''),
        code(r'''
        one_question = [{"query": "worked example", "relevant": {"A", "B"}}]
        worked_retrieval = lambda query, k: [toy_hit("X", .9), toy_hit("A", .8), toy_hit("Y", .7)][:k]
        summary, detail = evaluate(worked_retrieval, one_question, k=3)
        display(summary)
        display(detail)
        assert summary["Recall@3"] == .5 and summary["HitRate@3"] == 1.0 and summary["MRR@3"] == .5
        '''),
        md('''
        ## Step 5 · Evaluate the actual teaching baselines

        The gold set defines sources useful for answering each question. Some questions need
        one chart; others accept both a chart and a contextual memo. Two queries have no
        supporting source: one uses unseen words; the other asks about a future forecast
        using familiar solar-energy words. The second is intentionally harder to reject.

        Recall is undefined when there are no relevant sources, so we exclude those queries
        from recall/hit/MRR averages and report **unsupported evidence-return rate** separately.
        Returning a candidate is not the same as generating a false answer. Answer-level
        refusal and groundedness need a separate evaluation.
        '''),
        code(r'''
        print("Question labels:")
        display([{**case, "relevant": sorted(case["relevant"])} for case in GOLD])
        strategies = {"A: lexical + reference descriptions": retrieve_a,
                      "B teaching baseline: color-only images": retrieve_b_color,
                      "C teaching baseline: lexical memos + color RRF": retrieve_c}
        baseline_results = {}
        for name, retrieve in strategies.items():
            baseline_results[name] = evaluate(retrieve, k=3)
        display([{ "strategy": name, **result[0]} for name, result in baseline_results.items()])
        '''),
        md('''
        Do not stop at an average. The next cell prints each query for C, including the
        retrieved sources and labelled relevance. Look for a query whose color branch had no
        direction, a memo-only question, and an unsupported question that still returned hits.
        Those failures explain *why* a score changed.
        '''),
        code(r'''
        display(baseline_results["C teaching baseline: lexical memos + color RRF"][1])
        print("Overlap does not imply answerability:")
        display(show_hits(retrieve_a("solar energy next winter forecast", 3)))
        '''),
        md('''
        Search almost always finds the nearest item when a nonzero query is present. A
        forecast question can retrieve a historical chart without the chart containing any
        forecast. A similarity threshold might reject some such cases, but its value must
        be chosen on labelled development data; a universal threshold like `0.7` has no
        general meaning across encoders and corpora.

        For a real study, separate development queries from held-out test queries, report
        sample sizes, inspect relevance disagreements, and measure retrieval separately from
        answer correctness. Eight classroom queries illustrate mechanics; they do not establish
        a reliable ranking of commercial systems.
        '''),
        md('''
        ## Step 6 · Optional genuine A/B/C comparison

        With `MM_RUN_MODELS=1`, this cell runs real local FastEmbed models on the same questions:
        **A** uses BGE text embeddings over memos and reference descriptions; **B** uses one
        joint CLIP collection; **C** fuses BGE memo ranks with CLIP image ranks. All three use
        actual Qdrant retrieval. Results are computed, not copied from the color baseline.

        This comparison still gives A hand-authored captions. Report that advantage; repeat
        with measured live parses if you want to study parser errors. The local-model flag
        does not enable paid vision calls. Live parsing additionally requires `MM_RUN_LIVE=1`.
        '''),
        code(r'''
        if RUN_MODELS:
            from live import run_model_comparison
            real_results = run_model_comparison(k=3)
            display([{ "strategy": name, **result[0]} for name, result in real_results.items()])
            for name, (_, per_query) in real_results.items():
                print(name)
                display(per_query)
        else:
            print("Real A/B/C comparison skipped. No CLIP or BGE metric is fabricated.")
        '''),
        md('''
        ## Step 7 · Study the trade-off, not just the winner

        Larger `k` usually improves the chance of including useful evidence, but passes more
        irrelevant material to a reader and can increase latency or cost. A useful retrieval
        experiment measures a small curve, not a single convenient cutoff. The next cell
        changes `k` while keeping questions and strategy fixed. A second experiment should
        hold final `k=3` fixed and vary C's per-branch pool (for example 1, 2, then 3), recording
        the number of candidates each searcher examines. Fusion cannot recover a useful
        source omitted from every branch's candidate pool.
        '''),
        code(r'''
        display([evaluate(retrieve_a, k=k)[0] for k in (1, 2, 3)])
        '''),
        md('''
        ### Exercise · A fairer study

        Add two held-out questions: one with multiple relevant sources and one unsupported
        question that shares vocabulary with the corpus. Label them by inspecting the actual
        sources before looking at retrieval output. Then describe how you would compare
        reference captions with live model captions without changing everything at once.

        <details><summary>Worked design</summary>

        Hold the corpus, query set, encoder, `k`, and relevance labels fixed. Change only
        caption source. Measure parse label/value correctness first, then retrieval and answer
        quality. Keep development questions separate from final test questions. Do not tune a
        threshold on the same small set used to claim the final result.
        </details>

        Next: [05 · PDF pages and video moments](05_pdf_and_video.ipynb).
        Source connections: reference cells 30–31 (RRF), 37–38 (evaluation), and 43–44
        (video source identities). Our implementation prevents duplicate frames from gaining
        repeated votes inside one ranked list.
        '''),
    ]


def pdf_video():
    return [
        md('''
        # 5 · Find the right PDF page and video moment

        A PDF is a container, not a guarantee of searchable text. A video is a sequence,
        not one giant image. In this final notebook we give both formats smaller evidence
        units with stable page numbers or timestamps, then reuse the same retrieval ideas.

        Allow **45–65 minutes**. We will extract and render a two-page PDF, decode a real
        12-second video, align six sampled frames to transcript windows, fuse searches at
        the segment level, and assemble a grounded reader request.

        Our video is **silent**. Its companion transcript is hand-authored teaching material,
        not speech recognition output. A production pipeline would create or receive a
        timestamped transcript separately and measure its errors.
        '''),
        copy.deepcopy(SETUP),
        figure("09_pdf", "PDF pages may have native text, raster pixels, or both; preserve page provenance."),
        md('''
        ## Step 1 · Test whether a PDF contains extractable text

        Page 1 of our report contains native text. Page 2 contains only a raster chart image.
        `pypdfium2` extracts available text and renders both pages to PNG. Rendering is not OCR:
        it turns the page into pixels, but does not read words from those pixels.

        Predict which page has almost no extracted characters. Keep a rendered image even
        for text-rich pages: extraction can lose columns, reading order, equations, or the
        relationship between a chart and its legend.
        '''),
        code(r'''
        pages = read_pdf(CORPUS / "solar_lab_report.pdf")
        display([{"id": p.id, "page": p.meta["page"], "text_characters": len(p.text), "parent": p.parent_id} for p in pages])
        print("Page 1 extracted text:\n", pages[0].text)
        display(Image(filename=pages[1].path, width=450))
        '''),
        md('''
        The raster chart page has no native text layer, although you can read it visually.
        A text-only PDF extractor did not fail to download the report; it reached a boundary
        of what the file stores as text. Options include OCR, vision parsing, or a visual
        page-retrieval model. More advanced document retrieval can represent many patches
        per page, but this tutorial focuses on one representation per evidence unit.
        '''),
        md('''
        ## Step 2 · Index PDF evidence with page citations

        For an offline Strategy A demonstration, page 1 uses extracted text and page 2 uses
        the **already labelled hand-authored energy-chart fixture**. This is not automatic
        OCR of the PDF. The fixture is explicitly attached because our corpus builder put
        that known chart on page 2. A new unknown PDF would need a real extraction path.

        Every page retains a parent snapshot ID, its page number, and a rendered image path.
        The snapshot ID includes the filename stem and a short source hash. Two different
        files named `report.pdf` can coexist; a corrected file gets a new version identity.
        A larger catalog would separately track a logical document ID across versions.
        A citation to an entire 200-page document is less helpful than a citation to the page
        that contains the evidence.

        Rendered pages live in a folder named from the PDF's name and content hash, so
        rendering another report—or a corrected version—does not overwrite old evidence.
        '''),
        code(r'''
        from dataclasses import replace
        page_two_caption = flatten_parse(load_reference_parses()["chart_energy"])
        pdf_records = [replace(pages[0], text=pages[0].text),
                       replace(pages[1], text=page_two_caption,
                               meta={**pages[1].meta, "parse_source": "hand-authored known-chart fixture"})]
        pdf_encoder = LexicalEncoder([p.text for p in pdf_records])
        pdf_index = QdrantIndex(pdf_records, pdf_encoder.embed([p.text for p in pdf_records]), "pdf_pages")
        pdf_hits = pdf_index.search(pdf_encoder.embed(["May energy kWh"])[0], k=1)
        display(show_hits(pdf_hits))
        print("Cite:", pdf_hits[0]["source_id"], "from", pdf_hits[0]["parent_id"])
        '''),
        md('''
        An image query path could index the rendered page directly instead. Different pages
        can need different representations, but record their provenance. Do not call a
        manually supplied caption “extracted PDF text.” When reading an answer, attach the
        rendered retrieved page so the model can inspect the chart itself.
        '''),
        code(r'''
        pdf_bundle = evidence_bundle(pdf_hits, max_images=1)
        print("Retrieved-page image attached:", pdf_bundle[0]["images"])
        if RUN_LIVE:
            from live import answer_from_bundle
            print(answer_from_bundle("Which month produced the most energy, and how much? Cite the page.", pdf_bundle))
        else:
            print("PDF reader skipped. The retrieved page above is available for human inspection.")
        '''),
        figure("10_video", "Timestamped transcript windows align with sampled visual frames without implying exact event onset."),
        md('''
        ## Step 3 · Treat video as two aligned streams

        One stream is what was said: transcript text with start/end times. The other is what
        appeared: sampled frames with observation times. They can agree, complement each
        other, or conflict. A speaker might discuss the next slide while the previous slide
        is still visible.

        Our three transcript windows are `[0,4)`, `[4,8)`, and `[8,12)` seconds. The square
        bracket includes the start; the parenthesis excludes the end. Therefore a frame at
        exactly 4 seconds belongs to the second window, not both.
        '''),
        code(r'''
        companion = json.loads((CORPUS / "transcript.json").read_text())
        segments = companion["segments"]
        print("Transcript provenance:", companion["provenance"])
        display(segments)
        '''),
        md('''
        ## Step 4 · Decode and sample actual frames

        OpenCV reads the AVI file. We retain one frame every two seconds and compute the
        frame's timestamp from its decoded index and the video's frame rate. This works for
        our constant-frame-rate teaching video. Variable-frame-rate media needs timestamps
        from the media stream rather than assuming `frame_index / fps` is universally exact.

        Six frames are enough to inspect three static scenes. Sampling every two seconds
        could miss a warning visible for only half a second. Scene-change detection or denser
        sampling may help, but each adds cost and still requires evaluation.
        '''),
        code(r'''
        frames = sample_keyframes(CORPUS / "solar_lab_review.avi", every_seconds=2)
        display([{ "id": f["id"], "seconds": f["time"] } for f in frames])
        for frame in frames[::2]:
            print("Observed at", timestamp(frame["time"]))
            display(Image(filename=frame["path"], width=440))
        '''),
        md('''
        ### Open sampling: frame index → time → selection

        The full helper saves PNG files and validates failures. Its core loop is below, run
        directly against the actual video. `capture.read()` decodes the next frame; dividing
        its index by FPS gives time for this constant-rate fixture. We retain a sample when
        its time reaches the next two-second target. `finally` releases the decoder even
        if an error occurs.
        '''),
        code(r'''
        import cv2
        capture = cv2.VideoCapture(str(CORPUS / "solar_lab_review.avi"))
        sampled_times = []
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not capture.isOpened() or fps <= 0:
                raise ValueError("The teaching video could not establish a valid frame rate.")
            frame_number, next_target = 0, 0.0
            while True:
                ok, decoded_pixels = capture.read()
                if not ok:
                    break
                seconds = frame_number / fps
                if seconds + 1e-8 >= next_target:
                    sampled_times.append(seconds)
                    next_target += 2.0
                frame_number += 1
        finally:
            capture.release()
        print("Manually sampled times:", sampled_times)
        assert sampled_times == [f["time"] for f in frames]
        # Alignment is just the half-open interval rule, applied to observations.
        print({s["id"]: [t for t in sampled_times if s["start"] <= t < s["end"]]
               for s in segments})
        '''),
        md('''
        If decoding fails, the helper raises an error. It does not silently substitute
        unrelated stills and claim that video sampling worked. If a transcript window has
        no observed frame, we keep that gap explicit. A “nearest” frame from a different
        window can be convenient context, but must not be presented as an observation
        inside the requested interval.
        '''),
        code(r'''
        transcript_records, frame_records = build_video_records(segments, frames)
        display([{"chunk_id": r.id, "segment_source": r.source_id,
                  "whole_video_parent": r.parent_id, "sample_time": r.meta["sample_time"]}
                 for r in frame_records])
        print("Unique frame IDs:", len({r.id for r in frame_records}) == len(frame_records))
        no_frame_text, no_frame_visual = build_video_records(segments, [])
        print("Empty-frame case:", len(no_frame_text), "text windows;", len(no_frame_visual), "visual records")
        '''),
        md('''
        ## Step 5 · Index streams separately, fuse at the segment level

        Transcript chunks use word-count vectors. Frame chunks use measured color vectors.
        Each frame has a unique ID for storage, but frames in one transcript window share
        a **segment source ID**. The complete video is their parent. RRF gives at most one
        vote per segment in each list, so two nearly identical frames do not double its vote.

        The query below includes both “teal” (a visual clue) and “controller warning” (words
        from the transcript). This deliberately fits our teaching baselines. A general natural
        language image query would need the optional CLIP encoder from notebook 3.
        '''),
        code(r'''
        transcript_encoder = LexicalEncoder([r.text for r in transcript_records])
        transcript_index = QdrantIndex(transcript_records,
                                      transcript_encoder.embed([r.text for r in transcript_records]), "video_text")
        frame_encoder = ColorEncoder()
        frame_index = QdrantIndex(frame_records, frame_encoder.embed_images([r.path for r in frame_records]), "video_frames")

        def retrieve_video(query, k=2):
            text_hits = transcript_index.search(transcript_encoder.embed([query])[0], k)
            # Request all six frames before source-level dedupe for this tiny corpus.
            visual_hits = frame_index.search(frame_encoder.embed_text([query])[0], len(frame_records))
            return rrf([text_hits, visual_hits])[:k]

        video_question = "When does the teal controller warning screen appear?"
        video_hits = retrieve_video(video_question, k=2)
        display(show_hits(video_hits))
        for hit in video_hits:
            print(hit["source_id"], hit["votes"], "distinct evidence chunks:", len(hit["evidence"]))
        '''),
        md('''
        The selected controller segment should be supported by both streams. The additional
        candidate is not automatically useful. For a production system, choose a candidate
        budget, deduplicate near-identical frames, and evaluate whether visual search improves
        over the transcript-only baseline on genuinely visual questions.

        Here transcript-only retrieval can find a warning discussion, but it cannot read the
        exact code because the companion transcript omits it. Visual evidence is needed for
        **“Which error code is on screen?”** A color feature can help find the screen; it
        cannot read that code itself.
        '''),
        md('''
        ## Step 6 · Attach only evidence from selected segments

        Our reader bundle joins transcript text and one sampled frame for each **retrieved**
        segment. It prefers the actual frame returned by visual retrieval. When only text
        retrieves a segment, it may attach the earliest available frame from that window,
        labelled `representative_for_retrieved_transcript`. That label matters: a representative
        frame is context, not a frame chosen for matching the visual question.
        The bundle does not scan every frame to find the desired answer after retrieval.
        The bundle carries the segment interval and each frame's actual sample time.

        Inspect those fields before calling a model. A sample observed at 4 seconds supports
        “visible in the frame at 00:04.0.” In general, it does not prove the warning first
        appeared at exactly 4 seconds. Exact onset needs finer temporal evidence.
        '''),
        code(r'''
        video_bundle = evidence_bundle(video_hits, transcript_records, frame_records, max_images=2)
        display([{ "source_id": e["source_id"], "window": (e["meta"]["start"], e["meta"]["end"]),
                   "sample_times": [i["sample_time"] for i in e["images"]],
                   "frame_selection": [i["selection"] for i in e["images"]], "transcript": e["text"]}
                  for e in video_bundle])
        selected = {hit["source_id"] for hit in video_hits}
        assert all(item["source_id"] in selected for item in video_bundle)
        if RUN_LIVE:
            from live import answer_from_bundle
            print(answer_from_bundle("What error code is visible during the controller warning discussion? Cite the observed frame time and explain what the transcript says happened.", video_bundle))
        else:
            print("Video reader skipped. No generated answer or model-read timestamp is claimed.")
        '''),
        md('''
        ## Step 7 · Check time alignment, not only retrieved IDs

        A system can retrieve the correct video but the wrong moment. Our tiny check asks
        whether the top retrieved segment overlaps the labelled discussion window `[4,8)`.
        **Intersection over union (IoU)** divides the overlapping duration by the combined
        duration. Exact matching windows score 1; disjoint windows score 0.

        This evaluates segment localization, not whether the model correctly read `E17`.
        Answer correctness, code transcription, and citation support need their own checks.
        '''),
        code(r'''
        def temporal_iou(predicted, expected):
            intersection = max(0., min(predicted[1], expected[1]) - max(predicted[0], expected[0]))
            union = max(predicted[1], expected[1]) - min(predicted[0], expected[0])
            return intersection / union if union > 0 else 0.

        top = video_hits[0]
        predicted_window = (top["meta"]["start"], top["meta"]["end"])
        expected_window = (4., 8.)  # Evaluation label, never sent to the reader prompt.
        print("Top predicted window:", predicted_window)
        print("Localization IoU:", temporal_iou(predicted_window, expected_window))
        print("Shifted by two seconds:", temporal_iou((6., 10.), expected_window))
        '''),
        md('''
        ## Step 8 · Estimate the frame budget

        Thirty minutes at thirty frames per second contains 54,000 frames. Sampling once
        every two seconds leaves roughly 900. Scene-based deduplication may reduce that
        further, but could remove a small changing detail. Decide which losses matter for
        your questions before optimizing the count.

        Audio/transcript search often supplies most of the value for spoken explanations.
        Video frames become especially useful when something important is shown but never
        spoken. Compare transcript-only and combined retrieval on both kinds of questions.
        '''),
        code(r'''
        duration_seconds, fps, interval = 30*60, 30, 2
        print({"all_frames": duration_seconds*fps,
               "uniform_samples": int(np.ceil(duration_seconds/interval))})
        '''),
        md('''
        ### Capstone · Write a one-page design decision

        Pick a small collection you can inspect safely: science charts, public museum images,
        or your own project slides. Write down (1) the questions people ask, (2) where the
        evidence actually lives, (3) your retrieval strategy, (4) what a reader must verify,
        and (5) how you will measure failure. Include one unsupported query and one detail
        that appears only in pixels.

        <details><summary>Example decision structure</summary>

        “For class energy reports, start with extracted PDF text plus checked chart captions.
        Keep page images for numeric reading. Use separate visual retrieval only if held-out
        visual questions improve enough to justify its cost. Cite page IDs and video windows.
        Evaluate multi-source recall, hit rate, reciprocal rank, label–value–unit accuracy,
        unsupported-answer behavior, and time localization separately. Record source/model
        versions so updates invalidate stale features.”
        </details>

        You have now built the whole path: source media → representations → Qdrant search →
        rank fusion → selected evidence → optional reading → measurement. Revisit
        [THEORY.md](THEORY.md) and [FEATURE_STORES.md](FEATURE_STORES.md) to connect the code
        with the design principles. Source connections: reference cells 41–48 (video);
        PDF extraction/rendering and temporal-IoU experiments are additional material.
        '''),
    ]


if __name__ == "__main__":
    write("01_foundations.ipynb", foundations())
    write("02_describe_and_search.ipynb", strategy_a())
    write("03_shared_space.ipynb", shared_space())
    write("04_fusion_and_evaluation.ipynb", fusion())
    write("05_pdf_and_video.ipynb", pdf_video())
