# Multimodal retrieval: search words, pictures, pages, and moments

Imagine your school solar project has a safety memo, a chart of energy production, a screenshot with a warning code, and a short presentation recording. You ask, “What happened, and where is the evidence?” A useful search system must find the right source even when the answer is visible only in a picture.

This course builds that system in small steps. You will inspect the numbers that make search work, compare three retrieval designs, and learn when a result is strong enough to support an answer. The examples use original, fictional school data, so you can inspect every source and deliberately break the system without exposing personal information.

![The complete retrieval pipeline](assets/01_pipeline.svg)

## Start here

You need basic Python: variables, lists, dictionaries, functions, and `for` loops. No neural-network training, calculus, GPU, account, or API key is needed for the default lessons. Budget roughly **five sessions of 45–70 minutes**, plus time for the capstone. The small mathematics examples introduce their notation before using it.

From the repository root:

```bash
cd evals/multimodal
uv sync
uv run jupyter lab
```

Open `01_foundations.ipynb` and select the Python kernel from this module's `.venv`. Run cells from top to bottom. Each notebook starts its own state; you do not need to keep another notebook's kernel running. The committed outputs show a completed default run, but predicting the output before rerunning a cell is part of the lesson.

`uv sync` installs the packages declared in [pyproject.toml](pyproject.toml), using exact resolutions from [uv.lock](uv.lock). Installation requires internet access initially. The default lesson computations then use local files and local Qdrant; they do not download neural-model weights or call a hosted model. See [uv's project guide](https://docs.astral.sh/uv/guides/projects/) for installation and environment details.

## Choose your route

| Lesson | What you will do | The question you should be able to answer |
|---|---|---|
| [01 · Foundations](01_foundations.ipynb) | Inspect the corpus, calculate cosine similarity, preview LlamaIndex | What does a vector search score actually mean? |
| [02 · Describe and search](02_describe_and_search.ipynb) | Validate chart parses; build Strategy A in Qdrant | What information can a caption lose or invent? |
| [03 · Shared space](03_shared_space.ipynb) | Build a tiny visual encoder, then optionally use real CLIP | Why do two vectors need matching coordinates? |
| [04 · Fusion and evaluation](04_fusion_and_evaluation.ipynb) | Build Strategy C, derive RRF, evaluate retrieval and prepare pixel evidence | Did we find the right source, and can it support the answer? |
| [05 · PDFs and video](05_pdf_and_video.ipynb) | Extract and render pages, sample frames, align transcript windows | Can someone follow the citation back to the exact evidence? |

Read [THEORY.md](THEORY.md) alongside the notebooks for longer explanations, worked examples, cautions, and exercises. [FEATURE_STORES.md](FEATURE_STORES.md) explains how reusable features, time, freshness, and versions relate to retrieval. A vector database and a feature store have different jobs; the guide makes that distinction concrete.

Open [slides.html](slides.html) for the presentation. Use arrow keys to move and the notes control for the fuller teaching explanation. The [slide guide](slides/README.md) describes the standalone pages, print behavior, and authoring workflow.

[Download the PowerPoint deck](slides/outputs/pptx/multimodal-retrieval.pptx) with all 42 slides, editable diagrams and charts, and speaker notes.

## Know which experiment you are running

The default text encoder counts words. The default shared visual space measures colors and maps color words to the same coordinates. These choices make every step visible. **They are teaching models, not CLIP, and their scores do not measure neural-model quality.** Hand-authored chart descriptions are labeled fixtures, not recorded vision-model responses.

To install the real local neural-model path:

```bash
uv sync --extra models
MM_RUN_MODELS=1 uv run --extra models jupyter lab
```

This enables optional notebook cells using FastEmbed's BGE text model and the paired CLIP text/vision towers as ONNX models. The first run downloads weights; later inference runs locally. Allow several hundred megabytes of download and additional memory. If downloads fail, the optional cell reports the failure; the default demonstrations remain separate. Do not report skipped experiments as successful ones.

To configure optional LiteLLM vision parsing, reading, and hosted text embeddings:

```bash
uv sync --all-extras
cp -n .env.example .env
# Edit .env locally, then launch Jupyter.
uv run --all-extras jupyter lab
```

Keep keys in `.env`, which Git ignores. The notebook setup uses `load_dotenv(find_dotenv())`; existing shell variables take precedence. Set `MM_RUN_LIVE=1` only when you intend to send the selected content to your configured endpoint. Set `MM_RUN_MODELS=1` independently if you also want local neural-model experiments. Restart the kernel after changing environment configuration.

The example `VLM_MODEL=openai/gpt-5.6-sol` is a provider-qualified configurable name. Your gateway must actually serve that model or alias and support images. The text embedding endpoint uses `EMBED_MODEL`, a separate embedding model. LiteLLM standardizes request formats; it does not give a text-only model vision or make every provider accept identical parameters. See its [vision](https://docs.litellm.ai/docs/completion/vision) and [embedding](https://docs.litellm.ai/docs/embedding/supported_embedding) documentation.

## What is in the folder?

| Item | Purpose |
|---|---|
| `assets/*.svg` | Accessible vector diagrams used in the lessons and guides |
| `assets/corpus/` | Original charts, screenshot, memos, PDF, video, reference parses, and separate answer key |
| [lab.py](lab.py) | Small inspectable retrieval, fusion, evaluation, PDF, and video functions |
| [live.py](live.py) | Explicitly enabled FastEmbed and LiteLLM adapters |
| [framework.py](framework.py) | LlamaIndex adapters and framework preview |
| [SOURCE_MAP.md](SOURCE_MAP.md) | Exact connections to the supplied reference notebook |
| [SOURCES.md](SOURCES.md) | Primary papers and official API documentation |
| [VALIDATION.md](VALIDATION.md) | What ran, what was reviewed, and the limits of that evidence |
| [REVIEW_LOG.md](REVIEW_LOG.md) | Adversarial findings, root causes, fixes, and verification |

The video is a small authored visual teaching fixture with an aligned, hand-authored transcript. It does not demonstrate speech recognition. The PDF includes a page whose chart values need the rendered image. Keep source IDs, page numbers, frame times, and parsing provenance with every derived representation.

## Check your understanding

Before finishing, you should be able to explain all of these in your own words:

1. Why a text embedding model cannot directly search raw image pixels.
2. Why parsing an image during ingestion differs from reading it after retrieval.
3. When captions, a shared image–text space, or separate fused stores are useful.
4. Why identical vector lengths do not establish compatible embeddings.
5. Why Recall@k, Hit@k, and answer accuracy measure different things.
6. Why a timestamp window is a source location, not proof of an exact event time.
7. How a changed image or embedding model makes stored features stale.

For a capstone, add one original image and one memo, then write questions **before** inspecting search results. Include a question the sources cannot answer. Compare strategies using the same final evidence count, record how many candidates each branch searches, inspect every miss, and write a short decision explaining your evidence and uncertainty. Strategy C can examine more candidates because it has two branches; that extra work belongs in your comparison. The final notebook supplies a worked path and extension prompts.

## Reproduce the checks

Run from this folder:

```bash
uv sync --all-extras
uv run --all-extras pytest -q
uv run --all-extras python tools/execute_notebooks.py
uv run --all-extras python tools/validate_materials.py
uv run --all-extras python tools/validate_visuals.py
```

The notebook execution script forces both optional model switches to `0` and runs each lesson in a fresh kernel. The visual checker uses Playwright Chromium; install it once with `uv run playwright install chromium` if needed. Reports and review screenshots go to ignored `artifacts/`.

The build scripts regenerate authored material and may replace saved notebook outputs. Keep personal experiment results separately before rebuilding. Re-run the checks after changes that affect explanations, examples, or code.

© mui-group
