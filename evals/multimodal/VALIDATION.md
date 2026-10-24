# What was checked

Validation completed on **2026-10-22**, using Python **3.12.13** on macOS arm64 and the exact package resolutions in [uv.lock](uv.lock). The review and fixes are documented in [REVIEW_LOG.md](REVIEW_LOG.md).

## Repository integration verification — 2026-10-24

The course was integrated into `agents2026/evals/multimodal`. Missing runtime
modules and companion guides were restored from the matching local course
source; notebook setup and authoring instructions now use this repository's
path. The presentation shell and PowerPoint were placed at the locations
expected by their links and build scripts.

Fresh checks in this checkout passed:

- `uv run --all-extras python -m pytest -q`: **47 passed**.
- `uv run --all-extras python tools/execute_notebooks.py`: all five notebooks,
  **55 code cells**, completed in fresh kernels with both optional model
  switches disabled; committed outputs were refreshed.
- `uv run --all-extras python tools/validate_materials.py`: **159 local links**,
  five notebooks, 14 SVGs, and 42 static slides; no reported errors.
- `tools/validate_visuals.py`: **56 rendered artifacts**, no geometry issues.
- `tools/render_reading.py`: nine documents, **36 images**, no broken images
  or page-wide overflow.

The browser checks used installed Google Chrome **156.0.8078.48** through
`MM_CHROMIUM_EXECUTABLE`. The Playwright-managed browser was unavailable.
The copied virtual environment's `pytest` launcher had an obsolete interpreter
path, so tests were invoked through `python -m pytest`. A fresh `uv sync`
environment is recommended for new checkouts. No hosted model calls or
pretrained-model downloads were performed. The earlier review below is retained
as historical evidence; its manual review was not repeated during integration.

## Notebook execution

All five notebooks ran from beginning to end in **separate fresh kernels**, with no saved exception outputs. All **55 code cells** executed; the complete sequence contains **147 cells**. The saved outputs were produced by this run, not invented or copied from the reference.

| Notebook | Total cells | Executed code cells |
|---|---:|---:|
| 01 · Foundations | 30 | 9 |
| 02 · Describe and search | 30 | 12 |
| 03 · Shared space | 30 | 11 |
| 04 · Fusion and evaluation | 28 | 11 |
| 05 · PDFs and video | 29 | 12 |

The execution script forces `MM_RUN_MODELS=0` and `MM_RUN_LIVE=0`. Thus these results establish that the local teaching path works: original corpus files, word-count and color representations, real in-process Qdrant queries, rank fusion, metrics, PDF text extraction/rendering, and actual video decoding. Optional model cells report that they were skipped.

Reproduce from this module directory:

```bash
uv sync --all-extras
uv run --all-extras python tools/execute_notebooks.py
```

The machine-readable execution record is written to ignored `artifacts/notebook_execution.json`.

## Regression and integration checks

**47 tests passed.** They cover cosine normalization and zero vectors; invalid vector shapes; Qdrant payloads and modality filters; RRF score independence, deduplication, ties, and segment agreement; distinct recall/hit-rate/MRR calculations; unsupported-query handling; chart label swaps, units, duplicates, and invalid values; feature fingerprints; historical feature availability; PDF evidence isolation; actual frame decoding; selected-frame preservation; and timestamp rollover.

The real installed LlamaIndex package accepted both adapters and returned text/image hits using **controlled local embedding vectors**. FastEmbed's installed registry contains the requested BGE model and the matching 512-dimensional CLIP towers. The A/B/C integration paths were also exercised with controlled vectors. These are API-contract checks, not pretrained-model accuracy measurements.

LiteLLM request construction and response validation were checked with local stubs: image bytes, evidence IDs, time annotations, separate vision/embedding model settings, row ordering, and malformed embedding responses. No hosted API request was made.

```bash
uv run --all-extras pytest -q
```

## Visual and reading review

All **42 standalone slides** and **14 concept SVGs** were rendered. The final geometry check found no text outside the content frame, overlapping SVG text labels, or sampled open paths crossing figure labels. Separate slide checks also covered projected label sizes and code overflow. Contact sheets and selected full-size images were inspected; that inspection caught an arrow crossing a cosine label that the initial text-only geometry check had missed. The label placement was corrected and the line-intersection check was added.

The slide shell passed navigation, Home/End, arrows after clicking the slide, speaker-note controls, a 390-pixel mobile-width check, and a no-JavaScript fallback check. Every slide includes `© mui-group`. See [the detailed slide record](slides/REVIEW.md).

Reading previews of all five executed notebooks and four Markdown documents loaded **36 images** without broken images or page-wide horizontal overflow. The four documents were README, THEORY, FEATURE_STORES, and SOURCE_MAP. These local review previews block external CDN requests; they check content and images, while JupyterLab supplies its normal mathematics renderer for study.

The final lead review used Chromium **153.0.8010.12**, already installed on the machine, through an explicit `MM_CHROMIUM_EXECUTABLE` override. For a new machine, install the browser expected by Playwright once:

```bash
uv run playwright install chromium
uv run --all-extras python tools/validate_visuals.py
uv run --all-extras python tools/render_reading.py
```

Alternatively, set `MM_CHROMIUM_EXECUTABLE` to a compatible browser executable you already have. Screenshots, contact sheets, and geometry/reading reports are kept in ignored `artifacts/`. Delivered slide previews are under `assets/slide_previews/`.

## Reproducible package environment

| Package | Tested version |
|---|---|
| NumPy | 2.5.4 |
| qdrant-client | 1.19.1 |
| FastEmbed | 0.9.0 |
| llama-index-core | 0.14.25 |
| LiteLLM | 1.106.2 |
| Pillow | 12.3.0 |
| opencv-python-headless | 4.14.0.94 |
| pypdfium2 | 5.14.0 |
| nbclient | 0.11.0 |

The lockfile pins package resolutions. Model weight revisions and endpoint deployments are separate artifacts; a Python lockfile does not freeze them.

## What these checks do not establish

No full pretrained CLIP/BGE experiment or hosted vision response was measured in this run. The lesson includes their opt-in implementations, but the delivered numeric outputs are from labeled local teaching baselines. Do not cite them as CLIP quality, vision-model accuracy, or an A/B/C production benchmark.

The supplied video is a silent constant-frame-rate fixture with a hand-authored companion transcript. There is no speech-recognition quality claim. The PDF lesson distinguishes native text from a raster page and uses a labeled caption fixture; it does not claim to have run an OCR engine. The feature-store guide teaches lifecycle and historical-selection principles; it does not implement a production feature-store service.

Schema checks cannot establish factual truth, a citation cannot establish that the cited source supports every claim, and a small synthetic corpus cannot establish performance on real users' documents. Those limits are part of the lessons and capstone design.
