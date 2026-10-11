# Evaluating RAG with HotpotQA and LangGraph

Learn how to tell whether a RAG system found the right evidence, answered correctly, and cited the right sentences. The tutorial starts with a complete invented JSON example, moves to actual HotpotQA records, and builds a bounded LangGraph agent with retrieval, evidence grading, query rewriting and generation.

## Start the notebook

Prerequisites: basic Python, Python 3.11–3.13, and [uv](https://docs.astral.sh/uv/getting-started/installation/). From the repository root:

```bash
cd evals/hotpotqa
uv sync --locked
uv run jupyter lab hotpotqa_rag_evaluation.ipynb
```

Use the project's Python kernel and run cells top to bottom with Shift+Enter. The notebook contains explanations, code, vector diagrams and stored outputs. Allow 60–90 minutes. After installation, the default path needs neither an API key nor network access.

Open [the notebook](hotpotqa_rag_evaluation.ipynb), [the illustrated HTML slides](slides/index.html), or [the 34-page slide PDF](slides/hotpotqa_rag_evaluation.pdf). The slides run locally in a browser and include a reading view and print layout. Keep the module's directory structure intact so relative links and graphics work.

## What the experiments establish

The four-paragraph Lantern Atlas story is invented and explicitly labeled synthetic. A scripted backend walks through a real LangGraph graph using predetermined decisions and an answer. This demonstrates wiring and scoring, **not model accuracy**.

Ten real development records support a local TF-IDF retrieval comparison. They contain five bridge questions and five comparison questions selected for teaching. All five comparison answers in this small slice are yes/no, and the slice is not representative. Each real question searches only its ten supplied distractor paragraphs. This is a distractor teaching exercise, not a fullwiki benchmark.

The optional live path replaces the scripted backend with a structured-output language model. It evaluates short answers, exact sentence citations and joint scores, and saves predictions in official HotpotQA format. Model failures remain in the evaluation denominator.

## Lesson map

| Notebook sections | You will learn |
|---|---|
| 1–5 | Setup, RAG, HotpotQA settings, raw JSON and sentence addresses |
| 6–9 | Evaluation boundaries, TF-IDF retrieval, LangGraph nodes and bounded routing |
| 10–13 | Answer EM/F1, supporting-fact scores, joint scores and prediction JSON |
| 14 | Actual-record retrieval comparisons and missed-evidence inspection |
| 15–17 | Optional hosted-model run, failure analysis and official evaluator export |
| 18–20 | Fair comparisons, limitations, exercises and references |

The main files are:

| File | Purpose |
|---|---|
| [hotpotqa_rag_evaluation.ipynb](hotpotqa_rag_evaluation.ipynb) | Executed, step-by-step tutorial |
| [slides/index.html](slides/index.html) | Illustrated companion presentation |
| [rag.py](rag.py) | Sentence-preserving retrieval and real LangGraph agent |
| [metrics.py](metrics.py) | Official-compatible scoring functions |
| [evaluation.py](evaluation.py) | Batch harness and reproducible export |
| [data/README.md](data/README.md) | Raw sample files, attribution, checksums and selection |
| [SOURCES.md](SOURCES.md) | Primary references and provenance |
| [VALIDATION.md](VALIDATION.md) | Checks performed and remaining limits |
| [tools/build_notebook.py](tools/build_notebook.py) | Editable notebook authoring source |

## Optional live model

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and `OPENAI_MODEL`. Choose an accessible model with structured-output support. The notebook loads only this module's `.env`; existing environment variables take precedence. Restart the kernel after changing configuration.

Set the notebook's `RUN_LIVE` variable to `True`, then run the cells in order. The default selects two real questions. With two retrieval rounds, each question uses at most four model calls. Calls have a 45-second timeout, a 1,200-token output cap and client retries disabled. The model's token prices and input lengths determine cost. Errors appear in result rows instead of being replaced with fixture answers. No hosted calls are needed to read or execute the offline tutorial.

Each live run writes a new ignored directory under `artifacts/`, containing predictions, matching gold records, traces, metrics and a manifest. A later cell invokes the pinned official evaluator on that exact subset. Keep run directories if you need to reproduce or audit an experiment. Review traces before sharing them.

## Verify or modify the tutorial

Run from this module:

```bash
uv run pytest -q
uv run python tools/validate.py
```

After editing the authoring source, regenerate and execute the notebook:

```bash
uv run python tools/build_diagrams.py
uv run python tools/build_notebook.py
uv run jupyter nbconvert --to notebook --execute hotpotqa_rag_evaluation.ipynb --inplace
uv run python tools/validate.py
```

Regeneration replaces notebook cells and outputs. Save personal experiments separately. Keep `RUN_LIVE = False` for offline verification. Helper code is imported into a kernel once, so restart the kernel after editing `rag.py`, `metrics.py` or `evaluation.py`.

The official CMU data endpoint failed during preparation. The bundled records came from a pinned mirror in the HotpotQA Hugging Face organization and were converted losslessly into the original pair-based JSON layout; serialization differs from the canonical JSON download. The preparation script and exact hashes document this distinction. Data attribution and licensing travel with the sample. The tutorial preserves original titles, sentence text and indices.
