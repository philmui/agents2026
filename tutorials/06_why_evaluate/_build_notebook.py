"""Builds 06_why_evaluate.ipynb from a list of (type, source) cells.

Run:  python3 _build_notebook.py
This keeps the notebook JSON well-formed and easy to regenerate. The script
itself is not part of the tutorial; it is a build tool. Edit THIS file, never
the generated .ipynb.

Pattern (identical across every module in tutorials/):
  - md(r'''...''')   adds a markdown cell   (use raw strings for LaTeX/backslashes)
  - code(r'''...''') adds a code cell
  - the EMIT block at the bottom writes the .ipynb. Change only OUT below.
"""
import json

# Each entry is ("md", "markdown text") or ("code", "python source").
CELLS = []
def md(text):   CELLS.append(("md",   text.strip("\n")))
def code(text): CELLS.append(("code", text.strip("\n")))

# ============================================================================
# TITLE + SUMMARY
# ============================================================================
md(r"""
# Module 06 · Why Evaluate? + RAGAS Setup

### A hands-on, build-it-yourself module for advanced high school researchers

![MDD loop](slides/assets/02_mdd_loop.svg)

Module 05 gave you a working RAG pipeline — an Ollama embedder, a Qdrant
vector store, and a prompt-stuffing generator that answers questions about
precious metals. That pipeline *works*, but "it works" is not a measurement.
This module asks the harder question: *how do you know it is working well?*
You will install the **evaluation mindset**, learn the
**Metrics-Driven Development (MDD)** loop, and wire up the RAGAS data
structures (`SingleTurnSample`, `EvaluationDataset`) the rest of the track
will use to score every component. Module 07 computes the first real
metric scores on the dataset you build here.

This is Module 06 of a twelve-part track that ends in a full
**Agentic RAG Evaluation** capstone.
""")

md(r"""
## 📋 Summary: the one-paragraph version

You cannot improve a system you cannot measure. This module gives you the
tools to measure: a fixed **golden test set** (8 questions with reference
answers), a **RAGAS evaluation harness** (the `SingleTurnSample` /
`EvaluationDataset` data structures plus the `judge_llm` object that will
score them), and the **MDD loop** discipline of running the same metrics
before and after every pipeline change. By the end of this notebook you will
have an `EvaluationDataset` ready to hand to any RAGAS metric.
""")

md(r"""
## 🗺️ What you will build, step by step

| Step | What you do | Key tool |
| ---: | --- | --- |
| 0.1 | Install libraries | `uv sync` |
| 0.2 | Load shared API keys; set `HAVE_KEYS` guard | `python-dotenv` |
| 1 | RAGAS import stub + `nest_asyncio` | `sys.modules`, `nest_asyncio` |
| 2 | Re-build the M5 RAG (embedder → vector store → retriever → generator) | `langchain-ollama`, `langchain-qdrant` |
| 3 | Create RAGAS model objects (`generator_llm`, `judge_llm`, `ragas_embeddings`) | `ragas`, `litellm` |
| 4 | Load golden questions; run `rag_answer` on each; build `EvaluationDataset` | `ragas.dataset_schema` |
| 5 | Inspect the dataset — verify shape, peek at a sample row | `pandas` |

### 🎓 What you will *learn* (the concepts)

- **The evaluation mindset** — why eyeballing one answer is not measurement.
- **The MDD loop** — Build → Measure → Diagnose → Improve, one variable at a time.
- **Retriever vs. generator** — wrong answers come from two places; the fix is different each time.
- **RAGAS data structures** — `SingleTurnSample`, `EvaluationDataset`, and the generator ≠ judge split.

### ✅ Prerequisites

- Module 05 completed: you understand cloud-Ollama (LiteLLM), Qdrant in-memory, and the RAG answer function.
- `OLLAMA_API_KEY` set in `tutorials/.env` — or follow along with the frozen fallback.
- Curiosity about *why* a number might be lying to you.
""")

# ============================================================================
# STEP 0.1 — INSTALL
# ============================================================================
md(r"""
---
# Step 0 · Setup

## 0.1 Install the libraries

The exact dependency list lives in **`pyproject.toml`** next to this notebook,
so the environment is **reproducible**. Install everything with one command,
**from this module's folder** (`tutorials/06_why_evaluate/`), using
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync            # reads pyproject.toml, creates .venv/, installs everything
uv run jupyter lab # launch Jupyter inside that environment
```

When the notebook opens, pick the kernel **`Python 3 (ipykernel)`** (top-right
kernel picker). That is the interpreter from `.venv`, so every `import` below
resolves against what `uv sync` installed.
""")

# ============================================================================
# STEP 0.2 — KEY LOADING (shared .env)
# ============================================================================
md(r"""
## 0.2 Provide your API key (shared `.env`)

All twelve modules read their keys from a **single** `.env` file in the
`tutorials/` folder — the parent of this module. Create it once if you have
not already:

```
# tutorials/.env
OLLAMA_API_KEY=your_key_here
```

`find_dotenv()` walks UP the directory tree from this notebook and locates
that shared file automatically. You never copy keys into individual modules.
`.env` is gitignored — never commit it.

**No key?** The notebook falls back to `frozen/sample_dataset.json`, a
two-row illustrative dataset that lets you follow every downstream cell
without spending any credits.

> **Cost note (M6+):** Building the live `EvaluationDataset` makes
> **8 generator LLM calls** (one per golden question). RAGAS model objects
> are created without any API call — costs only start when you call
> `rag_answer()` in Step 4.
""")

code(r'''
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())          # resolves to tutorials/.env automatically

HAVE_KEYS = bool(os.environ.get("OLLAMA_API_KEY"))
if not HAVE_KEYS:
    print(
        "⚠  No OLLAMA_API_KEY found in tutorials/.env.\n"
        "   This notebook will use the cached illustrative results in\n"
        "   frozen/sample_dataset.json so you can still follow along.\n"
        "   Set OLLAMA_API_KEY to run the live pipeline."
    )
else:
    print("✓ OLLAMA_API_KEY found — live pipeline will run.")
    # The cloud Ollama base URL defaults to localhost relay for :cloud models.
    os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11434")
''')

# ============================================================================
# STEP 1 — RAGAS IMPORT STUB + nest_asyncio
# ============================================================================
md(r"""
---
# Step 1 · RAGAS import stub + `nest_asyncio`

RAGAS 0.4.3 has a hard import for a `langchain_community` module that
`langchain-community` 1.x removed. The litellm path we use never calls that
code, but Python tries to import it at module load time and crashes. The fix
is to **stub** the missing module *before* importing ragas.

We also need `nest_asyncio` because Jupyter already runs an event loop and
RAGAS uses `asyncio` internally. Without the patch, nested `await` calls raise
a `RuntimeError`.

Run this cell before any `import ragas` anywhere in this notebook.
""")

code(r'''
import sys, types

# ── RAGAS import stub ─────────────────────────────────────────────────────
# RAGAS 0.4.3 hard-imports langchain_community.chat_models.vertexai, which
# was removed in langchain-community 1.x. Stub it so the litellm path works.
_vx = types.ModuleType("langchain_community.chat_models.vertexai")

class ChatVertexAI:  # placeholder — intentionally non-functional
    pass

_vx.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vx

# ── nest_asyncio ──────────────────────────────────────────────────────────
# Jupyter's event loop is already running; RAGAS needs to await inside it.
import nest_asyncio
nest_asyncio.apply()

# ── Safe ragas imports ────────────────────────────────────────────────────
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

print("✓ RAGAS imported successfully")
print(f"  SingleTurnSample fields: {list(SingleTurnSample.model_fields.keys())}")
''')

# ============================================================================
# STEP 2 — RE-BUILD THE M5 RAG
# ============================================================================
md(r"""
---
# Step 2 · Re-build the M5 RAG

We need a working RAG so we can produce `retrieved_contexts` and `response`
for each golden question. This is the same stack you built in Module 05:

1. **Ollama embeddings** via `langchain-ollama`
2. **Qdrant in-memory** vector store via `langchain-qdrant`
3. **Corpus loader + chunker** from `corpus/`
4. **Chat LLM + prompt** for the generator

If `HAVE_KEYS` is `False`, skip to Step 3 — we will load the frozen dataset
instead of running `rag_answer`.
""")

code(r'''
if HAVE_KEYS:
    from pathlib import Path
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_qdrant import QdrantVectorStore
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from langchain_core.prompts import ChatPromptTemplate

    # ── Model names ───────────────────────────────────────────────────────
    LLM_NAME_OLLAMA       = "nemotron-3-super:cloud"
    EMBEDDING_NAME_OLLAMA = "qwen3-embedding:0.6b"
    OLLAMA_BASE           = os.environ["OLLAMA_API_BASE"]

    # ── LangChain models ──────────────────────────────────────────────────
    chat_llm      = ChatOllama(model=LLM_NAME_OLLAMA,       base_url=OLLAMA_BASE, temperature=0.0)
    lc_embeddings = OllamaEmbeddings(model=EMBEDDING_NAME_OLLAMA, base_url=OLLAMA_BASE)

    # ── Load corpus + chunk ───────────────────────────────────────────────
    raw_docs = [
        {"source": p.name, "page_content": p.read_text()}
        for p in sorted(Path("corpus").glob("*.md"))
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    lc_docs = [
        Document(page_content=piece, metadata={"source": d["source"]})
        for d in raw_docs
        for piece in splitter.split_text(d["page_content"])
    ]
    print(f"Corpus: {len(raw_docs)} files → {len(lc_docs)} chunks")

    # ── Vector store ──────────────────────────────────────────────────────
    vector_store   = QdrantVectorStore.from_documents(
        lc_docs, embedding=lc_embeddings,
        location=":memory:", collection_name="metals_kb"
    )
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # ── RAG prompt + answer function ──────────────────────────────────────
    RAG_PROMPT = ChatPromptTemplate.from_template(
        "You are a precise metals-markets tutor. Answer using ONLY the context passages. "
        "If the context does not contain the answer, say you do not know.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    def rag_answer(question: str, k: int = 10) -> dict:
        """Retrieve top-k passages and generate an answer. Returns a dict with
        'response' (str) and 'retrieved_contexts' (list[str])."""
        candidates = [d.page_content for d in base_retriever.invoke(question)]
        contexts   = candidates[:3]          # top-3 without reranking (added in M10)
        block      = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
        response   = chat_llm.invoke(
            RAG_PROMPT.format_messages(context=block, question=question)
        ).content.strip()
        return {"response": response, "retrieved_contexts": contexts}

    print("✓ RAG pipeline ready")
else:
    print("(skipping RAG build — no API key)")
''')

# ============================================================================
# STEP 3 — RAGAS MODEL OBJECTS
# ============================================================================
md(r"""
---
# Step 3 · RAGAS model objects

RAGAS wraps LLMs and embedding models in thin adapters so its metric code
stays model-agnostic. We create **three** objects:

| Object | Model | Purpose |
| --- | --- | --- |
| `generator_llm` | `nemotron-3-super:cloud` | Wrote the answers we are grading |
| `judge_llm` | `gemma4:31b-cloud` | Grades those answers (different model!) |
| `ragas_embeddings` | `qwen3-embedding:0.6b` | Semantic similarity for relevancy metrics |

> **⚠ Generator ≠ judge.** Using the same model to write *and* grade answers
> creates a sycophancy loop — the model gives high scores to outputs it would
> produce itself. Always keep these two objects pointing at different models.

Creating these objects makes **no API calls** — costs start only when metrics
are evaluated in Module 07.
""")

code(r'''
if HAVE_KEYS:
    import litellm
    from ragas.llms import llm_factory
    from ragas.embeddings.base import embedding_factory

    LLM_MODEL       = "ollama_chat/nemotron-3-super:cloud"
    JUDGE_MODEL     = "ollama_chat/gemma4:31b-cloud"
    EMBEDDING_MODEL = "ollama/qwen3-embedding:0.6b"

    generator_llm    = llm_factory(
        LLM_MODEL,   provider="litellm",
        client=litellm.completion, temperature=0.3
    )
    judge_llm        = llm_factory(
        JUDGE_MODEL, provider="litellm",
        client=litellm.completion, temperature=0.0
    )
    ragas_embeddings = embedding_factory(
        "litellm", model=EMBEDDING_MODEL,
        api_base=os.environ["OLLAMA_API_BASE"]
    )
    print("✓ generator_llm :", LLM_MODEL)
    print("✓ judge_llm     :", JUDGE_MODEL)
    print("✓ ragas_embeddings:", EMBEDDING_MODEL)
else:
    generator_llm = judge_llm = ragas_embeddings = None
    print("(RAGAS model objects set to None — frozen data path active)")
''')

# ============================================================================
# STEP 4 — BUILD EvaluationDataset
# ============================================================================
md(r"""
---
# Step 4 · Build the `EvaluationDataset`

Each `SingleTurnSample` captures one question's full context:
- `user_input` — the question from `golden_questions.json`
- `retrieved_contexts` — the list of passages `rag_answer` returned
- `response` — the generator's answer
- `reference` — the ground-truth answer from the golden set

An `EvaluationDataset` is simply a list of these samples. Module 07 will call
`ragas.evaluate(dataset=eval_dataset, metrics=[...], llm=judge_llm, ...)` on
this object to score the retriever.

If `HAVE_KEYS` is `False`, we load the illustrative frozen dataset instead.
The frozen file is clearly labelled as illustrative so students can follow
downstream cells without any API credits.

> *Illustrative output (frozen path):* the frozen dataset ships 2 rows to
> keep the file tiny. A live run produces all 8 rows.
""")

code(r'''
import json

if HAVE_KEYS:
    golden  = json.loads(open("golden_questions.json").read())
    samples = []
    for g in golden:
        print(f"  → {g['question'][:60]}…")
        out = rag_answer(g["question"])
        samples.append(SingleTurnSample(
            user_input         = g["question"],
            retrieved_contexts = out["retrieved_contexts"],
            response           = out["response"],
            reference          = g["reference"],
        ))
    eval_dataset = EvaluationDataset(samples=samples)
    print(f"\n✓ EvaluationDataset: {len(eval_dataset.samples)} samples (live run)")
else:
    raw = json.load(open("frozen/sample_dataset.json"))
    eval_dataset = EvaluationDataset(samples=[
        SingleTurnSample(**s) for s in raw["samples"]
    ])
    print("(using cached illustrative result — set OLLAMA_API_KEY in "
          "tutorials/.env to run live)")
    print(f"✓ EvaluationDataset: {len(eval_dataset.samples)} samples (frozen)")
''')

# ============================================================================
# STEP 5 — INSPECT THE DATASET
# ============================================================================
md(r"""
---
# Step 5 · Inspect the dataset

Before we hand this dataset to any metric in Module 07, let's verify its
shape and read one sample in full.
""")

code(r'''
import pandas as pd

rows = [
    {
        "question":    s.user_input[:60] + "…",
        "n_contexts":  len(s.retrieved_contexts),
        "response_len": len(s.response),
        "has_reference": bool(s.reference),
    }
    for s in eval_dataset.samples
]
df = pd.DataFrame(rows)
print(df.to_string(index=False))
''')

code(r'''
# Print the first sample in full so we can read what RAGAS will score.
s0 = eval_dataset.samples[0]
print("── user_input ──────────────────────────────────────────")
print(s0.user_input)
print("\n── retrieved_contexts (first passage) ──────────────────")
print(s0.retrieved_contexts[0] if s0.retrieved_contexts else "(none)")
print("\n── response ────────────────────────────────────────────")
print(s0.response)
print("\n── reference ───────────────────────────────────────────")
print(s0.reference)
''')

# ============================================================================
# RECAP
# ============================================================================
md(r"""
---
# Recap & what's next

You learned:

- **Evaluation mindset** — why a fixed golden test set + consistent scoring
  procedure is non-negotiable.
- **MDD loop** — Build → Measure → Diagnose → Improve, always one variable at
  a time. Absolute scores matter less than the direction and size of changes.
- **Retriever vs. generator split** — wrong answers come from two distinct
  places and require different fixes.
- **RAGAS harness** — the import stub + `nest_asyncio` trick, the three model
  objects (`generator_llm`, `judge_llm`, `ragas_embeddings`), and the
  `SingleTurnSample` / `EvaluationDataset` structures.

**Next module (07 — Retriever Metrics):** takes the `eval_dataset` you just
built and runs the first real metrics —
`LLMContextPrecisionWithReference` and `LLMContextRecall` — on the
retriever half of the pipeline.
""")

# ============================================================================
# EMIT NOTEBOOK  (do not change below except OUT)
# ============================================================================
def to_cell(kind: str, src: str) -> dict:
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

nb = {
    "cells": [to_cell(k, s) for k, s in CELLS],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = "06_why_evaluate.ipynb"   # <-- this module's notebook filename
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

n_md   = sum(1 for k, _ in CELLS if k == "md")
n_code = sum(1 for k, _ in CELLS if k == "code")
print(f"Wrote {OUT}: {len(CELLS)} cells ({n_md} markdown, {n_code} code)")
