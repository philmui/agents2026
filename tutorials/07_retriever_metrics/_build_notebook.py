"""Builds 07_retriever_metrics.ipynb from a list of (type, source) cells.

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
# Module 07 · Retriever Metrics

### A hands-on, build-it-yourself module for advanced high school researchers

![Context precision and recall diagram](slides/assets/04_context_precision_recall.svg)

In Module 06 you wired up RAGAS and built your first `EvaluationDataset`. Now
it is time to actually **score your retriever**. This module introduces the two
most fundamental retrieval metrics: **LLMContextPrecisionWithReference** and
**LLMContextRecall**. By the end you will have run both metrics over the full
8-question golden set, printed a per-question breakdown, and debugged a single
low-recall sample with `single_turn_ascore()`.

This is Module 07 of a twelve-part track that ends in a full **Agentic RAG
Evaluation** capstone. Module 08 adds two more retriever metrics —
`ContextEntityRecall` and `NoiseSensitivity` — to complete the retrieval picture.
""")

md(r"""
## 📋 Summary: the one-paragraph version

A retriever's quality has two independent dimensions. **Precision** asks whether
the chunks it returned were relevant and well-ranked: are the useful passages
near the top, or buried under noise? **Recall** asks whether the retriever
surfaced *all* the evidence needed to answer the question fully: did you miss any
important passages? These two metrics pull in opposite directions — a system that
retrieves very few chunks can score high on precision but collapse on recall, and
vice versa. Measuring both together gives you a complete picture of where
retrieval is helping or hurting your pipeline.
""")

md(r"""
## 🗺️ What you will build, step by step

| Step | What you do | Key tool |
| ---: | --- | --- |
| 0 | Load shared API keys; apply RAGAS import stub + `nest_asyncio` | `python-dotenv`, `nest_asyncio` |
| 1 | Wire the judge LLM and RAGAS embeddings | `litellm`, `ragas.llms`, `ragas.embeddings` |
| 2 | Rebuild the RAG answer function + build an `EvaluationDataset` | `langchain`, `ragas.dataset_schema` |
| 3 | Run `evaluate()` with both precision and recall metrics | `ragas.evaluate` |
| 4 | Debug a single low-recall sample with `single_turn_ascore()` | `ragas.metrics` |

### 🎓 What you will *learn* (the concepts)

- **LLMContextPrecisionWithReference** — position-weighted relevance of retrieved chunks
- **LLMContextRecall** — fraction of reference statements covered by retrieved chunks
- Why precision and recall are complementary, not redundant
- How to inspect individual samples when aggregate scores hide a problem

### ✅ Prerequisites

- Module 06 completed (RAGAS setup + `EvaluationDataset`)
- Module 05 completed (cloud Ollama stack + `rag_answer` function)
- `OLLAMA_API_KEY` set in `tutorials/.env` (or follow along with the cached `frozen/` results)
""")

# ============================================================================
# STEP 0 — SETUP
# ============================================================================
md(r"""
---
# Step 0 · Setup

## 0.1 Install the libraries

The exact dependency list lives in **`pyproject.toml`** next to this notebook.
Install everything with one command, **from this module's folder**:

```bash
uv sync            # reads pyproject.toml, creates .venv/, installs everything
uv run jupyter lab # launch Jupyter inside that environment
```

When the notebook opens, pick the kernel **`Python 3 (ipykernel)`** (top-right
kernel picker).
""")

md(r"""
## 0.2 Provide your API keys (shared `.env`)

All twelve modules read their keys from a **single** `.env` file in the
`tutorials/` folder (the parent of this module). Create `tutorials/.env` once:

```
OLLAMA_API_KEY=...      # cloud Ollama — needed for judge LLM + embeddings
OLLAMA_API_BASE=http://localhost:11434
```

`find_dotenv()` walks UP from this notebook and locates that shared file, so you
never copy keys into each module. `.env` is gitignored — never commit it.

If you do not have a key yet, the notebook falls back to the illustrative scores
in `frozen/retriever_scores.json` so you can still follow along.
""")

code(r'''
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())          # resolves to tutorials/.env automatically
os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11434")
HAVE_KEYS = bool(os.environ.get("OLLAMA_API_KEY"))

if HAVE_KEYS:
    print("✓ OLLAMA_API_KEY found — will run live.")
else:
    print("No API key found. This notebook will use the cached results in "
          "frozen/ so you can still follow along.")
''')

md(r"""
## 0.3 RAGAS 0.4.3 import stub + asyncio patch

RAGAS 0.4.3 hard-imports a module that `langchain-community` 1.x removed.
The LiteLLM path never actually uses it, so we stub it out **before** importing
RAGAS. This must come first in every RAGAS notebook.

Also: Jupyter already runs an asyncio event loop, but RAGAS uses `asyncio.run()`
internally. `nest_asyncio.apply()` patches the loop so both can coexist.
""")

code(r'''
# -- RAGAS import stub (must come before any `from ragas import ...`) --
import sys, types

_vx = types.ModuleType("langchain_community.chat_models.vertexai")

class ChatVertexAI:  # placeholder, intentionally non-functional
    pass

_vx.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vx

# -- asyncio compatibility --
import nest_asyncio
nest_asyncio.apply()

print("RAGAS stub applied. nest_asyncio patched.")
''')

# ============================================================================
# STEP 1 — JUDGE LLM + RAGAS EMBEDDINGS
# ============================================================================
md(r"""
---
# Step 1 · Wire the judge LLM and RAGAS embeddings

RAGAS needs two model objects:

- **`judge_llm`** — evaluates relevance verdicts and decomposes reference answers
  into statements. We use `gemma4:31b-cloud`, deliberately **different** from the
  generator. Using the same model to generate and grade creates a conflict of
  interest: it tends to rate its own outputs highly.
- **`ragas_embeddings`** — used internally by some metrics for semantic
  similarity. We use `qwen3-embedding:0.6b`.

Both are served via LiteLLM, which translates the standard OpenAI API shape into
the Ollama backend format.

> **⚠ Cost note:** Every `evaluate()` call sends approximately
> `n_questions × k_chunks` LLM calls to the judge (one per chunk per question).
> With 8 questions and k=10, that is ~80 calls per metric. **Keep k small during
> development** to limit cost.
""")

code(r'''
import litellm
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

LLM_NAME_OLLAMA       = "nemotron-3-super:cloud"
JUDGE_MODEL           = "ollama_chat/gemma4:31b-cloud"
EMBEDDING_MODEL       = "ollama/qwen3-embedding:0.6b"

if HAVE_KEYS:
    judge_llm = llm_factory(
        JUDGE_MODEL,
        provider="litellm",
        client=litellm.completion,
        temperature=0.0,
    )
    ragas_embeddings = embedding_factory(
        "litellm",
        model=EMBEDDING_MODEL,
        api_base=os.environ["OLLAMA_API_BASE"],
    )
    print("Judge LLM and embeddings initialised.")
else:
    judge_llm = None
    ragas_embeddings = None
    print("(skipping model init — no keys)")
''')

# ============================================================================
# STEP 2 — RAG ANSWER + EVALUATION DATASET
# ============================================================================
md(r"""
---
# Step 2 · Rebuild the RAG answer function + build the EvaluationDataset

We need to run the RAG pipeline over every golden question so that each
`SingleTurnSample` holds the actual retrieved contexts and generated response.
This mirrors the setup from Module 06: the same corpus, the same vector store,
the same prompt — we are evaluating the **real output**, not a placeholder.
""")

code(r'''
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# -- Embedder for the vector store (same model as Module 05/06) --
EMBEDDING_NAME_OLLAMA = "qwen3-embedding:0.6b"

if HAVE_KEYS:
    lc_embeddings = OllamaEmbeddings(
        model=EMBEDDING_NAME_OLLAMA,
        base_url=os.environ["OLLAMA_API_BASE"],
    )
    chat_llm = ChatOllama(
        model=LLM_NAME_OLLAMA,
        base_url=os.environ["OLLAMA_API_BASE"],
        temperature=0.0,
    )

    # -- Load and chunk the 8-file metals corpus --
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
    vector_store = QdrantVectorStore.from_documents(
        lc_docs,
        embedding=lc_embeddings,
        location=":memory:",
        collection_name="metals_kb",
    )
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    print(f"Corpus loaded: {len(raw_docs)} files → {len(lc_docs)} chunks")
else:
    vector_store = None
    base_retriever = None
    chat_llm = None
    print("(skipping corpus load — no keys)")
''')

code(r'''
RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a precise metals-markets tutor. Answer using ONLY the context passages. "
    "If the context does not contain the answer, say you do not know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

def rag_answer(question: str, k: int = 10) -> dict:
    """Run the RAG pipeline and return response + retrieved contexts."""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    candidates = [d.page_content for d in retriever.invoke(question)]
    block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(candidates[:3], 1))
    response = chat_llm.invoke(
        RAG_PROMPT.format_messages(context=block, question=question)
    ).content.strip()
    return {"response": response, "retrieved_contexts": candidates}

print("rag_answer() defined.")
''')

md(r"""
Now build the `EvaluationDataset`. Each `SingleTurnSample` bundles:

- `user_input` — the question
- `retrieved_contexts` — what the retriever actually fetched
- `response` — what the generator actually said
- `reference` — the gold-standard answer we grade against

The `reference` field is what `LLMContextRecall` decomposes into statements.
""")

code(r'''
import json
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

golden = json.load(open("golden_questions.json"))

if HAVE_KEYS:
    samples = []
    for g in golden:
        result = rag_answer(g["question"])
        samples.append(SingleTurnSample(
            user_input=g["question"],
            retrieved_contexts=result["retrieved_contexts"],
            response=result["response"],
            reference=g["reference"],
        ))
    eval_dataset = EvaluationDataset(samples=samples)
    print(f"EvaluationDataset built: {len(samples)} samples")
    for s in samples:
        print(f"  Q: {s.user_input[:60]}…  contexts={len(s.retrieved_contexts)}")
else:
    samples = []
    eval_dataset = None
    print("(skipping dataset build — no keys)")
''')

# ============================================================================
# STEP 3 — EVALUATE
# ============================================================================
md(r"""
---
# Step 3 · Run evaluate() over the full dataset

`evaluate()` runs both metrics against every sample in the dataset and returns a
`Result` object. We convert it to a pandas DataFrame for easy inspection.

The two metrics we pass here are the **only** retriever metrics this module covers:
- `LLMContextPrecisionWithReference()` — precision + ranking
- `LLMContextRecall()` — coverage against the reference

Modules 08 and 09 add `ContextEntityRecall`, `NoiseSensitivity`, and the generator
metrics — don't jump ahead yet.
""")

code(r'''
import json

if HAVE_KEYS:
    from ragas import evaluate
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

    retriever_metrics = [
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    results = evaluate(
        dataset=eval_dataset,
        metrics=retriever_metrics,
        llm=judge_llm,
        embeddings=ragas_embeddings,
    )

    df = results.to_pandas()
    print(df[["user_input", "context_precision", "context_recall"]].to_string())

    scores = {
        "context_precision": float(df["context_precision"].mean()),
        "context_recall":    float(df["context_recall"].mean()),
    }
else:
    scores = json.load(open("frozen/retriever_scores.json"))
    print("(using cached illustrative result — set keys in tutorials/.env to run live)")

print(f"\nMean scores: precision={scores['context_precision']:.3f}  "
      f"recall={scores['context_recall']:.3f}")
''')

md(r"""
### Reading the output

| Score | What it means |
| --- | --- |
| `context_precision ≈ 0.78` | Most returned chunks were relevant and ranked near the top. Some noise chunks crept into the top-10 on a few questions. |
| `context_recall ≈ 0.71` | About 71% of the reference statements were covered by retrieved chunks. Multi-hop questions account for most of the gap. |

The gap between the two numbers is normal: precision rewards clean, well-ranked
retrieval, while recall catches the cases where an important passage wasn't
fetched at all.
""")

# ============================================================================
# STEP 4 — SINGLE-SAMPLE DEBUG
# ============================================================================
md(r"""
---
# Step 4 · Single-sample debug with single_turn_ascore()

Aggregate means hide per-question variation. The two multi-hop questions in the
golden set (questions 7 and 8, indices 6 and 7) typically show lower recall
because a single retrieval pass may miss one of the two required source passages.

`single_turn_ascore()` lets you score one sample at a time — useful for debugging
and for understanding *why* a question scored low.
""")

code(r'''
import asyncio

if HAVE_KEYS and len(samples) > 6:
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

    # The multi-hop silver question (index 6) is a good stress-test for recall
    sample = samples[6]

    precision_score = asyncio.run(
        LLMContextPrecisionWithReference(llm=judge_llm).single_turn_ascore(sample)
    )
    recall_score = asyncio.run(
        LLMContextRecall(llm=judge_llm).single_turn_ascore(sample)
    )

    print(f"Question (index 6): {sample.user_input}")
    print(f"  Retrieved chunks : {len(sample.retrieved_contexts)}")
    print(f"  Precision        : {precision_score:.3f}")
    print(f"  Recall           : {recall_score:.3f}")
    print()
    print("Retrieved contexts (first 120 chars each):")
    for i, ctx in enumerate(sample.retrieved_contexts, 1):
        print(f"  [{i}] {ctx[:120]}…")
else:
    print("(skipping single-sample debug — no keys or dataset not built)")
    print("Illustrative output:")
    print("  Question: Why might silver outperform gold in a strong economy...")
    print("  Precision: 0.81    Recall: 0.60")
    print("  (Recall is lower because this multi-hop question needs two source passages.)")
''')

md(r"""
### What low recall on a multi-hop question looks like

The silver question requires two facts from two different corpus files:
1. Silver has heavy industrial demand (from `03_silver_and_industrial_demand.md`)
2. Silver is more volatile than gold (also from `03_silver_and_industrial_demand.md`)

A single vector search may rank one passage high and the other lower, especially
if the query embedding aligns more strongly with one sub-topic than the other.
When recall drops below 0.7 on a question, always check:

1. **Are both required passages actually in the corpus?** (They are here.)
2. **Did the retriever fetch them, just not in the top-k returned?** (Try raising k.)
3. **Is the reference answer based on passages the corpus doesn't actually contain?**
   (Check the `reference_contexts` field in `golden_questions.json`.)
""")

# ============================================================================
# RECAP
# ============================================================================
md(r"""
---
# Recap & what's next

You learned:

- **LLMContextPrecisionWithReference** grades whether retrieved chunks were
  relevant and whether the relevant ones were ranked high. It is position-weighted
  — a good chunk at rank 1 counts more than the same chunk at rank 8.
- **LLMContextRecall** grades coverage: what fraction of the reference answer's
  statements were supported by at least one retrieved chunk. It is not sensitive
  to rank, only to presence.
- These two metrics are complementary: run both, read both. A system with
  0.95 precision and 0.45 recall has a different, equally serious problem from
  one with 0.45 precision and 0.95 recall.
- `single_turn_ascore()` lets you drill into individual samples when aggregate
  numbers hide the real trouble.

**Next module (08):** *More Retriever Metrics* — adds `ContextEntityRecall`
(did the retriever surface the specific named entities the answer needs?) and
`NoiseSensitivity` (how much does the generator degrade when noisy chunks are
mixed in?). Note that `NoiseSensitivity` uses an **inverted scale** — lower is
better.
""")

# ============================================================================
# EMIT NOTEBOOK  (do not change below except OUT)
# ============================================================================
def to_cell(kind: str, src: str) -> dict:
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


nb = {
    "cells": [to_cell(k, s) for k, s in CELLS],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = "07_retriever_metrics.ipynb"   # <-- this module's notebook filename
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

n_md   = sum(1 for k, _ in CELLS if k == "md")
n_code = sum(1 for k, _ in CELLS if k == "code")
print(f"Wrote {OUT}: {len(CELLS)} cells ({n_md} markdown, {n_code} code)")
