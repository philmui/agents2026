"""Builds 08_more_retriever_metrics.ipynb from a list of (type, source) cells.

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
# Module 08 · More Retriever Metrics

### A hands-on, build-it-yourself module for advanced high school researchers

![Context Entity Recall diagram](slides/assets/05_context_entities_recall.svg)

In Module 07 you measured retrieval quality at the *passage* level with Precision and Recall.
This module adds two complementary metrics: **ContextEntityRecall** checks whether retrieved
passages actually contain the *specific named entities* (metals, units, instruments) the correct
answer requires, and **NoiseSensitivity** measures how often irrelevant retrieved passages
actively mislead the generator into producing incorrect claims.

This is Module 08 of a twelve-part track that ends in a full **Agentic RAG Evaluation** capstone.
""")

md(r"""
## 📋 Summary: the one-paragraph version

Precision and Recall (Module 07) tell you whether the right *chunks* came back. Entity Recall
tells you whether those chunks carry the right *facts* — did the retriever surface passages that
mention "troy ounce", "LBMA", "contango", and the other domain-specific entities the answer
needs? NoiseSensitivity flips the question: even if precision is high, do the few irrelevant
passages that slipped through corrupt the generator's output? Together these two metrics complete
the retriever diagnostic picture and point directly at two different kinds of fix (better
chunking/k-sizing vs. reranking).
""")

md(r"""
## 🗺️ What you will build, step by step

| Step | What you do | Key tool |
| ---: | --- | --- |
| 0 | Set up the environment + load shared API key | `uv`, `python-dotenv` |
| 1 | Stub the missing RAGAS import + patch event loop | `sys.modules`, `nest_asyncio` |
| 2 | Build RAGAS judge LLM and embeddings | `ragas.llms`, `ragas.embeddings` |
| 3 | Load corpus, build vector store, build `eval_dataset` | `langchain-qdrant`, `ragas` |
| 4 | Run `ContextEntityRecall` on 8 golden questions | `ragas.metrics` |
| 5 | Run `NoiseSensitivity` (⚠ lower is better) | `ragas.metrics` |
| 6 | Run both metrics together with `evaluate()` | `ragas.evaluate` |
| 7 | Single-sample debug: inspect one question in detail | `single_turn_ascore` |

### 🎓 What you will *learn* (the concepts)

- **ContextEntityRecall**: how to measure whether retrieved passages contain the domain-specific
  named entities a correct answer requires.
- **NoiseSensitivity (⚠ inverted scale)**: how to detect whether irrelevant retrieved passages
  are corrupting the generator's output — and why **lower is better** here.
- How to read a RAGAS results table when one column points in the opposite direction from all
  the others.
- How to use `single_turn_ascore()` to debug a single question instead of paying for a full run.

### ✅ Prerequisites

- Module 07 (retriever metrics: Precision and Recall). You should be comfortable building an
  `EvaluationDataset` and calling `evaluate()`.
- `OLLAMA_API_KEY` in `tutorials/.env` (the parent folder). See Step 0.2 below.
- Basic Python. No machine learning background required.
""")


# ============================================================================
# STEP 0 — SETUP
# ============================================================================
md(r"""
---
# Step 0 · Setup

## 0.1 Install the libraries

The exact dependency list lives in **`pyproject.toml`** next to this notebook,
so the environment is **reproducible**. Install everything with one command,
**from this module's folder**, using [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync            # reads pyproject.toml, creates .venv/, installs everything
uv run jupyter lab # launch Jupyter inside that environment
```

When the notebook opens, pick the kernel **`Python 3 (ipykernel)`** (top-right
kernel picker). That is the interpreter from `.venv`, so every `import` below
resolves against what `uv sync` installed.
""")

md(r"""
## 0.2 Provide your API key (shared `.env`)

All twelve modules read their keys from a **single** `.env` file in the
`tutorials/` folder (the parent of this module). Create `tutorials/.env` once
if it does not already exist:

```
OLLAMA_API_KEY=your-ollama-cloud-key-here
OLLAMA_API_BASE=http://localhost:11434
```

`find_dotenv()` walks UP from this notebook and locates that shared file, so you
never copy keys into each module. `.env` is gitignored — never commit it.

> **Cost note:** Both metrics require LLM judge calls (entity extraction for
> ContextEntityRecall, claim tracing for NoiseSensitivity). Running 8 golden
> questions with both metrics costs roughly 16–24 judge calls. Use the
> single-sample debug in Step 7 to validate setup before a full run.
""")

code(r'''
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())                           # resolves to tutorials/.env
os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11434")

HAVE_KEYS = bool(os.environ.get("OLLAMA_API_KEY"))
if not HAVE_KEYS:
    print("⚠  No OLLAMA_API_KEY found in tutorials/.env")
    print("   This notebook will use cached illustrative results from frozen/")
    print("   so you can still follow along. Set the key to run live.")
else:
    print("✓ OLLAMA_API_KEY loaded — notebook will run live.")
''')


# ============================================================================
# STEP 1 — RAGAS IMPORT STUB + NEST_ASYNCIO
# ============================================================================
md(r"""
---
# Step 1 · RAGAS import stub + event-loop patch

RAGAS 0.4.3 hard-imports a module that `langchain-community` 1.x removed. The
LiteLLM path we use never needs it, so we stub the missing module before RAGAS
loads. We also patch the event loop so RAGAS's asyncio calls work inside Jupyter.
""")

code(r'''
import sys, types

# Stub the removed langchain-community module that RAGAS 0.4.3 still imports.
# The LiteLLM path never calls it; this just satisfies the import machinery.
_vx = types.ModuleType("langchain_community.chat_models.vertexai")

class ChatVertexAI:  # placeholder — intentionally non-functional
    pass

_vx.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vx

# Jupyter already runs an event loop; nest_asyncio lets RAGAS's async calls
# work inside it without raising "This event loop is already running".
import nest_asyncio
nest_asyncio.apply()

print("✓ RAGAS stub applied; event loop patched.")
''')


# ============================================================================
# STEP 2 — RAGAS JUDGE LLM + EMBEDDINGS
# ============================================================================
md(r"""
---
# Step 2 · Build the RAGAS judge LLM and embeddings

Both metrics use an LLM as judge:
- **ContextEntityRecall** — the judge extracts named entities from the reference
  answer and checks how many appear in the retrieved context.
- **NoiseSensitivity** — the judge identifies incorrect claims in the generated
  answer and traces each one back to a noisy retrieved passage.

We use a separate, stronger judge model (`gemma4:31b-cloud`) from the generator
(`nemotron-3-super:cloud`) so the evaluator is not grading its own output.
""")

code(r'''
import json

# ---- Model identifiers (match the capstone exactly) ----
LLM_MODEL       = "ollama_chat/nemotron-3-super:cloud"
JUDGE_MODEL     = "ollama_chat/gemma4:31b-cloud"
EMBEDDING_MODEL = "ollama/qwen3-embedding:0.6b"

if HAVE_KEYS:
    import litellm
    from ragas.llms import llm_factory
    from ragas.embeddings.base import embedding_factory

    generator_llm    = llm_factory(LLM_MODEL,   provider="litellm",
                                   client=litellm.completion, temperature=0.3)
    judge_llm        = llm_factory(JUDGE_MODEL,  provider="litellm",
                                   client=litellm.completion, temperature=0.0)
    ragas_embeddings = embedding_factory("litellm", model=EMBEDDING_MODEL,
                                         api_base=os.environ["OLLAMA_API_BASE"])
    print("✓ RAGAS judge LLM and embeddings ready.")
else:
    generator_llm = judge_llm = ragas_embeddings = None
    print("(skipped — no keys; frozen results will be used in Steps 4–6)")
''')


# ============================================================================
# STEP 3 — CORPUS + VECTOR STORE + EVALUATION DATASET
# ============================================================================
md(r"""
---
# Step 3 · Load corpus, build vector store, assemble `eval_dataset`

We build the same retrieval pipeline used throughout the track: 8 metals corpus
files split into 500-character chunks, embedded with the cloud Ollama embedder,
stored in an in-memory Qdrant collection. Then we run each of the 8 golden
questions through the retriever and record the retrieved contexts to build an
`EvaluationDataset` for RAGAS.
""")

code(r'''
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the 8 metals corpus files from the local corpus/ folder.
raw_docs = [
    {"source": p.name, "page_content": p.read_text()}
    for p in sorted(Path("corpus").glob("*.md"))
]
print(f"Loaded {len(raw_docs)} corpus files.")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
lc_docs = [
    Document(page_content=piece, metadata={"source": d["source"]})
    for d in raw_docs
    for piece in splitter.split_text(d["page_content"])
]
print(f"Split into {len(lc_docs)} chunks.")
''')

code(r'''
if HAVE_KEYS:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from langchain_qdrant import QdrantVectorStore

    lc_embeddings = OllamaEmbeddings(
        model="qwen3-embedding:0.6b",
        base_url=os.environ["OLLAMA_API_BASE"],
    )
    vector_store = QdrantVectorStore.from_documents(
        lc_docs, embedding=lc_embeddings,
        location=":memory:", collection_name="metals_kb_m08",
    )
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    print("✓ Vector store built.")
else:
    base_retriever = None
    print("(vector store skipped — no keys)")
''')

code(r'''
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a precise metals-markets tutor. Answer using ONLY the context passages. "
    "If the context does not contain the answer, say you do not know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

def rag_answer(question: str, k: int = 10) -> dict:
    """Retrieve k passages and generate an answer; return response + contexts."""
    candidates = [d.page_content for d in base_retriever.invoke(question)]
    contexts   = candidates[:k]
    block      = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    from langchain_ollama import ChatOllama
    chat_llm = ChatOllama(
        model="nemotron-3-super:cloud",
        base_url=os.environ["OLLAMA_API_BASE"],
        temperature=0.0,
    )
    response = chat_llm.invoke(
        RAG_PROMPT.format_messages(context=block, question=question)
    ).content.strip()
    return {"response": response, "retrieved_contexts": contexts}
''')

code(r'''
import json as _json
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

golden = _json.load(open("golden_questions.json"))

if HAVE_KEYS:
    samples = []
    for g in golden:
        out = rag_answer(g["question"])
        samples.append(SingleTurnSample(
            user_input=g["question"],
            response=out["response"],
            retrieved_contexts=out["retrieved_contexts"],
            reference=g["reference"],
        ))
    eval_dataset = EvaluationDataset(samples=samples)
    print(f"✓ EvaluationDataset built with {len(samples)} samples.")
else:
    # Build a minimal illustrative dataset so RAGAS imports still work.
    samples = [
        SingleTurnSample(
            user_input=g["question"],
            response="(illustrative — no live run)",
            retrieved_contexts=g.get("reference_contexts", ["(no context)"]),
            reference=g["reference"],
        )
        for g in golden
    ]
    eval_dataset = EvaluationDataset(samples=samples)
    print("(using illustrative dataset — set OLLAMA_API_KEY to run live)")
''')


# ============================================================================
# STEP 4 — CONTEXT ENTITY RECALL
# ============================================================================
md(r"""
---
# Step 4 · ContextEntityRecall

**ContextEntityRecall** measures whether the retrieved context contains the
specific named entities the correct answer requires.

The RAGAS judge:
1. Reads the *reference answer* and extracts named entities
   (e.g., "troy ounce", "LBMA", "contango", "South Africa").
2. Checks how many of those entities appear anywhere in the *retrieved context*.
3. Returns the fraction: `|found| / |reference entities|`.

**Higher is better** (range 0 – 1).
""")

code(r'''
from ragas.metrics import ContextEntityRecall
from ragas import evaluate

if HAVE_KEYS:
    entity_results = evaluate(
        dataset=eval_dataset,
        metrics=[ContextEntityRecall()],
        llm=judge_llm,
        embeddings=ragas_embeddings,
    )
    df_entity = entity_results.to_pandas()
    print(df_entity[["user_input", "context_entity_recall"]].to_string())
else:
    entity_result_data = _json.load(open("frozen/step3_entity_recall.json"))
    print("(using cached illustrative result — set OLLAMA_API_KEY to run live)\n")
    print(f"Mean context_entity_recall: {entity_result_data['context_entity_recall']:.3f}")
    import pandas as pd
    df_entity = pd.DataFrame(entity_result_data["per_question"])
    print(df_entity.to_string())
''')

md(r"""
### What the scores tell you

A `context_entity_recall` of **0.64** means that on average, about two-thirds of
the named entities in the reference answers were present somewhere in the 10
retrieved passages. The multi-hop questions (Q7, Q8) tend to score lower because
they require entities from *two different source documents* to both appear in the
retrieved context at the same time.

**If entity recall is low:**
- The chunks may be splitting a key entity mention across a boundary → reduce
  `chunk_overlap` or reduce `chunk_size`.
- The retriever may not be surfacing the entity-rich passages → try a larger `k`.
""")


# ============================================================================
# STEP 5 — NOISE SENSITIVITY
# ============================================================================
md(r"""
---
# Step 5 · NoiseSensitivity

**NoiseSensitivity** answers: when the generator makes an incorrect claim, how
often was it led there by an irrelevant passage in the retrieved context?

## ⚠ Caution — Inverted Scale

> **NoiseSensitivity uses an INVERTED scale. LOWER is better.**
>
> - Score = 0 → the generator was *never* misled by noisy context.
> - Score = 1 → *every* incorrect claim came from a noisy passage.
>
> **Do NOT celebrate a high NoiseSensitivity score.** When sorting results,
> always sort *ascending* to put the best-performing questions at the top.
> Label this column clearly in any dashboard or report.

This is the one metric in the retriever suite where DOWN is the direction you want.
""")

code(r'''
from ragas.metrics import NoiseSensitivity

# NoiseSensitivity: LOWER is better — measures how often the generator
# is misled by irrelevant retrieved passages.
if HAVE_KEYS:
    noise_results = evaluate(
        dataset=eval_dataset,
        metrics=[NoiseSensitivity()],
        llm=judge_llm,
        embeddings=ragas_embeddings,
    )
    df_noise = noise_results.to_pandas()
    # Sort ASCENDING — lowest (best) noise sensitivity at the top.
    df_noise_sorted = df_noise.sort_values("noise_sensitivity_relevant", ascending=True)
    print(df_noise_sorted[["user_input", "noise_sensitivity_relevant"]].to_string())
else:
    noise_result_data = _json.load(open("frozen/step4_noise_sensitivity.json"))
    print("(using cached illustrative result — set OLLAMA_API_KEY to run live)")
    print(noise_result_data["_note"])
    print(f"\nMean noise_sensitivity_relevant: {noise_result_data['noise_sensitivity_relevant']:.3f}")
    import pandas as pd
    df_noise = pd.DataFrame(noise_result_data["per_question"])
    # Sort ASCENDING — best (lowest) at top
    df_noise_sorted = df_noise.sort_values("noise_sensitivity_relevant", ascending=True)
    print(df_noise_sorted.to_string())
''')

md(r"""
### What the scores tell you

An aggregate `noise_sensitivity_relevant` of **0.18** is healthy — only about
one-in-five incorrect answer claims traced back to noisy context. If you see
scores above 0.4, the retriever is returning too many loosely related chunks.

**To reduce NoiseSensitivity:**
- Decrease `k` (fewer retrieved passages = less noise surface area).
- Add reranking (Module 10) to filter the top-10 candidates down to only the
  most relevant 3 before the generator sees them.
""")


# ============================================================================
# STEP 6 — BOTH METRICS TOGETHER
# ============================================================================
md(r"""
---
# Step 6 · Run both metrics together with `evaluate()`

It is more efficient (and easier to cross-reference) to run both metrics in a
single `evaluate()` call. RAGAS batches the judge calls across metrics.
""")

code(r'''
from ragas.metrics import ContextEntityRecall, NoiseSensitivity
from ragas import evaluate

if HAVE_KEYS:
    combined_results = evaluate(
        dataset=eval_dataset,
        metrics=[
            ContextEntityRecall(),
            NoiseSensitivity(),   # LOWER is better
        ],
        llm=judge_llm,
        embeddings=ragas_embeddings,
    )
    df_combined = combined_results.to_pandas()
    print(df_combined[[
        "user_input",
        "context_entity_recall",
        "noise_sensitivity_relevant",   # lower is better
    ]].to_string())
    print(f"\nMean context_entity_recall:   {df_combined['context_entity_recall'].mean():.3f}  (↑ higher better)")
    print(f"Mean noise_sensitivity_relevant: {df_combined['noise_sensitivity_relevant'].mean():.3f}  (↓ lower better)")
else:
    import pandas as pd
    er  = pd.DataFrame(entity_result_data["per_question"])
    ns  = pd.DataFrame(noise_result_data["per_question"])
    df_combined = er.merge(ns, on="question")
    df_combined.columns = ["user_input", "context_entity_recall", "noise_sensitivity_relevant"]
    print("(illustrative combined results)\n")
    print(df_combined.to_string())
    print(f"\nMean context_entity_recall:      {entity_result_data['context_entity_recall']:.3f}  (↑ higher better)")
    print(f"Mean noise_sensitivity_relevant: {noise_result_data['noise_sensitivity_relevant']:.3f}  (↓ lower better)")
''')

md(r"""
### Reading the combined table — direction matters

| Metric | Good score | Scale direction |
|--------|-----------|-----------------|
| `context_entity_recall` | High → 1.0 | ↑ Higher is better |
| `noise_sensitivity_relevant` | **Low → 0** | **↓ Lower is better ⚠** |

Together with Module 07's Precision and Recall, you now have four retriever
metrics that diagnose four distinct failure modes:

- **Precision** — are the returned chunks relevant at all?
- **Recall** — are the key chunks present?
- **EntityRecall** — do the chunks carry the right named facts?
- **NoiseSensitivity** — are the wrong chunks corrupting the answer?
""")


# ============================================================================
# STEP 7 — SINGLE-SAMPLE DEBUG
# ============================================================================
md(r"""
---
# Step 7 · Single-sample debug with `single_turn_ascore()`

Before paying for a full `evaluate()` run on 8 questions (16–24 judge calls),
use `single_turn_ascore()` to validate your setup on one sample and inspect the
intermediate output. This is also useful when a particular question scores
unexpectedly low and you want to understand why.
""")

code(r'''
import asyncio

if HAVE_KEYS and samples:
    # Inspect question index 3 (the platinum/palladium supply question).
    sample_to_debug = samples[3]

    entity_score = asyncio.run(
        ContextEntityRecall(llm=judge_llm).single_turn_ascore(sample_to_debug)
    )
    noise_score = asyncio.run(
        NoiseSensitivity(llm=judge_llm).single_turn_ascore(sample_to_debug)
    )

    print(f"Question:              {sample_to_debug.user_input}")
    print(f"context_entity_recall: {entity_score:.3f}  (↑ higher better)")
    print(f"noise_sensitivity:     {noise_score:.3f}  (↓ lower better ⚠)")
else:
    print("Single-sample debug requires OLLAMA_API_KEY.")
    print("Illustrative output:")
    print("  Question:              Why is platinum and palladium supply considered risky?")
    print("  context_entity_recall: 0.710  (↑ higher better)")
    print("  noise_sensitivity:     0.120  (↓ lower better ⚠)")
''')


# ============================================================================
# RECAP
# ============================================================================
md(r"""
---
# Recap & what's next

You have now added two more retriever metrics to your evaluation toolkit:

- **ContextEntityRecall** measures whether the retrieved context contains the
  specific named entities (metals, units, instruments) the correct answer
  requires. Higher is better (0–1).

- **NoiseSensitivity** ⚠ measures how often irrelevant retrieved passages
  corrupt the generator's output. **Lower is better** — this is the one
  inverted-scale metric in the retriever suite. Remember to sort ascending and
  label it clearly in any dashboard.

Together with Precision and Recall from Module 07, you now have four orthogonal
views of retrieval quality. If entity recall is low, improve chunking or
increase `k`. If noise sensitivity is high, decrease `k` or add reranking.

**Next module (09 — Generator Metrics):** we shift the lens from retrieval to
generation. You will learn Faithfulness, ResponseRelevancy, and FactualCorrectness
— three RAGAS metrics that judge the *answer* rather than the *context* — and
encounter a second LLM-as-judge pattern where the evaluator decomposes the
answer into individual claims before scoring each one.
""")


# ============================================================================
# EMIT NOTEBOOK  (do not change below except OUT)
# ============================================================================
def to_cell(kind: str, src: str) -> dict:
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}

nb = {
    "cells": [to_cell(k, s) for k, s in CELLS],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = "08_more_retriever_metrics.ipynb"
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

n_md   = sum(1 for k, _ in CELLS if k == "md")
n_code = sum(1 for k, _ in CELLS if k == "code")
print(f"Wrote {OUT}: {len(CELLS)} cells ({n_md} markdown, {n_code} code)")
