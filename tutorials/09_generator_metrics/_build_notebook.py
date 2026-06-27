"""Builds 09_generator_metrics.ipynb from a list of (type, source) cells.

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
# Module 09 · Generator Metrics + LLM-as-Judge

### A hands-on, build-it-yourself module for advanced high school researchers

One-paragraph orientation: Modules 07–08 taught you how to score the
**retriever** — did the right passages end up in the context window? This
module shifts focus to the other half of the pipeline: the **generator**.
You will measure whether the language model turned those passages into a
faithful, relevant, and factually correct answer, using three RAGAS metrics
(Faithfulness, ResponseRelevancy, FactualCorrectness). You will also learn
about **LLM-as-judge** — the technique every one of these metrics relies on —
and the biases you need to guard against. This is module 09 of a twelve-part
track that ends in a full **Agentic RAG Evaluation** capstone.
""")

md(r"""
## Summary: the one-paragraph version

A RAG pipeline can retrieve perfect passages and still produce a hallucinated
or off-topic answer. Generator metrics let you catch that. Faithfulness checks
whether every claim in the answer is grounded in the retrieved context.
Response Relevancy checks whether the answer actually addresses the question.
Factual Correctness compares the answer against a human-written reference to
catch errors the context itself may contain. All three delegate scoring to a
second LLM — the **judge** — which introduces well-known biases (verbosity,
position, self-preference) that you must mitigate by design.
""")

md(r"""
## What you will build, step by step

| Step | What you do | Key tool |
| ---: | --- | --- |
| 0 | Set up the environment and load your API key | `uv`, `python-dotenv` |
| 1 | Compatibility stub + event-loop patch | `sys.modules`, `nest_asyncio` |
| 2 | Load the corpus and build the in-memory vector store | `langchain-qdrant`, `langchain-ollama` |
| 3 | Build RAGAS LLM and embedding objects (judge at T=0) | `ragas`, `litellm` |
| 4 | Load golden questions and build the evaluation dataset | `ragas.dataset_schema` |
| 5 | Run Faithfulness — score + per-sample deep-dive | `Faithfulness` |
| 6 | Run Response Relevancy | `ResponseRelevancy` |
| 7 | Run Factual Correctness — F1, precision, recall modes | `FactualCorrectness` |
| 8 | Combine all three generator metrics in one run | `evaluate` |
| 9 | LLM-as-judge: biases, mitigations, design rules | prose + discussion |

### What you will *learn* (the concepts)

- **Faithfulness** — claim-level grounding vs the retrieved context; why
  faithfulness ≠ factual truth.
- **Response Relevancy** — back-question cosine similarity; why a faithful
  answer can still miss the question.
- **Factual Correctness** — F1/precision/recall against a golden reference;
  the only metric that catches errors in the context itself.
- **LLM-as-judge** — how the scoring works; verbosity bias, position bias,
  self-preference; temperature=0 discipline; generator ≠ judge.

### Prerequisites

- Modules 01–08 of this track, or equivalent familiarity with RAG, RAGAS
  setup, and retriever metrics.
- `OLLAMA_API_KEY` in `tutorials/.env` (the parent folder of this module).
- Basic Python. No deep ML background required.
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
`tutorials/` folder (the parent of this module). Create `tutorials/.env` once:

```
OLLAMA_API_KEY=...      # cloud Ollama (chat, judge, embeddings)
```

`find_dotenv()` walks UP from this notebook and locates that shared file
automatically. **Do NOT create a per-module `.env`.** `.env` is gitignored —
never commit it.

> **Cost note:** each evaluation run sends the question + context + answer to
> the judge LLM for scoring. With 8 golden questions and 3 metrics, expect
> roughly 24–30 judge calls. Check your metered-endpoint usage before looping.
""")

code(r'''
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())          # resolves to tutorials/.env automatically

HAVE_KEYS = bool(os.environ.get("OLLAMA_API_KEY"))

if not HAVE_KEYS:
    print(
        "No OLLAMA_API_KEY found.\n"
        "This notebook will use the cached results in frozen/ so you can still\n"
        "follow along. To run live, add your key to tutorials/.env and re-run."
    )
else:
    print("Keys loaded. Ready to run live evaluations.")
''')

# ============================================================================
# STEP 1 — COMPATIBILITY STUB + EVENT-LOOP PATCH
# ============================================================================
md(r"""
---
# Step 1 · Compatibility stub + event-loop patch

RAGAS 0.4.3 hard-imports a module that `langchain-community` 1.x removed. The
LiteLLM path never uses it, so we stub it **before** importing ragas to avoid
an `ImportError`.

Jupyter notebooks also run an event loop already; RAGAS uses `asyncio`
internally, so we apply `nest_asyncio` to allow nested event loops.
""")

code(r'''
import sys
import types

# ---- RAGAS 0.4.3 compatibility stub ------------------------------------
_vx = types.ModuleType("langchain_community.chat_models.vertexai")

class ChatVertexAI:  # placeholder, intentionally non-functional
    pass

_vx.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vx
# ------------------------------------------------------------------------

import nest_asyncio
nest_asyncio.apply()   # allow RAGAS async calls inside Jupyter's event loop

print("Stub installed. nest_asyncio applied.")
''')

# ============================================================================
# STEP 2 — CORPUS + VECTOR STORE
# ============================================================================
md(r"""
---
# Step 2 · Load the corpus and build the vector store

We load the same 8-file metals knowledge base from `corpus/` that you used in
Modules 05–08, chunk it, and index it into an in-memory Qdrant collection using
cloud Ollama embeddings.

This step is identical to Module 08 — if you already have it running you can
skip straight to Step 3.
""")

code(r'''
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11434")

LLM_NAME_OLLAMA       = "nemotron-3-super:cloud"
EMBEDDING_NAME_OLLAMA = "qwen3-embedding:0.6b"

chat_llm = ChatOllama(
    model=LLM_NAME_OLLAMA,
    base_url=os.environ["OLLAMA_API_BASE"],
    temperature=0.0,
)
lc_embeddings = OllamaEmbeddings(
    model=EMBEDDING_NAME_OLLAMA,
    base_url=os.environ["OLLAMA_API_BASE"],
)

# Load + chunk the metals corpus
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

# In-memory Qdrant vector store
if HAVE_KEYS:
    vector_store = QdrantVectorStore.from_documents(
        lc_docs,
        embedding=lc_embeddings,
        location=":memory:",
        collection_name="metals_kb",
    )
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    print(f"Vector store ready. {len(lc_docs)} chunks indexed.")
else:
    print("(skipping vector store — no API key)")
''')

# ============================================================================
# STEP 3 — RAGAS MODEL OBJECTS
# ============================================================================
md(r"""
---
# Step 3 · Build RAGAS LLM and embedding objects

RAGAS needs its own wrappers around the LLM and embeddings — not the LangChain
objects from Step 2, but the `llm_factory` / `embedding_factory` equivalents
from RAGAS itself (both backed by LiteLLM).

> **Design choice — judge at `temperature=0`:**
> The judge LLM must be deterministic so scores are reproducible across
> re-runs. We deliberately use a *different* model for judging
> (`gemma4:31b-cloud`) than for generating (`nemotron-3-super:cloud`) to avoid
> self-preference bias. See Step 9 for more on LLM-as-judge biases.
""")

code(r'''
import litellm
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

LLM_MODEL       = "ollama_chat/nemotron-3-super:cloud"   # generator
JUDGE_MODEL     = "ollama_chat/gemma4:31b-cloud"          # judge (different model!)
EMBEDDING_MODEL = "ollama/qwen3-embedding:0.6b"

if HAVE_KEYS:
    generator_llm = llm_factory(
        LLM_MODEL, provider="litellm",
        client=litellm.completion, temperature=0.3
    )
    judge_llm = llm_factory(
        JUDGE_MODEL, provider="litellm",
        client=litellm.completion, temperature=0.0    # deterministic judge
    )
    ragas_embeddings = embedding_factory(
        "litellm",
        model=EMBEDDING_MODEL,
        api_base=os.environ["OLLAMA_API_BASE"],
    )
    print("RAGAS model objects ready.")
else:
    generator_llm = judge_llm = ragas_embeddings = None
    print("(no API key — judge and embeddings set to None; frozen results will be used)")
''')

# ============================================================================
# STEP 4 — GOLDEN DATASET
# ============================================================================
md(r"""
---
# Step 4 · Load golden questions and build the evaluation dataset

We use the same 8 human-authored questions from `golden_questions.json` that
earlier modules introduced. For generator metrics each sample needs:

- `user_input` — the question
- `response` — the RAG answer (generated on the fly, or loaded from frozen)
- `retrieved_contexts` — the passages actually given to the generator
- `reference` — the human-authored correct answer (needed for Factual Correctness)
""")

code(r'''
import json
from langchain_core.prompts import ChatPromptTemplate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

with open("golden_questions.json") as f:
    golden = json.load(f)

RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a precise metals-markets tutor. Answer using ONLY the context passages. "
    "If the context does not contain the answer, say you do not know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

def rag_answer(question: str, k: int = 10) -> dict:
    """Retrieve k passages and generate an answer. Returns response + contexts."""
    candidates = [d.page_content for d in base_retriever.invoke(question)][:k]
    block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(candidates[:3], 1))
    response = chat_llm.invoke(
        RAG_PROMPT.format_messages(context=block, question=question)
    ).content.strip()
    return {"response": response, "retrieved_contexts": candidates[:3]}


if HAVE_KEYS:
    print("Building evaluation dataset (generating RAG answers)...")
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
    print(f"Dataset ready: {len(samples)} samples.")
else:
    print("(no API key — loading frozen illustrative dataset)")
    frozen_scores = json.load(open("frozen/generator_scores.json"))
    eval_dataset = None
    print("Frozen scores loaded:", frozen_scores)
''')

# ============================================================================
# STEP 5 — FAITHFULNESS
# ============================================================================
md(r"""
---
# Step 5 · Faithfulness

**Faithfulness** measures whether every claim in the generated answer is
supported by the retrieved context passages.

The judge LLM:
1. Decomposes the answer into atomic statements ("A troy ounce is 31.1 grams",
   "It is heavier than a regular ounce", …)
2. Checks each statement against the contexts
3. Returns `grounded statements / total statements`

**Score range:** 0 – 1. Higher is better.

> ⚠ **Faithfulness ≠ correctness.** A score of 1.0 means every claim is
> grounded in the context — but if the retrieved passages themselves are wrong,
> a fully faithful answer is still wrong. Factual Correctness (Step 7) catches
> this.
""")

code(r'''
from ragas import evaluate
from ragas.metrics import Faithfulness

if HAVE_KEYS and eval_dataset is not None:
    faith_results = evaluate(
        dataset=eval_dataset,
        metrics=[Faithfulness()],
        llm=judge_llm,
        embeddings=ragas_embeddings,
    )
    faith_df = faith_results.to_pandas()
    print(faith_df[["user_input", "faithfulness"]].to_string(index=False))
    print(f"\nMean Faithfulness: {faith_df['faithfulness'].mean():.3f}")
else:
    # Frozen illustrative result
    faith_score = frozen_scores["faithfulness"]
    print(f"(using cached illustrative result — set keys in tutorials/.env to run live)")
    print(f"Mean Faithfulness: {faith_score:.3f}")
''')

md(r"""
### Single-sample deep-dive

You can score one sample at a time without building a full dataset — useful
for debugging why a specific answer was marked unfaithful:
""")

code(r'''
import asyncio

if HAVE_KEYS and eval_dataset is not None:
    sample_0 = eval_dataset.samples[0]
    score_0 = asyncio.get_event_loop().run_until_complete(
        Faithfulness(llm=judge_llm).single_turn_ascore(sample_0)
    )
    print(f"Q: {sample_0.user_input}")
    print(f"Faithfulness for sample 0: {score_0:.3f}")
else:
    print("(single-sample scoring requires API keys)")
''')

# ============================================================================
# STEP 6 — RESPONSE RELEVANCY
# ============================================================================
md(r"""
---
# Step 6 · Response Relevancy

**Response Relevancy** measures whether the answer actually addresses the
question that was asked, using a *back-question* technique:

1. The judge generates *n* synthetic questions that the answer *could* be
   answering.
2. Cosine similarity is computed between those back-questions and the original
   question using the embedding model.
3. The mean similarity is the score.

**Score range:** 0 – 1. Higher is better.

> A faithful answer (every claim grounded) can still score low here if it
> drifts off-topic — for example, accurately quoting context about platinum
> when the question was about gold.
""")

code(r'''
from ragas.metrics import ResponseRelevancy

if HAVE_KEYS and eval_dataset is not None:
    rel_results = evaluate(
        dataset=eval_dataset,
        metrics=[ResponseRelevancy()],
        llm=judge_llm,
        embeddings=ragas_embeddings,
    )
    rel_df = rel_results.to_pandas()
    print(rel_df[["user_input", "answer_relevancy"]].to_string(index=False))
    print(f"\nMean Response Relevancy: {rel_df['answer_relevancy'].mean():.3f}")
else:
    rel_score = frozen_scores["answer_relevancy"]
    print(f"(using cached illustrative result — set keys in tutorials/.env to run live)")
    print(f"Mean Response Relevancy: {rel_score:.3f}")
''')

# ============================================================================
# STEP 7 — FACTUAL CORRECTNESS
# ============================================================================
md(r"""
---
# Step 7 · Factual Correctness

**Factual Correctness** compares the answer against the human-authored
**reference** in the golden dataset (not against the retrieved context).

The judge:
1. Extracts atomic claims from both the answer and the reference.
2. Computes **precision** = correct answer claims / all answer claims.
3. Computes **recall** = correct answer claims / all reference claims.
4. Default mode returns the **F1** (harmonic mean).

This is the only generator metric that catches errors the retriever already
introduced — because it compares to a ground-truth reference, not to whatever
passages happened to be retrieved.

You can also run precision-only or recall-only modes to diagnose the failure:
- **Low precision** → model adds wrong facts not in the reference
- **Low recall** → model omits key facts the reference covers
""")

code(r'''
from ragas.metrics import FactualCorrectness

if HAVE_KEYS and eval_dataset is not None:
    # Default F1 mode
    fc_results = evaluate(
        dataset=eval_dataset,
        metrics=[FactualCorrectness()],
        llm=judge_llm,
        embeddings=ragas_embeddings,
    )
    fc_df = fc_results.to_pandas()
    print(fc_df[["user_input", "factual_correctness"]].to_string(index=False))
    print(f"\nMean Factual Correctness (F1): {fc_df['factual_correctness'].mean():.3f}")

    # Precision and recall modes
    fc_p = FactualCorrectness(llm=judge_llm, mode="precision")
    fc_r = FactualCorrectness(llm=judge_llm, mode="recall")
    s0 = eval_dataset.samples[0]
    p0 = asyncio.get_event_loop().run_until_complete(fc_p.single_turn_ascore(s0))
    r0 = asyncio.get_event_loop().run_until_complete(fc_r.single_turn_ascore(s0))
    print(f"\nSample 0 — Precision: {p0:.3f}  Recall: {r0:.3f}")
else:
    fc_f1   = frozen_scores["factual_correctness"]
    fc_prec = frozen_scores["factual_correctness_precision"]
    fc_rec  = frozen_scores["factual_correctness_recall"]
    print(f"(using cached illustrative result — set keys in tutorials/.env to run live)")
    print(f"Mean Factual Correctness (F1): {fc_f1:.3f}")
    print(f"Illustrative sample 0 — Precision: {fc_prec:.3f}  Recall: {fc_rec:.3f}")
''')

# ============================================================================
# STEP 8 — ALL THREE TOGETHER
# ============================================================================
md(r"""
---
# Step 8 · All three generator metrics in one run

Running all metrics in a single `evaluate()` call is more efficient than three
separate calls because RAGAS can share judge calls where metrics overlap.
""")

code(r'''
import pandas as pd
from ragas.metrics import Faithfulness, ResponseRelevancy, FactualCorrectness

if HAVE_KEYS and eval_dataset is not None:
    all_results = evaluate(
        dataset=eval_dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            FactualCorrectness(),
        ],
        llm=judge_llm,
        embeddings=ragas_embeddings,
    )
    all_df = all_results.to_pandas()
    cols = ["user_input", "faithfulness", "answer_relevancy", "factual_correctness"]
    print(all_df[cols].to_string(index=False))
    print("\nMeans:")
    print(all_df[cols[1:]].mean().to_string())
else:
    # Show the frozen illustrative scores
    illustrative = {
        "faithfulness":         frozen_scores["faithfulness"],
        "answer_relevancy":     frozen_scores["answer_relevancy"],
        "factual_correctness":  frozen_scores["factual_correctness"],
    }
    print("(using cached illustrative result — set keys in tutorials/.env to run live)")
    summary_df = pd.DataFrame([illustrative])
    print(summary_df.to_string(index=False))
''')

# ============================================================================
# STEP 9 — LLM-AS-JUDGE: BIASES AND MITIGATIONS
# ============================================================================
md(r"""
---
# Step 9 · LLM-as-judge — biases and mitigations

Every generator metric in this module delegates scoring to a second LLM — the
**judge**. Understanding the judge's failure modes is as important as
understanding the metric itself.

## What the judge does

The judge LLM receives a structured prompt containing the question, the
retrieved context, the generated answer, and (for some metrics) the reference.
It returns a numeric score or a verdict on individual claims. This is far more
flexible than string-matching metrics like ROUGE or BLEU, which reward exact
wording rather than meaning.

## The three main biases

| Bias | What happens | Mitigation |
|---|---|---|
| **Verbosity bias** | The judge prefers longer, more elaborate answers even when they contain errors | Evaluate concise and verbose answers separately; instruct the judge to penalise unsupported claims |
| **Position bias** | In A/B comparisons the judge favours the first candidate it sees | Randomise candidate order; average scores from both orderings |
| **Self-preference** | A model rates its own output higher than output from other models | **Always use a different model as judge and generator** |

## Design rules we follow in this track

1. **Generator** = `nemotron-3-super:cloud` at `temperature=0.3` (some variation
   is fine when generating answers)
2. **Judge** = `gemma4:31b-cloud` at `temperature=0.0` (deterministic, different
   model family)
3. Never run the same model as both generator and judge.
4. The judge is the same across Modules 06–12 so scores are comparable across modules.

## Why `temperature=0` for the judge?

Reproducibility. If you run the same evaluation twice and get different scores,
you cannot tell whether the pipeline improved or the judge changed its mind. A
deterministic judge eliminates that source of noise.
""")

code(r'''
# This cell is a summary illustration — no live API calls needed.
import json

judge_config = {
    "model":        "ollama_chat/gemma4:31b-cloud",
    "temperature":  0.0,          # deterministic
    "role":         "judge only", # never used as generator
}

generator_config = {
    "model":       "ollama_chat/nemotron-3-super:cloud",
    "temperature": 0.3,
    "role":        "generator only",
}

print("Judge config:")
print(json.dumps(judge_config, indent=2))
print("\nGenerator config:")
print(json.dumps(generator_config, indent=2))
print("\nKey rule: judge model != generator model  ->", judge_config["model"] != generator_config["model"])
''')

# ============================================================================
# RECAP
# ============================================================================
md(r"""
---
# Recap & what's next

## What you learned

- **Faithfulness** decomposes the answer into claims and checks each one
  against the retrieved context. A score of 1.0 means no hallucination — but
  does not mean the answer is correct if the context was already wrong.

- **Response Relevancy** uses a back-question cosine technique to check
  whether the answer actually addresses the question. A faithful but off-topic
  answer will score low here.

- **Factual Correctness** compares the answer to a golden reference using F1,
  precision, and recall modes. It is the only metric that catches errors the
  retriever introduced.

- **LLM-as-judge** is the scoring technique behind all three. Keep the judge
  deterministic (`temperature=0`), keep the judge model different from the
  generator model, and be aware of verbosity, position, and self-preference
  biases.

## Illustrative generator scores (frozen)

| Metric | Score |
|---|---|
| Faithfulness | 0.88 |
| Response Relevancy | 0.82 |
| Factual Correctness (F1) | 0.69 |

*Illustrative output — set `OLLAMA_API_KEY` in `tutorials/.env` to run live.*

**Next module (10):** Reranking — you will add Cohere rerank-v3.5 to the
retrieval step, retrieve k=10 candidates, rerank to top 3, and measure whether
the generator metrics you learned here improve as a result.
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
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = "09_generator_metrics.ipynb"
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

n_md   = sum(1 for k, _ in CELLS if k == "md")
n_code = sum(1 for k, _ in CELLS if k == "code")
print(f"Wrote {OUT}: {len(CELLS)} cells ({n_md} markdown, {n_code} code)")
