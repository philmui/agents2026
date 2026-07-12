"""Builds 10_reranking.ipynb from a list of (type, source) cells.

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
# Module 10 · Reranking

### A hands-on, build-it-yourself module for advanced high school researchers

![Reranking pipeline](slides/assets/11_reranking_pipeline.svg)

In Module 09 you measured generator quality — Faithfulness, ResponseRelevancy,
and FactualCorrectness. Now we step back to the **retriever** side and ask: can
we hand the generator better raw material? The answer is a *reranker*: retrieve
**k=10** candidates cheaply with cosine similarity, then let a cross-encoder
rescore them and keep only **top_n=3** for the prompt.

This is Module 10 of a twelve-part track that ends in a full **Agentic RAG
Evaluation** capstone. Module 11 will wrap this pipeline in a LangGraph agent
with live tool calls.
""")

md(r"""
## 📋 Summary: the one-paragraph version

Embedding-based retrieval is fast and broad, but cosine similarity measures
*general semantic overlap* rather than *answer relevance*. A **cross-encoder
reranker** (Cohere rerank-v3.5) takes the query and each candidate passage
*together* and scores them with fine-grained token-level attention. Because
that is expensive, we only run it on the small shortlist the bi-encoder already
produced. The result: Context Precision rises, the generator receives tighter
context, and answers improve — at the cost of one extra API call per query.
""")

md(r"""
## 🗺️ What you will build, step by step

| Step | What you do | Key tool |
| ---: | --- | --- |
| 0 | Install deps, load keys from `tutorials/.env` | `uv`, `python-dotenv` |
| 1 | Import libraries, load corpus, build in-memory vector store | `langchain-qdrant`, `langchain-ollama` |
| 2 | Implement `rerank()` using Cohere rerank-v3.5 | `cohere.ClientV2` |
| 3 | Implement `rag_answer()` with `use_rerank` toggle | `ChatOllama`, `ChatPromptTemplate` |
| 4 | Before/after comparison on a golden question | side-by-side print |
| 5 | Connect to retriever metrics (precision/recall lift) | illustrative table |
| 6 | Recap and pointer to Module 11 | — |

### 🎓 What you will *learn* (the concepts)

- Why embedding retrieval is high-recall but low-precision
- How cross-encoders differ from bi-encoders and why we use both
- The **retrieve-wide-then-narrow** pattern (k=10 → top_n=3)
- How to call Cohere `rerank-v3.5` and interpret relevance scores
- How to measure the before/after lift in Context Precision

### ✅ Prerequisites

- Modules 05–09 completed (cloud-Ollama RAG pipeline + RAGAS metrics).
- `tutorials/.env` with **both** `OLLAMA_API_KEY` and `COHERE_API_KEY`.
- Students without keys can follow along using `frozen/rerank_comparison.json`.
""")

# ============================================================================
# STEP 0 — SETUP
# ============================================================================
md(r"""
---
# Step 0 · Setup

## 0.1 Install the libraries

The exact dependency list lives in **`pyproject.toml`** next to this notebook.
Install everything with one command from this module's folder:

```bash
uv sync            # reads pyproject.toml, creates .venv/, installs everything
uv run jupyter lab # launch Jupyter inside that environment
```

When the notebook opens, pick the kernel **`Python 3 (ipykernel)`** (top-right
kernel picker). That interpreter comes from `.venv`, so every `import` below
resolves against what `uv sync` installed.
""")

md(r"""
## 0.2 Provide your API keys (shared `.env`)

This module needs **two** paid API keys:

| Key | Service | Where to get one |
| --- | --- | --- |
| `OLLAMA_API_KEY` | Cloud Ollama — chat + embeddings | https://ollama.com |
| `COHERE_API_KEY` | Cohere reranking (new in M10) | https://dashboard.cohere.com/api-keys |

Both live in the **single shared** `tutorials/.env` file (the parent of this
module). Create it once:

```
OLLAMA_API_KEY=your_ollama_key_here
COHERE_API_KEY=your_cohere_key_here
```

`find_dotenv()` walks UP from this notebook and finds that file automatically.
**Do NOT create a per-module `.env`.** `.env` is gitignored — never commit keys.

> **Cost note:** Cohere's free tier allows a generous number of rerank calls per
> month. The 8-question golden set uses 8 calls × 10 documents each. Well within
> free tier limits for a tutorial run.
""")

code(r'''
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())   # resolves to tutorials/.env automatically

HAVE_KEYS = bool(os.environ.get("OLLAMA_API_KEY")) and bool(os.environ.get("COHERE_API_KEY"))
if not HAVE_KEYS:
    print(
        "⚠  One or both API keys are missing.\n"
        "   This notebook will run using the cached illustrative results\n"
        "   in frozen/rerank_comparison.json so you can still follow along.\n"
        "   Set OLLAMA_API_KEY and COHERE_API_KEY in tutorials/.env to run live."
    )
else:
    print("✓ Both API keys loaded.")
''')

# ============================================================================
# STEP 1 — IMPORTS + VECTOR STORE
# ============================================================================
md(r"""
---
# Step 1 · Imports and vector store

We reuse the same cloud-Ollama setup from Module 05 and the in-memory Qdrant
store from Modules 04–09. Nothing new here — this step just makes the notebook
self-contained so you do not need to import from a sibling folder.
""")

code(r'''
import os
from pathlib import Path

# LangChain + Qdrant
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Model names (same as capstone) ──────────────────────────────────────────
LLM_NAME_OLLAMA       = "nemotron-3-super:cloud"
EMBEDDING_NAME_OLLAMA = "qwen3-embedding:0.6b"
OLLAMA_API_BASE       = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")

chat_llm = ChatOllama(
    model=LLM_NAME_OLLAMA,
    base_url=OLLAMA_API_BASE,
    temperature=0.0,
) if HAVE_KEYS else None

lc_embeddings = OllamaEmbeddings(
    model=EMBEDDING_NAME_OLLAMA,
    base_url=OLLAMA_API_BASE,
) if HAVE_KEYS else None
''')

code(r'''
# ── Load corpus + build in-memory vector store ───────────────────────────────
import json

if HAVE_KEYS:
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
    print(f"Vector store ready: {len(lc_docs)} chunks from {len(raw_docs)} files.")
else:
    print("(keyless mode — skipping vector store build)")

# ── Load golden questions ────────────────────────────────────────────────────
golden = json.loads(Path("golden_questions.json").read_text())
print(f"Loaded {len(golden)} golden questions ({sum(1 for q in golden if q['hop']=='single')} single-hop, "
      f"{sum(1 for q in golden if q['hop']=='multi')} multi-hop).")
''')

# ============================================================================
# STEP 2 — THE RERANK HELPER
# ============================================================================
md(r"""
---
# Step 2 · The `rerank()` helper

A **cross-encoder** sees the query and a candidate passage *together*, letting
it compute token-level attention between them. This is far more expensive than
an embedding lookup (a single dot product), but since we only run it on the
small k=10 shortlist the total latency is still acceptable (~100–300 ms extra).

Cohere's `rerank-v3.5` is the capstone's reranker. We initialise a `ClientV2`
with our key, call `co.rerank()`, and unpack the scored results.

> **⚠ Caution:** `co.rerank()` is a network call that costs tokens. Do not
> call it inside a tight loop without caching results.
""")

code(r'''
import cohere

co = cohere.ClientV2(os.environ["COHERE_API_KEY"]) if HAVE_KEYS else None

def rerank(query: str, docs: list[str], top_n: int = 3) -> list[tuple[str, float]]:
    """Return the top_n most relevant (doc_text, relevance_score) pairs.

    Uses Cohere rerank-v3.5. If docs is empty, returns an empty list.
    relevance_score is in [0, 1]; higher means more relevant.
    """
    if not docs:
        return []
    result = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=docs,
        top_n=top_n,
    )
    return [(docs[r.index], r.relevance_score) for r in result.results]
''')

md(r"""
### Quick smoke-test of `rerank()`

Let us run reranking on a single golden question to see the scores.
""")

code(r'''
import json as _json

q0 = golden[2]["question"]
print(f"Question: {q0}\n")

if HAVE_KEYS:
    # Retrieve k=10 candidates from the vector store
    candidates = [d.page_content for d in base_retriever.invoke(q0)]
    ranked = rerank(q0, candidates, top_n=3)
    for i, (text, score) in enumerate(ranked, 1):
        print(f"[{i}] score={score:.4f}  {text[:100]}…")
else:
    # Use frozen illustrative output
    frozen = _json.loads(open("frozen/rerank_comparison.json").read())
    print("(using cached illustrative result — set keys in tutorials/.env to run live)\n")
    for i, (text, score) in enumerate(
        zip(frozen["reranked_top3"], frozen["reranked_scores"]), 1
    ):
        print(f"[{i}] score={score:.4f}  {text[:100]}…")
''')

# ============================================================================
# STEP 3 — RAG ANSWER WITH USE_RERANK TOGGLE
# ============================================================================
md(r"""
---
# Step 3 · `rag_answer()` with the `use_rerank` toggle

We now wire `rerank()` into the full RAG pipeline. The `use_rerank=False` path
simply takes the first `top_n` cosine-ranked candidates — a clean baseline.
The `use_rerank=True` path (the default) passes the k=10 shortlist through the
cross-encoder first.

This toggle is exactly what Module 12 uses to run the MDD (Model-Data-Design)
loop: build one dataset with reranking off, another with it on, compare metrics.
""")

code(r'''
RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a precise metals-markets tutor. Answer using ONLY the context passages. "
    "If the context does not contain the answer, say you do not know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

def rag_answer(
    question: str, k: int = 10, top_n: int = 3, use_rerank: bool = True
) -> dict:
    """Run the full RAG pipeline and return the response + retrieved contexts.

    Args:
        question:   The user's question.
        k:          Number of candidates to retrieve from the vector store.
        top_n:      Number of passages to pass to the LLM.
        use_rerank: If True, rerank candidates before selecting top_n.

    Returns:
        dict with keys "response" and "retrieved_contexts".
    """
    candidates = [d.page_content for d in base_retriever.invoke(question)]
    if use_rerank:
        contexts = [t for t, _ in rerank(question, candidates, top_n=top_n)]
    else:
        contexts = candidates[:top_n]   # cosine-ranked, no reranker
    block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    response = chat_llm.invoke(
        RAG_PROMPT.format_messages(context=block, question=question)
    ).content.strip()
    return {"response": response, "retrieved_contexts": contexts}
''')

# ============================================================================
# STEP 4 — BEFORE / AFTER COMPARISON
# ============================================================================
md(r"""
---
# Step 4 · Before / after comparison on a golden question

The best way to *see* what the reranker does is to run both pipelines on the
same question and compare the top-3 passages side by side.

We use question index 2: *"Why does gold tend to do well when real interest
rates are low?"* — a single-hop question with a clear answer passage in the
corpus.
""")

code(r'''
import json as _json

q = golden[2]["question"]
ref = golden[2]["reference"]
print(f"Question : {q}")
print(f"Reference: {ref}\n")

if HAVE_KEYS:
    baseline = rag_answer(q, k=10, use_rerank=False)
    reranked  = rag_answer(q, k=10, use_rerank=True)

    print("── BASELINE top-3  (cosine order, no reranker) ─────────────────────────")
    for i, ctx in enumerate(baseline["retrieved_contexts"], 1):
        print(f"  [{i}] {ctx[:120]}…")

    print("\n── RERANKED top-3  (cross-encoder order) ────────────────────────────────")
    for i, ctx in enumerate(reranked["retrieved_contexts"], 1):
        print(f"  [{i}] {ctx[:120]}…")

    print("\n── LLM answers ──────────────────────────────────────────────────────────")
    print(f"  Baseline : {baseline['response'][:200]}")
    print(f"  Reranked : {reranked['response'][:200]}")
else:
    frozen = _json.loads(open("frozen/rerank_comparison.json").read())
    print("(using cached illustrative result — set keys in tutorials/.env to run live)\n")

    print("── BASELINE top-3  (cosine order, no reranker) ─────────────────────────")
    for i, ctx in enumerate(frozen["baseline_top3"], 1):
        print(f"  [{i}] {ctx[:120]}…")

    print("\n── RERANKED top-3  (cross-encoder order) ────────────────────────────────")
    for i, ctx in enumerate(frozen["reranked_top3"], 1):
        score = frozen["reranked_scores"][i - 1]
        print(f"  [{i}] score={score:.4f}  {ctx[:120]}…")
''')

md(r"""
### What to look for

The passage that directly says *"Gold pays no interest or dividend, so when
real rates are low or negative, that cost shrinks and gold often does well"*
should appear as rank **#1** after reranking. In the cosine baseline it may
appear at rank #2 or #3 because other passages about gold and interest rates
have similar embedding distances but do not directly answer the question.

> **⚠ Caution — latency:** The reranked call takes ~200–400 ms longer than the
> baseline because of the extra Cohere API round-trip. In interactive demos this
> is fine; in batch evaluation over hundreds of questions you should budget for
> this cost.
""")

# ============================================================================
# STEP 5 — CONNECTING TO RETRIEVER METRICS
# ============================================================================
md(r"""
---
# Step 5 · Connecting the lift to retriever metrics

In Modules 07–08 you computed Context Precision and Context Recall with RAGAS.
Reranking is specifically designed to lift **Context Precision**: the fraction
of retrieved passages that are actually relevant. Because we select 3 out of 10
more carefully, the 3 that remain are more likely to contain the answer.

The table below shows **illustrative** before/after numbers matching the shape
of what a real run typically produces. Run live to see your actual values.
""")

code(r'''
import json as _json

frozen = _json.loads(open("frozen/rerank_comparison.json").read())
print(frozen["_note"])
print()
print(f"{'Metric':<30} {'Baseline':>10} {'Reranked':>10}")
print("-" * 52)
print(f"{'Context Precision':<30} {frozen['precision_before']:>10.2f} {frozen['precision_after']:>10.2f}")
print(f"{'Context Recall':<30} {frozen['recall_before']:>10.2f} {frozen['recall_after']:>10.2f}")
print()
print("Precision improved because the cross-encoder selected the answering")
print("passage more reliably than cosine similarity alone.")
''')

md(r"""
### The MDD perspective

In the **Measure → Diagnose → Decide** loop (introduced in Module 06):

1. **Measure** — Context Precision was low on certain questions.
2. **Diagnose** — The bi-encoder was ranking topically similar but
   non-answering passages above the true answer passage.
3. **Decide** — Add a cross-encoder reranker to the retrieval step.

This is exactly the intervention Module 12 formalises across the full golden
set with all retriever and generator metrics.

> **⚠ Caution — reranking does not fix recall failures.** If the relevant
> passage is not in the k=10 shortlist at all (Context Recall is low), the
> reranker has nothing to surface. Monitor Recall separately and increase k
> if needed — at the cost of more reranker calls.
""")

# ============================================================================
# RECAP
# ============================================================================
md(r"""
---
# Recap & what's next

### What you built

A **retrieve-wide-then-narrow** pipeline: `base_retriever` fetches k=10
candidates by cosine similarity; `rerank()` calls Cohere rerank-v3.5 to
rescore them with a cross-encoder; `rag_answer()` passes the top_n=3 passages
to the LLM. You measured a concrete Context Precision lift on a golden question.

### Key concepts

| Concept | One-line summary |
| --- | --- |
| Bi-encoder | Independent embeddings → fast but imprecise |
| Cross-encoder | Joint query+doc scoring → precise but expensive |
| Retrieve-wide-then-narrow | k=10 bi-encoder → top_n=3 cross-encoder |
| Cohere rerank-v3.5 | `co.rerank(model=..., query=..., documents=..., top_n=3)` |
| Precision lift | More of the top-3 passages actually contain the answer |

### Cautions to remember

- Reranking adds ~200–400 ms and a second paid API call per query.
- Setting `top_n` too small can drop needed context if the reranker is imperfect.
- Reranking *refines* the shortlist — it cannot surface passages that were never retrieved.

**Next module (11):** *From RAG to Agent* — wraps this reranked pipeline in a
`create_react_agent` (LangGraph ReAct loop) that can call live metal-price tools
and hold multi-turn conversations. You will also measure three new agent-level
metrics: ToolCallAccuracy, TopicAdherence, and AgentGoalAccuracy.
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

OUT = "10_reranking.ipynb"   # <-- this module's notebook filename
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

n_md   = sum(1 for k, _ in CELLS if k == "md")
n_code = sum(1 for k, _ in CELLS if k == "code")
print(f"Wrote {OUT}: {len(CELLS)} cells ({n_md} markdown, {n_code} code)")
