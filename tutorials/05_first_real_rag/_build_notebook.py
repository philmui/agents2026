"""Builds 05_first_real_rag.ipynb from a list of (type, source) cells.

Run:  python3 _build_notebook.py
This keeps the notebook JSON well-formed and easy to regenerate. Edit THIS file,
never the generated .ipynb.

Pattern (identical across every module in tutorials/):
  - md(r'''...''')   adds a markdown cell
  - code(r'''...''') adds a code cell
  - the EMIT block at the bottom writes the .ipynb. Change only OUT below.
"""
import json

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("md", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ============================================================================
# TITLE + SUMMARY
# ============================================================================
md(r"""
# Module 05 · Stack Migration + First Real RAG

### A hands-on, build-it-yourself module for advanced high school researchers

![RAG architecture](slides/assets/01_agentic_rag_architecture.svg)

In the previous module you chunked the eight-file precious-metals corpus and
stored it in Qdrant using the free local `sentence-transformers` embedder. This
module is **the pivot**: you swap that local embedder for cloud Ollama, load API
keys from the shared `tutorials/.env` file, and then wire everything together
into your first complete **prompt-stuffing RAG answer** — retrieve relevant
passages → stuff them into a prompt → generate a grounded response. Every module
from 06 onward builds directly on this stack.

This is Module 05 of a twelve-part track that ends in a full
**Agentic RAG Evaluation** capstone in `topics/06_rag_eval/`.
""")

md(r"""
## 📋 Summary: the one-paragraph version

Module 05 has two jobs. First, it migrates the stack from a free local
embedding model to cloud Ollama embeddings and a cloud Ollama chat model —
the same models used throughout the capstone. Second, it builds `rag_answer()`:
a function that retrieves the top-k passages from Qdrant, selects the best
`top_n` (no reranking yet — that arrives in Module 10), stuffs them into a
carefully worded prompt, and asks the chat model to answer **only from the
provided context**. The result is a live, end-to-end RAG system you can query
with any metals-markets question.
""")

md(r"""
## 🗺️ What you will build, step by step

| Step | What you do | Key tool |
| ---: | --- | --- |
| 0.1 | Install dependencies | `uv sync` |
| 0.2 | Load the shared API key file | `load_dotenv(find_dotenv())` |
| 1 | Understand **why** we migrate the embedder | conceptual framing |
| 2 | Build the cloud LLM + embedder objects | `ChatOllama`, `OllamaEmbeddings` |
| 3 | Chunk the corpus and index it in Qdrant | `RecursiveCharacterTextSplitter`, `QdrantVectorStore` |
| 4 | Write `rag_answer()` — prompt-stuffing RAG | `ChatPromptTemplate` |
| 5 | Run a sample question end-to-end | live or frozen result |

### 🎓 What you will *learn* (the concepts)

- Why embedding quality matters and why we switch from local to cloud Ollama
- How the shared `tutorials/.env` pattern works with `find_dotenv()`
- What prompt-stuffing RAG is and how the `RAG_PROMPT` template constrains the LLM
- Why retrieval quality directly limits answer quality

### ✅ Prerequisites

- Module 04: chunking + the 8-file metals corpus + Qdrant in-memory
- Comfort reading basic Python — no ML background required
- An `OLLAMA_API_KEY` in `tutorials/.env` (or follow along with frozen results)
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
**from this module's folder** (`tutorials/05_first_real_rag/`), using
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync            # reads pyproject.toml, creates .venv/, installs everything
uv run jupyter lab # launch Jupyter inside that environment
```

When the notebook opens, pick the kernel **`Python 3 (ipykernel)`** (top-right
kernel picker). That kernel uses the interpreter from `.venv`, so every `import`
below resolves against what `uv sync` installed.

New packages added in this module (not present in M01–M04):

| Package | Purpose |
| --- | --- |
| `langchain-ollama` | `ChatOllama` chat model + `OllamaEmbeddings` |
| `litellm` | LiteLLM provider bridge (used by RAGAS in M06+) |
| `langchain-qdrant` | `QdrantVectorStore` LangChain integration |
| `qdrant-client` | Qdrant `:memory:` backend (no server needed) |
""")

# --- Step 0.2: shared .env + HAVE_KEYS guard ---
md(r"""
## 0.2 Provide your API key (shared `tutorials/.env`)

All twelve modules read their keys from a **single** `.env` file in the
`tutorials/` folder — the *parent* of this module. Create it once:

```
tutorials/.env
──────────────
OLLAMA_API_KEY=<your cloud Ollama key>
```

The call `load_dotenv(find_dotenv())` below walks **up** the directory tree from
this notebook and finds that shared file automatically. You never copy keys into
each module, and `.env` is already gitignored at `tutorials/.gitignore` — it
will never be committed.

**No key?** The notebook detects a missing key and falls back to the small
illustrative results in `frozen/rag_answer.json`, so you can still follow along.

> ⚠️ **Cost note:** each call to `rag_answer()` sends one embedding request (for
> the question) and one chat-completion request to cloud Ollama. Indexing the
> corpus sends ~50–80 embedding calls. These are small requests, but they do
> consume quota. If you are on a free-tier key, run live only for the final
> demo cell.
""")

code(r'''
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())          # resolves to tutorials/.env automatically

HAVE_KEYS = bool(os.environ.get("OLLAMA_API_KEY"))

if HAVE_KEYS:
    print("OLLAMA_API_KEY found — live calls enabled.")
else:
    print(
        "No OLLAMA_API_KEY found in tutorials/.env.\n"
        "This notebook will use the cached results in frozen/ so you can still\n"
        "follow along. Add your key to tutorials/.env to run live."
    )
''')

# ============================================================================
# STEP 1 — WHY MIGRATE?
# ============================================================================
md(r"""
---
# Step 1 · Why migrate away from the local embedder?

In Modules 02–04 you used `sentence-transformers` (`all-MiniLM-L6-v2`, 384
dimensions) — a free model that runs entirely offline. It is excellent for
prototyping. So why swap it out?

**Two reasons:**

1. **Embedding space consistency.** Cosine similarity is only meaningful when
   the query vector and the corpus vectors live in the *same embedding space*.
   The model that encodes your documents at *index time* must be identical to the
   model that encodes the user's question at *query time*. RAGAS (the evaluation
   library arriving in Module 06) uses its own embedding model to score retrieval
   quality — and that model must also match. Using cloud Ollama's
   `qwen3-embedding:0.6b` everywhere guarantees this.

2. **Representation quality.** Larger, domain-trained models produce better
   representations for financial and commodity text. Switching now means every
   downstream evaluation metric is measuring a system that actually resembles the
   capstone, not an approximation of it.

The diagram below shows the full RAG architecture — Module 05 completes the
**Retrieve → Augment → Generate** loop for the first time with the real stack.
""")

code(r'''
# No executable code in Step 1 — this is a conceptual framing step.
# The migration happens in Step 2 (building the LLM and embedder objects).
print("Step 1: conceptual framing complete.")
print("New stack: ChatOllama (generator) + OllamaEmbeddings (retrieval embedder)")
print("Shared key file: tutorials/.env  →  loaded via find_dotenv()")
''')

# ============================================================================
# STEP 2 — BUILD LLM + EMBEDDER
# ============================================================================
md(r"""
---
# Step 2 · Build the cloud Ollama LLM and embedder

`ChatOllama` wraps the cloud Ollama chat endpoint.  `OllamaEmbeddings` wraps the
embedding endpoint. Both are initialized with the same `base_url`, read from the
environment variable `OLLAMA_API_BASE` (defaulting to the local daemon URL if
not set).

Model names:
- **Generator:** `nemotron-3-super:cloud` — a capable instruction-following model
- **Embedder:** `qwen3-embedding:0.6b` — matches the capstone's embedding space

> ⚠️ **Caution — model names are cloud-specific.** The `:cloud` and `:0.6b`
> suffixes route requests through the cloud Ollama relay. If you are running a
> local Ollama daemon instead, substitute the model names that daemon serves.
""")

code(r'''
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Defaults to local daemon; override by setting OLLAMA_API_BASE in tutorials/.env
os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11434")

LLM_NAME_OLLAMA       = "nemotron-3-super:cloud"
EMBEDDING_NAME_OLLAMA = "qwen3-embedding:0.6b"

if HAVE_KEYS:
    chat_llm = ChatOllama(
        model=LLM_NAME_OLLAMA,
        base_url=os.environ["OLLAMA_API_BASE"],
        temperature=0.0,
    )
    lc_embeddings = OllamaEmbeddings(
        model=EMBEDDING_NAME_OLLAMA,
        base_url=os.environ["OLLAMA_API_BASE"],
    )
    print(f"LLM ready:        {LLM_NAME_OLLAMA}")
    print(f"Embedder ready:   {EMBEDDING_NAME_OLLAMA}")
    print(f"Ollama base URL:  {os.environ['OLLAMA_API_BASE']}")
else:
    chat_llm       = None
    lc_embeddings  = None
    print("(skipping model init — no API key; will use frozen results in Step 5)")
''')

# ============================================================================
# STEP 3 — CHUNK CORPUS + BUILD VECTOR STORE
# ============================================================================
md(r"""
---
# Step 3 · Chunk the corpus and build the vector store

This step reuses the chunking logic from Module 04. The 8 `.md` files in
`corpus/` are loaded, split into 500-character chunks with 60-character overlap,
and indexed in a Qdrant `:memory:` vector store using the cloud embedder.

Parameters to know:
- `chunk_size=500` — maximum characters per chunk
- `chunk_overlap=60` — how many characters the next chunk re-reads from the end
  of the previous one (prevents cutting a thought mid-sentence)
- `k=10` — how many candidate chunks the retriever returns per query
- `location=":memory:"` — Qdrant runs in-process; no server, no persistence

> ⚠️ **Caution — indexing costs quota.** Each chunk is encoded by
> `OllamaEmbeddings` in a real API call. With ~50–80 chunks across 8 files this
> is small, but it does consume your key's quota. If `HAVE_KEYS` is False, this
> step is skipped and retrieval falls back to the frozen output in Step 5.
""")

code(r'''
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

if HAVE_KEYS:
    # Load the 8 corpus files
    raw_docs = [
        {"source": p.name, "page_content": p.read_text()}
        for p in sorted(Path("corpus").glob("*.md"))
    ]
    print(f"Loaded {len(raw_docs)} corpus files.")

    # Chunk each file
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    lc_docs = [
        Document(page_content=piece, metadata={"source": d["source"]})
        for d in raw_docs
        for piece in splitter.split_text(d["page_content"])
    ]
    print(f"Total chunks: {len(lc_docs)}")

    # Build in-memory Qdrant vector store (one embedding call per chunk)
    vector_store = QdrantVectorStore.from_documents(
        lc_docs,
        embedding=lc_embeddings,
        location=":memory:",
        collection_name="metals_kb",
    )
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    print("Vector store built.  Ready to retrieve.")
else:
    vector_store   = None
    base_retriever = None
    print("(skipping index build — no API key; retrieval bypassed in Step 5)")
''')

# ============================================================================
# STEP 4 — rag_answer()
# ============================================================================
md(r"""
---
# Step 4 · Write `rag_answer()` — the prompt-stuffing RAG function

The function has three stages:

1. **Retrieve** — ask the vector store for the `k` most similar chunks.
2. **Select** — take the first `top_n` candidates (no reranking yet; Module 10
   replaces this line with a Cohere rerank call).
3. **Generate** — inject those passages into `RAG_PROMPT` and ask the LLM to
   answer **only from the provided context**.

The `RAG_PROMPT` is critical: it explicitly constrains the LLM to cite only the
provided context. If the context does not contain the answer, the model should
say so. This is what separates RAG from pure generation and is the basis for the
*Faithfulness* metric in Module 09.

> ⚠️ **Grounding ≠ correctness.** Telling the model to stay grounded reduces
> hallucination but does not eliminate it. If the retrieved passages are
> inaccurate, or if the model paraphrases ambiguously, the answer can still be
> wrong. Module 09 measures this gap with Faithfulness and Factual Correctness.
""")

code(r'''
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a precise metals-markets tutor. Answer using ONLY the context passages. "
    "If the context does not contain the answer, say you do not know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


def rag_answer(question: str, k: int = 10, top_n: int = 3) -> dict:
    """Retrieve top-k chunks, select top_n (no rerank), generate a grounded answer.

    Returns:
        dict with keys "response" (str) and "retrieved_contexts" (list[str]).
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    candidates = [d.page_content for d in retriever.invoke(question)]
    # Module 10 replaces this slice with a Cohere rerank call:
    contexts = candidates[:top_n]
    block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    response = chat_llm.invoke(
        RAG_PROMPT.format_messages(context=block, question=question)
    ).content.strip()
    return {"response": response, "retrieved_contexts": contexts}


print("rag_answer() defined.  k=10 candidates → top_n=3 contexts → LLM.")
''')

# ============================================================================
# STEP 5 — LIVE DEMO (with frozen fallback)
# ============================================================================
md(r"""
---
# Step 5 · Run a sample question end-to-end

Try the RAG system on a metals-markets question.  If your `OLLAMA_API_KEY` is
present the call goes live; otherwise the notebook loads the illustrative result
from `frozen/rag_answer.json`.

The frozen output was hand-authored to show the **realistic shape** of a live
answer — plausible content, correct structure, clearly labelled as illustrative.
It is not a real LLM response.
""")

code(r'''
import json

SAMPLE_QUESTION = "What factors drive gold prices higher?"

if HAVE_KEYS:
    result = rag_answer(SAMPLE_QUESTION)
else:
    with open("frozen/rag_answer.json") as fh:
        result = json.load(fh)
    print("(using cached illustrative result — set OLLAMA_API_KEY in tutorials/.env to run live)\n")

print("Question:", SAMPLE_QUESTION)
print()
print("Answer:")
print(result["response"])
print()
print(f"Retrieved contexts ({len(result['retrieved_contexts'])} passages):")
for i, ctx in enumerate(result["retrieved_contexts"], 1):
    preview = ctx[:120].replace("\n", " ")
    print(f"  [{i}] {preview}…")
''')

md(r"""
### What to notice in the output

- The answer should cite information that appears *verbatim or paraphrased* from
  the retrieved passages — that is grounding.
- If you ask a question outside the corpus (e.g., "What is the GDP of France?"),
  a well-behaved RAG system should say it does not know rather than hallucinate.
- The three retrieved passages are the *raw* top-3 from the retriever — not
  ranked by relevance to the specific question.  Module 10 fixes this.
""")

# ============================================================================
# RECAP
# ============================================================================
md(r"""
---
# Recap & what's next

### What you built

- A cloud Ollama **LLM** (`ChatOllama`) and **embedder** (`OllamaEmbeddings`)
  that mirror the capstone stack.
- A Qdrant `:memory:` vector store indexed with cloud embeddings of the 8-file
  precious-metals corpus.
- `rag_answer()` — your first complete, end-to-end RAG function: retrieve →
  select top-3 → stuff prompt → generate grounded answer.

### The three ideas to keep

1. Embedding consistency: the same model must encode corpus and query.
2. The shared `tutorials/.env` + `find_dotenv()` pattern scales to all 12 modules.
3. Grounding constrains the LLM but does not guarantee correctness — you need
   metrics to measure how well it actually works.

**Next module (06):** *Why Evaluate? + RAGAS Setup* — introduces the
Metrics-Driven Development loop, `SingleTurnSample`, `EvaluationDataset`, and
the first RAGAS evaluation run on the system you just built.
""")

# ============================================================================
# EMIT NOTEBOOK  (do not change below except OUT)
# ============================================================================
def to_cell(kind: str, src: str) -> dict:
    lines = src.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
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

OUT = "05_first_real_rag.ipynb"
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

n_md = sum(1 for k, _ in CELLS if k == "md")
n_code = sum(1 for k, _ in CELLS if k == "code")
print(f"Wrote {OUT}: {len(CELLS)} cells ({n_md} markdown, {n_code} code)")
