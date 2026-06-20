"""Builds 04_chunking_corpus.ipynb from a list of (type, source) cells.

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
# Module 04 · Chunking & the Corpus

### A hands-on, build-it-yourself module for advanced high school researchers

One sentence can defeat a whole-document embedding — once you split a 2 000-word
file into overlapping 500-character windows, each window gets its own vector and
your searches become dramatically more precise. In this module you will chunk the
**8-file metals-market corpus**, embed every chunk with a local
`sentence-transformers` model (no API key), index the vectors in Qdrant
`:memory:`, and run your first real retrieval queries. This is Module 04 of a
twelve-part track that ends in a full **Agentic RAG Evaluation** capstone.
""")

md(r"""
## 📋 Summary: the one-paragraph version

Long documents are bad inputs for retrieval: a single embedding averages every
idea in the file, diluting specific facts into noise. **Chunking** solves this by
slicing each document into short, overlapping windows — each window becomes its
own vector, so a query about "LBMA auctions" lands near the right passage instead
of competing against everything else in a 2 000-word file. We use LangChain's
`RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)`, the exact
setting the capstone uses, with a local `all-MiniLM-L6-v2` embedder that requires
no API key.
""")

md(r"""
## 🗺️ What you will build, step by step

| Step | What you do | Key tool |
| ---: | --- | --- |
| 0 | Set up the environment | `uv`, `jupyterlab` |
| 1 | Inspect the corpus — count files, preview one document | `pathlib.Path` |
| 2 | Chunk all 8 files with RecursiveCharacterTextSplitter | `langchain-text-splitters` |
| 3 | Embed all chunks with the local sentence-transformers model | `sentence-transformers` |
| 4 | Build a Qdrant `:memory:` collection and upsert all vectors | `qdrant-client` |
| 5 | Run sample queries and inspect top-k results | `qdrant_client.QdrantClient` |
| 6 | Experiment: change chunk_size and observe the effect | your own exploration |

### 🎓 What you will *learn* (the concepts)

- **Why chunking matters**: how whole-document embeddings dilute retrieval quality.
- **RecursiveCharacterTextSplitter**: what `chunk_size` and `chunk_overlap` do, and
  how the splitter chooses where to break.
- **The 8-file metals corpus**: the real knowledge base all later modules evaluate.
- **End-to-end keyless pipeline**: read → split → embed → index → query without
  any cloud API.

### ✅ Prerequisites

- Modules 01–03 of this track (you understand embeddings and cosine search).
- Comfort reading basic Python.
- Curiosity.
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

> **No API keys needed.** This module is entirely keyless — the local
> `sentence-transformers` model downloads from HuggingFace once (~90 MB) and
> then works offline. Qdrant runs fully in-process.
""")

# ============================================================================
# STEP 1 — INSPECT THE CORPUS
# ============================================================================
md(r"""
---
# Step 1 · Inspect the corpus

The `corpus/` folder next to this notebook contains 8 Markdown files — the
same knowledge base the capstone uses to answer questions about precious-metals
markets. Let's load them and take a look before splitting anything.
""")

code(r'''
from pathlib import Path

CORPUS_DIR = Path("corpus")
corpus_paths = sorted(CORPUS_DIR.glob("*.md"))

print(f"Found {len(corpus_paths)} corpus files:\n")
for p in corpus_paths:
    word_count = len(p.read_text().split())
    print(f"  {p.name:<45} {word_count:>5} words")
''')

md(r"""
Preview the first 500 characters of one file so we know what the text looks like
before splitting.
""")

code(r'''
sample_path = corpus_paths[0]
sample_text = sample_path.read_text()

print(f"=== {sample_path.name} (first 500 chars) ===\n")
print(sample_text[:500])
print("\n[…truncated…]")
''')

# ============================================================================
# STEP 2 — CHUNKING
# ============================================================================
md(r"""
---
# Step 2 · Chunk with RecursiveCharacterTextSplitter

`RecursiveCharacterTextSplitter` breaks text on the most natural boundary it
can find — double newlines first (paragraph breaks), then single newlines, then
spaces, then characters as a last resort. This keeps chunks coherent: they
almost always end at a sentence boundary.

Two parameters control the split:

- **`chunk_size=500`** — maximum characters per chunk (≈ one dense paragraph).
- **`chunk_overlap=60`** — characters shared between consecutive chunks; prevents
  a key sentence from being cut in half between two windows.

> ⚠ **Caution — size matters a lot.**
> `chunk_size=50` gives fragments with no context; `chunk_size=2000` buries
> specific facts in a whole-page average. 500 / 60 is the capstone's tuned
> default. Always inspect a few chunks before committing to a setting.
""")

code(r'''
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=60,
)

# Load all corpus files, then split each one
raw_docs = [
    {"source": p.name, "page_content": p.read_text()}
    for p in corpus_paths
]

# chunks is a list of (source_filename, chunk_text) tuples
chunks: list[tuple[str, str]] = [
    (doc["source"], piece)
    for doc in raw_docs
    for piece in splitter.split_text(doc["page_content"])
]

print(f"Total chunks: {len(chunks)}")
print(f"Average chunk length: {sum(len(t) for _, t in chunks) / len(chunks):.0f} chars")
''')

md(r"""
Let's peek at a few chunks to confirm they look like coherent passages.
""")

code(r'''
import random

random.seed(42)
sample_indices = random.sample(range(len(chunks)), k=3)

for idx in sample_indices:
    source, text = chunks[idx]
    print(f"--- Chunk #{idx}  (source: {source}) ---")
    print(text)
    print()
''')

md(r"""
Notice that consecutive chunks share their last ~60 characters with the next
chunk's first ~60 characters (the `chunk_overlap`). This overlap is what
prevents a sentence straddling a boundary from being invisible to retrieval.
""")

code(r'''
# Demonstrate overlap between two consecutive chunks from the same document
first_file_chunks = [t for s, t in chunks if s == corpus_paths[0].name]

if len(first_file_chunks) >= 2:
    c0 = first_file_chunks[0]
    c1 = first_file_chunks[1]
    print(f"Chunk 0 ends with:    …{c0[-70:]!r}")
    print(f"Chunk 1 starts with:  {c1[:70]!r}…")
    print()
    # Compute the actual overlap
    overlap_len = 0
    for n in range(min(len(c0), len(c1)), 0, -1):
        if c0[-n:] == c1[:n]:
            overlap_len = n
            break
    print(f"Exact overlap found: {overlap_len} characters")
''')

# ============================================================================
# STEP 3 — EMBED
# ============================================================================
md(r"""
---
# Step 3 · Embed all chunks locally

We use `all-MiniLM-L6-v2` from `sentence-transformers` — the same 384-dimension
model from Modules 02 and 03. It downloads from HuggingFace the first time
(~90 MB) and then runs entirely offline.

Encoding all chunks at once in a single `.encode()` call is much faster than
encoding them one by one — the library batches them internally.
""")

code(r'''
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

texts = [text for _, text in chunks]
vectors: np.ndarray = embedder.encode(texts, show_progress_bar=True)

print(f"\nEmbedded {len(vectors)} chunks")
print(f"Vector shape: {vectors.shape}  (chunks × dimensions)")
print(f"Each vector is {vectors.shape[1]}-dimensional (all-MiniLM-L6-v2 default)")
''')

# ============================================================================
# STEP 4 — INDEX IN QDRANT
# ============================================================================
md(r"""
---
# Step 4 · Index in Qdrant `:memory:`

Qdrant is a vector database. When you pass `":memory:"` as the location it
runs entirely in RAM — no server to start, no files on disk. Perfect for
learning and experimentation.

We store each vector alongside a small **payload**: the chunk text and its
source filename. Payloads let us inspect which document a result came from.
""")

code(r'''
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION = "metals_kb"
VECTOR_SIZE = vectors.shape[1]   # 384

client = QdrantClient(":memory:")
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)

points = [
    PointStruct(
        id=i,
        vector=v.tolist(),
        payload={"text": text, "source": source},
    )
    for i, ((source, text), v) in enumerate(zip(chunks, vectors))
]
client.upsert(collection_name=COLLECTION, points=points)

info = client.get_collection(COLLECTION)
print(f"Collection '{COLLECTION}' ready.")
print(f"  Points indexed: {info.points_count}")
print(f"  Vector size:    {info.config.params.vectors.size}")
print(f"  Distance:       {info.config.params.vectors.distance}")
''')

# ============================================================================
# STEP 5 — QUERY
# ============================================================================
md(r"""
---
# Step 5 · Retrieve top-k chunks for a query

Retrieval works the same way as in Module 03: encode the query into a vector,
then ask Qdrant for the nearest `k` neighbours by cosine similarity.

The difference now is that each result is a focused *chunk* — roughly one
paragraph — rather than a whole document. That precision is exactly what lets a
later LLM answer questions accurately.
""")

code(r'''
def retrieve(query: str, k: int = 5) -> list[dict]:
    """Return the top-k matching chunks with source and score."""
    q_vec = embedder.encode(query).tolist()
    hits = client.search(
        collection_name=COLLECTION,
        query_vector=q_vec,
        limit=k,
        with_payload=True,
    )
    return [
        {
            "score": round(hit.score, 4),
            "source": hit.payload["source"],
            "text": hit.payload["text"],
        }
        for hit in hits
    ]
''')

code(r'''
# --- Query 1: LBMA auction mechanics ---
query_1 = "How is the LBMA gold price set?"
results_1 = retrieve(query_1, k=5)

print(f"Query: {query_1!r}\n")
for i, r in enumerate(results_1, 1):
    print(f"[{i}] score={r['score']}  source={r['source']}")
    print(f"    {r['text'][:150].replace(chr(10), ' ')}…")
    print()
''')

code(r'''
# --- Query 2: Silver industrial demand ---
query_2 = "What role does silver play in solar panels?"
results_2 = retrieve(query_2, k=5)

print(f"Query: {query_2!r}\n")
for i, r in enumerate(results_2, 1):
    print(f"[{i}] score={r['score']}  source={r['source']}")
    print(f"    {r['text'][:150].replace(chr(10), ' ')}…")
    print()
''')

code(r'''
# --- Query 3: Portfolio / risk ---
query_3 = "How does gold help diversify an investment portfolio?"
results_3 = retrieve(query_3, k=5)

print(f"Query: {query_3!r}\n")
for i, r in enumerate(results_3, 1):
    print(f"[{i}] score={r['score']}  source={r['source']}")
    print(f"    {r['text'][:150].replace(chr(10), ' ')}…")
    print()
''')

# ============================================================================
# STEP 6 — EXPERIMENT: VARY CHUNK SIZE
# ============================================================================
md(r"""
---
# Step 6 · Experiment — how does chunk_size change results?

Let's rebuild the index with a *larger* chunk size and compare. This is the
kind of quick experiment you would run before committing to a chunking strategy
in a real project.

> ⚠ **Caution — there is no universal best chunk size.**
> The right value depends on your embedding model's ideal input length, your
> documents' structure, and your retrieval task. Always run an evaluation (see
> Module 07) rather than trusting intuition alone.
""")

code(r'''
def build_index(chunk_size: int, chunk_overlap: int) -> QdrantClient:
    """Build a fresh in-memory Qdrant index with the given split parameters."""
    sp = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    ch = [
        (doc["source"], piece)
        for doc in raw_docs
        for piece in sp.split_text(doc["page_content"])
    ]
    vecs = embedder.encode([t for _, t in ch])
    cl = QdrantClient(":memory:")
    cl.create_collection(
        "exp",
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    cl.upsert("exp", points=[
        PointStruct(id=i, vector=v.tolist(),
                    payload={"text": t, "source": s})
        for i, ((s, t), v) in enumerate(zip(ch, vecs))
    ])
    return cl, len(ch)


print("Rebuilding index with chunk_size=1000, chunk_overlap=100 …")
client_large, n_large = build_index(1000, 100)
print(f"  Large chunks: {n_large} total (was {len(chunks)} at 500/60)")
''')

code(r'''
# Compare retrieval for the same query under both chunk sizes
test_query = "How is the LBMA gold price set?"
q_vec = embedder.encode(test_query).tolist()

hits_default = client.search(COLLECTION, query_vector=q_vec, limit=3, with_payload=True)
hits_large   = client_large.search("exp",     query_vector=q_vec, limit=3, with_payload=True)

print(f"Query: {test_query!r}\n")
print("=== Default (chunk_size=500) ===")
for h in hits_default:
    print(f"  score={h.score:.4f}  len={len(h.payload['text'])} chars")
    print(f"  {h.payload['text'][:120].replace(chr(10), ' ')}…\n")

print("=== Large (chunk_size=1000) ===")
for h in hits_large:
    print(f"  score={h.score:.4f}  len={len(h.payload['text'])} chars")
    print(f"  {h.payload['text'][:120].replace(chr(10), ' ')}…\n")
''')

md(r"""
Larger chunks tend to produce *lower* cosine scores because the embedding
averages more content — the query signal is diluted. Smaller chunks have
higher scores but may lack the surrounding context an LLM needs to give
a complete answer. The 500 / 60 setting balances both concerns.
""")

# ============================================================================
# RECAP
# ============================================================================
md(r"""
---
# Recap & what's next

You built a complete keyless retrieval pipeline over real documents:

1. **Loaded** 8 metals-market Markdown files with a `Path` glob.
2. **Chunked** them with `RecursiveCharacterTextSplitter(500, 60)` — natural
   paragraph-level windows with 60-character overlap.
3. **Embedded** every chunk locally with `all-MiniLM-L6-v2` (384-dim, no API key).
4. **Indexed** the vectors in Qdrant `:memory:` with source payloads.
5. **Retrieved** focused passages for three different queries.
6. **Experimented** with a larger chunk size to see precision vs. context trade-offs.

**Next module (05 — Stack migration + first real RAG):** you will swap the local
`sentence-transformers` embedder for cloud Ollama via LiteLLM, load your
`OLLAMA_API_KEY` from the shared `tutorials/.env`, and generate the first full
RAG answer — query → retrieve chunks → stuff prompt → LLM response.
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

OUT = "04_chunking_corpus.ipynb"
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

n_md   = sum(1 for k, _ in CELLS if k == "md")
n_code = sum(1 for k, _ in CELLS if k == "code")
print(f"Wrote {OUT}: {len(CELLS)} cells ({n_md} markdown, {n_code} code)")
