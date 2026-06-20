"""Builds 03_similarity_search.ipynb from a list of (type, source) cells.

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
def md(text):   CELLS.append(("md", text.strip("\n")))
def code(text): CELLS.append(("code", text.strip("\n")))


# ============================================================================
# TITLE + SUMMARY
# ============================================================================
md(r"""
# Module 03 · Similarity & Vector Search

### A hands-on, build-it-yourself module for advanced high school researchers

In Module 02 you learned how to turn a sentence into a 384-dimensional
numeric vector (an *embedding*). Now you'll answer the key follow-up question:
**given a query vector, which stored vectors are most similar?** You'll
implement cosine similarity by hand, then delegate the heavy lifting to
**Qdrant** — a vector database that indexes millions of vectors and retrieves
the top-k nearest neighbours in milliseconds. No API key is required; the
local `all-MiniLM-L6-v2` model plus Qdrant's `:memory:` mode run entirely
on your laptop.

This is Module 03 of a twelve-part track that ends in a full
**Agentic RAG Evaluation** capstone.
""")

md(r"""
## Summary: the one-paragraph version

**Cosine similarity** measures the *angle* between two vectors, not their
Euclidean distance. Two sentences that share meaning point in almost the same
direction in embedding space, so their cosine score is close to 1 — regardless
of how long or short each sentence is. **Qdrant** stores those vectors in a
*collection*, builds an HNSW index for fast approximate search, and answers
*top-k* queries in sub-millisecond time. Together, cosine similarity and a
vector store give you the retrieval backbone that every RAG pipeline depends on.
""")

md(r"""
## What you will build, step by step

| Step | What you do | Key tool |
| ---: | --- | --- |
| 0 | Set up the environment | `uv`, `jupyterlab` |
| 1 | Load the model; inspect one embedding | `SentenceTransformer` |
| 2 | Implement cosine similarity by hand; explore pairwise scores | `numpy` |
| 3 | Create a Qdrant `:memory:` collection | `QdrantClient`, `VectorParams` |
| 4 | Encode the toy corpus; upsert `PointStruct`s | `model.encode`, `upsert` |
| 5 | Query "What is a troy ounce?" → top-3 results | `query_points` |
| 6 | Experiment: vary query, add sentences, change `limit` | — |

### What you will *learn* (the concepts)

- **Cosine similarity**: measures angle between vectors; insensitive to vector magnitude
- **Top-k retrieval**: returning the k closest vectors to a query vector
- **Qdrant `:memory:`**: an in-process vector database — no server, no disk

### Prerequisites

- Module 02 (embeddings) or comfort with the idea that text → numeric vector
- Basic Python (lists, loops, functions)
- Curiosity about how search engines find relevant passages
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

> **No API keys needed.** `sentence-transformers` downloads `all-MiniLM-L6-v2`
> once (~90 MB) on first use and caches it locally. After that the model runs
> completely offline.
""")


# ============================================================================
# STEP 1 — Load model; inspect one embedding
# ============================================================================
md(r"""
---
# Step 1 · Load the model and inspect one embedding

We use the same 384-dimensional model from Module 02.
""")

code(r'''
from sentence_transformers import SentenceTransformer
import numpy as np

# Downloads once (~90 MB), then cached locally — no internet needed after that.
model = SentenceTransformer("all-MiniLM-L6-v2")

sample = model.encode("What is a troy ounce?")
print(f"Vector shape : {sample.shape}")          # (384,)
print(f"First 8 dims : {sample[:8].round(4)}")
print(f"L2 norm      : {np.linalg.norm(sample):.4f}")
''')

md(r"""
> **Observe:** The L2 norm is close to 1 because `sentence-transformers`
> normalises its output by default. Cosine similarity between two
> unit-normalised vectors simplifies to a plain dot product — but we'll
> write the full formula anyway so the math is explicit.
""")


# ============================================================================
# STEP 2 — Cosine similarity by hand
# ============================================================================
md(r"""
---
# Step 2 · Cosine similarity by hand

### The formula

$$
\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}
$$

The numerator is the **dot product** (sum of element-wise products).
The denominator is the product of each vector's **L2 norm** (its length).
Dividing removes the effect of vector magnitude, leaving a score in \[−1, 1\].
""")

code(r'''
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Score in [-1, 1]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# Encode three sentences with very different meanings
sent_troy   = model.encode("A troy ounce equals 31.1 grams.")
sent_gold   = model.encode("Gold is priced in troy ounces.")
sent_ml     = model.encode("Machine learning optimises loss functions.")
query       = model.encode("What is a troy ounce?")

print("Pairwise cosine scores")
print(f"  query ↔ troy   : {cosine(query, sent_troy):.4f}  (expect HIGH)")
print(f"  query ↔ gold   : {cosine(query, sent_gold):.4f}  (expect MEDIUM)")
print(f"  query ↔ ML     : {cosine(query, sent_ml):.4f}   (expect LOW)")
print()
print(f"  troy  ↔ gold   : {cosine(sent_troy, sent_gold):.4f}")
print(f"  troy  ↔ ML     : {cosine(sent_troy, sent_ml):.4f}")
''')

md(r"""
> **Key insight:** Sentences about troy ounces and gold cluster together in
> embedding space; the machine-learning sentence is far away. That's cosine
> similarity at work — it captures *semantic proximity*, not keyword overlap.

> ⚠ **Caution:** In practice you never call `cosine()` manually in a RAG
> pipeline. Qdrant runs it over all stored vectors in parallel, using an HNSW
> index. The manual version here is purely for building intuition.
""")


# ============================================================================
# STEP 3 — Create a Qdrant :memory: collection
# ============================================================================
md(r"""
---
# Step 3 · Create a Qdrant `:memory:` collection

A Qdrant **collection** is a named table of vectors. Every vector in a
collection must have the same dimension. The `Distance.COSINE` setting tells
Qdrant to normalise vectors on insert and use cosine distance internally.
""")

code(r'''
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# :memory: runs entirely inside this Python process — no server, no files.
client = QdrantClient(":memory:")

COLLECTION = "demo"
DIM = 384  # all-MiniLM-L6-v2 output dimension

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
)

info = client.get_collection(COLLECTION)
print(f"Collection '{COLLECTION}' created.")
print(f"  vectors_count : {info.vectors_count}")
print(f"  status        : {info.status}")
''')


# ============================================================================
# STEP 4 — Encode toy corpus and upsert
# ============================================================================
md(r"""
---
# Step 4 · Encode the toy corpus and upsert points

A Qdrant **PointStruct** bundles:
- `id` — a unique integer (or UUID) for this point
- `vector` — a plain Python list of floats (the embedding)
- `payload` — any JSON-serialisable metadata (we store the original text)
""")

code(r'''
from qdrant_client.models import PointStruct

# Small, hand-picked toy corpus (5 sentences, 3 about precious metals)
SENTENCES = [
    "Gold is priced in troy ounces.",
    "Silver has significant industrial demand.",
    "A troy ounce equals 31.1 grams.",
    "Platinum is rarer than gold.",
    "Machine learning optimises loss functions.",
]

# Encode all sentences in one batch call (faster than one-by-one)
vectors = model.encode(SENTENCES)  # shape: (5, 384)

points = [
    PointStruct(id=i, vector=vec.tolist(), payload={"text": s})
    for i, (vec, s) in enumerate(zip(vectors, SENTENCES))
]

client.upsert(collection_name=COLLECTION, points=points)

info = client.get_collection(COLLECTION)
print(f"Points in collection: {info.vectors_count}")
''')

md(r"""
> **Tip:** `model.encode(list_of_strings)` returns a 2-D NumPy array.
> Qdrant expects a plain Python list, so we call `.tolist()` on each row.
""")


# ============================================================================
# STEP 5 — Query: top-k retrieval
# ============================================================================
md(r"""
---
# Step 5 · Query "What is a troy ounce?" and retrieve top-3 results

`client.query_points` encodes the similarity search in one call:
1. Qdrant computes the cosine between the query vector and every stored vector.
2. It returns the `limit` points with the *highest* cosine scores.

Scores are in \[0, 1\] (Qdrant rescales cosine from \[−1,1\] to \[0,1\] for
the COSINE distance mode).
""")

code(r'''
QUERY = "What is a troy ounce?"
q_vec = model.encode(QUERY).tolist()

results = client.query_points(
    collection_name=COLLECTION,
    query=q_vec,
    limit=3,
).points

print(f'Query: "{QUERY}"\n')
print(f"{'Rank':<6} {'Score':<8} Text")
print("-" * 60)
for rank, hit in enumerate(results, 1):
    print(f"#{rank:<5} {hit.score:.4f}   {hit.payload['text']}")
''')

md(r"""
> **Expected output (approximate — exact scores depend on model version):**
>
> ```
> #1     0.8831   A troy ounce equals 31.1 grams.
> #2     0.7245   Gold is priced in troy ounces.
> #3     0.5102   Silver has significant industrial demand.
> ```
>
> Notice that the ML sentence never appears in top-3 — its direction in
> embedding space is orthogonal to the query. That's cosine similarity working
> exactly as intended.

> ⚠ **Caution:** Result #2 ("Gold is priced in troy ounces") scores highly
> because it is *topically close* to the query — but it doesn't actually
> *define* a troy ounce. **Cosine similarity measures relevance, not factual
> accuracy.** The RAG generator in Module 05 must handle this; the evaluators
> in Modules 07–09 will measure how well it does.
""")


# ============================================================================
# STEP 6 — Experiment
# ============================================================================
md(r"""
---
# Step 6 · Experiment: vary query, sentences, and k

Now it's your turn. Try each of the suggestions below and observe how the
ranked results change.
""")

code(r'''
# --- Experiment 6a: try a different query ---
QUERY_2 = "What metals are used in electronics?"
q2_vec = model.encode(QUERY_2).tolist()

hits2 = client.query_points(collection_name=COLLECTION, query=q2_vec, limit=3).points
print(f'Query: "{QUERY_2}"\n')
for rank, hit in enumerate(hits2, 1):
    print(f"#{rank}  {hit.score:.4f}  {hit.payload['text']}")
''')

code(r'''
# --- Experiment 6b: add more sentences and re-index ---
extra_sentences = [
    "Silver is widely used in solar panels and electronics.",
    "Palladium is a key component in catalytic converters.",
    "The London Bullion Market Association sets benchmark prices.",
]
extra_vectors = model.encode(extra_sentences)

# Qdrant ids must be unique integers; offset by the existing count
existing_count = len(SENTENCES)
extra_points = [
    PointStruct(id=existing_count + i, vector=vec.tolist(), payload={"text": s})
    for i, (vec, s) in enumerate(zip(extra_vectors, extra_sentences))
]
client.upsert(collection_name=COLLECTION, points=extra_points)

print(f"Collection now has {client.get_collection(COLLECTION).vectors_count} points\n")

# Re-run the electronics query against the enlarged corpus
hits3 = client.query_points(collection_name=COLLECTION, query=q2_vec, limit=3).points
print(f'Query: "{QUERY_2}"\n')
for rank, hit in enumerate(hits3, 1):
    print(f"#{rank}  {hit.score:.4f}  {hit.payload['text']}")
''')

code(r'''
# --- Experiment 6c: see what happens with k=1 vs k=5 ---
for k in (1, 3, 5):
    hits_k = client.query_points(collection_name=COLLECTION,
                                  query=model.encode("What is a troy ounce?").tolist(),
                                  limit=k).points
    print(f"\nTop-{k}:")
    for rank, hit in enumerate(hits_k, 1):
        print(f"  #{rank}  {hit.score:.4f}  {hit.payload['text']}")
''')

md(r"""
> **Reflection questions:**
> - Did adding domain-specific sentences improve the results for the
>   "electronics" query?
> - What happens if you set `limit=8` (more than the corpus size)?
> - Can you construct a query where the top result is clearly wrong — i.e.,
>   high cosine similarity but factually irrelevant?
""")


# ============================================================================
# RECAP
# ============================================================================
md(r"""
---
# Recap & what's next

You built a complete similarity-search pipeline from scratch:

1. **Cosine similarity** — the formula that measures angle between vectors,
   stripping out magnitude bias.
2. **Qdrant `:memory:`** — create a collection, upsert `PointStruct`s, and
   query with `query_points`.
3. **Top-k retrieval** — the engine that every RAG pipeline's retriever uses.

**Key cautions to carry forward:**
- High cosine score ≠ correct or faithful passage.
- Top-k is a hard cut; ranks just below k are silently dropped.
- The embedding model's training domain shapes what "similar" means.

---

**Next module — Module 04: Chunking & the Corpus**

Module 04 scales everything up. You will load the real 8-file precious-metals
knowledge base, split long documents into overlapping chunks with
`RecursiveCharacterTextSplitter`, and build a full-scale Qdrant collection
ready for the RAG pipeline that Module 05 assembles.
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

OUT = "03_similarity_search.ipynb"
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

n_md   = sum(1 for k, _ in CELLS if k == "md")
n_code = sum(1 for k, _ in CELLS if k == "code")
print(f"Wrote {OUT}: {len(CELLS)} cells ({n_md} markdown, {n_code} code)")
