"""Builds 02_embeddings.ipynb from a list of (type, source) cells.

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
# Module 02 · Embeddings & Meaning

### A hands-on, build-it-yourself module for advanced high school researchers

One-paragraph orientation: Module 01 introduced the RAG pipeline and explained
*why* a system retrieves passages before generating an answer.  This module
answers the deeper question: **how does "relevant" get measured?**  You will
load a free, keyless sentence-transformer model, encode English sentences into
384-number vectors, and prove — with arithmetic — that semantically similar
sentences land geometrically close in the resulting *embedding space*.  Module 03
will drop those same vectors into Qdrant and run nearest-neighbor search, turning
the similarity measure you learn here into a working retriever.

This is module **02** of a twelve-part track that ends in a full
**Agentic RAG Evaluation** capstone.
""")

md(r"""
## 📋 Summary: the one-paragraph version

A sentence embedding is a fixed-length list of numbers — 384 for the model used
here — where the numbers encode what a sentence *means* rather than which words
it uses.  Sentences about related topics end up as nearby points in the
384-dimensional space, and we measure "nearby" with **cosine similarity**: the
dot product of two unit-norm vectors.  Values near 1 mean semantically similar;
values near 0 mean unrelated.  No API key is needed: the model downloads once
(~90 MB) and runs entirely on your laptop.
""")

md(r"""
## 🗺️ What you will build, step by step

| Step | What you do | Key tool |
| ---: | --- | --- |
| 0 | Install dependencies with `uv sync`; launch Jupyter | `uv`, `ipykernel` |
| 1 | Load `all-MiniLM-L6-v2` and encode a single sentence | `SentenceTransformer` |
| 2 | Inspect the 384-dim vector (shape, norm, first values) | `numpy` |
| 3 | Encode a batch of metals-related sentences | `SentenceTransformer` |
| 4 | Compute pair-wise cosine similarity; build a similarity matrix | `numpy`, `pandas` |
| 5 | Visualise with a PCA 2D scatter | `sklearn`, `matplotlib` |
| 6 | Recap and pointer to Module 03 | — |

### 🎓 What you will *learn* (the concepts)

- What an **embedding** is and why 384 numbers can encode sentence meaning
- What **embedding space** means geometrically
- How **cosine similarity** quantifies semantic closeness
- Why embeddings encode *correlation*, not truth — and why that matters for RAG

### ✅ Prerequisites

- Comfort reading basic Python.  No machine learning background required.
- Module 01 (What is RAG?) — helpful but not strictly required.
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
so the environment is **reproducible**.  Install everything with one command,
**from this module's folder**, using [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync            # reads pyproject.toml, creates .venv/, installs everything
uv run jupyter lab # launch Jupyter inside that environment
```

When the notebook opens, pick the kernel **`Python 3 (ipykernel)`** (top-right
kernel picker).  That is the interpreter from `.venv`, so every `import` below
resolves against what `uv sync` installed.

> **No API key needed.**  This module runs entirely locally.  The model
> (`all-MiniLM-L6-v2`) downloads from HuggingFace Hub the first time (~90 MB)
> and is then cached in `~/.cache/huggingface/`.  Subsequent runs are instant.
""")

# ============================================================================
# STEP 1 — LOAD THE MODEL & ENCODE ONE SENTENCE
# ============================================================================
md(r"""
---
# Step 1 · Load the model and encode one sentence

`SentenceTransformer` is a small library that wraps a pre-trained transformer
and adds a pooling layer to produce a single fixed-length vector per input text.

The model we use, **`all-MiniLM-L6-v2`**, is a 6-layer distilled transformer
fine-tuned on over 1 billion sentence pairs.  It produces **384-dimensional**
unit-norm vectors — compact enough to be fast, rich enough to capture nuanced
meaning.

The call `model.encode(text)` runs a forward pass and returns a NumPy array.
""")

code(r'''
from sentence_transformers import SentenceTransformer

# Downloads the model the first time (~90 MB), then cached locally.
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode a single sentence.
sentence = "Gold is a safe-haven asset."
vec = model.encode(sentence)

print(f"Input : {sentence!r}")
print(f"Shape : {vec.shape}")        # (384,)
print(f"Dtype : {vec.dtype}")        # float32
print(f"First 8 values: {vec[:8].round(4)}")
''')

# ============================================================================
# STEP 2 — INSPECT THE VECTOR
# ============================================================================
md(r"""
---
# Step 2 · Inspect the 384-dimensional vector

A few things worth noting about the raw vector before we compare sentences.
""")

code(r'''
import numpy as np

# The model outputs unit-norm vectors (norm ≈ 1.0).
norm = np.linalg.norm(vec)
print(f"‖vec‖ = {norm:.6f}  (should be ≈ 1.0)")

# Values are spread across the full [-1, 1] range.
print(f"min={vec.min():.4f}  max={vec.max():.4f}  mean={vec.mean():.4f}")

# Because the norm is 1, cosine similarity of a vector with itself is 1.
dot_self = float(vec @ vec)
print(f"cos(vec, vec) = {dot_self:.6f}  (should be ≈ 1.0)")
''')

md(r"""
> **Why unit norm?**  When all vectors have the same length (magnitude = 1), the
> angle between them is the *only* thing that varies.  Cosine similarity then
> simplifies to a plain dot product — no division needed.  The model normalises
> by design so that length differences don't distort similarity scores.
""")

# ============================================================================
# STEP 3 — ENCODE A BATCH OF METALS SENTENCES
# ============================================================================
md(r"""
---
# Step 3 · Encode a batch of sentences

`model.encode(list_of_strings)` encodes all sentences in a single forward pass
(with internal batching on longer lists).  The output is a **2-D NumPy array**
of shape `(N, 384)` where row `i` is the embedding for sentence `i`.
""")

code(r'''
sentences = [
    "Gold is a safe-haven asset.",                     # 0
    "Investors buy gold during recessions.",            # 1 — semantically close to 0
    "Silver is widely used in solar panel production.", # 2
    "Platinum group metals are key catalysts.",         # 3 — somewhat related to 2
    "The weather is nice today.",                       # 4 — off-topic
]

embeddings = model.encode(sentences)   # shape (5, 384)
print(f"embeddings.shape = {embeddings.shape}")

# Confirm each row has unit norm.
norms = np.linalg.norm(embeddings, axis=1)
print(f"Row norms: {norms.round(4)}")
''')

# ============================================================================
# STEP 4 — COSINE SIMILARITY MATRIX
# ============================================================================
md(r"""
---
# Step 4 · Cosine similarity: measuring semantic distance

The cosine similarity between two vectors `a` and `b` is:

$$\cos(\theta) = \frac{a \cdot b}{\|a\| \cdot \|b\|}$$

Because our vectors are already unit-norm, this reduces to just `a · b`.  Values
range from **−1** (opposite directions) to **1** (identical directions).

We'll implement the helper from the capstone codebase and build a full 5 × 5
similarity matrix.
""")

code(r'''
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# Build the 5×5 pairwise similarity matrix.
n = len(sentences)
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = cosine(embeddings[i], embeddings[j])

# Display as a labelled DataFrame.
import pandas as pd

short_labels = [s[:32] + ("…" if len(s) > 32 else "") for s in sentences]
df = pd.DataFrame(sim_matrix.round(3), index=short_labels, columns=short_labels)
print(df.to_string())
''')

md(r"""
**What to look for in the matrix:**

- The diagonal is always **1.000** (every sentence is identical to itself).
- Sentences 0 and 1 ("Gold … safe-haven" vs "Investors buy gold …") should score
  high — they are semantically close even though they share few words.
- Sentence 4 ("The weather …") should score near **0** against all metals
  sentences — it lives in a completely different region of embedding space.

> ⚠ **Caution — correlation, not truth.**  The embedding captures *statistical
> co-occurrence* from training data, not logical meaning.  "Gold prices always
> rise in recessions" and "Gold prices never rise in recessions" may score a
> *high* similarity because the same financial vocabulary appears in both — the
> model does not understand negation the way a human does.
""")

# ============================================================================
# STEP 5 — PCA VISUALISATION
# ============================================================================
md(r"""
---
# Step 5 · Visualising embedding space with PCA

We can't draw 384 axes, but we can project the points onto the 2 directions of
greatest variance using **Principal Component Analysis (PCA)**.  The result is a
lossy but intuitive scatter plot.
""")

code(r'''
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Project 384 dimensions → 2 for visualisation.
pca = PCA(n_components=2, random_state=42)
pts = pca.fit_transform(embeddings)   # shape (5, 2)

var_explained = pca.explained_variance_ratio_ * 100
print(f"Variance explained: PC1={var_explained[0]:.1f}%  PC2={var_explained[1]:.1f}%")
print("(PCA is lossy — most variance lives in the remaining 382 dimensions)\n")

# Colour scheme: blue for metals, red for off-topic.
colors = ["#2563eb", "#2563eb", "#7c3aed", "#7c3aed", "#dc2626"]
labels_short = ["Gold: safe-haven", "Gold: recessions",
                "Silver: solar", "Platinum: catalysts", "Weather"]

fig, ax = plt.subplots(figsize=(6, 4.5))
for i, (x, y) in enumerate(pts):
    ax.scatter(x, y, color=colors[i], s=90, zorder=3)
    ax.annotate(
        labels_short[i], (x, y),
        textcoords="offset points", xytext=(6, 3),
        fontsize=8, color=colors[i],
    )

ax.set_title("PCA projection of sentence embeddings (2 of 384 dims)", fontsize=10)
ax.set_xlabel(f"PC 1  ({var_explained[0]:.1f}% var.)")
ax.set_ylabel(f"PC 2  ({var_explained[1]:.1f}% var.)")
ax.axhline(0, color="#e2e8f0", linewidth=0.8)
ax.axvline(0, color="#e2e8f0", linewidth=0.8)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.show()
''')

md(r"""
**Interpreting the plot:**

- The two "gold" sentences (blue) should cluster together in the top-left (or
  wherever the metals region lands).
- The "weather" sentence (red) should sit alone, far from the metals cluster.
- "Silver" and "Platinum" (violet) are both metals but different sub-topics —
  they may land between the gold cluster and the weather sentence.

> ⚠ **PCA is lossy.** Two points that look far apart in 2D might actually be
> close in the full 384-dim space.  Always measure cosine similarity in the
> original space for any real decision — the plot is only for intuition.
""")

# ============================================================================
# STEP 6 (BONUS) — HIGHLIGHT: MODEL-DEPENDENT SCORES
# ============================================================================
md(r"""
---
# Step 6 · Caution: scores are model-dependent

This step does not require any new library; it just illustrates an important
pitfall by computing a similarity with a simpler representation.
""")

code(r'''
# Bag-of-words similarity using raw token overlap (Jaccard on word sets).
def jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity (case-insensitive)."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


pairs = [
    (sentences[0], sentences[1]),   # "Gold safe-haven" vs "gold recessions"
    (sentences[0], sentences[4]),   # "Gold safe-haven" vs "weather"
]

print(f"{'Pair':<50}  {'Jaccard':>8}  {'Cosine':>8}")
print("-" * 70)
for a, b in pairs:
    j = jaccard(a, b)
    c = cosine(model.encode(a), model.encode(b))
    print(f"{a[:24]!r} ↔ {b[:24]!r}   {j:8.3f}   {c:8.3f}")

print()
print("Jaccard sees only shared words; cosine sees shared meaning.")
print("Note: cosine values are specific to all-MiniLM-L6-v2.")
print("Switching to a different model would change the numbers.")
''')

# ============================================================================
# RECAP
# ============================================================================
md(r"""
---
# Recap & what's next

### What you learned in Module 02

| Concept | One-line summary |
| --- | --- |
| **Embedding** | A sentence → a fixed-length vector (384 floats) encoding meaning |
| **Embedding space** | Sentences as points; similar meanings cluster together |
| **Cosine similarity** | Dot product of unit-norm vectors; 1 = identical, 0 = unrelated |
| **PCA** | A lossy 2D projection for visualising high-dimensional clusters |

### ⚠ Key cautions to remember

1. **Correlation ≠ truth** — similar wording can produce high similarity even if
   one sentence is false.
2. **Model-dependent** — similarity scores are not comparable across different
   models.  Use one model consistently.
3. **Domain shift** — `all-MiniLM-L6-v2` works well for general financial prose
   but may struggle with highly specialised notation.

---

**Next module (03):** *Similarity & Vector Search* — you will insert the same
embeddings you computed here into a **Qdrant** in-memory vector database and run
top-k nearest-neighbor search.  That is the actual retrieval step the capstone
uses to find relevant passages before generating an answer.
""")

# ============================================================================
# EMIT NOTEBOOK  (do not change below except OUT)
# ============================================================================
def to_cell(kind, src):
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

OUT = "02_embeddings.ipynb"   # <-- this module's notebook filename
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

n_md   = sum(1 for k, _ in CELLS if k == "md")
n_code = sum(1 for k, _ in CELLS if k == "code")
print(f"Wrote {OUT}: {len(CELLS)} cells ({n_md} markdown, {n_code} code)")
