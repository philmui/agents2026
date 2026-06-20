# Module 02 · Embeddings & Meaning — Explanation

> Per-module markdown companion to the notebook + slides.
> Audience: advanced high-school researchers. Tone: clear, concrete, encouraging,
> honest about where ideas break.

## Where this module sits

Module 01 introduced the RAG idea — *Retrieve, Augment, Generate* — and explained
*why* a system would look up passages before answering a question. But it left one
mechanism unexplained: how does the retriever decide which passages are relevant?
Module 02 answers that question. By the end you will be able to turn any sentence
into a 384-number vector and measure, arithmetically, how similar two sentences
are. Module 03 will build on this by dropping those vectors into a vector database
(Qdrant) and running top-k nearest-neighbor search at scale — the actual retrieval
step of the capstone RAG pipeline.

---

## The big idea

### 1. Text → numbers

A neural language model has learned, from billions of sentences, that words
appearing in similar contexts have similar meanings. When you pass a sentence to
the `SentenceTransformer` encoder, it runs the text through a transformer network
and returns a single fixed-length vector — 384 numbers for `all-MiniLM-L6-v2`.
Those numbers are not arbitrary; they encode the *meaning* of the sentence as
the model understood it during training.

The key insight: **the same meaning produces similar vectors**. "Gold is a
safe-haven asset" and "Investors flee to gold in a downturn" share no words, yet
the model places them close together in the 384-dimensional space because it has
seen them used in similar contexts many times.

### 2. Embedding space

Think of every sentence as a point in a very high-dimensional room. Two sentences
about gold prices sit near each other; a sentence about the weather sits far away.
The model has arranged the room so that *semantic proximity equals geometric
proximity*. We call this arrangement *embedding space*.

Because `all-MiniLM-L6-v2` normalises each vector to unit length (magnitude 1),
angles between vectors are all that matter. The standard measure is **cosine
similarity**: the dot product of two unit vectors equals the cosine of the angle
between them, and ranges from −1 (opposite directions) to 1 (identical direction).

### 3. Why 384 dimensions?

Each dimension loosely captures one independent facet of meaning — topic, tense,
entity type, sentiment, formality, and hundreds of other subtle patterns. You
cannot label them individually (they emerge from training, not design), but
collectively they let the model distinguish thousands of semantic distinctions
simultaneously. Two sentences can be similar along the "precious metals" axis but
different along the "price prediction" axis, and the cosine of the full 384-dim
vector captures the net result.

### 4. PCA as a sanity check

PCA (Principal Component Analysis) compresses 384 numbers down to 2 for plotting.
The resulting scatter plot is *lossy* — many dimensions are discarded — but it
gives a useful visual intuition: metal-market sentences should cluster away from
unrelated sentences. If your clusters don't look right in 2D, measure cosine
similarity in the original space before concluding there's a problem.

---

## Code preview

Load the model and encode a sentence:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # ~90 MB download, then cached
vec = model.encode("Gold is a safe-haven asset.")
print(vec.shape)   # (384,)
```

Encode a batch (returns a matrix):

```python
sentences = [
    "Gold is a safe-haven asset.",
    "Investors buy gold during recessions.",
    "The weather is nice today.",
]
embeddings = model.encode(sentences)   # shape (3, 384)
```

Cosine similarity helper (from MODULE_SPEC §8a):

```python
import numpy as np

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

print(cosine(embeddings[0], embeddings[1]))   # e.g. 0.74 — related
print(cosine(embeddings[0], embeddings[2]))   # e.g. 0.13 — unrelated
```

PCA scatter:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pts = PCA(n_components=2).fit_transform(embeddings)
plt.scatter(pts[:, 0], pts[:, 1])
# label each point with a short sentence excerpt
```

---

## Notebook preview

| Step | What you do | Key tool |
| ---: | --- | --- |
| 0 | Install deps with `uv sync`; launch Jupyter | `uv`, `ipykernel` |
| 1 | Load `all-MiniLM-L6-v2` and encode a single sentence | `SentenceTransformer` |
| 2 | Inspect the 384-dim vector (shape, norm, first few values) | `numpy` |
| 3 | Encode a batch of metals-related sentences | `SentenceTransformer` |
| 4 | Compute pair-wise cosine similarity; build a similarity matrix | `numpy`, `pandas` |
| 5 | Visualise with PCA scatter (matplotlib) | `sklearn`, `matplotlib` |
| 6 | Recap & pointer to Module 03 | — |

---

## Cautions

⚠ **Correlation, not truth.** An embedding reflects statistical patterns from
training text, not factual correctness. Two sentences can score high similarity
because they share the same vocabulary and register even if one is true and the
other is false. A retriever that ranks by cosine similarity will happily surface
a plausible-sounding but wrong passage.

⚠ **Model-dependent scores.** Cosine similarity values are meaningful only within
a single model. If you switch from `all-MiniLM-L6-v2` to a cloud model in
Module 05, previously computed similarity scores are not comparable. Always use
the same model for both indexing and querying.

⚠ **Domain shift.** `all-MiniLM-L6-v2` was fine-tuned on general-web text.
It handles everyday financial prose well, but dense specialist notation (futures
contract specifications, derivative pricing formulas) may embed poorly relative
to a domain-fine-tuned model. For the capstone's metals corpus this is not a
significant problem, but it is worth remembering for other domains.

⚠ **PCA is lossy.** The 2D scatter is a projection; two points that look far
apart in 2D may actually be close in 384D. Always measure distance or similarity
in the original high-dimensional space for any real decision.

---

## References

- Capstone theory document (general background on embeddings in RAG):
  `topics/06_rag_eval/agentic_rag_evaluation_theory.md` — "Dense Retrieval" section
- Sentence-Transformers documentation and `all-MiniLM-L6-v2` model card:
  <https://www.sbert.net/docs/sentence_transformer/pretrained_models.html>
- Original MiniLM paper: Wang et al. (2020), "MiniLM: Deep Self-Attention
  Distillation for Task-Agnostic Compression of Pre-Trained Transformers"
- NumPy `linalg.norm` and dot-product reference:
  <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>
- Scikit-learn PCA: <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>

**Next module:** `tutorials/03_similarity_search/` — cosine similarity at scale
with Qdrant `:memory:`, top-k retrieval, and why approximate nearest-neighbor
search is needed for large corpora.
