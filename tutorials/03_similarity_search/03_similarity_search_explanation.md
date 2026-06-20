# Module 03 · Similarity & Vector Search — Explanation

> Audience: advanced high-school researchers. Tone: clear, concrete, encouraging,
> honest about where ideas break.

## Where this module sits

Module 02 showed you how to turn text into numeric vectors (embeddings) with
`sentence-transformers`. You now have a bag of 384-dimensional arrows floating
in space. Module 03 answers the natural next question: *given a query arrow,
which stored arrows point most nearly the same direction?* That question is
**similarity search**, and answering it efficiently requires both a similarity
function (cosine) and a vector store (Qdrant). Module 04 will scale this up by
loading a real 8-file corpus and chunking each document before indexing —
everything you build here carries forward unchanged.

## The big idea

### 1. Cosine similarity — angle, not distance

Two sentences embedded by the same model land in the same 384-dimensional
space. Their *meaning* is encoded in the *direction* of their vectors, not in
how far each vector stretches from the origin. Cosine similarity measures the
angle θ between two vectors:

```
cos θ = (a · b) / (‖a‖ × ‖b‖)
```

Dividing by the norms strips out vector length, leaving a score in \[−1, 1\]:
1 means identical direction (identical meaning), 0 means orthogonal
(unrelated), −1 means perfectly opposite. In practice, sentence-transformer
scores cluster between 0.2 and 0.95 for English text.

### 2. Direction encodes meaning, length doesn't

"Gold is priced in troy ounces" and "Gold is measured in troy ounces per
unit" are paraphrases: their arrows point almost the same way even though one
string is longer than the other. A naive dot-product score would penalise the
shorter sentence; cosine cancels that length bias. That is why
`Distance.COSINE` is the standard choice for semantic text retrieval.

### 3. Qdrant `:memory:` — a vector database in a Python process

Qdrant is a purpose-built vector database. The `:memory:` mode runs entirely
inside your Python process with no external server, no disk, and no
configuration — ideal for learning. You create a *collection* (analogous to a
database table, but every row is a vector), *upsert* points (each point is a
vector + an integer id + optional payload), and then *query* by passing a query
vector and a `limit=k`.

Internally Qdrant builds an HNSW (Hierarchical Navigable Small World) index
that finds approximate nearest neighbours in O(log n) time instead of
brute-forcing all O(n) pairwise cosines.

### 4. Top-k retrieval

Querying with `limit=k` returns the k points whose vectors are closest in
cosine to the query vector. This is the "retrieval" step that Module 05 will
plug into a full prompt-stuffing RAG pipeline. The k results are the
"context passages" that get handed to the language model.

## Code preview

Embed a small toy corpus and compute a manual cosine:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

a = model.encode("What is a troy ounce?")
b = model.encode("A troy ounce equals 31.1 grams.")

def cosine(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

print(cosine(a, b))   # ~0.88
```

Create a Qdrant collection and upsert points:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient(":memory:")
client.create_collection(
    collection_name="demo",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

sentences = [
    "Gold is priced in troy ounces.",
    "Silver has significant industrial demand.",
    "A troy ounce equals 31.1 grams.",
    "Platinum is rarer than gold.",
    "Machine learning optimises loss functions.",
]
vectors = model.encode(sentences)

client.upsert(
    collection_name="demo",
    points=[
        PointStruct(id=i, vector=vec.tolist(), payload={"text": s})
        for i, (vec, s) in enumerate(zip(vectors, sentences))
    ],
)
```

Query for top-3 results:

```python
query = "What is a troy ounce?"
hits = client.query_points(
    collection_name="demo",
    query=model.encode(query).tolist(),
    limit=3,
).points

for rank, hit in enumerate(hits, 1):
    print(f"#{rank}  {hit.score:.4f}  {hit.payload['text']}")
```

## Notebook preview

| Step | What you do |
| ---: | --- |
| 0 | Install libraries with `uv sync`; launch `uv run jupyter lab` |
| 1 | Load the model and inspect one embedding |
| 2 | Implement `cosine()` by hand and explore pairwise scores |
| 3 | Create a Qdrant `:memory:` collection with `VectorParams(size=384, distance=Distance.COSINE)` |
| 4 | Encode the toy corpus and upsert `PointStruct`s |
| 5 | Query "What is a troy ounce?" and print top-3 scored results |
| 6 | Experiment: change the query, add more sentences, vary `limit` |

## Cautions

**⚠ High cosine similarity does not mean the passage is the correct answer.**
"Gold is priced in troy ounces" is topically similar to "What is a troy
ounce?" and will score highly — but it doesn't *define* a troy ounce. The
retriever's job is to surface *candidate* passages, not to verify their
factual accuracy. That distinction matters enormously from Module 06 onward
when you evaluate *faithfulness* and *factual correctness* separately from
*context recall*.

**⚠ The embedding model is a fixed snapshot.**
`all-MiniLM-L6-v2` was trained on general English text. Finance-specific
jargon ("contango", "backwardation", "LBMA AM fix") may land in unexpected
regions of the embedding space, degrading retrieval quality. Module 05
migrates to a domain-aware cloud embedder to address this.

**⚠ Top-k is a hard cut.**
If scores at ranks 3 and 4 are 0.71 and 0.70 — nearly identical — top-3
retrieval arbitrarily discards rank 4. Module 10 introduces Cohere reranking
to handle borderline cases more gracefully.

## References

- Capstone theory: `topics/rag-eval/agentic_rag_evaluation_theory.md`
  (see the "Retrieval" section and the context-precision/recall discussion)
- Capstone notebook: `topics/rag-eval/agentic_rag_evaluation_tutorial.ipynb`
  (Step 3 — vector store construction with Qdrant)
- Qdrant docs: <https://qdrant.tech/documentation/>
- `sentence-transformers` model card: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- Johnson et al. (2019). *Billion-scale similarity search with GPUs* (FAISS / HNSW background)

**Next module:** Module 04 — Chunking & the Corpus. You will apply everything
from this module to the real 8-file precious-metals knowledge base,
introducing `RecursiveCharacterTextSplitter` to handle long documents before
indexing.
