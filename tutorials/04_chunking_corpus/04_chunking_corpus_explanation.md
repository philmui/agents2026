# Module 04 · Chunking & the Corpus — Explanation

> Audience: advanced high-school researchers. Tone: clear, concrete, encouraging,
> honest about where ideas break.

## Where this module sits

Module 03 left you with a working cosine-similarity search over a handful of toy sentences stored in Qdrant `:memory:`. You saw how vectors encode meaning and how nearest-neighbor lookup finds related text. Module 04 takes the next essential step: replacing those toy sentences with a real, eight-file knowledge base about precious-metals markets, and introducing **chunking** — the practice of splitting long documents into overlapping windows before embedding them. That combination (chunked corpus + local embedder + Qdrant) is the exact foundation the capstone (Module 12) builds on, even though later modules will swap the local embedder for a cloud model.

## The big idea

### Why not embed a whole document?

An embedding model turns text into a single fixed-size vector. When you feed it a 2 000-word document, that vector is a weighted average of every idea in the document — gold prices, LBMA auctions, bid-ask spreads, troy ounces — all blended together. If a query asks specifically about LBMA auctions, the whole-document vector may be close enough in embedding space to retrieve the right file, but it will also compete with every other file that happens to discuss gold in any way. The signal-to-noise ratio is poor.

Chunking fixes this by giving each focused passage its own vector. A 500-character chunk about "the LBMA twice-daily electronic auction" becomes its own point in embedding space, close to queries about auctions and far from queries about silver photovoltaics. Retrieval precision improves dramatically.

### chunk_size and chunk_overlap

`chunk_size` is the maximum number of characters in one chunk. `chunk_overlap` is the number of characters shared between consecutive chunks. Overlap matters because a key sentence might straddle the boundary between two windows; without overlap, it would appear in neither. The capstone uses `chunk_size=500, chunk_overlap=60` — roughly one dense paragraph per chunk with a two-sentence safety margin.

`RecursiveCharacterTextSplitter` from `langchain-text-splitters` tries to respect natural text boundaries. It attempts to break on double newlines (paragraph boundaries) first, then single newlines, then spaces, and only as a last resort splits mid-word. This means chunks almost always end at a sentence boundary, which keeps embeddings coherent.

### The 8-file metals corpus

The corpus lives in `corpus/` and consists of eight Markdown files covering different facets of precious-metals markets: spot prices and quotes, gold fundamentals, silver and industrial demand, platinum and palladium, investment vehicles, macro drivers, risk and portfolio construction, and futures and derivatives. Each file is approximately 2 000 words. Together they give the retriever enough depth that different questions pull back genuinely different chunks — which is exactly what the evaluation modules (07–09) need to measure.

### From chunks to Qdrant

After splitting, each chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, downloads once, then fully offline). The vectors are stored as `PointStruct` objects in a Qdrant in-memory collection. Querying is the same cosine nearest-neighbor lookup from Module 03, but now it returns targeted passages instead of whole documents.

## Code preview

Load the corpus with a `Path` glob — no file names hard-coded:

```python
from pathlib import Path
raw_docs = [{"source": p.name, "page_content": p.read_text()}
            for p in sorted(Path("corpus").glob("*.md"))]
```

Split every document into overlapping chunks:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
chunks = [(d["source"], piece)
          for d in raw_docs
          for piece in splitter.split_text(d["page_content"])]
```

Embed all chunks at once (batched, fast on CPU):

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode([text for _, text in chunks], show_progress_bar=True)
```

Index and query:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(":memory:")
client.create_collection("metals_kb",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE))
client.upsert("metals_kb", points=[
    PointStruct(id=i, vector=v.tolist(), payload={"text": t, "source": s})
    for i, ((s, t), v) in enumerate(zip(chunks, vectors))])

q_vec = model.encode("How is the LBMA gold price set?").tolist()
hits = client.search("metals_kb", query_vector=q_vec, limit=5)
for h in hits:
    print(h.payload["text"][:120])
```

## Notebook preview

| Step | What you do |
| ---: | --- |
| 0 | Install deps with `uv sync`; launch Jupyter Lab |
| 1 | Inspect the corpus — count files, preview one document |
| 2 | Chunk all 8 files with RecursiveCharacterTextSplitter (500 / 60) |
| 3 | Embed all chunks with the local sentence-transformers model |
| 4 | Build a Qdrant `:memory:` collection and upsert all vectors |
| 5 | Run sample queries and inspect the top-k chunk results |
| 6 | Experiment: change chunk_size and observe the effect on result count and content |

## Cautions

⚠ **Chunk too small — you lose context.** A chunk of 50 characters might contain a single noun phrase with no surrounding sentence. The embedding captures a fragment of meaning, and queries that phrase things differently will miss it entirely. As a rule of thumb, a chunk should contain at least one complete sentence, preferably two or three.

⚠ **Chunk too large — you dilute retrieval.** A 2 000-character chunk encodes multiple ideas. The embedding is pulled toward the dominant topic, so a query about a minor detail in that passage will score low even though the answer is there. You get recall (the document is retrieved) but poor precision (most of the text is irrelevant to the question).

⚠ **chunk_overlap does not eliminate boundary artifacts.** Overlap reduces the chance that a key sentence falls entirely outside all chunks, but two consecutive chunks still share only 60 characters. A multi-sentence fact that spans 80 characters could still be split. This is one reason Module 10 adds Cohere reranking — to recover precision after retrieving a broader initial set.

⚠ **All chunks are not equal in size.** `RecursiveCharacterTextSplitter` tries to respect boundaries, so some chunks may be shorter than `chunk_size`. Always inspect `len(chunks)` and a few random samples to confirm the distribution looks reasonable.

## References

- Capstone theory doc: `topics/rag-eval/agentic_rag_evaluation_theory.md` — see the "Corpus and Retriever Setup" section for how chunking fits into the full evaluation pipeline.
- Capstone notebook: `topics/rag-eval/agentic_rag_evaluation_tutorial.ipynb` — the corpus load and splitter call appear in the "Vector Store" cell.
- LangChain text splitters docs: https://python.langchain.com/docs/how_to/recursive_text_splitter/
- `sentence-transformers` model card for `all-MiniLM-L6-v2`: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Qdrant Python client docs: https://python-client.qdrant.tech/

**Next module:** Module 05 — *Stack migration + first real RAG* — replaces the local `sentence-transformers` embedder with cloud Ollama via LiteLLM, introduces the shared `tutorials/.env` key file, and builds the first complete prompt-stuffing RAG answer over this same corpus.
