# Module 05 · Stack Migration + First Real RAG — Explanation

> Audience: advanced high-school researchers. Tone: clear, concrete, encouraging,
> honest about where ideas break.

## Where this module sits

Module 04 left you with a chunked version of the eight-file precious-metals
corpus, stored in a Qdrant in-memory vector store using the free local
`sentence-transformers` embedder. That pipeline was self-contained and required
no API keys — ideal for learning the mechanics of chunking and retrieval. Module
05 is **the pivot**: you swap the local embedder for cloud Ollama embeddings, set
up the shared `tutorials/.env` that all subsequent modules will read, and then
wire everything together into your first **end-to-end RAG answer** — retrieve →
augment prompt → generate grounded response. Every module from here (06–12)
builds directly on this stack.

## The big idea

### 1 · Why leave the local embedder?

The `all-MiniLM-L6-v2` model (384 dimensions, ~90 MB) is excellent for
prototyping because it runs offline. But for the capstone evaluation system we
need **quality and consistency**: the same embedding model must be used at index
time (encoding the corpus) *and* at query time (encoding the user question),
because cosine similarity is only meaningful when both vectors live in the same
embedding space. Cloud Ollama's `qwen3-embedding:0.6b` model produces
higher-quality representations for domain text, and it is also the model used by
RAGAS (the evaluation library introduced in Module 06) when scoring retrieval
precision and recall. Switching now means every downstream metric is measuring
the right thing.

### 2 · The shared `.env` and `find_dotenv()`

Starting in Module 05, calls go over the network, so you need an API key.
Rather than creating a separate `.env` file in every module folder (error-prone
and hard to keep in sync), the track uses **one shared file** at `tutorials/.env`.
The call `load_dotenv(find_dotenv())` walks *up* the directory tree from wherever
the notebook lives and finds that single file automatically. You create it once;
all twelve modules use it. The file is gitignored — you will never accidentally
commit your keys.

### 3 · Prompt-stuffing RAG

"Prompt stuffing" is the simplest possible RAG answer strategy: take the top-k
retrieved passages, concatenate them into a single `[1] … [2] … [3] …` block,
and inject that block into a prompt template alongside the user question. The
LLM is instructed to answer *only from the provided context* — if the context
does not contain the answer, the model should say so rather than invent one. This
constraint is what separates RAG from pure generation and is the foundation for
the faithfulness metrics you will measure in Module 09.

In Module 05 we set `use_rerank=False` and simply take the first `top_n=3`
candidates from the retriever's top-k results. Module 10 will add Cohere
reranking to improve which three passages are selected.

## Code preview

**Load keys and build the LLM + embedder:**

```python
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())          # walks up → tutorials/.env
os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11434")

from langchain_ollama import ChatOllama, OllamaEmbeddings
chat_llm = ChatOllama(
    model="nemotron-3-super:cloud",
    base_url=os.environ["OLLAMA_API_BASE"],
    temperature=0.0,
)
lc_embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url=os.environ["OLLAMA_API_BASE"],
)
```

**Chunk the corpus and build the vector store:**

```python
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

raw_docs = [{"source": p.name, "page_content": p.read_text()}
            for p in sorted(Path("corpus").glob("*.md"))]
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
lc_docs = [Document(page_content=piece, metadata={"source": d["source"]})
           for d in raw_docs for piece in splitter.split_text(d["page_content"])]
vector_store = QdrantVectorStore.from_documents(
    lc_docs, embedding=lc_embeddings, location=":memory:", collection_name="metals_kb")
```

**Generate a RAG answer (no reranking yet):**

```python
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a precise metals-markets tutor. Answer using ONLY the context passages. "
    "If the context does not contain the answer, say you do not know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:")

def rag_answer(question: str, k: int = 10, top_n: int = 3) -> dict:
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    candidates = [d.page_content for d in retriever.invoke(question)]
    contexts = candidates[:top_n]          # no reranking in M5
    block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    response = chat_llm.invoke(
        RAG_PROMPT.format_messages(context=block, question=question)
    ).content.strip()
    return {"response": response, "retrieved_contexts": contexts}
```

## Notebook preview

| Step | What you do |
| ---: | --- |
| 0.1 | Install dependencies with `uv sync` |
| 0.2 | Load the shared `tutorials/.env` key file; confirm `HAVE_KEYS` |
| 1 | Understand **why** we migrate away from the local embedder |
| 2 | Build the cloud Ollama LLM and embedder objects |
| 3 | Chunk the corpus and index it in Qdrant `:memory:` |
| 4 | Write `rag_answer()` — the prompt-stuffing RAG function |
| 5 | Run a sample question end-to-end and inspect the output |

## Cautions

⚠ **Grounding ≠ correctness.** Instructing the LLM to answer only from the
provided context reduces hallucination but does not eliminate it. If the
retrieved passages themselves contain an error, or if the model paraphrases
ambiguously, the answer can still be wrong. Module 09 introduces *Faithfulness*
and *Factual Correctness* metrics that quantify this gap.

⚠ **RAG quality is retrieval-limited.** If the right passage is not in the top-k
results, even a perfect generator cannot produce a correct answer. The number k
(how many candidates you retrieve before selecting top_n) is a critical
hyperparameter: too small and you miss relevant passages; too large and you flood
the prompt with noise. Module 07 measures retrieval precision and recall so you
can tune k empirically.

⚠ **Cost note.** Every call to `rag_answer()` sends one embedding request (for
the question) and one chat-completion request (for the answer) to the cloud
Ollama endpoint. Indexing the corpus sends one embedding per chunk (~50–80
calls). If `OLLAMA_API_KEY` is missing, the notebook falls back to illustrative
results in `frozen/rag_answer.json` so you can follow along without spending
credits.

## References

- Capstone theory: `topics/06_rag_eval/agentic_rag_evaluation_theory.md` —
  Section "RAG Pipeline Overview" and "Retrieval vs Generation" diagrams.
- Capstone notebook: `topics/06_rag_eval/agentic_rag_evaluation_tutorial.ipynb`
  — the `rag_answer()` function and `QdrantVectorStore.from_documents` call.
- LangChain-Ollama docs: https://python.langchain.com/docs/integrations/chat/ollama
- LangChain-Qdrant docs: https://python.langchain.com/docs/integrations/vectorstores/qdrant
- `python-dotenv` `find_dotenv()` docs: https://pypi.org/project/python-dotenv/
- **Next module:** Module 06 — *Why Evaluate? + RAGAS Setup* — introduces the
  Metrics-Driven Development loop and the first RAGAS evaluation dataset.
