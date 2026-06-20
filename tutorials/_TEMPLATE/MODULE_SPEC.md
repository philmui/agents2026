# MODULE_SPEC — master brief for building one tutorial module

You are building **one module** of a 12-part tutorial track for advanced
high-school researchers. The track gently rebuilds the capstone in
`topics/06_rag_eval/` (an Agentic RAG **evaluation** system) one concept at a
time. Module 12 reassembles the whole thing.

Read this entire file before writing. Everything you need is here or in the
referenced paths. Do NOT invent APIs — all real code is reproduced below or
lives in the capstone notebook.

---

## 1. Your deliverables (the artifact contract)

Write these into `tutorials/NN_slug/` (your assigned folder):

```
NN_slug/
├── slides/index.html          # reveal.js deck, house style
├── slides/assets/*.svg         # SVGs this deck references (copy from staging)
├── NN_slug_explanation.md      # concepts → code preview → notebook preview → cautions → refs
├── _build_notebook.py          # the notebook builder (md()/code() cells)
├── NN_slug.ipynb               # GENERATED — run `python3 _build_notebook.py`
├── pyproject.toml              # requires-python>=3.13; [tool.uv] package=false; minimal deps
├── .gitignore                  # copy from _TEMPLATE/.gitignore (NO .env line; .env is shared at parent)
├── corpus/                     # own copy of the 8 metals .md files (modules that use the corpus: M4+)
└── frozen/*.json               # cached real-run outputs for keyed modules (M5+) — see §6
```

After writing, run `python3 _build_notebook.py` from your folder so the `.ipynb`
exists and is valid JSON.

### Templates to start from (in `tutorials/_TEMPLATE/`)
- `slides/index.html` — full house-style deck: copy it, keep the `<head>`/`<style>`
  and the trailing `<script>` (Reveal.initialize + auto-fit) **verbatim**, replace
  only the body slides (title, Contents, content, Summary).
- `_build_notebook.py` — copy and fill in cells; keep the EMIT block; set `OUT`.
- `pyproject.toml`, `.gitignore`, `EXPLANATION_OUTLINE.md` — copy and fill in.

---

## 2. Shared assets (copy FROM, don't move)

- **Corpus** (8 files): `tutorials/_assets/corpus/*.md` → copy into your `corpus/`
  if your module uses it (M4+). Files: `01_spot_price_and_quotes.md`,
  `02_gold_fundamentals.md`, `03_silver_and_industrial_demand.md`,
  `04_platinum_palladium.md`, `05_investment_vehicles.md`, `06_drivers_and_macro.md`,
  `07_risk_and_portfolio.md`, `08_futures_and_derivatives.md`.
- **Capstone SVGs** (14): `tutorials/_assets/svg/*.svg` → copy the ones your deck
  uses into `slides/assets/`. Names listed in the per-module table (§7).
- **Golden questions**: `tutorials/_assets/golden_questions.json` (8 Q's, 6 single-hop
  + 2 multi-hop). Copy into your folder if your module uses it (M6+). The 2
  multi-hop questions are reserved to be **highlighted** in M12 but they exist in
  the file throughout.
- If a foundational module (M1–M3) needs a diagram the capstone lacks (embeddings,
  cosine, chunking), **hand-author a small inline SVG** in the deck (see the
  `.flow` inline-SVG example in the reference deck `topics/01_dense_retrieval/slides/index.html`).

---

## 3. House conventions (non-negotiable)

### Slides (full recipe: `skills/reveal-content-deck/SKILL.md`)
- Self-contained `index.html`; reveal.js + highlight.js from jsDelivr CDN; images in `slides/assets/`.
- `center:false` + top-align CSS (already in the template). SVG `max-height:540px` cap. Auto-fit engine after `Reveal.initialize` (already in template). Keep both.
- Footer `mui-group@asdrp ©` on every slide (already in template).
- MUST open with: an impactful **title slide** (kicker "ASDRP · Agentic RAG Track · Module NN", attribution "Built by **mui-group @ ASDRP**") then a **Contents** slide (3 color-coded tracks grouping THIS deck's real sections). End with a **Summary** slide (3-track recap) + a closing "run the notebook" line naming the `.ipynb`.
- Lead each concept slide with a diagram or code, not prose. Use the idioms: `.cols`/`.cols.w-svg`, `.card`(`.violet`/`.green`/`.amber`/`.red`), `.caution`, `.equation` (one hero formula), `.pill` tags, `.metric-table`.

### Notebook (`_build_notebook.py` → `.ipynb`)
- Spine: title + SVG → one-paragraph summary → "What you will build" step table → "What you will learn" → prerequisites → (keyed only) cost/safety note → **Step 0 setup** (uv sync + `uv run jupyter lab` + kernel picker) → key load → imports → numbered steps (each ≤ a couple of ideas, every concept followed by a ⚠ caution where apt) → recap + "next module" pointer.
- Use raw strings `md(r"""...""")` / `code(r'''...''')`.
- Notebook references sibling modules ONLY in prose ("In Module 3 you built…"). No cross-folder imports.

### Keys & secrets — SHARED `.env` at the parent (IMPORTANT, changed convention)
- There is **one** shared keys file: `tutorials/.env` (the parent of your module). It is gitignored at `tutorials/.gitignore`.
- Keyed notebooks (M5+) load keys with `load_dotenv(find_dotenv())` — this walks UP and resolves to `tutorials/.env` automatically. **Do NOT create a per-module `.env` or `.env.example`.** Do NOT add a `.env` line to your module `.gitignore` (the parent handles it).
- NEVER hard-code a key. NEVER read or copy real values from `topics/06_rag_eval/.env`.
- Keyed cells must GUARD a missing key with a friendly message and fall back to `frozen/*.json` (see §6) — never let a raw stack trace hit a student.

### Code quality
- PEP 8, type hints on function signatures, small focused cells. Comments match the capstone's explain-as-you-go density. No `print`-debugging cruft.

---

## 4. The stack ladder (which tools each module uses)

- **M1–M3: free & keyless.** Local embedder `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) + Qdrant `:memory:`. No API keys, no cost. M1 may be almost pure prose + a tiny local demo.
- **M4: still keyless** but introduces chunking + the real 8-file metals corpus with `langchain-text-splitters`, still on the local embedder + Qdrant.
- **M5: the migration.** Swap local embedder → **cloud Ollama via LiteLLM** (chat generator + embeddings). First module needing `OLLAMA_API_KEY`. Build the first real prompt-stuffing RAG answer. From here the stack mirrors the capstone exactly.
- **M6–M9: + RAGAS** evaluation (needs Ollama judge + embeddings).
- **M10: + Cohere** reranking (`COHERE_API_KEY`).
- **M11: + LangGraph** agent + agent metrics + `METALS_API_KEY` (live tools).
- **M12: everything**, the full capstone assembly.

### Real dependency versions (from `topics/06_rag_eval/pyproject.toml`)
`cohere>=7.0.4`, `ipykernel>=7.3.0`, `langchain>=1.3.9`, `langchain-ollama>=1.1.0`,
`langchain-qdrant>=1.1.0`, `langchain-text-splitters>=1.1.2`, `langgraph>=1.2.5`,
`litellm>=1.89.2`, `nest-asyncio>=1.6.0`, `pandas>=3.0.3`, `python-dotenv>=1.2.2`,
`qdrant-client>=1.18.0`, `ragas>=0.4.3`, `rapidfuzz>=3.14.5`, `requests>=2.34.2`.
Put in YOUR `pyproject.toml` ONLY the subset your module imports, plus
`jupyterlab` + `ipykernel`. (Free modules add `sentence-transformers`, `qdrant-client`,
and — for M2/M3 plotting — optionally `matplotlib`; they do NOT add ragas/litellm/etc.)

---

## 5. RAGAS 0.4.3 import-stub gotcha (M6+ only)

RAGAS 0.4.3 hard-imports a module langchain-community 1.x removed. The litellm
path never uses it, so stub it BEFORE importing ragas:

```python
import sys, types
_vx = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI:  # placeholder, intentionally non-functional
    pass
_vx.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vx
```

Also: notebooks already run an event loop; RAGAS uses asyncio, so M6+ needs:
```python
import nest_asyncio; nest_asyncio.apply()
```

---

## 6. Frozen fallback (keyed modules, M5+)

Each keyed module ships `frozen/` with a small JSON capturing the realistic
*shape* of the output its keyed cells would produce (hand-authored, plausible
numbers — clearly labelled as illustrative, matching the capstone's "*Illustrative
output*" markdown cells). The keyed cell pattern:

```python
import json, os
if HAVE_KEYS:
    result = ...real call...
else:
    result = json.load(open("frozen/step_name.json"))
    print("(using cached illustrative result — set keys in tutorials/.env to run live)")
```

Keep frozen JSON tiny. Its job is to let a keyless student read sensible output
and keep going, not to fake a full live run.

---

## 7. Per-module assignments

Build ONLY your assigned module. `slug` is the folder name.

| NN | slug | Title | New concepts (own ≤2–4) | Stack / deps to add | Keys | SVGs to reuse (from _assets/svg) | New inline SVGs to author |
|----|------|-------|--------------------------|---------------------|------|----------------------------------|---------------------------|
| 01 | 01_what_is_rag | What is RAG? | the RAG idea; why retrieve; in-context learning; confabulation | none beyond dotenv (tiny/no local model) | none | 03_retriever_vs_generator (optional) | RAG 5-step / retrieve→augment→generate flow |
| 02 | 02_embeddings | Embeddings & meaning | text→vectors; what an embedding is; embedding space | sentence-transformers | none | — | text→vector; embedding-space arrows |
| 03 | 03_similarity_search | Similarity & vector search | cosine similarity; top-k; Qdrant `:memory:` | sentence-transformers, qdrant-client | none | — | cosine angle (not distance) |
| 04 | 04_chunking_corpus | Chunking & the corpus | chunking strategy; RecursiveCharacterTextSplitter; the 8-file metals corpus | + langchain-text-splitters | none | — | long doc → overlapping chunks |
| 05 | 05_first_real_rag | Stack migration + first real RAG | cloud-Ollama via LiteLLM; shared `.env`/find_dotenv; prompt-stuffing answer | langchain-ollama, litellm, langchain-qdrant, qdrant-client | OLLAMA | 01_agentic_rag_architecture (the RAG part) | local→cloud migration arrow |
| 06 | 06_why_evaluate | Why evaluate? + RAGAS setup | eval mindset; MDD loop; SingleTurnSample/EvaluationDataset | + ragas, nest-asyncio | OLLAMA | 02_mdd_loop, 03_retriever_vs_generator | — |
| 07 | 07_retriever_metrics | Retriever metrics | LLMContextPrecisionWithReference; LLMContextRecall | ragas | OLLAMA | 04_context_precision_recall | — |
| 08 | 08_more_retriever_metrics | More retriever metrics | ContextEntityRecall; NoiseSensitivity (inverted scale!) | ragas | OLLAMA | 05_context_entities_recall, 06_noise_sensitivity | — |
| 09 | 09_generator_metrics | Generator metrics + LLM-as-judge | Faithfulness; ResponseRelevancy; FactualCorrectness; judge biases | ragas | OLLAMA | 07_faithfulness, 08_response_relevancy, 09_factual_correctness, 10_llm_as_judge | — |
| 10 | 10_reranking | Reranking | Cohere rerank-v3.5; retrieve k=10 → top_n=3; before/after lift | + cohere | OLLAMA + COHERE | 11_reranking_pipeline | — |
| 11 | 11_rag_to_agent | From RAG to Agent + agent metrics | create_react_agent (ReAct); multi-turn; ToolCallAccuracy/F1; TopicAdherence; AgentGoalAccuracy; convert_to_ragas_messages | + langgraph, requests, rapidfuzz | OLLAMA + COHERE + METALS | 01_agentic_rag_architecture, 12_topic_adherence, 13_tool_call_metrics, 14_agent_goal_accuracy | — |
| 12 | 12_capstone | Capstone | assemble all: full pipeline, all 12 metrics, MDD baseline-vs-reranked, the 2 multi-hop golden Qs, Goodhart cautions | full capstone set | OLLAMA + COHERE + METALS | all 14 | — |

---

## 8. Canonical code (lift from these — they are the REAL capstone)

### 8a. Free local embedder (M2–M4) — replaces the cloud embedder for keyless modules
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")   # 384-dim, downloads once (~90MB), then offline
vec = model.encode("How is a troy ounce defined?")           # one vector
mat = model.encode([d for d in texts])                        # many at once
# cosine similarity by hand:
import numpy as np
def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
```
For Qdrant with a local embedder, either compute vectors yourself and use the
qdrant-client directly, or wrap the model in a LangChain-compatible embeddings
shim. Simplest for M3: use `qdrant_client` directly with `VectorParams(size=384,
distance=Distance.COSINE)` and `model.encode`.

### 8b. Cloud-Ollama migration (M5) — the capstone's real setup
```python
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())                      # resolves to tutorials/.env
os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11434")

LLM_NAME_OLLAMA       = "nemotron-3-super:cloud"
EMBEDDING_NAME_OLLAMA = "qwen3-embedding:0.6b"

from langchain_ollama import ChatOllama, OllamaEmbeddings
chat_llm = ChatOllama(model=LLM_NAME_OLLAMA, base_url=os.environ["OLLAMA_API_BASE"], temperature=0.0)
lc_embeddings = OllamaEmbeddings(model=EMBEDDING_NAME_OLLAMA, base_url=os.environ["OLLAMA_API_BASE"])
```

### 8c. Vector store + corpus load (M4/M5)
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
base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
```

### 8d. RAGAS model objects (M6+)
```python
import litellm
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
LLM_MODEL       = "ollama_chat/nemotron-3-super:cloud"
JUDGE_MODEL     = "ollama_chat/gemma4:31b-cloud"
EMBEDDING_MODEL = "ollama/qwen3-embedding:0.6b"
generator_llm   = llm_factory(LLM_MODEL, provider="litellm", client=litellm.completion, temperature=0.3)
judge_llm       = llm_factory(JUDGE_MODEL, provider="litellm", client=litellm.completion, temperature=0.0)
ragas_embeddings= embedding_factory("litellm", model=EMBEDDING_MODEL, api_base=os.environ["OLLAMA_API_BASE"])
```

### 8e. RAG answer + dataset (M5 builds the answer; M6+ builds the dataset)
```python
from langchain_core.prompts import ChatPromptTemplate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a precise metals-markets tutor. Answer using ONLY the context passages. "
    "If the context does not contain the answer, say you do not know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:")
def rag_answer(question, k=10, top_n=3, use_rerank=True):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    candidates = [d.page_content for d in retriever.invoke(question)]
    contexts = [t for t, _ in rerank(question, candidates, top_n=top_n)] if use_rerank else candidates[:top_n]
    block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    response = chat_llm.invoke(RAG_PROMPT.format_messages(context=block, question=question)).content.strip()
    return {"response": response, "retrieved_contexts": contexts}
```

### 8f. Retriever metrics (M7/M8)
```python
from ragas import evaluate
from ragas.metrics import (LLMContextPrecisionWithReference, LLMContextRecall,
                           ContextEntityRecall, NoiseSensitivity)
retriever_metrics = [LLMContextPrecisionWithReference(), LLMContextRecall(),
                     ContextEntityRecall(), NoiseSensitivity()]  # NoiseSensitivity: LOWER is better
results = evaluate(dataset=eval_dataset, metrics=retriever_metrics, llm=judge_llm, embeddings=ragas_embeddings)
# single-sample debug:  score = await ContextEntityRecall(llm=judge_llm).single_turn_ascore(samples[3])
```

### 8g. Generator metrics (M9)
```python
from ragas.metrics import Faithfulness, ResponseRelevancy, FactualCorrectness
generator_metrics = [Faithfulness(), ResponseRelevancy(), FactualCorrectness()]
results = evaluate(dataset=eval_dataset, metrics=generator_metrics, llm=judge_llm, embeddings=ragas_embeddings)
fc_precision = FactualCorrectness(llm=judge_llm, mode="precision")
fc_recall    = FactualCorrectness(llm=judge_llm, mode="recall")
```

### 8h. Cohere reranking (M10)
```python
import cohere
co = cohere.ClientV2(os.environ["COHERE_API_KEY"])
def rerank(query, docs, top_n=3):
    if not docs: return []
    result = co.rerank(model="rerank-v3.5", query=query, documents=docs, top_n=top_n)
    return [(docs[r.index], r.relevance_score) for r in result.results]
```

### 8i. Agent + tools + agent metrics (M11)
```python
import requests
from langchain_core.tools import tool
METALS_BASE = "https://api.metals.dev/v1"

@tool
def get_metal_price(metal: str, currency: str = "USD") -> str:
    """Get the current spot price of a precious or industrial metal."""
    resp = requests.get(f"{METALS_BASE}/metal/spot",
        params={"api_key": os.environ["METALS_API_KEY"], "metal": metal.lower(), "currency": currency.upper()}, timeout=20)
    data = resp.json()
    if data.get("status") != "success":
        return f"Could not fetch price for {metal}: {data.get('error_message','unknown error')}."
    return f"The current spot price of {metal} is {data['rate']['price']} {currency.upper()} per troy ounce."

@tool
def search_metal_knowledge(query: str) -> str:
    """Look up background knowledge about metals markets (NOT live prices)."""
    candidates = [d.page_content for d in base_retriever.invoke(query)]
    top = rerank(query, candidates, top_n=3)
    return "\n\n".join(f"[Passage {i}] {t}" for i, (t, _) in enumerate(top, 1))

# (convert_currency tool exists in the capstone too — include if M11 covers it.)
tools = [get_metal_price, convert_currency, search_metal_knowledge]

from langgraph.prebuilt import create_react_agent
SYSTEM_PROMPT = ("You are MetalDesk, a precious-metals research assistant. ... "
                 "If a request is outside metals and markets, politely decline. "
                 "Ground your answers in tool results and do not invent prices.")
agent = create_react_agent(model=chat_llm, tools=tools, prompt=SYSTEM_PROMPT)
result = agent.invoke({"messages": [{"role": "user", "content": question}]})

from ragas.integrations.langgraph import convert_to_ragas_messages
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
from ragas.metrics import (TopicAdherenceScore, ToolCallAccuracy, ToolCallF1,
                           AgentGoalAccuracyWithReference, AgentGoalAccuracyWithoutReference)
ragas_trace = convert_to_ragas_messages(result["messages"])
```

### 8j. MDD loop (M12)
```python
def build_dataset(use_rerank, k=10, top_n=3):
    rows = [SingleTurnSample(user_input=g["question"], response=(o:=rag_answer(g["question"], k=k, top_n=top_n, use_rerank=use_rerank))["response"],
            retrieved_contexts=o["retrieved_contexts"], reference=g["reference"]) for g in golden]
    return EvaluationDataset(samples=rows)
mdd_metrics = [LLMContextPrecisionWithReference(), LLMContextRecall(), Faithfulness(), FactualCorrectness()]
baseline = evaluate(build_dataset(use_rerank=False, top_n=3), metrics=mdd_metrics, llm=judge_llm, embeddings=ragas_embeddings)
improved = evaluate(build_dataset(use_rerank=True, k=10, top_n=3), metrics=mdd_metrics, llm=judge_llm, embeddings=ragas_embeddings)
```

---

## 9. Definition of done (self-check before you finish)
- [ ] All files in §1 exist; `python3 _build_notebook.py` ran and produced a valid `.ipynb`.
- [ ] Deck opens with title (attributing mui-group @ ASDRP) + Contents, ends with Summary; footer on every slide; only the body changed from the template.
- [ ] Notebook follows the Step-0 spine; keyed modules use shared-`.env` + frozen fallback with friendly guards (no raw tracebacks).
- [ ] ≤2–4 new concepts; doesn't pre-teach later modules; prose references neighbors correctly; explanation.md has a ⚠ caution + references + next-module pointer.
- [ ] No hard-coded keys; no per-module `.env`/`.env.example`; corpus/golden copied in if used.
- [ ] Code lifted faithfully from §8 / the capstone — no invented APIs.
