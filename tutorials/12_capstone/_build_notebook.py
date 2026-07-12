"""Builds 12_capstone.ipynb from a list of (type, source) cells.

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
def md(text):  CELLS.append(("md", text.strip("\n")))
def code(text): CELLS.append(("code", text.strip("\n")))

# ============================================================================
# TITLE + SUMMARY
# ============================================================================
md(r"""
# Module 12 · Capstone: the Full Agentic RAG Evaluation System

### A hands-on, build-it-yourself module for advanced high school researchers

![Agentic RAG architecture](slides/assets/01_agentic_rag_architecture.svg)

In Modules 1–11 you built every individual piece of an Agentic RAG Evaluation
system: a chunked vector store (M4), cloud Ollama embeddings (M5), RAGAS
evaluation infrastructure (M6), all four retriever metrics (M7–M8), all three
generator metrics (M9), Cohere reranking with a measurable MDD lift (M10), and
a LangGraph ReAct agent with all five agent-tier metrics (M11). This is Module
12 — the Capstone. It assembles every piece into one unified system, runs all
12 metrics, executes the full Metrics-Driven Development loop, spotlights the
two hardest questions in the golden set, and closes with the cautions that
matter most. There is nothing new to teach; this is the finished wall built from
the bricks you already laid.

This is Module 12 of a twelve-part track that ends in a full **Agentic RAG
Evaluation** capstone.
""")

md(r"""
## 📋 Summary: the one-paragraph version

A complete Agentic RAG Evaluation system has three layers. The **pipeline**
layer loads the corpus, chunks and embeds it, retrieves candidate passages,
reranks them with Cohere, generates an answer, and — when the question needs
live data — routes through a LangGraph agent that can call three tools.
The **metrics** layer grades the output with 12 RAGAS metrics divided across
three tiers: four retriever metrics (precision, recall, entities recall, noise
sensitivity), three generator metrics (faithfulness, response relevancy, factual
correctness), and five agent metrics (topic adherence, tool-call accuracy,
tool-call F1, goal accuracy with and without reference). The **development**
layer is Metrics-Driven Development (MDD): baseline, change one thing (turn on
reranking), recompute, compare. The capstone runs all three layers, spotlights
the two multi-hop golden questions where retrieval still struggles, and ends
with Goodhart's Law — because knowing when a number is lying is the most
valuable evaluation skill of all.
""")

md(r"""
## 🗺️ What you will build, step by step

| Step | What you do | Key tool |
| ---: | --- | --- |
| 0 | Set up the environment; load three API keys from `tutorials/.env` | `python-dotenv`, `uv` |
| 1 | RAGAS import stub + `nest_asyncio`; cost/safety note | `nest_asyncio` |
| 2 | Connect cloud Ollama: chat LLM, RAGAS judge, embeddings | `langchain-ollama`, `litellm` |
| 3 | Load 8-file corpus; build Qdrant in-memory vector store | `langchain-qdrant` |
| 4 | Build Cohere reranker; define `rag_answer()` | `cohere` |
| 5 | Define three `@tool` functions; assemble the MetalDesk ReAct agent | `langgraph`, `requests` |
| 6 | Load golden questions; build `EvaluationDataset` | `ragas.dataset_schema` |
| 7 | Score all 4 **retriever** metrics | `ragas` |
| 8 | Score all 3 **generator** metrics | `ragas` |
| 9 | Run agent on a multi-tool question; convert trace; score all 5 **agent** metrics | `ragas` |
| 10 | **MDD loop**: baseline vs. reranked comparison | `ragas.evaluate` |
| 11 | **Multi-hop spotlight**: inspect the 2 hardest golden questions | — |
| 12 | Full 12-metric scoreboard; NoiseSensitivity inversion reminder | `pandas` |
| Recap | Goodhart's Law cautions; where to go from here | — |

### 🎓 What you will *learn* (the concepts)

- How all 12 RAGAS metrics assemble into a three-tier evaluation framework
- How to run Metrics-Driven Development with a clean one-variable comparison
- Why multi-hop questions are the stress test for any retrieval improvement
- Goodhart's Law and why optimising a metric can destroy its usefulness
- The NoiseSensitivity inversion and the LLM-judge biases to watch for

### ✅ Prerequisites

- Modules 1–11 of this track (or the capstone notebook in `topics/06_rag_eval/`)
- Three API keys in `tutorials/.env`: `OLLAMA_API_KEY`, `COHERE_API_KEY`, `METALS_API_KEY`
- Curiosity about what happens when you put the whole system together
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
**from this module's folder** (`tutorials/12_capstone/`), using
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync            # reads pyproject.toml, creates .venv/, installs everything
uv run jupyter lab # launch Jupyter inside that environment
```

When the notebook opens, pick the kernel **`Python 3 (ipykernel)`** (top-right
kernel picker). That is the interpreter from `.venv`, so every `import` below
resolves against what `uv sync` installed.
""")

md(r"""
## 0.2 Provide your API keys (shared `.env`)

This module needs **three keys**. All twelve modules read their keys from a
**single** `.env` file in the `tutorials/` folder (the parent of this module).
Create `tutorials/.env` once (copy from `tutorials/.env.example`):

```
OLLAMA_API_KEY=...      # cloud Ollama — chat LLM, judge, embeddings
COHERE_API_KEY=...      # Cohere Rerank v3.5 (sharpens retrieval)
METALS_API_KEY=...      # Metals.dev live prices (agent tools)
```

`find_dotenv()` walks UP from this notebook and locates that shared file
automatically — you never copy keys into each module. `.env` is gitignored at
`tutorials/.gitignore`. **Never commit it.**

> 💡 **No keys?** No problem. The notebook detects missing keys and falls back
> to cached illustrative results in `frozen/` so you can follow every step
> without spending money or running live calls.
""")

code(r'''
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())          # resolves to tutorials/.env automatically

# Three keys: Ollama (LLM + judge + embeddings), Cohere (reranking), Metals.dev (live tools).
HAVE_OLLAMA  = bool(os.environ.get("OLLAMA_API_KEY"))
HAVE_COHERE  = bool(os.environ.get("COHERE_API_KEY"))
HAVE_METALS  = bool(os.environ.get("METALS_API_KEY"))
HAVE_KEYS    = HAVE_OLLAMA and HAVE_COHERE and HAVE_METALS

if HAVE_KEYS:
    print("All three API keys found — notebook will run live calls.")
else:
    missing = [k for k, v in [("OLLAMA_API_KEY", HAVE_OLLAMA),
                               ("COHERE_API_KEY", HAVE_COHERE),
                               ("METALS_API_KEY", HAVE_METALS)] if not v]
    print(f"Missing keys: {', '.join(missing)}")
    print("This notebook will use cached results in frozen/ so you can follow along.")
''')

# ============================================================================
# STEP 1 — RAGAS STUB + COST NOTE
# ============================================================================
md(r"""
---
# Step 1 · RAGAS import stub + event-loop setup

## 1.1 RAGAS 0.4.3 compatibility shim

RAGAS 0.4.3 hard-imports a module that `langchain-community` 1.x removed. The
LiteLLM path we use never touches it, so we stub it before importing RAGAS.
You first encountered this in Module 6.
""")

code(r'''
import sys
import types

# RAGAS 0.4.3 compatibility shim — stub the removed langchain_community module
# before ragas imports it. This is safe: the litellm path never calls ChatVertexAI.
_vx = types.ModuleType("langchain_community.chat_models.vertexai")

class ChatVertexAI:   # placeholder, intentionally non-functional
    pass

_vx.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vx
print("RAGAS compatibility stub installed.")
''')

md(r"""
## 1.2 Allow async in notebooks
""")

code(r'''
import nest_asyncio
nest_asyncio.apply()
print("nest_asyncio applied — async/await cells will work inside Jupyter.")
''')

md(r"""
## ⚠ Cost and safety note

This capstone touches **three paid external services**:

| Service | Used for | Billed per |
| --- | --- | --- |
| Cloud Ollama | Chat LLM (agent brain), RAGAS judge, embeddings | Tokens |
| Cohere Rerank v3.5 | Reranking inside `search_metal_knowledge` and `rag_answer` | API call |
| Metals.dev | Live spot prices (`get_metal_price`, `convert_currency`) | API call |

**Practical tips:**
- Run each live section once, then reuse the printed output for exploration.
- If any key is missing the notebook falls back to `frozen/` cached results.
- Agent runs are nondeterministic — the same question may trigger different tool
  sequences on different runs.
- NEVER hard-code API keys in notebook cells. Always read from `os.environ`.
""")

# ============================================================================
# STEP 2 — CONNECT CLOUD OLLAMA
# ============================================================================
md(r"""
---
# Step 2 · Connect cloud Ollama

We need three model objects (first assembled in Module 5, reused throughout):

- **`chat_llm`** — the agent's brain and RAG generator (ChatOllama, low temperature)
- **`judge_llm`** — the RAGAS judge (*different* model; Module 9 explains why)
- **`ragas_embeddings`** — for ResponseRelevancy's cosine-similarity step
""")

code(r'''
import os
import litellm
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11434")

LLM_NAME_OLLAMA       = "nemotron-3-super:cloud"
EMBEDDING_NAME_OLLAMA = "qwen3-embedding:0.6b"
LLM_MODEL             = "ollama_chat/nemotron-3-super:cloud"
JUDGE_MODEL           = "ollama_chat/gemma4:31b-cloud"
EMBEDDING_MODEL       = "ollama/qwen3-embedding:0.6b"

if HAVE_OLLAMA:
    chat_llm = ChatOllama(
        model=LLM_NAME_OLLAMA,
        base_url=os.environ["OLLAMA_API_BASE"],
        temperature=0.0,
    )
    lc_embeddings = OllamaEmbeddings(
        model=EMBEDDING_NAME_OLLAMA,
        base_url=os.environ["OLLAMA_API_BASE"],
    )
    judge_llm = llm_factory(
        JUDGE_MODEL, provider="litellm", client=litellm.completion, temperature=0.0
    )
    ragas_embeddings = embedding_factory(
        "litellm", model=EMBEDDING_MODEL, api_base=os.environ["OLLAMA_API_BASE"]
    )
    print(f"LLM   : {LLM_NAME_OLLAMA}")
    print(f"Judge : {JUDGE_MODEL}  (different model — avoids self-preference bias)")
    print(f"Embed : {EMBEDDING_NAME_OLLAMA}")
else:
    chat_llm = lc_embeddings = judge_llm = ragas_embeddings = None
    print("Ollama key missing — model objects set to None; frozen/ fallback will be used.")
''')

# ============================================================================
# STEP 3 — CORPUS + VECTOR STORE
# ============================================================================
md(r"""
---
# Step 3 · Load corpus and build the vector store

The same 8-file metals corpus you chunked in Module 4, embedded in Module 5, and
have been using ever since. Nothing new here — this is the carry-over foundation.
""")

code(r'''
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

raw_docs = [
    {"source": p.name, "page_content": p.read_text()}
    for p in sorted(Path("corpus").glob("*.md"))
]
print(f"Loaded {len(raw_docs)} documents:")
for d in raw_docs:
    print(f"  - {d['source']} ({len(d['page_content'])} chars)")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
lc_docs = [
    Document(page_content=piece, metadata={"source": d["source"]})
    for d in raw_docs
    for piece in splitter.split_text(d["page_content"])
]
print(f"\nSplit into {len(lc_docs)} chunks (size=500, overlap=60).")
''')

code(r'''
from langchain_qdrant import QdrantVectorStore

if HAVE_OLLAMA:
    vector_store = QdrantVectorStore.from_documents(
        lc_docs, embedding=lc_embeddings, location=":memory:",
        collection_name="metals_kb",
    )
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    test_hits = base_retriever.invoke("what drives the gold price?")
    print(f"Vector store ready. Quick retrieval test ({len(test_hits)} hits):")
    print(f"  Top chunk: {test_hits[0].page_content[:100]}...")
else:
    vector_store = base_retriever = None
    print("(Skipping vector store — no Ollama key. Using frozen fallback.)")
''')

# ============================================================================
# STEP 4 — COHERE RERANKER + RAG ANSWER
# ============================================================================
md(r"""
---
# Step 4 · Cohere reranker and `rag_answer()`

The reranker (Module 10) and the complete `rag_answer()` function. The function
takes a `use_rerank` flag so we can switch it on and off cleanly in the MDD loop
(Step 10).

The pattern: **retrieve wide** (k=10) → **rerank narrow** (top_n=3) → **generate**.
""")

code(r'''
import json
import cohere

if HAVE_COHERE:
    co = cohere.ClientV2(os.environ["COHERE_API_KEY"])
    def rerank(query: str, docs: list[str], top_n: int = 3) -> list[tuple[str, float]]:
        """Return (text, relevance_score) pairs for the top_n most relevant docs."""
        if not docs:
            return []
        result = co.rerank(model="rerank-v3.5", query=query, documents=docs, top_n=top_n)
        return [(docs[r.index], r.relevance_score) for r in result.results]
    print("Cohere reranker ready (rerank-v3.5).")
else:
    def rerank(query: str, docs: list[str], top_n: int = 3) -> list[tuple[str, float]]:
        """Frozen fallback: return the first top_n docs with a placeholder score."""
        return [(d, 0.9 - i * 0.05) for i, d in enumerate(docs[:top_n])]
    print("(Cohere key missing — rerank() returns the first top_n docs unscored.)")
''')

code(r'''
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a precise metals-markets tutor. Answer using ONLY the context passages. "
    "If the context does not contain the answer, say you do not know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

def rag_answer(question: str, k: int = 10, top_n: int = 3,
               use_rerank: bool = True) -> dict:
    """Run the full RAG pipeline: retrieve wide → rerank narrow → generate."""
    if not HAVE_OLLAMA:
        frozen = json.load(open("frozen/mdd_baseline.json"))
        # Return a plausible frozen structure so downstream code runs.
        return {
            "response": "(frozen — set OLLAMA_API_KEY to run live)",
            "retrieved_contexts": ["(frozen context — set keys to run live)"],
        }
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    candidates = [d.page_content for d in retriever.invoke(question)]
    if use_rerank and HAVE_COHERE:
        contexts = [t for t, _ in rerank(question, candidates, top_n=top_n)]
    else:
        contexts = candidates[:top_n]
    block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    response = chat_llm.invoke(
        RAG_PROMPT.format_messages(context=block, question=question)
    ).content.strip()
    return {"response": response, "retrieved_contexts": contexts}

print("rag_answer() defined — use_rerank=True uses Cohere, False uses vector-search top-k.")
''')

# ============================================================================
# STEP 5 — TOOLS + AGENT
# ============================================================================
md(r"""
---
# Step 5 · Three `@tool` functions and the MetalDesk agent

The three tools and the ReAct agent from Module 11. The agent decides at runtime
which tool to call, in what order, and how many times — this is the "agentic"
part of the architecture.
""")

code(r'''
import requests
from langchain_core.tools import tool

METALS_BASE = "https://api.metals.dev/v1"

@tool
def get_metal_price(metal: str, currency: str = "USD") -> str:
    """Get the current spot price of a precious or industrial metal.

    Args:
        metal: one of gold, silver, platinum, palladium, copper, aluminum, nickel, lead, zinc.
        currency: a 3-letter currency code such as USD, EUR, GBP. Defaults to USD.
    Returns a short sentence with the live price per troy ounce.
    """
    if not HAVE_METALS:
        frozen = json.load(open("frozen/agent_trace_and_metrics.json"))
        return frozen["tool_smoke_test"]["get_metal_price_gold"]
    resp = requests.get(
        f"{METALS_BASE}/metal/spot",
        params={"api_key": os.environ["METALS_API_KEY"],
                "metal": metal.lower(), "currency": currency.upper()},
        timeout=20,
    )
    data = resp.json()
    if data.get("status") != "success":
        return f"Could not fetch price for {metal}: {data.get('error_message', 'unknown error')}."
    return f"The current spot price of {metal} is {data['rate']['price']} {currency.upper()} per troy ounce."


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount of money from one currency to another using live rates.

    Args:
        amount: how much to convert, e.g. 100.
        from_currency: source 3-letter code, e.g. USD.
        to_currency: target 3-letter code, e.g. EUR.
    """
    if not HAVE_METALS:
        frozen = json.load(open("frozen/agent_trace_and_metrics.json"))
        return frozen["tool_smoke_test"]["convert_currency_100usd_eur"]
    resp = requests.get(
        f"{METALS_BASE}/currencies",
        params={"api_key": os.environ["METALS_API_KEY"], "base": "USD"},
        timeout=20,
    )
    data = resp.json()
    rates = data.get("currencies", {})
    fr, to = from_currency.upper(), to_currency.upper()
    if fr not in rates or to not in rates:
        return f"Unsupported currency: {from_currency} or {to_currency}."
    usd_value = amount * rates[fr]
    converted = usd_value / rates[to]
    return f"{amount} {fr} is about {converted:.2f} {to}."


@tool
def search_metal_knowledge(query: str) -> str:
    """Look up background knowledge about metals markets: how prices work, what drives them,
    investment vehicles, risk, and futures. Use this for conceptual or explanatory questions,
    NOT for live prices. Returns the most relevant passages from the knowledge base.
    """
    if not (HAVE_OLLAMA and HAVE_COHERE):
        frozen = json.load(open("frozen/agent_trace_and_metrics.json"))
        return frozen["tool_smoke_test"]["search_metal_knowledge_snippet"]
    candidates = [d.page_content for d in base_retriever.invoke(query)]
    top = rerank(query, candidates, top_n=3)
    return "\n\n".join(f"[Passage {i}] {text}" for i, (text, _) in enumerate(top, 1))


tools = [get_metal_price, convert_currency, search_metal_knowledge]
print("Tools defined:", [t.name for t in tools])
''')

code(r'''
from langgraph.prebuilt import create_react_agent

SYSTEM_PROMPT = (
    "You are MetalDesk, a precious-metals research assistant. "
    "You help users with precious and industrial metals: live prices, currency conversions, "
    "and background knowledge about how metals markets work, what drives prices, ways to invest, "
    "risk, and futures. "
    "Use get_metal_price for live prices, convert_currency for currency math, and "
    "search_metal_knowledge for conceptual or explanatory questions. "
    "If a request is outside metals and markets (for example cooking, sports, or medical advice), "
    "politely decline and explain that it is outside your scope. "
    "Ground your answers in tool results and do not invent prices."
)

if HAVE_OLLAMA:
    agent = create_react_agent(model=chat_llm, tools=tools, prompt=SYSTEM_PROMPT)
    print("MetalDesk agent ready (create_react_agent + 3 tools).")
else:
    agent = None
    print("(Agent skipped — no Ollama key. Steps 9+ will use frozen/ fallback.)")
''')

# ============================================================================
# STEP 6 — GOLDEN QUESTIONS + DATASET
# ============================================================================
md(r"""
---
# Step 6 · Load golden questions and build the evaluation dataset

The 8-question golden test set from `golden_questions.json` — 6 single-hop
questions and **2 multi-hop** questions (first introduced in Module 6, used as
the stress test throughout). We build a `SingleTurnSample` for each question
by running `rag_answer()` with reranking on.

> ⚠ **Cost note:** building the dataset makes 8 calls to the RAG pipeline
> (retrieval + reranking + generation). Use the frozen fallback if keys are
> missing.
""")

code(r'''
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

with open("golden_questions.json") as f:
    golden = json.load(f)

print(f"Loaded {len(golden)} golden questions:")
for g in golden:
    hop = g.get("hop", "single")
    star = "★ MULTI-HOP" if hop == "multi" else ""
    print(f"  [{hop}] {star} {g['question'][:70]}...")

# Identify multi-hop questions for the spotlight in Step 11.
multi_hop_qs = [g for g in golden if g.get("hop") == "multi"]
print(f"\n{len(multi_hop_qs)} multi-hop questions will be spotlighted in Step 11.")
''')

code(r'''
def build_dataset(use_rerank: bool, k: int = 10,
                  top_n: int = 3) -> EvaluationDataset:
    """Run the RAG pipeline over the golden set and package results for RAGAS."""
    rows = []
    for g in golden:
        out = rag_answer(g["question"], k=k, top_n=top_n, use_rerank=use_rerank)
        rows.append(SingleTurnSample(
            user_input=g["question"],
            response=out["response"],
            retrieved_contexts=out["retrieved_contexts"],
            reference=g["reference"],
        ))
    return EvaluationDataset(samples=rows)

if HAVE_KEYS:
    eval_dataset = build_dataset(use_rerank=True, k=10, top_n=3)
    print(f"Built evaluation dataset with {len(eval_dataset.samples)} samples.")
    # Illustrative output from one sample:
    s0 = eval_dataset.samples[0]
    print(f"\nExample:\n  Q: {s0.user_input}")
    print(f"  A (first 100 chars): {s0.response[:100]}")
    print(f"  Contexts: {len(s0.retrieved_contexts)} passages")
else:
    eval_dataset = None
    print("(No keys — eval_dataset is None. Each scoring step uses frozen/ fallback.)")
''')

# ============================================================================
# STEP 7 — RETRIEVER METRICS
# ============================================================================
md(r"""
---
# Step 7 · Retriever metrics (Tier 1)

You built these in Modules 7 and 8. Now we score all four at once.

| Metric | What it grades | Higher = better? |
| --- | --- | --- |
| `LLMContextPrecisionWithReference` | Relevant chunks ranked near top | ✓ |
| `LLMContextRecall` | Did retrieval find all needed passages? | ✓ |
| `ContextEntityRecall` | Named entities from reference present in context? | ✓ |
| `NoiseSensitivity` | Fraction of claims traceable to noise | **✗ LOWER IS BETTER** |

![Retriever metrics](slides/assets/04_context_precision_recall.svg)
""")

code(r'''
from ragas import evaluate
from ragas.metrics import (
    LLMContextPrecisionWithReference, LLMContextRecall,
    ContextEntityRecall, NoiseSensitivity,
)

retriever_metrics = [
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
    ContextEntityRecall(),
    NoiseSensitivity(),       # LOWER IS BETTER — read this column carefully!
]

if HAVE_KEYS and eval_dataset is not None:
    retriever_results = evaluate(
        dataset=eval_dataset,
        metrics=retriever_metrics,
        llm=judge_llm,
        embeddings=ragas_embeddings,
    )
    print("Retriever metric scores:")
    print(retriever_results)
else:
    retriever_results = json.load(open("frozen/full_scoreboard.json"))
    r = retriever_results["retriever_tier"]
    print("(using cached illustrative scores — set keys in tutorials/.env to run live)")
    print(f"  llm_context_precision_with_reference : {r['llm_context_precision_with_reference']}")
    print(f"  context_recall                       : {r['context_recall']}")
    print(f"  context_entity_recall                : {r['context_entity_recall']}")
    print(f"  noise_sensitivity                    : {r['noise_sensitivity']}  ← LOWER IS BETTER")
    print(f"\n  Note: {r['note_noise_sensitivity']}")
''')

md(r"""
> ⚠ **NoiseSensitivity is the one inverted metric.** Every other number in this
> scoreboard rewards a higher value. Noise sensitivity measures the fraction of
> claims that are wrong and traceable to retrieved noise — 0 is perfect.
> Always relabel this column deliberately when you are scanning a results table.
""")

# ============================================================================
# STEP 8 — GENERATOR METRICS
# ============================================================================
md(r"""
---
# Step 8 · Generator metrics (Tier 2)

You built these in Module 9. These grade what the LLM did with the context.

| Metric | What it grades | Key caution |
| --- | --- | --- |
| `Faithfulness` | Every claim grounded in context? | Faithful ≠ correct |
| `ResponseRelevancy` | Answer actually addresses the question? | Rewards on-topic, not right |
| `FactualCorrectness` | Claims match ground-truth reference? | Only metric comparing to truth |

![Faithfulness](slides/assets/07_faithfulness.svg)
""")

code(r'''
from ragas.metrics import Faithfulness, ResponseRelevancy, FactualCorrectness

generator_metrics = [
    Faithfulness(),
    ResponseRelevancy(),
    FactualCorrectness(),
]

if HAVE_KEYS and eval_dataset is not None:
    generator_results = evaluate(
        dataset=eval_dataset,
        metrics=generator_metrics,
        llm=judge_llm,
        embeddings=ragas_embeddings,
    )
    print("Generator metric scores:")
    print(generator_results)
else:
    generator_results = json.load(open("frozen/full_scoreboard.json"))
    g = generator_results["generator_tier"]
    print("(using cached illustrative scores — set keys in tutorials/.env to run live)")
    print(f"  faithfulness       : {g['faithfulness']}")
    print(f"  response_relevancy : {g['response_relevancy']}")
    print(f"  factual_correctness: {g['factual_correctness']}")
''')

md(r"""
*Illustrative output (your numbers will vary):*
```
faithfulness        : 0.89
response_relevancy  : 0.91
factual_correctness : 0.74
```

> ⚠ **Faithful ≠ correct.** A system that retrieves a wrong passage and faithfully
> repeats it scores 1.0 on faithfulness and low on factual correctness. Both
> numbers together tell the story; either one alone misleads.
""")

# ============================================================================
# STEP 9 — AGENT METRICS
# ============================================================================
md(r"""
---
# Step 9 · Agent metrics (Tier 3)

You built these in Module 11. Here we assemble all five in the capstone context.
The agent answers a question that requires two tool calls — one live price lookup
and one knowledge search.

![Agentic RAG](slides/assets/01_agentic_rag_architecture.svg)
""")

code(r'''
from ragas.integrations.langgraph import convert_to_ragas_messages
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall

AGENT_QUERY = "What is the current price of platinum in USD, and why is its supply risky?"

def run_agent(question: str) -> dict:
    """Invoke the agent and return its full result dict."""
    if HAVE_KEYS and agent is not None:
        return agent.invoke({"messages": [{"role": "user", "content": question}]})
    else:
        frozen = json.load(open("frozen/agent_trace_and_metrics.json"))
        print("(using cached illustrative agent trace — set all three keys to run live)")
        return {"messages": frozen["agent_run"]["messages"], "_frozen": True}

agent_result = run_agent(AGENT_QUERY)

# Convert trace to RAGAS message format
if HAVE_KEYS and not agent_result.get("_frozen"):
    ragas_trace = convert_to_ragas_messages(agent_result["messages"])
else:
    # Build an illustrative RAGAS trace from the frozen data.
    ragas_trace = [
        HumanMessage(content=AGENT_QUERY),
        AIMessage(
            content="Let me fetch the live price and look up the supply context.",
            tool_calls=[
                ToolCall(name="get_metal_price", args={"metal": "platinum", "currency": "USD"}),
                ToolCall(name="search_metal_knowledge", args={"query": "platinum supply risk South Africa"}),
            ],
        ),
        ToolMessage(content="The current spot price of platinum is 1012.50 USD per troy ounce."),
        ToolMessage(content="[Passage 1] Platinum supply is concentrated in South Africa (~70%) and Russia (~12%), making it acutely sensitive to labour unrest, power shortages, and geopolitical shocks."),
        AIMessage(content="Platinum is currently trading at $1,012.50 USD per troy ounce. Its supply is risky because roughly 70% comes from South Africa and ~12% from Russia, exposing it to strikes, power shortages, and sanctions."),
    ]

print(f"Trace: {len(ragas_trace)} messages.")
for m in ragas_trace:
    kind = type(m).__name__
    extra = ""
    if getattr(m, "tool_calls", None):
        extra = " -> calls: " + ", ".join(tc.name for tc in m.tool_calls)
    print(f"  {kind}{extra}")
''')

code(r'''
from ragas.metrics import (
    TopicAdherenceScore,
    ToolCallAccuracy, ToolCallF1,
    AgentGoalAccuracyWithReference, AgentGoalAccuracyWithoutReference,
)

allowed_topics = ["precious metals", "metals markets", "commodity prices", "investing in metals"]

# Build a two-turn sample: one in-scope (answered), one out-of-scope (declined).
convo = [
    HumanMessage(content="What is the current price of platinum in USD, and why is its supply risky?"),
    AIMessage(
        content="Fetching price and knowledge...",
        tool_calls=[
            ToolCall(name="get_metal_price", args={"metal": "platinum", "currency": "USD"}),
            ToolCall(name="search_metal_knowledge", args={"query": "platinum supply risk"}),
        ],
    ),
    ToolMessage(content="The current spot price of platinum is 1012.50 USD per troy ounce."),
    ToolMessage(content="[Passage 1] Platinum supply is concentrated in South Africa (~70%) and Russia (~12%)."),
    AIMessage(content="Platinum is $1,012.50 per troy ounce. Supply is risky due to concentration in South Africa and Russia."),
    HumanMessage(content="Can you recommend a good recipe for chicken soup?"),
    AIMessage(content="That is outside my scope. I can only help with metals and markets."),
]
ta_sample = MultiTurnSample(user_input=convo, reference_topics=allowed_topics)

expected_calls = [
    ToolCall(name="get_metal_price", args={"metal": "platinum", "currency": "USD"}),
    ToolCall(name="search_metal_knowledge", args={"query": "platinum supply risk South Africa"}),
]
tool_sample = MultiTurnSample(user_input=ragas_trace, reference_tool_calls=expected_calls)
goal_sample = MultiTurnSample(
    user_input=ragas_trace,
    reference="Report the current platinum price in USD and explain why its supply is risky.",
)

if HAVE_OLLAMA and judge_llm is not None:
    ta_f1 = await TopicAdherenceScore(llm=judge_llm, mode="f1").multi_turn_ascore(ta_sample)
    tca   = await ToolCallAccuracy().multi_turn_ascore(tool_sample)
    tcf1  = await ToolCallF1().multi_turn_ascore(tool_sample)
    aga_w = await AgentGoalAccuracyWithReference(llm=judge_llm).multi_turn_ascore(goal_sample)
    aga_n = await AgentGoalAccuracyWithoutReference(llm=judge_llm).multi_turn_ascore(
        MultiTurnSample(user_input=ragas_trace))
    print(f"TopicAdherenceScore (f1)         : {ta_f1:.2f}")
    print(f"ToolCallAccuracy                 : {tca:.2f}")
    print(f"ToolCallF1                       : {tcf1:.2f}")
    print(f"AgentGoalAccuracy (with ref)     : {aga_w}")
    print(f"AgentGoalAccuracy (without ref)  : {aga_n}")
else:
    frozen = json.load(open("frozen/agent_trace_and_metrics.json"))
    am = frozen["agent_metrics"]
    print("(using cached illustrative scores — set OLLAMA_API_KEY to run live)")
    print(f"TopicAdherenceScore (f1)         : {am['topic_adherence_f1']:.2f}")
    print(f"ToolCallAccuracy                 : {am['tool_call_accuracy']:.2f}")
    print(f"ToolCallF1                       : {am['tool_call_f1']:.2f}")
    print(f"AgentGoalAccuracy (with ref)     : {am['agent_goal_accuracy_with_reference']}")
    print(f"AgentGoalAccuracy (without ref)  : {am['agent_goal_accuracy_without_reference']}")
''')

# ============================================================================
# STEP 10 — MDD LOOP
# ============================================================================
md(r"""
---
# Step 10 · Metrics-Driven Development loop

This is the payoff of Module 6's MDD mindset. We change **exactly one thing**:
Cohere reranking on vs. off. Both configurations hand the generator the same
number of passages (`top_n=3`). Any improvement in the metrics is attributable
to the reranker.

![MDD loop](slides/assets/02_mdd_loop.svg)
""")

code(r'''
mdd_metrics = [
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
    Faithfulness(),
    FactualCorrectness(),
]

if HAVE_KEYS:
    print("Running baseline (use_rerank=False)...")
    baseline_results = evaluate(
        build_dataset(use_rerank=False, top_n=3),
        metrics=mdd_metrics,
        llm=judge_llm,
        embeddings=ragas_embeddings,
    )
    print("Running improved (use_rerank=True, k=10, top_n=3)...")
    improved_results = evaluate(
        build_dataset(use_rerank=True, k=10, top_n=3),
        metrics=mdd_metrics,
        llm=judge_llm,
        embeddings=ragas_embeddings,
    )
    print("Done.")
else:
    baseline_results = None
    improved_results = None
    print("(No keys — using frozen MDD data for comparison in the next cell.)")
''')

code(r'''
import pandas as pd

if HAVE_KEYS and baseline_results is not None:
    def means(results: object) -> dict:
        df = results.to_pandas()
        return {c: round(df[c].mean(), 3) for c in df.columns if df[c].dtype.kind in "fc"}

    base, imp = means(baseline_results), means(improved_results)
    comparison = pd.DataFrame({"baseline": base, "improved (rerank)": imp})
    comparison["change"] = (comparison["improved (rerank)"] - comparison["baseline"]).round(3)
    print("MDD comparison:")
    print(comparison.to_string())
else:
    base_f = json.load(open("frozen/mdd_baseline.json"))
    imp_f  = json.load(open("frozen/mdd_reranked.json"))
    mdd_f  = json.load(open("frozen/full_scoreboard.json"))["mdd_comparison"]
    print("(using cached illustrative MDD data — set keys in tutorials/.env to run live)")
    print()
    print(f"{'Metric':<42} {'Baseline':>10} {'Reranked':>10} {'Change':>8}")
    print("-" * 74)
    for metric, base_v, imp_v, chg in zip(
        mdd_f["metric"], mdd_f["baseline"], mdd_f["improved_rerank"], mdd_f["change"]
    ):
        sign = "+" if chg > 0 else ""
        print(f"{metric:<42} {base_v:>10.2f} {imp_v:>10.2f} {sign}{chg:>7.2f}")
''')

md(r"""
*Illustrative output (your numbers will vary):*

| metric | baseline | improved (rerank) | change |
| --- | --- | --- | --- |
| llm_context_precision_with_reference | 0.74 | 0.92 | +0.18 |
| context_recall | 0.69 | 0.84 | +0.15 |
| faithfulness | 0.83 | 0.89 | +0.06 |
| factual_correctness | 0.62 | 0.74 | +0.12 |

The biggest gains are on the **retriever side** (precision and recall), exactly
where a reranker should help. Generator metrics rose as a downstream consequence:
better context in, better answers out. This is the satisfying version of MDD —
a single well-chosen change moved the numbers in the direction the theory predicts.

> ⚠ **Goodhart's Law.** If you keep tweaking until one number goes up, you are
> fitting noise. With eight questions, a 0.02 difference is random variation.
> Prefer large, theory-aligned gaps and confirm wins by reading actual answers.
""")

# ============================================================================
# STEP 11 — MULTI-HOP SPOTLIGHT
# ============================================================================
md(r"""
---
# Step 11 · Multi-hop question spotlight ★

The golden set has 6 single-hop questions and **2 multi-hop** questions. Single-hop
questions can be answered from one corpus passage. Multi-hop questions require
synthesising information from **two separate** passages, which stresses every
retrieval metric.

These are the two hardest questions in the golden set.
""")

code(r'''
print("=" * 70)
print("MULTI-HOP QUESTIONS — the hardest retrieval cases in the golden set")
print("=" * 70)
for i, q in enumerate(multi_hop_qs, 1):
    print(f"\n[Multi-hop {i}]")
    print(f"  Q: {q['question']}")
    print(f"  Reference: {q['reference']}")
    print(f"  Requires contexts from:")
    for ctx in q.get("reference_contexts", []):
        print(f"    - {ctx[:80]}...")
print()
print("These questions consistently score lower on context_recall because")
print("the top-k window may not capture BOTH needed passages simultaneously.")
print("Widening k and reranking helps, but multi-hop retrieval remains hard.")
''')

code(r'''
# Show baseline vs. reranked scores specifically for multi-hop questions.
frozen_base = json.load(open("frozen/mdd_baseline.json"))
frozen_imp  = json.load(open("frozen/mdd_reranked.json"))

print("Illustrative faithfulness scores — baseline vs. reranked:\n")
print(f"{'Question':<55} {'Base FF':>8} {'Rerank FF':>10}")
print("-" * 76)
for b, r in zip(frozen_base["per_question"], frozen_imp["per_question"]):
    tag = " ★" if b.get("hop") == "multi" else ""
    print(f"{b['question'][:55]:<55} {b['faithfulness']:>8.2f} {r['faithfulness']:>10.2f}{tag}")

print()
print("★ = multi-hop question — note how reranking helps more on multi-hop cases")
print("    but they still score lower than single-hop questions on average.")
''')

# ============================================================================
# STEP 12 — FULL 12-METRIC SCOREBOARD
# ============================================================================
md(r"""
---
# Step 12 · The full 12-metric scoreboard

All three tiers assembled into one view. This is the complete evaluation picture
for the Agentic RAG system.

> ⚠ Remember: **NoiseSensitivity is the one inverted metric** — lower is better.
> Every other number rewards a higher value. Keep this in mind when scanning the
> table.
""")

code(r'''
# Assemble the three-tier scoreboard from the scored results (or frozen data).
sb = json.load(open("frozen/full_scoreboard.json"))
r_tier = sb["retriever_tier"]
g_tier = sb["generator_tier"]
a_tier = sb["agent_tier"]

rows = [
    # ---- Retriever tier ----
    ("RETRIEVER", "llm_context_precision_with_reference",  r_tier["llm_context_precision_with_reference"],  "↑ higher = better"),
    ("RETRIEVER", "context_recall",                        r_tier["context_recall"],                        "↑ higher = better"),
    ("RETRIEVER", "context_entity_recall",                 r_tier["context_entity_recall"],                 "↑ higher = better"),
    ("RETRIEVER", "noise_sensitivity",                     r_tier["noise_sensitivity"],                     "↓ LOWER = better"),
    # ---- Generator tier ----
    ("GENERATOR", "faithfulness",                          g_tier["faithfulness"],                          "↑ higher = better"),
    ("GENERATOR", "response_relevancy",                    g_tier["response_relevancy"],                    "↑ higher = better"),
    ("GENERATOR", "factual_correctness",                   g_tier["factual_correctness"],                   "↑ higher = better"),
    # ---- Agent tier ----
    ("AGENT",     "topic_adherence_f1",                    a_tier["topic_adherence_f1"],                    "↑ higher = better"),
    ("AGENT",     "tool_call_accuracy",                    a_tier["tool_call_accuracy"],                    "↑ higher = better"),
    ("AGENT",     "tool_call_f1",                          a_tier["tool_call_f1"],                          "↑ higher = better"),
    ("AGENT",     "agent_goal_accuracy_with_reference",    a_tier["agent_goal_accuracy_with_reference"],    "1 = success"),
    ("AGENT",     "agent_goal_accuracy_without_reference", a_tier["agent_goal_accuracy_without_reference"], "1 = success"),
]

print("(Illustrative scores — run with real keys for live values)\n")
print(f"{'Tier':<10} {'Metric':<44} {'Score':>7}  {'Direction'}")
print("-" * 78)
prev_tier = None
for tier, metric, score, direction in rows:
    if tier != prev_tier and prev_tier is not None:
        print()
    prev_tier = tier
    inv = "  ← INVERTED" if "lower" in direction.lower() else ""
    print(f"{tier:<10} {metric:<44} {score:>7.2f}  {direction}{inv}")
''')

# ============================================================================
# RECAP — GOODHART'S LAW + WHERE TO GO
# ============================================================================
md(r"""
---
# Recap: what you built in Module 12 (and in the full track)

### The full capstone system

1. **8-file metals corpus** loaded and chunked (500/60)
2. **Cloud Ollama embeddings** → Qdrant in-memory vector store
3. **Retrieve k=10 → Cohere rerank → top_n=3** context passages
4. **`rag_answer()`** — the complete RAG generation function with a `use_rerank` flag
5. **MetalDesk agent** — `create_react_agent` + 3 tools (price, currency, knowledge)
6. **All 12 RAGAS metrics** across 3 tiers (4 retriever + 3 generator + 5 agent)
7. **MDD loop** — baseline vs. reranked, comparing 4 metrics on 8 golden questions
8. **Multi-hop spotlight** — the 2 hardest questions that stress every retrieval metric

### The Goodhart's Law caution — carry it everywhere

When a metric becomes the target, it stops being a good metric.

- 8 questions is too few to tell signal from noise.
- Tweaking until one number goes up is fitting the test, not improving the system.
- A faithfulness of 1.0 is not a victory if the context was wrong.
- A tool-call accuracy of 0.0 is not a failure if the agent took a different valid path.
- Keep a human in the loop. Read the actual answers. One number never tells the whole story.

### Where to go from here

- **Extend the golden set** to 20–50 questions (more multi-hop) for statistical significance.
- **Try hybrid retrieval** (BM25 + dense) to improve multi-hop recall.
- **Swap the judge model** and compare scores to understand judge variance.
- **Add an adversarial test set** to stress NoiseSensitivity and Faithfulness.
- **Read every caution** in `topics/06_rag_eval/agentic_rag_evaluation_theory.md` — each
  one points to a real open research question.

The authoritative capstone with full live-run outputs:
`topics/06_rag_eval/agentic_rag_evaluation_tutorial.ipynb`

You have completed the 12-module ASDRP Agentic RAG Evaluation track.
""")

# ============================================================================
# EMIT NOTEBOOK  (do not change below except OUT)
# ============================================================================
def to_cell(kind: str, src: str) -> dict:
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

OUT = "12_capstone.ipynb"   # <-- this module's notebook filename
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

n_md = sum(1 for k, _ in CELLS if k == "md")
n_code = sum(1 for k, _ in CELLS if k == "code")
print(f"Wrote {OUT}: {len(CELLS)} cells ({n_md} markdown, {n_code} code)")
