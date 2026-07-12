"""Builds 11_rag_to_agent.ipynb from a list of (type, source) cells.

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
# Module 11 · From RAG to Agent + Agent Metrics

### A hands-on, build-it-yourself module for advanced high school researchers

![Agentic RAG architecture](slides/assets/01_agentic_rag_architecture.svg)

In Module 10 you sharpened retrieval with Cohere reranking. You had a clean,
high-quality RAG pipeline — but it was a **fixed sequence**: retrieve, rerank,
generate, always in that order, for every question. Module 11 breaks that
rigidity. You will wrap your retriever and two live-data tools inside a
**LangGraph ReAct agent** that *decides at runtime* which tool to call, in what
order, and how many times. Then you will add five RAGAS metrics designed
specifically for agent traces: `TopicAdherenceScore`, `ToolCallAccuracy`,
`ToolCallF1`, `AgentGoalAccuracyWithReference`, and
`AgentGoalAccuracyWithoutReference`. Module 12 (the capstone) assembles every
metric from the track into one MDD dashboard.

This is Module 11 of a twelve-part track that ends in a full **Agentic RAG
Evaluation** capstone.
""")

md(r"""
## 📋 Summary: the one-paragraph version

A **ReAct agent** runs a reason → act → observe loop: at each step the model
reads the conversation so far, decides whether to call a tool (and which one,
with what arguments), receives the tool output, then reasons again. LangGraph's
`create_react_agent` builds this loop for you. You define three `@tool`
functions — a live metals price fetcher, a currency converter, and a RAG
knowledge search — hand them to the agent, then invoke it with a question. The
agent produces a *trace*: a sequence of human, AI, and tool messages.
`convert_to_ragas_messages` translates that trace into the typed RAGAS format so
you can score topic adherence (did the agent stay in its domain?), tool-call
accuracy and F1 (did it call the right tools with the right arguments?), and
goal accuracy (did it actually help the user?).
""")

md(r"""
## 🗺️ What you will build, step by step

| Step | What you do | Key tool |
| ---: | --- | --- |
| 0 | Set up the environment; load three API keys from `tutorials/.env` | `python-dotenv`, `uv` |
| 1 | RAGAS import stub + `nest_asyncio`; read the cost/safety note | `nest_asyncio` |
| 2 | Connect cloud Ollama (chat LLM, RAGAS judge, embeddings) | `langchain-ollama`, `litellm` |
| 3 | Load corpus and build vector store | `langchain-qdrant` |
| 4 | Build the Cohere reranker | `cohere` |
| 5 | Define three `@tool` functions; smoke-test each in isolation | `langchain-core`, `requests` |
| 6 | Assemble the ReAct agent with `create_react_agent` | `langgraph` |
| 7 | Run the agent on a multi-tool question; pretty-print the trace | `langgraph` |
| 8 | Convert the trace with `convert_to_ragas_messages`; inspect RAGAS types | `ragas` |
| 9 | Score `TopicAdherenceScore` (precision, recall, F1) | `ragas` |
| 10 | Score `ToolCallAccuracy` | `ragas` |
| 11 | Score `ToolCallF1`; compare a clean trace vs. a noisy one | `ragas` |
| 12 | Score `AgentGoalAccuracyWithReference` and `WithoutReference` | `ragas` |
| Recap | Summary table of all five agent metrics; pointer to Module 12 | — |

### 🎓 What you will *learn* (the concepts)

- The **ReAct loop** (reason → act → observe) and how it differs from a fixed pipeline
- How `create_react_agent` builds the loop from a model + tools + system prompt
- Why **multi-turn evaluation** needs different metrics from single-turn RAG
- `TopicAdherenceScore` — measuring scope discipline
- `ToolCallAccuracy` (strict) vs. `ToolCallF1` (partial credit)
- `AgentGoalAccuracy` — judging the outcome, not the path

### ✅ Prerequisites

- Modules 5–10 (cloud Ollama, vector store, Cohere reranking, RAGAS single-turn metrics)
- Three API keys in `tutorials/.env`: `OLLAMA_API_KEY`, `COHERE_API_KEY`, `METALS_API_KEY`
- Curiosity about what the agent actually *does* when it decides for itself
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
**from this module's folder** (`tutorials/11_rag_to_agent/`), using
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
OLLAMA_API_KEY=...      # cloud Ollama — chat, judge, embeddings
COHERE_API_KEY=...      # Cohere reranking (used inside search_metal_knowledge)
METALS_API_KEY=...      # Metals.dev live prices (get_metal_price, convert_currency)
```

`find_dotenv()` walks UP from this notebook and locates that shared file
automatically — you never copy keys into each module. `.env` is gitignored
at `tutorials/.gitignore`. **Never commit it.**

> 💡 **No keys?** No problem. The notebook detects missing keys and falls back
> to cached illustrative results in `frozen/` so you can follow every step
> without spending money or running live calls.
""")

code(r'''
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())          # resolves to tutorials/.env automatically

# This module uses three keys: Ollama (LLM + judge + embeddings),
# Cohere (reranking inside search_metal_knowledge), and Metals.dev (live prices).
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
# STEP 1 — RAGAS STUB + nest_asyncio + COST NOTE
# ============================================================================
md(r"""
---
# Step 1 · RAGAS import stub + event-loop setup

## 1.1 Why RAGAS needs a compatibility stub

RAGAS 0.4.3 hard-imports a module that `langchain-community` 1.x removed.
The LiteLLM path we use never touches it, so we stub it out *before* importing
RAGAS. This is a one-time fix — future RAGAS releases will not need it.
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

Jupyter already runs an event loop, and RAGAS metrics use `asyncio`.
`nest_asyncio.apply()` lets you call `await` directly in a notebook cell
without restarting the kernel.
""")

code(r'''
import nest_asyncio
nest_asyncio.apply()
print("nest_asyncio applied — async/await cells will work inside Jupyter.")
''')

md(r"""
## ⚠ Cost and safety note

This module touches **three paid external services**:

| Service | Used for | Billed per |
| --- | --- | --- |
| Cloud Ollama | Chat LLM (agent brain), RAGAS judge, embeddings | Tokens |
| Cohere Rerank v3.5 | Reranking inside `search_metal_knowledge` | API call |
| Metals.dev | Live spot prices (`get_metal_price`, `convert_currency`) | API call |

**Practical tips:**
- Run live cells once, then reuse the printed output for exploration.
- If a key is missing the notebook falls back to `frozen/` cached results.
- Agent runs are nondeterministic — the same question may trigger different tool
  sequences and incur different costs on different runs.
- NEVER hard-code API keys in notebook cells. Always read from `os.environ`.
""")

# ============================================================================
# STEP 2 — CONNECT CLOUD OLLAMA
# ============================================================================
md(r"""
---
# Step 2 · Connect cloud Ollama

We need three model objects:
- **`chat_llm`** — the agent's "brain" (ChatOllama, low temperature for reliable tool-calling)
- **`judge_llm`** — the RAGAS judge (a different, slower model via LiteLLM)
- **`ragas_embeddings`** — RAGAS embeddings for semantic similarity

These mirror the setup from Module 5 onward. The models whose names end in
`:cloud` are routed by the local Ollama daemon to the cloud.
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
        temperature=0.0,   # deterministic tool-calling
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
    print(f"Judge : {JUDGE_MODEL}")
    print(f"Embed : {EMBEDDING_NAME_OLLAMA}")
else:
    chat_llm = None
    lc_embeddings = None
    judge_llm = None
    ragas_embeddings = None
    print("Ollama key missing — model objects set to None; frozen/ fallback will be used.")
''')

# ============================================================================
# STEP 3 — LOAD CORPUS + VECTOR STORE
# ============================================================================
md(r"""
---
# Step 3 · Load corpus and build the vector store

This step is identical to Module 10 (carry-over). We load the 8-file metals
corpus, chunk it, embed it, and store it in Qdrant in-memory.

> 💡 If `lc_embeddings` is None (no Ollama key), we skip the vector store and
> the knowledge-search tool will fall back to frozen output.
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
    # Quick sanity-check retrieval
    test_hits = base_retriever.invoke("what drives the gold price?")
    print(f"Vector store ready. Quick retrieval test ({len(test_hits)} hits):")
    print(f"  Top chunk: {test_hits[0].page_content[:100]}...")
else:
    vector_store = None
    base_retriever = None
    print("(Skipping vector store — no Ollama key. Using frozen fallback.)")
''')

# ============================================================================
# STEP 4 — COHERE RERANKER
# ============================================================================
md(r"""
---
# Step 4 · Build the Cohere reranker

`search_metal_knowledge` calls `rerank()` internally, so we define it here.
This is the same reranker from Module 10.
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
    print("(Cohere key missing — rerank() will return the first top_n docs unscored.)")
''')

# ============================================================================
# STEP 5 — DEFINE TOOLS + SMOKE TEST
# ============================================================================
md(r"""
---
# Step 5 · Define the three MetalDesk tools

An agent is only as capable as the tools it can call. We define three using
LangChain's `@tool` decorator. Each tool is a regular Python function with a
docstring — the agent reads that docstring to decide *when* to call the tool.

The three tools:
- **`get_metal_price`** — live spot price from Metals.dev REST API
- **`convert_currency`** — live currency conversion from Metals.dev
- **`search_metal_knowledge`** — vector search + Cohere rerank over the corpus

> ⚠ **Caution:** each live tool call costs money and network time. In a
> production agent you would add caching and timeouts. Here we keep it simple.
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
        frozen = json.load(open("frozen/tool_smoke_test.json"))
        return frozen["get_metal_price_gold"]
    resp = requests.get(
        f"{METALS_BASE}/metal/spot",
        params={
            "api_key": os.environ["METALS_API_KEY"],
            "metal": metal.lower(),
            "currency": currency.upper(),
        },
        timeout=20,
    )
    data = resp.json()
    if data.get("status") != "success":
        return f"Could not fetch price for {metal}: {data.get('error_message', 'unknown error')}."
    price = data["rate"]["price"]
    return f"The current spot price of {metal} is {price} {currency.upper()} per troy ounce."


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount of money from one currency to another using live rates.

    Args:
        amount: how much to convert, e.g. 100.
        from_currency: source 3-letter code, e.g. USD.
        to_currency: target 3-letter code, e.g. EUR.
    """
    if not HAVE_METALS:
        frozen = json.load(open("frozen/tool_smoke_test.json"))
        return frozen["convert_currency_100usd_eur"]
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
        frozen = json.load(open("frozen/tool_smoke_test.json"))
        return frozen["search_metal_knowledge_snippet"]
    candidates = [d.page_content for d in base_retriever.invoke(query)]
    top = rerank(query, candidates, top_n=3)
    return "\n\n".join(f"[Passage {i}] {text}" for i, (text, _) in enumerate(top, 1))


tools = [get_metal_price, convert_currency, search_metal_knowledge]
print("Tools defined:", [t.name for t in tools])
''')

md(r"""
## 5.1 Smoke-test each tool in isolation

Always test tools *before* wiring them into the agent. A broken tool causes
confusing agent behaviour that is hard to debug from the trace alone.
""")

code(r'''
# Smoke-test each tool directly (bypassing the agent) so we know they work in isolation.
# With live keys these call the real APIs; without keys they return frozen strings.
print("--- get_metal_price ---")
print(get_metal_price.invoke({"metal": "gold"}))

print("\n--- convert_currency ---")
print(convert_currency.invoke({"amount": 100, "from_currency": "USD", "to_currency": "EUR"}))

print("\n--- search_metal_knowledge (first 200 chars) ---")
snippet = search_metal_knowledge.invoke({"query": "what makes platinum supply risky?"})
print(snippet[:200], "...")
''')

# ============================================================================
# STEP 6 — ASSEMBLE THE AGENT
# ============================================================================
md(r"""
---
# Step 6 · Assemble the ReAct agent with `create_react_agent`

`create_react_agent` builds the reason → act → observe loop for you. You supply:
- **`model`** — the chat LLM (the agent's "brain")
- **`tools`** — the list of `@tool` functions it can call
- **`prompt`** — a system prompt that tells it its role and constraints

The agent reads the tool docstrings at every loop iteration to decide which
tool (if any) to call next.

> ⚠ **Key insight:** `create_react_agent` makes no API call at build time.
> The LLM is invoked only when `.invoke()` runs — so building the agent is free.
""")

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
    print("Agent ready.")
else:
    agent = None
    print("(Agent skipped — no Ollama key. Steps 7–8 will use frozen/agent_run.json.)")
''')

# ============================================================================
# STEP 7 — RUN THE AGENT
# ============================================================================
md(r"""
---
# Step 7 · Run the agent on a multi-tool question

We ask a question that should trigger two `get_metal_price` calls. The agent
reasons, calls both tools, reads the results, and writes a final answer. We
pretty-print the full message trace so you can see every step.

> ⚠ **Nondeterminism:** the same question can produce a different tool sequence
> on different runs. LLMs are probabilistic, even at temperature 0.
""")

code(r'''
def run_agent(question: str) -> dict:
    """Invoke the agent and return its full result (a dict with a 'messages' list)."""
    if HAVE_KEYS and agent is not None:
        return agent.invoke({"messages": [{"role": "user", "content": question}]})
    else:
        # Frozen fallback — return the illustrative trace as a plausible dict.
        frozen = json.load(open("frozen/agent_run.json"))
        print("(using cached illustrative agent trace — set all three keys to run live)")
        # Wrap frozen messages in a dict to match the real structure.
        return {"messages": frozen["messages"], "_frozen": True}

QUERY = "What are the current prices of gold and silver in US dollars?"
result = run_agent(QUERY)

# Pretty-print the trace when running live (frozen dict has no pretty_print method).
if not result.get("_frozen"):
    for m in result["messages"]:
        m.pretty_print()
else:
    for m in result["messages"]:
        role = m.get("type", "?")
        content = m.get("content", "")
        calls = m.get("tool_calls", [])
        if calls:
            call_str = ", ".join(f'{c["name"]}({c["args"]})' for c in calls)
            print(f"[{role}] (tool calls: {call_str})")
        else:
            print(f"[{role}] {content[:120]}")
''')

md(r"""
*Illustrative output (live values will differ):*

```
================================ Human Message =================================
What are the current prices of gold and silver in US dollars?

================================== Ai Message ==================================
Tool Calls:
  get_metal_price (id: ...)
  Call ID: ...
    Args:
      metal: gold
      currency: USD
  get_metal_price (id: ...)
  Call ID: ...
    Args:
      metal: silver
      currency: USD

================================= Tool Message =================================
Name: get_metal_price
The current spot price of gold is 2998.41 USD per troy ounce.

================================= Tool Message =================================
Name: get_metal_price
The current spot price of silver is 33.72 USD per troy ounce.

================================== Ai Message ==================================
Here are the current precious metal spot prices:
- **Gold**: $2,998.41 USD per troy ounce
- **Silver**: $33.72 USD per troy ounce
```
""")

# ============================================================================
# STEP 8 — CONVERT TRACE TO RAGAS MESSAGES
# ============================================================================
md(r"""
---
# Step 8 · Convert the trace with `convert_to_ragas_messages`

Single-turn RAGAS metrics expect a `(question, answer, contexts)` triple.
Agent traces are *multi-turn*: the conversation weaves human turns, model
reasoning, tool calls, and tool results. RAGAS represents this with typed
message objects:

| RAGAS type | Corresponds to |
| --- | --- |
| `HumanMessage` | User's question |
| `AIMessage` | Model response; may carry `tool_calls` list |
| `ToolMessage` | String output from a tool |
| `ToolCall` | A named call with `args` dict embedded in an `AIMessage` |

`convert_to_ragas_messages` does the translation from LangGraph's internal
message format to these RAGAS types.
""")

code(r'''
from ragas.integrations.langgraph import convert_to_ragas_messages
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall

if HAVE_KEYS and not result.get("_frozen"):
    ragas_trace = convert_to_ragas_messages(result["messages"])
    print(f"Converted {len(ragas_trace)} messages. Types in order:")
    for m in ragas_trace:
        kind = type(m).__name__
        extra = ""
        if getattr(m, "tool_calls", None):
            extra = " -> calls: " + ", ".join(tc.name for tc in m.tool_calls)
        print(f"  {kind}{extra}")
else:
    # Build an illustrative RAGAS trace from the frozen data to use in later steps.
    frozen = json.load(open("frozen/agent_run.json"))
    ragas_trace = [
        HumanMessage(content="What are the current prices of gold and silver in US dollars?"),
        AIMessage(
            content="Fetching current spot prices for you.",
            tool_calls=[
                ToolCall(name="get_metal_price", args={"metal": "gold",   "currency": "USD"}),
                ToolCall(name="get_metal_price", args={"metal": "silver", "currency": "USD"}),
            ],
        ),
        ToolMessage(content="The current spot price of gold is 2998.41 USD per troy ounce."),
        ToolMessage(content="The current spot price of silver is 33.72 USD per troy ounce."),
        AIMessage(content="Here are the current precious metal spot prices:\n"
                          "- **Gold**: $2,998.41 USD per troy ounce\n"
                          "- **Silver**: $33.72 USD per troy ounce"),
    ]
    print("(using cached illustrative RAGAS trace — set all three keys to run live)")
    print(f"Illustrative trace: {len(ragas_trace)} messages. Types in order:")
    for m in ragas_trace:
        kind = type(m).__name__
        extra = ""
        if getattr(m, "tool_calls", None):
            extra = " -> calls: " + ", ".join(tc.name for tc in m.tool_calls)
        print(f"  {kind}{extra}")
''')

# ============================================================================
# STEP 9 — TOPIC ADHERENCE
# ============================================================================
md(r"""
---
# Step 9 · TopicAdherenceScore

`TopicAdherenceScore` checks whether the agent stayed within its allowed domain.
You supply a `reference_topics` list describing the agent's intended scope. The
metric counts how many turns the agent correctly answered in-scope questions and
correctly declined out-of-scope ones, then reports precision, recall, or F1.

We construct a two-turn conversation: one metals question (should be answered)
and one off-topic request (should be declined).
""")

code(r'''
from ragas.metrics import TopicAdherenceScore

# A conversation with one in-scope turn (answered) and one out-of-scope turn (declined).
convo = [
    HumanMessage(content="What drives the price of gold?"),
    AIMessage(
        content="Let me look that up.",
        tool_calls=[ToolCall(name="search_metal_knowledge",
                             args={"query": "what drives the gold price"})],
    ),
    ToolMessage(content="Gold is driven mainly by real interest rates, the US dollar, and safe-haven demand."),
    AIMessage(content="Gold is driven mainly by real interest rates, the strength of the US dollar, "
                      "and safe-haven demand from investors during uncertainty."),
    HumanMessage(content="Nice. Can you also give me a good chocolate cake recipe?"),
    AIMessage(content="That is outside my scope. I can only help with metals and markets."),
]

allowed_topics = ["precious metals", "metals markets", "investing in metals", "commodity prices"]
ta_sample = MultiTurnSample(user_input=convo, reference_topics=allowed_topics)

if HAVE_OLLAMA and judge_llm is not None:
    for mode in ["precision", "recall", "f1"]:
        scorer = TopicAdherenceScore(llm=judge_llm, mode=mode)
        score = await scorer.multi_turn_ascore(ta_sample)
        print(f"Topic adherence ({mode}): {score:.2f}")
else:
    frozen = json.load(open("frozen/agent_metrics.json"))
    print("(using cached illustrative scores — set OLLAMA_API_KEY to run live)")
    print(f"Topic adherence (precision): {frozen['topic_adherence_precision']:.2f}")
    print(f"Topic adherence (recall):    {frozen['topic_adherence_recall']:.2f}")
    print(f"Topic adherence (f1):        {frozen['topic_adherence_f1']:.2f}")
''')

md(r"""
*Illustrative output:*
```
Topic adherence (precision): 1.00
Topic adherence (recall):    1.00
Topic adherence (f1):        1.00
```

> ⚠ **Caution:** the metric is only as good as your topic list. Define it too
> narrowly and you penalise reasonable adjacent answers (e.g. "what is a basis
> point?" is adjacent to markets). Define it too broadly and you never catch real
> scope drift. Precision and recall encode different risks: a high-precision agent
> refuses anything uncertain (safe but frustrating); a high-recall agent answers
> everything (helpful but prone to going out of bounds). Decide which failure your
> application can tolerate before you read the number.
""")

# ============================================================================
# STEP 10 — TOOL-CALL ACCURACY
# ============================================================================
md(r"""
---
# Step 10 · ToolCallAccuracy — strict matching

`ToolCallAccuracy` compares the actual tool calls the agent made against a list
of **expected** calls you supply as `reference_tool_calls`. It is strict:
right tools, right arguments, right order. One mismatch can sink the score.

Use it as a pass/fail gate when the exact sequence is a requirement (e.g. in a
regulated workflow where every call must be logged).
""")

code(r'''
from ragas.metrics import ToolCallAccuracy

# What we expected the agent to do for the gold + silver query.
expected_calls = [
    ToolCall(name="get_metal_price", args={"metal": "gold",   "currency": "USD"}),
    ToolCall(name="get_metal_price", args={"metal": "silver", "currency": "USD"}),
]

tool_sample = MultiTurnSample(user_input=ragas_trace, reference_tool_calls=expected_calls)

if HAVE_OLLAMA and judge_llm is not None:
    tca = ToolCallAccuracy()
    acc = await tca.multi_turn_ascore(tool_sample)
    print(f"Tool-call accuracy: {acc:.2f}")
else:
    frozen = json.load(open("frozen/agent_metrics.json"))
    print("(using cached illustrative score — set OLLAMA_API_KEY to run live)")
    print(f"Tool-call accuracy: {frozen['tool_call_accuracy']:.2f}")
''')

md(r"""
*Illustrative output:*
```
Tool-call accuracy: 1.00
```

> ⚠ **Caution — exact argument matching:** `get_metal_price(metal="Gold")` and
> `get_metal_price(metal="gold")` are different as far as this metric is
> concerned. Write reference calls to match the exact string form the agent will
> produce. Also: strict accuracy can report 0.0 for a perfectly good answer that
> simply took a different valid path — in that case you are measuring conformity
> to your script, not success. Use goal accuracy (Step 12) alongside it.
""")

# ============================================================================
# STEP 11 — TOOL-CALL F1
# ============================================================================
md(r"""
---
# Step 11 · ToolCallF1 — partial credit

`ToolCallF1` is forgiving where `ToolCallAccuracy` is strict: it ignores call
order and gives partial credit through precision and recall over the set of
calls. An agent that made both expected calls *plus one extra* scores roughly
0.8, not 0.0.

Use F1 during development to track whether the agent is getting closer to the
reference, even if it hasn't nailed the exact sequence yet.
""")

code(r'''
from ragas.metrics import ToolCallF1

f1_metric = ToolCallF1()

# Case 1: the clean trace from before (should score 1.0).
if HAVE_OLLAMA and judge_llm is not None:
    clean_f1 = await f1_metric.multi_turn_ascore(tool_sample)
    print(f"Tool-call F1 (clean run): {clean_f1:.2f}")
else:
    frozen = json.load(open("frozen/agent_metrics.json"))
    print("(using cached illustrative score — set OLLAMA_API_KEY to run live)")
    print(f"Tool-call F1 (clean run): {frozen['tool_call_f1_clean']:.2f}")

# Case 2: simulate an over-eager agent that made one extra (unnecessary) call.
noisy_trace = [
    HumanMessage(content="What are the current prices of gold and silver in US dollars?"),
    AIMessage(
        content="Fetching prices...",
        tool_calls=[
            ToolCall(name="get_metal_price", args={"metal": "gold",     "currency": "USD"}),
            ToolCall(name="get_metal_price", args={"metal": "silver",   "currency": "USD"}),
            ToolCall(name="get_metal_price", args={"metal": "platinum", "currency": "USD"}),  # extra
        ],
    ),
    ToolMessage(content="Gold: 2998.41 USD per troy ounce."),
    ToolMessage(content="Silver: 33.72 USD per troy ounce."),
    ToolMessage(content="Platinum: 1012.50 USD per troy ounce."),
    AIMessage(content="Gold and silver prices retrieved."),
]
noisy_sample = MultiTurnSample(user_input=noisy_trace, reference_tool_calls=expected_calls)

if HAVE_OLLAMA and judge_llm is not None:
    noisy_f1 = await f1_metric.multi_turn_ascore(noisy_sample)
    print(f"Tool-call F1 (one extra call): {noisy_f1:.2f}")
else:
    print(f"Tool-call F1 (one extra call): {frozen['tool_call_f1_noisy']:.2f}")
''')

md(r"""
*Illustrative output:*
```
Tool-call F1 (clean run): 1.00
Tool-call F1 (one extra call): 0.80
```

In the noisy case the agent made two of the two expected calls (recall = 1.0)
but also made one extra call (precision = 2/3 ≈ 0.67), giving F1 ≈ 0.80.
""")

# ============================================================================
# STEP 12 — AGENT GOAL ACCURACY
# ============================================================================
md(r"""
---
# Step 12 · AgentGoalAccuracy — did it actually help?

The previous two metrics judge the *path* (which tools, which arguments). Goal
accuracy judges the *outcome*: did the agent accomplish what the user actually
wanted, regardless of how it got there?

It is binary — 1 for success, 0 for failure.

- **`AgentGoalAccuracyWithReference`** — compare the end state to an ideal
  outcome you supply as a `reference` string.
- **`AgentGoalAccuracyWithoutReference`** — the judge infers the goal from
  the conversation itself. More flexible, but adds a second layer of LLM
  judgment (and therefore a second source of error).
""")

code(r'''
from ragas.metrics import AgentGoalAccuracyWithReference, AgentGoalAccuracyWithoutReference

if HAVE_OLLAMA and judge_llm is not None:
    # With a reference outcome:
    goal_with_ref = AgentGoalAccuracyWithReference(llm=judge_llm)
    goal_sample = MultiTurnSample(
        user_input=ragas_trace,
        reference="Report the current prices of gold and silver in US dollars.",
    )
    score_with_ref = await goal_with_ref.multi_turn_ascore(goal_sample)
    print(f"Agent goal accuracy (with reference):    {score_with_ref:.0f}")

    # Without a reference: the judge reads the conversation and infers the goal.
    goal_without_ref = AgentGoalAccuracyWithoutReference(llm=judge_llm)
    score_without_ref = await goal_without_ref.multi_turn_ascore(
        MultiTurnSample(user_input=ragas_trace)
    )
    print(f"Agent goal accuracy (without reference): {score_without_ref:.0f}")
else:
    frozen = json.load(open("frozen/agent_metrics.json"))
    print("(using cached illustrative scores — set OLLAMA_API_KEY to run live)")
    print(f"Agent goal accuracy (with reference):    {frozen['agent_goal_accuracy_with_reference']}")
    print(f"Agent goal accuracy (without reference): {frozen['agent_goal_accuracy_without_reference']}")
''')

md(r"""
*Illustrative output:*
```
Agent goal accuracy (with reference):    1
Agent goal accuracy (without reference): 1
```

> ⚠ **Caution — binary and blunt:** a score of 1 hides *how well* the goal was
> met (was the answer complete? well explained?), and a 0 hides *how close* the
> agent came. Never use goal accuracy in isolation. Pair it with topic adherence
> (was the agent on-topic?), tool-call F1 (how close was the action sequence?),
> and the single-turn generator metrics from Module 9 (was the answer faithful
> and relevant?). One number never tells the whole story.
""")

# ============================================================================
# RECAP
# ============================================================================
md(r"""
---
# Recap: the five agent metrics at a glance

| Metric | What it grades | Output | Best used when |
| --- | --- | --- | --- |
| `TopicAdherenceScore` | Scope discipline | 0–1 (precision / recall / F1) | Domain boundary matters |
| `ToolCallAccuracy` | Right tools, args, order (strict) | 0 or 1 | Exact sequence required |
| `ToolCallF1` | Tool precision & recall (partial credit) | 0–1 | During development; tracking progress |
| `AgentGoalAccuracyWithReference` | Outcome vs. supplied reference | 0 or 1 | You can define the ideal outcome |
| `AgentGoalAccuracyWithoutReference` | Outcome (judge infers goal) | 0 or 1 | Goal is implicit or hard to specify |

### What you built in Module 11

1. Three `@tool` functions (live price, currency conversion, RAG knowledge search)
2. A MetalDesk ReAct agent using `create_react_agent`
3. A live agent run with a multi-tool question
4. A RAGAS message trace via `convert_to_ragas_messages`
5. Scores on all five agent-tier RAGAS metrics

**Next module (12 — Capstone):** Module 12 assembles everything from the
entire track — all 12 metrics, the MDD baseline-vs-reranked comparison loop,
and the two multi-hop golden questions — into one complete Agentic RAG
Evaluation system. You built every brick; Module 12 lays them into the
finished wall.
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

OUT = "11_rag_to_agent.ipynb"   # <-- this module's notebook filename
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

n_md = sum(1 for k, _ in CELLS if k == "md")
n_code = sum(1 for k, _ in CELLS if k == "code")
print(f"Wrote {OUT}: {len(CELLS)} cells ({n_md} markdown, {n_code} code)")
