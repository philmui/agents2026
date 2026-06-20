"""Builds 01_what_is_rag.ipynb from a list of (type, source) cells.

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
# Module 01 · What is RAG?

### A hands-on, build-it-yourself module for advanced high school researchers

One-paragraph orientation: Large Language Models are powerful but frozen — they
can confabulate facts they never knew or that have changed since training.
**Retrieval-Augmented Generation (RAG)** solves this by inserting retrieved
passages directly into the model's prompt so it reads before it answers.
This is Module 01 of a twelve-part track that ends in a full
**Agentic RAG Evaluation** capstone.

> **No API keys required.** This module uses only the Python standard library.
""")

md(r"""
## 📋 Summary: the one-paragraph version

RAG = Retrieve + Augment + Generate. Instead of asking an LLM to remember a
fact from its frozen training weights, we fetch relevant text from a knowledge
base and place it in the prompt window — exploiting *in-context learning*.
The model reads the passages, then writes an answer grounded in them.
This sidesteps both the knowledge-cutoff problem and confabulation, but it
introduces a new failure mode: bad retrieval produces bad answers, regardless
of how good the generator is.
""")

md(r"""
## 🗺️ What you will build, step by step

| Step | What you do | Key idea |
| ---: | --- | --- |
| 0 | Set up the environment | `uv sync` + `uv run jupyter lab` |
| 1 | The frozen-library problem | Why LLMs confabulate |
| 2 | In-context learning | How prompts carry new knowledge |
| 3 | A toy RAG loop | Retrieve → Augment → Generate (keyless) |
| 4 | Two failure modes | Retriever vs generator intuition |

### 🎓 What you will *learn* (the concepts)

- **The RAG idea**: why retrieval is the right solution to the LLM knowledge-gap problem
- **Confabulation / hallucination**: fluency ≠ accuracy; the model invents facts confidently
- **In-context learning**: LLMs can use new information placed in the prompt — no retraining needed
- **Two-part architecture**: retriever and generator as separate components with independent failure modes

### ✅ Prerequisites

- Comfort reading basic Python. No machine learning required.
- No API keys, no local GPU, no special hardware.
- Curiosity about how AI research assistants actually work.
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

> This module has **no API keys**. Module 5 is the first module that needs one.
> When that time comes, all keys go in a single shared `tutorials/.env` file —
> never in a per-module file, never committed to git.
""")

code(r'''
# Step 0 verification: confirm we are running Python 3.10+
import sys
assert sys.version_info >= (3, 10), f"Need Python 3.10+, got {sys.version}"
print(f"Python {sys.version}")
print("Environment ready.")
''')

# ============================================================================
# STEP 1 — THE FROZEN-LIBRARY PROBLEM
# ============================================================================
md(r"""
---
# Step 1 · The frozen-library problem

A large language model learns by reading enormous amounts of text — web pages,
books, research papers, forum posts — collected up to a fixed date. After
training, the model's weights are **frozen**: no new information can enter
without a full retraining run.

This creates three concrete gaps:

| Gap | What it means | Example |
| --- | --- | --- |
| **Temporal** | Events after the training cutoff | Today's gold spot price |
| **Private** | Data the model was never trained on | Your lab's internal notes |
| **Precision** | Specific facts that need exact sourcing | A contract clause |

The cell below builds a tiny simulation: a "frozen" dictionary representing
the model's parametric memory, and a question it cannot answer.
""")

code(r'''
# Simulate an LLM's frozen parametric memory as a Python dict.
# In a real LLM the "knowledge" is distributed across billions of weights —
# but the frozen-knowledge problem is the same.

FROZEN_MEMORY: dict[str, str] = {
    "troy_ounce":     "A troy ounce equals 31.1035 grams. Used for precious metals.",
    "gold_symbol":    "The chemical symbol for gold is Au (Latin: aurum).",
    "silver_use":     "Silver is widely used in electronics and solar panels.",
}

def frozen_lm_answer(question: str) -> str:
    """
    A comically simplified stand-in for an LLM without retrieval.
    If no key matches, it confabulates a plausible-sounding number.
    """
    q = question.lower()
    for key, fact in FROZEN_MEMORY.items():
        if key.replace("_", " ") in q or key in q:
            return fact
    # The model doesn't know — but it won't admit it.
    return "(confabulated) The current price is approximately $1,842.50."

print(frozen_lm_answer("What is a troy ounce?"))         # known → correct
print(frozen_lm_answer("What is today's gold price?"))   # unknown → confabulated
''')

md(r"""
> **⚠ Caution — Confabulation is not lying; it is a fundamental property of
> statistical language models.**
>
> The model is not trying to deceive you. It is doing exactly what it was
> trained to do: produce the most statistically likely continuation of your
> prompt. When it does not have the answer, a plausible-sounding number *is*
> the most likely continuation. Fluency and accuracy are **independent**.
""")

# ============================================================================
# STEP 2 — IN-CONTEXT LEARNING
# ============================================================================
md(r"""
---
# Step 2 · In-context learning: read before you answer

Here is the key property we exploit: LLMs can **use new information placed
directly in the prompt window** — with no retraining, no fine-tuning, no
weight updates. This is called *in-context learning*.

The prompt window is like a page of reading the model gets right before it
answers. If we put the relevant facts on that page, the model can use them.

```
┌───────────────────────────────────────────────────────┐
│  PROMPT (context window)                              │
│  ─────────────────────────────────────────────────    │
│  [Retrieved Passages]                                 │
│    [1] A troy ounce equals 31.1035 grams ...          │
│    [2] The spot price is the current market price ... │
│                                                       │
│  [Question]                                           │
│    How many grams is a troy ounce?                    │
│                                                       │
│  [Instruction]                                        │
│    Answer using ONLY the passages above.              │
└───────────────────────────────────────────────────────┘
               ↓
     LLM writes a grounded answer
```

The cell below shows what this augmented prompt looks like for a simple case.
""")

code(r'''
def build_augmented_prompt(question: str, passages: list[str]) -> str:
    """
    Construct the prompt that RAG sends to the LLM.
    The model reads the passages and answers from them.
    """
    if not passages:
        return (
            f"Answer the following question to the best of your ability.\n\n"
            f"Question: {question}\nAnswer:"
        )
    context_block = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    return (
        "You are a precise research assistant. "
        "Answer using ONLY the passages below.\n"
        "If the passages do not contain the answer, say you do not know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\nAnswer:"
    )

# Show the prompt that would be sent to the LLM
sample_passage = ("A troy ounce is a unit of weight equal to 31.1035 grams, "
                  "used internationally for precious metals trading.")
prompt = build_augmented_prompt("How many grams is a troy ounce?", [sample_passage])
print(prompt)
''')

md(r"""
> **Key insight:** the model does **not** permanently learn from these passages.
> The next time you call it (with a fresh prompt containing no context), it is
> back to its frozen state. In-context learning is **per-call**, not persistent.
""")

# ============================================================================
# STEP 3 — A TOY RAG LOOP
# ============================================================================
md(r"""
---
# Step 3 · A toy RAG loop: retrieve → augment → generate

Now let us wire the three steps together into a complete (though toy) RAG
pipeline. The retriever here uses simple keyword matching — Module 2 will
replace it with proper semantic vector search.

The generate step is simulated (we just print the prompt rather than calling
a real LLM) because this module is keyless. Module 5 plugs in the real LLM.
""")

code(r'''
# A small in-memory knowledge base (stand-in for a vector store)
KNOWLEDGE_BASE: dict[str, str] = {
    "troy_ounce":      ("A troy ounce is a unit of weight equal to 31.1035 grams, "
                        "used internationally for precious metals."),
    "spot_price":      ("The spot price is the current market price at which a commodity "
                        "such as gold or silver can be bought or sold for immediate delivery."),
    "gold_symbol":     "The chemical symbol for gold is Au, from the Latin word 'aurum'.",
    "silver_solar":    ("Silver is the most electrically conductive metal and is widely used "
                        "in photovoltaic (solar) panels and electronics."),
    "platinum_auto":   ("Platinum is used in catalytic converters in automobiles because it "
                        "efficiently oxidises harmful exhaust gases."),
}


def retrieve(question: str, kb: dict[str, str], top_k: int = 2) -> list[str]:
    """
    Keyword-based retrieval — a conceptual stand-in for semantic vector search.
    Returns the top_k passages with the most keyword overlaps.
    """
    tokens = set(question.lower().split())
    scored: list[tuple[int, str]] = []
    for text in kb.values():
        overlap = sum(1 for tok in tokens if tok in text.lower())
        scored.append((overlap, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for score, text in scored[:top_k] if score > 0]


def rag_loop(question: str, kb: dict[str, str], verbose: bool = True) -> str:
    """
    Full toy RAG pipeline:
      1. Retrieve relevant passages
      2. Augment the prompt
      3. (Simulated) Generate an answer

    In Module 5, step 3 is replaced with a real LLM call.
    """
    # --- Step 1: Retrieve ---
    passages = retrieve(question, kb)
    if verbose:
        print(f"Retrieved {len(passages)} passage(s).")

    # --- Step 2: Augment ---
    prompt = build_augmented_prompt(question, passages)

    # --- Step 3: Simulate generation (print the prompt that would go to the LLM) ---
    if verbose:
        print("\n" + "─" * 60)
        print(prompt)
        print("─" * 60)

    if not passages:
        return "[No relevant passages found — the LLM would confabulate here]"
    return "[In Module 5 a real LLM call replaces this line and returns an answer]"


# Try two questions: one the KB can answer, one it cannot
print("=== Question the knowledge base can answer ===")
rag_loop("What is a troy ounce?")

print("\n=== Question the knowledge base CANNOT answer ===")
rag_loop("What is the current gold price today?")
''')

md(r"""
**Notice what happened with the unanswerable question:**
no relevant passages were retrieved, so the augmented prompt contains no
context. A real LLM receiving that prompt would be likely to confabulate.
This is the *garbage-in-garbage-out* property of RAG: the generator is only
as good as what the retriever gives it.
""")

# ============================================================================
# STEP 4 — TWO FAILURE MODES
# ============================================================================
md(r"""
---
# Step 4 · Two failure modes: retriever vs generator

A RAG pipeline has two independent components. Each can fail separately:

```
   Question
      │
      ▼
 ┌──────────┐   contexts   ┌───────────┐
 │ RETRIEVER│ ──────────► │ GENERATOR │ ──► Answer
 └──────────┘             └───────────┘
      │                        │
      ▼                        ▼
 Wrong passages           Ignores passages /
 Missing passages         unfaithful summary
 Noisy passages           hallucination anyway
```

| Failure | Symptom | Example |
| --- | --- | --- |
| Retriever returns **wrong** passages | Answer confidently wrong | Asked about platinum, got silver passages, answered about silver |
| Retriever returns **incomplete** passages | Answer partially right | Missed a key context chunk |
| Generator **ignores** context | Answer reverts to parametric memory | LLM "knows better" and overrides the passage |
| Generator **misreads** context | Answer subtly wrong | Misinterprets a number or unit |

This is why Modules 7–9 of this track measure the two halves separately:
- **Retriever metrics** (Modules 7–8): context precision, recall, entity recall, noise sensitivity
- **Generator metrics** (Module 9): faithfulness, response relevancy, factual correctness
""")

code(r'''
# Demonstrate retriever failure: fetch wrong passages on purpose

BAD_KB: dict[str, str] = {
    "silver_solar":  "Silver is the most electrically conductive metal.",
    "silver_coin":   "Silver coins have been used as currency for millennia.",
}

print("=== Retriever failure: asking about platinum, KB only has silver passages ===")
results = retrieve("What is platinum used for?", BAD_KB, top_k=2)
print(f"Retrieved passages: {results}")
print()
print("A real LLM given these passages would answer ABOUT SILVER, not platinum.")
print("The answer would be fluent and confident — and completely wrong.")
''')

md(r"""
> **⚠ Caution — Measuring retrieval quality is non-trivial.**
>
> You cannot simply count keyword hits. A passage about "silver and solar
> panels" might be tangentially related to a question about "renewable energy
> investments and precious metals" — but is it truly relevant? This is why
> Modules 7–8 use an LLM *judge* to evaluate whether retrieved contexts
> actually support answering the question. Human-level relevance judgment
> requires semantic understanding, not just string matching.
""")

# ============================================================================
# RECAP
# ============================================================================
md(r"""
---
# Recap & what's next

## What you built in Module 01

- A mental model of the **three gaps** a frozen LLM cannot close: temporal,
  private-data, and precision
- An understanding of **confabulation**: why fluency ≠ accuracy
- The **RAG loop**: retrieve → augment (build prompt with passages) → generate
- An understanding of **in-context learning**: the model reads, it doesn't remember
- Intuition for the **two-part failure surface**: retriever and generator fail independently

## What you **did not** build yet

This module was intentionally conceptual. The retriever here is trivial keyword
matching; the generator is simulated. The real machinery comes in:
- **Module 02**: turning text into vectors with sentence embeddings
- **Module 03**: finding nearest-neighbor passages with Qdrant
- **Module 05**: swapping in a real cloud LLM for actual generated answers

**Next module (Module 02) — Embeddings & meaning:** you will learn how text
is converted into numeric vectors so that a retriever can find semantically
similar passages — not just keyword matches. This is the mathematical engine
behind the "Retrieve" step.
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

OUT = "01_what_is_rag.ipynb"   # <-- this module's notebook filename
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

n_md   = sum(1 for k, _ in CELLS if k == "md")
n_code = sum(1 for k, _ in CELLS if k == "code")
print(f"Wrote {OUT}: {len(CELLS)} cells ({n_md} markdown, {n_code} code)")
