"""Builds NN_slug.ipynb from a list of (type, source) cells.

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
# Module NN · Title

### A hands-on, build-it-yourself module for advanced high school researchers

![alt text](assets/diagram.svg)

One-paragraph orientation: what this module teaches, how it builds on the
previous module, and what the next module will add. This is module NN of a
twelve-part track that ends in a full **Agentic RAG Evaluation** capstone.
""")

md(r"""
## 📋 Summary: the one-paragraph version

Plain-language summary of the single big idea of this module.
""")

md(r"""
## 🗺️ What you will build, step by step

| Step | What you do | Key tool |
| ---: | --- | --- |
| 0 | Set up the environment | `uv`, `python-dotenv` |
| 1 | … | … |

### 🎓 What you will *learn* (the concepts)

- …

### ✅ Prerequisites

- Comfort reading basic Python. No machine learning required.
- (Keyed modules only) the relevant API key — see Step 0.
- Curiosity.
""")

# ============================================================================
# STEP 0 — SETUP  (every module includes this, adapted to its own deps/keys)
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
""")

# --- KEYED MODULES ONLY (M5+): key-loading + friendly guard + frozen fallback ---
# Keys live in ONE shared file: tutorials/.env (the parent folder of every module).
# load_dotenv(find_dotenv()) walks UP the directory tree and finds it automatically,
# so no per-module .env is needed.
# md(r'''
# ## 0.2 Provide your API keys (shared `.env`)
#
# All twelve modules read their keys from a **single** `.env` file in the
# `tutorials/` folder (the parent of this module). Create `tutorials/.env` once:
#
# ```
# OLLAMA_API_KEY=...      # cloud Ollama
# COHERE_API_KEY=...      # Cohere reranking (M10+)
# METALS_API_KEY=...      # Metals.dev live prices (M11+)
# ```
#
# `find_dotenv()` walks UP from this notebook and locates that shared file, so you
# never copy keys into each module. `.env` is gitignored — never commit it.
# ''')
# code(r'''
# import os
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())          # resolves to tutorials/.env automatically
# HAVE_KEYS = bool(os.environ.get("OLLAMA_API_KEY"))  # adapt per module
# if not HAVE_KEYS:
#     print("No API key found. This notebook will run using the cached "
#           "results in frozen/ so you can still follow along.")
# ''')

# ============================================================================
# STEPS — one per concept, each ≤ a couple of ideas
# ============================================================================
md(r"""
---
# Step 1 · …
""")

# code(r'''
# ...
# ''')

# ============================================================================
# RECAP
# ============================================================================
md(r"""
---
# Recap & what's next

You learned …

**Next module (NN+1):** … — it adds … on top of what you built here.
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

OUT = "NN_slug.ipynb"   # <-- set to this module's notebook filename
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

n_md = sum(1 for k, _ in CELLS if k == "md")
n_code = sum(1 for k, _ in CELLS if k == "code")
print(f"Wrote {OUT}: {len(CELLS)} cells ({n_md} markdown, {n_code} code)")
