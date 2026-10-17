# Beam + LangGraph overview

Start with [tutorial.ipynb](tutorial.ipynb). It explains and defines the entire agent, then adds interactive chat, a Beam versus `gpt-5.6-sol` experiment, latency plots, answer review, and CSV export. No code imports from `src/` or other course lessons are required. Run All creates controls without making model or search API calls. The graph visualization requests an image from Mermaid.ink and requires internet access.

Use Python 3.12+ and a separate environment, from this folder:

```sh
uv sync
uv run jupyter lab tutorial.ipynb
```

Copy [.env.example](.env.example) to `.env` and fill it in with a local editor. The CLI uses `load_dotenv(find_dotenv())` to find the nearest `.env` starting from the script's directory and walking up its parents. Existing environment variables take precedence. The notebook has an explicit-path control and reads values directly from its selected file without interpolation. Never commit `.env`.

[beam_agent.py](beam_agent.py) is the standalone CLI. It needs only `langgraph`, `openai`, `python-dotenv`, and `httpx`:

```sh
python beam_agent.py
python beam_agent.py --effort medium
```

It uses Beam for conversation and Tavily for web search. `/reset` clears the in-memory conversation; `/quit`, Ctrl-D, or Ctrl-C exits. A failed turn preserves the last successful history. Each turn allows at most four model requests and three searches, with retries disabled. Output and transport limits are explained in the notebook.

The notebook compares Beam's Chat Completions API with OpenAI's Responses API: our live check found that GPT-5.6 Sol rejects reasoning-enabled function tools through Chat Completions. Both use the same graph, search function, questions, low effort, and output cap. API endpoints remain a difference in the measured provider workflows.

A six-trial live run on October 17, 2026 completed successfully: mean total latency was **2.81 s for Beam** and **5.98 s for GPT-5.6 Sol**; both matched **3/3 short factual answers**. This tiny run measures the complete agent workflow and cannot establish general model quality. One Beam answer omitted requested source URLs, illustrating why the notebook includes full-answer and citation review. A self-contained historical snapshot and its limitations appear inside the notebook; raw exports are not included.

Validation performed: the notebook executed from an empty directory with HTTP disabled; the actual benchmark callback completed all six live trials; the Beam CLI handled a live question, a contextual follow-up, reset, and exit. Offline behavior checks cover history preservation, tool validation, call budgets, API failures, and the notebook's Responses adapter:

```sh
uv run pytest -q
```

The notebook is hand-authored, independent of the generated Marimo exports elsewhere in this course.
