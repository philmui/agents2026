# `backend/`: the voice-app backend (OpenAI Agents SDK + Langfuse + FastAPI)

This is the **one backend** the web app talks to. It handles every server-side
operation that requires the permanent OpenAI key, so the key never reaches the
browser, and it traces everything to Langfuse.

| Job | Route | Used by | Notes |
|---|---|---|---|
| Mint ephemeral `ek_` tokens | `POST /token` (and `GET /token`) | **Transcribe**, **Assist** | The browser opens WebRTC to OpenAI directly, but only with a short-lived token. The real key stays here. |
| Run web search | `POST /web-search` | **Assist** | An **OpenAI Agents SDK** agent with a hosted `WebSearchTool`; returns only the grounded answer. |
| Proxy live translation + source captions | `WS /translate` | **Translate** | OpenAI's translation socket needs an `Authorization` header a browser cannot set on a WebSocket, so this server relays it. |
| Health + config check | `GET /health` | you (debugging) | Reports whether the key loaded, tracing is on, and a caller token is required (never reveals the key). |

The real `OPENAI_API_KEY` (and the optional `LANGFUSE_*` and `CAPSTONE_*` keys) are
loaded from the shared `topics/voice_agents/.env` via `python-dotenv`
(`load_dotenv(find_dotenv())`). You set your keys once, in one file.

## Setup and run (step by step)

```bash
# 1) From the course root, create the shared .env (if you have not already):
cd topics/voice_agents
cp .env.example .env
#    Then edit .env and paste your paid-tier OPENAI_API_KEY (starts with sk-).
#    OPTIONAL: paste LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY to turn on tracing.

# 2) Install and start the backend:
cd 09_capstone_openai/backend
uv sync                                              # creates .venv/, installs deps, writes uv.lock
uv run uvicorn src.main:app --reload --port 8000     # http://localhost:8000
```

`uv sync` picks a Python 3.11 - 3.14 interpreter (the `openinference` instrumentation
package requires `< 3.15`, which `requires-python` in `pyproject.toml` enforces).

Smoke-test it:

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"gpt-realtime-2.1","translate_model":"gpt-realtime-translate",
#  "has_api_key":true,"telemetry":false,"auth":false}
```

- `has_api_key:false` -> your `.env` was not found or the `OPENAI_API_KEY` line is
  empty. Fix `topics/voice_agents/.env` and restart.
- `telemetry:false` -> Langfuse is off (no keys, or the keys failed an auth check).
  Everything still works; you just do not get traces. Add valid `LANGFUSE_*` keys.
- `auth:false` -> no shared caller token is required (open on localhost).

Interactive API docs live at `http://localhost:8000/docs`.

## The web-search agent, in code

```python
from agents import Agent, Runner, WebSearchTool, set_default_openai_key

# Give the SDK the real key for MODEL calls, but NOT for its own trace export.
set_default_openai_key(OPENAI_API_KEY, use_for_tracing=False)

agent = Agent(
    name="Web Search Delegate",
    instructions="Always use web_search; answer in short plain text a voice "
                 "assistant can read aloud; name sources; no Markdown or long URLs.",
    model="gpt-5.6",
    tools=[WebSearchTool()],                     # OpenAI's HOSTED web search tool
)

result = await Runner.run(agent, query)          # runs reason -> search -> answer
answer = result.final_output                     # the grounded text we return
```

The permanent key stays on the server; the browser only ever receives the `answer`
string, in the shape `{"answer": "..."}`.

## Langfuse tracing (optional but recommended)

1. Create a free account and project at <https://cloud.langfuse.com>.
2. Open **Settings -> API Keys** and copy the **Public key** (`pk-lf-...`) and
   **Secret key** (`sk-lf-...`).
3. Paste both into `topics/voice_agents/.env`:
   ```bash
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or https://us.cloud.langfuse.com
   ```
4. Restart the backend. `GET /health` now shows `"telemetry":true`.
5. Use Assist or Translate, then open the Langfuse **Traces** view. You will see
   traces named `assist-web-search` and `translate-session`, grouped into sessions
   and tagged by feature.

### How the tracing is wired (`src/telemetry.py`)

- Langfuse is imported **after** `load_dotenv()`, so the client sees the credentials.
- It uses the framework integration the Langfuse docs recommend:
  `OpenAIAgentsInstrumentor().instrument()` auto-captures every `Runner.run(...)` as
  OpenTelemetry spans (model name, token usage, tool calls) that
  `langfuse.get_client()` exports.
- Each request is wrapped in ONE trace with a **low-cardinality** name
  (`assist-web-search`), an **explicit input** (just the user query), an **explicit
  output** (the final answer), a **`session_id`**, **tags**, and an **environment**
  label. The SDK's nested model/tool spans land inside this trace.
- It **flushes** after each short request.
- If the Langfuse packages are missing OR the keys are absent/invalid, telemetry
  stays disabled and the app runs normally.

### Two tracing gotchas (both handled in `main.py`)

- **Do not disable SDK tracing while Langfuse is ON.** The instrumentor listens to
  the SDK's own trace pipeline; disabling it flattens Langfuse traces (drops the
  nested model/tool spans). Verified: with the pipeline off a web-search trace had
  **1** observation; with it on, **6** (`GENERATION` + `AGENT` + `CHAIN`).
- **When Langfuse is OFF, disable tracing so nothing leaks.** We pass
  `use_for_tracing=False` to `set_default_openai_key` (the real key is never used to
  export traces to OpenAI's dashboard), and when no Langfuse keys are set we call
  `set_tracing_disabled(True)`. So "telemetry disabled" truly means no trace export.

## Protecting the paid routes (`src/security.py`)

`/token`, `/web-search`, and `/translate` spend real money. On localhost the backend
is open; for a deployed demo, three guards keep it from being an open wallet. All
degrade gracefully, with nothing configured the app runs open on localhost:

- **Per-caller rate limit** (always on): an in-process counter caps paid requests per
  caller IP per minute. Configure with `CAPSTONE_RATE_LIMIT_PER_MIN` (default 60).
  Over the cap returns `429`.
- **WebSocket Origin check** (always on): `/translate` accepts a missing Origin
  (non-browser clients), a localhost Origin, or one listed in
  `CAPSTONE_ALLOWED_ORIGINS`. Everything else is rejected.
- **Optional shared caller token**: set `CAPSTONE_API_TOKEN` and every paid route
  requires it, HTTP via `Authorization: Bearer <token>`, and the `/translate`
  WebSocket via a `{"token": "..."}` field in its first message. Missing/wrong token
  returns `401`. The frontend sends it when `NEXT_PUBLIC_CAPSTONE_API_TOKEN` is set to
  the same value.

`GET /health` reports `"auth": true` when a token is required. A `NEXT_PUBLIC_` token
is visible in the browser bundle, so it is a light gate to deter casual abuse of a
public URL, not real per-user login.

## The translation proxy

The translation and transcription WebSockets are **not** part of the Agents SDK, so
`WS /translate` uses a direct relay. Its narrow `session.update` sets only
`session.audio.output.language`; do not add standard Realtime audio fields. The live
translation session does not reliably emit a source-language caption, so `/translate`
also opens a `gpt-realtime-whisper` transcription session (`?intent=transcription`),
copies each mic chunk to it, detects pauses (that model has no server VAD), commits
each phrase, and relays `conversation.item.input_audio_transcription.delta`/`.completed`
for the "You said (source)" caption. The whole session is wrapped in one Langfuse
`translate-session` trace.

## Dependencies (all declared in `pyproject.toml`)

- `fastapi` + `uvicorn[standard]` - the web server and ASGI runtime. `[standard]`
  includes the `websockets` library uvicorn needs to accept the browser's
  `/translate` WebSocket.
- `openai-agents` - the OpenAI Agents SDK (`Agent`, `Runner`, `WebSearchTool`).
- `httpx` - async HTTP client for the server-to-server `/token` call.
- `websockets` - async WebSocket **client** for `/translate`'s upstream sockets.
- `langfuse` + `openinference-instrumentation-openai-agents` - the telemetry pair the
  Langfuse docs recommend.
- `python-dotenv` - loads the shared `.env`.
- `pytest` + `httpx2` (dev group) - run tests with `uv run pytest`.

`uv sync` reads `pyproject.toml`, creates `.venv/`, resolves packages, and writes
`uv.lock`. Commit `uv.lock` (it is force-included in `.gitignore`) so a classmate
gets a byte-for-byte identical install.

## Tests

```bash
cd topics/voice_agents/09_capstone_openai/backend
uv run pytest -q
```

- `tests/test_web_search.py` - patches `Runner.run` (no network) and checks the route
  runs the agent, strips/validates the query, extracts `final_output`, and maps
  failures to clean HTTP errors.
- `tests/test_translation.py` - drives the `/translate` proxy end to end with
  in-memory fake OpenAI sockets (source-caption sidecar, phrase detection, drain on
  close).
- `tests/test_telemetry.py` - pins the "always safe" contract: with no Langfuse keys,
  telemetry is disabled and its context managers/flush are no-ops.
- `tests/test_security.py` - pins the guards: the `/health` `auth` flag, the shared
  caller token on HTTP and WebSocket routes, the per-IP rate limit (`429`), and the
  WebSocket Origin allow-list.
