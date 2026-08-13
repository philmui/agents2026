# `backend/`: the capstone backend (FastAPI + python-dotenv)

This is the **one backend** the capstone web app talks to. It handles every
server-side operation that requires the permanent OpenAI key:

| Job | Route | Used by | Why the backend must do it |
|---|---|---|---|
| Mint ephemeral `ek_` tokens | `POST /token` (and `GET /token`) | **Transcribe**, **Assist** | The browser opens WebRTC to OpenAI directly, but only with a short-lived token. The real key never leaves the server. |
| Run hosted web search | `POST /web-search` | **Assist** | The Realtime function tool delegates to the Responses API; the browser receives only the grounded answer. |
| Proxy live translation and source captions | `WS /translate` | **Translate** | OpenAI's sockets need an `Authorization: Bearer` header, which a browser **cannot** set on a WebSocket. The backend sends each mic chunk to translation and transcription sessions, then relays the results. |
| Health + config check | `GET /health` | you (debugging) | Confirms the server is up and that `.env` loaded (without revealing the key). |

The real `OPENAI_API_KEY` is loaded from the **one shared** `topics/voice_agents/.env`
via `python-dotenv` (`load_dotenv(find_dotenv())`), exactly like every other Python
module in this course. You set your key once, in one file.

## Setup and run

```bash
# 1) From the course root, create the ONE shared .env (if you have not already):
cd topics/voice_agents
cp .env.example .env            # then paste your paid-tier OPENAI_API_KEY into .env

# 2) Install and start the backend:
cd 08_capstone_multimode/backend
uv sync                                              # creates .venv/, installs deps, writes uv.lock
uv run uvicorn src.main:app --reload --port 8000     # http://localhost:8000
```

Smoke-test it in a browser or with curl:

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"gpt-realtime-2.1","translate_model":"gpt-realtime-translate","has_api_key":true}
```

If `has_api_key` is `false`, your `.env` was not found or the key line is empty.
Fix `topics/voice_agents/.env` and restart. Interactive API docs live at
`http://localhost:8000/docs`.

## How the three routes map to the three modes

```
Transcribe  browser --GET /token--> backend --real key--> OpenAI (ek_ token)
            browser --WebRTC ek_-------------------------> OpenAI  (speech -> text)

Assist      browser --GET /token--> backend --real key--> OpenAI (ek_ token)
            browser --WebRTC ek_-------------------------> OpenAI  (talk to tool-using agent)
            browser --POST /web-search--> backend --Responses web_search--> OpenAI

Translate   browser --WS /translate--> backend --auth WS--> OpenAI translation
                                      `--auth WS--> OpenAI source transcription
            (the same mic PCM16 feeds both; the backend relays source text,
             translated text, and translated audio back)
```

Only **Translate** keeps a live socket open to the backend. Transcribe uses the
backend once to mint a token. Assist talks to OpenAI directly over WebRTC after
minting its token, and calls the backend again whenever its `web_search` function
tool runs.

The `/web-search` route uses `gpt-5.6` with the Responses API hosted
`{"type":"web_search"}` tool and `tool_choice:"required"`. It returns a short
plain-text answer for the voice agent to observe and speak.

The translation endpoint has a deliberately narrow `session.update` schema. Its
audio wire format is already PCM16 at 24 kHz, so the backend sends only
`session.audio.output.language`. Do not add standard Realtime fields such as
`session.audio.input.format`, `session.audio.output.format`, or input transcription
settings; the dedicated translation endpoint rejects them.

The live translation session does not reliably emit its documented
`session.input_transcript.delta` event. To keep **You said (source)** populated,
`/translate` also opens a `gpt-realtime-whisper` transcription session using
`?intent=transcription` and copies each mic chunk to it. Because this transcription
model does not support VAD, the backend detects pauses in PCM16 audio, commits each
phrase, and relays `conversation.item.input_audio_transcription.delta` and
`.completed` events. This sidecar adds transcription usage while Translate runs.

## Dependencies (all instantiated in `pyproject.toml`)

- `fastapi` + `uvicorn[standard]` - the web server and the ASGI runtime. The
  `[standard]` extra includes the `websockets` library uvicorn needs to accept the
  browser's `/translate` WebSocket.
- `httpx` - async HTTP client for the server-to-server `/token` and
  `/web-search` calls.
- `websockets` - async WebSocket **client** used by `/translate` to open the
  authenticated OpenAI translation and source-transcription sockets.
- `python-dotenv` - loads the shared `.env`.
- `pytest` + `httpx2` (development group) - run backend regression tests with
  `uv run pytest`; `httpx2` is the transport required by current Starlette's
  `TestClient`, while the application continues to use `httpx` for OpenAI calls.

`uv sync` reads `pyproject.toml`, creates `.venv/`, resolves the packages, and
writes `uv.lock`. Commit `uv.lock` so classmates get an identical install.

## Alternative: the Module 06 backend

Module 06 built a token-only version of this server (no `/translate` proxy). It can
still mint tokens for Transcribe/Assist, but Translate mode needs the `/translate`
route, which only exists here. For the full three-mode capstone, run **this**
backend. See `../capstone_multimode_tutorial.md` and `../../docs/API_FACTS.md`.
