# Module 02 - The Realtime Handshake

Open a long-lived WebSocket **session** to OpenAI's Realtime API, send one
`session.update`, and print every **server event** as it arrives. No audio yet.
This module teaches the mental model the whole course runs on: **sessions +
events + an event loop**, and how to read that event stream to debug everything
later.

Full walkthrough: [`realtime_handshake_tutorial.md`](./realtime_handshake_tutorial.md).
Slides: [`slides/index.html`](./slides/index.html).

## Quickstart

```bash
# 1) One-time: put your OpenAI key in the SHARED secrets file (paid tier required).
cd ../                       # topics/voice_agents
cp .env.example .env         # then edit .env and paste OPENAI_API_KEY=sk-...

# 2) Install this module's deps (creates .venv from pyproject.toml).
cd 02_realtime_handshake
uv sync

# 3) Run the handshake demo. Prints the server event stream, then exits.
uv run python src/handshake_ws.py
```

Expected output (event log is the last two lines; status lines go to stderr):

```text
[connect] wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1
[open] WebSocket connected. Sending session.update ...
session.created                        server opened session id=sess_...
session.updated                        server accepted our session.update
[done] Handshake complete. Closing.
```

The script finds the shared `../.env` automatically via
`load_dotenv(find_dotenv())`, so you only set your key once for the whole course.

## What's inside

| File | Purpose |
|---|---|
| `src/handshake_ws.py` | Runnable demo: connect, `session.update`, print every server event |
| `realtime_handshake_tutorial.md` | Step-by-step tutorial with mermaid sequence diagram |
| `slides/index.html` | Single-file reveal.js deck for the module |
| `pyproject.toml` | `uv` project: deps are `websocket-client` + `python-dotenv` |

## Troubleshooting

- **`OPENAI_API_KEY is not set`** - you skipped step 1, or your `.env` is not
  under `topics/voice_agents/`. Copy `.env.example` to `.env` and paste your key.
- **An `error` event with code `401`** - the key is wrong or on the free tier.
  Realtime requires a paid tier.
- **It hangs** - press `Ctrl+C`. The demo normally closes itself right after
  `session.updated`.

Next: **Module 03** adds a microphone and turns this event stream into a live
transcript with `gpt-realtime-whisper`.
