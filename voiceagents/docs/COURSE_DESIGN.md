# Voice Agents Minicourse — Course Design (blueprint for all module authors)

**Audience:** advanced high-school students with *basic* Python. Explain every concept and line.
**Goal:** build, from the ground up, a web app (NextJS + React frontend, Python backend) that
demonstrates 3 OpenAI Realtime capabilities: transcription, translation, voice assistant.

**Confirmed design decisions (2026-07-25):**
1. Canonical voice-assistant model = `gpt-realtime-2.1` (note `gpt-realtime-2` as the older DataCamp name).
2. Browser path teaches the **official `@openai/agents/realtime` SDK first, then the raw WebRTC** internals.
3. Transcription + translation are **server-side Python WebSocket CLIs**; the assistant + capstone run in
   the **browser over WebRTC**. (Matches OpenAI's "WebRTC for browsers, WebSocket for servers" guidance.)

All API specifics come from `docs/API_FACTS.md` — the single source of truth. Do not contradict it.

---

## The 8 modules (each an independent subfolder under `topics/voice_agents/`)

| # | Folder | Teaches | Runnable artifact | Stack |
|---|---|---|---|---|
| 01 | `01_voice_foundations` | What voice audio *is* (analog→sampling→PCM16→24kHz→base64); WebSocket vs WebRTC vs PSTN vs HTTP; what "real-time"/latency mean | Python: record mic, print raw samples, play back | `sounddevice`, `numpy` |
| 02 | `02_realtime_handshake` | Realtime API mental model: sessions + events + the event loop; connect, `session.update`, read the event stream (no audio yet) | Python: open WS, print server events | `websocket-client` |
| 03 | `03_transcription` | Capability #1 — `gpt-realtime-whisper`, `session.type:"transcription"`, mic→append→`...transcription.completed` | Python CLI: live transcript | `websocket-client`, `sounddevice` |
| 04 | `04_translation` | Capability #2 — `gpt-realtime-translate`, `/v1/realtime/translations`, `session.`-prefixed events, output language | Python CLI: speak → translated text/audio | `websocket-client`, `sounddevice` |
| 05 | `05_voice_assistant_cli` | Capability #3 server-side — `gpt-realtime-2.1` speech-to-speech, VAD, `response.output_audio.delta`, barge-in | Python full-duplex terminal assistant | `websocket-client`, `sounddevice` |
| 06 | `06_python_backend` | The web app's backend: FastAPI `/token` route minting ephemeral `ek_` secrets via `client_secrets`; why the browser never sees the real key; CORS | FastAPI server + curl test | `fastapi`, `uvicorn`, `httpx` |
| 07 | `07_nextjs_frontend` | NextJS+React frontend: `@openai/agents/realtime` (RealtimeAgent/RealtimeSession) THEN raw `RTCPeerConnection`+SDP+`oai-events`; fetch token from module 06 | NextJS app: talk button + live transcript, talks to `gpt-realtime-2.1` | Next.js, React, `@openai/agents` |
| 08 | `08_capstone_multimode` | Unify all 3 modes (Transcribe / Translate / Assist) in one app + a ReAct tool call on the RealtimeAgent; deploy notes | Full multi-mode web app | Next.js + FastAPI |

### Dependency story (why this order)
Concepts (01) → prove a connection (02) → each capability in isolation, easiest transport first
(03, 04, 05) → stand up the backend that makes a browser safe (06) → the browser frontend (07) →
compose everything + tools (08). Each module still runs **standalone** with its own `uv`/`pyproject.toml`.

---

## Per-module deliverables (EVERY module folder must contain)

```
NN_name/
  pyproject.toml          # [tool.uv] package=false, >= lower bounds, heavy comments re: uv sync / uv.lock
  README.md               # 5-line quickstart: uv sync; set ../.env; uv run <thing>
  <name>_tutorial.md      # the step-by-step tutorial (see below)
  slides/index.html       # single-file reveal.js 5.1.0 deck (see house style below)
  src/ or app/            # runnable code (Python modules 01-06; Next.js app 07-08)
  .gitignore              # .env, .venv/, __pycache__/, node_modules/, .next/
```

**Secrets:** ONE shared `.env` at `topics/voice_agents/.env` (see `.env.example`). Python loads it with
`from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv())` so any module finds the parent
`.env` by walking up. The Next.js app reads the key only on the server (module 06 backend), never client-side.

### Tutorial markdown shape (house style, from `02_agentic_rag`/`07_advanced_retrievers`)
- Open with **the one idea** for the module (1-2 sentences), then a small **concept map table**
  (cols: concept / what it does / when it matters).
- Teach each concept in its own `##` section with plenty of **code snippets** and a
  **Caution / gotcha** callout where the API bites (e.g. `response.output_audio.delta` naming).
- Use **mermaid** diagrams for every workflow (see mermaid rules below). Back-link concepts to slide numbers.
- No em-dashes. Define jargon on first use. Assume basic Python only.

### Slide deck house style (single-file `slides/index.html`)
- reveal.js **5.1.0** from CDN, `theme/white.css`, `atom-one-light` highlight.
- **Font: Inter** (Google Fonts) with a system-sans fallback — this course leans into the user's
  Inter/pastel aesthetic. Bold (700+) titles, light/regular body. Letter-spacing -0.5px on headings.
- **Pastel palette** on stark white / off-white (and optionally a charcoal title slide). Suggested tokens
  (harmonize across all 8 decks): ink `#0f172a`, body `#334155`, muted `#64748b`,
  primary `#6366f1` (indigo pastel), accent `#ec4899` (pink pastel) / `#14b8a6` (teal),
  surfaces `#eef2ff` / `#fdf2f8` / `#f0fdfa`, line `#e2e8f0`, ok `#059669`, warn `#d97706`, bad `#dc2626`.
- **Left-aligned, top-pinned** flex sections; include the auto-fit shrink script (copy from
  `07_advanced_retrievers/slides/index.html`).
- Components: `.kicker` eyebrow (e.g. "Voice Agents · Module 03"), `.card`, `.caution`, `.cols`
  (asymmetric two-column), pills. **Never** put an accent line/underline beneath a title — use whitespace.
- **7×7 rule**: ≤7 lines/slide, ≤7 words/line. Prefer diagrams, code snippets, and SVG/mermaid over prose.
- Footer attribution: **© mui-group**.
- Embed workflow diagrams as inline SVG or a pre-rendered mermaid image in `slides/assets/` or the
  shared `topics/voice_agents/assets/`. (reveal + mermaid can be wired via the mermaid plugin, but a
  clean inline SVG is safest for offline viewing.)

### Mermaid rules (avoid overlap — the user called this out twice)
- Prefer `flowchart LR` or `TD` with short node labels; break long labels across `<br/>`.
- Keep to ≤7 nodes per diagram; split complex flows into two diagrams.
- Use `sequenceDiagram` for the client↔server event exchanges (append/commit/delta) — it naturally
  avoids line overlap and reads like a conversation, which fits the audio metaphor.
- No crossing edges where avoidable; label edges tersely (`audio`, `token`, `SDP`).

### Recurring visuals to reuse across modules (put shared ones in `topics/voice_agents/assets/`)
- `transport-compare` (HTTP vs WebSocket vs WebRTC vs PSTN) — module 01, referenced later.
- `audio-pipeline` (analog → sample → PCM16 → base64 → wire) — module 01.
- `event-loop` sequence (client append → server delta) — modules 02/05.
- `webrtc-handshake` sequence (browser → your backend token → SDP → /v1/realtime/calls) — modules 06/07.
