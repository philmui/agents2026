"""
Voice Agents - Module 08 (capstone) - the ONE backend for the whole web app.

WHAT THIS FILE IS
-----------------
A single FastAPI server that powers all three modes of the capstone web app.
It has four jobs:

  1) GET  /health     -> "I am alive" + whether the API key loaded (never the key).
  2) POST /token      -> mint a SHORT-LIVED, scoped ephemeral "ek_..." token so the
     (and GET /token)     browser can open a WebRTC session with OpenAI directly.
                          Used by TRANSCRIBE mode and ASSIST mode.
  3) WS   /translate  -> a live TRANSLATION PROXY. The browser cannot open OpenAI's
                         translation WebSocket itself (that socket needs an
                         Authorization: Bearer header, and browsers cannot set
                         headers on a WebSocket). So the browser opens a plain
                         WebSocket to THIS server, we open the authenticated OpenAI
                         translation and source-transcription sockets with the real
                         key, and we relay audio + transcripts both ways. Used by
                         TRANSLATE mode.
  4) POST /web-search -> run OpenAI's hosted web search for the ASSIST agent. The
                         Realtime agent calls this as a function tool; this server
                         makes the authenticated Responses API request and returns
                         only the grounded answer.

WHY A BACKEND AT ALL?  (this is the whole security point of the course)
----------------------------------------------------------------------
Your real OpenAI key (OPENAI_API_KEY, the "sk-..." one) is the master key to your
account: it can spend money and lasts for months. If it ever reached the browser,
anyone could open DevTools, read it, and use it as their own. So the browser must
NEVER see it. This server keeps the real key on the SERVER and hands the browser
only what is safe:
  - Transcribe / Assist voice session: a temporary "ek_..." token that expires in
    ~1 minute and can only start one Realtime session. Leaking it is cheap.
  - Assist web search: a normal HTTPS call to this server. The server uses the real
    key to call the Responses API, but returns only the search answer.
  - Translate: nothing secret at all. The browser talks to THIS server; only this
    server holds the real key when it opens the OpenAI upstream sockets.

WHERE THE SECRET COMES FROM  (python-dotenv)
--------------------------------------------
We never hard-code the key. At startup we call `load_dotenv(find_dotenv())`:
find_dotenv() walks UP the folder tree until it finds a ".env" file, and
load_dotenv() reads it into the process environment. In this course that ONE
shared file lives at  topics/voice_agents/.env  (copied from .env.example). Every
Python module in the course loads it the same way, so you set your key exactly
once. See _shared/API_FACTS.md and topics/voice_agents/.env.example.
"""

# ---------------------------------------------------------------------------
# IMPORTS - every import explained
# ---------------------------------------------------------------------------

import asyncio  # run two directions of the translation relay at once (browser->OpenAI
                # and OpenAI->browser) without one blocking the other.
import base64   # inspect PCM16 chunks for lightweight source-phrase detection.
import json     # every Realtime message is a JSON object we build or parse.
import os       # read environment variables (where our secret key lives at runtime).
import sys      # PCM16 is little-endian; byteswap only on rare big-endian servers.
from array import array  # efficiently interpret PCM16 bytes as signed 16-bit samples.
from typing import Literal

import httpx    # async HTTP client. We use it to call OpenAI's /client_secrets
#               endpoint FROM our server (server-to-server) when minting a token.

import websockets  # async WebSocket CLIENT. We use it to open the authenticated
#                    OpenAI translation and transcription sockets and relay frames.
#                    (The browser side uses FastAPI's WebSocket support below.)

from dotenv import find_dotenv, load_dotenv  # load secrets from the shared .env.
#   find_dotenv() searches UPWARD for a ".env"; load_dotenv() reads it into os.environ.
#   This is how the ONE shared topics/voice_agents/.env is found no matter which
#   folder you launch uvicorn from.

from fastapi import (
    FastAPI,          # the app object; you attach "routes" (URLs) to it.
    HTTPException,    # raise this to return a clean HTTP error (like 502) to the caller.
    WebSocket,        # the server side of a browser WebSocket connection.
    WebSocketDisconnect,  # raised when the browser hangs up; we catch it to clean up.
)
from fastapi.middleware.cors import CORSMiddleware  # lets the browser (http://localhost:3000)
#   call this server (http://localhost:8000) despite the different origin. See below.

from pydantic import BaseModel  # tiny classes that describe the SHAPE of JSON, used
#   by FastAPI to validate/return data and to auto-generate the /docs page.


# ---------------------------------------------------------------------------
# LOAD SECRETS - do this once, at import time (server startup)
# ---------------------------------------------------------------------------

# Read topics/voice_agents/.env into environment variables. find_dotenv() searches
# upward from this file, so it works no matter which folder you launch uvicorn from.
load_dotenv(find_dotenv())

# Pull the real key out of the environment. os.environ.get returns None if missing;
# we check for that in each route and fail loudly, instead of calling OpenAI with an
# empty key and getting a confusing error.
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
# Be forgiving of the common dotenv typo
# OPENAI_API_KEY=OPENAI_API_KEY=sk-...
while OPENAI_API_KEY.startswith("OPENAI_API_KEY="):
    OPENAI_API_KEY = OPENAI_API_KEY.removeprefix("OPENAI_API_KEY=").strip()


def has_openai_api_key() -> bool:
    """Return whether the configured value looks like a public OpenAI API key."""
    return bool(OPENAI_API_KEY and OPENAI_API_KEY.strip().startswith("sk-"))

# Optional: a stable, hashed per-user id OpenAI can use to spot abuse per user.
# Blank for this course; we only forward it if you set it in .env.
OPENAI_SAFETY_IDENTIFIER = os.environ.get("OPENAI_SAFETY_IDENTIFIER")


# ---------------------------------------------------------------------------
# CONSTANTS - exact OpenAI endpoints/models/events. These MUST match
#             _shared/API_FACTS.md. Do NOT guess these strings.
# ---------------------------------------------------------------------------

# Mints ephemeral browser tokens. We POST to it with the REAL key; it returns an
# "ek_..." key at data.value (API_FACTS.md section 5).
CLIENT_SECRETS_URL = "https://api.openai.com/v1/realtime/client_secrets"

# Runs the hosted web_search tool for Assist mode. Realtime supports function
# tools, so the browser function calls this server route; the server delegates the
# actual internet lookup to the Responses API without exposing the permanent key.
RESPONSES_URL = "https://api.openai.com/v1/responses"

# The speech-to-speech voice model for ASSIST mode (GA canonical id). API_FACTS.md:
# "gpt-realtime-2.1" (the older DataCamp name was "gpt-realtime-2").
REALTIME_MODEL = "gpt-realtime-2.1"

# Current Responses model used for the small, latency-sensitive search delegate.
# The gpt-5.6 alias resolves to GPT-5.6 Sol and supports the hosted web_search tool.
WEB_SEARCH_MODEL = "gpt-5.6"

# The assistant's voice, chosen ONCE when minting the token; cannot change mid-session.
REALTIME_VOICE = "marin"

# The dedicated live-translation endpoint and model (API_FACTS.md sections 1-2).
# Note the path ".../realtime/translations" and the model fixed in the query string.
TRANSLATE_URL = "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"

# Source-caption sidecar for Translate mode. The translation endpoint documents
# session.input_transcript.delta, but the live service does not currently emit it
# for this app. A dedicated transcription session receives the same mic chunks and
# reliably returns the speaker's original words.
SOURCE_TRANSCRIBE_URL = "wss://api.openai.com/v1/realtime?intent=transcription"

# Translation WebSocket audio is fixed to PCM16 (16-bit signed samples), 24000 Hz,
# mono. The browser captures and plays audio at this rate. Unlike a standard
# Realtime session, a translation session does NOT accept audio format fields in
# session.update; the dedicated endpoint already defines the wire format.
SAMPLE_RATE = 24000


def build_translation_session_update(target_language: str) -> dict[str, object]:
    """Build the deliberately narrow session.update accepted by translation."""
    return {
        "type": "session.update",
        "session": {
            "audio": {
                "output": {
                    "language": target_language,
                }
            }
        },
    }


def build_source_transcription_session_update() -> dict[str, object]:
    """Configure the sidecar that captions Translate input in its source language."""
    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "transcription": {
                        "model": "gpt-realtime-whisper",
                        "delay": "low",
                    },
                }
            },
        },
    }


class SourcePhraseDetector:
    """Detect pauses in PCM16 mic audio so the transcription sidecar can commit."""

    # A deliberately modest energy threshold. The max-duration fallback below
    # still commits continuous speech even when a microphone never becomes quiet.
    SPEECH_RMS = 300
    SILENCE_SAMPLES = int(SAMPLE_RATE * 0.7)
    MIN_SPEECH_SAMPLES = int(SAMPLE_RATE * 0.2)
    MAX_SEGMENT_SAMPLES = SAMPLE_RATE * 4

    def __init__(self) -> None:
        self.buffered_samples = 0
        self.speech_samples = 0
        self.silence_samples = 0
        self.has_speech = False

    def add_chunk(self, encoded_audio: str) -> bool:
        """Return True when the accumulated source phrase should be committed."""
        try:
            pcm_bytes = base64.b64decode(encoded_audio, validate=True)
        except (ValueError, TypeError):
            return False

        # Ignore a trailing odd byte; valid PCM16 always has two bytes per sample.
        pcm_bytes = pcm_bytes[: len(pcm_bytes) - (len(pcm_bytes) % 2)]
        if not pcm_bytes:
            return False

        samples = array("h")
        samples.frombytes(pcm_bytes)
        if sys.byteorder != "little":
            samples.byteswap()

        sample_count = len(samples)
        rms = int((sum(sample * sample for sample in samples) / sample_count) ** 0.5)

        if rms >= self.SPEECH_RMS:
            # Do not let a long quiet period make the first speech chunk look like
            # a max-duration segment. Duration starts when speech actually starts.
            if not self.has_speech:
                self.buffered_samples = 0
            self.has_speech = True
            self.speech_samples += sample_count
            self.silence_samples = 0
        elif self.has_speech:
            self.silence_samples += sample_count

        if self.has_speech:
            self.buffered_samples += sample_count

        ended_on_pause = (
            self.speech_samples >= self.MIN_SPEECH_SAMPLES
            and self.silence_samples >= self.SILENCE_SAMPLES
        )
        reached_max_duration = (
            self.has_speech
            and self.buffered_samples >= self.MAX_SEGMENT_SAMPLES
        )
        return ended_on_pause or reached_max_duration

    def has_uncommitted_speech(self) -> bool:
        return self.has_speech and self.buffered_samples > 0

    def reset(self) -> None:
        self.buffered_samples = 0
        self.speech_samples = 0
        self.silence_samples = 0
        self.has_speech = False


# ---------------------------------------------------------------------------
# THE APP - create the FastAPI application object.
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Voice Agents Capstone Backend",
    description=(
        "One backend for the Module 08 capstone: mints ephemeral ek_ tokens "
        "(Transcribe + Assist), runs hosted web search for Assist, and proxies "
        "the live-translation WebSocket (Translate)."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# CORS - Cross-Origin Resource Sharing.
# ---------------------------------------------------------------------------
# Browsers enforce the "same-origin policy": by default, JavaScript on
# http://localhost:3000 (the Next.js dev site) is NOT allowed to call
# http://localhost:8000 (this server) because the origins differ (different port).
# CORS is how the SERVER says "these origins may call me." This applies to the
# /token fetch; browser WebSockets are not blocked by CORS the same way, but we
# keep a permissive DEV policy here for simplicity. In production, replace
# allow_origins with your real site's URL, not localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],   # GET, POST, OPTIONS, ... (the browser preflights POST with OPTIONS)
    allow_headers=["*"],   # allow Content-Type and any other request headers
)


# ---------------------------------------------------------------------------
# RESPONSE SHAPES - what our JSON looks like, so callers (and /docs) know.
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Shape of GET /health. Sanity-checks the server AND the config."""
    status: str            # always "ok" when the server is running
    model: str             # the realtime model this backend mints tokens for
    translate_model: str   # the translation model the /translate proxy uses
    has_api_key: bool      # True if OPENAI_API_KEY was found (does NOT reveal the key)


class TokenResponse(BaseModel):
    """Shape of POST /token (and GET /token). This is what the browser receives."""
    value: str             # the ephemeral key, e.g. "ek_abc123...". The ONLY secret we return.
    model: str             # the model the browser should connect with ("gpt-realtime-2.1")
    expires_at: int | None = None  # unix time the ek_ key dies, if OpenAI told us (else None)


class WebSearchRequest(BaseModel):
    """Shape of POST /web-search input from the Realtime function tool."""
    query: str


class WebSearchResponse(BaseModel):
    """The grounded text returned to the Realtime agent as its observation."""
    answer: str


def extract_response_output_text(data: dict[str, object]) -> str | None:
    """Extract assistant text from a raw Responses API JSON response."""
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    text_parts: list[str] = []
    output = data.get("output")
    if not isinstance(output, list):
        return None

    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in {"output_text", "text"}:
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())

    return "\n".join(text_parts) or None


# ---------------------------------------------------------------------------
# ROUTE 1:  GET /health  - a liveness + config check.
# ---------------------------------------------------------------------------
# A "route" maps a URL + HTTP method to a function. @app.get("/health") means
# "when someone does GET /health, run this and send back what it returns as JSON."
# response_model tells FastAPI the output shape (used for /docs and validation).
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    # Report whether the key loaded WITHOUT ever printing the key. bool(...) turns
    # the key string (or None) into True/False, so you can debug "did my .env load?"
    # safely from a browser at http://localhost:8000/health.
    return HealthResponse(
        status="ok",
        model=REALTIME_MODEL,
        translate_model="gpt-realtime-translate",
        has_api_key=has_openai_api_key(),
    )


# ---------------------------------------------------------------------------
# ROUTE 2:  POST /token  - mint ONE ephemeral key for the browser.
# ---------------------------------------------------------------------------
# Used by TRANSCRIBE and ASSIST mode. Step by step:
#   1. Make sure we actually have the real key (else fail with a clear message).
#   2. Call OpenAI's /client_secrets with the REAL key in the Authorization header.
#   3. Ask for a "realtime" session on gpt-realtime-2.1 with voice "marin".
#   4. Read the ephemeral key from data.value and return ONLY that to the browser.
#
# Why POST? Minting a credential CREATES something on OpenAI's side, so POST is the
# correct verb. (A GET alias at the bottom lets you test in a browser address bar.)
@app.post("/token", response_model=TokenResponse)
async def mint_token(
    mode: Literal["assist", "transcribe"] = "assist",
) -> TokenResponse:
    # --- Step 1: guard against a missing key --------------------------------
    if not has_openai_api_key():
        raise HTTPException(
            status_code=500,
            detail=(
                "OPENAI_API_KEY is missing or malformed. Set it to a paid-tier "
                "OpenAI key beginning with 'sk-' in topics/voice_agents/.env."
            ),
        )

    # --- Step 2: build the request to OpenAI --------------------------------
    # The Authorization header carries the REAL key, server-to-server over HTTPS,
    # never exposed to the browser. At GA there is NO "OpenAI-Beta" header (API_FACTS).
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENAI_SAFETY_IDENTIFIER:
        headers["OpenAI-Safety-Identifier"] = OPENAI_SAFETY_IDENTIFIER

    if mode == "transcribe":
        session = {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "transcription": {
                        "model": "gpt-realtime-whisper",
                        "delay": "low",
                    },
                }
            },
        }
        response_model = "gpt-realtime-whisper"
    else:
        session = {
            "type": "realtime",
            "model": REALTIME_MODEL,
            "audio": {"output": {"voice": REALTIME_VOICE}},
        }
        response_model = REALTIME_MODEL
    payload = {"session": session}

    # --- Step 3: actually call OpenAI (asynchronously) ----------------------
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            openai_response = await client.post(
                CLIENT_SECRETS_URL, headers=headers, json=payload
            )
    except httpx.RequestError as exc:
        # Network-level failure (DNS, refused, timeout) -> 502 Bad Gateway.
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach OpenAI to mint a token: {exc}",
        )

    # If OpenAI answered with an error status, forward a clean error for debugging.
    if openai_response.status_code != 200:
        raise HTTPException(
            status_code=openai_response.status_code,
            detail=f"OpenAI returned {openai_response.status_code}: {openai_response.text}",
        )

    # --- Step 4: extract the ephemeral key and return ONLY it ---------------
    data = openai_response.json()
    ephemeral_key = data.get("value")  # per API_FACTS.md, the ek_ key is at data.value
    if not ephemeral_key:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI response missing 'value'. Got keys: {list(data.keys())}",
        )

    # Return ONLY the ephemeral key (plus the model). The real key never appears here.
    return TokenResponse(
        value=ephemeral_key,
        model=response_model,
        expires_at=data.get("expires_at"),
    )


# ---------------------------------------------------------------------------
# CONVENIENCE:  GET /token  - same as POST, so you can test in a browser bar.
# ---------------------------------------------------------------------------
# POST is canonical and is what the frontend uses; this GET alias just calls the
# same logic so you can hit http://localhost:8000/token in a browser to smoke-test.
@app.get("/token", response_model=TokenResponse)
async def mint_token_get(
    mode: Literal["assist", "transcribe"] = "assist",
) -> TokenResponse:
    return await mint_token(mode)


# ---------------------------------------------------------------------------
# ROUTE 3: POST /web-search - hosted search for the Assist ReAct loop.
# ---------------------------------------------------------------------------
# The Realtime API exposes function tools to this browser session. Its web_search
# function posts the query here, where the permanent API key is safe. We then call
# the Responses API with its hosted web_search tool forced on, return the concise
# grounded answer, and let the Realtime model observe and speak that result.
@app.post("/web-search", response_model=WebSearchResponse)
async def web_search(request: WebSearchRequest) -> WebSearchResponse:
    if not has_openai_api_key():
        raise HTTPException(
            status_code=500,
            detail=(
                "OPENAI_API_KEY is missing or malformed. Set it to a paid-tier "
                "OpenAI key beginning with 'sk-' in topics/voice_agents/.env."
            ),
        )

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be blank")
    if len(query) > 1000:
        raise HTTPException(
            status_code=422,
            detail="query must be at most 1000 characters",
        )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENAI_SAFETY_IDENTIFIER:
        headers["OpenAI-Safety-Identifier"] = OPENAI_SAFETY_IDENTIFIER

    payload = {
        "model": WEB_SEARCH_MODEL,
        "reasoning": {"effort": "low"},
        "tools": [{"type": "web_search"}],
        # This route exists specifically to search, so do not let the delegate
        # answer from model memory without actually invoking web_search.
        "tool_choice": "required",
        "instructions": (
            "Use live web search to answer the query. Return a concise, factual, "
            "plain-text answer suitable for a voice assistant. Mention the names "
            "of important sources, but do not use Markdown or read out long URLs."
        ),
        "input": query,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            openai_response = await client.post(
                RESPONSES_URL,
                headers=headers,
                json=payload,
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach OpenAI for web search: {exc}",
        )

    if openai_response.status_code != 200:
        raise HTTPException(
            status_code=openai_response.status_code,
            detail=(
                f"OpenAI web search returned {openai_response.status_code}: "
                f"{openai_response.text}"
            ),
        )

    try:
        data = openai_response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="OpenAI web search returned invalid JSON",
        ) from exc

    answer = extract_response_output_text(data)
    if not answer:
        raise HTTPException(
            status_code=502,
            detail="OpenAI web search response contained no assistant text",
        )
    return WebSearchResponse(answer=answer)


# ---------------------------------------------------------------------------
# ROUTE 4:  WS /translate  - the LIVE TRANSLATION PROXY (Translate mode).
# ---------------------------------------------------------------------------
# THE PROBLEM this solves
# -----------------------
# OpenAI's translation endpoint is a WebSocket that authenticates with an
# "Authorization: Bearer <key>" HEADER. Browsers CANNOT set headers on a WebSocket
# (the WebSocket constructor has no headers argument), and the ephemeral ek_ token
# is scoped to a "realtime" voice session, not a translation session. So the browser
# cannot open that socket itself. Instead:
#
#     [ browser ] --plain WS--> [ THIS server ] --authenticated WS--> [ OpenAI ]
#
# The browser sends us small JSON messages carrying base64 PCM16 mic audio. We hold
# the real key and open two OpenAI sockets. The translation socket produces target
# text/audio; a transcription sidecar receives the same mic audio and produces the
# reliable original-language caption. The real key stays here the whole time.
#
# MESSAGE SHAPES on the BROWSER <-> SERVER leg (our own tiny protocol)
# -------------------------------------------------------------------
#   browser -> server : {"type": "start", "language": "es"}   (once, first)
#                       {"type": "audio", "audio": "<base64 PCM16>"}  (many)
#   server -> browser : {"type": "ready"}                     (OpenAI socket is open)
#                       {"type": "source", "item_id": "...",
#                        "delta": "..."}                       (partial source text)
#                       {"type": "source", "item_id": "...",
#                        "transcript": "...", "completed": true} (final source text)
#                       {"type": "target", "delta": "..."}    (the translation, text)
#                       {"type": "audio",  "delta": "<base64 PCM16>"} (translated speech)
#                       {"type": "error",  "message": "..."}  (something went wrong)
#
# We deliberately RE-SHAPE OpenAI's events into these simple types so the browser
# code stays tiny and does not need to know OpenAI's exact event names. The mappings
# from translation `session.*` and transcription `conversation.*` events are below.
@app.websocket("/translate")
async def translate(ws: WebSocket) -> None:
    # Accept the browser's WebSocket handshake. After this we can send/receive JSON.
    await ws.accept()

    # If the server has no key, tell the browser cleanly and stop. (Do this AFTER
    # accept() so the browser receives our message instead of a bare handshake fail.)
    if not has_openai_api_key():
        await ws.send_json(
            {
                "type": "error",
                "message": (
                    "OPENAI_API_KEY is missing or malformed. Set a paid-tier "
                    "OpenAI key beginning with 'sk-' in topics/voice_agents/.env."
                ),
            }
        )
        await ws.close()
        return

    # --- Step 1: wait for the browser's "start" message with the target language --
    # The browser sends this first so we know which language to translate INTO.
    try:
        first = await ws.receive_json()
    except WebSocketDisconnect:
        return  # browser hung up before starting; nothing to clean up yet.
    target_language = (first or {}).get("language") or "es"

    # --- Step 2: open authenticated translation + source-caption sockets ----------
    # Both upstream connections receive the same PCM16 mic audio. Translation
    # supplies target text/audio; the transcription sidecar supplies reliable text
    # for "You said" in the automatically detected source language.
    oai_headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    if OPENAI_SAFETY_IDENTIFIER:
        oai_headers["OpenAI-Safety-Identifier"] = OPENAI_SAFETY_IDENTIFIER

    try:
        # max_size=None: audio frames can be large; do not cap frame size.
        async with (
            websockets.connect(
                TRANSLATE_URL,
                additional_headers=oai_headers,
                max_size=None,
            ) as translator,
            websockets.connect(
                SOURCE_TRANSCRIBE_URL,
                additional_headers=oai_headers,
                max_size=None,
            ) as source_transcriber,
        ):
            # --- Step 3: configure both upstream sessions ---------------------------
            # Translation sessions intentionally expose a smaller schema than
            # standard Realtime sessions, so its update still contains ONLY the
            # target language. The separate transcription session accepts the
            # standard PCM format and gpt-realtime-whisper settings.
            await translator.send(
                json.dumps(build_translation_session_update(target_language))
            )
            await source_transcriber.send(
                json.dumps(build_source_transcription_session_update())
            )

            async def wait_for_configuration(socket, label: str) -> None:
                async with asyncio.timeout(10):
                    while True:
                        event = json.loads(await socket.recv())
                        etype = event.get("type", "")
                        if etype == "session.updated":
                            return
                        if etype == "error":
                            message = event.get("error", {}).get(
                                "message", f"OpenAI rejected the {label} session"
                            )
                            raise RuntimeError(message)

            # Do not start the microphone until BOTH upstream sessions acknowledge
            # their settings. Otherwise the first audio chunks or source captions
            # could be lost.
            try:
                await asyncio.gather(
                    wait_for_configuration(translator, "translation"),
                    wait_for_configuration(source_transcriber, "source transcription"),
                )
            except RuntimeError as exc:
                await ws.send_json({"type": "error", "message": str(exc)})
                return
            await ws.send_json({"type": "ready"})

            # --- Step 4: relay audio and all output concurrently --------------------
            detector = SourcePhraseDetector()
            source_pending = 0
            source_idle = asyncio.Event()
            source_idle.set()
            translation_closed = asyncio.Event()

            async def commit_source_phrase() -> None:
                """Commit one detected phrase to gpt-realtime-whisper."""
                nonlocal source_pending
                if not detector.has_uncommitted_speech():
                    return
                detector.reset()
                source_pending += 1
                source_idle.clear()
                try:
                    await source_transcriber.send(
                        json.dumps({"type": "input_audio_buffer.commit"})
                    )
                except Exception:
                    source_pending -= 1
                    if source_pending == 0:
                        source_idle.set()
                    raise

            async def pump_to_openai() -> None:
                """Forward every mic chunk to translation and source transcription."""
                while True:
                    msg = await ws.receive_json()  # raises WebSocketDisconnect on hangup
                    if msg.get("type") == "audio" and msg.get("audio"):
                        encoded_audio = msg["audio"]
                        await asyncio.gather(
                            translator.send(
                                json.dumps(
                                    {
                                        "type": "session.input_audio_buffer.append",
                                        "audio": encoded_audio,
                                    }
                                )
                            ),
                            source_transcriber.send(
                                json.dumps(
                                    {
                                        "type": "input_audio_buffer.append",
                                        "audio": encoded_audio,
                                    }
                                )
                            ),
                        )
                        if detector.add_chunk(encoded_audio):
                            await commit_source_phrase()
                    elif msg.get("type") == "stop":
                        # Flush the last source phrase and translated output. The
                        # orchestrator below waits for both before telling the
                        # browser it is safe to close.
                        await commit_source_phrase()
                        await translator.send(json.dumps({"type": "session.close"}))
                        return

            async def pump_translation_to_browser() -> None:
                """Relay target transcript/audio from the translation session."""
                async for raw in translator:
                    event = json.loads(raw)
                    etype = event.get("type", "")

                    if etype == "session.output_transcript.delta":
                        await ws.send_json(
                            {"type": "target", "delta": event.get("delta", "")}
                        )
                    elif etype == "session.output_audio.delta":
                        await ws.send_json(
                            {"type": "audio", "delta": event.get("delta", "")}
                        )
                    elif etype == "error":
                        raise RuntimeError(
                            event.get("error", {}).get(
                                "message", "Unknown translation error"
                            )
                        )
                    elif etype == "session.closed":
                        translation_closed.set()
                        return

            async def pump_source_to_browser() -> None:
                """Relay original-language deltas and finalized phrases."""
                nonlocal source_pending
                async for raw in source_transcriber:
                    event = json.loads(raw)
                    etype = event.get("type", "")
                    item_id = event.get("item_id", "source")

                    if etype == "conversation.item.input_audio_transcription.delta":
                        await ws.send_json(
                            {
                                "type": "source",
                                "item_id": item_id,
                                "delta": event.get("delta", ""),
                            }
                        )
                    elif (
                        etype
                        == "conversation.item.input_audio_transcription.completed"
                    ):
                        await ws.send_json(
                            {
                                "type": "source",
                                "item_id": item_id,
                                "transcript": event.get("transcript", ""),
                                "completed": True,
                            }
                        )
                        if source_pending > 0:
                            source_pending -= 1
                        if source_pending == 0:
                            source_idle.set()
                    elif etype == "error":
                        raise RuntimeError(
                            event.get("error", {}).get(
                                "message", "Unknown source transcription error"
                            )
                        )
                raise RuntimeError("Source transcription connection closed unexpectedly")

            input_task = asyncio.create_task(pump_to_openai())
            translation_task = asyncio.create_task(pump_translation_to_browser())
            source_task = asyncio.create_task(pump_source_to_browser())
            tasks = {input_task, translation_task, source_task}

            try:
                done, _ = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # A normal session first completes the browser-input task after a
                # "stop" message. If an upstream task ended first, surface its error.
                if input_task not in done:
                    for task in done:
                        task.result()
                    raise RuntimeError("An upstream translation task ended unexpectedly")

                input_task.result()
                for task in done - {input_task}:
                    task.result()

                # Translation session.close flushes target output. The transcription
                # sidecar has no session.close event, so wait for its last committed
                # phrase before closing the browser socket.
                async with asyncio.timeout(10):
                    await translation_closed.wait()
                if source_pending:
                    async with asyncio.timeout(10):
                        await source_idle.wait()
                await ws.send_json({"type": "closed"})
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    except WebSocketDisconnect:
        # The browser hung up (switched modes, closed the tab). Normal; just return.
        return
    except Exception as exc:  # any upstream/socket failure: tell the browser, then close.
        try:
            await ws.send_json({"type": "error", "message": f"Translation failed: {exc}"})
        except Exception:
            pass  # browser may already be gone; nothing more we can do.
    finally:
        # Always close the browser side cleanly (safe to call even if already closed).
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# RUN DIRECTLY:  `uv run python src/main.py`  (or the uvicorn command below).
# ---------------------------------------------------------------------------
# This block runs only when you execute the file directly (not when imported).
# The recommended command is still:
#     uv run uvicorn src.main:app --reload --port 8000
# but this makes `uv run python src/main.py` work too.
if __name__ == "__main__":
    import uvicorn

    # host="0.0.0.0" listens on all interfaces; port 8000 is our dev port.
    # reload=True restarts the server automatically when you edit the code.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
