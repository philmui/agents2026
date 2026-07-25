"""
Voice Agents - the ONE backend for the whole web app.

WHAT THIS FILE IS
-----------------
A single FastAPI server that powers all three modes of the voice web app. It
exposes four routes; the browser UI talks only to these:

  1) GET  /health     -> "I am alive" + whether the API key loaded (never the key)
                          + whether Langfuse telemetry is on + whether a caller
                          token is required.
  2) POST /token      -> mint a SHORT-LIVED, scoped ephemeral "ek_..." token so the
     (and GET /token)     browser can open a WebRTC session with OpenAI directly.
                          Used by TRANSCRIBE mode and ASSIST mode.
  3) POST /web-search -> run an OpenAI Agents SDK web-search agent for ASSIST. The
                          browser's Realtime function tool posts a query here; the
                          Agent uses the hosted web_search tool and returns only the
                          grounded answer.
  4) WS   /translate  -> a live TRANSLATION PROXY (Translate mode). The browser
                          cannot set the Authorization header OpenAI's translation
                          WebSocket needs, so this server holds the key and relays
                          audio + transcripts both ways.

HOW WEB SEARCH WORKS (the OpenAI Agents SDK)
--------------------------------------------
Instead of hand-building a Responses API request, /web-search uses the OpenAI
Agents SDK: an `Agent` OWNS its tools and a `Runner` executes the reason -> act ->
observe -> respond loop. The agent's only tool is the HOSTED `WebSearchTool`, so
OpenAI runs the real internet lookup on its side. The SDK also emits structured
trace events, so the run shows up in Langfuse (see telemetry.py) with no extra code.

WHY A BACKEND AT ALL?  (the security point)
-------------------------------------------
Your real OpenAI key (OPENAI_API_KEY, the "sk-..." one) is the master key to your
account. If it ever reached the browser, anyone could read it in DevTools. So the
browser must NEVER see it. This server keeps the real key on the SERVER and hands
the browser only what is safe: a ~1-minute ephemeral "ek_..." token (Transcribe /
Assist voice), a plain HTTPS answer (Assist web search), or nothing secret at all
(Translate talks only to this server, which holds the key upstream). The paid
routes are additionally guarded (rate limit, Origin check, optional token) by
security.py.

WHERE THE SECRETS COME FROM  (python-dotenv)  -- AND IMPORT ORDER MATTERS
-------------------------------------------------------------------------
At startup we call `load_dotenv(find_dotenv())` FIRST, then import the Agents SDK
and telemetry. This ordering matters: Langfuse must be imported only AFTER the
environment is loaded, or its client would initialize with missing credentials.
find_dotenv() walks UP the folder tree until it finds a ".env"; the shared file
lives at topics/voice_agents/.env. Add your Langfuse keys to that same file (see
.env.example) and telemetry turns on automatically.
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
from contextlib import asynccontextmanager  # define the FastAPI startup/shutdown hook.
from typing import Literal

import httpx    # async HTTP client. We use it to call OpenAI's /client_secrets
#               endpoint FROM our server (server-to-server) when minting a token.

import websockets  # async WebSocket CLIENT. We use it to open the authenticated
#                    OpenAI translation and transcription sockets and relay frames.
#                    (The browser side uses FastAPI's WebSocket support below.)

from dotenv import find_dotenv, load_dotenv  # load secrets from the shared .env.

from fastapi import (
    FastAPI,          # the app object; you attach "routes" (URLs) to it.
    HTTPException,    # raise this to return a clean HTTP error (like 502) to the caller.
    Request,          # the incoming HTTP request; the access guard reads its headers/IP.
    WebSocket,        # the server side of a browser WebSocket connection.
    WebSocketDisconnect,  # raised when the browser hangs up; we catch it to clean up.
)
from fastapi.middleware.cors import CORSMiddleware  # lets the browser (localhost:3000)
#   call this server (localhost:8000) despite the different origin.

from pydantic import BaseModel  # tiny classes that describe the SHAPE of JSON.


# ---------------------------------------------------------------------------
# LOAD SECRETS FIRST - do this once, at import time (server startup), BEFORE we
# import anything that reads the environment (the Agents SDK and telemetry).
# The Langfuse skill: "Import Langfuse AFTER loading environment variables."
# ---------------------------------------------------------------------------

# Read topics/voice_agents/.env into environment variables. find_dotenv() searches
# upward from this file, so it works no matter which folder you launch uvicorn from.
load_dotenv(find_dotenv())

# --- The OpenAI Agents SDK (imported AFTER load_dotenv) ---
# Agent           : a named model + instructions + the tools it may call.
# Runner          : executes an agent to completion (the reason/act/observe loop).
# WebSearchTool   : OpenAI's HOSTED web search tool. The agent calls it and OpenAI
#                   runs the actual internet lookup; we never wire a search API.
# set_default_openai_key : hand the SDK the real key ONCE, at startup.
# set_tracing_disabled   : turn the SDK's whole trace pipeline off.
#
# HOW TRACING IS CONTROLLED (this is subtle, read it once):
# The SDK has its OWN tracing pipeline that, by default, uploads traces to OpenAI's
# dashboard using your API key. We do NOT want that. Instead:
#   * We pass use_for_tracing=False to set_default_openai_key so the real key is
#     NEVER used to export traces to OpenAI.
#   * When Langfuse IS on, we LEAVE the trace pipeline enabled, because the Langfuse
#     instrumentor listens to it and re-exports the nested model/tool spans to
#     Langfuse. Disabling it would flatten Langfuse traces (drop every nested span).
#   * When Langfuse is OFF, we call set_tracing_disabled(True) so nothing is traced
#     anywhere. This makes "telemetry disabled" truly mean no trace export at all.
# See configure_agent_tracing() below, called at startup after telemetry is set up.
from agents import (  # noqa: E402 - intentionally after load_dotenv()
    Agent,
    Runner,
    WebSearchTool,
    set_default_openai_key,
    set_tracing_disabled,
)

# Our own tiny, always-safe Langfuse wrapper (see telemetry.py). Imported AFTER
# load_dotenv so the Langfuse client sees the credentials. configure_telemetry()
# turns tracing on at startup if the packages are installed and keys are set;
# `telemetry` gives us .trace(...) and .flush() that are no-ops when it is off.
from src.telemetry import configure_telemetry, telemetry  # noqa: E402

# Lightweight access controls for the paid routes (see security.py). All degrade
# gracefully: with no env vars set, only a generous per-IP rate limit and a
# localhost Origin check apply, so the tutorial runs locally with zero setup.
from src.security import (  # noqa: E402
    auth_enabled,
    guard_http,
    ws_reject_reason,
)


# ---------------------------------------------------------------------------
# READ THE KEY out of the environment we just loaded.
# ---------------------------------------------------------------------------

# os.environ.get returns None if missing; we default to "" and strip whitespace.
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
# Be forgiving of the common dotenv typo: OPENAI_API_KEY=OPENAI_API_KEY=sk-...
while OPENAI_API_KEY.startswith("OPENAI_API_KEY="):
    OPENAI_API_KEY = OPENAI_API_KEY.removeprefix("OPENAI_API_KEY=").strip()


def has_openai_api_key() -> bool:
    """Return whether the configured value looks like a real OpenAI API key."""
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

# The speech-to-speech voice model for ASSIST mode (GA canonical id, see API_FACTS.md).
REALTIME_MODEL = "gpt-realtime-2.1"

# Text model that backs the Assist web-search AGENT. The gpt-5.6 alias resolves to
# GPT-5.6 Sol and supports the hosted web_search tool (see API_FACTS.md).
WEB_SEARCH_MODEL = "gpt-5.6"

# The assistant's voice, chosen ONCE when minting the token; cannot change mid-session.
REALTIME_VOICE = "marin"

# The dedicated live-translation endpoint and model (see API_FACTS.md).
TRANSLATE_URL = "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"

# Source-caption sidecar for Translate mode (see API_FACTS.md compatibility note). The
# translation endpoint does not reliably emit source captions, so a dedicated
# transcription session receives the same mic chunks and returns the original words.
SOURCE_TRANSCRIBE_URL = "wss://api.openai.com/v1/realtime?intent=transcription"

# Translation WebSocket audio is fixed to PCM16, 24000 Hz, mono (see API_FACTS.md).
SAMPLE_RATE = 24000

# The instructions the web-search Agent follows: search live, answer in short plain
# text a voice assistant can read aloud, name sources, no Markdown or long URLs.
WEB_SEARCH_AGENT_INSTRUCTIONS = (
    "You are a web-search delegate for a voice assistant. Always use the "
    "web_search tool to answer the user's query with live information; never "
    "answer from memory alone. Return a concise, factual, plain-text answer "
    "suitable for reading out loud. Mention the names of important sources, "
    "but do not use Markdown or read out long URLs."
)


# ---------------------------------------------------------------------------
# THE WEB-SEARCH AGENT - built ONCE and reused for every /web-search request.
# ---------------------------------------------------------------------------
# An Agent bundles a persona (instructions), a model, and the tools it may call.
# Here the only tool is OpenAI's HOSTED WebSearchTool: when the agent decides to
# search, OpenAI runs the real internet lookup on its side and hands the results
# back to the model, which then writes the grounded answer. `Runner.run(agent, q)`
# executes the whole reason -> act (search) -> observe -> respond loop for us.
#
# We build the Agent lazily (on first use) so importing this module never requires
# a key. `set_default_openai_key` gives the SDK the real key the first time.
_web_search_agent: Agent | None = None
_sdk_key_configured = False


def configure_agent_tracing() -> None:
    """Point the SDK's key/tracing at the right place, based on telemetry state.

    Called once at startup AFTER configure_telemetry() so telemetry.enabled is
    known. Two independent decisions:
      1. Give the SDK the real key for MODEL calls, but never for its own trace
         export (use_for_tracing=False), so the key is not used to ship traces to
         OpenAI's dashboard.
      2. If Langfuse is OFF, disable the SDK trace pipeline entirely, so a backend
         with no Langfuse keys emits NO traces anywhere. If Langfuse is ON, leave
         the pipeline enabled so the instrumentor can re-export nested spans.
    """
    global _sdk_key_configured
    if not has_openai_api_key():
        return
    set_default_openai_key(OPENAI_API_KEY, use_for_tracing=False)
    if not telemetry.enabled:
        set_tracing_disabled(True)
    _sdk_key_configured = True


def get_web_search_agent() -> Agent:
    """Return the shared web-search Agent, creating and configuring it once."""
    global _web_search_agent

    # If startup did not configure the key (e.g. the key arrived later), do it now.
    # configure_agent_tracing() is idempotent and safe to call again.
    if not _sdk_key_configured:
        configure_agent_tracing()

    if _web_search_agent is None:
        _web_search_agent = Agent(
            name="Web Search Delegate",
            instructions=WEB_SEARCH_AGENT_INSTRUCTIONS,
            model=WEB_SEARCH_MODEL,
            tools=[WebSearchTool()],
        )
    return _web_search_agent


# ---------------------------------------------------------------------------
# Translation session builders. The translation and transcription WebSockets are
# not part of the Agents SDK, so /translate uses a direct relay, wrapped in a
# Langfuse trace for visibility.
# ---------------------------------------------------------------------------

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
# THE APP - create the FastAPI application object, with a startup hook.
# ---------------------------------------------------------------------------
# The `lifespan` runs code once when the server starts (and once on shutdown).
# We use it to turn on Langfuse telemetry (safe no-op if not configured) and to
# flush any buffered spans cleanly when the server stops.
@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_telemetry()       # enable Langfuse tracing if packages + keys are present
    configure_agent_tracing()   # point the SDK key/tracing at Langfuse-or-nowhere
    yield
    telemetry.flush()           # push any remaining spans before shutdown


app = FastAPI(
    title="Voice Agents Capstone Backend (OpenAI Agents SDK + Langfuse)",
    description=(
        "One backend for the voice web app: mints ephemeral ek_ tokens "
        "(Transcribe + Assist), runs an OpenAI Agents SDK web-search agent for "
        "Assist, and proxies the live-translation WebSocket (Translate). Every "
        "agent run and translation session is traced to Langfuse."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS - Cross-Origin Resource Sharing (a permissive localhost dev policy).
# ---------------------------------------------------------------------------
# Browsers block JavaScript on http://localhost:3000 from calling
# http://localhost:8000 unless the SERVER says the origin is allowed. This
# permissive DEV policy allows any localhost origin. In production, replace
# allow_origin_regex with your real site's exact URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    telemetry: bool        # True if Langfuse tracing is active
    auth: bool             # True if a shared caller token is required (CAPSTONE_API_TOKEN set)


class TokenResponse(BaseModel):
    """Shape of POST /token (and GET /token). This is what the browser receives."""
    value: str             # the ephemeral key, e.g. "ek_abc123...". The ONLY secret we return.
    model: str             # the model the browser should connect with
    expires_at: int | None = None  # unix time the ek_ key dies, if OpenAI told us


class WebSearchRequest(BaseModel):
    """Shape of POST /web-search input from the Realtime function tool."""
    query: str
    # Optional: the browser passes a stable id per voice conversation so all of a
    # conversation's searches group into one Langfuse SESSION. Harmless when
    # omitted (a plain curl need not send it). Best practice: session_id.
    session_id: str | None = None


class WebSearchResponse(BaseModel):
    """The grounded text returned to the Realtime agent as its observation."""
    answer: str


# ---------------------------------------------------------------------------
# ROUTE 1:  GET /health  - a liveness + config check.
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    # Report whether the key loaded WITHOUT ever printing the key, and whether
    # Langfuse telemetry is active. Safe to open at http://localhost:8000/health.
    return HealthResponse(
        status="ok",
        model=REALTIME_MODEL,
        translate_model="gpt-realtime-translate",
        has_api_key=has_openai_api_key(),
        telemetry=telemetry.enabled,
        auth=auth_enabled(),
    )


# ---------------------------------------------------------------------------
# ROUTE 2:  POST /token  - mint ONE ephemeral key for the browser.
# ---------------------------------------------------------------------------
# Used by TRANSCRIBE and ASSIST mode. We call OpenAI's /client_secrets with the
# REAL key and return ONLY the ephemeral "ek_..." value.
@app.post("/token", response_model=TokenResponse)
async def mint_token(
    request: Request,
    mode: Literal["assist", "transcribe"] = "assist",
) -> TokenResponse:
    # Access guard: rate-limit every caller, and require the shared caller token
    # when CAPSTONE_API_TOKEN is set (no-op on localhost with no token). Minting a
    # credential spends money, so this is the first thing we check.
    guard_http(request)
    return await _mint_token_impl(mode)


async def _mint_token_impl(
    mode: Literal["assist", "transcribe"] = "assist",
) -> TokenResponse:
    """The real token-minting logic, shared by the POST route and the GET alias.
    (Kept separate so the access guard is applied once, in the route handlers.)"""
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
    # never exposed to the browser. At GA there is NO "OpenAI-Beta" header.
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
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach OpenAI to mint a token: {exc}",
        )

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

    return TokenResponse(
        value=ephemeral_key,
        model=response_model,
        expires_at=data.get("expires_at"),
    )


# ---------------------------------------------------------------------------
# CONVENIENCE:  GET /token  - same as POST, so you can test in a browser bar.
# ---------------------------------------------------------------------------
@app.get("/token", response_model=TokenResponse)
async def mint_token_get(
    request: Request,
    mode: Literal["assist", "transcribe"] = "assist",
) -> TokenResponse:
    guard_http(request)  # same access guard as the POST route
    return await _mint_token_impl(mode)


# ---------------------------------------------------------------------------
# ROUTE 3: POST /web-search - the OpenAI Agents SDK web-search agent (Assist).
# ---------------------------------------------------------------------------
# The browser's Realtime session exposes a `web_search` FUNCTION tool. When the
# voice model decides to search, the browser posts the query here. We run our
# server-side Agent (which owns the HOSTED WebSearchTool) with Runner.run(...),
# and return only the grounded answer. The permanent key stays on the server.
#
# TELEMETRY (Langfuse best practices, from the Langfuse skill):
#   * ONE trace per request, named "assist-web-search" (verb-first, low-cardinality;
#     the query is NOT in the name).
#   * Typed as_type="agent": this server-side agent is a SUBAGENT dispatched by the
#     browser's voice agent, so it shows as its own node in the Langfuse Agent Graph.
#   * Explicit input = just the user query (not a dump of every function arg); the
#     final answer is set as the explicit output.
#   * session_id groups all searches from one voice conversation; a "web-search" tag
#     enables per-feature filtering.
#   * The SDK's own tool/model spans nest INSIDE this trace automatically.
#   * flush() right after, because this is a short request.
@app.post("/web-search", response_model=WebSearchResponse)
async def web_search(request: Request, body: WebSearchRequest) -> WebSearchResponse:
    # Access guard first: rate-limit and (if configured) require the caller token.
    # Web search spends money, so we gate it before doing any work.
    guard_http(request)

    if not has_openai_api_key():
        raise HTTPException(
            status_code=500,
            detail=(
                "OPENAI_API_KEY is missing or malformed. Set it to a paid-tier "
                "OpenAI key beginning with 'sk-' in topics/voice_agents/.env."
            ),
        )

    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be blank")
    if len(query) > 1000:
        raise HTTPException(
            status_code=422,
            detail="query must be at most 1000 characters",
        )

    agent = get_web_search_agent()

    try:
        with telemetry.trace(
            "assist-web-search",
            input=query,                       # explicit input: just the query
            session_id=body.session_id,        # groups a conversation's searches
            tags=["assist", "web-search"],     # per-feature filtering
            as_type="agent",                   # a dispatched subagent (Agent Graph)
        ) as span:
            # Inside the trace, Runner.run executes the agent's reason -> search ->
            # observe -> answer loop; the SDK instrumentation records each step
            # (model, tokens, the web_search tool call) as nested observations.
            result = await Runner.run(agent, query)
            answer = (result.final_output or "").strip() if result else ""
            span.set_output(answer)            # explicit output: the final answer
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - any SDK/model/tool failure -> 502.
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI Agents web search failed: {exc}",
        ) from exc
    finally:
        # Short request: push the trace to Langfuse promptly (no-op when off).
        telemetry.flush()

    if not answer:
        raise HTTPException(
            status_code=502,
            detail="OpenAI Agents web search returned no answer text",
        )
    return WebSearchResponse(answer=answer)


# ---------------------------------------------------------------------------
# ROUTE 4:  WS /translate  - the LIVE TRANSLATION PROXY (Translate mode).
# ---------------------------------------------------------------------------
# OpenAI's translation endpoint is a
# WebSocket that authenticates with an "Authorization: Bearer <key>" HEADER, and
# browsers CANNOT set headers on a WebSocket. So the browser opens a plain socket
# to THIS server, and the server (holding the real key) opens the authenticated
# OpenAI translation + transcription sockets and relays frames both ways:
#
#     [ browser ] --plain WS--> [ THIS server ] --authenticated WS--> [ OpenAI ]
#
# The translation and transcription WebSockets are NOT part of the Agents SDK, so
# we keep the proven raw-relay code and simply wrap the whole session in one
# Langfuse trace ("translate-session", typed as a plain span) so it shows up on
# the dashboard too.
#
# Browser <-> server protocol (our own tiny message shapes, unchanged):
#   browser -> server : {"type": "start", "language": "es"}   (once, first)
#                       {"type": "audio", "audio": "<base64 PCM16>"}  (many)
#                       {"type": "stop"}                        (flush + close)
#   server -> browser : {"type": "ready"}
#                       {"type": "source", "item_id", "delta"}          (partial)
#                       {"type": "source", "item_id", "transcript", "completed"}
#                       {"type": "target", "delta"}             (translation text)
#                       {"type": "audio",  "delta": "<base64 PCM16>"} (translated speech)
#                       {"type": "closed"} | {"type": "error", "message"}
@app.websocket("/translate")
async def translate(ws: WebSocket) -> None:
    await ws.accept()

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
    # The first message also carries the optional caller token, so we read it
    # before applying the access guard.
    try:
        first = await ws.receive_json()
    except WebSocketDisconnect:
        return  # browser hung up before starting; nothing to clean up yet.

    # Access guard: reject disallowed Origins, rate-limit the caller IP, and (if
    # CAPSTONE_API_TOKEN is set) require a matching token in this first message.
    # On localhost with no token configured, this passes silently.
    reject = ws_reject_reason(ws, first if isinstance(first, dict) else None)
    if reject:
        await ws.send_json({"type": "error", "message": reject})
        await ws.close()
        return

    target_language = (first or {}).get("language") or "es"
    session_id = (first or {}).get("session_id")  # optional grouping id

    # --- Step 2: open authenticated translation + source-caption sockets ----------
    oai_headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    if OPENAI_SAFETY_IDENTIFIER:
        oai_headers["OpenAI-Safety-Identifier"] = OPENAI_SAFETY_IDENTIFIER

    # One Langfuse trace for the whole translation session (no-op when disabled).
    # Named "translate-session" (verb-first, low-cardinality); the target language
    # is metadata/input, not part of the name.
    try:
        with telemetry.trace(
            "translate-session",
            input={"target_language": target_language},
            session_id=session_id,
            tags=["translate"],
            as_type="span",
        ):
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
                # --- Step 3: configure both upstream sessions -------------------
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

                # Do not start the microphone until BOTH upstream sessions ack.
                try:
                    await asyncio.gather(
                        wait_for_configuration(translator, "translation"),
                        wait_for_configuration(
                            source_transcriber, "source transcription"
                        ),
                    )
                except RuntimeError as exc:
                    await ws.send_json({"type": "error", "message": str(exc)})
                    return
                await ws.send_json({"type": "ready"})

                # --- Step 4: relay audio and all output concurrently ------------
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
                    """Forward every mic chunk to translation and transcription."""
                    while True:
                        msg = await ws.receive_json()  # raises WebSocketDisconnect
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
                            await commit_source_phrase()
                            await translator.send(
                                json.dumps({"type": "session.close"})
                            )
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
                    raise RuntimeError(
                        "Source transcription connection closed unexpectedly"
                    )

                input_task = asyncio.create_task(pump_to_openai())
                translation_task = asyncio.create_task(pump_translation_to_browser())
                source_task = asyncio.create_task(pump_source_to_browser())
                tasks = {input_task, translation_task, source_task}

                try:
                    done, _ = await asyncio.wait(
                        tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # A normal session first completes the browser-input task after
                    # a "stop". If an upstream task ended first, surface its error.
                    if input_task not in done:
                        for task in done:
                            task.result()
                        raise RuntimeError(
                            "An upstream translation task ended unexpectedly"
                        )

                    input_task.result()
                    for task in done - {input_task}:
                        task.result()

                    # Translation session.close flushes target output. The
                    # transcription sidecar has no session.close event, so wait
                    # for its last committed phrase before closing the browser.
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
    except Exception as exc:  # any upstream/socket failure: tell the browser, close.
        try:
            await ws.send_json(
                {"type": "error", "message": f"Translation failed: {exc}"}
            )
        except Exception:
            pass  # browser may already be gone; nothing more we can do.
    finally:
        telemetry.flush()  # push the translate-session trace (no-op when off)
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# RUN DIRECTLY:  `uv run python -m src.main`  (or the uvicorn command in README).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    # host="0.0.0.0" listens on all interfaces; port 8000 is our dev port.
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
