"""
Voice Agents · Module 06 — the token-minting backend.

WHAT THIS FILE IS
-----------------
A tiny web server (an "API") with exactly two jobs:

  1) GET  /health  -> say "I am alive" so you can check the server started.
  2) POST /token   -> hand the browser a SHORT-LIVED, SCOPED "ephemeral" key
                      (it starts with "ek_...") that the browser can safely use
                      to open a WebRTC voice session with OpenAI.

WHY A BACKEND AT ALL?  (read this, it is the whole point of the module)
----------------------------------------------------------------------
Your real OpenAI key (OPENAI_API_KEY, the "sk-..." one) is like the master key
to your house. It can spend your money and is meant to last for months. If you
put it in the browser, ANYONE who opens the page can read it (View Source, the
Network tab, etc.) and use it as their own. That is a leaked secret.

So the browser must NEVER see the real key. Instead:
  - This backend keeps the real key on the SERVER, where visitors cannot read it.
  - When the browser wants to talk, it asks THIS server for a token.
  - This server calls OpenAI with the real key and asks for an EPHEMERAL key:
    a temporary, single-purpose "ek_..." key that expires in about a minute and
    only lets the holder start one realtime voice session. Losing it is cheap.
  - The browser gets only that ephemeral key and uses it to connect to OpenAI.

Analogy: the real key is your credit card. The ephemeral key is a one-time gift
card with $5 on it that self-destructs in a minute. You hand out gift cards, not
your credit card.

This file is the "bridge" that module 07 (the Next.js frontend) fetches from.
"""

# ---------------------------------------------------------------------------
# IMPORTS  — every import explained
# ---------------------------------------------------------------------------

import os  # read environment variables (where our secret key lives at runtime)

import httpx  # an HTTP client that can make requests *asynchronously* (see note below).
#            We use it to call OpenAI's server FROM our server (server-to-server).

from dotenv import find_dotenv, load_dotenv  # load secrets from the shared .env file.
#   find_dotenv() walks UP the folder tree until it finds a ".env"; load_dotenv reads it
#   into os.environ. This is how every module in this course finds the ONE shared
#   topics/voice_agents/.env without hard-coding a path.

from fastapi import FastAPI, HTTPException  # FastAPI = the web framework.
#   FastAPI  -> the app object; you attach "routes" (URLs) to it.
#   HTTPException -> raise this to return a clean error (like 502) to the caller.

from fastapi.middleware.cors import CORSMiddleware  # lets the browser (a different
#   origin, e.g. http://localhost:3000) call this server (http://localhost:8000).
#   Without CORS the browser BLOCKS the request. More on this below.

from pydantic import BaseModel  # tiny classes that describe the SHAPE of JSON.
#   FastAPI uses them to validate input and to document the response, and to
#   auto-generate interactive docs at /docs.


# ---------------------------------------------------------------------------
# LOAD SECRETS  — do this once, when the module is imported (server startup)
# ---------------------------------------------------------------------------

# Read topics/voice_agents/.env into environment variables. find_dotenv() searches
# upward from this file, so it works no matter which folder you launch uvicorn from.
load_dotenv(find_dotenv())

# Pull the real key out of the environment. os.environ.get returns None if it is
# missing (we check for that below and fail loudly, instead of calling OpenAI
# with an empty key and getting a confusing error).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Optional: a stable, hashed per-user id. OpenAI can use it to spot abuse per user.
# It is blank for this course; we only forward it if you set it in .env.
OPENAI_SAFETY_IDENTIFIER = os.environ.get("OPENAI_SAFETY_IDENTIFIER")


# ---------------------------------------------------------------------------
# CONSTANTS  — the exact OpenAI endpoint and session settings we ask for.
#             These MUST match docs/API_FACTS.md. Do not guess these strings.
# ---------------------------------------------------------------------------

# The one OpenAI endpoint that mints ephemeral browser tokens. We POST to it
# with the REAL key; it returns an "ek_..." key at data.value in the response.
CLIENT_SECRETS_URL = "https://api.openai.com/v1/realtime/client_secrets"

# The speech-to-speech voice model for the assistant (GA name). API_FACTS.md:
# canonical id is "gpt-realtime-2.1" (the older DataCamp name was "gpt-realtime-2").
REALTIME_MODEL = "gpt-realtime-2.1"

# The assistant's voice. Chosen ONCE here and cannot change mid-session.
# "marin" is one of OpenAI's named voices.
REALTIME_VOICE = "marin"


# ---------------------------------------------------------------------------
# THE APP  — create the FastAPI application object.
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Voice Agents Token Backend",
    description="Mints short-lived ephemeral (ek_) tokens so the browser never sees the real OpenAI key.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# CORS  — Cross-Origin Resource Sharing.
# ---------------------------------------------------------------------------
# Browsers enforce the "same-origin policy": by default, JavaScript on
# http://localhost:3000 (your Next.js dev site) is NOT allowed to fetch from
# http://localhost:8000 (this server) because the origins differ (different port).
# CORS is how the SERVER says "these origins are allowed to call me." We enable a
# PERMISSIVE policy suitable for LOCAL DEVELOPMENT only. In production you would
# replace allow_origins with your real site's URL, not "*"/localhost.
app.add_middleware(
    CORSMiddleware,
    # The frontends allowed to call us during development. Next.js dev usually
    # runs on port 3000; add 127.0.0.1 too because it is a different origin string.
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,   # allow cookies/auth headers on cross-origin requests
    allow_methods=["*"],      # allow GET, POST, OPTIONS, ... (the browser sends a
    #                           preflight OPTIONS request before POST; "*" covers it)
    allow_headers=["*"],      # allow any request headers (e.g. Content-Type: application/json)
)


# ---------------------------------------------------------------------------
# RESPONSE SHAPES  — what our JSON looks like, so callers (and /docs) know.
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Shape of GET /health. Handy to sanity-check the server AND the config."""
    status: str            # always "ok" when the server is running
    model: str             # which realtime model this backend mints tokens for
    has_api_key: bool      # True if OPENAI_API_KEY was found (does NOT reveal the key)


class TokenResponse(BaseModel):
    """Shape of POST /token. This is what the browser receives."""
    value: str             # the ephemeral key, e.g. "ek_abc123...". This is the ONLY secret we return.
    model: str             # the model the browser should connect with ("gpt-realtime-2.1")
    expires_at: int | None = None  # unix timestamp when the ek_ key dies, if OpenAI told us (else None)


# ---------------------------------------------------------------------------
# ROUTE 1:  GET /health  — a liveness + config check.
# ---------------------------------------------------------------------------
# A "route" maps a URL + HTTP method to a Python function. The decorator
# @app.get("/health") means: "when someone does GET /health, run this function
# and send back whatever it returns as JSON." response_model tells FastAPI the
# output shape (used for validation and the auto-generated /docs page).
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    # We report whether the key was loaded WITHOUT ever printing the key itself.
    # bool(...) turns the key string (or None) into True/False. This lets you
    # debug "did my .env load?" safely.
    return HealthResponse(
        status="ok",
        model=REALTIME_MODEL,
        has_api_key=bool(OPENAI_API_KEY),
    )


# ---------------------------------------------------------------------------
# ROUTE 2:  POST /token  — mint one ephemeral key for the browser.
# ---------------------------------------------------------------------------
# This is the important one. Step by step:
#   1. Make sure we actually have the real key (else fail with a clear message).
#   2. Call OpenAI's /v1/realtime/client_secrets with the REAL key in the header.
#   3. Ask for a "realtime" session on gpt-realtime-2.1 with voice "marin".
#   4. Read the ephemeral key from data.value and return ONLY that to the browser.
#
# Why POST and not GET? Minting a credential is an ACTION that creates something
# new on OpenAI's side, so POST is the correct verb. (We add a GET alias at the
# bottom purely so you can test in a browser address bar; POST is canonical.)
@app.post("/token", response_model=TokenResponse)
async def mint_token() -> TokenResponse:
    # --- Step 1: guard against a missing key --------------------------------
    # If .env was not set up, stop NOW with a helpful 500 error instead of
    # sending an empty "Bearer " to OpenAI and getting a cryptic 401 back.
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set. Copy topics/voice_agents/.env.example to .env and add your key.",
        )

    # --- Step 2: build the request to OpenAI --------------------------------
    # The Authorization header carries the REAL key. This travels server->server
    # over HTTPS and is never exposed to the browser. Note: at GA there is NO
    # "OpenAI-Beta: realtime=v1" header anymore (see API_FACTS.md) — do not add it.
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    # Only if the operator set a safety id do we forward it (optional per-user tag).
    if OPENAI_SAFETY_IDENTIFIER:
        headers["OpenAI-Safety-Identifier"] = OPENAI_SAFETY_IDENTIFIER

    # The request body. This is the "session template" the ephemeral key inherits:
    # a realtime speech-to-speech session, on our model, with the assistant voice.
    # This shape matches API_FACTS.md section 5 exactly.
    payload = {
        "session": {
            "type": "realtime",          # a speech-to-speech voice session (not transcription/translation)
            "model": REALTIME_MODEL,     # "gpt-realtime-2.1"
            "audio": {
                "output": {
                    "voice": REALTIME_VOICE,  # "marin" — locked in for the session
                }
            },
        }
    }

    # --- Step 3: actually call OpenAI (asynchronously) ----------------------
    # "async" means this function can pause while waiting for the network without
    # blocking the whole server, so it can serve other requests meanwhile.
    # httpx.AsyncClient is an HTTP client that supports that "await" style.
    # The `async with` block makes sure the network connection is cleaned up even
    # if something goes wrong. timeout=10 -> give up after 10 seconds.
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            openai_response = await client.post(
                CLIENT_SECRETS_URL,
                headers=headers,
                json=payload,  # httpx serializes this dict to JSON and sets the body
            )
    except httpx.RequestError as exc:
        # Network-level failure (DNS, connection refused, timeout). Tell the
        # browser it was an upstream problem (502 = "bad gateway", i.e. the
        # server we depend on failed), and include a short reason.
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach OpenAI to mint a token: {exc}",
        )

    # If OpenAI answered but with an error status (e.g. 401 bad key, 429 rate
    # limit), forward a clean error. .text is the raw body for debugging. We do
    # NOT leak our request; only OpenAI's reply is shown.
    if openai_response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI returned {openai_response.status_code}: {openai_response.text}",
        )

    # --- Step 4: extract the ephemeral key and return ONLY it ---------------
    # Parse the JSON body OpenAI sent back into a Python dict.
    data = openai_response.json()

    # Per API_FACTS.md, the ephemeral key lives at data["value"] and looks like
    # "ek_...". .get(...) returns None if the field is missing so we can check it.
    ephemeral_key = data.get("value")
    if not ephemeral_key:
        # Defensive: OpenAI changed the response shape or sent something odd.
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI response did not contain an ephemeral key at 'value'. Got keys: {list(data.keys())}",
        )

    # OpenAI may include when the token expires. It is optional; default to None.
    expires_at = data.get("expires_at")

    # Return ONLY the ephemeral key (plus the model to connect with). The real
    # OPENAI_API_KEY never appears in this response, so the browser cannot learn it.
    return TokenResponse(
        value=ephemeral_key,
        model=REALTIME_MODEL,
        expires_at=expires_at,
    )


# ---------------------------------------------------------------------------
# CONVENIENCE:  GET /token  — same thing, so you can test in a browser bar.
# ---------------------------------------------------------------------------
# POST is the correct verb for minting a credential, and it is what the frontend
# uses. But typing a POST into a browser address bar is awkward, so we add a GET
# alias that simply calls the same logic. Great for a quick manual check.
@app.get("/token", response_model=TokenResponse)
async def mint_token_get() -> TokenResponse:
    return await mint_token()


# ---------------------------------------------------------------------------
# RUN DIRECTLY:  `uv run python src/main.py`  (or use the uvicorn command).
# ---------------------------------------------------------------------------
# This block only runs when you execute the file directly (not when it is
# imported). It starts uvicorn, the ASGI server that actually listens on a port
# and calls into our `app`. The recommended command is still:
#     uv run uvicorn src.main:app --reload --port 8000
# but this makes `python src/main.py` work too.
if __name__ == "__main__":
    import uvicorn  # imported here so it is only needed when running directly

    # host="0.0.0.0" listens on all network interfaces; port 8000 is our dev port.
    # reload=True restarts the server automatically when you edit the code.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
