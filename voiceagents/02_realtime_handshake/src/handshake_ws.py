"""
Voice Agents - Module 02: the Realtime API handshake (no audio yet).

WHAT THIS SCRIPT DOES, IN ONE SENTENCE:
    It opens a long-lived WebSocket to OpenAI's Realtime API, sends one
    configuration message (a "session.update"), and then prints every event the
    server sends back so you can SEE the conversation the two computers are having.

WHY THIS MATTERS:
    Everything later in the course (transcription, translation, the voice
    assistant) is just this same pattern with audio added: connect once, then
    trade small JSON messages called "events" back and forth. If you can read
    the event stream, you can debug anything. This module teaches you to read it.

MENTAL MODEL (keep this picture in your head):
    - A SESSION is one open WebSocket connection that stays alive.
    - On that connection you exchange EVENTS: little JSON objects, each with a
      "type" string like "session.created" or "session.update".
    - There is an EVENT LOOP: you SEND client events, and you RECEIVE server
      events, in no fixed order, for as long as the socket is open.

Run it with:
    uv run python src/handshake_ws.py
Stop it any time with Ctrl+C.
"""

# ---------------------------------------------------------------------------
# Imports (each line, explained)
# ---------------------------------------------------------------------------
import json      # to turn Python dicts into JSON text (dumps) and back (loads).
import os        # to read the API key out of the environment after dotenv loads it.
import sys       # to print to "stderr" (the error stream) for messages, keeping
                 #   the normal output stream clean for the event log.

# websocket-client gives us WebSocketApp: a WebSocket client that calls our
# functions ("callbacks") when things happen (open, message, error, close).
import websocket

# python-dotenv finds and loads the shared .env file so OPENAI_API_KEY becomes
# available via os.environ, without us ever writing the key in this file.
from dotenv import find_dotenv, load_dotenv


# ---------------------------------------------------------------------------
# Constants: the exact model + endpoint, quoted from docs/API_FACTS.md
# ---------------------------------------------------------------------------
# The canonical speech-to-speech model id at GA. (An older DataCamp tutorial
# calls it "gpt-realtime-2"; the current name is "gpt-realtime-2.1".)
MODEL = "gpt-realtime-2.1"

# The Realtime WebSocket endpoint. The model is chosen with a "?model=" query
# parameter right in the URL, the same way a web address carries "?key=value".
# Note the "wss://" scheme: that is WebSocket-over-TLS (the secure, encrypted
# form of "ws://", just like "https" is the secure form of "http").
WS_URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"


# ---------------------------------------------------------------------------
# Load the secret key from the shared .env (topics/voice_agents/.env)
# ---------------------------------------------------------------------------
# find_dotenv() walks UP the folder tree from this file until it finds a file
# named ".env". Because the whole course shares ONE .env at topics/voice_agents/,
# every module finds the same key without hard-coding a path. load_dotenv() then
# reads that file and copies its KEY=VALUE lines into the environment.
load_dotenv(find_dotenv())

# Pull the key back out of the environment. os.environ.get returns None if it is
# missing, so we can print a friendly message instead of a confusing crash.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# An optional, hashed per-user id for abuse tracking in production. Blank in the
# course. If present, we will send it as an extra header (OpenAI-Safety-Identifier).
SAFETY_ID = os.environ.get("OPENAI_SAFETY_IDENTIFIER", "").strip()

if not OPENAI_API_KEY:
    # Fail early and clearly. Beginners should never have to guess why a network
    # call 401'd; tell them exactly which file to fix.
    sys.exit(
        "ERROR: OPENAI_API_KEY is not set.\n"
        "Copy topics/voice_agents/.env.example to topics/voice_agents/.env and\n"
        "paste your key, then run this again. (The free tier cannot use Realtime.)"
    )


# ---------------------------------------------------------------------------
# The session configuration we will send once, right after the socket opens.
# ---------------------------------------------------------------------------
# A "session.update" is a CLIENT event (a message we send). It configures the
# session: who the assistant is, what it may output, and (later) audio settings.
#
# In this module we deliberately send NO audio config. We set two things:
#   - "instructions": the assistant's system prompt (its personality / rules).
#   - "output_modalities": ["text"]. A "modality" is a KIND of output. Asking for
#     text only keeps this first demo simple: there is no microphone and no
#     speaker here, so we do not want the server preparing to speak.
#
# IMPORTANT: at GA the session object carries a "type" field. For a normal
# speech/text session that value is "realtime". (Transcription sessions use
# "transcription"; we meet that in Module 03.)
SESSION_UPDATE = {
    "type": "session.update",          # the event type (client -> server)
    "session": {
        "type": "realtime",            # a normal realtime session (not transcription)
        "instructions": (
            "You are a friendly assistant for a high-school coding class. "
            "Keep answers short and encouraging."
        ),
        "output_modalities": ["text"],  # text only for this no-audio handshake demo
    },
}


# ---------------------------------------------------------------------------
# Callback: runs ONCE when the WebSocket has connected.
# ---------------------------------------------------------------------------
def on_open(ws):
    """Called by websocket-client the moment the connection is established.

    `ws` is the live socket. To SEND a client event we serialize our Python
    dict to a JSON string with json.dumps and hand it to ws.send.
    """
    print("[open] WebSocket connected. Sending session.update ...", file=sys.stderr)

    # json.dumps(...) turns the SESSION_UPDATE dict into the JSON TEXT the API
    # expects. Every event on this socket is a single JSON text message.
    ws.send(json.dumps(SESSION_UPDATE))


# ---------------------------------------------------------------------------
# Callback: runs for EVERY message the server sends. This is the event loop.
# ---------------------------------------------------------------------------
def on_message(ws, message):
    """Called once per server event. `message` is a JSON string.

    Reading these events is the whole point of the module: it is how you learn
    what the server is doing and how you debug every later feature.
    """
    # Parse the JSON text into a Python dict so we can inspect its fields.
    event = json.loads(message)

    # Every Realtime event has a "type" string. That is the first thing to read.
    event_type = event.get("type", "<no type>")

    # Build a short, human-readable summary that depends on the event type, so
    # the log teaches you what each event means instead of dumping raw JSON.
    summary = summarize(event)

    # Print "type  ->  summary". We keep this on the normal output stream so you
    # could pipe it to a file, while status/errors go to stderr.
    print(f"{event_type:<38} {summary}")

    # An "error" event means the server rejected something we sent. Surfacing it
    # loudly (and its message) is the single most useful debugging habit here.
    if event_type == "error":
        err = event.get("error", {})
        print(
            "  !! server error: "
            f"{err.get('type')} / {err.get('code')}: {err.get('message')}",
            file=sys.stderr,
        )

    # For THIS teaching demo we are done as soon as the server confirms our
    # configuration with "session.updated": we have proven the full handshake
    # (connect -> session.created -> session.update -> session.updated). Close
    # the socket so the script exits cleanly instead of waiting forever.
    if event_type == "session.updated":
        print("\n[done] Handshake complete. Closing.", file=sys.stderr)
        ws.close()


# ---------------------------------------------------------------------------
# Callback: runs if the connection errors out.
# ---------------------------------------------------------------------------
def on_error(ws, error):
    """Called on a transport-level error (bad key, no network, DNS, etc.)."""
    print(f"[error] {error}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Callback: runs when the socket closes (by us, by the server, or by a drop).
# ---------------------------------------------------------------------------
def on_close(ws, status_code, msg):
    """Called once when the connection ends. The args tell us why it closed."""
    print(f"[close] code={status_code} reason={msg!r}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Helper: turn one event dict into a short, friendly one-line summary.
# ---------------------------------------------------------------------------
def summarize(event):
    """Return a brief description tailored to a few common server event types.

    This is purely for readability. If we do not recognize a type, we fall back
    to listing the event's top-level keys so nothing is ever hidden from you.
    """
    t = event.get("type", "")

    if t == "session.created":
        # First server event. It hands us the session's id and its defaults
        # (the voice, model, and audio settings the server chose for us).
        sid = event.get("session", {}).get("id", "?")
        return f"server opened session id={sid}"

    if t == "session.updated":
        # Confirms our session.update was accepted. The echoed "session" object
        # now reflects the instructions/modalities we asked for.
        return "server accepted our session.update"

    if t == "error":
        # The important fields live under "error"; on_message prints the detail.
        return "server reported an error (see stderr)"

    if t == "rate_limits.updated":
        # The server often tells us our remaining request/token budget.
        limits = event.get("rate_limits", [])
        names = ", ".join(item.get("name", "?") for item in limits)
        return f"remaining budget for: {names}" if names else "rate limits updated"

    # Fallback: we did not special-case this type. Show its keys so you can see
    # the shape of any event we did not anticipate. This is deliberately generic.
    keys = ", ".join(k for k in event.keys() if k != "type")
    return f"(keys: {keys})" if keys else "(no extra fields)"


# ---------------------------------------------------------------------------
# main(): wire the callbacks to a WebSocketApp and run the loop.
# ---------------------------------------------------------------------------
def main():
    # The HTTP headers sent during the WebSocket "upgrade" handshake.
    #
    #   Authorization: Bearer <key>   proves who we are (server-side only!).
    #
    # CAUTION: at GA there is NO "OpenAI-Beta: realtime=v1" header. The old beta
    # required it; sending it now is unnecessary. Do not add it.
    headers = [f"Authorization: Bearer {OPENAI_API_KEY}"]

    # Optionally attach the hashed user id for per-user abuse tracking.
    if SAFETY_ID:
        headers.append(f"OpenAI-Safety-Identifier: {SAFETY_ID}")

    print(f"[connect] {WS_URL}", file=sys.stderr)

    # WebSocketApp bundles the URL + headers + our four callbacks into one
    # object. It does NOT connect yet; that happens in run_forever() below.
    ws_app = websocket.WebSocketApp(
        WS_URL,
        header=headers,        # note: the parameter is "header" (singular)
        on_open=on_open,       # called once, when connected
        on_message=on_message, # called for every server event (the event loop)
        on_error=on_error,     # called on a transport error
        on_close=on_close,     # called once, when the socket ends
    )

    # run_forever() opens the connection and then blocks, pumping the event loop
    # (dispatching each incoming message to on_message) until the socket closes.
    # We catch Ctrl+C so quitting looks tidy rather than like a crash.
    try:
        ws_app.run_forever()
    except KeyboardInterrupt:
        print("\n[quit] Ctrl+C - closing the socket.", file=sys.stderr)
        ws_app.close()


# Standard Python entry-point guard: run main() only when this file is executed
# directly (uv run python src/handshake_ws.py), not when it is imported.
if __name__ == "__main__":
    main()
