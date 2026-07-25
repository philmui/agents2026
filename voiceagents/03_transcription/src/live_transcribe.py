"""
live_transcribe.py: Module 03 - live speech-to-text with the OpenAI Realtime API.

WHAT THIS PROGRAM DOES
----------------------
1. Opens a WebSocket to OpenAI and turns it into a *transcription session*
   (session.type = "transcription"), driven by the model "gpt-realtime-whisper".
2. Captures your microphone in tiny ~50 ms chunks of raw PCM16 audio at 24 kHz.
3. Base64-encodes each chunk and streams it to the server as
   `input_audio_buffer.append` events (the bytes live in the "audio" field).
4. Prints what you said, live, from the server's transcription events:
       - conversation.item.input_audio_transcription.delta      (streaming, partial)
       - conversation.item.input_audio_transcription.completed   (final for a turn)

It is a command-line program: run it, talk, watch your words appear, press Ctrl-C to stop.

HOW TO RUN
----------
    uv sync
    uv run python src/live_transcribe.py

Your OpenAI key is read from the ONE shared .env at topics/voice_agents/.env
(see that folder's .env.example). This module never puts the key in the browser.

WHY A WEBSOCKET (not plain HTTP)?
---------------------------------
HTTP is one round-trip: you ask once, you get one answer, the line closes. Transcription
is a *stream*: audio keeps flowing in and text keeps flowing back for as long as you talk.
A WebSocket is a single connection that stays open so both sides can send messages whenever
they want. That is exactly the shape of a live conversation, so the Realtime API uses it.
"""

# ---------------------------------------------------------------------------
# Imports. Each line is explained so nothing is magic.
# ---------------------------------------------------------------------------
import base64          # turns raw audio bytes into ASCII text so they fit inside JSON
import json            # every message on the wire is a JSON object (a dict we encode/decode)
import os              # to read the OPENAI_API_KEY environment variable
import queue           # a thread-safe mailbox: the mic thread drops audio, the main loop picks it up
import sys             # sys.stdout lets us print without an automatic newline (for streaming text)
import threading       # we send mic audio from a background thread so reading the socket never blocks

import numpy as np             # audio samples are just numbers; numpy holds them efficiently
import sounddevice as sd       # cross-platform microphone capture (records from your default input)
import websocket               # the "websocket-client" package: a simple, blocking WebSocket client
from dotenv import load_dotenv, find_dotenv   # finds and loads the shared .env for the API key


# ---------------------------------------------------------------------------
# Constants. Editing these in ONE place keeps the rest of the file readable.
# ---------------------------------------------------------------------------

# The transcription-only model. This is REALTIME transcription billed by the audio minute.
# It is NOT the file-based "whisper-1" endpoint where you upload a finished .wav/.mp3.
# We name the model in ONE place: session.audio.input.transcription.model (see SESSION_CONFIG).
MODEL = "gpt-realtime-whisper"

# The Realtime WebSocket endpoint. For a transcription session the canonical query is
# "?intent=transcription": it tells OpenAI up front that this connection only listens and
# returns text (the model itself is named in SESSION_CONFIG, not in the URL).
# NOTE: at GA there is NO "OpenAI-Beta: realtime=v1" header anymore. Do not add it.
WS_URL = "wss://api.openai.com/v1/realtime?intent=transcription"

# Audio format required by the Realtime API: PCM16, mono, 24000 Hz (24 kHz).
#   - PCM16      : each sample is a 16-bit signed integer (numpy dtype "int16").
#   - mono       : one channel (CHANNELS = 1), not stereo.
#   - 24000 Hz   : 24000 samples captured per second.
SAMPLE_RATE = 24000       # samples per second (Hz)
CHANNELS = 1              # mono

# We send audio in ~50 ms chunks (OpenAI's recommended size: small enough to feel live,
# large enough to avoid spamming the socket). 50 ms of 24 kHz mono audio is:
#     24000 samples/sec * 0.050 sec = 1200 samples per chunk.
FRAMES_PER_CHUNK = 1200   # = SAMPLE_RATE * 0.050


# ---------------------------------------------------------------------------
# 1) Load the API key from the shared .env (walking UP the folder tree to find it).
# ---------------------------------------------------------------------------
# find_dotenv() starts in this file's folder and walks upward until it sees a ".env",
# so it discovers topics/voice_agents/.env even though we run from 03_transcription/.
load_dotenv(find_dotenv())

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    # Fail early with a friendly message instead of a confusing error deep inside the socket.
    sys.exit(
        "OPENAI_API_KEY is not set.\n"
        "Copy topics/voice_agents/.env.example to .env and paste your key, then rerun."
    )

# Optional per-user abuse-tracking header (safe to leave blank for the course).
SAFETY_ID = os.environ.get("OPENAI_SAFETY_IDENTIFIER", "").strip()


# ---------------------------------------------------------------------------
# 2) Describe the transcription session we want (sent once, right after connecting).
# ---------------------------------------------------------------------------
# This is a "session.update" event. Its job is to configure the connection:
#   - session.type = "transcription"     -> this socket transcribes; it will NOT talk back.
#   - audio.input.format                 -> tell the server our bytes are 24 kHz PCM16.
#   - audio.input.transcription.model    -> which transcription model to use.
#   - audio.input.turn_detection         -> HOW the server decides a "turn" (a phrase) ended.
#
# TURN DETECTION: three choices (only one at a time):
#   "server_vad"    : classic Voice Activity Detection. The server watches the audio energy
#                     and cuts a turn when you go quiet for ~200 ms. Simple and predictable.
#   "semantic_vad"  : a smarter model that ends a turn when your SENTENCE sounds finished,
#                     not merely when you pause. Has an "eagerness" knob (how quickly to cut in).
#   null (Python None): MANUAL mode. The server never auto-ends a turn; YOU decide when a
#                     phrase is done by sending an "input_audio_buffer.commit" event yourself.
#
# We use server_vad here because it is the easiest mental model for a first transcription CLI.
SESSION_CONFIG = {
    "type": "session.update",
    "session": {
        "type": "transcription",            # <-- flips this connection into transcription mode
        "audio": {
            "input": {
                # Our microphone bytes are PCM at 24000 Hz. This MUST match what we actually send.
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                # Which transcription model turns the audio into text.
                "transcription": {"model": MODEL},
                # Let the server auto-detect the end of each spoken phrase.
                # Swap this line to try the other two modes:
                #   "turn_detection": {"type": "semantic_vad", "eagerness": "medium"},
                #   "turn_detection": None,   # manual: you must send input_audio_buffer.commit
                "turn_detection": {"type": "server_vad"},
            }
        },
    },
}


# ---------------------------------------------------------------------------
# 3) Microphone -> a thread-safe queue of audio chunks.
# ---------------------------------------------------------------------------
# sounddevice runs its own high-priority audio thread and calls `mic_callback` every time
# it has FRAMES_PER_CHUNK new samples. That thread must return quickly, so it does the
# absolute minimum: copy the samples into a queue. The main thread reads the queue and
# does the slow work (base64 + send). This hand-off keeps audio glitch-free.
audio_q: "queue.Queue[bytes]" = queue.Queue()


def mic_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    """Called by sounddevice on its audio thread with each new block of mic samples.

    indata : numpy array of int16 samples, shape (frames, CHANNELS).
    We flatten it to raw bytes and hand it to the queue for the main loop to send.
    """
    if status:
        # Non-fatal warnings such as "input overflow" if the machine briefly can't keep up.
        print(f"[mic warning] {status}", file=sys.stderr)
    # indata is int16 in the range -32768..32767 (that IS PCM16). .tobytes() gives the raw
    # little-endian bytes the API expects. .copy() detaches from sounddevice's reused buffer.
    audio_q.put(indata.copy().tobytes())


def sender_loop(ws: websocket.WebSocket, stop: threading.Event) -> None:
    """Background thread: pull audio chunks off the queue and stream them to the server.

    Each chunk becomes ONE `input_audio_buffer.append` event. The audio bytes are base64-
    encoded into ASCII text and placed in the "audio" field (this field name is a common
    trap: see the Caution in the tutorial).
    """
    while not stop.is_set():
        try:
            # Wait up to 100 ms for the next chunk; the timeout lets us re-check `stop`.
            chunk = audio_q.get(timeout=0.1)
        except queue.Empty:
            continue
        # base64.b64encode returns bytes; .decode("ascii") turns it into a JSON-safe string.
        b64_audio = base64.b64encode(chunk).decode("ascii")
        event = {
            "type": "input_audio_buffer.append",
            "audio": b64_audio,          # <-- the field is "audio" (NOT "delta") on the way IN
        }
        try:
            ws.send(json.dumps(event))   # json.dumps turns the dict into the text we transmit
        except websocket.WebSocketConnectionClosedException:
            break                        # the socket closed (e.g. Ctrl-C); stop quietly


# ---------------------------------------------------------------------------
# 4) Read server events and print the transcript.
# ---------------------------------------------------------------------------
def handle_server_event(raw: str) -> None:
    """Decode one server message (a JSON string) and react to the event types we care about."""
    event = json.loads(raw)           # text on the wire -> a Python dict
    etype = event.get("type", "")     # every event has a "type" string that tells us what it is

    if etype == "input_audio_buffer.speech_started":
        # Server VAD heard you START talking. Nice place to show a live cue.
        print("\n[listening...] ", end="", flush=True)

    elif etype == "input_audio_buffer.speech_stopped":
        # Server VAD heard you STOP talking; the finished transcript arrives moments later.
        pass

    elif etype == "conversation.item.input_audio_transcription.delta":
        # STREAMING partial text: words as they are recognized. Print WITHOUT a newline so
        # the phrase grows in place. flush=True forces it to the screen immediately.
        sys.stdout.write(event.get("delta", ""))
        sys.stdout.flush()

    elif etype == "conversation.item.input_audio_transcription.completed":
        # FINAL transcript for this turn. The full text is in "transcript".
        # We reprint it cleanly on its own line so the final version is unambiguous.
        print(f"\nYOU SAID: {event.get('transcript', '').strip()}\n")

    elif etype == "error":
        # The server rejected something (bad config, bad audio, rate limit, etc.). Show it.
        err = event.get("error", {})
        print(f"\n[server error] {err.get('message', err)}", file=sys.stderr)

    # Any other event types (session.created, session.updated, ...) are ignored for clarity.


# ---------------------------------------------------------------------------
# 5) Tie it all together: connect, configure, stream mic, print transcripts.
# ---------------------------------------------------------------------------
def main() -> None:
    # The Authorization header proves who we are. The API key stays server-side (this CLI),
    # never in a browser. Headers are passed to websocket-client as a list of "Key: value".
    headers = [f"Authorization: Bearer {API_KEY}"]
    if SAFETY_ID:
        headers.append(f"OpenAI-Safety-Identifier: {SAFETY_ID}")

    print(f"Connecting to {WS_URL} ...")
    # create_connection opens the WebSocket and blocks until the handshake completes.
    ws = websocket.create_connection(WS_URL, header=headers)
    print("Connected. Configuring transcription session...")

    # Send our session configuration first, before any audio.
    ws.send(json.dumps(SESSION_CONFIG))

    # `stop` is a flag both threads watch so we can shut down cleanly on Ctrl-C.
    stop = threading.Event()

    # Start the background sender thread (mic queue -> append events). daemon=True means it
    # will not keep the program alive on its own once main() returns.
    sender = threading.Thread(target=sender_loop, args=(ws, stop), daemon=True)
    sender.start()

    # Open the microphone. `sd.InputStream` calls mic_callback repeatedly on its own thread.
    #   dtype="int16"           -> capture as PCM16 (the format the API wants)
    #   blocksize=FRAMES_PER_CHUNK -> deliver ~50 ms per callback
    mic = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=FRAMES_PER_CHUNK,
        callback=mic_callback,
    )

    print("\nSpeak into your microphone. Press Ctrl-C to stop.\n")
    try:
        with mic:                       # the "with" block starts the mic and guarantees it stops
            while True:
                # The main thread just relays server events to our handler. recv() blocks
                # until the next message arrives, which is fine: the mic thread and sender
                # thread keep running independently.
                message = ws.recv()
                if message:             # recv() returns "" when the connection closes
                    handle_server_event(message)
                else:
                    break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Shut down in a tidy order: tell the sender to quit, then close the socket.
        stop.set()
        sender.join(timeout=1.0)
        try:
            ws.close()
        except Exception:
            pass
        print("Closed. Goodbye.")


if __name__ == "__main__":
    # Only run main() when this file is executed directly (not when imported for its functions).
    main()
