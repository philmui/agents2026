"""
Voice Agents · Module 05: a full-duplex terminal voice assistant.
====================================================================

Run it, then just TALK. When you stop, the assistant answers out loud and prints
what it is saying. Start talking again while it speaks and it stops to listen
(this is called "barge-in", exactly like interrupting a person mid-sentence).

"Full duplex" means both directions are open at once: your microphone is
streaming UP to OpenAI at the same time the assistant's voice is streaming DOWN
to your speakers. A walkie-talkie is half duplex (one side talks at a time); a
phone call is full duplex. We are building a phone call with an AI.

The model is `gpt-realtime-2.1`. It is "speech-to-speech": we send it raw audio
and it sends back raw audio, with no separate speech-to-text or text-to-speech
step in between. That is what makes it fast enough to feel like a conversation.

You only need BASIC Python to follow this. Every concept is explained in comments.

--------------------------------------------------------------------------------
THE FOUR MOVING PARTS (each runs at the same time, on its own thread):

  1. The WebSocket           a always-open two-way pipe to OpenAI. We SEND mic
                             audio events and RECEIVE audio + transcript events.
  2. The microphone stream   sounddevice hands us ~50 ms of raw audio at a time;
                             we base64-encode it and push it up the WebSocket.
  3. The speaker stream      sounddevice repeatedly asks us "give me more audio
                             to play"; we hand it bytes from a playback queue.
  4. The message loop        reads events coming DOWN from OpenAI and reacts:
                             queue audio to play, print transcript, handle
                             barge-in.

A "thread" is just a helper worker that runs alongside the main program so
nothing has to wait in line. sounddevice and websocket-client each start their
own threads for us; we only have to write the small callback functions they call.
--------------------------------------------------------------------------------
"""

# --- Standard-library imports (already installed with Python) ----------------
import base64          # audio bytes must travel as text over the WebSocket; base64 does that
import json            # every Realtime message is a JSON object (a Python dict on the wire)
import os              # to read OPENAI_API_KEY out of the environment
import queue           # a thread-safe FIFO buffer that holds audio waiting to be played
import sys             # to print without a trailing newline and flush immediately

# --- Third-party imports (installed by `uv sync`) ----------------------------
import numpy as np                 # to view raw mic bytes AS NUMBERS (see mic_callback)
import sounddevice as sd           # microphone in / speakers out (wraps PortAudio)
import websocket                   # the `websocket-client` package; NOTE: singular module name
from dotenv import load_dotenv, find_dotenv   # find and load the shared ../.env


# =============================================================================
# 1. CONFIGURATION: every value here is verified in _shared/API_FACTS.md
# =============================================================================

# The model that does speech-to-speech. OpenAI's GA docs use "-2.1".
# (An earlier DataCamp tutorial called it "gpt-realtime-2"; "-2.1" is canonical.)
MODEL = "gpt-realtime-2.1"

# Server-side sessions connect over a WebSocket. The model goes in the URL.
# CAUTION: at GA there is NO "OpenAI-Beta: realtime=v1" header anymore. Sending
# it is a leftover from the beta and is not needed.
WS_URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"

# Realtime audio is PCM16 at 24 kHz, mono. Let's unpack that:
#   PCM      = the raw, uncompressed audio numbers (no MP3-style compression).
#   16       = each number (each "sample") is a 16-bit signed integer, so 2 bytes.
#   24 kHz   = we take 24000 of those samples every second.
#   mono     = one channel (one microphone), not stereo.
# Both what we SEND and what we RECEIVE use exactly this format.
SAMPLE_RATE = 24000     # samples per second
CHANNELS = 1            # mono
SAMPLE_WIDTH = 2        # bytes per sample (16-bit = 2 bytes)
DTYPE = "int16"         # how sounddevice should label those 16-bit samples

# We stream the mic in small chunks. ~50 ms per chunk is OpenAI's recommendation:
# small enough to feel instant, large enough not to spam tiny messages.
#   0.05 seconds * 24000 samples/second = 1200 samples per chunk.
CHUNK_MS = 50
FRAMES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_MS / 1000)   # = 1200 samples

# The assistant's voice is chosen ONCE, before it ever speaks, and cannot be
# changed mid-session (see the Caution in the tutorial). "marin" is one of the
# OpenAI realtime voices.
VOICE = "marin"

# Optional: draw a tiny live volume meter from your mic in the terminal. It is a
# fun way to SEE that audio is just numbers, but it competes with the assistant
# transcript for the same line, so it is off by default. Flip to True to watch it.
SHOW_MIC_METER = False

# The system prompt: who the assistant is and how it should behave. Keep spoken
# answers short, because long monologues are tedious to listen to.
INSTRUCTIONS = (
    "You are a friendly, concise voice assistant for a high-school coding class. "
    "Speak in short, natural sentences. If the user interrupts you, stop and "
    "listen. Keep answers under about three sentences unless asked for more."
)


# =============================================================================
# 2. A PLAYBACK QUEUE: audio we have received but not yet played
# =============================================================================
#
# Audio arrives from OpenAI in a burst of little pieces, faster than we can play
# them. So we drop each piece into a queue (a line). The speaker stream (part 3)
# pulls from the FRONT of that line at exactly the right speed. A Queue is
# "thread-safe": the message loop can add to it while the speaker thread removes
# from it, with no crashes or corruption.
#
# We store bytes. `bytearray()` is a growable buffer we can slice cheaply, which
# is perfect because the speaker asks for a fixed number of frames each time and
# our chunks will not line up perfectly with that size.

audio_queue: "queue.Queue[bytes]" = queue.Queue()   # holds base64-decoded PCM pieces
_leftover = bytearray()   # bytes pulled from the queue but not yet handed to the speaker


def enqueue_audio(pcm_bytes: bytes) -> None:
    """Add one decoded audio piece to the back of the playback line."""
    audio_queue.put(pcm_bytes)


def clear_playback() -> None:
    """Throw away everything not yet played. Called when the user barges in:
    the moment the human starts talking again, we want silence immediately, not
    the tail end of the assistant's previous sentence."""
    global _leftover
    _leftover = bytearray()                 # forget the partial chunk in hand
    while not audio_queue.empty():          # and empty the whole line
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


# =============================================================================
# 3. THE SPEAKER (output) CALLBACK: sounddevice asks us for audio to play
# =============================================================================
#
# sounddevice runs a RawOutputStream on its own thread. Many times per second it
# calls this function saying: "I need `frames` samples RIGHT NOW to send to the
# speakers. Write them into `outdata`." Our job is to fill `outdata` from the
# queue, and to output silence (zeros) whenever the queue has run dry.
#
# Because we chose a *Raw* stream, `outdata` is a plain writable byte buffer of
# length frames * SAMPLE_WIDTH * CHANNELS (no NumPy needed). We assign EXACTLY
# that many bytes into it with `outdata[:] = chunk`. If we do not have enough
# audio, the rest MUST be zeros (silence) or the speaker will click and replay
# stale bytes.

def speaker_callback(outdata, frames, time_info, status) -> None:
    global _leftover

    if status:
        # `status` reports underflows/overflows. Printing helps you debug audio
        # dropouts, but it is not fatal, so we just note it.
        print(f"\n[speaker status] {status}", file=sys.stderr)

    needed_bytes = frames * SAMPLE_WIDTH * CHANNELS   # how many raw bytes to output

    # Keep pulling pieces off the queue until we have at least `needed_bytes`,
    # or until the queue is empty (meaning the assistant is momentarily silent).
    while len(_leftover) < needed_bytes:
        try:
            _leftover.extend(audio_queue.get_nowait())
        except queue.Empty:
            break   # nothing more available right now; we will pad with silence

    if len(_leftover) >= needed_bytes:
        # We have enough real audio: hand over exactly `needed_bytes`...
        chunk = bytes(_leftover[:needed_bytes])
        del _leftover[:needed_bytes]          # ...and keep the remainder for next time
    else:
        # Not enough: use whatever we have and pad the rest with silence (zeros).
        chunk = bytes(_leftover) + b"\x00" * (needed_bytes - len(_leftover))
        _leftover = bytearray()

    # Write the raw PCM16 bytes straight into the buffer sounddevice gave us.
    # `outdata[:] = chunk` requires len(chunk) == needed_bytes, which we ensured.
    outdata[:] = chunk


# =============================================================================
# 4. STATE we share between threads (kept tiny and obvious on purpose)
# =============================================================================
#
# `ws_app` is the WebSocket object once it is connected, so the mic thread can
# send on it. `session_ready` is False until on_open has sent our session.update,
# so the mic thread does not push audio before the session is configured (audio
# that arrives too early is rejected by the server). `assistant_speaking` tracks
# whether the assistant currently has audio playing, so we only fire the barge-in
# logic when it actually matters. `played_ms` counts how many milliseconds of
# assistant audio we have QUEUED in the CURRENT turn. It is an UPPER BOUND on what
# you actually heard (the playback queue usually holds a little more than has left
# the speaker), and it is the number `conversation.item.truncate` wants. The server
# clamps it if it is slightly too large, so a small overshoot is safe.

ws_app: "websocket.WebSocketApp | None" = None
session_ready = False                   # True once session.update has been sent
assistant_speaking = False
current_item_id: "str | None" = None    # id of the assistant message being spoken
played_ms = 0                           # ms of the current answer QUEUED so far (upper bound on heard)


# =============================================================================
# 5. THE MICROPHONE (input) CALLBACK: sounddevice hands us mic audio
# =============================================================================
#
# sounddevice runs an InputStream on its own thread too. Every ~50 ms it calls
# this with `indata`: a NumPy array of the latest mic samples. We turn those
# samples into bytes, base64-encode them (because JSON/WebSocket carry text, not
# raw bytes), and send an `input_audio_buffer.append` event up to OpenAI.
#
# We do NOT decide when a "turn" ends. Semantic VAD on OpenAI's side listens to
# this stream and decides when you have stopped talking (see the tutorial).

def mic_callback(indata, frames, time_info, status) -> None:
    if status:
        print(f"\n[mic status] {status}", file=sys.stderr)

    if ws_app is None or not session_ready:
        return   # not configured yet; drop this chunk (happens only at startup)

    # A RawInputStream hands us `indata` as a raw byte buffer: those bytes ARE
    # the PCM16 samples (2 bytes each). bytes(indata) copies them out.
    pcm_bytes = bytes(indata)

    # To make "audio is just numbers" concrete, view the SAME bytes as int16
    # numbers with NumPy and find the loudest sample in this 50 ms chunk. When
    # SHOW_MIC_METER is on, draw a little bar so you can watch your own volume.
    if SHOW_MIC_METER:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        peak = int(np.abs(samples).max()) if samples.size else 0   # 0..32767
        bar = "#" * (peak // 2000)                                 # 0..16 hashes
        print(f"\rmic |{bar:<16}|", end="", flush=True)

    # base64.b64encode turns raw bytes into ASCII text (JSON/WebSocket carry
    # text, not raw bytes); .decode() makes it a normal Python string.
    b64_audio = base64.b64encode(pcm_bytes).decode("ascii")

    # CAUTION: the base64 audio goes in the "audio" field for the APPEND event.
    # (Do not confuse this with the assistant's audio, which arrives in a "delta"
    # field, see handle_event below.)
    append_event = {
        "type": "input_audio_buffer.append",
        "audio": b64_audio,
    }
    try:
        ws_app.send(json.dumps(append_event))
    except Exception:
        # If the socket closed mid-stream, stop trying to send. The main loop
        # will report the real error; we do not want a wall of tracebacks here.
        pass


# =============================================================================
# 6. HANDLE ONE SERVER EVENT: the brain of the assistant
# =============================================================================
#
# This runs for every JSON message OpenAI sends down. We only act on the handful
# of event types we care about; everything else we ignore (there are many).

def handle_event(event: dict) -> None:
    global assistant_speaking, current_item_id, played_ms

    etype = event.get("type", "")

    # ---- The assistant's VOICE arrives here, piece by piece -----------------
    # CAUTION: the event is "response.output_audio.delta", NOT
    # "response.audio.delta". Guessing "response.audio.delta" is the single most
    # common mistake with this API and you will just hear silence.
    if etype == "response.output_audio.delta":
        # The base64 audio is in the "delta" field (NOT "audio"; that field name
        # is only for what WE send up). Decode it back to raw PCM16 bytes and
        # queue it for the speaker callback to play.
        pcm = base64.b64decode(event["delta"])
        enqueue_audio(pcm)
        assistant_speaking = True
        current_item_id = event.get("item_id", current_item_id)

        # Track how many ms of this answer we have QUEUED (not yet necessarily
        # played). Each sample is SAMPLE_WIDTH bytes; there are SAMPLE_RATE samples
        # per second. This is an upper bound on what the user has actually heard.
        samples = len(pcm) // SAMPLE_WIDTH
        played_ms += int(samples * 1000 / SAMPLE_RATE)
        return

    # ---- The assistant's WORDS arrive here, piece by piece ------------------
    # This is the text of what it is saying, streamed as it speaks. We print it
    # with no newline so it reads like live captions.
    if etype == "response.output_audio_transcript.delta":
        sys.stdout.write(event.get("delta", ""))
        sys.stdout.flush()
        return

    # ---- The user started talking: BARGE-IN ---------------------------------
    # Semantic VAD detected fresh human speech. If the assistant is mid-answer we
    # (a) stop OpenAI from generating more of that answer,
    # (b) throw away the audio we have queued but not yet played, and
    # (c) tell the server to forget the tail past what we queued, so its memory of
    #     the conversation roughly matches what you heard (played_ms is an upper bound).
    if etype == "input_audio_buffer.speech_started":
        if assistant_speaking:
            print("\n[you interrupted, assistant stops]")
            # (a) cancel the in-progress response on the server
            send_event({"type": "response.cancel"})
            # (b) silence locally, immediately
            clear_playback()
            # (c) truncate the server's record of the assistant message to the
            #     audio we had QUEUED (played_ms, an upper bound on what you heard).
            #     Over a WebSocket the event is conversation.item.truncate; WebRTC
            #     uses output_audio_buffer.clear.
            if current_item_id is not None:
                send_event({
                    "type": "conversation.item.truncate",
                    "item_id": current_item_id,
                    "content_index": 0,
                    "audio_end_ms": played_ms,
                })
            assistant_speaking = False
        return

    # ---- A turn finished cleanly --------------------------------------------
    if etype == "response.done":
        assistant_speaking = False
        current_item_id = None
        played_ms = 0
        print()   # end the caption line with a newline
        return

    # ---- What YOU said, transcribed (nice to see; optional) -----------------
    if etype == "conversation.item.input_audio_transcription.completed":
        print(f"\n[you said] {event.get('transcript', '').strip()}")
        return

    # ---- The server will tell us if we misconfigured something --------------
    if etype == "error":
        print(f"\n[server error] {json.dumps(event.get('error', event), indent=2)}")
        return


# =============================================================================
# 7. WEBSOCKET CALLBACKS: connect, receive, error, close
# =============================================================================

def send_event(event: dict) -> None:
    """Send one JSON event up to OpenAI (a thin, safe wrapper around ws.send)."""
    if ws_app is not None:
        ws_app.send(json.dumps(event))


def on_open(ws: "websocket.WebSocketApp") -> None:
    """Fires once the WebSocket handshake succeeds. This is where we configure
    the session by sending a single `session.update` event. Everything about how
    the assistant listens and speaks is set here."""
    global session_ready
    print("Connected. Configuring session...")

    # CAUTION: at GA the audio format and turn detection are NESTED under
    # session.audio.input / session.audio.output. The old FLAT fields
    # (input_audio_format: "pcm16") are legacy and will not configure GA sessions.
    session_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": INSTRUCTIONS,
            # Lower reasoning effort = lower latency, which is what you want for a
            # snappy back-and-forth voice chat (recommended for most voice agents).
            "reasoning": {"effort": "low"},
            "audio": {
                "input": {
                    # The mic audio WE send: PCM at 24 kHz.
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    # Turn ON transcription of YOUR speech (same as Module 03). In a
                    # speech-to-speech session this is OFF by default, so without this
                    # line the server never emits
                    # conversation.item.input_audio_transcription.completed and the
                    # "[you said] ..." line below would never print. Opting in makes it fire.
                    "transcription": {"model": "gpt-realtime-whisper"},
                    # semantic_vad decides when you have finished a sentence using
                    # meaning, not just silence, so it does not cut you off during
                    # a thoughtful pause. When it decides you are done, the server
                    # AUTOMATICALLY creates a response, so we never send response.create.
                    "turn_detection": {"type": "semantic_vad"},
                },
                "output": {
                    # The assistant audio we RECEIVE: also PCM at 24 kHz.
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    # Voice is chosen ONCE here and cannot change mid-session.
                    "voice": VOICE,
                },
            },
        },
    }
    send_event(session_update)
    # Only now let the mic thread start appending audio: the session is configured.
    session_ready = True
    print("Session ready. Start talking! (Press Ctrl+C to quit.)\n")


def on_message(ws: "websocket.WebSocketApp", message: str) -> None:
    """Fires for every message from OpenAI. Each message is a JSON string; we
    parse it into a dict and hand it to our event brain."""
    try:
        event = json.loads(message)
    except json.JSONDecodeError:
        return   # ignore anything that is not valid JSON (should not happen)
    handle_event(event)


def on_error(ws: "websocket.WebSocketApp", error: Exception) -> None:
    print(f"\n[websocket error] {error}", file=sys.stderr)


def on_close(ws: "websocket.WebSocketApp", status_code, msg) -> None:
    global session_ready
    session_ready = False   # stop the mic thread from sending on a dead socket
    print(f"\nConnection closed (code={status_code}).")


# =============================================================================
# 8. MAIN: wire everything together and run
# =============================================================================

def main() -> None:
    global ws_app

    # Load the shared secrets file. find_dotenv() walks UP the folder tree from
    # here until it finds a .env, so it discovers topics/voice_agents/.env even
    # though we run from the module subfolder.
    load_dotenv(find_dotenv())
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "OPENAI_API_KEY is not set. Copy topics/voice_agents/.env.example to "
            ".env and paste your key. (Realtime needs a paid tier.)"
        )

    # Build the WebSocketApp. The only header we need is Authorization. We attach
    # our four callbacks; websocket-client will call them on its own thread.
    # CAUTION: no "OpenAI-Beta" header at GA.
    ws_app = websocket.WebSocketApp(
        WS_URL,
        header=[f"Authorization: Bearer {api_key}"],
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    print("Voice Agents - Module 05 - terminal voice assistant")
    print(f"Model: {MODEL}   Voice: {VOICE}   Audio: PCM16 @ {SAMPLE_RATE} Hz mono")
    print("Connecting to OpenAI Realtime...")

    # Open BOTH audio streams and keep them open for the whole program using a
    # `with` block. RawInputStream / RawOutputStream deal in raw bytes, which is
    # exactly what we base64-encode (mic) and decode into (speaker).
    #
    #   samplerate  : must match the 24 kHz the API expects
    #   blocksize   : FRAMES_PER_CHUNK so each mic callback is ~50 ms of audio
    #   dtype       : int16, i.e. PCM16
    #   channels    : 1 (mono)
    #   callback    : the function sounddevice calls to move audio
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAMES_PER_CHUNK,
        dtype=DTYPE,
        channels=CHANNELS,
        callback=mic_callback,
    ), sd.RawOutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAMES_PER_CHUNK,
        dtype=DTYPE,
        channels=CHANNELS,
        callback=speaker_callback,
    ):
        try:
            # run_forever() blocks here, pumping the WebSocket, until the socket
            # closes or we press Ctrl+C. Meanwhile the two audio streams and the
            # WebSocket's receive thread are all running in the background.
            ws_app.run_forever()
        except KeyboardInterrupt:
            print("\nGoodbye!")
        finally:
            ws_app.close()


if __name__ == "__main__":
    main()
