"""
Voice Agents · Module 05 (bonus): the SAME assistant, but MANUAL turns.
========================================================================

`voice_assistant.py` uses semantic VAD: OpenAI listens to your mic, decides on
its own when you have finished a sentence, and then AUTOMATICALLY asks the model
to reply. You never say "your turn is over" and you never say "now respond".

This file removes that magic so you can see what VAD was doing for you. Here we
set turn detection to null (`turn_detection = None`), which means:

  * The server never decides your turn is over.
  * The server never auto-creates a response.
  * YOU are in charge. This is a classic "push to talk" walkie-talkie:
      1. Hold nothing, just talk.
      2. Press ENTER when you are done. That sends TWO events:
           - input_audio_buffer.commit  ("that phrase is complete")
           - response.create            ("now answer me")
      3. Listen to the reply, then talk again and press ENTER again.

Everything else (audio format, voice, speaker playback) is identical to
voice_assistant.py. Compare the two files side by side to feel the difference
between automatic (VAD) and manual turn-taking. See the tutorial, section
"VAD-driven auto response vs manual".

NOTE ON BARGE-IN: with manual turns there is no speech_started event to trigger
an interruption, so this simple version does not implement barge-in. Barge-in is
a VAD feature; it lives in voice_assistant.py.
"""

# --- Standard-library imports ------------------------------------------------
import base64          # audio bytes travel as text over the WebSocket
import json            # every Realtime message is a JSON object
import os              # to read OPENAI_API_KEY
import queue           # thread-safe buffer of audio waiting to be played
import sys             # print without a newline, flush immediately, exit cleanly
import threading       # to read keyboard ENTER without blocking the audio/socket

# --- Third-party imports (installed by `uv sync`) ----------------------------
import sounddevice as sd           # microphone in / speakers out
import websocket                   # the `websocket-client` package (singular module)
from dotenv import load_dotenv, find_dotenv


# =============================================================================
# 1. CONFIGURATION (identical values to voice_assistant.py)
# =============================================================================

MODEL = "gpt-realtime-2.1"                                   # speech-to-speech model
WS_URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"   # server-side WebSocket

SAMPLE_RATE = 24000     # samples per second (24 kHz)
CHANNELS = 1            # mono
SAMPLE_WIDTH = 2        # bytes per PCM16 sample
DTYPE = "int16"         # PCM16
FRAMES_PER_CHUNK = int(SAMPLE_RATE * 50 / 1000)   # ~50 ms per mic chunk = 1200 samples

VOICE = "marin"         # chosen once; cannot change mid-session

INSTRUCTIONS = (
    "You are a friendly, concise voice assistant for a high-school coding class. "
    "Speak in short, natural sentences. Keep answers to about three sentences."
)


# =============================================================================
# 2. PLAYBACK QUEUE + SPEAKER CALLBACK (same design as voice_assistant.py)
# =============================================================================

audio_queue: "queue.Queue[bytes]" = queue.Queue()
_leftover = bytearray()


def speaker_callback(outdata, frames, time_info, status) -> None:
    """sounddevice calls this many times per second asking for `frames` samples.
    We fill `outdata` from the queue and pad with silence (zeros) when empty."""
    global _leftover
    if status:
        print(f"\n[speaker status] {status}", file=sys.stderr)

    needed = frames * SAMPLE_WIDTH * CHANNELS
    while len(_leftover) < needed:
        try:
            _leftover.extend(audio_queue.get_nowait())
        except queue.Empty:
            break

    if len(_leftover) >= needed:
        outdata[:] = bytes(_leftover[:needed])
        del _leftover[:needed]
    else:
        outdata[:] = bytes(_leftover) + b"\x00" * (needed - len(_leftover))
        _leftover = bytearray()


# =============================================================================
# 3. SHARED STATE
# =============================================================================

ws_app: "websocket.WebSocketApp | None" = None
session_ready = False   # True once session.update has been sent


def send_event(event: dict) -> None:
    """Send one JSON event up to OpenAI."""
    if ws_app is not None:
        ws_app.send(json.dumps(event))


# =============================================================================
# 4. MIC CALLBACK: stream audio up as input_audio_buffer.append
# =============================================================================
#
# This is the same as voice_assistant.py. The mic keeps flowing UP the whole
# time; the difference is only in WHO ends the turn. Here the server never ends
# it, so the audio simply accumulates in the server's input buffer until we
# commit it by pressing ENTER.

def mic_callback(indata, frames, time_info, status) -> None:
    if status:
        print(f"\n[mic status] {status}", file=sys.stderr)
    if ws_app is None or not session_ready:
        return
    b64_audio = base64.b64encode(bytes(indata)).decode("ascii")
    try:
        send_event({"type": "input_audio_buffer.append", "audio": b64_audio})
    except Exception:
        pass   # socket closed mid-stream; the main loop reports the real error


# =============================================================================
# 5. HANDLE SERVER EVENTS (only the ones we need to hear the reply)
# =============================================================================

def handle_event(event: dict) -> None:
    etype = event.get("type", "")

    # The assistant's VOICE. CAUTION: the type is response.output_audio.delta
    # (NOT response.audio.delta) and the bytes are in the "delta" field.
    if etype == "response.output_audio.delta":
        audio_queue.put(base64.b64decode(event["delta"]))
        return

    # The assistant's WORDS, streamed as live captions.
    if etype == "response.output_audio_transcript.delta":
        sys.stdout.write(event.get("delta", ""))
        sys.stdout.flush()
        return

    # Turn finished: newline, then re-print the prompt so you know it is your go.
    if etype == "response.done":
        print("\n\n(talk, then press ENTER to send)")
        return

    if etype == "error":
        print(f"\n[server error] {json.dumps(event.get('error', event), indent=2)}")
        return


# =============================================================================
# 6. THE KEYBOARD THREAD: ENTER commits the turn and asks for a response
# =============================================================================
#
# input() blocks until you press ENTER, which would freeze the whole program if
# we called it on the main thread. So we run it on its own thread. Each ENTER
# does the two manual steps that VAD would otherwise do for us.

def keyboard_loop(stop: "threading.Event") -> None:
    while not stop.is_set():
        try:
            input()   # wait for ENTER (the text you type is ignored)
        except EOFError:
            return
        if stop.is_set():
            return
        # Step 1: "that phrase is complete." Only valid because turn_detection is
        # null; with VAD on, committing manually fights the automatic detector.
        send_event({"type": "input_audio_buffer.commit"})
        # Step 2: "now respond." With VAD this is automatic; here we ask for it.
        send_event({"type": "response.create"})


# =============================================================================
# 7. WEBSOCKET CALLBACKS
# =============================================================================

def on_open(ws: "websocket.WebSocketApp") -> None:
    global session_ready
    print("Connected. Configuring session (manual turns)...")

    session_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": INSTRUCTIONS,
            "reasoning": {"effort": "low"},
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    # THE KEY LINE: null turn detection = no VAD. The server will
                    # not end your turn and will not auto-create a response. That
                    # is now YOUR job (see keyboard_loop). In JSON, Python None
                    # becomes null.
                    "turn_detection": None,
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "voice": VOICE,
                },
            },
        },
    }
    send_event(session_update)
    session_ready = True
    print("Session ready. Talk, then press ENTER to get a reply. Ctrl+C to quit.\n")


def on_message(ws, message: str) -> None:
    try:
        handle_event(json.loads(message))
    except json.JSONDecodeError:
        pass


def on_error(ws, error: Exception) -> None:
    print(f"\n[websocket error] {error}", file=sys.stderr)


def on_close(ws, status_code, msg) -> None:
    global session_ready
    session_ready = False
    print(f"\nConnection closed (code={status_code}).")


# =============================================================================
# 8. MAIN
# =============================================================================

def main() -> None:
    global ws_app

    load_dotenv(find_dotenv())
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "OPENAI_API_KEY is not set. Copy topics/voice_agents/.env.example to "
            ".env and paste your key. (Realtime needs a paid tier.)"
        )

    ws_app = websocket.WebSocketApp(
        WS_URL,
        header=[f"Authorization: Bearer {api_key}"],   # no OpenAI-Beta header at GA
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    print("Voice Agents - Module 05 (bonus) - PUSH TO TALK (manual turns)")
    print(f"Model: {MODEL}   Voice: {VOICE}   Audio: PCM16 @ {SAMPLE_RATE} Hz mono")

    # Start the keyboard thread. It is a daemon so it dies with the program.
    stop = threading.Event()
    kb = threading.Thread(target=keyboard_loop, args=(stop,), daemon=True)
    kb.start()

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=FRAMES_PER_CHUNK,
        dtype=DTYPE, channels=CHANNELS, callback=mic_callback,
    ), sd.RawOutputStream(
        samplerate=SAMPLE_RATE, blocksize=FRAMES_PER_CHUNK,
        dtype=DTYPE, channels=CHANNELS, callback=speaker_callback,
    ):
        try:
            ws_app.run_forever()
        except KeyboardInterrupt:
            print("\nGoodbye!")
        finally:
            stop.set()
            ws_app.close()


if __name__ == "__main__":
    main()
