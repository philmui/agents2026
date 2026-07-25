"""
manual_commit.py: Module 03 bonus - MANUAL turn detection (turn_detection = null).

WHY THIS EXISTS
---------------
`live_transcribe.py` lets the SERVER decide when your phrase ends (server_vad). This file
shows the OPPOSITE: turn_detection is turned OFF (null), so the server never auto-cuts a
turn. YOU decide when a phrase is finished by pressing ENTER, which sends an
`input_audio_buffer.commit` event. Only then does the server transcribe what it has buffered.

Use manual commit when you want exact control over phrase boundaries (for example, a
push-to-talk button) instead of trusting an automatic pause detector.

HOW TO RUN
----------
    uv run python src/manual_commit.py

Then: talk, press ENTER to finish a phrase (that commits + transcribes), talk again,
press ENTER again, ... and Ctrl-C to quit.
"""

import base64
import json
import os
import queue
import sys
import threading

import numpy as np
import sounddevice as sd
import websocket
from dotenv import load_dotenv, find_dotenv

# ---- Same constants and setup as live_transcribe.py (kept here so this file stands alone) ----
MODEL = "gpt-realtime-whisper"                              # named in SESSION_CONFIG, not the URL
WS_URL = "wss://api.openai.com/v1/realtime?intent=transcription"   # canonical transcription query
SAMPLE_RATE = 24000
CHANNELS = 1
FRAMES_PER_CHUNK = 1200                 # ~50 ms of 24 kHz mono audio

load_dotenv(find_dotenv())
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    sys.exit("OPENAI_API_KEY is not set. See topics/voice_agents/.env.example.")

# The ONLY meaningful difference from live_transcribe.py: turn_detection is None (null).
# With no automatic VAD, the buffer keeps growing until WE send a commit event.
SESSION_CONFIG = {
    "type": "session.update",
    "session": {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                "transcription": {"model": MODEL},
                "turn_detection": None,     # <-- MANUAL mode: no auto turn-taking
            }
        },
    },
}

audio_q: "queue.Queue[bytes]" = queue.Queue()


def mic_callback(indata, frames, time_info, status):
    """Copy each mic block (int16 PCM16) into the queue for the sender thread."""
    if status:
        print(f"[mic warning] {status}", file=sys.stderr)
    audio_q.put(indata.copy().tobytes())


def sender_loop(ws, stop):
    """Stream mic chunks as input_audio_buffer.append events (bytes in the 'audio' field)."""
    while not stop.is_set():
        try:
            chunk = audio_q.get(timeout=0.1)
        except queue.Empty:
            continue
        event = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("ascii"),
        }
        try:
            ws.send(json.dumps(event))
        except websocket.WebSocketConnectionClosedException:
            break


def reader_loop(ws, stop):
    """Print transcripts from the server until the socket closes or we are told to stop."""
    while not stop.is_set():
        try:
            raw = ws.recv()
        except (websocket.WebSocketConnectionClosedException, OSError):
            break
        if not raw:
            break
        event = json.loads(raw)
        etype = event.get("type", "")
        if etype == "conversation.item.input_audio_transcription.completed":
            print(f"\nYOU SAID: {event.get('transcript', '').strip()}\n")
        elif etype == "error":
            err = event.get("error", {})
            print(f"\n[server error] {err.get('message', err)}", file=sys.stderr)


def main():
    headers = [f"Authorization: Bearer {API_KEY}"]
    ws = websocket.create_connection(WS_URL, header=headers)
    ws.send(json.dumps(SESSION_CONFIG))

    stop = threading.Event()
    threading.Thread(target=sender_loop, args=(ws, stop), daemon=True).start()
    threading.Thread(target=reader_loop, args=(ws, stop), daemon=True).start()

    mic = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=FRAMES_PER_CHUNK,
        callback=mic_callback,
    )

    print("MANUAL mode. Talk, then press ENTER to finish a phrase. Ctrl-C to quit.\n")
    try:
        with mic:
            while True:
                # input() blocks until you press ENTER. That is our "phrase is done" signal.
                input()
                # commit finalizes the buffered audio as one turn; the server transcribes it.
                ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
    except (KeyboardInterrupt, EOFError):
        print("\nStopping...")
    finally:
        stop.set()
        try:
            ws.close()
        except Exception:
            pass
        print("Closed. Goodbye.")


if __name__ == "__main__":
    main()
