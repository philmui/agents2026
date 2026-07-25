#!/usr/bin/env python3
"""
Live voice translator (Module 04 of the Voice Agents minicourse).

Speak into your microphone in almost any language. This program streams your
voice to OpenAI's dedicated translation model, `gpt-realtime-translate`, and:

  * prints the SOURCE text it heard    (session.input_transcript.delta)
  * prints the TARGET text it produced (session.output_transcript.delta)
  * plays the TRANSLATED speech back    (session.output_audio.delta)

You choose ONLY the target language. The source language is auto-detected.

--------------------------------------------------------------------------
Run it:
    uv run python src/translate.py                 # asks you for a language
    uv run python src/translate.py --to Spanish    # or pass one directly
    uv run python src/translate.py --to French --list-devices   # debug audio

Press Ctrl+C to stop.
--------------------------------------------------------------------------

Everything here is explained line by line in translation_tutorial.md.
The audience is assumed to know only basic Python.
"""

# ----------------------------------------------------------------------------
# Imports. Standard-library first, then the third-party packages that
# `uv sync` installed from pyproject.toml.
# ----------------------------------------------------------------------------
import argparse          # parse command-line flags like  --to Spanish
import base64            # audio travels over the wire as base64 text; we decode/encode it
import json              # every Realtime message is a JSON object
import os                # read the OPENAI_API_KEY environment variable
import queue             # a thread-safe mailbox to hand mic audio to the sender
import sys               # write to stderr and exit cleanly
import threading         # the mic runs on its own thread while we read the socket

import numpy as np              # sounddevice hands us audio as NumPy arrays
import sounddevice as sd        # capture the microphone and play speakers
import websocket                # the "websocket-client" package (NOT "websockets")
from dotenv import find_dotenv, load_dotenv   # locate + load the shared .env file


# ----------------------------------------------------------------------------
# Constants that come straight from _shared/API_FACTS.md. Do not guess these.
# ----------------------------------------------------------------------------

# The dedicated translation endpoint. Note the path is ".../translations"
# (module 03 transcription uses ".../realtime?..." with session.type instead).
# The model is fixed to gpt-realtime-translate via the query string.
TRANSLATE_URL = "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"

# The Realtime API speaks PCM16 audio at 24000 Hz, mono. "PCM16" means each
# audio sample is a 16-bit signed integer. 24000 Hz means 24000 samples per
# second. Mono means one channel. These three numbers MUST match the API.
SAMPLE_RATE = 24000          # samples per second (Hz)
CHANNELS = 1                 # mono (one microphone channel)
DTYPE = "int16"              # 16-bit signed integer per sample == "PCM16"

# We send the mic in small chunks. ~50 ms is the OpenAI-recommended size.
# 24000 samples/sec * 0.05 sec = 1200 samples per chunk.
CHUNK_MS = 50
FRAMES_PER_CHUNK = SAMPLE_RATE * CHUNK_MS // 1000     # = 1200 samples


# ----------------------------------------------------------------------------
# Helper: turn a friendly language name into the model's language code.
#
# gpt-realtime-translate auto-detects the input from 70+ languages and
# translates into 13. It accepts either a BCP-47 language code (e.g. "es") or,
# in practice, a plain English name. To keep the CLI beginner-friendly we let
# the user type "Spanish" and map it to the "es" code, but we also pass
# anything we do not recognise straight through, so "es" or "pt-BR" also work.
#
# NOTE: the map below is just a convenience list of common targets, NOT an
# authoritative enumeration of the supported set. If a code is not supported
# the server rejects it with an "error" event (we print it), so the server is
# the source of truth, not this dict.
# ----------------------------------------------------------------------------
LANGUAGE_CODES = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "dutch": "nl",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "hindi": "hi",
    "arabic": "ar",
    "polish": "pl",
}


def to_language_code(name: str) -> str:
    """Return a language code the model understands.

    "Spanish" -> "es"; an unknown value (like "es" or "pt-BR") is returned
    unchanged and lowercased, so power users can pass a raw code.
    """
    cleaned = name.strip().lower()
    return LANGUAGE_CODES.get(cleaned, cleaned)


# ----------------------------------------------------------------------------
# Command-line arguments.
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live voice translator using OpenAI gpt-realtime-translate."
    )
    parser.add_argument(
        "--to",
        dest="target",
        default=None,
        help='Target language, e.g. "Spanish" or a code like "es". '
        "If omitted, the program asks you.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print your audio input/output devices and exit (debugging).",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------
# A small container for everything the network callbacks need to share.
# Using a class keeps globals out of the way and makes the code readable.
# ----------------------------------------------------------------------------
class Translator:
    def __init__(self, target_language_code: str, api_key: str):
        self.target_language_code = target_language_code
        self.api_key = api_key

        # A thread-safe queue. The microphone thread PUTS raw audio bytes in;
        # the sender thread GETS them out and ships them over the socket.
        self.mic_queue: "queue.Queue[bytes]" = queue.Queue()

        # sounddevice objects we start once the socket is open.
        self.mic_stream: sd.RawInputStream | None = None
        self.speaker_stream: sd.RawOutputStream | None = None

        # A flag so every thread knows when to stop (set on Ctrl+C or error).
        self.running = True

        # Which text stream we printed last: "source", "target", or None.
        # The source transcript and the target (translated) transcript both
        # arrive as streaming deltas with no newline, so without a marker they
        # would run together on screen. We remember the last stream and print a
        # labeled header ONLY when it changes, keeping live text uncluttered.
        self._last_stream: str | None = None

    # ---- Pretty-print the two live transcript streams, clearly separated ----
    def _print_delta(self, stream: str, label: str, text: str) -> None:
        """Print a streaming transcript piece, tagging it when the stream flips.

        stream : a stable key, "source" or "target".
        label  : the human header to show when we switch to this stream.
        text   : the newest slice of text from the server's "delta" field.
        """
        if not text:
            return
        if stream != self._last_stream:
            # New stream: break the previous line and print a fresh label so the
            # source text and the translation never blur into one another.
            print(f"\n{label} ", end="", flush=True)
            self._last_stream = stream
        print(text, end="", flush=True)

    # ---- Microphone: called by sounddevice for each new block of audio ----
    def _on_mic_block(self, indata, frames, time_info, status):
        """sounddevice calls this on its OWN thread with fresh mic samples.

        `indata` is a raw bytes buffer of int16 samples. We copy it into our
        queue as quickly as possible and return; this callback must be fast.
        """
        if status:
            # Non-fatal warnings like input overflow. Print and keep going.
            print(f"[mic] {status}", file=sys.stderr)
        if self.running:
            # bytes(indata) copies the buffer so it stays valid after we return.
            self.mic_queue.put(bytes(indata))

    # ---- Sender thread: mic queue -> WebSocket ----
    def _sender_loop(self, ws: websocket.WebSocket):
        """Pull mic chunks off the queue and append them to the session.

        Runs on its own thread so reading the socket is never blocked by the
        microphone (and vice versa).
        """
        while self.running:
            try:
                # Wait up to 0.1s for audio so we can re-check self.running.
                chunk = self.mic_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Base64-encode the raw PCM16 bytes into ASCII text for JSON.
            b64_audio = base64.b64encode(chunk).decode("ascii")

            # CAUTION: on a TRANSLATION session the event name is prefixed
            # with "session." -> "session.input_audio_buffer.append".
            # (A normal Realtime session would send "input_audio_buffer.append".)
            # The audio bytes go in the "audio" field of THIS outgoing event.
            event = {
                "type": "session.input_audio_buffer.append",
                "audio": b64_audio,
            }
            try:
                ws.send(json.dumps(event))
            except Exception as exc:            # socket closed mid-send, etc.
                if self.running:
                    print(f"\n[sender] stopped: {exc}", file=sys.stderr)
                break

    # ---- Configure the session right after the socket opens ----
    def _configure_session(self, ws: websocket.WebSocket):
        """Tell the server our audio format and the TARGET language.

        We set the output language; the SOURCE language is auto-detected, so
        there is nothing to configure for the input side except its format.
        """
        session_update = {
            "type": "session.update",
            "session": {
                # GA nests audio format under session.audio.input/output.
                "audio": {
                    "input": {
                        # PCM16 @ 24 kHz, exactly what we capture from the mic.
                        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                        # THE knob for this whole module: the target language.
                        # Source is auto-detected, so we only set output.language.
                        "language": self.target_language_code,
                    },
                },
            },
        }
        ws.send(json.dumps(session_update))

    # ---- Receiver: called for every text frame the server sends ----
    def _on_message(self, ws: websocket.WebSocket, message: str):
        """Handle one server event (a JSON string).

        The three events we care about are all prefixed with "session." and
        carry their payload in the "delta" field (NOT "audio").
        """
        event = json.loads(message)
        etype = event.get("type", "")

        if etype == "session.input_transcript.delta":
            # Streaming text of what YOU said (the detected source language).
            # _print_delta prints it in place and labels it "YOU (source):" the
            # first time, so it never blurs into the translation stream below.
            self._print_delta("source", "YOU (source):", event.get("delta", ""))

        elif etype == "session.output_transcript.delta":
            # Streaming text of the TRANSLATION (the target language). Labeled
            # "TRANSLATION:" whenever we switch from the source stream to this one.
            self._print_delta("target", "TRANSLATION:", event.get("delta", ""))

        elif etype == "session.output_audio.delta":
            # CAUTION: the translated AUDIO bytes are in event["delta"],
            # NOT event["audio"]. This is the single most common mistake.
            audio_bytes = base64.b64decode(event["delta"])
            if self.speaker_stream is not None:
                self.speaker_stream.write(audio_bytes)   # play it immediately

        elif etype == "session.output_transcript.done":
            # One translated segment finished. Reset the stream tracker so the
            # next utterance re-prints its "YOU (source):" / "TRANSLATION:"
            # labels from scratch. We do not print a newline here because the
            # next label already begins with one (see _print_delta).
            self._last_stream = None

        elif etype == "error":
            # The server rejected something (bad language, bad format, ...).
            err = event.get("error", {})
            print(f"\n[server error] {err.get('message', event)}", file=sys.stderr)

        # Any other event types (session.created, session.updated, ...) are
        # ignored here to keep the on-screen output focused on the translation.

    def _on_error(self, ws: websocket.WebSocket, error):
        print(f"\n[websocket error] {error}", file=sys.stderr)

    def _on_close(self, ws: websocket.WebSocket, status_code, msg):
        self.running = False
        print("\n[closed] translation session ended.", file=sys.stderr)

    def _on_open(self, ws: websocket.WebSocket):
        """Runs once the WebSocket handshake succeeds.

        Order matters: configure the session, open the speakers so we can play
        replies, open the mic, then start the sender thread that drains the
        mic queue into the socket.
        """
        print("[open] connected. Configuring translation session...", file=sys.stderr)
        self._configure_session(ws)

        # Speakers: a raw OUTPUT stream we .write() decoded PCM16 bytes into.
        self.speaker_stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE
        )
        self.speaker_stream.start()

        # Microphone: a raw INPUT stream that calls _on_mic_block for each block.
        self.mic_stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=FRAMES_PER_CHUNK,     # ~50 ms of audio per callback
            callback=self._on_mic_block,
        )
        self.mic_stream.start()

        # Start streaming mic audio to OpenAI on a background thread.
        threading.Thread(target=self._sender_loop, args=(ws,), daemon=True).start()

        print("[ready] Speak now. Ctrl+C to quit.\n", file=sys.stderr)

    # ---- Public entry point ----
    def run(self):
        """Open the WebSocket and block until the connection closes."""
        # The Authorization header carries the real API key. At GA there is NO
        # "OpenAI-Beta: realtime=v1" header any more (see API_FACTS.md).
        headers = [f"Authorization: Bearer {self.api_key}"]

        ws = websocket.WebSocketApp(
            TRANSLATE_URL,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        # run_forever() blocks this (main) thread, reading frames and firing the
        # on_* callbacks above, until the socket closes or we KeyboardInterrupt.
        try:
            ws.run_forever()
        except KeyboardInterrupt:
            print("\n[quit] Ctrl+C received, shutting down...", file=sys.stderr)
        finally:
            # Always release the audio hardware, even on error.
            self.running = False
            if self.mic_stream is not None:
                self.mic_stream.stop()
                self.mic_stream.close()
            if self.speaker_stream is not None:
                self.speaker_stream.stop()
                self.speaker_stream.close()
            try:
                ws.close()
            except Exception:
                pass


# ----------------------------------------------------------------------------
# Program start.
# ----------------------------------------------------------------------------
def main():
    args = parse_args()

    # --list-devices: print the audio devices sounddevice can see, then exit.
    # Handy when the mic or speakers are not the ones you expect.
    if args.list_devices:
        print(sd.query_devices())
        return

    # Load the ONE shared .env at topics/voice_agents/.env. find_dotenv walks
    # UP the directory tree from here until it finds a .env file, so every
    # module shares the same key without copying it around.
    load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: OPENAI_API_KEY is not set.\n"
            "Copy topics/voice_agents/.env.example to .env and paste your key."
        )

    # Decide the target language: from --to, or ask interactively.
    target_name = args.target
    if not target_name:
        target_name = input("Translate your speech INTO which language? ").strip()
    if not target_name:
        sys.exit("ERROR: no target language given.")

    target_code = to_language_code(target_name)
    print(
        f"[config] Target language: {target_name} (code '{target_code}'). "
        "Source will be auto-detected.",
        file=sys.stderr,
    )

    Translator(target_code, api_key).run()


if __name__ == "__main__":
    main()
