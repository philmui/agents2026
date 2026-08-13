"""Shared audio constants for the whole Voice Agents course.

Every later module (transcription, translation, the voice assistant) sends
audio to OpenAI's Realtime API in EXACTLY one format. We define that format
here, once, so the numbers never drift between modules.

The format (verified in docs/API_FACTS.md):
    PCM16, 24000 Hz, mono, base64-encoded on the wire.

Read the tutorial (voice_foundations_tutorial.md) for what each word means.
This module (01) does NOT call OpenAI; it just proves you can capture audio
in this exact shape on your own machine first.
"""

# How many audio samples we capture per second. "24000 Hz" means the
# microphone's continuous sound wave is measured 24,000 times every second.
# OpenAI's Realtime API expects 24 kHz, so we use it everywhere.
SAMPLE_RATE = 24000

# One audio channel = "mono" (a single microphone's-eye view of the room).
# Two channels would be "stereo" (left + right). Realtime audio is mono.
CHANNELS = 1

# Each sample is stored as a signed 16-bit integer ("PCM16"): a whole number
# from -32768 to +32767. "int16" is NumPy's name for that type. 16 bits = 2
# bytes, so every single sample costs exactly 2 bytes on disk or on the wire.
SAMPLE_DTYPE = "int16"
BYTES_PER_SAMPLE = 2  # 16 bits / 8 bits-per-byte = 2 bytes

# The Realtime API likes small audio "chunks" of roughly 50 milliseconds.
# 50 ms is 1/20th of a second, so at 24000 samples/second that is:
#     24000 samples/sec * 0.050 sec = 1200 samples per chunk.
# We reuse this later when we stream audio; module 01 just demonstrates it.
CHUNK_MS = 50
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # -> 1200
