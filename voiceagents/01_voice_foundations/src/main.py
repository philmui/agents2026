"""Voice Agents · Module 01 — What voice audio IS, on your own machine.

This script does five things, in order, and prints what happens at each step
so you can SEE the ideas from the tutorial with your own eyes:

    1. Record ~3 seconds from your microphone.
    2. Print the NumPy array's shape and dtype (its "size and kind").
    3. Print a few raw int16 sample values (the actual numbers on the wire).
    4. Count how many BYTES that audio is, and base64-encode ONE 50 ms chunk
       to show the exact text that later modules send to OpenAI.
    5. Play the audio back through your speakers.

There are NO OpenAI API calls here. This is the conceptual foundation for the
whole course: once you can capture audio in the right shape locally, sending it
to the Realtime API (module 02 onward) is just "put these bytes on a socket".

Run it:
    uv run python src/main.py            # record from the mic, then play back
    uv run python src/main.py --seconds 5
    uv run python src/main.py --tone     # no mic? synthesize a 440 Hz beep instead

Press Ctrl+C at any time to stop.
"""

# --- Standard-library imports (ship with Python, nothing to install) --------
import argparse   # parse command-line options like --seconds and --tone
import base64     # turn raw bytes into safe ASCII text (the "on the wire" form)
import sys        # write clean messages to the terminal / exit codes

# --- Third-party imports (installed by `uv sync` from pyproject.toml) -------
import numpy as np           # arrays of numbers: our audio lives in one
import sounddevice as sd     # record from the mic and play to the speakers

# We loaded python-dotenv in pyproject.toml to keep every module's setup
# identical, but module 01 never talks to OpenAI, so there is no key to read.
# Later modules will do: from dotenv import load_dotenv, find_dotenv.

# Our one source of truth for the audio format (see audio_config.py).
from audio_config import (
    SAMPLE_RATE,
    CHANNELS,
    SAMPLE_DTYPE,
    BYTES_PER_SAMPLE,
    CHUNK_MS,
    CHUNK_SAMPLES,
)


def record(seconds: float) -> np.ndarray:
    """Capture `seconds` of mono PCM16 audio from the default microphone.

    Returns a 1-D NumPy array of int16 samples (one number per sample).
    """
    # How many individual samples we need = seconds * samples-per-second.
    # int(...) drops any fraction so we ask for a whole number of samples.
    num_samples = int(seconds * SAMPLE_RATE)

    print(f"Recording {seconds:.1f}s at {SAMPLE_RATE} Hz... speak now.")

    # sd.rec() starts recording immediately and returns RIGHT AWAY (before the
    # recording is finished), so we must sd.wait() for it to complete.
    #   - num_samples: how many samples to grab
    #   - samplerate : measure 24000 times per second
    #   - channels=1 : mono (a single audio track)
    #   - dtype      : store each sample as a signed 16-bit integer (PCM16)
    recording = sd.rec(
        num_samples,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=SAMPLE_DTYPE,
    )
    sd.wait()  # block here until all `num_samples` have been captured

    # sd.rec() hands back a 2-D array shaped (num_samples, channels), e.g.
    # (72000, 1). Because we are mono, we flatten it to a plain 1-D list of
    # samples with .reshape(-1): "one long line of numbers".
    return recording.reshape(-1)


def synth_tone(seconds: float, hz: float = 440.0) -> np.ndarray:
    """Build a pure `hz` sine-wave beep as int16 samples (a mic-free fallback).

    Handy on machines with no microphone, or in CI. It produces audio in the
    SAME PCM16 / 24 kHz / mono shape as a real recording.
    """
    num_samples = int(seconds * SAMPLE_RATE)

    # t is the timestamp (in seconds) of each sample: 0, 1/24000, 2/24000, ...
    # np.linspace(0, seconds, num_samples, endpoint=False) makes that ramp.
    t = np.linspace(0.0, seconds, num_samples, endpoint=False)

    # A sine wave swings smoothly between -1.0 and +1.0. That is the shape of a
    # steady musical tone at `hz` cycles per second.
    wave = np.sin(2.0 * np.pi * hz * t)

    # PCM16 samples are integers from -32768..32767, so scale the -1..1 wave up.
    # We use 0.3 * 32767 (about a third of full volume) so the beep is audible
    # but not painfully loud, then cast to int16 to match the recording format.
    return (0.3 * 32767 * wave).astype(np.int16)


def describe(samples: np.ndarray) -> None:
    """Print the array's shape/dtype, a few raw values, and the byte count."""
    print("\n--- What did we actually capture? ---")

    # SHAPE = how many numbers, in how many dimensions. For 3s of mono audio at
    # 24 kHz this is (72000,): a flat line of 72,000 samples, one dimension.
    print(f"shape : {samples.shape}   (samples, )")

    # DTYPE = the kind of each number. int16 = signed 16-bit integer = PCM16.
    print(f"dtype : {samples.dtype}   (int16 == PCM16, range -32768..32767)")

    # RAW VALUES: audio really is just a list of integers. Near silence they
    # hover around 0; louder sound swings toward the +/-32767 extremes.
    first_ten = samples[:10]
    print(f"first 10 raw samples: {first_ten.tolist()}")
    print(f"loudest sample magnitude: {int(np.abs(samples).max())} (max possible 32767)")

    # BYTES: each int16 sample is exactly 2 bytes, so total bytes = N * 2.
    # samples.nbytes asks NumPy directly; we also show the hand calculation.
    total_bytes = samples.nbytes
    hand = len(samples) * BYTES_PER_SAMPLE
    print(
        f"bytes : {total_bytes:,}  "
        f"(= {len(samples):,} samples x {BYTES_PER_SAMPLE} bytes = {hand:,})"
    )

    # A quick reality check on bandwidth: 24000 samples/s * 2 bytes = 48000
    # bytes/s = about 46.9 KB per second of mono PCM16 audio. This is WHY
    # real-time voice needs a steady, low-latency connection, not one big
    # request: you are moving ~48 KB every second, forever, in both directions.
    print(f"data rate: {SAMPLE_RATE * BYTES_PER_SAMPLE:,} bytes/sec of raw audio")


def show_wire_format(samples: np.ndarray) -> None:
    """Base64-encode ONE ~50 ms chunk to reveal the exact text sent on the wire.

    Networks and JSON move text, not raw binary, so audio bytes are wrapped in
    base64: a scheme that rewrites any bytes using 64 safe ASCII characters
    (A-Z, a-z, 0-9, + and /). Later modules put this string in a JSON event's
    "audio" field. Here we just look at it.
    """
    print("\n--- What goes 'on the wire'? ---")

    # Slice out the first CHUNK_SAMPLES samples (~50 ms of audio) so the printed
    # string is short. Real streaming sends many such chunks back to back.
    chunk = samples[:CHUNK_SAMPLES]

    # .tobytes() gives us the raw binary: 1200 samples * 2 bytes = 2400 bytes.
    raw = chunk.tobytes()

    # base64.b64encode() -> bytes of ASCII; .decode("ascii") -> a normal string.
    b64 = base64.b64encode(raw).decode("ascii")

    print(f"one chunk = {CHUNK_MS} ms = {len(chunk)} samples = {len(raw):,} raw bytes")
    print(f"base64 length: {len(b64)} characters (base64 grows bytes by ~33%)")
    print("first 80 base64 chars (this is what later modules put in JSON):")
    print(f"  {b64[:80]}...")

    # PROOF it is loss-free: decode the base64 back to bytes, rebuild the int16
    # array, and confirm it equals the chunk we started with. base64 changes the
    # ENCODING, never the audio itself.
    decoded = np.frombuffer(base64.b64decode(b64), dtype=np.int16)
    matches = np.array_equal(decoded, chunk)
    print(f"decoded back to int16 and matches the original chunk? {matches}")


def playback(samples: np.ndarray) -> None:
    """Play the samples through the default speakers."""
    print("\nPlaying it back...")
    # sd.play() also returns immediately, so sd.wait() lets the sound finish.
    sd.play(samples, samplerate=SAMPLE_RATE)
    sd.wait()
    print("Done.")


def main() -> int:
    # --- Command-line options -------------------------------------------------
    parser = argparse.ArgumentParser(description="Record, inspect, and play back PCM16 audio.")
    parser.add_argument(
        "--seconds", type=float, default=3.0,
        help="How many seconds to record or synthesize (default: 3).",
    )
    parser.add_argument(
        "--tone", action="store_true",
        help="Skip the mic and synthesize a 440 Hz beep instead (no microphone needed).",
    )
    args = parser.parse_args()

    # --- Step 1: get some audio (from the mic, or a synthesized beep) ---------
    if args.tone:
        print("Synthesizing a 440 Hz tone (no microphone used).")
        samples = synth_tone(args.seconds)
    else:
        try:
            samples = record(args.seconds)
        except Exception as err:
            # Most failures here are "no input device" or a denied mic
            # permission. Guide the student to the --tone fallback instead of
            # dumping a raw traceback.
            print(f"\nCould not record from a microphone: {err}", file=sys.stderr)
            print("Tip: re-run with --tone to synthesize audio instead:", file=sys.stderr)
            print("     uv run python src/main.py --tone", file=sys.stderr)
            return 1

    # --- Steps 2-4: inspect the array, then the on-the-wire form --------------
    describe(samples)       # shape, dtype, raw values, byte count
    show_wire_format(samples)  # base64 one chunk, prove it round-trips

    # --- Step 5: hear it -------------------------------------------------------
    playback(samples)

    print("\nThat is all voice audio is: a stream of int16 numbers, 24000 per")
    print("second, wrapped in base64 to travel as text. Module 02 puts these")
    print("exact bytes onto a WebSocket to OpenAI's Realtime API.")
    return 0


if __name__ == "__main__":
    # Return code 0 = success, nonzero = something went wrong (mic, etc.).
    # We also catch Ctrl+C so quitting mid-recording looks clean, not scary.
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Bye.")
        raise SystemExit(130)
