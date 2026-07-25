# Module 01 · Voice Foundations

**What voice audio really is** (analog sound to sampling to PCM16 to 24 kHz to
mono to base64) and **how it travels** (HTTP vs WebSocket vs WebRTC vs PSTN).
This module has **no OpenAI API calls**. It is the conceptual bedrock the rest
of the course stands on.

You will record about 3 seconds from your microphone, print the raw sample
numbers, count the bytes, base64-encode one chunk to see the exact text that
later modules put on the wire, then play the audio back.

## Quickstart

```bash
# From this folder (topics/voice_agents/01_voice_foundations):
uv sync                                   # create .venv from pyproject.toml
uv run python src/main.py                 # record ~3s, inspect it, play it back
uv run python src/main.py --seconds 5     # record longer
uv run python src/main.py --tone          # no mic? synthesize a 440 Hz beep instead
```

No `.env` or API key is needed for this module. (The shared `../.env` matters
starting in module 02; we install `python-dotenv` here only to keep every
module's setup identical.)

## What you get

- `src/audio_config.py`: the ONE place the audio format is defined
  (PCM16, 24000 Hz, mono, 50 ms chunks). Every later module reuses these numbers.
- `src/main.py`: record, describe the NumPy array, show the on-the-wire base64,
  and play it back. Runs even with no microphone via `--tone`.
- `voice_foundations_tutorial.md`: the full explanation, one concept at a time.
- `slides/index.html`: the reveal.js deck for this module.

## Troubleshooting

- **No microphone / permission denied:** run with `--tone`. On macOS, the first
  run may prompt for microphone access; grant it, then re-run.
- **`uv: command not found`:** install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **PortAudio errors on Linux:** install the system library, e.g.
  `sudo apt-get install libportaudio2`, then `uv sync` again.

---

Built by **mui-group** for advanced high-school students.
