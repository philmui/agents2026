# Module 03: Live Transcription (`gpt-realtime-whisper`)

Capability #1 of the Voice Agents course: turn your **speech into text, live**, as you talk.
A command-line program opens a Realtime **transcription session**, streams your microphone
to OpenAI in tiny PCM16 chunks, and prints your words the moment they are recognized.

Full walkthrough: [`transcription_tutorial.md`](./transcription_tutorial.md) · Slides: [`slides/index.html`](./slides/index.html)

## Quickstart

```bash
cp ../.env.example ../.env      # 1) once: paste your OpenAI key into the shared ../.env
uv sync                        # 2) create .venv and install deps from pyproject.toml
uv run python src/live_transcribe.py   # 3) talk; your transcript prints live. Ctrl-C to stop.
```

Bonus: manual turn-taking (`turn_detection: null`), press ENTER to finish each phrase:

```bash
uv run python src/manual_commit.py
```

## What's here

| File | What it is |
|---|---|
| `src/live_transcribe.py` | Main CLI. Mic to `input_audio_buffer.append`; prints `...transcription.completed`. Uses **server_vad**. |
| `src/manual_commit.py` | Same, but **manual** turns: you send `input_audio_buffer.commit` on ENTER. |
| `transcription_tutorial.md` | Concept-by-concept explanation of every line and every API gotcha. |
| `slides/index.html` | reveal.js deck for teaching the module. |

## Requirements

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).
- A working microphone (grant mic permission on first run).
- `sounddevice` needs PortAudio: macOS has it; on Debian/Ubuntu run
  `sudo apt-get install libportaudio2`.
- An OpenAI key on a **paid tier** (the free tier cannot use Realtime).

## Gotchas (see the tutorial for details)

- On the way **in**, mic bytes go in the **`audio`** field of `input_audio_buffer.append`
  (not `delta`). `delta` is only for streaming events on the way **out**.
- This is realtime transcription **billed by the audio minute**. It is **not** the
  file-upload `whisper-1` endpoint.
- At GA there is **no** `OpenAI-Beta: realtime=v1` header. Do not add it.

---

Part of the **Voice Agents** minicourse. Built by **mui-group**.
