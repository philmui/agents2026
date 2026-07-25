# Module 05: Terminal Voice Assistant (`gpt-realtime-2.1`)

Capability #3 of the Voice Agents course, server-side first: a **full-duplex** command-line
voice assistant. Your mic streams **up** to OpenAI while the assistant's voice streams **down**
to your speakers, both at once, so you can **interrupt it mid-sentence** (barge-in). This is the
speech-to-speech model `gpt-realtime-2.1` with **semantic VAD** deciding when your turn ends.

Full walkthrough: [`voice_assistant_cli_tutorial.md`](./voice_assistant_cli_tutorial.md) · Slides: [`slides/index.html`](./slides/index.html)

## Quickstart

```bash
cp ../.env.example ../.env      # 1) once: paste your OpenAI key into the shared ../.env
uv sync                        # 2) create .venv and install deps from pyproject.toml
uv run python src/voice_assistant.py   # 3) just talk. It answers out loud. Ctrl-C to stop.
```

Bonus: **manual turns** (push-to-talk): `turn_detection: null`, press ENTER to send each phrase:

```bash
uv run python src/push_to_talk.py
```

## What's here

| File | What it is |
|---|---|
| `src/voice_assistant.py` | Main CLI. Full-duplex speech-to-speech with **semantic VAD** + **barge-in**. |
| `src/push_to_talk.py` | Same assistant, but **manual** turns: ENTER sends `commit` + `response.create`. |
| `voice_assistant_cli_tutorial.md` | Concept-by-concept explanation of every line and every API gotcha. |
| `slides/index.html` | reveal.js deck for teaching the module. |

## Requirements

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).
- A working **microphone and speakers** (grant mic permission on first run).
- **Headphones recommended.** Without them, your speakers leak into your mic and the assistant
  hears itself and interrupts itself. Headphones give clean barge-in.
- `sounddevice` needs PortAudio: macOS has it; on Debian/Ubuntu run
  `sudo apt-get install libportaudio2`.
- An OpenAI key on a **paid tier** (the free tier cannot use Realtime).

## Gotchas (see the tutorial for details)

- The assistant's audio arrives in **`response.output_audio.delta`** (bytes in the `delta`
  field), **NOT** `response.audio.delta`. Guessing `response.audio.delta` is the #1 mistake here
  and you will just hear silence.
- The **voice is chosen once** (`marin`) in `session.update` and cannot change mid-session.
- Audio is **PCM16 @ 24 kHz mono**, nested under `session.audio.input` / `session.audio.output`
  as `{"type": "audio/pcm", "rate": 24000}`. The old flat `input_audio_format` is legacy.
- With **semantic VAD**, the server auto-fires `response.create` when your turn ends. You only
  send `response.create` yourself in **manual** mode (`push_to_talk.py`).
- **Barge-in** over a WebSocket uses `conversation.item.truncate`; browser WebRTC would use
  `output_audio_buffer.clear` instead.
- At GA there is **no** `OpenAI-Beta: realtime=v1` header. Do not add it.

---

Part of the **Voice Agents** minicourse. Built by **mui-group**.
