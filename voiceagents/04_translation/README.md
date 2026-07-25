# Module 04: Live Translation (`gpt-realtime-translate`)

Speak into your mic and hear (and read) your words in another language, in real
time. This is capability #2 of the Voice Agents minicourse. Full walkthrough:
[`translation_tutorial.md`](./translation_tutorial.md). Slides: [`slides/index.html`](./slides/index.html).

## Quickstart

```bash
# 0) One time, from topics/voice_agents/: cp .env.example .env  and paste your key.
cd topics/voice_agents/04_translation
uv sync                                   # create .venv from pyproject.toml
uv run python src/translate.py --to Spanish   # speak; hear Spanish back
```

- No `--to`? The program asks you which language to translate INTO.
- Use a name (`Spanish`, `Japanese`) or a code (`es`, `ja`). Source is auto-detected.
- Audio wrong device? `uv run python src/translate.py --list-devices`.
- Stop with `Ctrl+C`.

## What it does

Connects to `wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate`,
sets the **target** language via `session.audio.output.language`, streams your mic
as `session.input_audio_buffer.append`, and handles the `session.`-prefixed replies:
`session.input_transcript.delta` (source text), `session.output_transcript.delta`
(target text), and `session.output_audio.delta` (translated audio, played back).

> **Heads-up (vs Module 03):** every event here is prefixed with `session.`, the
> translated audio bytes arrive in `event["delta"]` (not `event["audio"]`), and there
> is **no** `response.create` loop. See the tutorial's Caution boxes.

Requires a paid OpenAI tier (the free tier cannot use Realtime) and a working mic + speakers.
