# OpenAI Realtime API — Verified Ground Truth (single source of truth)

> **Every module author MUST read this file and match it exactly.** These facts were
> cross-checked on 2026-07-25 against OpenAI's live developer docs
> (`developers.openai.com/api/docs/guides/realtime`, `.../realtime-webrtc`,
> `.../realtime-websocket`, `.../realtime-conversations`, `.../voice-agents`) and the
> DataCamp tutorial `datacamp.com/tutorial/gpt-realtime-2-api`.
>
> Where the DataCamp tutorial and OpenAI's live docs disagree, **OpenAI's live docs win**
> and the discrepancy is called out below so students learn to read primary sources.

---

## 1. Model IDs (quote these verbatim in every module)

| Capability | Canonical model ID | Notes |
|---|---|---|
| Speech-to-speech voice assistant | `gpt-realtime-2.1` | OpenAI GA docs + official JS SDK use `-2.1`. **DataCamp says `gpt-realtime-2`** (older). Teach `-2.1` as canonical; mention `-2` as the earlier name. |
| Live translation | `gpt-realtime-translate` | Dedicated translation endpoint. 70+ input languages, 13 output languages. |
| Transcription only | `gpt-realtime-whisper` | Realtime, billed by audio minute. NOT the same as file-based `whisper-1`. |
| Text agent companion (chained pipelines or hosted-search delegate) | `gpt-5.6` | Used behind speech-to-speech when a Realtime function tool delegates text work to the Responses API. |

Do **not** invent `gpt-realtime-mini` or `gpt-4o-realtime-preview` — they are not part of this course.

## 2. Transports & endpoints

- **WebSocket (server-side)**: `wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1`
- **WebRTC (browser)**: POST the SDP offer to `https://api.openai.com/v1/realtime/calls`
- **Translation session**: `wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate`
- **Transcription session**: set `session.type: "transcription"` (or connect with `?intent=transcription`)
- **Ephemeral tokens (mint on YOUR backend)**: `POST https://api.openai.com/v1/realtime/client_secrets`

At **GA, the `OpenAI-Beta: realtime=v1` header is GONE** — do not include it. Optionally send
`OpenAI-Safety-Identifier: <hashed-user-id>` for per-user abuse tracking.

## 3. Audio format (GA nesting — this changed from the beta!)

PCM16, **24 kHz**, mono, base64-encoded. At GA, format/turn-detection are **nested** under
`session.audio.input` and `session.audio.output` (the old flat `input_audio_format: "pcm16"` is legacy):

```jsonc
{
  "type": "session.update",
  "session": {
    "type": "realtime",
    "audio": {
      "input":  {
        "format": { "type": "audio/pcm", "rate": 24000 },
        "turn_detection": { "type": "semantic_vad" }   // or "server_vad", or null for manual
      },
      "output": {
        "format": { "type": "audio/pcm", "rate": 24000 },
        "voice": "marin"                                // chosen ONCE, cannot switch mid-session
      }
    }
  }
}
```
`audio/pcmu` (G.711 µ-law) exists for telephony/PSTN; default to `audio/pcm` for web.

**Translation exception:** the dedicated `/v1/realtime/translations` endpoint
already fixes WebSocket audio to 24 kHz PCM16 and exposes a narrower update
schema. Its `session.update` should set only
`session.audio.output.language`. Do not send `session.audio.input.format`,
`session.audio.output.format`, or input transcription settings; those standard
Realtime fields are rejected by translation sessions.

## 4. Event catalog (exact strings — the gotchas are real)

**Client → server**
- `session.update` — configure the session (audio, instructions, tools, modalities)
- `input_audio_buffer.append` — mic audio; base64 bytes go in the **`audio`** field. Max 15 MB/chunk; ~50 ms chunks recommended.
- `input_audio_buffer.commit` — finalize a manual turn (only when `turn_detection: null`)
- `response.create` — ask the model to respond (auto-fired by VAD unless disabled)
- `response.cancel` — interrupt a response in progress
- `conversation.item.truncate` — (WebSocket) drop unplayed assistant audio; WebRTC uses `output_audio_buffer.clear`

**Server → client**
- `response.output_audio.delta` — assistant audio; base64 in the **`delta`** field.
  ⚠️ It is **`response.output_audio.delta`, NOT `response.audio.delta`** (a common wrong guess).
- `response.output_audio_transcript.delta` / `.done` — text of what the assistant is saying
- `response.output_text.delta` — text output (text modality)
- `response.done` — turn complete (carries transcripts, **not** audio bytes)
- `input_audio_buffer.speech_started` / `input_audio_buffer.speech_stopped` — VAD boundaries
- `conversation.item.input_audio_transcription.completed` — transcription of the **user's** speech

**Translation session events (note the `session.` prefix and the `delta` field):**
- send `session.input_audio_buffer.append`
- receive `session.output_audio.delta` (translated audio; bytes in **`event["delta"]`**, NOT `event["audio"]`)
- receive `session.output_transcript.delta` (target)
- target language set via `session.audio.output.language`; source auto-detected. No `response.create` loop.

**Source-caption compatibility note (live-verified 2026-07-25):** OpenAI's
translation guide documents `session.input_transcript.delta`, but the live
translation WebSocket did not emit it during end-to-end verification. Apps that
must display the speaker's original words should feed the same PCM16 chunks to a
parallel transcription connection at `?intent=transcription`, configured with
`gpt-realtime-whisper`. That model does not support VAD, so commit detected phrases
manually with `input_audio_buffer.commit`, then consume
`conversation.item.input_audio_transcription.delta` and `.completed`.

## 5. Browser WebRTC recipe (verified)

**Backend (Node/Next route or Python) mints the token:**
```js
// POST https://api.openai.com/v1/realtime/client_secrets
// headers: Authorization: Bearer <OPENAI_API_KEY>, Content-Type: application/json
// body:
{ "session": { "type": "realtime", "model": "gpt-realtime-2.1",
               "audio": { "output": { "voice": "marin" } } } }
// response: the ephemeral key is at data.value  ->  starts with "ek_..."
```

**Browser connects with the `ek_` key (never the real API key):**
```js
const pc = new RTCPeerConnection();
pc.ontrack = (e) => (audioEl.srcObject = e.streams[0]);          // hear the assistant
const ms = await navigator.mediaDevices.getUserMedia({ audio: true });
pc.addTrack(ms.getTracks()[0]);                                   // send the mic
const dc = pc.createDataChannel("oai-events");                    // JSON events channel
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);
const sdpRes = await fetch("https://api.openai.com/v1/realtime/calls", {
  method: "POST", body: offer.sdp,
  headers: { Authorization: `Bearer ${EPHEMERAL_KEY}`, "Content-Type": "application/sdp" },
});
await pc.setRemoteDescription({ type: "answer", sdp: await sdpRes.text() });
```

## 6. Official SDKs

- **Browser/TS**: `@openai/agents/realtime` → `RealtimeAgent` + `RealtimeSession`.
  `new RealtimeSession(agent, { model: "gpt-realtime-2.1" })`, then `session.connect({ apiKey: "ek_..." })`.
  Attach `tools`, `handoffs`, `guardrails` to the `RealtimeAgent` (same as a text agent).
  Realtime voice supports function tools and hosted MCP tools. For public web search,
  expose a strict `web_search` function tool that calls your backend; the backend
  sends `{"model":"gpt-5.6","tools":[{"type":"web_search"}],"tool_choice":"required"}`
  to `POST https://api.openai.com/v1/responses`. Do not expose the permanent API
  key in the browser or attach the Responses-only hosted tool directly to the
  Realtime transport.
- **Python (server / this course's transcription+translation modules)**: raw WebSocket via
  `websocket-client` (`import websocket`) or `websockets` (async). DataCamp uses `websocket-client` +
  `sounddevice` + `numpy`. Either is fine; be consistent within a module.

## 7. Limits & operational facts

- Sessions end after ~60 min (OpenAI) / 30 min (Azure). Reconnect around 55 min.
- Tier-1 rate limits ~200 req/min, 40k tokens/min; the free tier cannot use Realtime.
- `reasoning.effort: "low"` is recommended for most production voice agents (lower latency).
- Turn detection: `semantic_vad` (has `eagerness`), `server_vad`, or `null` (you commit manually).
- Voice must be selected before the first audio output and cannot change mid-session.
- Context window 128K.

## 8. Pedagogy reminders (audience = advanced high-school students, basic Python)

- Explain **WebSocket vs WebRTC vs PSTN** and **what "voice audio" is** (analog → sampling →
  PCM16 → 24 kHz → base64) plainly, with analogies. This is a required teaching point.
- Every code line should be explainable. Prefer clarity over cleverness.
- No em-dashes in prose (author house style). Use commas or parentheses.
