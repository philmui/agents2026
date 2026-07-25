export const meta = {
  name: 'voice-agents-authoring',
  description: 'Author + adversarially verify + fix the 8-module voice_agents minicourse',
  phases: [
    { title: 'Author', detail: 'one subagent per module writes pyproject, tutorial MD, slides, code' },
    { title: 'Verify', detail: 'adversarial check of API facts + slide-style rules per module' },
    { title: 'Fix', detail: 'apply corrections to modules with findings' },
  ],
}

const ROOT = '/Users/pmui/dev/agent-tutorials/topics/voice_agents'
const REF = '/Users/pmui/dev/agent-tutorials/topics/07_advanced_retrievers'

// Shared preamble every author + verifier reads first.
const COMMON = `
You are authoring one module of an 8-module minicourse "Voice Agents" at ${ROOT}.
Audience: advanced high-school students with BASIC Python. Explain every concept and every line.

BEFORE writing anything, READ these three files in full and obey them:
  1. ${ROOT}/_shared/API_FACTS.md        (verified OpenAI Realtime API ground truth — the single source of truth)
  2. ${ROOT}/_shared/COURSE_DESIGN.md    (the course blueprint, per-module deliverables, house style, mermaid + slide rules)
  3. ${REF}/slides/index.html            (reference reveal.js deck — COPY its structure + the auto-fit <script> at the bottom verbatim; then adapt palette/font per COURSE_DESIGN)

Also skim ${REF}/pyproject.toml for the pyproject house style (comments about uv sync / uv.lock, [tool.uv] package=false, >= lower bounds).

NON-NEGOTIABLE FACTS (from API_FACTS.md — do not deviate):
- Voice assistant model = "gpt-realtime-2.1" (note "gpt-realtime-2" as the older DataCamp name).
- Transcription model = "gpt-realtime-whisper"; Translation model = "gpt-realtime-translate".
- Assistant audio event is response.output_audio.delta (base64 in "delta"), NOT response.audio.delta.
- Audio is PCM16 @ 24000 Hz mono base64; GA nests format under session.audio.input/output as { "type":"audio/pcm","rate":24000 }.
- WebSocket: wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1 ; NO OpenAI-Beta header at GA.
- Browser WebRTC: mint ek_ token via POST https://api.openai.com/v1/realtime/client_secrets (key at data.value),
  then POST SDP to https://api.openai.com/v1/realtime/calls with Content-Type: application/sdp; datachannel "oai-events".
- Secrets: ONE shared .env at ${ROOT}/.env, loaded in Python via load_dotenv(find_dotenv()). Browser never sees the real key.

HOUSE STYLE:
- Tutorial markdown: open with the ONE idea + a concept-map table, then a section per concept with code snippets and a
  "Caution" callout at each API gotcha. Use mermaid for workflows (flowchart LR/TD or sequenceDiagram; <=7 nodes; short labels;
  break long labels with <br/>; no crossing edges; use sequenceDiagram for client<->server event exchanges). No em-dashes in prose.
- Slides slides/index.html: single-file reveal.js 5.1.0 (CDN white theme + atom-one-light). Font Inter (Google Fonts) + system fallback.
  Pastel palette on white per COURSE_DESIGN tokens. Left-aligned, top-pinned flex sections. Copy the auto-fit shrink <script>.
  Kicker eyebrow "Voice Agents · Module NN". 7x7 rule (<=7 lines/slide, <=7 words/line): favor diagrams, code, SVG over prose.
  NEVER put an underline/accent line beneath a title (use whitespace). Footer attribution EXACTLY: © mui-group.
- Every module folder gets: pyproject.toml ([tool.uv] package=false, heavy comments), README.md (uv sync + run quickstart),
  <name>_tutorial.md, slides/index.html, runnable code (src/ for Python, app/ for Next.js), and a .gitignore.
- Code MUST be runnable and correct against API_FACTS.md. Prefer clarity over cleverness. Comment generously for beginners.

Write real files with the Write tool. Do NOT ask questions; make sensible choices and proceed.
`

const MODULES = [
  { n: '01', folder: '01_voice_foundations', slug: 'voice_foundations',
    brief: `Teach what voice audio IS (analog sound -> sampling -> PCM16 -> 24kHz -> mono -> base64) and the transport landscape:
      plain HTTP/request-response vs WebSocket (persistent server duplex) vs WebRTC (browser realtime media + NAT traversal) vs
      PSTN (the old phone network, G.711/µ-law, why telephony matters). Explain "real-time" and latency budgets for natural speech.
      NO OpenAI API calls in this module. Runnable code (src/): record ~3s from the mic with sounddevice, print the numpy sample
      array shape/dtype and a few raw int16 values, show how many bytes that is, base64-encode one chunk to show what goes on the
      wire, then play the audio back. Include a mermaid audio-pipeline diagram and a mermaid/table transport comparison.
      pyproject deps: sounddevice, numpy, python-dotenv (dotenv unused here but keeps the pattern consistent). This module is the
      conceptual foundation the whole course references, so the tutorial + slides must be especially clear and visual.` },

  { n: '02', folder: '02_realtime_handshake', slug: 'realtime_handshake',
    brief: `Teach the Realtime API mental model: a SESSION is a long-lived WebSocket; you exchange JSON EVENTS on it; there is an
      event loop (you send client events, you receive server events). NO audio yet. Runnable code (src/): connect to
      wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1 with Authorization: Bearer <key> (from shared .env via
      load_dotenv(find_dotenv())), on open send a session.update with instructions, then print every server event type and a short
      summary as it arrives (session.created, session.updated, etc.). Show BOTH websocket-client (WebSocketApp with on_open/
      on_message) since the rest of the Python modules use it. Explain there is NO OpenAI-Beta header at GA. Include a mermaid
      sequenceDiagram of the handshake (client connect -> session.created -> session.update -> session.updated). pyproject deps:
      websocket-client, python-dotenv. Emphasize reading server events is how you debug everything later.` },

  { n: '03', folder: '03_transcription', slug: 'transcription',
    brief: `Capability #1 — live transcription with gpt-realtime-whisper. Runnable Python CLI (src/): open a transcription session
      (session.type:"transcription"), capture mic with sounddevice in ~50ms PCM16 24kHz chunks, base64-encode and send each as
      input_audio_buffer.append (bytes in the "audio" field), and print the user's transcript from
      conversation.item.input_audio_transcription.completed (and show the .delta streaming variant if used). Explain server_vad vs
      semantic_vad vs manual commit. Caution callouts: append field is "audio", transcription billed by audio minute, this is NOT
      file-based whisper-1. Include a mermaid sequenceDiagram (mic -> append -> speech_started/stopped -> transcription.completed).
      pyproject deps: websocket-client, sounddevice, numpy, python-dotenv.` },

  { n: '04', folder: '04_translation', slug: 'translation',
    brief: `Capability #2 — live translation with gpt-realtime-translate. Runnable Python CLI (src/): connect to
      wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate, set target via session.audio.output.language
      (source auto-detected), stream mic audio as session.input_audio_buffer.append, and handle the session.-PREFIXED events:
      session.output_audio.delta (translated audio bytes in event["delta"] — play it back), session.input_transcript.delta
      (source text) and session.output_transcript.delta (target text). Strong caution callouts contrasting with module 03: the
      "session." prefix, delta bytes in event["delta"] NOT event["audio"], no response.create loop. Support choosing the output
      language via a CLI arg or prompt. Include a mermaid sequenceDiagram. pyproject deps: websocket-client, sounddevice, numpy,
      python-dotenv.` },

  { n: '05', folder: '05_voice_assistant_cli', slug: 'voice_assistant_cli',
    brief: `Capability #3, server-side first — a full-duplex terminal voice assistant with gpt-realtime-2.1. Runnable Python CLI
      (src/): connect over WebSocket, session.update with audio.input (pcm16@24k, turn_detection semantic_vad) and audio.output
      (voice "marin"), instructions, and reasoning.effort "low". Stream mic via input_audio_buffer.append; play assistant audio from
      response.output_audio.delta (base64 in "delta"); print assistant transcript from response.output_audio_transcript.delta.
      Demonstrate barge-in/interruption (when the user starts talking, stop playback; mention conversation.item.truncate). Explain
      VAD-driven auto response.create vs manual. Caution callouts: response.output_audio.delta (NOT response.audio.delta), voice
      chosen once. Include a mermaid sequenceDiagram of a full turn with barge-in. pyproject deps: websocket-client, sounddevice,
      numpy, python-dotenv. This isolates assistant logic before the browser.` },

  { n: '06', folder: '06_python_backend', slug: 'python_backend',
    brief: `The web app's backend — a FastAPI service whose job is to safely mint EPHEMERAL browser tokens. Runnable code (app/ or
      src/): a FastAPI app with GET /health and POST (or GET) /token that server-side calls
      POST https://api.openai.com/v1/realtime/client_secrets with the real OPENAI_API_KEY (from shared .env via
      load_dotenv(find_dotenv())) and body { session: { type:"realtime", model:"gpt-realtime-2.1", audio:{ output:{ voice:"marin" }}}},
      returning the ephemeral key from data.value (the "ek_..." token) to the browser. Add permissive CORS for localhost dev. Explain
      CLEARLY why the browser must never receive the real key and what an ephemeral token is (short-lived, scoped). Provide a curl
      test in the README. Use httpx (async) for the outbound call. Include a mermaid sequenceDiagram (browser -> /token -> OpenAI
      client_secrets -> ek_ -> browser). pyproject deps: fastapi, uvicorn[standard], httpx, python-dotenv. This is the bridge to the
      frontend (module 07 fetches from here).` },

  { n: '07', folder: '07_nextjs_frontend', slug: 'nextjs_frontend',
    brief: `The NextJS + React frontend that talks to gpt-realtime-2.1 in the BROWSER over WebRTC. Teach the OFFICIAL SDK FIRST, then
      the raw internals. Build a Next.js (App Router, TypeScript) app in app/ with a clean, minimal UI: a big "Talk" button, a live
      transcript panel, and a connection status pill. PRIMARY path: use @openai/agents/realtime (RealtimeAgent + RealtimeSession),
      fetching the ephemeral token from the module-06 backend (NEXT_PUBLIC_TOKEN_ENDPOINT), then session.connect({ apiKey: ek }).
      SECOND path (a "How it really works" section in the tutorial + a commented lib/rawWebrtc.ts): raw RTCPeerConnection,
      getUserMedia({audio:true}), pc.addTrack, pc.createDataChannel("oai-events"), createOffer/setLocalDescription, POST offer.sdp to
      https://api.openai.com/v1/realtime/calls with Authorization: Bearer ek + Content-Type: application/sdp, setRemoteDescription
      with the answer. Explain WHY browsers use WebRTC not WebSocket (media transport, echo cancellation, NAT). The frontend NEVER
      holds the real API key. Provide package.json (next, react, react-dom, @openai/agents). NOTE: there is no pyproject here (it is a
      Node project) — but STILL include README (npm install / npm run dev, and that it needs module 06 running) and a .gitignore
      (node_modules, .next). Include a mermaid sequenceDiagram of the full browser handshake. Also write the tutorial .md and
      slides/index.html per house style. Keep UI styling minimalist/pastel to match the slide aesthetic.` },

  { n: '08', folder: '08_capstone_multimode', slug: 'capstone_multimode',
    brief: `Capstone — ONE web app unifying all three modes plus a tool-calling assistant. Build on modules 06+07. In app/: a Next.js
      UI with a mode switch (Transcribe | Translate | Assist). Assist mode uses RealtimeAgent/RealtimeSession (gpt-realtime-2.1) over
      WebRTC and demonstrates ReAct-style TOOLS: define get_time plus a strict web_search function tool on the RealtimeAgent. The
      browser tool must call the backend, which securely runs Responses hosted web_search with the permanent key. Show the model
      calling each tool and speaking the result — explain the reason->act->observe->respond loop plainly. Transcribe/Translate modes
      can either reuse the browser WebRTC data or clearly document calling the module-03/04 Python tools; pick the simplest correct
      story and state it explicitly (do not hand-wave). Include deploy notes (env vars, that the token backend must run, session
      length limits from API_FACTS). This is a Node/Next project: package.json (next, react, react-dom, @openai/agents), README,
      .gitignore, tutorial .md, slides/index.html. Include a mermaid architecture diagram tying frontend + backend + OpenAI together
      and a sequenceDiagram for a tool call. Reference earlier modules by number. End the course with a "where to go next" slide.` },
]

phase('Author')
const results = await pipeline(
  MODULES,
  // Stage 1: author the module
  (m) => agent(
    `${COMMON}\n\n=== YOUR MODULE: ${m.n} (${m.folder}) ===\nSlug for filenames: ${m.slug} (so the tutorial is ${m.slug}_tutorial.md).\nModule folder: ${ROOT}/${m.folder}\n\nWHAT TO TEACH / BUILD:\n${m.brief}\n\nProduce ALL deliverables listed in the house style for this module now. Make the code runnable and correct against API_FACTS.md. When done, reply with a one-paragraph summary of what you created and the exact file paths.`,
    { label: `author:${m.n}`, phase: 'Author' }
  ).then((summary) => ({ ...m, summary })),

  // Stage 2: adversarially verify THIS module against ground truth + house style
  (authored) => agent(
    `${COMMON}\n\n=== VERIFY MODULE ${authored.n} (${authored.folder}) ===\nYou are an adversarial reviewer. The module was just authored. Your job is to FIND problems, not to praise.\nRead every file under ${ROOT}/${authored.folder} and check against ${ROOT}/_shared/API_FACTS.md and COURSE_DESIGN.md.\n\nCHECK RUTHLESSLY:\n1) MODEL IDS: any occurrence of gpt-realtime-2 WITHOUT the .1 (except when explicitly labeled the older name), or any wrong/invented model id. gpt-realtime-2.1 / gpt-realtime-translate / gpt-realtime-whisper must be exact.\n2) EVENT NAMES: response.output_audio.delta (never response.audio.delta); input_audio_buffer.append with bytes in "audio"; translation events carry the session. prefix with bytes in event["delta"]; conversation.item.input_audio_transcription.completed for user transcript.\n3) ENDPOINTS: /v1/realtime, /v1/realtime/translations, /v1/realtime/calls, /v1/realtime/client_secrets exactly; ephemeral key read from data.value; NO OpenAI-Beta header.\n4) AUDIO: pcm16 @ 24000 mono; standard Realtime/transcription sessions use GA nesting under session.audio.input/output with { "type":"audio/pcm","rate":24000 }. Translation is the exception: session.update sets only session.audio.output.language and MUST NOT include input/output format or input transcription fields.\n5) SECRETS: Python uses load_dotenv(find_dotenv()); the shared parent .env story is correct; the browser never sees the real key.\n6) SLIDES: single-file reveal.js 5.1.0; Inter font; pastel-on-white; left-aligned; NO accent line under any title; footer EXACTLY "© mui-group"; kicker "Voice Agents · Module ${authored.n}"; roughly obeys 7x7; the auto-fit script is present.\n7) MERMAID: diagrams present for workflows; <=7 nodes; short labels; likely to render without overlap; sequenceDiagram used for event exchanges.\n8) FILES: pyproject.toml (package=false) for Python modules OR package.json for Node modules (07,08); README with runnable quickstart; .gitignore; tutorial .md; slides/index.html. Code is plausibly runnable and beginner-clear. No em-dashes in prose.\n\nReturn STRICT JSON only: { "module":"${authored.n}", "pass": <true|false>, "findings": [ { "file":"relative/path", "severity":"high|medium|low", "issue":"...", "fix":"concrete correction" } ] }. pass=true ONLY if there are no high or medium findings.`,
    { label: `verify:${authored.n}`, phase: 'Verify', schema: {
      type: 'object', additionalProperties: false,
      required: ['module', 'pass', 'findings'],
      properties: {
        module: { type: 'string' },
        pass: { type: 'boolean' },
        findings: { type: 'array', items: {
          type: 'object', additionalProperties: false,
          required: ['file', 'severity', 'issue', 'fix'],
          properties: {
            file: { type: 'string' },
            severity: { enum: ['high', 'medium', 'low'] },
            issue: { type: 'string' },
            fix: { type: 'string' },
          },
        } },
      },
    } }
  ).then((verdict) => ({ ...authored, verdict }))
)

// Barrier: decide which modules need fixing (need ALL verdicts to summarize + drive fixes).
const needFix = results.filter(Boolean).filter((r) => r.verdict && r.verdict.pass === false)
log(`Authored ${results.filter(Boolean).length}/8 modules. ${needFix.length} need fixes: ${needFix.map((r) => r.n).join(', ') || 'none'}`)

phase('Fix')
const fixed = await parallel(needFix.map((r) => () =>
  agent(
    `${COMMON}\n\n=== FIX MODULE ${r.n} (${r.folder}) ===\nAn adversarial review found problems. Apply EVERY fix below by editing the files under ${ROOT}/${r.folder}. Re-read API_FACTS.md if unsure. Do not introduce new deviations. Preserve the teaching quality.\n\nFINDINGS (JSON):\n${JSON.stringify(r.verdict.findings, null, 2)}\n\nAfter fixing, reply with the list of files you changed and a one-line confirmation that each high/medium finding is resolved.`,
    { label: `fix:${r.n}`, phase: 'Fix' }
  ).then((summary) => ({ module: r.n, summary }))
))

return {
  authored: results.filter(Boolean).map((r) => ({ module: r.n, folder: r.folder, pass: r.verdict?.pass, findingCount: r.verdict?.findings?.length ?? null })),
  fixedModules: fixed.filter(Boolean).map((f) => f.module),
}
