// =============================================================================
// lib/translateClient.ts  -  TRANSLATE MODE, in the browser, for real.
// =============================================================================
//
// WHAT THIS DOES
// --------------
// Speak one language, hear another, entirely inside the web app. The catch (and
// the whole reason the backend exists) is that OpenAI's translation endpoint is a
// WebSocket that authenticates with an "Authorization: Bearer <key>" HEADER, and
// browsers CANNOT set headers on a WebSocket. So we do NOT talk to OpenAI directly
// here. Instead:
//
//     [ this browser code ] --plain WS--> [ our FastAPI backend ] --auth WS--> [ OpenAI ]
//
// The backend (backend/src/main.py, route WS /translate) holds the real key, opens
// the authenticated OpenAI socket, and relays messages both ways. This file:
//   1) opens a plain WebSocket to the backend,
//   2) captures the microphone and converts it to PCM16 @ 24 kHz (what OpenAI wants),
//   3) streams that audio as base64 to the backend,
//   4) plays the translated audio the backend sends back, and
//   5) reports the source + target transcripts to the UI. The backend obtains
//      reliable source captions from a parallel Realtime transcription sidecar.
//
// AUDIO, PLAINLY (the required teaching point, see docs/API_FACTS.md)
// ------------------------------------------------------------------------
// A microphone turns sound into a stream of numbers ("samples"). "PCM16 @ 24 kHz
// mono" means: 24000 samples per second, one channel, each sample a 16-bit signed
// integer. The browser's Web Audio API gives us samples as floats in [-1, 1]; we
// convert those to 16-bit integers before sending, and convert the integers we
// receive back to floats before playing. base64 just packs those raw bytes into
// plain text so they fit inside a JSON message.
//
// The tiny message protocol between this file and the backend (mirrors main.py):
//   browser -> backend : {type:"start", language:"es"}   then many {type:"audio", audio:"<b64>"}
//   backend -> browser : {type:"ready"} | {type:"source",item_id,delta}
//                        | {type:"source",item_id,transcript,completed:true}
//                        | {type:"target",delta} | {type:"audio",delta:"<b64>"}
//                        | {type:"error",message}
// =============================================================================

// The audio format OpenAI's Realtime API uses (see API_FACTS.md). We create our
// AudioContext at exactly this rate so the browser resamples the mic for us and
// the samples we send are already at 24 kHz.
const SAMPLE_RATE = 24000;

// The backend may wait up to ten seconds each for translation and source-caption
// flushes. Keep the browser socket alive long enough to receive the final phrase.
const STOP_TIMEOUT_MS = 22000;

// Callbacks the UI hands us so we can push updates back into React state.
export type TranslateHandlers = {
  onStatus: (status: string) => void; // human-readable connection status
  onSource: (text: string) => void; // complete SOURCE transcript assembled so far
  onTarget: (text: string) => void; // a slice of the TARGET transcript (the translation)
  onError: (message: string) => void;
};

// A base URL for the backend, read from the environment (see lib/backend.ts).
import { BACKEND_WS_URL, BACKEND_CALLER_TOKEN } from "@/lib/backend";

export class TranslateClient {
  private ws: WebSocket | null = null;
  private handlers: TranslateHandlers;

  // Web Audio objects. One AudioContext runs both capture and playback at 24 kHz.
  private audioCtx: AudioContext | null = null;
  private mic: MediaStream | null = null;
  private micSource: MediaStreamAudioSourceNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private stopping = false;
  private closeTimer: number | null = null;

  // Source transcription arrives as deltas plus a corrected completed phrase.
  // Keep each item separate so completion can replace its own partial text
  // without duplicating words already shown.
  private sourceOrder: string[] = [];
  private sourceSegments = new Map<string, string>();

  // Playback scheduling: we line translated audio chunks up back-to-back so they
  // play as one smooth stream instead of overlapping. `playHead` is the time (on
  // the AudioContext clock) at which the next chunk should start.
  private playHead = 0;

  constructor(handlers: TranslateHandlers) {
    this.handlers = handlers;
  }

  // ---- Public: open the backend socket, the mic, and start translating -------
  // `language` is the TARGET language code (e.g. "es"). Source is auto-detected.
  async start(language: string) {
    this.handlers.onStatus("connecting");
    this.stopping = false;
    this.sourceOrder = [];
    this.sourceSegments.clear();

    // Create and resume audio while this method is still running from the click
    // gesture. Waiting for the backend before doing this can trigger autoplay
    // blocking in Safari and Chrome.
    const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
    this.audioCtx = audioCtx;
    this.playHead = audioCtx.currentTime;
    await audioCtx.resume();

    // 1) Open a plain WebSocket to OUR backend. No secret needed here: this leg
    //    is browser <-> our server. The backend adds the real key on its side.
    const ws = new WebSocket(`${BACKEND_WS_URL}/translate`);
    this.ws = ws;

    // When the socket opens, tell the backend which language to translate INTO.
    // Include the optional shared caller token (a WebSocket cannot set an
    // Authorization header, so the backend reads it from this first message).
    ws.addEventListener("open", () => {
      const start: { type: "start"; language: string; token?: string } = {
        type: "start",
        language,
      };
      if (BACKEND_CALLER_TOKEN) start.token = BACKEND_CALLER_TOKEN;
      ws.send(JSON.stringify(start));
    });

    // Handle every message the backend relays back to us.
    ws.addEventListener("message", (e) => {
      void this.onBackendMessage(e.data).catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        this.handlers.onError(`Could not start audio: ${message}`);
        this.closeNow(false);
      });
    });

    // If the socket errors or closes unexpectedly, surface it and clean up.
    ws.addEventListener("error", () => {
      this.handlers.onError("Connection to the translation backend failed.");
      this.closeNow(false);
    });
    ws.addEventListener("close", () => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.cleanupAudio();
      if (this.closeTimer !== null) window.clearTimeout(this.closeTimer);
      this.closeTimer = null;
      if (this.stopping) {
        this.handlers.onStatus("stopped");
      } else {
        this.handlers.onError("The translation connection closed unexpectedly.");
      }
    });
  }

  // ---- Handle one message from the backend ----------------------------------
  private async onBackendMessage(raw: string) {
    let msg: {
      type?: string;
      delta?: string;
      transcript?: string;
      item_id?: string;
      completed?: boolean;
      message?: string;
    };
    try {
      msg = JSON.parse(raw);
    } catch {
      return; // ignore anything that is not JSON
    }

    switch (msg.type) {
      // The backend opened the OpenAI socket. Now it is safe to start the mic.
      case "ready":
        this.handlers.onStatus("listening");
        await this.startMic();
        break;

      // Source captions come from the backend's transcription sidecar. Deltas
      // stream quickly; the completed phrase replaces that item's partial text.
      case "source": {
        const itemId = msg.item_id || "source";
        if (!this.sourceSegments.has(itemId)) {
          this.sourceOrder.push(itemId);
          this.sourceSegments.set(itemId, "");
        }
        if (msg.completed && typeof msg.transcript === "string") {
          this.sourceSegments.set(itemId, msg.transcript.trim());
        } else if (msg.delta) {
          this.sourceSegments.set(
            itemId,
            (this.sourceSegments.get(itemId) || "") + msg.delta,
          );
        }
        const transcript = this.sourceOrder
          .map((id) => this.sourceSegments.get(id)?.trim() || "")
          .filter(Boolean)
          .join(" ");
        this.handlers.onSource(transcript);
        break;
      }

      // A slice of the target transcript (the translation, as text).
      case "target":
        if (msg.delta) this.handlers.onTarget(msg.delta);
        break;

      // Translated speech. The bytes are base64 PCM16 @ 24 kHz; play them.
      case "audio":
        if (msg.delta) this.playPcm16(msg.delta);
        break;

      // The backend (or OpenAI) reported a problem.
      case "error":
        this.handlers.onError(msg.message || "Unknown translation error.");
        this.closeNow(false);
        break;

      // The backend waited for OpenAI to flush its remaining translated output.
      case "closed":
        this.closeNow(true);
        break;
    }
  }

  // ---- Microphone capture: mic floats -> PCM16 -> base64 -> backend ----------
  private async startMic() {
    // One AudioContext for the whole session, forced to 24 kHz so the browser
    // resamples the mic to the rate OpenAI expects. (Playback uses it too.)
    const audioCtx =
      this.audioCtx ?? new AudioContext({ sampleRate: SAMPLE_RATE });
    this.audioCtx = audioCtx;
    if (audioCtx.state === "suspended") await audioCtx.resume();
    // Start the playback clock a hair in the future so the first chunk is not late.
    this.playHead = audioCtx.currentTime;

    // Ask for the microphone (pops the browser's "allow microphone?" prompt).
    const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.mic = mic;

    // Wire the mic into the audio graph: mic source -> processor.
    const micSource = audioCtx.createMediaStreamSource(mic);
    this.micSource = micSource;

    // A ScriptProcessorNode calls us with small blocks of mic samples. It is the
    // simplest capture node to teach (the newer AudioWorklet is more code for the
    // same result). 4096 samples per block, 1 input channel, 1 output channel.
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    this.processor = processor;

    processor.onaudioprocess = (event) => {
      // Guard: if we are shutting down or the socket is not open, do nothing.
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

      // The mic block as Float32 samples in [-1, 1] at 24 kHz.
      const floats = event.inputBuffer.getChannelData(0);
      // Convert to base64 PCM16 and send it as one "audio" message.
      const b64 = floatsToBase64Pcm16(floats);
      this.ws.send(JSON.stringify({ type: "audio", audio: b64 }));
    };

    // Connect the graph. The processor must reach a destination to run, but we do
    // NOT want to hear our own mic, so we route it through a muted gain node.
    const mute = audioCtx.createGain();
    mute.gain.value = 0;
    micSource.connect(processor);
    processor.connect(mute);
    mute.connect(audioCtx.destination);
  }

  // ---- Playback: base64 PCM16 -> floats -> scheduled AudioBuffer -------------
  private playPcm16(base64: string) {
    const audioCtx = this.audioCtx;
    if (!audioCtx) return;

    // Decode base64 -> raw bytes -> Int16 samples -> Float32 in [-1, 1].
    const floats = base64Pcm16ToFloats(base64);
    if (floats.length === 0) return;

    // Wrap the floats in an AudioBuffer at our 24 kHz rate. We copy into the
    // buffer's own channel array with .set(...) (rather than copyToChannel) so we
    // do not depend on the exact typed-array backing-buffer type the DOM lib wants.
    const buffer = audioCtx.createBuffer(1, floats.length, SAMPLE_RATE);
    buffer.getChannelData(0).set(floats);

    // Schedule this chunk to start where the previous one ended, so the stream is
    // gapless. If we have fallen behind (playHead in the past), catch up to now.
    const startAt = Math.max(this.playHead, audioCtx.currentTime);
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);
    source.start(startAt);
    this.playHead = startAt + buffer.duration;
  }

  // ---- Tear everything down: mic, audio graph, and the socket ----------------
  stop() {
    const ws = this.ws;
    this.stopping = true;
    this.cleanupAudio();

    if (ws && ws.readyState === WebSocket.OPEN) {
      // Ask the backend to flush both OpenAI sessions. It replies "closed" after
      // target output and the final source phrase arrive; the timeout prevents a
      // stuck shutdown.
      try {
        ws.send(JSON.stringify({ type: "stop" }));
      } catch {
        this.closeNow(true);
        return;
      }
      this.handlers.onStatus("stopping");
      this.closeTimer = window.setTimeout(
        () => this.closeNow(true),
        STOP_TIMEOUT_MS,
      );
      return;
    }

    this.closeNow(true);
  }

  private closeNow(intentional: boolean) {
    const ws = this.ws;
    this.ws = null;
    this.stopping = intentional;
    if (this.closeTimer !== null) window.clearTimeout(this.closeTimer);
    this.closeTimer = null;
    if (ws && ws.readyState < WebSocket.CLOSING) ws.close();
    this.cleanupAudio();
    if (intentional) this.handlers.onStatus("stopped");
  }

  private cleanupAudio() {
    // Stop the mic hardware (turns off the browser's mic indicator).
    if (this.mic) this.mic.getTracks().forEach((t) => t.stop());

    // Disconnect the audio graph nodes.
    if (this.processor) this.processor.disconnect();
    if (this.micSource) this.micSource.disconnect();

    // Close the AudioContext to free the audio device.
    if (this.audioCtx && this.audioCtx.state !== "closed") {
      void this.audioCtx.close();
    }

    this.mic = null;
    this.processor = null;
    this.micSource = null;
    this.audioCtx = null;
  }
}

// =============================================================================
// Small pure helpers for the audio <-> base64 conversions. Kept out of the class
// so they are easy to read and test in isolation.
// =============================================================================

/**
 * Convert Float32 mic samples in [-1, 1] to a base64 string of little-endian
 * PCM16 bytes (what OpenAI's Realtime API expects).
 */
function floatsToBase64Pcm16(floats: Float32Array): string {
  // Each float becomes one 16-bit integer, so the byte buffer is 2x the length.
  const pcm = new Int16Array(floats.length);
  for (let i = 0; i < floats.length; i++) {
    // Clamp to [-1, 1] then scale to the 16-bit signed range. 0x7fff = 32767.
    const s = Math.max(-1, Math.min(1, floats[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  // Interpret the Int16Array's memory as raw bytes and base64-encode them.
  return bytesToBase64(new Uint8Array(pcm.buffer));
}

/**
 * Convert a base64 string of little-endian PCM16 bytes back to Float32 samples
 * in [-1, 1], ready to drop into an AudioBuffer for playback.
 */
function base64Pcm16ToFloats(base64: string): Float32Array {
  const bytes = base64ToBytes(base64);
  // View the same bytes as 16-bit signed integers (2 bytes each).
  const pcm = new Int16Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 2));
  const floats = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) {
    // Scale the 16-bit integer back into the [-1, 1] float range.
    floats[i] = pcm[i] / (pcm[i] < 0 ? 0x8000 : 0x7fff);
  }
  return floats;
}

/** Base64-encode a byte array (browser-safe, no Node Buffer). */
function bytesToBase64(bytes: Uint8Array): string {
  // Build a binary string in chunks so we do not blow the call-stack on big arrays.
  let binary = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

/** Decode a base64 string to a byte array (browser-safe, no Node Buffer). */
function base64ToBytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

// =============================================================================
// The 13 target languages gpt-realtime-translate supports, with codes. Shown in
// the TranslatePanel dropdown. Kept in sync with 04_translation/src/translate.py.
// =============================================================================
export const TRANSLATE_LANGUAGES: { name: string; code: string }[] = [
  { name: "Spanish", code: "es" },
  { name: "Portuguese", code: "pt" },
  { name: "French", code: "fr" },
  { name: "Japanese", code: "ja" },
  { name: "Russian", code: "ru" },
  { name: "Chinese", code: "zh" },
  { name: "German", code: "de" },
  { name: "Korean", code: "ko" },
  { name: "Hindi", code: "hi" },
  { name: "Indonesian", code: "id" },
  { name: "Vietnamese", code: "vi" },
  { name: "Italian", code: "it" },
  { name: "English", code: "en" },
];
