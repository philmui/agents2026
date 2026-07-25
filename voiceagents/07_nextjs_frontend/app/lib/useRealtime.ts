// =============================================================================
// lib/useRealtime.ts  —  PRIMARY PATH: the official @openai/agents SDK.
// =============================================================================
//
// This React hook wraps the OpenAI Agents SDK so the UI (app/page.tsx) stays
// tiny: it just calls connect()/disconnect() and reads `status` + `transcript`.
//
// The SDK does an enormous amount for us in the browser:
//   - opens a WebRTC connection to gpt-realtime-2.1,
//   - asks for the microphone and streams it up,
//   - plays the assistant's voice back through your speakers,
//   - keeps a running `history` (the transcript) we can render.
//
// We only teach it the two things it cannot guess: WHICH agent to be
// (a RealtimeAgent with instructions + a voice) and HOW to authenticate
// (the ephemeral ek_ token from our backend).
//
// The RAW version of all this plumbing lives in lib/rawWebrtc.ts so you can
// see what the SDK is doing under the hood.
// =============================================================================

"use client"; // This hook uses the microphone and WebRTC, which only exist in the browser.

import { useCallback, useEffect, useRef, useState } from "react";
import { RealtimeAgent, RealtimeSession } from "@openai/agents/realtime";
import type { RealtimeItem } from "@openai/agents/realtime";
import { fetchEphemeralToken } from "@/lib/token";

// The four states our little connection can be in. A plain string union keeps
// the UI logic (and the status pill) simple and type-safe.
export type ConnectionStatus = "idle" | "connecting" | "connected" | "error";

// One line of transcript, flattened from the SDK's richer history items so the
// UI never has to know about the SDK's internal shapes.
export type TranscriptLine = {
  id: string; // stable key for React lists (the SDK's itemId)
  role: "user" | "assistant";
  text: string; // the spoken words, transcribed to text
  done: boolean; // false while the words are still streaming in
};

// The model id is fixed for the whole course. Quoting API_FACTS.md:
// the voice assistant model is "gpt-realtime-2.1" (older DataCamp name: "gpt-realtime-2").
const MODEL = "gpt-realtime-2.1";

/**
 * Turn the SDK's `history` (an array of RealtimeItem) into simple transcript
 * lines. We only keep user/assistant *messages* and pull out their text.
 *
 * Why this is needed: each history item can hold audio, text, or a tool call.
 * For a message, `content` is an array of parts; a spoken turn shows up as an
 * "input_audio" (user) or "output_audio" (assistant) part whose `transcript`
 * field fills in as the words are recognized. We concatenate any text/transcript
 * we find so both typed and spoken turns render.
 */
function historyToTranscript(history: RealtimeItem[]): TranscriptLine[] {
  const lines: TranscriptLine[] = [];

  for (const item of history) {
    // Skip tool calls / approvals: this UI only shows the conversation.
    if (item.type !== "message") continue;
    if (item.role !== "user" && item.role !== "assistant") continue;

    // Join every text-bearing part of the message into one string.
    let text = "";
    for (const part of item.content) {
      if (part.type === "input_text" || part.type === "output_text") {
        text += part.text ?? "";
      } else if (part.type === "input_audio" || part.type === "output_audio") {
        // `transcript` is null until speech recognition catches up.
        text += part.transcript ?? "";
      }
    }

    // A message is "done" once its status is no longer "in_progress".
    // (System messages have no status; we already filtered them out above.)
    const status = (item as { status?: string }).status;
    lines.push({
      id: item.itemId,
      role: item.role,
      text: text.trim(),
      done: status ? status !== "in_progress" : true,
    });
  }

  return lines;
}

/**
 * The hook. Returns everything the UI needs and nothing it does not.
 */
export function useRealtime() {
  const [status, setStatus] = useState<ConnectionStatus>("idle");
  const [transcript, setTranscript] = useState<TranscriptLine[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [muted, setMuted] = useState(false);

  // We keep the live session in a ref (not state) because it is a long-lived
  // object, not UI data: changing it should NOT trigger a re-render.
  const sessionRef = useRef<RealtimeSession | null>(null);

  /** Open the connection: get a token, build the agent, connect over WebRTC. */
  const connect = useCallback(async () => {
    // Guard against double-clicks on the Talk button.
    if (status === "connecting" || status === "connected") return;

    setErrorMessage(null);
    setStatus("connecting");

    try {
      // 1) Ask OUR backend for a short-lived ek_ token (see lib/token.ts).
      const ephemeralKey = await fetchEphemeralToken();

      // 2) Describe the agent: its name, its personality (instructions), and
      //    its voice. The voice is chosen ONCE and cannot change mid-session
      //    (API_FACTS.md). "marin" is the default voice used across this course.
      const agent = new RealtimeAgent({
        name: "Voice Tutor",
        instructions:
          "You are a friendly voice tutor for high-school students. " +
          "Keep answers short and spoken-aloud friendly. Ask a follow-up question when helpful.",
        voice: "marin",
      });

      // 3) Create the session. In the browser the SDK defaults to WebRTC, but
      //    we name the transport explicitly so beginners can see the choice.
      const session = new RealtimeSession(agent, {
        model: MODEL,
        transport: "webrtc",
      });
      sessionRef.current = session;

      // 4) Wire up events BEFORE connecting so we never miss the first update.
      //    (event names verified against @openai/agents-realtime .d.ts)

      // The full conversation, re-emitted whenever anything changes. We flatten
      // it to transcript lines for the UI.
      session.on("history_updated", (history) => {
        setTranscript(historyToTranscript(history));
      });

      // Any SDK/transport error. We surface it and flip the pill to "error".
      session.on("error", (event) => {
        console.error("RealtimeSession error:", event.error);
        setErrorMessage(String((event.error as Error)?.message ?? event.error));
        setStatus("error");
      });

      // Optional teaching hook: barge-in. When you start talking over the
      // assistant, the SDK fires this and stops its own playback for you.
      session.on("audio_interrupted", () => {
        console.log("You interrupted the assistant (barge-in).");
      });

      // 5) Connect. This is where the WebRTC handshake + mic capture happen.
      //    We pass ONLY the ephemeral key, never the real API key.
      await session.connect({ apiKey: ephemeralKey });

      setStatus("connected");
    } catch (err) {
      console.error(err);
      setErrorMessage(err instanceof Error ? err.message : String(err));
      setStatus("error");
      // Clean up a half-open session if connect() failed partway.
      sessionRef.current?.close();
      sessionRef.current = null;
    }
  }, [status]);

  /** Close the connection and reset the UI. */
  const disconnect = useCallback(() => {
    sessionRef.current?.close(); // stops mic, closes WebRTC, ends the session
    sessionRef.current = null;
    setStatus("idle");
    setMuted(false);
  }, []);

  /** Mute/unmute the microphone without tearing down the connection. */
  const toggleMute = useCallback(() => {
    const session = sessionRef.current;
    if (!session) return;
    const next = !muted;
    session.mute(next); // SDK stops sending mic audio while muted
    setMuted(next);
  }, [muted]);

  // Safety net: if the component unmounts (user navigates away) while still
  // connected, close the session so the microphone light turns off.
  useEffect(() => {
    return () => {
      sessionRef.current?.close();
      sessionRef.current = null;
    };
  }, []);

  return { status, transcript, errorMessage, muted, connect, disconnect, toggleMute };
}
