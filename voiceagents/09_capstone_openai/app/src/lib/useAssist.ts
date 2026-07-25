// =============================================================================
// lib/useAssist.ts  -  the React hook behind ASSIST MODE.
// =============================================================================
//
// WHAT THIS FILE IS
// -----------------
// A small React hook that drives the tool-calling voice assistant. It hides all
// of the @openai/agents/realtime plumbing so the UI component (the AssistPanel)
// only has to call connect() / disconnect() and then render three things:
//
//   status      : "idle" | "connecting" | "connected" | "error"
//   transcript  : the spoken conversation, flattened to simple lines
//   toolEvents  : a running log of the ReAct loop (act -> observe steps)
//
// It wraps the @openai/agents/realtime session. The key thing here is that
// we watch the session history for TOOL CALLS and surface them, so students can
// literally see the model reason -> act -> observe -> respond.
//
// WHY A HOOK (and not code in the component)?
//   A React hook lets us keep long-lived objects (the live session) and event
//   wiring OUT of the render function. The component stays declarative: it reads
//   state and shows it; the hook owns the messy side effects.
// =============================================================================

"use client"; // Uses the microphone + WebRTC, which only exist in the browser.

import { useCallback, useEffect, useRef, useState } from "react";
import type { RealtimeItem, RealtimeSession } from "@openai/agents/realtime";
import { makeAssistant, makeSession } from "@/lib/assistAgent";
import { getEphemeralToken } from "@/lib/token";

// The four states the connection can be in. A plain string union keeps the
// status pill in the UI trivial and type-safe.
export type AssistStatus = "idle" | "connecting" | "connected" | "error";

// One line of spoken conversation, flattened from the SDK's richer history.
export type TranscriptLine = {
  id: string; // stable React key (the SDK's itemId)
  role: "user" | "assistant";
  text: string; // the words, transcribed to text
  done: boolean; // false while the words are still streaming in
};

// One entry in the ReAct log. "act" = the model decided to call the tool;
// "observe" = our code returned a result the model then reads back.
export type ToolEvent = {
  id: string; // React key
  phase: "act" | "observe";
  label: string; // e.g. a get_time/web_search call or its returned observation
};

/**
 * Flatten the SDK's history into (a) transcript lines and (b) tool events.
 *
 * Each history item is one of a few kinds. We care about two:
 *   - "message"       -> a spoken/typed turn (user or assistant)
 *   - "function_call" -> a tool call. This SINGLE item carries BOTH halves of
 *                        the loop: `name` + `arguments` are the "act", and its
 *                        `output` field (null until the tool runs, then a
 *                        string) is the "observe".
 * Everything else (MCP calls, approvals) is ignored for this teaching UI.
 */
function flattenHistory(history: RealtimeItem[]): {
  lines: TranscriptLine[];
  tools: ToolEvent[];
} {
  const lines: TranscriptLine[] = [];
  const tools: ToolEvent[] = [];

  for (const item of history) {
    // ---- a spoken or typed message ----
    if (item.type === "message") {
      if (item.role !== "user" && item.role !== "assistant") continue;

      // A message's `content` is an array of parts. Spoken turns arrive as
      // audio parts whose `transcript` fills in as recognition catches up;
      // typed turns arrive as text parts. We concatenate whatever text we find.
      let text = "";
      for (const part of item.content) {
        if (part.type === "input_text" || part.type === "output_text") {
          text += part.text ?? "";
        } else if (
          part.type === "input_audio" ||
          part.type === "output_audio"
        ) {
          text += part.transcript ?? "";
        }
      }

      // A user/assistant message has a `status`; "in_progress" means the words
      // are still streaming in. (System messages have none, but we filtered
      // those out above via the role check.)
      const status = (item as { status?: string }).status;
      lines.push({
        id: item.itemId,
        role: item.role,
        text: text.trim(),
        done: status ? status !== "in_progress" : true,
      });
      continue;
    }

    // ---- a tool call: BOTH the ACT and (once done) the OBSERVE step ----
    if (item.type === "function_call") {
      // ACT: `name` is the tool ("get_time" or "web_search"); `arguments` is its JSON
      // the arguments the model chose. We show them verbatim so the act is clear.
      const args = item.arguments ? `(${item.arguments})` : "()";
      tools.push({
        id: item.itemId + ":act",
        phase: "act",
        label: `${item.name}${args}`,
      });

      // OBSERVE: `output` is null while the tool runs, then becomes the string
      // our execute() returned. When it is present, that is what the model
      // observed, so we add the observe step right after the act.
      if (item.output != null) {
        tools.push({
          id: item.itemId + ":observe",
          phase: "observe",
          label: item.output,
        });
      }
      continue;
    }
  }

  return { lines, tools };
}

/**
 * The hook. Returns exactly what the AssistPanel needs and nothing else.
 */
export function useAssist() {
  const [status, setStatus] = useState<AssistStatus>("idle");
  const [transcript, setTranscript] = useState<TranscriptLine[]>([]);
  const [toolEvents, setToolEvents] = useState<ToolEvent[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [muted, setMuted] = useState(false);

  // The live session is a long-lived object, not UI data, so it lives in a ref
  // (updating a ref does NOT trigger a re-render).
  const sessionRef = useRef<RealtimeSession | null>(null);

  const connect = useCallback(async () => {
    // Guard against double-clicks while already connecting/connected.
    if (status === "connecting" || status === "connected") return;

    setErrorMessage(null);
    setToolEvents([]);
    setTranscript([]);
    setStatus("connecting");

    try {
      // 1) Build the agent (persona + its function tools) and its session.
      //    Both are defined in lib/assistAgent.ts. makeAssistant() also mints a
      //    per-conversation id so every web_search this connection makes groups
      //    into one Langfuse session; we do not need the id in the UI, so we
      //    discard it here.
      const { agent } = makeAssistant();
      const session = makeSession(agent);
      sessionRef.current = session;

      // 2) Wire events BEFORE connecting so we never miss the first update.
      //    history_updated re-emits the WHOLE conversation whenever anything
      //    changes; we flatten it into transcript lines + tool events.
      session.on("history_updated", (history) => {
        const { lines, tools } = flattenHistory(history);
        setTranscript(lines);
        setToolEvents(tools);
      });

      // Any transport/SDK error flips the pill to "error" and shows the message.
      session.on("error", (event) => {
        console.warn("RealtimeSession error:", event);
        const err = (event as { error?: unknown }).error ?? event;
        setErrorMessage(err instanceof Error ? err.message : String(err));
        setStatus("error");
        if (sessionRef.current === session) {
          session.close();
          sessionRef.current = null;
        }
      });

      // Teaching hook: when you talk over the assistant, the SDK stops its own
      // playback (barge-in). We just log it.
      session.on("audio_interrupted", () => {
        console.log("Barge-in: you interrupted the assistant.");
      });

      // 3) Ask OUR backend for a short-lived ek_ token, then connect. The
      //    WebRTC handshake and mic capture happen inside connect(). We pass
      //    ONLY the ephemeral key, never the real API key.
      const ephemeralKey = await getEphemeralToken("assist");
      await session.connect({ apiKey: ephemeralKey });

      setStatus("connected");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
      setStatus("error");
      sessionRef.current?.close();
      sessionRef.current = null;
    }
  }, [status]);

  const disconnect = useCallback(() => {
    sessionRef.current?.close(); // stops mic, closes WebRTC, ends the session
    sessionRef.current = null;
    setStatus("idle");
    setMuted(false);
  }, []);

  const toggleMute = useCallback(() => {
    const session = sessionRef.current;
    if (!session) return;
    const next = !muted;
    session.mute(next); // SDK stops sending mic audio while muted
    setMuted(next);
  }, [muted]);

  // Safety net: if the component unmounts (e.g. you switch modes) while still
  // connected, close the session so the microphone light turns off.
  useEffect(() => {
    return () => {
      sessionRef.current?.close();
      sessionRef.current = null;
    };
  }, []);

  return {
    status,
    transcript,
    toolEvents,
    errorMessage,
    muted,
    connect,
    disconnect,
    toggleMute,
  };
}
