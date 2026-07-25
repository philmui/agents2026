// =============================================================================
// components/TranscribePanel.tsx  —  TRANSCRIBE MODE (live speech-to-text).
// =============================================================================
//
// This mode reuses the browser WebRTC path from Module 07, but flips the session
// into "transcription" so the model returns TEXT of what you said and never
// speaks back. The connection logic lives in lib/transcribeClient.ts (a small
// class); this component just wires its callbacks to React state and renders.
//
// The exact events we rely on (verified in _shared/API_FACTS.md section 4):
//   conversation.item.input_audio_transcription.delta      -> partial words
//   conversation.item.input_audio_transcription.completed   -> a finished segment
// =============================================================================

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { TranscribeClient } from "@/lib/transcribeClient";
import { getEphemeralToken } from "@/lib/token";
import { StatusPill } from "@/components/StatusPill";

export function TranscribePanel() {
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState<string | null>(null);
  // Finished segments accumulate here; `partial` is the in-progress words.
  const [finals, setFinals] = useState<string[]>([]);
  const [partial, setPartial] = useState("");

  // The live client is a long-lived object, so it lives in a ref, not state.
  const clientRef = useRef<TranscribeClient | null>(null);
  const isRunning = status !== "idle" && status !== "stopped" && status !== "error";
  const canCommit = status === "listening" || status === "hearing you...";

  const start = useCallback(async () => {
    setError(null);
    setFinals([]);
    setPartial("");
    setStatus("connecting");

    // Create the client and give it callbacks that update our React state.
    const client = new TranscribeClient({
      onPartial: (text) => setPartial((prev) => prev + text),
      onFinal: (text) => {
        // A segment finished: push it into the list and clear the partial line.
        setFinals((prev) => [...prev, text]);
        setPartial("");
      },
      onStatus: (s) => setStatus(s),
      onError: (message) => {
        setError(message);
        setStatus("error");
      },
    });
    clientRef.current = client;

    try {
      // Fetch a short-lived ek_ token from our backend, then do the handshake.
      const token = await getEphemeralToken("transcribe");
      await client.start(token);
    } catch (err) {
      client.stop();
      clientRef.current = null;
      setError(err instanceof Error ? err.message : String(err));
      setStatus("error");
    }
  }, []);

  const stop = useCallback(() => {
    clientRef.current?.stop();
    clientRef.current = null;
    setStatus("stopped");
  }, []);

  const commit = useCallback(() => {
    clientRef.current?.commit();
  }, []);

  useEffect(() => {
    return () => {
      clientRef.current?.stop();
      clientRef.current = null;
    };
  }, []);

  return (
    <div>
      <div className="card teal">
        <h3 style={{ marginTop: 0 }}>Transcribe · live speech-to-text</h3>
        <p className="small">
          Browser WebRTC with <code>session.type:&nbsp;&quot;transcription&quot;</code>{" "}
          and <code>gpt-realtime-whisper</code> (Module 03&rsquo;s capability, in
          the browser). It listens and writes; it never talks back. Press{" "}
          <b>Finish phrase</b> after each spoken phrase to finalize that segment.
        </p>

        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          {!isRunning ? (
            <button className="btn primary" onClick={start}>
              Start transcribing
            </button>
          ) : (
            <>
              <button className="btn primary" onClick={commit} disabled={!canCommit}>
                Finish phrase
              </button>
              <button className="btn danger" onClick={stop}>
                Stop
              </button>
            </>
          )}
          <StatusPill status={status} />
        </div>

        {error && (
          <div className="caution" style={{ marginBottom: 0 }}>
            <b>Error:</b> {error}
          </div>
        )}
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Transcript</h3>
        <div className="transcript">
          {finals.length === 0 && !partial ? (
            <span className="partial">
              (Press &ldquo;Start transcribing&rdquo;, allow the mic, then speak.)
            </span>
          ) : (
            <>
              {finals.map((line, i) => (
                <p key={i} className="turn">
                  {line}
                </p>
              ))}
              {partial && <p className="turn partial">{partial}</p>}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
