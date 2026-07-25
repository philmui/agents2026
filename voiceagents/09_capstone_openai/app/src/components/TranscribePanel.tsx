// =============================================================================
// components/TranscribePanel.tsx  -  TRANSCRIBE MODE (live speech-to-text).
// =============================================================================
//
// This mode uses the browser's WebRTC path, with the session set to
// "transcription" so the model returns TEXT of what you said and never
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
  // A monotonically increasing "generation". Every start() captures the value it
  // began with; stop(), a terminal error, unmount, or a NEW start() all bump it.
  // A pending start() whose captured generation no longer matches must abort and
  // never open (or must immediately close) a microphone. This is what makes the
  // lifecycle cancellable and idempotent: only the LATEST start() owns the mic.
  const runIdRef = useRef(0);
  const isRunning = status !== "idle" && status !== "stopped" && status !== "error";
  const canCommit = status === "listening" || status === "hearing you...";

  // Stop the EXACT client passed in, but only if it is still the current one, and
  // clear the ref so a later stop()/unmount cannot double-close a reused slot.
  const stopClient = useCallback((client: TranscribeClient | null) => {
    if (!client) return;
    client.stop();
    if (clientRef.current === client) clientRef.current = null;
  }, []);

  const start = useCallback(async () => {
    // Bump the generation: this new run supersedes any in-flight start().
    const runId = ++runIdRef.current;
    // Tear down any previous client before starting a fresh one.
    stopClient(clientRef.current);

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
      onStatus: (s) => {
        if (runIdRef.current === runId) setStatus(s);
      },
      onError: (message) => {
        // A terminal transport error: STOP this exact client so its mic + peer
        // connection are released (not just flip a state flag), then surface it.
        stopClient(client);
        if (runIdRef.current !== runId) return; // superseded; stay quiet
        runIdRef.current++; // this run is over; ignore its late callbacks
        setError(message);
        setStatus("error");
      },
    });
    clientRef.current = client;

    try {
      // Fetch a short-lived ek_ token from our backend, then do the handshake.
      const token = await getEphemeralToken("transcribe");
      // If we were stopped/unmounted/superseded while the token was in flight,
      // do NOT open a microphone. Abort cleanly.
      if (runIdRef.current !== runId) {
        client.stop();
        if (clientRef.current === client) clientRef.current = null;
        return;
      }
      await client.start(token);
    } catch (err) {
      stopClient(client);
      if (runIdRef.current !== runId) return; // superseded; do not touch state
      runIdRef.current++;
      setError(err instanceof Error ? err.message : String(err));
      setStatus("error");
    }
  }, [stopClient]);

  const stop = useCallback(() => {
    runIdRef.current++; // invalidate any in-flight start()
    stopClient(clientRef.current);
    setStatus("stopped");
  }, [stopClient]);

  const commit = useCallback(() => {
    clientRef.current?.commit();
  }, []);

  useEffect(() => {
    return () => {
      // Unmount (e.g. switching modes): invalidate pending starts so a token
      // fetch that resolves after unmount cannot open an orphaned mic, and stop
      // whatever client currently exists.
      runIdRef.current++;
      stopClient(clientRef.current);
    };
  }, [stopClient]);

  return (
    <div>
      <div className="card teal">
        <h3 style={{ marginTop: 0 }}>Transcribe · live speech-to-text</h3>
        <p className="small">
          Browser WebRTC with <code>session.type:&nbsp;&quot;transcription&quot;</code>{" "}
          and <code>gpt-realtime-whisper</code> (live speech-to-text, in
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
