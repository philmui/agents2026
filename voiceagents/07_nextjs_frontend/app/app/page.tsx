// =============================================================================
// app/page.tsx  —  the ONE screen of the app.
// =============================================================================
//
// The whole UI is here: a big "Talk" button, a "Mute" button, a connection
// status pill, and a live transcript panel. A small toggle at the bottom lets
// you switch the ENGINE between:
//   - "SDK"  → the official @openai/agents SDK  (lib/useRealtime.ts)  ← primary
//   - "Raw"  → a hand-written WebRTC handshake  (lib/rawWebrtc.ts)    ← teaching
//
// Both engines produce the SAME transcript lines and the SAME status values, so
// the UI code below does not care which one is running. That is the point: the
// SDK is just a tidy wrapper around the raw plumbing.
// =============================================================================

"use client"; // This page uses the microphone, WebRTC, and React state: browser-only.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  useRealtime,
  type ConnectionStatus,
  type TranscriptLine,
} from "@/lib/useRealtime";
import { connectRawWebrtc, type RawConnection } from "@/lib/rawWebrtc";
import { fetchEphemeralToken, tokenEndpoint } from "@/lib/token";

// Which engine is selected. "sdk" is the recommended, primary path.
type Engine = "sdk" | "raw";

export default function Home() {
  // Which connection engine the user picked. Default to the official SDK.
  const [engine, setEngine] = useState<Engine>("sdk");

  return (
    <main className="page">
      {/* --- Header ---------------------------------------------------------- */}
      <header>
        <p className="kicker">Voice Agents · Module 07</p>
        <h1>Talk to gpt-realtime-2.1</h1>
        <p className="subtitle">
          A minimal browser voice agent. Click <strong>Talk</strong>, allow the
          microphone, and speak. Your voice streams to OpenAI over WebRTC and the
          reply comes back as audio and text.
        </p>
      </header>

      {/* The card swaps its inner engine when you flip the toggle. We give it a
          `key` so React fully remounts it on switch, guaranteeing the previous
          connection is torn down before the new engine mounts. */}
      {engine === "sdk" ? (
        <SdkPanel key="sdk" />
      ) : (
        <RawPanel key="raw" />
      )}

      {/* --- Engine toggle + notes ------------------------------------------ */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <span className="hint">Connection engine:</span>
        <div className="path-toggle" role="group" aria-label="Connection engine">
          <button
            className={engine === "sdk" ? "active" : ""}
            onClick={() => setEngine("sdk")}
          >
            SDK (recommended)
          </button>
          <button
            className={engine === "raw" ? "active" : ""}
            onClick={() => setEngine("raw")}
          >
            Raw WebRTC
          </button>
        </div>
      </div>

      <p className="hint">
        Token endpoint: <code>{tokenEndpoint}</code>. This is the module-06 backend
        that hands the browser a short-lived <code>ek_</code> token. Start it first
        (<code>uv run uvicorn src.main:app --reload --port 8000</code>). The browser
        never sees your real OpenAI key.
      </p>

      <p className="footer">
        Built by <b>mui-group</b> · advanced high-school students
      </p>
    </main>
  );
}

// =============================================================================
// Shared presentational pieces (used by BOTH engine panels).
// =============================================================================

/** The colored status pill. Pure display: it just reads the status string. */
function StatusPill({ status }: { status: ConnectionStatus }) {
  // Map each status to the words shown next to the colored dot.
  const label: Record<ConnectionStatus, string> = {
    idle: "Idle",
    connecting: "Connecting…",
    connected: "Connected",
    error: "Error",
  };
  return (
    <span className={`status-pill ${status}`}>
      <span className="dot" />
      {label[status]}
    </span>
  );
}

/** The transcript panel. Renders one bubble per turn; blinks while streaming. */
function Transcript({ lines }: { lines: TranscriptLine[] }) {
  // Auto-scroll to the newest line whenever the transcript grows.
  const endRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines]);

  if (lines.length === 0) {
    return (
      <div className="transcript">
        <p className="empty">
          Your conversation will appear here once you start talking.
        </p>
      </div>
    );
  }

  return (
    <div className="transcript">
      {lines.map((line) => (
        <div
          key={line.id}
          className={`line ${line.role} ${line.done ? "" : "streaming"}`}
        >
          <span className="who">{line.role}</span>
          {line.text}
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}

/** The control row: Talk / Stop button, Mute, and the status pill. */
function Controls({
  status,
  muted,
  onTalk,
  onStop,
  onToggleMute,
}: {
  status: ConnectionStatus;
  muted: boolean;
  onTalk: () => void;
  onStop: () => void;
  onToggleMute: () => void;
}) {
  const connected = status === "connected";
  const connecting = status === "connecting";
  return (
    <div className="controls">
      <button
        className={`talk-btn ${connected ? "stop" : ""}`}
        onClick={connected ? onStop : onTalk}
        disabled={connecting}
      >
        {connected ? "Stop" : connecting ? "Connecting…" : "Talk"}
      </button>

      {/* Mute only makes sense while connected. */}
      <button
        className={`ghost-btn ${muted ? "active" : ""}`}
        onClick={onToggleMute}
        disabled={!connected}
      >
        {muted ? "Unmute" : "Mute"}
      </button>

      <StatusPill status={status} />
    </div>
  );
}

/** Shared error banner (null-safe: renders nothing when there is no error). */
function ErrorBanner({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div className="error-banner" role="alert">
      <b>Something went wrong.</b> {message}
    </div>
  );
}

// =============================================================================
// PANEL 1 — the SDK engine (PRIMARY). Almost all logic lives in the hook.
// =============================================================================

function SdkPanel() {
  // The hook does the heavy lifting; the panel is just wiring to the UI pieces.
  const { status, transcript, errorMessage, muted, connect, disconnect, toggleMute } =
    useRealtime();

  return (
    <section className="card">
      <Controls
        status={status}
        muted={muted}
        onTalk={connect}
        onStop={disconnect}
        onToggleMute={toggleMute}
      />
      <ErrorBanner message={errorMessage} />
      <Transcript lines={transcript} />
    </section>
  );
}

// =============================================================================
// PANEL 2 — the RAW WebRTC engine (teaching). Same UI, hand-written plumbing.
// =============================================================================
//
// This panel manages the raw connection itself so you can see, with no SDK, how
// the same button drives the same transcript. The connection logic is imported
// from lib/rawWebrtc.ts; here we only translate its callbacks into React state.

function RawPanel() {
  const [status, setStatus] = useState<ConnectionStatus>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [muted, setMuted] = useState(false);

  // The transcript is stored keyed by item id so streaming deltas can UPDATE the
  // matching line in place. We convert it to an ordered array for rendering.
  const [byId, setById] = useState<Record<string, TranscriptLine>>({});
  const order = useRef<string[]>([]); // remembers the order lines first appeared

  // Long-lived objects kept in refs (they are not UI data):
  const connRef = useRef<RawConnection | null>(null); // the live WebRTC connection
  const audioRef = useRef<HTMLAudioElement | null>(null); // where we play the reply

  // Turn the id-keyed map into an array in first-seen order for <Transcript/>.
  const transcript = useMemo<TranscriptLine[]>(
    () => order.current.map((id) => byId[id]).filter(Boolean),
    [byId]
  );

  // Merge one streamed transcript fragment into state. For assistant deltas we
  // APPEND to the running text; for final lines we REPLACE with the full text.
  const applyTranscript = useCallback(
    (line: { id: string; role: "user" | "assistant"; text: string; done: boolean }) => {
      setById((prev) => {
        const existing = prev[line.id];
        if (!existing) order.current.push(line.id); // first time we see this turn
        const text = line.done
          ? line.text // final: authoritative full text
          : (existing?.text ?? "") + line.text; // streaming: append the delta
        return {
          ...prev,
          [line.id]: { id: line.id, role: line.role, text, done: line.done },
        };
      });
    },
    []
  );

  const onTalk = useCallback(async () => {
    if (status === "connecting" || status === "connected") return;
    setErrorMessage(null);
    setStatus("connecting");
    try {
      // 1) Ask our backend for the short-lived ek_ token (same helper as the SDK).
      const ephemeralKey = await fetchEphemeralToken();
      // 2) Run the hand-written handshake, feeding transcripts into React state.
      const conn = await connectRawWebrtc(
        ephemeralKey,
        audioRef.current!, // the <audio> element below; "!" = we know it is mounted
        applyTranscript
      );
      connRef.current = conn;
      setStatus("connected");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
      setStatus("error");
    }
  }, [status, applyTranscript]);

  const onStop = useCallback(() => {
    connRef.current?.close(); // stop mic, close the peer connection + data channel
    connRef.current = null;
    setStatus("idle");
    setMuted(false);
  }, []);

  // Mute by disabling the microphone track directly (no SDK to do it for us).
  const onToggleMute = useCallback(() => {
    const conn = connRef.current;
    if (!conn) return;
    const next = !muted;
    conn.micStream.getAudioTracks().forEach((t) => (t.enabled = !next));
    setMuted(next);
  }, [muted]);

  // Clean up if the user navigates away or flips the toggle while connected.
  useEffect(() => {
    return () => {
      connRef.current?.close();
      connRef.current = null;
    };
  }, []);

  return (
    <section className="card">
      <Controls
        status={status}
        muted={muted}
        onTalk={onTalk}
        onStop={onStop}
        onToggleMute={onToggleMute}
      />
      <ErrorBanner message={errorMessage} />
      <Transcript lines={transcript} />
      {/* Hidden player: the raw path points this element at the incoming audio.
          `autoPlay` lets the reply start on its own once a stream is attached. */}
      <audio ref={audioRef} autoPlay hidden />
    </section>
  );
}
