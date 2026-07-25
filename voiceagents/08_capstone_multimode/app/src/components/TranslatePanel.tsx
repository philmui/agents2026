// =============================================================================
// components/TranslatePanel.tsx  -  TRANSLATE MODE (live, in the browser).
// =============================================================================
//
// Speak one language, hear another, without leaving the app. Pick a target
// language, press Start, allow the mic, and talk. You will SEE the source text
// (what you said) and the target text (the translation) stream in, and HEAR the
// translated speech play back.
//
// All the audio + WebSocket work lives in lib/translateClient.ts. This component
// just wires that client's callbacks to React state and renders. The client talks
// to our FastAPI backend (WS /translate), which holds the real key and relays to
// OpenAI's gpt-realtime-translate endpoint. See translateClient.ts for exactly why
// this must go through the backend and cannot be a direct browser socket.
// =============================================================================

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { TranslateClient, TRANSLATE_LANGUAGES } from "@/lib/translateClient";
import { StatusPill } from "@/components/StatusPill";

export function TranslatePanel() {
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState<string | null>(null);
  const [language, setLanguage] = useState("es"); // target language CODE
  // The two live transcripts. Each streams in as small deltas we append.
  const [source, setSource] = useState(""); // what you said (auto-detected language)
  const [target, setTarget] = useState(""); // the translation (target language)

  // The live client is a long-lived object, so it lives in a ref, not state.
  const clientRef = useRef<TranslateClient | null>(null);
  const isRunning =
    status !== "idle" && status !== "stopped" && status !== "error";

  const start = useCallback(async () => {
    setError(null);
    setSource("");
    setTarget("");

    // Create the client and hand it callbacks that update our React state.
    const client = new TranslateClient({
      onStatus: (s) => setStatus(s),
      // Source updates are already assembled by item id so completed phrases can
      // replace partial text without duplicating it.
      onSource: (text) => setSource(text),
      onTarget: (delta) => setTarget((prev) => prev + delta),
      onError: (message) => {
        setError(message);
        setStatus("error");
      },
    });
    clientRef.current = client;

    try {
      await client.start(language);
    } catch (err) {
      client.stop();
      clientRef.current = null;
      setError(err instanceof Error ? err.message : String(err));
      setStatus("error");
    }
  }, [language]);

  const stop = useCallback(() => {
    clientRef.current?.stop();
    clientRef.current = null;
    setStatus("stopped");
  }, []);

  useEffect(() => {
    return () => {
      clientRef.current?.stop();
      clientRef.current = null;
    };
  }, []);

  return (
    <div>
      <div className="card pink">
        <h3 style={{ marginTop: 0 }}>Translate · speak one language, hear another</h3>
        <p className="small">
          Streams your mic to the backend, which relays it to{" "}
          <code>gpt-realtime-translate</code> and sends the translated speech back.
          Pick a target language, press Start, and talk. The source language is
          auto-detected.
        </p>

        {/* Target-language picker (source is auto-detected). Locked while running
            because the language is chosen once when the session starts. */}
        <label className="small" htmlFor="lang">
          Translate INTO:{" "}
        </label>
        <select
          id="lang"
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          disabled={isRunning}
          style={{
            font: "inherit",
            padding: "8px 12px",
            borderRadius: 10,
            border: "1px solid var(--line)",
            marginRight: 10,
          }}
        >
          {TRANSLATE_LANGUAGES.map((l) => (
            <option key={l.code} value={l.code}>
              {l.name} ({l.code})
            </option>
          ))}
        </select>

        <div
          style={{
            display: "flex",
            gap: 10,
            alignItems: "center",
            flexWrap: "wrap",
            marginTop: 12,
          }}
        >
          {!isRunning ? (
            <button className="btn primary" onClick={start}>
              Start translating
            </button>
          ) : (
            <button className="btn danger" onClick={stop}>
              Stop
            </button>
          )}
          <StatusPill status={status} />
        </div>

        {error && (
          <div className="caution" style={{ marginBottom: 0 }}>
            <b>Error:</b> {error}
          </div>
        )}
      </div>

      {/* The two live transcripts, side by side on wide screens. */}
      <div className="card teal">
        <h3 style={{ marginTop: 0 }}>You said (source)</h3>
        <div className="transcript">
          {source ? (
            <p className="turn">{source}</p>
          ) : (
            <span className="partial">
              (Press &ldquo;Start translating&rdquo;, allow the mic, then speak.)
            </span>
          )}
        </div>
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Translation (target)</h3>
        <div className="transcript">
          {target ? (
            <p className="turn">{target}</p>
          ) : (
            <span className="partial">(The translation appears here, and plays aloud.)</span>
          )}
        </div>
      </div>

      {/* The teaching point: why this goes through our backend, stated plainly. */}
      <div className="caution">
        <b>Why through the backend?</b> Translation uses a <b>WebSocket</b> that
        authenticates with an <code>Authorization: Bearer</code> header, and
        browsers cannot set headers on a WebSocket. So the browser opens a plain
        socket to our FastAPI backend (<code>WS /translate</code>), which holds the
        real key, opens the authenticated OpenAI socket, and relays audio and
        transcripts both ways. Your key never reaches the browser.
      </div>
    </div>
  );
}
