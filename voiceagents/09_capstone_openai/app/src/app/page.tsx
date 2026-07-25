// =============================================================================
// app/page.tsx  -  the home page: one app, three modes, one tool-using agent.
// =============================================================================
//
// This is the home page. It renders a header, a three-way MODE SWITCH
// (Transcribe | Translate | Assist), and whichever panel matches the mode.
//
// Each panel owns its own connection, so switching modes tears the old one down
// (React unmounts the panel, whose cleanup stops the mic) and mounts a fresh
// one. That keeps exactly ONE live session at a time, which is what we want.
//
// This file is a Client Component ("use client") because the panels use the
// microphone and WebRTC, which only exist in the browser.
// =============================================================================

"use client";

import { useState } from "react";
import { TranscribePanel } from "@/components/TranscribePanel";
import { TranslatePanel } from "@/components/TranslatePanel";
import { AssistPanel } from "@/components/AssistPanel";

// The three modes as a string union. Using a type (not loose strings) means a
// typo like "Assit" would be a compile error.
type Mode = "transcribe" | "translate" | "assist";

// The order + labels shown in the switch.
const MODES: { id: Mode; label: string }[] = [
  { id: "transcribe", label: "Transcribe" },
  { id: "translate", label: "Translate" },
  { id: "assist", label: "Assist" },
];

export default function HomePage() {
  // Which mode is showing. We start on Assist because it is the headline demo.
  const [mode, setMode] = useState<Mode>("assist");

  return (
    <main className="container">
      <p className="kicker">Voice Agents · A Complete Voice App</p>
      <h1 style={{ margin: "0 0 6px" }}>One app, three voices</h1>
      <p className="lead">
        Transcribe, translate, or talk to a tool-using assistant &mdash; all in
        the browser, all over the OpenAI Realtime API, all behind one FastAPI
        backend so your key stays secret.
      </p>

      {/* The mode switch. Clicking a button swaps the panel below it. The
          `key` on the mounted panel is the mode id, so React fully remounts on
          switch and any previous mic/session is cleaned up. */}
      <div className="mode-switch" role="tablist" aria-label="Choose a mode">
        {MODES.map((m) => (
          <button
            key={m.id}
            role="tab"
            aria-selected={mode === m.id}
            className={mode === m.id ? "active" : ""}
            onClick={() => setMode(m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* Render exactly one panel. The `key` guarantees a clean remount. */}
      {mode === "transcribe" && <TranscribePanel key="transcribe" />}
      {mode === "translate" && <TranslatePanel key="translate" />}
      {mode === "assist" && <AssistPanel key="assist" />}

      <div className="footer">
        Built for the Voice Agents minicourse &middot; &copy; <b>mui-group</b>.
        The FastAPI backend mints ephemeral <code>ek_</code> tokens and proxies
        translation; the browser never sees your real key.
      </div>
    </main>
  );
}
