// =============================================================================
// components/AssistPanel.tsx  —  ASSIST MODE (the capstone headline).
// =============================================================================
//
// A speech-to-speech voice assistant that can CALL A TOOL. All the hard work is
// in lib/useAssist.ts (which wraps lib/assistAgent.ts). This component is just
// the screen: a Talk button, a status pill, the live transcript, and a small
// "ReAct loop" panel that shows the model reason -> act -> observe -> respond.
//
// The SDK plays the assistant's voice for us through your speakers, so there is
// no <audio> element to manage here (contrast Transcribe mode, which uses raw
// WebRTC). We only render text.
// =============================================================================

"use client";

import { useAssist } from "@/lib/useAssist";
import { StatusPill } from "@/components/StatusPill";

export function AssistPanel() {
  // Everything the panel needs comes from the hook.
  const {
    status,
    transcript,
    toolEvents,
    errorMessage,
    muted,
    connect,
    disconnect,
    toggleMute,
  } = useAssist();

  const isConnected = status === "connected";
  const isBusy = status === "connecting";

  return (
    <div>
      <div className="card">
        <h3 style={{ marginTop: 0 }}>Assist · talk to a tool-using agent</h3>
        <p className="small">
          Uses <code>@openai/agents/realtime</code> over WebRTC.
          Try: <em>&ldquo;What time is it in Tokyo?&rdquo;</em> or{" "}
          <em>&ldquo;Search the web for today&rsquo;s top AI story.&rdquo;</em> Then
          watch the agent call <code>get_time</code> or <code>web_search</code>.
        </p>

        {/* Controls: connect / disconnect / mute, plus the live status pill. */}
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          {!isConnected ? (
            <button
              className="btn primary"
              onClick={connect}
              disabled={isBusy}
            >
              {isBusy ? "Connecting…" : "Start talking"}
            </button>
          ) : (
            <>
              <button className="btn danger" onClick={disconnect}>
                Stop
              </button>
              <button className="btn primary" onClick={toggleMute}>
                {muted ? "Unmute mic" : "Mute mic"}
              </button>
            </>
          )}
          <StatusPill status={status} />
        </div>

        {errorMessage && (
          <div className="caution" style={{ marginBottom: 0 }}>
            <b>Error:</b> {errorMessage}
          </div>
        )}
      </div>

      {/* The spoken conversation, transcribed live. */}
      <div className="card teal">
        <h3 style={{ marginTop: 0 }}>Conversation</h3>
        <div className="transcript">
          {transcript.length === 0 ? (
            <span className="partial">
              (Press &ldquo;Start talking&rdquo;, allow the mic, then say hello.)
            </span>
          ) : (
            transcript.map((line) => (
              <p key={line.id} className="turn">
                <b>{line.role === "user" ? "You" : "Assistant"}:</b>{" "}
                <span className={line.done ? "" : "partial"}>{line.text}</span>
              </p>
            ))
          )}
        </div>
      </div>

      {/* The ReAct loop, made visible: each tool call shows an act step and an
          observe step containing the clock result or grounded search answer. */}
      <div className="card pink">
        <h3 style={{ marginTop: 0 }}>ReAct loop · tool activity</h3>
        <p className="small" style={{ marginTop: 0 }}>
          reason &rarr; <b>act</b> (call the tool) &rarr; <b>observe</b> (read the
          result) &rarr; respond (speak it).
        </p>
        <div className="transcript" style={{ minHeight: 80 }}>
          {toolEvents.length === 0 ? (
            <span className="partial">
              (No tool calls yet. Ask for the time or current web information.)
            </span>
          ) : (
            toolEvents.map((ev) => (
              <p key={ev.id} className="turn">
                <b>{ev.phase === "act" ? "ACT" : "OBSERVE"}:</b>{" "}
                <code>{ev.label}</code>
              </p>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
