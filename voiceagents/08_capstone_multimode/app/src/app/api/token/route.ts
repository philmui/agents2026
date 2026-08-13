// ---------------------------------------------------------------------------
// /api/token  —  the TOKEN BACKEND, running INSIDE Next.js (server-only).
//
// WHY THIS EXISTS
// ---------------
// The browser must NEVER see your real OPENAI_API_KEY. If it did, anyone could
// open DevTools, copy it, and spend your money. Instead the browser asks THIS
// route for a short-lived "ephemeral" key that starts with "ek_...". That token
// can only open one Realtime session and expires in about a minute, so leaking
// it is harmless. This is exactly the job Module 06 (FastAPI) does; here we do
// the same thing in a Next.js Route Handler so the capstone runs as ONE app.
//
// This file runs on the SERVER only. Next.js never ships Route Handlers to the
// browser, so reading process.env.OPENAI_API_KEY here is safe.
// ---------------------------------------------------------------------------

import { NextResponse } from "next/server";

// Force the Node.js runtime (not the Edge runtime). We want a plain server
// environment with access to process.env and normal fetch. This also makes the
// route dynamic, so it never gets cached and always mints a FRESH token.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// The canonical voice-assistant model from docs/API_FACTS.md.
// (DataCamp's older name was "gpt-realtime-2"; "-2.1" is the GA canonical id.)
const MODEL = "gpt-realtime-2.1";
const TRANSCRIBE_MODEL = "gpt-realtime-whisper";

// GET /api/token  ->  { value: "ek_...", expires_at: <unix seconds> }
export async function GET(request: Request) {
  let apiKey = (process.env.OPENAI_API_KEY ?? "").trim();
  while (apiKey.startsWith("OPENAI_API_KEY=")) {
    apiKey = apiKey.slice("OPENAI_API_KEY=".length).trim();
  }
  const mode = new URL(request.url).searchParams.get("mode") ?? "assist";

  // Fail loudly and clearly if the key is missing, so students know to set it.
  if (!apiKey?.trim().startsWith("sk-")) {
    return NextResponse.json(
      {
        error:
          "OPENAI_API_KEY is missing or malformed. Set a paid-tier OpenAI key beginning with 'sk-'.",
      },
      { status: 500 },
    );
  }

  if (mode !== "assist" && mode !== "transcribe") {
    return NextResponse.json(
      { error: "mode must be 'assist' or 'transcribe'" },
      { status: 400 },
    );
  }

  const session =
    mode === "transcribe"
      ? {
          type: "transcription",
          audio: {
            input: {
              format: { type: "audio/pcm", rate: 24000 },
              transcription: { model: TRANSCRIBE_MODEL, delay: "low" },
            },
          },
        }
      : {
          type: "realtime",
          model: MODEL,
          audio: { output: { voice: "marin" } },
        };

  // Ask OpenAI to mint an ephemeral client secret scoped to the requested mode.
  // Endpoint + body shape are verified in docs/API_FACTS.md section 5.
  const resp = await fetch("https://api.openai.com/v1/realtime/client_secrets", {
    method: "POST",
    headers: {
      // The REAL key is used here, on the server, and only here.
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      // NOTE: at GA there is NO "OpenAI-Beta: realtime=v1" header. Do not add it.
    },
    body: JSON.stringify({ session }),
  });

  // If OpenAI rejected us (bad key, free tier, rate limit), forward the reason.
  if (!resp.ok) {
    const detail = await resp.text();
    return NextResponse.json(
      { error: "Failed to mint ephemeral token", detail },
      { status: resp.status },
    );
  }

  // The response JSON puts the ephemeral key at data.value (see API_FACTS §5).
  const data = await resp.json();
  const value: string | undefined = data?.value;
  const expiresAt: number | undefined = data?.expires_at;

  if (!value) {
    return NextResponse.json(
      { error: "Token response missing 'value'", raw: data },
      { status: 502 },
    );
  }

  // Hand the browser ONLY the ephemeral token (and when it expires). The real
  // key never leaves this server function.
  return NextResponse.json({ value, expires_at: expiresAt });
}
