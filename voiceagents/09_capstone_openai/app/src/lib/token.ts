// =============================================================================
// lib/token.ts  -  getEphemeralToken(): the browser asks for a short-lived key.
// =============================================================================
//
// TRANSCRIBE and ASSIST mode both connect to OpenAI over WebRTC, which needs a
// short-lived "ek_..." token (NOT your real key). This helper fetches one. The
// browser NEVER sees your real OpenAI key: whichever backend answers holds it and
// returns only the ephemeral token.
//
// The token comes from the FastAPI backend by default:
//     GET  {NEXT_PUBLIC_BACKEND_URL}/token         (default http://localhost:8000/token)
// That is the same backend that also powers Translate mode (WS /translate), so it
// is already running. If you would rather use this Next.js app's own built-in
// route (app/api/token/route.ts) and deploy the UI alone, set
//     NEXT_PUBLIC_TOKEN_ENDPOINT=/api/token
// and this helper will call that instead.
//
// NEXT_PUBLIC_ variables are readable in the browser. That is fine: an endpoint
// URL is not a secret. Only the API key is, and it stays on the backend.
// =============================================================================

import { BACKEND_URL, authHeader } from "@/lib/backend";

export type RealtimeMode = "assist" | "transcribe";

export async function getEphemeralToken(
  mode: RealtimeMode = "assist",
): Promise<string> {
  // Prefer an explicit override, else the FastAPI backend's /token route.
  const baseEndpoint =
    process.env.NEXT_PUBLIC_TOKEN_ENDPOINT || `${BACKEND_URL}/token`;
  const endpoint = new URL(baseEndpoint, window.location.origin);
  endpoint.searchParams.set("mode", mode);

  // GET works for both backends (each exposes a GET alias alongside POST).
  // Send the optional shared caller token if one is configured (see backend.ts).
  let res: Response;
  try {
    res = await fetch(endpoint, { method: "GET", headers: authHeader() });
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Could not reach the token server at ${endpoint.origin}: ${detail}`,
    );
  }
  if (!res.ok) {
    // Surface the backend's error text so the UI can show what went wrong
    // (e.g. "OPENAI_API_KEY is not set", a rate limit, or a free-tier key).
    const text = await res.text();
    throw new Error(`Token request failed (${res.status}): ${text}`);
  }

  const data: unknown = await res.json();
  // Both backends return the ephemeral key under "value" (see API_FACTS.md).
  if (
    typeof data !== "object" ||
    data === null ||
    !("value" in data) ||
    typeof data.value !== "string"
  ) {
    throw new Error("Token response did not contain a 'value' field.");
  }
  return data.value;
}
