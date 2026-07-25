// =============================================================================
// lib/token.ts  —  Fetch a short-lived ephemeral token from OUR backend.
// =============================================================================
//
// WHY THIS FILE EXISTS
// --------------------
// Your real OpenAI API key is a long-lived secret. If it ever reached the
// browser, anyone could open DevTools, copy it, and spend your money. So the
// browser is NEVER trusted with it.
//
// Instead, the module-06 Python backend holds the real key on the server and
// exchanges it for a short-lived "ephemeral" token that starts with "ek_".
// That token only works for the Realtime API, expires in about a minute, and
// is safe to hand to the browser. This file is the browser's request for that
// token.
//
// This one helper is reused by BOTH connection paths in this app:
//   - the official SDK path (lib/useRealtime.ts), and
//   - the raw WebRTC path (lib/rawWebrtc.ts).
// =============================================================================

// Read the backend URL from the environment. Anything named NEXT_PUBLIC_* is
// visible in the browser (that is fine here: a URL is not a secret). We default
// to the module-06 dev server if the variable is missing.
const TOKEN_ENDPOINT =
  process.env.NEXT_PUBLIC_TOKEN_ENDPOINT ?? "http://localhost:8000/token";

function nestedValue(value: unknown): unknown {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return undefined;
  }
  return (value as Record<string, unknown>).value;
}

/**
 * Ask our backend for a fresh ephemeral token.
 *
 * Returns the raw "ek_..." string. Throws a friendly Error if the backend is
 * unreachable (most commonly: module 06 is not running) or replies in an
 * unexpected shape.
 */
export async function fetchEphemeralToken(): Promise<string> {
  let res: Response;
  try {
    // A plain HTTP GET to our own backend. No secret leaves the browser here;
    // we are just asking the server to do the privileged work for us.
    res = await fetch(TOKEN_ENDPOINT, { method: "GET" });
  } catch (networkError) {
    // fetch() only throws for network-level failures (server down, CORS, DNS).
    throw new Error(
      `Could not reach the token backend at ${TOKEN_ENDPOINT}. ` +
        `Is module 06 (the FastAPI backend) running? Original error: ${networkError}`
    );
  }

  if (!res.ok) {
    // The server answered, but with an error status (e.g. 500 if its own
    // OpenAI key is missing). Surface the body to help you debug.
    const body = await res.text().catch(() => "");
    throw new Error(
      `Token backend returned HTTP ${res.status}. ${body}`.trim()
    );
  }

  // Parse the JSON body. We accept a few shapes so this frontend works no
  // matter exactly how module 06 formats its response:
  //   1) { "value": "ek_..." }                  (backend already unwrapped it)
  //   2) { "client_secret": { "value": "ek_" }} (raw OpenAI client_secrets echo)
  //   3) { "value": "ek_...", ... } nested under a "data" or "session" key
  const parsed: unknown = await res.json();
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("Token backend returned a non-object JSON response.");
  }

  const json = parsed as Record<string, unknown>;
  const token =
    json.value ??
    nestedValue(json.client_secret) ??
    nestedValue(json.data) ??
    nestedValue(json.session) ??
    json.client_secret;

  if (typeof token !== "string" || !token.startsWith("ek_")) {
    throw new Error(
      `Token backend responded, but no "ek_..." value was found. ` +
        `Got: ${JSON.stringify(json)}`
    );
  }

  return token;
}

/** Exported so the UI can show users which backend it is calling. */
export const tokenEndpoint = TOKEN_ENDPOINT;
