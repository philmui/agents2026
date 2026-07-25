// =============================================================================
// lib/backend.ts  -  where the browser finds the FastAPI backend.
// =============================================================================
//
// The capstone runs as TWO local processes:
//   - the Next.js UI      on  http://localhost:3000  (this app)
//   - the FastAPI backend on  http://localhost:8000  (backend/src/main.py)
//
// The browser needs three things from the backend:
//   1) an HTTP base URL, to fetch an ephemeral token  ->  GET  {BASE}/token
//   2) that HTTP URL, to run Assist web search         ->  POST {BASE}/web-search
//   3) a WebSocket base URL, for live translation      ->  WS   {WS}/translate
//
// Both are derived from ONE environment variable, NEXT_PUBLIC_BACKEND_URL, so you
// configure the backend location in a single place (.env.local). Anything that
// starts with NEXT_PUBLIC_ is readable in the browser, which is fine here: the
// backend's ADDRESS is not a secret (only the API key is, and that stays on the
// backend). If the variable is unset we default to the local dev backend.
// =============================================================================

// The backend's HTTP origin. Trailing slashes are trimmed so we can safely append
// paths like "/token". Default: the local FastAPI dev server.
export const BACKEND_URL = (
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"
).replace(/\/+$/, "");

// The same origin as a WebSocket URL: http -> ws, https -> wss. We reuse the HTTP
// origin so you only ever set NEXT_PUBLIC_BACKEND_URL, never a second ws:// value.
export const BACKEND_WS_URL = BACKEND_URL.replace(/^http/, "ws");
