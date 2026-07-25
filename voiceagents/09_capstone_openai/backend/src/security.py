"""
Voice Agents - lightweight access controls for the paid backend routes.

WHY THIS FILE EXISTS
--------------------
Three of this server's routes spend real money on your OpenAI account: /token
mints Realtime credentials, /web-search runs a hosted web search, and /translate
opens two paid upstream sockets. On your own laptop that is fine. But the moment
this backend is reachable by anyone else (a shared network, a deployment), an
unauthenticated caller could hammer those routes and run up your bill or exhaust
your quota. CORS does NOT stop that: CORS only restrains BROWSER JavaScript from
other web origins; it does nothing against a script using curl.

So this module adds two small, always-safe guards and one optional one:

  1) RATE LIMIT (always on): a simple in-process fixed-window counter caps how many
     paid requests one caller (by client IP) can make per minute. This protects
     even the local, no-auth case from a runaway loop.
  2) ORIGIN CHECK for the WebSocket (always on): /translate only accepts upgrades
     whose Origin header is a localhost origin or one you explicitly allow. A
     browser cannot forge Origin, so this blocks cross-site socket abuse.
  3) SHARED CALLER TOKEN (optional): if you set CAPSTONE_API_TOKEN, every paid
     route also requires that token (HTTP: "Authorization: Bearer <token>"; the
     /translate WebSocket: a {"token": "..."} field in the first message). Leave
     it UNSET for local class work and the routes stay open on localhost.

This is deliberately simple and honest: a shared token is a light gate, not real
per-user authentication (that needs a login system, which is out of scope for a
voice course). It is enough to keep a deployed demo from being an open wallet.

Everything here degrades gracefully: with no env vars set, only the generous rate
limit and the localhost origin check apply, so the tutorial runs with zero setup.
"""

from __future__ import annotations

import os
import time
from collections import deque

from fastapi import HTTPException, Request, WebSocket


# ---------------------------------------------------------------------------
# CONFIG (all optional; sensible course defaults)
# ---------------------------------------------------------------------------

def _caller_token() -> str:
    """The optional shared caller token. Empty string means 'auth disabled'."""
    return (os.environ.get("CAPSTONE_API_TOKEN") or "").strip()


def _rate_limit_per_minute() -> int:
    """Max paid requests per caller IP per minute. Generous default for a class."""
    raw = (os.environ.get("CAPSTONE_RATE_LIMIT_PER_MIN") or "").strip()
    try:
        value = int(raw)
        return value if value > 0 else 60
    except ValueError:
        return 60


def _extra_allowed_origins() -> set[str]:
    """Comma-separated exact origins allowed to open the /translate WebSocket,
    in addition to localhost. e.g. 'https://myapp.example.com'."""
    raw = os.environ.get("CAPSTONE_ALLOWED_ORIGINS") or ""
    return {o.strip() for o in raw.split(",") if o.strip()}


def auth_enabled() -> bool:
    """True when a shared caller token is configured (so routes require it)."""
    return bool(_caller_token())


# ---------------------------------------------------------------------------
# RATE LIMITER: a tiny in-process fixed-window counter, keyed by caller IP.
# ---------------------------------------------------------------------------
# We keep, per caller, the timestamps of recent requests and drop any older than
# 60 seconds. If what remains reaches the per-minute cap, we reject. This is not a
# distributed limiter (one process only), which is exactly right for a single-node
# course backend and keeps the teaching code readable.
_WINDOW_SECONDS = 60.0
_request_times: dict[str, deque[float]] = {}


def _caller_ip_from_request(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _caller_ip_from_ws(ws: WebSocket) -> str:
    return ws.client.host if ws.client else "unknown"


def _check_rate(caller_ip: str, now: float) -> bool:
    """Record one request for caller_ip and return True if still within the cap."""
    cap = _rate_limit_per_minute()
    bucket = _request_times.setdefault(caller_ip, deque())
    cutoff = now - _WINDOW_SECONDS
    while bucket and bucket[0] < cutoff:
        bucket.popleft()
    bucket.append(now)
    return len(bucket) <= cap


# ---------------------------------------------------------------------------
# HTTP GUARD: call at the top of each paid HTTP route.
# ---------------------------------------------------------------------------

def _bearer_token(request: Request) -> str:
    header = request.headers.get("Authorization") or ""
    prefix = "Bearer "
    return header[len(prefix):].strip() if header.startswith(prefix) else ""


def guard_http(request: Request) -> None:
    """Enforce rate limit (always) and the shared token (if configured).

    Raises HTTPException(401) on a bad/missing token and HTTPException(429) when
    the caller exceeds the per-minute cap.
    """
    # 1) Shared-token check (only when CAPSTONE_API_TOKEN is set).
    expected = _caller_token()
    if expected and _bearer_token(request) != expected:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid caller token (Authorization: Bearer ...).",
        )

    # 2) Rate limit (always on). time.monotonic() is immune to clock changes.
    if not _check_rate(_caller_ip_from_request(request), time.monotonic()):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded ({_rate_limit_per_minute()} requests/minute). "
                "Slow down, or raise CAPSTONE_RATE_LIMIT_PER_MIN."
            ),
        )


# ---------------------------------------------------------------------------
# WEBSOCKET GUARD: call right after ws.accept() for /translate.
# ---------------------------------------------------------------------------

def _origin_allowed(origin: str) -> bool:
    """True if there is no Origin (non-browser tools), it is localhost, or it is
    in the explicitly allowed set. Browsers always send a truthful Origin."""
    if not origin:
        return True  # curl / server-side clients send no Origin; token still gates them
    if origin in _extra_allowed_origins():
        return True
    # Allow http(s)://localhost[:port] and 127.0.0.1[:port].
    for host in ("localhost", "127.0.0.1"):
        if origin == f"http://{host}" or origin.startswith(f"http://{host}:"):
            return True
        if origin == f"https://{host}" or origin.startswith(f"https://{host}:"):
            return True
    return False


def ws_reject_reason(ws: WebSocket, first_message: dict | None) -> str | None:
    """Return a human error string if this WebSocket should be rejected, else None.

    Checks (in order): Origin allow-list, rate limit, and the shared token (read
    from the first message's optional 'token' field). The caller sends the reason
    to the browser as an {"type":"error"} message and closes.
    """
    origin = ws.headers.get("origin") or ""
    if not _origin_allowed(origin):
        return f"Origin not allowed: {origin or '(none)'}"

    if not _check_rate(_caller_ip_from_ws(ws), time.monotonic()):
        return (
            f"Rate limit exceeded ({_rate_limit_per_minute()} requests/minute). "
            "Slow down, or raise CAPSTONE_RATE_LIMIT_PER_MIN."
        )

    expected = _caller_token()
    if expected:
        provided = str((first_message or {}).get("token") or "").strip()
        if provided != expected:
            return "Missing or invalid caller token in the first message."

    return None
