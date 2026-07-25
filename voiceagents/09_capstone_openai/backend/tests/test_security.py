"""
Tests for the lightweight access controls (security.py) wired into main.py.

These pin the three guards: the optional shared caller token, the always-on
per-IP rate limit, and (for the WebSocket) the Origin allow-list. They must NOT
require any network or real keys, so /web-search patches Runner.run.
"""

import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src import main
from src import security


class FakeRunResult:
    def __init__(self, final_output):
        self.final_output = final_output


async def _fake_run(agent, query, *args, **kwargs):
    return FakeRunResult("ok answer")


def _reset_rate_limiter():
    """Clear the in-process rate-limiter state between tests."""
    security._request_times.clear()


class HealthAuthFlagTests(unittest.TestCase):
    def test_health_reports_auth_disabled_by_default(self):
        _reset_rate_limiter()
        with mock.patch.dict(os.environ, {"CAPSTONE_API_TOKEN": ""}, clear=False):
            with TestClient(main.app) as client:
                body = client.get("/health").json()
        self.assertIn("auth", body)
        self.assertFalse(body["auth"])

    def test_health_reports_auth_enabled_when_token_set(self):
        _reset_rate_limiter()
        with mock.patch.dict(os.environ, {"CAPSTONE_API_TOKEN": "s3cret"}, clear=False):
            with TestClient(main.app) as client:
                body = client.get("/health").json()
        self.assertTrue(body["auth"])


class CallerTokenTests(unittest.TestCase):
    def test_web_search_requires_token_when_configured(self):
        _reset_rate_limiter()
        with (
            mock.patch.dict(os.environ, {"CAPSTONE_API_TOKEN": "s3cret"}, clear=False),
            mock.patch.object(main, "OPENAI_API_KEY", "sk-test-only"),
            mock.patch.object(main.Runner, "run", _fake_run),
            TestClient(main.app) as client,
        ):
            # No token -> 401, and the agent must not run.
            unauthorized = client.post("/web-search", json={"query": "hi"})
            # Correct token -> 200.
            ok = client.post(
                "/web-search",
                json={"query": "hi"},
                headers={"Authorization": "Bearer s3cret"},
            )
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json(), {"answer": "ok answer"})

    def test_token_route_open_when_no_token_configured(self):
        _reset_rate_limiter()
        with (
            mock.patch.dict(os.environ, {"CAPSTONE_API_TOKEN": ""}, clear=False),
            mock.patch.object(main, "OPENAI_API_KEY", ""),  # forces the 500 key error, past the guard
            TestClient(main.app) as client,
        ):
            # No caller token needed; we reach the missing-key check (500), NOT 401.
            res = client.get("/token?mode=transcribe")
        self.assertEqual(res.status_code, 500)


class RateLimitTests(unittest.TestCase):
    def test_rate_limit_returns_429_after_cap(self):
        _reset_rate_limiter()
        env = {"CAPSTONE_API_TOKEN": "", "CAPSTONE_RATE_LIMIT_PER_MIN": "3"}
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.object(main, "OPENAI_API_KEY", ""),  # 500 past the guard when allowed
            TestClient(main.app) as client,
        ):
            statuses = [client.get("/token").status_code for _ in range(5)]
        # First 3 pass the guard (then hit the missing-key 500); the 4th+ are limited.
        self.assertEqual(statuses[:3], [500, 500, 500])
        self.assertEqual(statuses[3], 429)
        self.assertEqual(statuses[4], 429)


class WebSocketGuardTests(unittest.TestCase):
    def test_translate_rejects_bad_origin(self):
        _reset_rate_limiter()
        with (
            mock.patch.dict(os.environ, {"CAPSTONE_API_TOKEN": ""}, clear=False),
            mock.patch.object(main, "OPENAI_API_KEY", "sk-test-only"),
            TestClient(main.app) as client,
        ):
            with client.websocket_connect(
                "/translate", headers={"origin": "https://evil.example.com"}
            ) as ws:
                ws.send_json({"type": "start", "language": "es"})  # guard reads this
                msg = ws.receive_json()
        self.assertEqual(msg["type"], "error")
        self.assertIn("Origin not allowed", msg["message"])

    def test_translate_requires_token_when_configured(self):
        _reset_rate_limiter()
        with (
            mock.patch.dict(os.environ, {"CAPSTONE_API_TOKEN": "s3cret"}, clear=False),
            mock.patch.object(main, "OPENAI_API_KEY", "sk-test-only"),
            TestClient(main.app) as client,
        ):
            with client.websocket_connect("/translate") as ws:
                ws.send_json({"type": "start", "language": "es"})  # no token
                msg = ws.receive_json()
        self.assertEqual(msg["type"], "error")
        self.assertIn("token", msg["message"].lower())


if __name__ == "__main__":
    unittest.main()
