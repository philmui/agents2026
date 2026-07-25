"""
Tests for the OpenAI Agents SDK web-search route.

The route runs an OpenAI Agents SDK `Agent` via `Runner.run(...)`. So the seam we
patch is `main.Runner.run`: we replace it with a fake coroutine that returns an
object exposing `.final_output`, exactly like a real RunResult. That lets us
assert the route's behavior (validation, output extraction, error handling)
WITHOUT any network access or real key.

Telemetry is a no-op in tests: no Langfuse keys are set, so `telemetry.enabled`
is False and `telemetry.trace(...)` / `telemetry.flush()` do nothing.
"""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src import main


class FakeRunResult:
    """Minimal stand-in for the SDK's RunResult: only `.final_output` is read."""

    def __init__(self, final_output):
        self.final_output = final_output


class WebSearchTests(unittest.TestCase):
    def test_search_runs_agent_and_returns_final_output(self):
        captured = {}

        async def fake_run(agent, query, *args, **kwargs):
            # Record what the route handed the Runner so we can assert on it.
            captured["agent"] = agent
            captured["query"] = query
            return FakeRunResult("OpenAI announced a new product today.")

        with (
            patch.object(main, "OPENAI_API_KEY", "sk-test-only"),
            patch.object(main.Runner, "run", fake_run),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/web-search",
                json={"query": "  latest OpenAI announcement  "},
            )

        self.assertEqual(
            response.json(),
            {"answer": "OpenAI announced a new product today."},
        )
        # The query is stripped before being passed to the agent.
        self.assertEqual(captured["query"], "latest OpenAI announcement")
        # The agent handed to Runner.run is our configured web-search agent, whose
        # single tool is the hosted WebSearchTool.
        agent = captured["agent"]
        self.assertEqual(agent.name, "Web Search Delegate")
        self.assertEqual(agent.model, main.WEB_SEARCH_MODEL)
        self.assertEqual(len(agent.tools), 1)
        self.assertIsInstance(agent.tools[0], main.WebSearchTool)

    def test_search_rejects_blank_query_without_running_agent(self):
        ran = {"called": False}

        async def fake_run(agent, query, *args, **kwargs):
            ran["called"] = True
            return FakeRunResult("should not happen")

        with (
            patch.object(main, "OPENAI_API_KEY", "sk-test-only"),
            patch.object(main.Runner, "run", fake_run),
            TestClient(main.app) as client,
        ):
            response = client.post("/web-search", json={"query": "   "})

        self.assertEqual(response.status_code, 422)
        self.assertFalse(ran["called"])

    def test_search_rejects_missing_api_key(self):
        with (
            patch.object(main, "OPENAI_API_KEY", ""),
            TestClient(main.app) as client,
        ):
            response = client.post("/web-search", json={"query": "current news"})

        self.assertEqual(response.status_code, 500)
        self.assertIn("OPENAI_API_KEY", response.json()["detail"])

    def test_search_reports_missing_answer_text(self):
        async def fake_run(agent, query, *args, **kwargs):
            return FakeRunResult("")  # agent produced no text

        with (
            patch.object(main, "OPENAI_API_KEY", "sk-test-only"),
            patch.object(main.Runner, "run", fake_run),
            TestClient(main.app) as client,
        ):
            response = client.post("/web-search", json={"query": "current news"})

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "OpenAI Agents web search returned no answer text",
        )

    def test_search_maps_agent_failure_to_502(self):
        async def fake_run(agent, query, *args, **kwargs):
            raise RuntimeError("model exploded")

        with (
            patch.object(main, "OPENAI_API_KEY", "sk-test-only"),
            patch.object(main.Runner, "run", fake_run),
            TestClient(main.app) as client,
        ):
            response = client.post("/web-search", json={"query": "current news"})

        self.assertEqual(response.status_code, 502)
        self.assertIn(
            "OpenAI Agents web search failed", response.json()["detail"]
        )


if __name__ == "__main__":
    unittest.main()
