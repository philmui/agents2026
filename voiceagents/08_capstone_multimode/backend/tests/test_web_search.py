import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src import main


class FakeResponse:
    def __init__(self, data: dict[str, object], status_code: int = 200):
        self._data = data
        self.status_code = status_code
        self.text = "fake upstream response"

    def json(self) -> dict[str, object]:
        return self._data


class FakeAsyncClient:
    response = FakeResponse({})
    request: dict[str, object] = {}

    def __init__(self, *, timeout: float):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]):
        type(self).request = {
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": self.timeout,
        }
        return type(self).response


class WebSearchTests(unittest.TestCase):
    def test_search_forces_hosted_web_search_and_returns_message_text(self):
        FakeAsyncClient.response = FakeResponse(
            {
                "output": [
                    {
                        "type": "web_search_call",
                        "status": "completed",
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "OpenAI announced a new product today.",
                            }
                        ],
                    },
                ]
            }
        )

        with (
            patch.object(main, "OPENAI_API_KEY", "sk-test-only"),
            patch.object(main, "OPENAI_SAFETY_IDENTIFIER", None),
            patch.object(main.httpx, "AsyncClient", FakeAsyncClient),
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
        self.assertEqual(FakeAsyncClient.request["url"], main.RESPONSES_URL)
        self.assertEqual(FakeAsyncClient.request["timeout"], 30.0)
        self.assertEqual(
            FakeAsyncClient.request["json"],
            {
                "model": "gpt-5.6",
                "reasoning": {"effort": "low"},
                "tools": [{"type": "web_search"}],
                "tool_choice": "required",
                "instructions": (
                    "Use live web search to answer the query. Return a concise, "
                    "factual, plain-text answer suitable for a voice assistant. "
                    "Mention the names of important sources, but do not use "
                    "Markdown or read out long URLs."
                ),
                "input": "latest OpenAI announcement",
            },
        )

    def test_search_rejects_blank_query_without_calling_openai(self):
        FakeAsyncClient.request = {}
        with (
            patch.object(main, "OPENAI_API_KEY", "sk-test-only"),
            patch.object(main.httpx, "AsyncClient", FakeAsyncClient),
            TestClient(main.app) as client,
        ):
            response = client.post("/web-search", json={"query": "   "})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(FakeAsyncClient.request, {})

    def test_search_reports_missing_assistant_text(self):
        FakeAsyncClient.response = FakeResponse(
            {"output": [{"type": "web_search_call", "status": "completed"}]}
        )
        with (
            patch.object(main, "OPENAI_API_KEY", "sk-test-only"),
            patch.object(main.httpx, "AsyncClient", FakeAsyncClient),
            TestClient(main.app) as client,
        ):
            response = client.post("/web-search", json={"query": "current news"})

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "OpenAI web search response contained no assistant text",
        )


if __name__ == "__main__":
    unittest.main()
