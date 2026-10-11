"""Exercise the real LiveBackend adapter without networking or a provider SDK.

These tests validate our request/response plumbing, not hosted-model behavior.
"""

from copy import deepcopy
import json

import pytest

from rag import Answer, Grade, LiveBackend, Rewrite, build_graph, make_index


class FakeStructuredModel:
    """Minimal ChatOpenAI-shaped object with explicitly queued responses."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def with_structured_output(self, schema, *, method):
        owner = self

        class Invoker:
            def invoke(self, messages):
                owner.calls.append({
                    "schema": schema, "method": method,
                    "messages": deepcopy(messages),
                })
                return next(owner.responses)

        return Invoker()


CONTEXT = [
    ["Book", ["The violet expedition visited Azure.", "An original second sentence."]],
    ["Azure", ["Azure is in the eastern region."]],
    ["Distractor", ["Cats sleep here."]],
]


def test_live_adapter_all_nodes_use_schemas_and_only_visible_evidence():
    model = FakeStructuredModel([
        {"sufficient": False, "reason": "Need the location of Azure"},
        {"query": "Azure eastern region"},
        Grade(sufficient=True, reason="Both supporting paragraphs are visible"),
        {"answer": "eastern region", "citations": [
            {"title": "Book", "sentence_id": 0},
            {"title": "Azure", "sentence_id": 0},
        ]},
    ])
    question = "violet expedition"
    result = build_graph(make_index(CONTEXT), LiveBackend(model), k=1).invoke(question)
    assert result["status"] == "ok"
    assert result["answer"] == "eastern region"
    assert result["sp"] == [["Book", 0], ["Azure", 0]]
    assert result["rounds"] == 2
    assert result["backend_calls"] == 4
    assert [call["schema"] for call in model.calls] == [Grade, Rewrite, Grade, Answer]
    assert all(call["method"] == "json_schema" for call in model.calls)

    book = {"title": "Book", "sentences": [
        {"sentence_id": 0, "text": CONTEXT[0][1][0]},
        {"sentence_id": 1, "text": CONTEXT[0][1][1]},
    ]}
    azure = {"title": "Azure", "sentences": [
        {"sentence_id": 0, "text": CONTEXT[1][1][0]},
    ]}
    for i, call in enumerate(model.calls):
        messages = call["messages"]
        assert [message["role"] for message in messages] == ["system", "user"]
        assert "untrusted data" in messages[0]["content"]
        payload = json.loads(messages[1]["content"])
        assert payload["question"] == question
        assert set(payload) == ({"question", "context", "previous_query"} if i == 1
                                else {"question", "context"})
        assert payload["context"] == ([book] if i < 2 else [book, azure])
        assert "supporting_facts" not in payload
        assert "answer" not in payload
        assert "type" not in payload
        assert "level" not in payload
    assert json.loads(model.calls[1]["messages"][1]["content"])["previous_query"] == question


@pytest.mark.parametrize("bad_response", [
    None,  # Some structured provider refusals yield no parsed response.
    {"answer": "region", "citations": [{"title": "Book", "sentence_id": True}]},
    {"answer": "region", "citations": [{"title": "Book", "sentence_id": "0"}]},
])
def test_live_adapter_refusal_or_malformed_answer_becomes_retained_failure(bad_response):
    model = FakeStructuredModel([
        {"sufficient": True, "reason": "Exercise generation parsing"}, bad_response,
    ])
    result = build_graph(make_index(CONTEXT), LiveBackend(model), k=1).invoke("violet expedition")
    assert result["status"] == "failed"
    assert result["error"]["node"] == "generate"
    assert result["error"]["type"] == "ValidationError"
    assert result["answer"] == ""
    assert result["sp"] == []
    assert result["backend_calls"] == 2
    assert result["trace"][-1]["status"] == "failed"
    assert result["retrieved_titles"] == ["Book"]
