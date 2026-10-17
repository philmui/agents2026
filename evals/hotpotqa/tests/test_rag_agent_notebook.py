"""Exercise notebook model calls through the real SDK with mocked HTTP only."""

import ast
from collections import deque
import json
from pathlib import Path

import httpx
import nbformat
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import BadRequestError
from pydantic import ValidationError


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "rag_agent.ipynb"
ENDPOINT_ERROR = (
    "Function tools with reasoning_effort are not supported for gpt-5.6-luna "
    "in /v1/chat/completions. To use function tools, use /v1/responses "
    "or set reasoning_effort to 'none'."
)


def message_output(text):
    return [{
        "id": "msg_test", "type": "message", "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }]


@pytest.fixture
def notebook_runtime(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    requests = []
    outputs = deque()
    tool_queries = []

    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Search and return information about Lilian Weng blog posts."""
        tool_queries.append(query)
        return "Retrieved evidence."

    def handle_request(request):
        requests.append((request.url.path, json.loads(request.content)))
        if request.url.path != "/v1/responses":
            return httpx.Response(400, json={"error": {
                "message": ENDPOINT_ERROR, "type": "invalid_request_error",
                "param": "reasoning_effort", "code": None,
            }})
        return httpx.Response(200, json={
            "id": "resp_test", "created_at": 0, "model": "gpt-5.6-luna",
            "object": "response", "status": "completed", "error": None,
            "incomplete_details": None, "instructions": None, "metadata": {},
            "parallel_tool_calls": True, "temperature": None,
            "tool_choice": "auto", "tools": [], "top_p": None,
            "output": outputs.popleft(),
        })

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        def create_model(**kwargs):
            return ChatOpenAI(
                **kwargs, api_key="test-not-a-real-key",
                base_url="https://api.openai.com/v1", http_client=client,
            )

        monkeypatch.setattr("langchain_openai.ChatOpenAI", create_model)
        namespace = {"retriever_tool": retrieve_blog_posts}
        for name in ["generate_query_or_respond", "grade_documents"]:
            cell = next(
                cell for cell in notebook.cells
                if cell.cell_type == "code" and f"def {name}(" in cell.source
            )
            exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)
        yield namespace, requests, outputs, tool_queries, client


def test_legacy_endpoint_reproduces_reported_error(notebook_runtime):
    namespace, requests, outputs, tool_queries, client = notebook_runtime
    legacy_model = ChatOpenAI(
        model="gpt-5.6-luna", api_key="test-not-a-real-key",
        base_url="https://api.openai.com/v1", http_client=client,
        use_responses_api=False, reasoning_effort="low", max_retries=0,
    )
    with pytest.raises(BadRequestError, match="use /v1/responses"):
        legacy_model.bind_tools([namespace["retriever_tool"]]).invoke("Search blogs")
    assert requests[-1][0] == "/v1/chat/completions"


def test_tool_request_uses_responses_without_executing_retrieval(notebook_runtime):
    namespace, requests, outputs, tool_queries, client = notebook_runtime
    outputs.append([{
        "type": "function_call", "id": "fc_test", "call_id": "call_test",
        "name": "retrieve_blog_posts", "arguments": '{"query":"reward hacking"}',
        "status": "completed",
    }])
    result = namespace["generate_query_or_respond"]({
        "messages": [HumanMessage(content="What are types of reward hacking?")],
    })
    assert result["messages"][-1].tool_calls == [{
        "name": "retrieve_blog_posts", "args": {"query": "reward hacking"},
        "id": "call_test", "type": "tool_call",
    }]
    endpoint, payload = requests[-1]
    assert endpoint == "/v1/responses"
    assert payload["model"] == "gpt-5.6-luna"
    assert payload["tools"][0]["name"] == "retrieve_blog_posts"
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert tool_queries == []


def test_greeting_example_preserves_response_after_printing(notebook_runtime, capsys):
    namespace, requests, outputs, tool_queries, client = notebook_runtime
    outputs.append(message_output("Hello!"))
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    greeting = next(
        cell for cell in notebook.cells
        if cell.cell_type == "code" and 'HumanMessage(content="hello!")' in cell.source
    )
    exec(compile(greeting.source, str(NOTEBOOK), "exec"), namespace)
    assert isinstance(namespace["response"]["messages"][-1], AIMessage)
    assert "Hello!" in namespace["response"]["messages"][-1].text
    assert "Hello!" in capsys.readouterr().out
    assert requests[-1][0] == "/v1/responses"
    assert tool_queries == []


def test_tool_result_is_forwarded_on_followup_call(notebook_runtime):
    namespace, requests, outputs, tool_queries, client = notebook_runtime
    outputs.append(message_output("Here is the grounded answer."))
    namespace["generate_query_or_respond"]({"messages": [
        HumanMessage(content="Explain reward hacking."),
        AIMessage(content="", tool_calls=[{
            "name": "retrieve_blog_posts", "args": {"query": "reward hacking"},
            "id": "call_test", "type": "tool_call",
        }]),
        ToolMessage(content="Retrieved evidence.", tool_call_id="call_test"),
    ]})
    assert {
        "type": "function_call_output", "call_id": "call_test",
        "output": "Retrieved evidence.",
    } in requests[-1][1]["input"]


@pytest.mark.parametrize("score, expected", [
    ("yes", "generate_answer"), ("no", "rewrite_question"),
])
def test_grader_uses_strict_json_schema_on_responses(notebook_runtime, score, expected):
    namespace, requests, outputs, tool_queries, client = notebook_runtime
    outputs.append(message_output(json.dumps({"binary_score": score})))
    route = namespace["grade_documents"]({"messages": [
        HumanMessage(content="Explain reward hacking."),
        ToolMessage(content="Retrieved evidence.", tool_call_id="call_test"),
    ]})
    assert route == expected
    endpoint, payload = requests[-1]
    assert endpoint == "/v1/responses"
    assert "temperature" not in payload
    schema = payload["text"]["format"]
    assert schema["type"] == "json_schema"
    assert schema["strict"] is True
    assert schema["schema"]["properties"]["binary_score"]["enum"] == ["yes", "no"]


def test_grader_rejects_invalid_score_instead_of_rewriting(notebook_runtime):
    namespace, requests, outputs, tool_queries, client = notebook_runtime
    outputs.append(message_output('{"binary_score":"maybe"}'))
    with pytest.raises(ValidationError):
        namespace["grade_documents"]({"messages": [
            HumanMessage(content="Explain reward hacking."),
            ToolMessage(content="Retrieved evidence.", tool_call_id="call_test"),
        ]})


def test_all_notebook_code_compiles_and_has_no_stale_errors():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert not any(output.output_type == "error" for output in cell.outputs)
