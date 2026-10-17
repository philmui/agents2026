"""Offline behavior checks; no keys or paid requests are used.

Run: python -m pip install pytest
     python -m pytest test_beam_agent.py -q
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from beam_agent import SYSTEM, build_agent, load_keys, run_turn


def assistant(content=None, calls=None, reasoning="Returned Beam reasoning"):
    message = {"role": "assistant", "reasoning_content": reasoning}
    if content is not None:
        message["content"] = content
    if calls:
        message["tool_calls"] = calls
    return message


def tool_call(arguments='{"query":"NASA Webb launch"}', name="web_search", id="search-1"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


class ScriptedClient:
    """SDK-shaped test double. Scripts are fixtures, not measured model outputs."""

    def __init__(self, messages, finish=None):
        self.messages = iter(messages)
        self.requests = []
        self.finish = finish
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        message = next(self.messages)
        if isinstance(message, Exception):
            raise message
        finish = self.finish or ("tool_calls" if message.get("tool_calls") else "stop")
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=finish,
            message=SimpleNamespace(model_dump=lambda **_: deepcopy(message)),
        )])


def initial_history():
    return [{"role": "system", "content": SYSTEM}]


def test_search_then_answer_and_followup_preserve_provider_history():
    call_message = assistant(calls=[tool_call()])
    answer_message = assistant("December 25, 2021")
    client = ScriptedClient([call_message, answer_message, assistant("Ariane 5")])
    queries = []

    def search(query):
        queries.append(query)
        return [{"url": "https://science.nasa.gov/mission/webb/", "snippet": "Dec 25, 2021"}]

    graph = build_agent(client, "fixture", search)
    history = initial_history()
    first = run_turn(graph, history, "When did Webb launch?")
    assert first["status"] == "ok"
    assert history == initial_history()  # Input history is never mutated.
    assert client.requests[1]["messages"][2] == call_message
    assert client.requests[1]["messages"][3]["tool_call_id"] == "search-1"
    second = run_turn(graph, first["history"], "Which rocket launched it?")
    assert second["answer"] == "Ariane 5"
    assert client.requests[2]["messages"][4] == answer_message
    assert queries == ["NASA Webb launch"]
    assert [t["kind"] for t in first["trace"]] == ["model", "search", "model"]
    assert first["seconds"] >= sum(t["seconds"] for t in first["trace"])


def test_final_call_disables_tools_and_first_benchmark_call_requires_them():
    client = ScriptedClient([assistant(calls=[tool_call(id=str(i))]) for i in range(3)]
                            + [assistant("Final answer")])
    queries = []
    graph = build_agent(client, "fixture", lambda q: queries.append(q) or [], require_search=True)
    result = run_turn(graph, initial_history(), "Question")
    assert result["status"] == "ok"
    assert len(queries) == 3
    assert len(client.requests) == 4
    assert client.requests[0]["tool_choice"] == "required"
    assert "tools" not in client.requests[-1]


@pytest.mark.parametrize("call", [tool_call("not json"), tool_call('{"query":12}'),
                                  tool_call('{"query":""}'), tool_call(name="shell"),
                                  tool_call('{"query":"q","extra":true}')])
def test_invalid_tools_return_error_without_search(call):
    client = ScriptedClient([assistant(calls=[call]), assistant("No evidence")])

    def search(_):
        pytest.fail("Invalid arguments must not reach search")

    result = run_turn(build_agent(client, "fixture", search), initial_history(), "Question")
    assert result["status"] == "ok"
    assert result["trace"][1]["error"] is True
    assert '"error"' in result["history"][3]["content"]


def test_search_failure_keeps_tool_protocol_valid():
    client = ScriptedClient([assistant(calls=[tool_call()]), assistant("I could not verify it")])

    def search(_):
        raise TimeoutError()

    result = run_turn(build_agent(client, "fixture", search), initial_history(), "Question")
    assert result["status"] == "ok"
    assert result["trace"][1]["error"] is True
    assert client.requests[1]["messages"][3]["tool_call_id"] == "search-1"


@pytest.mark.parametrize("message, finish", [
    (assistant("", reasoning=""), None),
    (assistant("partial"), "length"),
    (TimeoutError(), None),
])
def test_failed_turn_does_not_commit_history(message, finish):
    client = ScriptedClient([message], finish=finish)
    history = initial_history()
    result = run_turn(build_agent(client, "fixture", lambda _: []), history, "Question")
    assert result["status"] == "error"
    assert result["history"] == history
    assert result["answer"] == ""


def test_dotenv_loads_environment_and_preserves_shell_values(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("REFLECTION_API_KEY=${DO_NOT_EXPAND}\nOPENAI_API_KEY=from-file\nTAVILY_API_KEY=search\n")
    monkeypatch.setenv("DO_NOT_EXPAND", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-value")
    monkeypatch.delenv("REFLECTION_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr("beam_agent.find_dotenv", lambda: str(path))
    keys = load_keys()
    assert keys["OPENAI_API_KEY"] == "ambient-value"
    assert keys["REFLECTION_API_KEY"] == "secret"
    assert keys["TAVILY_API_KEY"] == "search"


def test_extra_tool_call_gets_error_but_no_second_real_search():
    client = ScriptedClient([assistant(calls=[tool_call(id="one"), tool_call(id="two")]),
                             assistant("Answer")])
    queries = []
    result = run_turn(build_agent(client, "fixture", lambda q: queries.append(q) or []),
                      initial_history(), "Question")
    assert len(queries) == 1
    tool_messages = [m for m in result["history"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["one", "two"]
    assert '"error"' in tool_messages[1]["content"]


def notebook_definitions(*names):
    """Load selected definitions only, without rendering controls or reading .env."""
    import ast
    from pathlib import Path
    import nbformat
    import beam_agent

    namespace = vars(beam_agent).copy()
    notebook = nbformat.read(Path(__file__).with_name("tutorial.ipynb"), as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        selected = [node for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        if selected:
            exec(compile(ast.Module(body=selected, type_ignores=[]), "notebook", "exec"), namespace)
    return namespace


def test_responses_adapter_replays_reasoning_and_matches_tool_call_ids():
    ns = notebook_definitions("openai_turn")
    reasoning_item = {"type": "reasoning", "id": "reason-1", "encrypted_content": "opaque-fixture"}
    native_call = {"type": "function_call", "call_id": "call-1", "name": "web_search",
                   "arguments": '{"query":"NASA"}'}
    items = [SimpleNamespace(**item, model_dump=lambda item=item, **_: deepcopy(item))
             for item in [reasoning_item, native_call]]
    requests = []

    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        return SimpleNamespace(output=items, output_text="", status="completed")

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    history = initial_history() + [{"role": "user", "content": "Question"}]
    message, finish = ns["openai_turn"](client, "fixture", history, "low", 4096,
                                         {"tools": [], "tool_choice": "required"})
    assert finish == "tool_calls"
    assert message["tool_calls"][0]["id"] == "call-1"
    assert message["response_items"] == [reasoning_item, native_call]
    history += [message, {"role": "tool", "tool_call_id": "call-1", "content": "Evidence"}]
    ns["openai_turn"](client, "fixture", history, "low", 4096, {})
    assert requests[1]["input"][-3:] == [reasoning_item, native_call,
        {"type": "function_call_output", "call_id": "call-1", "output": "Evidence"}]
    assert requests[0]["store"] is False
    assert "reasoning.encrypted_content" in requests[0]["include"]
    assert requests[0]["tools"][0]["name"] == "web_search"
    assert "tools" not in requests[1]


def test_exact_score_rejects_explanation_only_and_failed_answers():
    import re

    ns = notebook_definitions("normalize", "short_answer", "exact_score")
    ns["re"] = re
    case = {"aliases": ["Canberra"]}
    score = ns["exact_score"]
    assert score({"status": "ok", "answer": "ANSWER: Canberra\nEvidence follows"}, case) == 1
    assert score({"status": "ok", "answer": "ANSWER: Sydney\nCanberra is also a city"}, case) == 0
    assert score({"status": "error", "answer": "ANSWER: Canberra"}, case) == 0
