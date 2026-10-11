"""Regression tests for evidence coordinates, routing bounds, and failures."""

import pytest

from rag import Answer, Citation, Grade, ScriptedBackend, build_graph, make_index


CONTEXT = [
    ["First bridge", ["The violet expedition visited Azure.", "Original sentence two."]],
    ["Azure", ["Azure is in the eastern region."]],
    ["Distractor", ["Cats sleep here."]],
]


def fixture_backend(grades=None, rewrites=None, citations=None):
    return ScriptedBackend(
        grades=grades or [True], rewrites=rewrites or [],
        answer=Answer(answer="eastern region", citations=citations or []),
    )


def test_original_sentence_coordinates_are_preserved():
    index = make_index(CONTEXT)
    assert index.documents[0].sentences == tuple(CONTEXT[0][1])
    assert index.documents[0].facts == [["First bridge", 0], ["First bridge", 1]]
    assert index.search("violet expedition", 1)[0].title == "First bridge"
    assert index.search("cats", 1)[0].title == "Distractor"
    assert [doc.title for doc in index.search("unknown-token", 3)] == [row[0] for row in CONTEXT]


def test_bounded_loop_union_and_original_question():
    class RecordingBackend(ScriptedBackend):
        observations = []

        def grade(self, question, documents):
            self.observations.append((question, [doc.title for doc in documents]))
            return Grade(sufficient=False, reason="Exercise hard round cap")

    backend = RecordingBackend(grades=[False], rewrites=["Azure eastern region"],
                               answer=Answer(answer="", citations=[]))
    result = build_graph(make_index(CONTEXT), backend, k=1).invoke("violet expedition")
    assert result["rounds"] == 2
    assert result["backend_calls"] == 4
    assert [event["node"] for event in result["trace"]] == ["retrieve", "grade", "rewrite", "retrieve", "grade", "generate"]
    assert result["retrieved_titles"] == ["First bridge", "Azure"]
    assert backend.observations == [
        ("violet expedition", ["First bridge"]),
        ("violet expedition", ["First bridge", "Azure"]),
    ]


def test_invalid_citations_are_flagged_and_preserved_for_scoring():
    citations = [Citation(title="First bridge", sentence_id=1),
                 Citation(title="First bridge", sentence_id=-1),
                 Citation(title="First bridge", sentence_id=99),
                 Citation(title="Azure", sentence_id=0)]
    result = build_graph(make_index(CONTEXT), fixture_backend(citations=citations), k=1).invoke("violet expedition")
    assert result["sp"] == [[c.title, c.sentence_id] for c in citations]
    assert result["invalid_citations"] == result["sp"][1:]
    assert result["status"] == "invalid_citations"


def test_inference_interface_does_not_accept_labeled_records():
    agent = build_graph(make_index(CONTEXT), fixture_backend())
    with pytest.raises(TypeError, match="question string"):
        agent.invoke({"question": "violet expedition", "answer": "gold", "supporting_facts": []})


@pytest.mark.parametrize("stage", ["grade", "rewrite", "generate"])
def test_backend_failure_retained_without_fallback(stage):
    backend = fixture_backend(grades=[False] if stage == "rewrite" else [True])

    def fail(*args):
        raise TimeoutError("simulated backend timeout")

    setattr(backend, stage, fail)
    result = build_graph(make_index(CONTEXT), backend).invoke("violet expedition")
    assert result["status"] == "failed"
    assert result["answer"] == ""
    assert result["sp"] == []
    assert result["error"]["node"] == stage
    assert result["trace"][-1]["status"] == "failed"
    assert result["backend_calls"] == (1 if stage == "grade" else 2)


def test_positive_grade_stops_after_one_round_and_can_reuse_agent():
    agent = build_graph(make_index(CONTEXT), fixture_backend())
    for _ in range(2):
        result = agent.invoke("violet expedition")
        assert result["rounds"] == 1
        assert result["backend_calls"] == 2
        assert len(result["trace"]) == 3


def test_one_round_cap_skips_rewrite_even_with_negative_grade():
    result = build_graph(make_index(CONTEXT), fixture_backend(grades=[False]), max_rounds=1).invoke("violet expedition")
    assert result["rounds"] == 1
    assert result["backend_calls"] == 2


@pytest.mark.parametrize("rounds", [0, 3, True])
def test_invalid_limits_rejected(rounds):
    with pytest.raises(ValueError):
        build_graph(make_index(CONTEXT), fixture_backend(), max_rounds=rounds)


def test_duplicate_titles_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        make_index([["same", ["One"]], ["same", ["Two"]]])


def test_actual_notebook_toy_walkthrough_finds_both_hops():
    """Use the shipped teaching asset: an unrelated fixture missed a lexical bug."""
    from pathlib import Path
    import json
    record = json.loads((Path(__file__).resolve().parents[1] / 'data/toy_example.json').read_text())
    index = make_index(record['context'])
    backend = ScriptedBackend(
        grades=[False, True], rewrites=['Mira Vale birthplace'],
        answer=Answer(answer='Harbor City', citations=[
            Citation(title='Lantern Atlas', sentence_id=0),
            Citation(title='Mira Vale', sentence_id=0),
        ]),
    )
    result = build_graph(index, backend, k=1, max_rounds=2).invoke(record['question'])
    assert index.search(record['question'], k=1)[0].title == 'Lantern Atlas'
    assert result['retrieved_titles'] == ['Lantern Atlas', 'Mira Vale']
    assert result['status'] == 'ok'
    assert result['invalid_citations'] == []
    assert result['rounds'] == 2 and result['backend_calls'] == 4
