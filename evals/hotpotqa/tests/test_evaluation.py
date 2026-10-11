"""Harness contracts: isolated inputs, unchanged data, and honest denominators."""

from copy import deepcopy
import json

import pytest

from evaluation import retrieval_metrics, run_experiment, save_experiment
from metrics import evaluate
from rag import Answer, Citation, ScriptedBackend


def record(qid="q1"):
    return {
        "_id": qid,
        "question": "Where is Azure?",
        "answer": "eastern region",
        "type": "bridge",
        "level": "test-only-label",
        "supporting_facts": [["Azure", 0]],
        "context": [["Azure", ["Azure is in the eastern region."]],
                    ["Unrelated", ["Cats sleep here."]]],
    }


def good_backend():
    # A scorer/harness fixture, not a real question-answering model.
    return ScriptedBackend(grades=[True], rewrites=[], answer=Answer(
        answer="eastern region", citations=[Citation(title="Azure", sentence_id=0)]))


def failing_backend():
    backend = good_backend()

    def fail(*args):
        raise TimeoutError("simulated provider failure")

    backend.grade = fail
    return backend


def test_backend_factory_receives_no_record_and_input_data_unchanged():
    records = [record()]
    before = deepcopy(records)
    seen = []
    backend = good_backend()
    original_grade = backend.grade

    def grade(question, documents):
        seen.append((question, [doc.as_context() for doc in documents]))
        return original_grade(question, documents)

    backend.grade = grade
    result = run_experiment(records, lambda: backend, k=1)
    assert records == before
    assert seen == [(records[0]["question"], [{
        "title": "Azure", "sentences": [
            {"sentence_id": 0, "text": "Azure is in the eastern region."}],
    }])]
    assert result["metrics"]["joint_f1"] == 1


def test_failed_request_stays_in_all_denominators():
    records = [record("success"), record("failure")]
    backends = iter([good_backend(), failing_backend()])
    result = run_experiment(records, lambda: next(backends), k=1)
    assert len(result["rows"]) == len(result["runs"]) == 2
    assert result["predictions"]["answer"] == {
        "success": "eastern region", "failure": ""}
    assert result["predictions"]["sp"]["failure"] == []
    assert result["rows"][1]["status"] == "failed"
    assert result["runs"][1]["backend_calls"] == 1
    assert result["metrics"]["em"] == 0.5
    assert result["metrics"]["sp_f1"] == 0.5
    assert result["metrics"]["joint_f1"] == 0.5
    # Successful retrieval can coexist with a failed model request.
    assert result["rows"][1]["support_fact_recall"] == 1


def test_missing_prediction_components_do_not_remove_a_question():
    records = [record("complete"), record("missing_answer"), record("missing_all")]
    predictions = {
        "answer": {"complete": "eastern region"},
        "sp": {"complete": [["Azure", 0]], "missing_answer": [["Azure", 0]]},
    }
    scores = evaluate(predictions, records)
    assert scores["em"] == pytest.approx(1 / 3)
    assert scores["sp_f1"] == pytest.approx(2 / 3)
    assert scores["joint_f1"] == pytest.approx(1 / 3)


def test_invalid_claims_are_exported_without_scorer_repair():
    backend = good_backend()
    backend.answer.citations.append(Citation(title="Azure", sentence_id=99))
    result = run_experiment([record()], lambda: backend, k=1)
    assert result["predictions"]["sp"]["q1"] == [["Azure", 0], ["Azure", 99]]
    assert result["rows"][0]["invalid_citation_count"] == 1
    assert result["rows"][0]["status"] == "invalid_citations"
    assert result["metrics"]["sp_f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("records", [[], [record(), record()]])
def test_empty_or_duplicate_ids_fail_before_backend_creation(records):
    def should_not_run():
        pytest.fail("backend should not be created for invalid batch")

    with pytest.raises(ValueError):
        run_experiment(records, should_not_run)


def test_fact_recall_and_title_recall_measure_different_coverage():
    gold = record()
    gold["supporting_facts"] = [["Azure", 0], ["Azure", 1], ["Bridge", 0]]
    scores = retrieval_metrics([["Azure", 0], ["Azure", 1], ["Azure", 1]], gold)
    assert scores["support_fact_recall"] == pytest.approx(2 / 3)
    assert scores["support_title_recall"] == 0.5
    assert scores["all_support_titles"] == 0


def test_save_records_failures_and_refuses_overwrite(tmp_path):
    records = [record()]
    result = run_experiment(records, failing_backend, k=1)
    directory = tmp_path / "run"
    save_experiment(result, records, directory, model="test-only-fixture", k=1, max_rounds=2)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["errors"] == 1
    assert manifest["question_ids"] == ["q1"]
    assert manifest["max_backend_calls_per_question"] == 4
    assert json.loads((directory / "predictions.json").read_text()) == result["predictions"]
    assert json.loads((directory / "gold.json").read_text()) == records
    with pytest.raises(FileExistsError):
        save_experiment(result, records, directory, model="test-only-fixture", k=1, max_rounds=2)
