"""Readable, dependency-free HotpotQA v1 scoring; all scores are in [0, 1].

Numerical semantics for valid official-format predictions match
data/upstream/hotpot_evaluate_v1.py (see SOURCES.md). The public dataset
entry point also rejects malformed prediction shapes rather than scoring them.
The metric code accepts the original JSON schema, not Hugging Face's columnar
nested dictionaries. Retrieval diagnostics belong in the evaluation harness;
``sp`` here means the final predicted supporting sentences, not all retrievals.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import re
import string
from typing import Any

METRIC_KEYS = (
    "em", "f1", "prec", "recall",
    "sp_em", "sp_f1", "sp_prec", "sp_recall",
    "joint_em", "joint_f1", "joint_prec", "joint_recall",
)


def normalize_answer(text: str) -> str:
    """Lowercase, remove ASCII punctuation/articles, and normalize whitespace.

    Punctuation is deleted (``New-York`` becomes ``newyork``), not replaced by
    spaces. Unicode punctuation, accents, synonyms, and aliases are not handled.
    """
    text = "".join(char for char in text.lower() if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _harmonic(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def answer_metrics(prediction: str, gold: str) -> dict[str, float]:
    """Normalized exact match and multiset token overlap, with yes/no safeguards.

    If either normalized answer is yes/no/noanswer, a mismatch gets zero token
    credit. The official edge case of two empty strings is EM=1 but F1=0.
    """
    pred, target = normalize_answer(prediction), normalize_answer(gold)
    result = {"em": float(pred == target), "f1": 0.0, "prec": 0.0, "recall": 0.0}
    special = {"yes", "no", "noanswer"}
    if (pred in special or target in special) and pred != target:
        return result
    pred_tokens, gold_tokens = pred.split(), target.split()
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap:
        precision, recall = overlap / len(pred_tokens), overlap / len(gold_tokens)
        result.update(f1=_harmonic(precision, recall), prec=precision, recall=recall)
    return result


def supporting_fact_metrics(prediction: Iterable, gold: Iterable) -> dict[str, float]:
    """Score sets of (exact title, zero-based sentence index) pairs.

    Duplicates count once. Titles are case-sensitive and not normalized.
    Empty predicted and gold sets have EM=1, but F1/precision/recall=0.
    """
    predicted, target = set(map(tuple, prediction)), set(map(tuple, gold))
    correct = len(predicted & target)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(target) if target else 0.0
    return {"em": float(predicted == target), "f1": _harmonic(precision, recall),
            "prec": precision, "recall": recall}


def score_example(pred_answer: str | None, pred_sp: Iterable | None,
                  gold_record: Mapping[str, Any]) -> dict[str, float]:
    """Return 12 official metrics. None denotes a missing prediction component.

    An absent answer does not erase available supporting-fact scores (and vice
    versa); either missing component forces all joint scores to zero.
    """
    scores = dict.fromkeys(METRIC_KEYS, 0.0)
    if pred_answer is not None:
        scores.update(answer_metrics(pred_answer, gold_record["answer"]))
    if pred_sp is not None:
        scores.update({f"sp_{key}": value for key, value in
                       supporting_fact_metrics(pred_sp, gold_record["supporting_facts"]).items()})
    if pred_answer is not None and pred_sp is not None:
        precision = scores["prec"] * scores["sp_prec"]
        recall = scores["recall"] * scores["sp_recall"]
        scores.update(joint_em=scores["em"] * scores["sp_em"],
                      joint_f1=_harmonic(precision, recall),
                      joint_prec=precision, joint_recall=recall)
    return scores


def evaluate(predictions: Mapping[str, Mapping],
             gold_records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Macro-average each official metric over *all* gold questions.

    ``predictions`` follows {"answer": {id: str}, "sp": {id: [[title, index]]}}.
    Missing IDs contribute zero to the affected component and joint metrics.
    Well-shaped extra prediction IDs are ignored, as in the official scorer. Empty gold
    data and duplicate gold IDs raise ValueError to catch harness mistakes.
    Present values must have the official prediction shape; malformed values
    raise TypeError, including JSON null (which is not a missing ID). Missing
    top-level answer/sp mappings are accepted as empty. Citation integers may
    be negative or out of range: they are scored unchanged, not repaired.
    """
    if not isinstance(predictions, Mapping):
        raise TypeError("Predictions must be a mapping with answer/sp mappings")
    answers, facts = predictions.get("answer", {}), predictions.get("sp", {})
    for name, component in (("answer", answers), ("sp", facts)):
        if not isinstance(component, Mapping):
            raise TypeError(f"Prediction {name!r} must be a mapping")
        if not all(isinstance(question_id, str) for question_id in component):
            raise TypeError(f"Prediction {name!r} IDs must be strings")
    if not all(isinstance(answer, str) for answer in answers.values()):
        raise TypeError("Every present answer prediction must be a string, not null")
    for citations in facts.values():
        if not isinstance(citations, (list, tuple)):
            raise TypeError("Every present sp prediction must be a list of citation pairs")
        for pair in citations:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or not isinstance(pair[0], str) or type(pair[1]) is not int):
                raise TypeError("Each citation must be [string title, integer index]; bool is invalid")
    records = list(gold_records)
    if not records:
        raise ValueError("Cannot evaluate an empty gold dataset")
    if len({record["_id"] for record in records}) != len(records):
        raise ValueError("Gold question IDs must be unique")
    totals = dict.fromkeys(METRIC_KEYS, 0.0)
    for record in records:
        question_id = record["_id"]
        scores = score_example(answers.get(question_id), facts.get(question_id), record)
        for key, value in scores.items():
            totals[key] += value
    return {key: value / len(records) for key, value in totals.items()}
