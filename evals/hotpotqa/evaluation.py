"""Small, explicit evaluation harness. Gold labels are used only after inference."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from metrics import evaluate, score_example
from rag import build_graph, make_index


def retrieval_metrics(retrieved_facts, gold_record):
    """Coverage of annotated facts and titles, not a faithfulness judgment.

    Paragraph retrieval exposes every original sentence in each returned paragraph.
    Fact recall therefore measures availability to the reader, not sentence selection.
    """
    retrieved = {tuple(pair) for pair in retrieved_facts}
    gold = {tuple(pair) for pair in gold_record['supporting_facts']}
    got_titles = {title for title, _ in retrieved}
    gold_titles = {title for title, _ in gold}
    return {
        'support_fact_recall': len(retrieved & gold) / len(gold) if gold else 0.0,
        'support_title_recall': len(got_titles & gold_titles) / len(gold_titles) if gold_titles else 0.0,
        'all_support_titles': float(gold_titles <= got_titles),
    }


def run_experiment(records, backend_factory: Callable, *, k=2, max_rounds=2):
    """Keep all attempted questions, including backend errors, in the denominator.

    backend_factory takes NO record, so it cannot accidentally receive gold labels.
    Index creation errors are dataset/configuration failures and raise immediately.
    A failed provider call is represented by rag.invoke's error result.
    """
    ids = [r['_id'] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate question IDs would overwrite predictions')
    if not records:
        raise ValueError('Select at least one evaluation record')
    predictions = {'answer': {}, 'sp': {}}
    rows, runs = [], []
    for record in records:
        agent = build_graph(make_index(record['context']), backend_factory(),
                            k=k, max_rounds=max_rounds)
        result = agent.invoke(record['question'])
        qid = record['_id']
        predictions['answer'][qid] = result['answer']
        predictions['sp'][qid] = result['sp']
        row = {'_id': qid, 'type': record['type'], 'status': result['status'],
               'rounds': result['rounds'], 'backend_calls': result['backend_calls'],
               'latency_seconds': result['latency_seconds'],
               'invalid_citation_count': len(result['invalid_citations'])}
        row.update(score_example(result['answer'], result['sp'], record))
        row.update(retrieval_metrics(result['retrieved_facts'], record))
        rows.append(row)
        runs.append({'_id': qid, **result})
    return {'predictions': predictions, 'rows': rows, 'runs': runs,
            'metrics': evaluate(predictions, records)}


def save_experiment(result, records, directory, *, model, k, max_rounds, client_settings=None):
    """Write official prediction JSON, matching gold, traces and provenance together."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)  # never overwrite a previous run
    canonical = json.dumps(records, ensure_ascii=False, sort_keys=True).encode()
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'setting': 'per-question distractor candidates; teaching subset',
        'model': model, 'k_per_round': k, 'max_rounds': max_rounds,
        'question_ids': [r['_id'] for r in records],
        'records_sha256': hashlib.sha256(canonical).hexdigest(),
        'python': platform.python_version(),
        'client_settings': client_settings or {},
        'packages': {p: importlib.metadata.version(p) for p in
                     ['langgraph', 'langchain-openai', 'scikit-learn', 'pydantic']},
        'source_hashes': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ['rag.py', 'metrics.py', 'evaluation.py', 'uv.lock']},
        'notebook_source_sha256': hashlib.sha256((Path(__file__).parent / 'tools' / 'build_notebook.py').read_bytes()).hexdigest(),
        'max_backend_calls_per_question': 2 * max_rounds,
        'errors': sum(r['status'] == 'failed' for r in result['rows']),
    }
    payloads = {'predictions.json': result['predictions'], 'gold.json': records,
                'rows.json': result['rows'], 'traces.json': result['runs'],
                'metrics.json': result['metrics'], 'manifest.json': manifest}
    for name, payload in payloads.items():
        (directory / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    return directory
