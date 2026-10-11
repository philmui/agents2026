"""Exercise the exact official command used by the notebook, without import shims."""
import ast
import json
from pathlib import Path
import subprocess
import sys

from metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_official_cli_runs_in_project_environment(tmp_path):
    gold = [json.loads((ROOT / 'data/toy_example.json').read_text())]
    record = gold[0]
    predictions = {'answer': {record['_id']: 'Harbor'},
                   'sp': {record['_id']: record['supporting_facts'] + [['Harbor City', 0]]}}
    for name, value in [('gold.json', gold), ('predictions.json', predictions)]:
        (tmp_path / name).write_text(json.dumps(value))
    completed = subprocess.run([sys.executable, str(ROOT / 'data/upstream/hotpot_evaluate_v1.py'),
                                str(tmp_path / 'predictions.json'), str(tmp_path / 'gold.json')],
                               capture_output=True, text=True, check=True)
    assert ast.literal_eval(completed.stdout) == evaluate(predictions, gold)
