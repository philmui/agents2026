"""Differential tests against an unchanged, pinned official HotpotQA scorer."""
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics import (answer_metrics, evaluate, normalize_answer, score_example,
                     supporting_fact_metrics)

SCORER = ROOT / "data/upstream/hotpot_evaluate_v1.py"
spec = importlib.util.spec_from_file_location("official_hotpot", SCORER)
official = importlib.util.module_from_spec(spec)
# ujson is only used for file I/O. Stdlib json reads these fixtures identically;
# the vendored source and all official metric functions stay unchanged.
with patch.dict(sys.modules, {"ujson": json}):
    spec.loader.exec_module(official)


class MetricsTests(unittest.TestCase):
    def test_answer_differential(self):
        pairs = [
            ("The Harbor City!", "harbor city"), ("New-York", "New York"),
            ("red red blue", "red blue blue"), ("yes", "yes indeed"),
            ("yes indeed", "yes"), ("no", "yes"), ("noanswer", "no answer"),
            (" YES ", "yes"), ("", ""), ("the", "a"), ("", "city"),
            ("Beyoncé", "Beyonce"), ("A—B", "A B"), ("city city", "city"),
        ]
        for pred, gold in pairs:
            with self.subTest(pred=pred, gold=gold):
                actual = answer_metrics(pred, gold)
                f1, precision, recall = official.f1_score(pred, gold)
                self.assertEqual(actual, {"em": float(official.exact_match_score(pred, gold)),
                                         "f1": f1, "prec": precision, "recall": recall})
                self.assertEqual(normalize_answer(pred), official.normalize_answer(pred))

    def test_supporting_fact_differential(self):
        pairs = [([], []), ([], [["A", 0]]), ([["A", 0]], []),
                 ([["A", 0], ["A", 0]], [["A", 0]]),
                 ([["a", 0], ["B", 1]], [["A", 0], ["B", 0]]),
                 ([["A", 0], ["C", 0]], [["A", 0], ["B", 0], ["C", 0]])]
        for pred, gold in pairs:
            with self.subTest(pred=pred, gold=gold):
                expected = dict.fromkeys(["sp_em", "sp_f1", "sp_prec", "sp_recall"], 0.0)
                official.update_sp(expected, pred, gold)
                self.assertEqual(supporting_fact_metrics(pred, gold),
                                 {key[3:]: value for key, value in expected.items()})

    def test_joint_uses_precision_and_recall_products(self):
        gold = {"answer": "Harbor City", "supporting_facts": [["A", 0], ["B", 0]]}
        scores = score_example("Harbor", [["A", 0], ["B", 0], ["C", 0]], gold)
        self.assertAlmostEqual(scores["joint_prec"], 2/3)
        self.assertAlmostEqual(scores["joint_recall"], 1/2)
        self.assertAlmostEqual(scores["joint_f1"], 4/7)
        self.assertNotAlmostEqual(scores["joint_f1"], scores["f1"] * scores["sp_f1"])

    def test_dataset_differential_missing_and_extra_predictions(self):
        gold = [{"_id": str(i), "answer": "Harbor City",
                 "supporting_facts": [["A", 0], ["B", 1]]} for i in range(5)]
        predictions = {
            "answer": {"0": "Harbor City", "1": "Harbor", "2": "Harbor City", "extra": "wrong"},
            "sp": {"0": [["A", 0], ["B", 1]], "1": [["A", 0], ["C", 0]],
                   "3": [["A", 0], ["B", 1]], "extra": []},
        }
        with tempfile.TemporaryDirectory() as folder:
            pred_path, gold_path = Path(folder)/"pred.json", Path(folder)/"gold.json"
            pred_path.write_text(json.dumps(predictions))
            gold_path.write_text(json.dumps(gold))
            output = io.StringIO()
            with redirect_stdout(output):
                official.eval(pred_path, gold_path)
            expected = ast.literal_eval(output.getvalue().splitlines()[-1])
        self.assertEqual(evaluate(predictions, gold), expected)
        self.assertAlmostEqual(expected["em"], 2/5)
        self.assertAlmostEqual(expected["sp_em"], 2/5)
        self.assertAlmostEqual(expected["joint_em"], 1/5)

    def test_gold_input_guards(self):
        with self.assertRaises(ValueError):
            evaluate({}, [])
        with self.assertRaises(ValueError):
            evaluate({}, [{"_id": "same"}, {"_id": "same"}])

    def test_malformed_prediction_shapes_are_rejected(self):
        gold = [{"_id": "q", "answer": "city", "supporting_facts": [["A", 0]]}]
        malformed = [
            None, [], {"answer": None}, {"answer": []}, {"sp": None}, {"sp": []},
            {"answer": {"q": None}}, {"answer": {"q": 42}},
            {"answer": {1: "city"}}, {"sp": {1: []}},
            {"sp": {"q": None}}, {"sp": {"q": "nonsense"}},
            {"sp": {"q": ["A"]}}, {"sp": {"q": [["A"]]}},
            {"sp": {"q": [["A", 0, "extra"]]}},
            {"sp": {"q": [[42, 0]]}}, {"sp": {"q": [["A", False]]}},
            {"sp": {"q": [["A", 0.0]]}}, {"sp": {"q": [["A", "0"]]}},
        ]
        for predictions in malformed:
            with self.subTest(predictions=predictions), self.assertRaises(TypeError):
                evaluate(predictions, gold)
        self.assertEqual(evaluate({}, gold), evaluate({"answer": {}, "sp": {}}, gold))

    def test_invalid_citation_addresses_are_not_repaired(self):
        gold = [{"_id": "q", "answer": "city", "supporting_facts": [["A", 0]]}]
        citations = [["A", 0], ["A", -1], ["A", 99], ["Invented title", 0]]
        actual = evaluate({"answer": {"q": "city"}, "sp": {"q": citations}}, gold)
        self.assertEqual(actual["sp_prec"], 0.25)
        self.assertEqual(actual["sp_recall"], 1.0)
        self.assertEqual(actual["sp_f1"], 0.4)
        self.assertEqual(actual["joint_f1"], 0.4)
        self.assertEqual(citations, [["A", 0], ["A", -1], ["A", 99], ["Invented title", 0]])

    def test_bundled_data_integrity(self):
        provenance = json.loads((ROOT/"data/provenance.json").read_text())
        for name, expected in provenance["file_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT/"data"/name).read_bytes()).hexdigest(), expected)
        records = json.loads((ROOT/"data/dev_sample.json").read_text())
        self.assertEqual(len(records), 10)
        self.assertEqual([x["_id"] for x in records], provenance["selected_ids"])
        self.assertEqual(sum(x["type"] == "bridge" for x in records), 5)
        for record in records:
            self.assertEqual(len(record["context"]), 10)
            contexts = dict(record["context"])
            for title, index in record["supporting_facts"]:
                self.assertIn(title, contexts)
                self.assertGreaterEqual(index, 0)
                self.assertLess(index, len(contexts[title]))


if __name__ == "__main__":
    unittest.main()
