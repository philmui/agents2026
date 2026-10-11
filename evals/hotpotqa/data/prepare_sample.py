"""Rebuild the checked-in teaching slice from the pinned HotpotQA mirror.

Run: uv run --no-project --python 3.12 --with pyarrow python data/prepare_sample.py
Downloads 27 MB once into data/.cache (ignored). No credentials are required.
The CMU original JSON endpoint was unreachable during preparation; the mirror
uses a different container schema. This script preserves text and IDs exactly
while restoring original HotpotQA JSON list/pair structure. See provenance.json.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request

HF_COMMIT = "1908d6afbbead072334abe2965f91bd2709910ab"
SCORER_COMMIT = "3635853403a8735609ee997664e1528f4480762a"
URL = (f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{HF_COMMIT}/"
       "distractor/validation-00000-of-00001.parquet?download=true")
PARQUET_SHA = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
ROOT = Path(__file__).resolve().parent


def convert(row):
    """Lossless structural conversion; no trimming, resplitting, or relabeling."""
    return {"_id": row["id"], **{k: row[k] for k in ("question", "answer", "type", "level")},
            "supporting_facts": [list(pair) for pair in zip(
                row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"], strict=True)],
            "context": [list(pair) for pair in zip(
                row["context"]["title"], row["context"]["sentences"], strict=True)]}


def main():
    import pyarrow.parquet as pq
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    parquet = cache / "validation.parquet"
    if not parquet.exists():
        with urllib.request.urlopen(URL, timeout=120) as response:
            parquet.write_bytes(response.read())
    assert hashlib.sha256(parquet.read_bytes()).hexdigest() == PARQUET_SHA, "Source checksum mismatch"
    rows = pq.read_table(parquet).to_pylist()
    records = [convert(row) for row in rows]
    # This is a reconstructed JSON representation, not a byte-for-byte CMU download.
    (cache / "hotpot_dev_distractor_v1.reconstructed.json").write_text(
        json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8")
    counts, selected, indices = Counter(), [], []
    for index, record in enumerate(records):
        kind = record["type"]
        if counts[kind] < 5:
            selected.append(record)
            indices.append(index)
            counts[kind] += 1
        if len(selected) == 10:
            break
    for name, data in [("dev_sample.json", selected),
                       ("example.json", next(x for x in selected if x["_id"] == "5abd94525542992ac4f382d2"))]:
        (ROOT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    provenance = {
        "dataset": "HotpotQA", "split": "development / validation", "setting": "distractor",
        "license": "CC BY-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, Christopher D. Manning (2018). HotpotQA.",
        "homepage": "https://hotpotqa.github.io/",
        "canonical_json_url": "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "source_url": URL, "source_revision": HF_COMMIT, "source_sha256": PARQUET_SHA,
        "source_rows": len(rows), "source_type_counts": dict(Counter(x["type"] for x in records)),
        "retrieved_date": "2026-10-10",
        "format_note": "CMU JSON endpoint timed out/reset. Downloaded pinned HotpotQA Hugging Face validation parquet mirror. Lossless structural conversion id->_id, supporting_facts dict->pairs, context dict->pairs; text, sentence ordering, titles, labels, IDs and all ten contexts retained. JSON serialization differs from original.",
        "selection_rule": "First five bridge and first five comparison records in source order; preserve relative source order. Teaching slice, not a random or representative performance benchmark.",
        "selected_source_indices_zero_based": indices, "selected_ids": [x["_id"] for x in selected],
        "selected_type_counts": dict(counts), "example_id": "5abd94525542992ac4f382d2",
        "scorer_revision": SCORER_COMMIT,
        "scorer_url": f"https://raw.githubusercontent.com/hotpotqa/hotpot/{SCORER_COMMIT}/hotpot_evaluate_v1.py",
        "file_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                        for name in ("dev_sample.json", "example.json", "toy_example.json", "upstream/hotpot_evaluate_v1.py")},
    }
    (ROOT/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "sample_indices": indices, "sample_type_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
