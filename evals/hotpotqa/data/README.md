# Data files and provenance

- `toy_example.json` is **entirely synthetic**. Its four short paragraphs explain the schema and exercise the offline graph. It is not a HotpotQA question, has no benchmark score, and uses `level: "toy"` (not an official difficulty value).
- `dev_sample.json` contains **10 real HotpotQA development distractor records**, selected as the first five bridge and first five comparison examples in source order. All ten contexts and every sentence are retained. This deliberately balanced teaching slice is not representative of the 7,405-question development set and must not be presented as leaderboard performance.
- `example.json` is one complete real record from that slice, ID `5abd94525542992ac4f382d2`. It has all ten contexts; it is not the shortened display excerpt used in explanatory material.
- `provenance.json` records source revisions, source/file SHA-256 digests, the selection rule, source indices, and IDs. The full development mirror has 5,918 bridge and 1,487 comparison records. Our sample's five comparison examples all have yes/no answers; the full dataset also contains non-boolean comparisons.
- `upstream/hotpot_evaluate_v1.py` is the unchanged official scorer, pinned to commit `3635853403a8735609ee997664e1528f4480762a`. Its Apache 2.0 license is included alongside it. It imports `ujson`, included in this module's locked environment. Unit tests substitute standard-library `json` for file I/O only and compare the unchanged scoring functions. A separate subprocess test invokes the actual unmodified command-line interface using `ujson`.

The original CMU JSON endpoint timed out/reset during preparation. We downloaded the [HotpotQA organization's pinned Hugging Face validation mirror](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab) instead. Its Parquet container stores `id` and dictionaries of parallel lists. `prepare_sample.py` restores `_id`, `[title, sentence_index]` pairs, and `[title, sentences]` pairs without changing text, labels, IDs, or order. These files preserve record content, but are not byte-identical copies of the CMU JSON serialization.

To reproduce, from `evals/hotpotqa` run:

```sh
uv run --no-project --python 3.12 --with pyarrow python data/prepare_sample.py
```

This downloads approximately 27 MB to ignored `data/.cache/`, verifies its pinned SHA-256 digest, and creates the sample plus a complete reconstructed JSON representation in that cache. It does not install PyArrow into the tutorial environment. The notebook runs on the checked-in files without downloading the full dataset.

## Attribution and reuse

Real data excerpts are from **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering**, by Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning (EMNLP 2018). [Paper](https://aclanthology.org/D18-1259/) · [Dataset homepage](https://hotpotqa.github.io/).

The dataset and these redistributed real-data excerpts are licensed under [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/); the [legal text](https://creativecommons.org/licenses/by-sa/4.0/legalcode) describes attribution and share-alike terms. Changes here are subset selection and JSON reserialization/structural conversion only. Retain this attribution, license link, and change description when reusing these excerpts. The synthetic fixture was created for this tutorial and is explicitly separate from the licensed benchmark data.

A `bridge` label expresses the intended question construction; it is not a guarantee that a model must perform two distinct retrieval steps. For example, the real album question has an alternate sentence mentioning the same organization as record label. Supporting-fact scores measure agreement with the annotated sentence set, not every possible logically valid rationale.
