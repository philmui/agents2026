# Sources and reproducibility

Primary references verified on 2026-10-10:

| Reference | What it establishes |
| --- | --- |
| [HotpotQA project homepage](https://hotpotqa.github.io/) | Dataset distribution, CC BY-SA 4.0 license, distractor versus fullwiki tasks, official evaluation links. |
| [Yang et al., EMNLP 2018](https://aclanthology.org/D18-1259/) | Wikipedia-based multi-document QA, sentence evidence supervision, bridge/comparison construction and benchmark design. |
| [Official repository README, pinned revision](https://github.com/hotpotqa/hotpot/blob/3635853403a8735609ee997664e1528f4480762a/README.md) | Original JSON and prediction schemas, zero-based evidence indices, train/dev/test conventions and code licensing. Fullwiki's supplied contexts are baseline retrieval results, not a dump of all Wikipedia. |
| [Official evaluator, pinned revision](https://github.com/hotpotqa/hotpot/blob/3635853403a8735609ee997664e1528f4480762a/hotpot_evaluate_v1.py) | Exact answer normalization, categorical-answer safeguards, supporting-fact set scoring, joint precision/recall products, per-question averaging and missing-ID behavior. |
| [HotpotQA Hugging Face mirror, pinned revision](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab) | Actual development distractor data used in the bundled teaching slice; source bytes pinned and verified by SHA-256. |
| [LangGraph agentic RAG tutorial](https://docs.langchain.com/oss/python/langgraph/agentic-rag) | Retrieve, assess evidence, rewrite, and generate as a graph with conditional routing. This tutorial adapts that pattern to HotpotQA and adds a finite retrieval budget. |

`data/provenance.json` records the exact input revision, hashes and selection rule. `data/README.md` explains the lossless mirror-to-original-schema conversion and the synthetic fixture. The ten real records preserve text and all candidate contexts. They are a teaching subset, not a statistical sample or an official evaluation run.

`metrics.py` returns official metric names on a 0–1 scale. `tests/test_metrics.py` checks it against the unchanged upstream evaluator, including empty strings/sets, repeated tokens, duplicate evidence, yes/no mismatches, missing components, and joint F1. Runtime diagnostics such as retrieval recall, graph steps, failures and latency are tutorial additions, not official HotpotQA leaderboard metrics.

The illustrative joint-score calculation uses answer precision = 1, answer recall = 1/2, evidence precision = 2/3, evidence recall = 1. Joint precision = 2/3, joint recall = 1/2, and joint F1 = 4/7 ≈ 0.5714. Multiplying the two F1 values instead gives 8/15 ≈ 0.5333, which is not the official formula.

## Citation

```bibtex
@inproceedings{yang2018hotpotqa,
  title={{HotpotQA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and
          Cohen, William W. and Salakhutdinov, Ruslan and Manning, Christopher D.},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year={2018},
  pages={2369--2380},
  url={https://aclanthology.org/D18-1259/},
  doi={10.18653/v1/D18-1259}
}
```
