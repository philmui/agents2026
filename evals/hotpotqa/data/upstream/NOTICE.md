# Upstream HotpotQA evaluator

`hotpot_evaluate_v1.py` is redistributed without modification from:
https://github.com/hotpotqa/hotpot/blob/3635853403a8735609ee997664e1528f4480762a/hotpot_evaluate_v1.py

The HotpotQA authors' [README at the same revision](https://github.com/hotpotqa/hotpot/blob/3635853403a8735609ee997664e1528f4480762a/README.md#license) licenses the code under Apache 2.0. A copy of the license is included in `LICENSE-2.0.txt`. That README states the evaluation script is adapted from SQuAD. The upstream script carries no additional file-level copyright or NOTICE text.

See `../provenance.json` for the source SHA-256. The readable implementation in `../../metrics.py` was rewritten for this tutorial; differential tests verify that its numerical scoring agrees with the upstream evaluator on the tested edge cases.
