# Validation record

Date: 2026-10-10. Final result: **35 tests and 39 parameterized subtests pass**, and all **23 code cells in the 51-cell notebook** execute successfully. This record distinguishes tested offline behavior from optional hosted behavior.

## Executed checks

- Installed the module environment from `uv.lock` with Python 3.13.
- Executed all 51 notebook cells in order with `RUN_LIVE = False` and stored their outputs.
- Compared answer, supporting-fact, joint and dataset metrics against the pinned, unchanged official scorer. Cases include special answers, repeated tokens, duplicate citations, empty values and missing predictions.
- Invoked the unmodified official evaluator in a separate Python process using its actual `ujson` dependency, and matched its result against the tutorial scorer.
- Tested real LangGraph execution, the retrieval-round limit, original-question preservation, sentence coordinates, invalid citations, failed provider calls and no scripted fallback.
- Tested the batch harness for inclusion of failed questions in score denominators, separation of gold labels from inference, duplicate-ID rejection, preservation of outputs, saved error counts and overwrite protection.
- Exercised the actual `LiveBackend` adapter using a mocked structured-output client, including valid multi-round responses, refusal-like `None` output, and invalid sentence-index types. No provider network request was made.
- Verified data checksums and all 24 annotated supporting-fact references across the 10 actual records. Each record retains all ten distractor paragraphs.

## Review and fixes

Separate agent reviews checked data/metric fidelity and notebook/harness contracts. Confirmed issues were fixed: abstention documentation now matches the empty-string live output contract, failure summaries count the actual `failed` status, and the saved manifest includes client request settings and the notebook source hash. The public scorer validates supplied prediction shapes while retaining invalid sentence coordinates as scoreable predictions.

A fresh-context adversarial reviewer read an isolated committed snapshot (base `0dc2d43`, head `a352063`) and found two concrete issues, both fixed:

1. The official CLI needed `ujson`, which was absent from the environment. It is now a locked dependency, and a separate-process regression test runs the exact command used by the notebook without a JSON import shim.
2. The synthetic walkthrough initially retrieved only the creator paragraph while its prose described book-then-creator retrieval. The invented book sentence now uses the question's word “creator”; the actual shipped JSON is covered by a regression test, and notebook assertions require both accumulated titles and valid citations. The notebook was reexecuted and the slide JSON/PDF regenerated.

The fresh-context reviewer reread the corrected code, executed notebook outputs, official CLI test and new adapter tests. The final verdict was **approve**, with no remaining blocker or should-fix finding in the reviewed core scope. Separate slide review also corrected the round-limit arrow, the distinction between successful execution and sufficient evidence, and retry/abstention wording.

The Codex CLI review was attempted against the same clean snapshot with the skill's adversarial framing and read-only sandbox. Its tool host timed out three times before it could read files, so its `NOT REACHED` report is **not counted as a completed review**. A fresh-context collaboration reviewer completed the substantive review instead. Snapshot HEAD and clean status were verified before and after. Suite freeze helper scripts were unavailable; the target was copied into an isolated temporary Git repository and committed without changing this workspace's Git history. These reviews use fresh contexts within the same model family and do not establish cross-family reviewer independence.

## Visual checks

- Inspected the notebook HTML preview and each of its three SVG diagrams.
- Inspected slide contact sheets and raw-JSON, real-record and workflow slides at full size.
- Browser checks exercised all 34 slides at 1500-, 1024- and 390-pixel viewport widths (102 slide/viewport cases), with no horizontal overflow, missing images or JavaScript errors.
- Checked keyboard navigation, notes, reading view, direct slide links and no-JavaScript fallback.
- Checked all 15 slide SVG text bounds and regenerated the final PDF with exactly 34 pages. A print-only trailing blank page was corrected.
- The final raw-JSON slide is parsed and compared with the actual synthetic JSON file. The PDF text contains the corrected example sentence.

## Scope limits

No hosted LLM calls were made. The live adapter was checked with mocked structured responses and failure paths, so provider availability, model-specific structured-output compatibility, token charges and actual answer quality remain unverified. The notebook keeps live calls behind explicit `RUN_LIVE` opt-in.

The scripted walkthrough is a test fixture. Its answer scores are not model results. The actual-record offline experiment measures TF-IDF retrieval coverage only. The selected ten records are not a representative benchmark sample. Browser validation applies to the local HTML deck, not PowerPoint.
