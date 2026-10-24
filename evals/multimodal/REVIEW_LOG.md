# Adversarial review: findings, causes, and repairs

The notebook, guide, and slide authors worked in separate agent contexts. The guide author challenged a frozen slide snapshot; the slide author challenged a frozen notebook/code snapshot; the notebook author checked the guides. The lead verified findings against the current files, added regression checks, and inspected rendered artifacts. These were separate review roles using the same model family, not independent scientific replication or model-family diversity.

Review questions were concrete: Can this lesson run from a clean kernel? Can an apparently correct metric conceal missing evidence? Can a citation lead to the wrong pixels? Does the picture imply a stronger claim than the code supports? What do the tests fail to exercise?

## Reference-notebook issues corrected in the new lessons

| Issue | Root cause | Correction |
|---|---|---|
| Reference cell 37 labels hit rate as recall | It counts any intersection with the gold set rather than the fraction of relevant sources found | Distinct Recall@k, HitRate@k, and MRR@k calculations; worked example with two relevant sources |
| Number overlap can pass after labels are swapped | A bag of numeric strings discards label/value/unit relationships | Structured triples, duplicate detection, and an intentionally swapped chart example |
| Normalization can sound like score calibration | Unit length and cross-model comparability are different properties | Explicit coordinate-system examples; rank fusion instead of comparing raw unrelated scores |
| Character slicing can sound like a token guarantee | Characters and tokenizer tokens have different boundaries | Explain the model's short text context and its own preprocessing; no `text[:300]` token guarantee |
| Timestamp windows can appear to prove event onset | Segment boundaries and observed frame times are different evidence | Half-open windows, observed sample times, and explicit limits on event-time claims |

## Introduced issues found and repaired

| Finding | Root cause | Repair and verification |
|---|---|---|
| Notebook image display would raise `TypeError` | `from lab import *` overwrote IPython's `Image` with the imported Pillow module | Use explicit helper imports; fresh-kernel execution is the regression gate |
| A selected video frame could be replaced by an earlier frame | Evidence assembly chose the first frame in a segment rather than the frame that retrieval actually selected | Preserve direct or fused visual evidence; label a representative-frame fallback for transcript-only retrieval; direct and fused-frame regression cases |
| A second PDF could overwrite the first PDF's rendered evidence | All pages were written to a shared `page_01.png` path | Namespace render paths and page identities by document stem and source hash; two same-named PDFs remain distinct and earlier image bytes remain intact |
| A timestamp near a minute boundary could display `00:60.0` | Seconds were rounded after splitting minutes | Round integer tenths before `divmod`; test minute rollover |
| Important mechanisms were hidden behind helper calls | Reusable helpers reduced visible implementation detail | Add manual Qdrant storage/querying, caption flattening, pixel-feature computation, an RRF accumulator, and frame/window code with explanations |
| Two retrieval branches could be described as an equal-budget experiment | Same final result count was mistaken for same candidate work | State C's per-branch candidate budget and caption advantages; separate fixture results from neural-model results |
| RRF examples could imply agreement across disjoint stores | Caption overlap was an unstated assumption | Label the caption variant; explain disjoint-source ties and the shared video-segment identity case |
| Guide and helper RRF ranks used different duplicate conventions | One kept original rank gaps; the helper compacted unique-source ranks | State both policies explicitly so learners can reproduce each calculation |
| A workflow arrow bypassed retrieval | Layout convenience displaced a dependency | Route index results back to the retrieval stage and re-render |
| A video diagram used unrelated intervals | A generic illustration was not aligned with the corpus | Match the 0–4, 4–8, and 8–12 second scenes; label illustrative coarse sampling and a hypothetical brief flash |
| Some slide source anchors were off by one or two cells | Heading ranges were reused as exact code references | Check actual zero-based reference cells and rebuild the manifest and notes |
| A video endpoint looked like a decoded sample | One filled glyph represented both a sample and an exclusive boundary | Use a hollow endpoint labeled `12 s end` |
| Two slide layouts had tiny or overlapping labels | A large diagram was scaled into a narrow column; a centered label was used as a record header | Recompose the geometry and check projected text sizes and intersections |
| A cosine diagram label crossed its arrow | Text-to-text geometry checks did not detect a text-to-line intersection | Move both vector labels clear of the arrows; inspect the full-size notebook rendering and corrected SVG |
| The slide shell could lose arrow-key navigation after a click | A local-file iframe could receive focus without usable cross-frame listeners | Keep the static iframe noninteractive; focus the parent stage and test post-click navigation |
| JavaScript-disabled navigation showed no lesson | The shell depended entirely on script-generated navigation | Add a static index containing every slide link and full explanation |

The detailed slide review and rendering results are in [slides/REVIEW.md](slides/REVIEW.md) and [slides/render_review.json](slides/render_review.json). The executed checks and remaining test boundaries are recorded in [VALIDATION.md](VALIDATION.md).

## Boundaries retained deliberately

The color encoder is deliberately unable to understand chart labels or semantics. It is a teaching model, not a substitute for a CLIP benchmark. Hand-authored parses are fixtures, not evidence that a model parsed accurately. The supplied video is silent; the aligned transcript is authored text, not speech-recognition output.

The video helper targets one constant-frame-rate teaching clip. Generalizing to multiple recordings requires video-specific identities and output namespaces; variable-frame-rate media requires decoded presentation timestamps. The tutorials explain these boundaries rather than pretending that a small classroom helper is a production media platform.

Request-contract tests use controlled local responses. They cannot establish that a particular hosted model supports the chosen alias, correctly reads charts, resists every instruction embedded in source material, or produces well-supported answers. Those are separate live evaluation tasks.
