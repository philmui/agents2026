# Where the ideas came from—and what this module changes

This module is an original, expanded teaching treatment built from the supplied reference notebook:

`/Users/pmui/dev/AIM/FDE1/The-AI-Forward-Deployed-Engineer-Certification/03_Retrieval/sessions/S6_Multimodal_Retrieval.ipynb`

The reference uses a support-operations story, marimo UI helpers, six charts and six memos, and optional model calls. This module uses Jupyter notebooks and a newly generated fictional **Cedar School solar lab** corpus. The source material is instructional inspiration and an implementation reference; the new corpus, explanations, illustrations, exercises, and feature-store treatment are original additions.

Cell numbers below are **zero-based positions in the reference notebook's `cells` array**, including Markdown cells. They are not notebook execution counts. They are stable for the supplied file as inspected; editing that source file can change them.

## Learning sequence map

| Reference location | Reference idea | Expanded destination |
|---|---|---|
| Cells 1–4: introduction | Mixed media, parser versus reader | [01 Foundations](01_foundations.ipynb), [THEORY §1–3](THEORY.md#1-what-the-database-canand-cannotsee) |
| Cells 6–9: “The one-step version first” | LlamaIndex `MultiModalVectorStoreIndex` and local adapters | [01 Foundations](01_foundations.ipynb): inspect the framework shortcut, then unpack its responsibilities |
| Cells 10–12: “Task 1” | `RawRecord`, text/image source identity | [01 Foundations](01_foundations.ipynb), generated Cedar School corpus |
| Cells 13–20: “Task 2” | Vision parsing, reference parses, validation, `Chunk` | [02 Describe and search](02_describe_and_search.ipynb), [THEORY §3](THEORY.md#3-keep-parsing-separate-from-reading) |
| Cells 21–24: “Task 3” | Text embeddings, Qdrant payloads, Strategy A | [02 Describe and search](02_describe_and_search.ipynb), [THEORY §4](THEORY.md#4-strategy-a-describe-once-index-as-text) |
| Cells 25–29: “Task 4” | CLIP towers, Strategy B, modality gap | [03 Shared space](03_shared_space.ipynb), [THEORY §5](THEORY.md#5-strategy-b-words-and-images-in-a-learned-shared-space) |
| Cells 29–31: “Task 5” | Separate stores and reciprocal rank fusion | [04 Fusion and evaluation](04_fusion_and_evaluation.ipynb), [THEORY §6](THEORY.md#6-strategy-c-separate-searches-merge-by-rank) |
| Cells 32–36: “Task 6” | Retrieve first, then read original pixels | [02 Describe and search](02_describe_and_search.ipynb) and [04 Fusion and evaluation](04_fusion_and_evaluation.ipynb), [THEORY §7](THEORY.md#7-narrow-the-search-then-read-the-evidence) |
| Cells 37–40: “Task 7” | Gold questions, visual queries, comparisons | [04 Fusion and evaluation](04_fusion_and_evaluation.ipynb), [THEORY §8](THEORY.md#8-evaluate-whether-the-right-evidence-was-found) |
| Cells 40–47: “Task 8” | Timestamped transcript and keyframes | [05 PDF and video](05_pdf_and_video.ipynb), [THEORY §10](THEORY.md#10-video-is-evidence-along-a-timeline) |
| Cells 48–51: conclusion and additional context | Costs, failure analysis, application choices | Every notebook's interpretation/exercise sections, [THEORY §11–12](THEORY.md#11-choose-a-design-by-evidence-not-fashion) |
| Cells 19–20 and 23: metadata and provenance | Source paths, parse metadata, collection rebuild | [FEATURE_STORES.md](FEATURE_STORES.md): original extension, not a reference implementation |
| Reference notes on scanned documents, especially cell 31 | OCR and document handling | [05 PDF and video](05_pdf_and_video.ipynb): concrete native-text/raster-page contrast |

## Code excerpts and adaptations in the guides

| Location in the new guides | Reference cell | Relationship |
|---|---:|---|
| `Chunk` in THEORY and FEATURE_STORES | 20 | Definition copied verbatim; standalone import added |
| `payload={...}` in FEATURE_STORES | 23 | Exact payload expression from `build_collection` |
| `unit_vector` in THEORY | 22 | Adaptation with finite-value and zero-vector validation |
| Structured parse examples in THEORY | 14, 16 | Adapted data contract using the new corpus; no fabricated live API result |
| `PointStruct` example in THEORY | 23 | Abbreviated explanatory adaptation; tiny vector explicitly illustrative |
| FastEmbed paired-tower loading in THEORY | 27 | Short adaptation preserving the reference's matching model names |
| Source-level fusion example in THEORY | 30 | Adaptation with canonical identity and within-list duplicate guard |
| `FeatureRecord` and `feature_as_of` | 20, 23 motivate the extension | Original teaching code, not an existing reference feature-store implementation |
| PDF-page workflow and availability-time example | — | New explanatory material |

Use the executable notebook helpers for full experiments. Guide snippets are either standalone small calculations or explicitly marked patterns with placeholders. None of the illustrative vectors or worked-example scores is presented as measured pretrained-model output.

## Deliberate corrections and clarifications

### A vector database can store image embeddings

Reference cell 4 says a text embedding model cannot see and a vision model cannot be indexed. The first point is useful; the second needs precision. A vision-language generator is not automatically an embedding encoder, but suitable vision encoders do produce indexable vectors. The database does not see pixels; the encoder creates their searchable representation.

### Hit rate and recall are different metrics

Reference cell 37 averages `bool(wanted & got)` and labels the result recall@k. That calculates **hit rate**: whether at least one relevant source appears. General recall divides the count of retrieved relevant sources by the count of all relevant sources. The new lesson teaches both, includes multi-source examples, and separately evaluates answer grounding.

### A set of numbers cannot validate a chart interpretation

Reference cell 18 extracts strings with a numeric regular expression. Its docstring suggests `27`, `27.0`, and `27k` meet, but the function does not fully canonicalize those formats: `27.0` remains different from `27`, and `27k` loses its magnitude suffix. Set membership also discards multiplicity, label relationships, signs in some formats, and units. The new material emphasizes label–value–unit checks and tests label swaps. A clean check against a hand-written fixture does not measure a live parser's accuracy.

### Normalization does not calibrate unrelated models

Reference cell 22 normalizes vectors and describes scores as comparable. Unit normalization makes dot product equal cosine for nonzero vectors within a compatible space. It does not align unrelated encoders or make their scores calibrated probabilities. The new material checks invalid vectors and distinguishes within-space similarity from cross-model rank fusion.

### Character slicing is not token counting

Reference cell 27 slices text to 300 characters with a comment about CLIP's token limit. Characters and model tokens are different units. The new material describes finite token context and requires model-aware preprocessing rather than treating this slice as a guaranteed token limit.

### The modality gap is a phenomenon to measure

Reference cell 29 describes text results dominating a mixed search. That may occur, but it is not a guaranteed outcome for every model, dataset, or query. The new lesson treats mixed versus image-only search as an experiment and separates a shared coordinate system from calibrated cross-modal scores.

### Fusion depends on evidence identity

Reference cell 30 keys by `(modality, id)`. That prevents text and image representations with the same ID from merging. In video, two representations may describe the same time window. The new material explains canonical evidence IDs and retains provenance so a fused result still points to both kinds of evidence. Within-list duplicate crops should not create repeated independent votes. If the text and image stores contain disjoint sources, equal-rank results tie; agreement bonuses do not magically appear.

### Nearby frames are weaker evidence than aligned frames

Reference cell 43 chooses the nearest frame when no frame lies inside a transcript interval, and it assumes a nonempty frame list. The new module explains missing-frame handling, weak alignment, interval boundaries, and the limits of sparse sampling. The synthetic video is silent, and the supplied companion transcript is explicitly hand-authored rather than ASR output.

### Local execution has costs and prerequisites

Zero reported provider tokens do not imply zero compute, memory, energy, or download cost. Missing API usage reporting is also different from known-zero usage. The new material keeps local teaching baselines, downloaded pretrained models, and optional hosted calls visibly distinct.

### A teaching rebuild is not a production migration

Reference cell 23 deletes a collection before rebuilding it. That is a reasonable small-notebook convenience, but a production system needs appropriate validation, versioning, publication, retention, access controls, and recovery. The feature-store guide adds these concepts without claiming the classroom project implements them all.

## Original corpus and evidence boundaries

The new corpus uses invented school-lab measurements: energy (April 42, May 57, June 51 kWh), temperatures (Roof 46, Garden 34, Shade 27 °C), water use (Hose 18, Bucket 8, Mist 3 L), and an E17 controller screenshot. These are instructional fixtures. Memos provide safety, schedule, and quality-caution text. The PDF and video reuse this synthetic evidence so students can trace representations across media.

The source answer key is kept separate from searchable content. Reference parses are labeled as curated fixtures. API/model branches are described as optional, and execution results must state which backend actually ran. The feature-store examples and hand-calculated RRF lists are separate hypothetical examples, not additional experimental observations.

## Primary reading

The guides link primary documentation where it supports a specific concept: Qdrant points, FastEmbed image embedding, the original CLIP paper, the modality-gap study, the RRF paper, and Feast's point-in-time joins. Package setup and current API verification are documented with the module's main source list. Versions and provider-specific model availability can change; the locally pinned environment and actual execution report determine what was tested here.
