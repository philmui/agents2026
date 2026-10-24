# Multimodal retrieval: speaker notes

These notes expand the intentionally concise slides. Reference cell numbers are zero-based indices into `S6_Multimodal_Retrieval.ipynb`. Code snippets on slides are shortened architecture excerpts; run the complete notebook cells.

## 01. Search beyond words

Imagine your school is studying a solar panel. The project folder contains lab notes, energy charts, controller screenshots, a PDF report, and a short video. A question such as ‘Which month produced the most energy?’ needs the right chart and a careful reading of its labels. Multimodal means working across more than one kind of information. Retrieval means finding useful evidence. We will build the search stage, inspect where it fails, and then connect it to an optional vision reader. The school, measurements, and media in this lesson are fictional teaching data. The chart shows 42, 57, and 51 kWh; its answer is May, 57 kWh. Ask learners what evidence they would need before answering a different question: why did May produce more? The chart alone does not explain causes.

Notebook: [01_foundations.ipynb](../01_foundations.ipynb)

## 02. Evidence has several shapes

A modality is a type of input, such as text, an image, or audio. A PDF is a container: one page might hold selectable text, while another contains a scanned chart. A video combines visual frames with time and may also contain audio. Our corpus includes text memos, three charts, a controller screenshot, a mixed PDF, and a silent twelve-second video with a hand-authored companion transcript. This variety is deliberate: a cleaning policy belongs in a memo, a controller code belongs on a screenshot, and an event's location in a video needs timestamps. Counterexample: a JPEG file full of printed words is technically an image, but ordinary OCR followed by text search may solve the task better than a general vision model.

Notebook: [01_foundations.ipynb](../01_foundations.ipynb)

Reference notebook cells: 4, 10–11 (zero-based).

## 03. Search compares representations

A search engine does not compare your sentence directly with the pixels in a file. It first creates a representation: searchable words, a numerical vector, or both. A vector database stores those vectors and finds nearby ones. It can store image embeddings perfectly well; the limitation belongs to a text-only encoder, which cannot accept image pixels. Likewise, a generative vision model is not automatically a retrieval index. Its captions or supported embedding outputs must be turned into records with stable identities. A counterexample helps: merely saving a PNG path in a database payload does not make that image semantically searchable. The payload points to the evidence; an appropriate representation makes it findable.

Notebook: [01_foundations.ipynb](../01_foundations.ipynb)

Reference notebook cells: 4 (zero-based).

## 04. Direction can measure similarity

An embedding is a list of numbers that a model produces from an input. The individual coordinates are usually learned features, not human labels such as ‘amount of solar energy.’ Think of an arrow from the origin. Cosine similarity compares the directions of two arrows rather than their lengths. Parallel directions have similarity 1; perpendicular directions have 0; opposite directions have -1. In the notebook, small hand-built vectors let you compute every step before any model download. Caution: a high cosine score means the representation considers two items similar. It does not measure truth, entailment, or the probability that an answer is correct. The two-dimensional drawing is an explanation, not a visualization of actual model embeddings.

Notebook: [01_foundations.ipynb](../01_foundations.ipynb)

## 05. Dimensions do not define compatibility

Three numbers written in one coordinate system cannot safely be compared with three numbers from an unrelated one. A position measured from your classroom door differs from a position measured from the city center, even when both contain two numbers. Similarly, equal embedding dimension is necessary for a direct dot product but does not establish semantic compatibility. Query and document vectors must use the intended compatible encoders, model versions, and preprocessing. CLIP is special because its text and image encoders were jointly trained to align their outputs. Counterexample: a 512-dimensional vector from a random text model is not interchangeable with a 512-dimensional CLIP image vector. Record model identity as well as shape.

Notebook: [01_foundations.ipynb](../01_foundations.ipynb)

## 06. One model, two different jobs

Ingestion means adding material to the collection. At ingestion, a vision model can describe a screenshot, extract text, or turn a chart into a table. You store the result and reuse it across later queries. At question time, the reader receives only a few retrieved images and answers the user's specific question from those pixels. These are separate jobs even when the same model performs both. The caption may have forgotten a small axis label; the reader can inspect it if retrieval found the source. The reader cannot rescue an image that was never retrieved. Reparse when the source or extraction process changes, rather than assuming ‘once’ means forever. On our fixture path, descriptions are authored teaching records, not claimed vision-model outputs.

Notebook: [02_describe_and_search.ipynb](../02_describe_and_search.ipynb)

Reference notebook cells: 4, 14, 33–36 (zero-based).

## 07. Unpack the framework shortcut

The reference notebook opens with LlamaIndex's MultiModalVectorStoreIndex. It is a useful first picture of the system: provide documents, a text embedder, and a compatible multimodal embedder; the framework maintains representations and retrieval paths. The code here is an architectural sketch; the notebook supplies the actual adapters and configuration. In particular, image search needs the text tower of the same multimodal model, rather than the unrelated text-only embedder used for memo search. A framework saves plumbing but does not decide whether a chart is accurate, whether two scores are comparable, or whether a result contains the answer. The next lessons make those decisions visible. Our offline path does not silently download models or call a hosted service.

Notebook: [01_foundations.ipynb](../01_foundations.ipynb)

Reference notebook cells: 8 (zero-based).

## 08. Make the workshop reproducible

Open a terminal in evals/multimodal and run uv sync. This installs the dependencies from the external pyproject.toml and lockfile, rather than modifying the environment in hidden notebook cells. Start Jupyter with uv run jupyter lab. Copy .env.example to .env only when configuring optional models, and keep secrets out of notebooks and version control. Python uses load_dotenv(find_dotenv()) to find configuration. The default lessons run with deterministic teaching data and deliberately simple baseline vectors; they are not benchmark results for a pretrained CLIP model or a hosted vision model. The optional live path downloads local ONNX models and can call your configured vision endpoint. A fully reproducible comparison also records model files and preprocessing, not just Python package versions.

Notebook: [01_foundations.ipynb](../01_foundations.ipynb)

## 09. Keep the route back to evidence

The original chart and a text caption of that chart are different records describing the same underlying evidence. Keep a source_id such as chart_energy, a representation identifier such as caption-v1, the source path, and useful provenance. A PDF record also needs a page number; a frame record needs a timestamp or time window. Stable source identity lets us fuse rankings without counting the same chart twice and lets an answer cite what the user can inspect. The database may use its own point ID, which is separate from the domain-level source ID. Counterexample: treating every image crop as unrelated evidence can fill the top three results with three copies of one chart and leave no room for a useful memo.

Notebook: [01_foundations.ipynb](../01_foundations.ipynb)

Reference notebook cells: 11, 20, 23 (zero-based).

## 10. A · Describe once, search words

Strategy A converts each image into searchable language. The ingestion parser might write ‘Monthly solar energy: April 42 kWh, May 57 kWh, June 51 kWh.’ Those words can share an index with lab memos, using the same compatible text encoder for records and queries. Store the link to the image so a later reader can inspect the source. This approach is practical when important visual content can be described in words. Its weakness is information loss: retrieval sees only what the parser included. A vague caption such as ‘a bar chart’ may omit both solar energy and months. The default notebook uses transparent, authored descriptions to isolate search behavior; optional model parsing is labeled separately.

Notebook: [02_describe_and_search.ipynb](../02_describe_and_search.ipynb)

Reference notebook cells: 14–24 (zero-based).

## 11. A caption decides what survives

Read the chart, then hide it and inspect the caption ‘A monthly solar chart.’ That caption helps find the topic, but it cannot answer which month was highest or what the units mean. Adding every visible value preserves more information, but creates more opportunities for extraction errors. Captions are lossy representations: they choose some details and discard others. OCR aims to recover visible text; a description also summarizes visual relationships; a chart-to-table extraction aims to preserve structured values. Those outputs have different uses. Do not infer that a rich caption is automatically correct. If a label is ambiguous, preserve the uncertainty and keep the source image available for inspection.

Notebook: [02_describe_and_search.ipynb](../02_describe_and_search.ipynb)

Reference notebook cells: 14–18 (zero-based).

## 12. Valid JSON can still be wrong

A parser returning {"month": "May", "value": 75, "unit": "kWh"} might satisfy the expected JSON schema while misreading the chart, whose value is 57. Schema validation checks structure and types. It cannot establish visual truth. Our generated corpus has an answer key, so a lesson can compare extracted numbers with known values. Real uploaded documents often do not have an answer key: use source inspection, cross-checks, trusted tables, or human review. A range check catches a negative energy value but cannot prove that 75 should be 57. Keep statuses such as verified, unverified, and invalid distinct. A second model agreeing is supporting evidence, not a guarantee of correctness.

Notebook: [02_describe_and_search.ipynb](../02_describe_and_search.ipynb)

Reference notebook cells: 15–18 (zero-based).

## 13. Store vectors; keep source metadata

Qdrant stores a vector plus a payload. The vector supports similarity search; the payload holds identifiers, modality, source paths, and provenance. A collection fixes its vector dimension and similarity metric. Query vectors must match the collection's intended encoder and dimension. The code sketch shows the query_points method used by current qdrant-client: its response contains points rather than being a raw list. In-memory local mode is ideal for a small exercise, but its records disappear when the process ends. Rebuilding after a notebook restart is expected. A persistent deployment needs explicit storage and lifecycle decisions. Metadata can also support filters; those filters should restrict eligible evidence before the result reaches a reader.

Notebook: [02_describe_and_search.ipynb](../02_describe_and_search.ipynb)

Reference notebook cells: 23 (zero-based).

## 14. Predict where descriptions will fail

The controller screenshot contains the exact code E17 and the warning that ventilation is blocked. A description mentioning blocked ventilation may retrieve the screenshot for ‘Why should we clear the vent?’ Exact code queries are more fragile: the parser may omit E17, a text embedding may blur it with E71, and the reader may mistake a small character. Add lexical matching or exact fields when identifiers matter. Another failure is unmentioned visual appearance: ‘the green chart’ cannot match a caption that never names color. The exercise is to name one caption omission, predict a query it will hurt, change only that caption, and rerun the query. That is a small controlled experiment, not evidence that A is universally best or worst.

Notebook: [02_describe_and_search.ipynb](../02_describe_and_search.ipynb)

## 15. B · Put words beside pictures

Strategy B embeds images and text into a compatible shared space. A text question passes through the model's text tower; an image passes through its image tower. Similarity can then rank images directly without requiring a caption first. The shared space is learned rather than manually specified. The notebook begins with an intentionally weak, color-only pixel feature exercise and manually mapped color queries, so everyone can run it offline and see its limits. It is not CLIP, not a semantic substitute, and not a performance benchmark. The optional FastEmbed path loads the actual paired CLIP encoders. The plotted points are a conceptual two-dimensional drawing; real embeddings have many dimensions and their geometry is not fully visible in this sketch.

Notebook: [03_shared_space.ipynb](../03_shared_space.ipynb)

Reference notebook cells: 25–29 (zero-based).

## 16. Training pulls matching pairs closer

CLIP was trained with image-text pairs. In a training batch, the model learns to give matching pairs higher compatibility than mismatched pairs. The text and image towers become translators into a related coordinate system. We use a pretrained model; we do not train CLIP in these notebooks. Shared space does not mean every capability is shared equally. Natural-image concepts can work well while small text, exact counts, fine spatial relations, and chart numerals remain difficult. It also does not guarantee that text records and image records have identical score distributions. Counterexample: recognizing that a picture is an energy bar chart is a different task from reading that May is 57 kWh. Retrieval can be useful without solving the second task.

Notebook: [03_shared_space.ipynb](../03_shared_space.ipynb)

Reference notebook cells: 25–27 (zero-based).

## 17. FastEmbed runs the paired towers

FastEmbed uses ONNX model files for local inference, allowing these embedding examples to avoid a large PyTorch setup. The two Qdrant CLIP model identifiers in the sketch select the intended text and vision towers. The first live run still downloads weights; ‘local inference’ does not mean ‘nothing is ever downloaded.’ After caching the weights, inference processes the inputs locally. Pin model versions or artifacts when reproducing results, and preserve preprocessing such as image resizing. TextEmbedding also supports text-only models; naming an arbitrary one here would break the shared-space design. The complete notebook guards the optional live path and reports missing models or blocked downloads instead of presenting fallback color features as CLIP results.

Notebook: [03_shared_space.ipynb](../03_shared_space.ipynb)

Reference notebook cells: 27 (zero-based).

## 18. Normalize before a dot product

The length of [3, 4] is 5 by the Pythagorean theorem. Dividing by 5 gives [0.6, 0.8], an arrow with length 1 pointing in the same direction. Once both vectors have length 1, their dot product equals cosine similarity. In numpy, axis=-1 computes one length per vector and keepdims=True preserves the dimensions needed for division. A zero vector has no direction, so reject it or define a clear fallback rather than quietly calling its cosine meaningful. Qdrant's cosine collection applies its own normalization behavior, while our numpy exercise makes the math explicit. Caution: normalizing does not make unrelated encoders compatible, and it does not calibrate scores into probabilities.

Notebook: [03_shared_space.ipynb](../03_shared_space.ipynb)

Reference notebook cells: 22, 27 (zero-based).

## 19. Finding a chart is not reading it

Ask two different questions: ‘Find the monthly solar-energy chart’ and ‘How much more did May produce than April?’ CLIP may help with the first. The second requires reading 57 and 42, checking the kWh unit, and subtracting to obtain 15 kWh. A high retrieval similarity does not perform that arithmetic or certify the extracted values. Even a vision reader should cite the chart and qualify any illegible label. In the fixture lesson, known chart data makes the expected answer checkable. In a real report, you may prefer a trusted data table or human verification for precise numbers. A useful error analysis labels separately whether the image was missed, the values were misread, or the calculation was wrong.

Notebook: [03_shared_space.ipynb](../03_shared_space.ipynb)

## 20. One space can favor one modality

The two towers can share compatible coordinates while text-text and text-image similarities follow different distributions. If all records compete in one mixed ranking, one modality may dominate the top results. Do not assume this must happen in your corpus; measure it. Examine representative queries whose gold evidence is a memo, an image, or both. A modality filter can reveal whether an image was useful but crowded out. Separate retrieval quotas or Strategy C are possible responses, each with tradeoffs. Counterexample: always reserving two image slots may hurt questions answered entirely by a safety memo. The notebook's simple color baseline cannot establish how a real CLIP model behaves; live results must be reported separately.

Notebook: [03_shared_space.ipynb](../03_shared_space.ipynb)

Reference notebook cells: 28–29 (zero-based).

## 21. C · Let each specialist search

Strategy C keeps a text-specialist collection and an image collection with its paired text query encoder. The two paths search independently. Each returns a ranked list of source IDs, and a fusion rule combines those lists. This keeps unrelated vector spaces separate. The image path may use CLIP, while the text path uses a model optimized for prose. In the offline teaching path the image baseline is intentionally weaker and explicitly labeled. Fusion can combine complementary evidence but cannot recover an item absent from every candidate list. The candidate pool size and final answer context size are different parameters: retrieve enough candidates for fusion, then send only the most useful sources to the reader.

Notebook: [04_fusion_and_evaluation.ipynb](../04_fusion_and_evaluation.ipynb)

Reference notebook cells: 29–31 (zero-based).

## 22. Raw scores use different rulers

Suppose the top text result has similarity 0.82 and the top image result has 0.31. It is tempting to declare the text result better. But these numbers come from different models and may have different distributions. They are not probabilities on a shared scale. One model can produce narrow score ranges and another broad ranges. Reciprocal rank fusion avoids that comparison by using order within each list. Proper score calibration is another approach, but it needs suitable labeled data and validation. Counterexample: multiplying image scores by three because one example looked too low is an arbitrary adjustment that may damage another query. These displayed numbers are a hypothetical demonstration, not measured scores from our notebooks.

Notebook: [04_fusion_and_evaluation.ipynb](../04_fusion_and_evaluation.ipynb)

Reference notebook cells: 29–31 (zero-based).

## 23. RRF adds rewards for good ranks

Reciprocal rank fusion, or RRF, rewards sources that appear near the top of one or more ranked lists. For every list containing source d, add 1 divided by c plus its rank. Start ranks at 1. A source missing from a list contributes zero for that list, not an invented last rank. The smoothing constant c controls how strongly first place beats lower places; 60 is a common starting value, not a universal optimum. We use c rather than k in this explanation so it is not confused with the number of results returned. Rank fusion deliberately discards similarity magnitudes. It can be robust across incompatible score scales, but does not prove relevance or require the lists to be statistically independent.

Notebook: [04_fusion_and_evaluation.ipynb](../04_fusion_and_evaluation.ipynb)

Reference notebook cells: 30 (zero-based).

## 24. Count each source once per list

A chart can be returned through several crops or captions. If one retrieval list counts every copy as separate support, a duplicated source unfairly gains score. Deduplicate each list by the chosen canonical source ID while preserving first occurrence, then assign one-based source ranks. The same source may legitimately contribute once from the text list and once from the image list. Decide the unit of relevance before evaluation: a PDF page, whole document, or video window may be the right unit for different tasks. Overly broad grouping can also be harmful; collapsing all video frames into a single video ID destroys temporal distinctions. The notebook helper makes the policy explicit and keeps provenance for the selected evidence.

Notebook: [04_fusion_and_evaluation.ipynb](../04_fusion_and_evaluation.ipynb)

Reference notebook cells: 30 (zero-based).

## 25. Work one fusion example by hand

Use c=60 and the two displayed lists. This example assumes captions let the chart and screenshot appear in the text list as well as the image list. If a text store contains only memos and its IDs never overlap image IDs, no source receives two contributions. The chart receives 1/62 from text and 1/61 from images, totaling about 0.032522. The screenshot receives 1/63 twice, totaling about 0.031746. The memo gets 1/61, about 0.016393, while the frame gets 1/62, about 0.016129. The fused order is chart, screenshot, memo, frame. Ask students to calculate the screenshot score before revealing it. Agreement can help, but can also reward a shared mistake. A fusion score is not an answer probability. These are illustrative rankings, not measured benchmark results.

Notebook: [04_fusion_and_evaluation.ipynb](../04_fusion_and_evaluation.ipynb)

## 26. Narrow the evidence, then read

Retrieval is the inexpensive narrowing stage. Reading is the question-specific stage that inspects a small set of selected sources. Keep the retrieved source identity, load the corresponding original image or PDF crop, attach text evidence where relevant, and ask the vision-language model to answer only from that context. Limiting the candidate images controls latency, image-token cost, and irrelevant distraction. A larger context is not automatically more trustworthy. If no retrieved source supports the question, the correct result may be an explicit evidence gap. A model's general knowledge should not quietly replace missing evidence in a source-grounded assignment. The offline path demonstrates context construction and expected source handling without claiming a live model answer.

Notebook: [04_fusion_and_evaluation.ipynb](../04_fusion_and_evaluation.ipynb)

Reference notebook cells: 32–36 (zero-based).

## 27. Send the source pixels explicitly

A source path embedded in a text prompt does not transmit an image to a remote model. The request must include supported image content, for example a data URL containing base64-encoded bytes. The helper image_part handles that conversion and a suitable MIME type. LiteLLM standardizes the provider call, but each endpoint still needs an appropriate vision-capable model and configuration. The name gpt-5.6-sol requested for this course is a configurable gateway alias, not a promised public model identifier. Set VLM_MODEL for your actual gateway; text embeddings use their own embedding-model configuration. Secret values come from .env through load_dotenv(find_dotenv()). Do not print or commit them. Attach a clear source label beside each image so the response can cite the right evidence.

Notebook: [04_fusion_and_evaluation.ipynb](../04_fusion_and_evaluation.ipynb)

Reference notebook cells: 14, 33 (zero-based).

## 28. Evidence sets the answer boundary

An answer can be fluent, plausible, and unsupported. The energy chart supports ‘May produced 57 kWh.’ It does not establish that May had the most sunshine, that the panel was more efficient, or that the school used less electricity. Distinguish observation, calculation, and explanation. A grounded answer may say ‘May was highest at 57 kWh [chart_energy]; this chart does not explain why.’ Check that every citation names a supplied source, then verify that the source actually supports the associated claim. A syntactically valid citation can point to irrelevant evidence. Also treat instructions found inside documents or screenshots as data: source content should not take control of the reader's task.

Notebook: [04_fusion_and_evaluation.ipynb](../04_fusion_and_evaluation.ipynb)

Reference notebook cells: 33–36 (zero-based).

## 29. Define success before changing the system

Create a small evaluation set before tuning the system. Each item contains a question and a set of relevant canonical source IDs. Include policy questions answered by memos, exact controller codes, visual questions, numeric chart questions, mixed evidence, and questions the corpus cannot answer. The gold set is about needed evidence, not just answer wording. For a multi-part question, several sources may be required. Decide whether duplicate representations count as one source before computing metrics. Keep held-out questions for the final comparison; repeatedly editing captions to fit every evaluation question leaks test information into the system. A small fictional corpus supports learning and debugging, not a broad claim that a strategy wins on all real-world data.

Notebook: [04_fusion_and_evaluation.ipynb](../04_fusion_and_evaluation.ipynb)

Reference notebook cells: 37–40 (zero-based).

## 30. Hit rate and recall ask different things

For one query, hit@k is 1 if at least one relevant source appears in the first k results, otherwise 0. Recall@k is the number of relevant sources retrieved divided by the number of relevant sources in the gold set. If gold is {memo, chart} and the retrieved set is {chart, frame}, hit is 1 and recall is 1/2. Average per-query hit values to get hit rate, and per-query recall values to get macro recall. With exactly one relevant source per question, these values coincide, which can hide an incorrect metric name. The reference notebook labels an any-hit calculation as recall; this lesson separates them. For questions with no relevant source, recall has a zero denominator: evaluate abstention separately rather than silently assigning an arbitrary recall.

Notebook: [04_fusion_and_evaluation.ipynb](../04_fusion_and_evaluation.ipynb)

Reference notebook cells: 37 (zero-based).

## 31. Compare like with like

A fair A/B/C comparison uses the same corpus, gold questions, source identity policy, and final k. Record candidate-pool sizes and model choices because they can also affect results. Fixture results demonstrate pipeline behavior with authored captions and simple vectors. Live results use real embedding or vision models and must be labeled as such. Never replace an unavailable model with a weaker fallback and keep the original model's name in the score table. Retrieval quality and answer quality are separate: a correct source can be misread, while a model can guess a correct answer without retrieving evidence. Also report costs relevant to the decision, such as ingestion calls, query latency, and images sent per answer. This deck intentionally shows no invented benchmark comparison.

Notebook: [04_fusion_and_evaluation.ipynb](../04_fusion_and_evaluation.ipynb)

Reference notebook cells: 37–40 (zero-based).

## 32. Turn a failure into an experiment

Suppose E17 queries miss the controller screenshot. First inspect whether the caption contains E17. If it does not, the likely cause is extraction or representation. If it does, inspect the candidate ranking; exact lexical matching may help more than another vision call. If the screenshot is retrieved but the answer says E71, the problem is reading. Change one component at a time and rerun the same question plus neighboring cases. A single successful retest is not enough if the change harms other queries. Useful controlled experiments include adding exact-code fields, increasing candidate depth before fusion, comparing transcript-only with transcript-plus-frames, or changing the reader's image resolution. Preserve observations and uncertainty instead of explaining every failure with ‘the model is bad.’

Notebook: [04_fusion_and_evaluation.ipynb](../04_fusion_and_evaluation.ipynb)

## 33. A PDF may need two routes

PDF is a format, not a single modality. Native text can often be extracted directly; scanned pages contain pixels and need OCR or visual processing. A chart may contain selectable axis labels but still require its visual arrangement to interpret it. Our tiny PDF has a native-text first page and a raster chart on page two so you can test both routes. Render pages at a readable resolution and use Pillow to inspect or crop images. Retain document ID and page number. A common counterexample is a two-column PDF whose extracted text interleaves unrelated sentences: text exists, but its reading order is broken. Inspect extracted output before treating it as reliable prose.

Notebook: [05_pdf_and_video.ipynb](../05_pdf_and_video.ipynb)

Reference notebook cells: 1, 4, 49 (zero-based).

## 34. A useful PDF citation is inspectable

A citation such as ‘the report’ is hard to check when the document is long. Preserve a document identifier, a user-facing page number, and optionally a crop or bounding box. Be explicit about indexing: PDF libraries often use zero-based page indices, while readers expect page 1, page 2, and so on. Region coordinates must record their coordinate system and page dimensions, especially if images are resized or rotated. A crop helps focus a reader but can remove the title, legend, or unit that gives the chart meaning. Keep enough surrounding context and a link to the full page. In this corpus, the energy chart is on user-facing page 2.

Notebook: [05_pdf_and_video.ipynb](../05_pdf_and_video.ipynb)

## 35. Video aligns two evidence streams

A video search system can index speech transcripts as text and sampled frames as images, preserving their time alignment. Our twelve-second teaching video is silent. The accompanying text is hand-authored, not speech-recognition output, so it demonstrates alignment without pretending to transcribe absent audio. It shows energy from 0 to 4 seconds, the controller from 4 to 8, and temperature from 8 to 12, using half-open intervals in code. A real recording could use automatic speech recognition, but speech and visuals may disagree or appear at different times. Search both streams when they add value; first measure whether a transcript alone answers the target questions. Neither a transcript nor a single frame captures everything in a video.

Notebook: [05_pdf_and_video.ipynb](../05_pdf_and_video.ipynb)

Reference notebook cells: 41–45 (zero-based).

## 36. Sampling saves work and loses moments

This is an alternative seeking sketch inspired by the reference. The runnable notebook uses sequential decoding and frame_number/fps timestamps for the known constant-frame-rate teaching clip. OpenCV can also seek to a target time and read a frame, as shown here. Always check the success flag: the read may fail or the target may be beyond the last frame. Seeking and reported timing are approximate; use actual presentation timestamps for variable-frame-rate media. Sampling every few seconds reduces work but can miss a brief warning. Scene-change detection and near-duplicate removal are alternatives, each with errors. The dot at 12 seconds marks the video boundary, not a guaranteed decodable frame. Convert OpenCV's BGR channel order before displaying through Pillow or another RGB-based viewer.

Notebook: [05_pdf_and_video.ipynb](../05_pdf_and_video.ipynb)

Reference notebook cells: 42–44 (zero-based).

## 37. A time window is not event proof

Retrieving the controller window supports saying that the relevant scene is in the 4–8 second interval. It does not automatically prove the warning first appeared at exactly 4.000 seconds. A sampled frame shows what was visible at its decoded time; a transcript segment shows words assigned to a time range. They are different observations. For an onset question, inspect adjacent frames at a finer interval and report uncertainty. Formatting a timestamp as minutes and seconds is useful but does not improve temporal accuracy. Keep both t_start and t_end in metadata, and cite the supported window. Counterexample: a narrator saying ‘the warning appeared earlier’ during the current scene does not place the warning's onset at that sentence's timestamp.

Notebook: [05_pdf_and_video.ipynb](../05_pdf_and_video.ipynb)

Reference notebook cells: 44–47 (zero-based).

## 38. Features need a lifecycle too

A feature is a derived property of an entity. In this project, features include OCR text, image dimensions, captions, and embeddings. A feature store manages reusable feature definitions and values, often with versioning, freshness, offline computation, and online serving. A vector database specializes in searching vectors and returning associated metadata. The concepts can connect, but they are not interchangeable, and this tutorial does not require installing a production feature platform. The local manifest is a small feature-registry teaching pattern. Keep the raw source separately; record how its features were produced; publish compatible vectors into the search index. Counterexample: a folder of arrays without source IDs or encoder versions is storage, but not a reliable feature lifecycle.

Notebook: [05_pdf_and_video.ipynb](../05_pdf_and_video.ipynb)

Reference notebook cells: 11, 20, 23, 27 (zero-based).

## 39. Reproduce what was known then

Suppose a lab observation happened at 10:00, but its verified chart caption became available at 10:20. A simulated query at 10:10 must not use that caption, even though the underlying event happened earlier. Track event time and availability time when historical correctness matters. This is the point-in-time problem familiar from feature stores: a later correction or reprocessing result should not leak into an earlier evaluation. Our small notebook records versions and demonstrates the principle; it is not a production point-in-time join engine. Also freeze the corpus and gold labels when comparing releases. Otherwise a model change may receive credit for a newer, easier corpus rather than better retrieval.

Notebook: [05_pdf_and_video.ipynb](../05_pdf_and_video.ipynb)

## 40. Changing an encoder changes the index

A new embedding model can move every item to a different coordinate system. Reusing old document vectors with new query vectors is unsafe even when their dimensions match. Version the encoder identity, model artifact or revision, preprocessing, parser prompt, source checksum, and output schema. Build and test a new index, then switch the query encoder and index together. Reparse only when that stage's inputs or logic changed; re-embed when the embedding representation changed. A source deletion should remove its searchable derived records and prevent stale citations. A lockfile fixes package versions, but model weights and prompts are separate artifacts. Counterexample: changing image cropping can alter the feature representation without changing the model name at all.

Notebook: [05_pdf_and_video.ipynb](../05_pdf_and_video.ipynb)

Reference notebook cells: 48 (zero-based).

## 41. Design your own controlled comparison

Choose a question family for the school corpus, such as exact controller codes, chart comparisons, or video scene location. Write your expected relevant sources before changing the system. Compare two retrieval strategies with fixed corpus, query set, and final k. Keep the fixture and live-model reports separate. Inspect one success and one failure in detail, then change only one component and rerun the evaluation. Your report should include the hypothesis, model and representation versions, source identity policy, retrieval metrics, answer-support checks, and the cost that matters for your setting. Add an unanswerable question and check abstention. A strong conclusion can be that OCR plus text retrieval is enough; complexity is justified only when it helps the actual task.

Notebook: [05_pdf_and_video.ipynb](../05_pdf_and_video.ipynb)

Reference notebook cells: 49–51 (zero-based).

## 42. Find evidence. Read carefully. Measure.

You now have a map of multimodal retrieval. Strategy A turns images into searchable words. Strategy B uses paired encoders with a shared space. Strategy C lets specialized paths retrieve independently and fuses their ranks. Parsing creates reusable representations; reading examines selected evidence for a specific question. PDF pages need provenance, video needs honest timestamps, and derived features need versioning. The next useful step is not automatically a larger model. It is a carefully chosen question, a clearly identified source, and a test that reveals where your pipeline succeeds or fails. Open the notebooks in order and record observations as you go. The theory guide explains the concepts in full, while FEATURE_STORES.md develops the lifecycle and historical-evaluation connections.

Notebook: [01_foundations.ipynb](../01_foundations.ipynb)
