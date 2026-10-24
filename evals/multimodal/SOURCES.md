# Where the ideas and APIs come from

The teaching explanations, diagrams, fictional corpus, and exercises in this module are new. The starting point is the user-supplied **S6_Multimodal_Retrieval.ipynb**, at:

```text
/Users/pmui/dev/AIM/FDE1/The-AI-Forward-Deployed-Engineer-Certification/
03_Retrieval/sessions/S6_Multimodal_Retrieval.ipynb
```

[SOURCE_MAP.md](SOURCE_MAP.md) identifies its cells, distinguishes excerpts from adaptations, and maps them to the expanded lessons. You do not need that separate repository to run this course.

The links below are primary papers or maintainers' documentation, checked on **2026-10-21**. The installed package versions are recorded in [uv.lock](uv.lock). A documentation page can describe a newer API than a locked environment; the execution record in [VALIDATION.md](VALIDATION.md) says what was actually tested.

| Concept or tool | Source | Use in this course |
|---|---|---|
| CLIP | [Radford et al., 2021](https://arxiv.org/abs/2103.00020) | Paired image/text training and shared coordinates; not a guarantee of precise chart reading |
| Modality gap | [Liang et al., NeurIPS 2022](https://arxiv.org/abs/2203.02053) | Why a shared space can still have different image/text score distributions |
| Reciprocal rank fusion | [Cormack, Clarke, and Büttcher, SIGIR 2009](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf) | Merge ranks from different retrieval systems; author-hosted paper |
| Qdrant client | [Official repository](https://github.com/qdrant/qdrant-client) | In-process local collections without a database server |
| Qdrant query API | [Search documentation](https://qdrant.tech/documentation/search/) | `query_points`, vectors, scores, and payloads |
| Qdrant filters | [Filtering](https://qdrant.tech/documentation/concepts/filtering/) | Metadata restrictions before candidate selection |
| FastEmbed | [Supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/) | BGE text encoder and matching CLIP text/vision towers |
| FastEmbed images | [Image embedding examples](https://qdrant.github.io/fastembed/examples/Image_Embedding/) | Image preprocessing and local ONNX inference |
| LlamaIndex | [MultiModalVectorStoreIndex API](https://developers.llamaindex.ai/python/framework-api-reference/indices/vector/) | Framework preview and explicit adapters |
| LiteLLM vision | [Vision inputs](https://docs.litellm.ai/docs/completion/vision) | Text and image content parts for parsing and reading |
| LiteLLM embeddings | [Embedding API](https://docs.litellm.ai/docs/embedding/supported_embedding) | Separate text embedding model and endpoint |
| NumPy | [Linear algebra reference](https://numpy.org/doc/stable/reference/routines.linalg.html) | Norms, normalization, dot products |
| Pillow | [Image reference](https://pillow.readthedocs.io/en/stable/reference/Image.html) | Original images, pixel arrays, and image loading |
| PDFium | [pypdfium2 Python API](https://pypdfium2.readthedocs.io/en/stable/python_api.html) | Page rendering and embedded text extraction |
| OpenCV | [VideoCapture reference](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) | Decode frames and preserve their timestamps |
| Feature stores | [Feast concepts](https://docs.feast.dev/getting-started/concepts) | Entities, features, historical and online retrieval |
| Point-in-time features | [Feast point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins) | Avoid using future information in historical examples |
| Environment management | [uv project guide](https://docs.astral.sh/uv/guides/projects/) | External `pyproject.toml`, `uv sync`, lockfile |
| Secret loading | [python-dotenv](https://bbc2.github.io/python-dotenv/) | `load_dotenv(find_dotenv())` |

## Boundaries worth keeping

The two CLIP towers must belong to the same trained model family and revision. A 512-dimensional vector from an unrelated model is not compatible merely because its length matches. FastEmbed's support for a model and its downloadable weights are separate from this course's default local demonstrations.

`gpt-5.6-sol` is the requested configurable vision model name. An endpoint may expose it as an alias; this course does not establish availability, price, or image support on your account. Configure the provider and endpoint explicitly. A vision-language model used through `completion()` is not the text embedding model used through `embedding()`.

Feast is discussed for context; it is not installed or claimed to be implemented here. The small feature-history examples teach the time and lineage requirements that a production feature platform must enforce.
