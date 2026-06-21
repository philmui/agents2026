# Module 08 · More Retriever Metrics — Explanation

> Audience: advanced high-school researchers following the ASDRP Agentic RAG Evaluation track.
> Tone: clear, concrete, encouraging, honest about where ideas break.

## Where this module sits

Module 07 introduced the first pair of retriever metrics — LLMContextPrecisionWithReference and
LLMContextRecall — which score whether the retrieved passages are relevant and complete at the
*passage* level. Module 08 digs one level deeper with two complementary metrics:
**ContextEntityRecall** asks whether those passages actually contain the specific named entities
the correct answer requires, and **NoiseSensitivity** asks whether any irrelevant passages that
slipped through are actively misleading the generator. Together, all four metrics from Modules 07
and 08 form a complete retriever health check. Module 09 will then turn the lens on the
*generator* side — Faithfulness, ResponseRelevancy, and FactualCorrectness.

## The big idea

### ContextEntityRecall — did retrieval capture the key named entities?

Passage-level recall tells you whether the right *chunks* came back. Entity recall tells you
whether those chunks carry the right *facts*. For a metals-markets corpus the critical entities
are things like "troy ounce", "LBMA", "contango", "gold-silver ratio", "South Africa", "platinum",
"allocated metal". A passage might be retrieved as relevant yet still fail to mention a specific
unit or instrument the question requires.

The metric works as follows: a judge LLM reads the reference answer and extracts every named
entity it can find. It then checks how many of those entities appear somewhere in the retrieved
context passages. The score is the fraction that were found:

```
ContextEntityRecall = |entities found in context| / |entities in reference answer|
```

Higher is better (range 0 – 1). A score of 0.64 means roughly two-thirds of the entities the
answer needed were present in the retrieved context. See the diagram in
`slides/assets/05_context_entities_recall.svg`.

### NoiseSensitivity — how badly does irrelevant context mislead the generator?

Even a well-tuned retriever returns `k = 10` candidates, some of which are marginally relevant or
off-topic. NoiseSensitivity measures the downstream damage: how often does the generator produce
an *incorrect claim that can be traced back to a noisy retrieved passage*?

The judge LLM:
1. Compares the generated answer to the reference and identifies every incorrect or unsupported
   claim.
2. For each such claim, checks whether it appears to derive from an irrelevant (noisy) passage
   in the retrieved context.
3. Returns the proportion of answer statements that are incorrect *and* grounded in noise.

See the diagram in `slides/assets/06_noise_sensitivity.svg`.

## ⚠ Caution — NoiseSensitivity uses an INVERTED scale

**This is the most important thing to remember from this module.**

Every other metric in the retrieval suite (Precision, Recall, EntityRecall) and most generator
metrics follow the same direction: *higher is better*. NoiseSensitivity is the exception.
A score of **0.0 is ideal** — the generator was never misled by noisy context. A score of **1.0
is the worst possible** — every incorrect claim in the answer came from noise.

Consequences of forgetting this:
- Sorting results descending and declaring the "top" questions "best" will actually show you the
  *worst* questions.
- A sudden improvement in NoiseSensitivity looks like a rising number, but it is actually the
  score *falling* toward zero.
- Dashboard colour codings that use green-for-high will display backwards for this metric.

Always label the column explicitly and add a comment in any code that reads it:
```python
# NoiseSensitivity: LOWER is better
df.sort_values("noise_sensitivity_relevant", ascending=True)   # best at top
```

## Cost note

Both metrics require LLM judge calls (entity extraction for ContextEntityRecall, claim tracing
for NoiseSensitivity). Running 8 golden questions with both metrics costs roughly **16–24 judge
calls** depending on answer length. On the shared cloud Ollama endpoint this is fast and cheap,
but if you are iterating quickly use `single_turn_ascore()` on a single sample first to verify
the setup before running the full `evaluate()`.

## Code preview

### 1. Stub the missing RAGAS import and patch the event loop

```python
import sys, types
_vx = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI:
    pass
_vx.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vx

import nest_asyncio; nest_asyncio.apply()
```

### 2. Load keys and set up RAGAS judge + embeddings

```python
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HAVE_KEYS = bool(os.environ.get("OLLAMA_API_KEY"))

import litellm
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

JUDGE_MODEL     = "ollama_chat/gemma4:31b-cloud"
EMBEDDING_MODEL = "ollama/qwen3-embedding:0.6b"
judge_llm        = llm_factory(JUDGE_MODEL, provider="litellm", client=litellm.completion, temperature=0.0)
ragas_embeddings = embedding_factory("litellm", model=EMBEDDING_MODEL, api_base=os.environ["OLLAMA_API_BASE"])
```

### 3. Run ContextEntityRecall

```python
from ragas.metrics import ContextEntityRecall
from ragas import evaluate

results = evaluate(
    dataset=eval_dataset,
    metrics=[ContextEntityRecall()],
    llm=judge_llm,
    embeddings=ragas_embeddings,
)
print(results.to_pandas()[["user_input", "context_entity_recall"]])
```

### 4. Run NoiseSensitivity (remember: lower is better)

```python
from ragas.metrics import NoiseSensitivity

# NoiseSensitivity: LOWER is better
results = evaluate(
    dataset=eval_dataset,
    metrics=[NoiseSensitivity()],
    llm=judge_llm,
    embeddings=ragas_embeddings,
)
df = results.to_pandas()
df.sort_values("noise_sensitivity_relevant", ascending=True)  # best at top
```

### 5. Single-sample debug

```python
import asyncio
score = asyncio.run(
    ContextEntityRecall(llm=judge_llm).single_turn_ascore(samples[3])
)
print(f"Entity recall for Q4: {score:.3f}")
```

## Notebook preview

| Step | What you do |
| ---: | --- |
| 0.1 | Install libraries with `uv sync` |
| 0.2 | Load `OLLAMA_API_KEY` from shared `tutorials/.env` |
| 1 | Stub the RAGAS VertexAI import + apply nest_asyncio |
| 2 | Build RAGAS judge LLM and embeddings |
| 3 | Load corpus, build vector store, assemble `eval_dataset` |
| 4 | Run `ContextEntityRecall` on all 8 golden questions |
| 5 | Run `NoiseSensitivity` on all 8 golden questions |
| 6 | Run both metrics together with `evaluate()` |
| 7 | Inspect results; use single-sample debug to investigate a low score |

## References

- **Capstone theory doc**: `topics/06_rag_eval/agentic_rag_evaluation_theory.md` — sections on
  Context Entity Recall and Noise Sensitivity.
- **Capstone notebook**: `topics/06_rag_eval/agentic_rag_evaluation_tutorial.ipynb` — the
  `retriever_metrics` evaluation cell (search for `ContextEntityRecall`).
- **RAGAS docs**: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_entity_recall/
- **RAGAS docs**: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/noise_sensitivity/
- **Module 07** (`07_retriever_metrics/`) — covers Precision and Recall; this module builds on top.
- **Next module**: `09_generator_metrics/` — Faithfulness, ResponseRelevancy, FactualCorrectness.
