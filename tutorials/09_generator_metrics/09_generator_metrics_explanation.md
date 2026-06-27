# Module 09 · Generator Metrics + LLM-as-Judge — Explanation

> Audience: advanced high-school researchers. Tone: clear, concrete, encouraging,
> honest about where ideas break.

## Where this module sits

Modules 07 and 08 gave you four *retriever* metrics: Context Precision,
Context Recall, Context Entity Recall, and Noise Sensitivity. Those scores
tell you how good the search step is — did the right passages end up in the
context window? Module 09 shifts attention to the other half of the pipeline:
the **generator**, the language model that reads those passages and writes an
answer. The three new metrics here — Faithfulness, Response Relevancy, and
Factual Correctness — measure the quality of that final answer. In Module 12
(the capstone) every metric from both sides of the pipeline runs together so
you can see the full picture.

---

## The big idea

### Faithfulness — does the answer stick to the sources?

A faithful answer is one where every claim can be traced back to at least one
of the retrieved context passages. RAGAS implements this by asking the judge
LLM to decompose the answer into atomic statements, then checking each one
against the contexts.

Score = (statements that are grounded in the context) ÷ (all statements)

A score of 1.0 means every sentence in the answer is directly supported. A
lower score means the model hallucinated — it made claims the retrieved
passages never said. See `slides/assets/07_faithfulness.svg`.

### Response Relevancy — is the answer actually about the question?

A faithful answer could still miss the point entirely: all its statements are
grounded, but none of them addresses what the user asked. Response Relevancy
measures *topical alignment*. The judge generates several synthetic questions
from the answer, then uses embeddings to measure cosine similarity between
those back-questions and the original question. A high score means the answer
stayed on-topic. See `slides/assets/08_response_relevancy.svg`.

### Factual Correctness — does the answer agree with the reference?

Unlike Faithfulness (which compares the answer to the retrieved passages),
Factual Correctness compares the answer to a human-authored **reference
answer** in your golden dataset. RAGAS extracts claims from both, then
computes precision and recall:

- **precision** = correct answer claims ÷ all answer claims (are the claims
  right?)
- **recall** = correct answer claims ÷ all reference claims (are all
  important points covered?)
- **default (F1)** = harmonic mean of the two

See `slides/assets/09_factual_correctness.svg`.

### LLM-as-judge — how the scoring actually works

All three metrics above delegate scoring to a second language model: the
**judge**. The judge reads the question, context, answer (and reference where
needed), then decides whether claims are grounded or correct. This is far more
flexible than brittle string-matching, but it introduces its own risks.

See `slides/assets/10_llm_as_judge.svg`.

**Known biases in LLM judges:**

| Bias | What happens | Why it matters |
|---|---|---|
| **Verbosity bias** | Judge prefers longer, more elaborate answers | A wordy but hallucinated answer can outscore a terse but accurate one |
| **Position bias** | Judge prefers whichever candidate appears first in a list | Comparison setups must randomize order |
| **Self-preference** | A model rates its own outputs higher than outputs from other models | Do not use the same model as generator and judge |

This module *owns* the LLM-as-judge concept — later modules rely on it
without re-explaining it.

---

## Code preview

**Stub + event-loop patch (required before importing RAGAS):**

```python
import sys, types
_vx = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI:
    pass
_vx.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vx

import nest_asyncio
nest_asyncio.apply()
```

**Building RAGAS model objects (judge at temperature 0):**

```python
import litellm
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

JUDGE_MODEL     = "ollama_chat/gemma4:31b-cloud"
EMBEDDING_MODEL = "ollama/qwen3-embedding:0.6b"

judge_llm = llm_factory(
    JUDGE_MODEL, provider="litellm",
    client=litellm.completion, temperature=0.0   # deterministic judge
)
ragas_embeddings = embedding_factory(
    "litellm", model=EMBEDDING_MODEL,
    api_base=os.environ["OLLAMA_API_BASE"]
)
```

**Running all three generator metrics:**

```python
from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, FactualCorrectness

results = evaluate(
    dataset=eval_dataset,
    metrics=[Faithfulness(), ResponseRelevancy(), FactualCorrectness()],
    llm=judge_llm,
    embeddings=ragas_embeddings,
)
print(results.to_pandas()[["faithfulness","answer_relevancy","factual_correctness"]])
```

**Precision vs recall mode for FactualCorrectness:**

```python
fc_precision = FactualCorrectness(llm=judge_llm, mode="precision")
fc_recall    = FactualCorrectness(llm=judge_llm, mode="recall")
```

---

## Notebook preview

| Step | What you do |
| ---: | --- |
| 0.1 | Install with `uv sync`, launch `uv run jupyter lab` |
| 0.2 | Load `OLLAMA_API_KEY` from shared `tutorials/.env` |
| 1 | Compatibility stub + `nest_asyncio` patch |
| 2 | Load corpus and build the vector store |
| 3 | Build RAGAS LLM / embeddings objects (judge at T=0) |
| 4 | Load golden questions and build the evaluation dataset |
| 5 | Run Faithfulness — score + per-sample deep-dive |
| 6 | Run Response Relevancy — score + interpretation |
| 7 | Run Factual Correctness (F1, precision, recall modes) |
| 8 | Compare all three generator scores in one DataFrame |
| 9 | LLM-as-judge discussion: biases + mitigation |

---

## Cautions

⚠ **Faithfulness ≠ Factual Correctness.** A perfectly faithful answer (score
1.0) can still be *wrong*. Faithfulness only asks "did the model stay within
the retrieved passages?" — if the retrieved passages themselves contain an
error, a faithful answer inherits that error and still scores 1.0. Factual
Correctness catches this because it compares to a human-authored reference,
not the retriever's output. Use both together.

⚠ **LLM-judge biases are real and measurable.** Keep the judge model
different from the generator model. Run the judge at `temperature=0` so scores
are reproducible. Be especially skeptical of Response Relevancy on very short
answers — the back-question technique has less signal when the answer is a
single sentence.

⚠ **Cost note.** Every evaluation call sends context + answer to the judge
LLM. With 8 golden questions and three metrics, expect roughly 24–30 judge
calls per full evaluation run. If you are using a metered cloud endpoint,
check your usage before running in a loop.

---

## References

- Capstone theory: `topics/06_rag_eval/agentic_rag_evaluation_theory.md`
  (Faithfulness §3.3, Response Relevancy §3.4, Factual Correctness §3.5,
  LLM-as-judge §4)
- Capstone notebook: `topics/06_rag_eval/agentic_rag_evaluation_tutorial.ipynb`
- RAGAS 0.4.x docs: <https://docs.ragas.io/en/v0.4.3/>
- RAGAS metrics reference: <https://docs.ragas.io/en/v0.4.3/concepts/metrics/>
- LLM-as-judge survey (Zheng et al., 2023): <https://arxiv.org/abs/2306.05685>

**Next module:** Module 10 — Reranking. You will add Cohere rerank-v3.5 to
the retrieval step and measure whether it lifts the generator metrics you
learned here.
