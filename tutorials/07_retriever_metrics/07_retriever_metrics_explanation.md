# Module 07 · Retriever Metrics — Explanation

> Per-module markdown companion to the notebook and slides.
> Audience: advanced high-school researchers. Tone: clear, concrete, encouraging,
> honest about where ideas break.

## Where this module sits

Module 06 introduced the evaluation mindset: why we measure instead of guess, what the MDD loop looks like, and how RAGAS's `SingleTurnSample` and `EvaluationDataset` structures hold the data. By the end of Module 06 you had RAGAS wired up but no metrics yet. Module 07 adds the first two retriever metrics — `LLMContextPrecisionWithReference` and `LLMContextRecall` — and shows you how to run them over the full 8-question golden set and drill into individual samples. Module 08 will extend the retriever picture with two more specialized metrics: `ContextEntityRecall` (entity-level coverage) and `NoiseSensitivity` (robustness to irrelevant chunks).

## The big idea

### Precision vs Recall: two complementary lenses

A retriever's job is to fetch the right passages from the knowledge base. But "right" has two different meanings, and one number can't capture both at once.

**Context precision** asks: of all the chunks the retriever returned, how many were actually useful — and were the useful ones near the top? Think of it as a quality-and-ranking score. If the retriever returns 10 chunks and only 2 are relevant, precision is low even if those 2 were exactly right. Worse, if the 2 relevant chunks are buried at positions 8 and 9, the score is further penalized because good evidence hidden at the bottom is harder for the generator to use.

**Context recall** asks the opposite question: of all the evidence that the reference answer needed, how much of it did the retriever actually surface? Think of it as a coverage score. If the reference answer requires three distinct facts and the retriever only returned chunks supporting two of them, recall is roughly 0.67 — no matter how clean and well-ranked those two chunks were.

These two measures pull in opposite directions. The diagram at `slides/assets/04_context_precision_recall.svg` (reproduced from the capstone theory document) shows both in a single picture: precision is about the fraction of retrieved chunks that are relevant, recall is about the fraction of reference statements that are covered.

### Why both numbers matter

It is easy to game one metric while wrecking the other. Returning only 1 perfectly relevant chunk pushes precision toward 1.0 but collapses recall if the question needed more. Returning every chunk in the corpus guarantees recall can't be below 1.0 but makes precision near zero. A good retriever needs high scores on both, which is why every evaluation in this track reports them together.

The fishing analogy helps: precision is "did I mostly catch fish, not old boots?" and recall is "did I catch all the fish in the lake?" You can be a careful but under-fishing angler (high precision, low recall) or an indiscriminate net-dragger (high recall, low precision). Only a well-tuned retriever lands near the top-right corner of both.

### How RAGAS implements these metrics

`LLMContextPrecisionWithReference` sends each (question, chunk, reference answer) triple to a judge LLM and asks whether that chunk is relevant to answering the question given the reference. It then applies position weighting: a relevant chunk at rank 1 contributes more to the final score than the same chunk at rank 5. This makes the score sensitive to ranking, not just to which chunks were included.

`LLMContextRecall` takes a different approach. It first decomposes the reference answer into individual statements (e.g., "the troy ounce is approximately 31.1 grams" and "it is heavier than the avoirdupois ounce"). For each statement it checks whether any retrieved chunk supports it. The recall score is the fraction of reference statements that were covered.

## Cost note

Every call to `evaluate()` sends each (question × chunk) pair to the judge LLM. With 8 golden questions and k=10 retrieved chunks, that is approximately 80 LLM calls per metric — 160 calls total for both metrics together. During development, reduce `k` (e.g., k=3) to keep costs down. Use the full k=10 only for final comparison runs.

## Code preview

### 1. RAGAS import stub + asyncio patch

```python
# Must come before any ragas import — see MODULE_SPEC §5
import sys, types
_vx = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI: pass
_vx.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vx

import nest_asyncio; nest_asyncio.apply()
```

### 2. Judge LLM and embeddings

```python
import litellm
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

judge_llm = llm_factory(
    "ollama_chat/gemma4:31b-cloud",
    provider="litellm", client=litellm.completion, temperature=0.0
)
ragas_embeddings = embedding_factory(
    "litellm", model="ollama/qwen3-embedding:0.6b",
    api_base=os.environ["OLLAMA_API_BASE"]
)
```

### 3. Full-dataset evaluation

```python
from ragas import evaluate
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

results = evaluate(
    dataset=eval_dataset,
    metrics=[LLMContextPrecisionWithReference(), LLMContextRecall()],
    llm=judge_llm,
    embeddings=ragas_embeddings,
)
print(results.to_pandas()[["user_input", "context_precision", "context_recall"]])
```

### 4. Single-sample debug

```python
import asyncio
score = asyncio.run(
    LLMContextRecall(llm=judge_llm).single_turn_ascore(samples[6])
)
print(f"Recall on multi-hop silver question: {score:.3f}")
```

## Notebook preview

| Step | What you do |
| ---: | --- |
| 0 | Set up: shared `.env`, RAGAS import stub, `nest_asyncio` |
| 1 | Load RAGAS judge LLM and embeddings |
| 2 | Rebuild RAG answer function + build `EvaluationDataset` from golden questions |
| 3 | Run `evaluate()` with `[LLMContextPrecisionWithReference(), LLMContextRecall()]` |
| 4 | Single-sample debug with `single_turn_ascore()` on a multi-hop question |
| 5 | Recap and next-module pointer |

## Cautions

⚠ **High precision with low recall is easy to misread as success.** Every chunk you returned was relevant — but you missed half of what the answer needed. A system that returns only 1 perfectly relevant chunk can score 1.0 on precision and 0.2 on recall. Precision alone tells you nothing about completeness.

⚠ **Recall grades your golden set as much as your retriever.** `LLMContextRecall` measures coverage against the reference answer you wrote. If that reference is vague, incomplete, or paraphrased from a different chunk than what's in the corpus, recall will appear lower than the retriever actually is. Before blaming the retriever, check whether the reference is a fair description of what the corpus actually contains.

⚠ **LLM-judge variance.** RAGAS uses a language model to decide whether each chunk is relevant and whether each reference statement is supported. Re-running the same dataset on the same corpus can yield slightly different scores due to LLM non-determinism (even at temperature 0). Treat differences smaller than ~0.05 with skepticism; prefer the direction of change over exact values.

⚠ **Precision is sensitive to the choice of k.** Retrieving k=3 vs k=10 changes the denominator and the ranked position of chunks, so precision scores are not directly comparable across different k values. Always hold k constant when comparing two retriever configurations.

## References

- **Capstone theory document**, §4 "Context precision and context recall": `topics/06_rag_eval/agentic_rag_evaluation_theory.md`
- **Capstone notebook**, Section 11 (retriever metrics): `topics/06_rag_eval/agentic_rag_evaluation_tutorial.ipynb`
- RAGAS documentation: <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/>
- RAGAS paper: "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (Es et al., 2023) — <https://arxiv.org/abs/2309.15217>
- **Next module**: `tutorials/08_more_retriever_metrics/` — `ContextEntityRecall` (entity-level coverage) and `NoiseSensitivity` (inverted scale — lower is better).
