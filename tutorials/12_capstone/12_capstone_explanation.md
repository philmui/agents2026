# Module 12 · Capstone — Explanation

> Audience: advanced high-school researchers who have completed Modules 1–11.
> Tone: clear, concrete, celebratory of everything built, honest about where the
> numbers can mislead. Mirror the house voice of the capstone theory doc.

## Where this module sits

Module 11 handed you a working ReAct agent equipped with three tools and five
RAGAS metrics that grade its behaviour. You had built every individual piece of
the Agentic RAG Evaluation system: a chunked vector store, cloud Ollama
embeddings, Cohere reranking, a golden test set, all four retriever metrics, all
three generator metrics, and all five agent-tier metrics. Module 12 is the
culmination. It lays every brick you built into the finished wall: one unified
notebook that assembles the full pipeline, runs all 12 metrics, executes the
Metrics-Driven Development loop, spotlights the two hardest questions in the
golden set, and closes with the cautions that stop you from misreading the
numbers. This is the system the entire track was pointing toward.

---

## The big idea (four concepts this module owns)

### 1. The full assembled pipeline

The capstone pipeline connects every subsystem in order:
1. **Corpus** — eight metals-domain markdown files loaded from `corpus/`.
2. **Chunking** — `RecursiveCharacterTextSplitter` (chunk=500, overlap=60).
3. **Embedding** — cloud Ollama `qwen3-embedding:0.6b` via `OllamaEmbeddings`.
4. **Vector store** — Qdrant in-memory collection `metals_kb`.
5. **Retrieval** — wide candidate pull at `k=10`.
6. **Cohere reranking** — `rerank-v3.5` narrows to `top_n=3` best passages.
7. **Generation** — `nemotron-3-super:cloud` via `ChatOllama` + the RAG prompt.
8. **Agent** — `create_react_agent` wrapping `get_metal_price`,
   `convert_currency`, and `search_metal_knowledge`.

Assembling these individually taught you what each piece does. Wiring them
together in one system teaches you how they interact and where failures
propagate.

### 2. All 12 RAGAS metrics across three tiers

The 12 metrics divide along the same retriever / generator / agent split you
learned in Module 3:

| Tier | Metrics |
| --- | --- |
| Retriever | `LLMContextPrecisionWithReference`, `LLMContextRecall`, `ContextEntityRecall`, `NoiseSensitivity` |
| Generator | `Faithfulness`, `ResponseRelevancy`, `FactualCorrectness` |
| Agent | `TopicAdherenceScore`, `ToolCallAccuracy`, `ToolCallF1`, `AgentGoalAccuracyWithReference`, `AgentGoalAccuracyWithoutReference` |

Running all 12 together for the first time makes one thing obvious: no single
number is sufficient. A high faithfulness can mask a retrieval problem; a high
tool-call accuracy can hide a goal failure.

### 3. The MDD baseline-vs-reranked comparison

Metrics-Driven Development means: establish a baseline, change exactly one
thing, recompute on the same test set, compare. In the capstone the single
change is turning Cohere reranking on. Both configurations hand the generator
the same number of passages (`top_n=3`); the only difference is whether those
three are the first three returned by vector search or the three the reranker
chose from a wider pool of ten. Because everything else is constant, any lift in
the metrics is attributable to reranking.

The illustrative results show the retriever metrics gaining the most (+0.15–0.18)
with generator metrics rising as a downstream consequence (+0.06–0.12). This is
the satisfying version of MDD: the improvement pattern matches the theory.

### 4. The two multi-hop questions

The golden set contains six single-hop questions and two multi-hop questions.
Single-hop questions can be answered from one corpus passage. Multi-hop questions
require synthesising information from two separate passages:

- **"Why might silver outperform gold in a strong economy but fall faster in a
  downturn?"** — requires the industrial-demand passage *and* the volatility
  passage.
- **"For diversification with low counterparty risk, what are the trade-offs of
  physical bullion versus ETFs?"** — requires the physical-storage passage *and*
  the correlation/portfolio passage.

Context recall consistently sags on these two questions even after reranking,
because the top-k window may not capture both needed passages simultaneously.
These are the hardest questions in the set and serve as a useful stress test of
any retrieval improvement you try.

---

## Code preview

The four key code patterns the notebook runs:

```python
# Full RAG answer function — retrieve wide, rerank narrow, generate
from langchain_core.prompts import ChatPromptTemplate
RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a precise metals-markets tutor. Answer using ONLY the context passages. "
    "If the context does not contain the answer, say you do not know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:")

def rag_answer(question: str, k: int = 10, top_n: int = 3, use_rerank: bool = True) -> dict:
    candidates = [d.page_content for d in vector_store.as_retriever(
        search_kwargs={"k": k}).invoke(question)]
    contexts = [t for t, _ in rerank(question, candidates, top_n=top_n)] \
               if use_rerank else candidates[:top_n]
    block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    response = chat_llm.invoke(
        RAG_PROMPT.format_messages(context=block, question=question)).content.strip()
    return {"response": response, "retrieved_contexts": contexts}
```

```python
# MDD loop — baseline vs. reranked, scored by the same 4 metrics
def build_dataset(use_rerank: bool, k: int = 10, top_n: int = 3) -> EvaluationDataset:
    rows = [SingleTurnSample(
        user_input=g["question"],
        response=(o := rag_answer(g["question"], k=k, top_n=top_n, use_rerank=use_rerank))["response"],
        retrieved_contexts=o["retrieved_contexts"],
        reference=g["reference"],
    ) for g in golden]
    return EvaluationDataset(samples=rows)

mdd_metrics = [LLMContextPrecisionWithReference(), LLMContextRecall(),
               Faithfulness(), FactualCorrectness()]
baseline_results = evaluate(build_dataset(use_rerank=False, top_n=3),
                            metrics=mdd_metrics, llm=judge_llm, embeddings=ragas_embeddings)
improved_results = evaluate(build_dataset(use_rerank=True, k=10, top_n=3),
                            metrics=mdd_metrics, llm=judge_llm, embeddings=ragas_embeddings)
```

```python
# All 12 metrics at once — three-tier scoreboard
from ragas.metrics import (
    LLMContextPrecisionWithReference, LLMContextRecall,
    ContextEntityRecall, NoiseSensitivity,          # retriever tier
    Faithfulness, ResponseRelevancy, FactualCorrectness,  # generator tier
    TopicAdherenceScore, ToolCallAccuracy, ToolCallF1,    # agent tier
    AgentGoalAccuracyWithReference, AgentGoalAccuracyWithoutReference,
)
```

```python
# Spotlight the two multi-hop questions
multi_hop_qs = [g for g in golden if g.get("hop") == "multi"]
for q in multi_hop_qs:
    sample = next(s for s in dataset.samples if s.user_input == q["question"])
    print(f"[MULTI-HOP] {q['question']}")
    print(f"  contexts retrieved: {len(sample.retrieved_contexts)}")
```

---

## Notebook preview

| Step | What you do |
| ---: | --- |
| 0 | Set up: install deps, load three API keys from `tutorials/.env` |
| 1 | RAGAS import stub + `nest_asyncio`; cost/safety note |
| 2 | Connect cloud Ollama: chat LLM, RAGAS judge, embeddings |
| 3 | Load corpus (8 files) and build Qdrant in-memory vector store |
| 4 | Build the Cohere reranker; define `rag_answer()` |
| 5 | Define three `@tool` functions; assemble the MetalDesk ReAct agent |
| 6 | Load golden questions; build `SingleTurnSample` dataset |
| 7 | Score all four **retriever** metrics; inspect per-sample scores |
| 8 | Score all three **generator** metrics |
| 9 | Score all five **agent-tier** metrics on a multi-tool agent run |
| 10 | **MDD loop**: baseline vs. reranked — compare all four MDD metrics |
| 11 | **Spotlight multi-hop questions** — show where recall still sags |
| 12 | Full 12-metric scoreboard; `NoiseSensitivity` inversion reminder |
| Recap | Goodhart's Law cautions; where to go from here |

---

## ⚠ Cautions

### Goodhart's Law — the most important caution in this whole track

When a metric becomes the target, it stops being a good metric. If you keep
tweaking your system until one number goes up, you risk two things: fitting noise
(with eight questions a 0.02 difference is random variation, not signal) and
gaming the metric without improving the underlying system. A retriever tuned to
maximise context precision on this exact golden set will likely overfit to the
phrasings in those eight questions and fail on new ones.

The right use of metrics is to track large, consistent, theory-aligned changes —
the kind where every metric moves up after a well-motivated intervention, like
the reranking switch in the MDD loop.

### NoiseSensitivity is inverted — lower is better

Every other metric in the 12-metric scoreboard rewards a higher number.
NoiseSensitivity is the exception: it measures the fraction of claims in the
answer that are wrong and traceable to retrieved context. A score of 0 is
perfect; a score of 0.2 means 20% of claims were noise-driven errors. This
inverted scale is a genuine trap when you are scanning a results table.

### LLM-judge biases

The RAGAS judge (`gemma4:31b-cloud` at temperature 0.0) is a different model
from the generator (`nemotron-3-super:cloud`) precisely to reduce self-preference
bias — but biases remain. The judge tends to favour longer answers (verbosity
bias), may score the same answer differently across runs (non-determinism), and
can emit invalid JSON on edge cases, producing `NaN` rather than a wrong score.
Always cross-check suspicious results by reading the actual answer.

### Faithful ≠ correct

Faithfulness only compares the answer to the retrieved context. A perfectly
faithful answer that loyally repeats a wrong passage earns a score of 1.0 while
being completely wrong. Always read faithfulness alongside factual correctness,
which compares against ground truth.

### Multi-hop questions stress every retrieval metric

The two multi-hop questions in the golden set consistently score lower on context
recall than the six single-hop questions because the relevant information is
spread across two corpus passages. A naive top-k window may capture one but not
both. This is not a bug in the metric — it is the metric correctly diagnosing a
real retrieval limitation. Widening `k` and using better reranking helps, but
multi-hop retrieval remains an open research problem.

---

## References

### Capstone source of truth

- **Theory guide** (all 14 sections): `topics/06_rag_eval/agentic_rag_evaluation_theory.md`
  — Sections 2 (MDD), 10 (LLM-as-judge biases), and the "Putting it together"
  closing section are especially relevant here.
- **Capstone notebook**: `topics/06_rag_eval/agentic_rag_evaluation_tutorial.ipynb`
  — Sections 19 (MDD loop), 20 (pitfalls and cautions), 21 (recap).

### Module back-references (concepts built in earlier modules)

| Module | Concept |
| --- | --- |
| 01 | What is RAG? The retrieve → augment → generate loop |
| 02 | Embeddings: text→vectors, embedding space |
| 03 | Cosine similarity, top-k search, Qdrant |
| 04 | Chunking with `RecursiveCharacterTextSplitter`, the 8-file metals corpus |
| 05 | Cloud-Ollama via LiteLLM, the real RAG answer function |
| 06 | Evaluation mindset, MDD loop, `SingleTurnSample`/`EvaluationDataset` |
| 07 | `LLMContextPrecisionWithReference`, `LLMContextRecall` |
| 08 | `ContextEntityRecall`, `NoiseSensitivity` (inverted scale) |
| 09 | `Faithfulness`, `ResponseRelevancy`, `FactualCorrectness`; LLM-as-judge biases |
| 10 | Cohere `rerank-v3.5`; retrieve-wide-rerank-narrow pattern |
| 11 | `create_react_agent`; `convert_to_ragas_messages`; all five agent metrics |

### External primary sources

- [RAGAS 0.4.x documentation](https://docs.ragas.io/)
- [LangGraph prebuilt agents](https://langchain-ai.github.io/langgraph/)
- [Cohere Rerank v3.5](https://docs.cohere.com/reference/rerank)
- [Qdrant in-memory store](https://qdrant.tech/documentation/)

---

## Where to go from here

This is the last module of the track. You have now built and evaluated a
complete Agentic RAG system. Some directions for further exploration:

- **Extend the golden set** — add 20–50 questions (including more multi-hop
  ones) to reduce noise and make metric differences statistically meaningful.
- **Try a different retrieval strategy** — hybrid BM25 + dense retrieval, or
  a parent-document retriever that fetches small chunks but returns full
  parent sections to the generator.
- **Swap the judge model** — run with two different judge models and compare
  their scores to understand judge variance.
- **Add an adversarial test set** — include questions with deliberately
  misleading context to stress-test NoiseSensitivity and Faithfulness.
- **Deploy the agent** — wrap it in a FastAPI endpoint and evaluate latency
  vs. quality trade-offs as you change `k` and `top_n`.
- **Read the capstone theory doc in full**: every section ends with a caution
  that points to a deeper research question. Any of them is a starting point
  for an original ASDRP research project.

**Capstone notebook for live runs**: `12_capstone.ipynb`
**Authoritative reference system**: `topics/06_rag_eval/agentic_rag_evaluation_tutorial.ipynb`
