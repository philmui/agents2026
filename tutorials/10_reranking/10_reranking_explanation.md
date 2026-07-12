# Module 10 · Reranking — Explanation

> Audience: advanced high-school researchers. Tone: clear, concrete, encouraging,
> honest about where ideas break.

## Where this module sits

Module 09 showed you three generator metrics — Faithfulness, ResponseRelevancy, and
FactualCorrectness — that judge whether an LLM's answer is accurate and on-topic.
Module 10 takes a step back to the *retriever* side of the pipeline and asks: can we
give the generator better raw material? By adding a **reranker** between the vector
search and the answer step, we push precision up before the LLM ever sees a passage.
Module 11 will take this refined pipeline and wrap it in a LangGraph agent capable of
multi-turn conversations and live tool calls.

## The big idea

### Why retrieval alone is not enough

Embedding-based retrieval is fast and cheap: it converts every passage in your
knowledge base to a vector and finds the geometrically nearest ones for a query. The
trouble is that cosine similarity measures *general semantic overlap*, not *answer
relevance*. A passage about "gold storage costs" and a passage about "interest-rate
theory" can sit nearly equally close to the query "why does gold do well when real
rates fall?" — but only one of them actually contains the answer.

### Cross-encoders vs. bi-encoders

Embedding models are **bi-encoders**: they map the query and each document to a vector
*independently*, which is why they scale to millions of passages. A **cross-encoder**
(the architecture inside Cohere's reranker) takes the query and a passage *together* as
a single input, letting the model compute fine-grained token-level interactions. This is
much more expensive per passage — but you only run it on the small *k=10* shortlist the
bi-encoder already retrieved.

### The retrieve-wide-then-narrow pattern

```
Bi-encoder:  retrieve k=10  →  cheap, high-recall shortlist
Cross-encoder: rerank k=10  →  return top_n=3  →  precise, high-precision context
```

The final prompt is stuffed with three passages of high confidence, not ten passages of
mixed quality. This typically improves Context Precision (measured in Module 07) while
keeping Context Recall roughly stable.

### Cohere rerank-v3.5

`cohere.ClientV2` exposes `co.rerank(model="rerank-v3.5", query=..., documents=...,
top_n=3)`. Each result carries an `index` (pointer back into the original list) and a
`relevance_score` (0–1). Higher is better. We unpack that into a ranked list of
`(text, score)` pairs and feed the texts into the RAG prompt.

## Code preview

**Initialising the Cohere client** (from `tutorials/.env` via `find_dotenv()`):

```python
import cohere, os
co = cohere.ClientV2(os.environ["COHERE_API_KEY"])
```

**The rerank helper** (lifted from §8h of MODULE_SPEC):

```python
def rerank(query: str, docs: list[str], top_n: int = 3) -> list[tuple[str, float]]:
    if not docs:
        return []
    result = co.rerank(model="rerank-v3.5", query=query, documents=docs, top_n=top_n)
    return [(docs[r.index], r.relevance_score) for r in result.results]
```

**Wiring rerank into rag_answer** (§8e of MODULE_SPEC):

```python
def rag_answer(question: str, k: int = 10, top_n: int = 3, use_rerank: bool = True) -> dict:
    candidates = [d.page_content for d in base_retriever.invoke(question)]
    contexts = [t for t, _ in rerank(question, candidates, top_n=top_n)] if use_rerank else candidates[:top_n]
    block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    response = chat_llm.invoke(RAG_PROMPT.format_messages(context=block, question=question)).content.strip()
    return {"response": response, "retrieved_contexts": contexts}
```

**Before/after comparison on a golden question**:

```python
q = golden[2]["question"]   # "Why does gold tend to do well when real interest rates are low?"
baseline = rag_answer(q, k=10, use_rerank=False)
reranked = rag_answer(q, k=10, use_rerank=True)
# Display side by side: baseline["retrieved_contexts"] vs reranked["retrieved_contexts"]
```

## Notebook preview

| Step | What you do |
| ---: | --- |
| 0 | Install deps (`uv sync`), load keys from `tutorials/.env` |
| 1 | Import libraries, load corpus, build vector store |
| 2 | Implement `rerank()` using Cohere rerank-v3.5 |
| 3 | Implement `rag_answer()` with `use_rerank` toggle |
| 4 | Before/after comparison: same question, same k=10, different top_n selection |
| 5 | Connect to retriever metrics: what precision and recall look like before vs after |
| 6 | Recap and pointer to Module 11 |

## Cautions

⚠ **Reranking adds latency and cost.** Each call to `co.rerank()` hits Cohere's API and
takes ~100–300 ms. In a demo with 8 golden questions that is manageable, but at
production scale you must weigh the precision lift against the per-query cost. The
`COHERE_API_KEY` key also implies a **second paid API** on top of Ollama — check your
account limits before running the full golden set.

⚠ **`top_n` too small drops needed context.** Setting `top_n=1` maximises precision in
theory but in practice the reranker is not perfect. If the one returned passage is
wrong, there is no fallback. `top_n=3` is a pragmatic balance: enough passages to cover
partial answers while keeping the prompt tight.

⚠ **Reranking does not fix bad retrieval.** If the bi-encoder's k=10 shortlist does not
contain the relevant passage at all (a recall failure), the reranker has nothing to
surface. Reranking *refines* the shortlist; it cannot *expand* it. Monitor Context Recall
(Module 07) separately.

⚠ **Cost note — two paid APIs.** This module is the first to need *both* `OLLAMA_API_KEY`
(chat/embeddings) and `COHERE_API_KEY` (reranking). Students without both keys can still
follow along using the illustrative outputs in `frozen/rerank_comparison.json`.

## References

- Capstone theory: `topics/06_rag_eval/agentic_rag_evaluation_theory.md` — Reranking /
  Cross-Encoder section.
- Capstone notebook: `topics/06_rag_eval/agentic_rag_evaluation_tutorial.ipynb` — the
  `rerank()` helper and `rag_answer(..., use_rerank=True)` cells.
- Cohere rerank-v3.5 docs: https://docs.cohere.com/docs/rerank-overview
- Original cross-encoder paper (Nogueira & Cho, 2019): https://arxiv.org/abs/1901.04085
- RAGAS retriever metrics (Context Precision / Recall): https://docs.ragas.io/en/latest/concepts/metrics/retriever.html

**Next module:** Module 11 — *From RAG to Agent* — wraps this pipeline in a ReAct
agent (LangGraph `create_react_agent`), adds live metal-price tool calls, and introduces
agent-level metrics: ToolCallAccuracy, TopicAdherence, and AgentGoalAccuracy.
