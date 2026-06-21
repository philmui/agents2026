# Module 06 · Why Evaluate? + RAGAS Setup — Explanation

> Conceptual companion to `06_why_evaluate.ipynb` and `slides/index.html`.
> Audience: advanced high-school researchers. Tone: clear, concrete, encouraging,
> honest about where ideas break.

## Where this module sits

Module 05 gave you a working RAG pipeline: a cloud-Ollama embedder, a Qdrant
vector store, and a prompt-stuffing generator that returns an answer to any
question about precious metals. That pipeline *works*, but "it works" is not a
measurement. Module 06 asks the harder question: *how do we know it is working
well?* It introduces the **evaluation mindset**, the **Metrics-Driven Development
(MDD)** loop, and the RAGAS data structures that the rest of the track
(`SingleTurnSample`, `EvaluationDataset`) will use to score every component.
Module 07 will compute the first real metric scores.

## The big idea

### 1. The evaluation mindset

You cannot improve a system you cannot measure. This sounds obvious, but in
practice it is easy to tweak a retriever, get a slightly better-sounding answer
on one question, and call it progress — without knowing whether the other seven
questions got worse. Rigorous evaluation requires a fixed **test set** of
representative questions, a consistent **scoring procedure**, and the discipline
to run that procedure *before and after* every change. Without all three, you
are reading tea leaves.

RAGAS provides the scoring procedure. `golden_questions.json` is the test set
(eight questions covering six single-hop and two multi-hop queries over the
metals-markets corpus). The discipline of running the same procedure on the same
questions is what this module installs.

### 2. The Metrics-Driven Development (MDD) loop

MDD is the evaluation mindset made operational. The loop has four phases:

1. **Build** — make a change to the pipeline (e.g., swap the retriever or add
   reranking).
2. **Measure** — run the same golden questions through the same metrics.
3. **Diagnose** — read the per-sample breakdown: which questions improved? which
   got worse? did precision rise while recall fell?
4. **Improve** — form a hypothesis about *why* a metric moved, make exactly one
   more change, and measure again.

The critical constraint is **one variable at a time**. If you turn on reranking
and swap the chunking strategy simultaneously, you cannot know which change
caused the improvement.

See `slides/assets/02_mdd_loop.svg` for the loop diagram.

### 3. Retriever vs. generator — which half failed?

A RAG answer is wrong because either the retriever brought back the wrong
passages or the generator produced an unfaithful or irrelevant response from
correct passages. These are completely different failure modes with completely
different fixes. Reranking helps the retriever; a better prompt or a stronger
generator model helps the generation side. RAGAS makes this split explicit:
retriever metrics (context precision, recall, entities recall, noise
sensitivity) and generator metrics (faithfulness, response relevancy, factual
correctness) point you to the right lever.

See `slides/assets/03_retriever_vs_generator.svg` for the split diagram.

### 4. RAGAS data structures: `SingleTurnSample` and `EvaluationDataset`

RAGAS organises a benchmark as a list of `SingleTurnSample` objects collected
into an `EvaluationDataset`. Each `SingleTurnSample` holds:

- `user_input` — the question.
- `retrieved_contexts` — the list of passages the retriever returned.
- `response` — the generator's answer.
- `reference` — the ground-truth answer from `golden_questions.json`.

```python
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

sample = SingleTurnSample(
    user_input="What is a troy ounce?",
    retrieved_contexts=["A troy ounce is 31.10 grams ..."],
    response="A troy ounce weighs 31.10 grams, slightly more than the common ounce.",
    reference="A troy ounce is the standard unit for precious metals, about 31.1 grams.",
)
dataset = EvaluationDataset(samples=[sample])
```

The `generator_llm` (the LLM that *wrote* the answers) and the `judge_llm` (the
LLM that *grades* them) are intentionally different models — this module sets up
both.

## Code preview

### RAGAS import stub (required before `import ragas`)

RAGAS 0.4.3 tries to import a `langchain_community` module that has been
removed. Stub it out first:

```python
import sys, types
_vx = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI:
    pass
_vx.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vx
import nest_asyncio; nest_asyncio.apply()
```

### Shared key loading

```python
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())          # walks up to tutorials/.env
HAVE_KEYS = bool(os.environ.get("OLLAMA_API_KEY"))
```

### RAGAS model objects

```python
import litellm
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

LLM_MODEL       = "ollama_chat/nemotron-3-super:cloud"
JUDGE_MODEL     = "ollama_chat/gemma4:31b-cloud"
EMBEDDING_MODEL = "ollama/qwen3-embedding:0.6b"

generator_llm    = llm_factory(LLM_MODEL,   provider="litellm",
                               client=litellm.completion, temperature=0.3)
judge_llm        = llm_factory(JUDGE_MODEL, provider="litellm",
                               client=litellm.completion, temperature=0.0)
ragas_embeddings = embedding_factory("litellm", model=EMBEDDING_MODEL,
                                     api_base=os.environ["OLLAMA_API_BASE"])
```

### Building the dataset

```python
import json
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

golden = json.loads(open("golden_questions.json").read())
samples = []
for g in golden:
    out = rag_answer(g["question"])
    samples.append(SingleTurnSample(
        user_input=g["question"],
        retrieved_contexts=out["retrieved_contexts"],
        response=out["response"],
        reference=g["reference"],
    ))
eval_dataset = EvaluationDataset(samples=samples)
```

## Notebook preview

| Step | What you do |
| ---: | --- |
| 0.1 | Install the libraries (`uv sync`) |
| 0.2 | Load keys from `tutorials/.env`; set `HAVE_KEYS` guard |
| 1 | Import stub + `nest_asyncio`; import RAGAS safely |
| 2 | Re-build the M5 RAG (embeddings, vector store, retriever, generator) |
| 3 | Create RAGAS model objects (`generator_llm`, `judge_llm`, `ragas_embeddings`) |
| 4 | Load golden questions; call `rag_answer` on each; build `EvaluationDataset` |
| 5 | Inspect the dataset — verify shape, peek at a sample row |

## Cost / safety note

Running all eight golden questions through the cloud-Ollama generator and
building the `EvaluationDataset` makes **8 generator LLM calls**. Creating the
RAGAS model objects makes no API calls at construction time. Scoring (the next
module) will add judge calls on top. If `OLLAMA_API_KEY` is missing, the
notebook falls back to `frozen/sample_dataset.json` — a two-row illustrative
dataset that lets you follow the rest of the cells without spending any credits.

## ⚠ Cautions

**You can't improve what you don't measure — but the wrong metric misleads.**
A high RAGAS score on a poorly chosen test set proves only that your system is
good at that test set. If your eight questions are too easy, too similar to each
other, or all single-hop, you are measuring the wrong thing. The golden set here
is deliberately diverse (single- and multi-hop, different metals, different
question types), but it is still only eight questions — treat absolute numbers
with healthy scepticism and focus on *relative changes* between pipeline
variants.

**Preview of Goodhart's Law (developed fully in Module 12):** when a metric
becomes the optimisation target, it stops being a reliable guide to quality.
Tuning your pipeline specifically against these eight questions risks over-fitting
to the test set rather than genuinely improving the system.

**Generator ≠ judge.** Using the same model to both write answers and grade
them creates a sycophancy loop: the model is likely to give high scores to
outputs it would have produced itself. This module deliberately separates
`generator_llm` (`nemotron-3-super`) from `judge_llm` (`gemma4:31b`). Always
maintain this split.

## References

- **Capstone theory doc** — `topics/06_rag_eval/agentic_rag_evaluation_theory.md`,
  sections 2 (MDD loop) and 3 (retriever vs. generator split). These are the
  source of the concepts in this module.
- **Capstone notebook** — `topics/06_rag_eval/agentic_rag_evaluation_tutorial.ipynb`,
  sections 8 (MDD) and 9 (RAGAS setup).
- **RAGAS 0.4.x docs** — https://docs.ragas.io/en/v0.4.3/
- **RAGAS dataset schema** — `ragas.dataset_schema.SingleTurnSample`,
  `ragas.dataset_schema.EvaluationDataset`
- **nest-asyncio** — https://github.com/erdewit/nest_asyncio

## Pointer to Module 07

Module 07 (`07_retriever_metrics`) takes the `EvaluationDataset` assembled here
and runs the first real metrics: **LLMContextPrecisionWithReference** and
**LLMContextRecall**. Those are the retriever half of the MDD loop — did the
right passages come back?
