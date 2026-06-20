# Module 01 · What is RAG? — Explanation

> Audience: advanced high-school researchers with basic Python familiarity. No ML background assumed.
> This is the first module in a twelve-part track that ends in a full **Agentic RAG Evaluation** capstone.

---

## Where this module sits

This is Module 01 — the starting point. There are no prerequisites beyond curiosity and basic Python comfort. By Module 12 you will have built a complete Agentic RAG evaluation system that measures a precious-metals research assistant across twelve different quality metrics. This module gives you the mental model that everything else builds on: *why* we need retrieval, and *what* RAG actually does.

---

## The big idea

### 1. LLMs have a frozen knowledge problem

A large language model is trained on a massive corpus of text collected up to a fixed date. After that training run, the model's weights are frozen. It is like a very well-read scholar who was locked in a library several months ago and has received no new information since. Ask it about historical events, general science, or timeless concepts and it performs remarkably well. Ask it about today's gold spot price, your company's internal research notes, or a paper published last week — and it either does not know, or worse, it invents a plausible-sounding answer.

This creates three concrete gaps:

- **Temporal gap**: events, prices, and facts that changed after the training cutoff
- **Private-data gap**: documents the model was never trained on (your corpus, your lab notes, proprietary datasets)
- **Precision gap**: specific facts that require exact sourcing rather than statistical memorization

### 2. Confabulation: fluency without accuracy

The most dangerous property of a frozen LLM is *confabulation* (also called hallucination). When the model does not know something, it often fills the gap with a confident, grammatically fluent, plausible-sounding invention. It does not say "I don't know" — it says "$1,842.50 per troy ounce" as if it looked it up. The problem is that fluency and accuracy are independent: the model is optimized to produce coherent text, not necessarily true text.

For research purposes this is critical to understand. Every LLM output is a draft hypothesis, not a citation. Retrieval-Augmented Generation is the engineering response to this problem.

### 3. The RAG idea: read before you answer

Retrieval-Augmented Generation addresses all three gaps by a simple principle: instead of asking the model to *remember* a fact, we give it the fact to *read* right before it answers.

At inference time (when you ask a question), the system:
1. **Retrieves** passages from a knowledge base that are relevant to the question
2. **Selects** the most relevant passages (the top-k results)
3. **Augments** the prompt: it inserts those passages directly into the text the model sees
4. **Generates** an answer instructed to use only the provided passages

The model never updates its weights. It simply uses a core property called *in-context learning*: LLMs can follow new information placed in the prompt window without any retraining.

### 4. Two moving parts: retriever and generator

A RAG pipeline has two independent failure modes (see diagram `slides/assets/03_retriever_vs_generator.svg`):

- The **retriever** can fail by returning irrelevant, incomplete, or noisy passages. No matter how good the generator is, bad context produces bad answers.
- The **generator** can fail by hallucinating beyond the context, misinterpreting the passages, or producing unfaithful summaries even when good context is available.

This two-part structure is why Modules 7–9 of this track measure the retriever and generator with separate metrics. You cannot improve what you do not measure separately.

---

## Code preview

Module 01 is primarily conceptual — there is no cloud API required and no heavy local model. The notebook illustrates the RAG idea with a small interactive Python example that constructs a toy "knowledge base" as a Python dictionary and demonstrates the retrieve → augment → generate loop conceptually using only the standard library plus a trivial keyword search.

```python
# A toy knowledge base: imagine this is your vector store
knowledge_base = {
    "troy_ounce": "A troy ounce is a unit of weight equal to 31.1035 grams, "
                  "used internationally for precious metals.",
    "spot_price":  "The spot price is the current market price at which a metal "
                   "can be bought or sold for immediate delivery.",
    "gold_symbol": "The chemical symbol for gold is Au, from the Latin 'aurum'.",
}

def retrieve(question: str, kb: dict, top_k: int = 2) -> list[str]:
    """Keyword-based retrieval (conceptual stand-in for vector search)."""
    keywords = question.lower().split()
    scored = [
        (sum(kw in text.lower() for kw in keywords), text)
        for text in kb.values()
    ]
    scored.sort(reverse=True)
    return [text for _, text in scored[:top_k] if _ > 0]
```

```python
def rag_answer(question: str, kb: dict) -> str:
    """Illustrate the RAG loop without calling any LLM API."""
    passages = retrieve(question, kb)
    if not passages:
        return "(No relevant passage found — the LLM would have to guess.)"
    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    # In a real system, this prompt goes to an LLM. Here we just show it.
    prompt = (
        f"Answer the question using ONLY the passages below.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return f"[Prompt that would be sent to the LLM]\n\n{prompt}"
```

---

## Notebook preview

| Step | What you do |
| ---: | --- |
| 0 | Set up the environment (`uv sync` + `uv run jupyter lab`) |
| 1 | The frozen-library problem — explore why LLMs confabulate |
| 2 | In-context learning — show the prompt window trick |
| 3 | Build a toy RAG loop — retrieve, augment, generate (keyless) |
| 4 | Two failure modes — retriever vs generator intuition |

---

## Cautions

> **⚠ Caution — RAG is not a cure for confabulation, it is a mitigation.**
>
> A RAG system can still confabulate if: (a) the retriever returns irrelevant passages, (b) the generator ignores the context and relies on its parametric memory anyway, or (c) the knowledge base itself contains incorrect information. Grounding an answer in retrieved text reduces hallucination rates significantly, but it does not eliminate them. Module 6 ("Why evaluate?") introduces the measurement loop you need to quantify how much confabulation remains.

> **⚠ Caution — "In-context learning" does not mean the model learns permanently.**
>
> When we say the model "learns from" the passages in its prompt, we mean it uses them for *that single inference call*. The model's weights do not update. The next time you call it with no context, it is back to its frozen state. Do not confuse in-context learning with fine-tuning or training.

---

## References

- **Capstone theory doc**: `topics/06_rag_eval/agentic_rag_evaluation_theory.md` — see §1 "What is Agentic RAG?" for the full system-level description this module conceptually introduces.
- **Capstone notebook**: `topics/06_rag_eval/agentic_rag_evaluation_tutorial.ipynb` — the complete pipeline this track rebuilds step by step.
- Lewis et al., 2020. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) — the original RAG paper.
- RAGAS documentation: <https://docs.ragas.io/> — the evaluation framework introduced in Module 6.
- LangChain documentation: <https://python.langchain.com/> — the orchestration framework used from Module 5 onward.

---

## Next module

**Module 02 — Embeddings & meaning**: you will learn how text is converted into vectors so that a retriever can find semantically similar passages (not just keyword matches). This is the mathematical engine that makes the "Retrieve" step in RAG work.
