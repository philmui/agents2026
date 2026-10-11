# Sources, interpretation, and evidence

References were read and checked on **2026-10-03**. This guide links every source supplied for the tutorial, along with primary papers and documentation used to verify technical distinctions. Examples and vector graphics in the teaching materials are original classroom illustrations.

## The eleven requested references

| Reference | Contribution to the tutorial | Where it matters |
| --- | --- | --- |
| [Raschka: classifier history and Jev](https://magazine.sebastianraschka.com/p/classifier-history-and-jev) | A historical path from feature-based classifiers to language-model classifiers, plus a discussion of Jev and possible open implementations. Its architecture and training proposals are explicitly speculative. | Historical map, proprietary-details discussion, and open-model research design. |
| [Production Jev and LangGraph](https://www.langchain.com/blog/building-prod-with-jev-and-langgraph) | Combines bounded model judgments with a graph runtime that manages state and routing. | A classroom graph with a classify node and explicit handlers. |
| [Building a harness with Jev](https://www.langchain.com/blog/building-a-harness-with-jev) | Introduces the LangChain classifier adapter and examples of model selection and tool-risk checks. | Adapter usage and the explanation of an agent harness. |
| [Jev in LangSmith Evals](https://www.langchain.com/blog/jev-is-now-available-in-langsmith-evals) | Shows evaluator configuration using state, typed questions, and feedback keys. | Explanation of how trace evidence becomes model feedback. |
| [Jev-as-a-Judge study](https://www.langchain.com/blog/jev-agent-evals-langsmith) | Reports agreement, repeated-call variability, latency, and cost for a limited fixed-trace experiment. | Reading the early evidence at its actual scale and designing broader judge studies. |
| [TypeSafe manifesto](https://typesafe.ai/manifesto) | Advocates composing small model judgments with inspectable software and exact computation. Its broad strategic claims are the company's position. | Motivation for keeping workflow rules in code. |
| [Quickstart](https://docs.typesafe.ai/introduction/quickstart) | Documents the API endpoint, authentication, and direct SDK entry point. | Optional live calls and setup instructions. |
| [Confidence routing](https://docs.typesafe.ai/patterns/confidence-routing) | Uses uncertainty to route decisions differently according to the consequences of mistakes. | Review branches and illustrative thresholds. |
| [Intent routing](https://docs.typesafe.ai/patterns/intent-routing) | Chooses among deterministic code, specialist models, and people based on intent and complexity. | Help-desk routing and specialist-agent research. |
| [Advanced structure](https://docs.typesafe.ai/primitives/advanced) | Documents JSON structures in instructions and criterion descriptions. | Structured question construction. |
| [How to build with System One](https://docs.typesafe.ai/concepts/how-to-build-with-system-one) | Recommends relevant state, narrow questions, batched evaluation, explicit composition, and tested uncertainty policies. | Overall notebook workflow and research on composing atomic decisions. |

## Technical distinctions verified against direct documentation

- **Choice** selects one supplied option and reports a distribution over those options. Its confidence statistic measures concentration relative to an even split. It is not identical to the winning probability. [Choice](https://docs.typesafe.ai/primitives/choice), [Confidence](https://docs.typesafe.ai/confidence).
- **Noul** returns the yes probability as `noul`. The documented answer has no separate confidence field. A value near zero favors no; a value near one favors yes. [Noul](https://docs.typesafe.ai/primitives/noul).
- **Score** uses ordered, described positions beginning at zero. The scalar is the probability-weighted average of level indices and can be fractional. Its confidence statistic accounts for distance between levels. JSON uses string level keys; the direct SDK uses integer level keys. [Score](https://docs.typesafe.ai/primitives/score), [SDK responses](https://docs.typesafe.ai/sdk/python/api/types/responses).
- Question IDs are response identifiers; the full meaning of a question belongs in its instructions. The state can be a text string, a JSON object, or an array. [API reference](https://docs.typesafe.ai/api).
- Independent questions share a state and can be evaluated together. A question that needs another answer requires a subsequent application step. Separately evaluated properties can still be statistically correlated. [Fan-out](https://docs.typesafe.ai/patterns/fan-out).
- The model guide listed `jev-1.13.0`, with `jev-latest` resolving to it when checked. Aliases move, and the response reports the actual version. The documented model accepts text and structured textual state rather than direct image, audio, or video input. [Models](https://docs.typesafe.ai/models).
- Literal interpretation, mathematical precision, irrelevant long context, adversarial state, and option ordering require care in version-specific tests. [Jev 1.13 limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

## What the training sources establish

TypeSafe identifies **RLCD** as reinforcement learning for calibrated decisions, with a goal of useful decisions and probability estimates. This public description does not supply the complete architecture, reward function, training dataset, or optimization procedure. [TypeSafe AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer).

Raschka's shared candidate-scoring example shows one plausible way to offer a flexible choice interface with an existing language model. It does not establish how Jev is implemented. His discussion of **RLCR** is an analogy to public calibration research, with no established connection to the proprietary RLCD method. The tutorial labels its candidate-scoring graphic as an illustrative research design. [Historical article](https://magazine.sebastianraschka.com/p/classifier-history-and-jev).

## How the empirical claims are used

The tutorials do not reproduce advertised speed or price ratios as measured results. Broad comparisons depend on model versions, task definitions, hardware, batching, network time, and pricing. The live notebook reports observed request duration and token usage if enabled, and leaves cross-model benchmarking to a controlled research design.

LangChain's early judge experiment used **five distinct captured weather-agent runs**, each judged repeatedly. It reported strong human-label agreement and low observed variability for Jev on those examples. Repetitions assess stability on fixed cases, while broad generalization needs diverse independent tasks. The article also describes provider-default generation settings and unavailable Jev service-version metadata. Its observations do not isolate a causal effect of Jev's training procedure. [Judge study](https://www.langchain.com/blog/jev-agent-evals-langsmith).

The notebook's retained results are offline toy calculations. Its synthetic Jev-shaped distributions are not observations of Jev. No real accuracy, calibration, price, or latency result is inferred from those fixtures.

## Primary references for earlier methods

| Topic | Primary or authoritative reference |
| --- | --- |
| Word counts, n-grams, TF-IDF | [scikit-learn feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html) |
| Multinomial naive Bayes and smoothing | [Stanford information-retrieval textbook](https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html) |
| Text CNNs, 2014 | [Kim: Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) |
| Transformers, 2017 | [Vaswani et al.: Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| ULMFiT, 2018 | [Howard and Ruder: Universal Language Model Fine-tuning](https://arxiv.org/abs/1801.06146) |
| BERT, 2018 | [Devlin et al.: BERT](https://arxiv.org/abs/1810.04805) |
| T5, 2019 | [Raffel et al.: A Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) |
| ModernBERT, 2024 | [Warner et al.: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) |
| Calibration | [Guo et al.: On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) |
| Graph state, nodes, and edges | [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) |

These sources support the concepts and historical dates. The teaching diagrams simplify operations to make their purpose visible; invented numerical examples are marked beside their outputs.
