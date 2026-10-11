# Research projects with System One decision models

These projects extend the tutorial into agentic systems, cooperating agents, and general machine learning. They are proposed experiments, with no completed findings. Choose one question and a modest dataset first; a precise small study is easier to interpret than a broad collection of unrelated demonstrations.

Use an evaluation unit that matches the claim: a distinct task for task success, a distinct trace for judge agreement, and a repeated call on the same trace for repeatability. Group related messages or task families in the same split. Freeze the test set before question engineering, threshold selection, or an automated research loop.

## 1. Confidence gates under distribution shift

**Question:** Does a gate chosen on ordinary messages still reduce errors on unfamiliar language, wording, or topics?

**Hypothesis:** a gate's accepted-case accuracy deteriorates under some shifts, even when automatic coverage remains similar.

Build in-domain training or reference data, a development set for choosing thresholds, and separate test sets for ordinary inputs, paraphrases, missing context, and multilingual variants. If task adaptation is entirely request-based, record the instructions and taxonomy as the fitted design. Compare confidence thresholds, top-probability thresholds, and a clarification fallback. Match coverage when comparing accepted-case quality, or report the complete coverage–error curve.

Measure accuracy, selective accuracy, review workload, classwise probability calibration, and Brier scores. Report group sizes and sampling uncertainty. Keep the fallback fixed and include high-confidence wrong predictions. A baseline that simply rejects more cases can appear better unless coverage is explicit.

**Agent connection:** an agent's evolving tool results can create a different input distribution from its initial user requests. Testing these stages separately can show whether one global gate is adequate.

## 2. Decision-based coordination of specialist agents

**Question:** Can typed routing reduce specialist work while preserving the team's final success?

**Hypothesis:** a coordinator helps when tasks have clear boundaries, but ambiguous or multi-skill tasks need escalation.

Use specialists with fixed capabilities and tool access. Compare an always-general agent, fixed rules, a generative coordinator, and a Jev coordinator with an explicit clarification or escalation route. Include tasks requiring one specialist, several specialists, and a task unsupported by the team. Hold out whole task families.

Measure final task completion, unnecessary invocations, incorrect handoffs, unresolved conflicting evidence, wall-clock time, and actual cost. Apply the same total budget and retry limits. Also measure the coordinator's label quality, but do not substitute it for end-to-end success. Shared model families can produce correlated failures; agreement between agents is not independent evidence.

**Useful failure analysis:** distinguish an incorrect assignment from a specialist failure after a correct assignment. If the specialist would fail under every coordinator, a routing change cannot explain the result.

## 3. Typed judges aligned with human review

**Question:** Do focused typed rubrics agree with human reviewers across diverse agent traces?

**Hypothesis:** repeatable scores can support useful monitoring, but their agreement varies by criterion and trace family.

Freeze traces from multiple tasks and agent architectures. Have at least two reviewers independently apply a written rubric. Record disagreements and adjudicate only under a stated procedure. Compare exact code checks, a generative judge, and Jev on identical traces. Evaluate different judge question designs on development traces, then freeze them.

Report agreement on unique test traces, repeated-call variability separately, and confusion matrices for consequential errors. Include fluent unsupported answers, unnecessarily long answers, missing tools, alternative valid tool sequences, and evaluator-directed instructions within evidence. Cluster uncertainty estimates by task or trace rather than treating repeated judgments as independent samples.

**Agentic research connection:** evaluator feedback can guide prompt or policy revision, but a biased judge can train a loop to optimize the wrong behavior. Use a separate human-labeled evaluation set outside that loop. The [early Jev judge study](https://www.langchain.com/blog/jev-agent-evals-langsmith) motivates this broader design without establishing its outcome.

## 4. Candidate generation and abstention in tool-using agents

**Question:** When does the list of available actions limit a decision model more than the selection itself?

**Hypothesis:** an incomplete candidate list causes errors even when the selector returns concentrated probabilities.

Choose a browser or tool task where allowed actions can be enumerated. Compare a complete verified candidate set, a deliberately reduced set, and a set containing explicit abstention or clarification. Keep the selector, state, and budget fixed. Counterbalance candidate ordering so one favorable list order does not explain the result.

Measure candidate recall, selection accuracy conditional on the correct candidate being present, invalid operations, escalation rate, and task completion. Include exact permission enforcement in code. Removing an action is a controlled intervention on proposal quality, while changing instructions is a different intervention and should be tested separately.

**Broader ML connection:** this separates retrieval or proposal failures from ranking failures. A highly accurate ranker cannot select the missing right answer.

## 5. Learning to combine atomic semantic features

**Question:** Can a small supervised model combine several typed judgments better than a broad model question or a hand-written policy?

**Hypothesis:** some atomic signals are complementary and improve downstream predictions, while redundant or poorly defined signals do not.

Define narrow questions about a document, then use their returned probabilities as numerical features. Compare a broad direct judgment, a fixed weighted sum, and a logistic regression or tree model trained on the features. Fit all downstream parameters on training data, choose features and thresholds on development data, and preserve an untouched test set. Keep related documents grouped.

Measure task accuracy, Brier scores, calibration, number of questions, call cost, and end-to-end latency. Remove one feature at a time to test its contribution. Compare against cheap lexical features so a model-derived feature pipeline has a credible simpler baseline.

**Automated research connection:** a research agent could propose new questions from development failures. Cap the number of searches, record every candidate, and never expose test feedback to the proposal loop. Many trials can overfit the development set too; nested evaluation or an additional selection holdout can help quantify that risk.

## 6. Open-model decision interfaces and calibration objectives

**Question:** What training helps an open model handle unseen instruction-defined choices with usable probabilities?

**Hypothesis:** diverse task training and a shared candidate scorer improve transfer beyond a fixed task head; probability-quality gains need separate evidence.

Start with a pretrained open model. One possible design scores the state, instruction, and each candidate description with shared parameters, then normalizes scores over candidates. This is an illustrative design, not a claim about Jev's architecture. Compare it with fixed-task classification heads and prompting for label text. Hold out entire tasks and label taxonomies, not just new rows from familiar tasks.

Compare a standard supervised objective with a justified probability-quality objective or post-hoc calibration. Use separate calibration data for temperature scaling. Report winning-label quality and probability metrics separately, along with model size, hardware, batch size, and end-to-end latency. Test candidate count, order, and description changes.

**Research boundary:** an API-compatible implementation establishes software compatibility. Cross-task generalization and calibration require measurements. The public description of RLCD does not provide its proprietary recipe, so proposed alternatives should be named according to what they actually implement.

## A minimal experiment record

| Field | What to record |
| --- | --- |
| Hypothesis | A prediction and an observation that could weaken it |
| Evaluation unit | A unique message, task, trace, or repeated call, as appropriate |
| Data provenance | Who created the examples, how labels were assigned, and known ambiguities |
| Splits | Grouping rule and what development decisions were made before testing |
| Model and question versions | Requested and returned model IDs, question definitions, and option order |
| Controls | Fixed tools, specialists, budgets, retries, context selection, and fallback |
| Measurements | Numerators, denominators, latency boundary, token accounting, and uncertainty |
| Failure categories | Misdefinition, missing evidence, missing candidate, misclassification, or downstream failure |
| Evidence status | Proposed, exploratory, development-selected, or final held-out result |

Begin with the notebook's `research_plan` dictionary and replace its fields with your group's design. The goal is a result that another student can assess without guessing what changed between conditions.
