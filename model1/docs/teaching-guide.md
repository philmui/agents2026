# Teaching the tutorial

This sequence assumes basic algebra and an interest in science or computing. The notebook additionally needs familiarity with Python lists, dictionaries, and functions. Students do not need previous machine-learning experience.

Begin with a message students can route themselves. Ask which words and facts support their decision before introducing a model. Return to that same task as representations become more sophisticated; students then see what changed and why it matters.

## Suggested pacing

| Session | Materials | Student activity |
| --- | --- | --- |
| 1: From messages to features, 45–60 minutes | Slides 1–22; notebook Steps 1–3 | Build a count vector, compare reordered sentences, run the toy classifier, and inspect unfamiliar wording. |
| 2: Typed decisions and workflows, 45–60 minutes | Slides 23–44; notebook Steps 4–9 | Read Choice, Noul, and Score; calculate a weighted score; route an ambiguous message. |
| 3: Agents and fair research, 45–60 minutes | Slides 45–52; notebook Steps 10–14 | Compare traces, interpret calibration and coverage, and draft an experiment with a held-out test set. |

The deck and notebook contain more material than a single short lesson. For younger students, spread the first session over two periods and treat naive Bayes's log calculation as an optional mathematical extension. For self-study, read the slide explanations and speaker notes before each corresponding notebook stage.

## Questions that reveal understanding

**Word counts:** If “screen” appears three times instead of twice, does the vector get longer? The count changes in its existing position; the vocabulary fixes the length. If two sentences have identical counts but different meanings, the classifier never receives the missing order information.

**Training and testing:** Why not build the vocabulary from every message? Using test examples to choose features lets future evaluation data influence the model. Some workflows intentionally use unlabeled deployment data, but that is a different protocol and must be described explicitly.

**Representations:** Why does “river bank” challenge a static embedding? The lookup vector for bank stays the same, although a later model can add context. A contextual model changes the representation using its surroundings. An embedding axis is usually a learned number rather than a named human concept.

**Choice and Noul:** Can three urgency properties all have probability 0.9? Yes, if they are three different questions. Probabilities sum to one across alternatives within one categorical question, not across unrelated properties.

**Score:** Can two responses both score 1.0 but express different uncertainty? Yes. Probability concentrated on level 1 differs from a split between levels 0 and 2. Read the distribution, not just its mean.

**Confidence:** Is Choice confidence an independently verified chance of correctness? No. It is a statistic computed from the distribution. A high value can accompany a wrong prediction, and threshold quality must be checked on labeled data.

**Agents:** If a model predicts that a tool operation is acceptable, who permits the action? The application's code and authorization rules. A semantic model prediction cannot create user permission.

**Judge evidence:** Why are five runs judged 100 times still only five distinct cases? Repetitions help estimate variability on fixed behavior; they do not add new task types. Agreement on held-out traces answers a different question from repeated-call consistency.

## Small activities

1. Give groups the same six messages and let them define three categories plus a fallback. Compare disagreements before discussing model predictions. This makes ambiguity in the answer key visible.
2. Have students draw vectors for “student helps teacher” and “teacher helps student,” then add bigram features. Ask which distinctions return and which still disappear.
3. Provide two Score distributions with the same average. Ask students to identify when the application should review the result.
4. Let students adjust the synthetic routing threshold. They should report both automatic coverage and accepted-case accuracy, including the undefined case when nothing is accepted.
5. Give students a fluent but unsupported agent answer. Have them distinguish tool-call presence, evidence agreement, and task completion.
6. Ask each group to write a hypothesis, a baseline, an evaluation unit, and one observation that would weaken the hypothesis. Use [the roadmap](research-roadmap.md) for project ideas.

## Evidence labels to preserve while teaching

The naive Bayes fit is an executed toy experiment. The Jev-shaped offline responses are invented fixtures. The calibration dataset is constructed so the first probabilities match frequencies. The live cells are optional examples rather than completed experiments in the supplied output. The proposed research projects have no claimed findings.

The title “System One” refers to TypeSafe's model category. Its human-cognition analogy does not demonstrate a particular internal cognitive process. Explain Jev through its documented inputs, outputs, and observable behavior, while keeping its proprietary training details separate.

At the end, ask students to explain a representation, a typed response, a routing policy, and a fair evaluation in their own words. An accurate explanation of those connections is more useful than recalling a sequence of model names.
