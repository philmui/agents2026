# The calibration dataset

**30 news articles and 480 machine-written summaries of them, each rated by
three human experts on four dimensions.**

You cannot calibrate a judge without something to calibrate against. This folder
is that something: real human judgement of real model output, which is otherwise
the single hardest thing to get hold of when you are learning to build evals. The
notebook loads it in the calibration section (see notebook §9 and `EVALS_THEORY.md`
section 8).

## Where it comes from

[**SummEval**](https://github.com/Yale-LILY/SummEval) (Fabbri et al., 2021), via
the `mteb/summeval` mirror. MIT licensed. The articles are from the CNN/DailyMail
news corpus; the summaries were produced by a range of summarisation models,
including deliberately weak ones, which is why the dataset contains more failures
than a healthy production system ever would.

## The files

| File | What is in it |
| --- | --- |
| `articles.jsonl` | `id`, `text`: the source article |
| `judgments.jsonl` | `article_id`, `summary_id`, `summary`, and four scores from 1 to 5 |

Each score is the **mean of three** expert annotators.

| Dimension | The question it asks |
| --- | --- |
| **consistency** | Is everything in the summary actually supported by the article? Hallucination, measured. |
| **relevance** | Does it capture what mattered? |
| **coherence** | Does it hold together as a piece of writing? |
| **fluency** | Are the sentences well formed? |

## Two things worth knowing before you use it

**The experts disagree with each other, and how much depends on the dimension.**
How much exactly is something this file cannot tell you, and that is the first
lesson hiding in it.

Each score here is the **mean** of three annotators; the individual ratings were
never published. A mean destroys information. Some means give the triple away and
most do not:

| The mean | What the three annotators must have said |
| --- | --- |
| `1.0` | `{1,1,1}` unanimous, provably |
| `5.0` | `{5,5,5}` unanimous, provably |
| `1.333` | `{1,1,2}` split, provably |
| `2.0` | `{2,2,2}` **or** `{1,2,3}` **or** `{1,1,4}`, unknowable |

So unanimity can only be **bracketed**:

| Dimension | Agreed at least | At most |
| --- | --- | --- |
| consistency | **80%** | 83% |
| fluency | **67%** | 72% |
| coherence | **6%** | 32% |
| relevance | **4%** | 35% |

**Read the left column.** The right one counts every whole-number mean, and three
annotators answering **completely at random** would produce a whole mean **33%**
of the time. So an "at most" near that number carries no information about
agreement whatsoever.

The ordering survives anyway, and now it is proved rather than asserted:
consistency's *lower* bound sits above coherence's *upper* bound, so the two
intervals cannot overlap however the hidden ratings fell. **Consistency is close
to a fact. Coherence is closer to an opinion.** That is not a flaw in the dataset,
it is the thing being measured, and it should change what you ask a judge to do.
Demanding that your judge agree with you about coherence is demanding more
agreement than three trained humans managed with each other.

> Worth keeping when you leave: the first question about any metric is what its
> value would read if the thing it measures were absent. Here that floor is 33%,
> and two of these four dimensions land on it.

**The distribution is lopsided, as real data always is.** 32 of 480 summaries
(6.7%) are rated 2 or below for consistency. Sample twenty at random to test
whether your judge catches hallucination and you would expect **about one** bad
example. You cannot measure a detector on one example, which is the whole argument
for generating your own failures (notebook section 10).

## Licence

MIT, from the SummEval repository. The underlying articles are CNN/DailyMail news
text, distributed for research use as part of that dataset.
