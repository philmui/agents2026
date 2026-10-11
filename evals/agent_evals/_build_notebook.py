"""Builds evals_tutorial.ipynb from a list of (type, source) cells.

Run:  python3 _build_notebook.py   (or: uv run python _build_notebook.py)

This keeps the notebook JSON well-formed and easy to regenerate. The script
itself is not part of the tutorial; it is a build tool. It mirrors the build
pattern used by ../18_tau2/_build_notebook.py so the modules stay stylistically
consistent.

Why a build script instead of hand-editing the .ipynb? A notebook is a big JSON
file. Editing JSON by hand is error prone (one missing comma breaks it). Here we
write the lesson as plain Python strings and let json.dump produce valid JSON
every time. To change the notebook, edit the md()/code() calls below and re-run.

IMPORTANT BUILD NOTE: several code cells contain triple-double-quoted Python
docstrings. To keep those inner docstring quotes from closing the outer string,
every md()/code() body below is wrapped in triple-single-quotes. Do not put a
triple-single-quote sequence inside any body.

House rules for this module: the audience is a curious learner (think an advanced
high-school researcher), so every concept and every line of code is explained.
There are NO em-dashes anywhere (commas, parentheses, and colons instead). The
notebook is self-contained: it depends only on the pinned pyproject deps and the
two data files in data/, never on any shared helpers package. The API-backed
sections use the OpenAI SDK and cache every result to artifacts/, so a second run
is instant and cheap. The conceptual companion is EVALS_THEORY.md; the notebook
cross-references it by section number (for example, see THEORY section 8).
"""
import json

# Each entry is ("md", "markdown text") or ("code", "python source").
CELLS = []
def md(text):   CELLS.append(("md", text.strip("\n")))
def code(text): CELLS.append(("code", text.strip("\n")))

# ============================================================================
# TITLE + OVERVIEW
# ============================================================================
md(r'''
# Evals from Scratch: Measuring LLMs and Agents You Can Defend

### A hands-on, build-it-yourself tutorial

![From an answer eval to a trajectory eval](assets/01-eval-loop.svg)

**Start with something familiar.** Imagine your school swaps in a new grading app and
a teacher says, "I think it grades essays better than the old one." That is a *feeling*.
It might be true. But the principal cannot spend money on a feeling, and you cannot tell
whether "better" means fewer mistakes, or just different mistakes. To settle it, you would
do the obvious thing: take a stack of essays a human already graded, feed them to both
apps, and count how often each one matches the human. Now "better" is a **number** you can
compare, repeat, and show to anyone.

That is the whole idea of this tutorial, and it has a name.

> An **eval** (short for *evaluation*) is a repeatable test that turns "it seems better"
> into a number you can trust and defend. We use it to measure large language models
> (**LLMs**, the AI systems like ChatGPT that read and write text) and the programs built
> on top of them.

The number is never the *whole* truth. But it is comparable across versions, it survives
the meeting you are not in, and it is the only thing that can tell you a change actually
helped rather than just moving the problem somewhere you were not looking.

> **Why this matters beyond a grade.** The same number a manager uses to decide whether
> to fund a project next month is the one an engineer uses to decide whether to ship a
> change today. Very few things speak to both the business and the builder. A good eval
> does. That is why learning to build one is a genuinely useful skill.

**What you will build in this notebook**, step by step, starting from nothing:

1. A tiny **dataset** (a list of questions with known-correct answers).
2. A **harness** (the small program that runs a system and scores its answers).
3. Simple **scorers** (functions that turn one answer into a number).
4. An **LLM judge** (using an AI to grade answers that plain code cannot check).
5. **Calibration** (the honesty check: does the AI judge agree with real humans?).
6. **Synthetic failures** (deliberately broken answers, because real mistakes are rare).
7. A **confidence interval** (a way to tell a real improvement from random luck).
8. **Agentic evals** (grading AI that *does things*, where the final answer is only part
   of the story).

You do not need any AI or machine-learning background. If you can read a little Python
(variables, functions, lists, and dictionaries), you can follow every line. Anything
beyond that, we explain the first time it shows up.

> The companion file [`EVALS_THEORY.md`](EVALS_THEORY.md) tells the same story from the
> *why* angle, with diagrams and, for every idea, a note on where it breaks. **This
> notebook is the hands-on half**: every idea there appears here as code you can run.
> The two point at each other by section number, so you can read them side by side.

The point of all of it is not the score. It is a number you can *defend*, and the
handful of habits that make it one. (One small note: there are no em-dashes anywhere,
on purpose, and every code cell is explained in the text around it.)
''')

md(r'''
## Table of contents

- **§0. Plain-language glossary** : every term this notebook uses, in one sentence each.
- **§1. Setup** : keys, models, data, and a tiny cache so reruns are free.
- **§2. Where ground truth comes from** : back-testing, and reading your own traces.
- **§3. The harness** : three functions and a loop.
- **§4. Scorers you can compute** : free, instant, deterministic checks.
- **§5. Two dumb baselines** : echo and oracle, which test the harness itself.
- **§6. Direct feedback** : the one field that makes a thumbs-down worth collecting.
- **§7. LLM as judge** : a scorer with a rubric, and why you cannot trust it yet.
- **§8. Calibrate the judge** : score real summaries against real human ratings.
- **§9. Separation versus agreement** : the two questions calibration answers.
- **§10. Generate the failures you need** : synthetic corruptions, pre-labeled by construction.
- **§11. Is the difference real?** : the bootstrap confidence interval.
- **§12. MVP to PoC to production** : one harness, three questions.
- **§13. Where a framework fits** : what Ragas adds, and what stays yours.
- **§14. Agentic evals** : the answer is not the whole story.
- **Conclusion, references, and next steps.**

If any bolded term below is new, the glossary in §0 defines each one in a sentence.
''')

# ============================================================================
# GLOSSARY
# ============================================================================
md(r'''
---
## §0. Plain-language glossary (read this first)

New fields come with new words, and evals are no exception. None of them are hard once
someone says them plainly, so here they all are, once, up front, in one sentence each.
Do not try to memorize this list. Skim it now so the words feel familiar, and come back
whenever one trips you up. Every term also gets a fuller, worked-out explanation in the
section where it first matters.

- **LLM (large language model).** An AI system trained on huge amounts of text that
  can read a prompt and write a reply. ChatGPT is one. It is the "system" we test.
- **Eval (evaluation).** A repeatable test that turns a system's output into a number
  you can compare across versions. Think of it as a vibe check made fair: run the same
  way every time, on examples you did not hand-pick to look good.
- **Ground truth.** The answer we treat as correct, decided by something *other* than
  the system being tested. Usually a human graded it, or it is a record of what really
  happened. It is the answer key.
- **Case (or example).** One input plus whatever we know about the correct output. A
  whole list of cases is a **dataset**.
- **Golden example.** A case you wrote by hand and are confident about. Your first and
  most trusted answer key.
- **Scorer.** A small function that reads the system's answer and returns a number
  (in this notebook, always from 0 to 1, where 1 means perfect). A dataset plus a few
  scorers *is* an eval.
- **Pass rate.** Out of all the cases, the fraction that cleared every scorer's bar.
  This is usually the single headline number, and the one most often quoted carelessly.
- **LLM judge.** When plain code cannot check something (like "is this summary faithful
  to the article?"), we hand the answer to *another* AI, give it a grading rubric, and
  let it score. That grader is the judge.
- **Calibration.** The honesty check: before trusting an AI judge, we compare its scores
  against real human scores. The step nearly everyone skips, and the heart of this notebook.
- **Separation.** Can the judge tell a clearly bad answer from a clearly good one at all?
  The easy bar to clear.
- **Agreement.** Does the judge score the way a human would? The hard bar.
- **Chance floor.** The score a metric would show even if the thing it claims to measure
  were completely absent (for example, if the graders were just guessing). Always find
  this floor before you trust a number.
- **Back-testing.** Running your system on inputs from the *past* and comparing its
  answers to what actually happened. Often the cheapest high-quality eval there is,
  because the answer key already exists.
- **Synthetic failure.** A wrong answer you created on purpose by breaking a known-good
  one. Because you broke it yourself, you already know it is wrong and exactly how.
- **Base rate.** How common something is in your real data. Real mistakes are usually
  rare (a low base rate), which is why we have to manufacture them to test a detector.
- **Confidence interval (CI).** A range, like [0.81, 0.94], that honestly says: if we
  had collected a different batch of examples the same size, this is roughly how much
  the number would wobble.
- **Bootstrap.** A clever trick for computing that range: re-shuffle the data you
  already have thousands of times and watch how much the answer moves. No fancy
  formulas or assumptions needed.
- **MDE (minimum detectable effect).** The smallest improvement your test set is big
  enough to actually notice. Chase a difference smaller than this and you are measuring
  luck, not skill.
- **Agent.** An LLM placed inside a loop so it can *do things*: call tools, read the
  results, and decide the next step, instead of just replying once.
- **Trajectory.** The complete recording of what an agent did: every tool it called,
  every result it got back, and its final answer. Agent evals grade this whole story,
  not just the last line.
- **pass^k.** The fraction of tasks an agent gets right on *all* k tries in a row. It
  catches flakiness that a single lucky run would hide.
''')

# ============================================================================
# SECTION 1: SETUP
# ============================================================================
md(r'''
---
## §1. Setup

Every project needs a little setup before the fun starts. We keep ours tiny and out in
the open so nothing feels like magic. Three things happen in the cell below, and that
is all:

1. **We load a secret key.** To talk to OpenAI's models we need an **API key** (a
   password that proves the request is yours). We keep it in a separate file named
   `.env` and load it from there, so it never gets pasted into the code or shared by
   accident. If you have not made that file yet, copy `.env.example` to `.env` and paste
   your key in. (No key? No problem, see the note below.)
2. **We pick two models.** One model plays the *system being tested* (it answers
   questions), and a **different** model plays the *judge* (it grades answers later).
   Keeping them different matters: an AI grading its own work tends to go easy on
   itself, exactly like a student marking their own test. We come back to this in §9.
3. **We set up a cache.** A **cache** just means "save each answer to a file so we never
   have to ask twice." The first run makes real requests and costs a few cents; every
   run after that reads the saved answers from the `artifacts/` folder, so it is instant
   and free.

> **You do not need a key to learn from this notebook.** Only two sections (§7 and §9)
> talk to an AI, and their results are already saved in `artifacts/`, so they work
> offline too. Everything else (building the harness, the scorers, the fake failures,
> the statistics) runs on your own machine with no key and no internet.
''')

code(r'''
# Standard library only, so far. Every import is used within a cell or two.
import json          # read/write the JSONL data and the cache
import os            # read environment variables (the API key, model overrides)
import re            # the deterministic scorers and the corruption functions
import hashlib       # turn a long prompt into a short, stable cache filename
from pathlib import Path            # filesystem paths that work on any OS
from collections import Counter     # token counting for the F1 scorer
from dataclasses import dataclass, field   # a small typed record for one result

import numpy as np   # sampling, correlation, and the bootstrap in §11

# python-dotenv reads a local .env into the environment. find_dotenv walks up the
# directory tree to the nearest .env, so this resolves your key no matter which
# folder you launched Jupyter from.
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Where the notebook lives, and where its data and cache go. Path.cwd() is the
# folder Jupyter started in, which for this project is the module folder itself.
HERE = Path.cwd()
DATA = HERE / "data"
ARTIFACTS = HERE / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)   # created on first run; git ignores it

# The models. A separate JUDGE_MODEL keeps "the thing being scored" and "the thing
# doing the scoring" distinct, which is a real calibration concern later. Both are
# small, cheap, widely available OpenAI models; override them in .env if you like.
SYSTEM_MODEL = os.environ.get("SYSTEM_MODEL", "gpt-4.1-mini")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1-nano")

print("Setup ready.")
print(f"  system model : {SYSTEM_MODEL}")
print(f"  judge model  : {JUDGE_MODEL}")
print(f"  data folder  : {DATA}  (exists: {DATA.exists()})")
''')

md(r'''
### A reader and a cache

Two tiny helpers do a lot of work in this notebook.

`read_jsonl` reads a **JSONL** file (one JSON object per line, a common format for
datasets because you can append to it without rewriting the whole file). Our data
lives in two such files: `articles.jsonl` and `judgments.jsonl`.

`cached_call` is the reason reruns are free. Every time we ask a model something, we
first hash the exact request into a short filename. If that file already exists in
`artifacts/`, we return the saved answer and make no network call. Otherwise we call
the API once and save the result. This is not just a cost optimization: a cached run
is also **reproducible**, which is the whole spirit of an eval.
''')

code(r'''
def read_jsonl(path: Path) -> list[dict]:
    """Read a file with one JSON object per line into a list of dicts."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()            # skip any blank lines
    ]

# Load the calibration data now so later cells can assume it is present.
# ARTICLES maps an article id to its full text; JUDGED is the list of 480 rated
# summaries. See data/README.md for provenance (SummEval, MIT licensed).
ARTICLES = {a["id"]: a["text"] for a in read_jsonl(DATA / "articles.jsonl")}
JUDGED = read_jsonl(DATA / "judgments.jsonl")

print(f"Loaded {len(JUDGED)} human-rated summaries of {len(ARTICLES)} articles.")

# The OpenAI client is created lazily so the offline sections need no key. We make
# it the first time cached_call actually has to hit the network.
_client = None
def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI   # imported here so import errors surface late
        _client = OpenAI()          # reads OPENAI_API_KEY from the environment
    return _client

def cached_call(model: str, prompt: str, temperature: float = 0.0) -> str:
    """Ask a model one question, but only ever once per (model, prompt).

    The result is cached to artifacts/ keyed by a hash of the exact request, so a
    second run reads from disk: free, instant, and identical to the first.
    """
    key = hashlib.sha256(f"{model}|{temperature}|{prompt}".encode()).hexdigest()[:16]
    cache_file = ARTIFACTS / f"call_{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())["reply"]

    # Cache miss: make the single API call. temperature=0 asks for the model's most
    # likely (least random) answer, which is what you want for a repeatable judge.
    reply = (
        _get_client()
        .chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        .choices[0]
        .message.content
    )
    cache_file.write_text(json.dumps({"reply": reply}))
    return reply
''')

# ============================================================================
# SECTION 2: GROUND TRUTH
# ============================================================================
md(r'''
---
## §2. Where the answer key comes from

To grade anything you need an answer key. In eval-speak that key is called **ground
truth**: the answer we treat as correct because a human decided it or because it is a
record of what really happened. The most common excuse for not building an eval is "but
we have no answer key, nobody labeled our data." That excuse is almost always wrong, and
seeing why is the first real lesson.

![Answer keys you already own, and back-testing](assets/02-ground-truth-sources.svg)

People sit on mountains of labeled data without realizing it, because the label was a
**side effect of doing the work**. You do not have to create the answer key; the past
already wrote it down for you:

| What you already have | The hidden answer key inside it |
| --- | --- |
| Support tickets that got resolved | Which team fixed it, and whether it came back |
| Documents that got approved | The fact that a human approved them |
| Past decisions with known results | How each one turned out |
| Email threads that ended | Who answered, and whether they had to follow up |
| Anything with a history log | Every step, with a timestamp |

Running your system on those *past* inputs and comparing its answers to what actually
happened is called **back-testing**. It is often the cheapest high-quality eval you can
get, because the answer key already exists. So before asking anyone to sit down and
label fresh data (which costs time, money, and goodwill), ask the free question first:
"what did we do last year, and did it work?"

### The other half: figuring out *what* to grade

Back-testing tells you *whether* an answer was right. It does not tell you **how your
system tends to go wrong**, and until you know that, every scorer you write is really
just a guess about which mistakes matter.

The way you find out is unglamorous, and it is most of the job: **read the records of
what your system actually did, one at a time, and write down what went wrong.** (The
full recording of one run, the input and the output and everything in between, is called
a **trace**.) Professionals who build evals spend **60 to 80% of their time** on this
reading, not on writing code. The steps are boring on purpose:

1. **Grab fifty to a hundred real traces.** If your data is lopsided (say 93% of answers
   are fine and only 7% are broken), deliberately pull *extra* broken ones. Picking
   purely at random from a mostly-fine pile teaches you almost nothing about the
   failures, and the failures are the point.
2. **Write one plain sentence per mistake, in your own words.** A sentence, not a label.
   "It answered using the old policy instead of the current one." If you invent tidy
   categories *before* reading, you will spend the whole time forcing real mistakes into
   boxes you guessed at, and miss the mistake you never imagined.
3. **Now group the sentences.** Let the categories rise up out of the sentences you
   actually wrote, instead of the other way around.
4. **Count each group and sort them.** Now you know what deserves a scorer, and (just as
   useful) what does not.
5. **Stop when it goes quiet.** Once about twenty traces in a row show nothing new, you
   have seen the shape of how your system fails.

> **That ranked list of failures is the real prize, more than any single score.** A list
> of how your system actually breaks, with counts, is the most useful thing you can hand
> the person deciding whether to keep funding the project. A lone number cannot be acted
> on, only argued about. (See THEORY section 2 for the full argument.)

Everything from §3 on assumes you have done this reading and know what you are trying to
measure. That is a big assumption, and skipping it is why so many beautifully built eval
systems end up measuring the wrong thing.
''')

# ============================================================================
# SECTION 3: THE HARNESS
# ============================================================================
md(r'''
---
## §3. The harness

Now we build the machine that does the grading. In the eval world this machine is
called a **harness**, and the good news is that it is tiny: **three small functions and
one loop.** Everything you have ever seen on top of evals (fancy dashboards, big
frameworks, tracing tools) is just convenience wrapped around these three pieces. We
keep it small on purpose, because you will be staring at its output for weeks and you
have to trust every number it prints.

![The harness: three functions and a loop](assets/03-harness.svg)

Here is what each piece does, in plain terms:

- **`Result`** holds one graded answer: the question (the *case*), what the system
  actually said, and the scores it earned. Think of it as one row in a gradebook, with
  enough detail that you can later ask "wait, why did this one fail?"
- **`run_eval`** is the loop. It walks through every case, runs the system once on each,
  and hands the answer to every scorer. Out comes a stack of `Result` rows.
- **`report`** adds it all up: the average score for each scorer, plus the overall pass
  rate. This is the summary you actually show people.

One detail is worth pausing on. Notice that `system` can be *any function* that takes an
input and returns an output. We keep it that loose on purpose. That single design choice
is what lets this same little harness grade a simple prompt today, and grade a full agent
tomorrow, without changing a line. Every comparison later in the notebook leans on it.
''')

code(r'''
@dataclass
class Result:
    """One case, one score set, and enough context to argue about it."""
    case: dict                              # the input, expected output, and flags
    actual: str                             # what the system actually produced
    scores: dict = field(default_factory=dict)   # scorer name -> number in 0..1

    @property
    def passed(self) -> bool:
        # A case passes only if EVERY scorer clears the bar. 0.5 is a convention;
        # the deterministic scorers below return exactly 0.0 or 1.0, so the bar is
        # really "did every hard check succeed".
        return all(v >= 0.5 for v in self.scores.values())

def run_eval(cases: list[dict], system, scorers: dict) -> list[Result]:
    """Run `system` over every case and score it every way.

    system:  a callable, input -> output. Loose on purpose (see the note above).
    scorers: a dict of {name: fn}, where fn(actual, case) -> float in 0..1.
    """
    results = []
    for case in cases:
        actual = system(case["input"])                      # run the system once
        scores = {name: fn(actual, case) for name, fn in scorers.items()}
        results.append(Result(case=case, actual=actual, scores=scores))
    return results

def report(results: list[Result]) -> dict:
    """Aggregate results into one row: the mean of each scorer, plus the pass rate."""
    if not results:
        return {}
    names = results[0].scores.keys()
    out = {
        name: sum(r.scores[name] for r in results) / len(results)
        for name in names
    }
    out["pass_rate"] = sum(r.passed for r in results) / len(results)
    out["n"] = len(results)
    return out

print("Harness ready: Result, run_eval, report.")
''')

# ============================================================================
# SECTION 4: SCORERS
# ============================================================================
md(r'''
---
## §4. Scorers you can compute

A **scorer** is just a small function that looks at one answer and returns a number
between 0 and 1, where 1 means "perfect" and 0 means "totally wrong." We start with the
scorers that need no AI at all. They are free, they run instantly, and they give the same
answer every single time (that last property has a fancy name, *deterministic*, but it
just means "no surprises"). People underestimate these plain scorers. They catch a
surprising amount.

![The scorer ladder](assets/04-scorer-ladder.svg)

We will write four of them, from strictest to most forgiving:

- **exact_match**: the pickiest. The answer has to match the expected answer *exactly*.
  This is right when "close" is still wrong, like an order ID, a category, or which team
  a ticket should go to.
- **token_f1**: measures how many words the answer and the expected answer share. It
  forgives different phrasing, so it fits free-form text where there is no single correct
  wording. (Do not worry about the name for now; the code below explains it.)
- **contains_required**: did the answer include the things it absolutely had to include?
  This one is quietly the most useful in real jobs. "Did the answer actually cite the
  policy number?" is a real requirement, and checking it is just looking for that text
  inside the answer.
- **declines**: did the system correctly refuse to answer when it should have? A system
  that *never* says "I don't know" is not brave, it is untrustworthy: sooner or later it
  will confidently make something up.

Every scorer returns a number from 0 to 1, so they all add up and average the same way.
That shared scale is what lets us mix them freely later.
''')

code(r'''
def _tokens(text: str) -> list[str]:
    """Lowercase a string and split it into word-and-number tokens."""
    return re.findall(r"[a-z0-9]+", str(text).lower())

def exact_match(actual: str, case: dict) -> float:
    """1.0 only if the outputs match exactly (after trimming and lowercasing)."""
    return float(
        str(actual).strip().lower() == str(case["output"]).strip().lower()
    )

def token_f1(actual: str, case: dict) -> float:
    """Harmonic mean of token precision and recall. Forgiving of wording."""
    a, b = Counter(_tokens(actual)), Counter(_tokens(case["output"]))
    common = sum((a & b).values())         # tokens shared, counting multiplicity
    if not common:
        return 0.0
    precision = common / sum(a.values())   # of what it said, how much was right
    recall = common / sum(b.values())      # of what was right, how much it said
    return 2 * precision * recall / (precision + recall)

def contains_required(actual: str, case: dict) -> float:
    """Fraction of the must-include strings that appear in the output."""
    required = case.get("must_include", [])
    if not required:
        return 1.0                          # nothing required, so trivially satisfied
    text = str(actual).lower()
    return sum(str(r).lower() in text for r in required) / len(required)

def declines(actual: str, case: dict) -> float:
    """1.0 if the output refuses, but only scored on cases marked should_decline."""
    if not case.get("should_decline"):
        return 1.0                          # not a decline case, so not penalized
    signals = (
        "i don't know", "i do not know", "cannot", "can't",
        "unsure", "escalate", "not enough information", "unable",
    )
    return float(any(s in str(actual).lower() for s in signals))

SCORERS = {
    "exact": exact_match,
    "f1": token_f1,
    "required": contains_required,
    "declines": declines,
}
print("Four deterministic scorers ready:", list(SCORERS))
''')

md(r'''
### A tiny "answer key" dataset

To grade anything, we need a small set of questions we already know the answers to. In
eval-speak this is called a **golden dataset** ("golden" as in the gold standard, the
answers we trust). Normally you would build it from the trace-reading work in §2. But so
that this notebook runs the same for everyone, with or without real data, here is a small
hand-written set about a made-up company's policies.

Each case has three parts: the `input` (the question), the expected `output` (the correct
answer), and sometimes an extra flag like `must_include` or `should_decline` that tells a
specific scorer what to check. In a real project you would keep these in a file called
something like `golden.jsonl`, and it would grow a little every time a user finds a
failure nobody imagined (that is the loop we build in §6).
''')

code(r'''
GOLDEN = [
    {
        "input": "What is the refund window for online orders?",
        "output": "30 days from delivery.",
        "must_include": ["30 days"],
    },
    {
        "input": "Which team handles a billing dispute over $500?",
        "output": "escalations",
    },
    {
        "input": "What is the CEO's personal mobile number?",
        "output": "I do not have that information.",
        "should_decline": True,
    },
    {
        "input": "Can I return a gift card for cash?",
        "output": "No. Gift cards are non-refundable.",
        "must_include": ["non-refundable"],
    },
    {
        "input": "How long are support tickets kept before archiving?",
        "output": "90 days.",
        "must_include": ["90 days"],
    },
]
print(f"{len(GOLDEN)} golden cases: your first ground truth.")
''')

# ============================================================================
# SECTION 5: BASELINES
# ============================================================================
md(r'''
---
## §5. Two dumb baselines, run first

Here is a question people almost never ask: *who grades the grader?* Before you trust the
harness to judge a real system, you should make sure the harness itself works. The
simplest way is to feed it two deliberately silly systems whose scores you can predict in
advance, and then check that the numbers come out the way you expect. If they do not, the
bug is in your scorer, not in whatever you were about to test.

![Two dumb baselines bracket the range](assets/05-baselines.svg)

These two "silly systems" are called **baselines**, because they set the floor and the
ceiling that any real system lives between:

- **echo** just repeats the question back as its answer. It should score near the
  *bottom*. If your real system cannot beat plain echo, something is broken before the
  answer ever reaches the model.
- **oracle** cheats: it looks up the correct answer and hands it back. It should score a
  perfect **1.0** on the answer scorers. And here is the useful part: if the oracle does
  *not* score 1.0, then your scorer is broken, because you literally gave it the right
  answer and it still marked it wrong. Far better to catch that now than to have a real,
  good change "fail" later for a reason that had nothing to do with the change.

Most teams never test their harness this way. We are about to.
''')

code(r'''
# echo: the terrible floor. Returns its input unchanged.
baseline = run_eval(GOLDEN, lambda x: x, SCORERS)

# oracle: cheats by looking up the expected answer for each input.
_by_input = {c["input"]: c["output"] for c in GOLDEN}
oracle = run_eval(GOLDEN, lambda x: _by_input[x], SCORERS)

def show(title, rep):
    fmt = {k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in rep.items()}
    print(f"{title:>16} | " + "  ".join(f"{k}={v}" for k, v in fmt.items()))

print("Sanity check: can the harness tell these two apart?\n")
show("echo the input", report(baseline))
show("oracle (cheats)", report(oracle))
''')

md(r'''
Read the two rows. The oracle scores 1.0 on `exact`, `f1`, and `required`, which
proves those three scorers are wired up correctly (an oracle that fed them the right
answer and still scored below 1.0 would mean the scorer, not the system, was broken).
Echo scores far lower, so the harness clearly separates a good system from a bad one.

> **In the enterprise:** these four scorers cost nothing per run, which is what lets
> them run on *every commit*, and a scorer that runs on every commit is the only kind
> that catches a regression before a user does. The judge you build in §7 costs a model
> call per case. The decision that forces is which scorers gate the merge and which run
> nightly on a sample. Teams that put a slow, paid judge in the commit gate watch it get
> disabled within a month, for cost or for flakiness, and then have no gate at all.
''')

# ============================================================================
# SECTION 6: FEEDBACK
# ============================================================================
md(r'''
---
## §6. Direct feedback, and the one field that matters

The scorers so far only check things you thought to check in advance. But your users will
hit problems you never dreamed of. The way you learn about those is the oldest one there
is: you ask them. This section is about doing that well, because most feedback is
collected in a way that throws away the useful part.

Think about the thumbs-up / thumbs-down buttons you have seen on AI answers. A lonely
thumbs-down tells you *something* went wrong, but not *what*, and "something was wrong
somewhere" is a complaint you can never reproduce or fix. The fix is one small rule:
**when someone gives a thumbs-down, require them to type a short reason.** That sentence
is the real data. The thumb is just the trigger that gets you the sentence.

Below is a stripped-down version of a feedback function. It is short on purpose. The one
line that carries all the weight is the guard that refuses to save a negative rating with
no reason attached.
''')

code(r'''
# A stand-in for a real feedback store. In production this writes to a database.
FEEDBACK_LOG = []

def record_feedback(answer_id: str, helpful: bool, reason: str = "") -> str:
    """Record one judgement on one answer. Requires a reason on negatives.

    Returns a short message to show the user. The `reason` requirement is the whole
    point: a negative thumb with no reason is a complaint you cannot reproduce.
    """
    if not helpful and not reason.strip():
        return "Please say what was wrong. That part is the useful bit."
    FEEDBACK_LOG.append({"answer_id": answer_id, "helpful": helpful, "reason": reason})
    return "Thanks."

# A user tries to thumbs-down with no reason (rejected), then adds one (accepted).
print(record_feedback("ans_017", helpful=False))
print(record_feedback("ans_017", helpful=False, reason="cited the 2019 policy, not the current one"))
print("feedback log now holds:", FEEDBACK_LOG)
''')

md(r'''
Three simple rules make feedback actually worth collecting:

1. **Ask right when the person is using the product**, not in a survey a week later when
   they have forgotten the details.
2. **Require the reason whenever the rating is negative.** One honest sentence beats a
   hundred anonymous thumbs.
3. **Save the question and the answer next to the reason.** Without them you have a
   complaint you can never reproduce, and you cannot fix what you cannot reproduce.

Here is the payoff, and it closes the loop of the whole course: a negative rating *with a
reason* is a brand-new eval case. Users keep finding failures you never imagined, and
each reason they write becomes a permanent test that guards against that failure forever.
The example above ("cited the 2019 policy, not the current one") is exactly the kind of
sentence that, once you have gathered a pile of them and grouped them like in §2, tells
you the next scorer to write.
''')

# ============================================================================
# SECTION 7: LLM AS JUDGE
# ============================================================================
md(r'''
---
## §7. LLM as judge, and why you cannot trust it yet

Some questions plain code just cannot answer. "Is this summary actually faithful to the
article it came from?" "Is this tone right for a worried customer?" There is no substring
you can search for and no set of words to overlap. Judging that kind of thing takes
reading and understanding. So we do the natural thing: we ask *another* AI to read the
answer and grade it. When an LLM plays the role of grader, we call it an **LLM judge**.

Here is the trap, and it is the heart of this whole section. Building a judge is easy:
you write a prompt asking it to score something, and it gives you a number. **The hard
part, the part almost everyone skips, is figuring out whether you should believe that
number at all.** We build the judge now, and then in §8 and §9 we put it on trial before
we trust it.

A **rubric** is just the grading instructions we give the judge (the same idea as a
rubric a teacher hands out). One more careful detail below: we ask the judge to reply in
a strict format called JSON so our code can read the score reliably. If the judge ever
breaks and sends back something we cannot read, we count that as a *failed* grade worth
0, never as a silent pass. A judge that quietly says "looks fine" whenever it malfunctions
is more dangerous than having no judge at all.
''')

code(r'''
JUDGE_PROMPT = """You are scoring one answer against a reference.

Question:
{input}

Reference answer (written by a domain expert):
{expected}

Answer to score:
{actual}

Score 0 to 1 on whether the answer conveys the same substance as the reference.
Differences in wording, length, or style do not matter. Missing or contradicting
a fact does.

Reply with JSON only: {{"score": <0-1>, "why": "<one sentence>"}}"""

def _parse_json(reply: str) -> dict:
    """Pull the first {...} block out of a model reply and parse it. {} on failure."""
    try:
        start, end = reply.index("{"), reply.rindex("}") + 1
        return json.loads(reply[start:end])
    except (ValueError, KeyError, TypeError):
        return {}

def judge(actual: str, case: dict) -> float:
    """Score one answer with the judge model. Returns a float in 0..1."""
    reply = cached_call(
        JUDGE_MODEL,
        JUDGE_PROMPT.format(
            input=case["input"], expected=case["output"], actual=actual
        ),
    )
    parsed = _parse_json(reply)
    try:
        return float(parsed["score"])
    except (KeyError, TypeError, ValueError):
        return 0.0     # an unparseable judge is a failed judgement, not a pass

print("Judge ready. It will not run until a cell calls it (needs an API key).")
''')

md(r'''
> **This cell needs your API key.** If you have not set one, skip the next cell; every
> offline section still works. The result is cached, so you pay for each unique
> (question, answer) pair exactly once.

The cell below gives the judge two answers to the same question from our golden set. One
answer is correct but worded differently; the other quietly contradicts a fact. A good
judge should score the first one high and the second one low. Passing this easy test has
a name, **separation**: can the judge tell an obviously good answer from an obviously bad
one? But passing it is only the *low* bar. The harder question, whether the judge scores
things the way a real human would, is called **agreement**, and §8 measures it properly.
Keep those two words in mind; the whole next two sections are about the difference
between them.
''')

code(r'''
_case = GOLDEN[0]   # "What is the refund window for online orders?" -> "30 days from delivery."

_paraphrase = "You can return online orders up to thirty days after they arrive."
_contradiction = "Online orders can be returned within 7 days of delivery."

try:
    s_ok = judge(_paraphrase, _case)
    s_bad = judge(_contradiction, _case)
    print(f"paraphrase (should be high): {s_ok:.2f}")
    print(f"contradiction (should be low): {s_bad:.2f}")
except Exception as e:
    print("Skipped (no API key or network). That is fine, the offline sections do not need it.")
    print("Error was:", type(e).__name__, e)
''')

# ============================================================================
# SECTION 8: CALIBRATE
# ============================================================================
md(r'''
---
## §8. Calibrate the judge against human judgement

"Calibrate" is a fancy word for a simple idea: check the judge's grades against grades we
know are trustworthy, and see how well they match. This is the single step that turns an
eval you can defend into one you cannot. It also explains why so many teams skip it:
**calibrating needs a batch of human-made grades to compare against, and most teams never
collected any.**

We ship you some so you can do it here. The `data/` folder holds 480 computer-written
summaries of 30 real news articles. Each summary was rated 1 to 5 by **three human
experts** on four qualities, from a public research dataset called
[SummEval](https://github.com/Yale-LILY/SummEval) (MIT licensed). Before we let any AI
judge loose on this data, we should first look at what the *humans* did, because that
tells us how hard the job even is.

![Human agreement, bracketed, against a chance floor](assets/07-agreement-interval.svg)

Here is a wrinkle we have to work around. The dataset only saved the **average** of the
three experts' ratings, not the three separate numbers. And an average loses information.
An average of `1.0` could only have come from three 1's. But an average of `2.0` might be
`{2,2,2}`, or `{1,2,3}`, or `{1,1,4}`, and there is no way to tell which. So we cannot
measure exactly how often the three experts agreed. What we *can* do is squeeze the true
answer between a floor and a ceiling (mathematicians call this **bracketing**):

- **At least** this often they were unanimous: how often the average is exactly 1.0 or
  5.0, since only three identical top or bottom scores can produce those.
- **At most** this often: how often the average is any whole number, which has to be true
  when they agree but can also happen by luck.

Then we compare both to a **chance floor**: if three people just guessed at random, their
average still lands on a whole number about a third of the time. Any real agreement has to
clear that floor to mean anything, and keeping that comparison in view is the main habit
this section is trying to build.
''')

code(r'''
DIMENSIONS = ("consistency", "fluency", "relevance", "coherence")

def bounds(dim: str) -> tuple[float, float]:
    """Bracket how often the three annotators agreed on this dimension."""
    vals = [r[dim] for r in JUDGED]
    at_least = sum(v in (1.0, 5.0) for v in vals) / len(vals)          # provably unanimous
    at_most = sum(abs(v - round(v)) < 1e-6 for v in vals) / len(vals)  # whole mean (upper bound)
    return at_least, at_most

# The chance floor: over all 5*5*5 = 125 equally likely triples, how often is the sum
# divisible by 3 (equivalently, the mean a whole number)?
_triples = [(a, b, c) for a in range(1, 6) for b in range(1, 6) for c in range(1, 6)]
CHANCE_FLOOR = sum((a + b + c) % 3 == 0 for a, b, c in _triples) / len(_triples)

print(f"chance floor (random raters give a whole mean): {CHANCE_FLOOR:.0%}\n")
print(f"{'dimension':>12}  {'agreed >=':>9}  {'at most':>8}  {'vs chance':>9}  {'mean':>5}")
for d in DIMENSIONS:
    lo, hi = bounds(d)
    mean = sum(r[d] for r in JUDGED) / len(JUDGED)
    print(f"{d:>12}  {lo:>9.0%}  {hi:>8.0%}  {hi / CHANCE_FLOOR:>8.1f}x  {mean:>5.2f}")
''')

md(r'''
**Two things in that table, and the second one is the lesson.**

First the finding. Three trained annotators, same summary, same rubric, agreed
unanimously about **consistency** at least four times in five. That is close to a fact,
because either the summary says something the article does not, or it does not. On
**coherence** they agreed at most a third of the time, and possibly far less. That is
closer to an opinion, and reasonable people hold different ones.

That ordering is not an assumption. Consistency's *lower* bound (about 80%) sits above
coherence's *upper* bound (about 32%), so the two intervals cannot overlap however the
hidden ratings actually fell. **You can prove the ordering without knowing either number.**

Now the second thing. Look at the `vs chance` column. For coherence and relevance the
upper bound is barely above what pure noise would score. A metric sitting on its own
chance floor is not a weak signal. It is not a signal.

> **The first question to ask about any metric: what would it read if the thing it
> measures were absent?** If you cannot answer that, you cannot interpret a single value
> of it. Two of these four dimensions land on the floor, and you would never see that
> from the number alone. (See THEORY section 8.)
''')

# ============================================================================
# SECTION 9: SEPARATION VS AGREEMENT
# ============================================================================
md(r'''
---
## §9. Separation versus agreement

Now for the real test. We let our judge grade some summaries and compare its grades to the
humans' grades, side by side. We only judge the **consistency** quality, because §8 just
showed us that is the one the experts actually agreed on, so it is the only one worth
judging. (Grading a quality the humans themselves cannot agree on would just be measuring
noise.)

![Separation and agreement are two questions](assets/06-judge-calibration.svg)

One thing about how we pick which summaries to test. We do *not* pick at random. About
93% of the summaries in this data are good, so a random handful would be almost all good
ones, and that would teach us nothing about whether the judge can catch a *bad* summary,
which is the whole point of a judge. Instead we deliberately grab half from the known-bad
pile (human score 2 or lower) and half from the known-good pile (4.5 or higher). Picking
on purpose like this is called **stratified sampling**, and it is the same lesson we hit
again in §10: to test whether something can catch a problem, you have to feed it enough
real problems.

> **This section needs your API key** (one AI call per summary, all cached so you pay
> once). If you do not have a key, skip the run cell. The idea above stands on its own,
> and the shipped cache will still show you real numbers.
''')

code(r'''
CONSISTENCY_PROMPT = """Does this summary state anything the article does not support?

ARTICLE:
{article}

SUMMARY:
{summary}

Score 1 to 5, where 5 means every claim is supported by the article and 1 means it
contains clear fabrications. Judge only factual support, not style, not completeness,
not writing quality.

Reply with JSON only: {{"score": <1-5>, "why": "<one sentence>"}}"""

def judge_consistency(article: str, summary: str) -> float:
    """Ask the judge model for a 1-to-5 consistency score. NaN if unparseable."""
    reply = cached_call(
        JUDGE_MODEL,
        CONSISTENCY_PROMPT.format(article=article[:3000], summary=summary),
    )
    parsed = _parse_json(reply)
    try:
        return float(parsed["score"])
    except (KeyError, TypeError, ValueError):
        return float("nan")

# Build the stratified sample now (this needs no API key: it just picks rows).
SAMPLE_N = 24
_rng = np.random.default_rng(0)
_bad = [r for r in JUDGED if r["consistency"] <= 2]
_good = [r for r in JUDGED if r["consistency"] >= 4.5]
_picks = (
    [_bad[i] for i in _rng.choice(len(_bad), SAMPLE_N // 2, replace=False)]
    + [_good[i] for i in _rng.choice(len(_good), SAMPLE_N // 2, replace=False)]
)
print(f"Sampled {len(_picks)} summaries: {SAMPLE_N // 2} known-bad, {SAMPLE_N // 2} known-good.")
print("Run the next cell to judge them (needs an API key).")
''')

code(r'''
try:
    CALIBRATION = []
    for r in _picks:
        score = judge_consistency(ARTICLES[r["article_id"]], r["summary"])
        CALIBRATION.append({"human": r["consistency"], "judge": score})

    # Keep only pairs the judge could score (drop NaN), then measure two things.
    pairs = [(c["human"], c["judge"]) for c in CALIBRATION if c["judge"] == c["judge"]]
    h = np.array([p[0] for p in pairs])
    j = np.array([p[1] for p in pairs])

    corr = float(np.corrcoef(h, j)[0, 1]) if len(pairs) > 2 else float("nan")
    within1 = float(np.mean(np.abs(h - j) <= 1))
    bad_avg = np.mean([c["judge"] for c in CALIBRATION if c["human"] <= 2])
    good_avg = np.mean([c["judge"] for c in CALIBRATION if c["human"] >= 4.5])

    print("AGREEMENT (does it score the way a human would?)")
    print(f"  correlation with the human mean : {corr:.2f}")
    print(f"  within 1 point of the humans    : {within1:.0%}")
    print("\nSEPARATION (can it tell bad from good at all?)")
    print(f"  average score on known-bad  (human <=2)  : {bad_avg:.2f}")
    print(f"  average score on known-good (human >=4.5): {good_avg:.2f}")
    print(f"  gap: {good_avg - bad_avg:.2f}  (bigger is better)")
except Exception as e:
    print("Skipped (no API key or network).")
    print("Error was:", type(e).__name__, e)
''')

md(r'''
Look at those two results, because they answer two very different questions.

**Separation** is the gap between the judge's average score on the bad summaries and its
average on the good ones. It answers the easy question: *can this judge tell bad from
good at all?* If those two averages are close together, the judge is basically blind to
the thing you care about, and nothing built on top of it is worth reading.

**Agreement** is the other pair of numbers (the correlation, and how often the judge
lands within one point of the humans). It answers the hard question: *does the judge grade
the way a person would?* This is the bar that actually matters, because the score you end
up reporting to other people rises and falls with it. And here is the catch that trips
everyone up: a judge can *pass* separation and still *fail* agreement. That is not rare,
it is the usual case. It is exactly why "I tried it on one good answer and one bad answer,
looks great" is not calibration.

The cached run above (a small, cheap judge model, on 24 stratified summaries) shows
exactly this split. **Separation is clear**: the judge averages about 1.7 on the
known-bad summaries and about 3.0 on the known-good, a gap of roughly 1.3 points, so it
is plainly tracking something real. **Agreement is only fair**: it lands within one
point of the human mean about three quarters of the time and correlates around 0.6.
That is a judge you could use to *flag* likely-bad summaries for a human to review, and
one you should *not* use to publish a precise consistency score. Knowing which of those
two claims your judge can support is the entire deliverable of calibration. (A stronger
judge model, or the rubric fixes below, would push agreement up; this one is left
deliberately modest so the gap between the two bars is visible.)

### What to do when calibration fails

It usually does the first time. In order of what actually works:

1. **Narrow the question.** "Is this good?" cannot be scored. "Does this state anything
   the article does not support?" can, which is why the prompt asks only about factual
   support and explicitly excludes style.
2. **Give the rubric examples** of a 1, a 3, and a 5.
3. **Ask for a decision, not a score.** Binary pass/fail is far more reliable than a
   float, and you can average the decisions.
4. **Use a different model for judging than for answering.** A model grading its own
   output is generous about its own habits. (This is why `JUDGE_MODEL` and `SYSTEM_MODEL`
   are separate settings.)
5. **Check whether the dimension is judgeable at all.** If humans agree no more often
   than chance, stop. You are not calibrating, you are fitting noise.

> **You cannot hold a judge to a standard humans do not meet.** If experts agree with
> each other at most a third of the time on a dimension, a judge that agrees with you a
> third of the time is performing at human level. Chasing it higher is chasing your own
> preferences, not correctness.

> **In the enterprise:** a calibrated judge is calibrated against *one model version*.
> The day the vendor upgrades the model behind your endpoint, every number the judge
> produces changes and nothing in your dashboard says why. Pin the judge model, and store
> its calibration result next to the pin, so a model change re-runs calibration before it
> re-runs anything else.
''')

# ============================================================================
# SECTION 10: SYNTHETIC FAILURES
# ============================================================================
md(r'''
---
## §10. Generate the failures you need

Time for a problem that sounds backwards but bites everyone. To test whether your system
catches mistakes, you need examples of those mistakes. But a good system produces very
few mistakes, so your data barely contains any. You are trying to build a smoke detector
in a house that almost never catches fire. Let us look at the dataset one more time and
count exactly how bad this shortage is.

![Make the failures your data barely contains](assets/08-synthetic-failures.svg)
''')

code(r'''
print(f"{'dimension':>12}  {'bad (<=2)':>9}  {'good (>=4.5)':>12}  {'bad in a random 20':>18}")
for d in DIMENSIONS:
    bad = sum(1 for r in JUDGED if r[d] <= 2) / len(JUDGED)
    good = sum(1 for r in JUDGED if r[d] >= 4.5) / len(JUDGED)
    print(f"{d:>12}  {bad:>9.1%}  {good:>12.1%}  {20 * bad:>18.1f}")
''')

md(r'''
**Read that table, because the problem is bigger than it looks.**

Only about one summary in fifteen is factually broken. So if you grab twenty summaries at
random to test whether your judge catches made-up facts, you get **roughly one** broken
example, and you simply cannot judge a detector from one example. To gather fifty broken
ones the honest way, you would have to read and hand-label around seven hundred summaries.
And remember, this dataset is *unusually* full of failures on purpose (the researchers
deliberately included output from weak models). Your own product's logs will be even
cleaner, because the entire point of your product is that it usually works.

> **The very failures you most want to catch are, by their nature, the ones your real
> data has the fewest of.** A test set that is 93% "everything went fine" simply cannot
> tell you whether your detector works. And collecting more real data does not save you:
> it just keeps the same lopsided ratio.

So instead of hunting for rare failures, we *manufacture* them. And this is not a
compromise, it is actually better, because of one lovely property:

> **A failure you created yourself comes with its answer key already attached.** You know
> it is broken, you know exactly which fact you broke, and you know precisely what a
> working detector should say about it. Real failures have to be hunted down and then
> labeled by a human. Manufactured ("synthetic") failures arrive pre-labeled, for free.
> (See THEORY section 10.)

Below we write four little "corruptions." Each one takes a good summary and breaks it in a
specific, realistic way, mirroring a real mistake AI systems actually make.
''')

code(r'''
# Names let us make a corruption that reads perfectly and is still wrong. The summaries
# are lowercased model output, but the source articles keep their capitalization, so we
# pull proper names from the articles and match case-insensitively.
_NAME = re.compile(r"\b[A-Z][a-z]{2,}(?: [A-Z][a-z]{2,})+\b")

def names_in(text: str) -> set[str]:
    return {m.group(0) for m in _NAME.finditer(text)}

ALL_NAMES = sorted({n for t in ARTICLES.values() for n in names_in(t)})

# Plausible sentences that are simply not in any article.
UNSUPPORTED = (
    " the ruling is expected to be appealed within the month .",
    " no criminal charges have been filed in connection with the case .",
    " a spokesperson declined to comment when contacted on tuesday .",
    " the figure represents a sharp increase on the previous year .",
)

def corrupt(summary: str, article: str, kind: str, rng) -> str | None:
    """Break a summary in one known way. Returns None if the corruption does not apply."""
    if kind == "number":
        # A changed quantity. Common, and invisible unless you check the source.
        nums = re.findall(r"\b\d[\d,\.]*\b", summary)
        if not nums:
            return None
        target = nums[int(rng.integers(len(nums)))]
        changed = str(int(rng.integers(2, 99))) if "." not in target else "9.9"
        return summary.replace(target, changed, 1)

    if kind == "negate":
        # A reversed claim. Reads perfectly and means the opposite.
        for a, b in ((" was ", " was not "), (" is ", " is not "),
                     (" will ", " will not "), (" had ", " had not ")):
            if a in summary:
                return summary.replace(a, b, 1)
        return None

    if kind == "unsupported":
        # A plausible sentence appended that the article never says.
        return summary.rstrip() + UNSUPPORTED[int(rng.integers(len(UNSUPPORTED)))]

    if kind == "entity":
        # A real name swapped for another real name from a different article.
        # Grammatical, fluent, and false: the hallucination that reaches production.
        local = sorted(names_in(article))
        present = [n for n in local if n.lower() in summary.lower()]
        if not present:
            return None
        old = present[int(rng.integers(len(present)))]
        pool = [n for n in ALL_NAMES if n not in local]
        if not pool:
            return None
        new = pool[int(rng.integers(len(pool)))]
        out = re.sub(re.escape(old.lower()), new.lower(), summary, count=1, flags=re.IGNORECASE)
        return out if out != summary else None

    raise ValueError(kind)

CORRUPTIONS = ("number", "negate", "unsupported", "entity")

# Start from known-good summaries only, so any inconsistency is one WE introduced.
_rng = np.random.default_rng(7)
_clean = [r for r in JUDGED if r["consistency"] >= 4.5]

SYNTHETIC = []
for r in _clean:
    for kind in CORRUPTIONS:
        broken = corrupt(r["summary"], ARTICLES[r["article_id"]], kind, _rng)
        if broken and broken != r["summary"]:
            SYNTHETIC.append({
                "article_id": r["article_id"],
                "summary": broken,
                "label": "inconsistent",   # free, by construction
                "failure": kind,
                "original": r["summary"],
            })

print(f"{len(SYNTHETIC)} labeled bad summaries, from {len(_clean)} known-good ones.")
print(f"(natural bad examples in the whole dataset: {sum(1 for r in JUDGED if r['consistency'] <= 2)})\n")
for k in CORRUPTIONS:
    print(f"  {k:12} {sum(1 for s in SYNTHETIC if s['failure'] == k):>4}")
''')

code(r'''
def diff_window(before: str, after: str, pad: int = 34) -> tuple[str, str]:
    """Show the two strings around their first point of difference.

    The corruption is often deep inside a long summary, so printing the first 90
    characters would show two identical prefixes. This finds where they diverge and
    prints a window around it, so the change is always visible.
    """
    i = next((k for k in range(min(len(before), len(after))) if before[k] != after[k]),
             min(len(before), len(after)))
    lo = max(0, i - pad)
    b = ("..." if lo else "") + before[lo:i + pad] + ("..." if i + pad < len(before) else "")
    a = ("..." if lo else "") + after[lo:i + pad] + ("..." if i + pad < len(after) else "")
    return b, a

# One example of EACH failure mode, showing the window around the change.
seen = set()
for s in SYNTHETIC:
    if s["failure"] in seen:
        continue
    seen.add(s["failure"])
    b, a = diff_window(s["original"], s["summary"])
    print(f"[{s['failure']}]")
    print(f"  before: {b}")
    print(f"  after : {a}")
    print()
    if len(seen) == len(CORRUPTIONS):
        break
''')

md(r'''
From a few hundred good summaries you now have far more labeled failures than the
dataset contained naturally, and each one carries **which** failure mode it is. A
detector that catches number changes but misses negations then shows up as a pattern,
not just a lower average. That per-failure breakdown is the thing natural data almost
never gives you.

### Where this stops working, and it does

| The catch | Why it matters |
| --- | --- |
| You test only the failures you thought of | The dangerous ones are the ones you did not imagine |
| Corruptions can be *too* easy | Swapping a name for a real name is fair. Swapping it for nonsense tests your fluency checker, not your fact checker |
| The distribution is now wrong | A 50/50 test set does not tell you the production false-positive rate |

**Use both, for different questions.** Synthetic failures answer *"can the detector
detect?"*: a balanced set, per-failure-mode, cheap and fast. Natural data answers *"what
will this cost me in production?"*: the real base rate, and therefore the real
false-positive burden. Report the second to stakeholders; use the first to improve the
system.
''')

# ============================================================================
# SECTION 11: CONFIDENCE INTERVALS
# ============================================================================
md(r'''
---
## §11. Is the difference real?

Sooner or later you will compare two versions of your system: an old prompt against a new
one, or the model you started with against a fine-tuned one. Someone will announce,
*"it went from 0.83 to 0.88, a five-point improvement!"* Here is the uncomfortable truth:
if your test set is small, that sentence usually means nothing at all. The five points
could easily be luck. This section shows you how to tell a real improvement from random
noise.

![The same true improvement, invisible at small n](assets/09-confidence-interval.svg)

The tool for this is a **confidence interval**, and the question it answers is simple: if
I had happened to test on a slightly different set of examples, how much would my score
have wobbled? A small wobble means the number is solid. A big wobble means you are mostly
measuring luck.

We compute it with a clever, assumption-free trick called the **bootstrap**. The idea:
take the results you already have, and build thousands of pretend "new" test sets by
drawing examples from your real set at random (allowing repeats). Score each pretend set,
and watch how much the scores spread out. That spread *is* the wobble. The beauty is that
it makes no assumption about the shape of your data, which matters because eval scores are
rarely a tidy bell curve.
''')

code(r'''
def bootstrap_ci(scores, n_resamples=5000, level=0.95, seed=0):
    """Confidence interval for a mean, by resampling with replacement."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores, dtype=float)
    # Draw n_resamples fresh datasets, each the same size, sampling rows with
    # replacement; take each one's mean. The spread of those means is the CI.
    means = arr[rng.integers(0, len(arr), size=(n_resamples, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(arr.mean()), float(lo), float(hi)

def difference_ci(a, b, n_resamples=5000, seed=0):
    """CI for the difference between two systems measured on the SAME cases.

    Paired on purpose: we resample the cases, then take both systems' means on the
    same resampled cases. Ignoring the pairing throws away the fact that both systems
    found the same cases hard, and widens the interval for no reason.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    idx = rng.integers(0, len(a), size=(n_resamples, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)

print("bootstrap_ci and difference_ci ready.")
''')

code(r'''
# Two systems that genuinely differ by five points (0.83 vs 0.88 pass rate), measured
# on eval sets of six different sizes. The TRUE difference is identical every time;
# only n changes. Watch the interval on the difference.
_rng = np.random.default_rng(3)

print(f"{'n cases':>8}  {'new mean':>9}  {'95% CI (new)':>18}  {'diff':>7}  {'CI on diff':>18}  real?")
for n in (5, 10, 20, 50, 200, 1000):
    old = (_rng.random(n) < 0.83).astype(float)
    new = (_rng.random(n) < 0.88).astype(float)
    m, lo, hi = bootstrap_ci(new)
    d, dlo, dhi = difference_ci(new, old)
    verdict = "no (spans 0)" if dlo <= 0 <= dhi else "yes"
    print(f"{n:>8}  {m:>9.2f}  [{lo:>5.2f}, {hi:>5.2f}]  {d:>+7.3f}  [{dlo:>+5.2f}, {dhi:>+5.2f}]  {verdict}")
''')

md(r'''
**The improvement is real in every row. You cannot see it in most of them.**

With five or ten cases the interval on the difference is enormous and includes zero, so
the honest report is *"we cannot tell yet."* That does not become *"no improvement"*
(absence of evidence is the whole problem), but it does mean you have not earned the
claim. Only around a few hundred cases does the interval clear zero.

Two consequences worth carrying:

**Five golden examples are enough to catch a broken system and not enough to compare two
working ones.** Both are true, and people use the first to justify the second. If a later
change has to *beat* a number, that number needs enough cases to be beatable in a
detectable way.

**Report the interval, not just the mean.** *"0.88, 95% CI [0.79, 0.95]"* is a sentence
an executive can act on and an engineer can check. *"0.88"* invites a comparison against
*"0.83"* that the data does not support.

> **A rough rule of thumb, not a law:** to reliably spot a 5-point difference you usually
> need a few hundred cases; a big 20-point difference shows up with just a few dozen. If
> you find yourself chasing tiny one-point changes on a small set, you are almost
> certainly measuring noise, not progress. The smallest gap your test set is actually big
> enough to notice has a name, the **minimum detectable effect** (or MDE), and it is a
> useful thing to know about your own eval before you start comparing.
''')

# ============================================================================
# SECTION 12: MVP TO POC TO PRODUCTION
# ============================================================================
md(r'''
---
## §12. MVP to PoC to production

Projects grow up in stages, and the *same* harness you just built means something
different at each one. The three stages have names you will hear constantly at work:
**MVP** (a first rough version, "minimum viable product"), **PoC** ("proof of concept,"
the version you show to convince someone to fund it), and **production** (the real thing,
serving real users).

![One harness, three stages, three questions](assets/10-mvp-poc-prod.svg)

| Stage | The question it answers | What you test it against |
| --- | --- | --- |
| **MVP** | Does it work at all? | Your golden set. A pass rate and your own eyeballs. |
| **PoC** | Is it good enough to fund? | Back-tested history, a calibrated judge, and intervals. |
| **Production** | Is it *still* working? | Live traffic, sampled and scored all the time. |

The jump that really matters is **MVP to PoC**, and here is the surprise: it is not a
technical step at all. It is the moment you stop saying "trust me, it looks good" and
start showing a real number, measured against data you did not get to hand-pick. That is
the first moment anyone outside your own team has a genuine reason to believe you.

> If you take one thing from this notebook: **the number has to come from data you did
> not pick, and it has to come with an interval.**
''')

# ============================================================================
# SECTION 13: FRAMEWORKS
# ============================================================================
md(r'''
---
## §13. Where a framework fits

You just built the entire thing by hand: a dataset, scorers, a judge, calibration, and
confidence intervals. That was on purpose. The pieces are small, and since you will be
the one defending the numbers they produce, it really helps to know exactly how they work.

Now, you may be wondering: was that all necessary? Are there not ready-made tools for
this? Yes, there are, and the most popular is a library called
[**Ragas**](https://docs.ragas.io/) (Apache 2.0). It packages up these same ideas. Once
you have built your own harness, it is worth knowing what a framework like Ragas adds on
top:

| Ragas gives you | Which otherwise you would write |
| --- | --- |
| Retrieval metrics | context precision and recall, answer relevancy, faithfulness |
| Synthetic test-set generation | the §10 idea, with more machinery |
| Integrations | LangChain, LlamaIndex, common observability platforms |

What it does *not* do is decide whether to believe its numbers. Its metrics are mostly
LLM-judged, which means **everything in §8 and §9 still applies**. A framework's judge is
exactly as uncalibrated as yours was before you checked it, and it is easier to skip the
check when the number arrives looking official.

Reasonable position: build your own for the handful of criteria that decide whether your
product works, and reach for Ragas when you want the standard retrieval metrics without
writing them. Do not adopt it to avoid understanding what it measures.

We do not install Ragas in this module (it pulls a large dependency tree). The concept to
carry is the one above: a framework metric is still an LLM judge, and a version number is
not a calibration.
''')

# ============================================================================
# SECTION 14: AGENTIC EVALS
# ============================================================================
md(r'''
---
## §14. Agentic evals: the answer is not the whole story

Everything up to now has graded a single answer. But an **agent** does far more than
answer a question. It decides which tools to use, takes several steps in a row, reads what
comes back, recovers when a step goes wrong, and (ideally) knows when to refuse a task it
should not attempt. Here is the twist that changes everything about grading: two agents
can arrive at the *exact same* correct final sentence while taking wildly different, and
very differently dangerous, routes to get there. A grade that only looks at the final
answer is completely blind to that.

![Agent evals: same answer, different paths](assets/11-answer-vs-trajectory.svg)

The full record of what an agent did along the way (every tool it called, every result it
got back, and the final answer) is called its **trajectory** (just a fancy word for "the
path it took"). Agentic evals grade that whole path, not only the destination, along four
things a single answer score can never see:

- **Tool choice.** Did it reach for the right tool, with the right inputs?
- **Task success.** Did the world actually end up in the correct state? (This is the
  outcome check, the closest thing to what we graded before.)
- **Efficiency and cost.** How many steps, how much time, how much money did it burn?
- **Safety and bounds.** Did it stay inside the fence, for example never opening data it
  had no business reading?

To make this concrete, below is a tiny pretend agent that runs completely offline. It has
three tools and quietly writes down every call it makes. Then we grade two things
*separately*: whether the final answer was right (the outcome), and whether the path it
took was clean (the trajectory).
''')

code(r'''
# A trace records everything the agent did. Each tool logs its own call here.
class Trace:
    def __init__(self):
        self.calls = []          # list of (tool_name, args, result)

    def log(self, name, args, result):
        self.calls.append({"tool": name, "args": args, "result": result})

    @property
    def tool_names(self):
        return [c["tool"] for c in self.calls]

# A tiny fake knowledge base and its tools. Every tool takes the trace so the call is
# recorded. read_customer_pii is the "tool it should not need" for this task.
POLICY = {"refund_window_days": 30, "gift_cards_refundable": False}

def make_tools(trace: Trace):
    def search_policy(query):
        result = f"policy fields: {list(POLICY)}"
        trace.log("search_policy", {"query": query}, result)
        return result
    def get_field(name):
        result = POLICY.get(name, "unknown")
        trace.log("get_field", {"name": name}, result)
        return result
    def read_customer_pii(customer_id):
        result = "SSN 000-00-0000, card 4111-1111-1111-1111"
        trace.log("read_customer_pii", {"customer_id": customer_id}, result)
        return result
    return {"search_policy": search_policy, "get_field": get_field,
            "read_customer_pii": read_customer_pii}

# Agent A: clean path. Searches, reads the one field it needs, answers.
def agent_a(question):
    trace = Trace()
    tools = make_tools(trace)
    tools["search_policy"](question)
    days = tools["get_field"]("refund_window_days")
    return f"The refund window is {days} days.", trace

# Agent B: reaches the SAME answer, but reads PII it never needed and searches wastefully.
def agent_b(question):
    trace = Trace()
    tools = make_tools(trace)
    tools["read_customer_pii"]("cust_42")     # unnecessary and unsafe
    tools["search_policy"](question)
    tools["search_policy"](question)          # wasteful repeat
    days = tools["get_field"]("refund_window_days")
    return f"The refund window is {days} days.", trace

ans_a, trace_a = agent_a("refund window?")
ans_b, trace_b = agent_b("refund window?")
print("Agent A answer:", ans_a, "| tools:", trace_a.tool_names)
print("Agent B answer:", ans_b, "| tools:", trace_b.tool_names)
''')

code(r'''
# Now score both. The OUTCOME scorer looks only at the final answer.
def outcome_correct(answer, expected="30 days"):
    return float(expected in answer)

# The TRAJECTORY scorers look at the path. These are just plain code over the trace,
# which is the point: most agentic checks are assertions about what happened, not a judge.
def used_forbidden_tool(trace, forbidden="read_customer_pii"):
    return float(forbidden in trace.tool_names)     # 1.0 means it did (bad)

def efficiency(trace, budget=3):
    # Fraction of the step budget left. 1.0 = used no more than budget; lower = wasteful.
    used = len(trace.calls)
    return max(0.0, min(1.0, (budget - used) / budget + 1.0)) if used <= budget else max(0.0, 1.0 - (used - budget) / budget)

for name, ans, trace in [("Agent A", ans_a, trace_a), ("Agent B", ans_b, trace_b)]:
    print(f"{name}")
    print(f"  outcome correct     : {outcome_correct(ans):.2f}   (final answer right?)")
    print(f"  used forbidden tool : {used_forbidden_tool(trace):.2f}   (1.0 = leaked PII, bad)")
    print(f"  efficiency          : {efficiency(trace):.2f}   (1.0 = within step budget)")
    print()
''')

md(r'''
Both agents score **1.0 on the outcome**: the final answer is identical and correct. An
answer-only eval would call them equal and move on. The trajectory scorers tell the real
story: Agent B leaked customer PII and made two wasteful calls. That is the whole case for
grading the path, not just the destination.

### Reliability: one good demo lies

There is one last thing to grade that only agents force you to think about:
**reliability.** An agent that nails a task once might flub the very same task the next
time, because the model does not answer identically every run. So a single impressive demo
can genuinely lie to you. The honest test is to run the same task **k** times in a row and
ask: how often does it succeed *every single time?* That measurement is called **pass^k**,
and it falls off a cliff faster than people expect. If one attempt works 90% of the time,
then getting it right three times in a row is only 0.9 x 0.9 x 0.9, which is about 73%.
''')

code(r'''
def pass_hat_k(single_pass_rate, k):
    """Probability of passing ALL k independent attempts, if each passes at this rate."""
    return single_pass_rate ** k

print("If one attempt passes 90% of the time:")
for k in (1, 2, 3, 5, 8):
    print(f"  pass^{k} = {pass_hat_k(0.90, k):.2%}")
print("\nOne good demo is pass^1. Production is pass^k for the k a real user will hit.")
''')

md(r'''
This is the bridge to a full agentic benchmark. The sibling module
[`18_tau2`](../18_tau2/) builds exactly this idea out: a user simulator that drives a
multi-turn conversation, an outcome verifier separate from a trajectory diagnostic, a
per-capability report, a regression gate, and pass^k over a fixed task suite. Everything
you built here (a dataset, deterministic scorers, a calibrated judge, intervals) is the
foundation those agentic pieces stand on. (See THEORY section 14.)
''')

# ============================================================================
# CONCLUSION
# ============================================================================
md(r'''
---
## Conclusion

Look back at how far you came. You wrote a grading machine in about forty lines, and
then, before trusting a word it said, you tested the machine itself with echo and oracle.
You built an AI judge and, instead of taking its scores on faith, you calibrated it
against real human ratings, and along the way you discovered that on some qualities the
human experts themselves cannot agree any better than random guessing, which is a fact
about the *question*, not a flaw in the judge. When your data did not have enough failures
to test with, you manufactured your own, each one arriving with its answer key attached.
You put an honest interval around a number and watched a genuine five-point improvement
disappear into the noise whenever the test set was small. And finally you carried the
whole idea over to agents, where two identical correct answers can hide the fact that one
of them leaked a secret and wasted two steps getting there.

Here is the sentence to walk away with: the number was never really the point. Knowing
exactly what would make it wrong is.

| Component | What we built | What production looks like |
| --- | --- | --- |
| Cases | A handful of golden examples | Thousands, sampled from live traffic, reviewed weekly |
| Scorers | A dict of four functions | A versioned registry, each scorer with its own tests |
| Judge | One rubric, calibrated once | Multiple judges, model pinned, re-calibrated on model change |
| Failures | Corruptions written by hand | A generator tuned so synthetic failures are as hard as real ones |
| Intervals | A bootstrap on the whole set | Intervals per slice, so you see which segment moved |
| Agents | A mock trajectory, scored by hand | A user simulator, outcome plus trajectory, pass^k over a suite |
| Cadence | Run when you remember | Model-free scorers on every commit; the judge nightly, on a sample |

The last row decides whether the harness is still running in six months.

### References

- [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/), Hamel Husain. The
  most useful practical piece on this topic; if you read one thing, this one.
- [Judging LLM-as-a-Judge (MT-Bench)](https://arxiv.org/abs/2306.05685), Zheng et al.,
  2023. Established the technique and documented its biases (position, verbosity,
  self-preference). §7 and §9 exist because of this paper.
- [Eugene Yan, LLM Evaluators](https://eugeneyan.com/writing/llm-evaluators/). A thorough
  survey of what works, with attention to the failure cases.
- [AI Agents That Matter](https://arxiv.org/abs/2407.01502), Kapoor et al., 2024. How
  benchmark numbers mislead when cost is not held constant. Directly relevant to §14.
- [SummEval](https://github.com/Yale-LILY/SummEval), Fabbri et al., 2021. The 480 rated
  summaries in `data/`, and the paper behind the agreement numbers in §8.
- [tau2-bench](https://arxiv.org/abs/2506.07982), the agentic benchmark the sibling module
  `18_tau2` builds toward.
- Companion: [`EVALS_THEORY.md`](EVALS_THEORY.md) for the *why* behind each step, with a
  caution and a counter-example on every concept.

### Keep building

1. **Count your own failures.** What fraction of your logged outputs are actually wrong?
   Under 10% and a random sample will not measure your detector; you need §10.
2. **Calibrate on your own data.** Hand-score twenty of your outputs, run the §7 judge
   over them, and compute the agreement and the chance floor for *your* task. That is the
   calibration half this notebook could not do for you.
3. **Put an interval on your headline number** before anyone quotes it back to you with a
   decimal point of confidence it never had.

*© mui-group*
''')

# ============================================================================
# EMIT THE NOTEBOOK
# ============================================================================
def to_cell(kind, source, index):
    lines = source.splitlines(keepends=True)
    base = {"id": f"{kind}-{index:03d}", "metadata": {}, "source": lines}
    if kind == "md":
        return {"cell_type": "markdown", **base}
    return {"cell_type": "code", "execution_count": None, "outputs": [], **base}


notebook = {
    "cells": [to_cell(kind, src, index) for index, (kind, src) in enumerate(CELLS)],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = "evals_tutorial.ipynb"
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"wrote {OUT} with {len(CELLS)} cells")
