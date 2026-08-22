# Feature Stores from First Principles with Chronon

A feature is a number a model uses, such as how much a customer spent in the last
30 days. The right answer depends on when you ask: include purchases made after the
prediction moment and your model looks great in testing, then fails in production.
Avoiding that is called point-in-time correctness, and it is what this project
teaches.

Over the tutorial you will:

- build customer features from a real retail transaction log that only ever look at
  the past;
- write genuine Chronon `GroupBy` and `Join` definitions;
- optionally run a small local Chronon Spark backfill, which recomputes historical
  feature values from the source history;
- catch leakage by training on a time-ordered split, then comparing an honest model
  against one that is allowed to peek at the future;
- add the contracts a real deployment needs, covering serving, offline and online
  agreement, freshness, ownership, and monitoring.

## Start here

```bash
cd feature_engineering
cp .env.example .env
uv sync
uv run python _build_notebook.py
uv run jupyter lab
```

This project intentionally pins Python 3.10 because `chronon-ai==0.0.114` pins
PySpark 3.3.1. PySpark 3.3.1's bundled serializer is incompatible with Python
3.11 and newer. If Jupyter was already open when you ran `uv sync`, restart it
with the command above so the notebook does not keep using a stale kernel.

The first notebook run downloads the UCI Online Retail CSV to `data/cache/`.
No API key is required. The notebook still calls
`load_dotenv(find_dotenv())`, so any future secrets belong only in the ignored
`.env` file.

## Files

- [`feature_store_tutorial.ipynb`](feature_store_tutorial.ipynb): detailed,
  executable tutorial.
- [`FEATURE_STORE_THEORY.md`](FEATURE_STORE_THEORY.md): illustrated concepts,
  counterexamples, cautions, and notebook cross-references.
- [`_build_notebook.py`](_build_notebook.py): reviewable notebook source.
- [`chronon/`](chronon/): real Chronon source, `GroupBy`, `Join`, and team config.
- [`assets/`](assets/): ten reusable SVG diagrams.
- [`slides/index.html`](slides/index.html): modern Reveal.js teaching deck.
- [`data/SOURCE.md`](data/SOURCE.md): dataset license, attribution, transforms,
  and limitations.
- [`pyproject.toml`](pyproject.toml): UV-managed Python environment.

## The case study

[UCI Online Retail](https://doi.org/10.24432/C5BW33) contains 541,909 real line
items from a UK online retailer. The tutorial asks, at each completed order,
whether that customer will place another completed order within 30 days.

Historical purchase and cancellation features use the half-open interval before
the decision, such as `[t - 30 days, t)`, so the decision moment itself is not
history. A customer who spends £10 on 1 January and £20 on 10 January has £10 of
prior spend at the 10 January order.

The target uses the future interval `(t, t + 30 days]`. Recent rows without a
complete target horizon are censored, not mislabeled as negatives, because their
outcome window has not finished; calling them negatives would teach the model that
recent customers never return.

## Three execution layers

These three layers are not interchangeable.

1. **Chronon authoring and compilation:** real definitions in `chronon/` use
   `chronon-ai==0.0.114`.
2. **Chronon Jupyter backfill:** optional §8 runs a tiny Spark-backed `JupyterJoin`
   with a UV-managed Java 17 runtime and a checksum-pinned Chronon assembly JAR.
   Spark is the engine doing the heavy data work, and it runs on the Java virtual
   machine.
3. **Transparent semantic reference:** NumPy computes the full UCI point-in-time
   training table so every boundary is easy to inspect. It is not described as the
   Chronon production engine.

Layer three exists so you can read every comparison in plain Python; layers one and
two are what a team actually operates. Mistake the NumPy code for the production
engine and you will badly underestimate what production costs.

Enable the optional real Chronon backfill in `.env`:

```dotenv
RUN_CHRONON_SPARK=1
```

The first enabled run downloads a 25 MB assembly JAR from Maven Central and checks
its SHA-256 digest, so a mismatch means stop and investigate. Version 0.0.114 needs
a documented, narrowly scoped `TableUtils` namespace compatibility shim. The
notebook asserts expected cutoff values after the job completes, so a wrong boundary
fails an assertion instead of looking plausible: the fixture checks £30 of prior
purchases and £7 of prior cancellations.

The local lab creates a fresh temporary Hive metastore and warehouse on every run
and removes its tutorial databases afterward. This makes the Spark cell safe to
rerun in one Jupyter kernel; without that isolation, a surviving Py4J Java process
can retain partition metadata that points to temporary files already deleted.

## Compile the Chronon project

```bash
cd chronon
PYTHONPATH=. uv run --project .. compile.py \
  --conf group_bys/retail/purchases.py --force-overwrite -y
PYTHONPATH=. uv run --project .. compile.py \
  --conf group_bys/retail/cancellations.py --force-overwrite -y
PYTHONPATH=. uv run --project .. compile.py \
  --conf joins/retail/repeat_purchase_training.py --force-overwrite -y
```

Compiled files land in ignored `chronon/production/`. Compiling is not running,
though. Actually executing these definitions in production also needs a warehouse to
store the historical tables, a stream to feed new events in as they arrive, a
scheduler, a key-value store that returns one key's value quickly, an implementation
of Chronon's online API, and the integration with your model service. None of that
lives in this repository.

## Validate

```bash
uv sync
uv run python _build_notebook.py
uv run python -m compileall -q _build_notebook.py chronon
uv run jupyter nbconvert --execute --to notebook --inplace \
  feature_store_tutorial.ipynb --ExecutePreprocessor.timeout=600
xmllint --noout assets/*.svg
rg -n -P '\x{2014}' . \
  -g '!data/cache/**' -g '!.venv/**'
```

Each line checks something different: `uv sync` pins versions, the rebuild keeps
notebook and source in step, `compileall` catches syntax errors, `xmllint` checks
each SVG, and `rg` fails if a forbidden em-dash reached a file. The `nbconvert` line
is the strongest test, because it runs every cell in a fresh kernel. A cell that
only worked because of something you typed by hand will fail there.

To validate the optional real Chronon path without changing `.env`:

```bash
RUN_CHRONON_SPARK=1 uv run jupyter nbconvert \
  --execute --to notebook feature_store_tutorial.ipynb \
  --output /tmp/feature_store_chronon_executed.ipynb \
  --ExecutePreprocessor.timeout=900
```

That run writes to `/tmp` instead of editing the notebook, and a cutoff assertion
failing only here means the two execution layers disagree.

## Core references

- [Chronon documentation](https://chronon.ai/contents.html)
- [Chronon GitHub repository](https://github.com/airbnb/chronon)
- [UCI Online Retail](https://doi.org/10.24432/C5BW33)

© mui-group
