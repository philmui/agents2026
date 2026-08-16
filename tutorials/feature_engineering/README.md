# Topic 19: Feature Stores from First Principles with Chronon

Build point-in-time-correct customer features from a real retail transaction log,
author genuine Chronon `GroupBy` and `Join` definitions, run an optional local
Chronon Spark backfill, expose leakage with a chronological model test, and add
serving, parity, freshness, ownership, and monitoring contracts.

## Start here

```bash
cd topics/19_feature_store
cp .env.example .env
uv sync
uv run python _build_notebook.py
uv run jupyter lab
```

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
the decision, such as `[t - 30 days, t)`. The target uses the future interval
`(t, t + 30 days]`. Recent rows without a complete target horizon are censored,
not mislabeled as negatives.

## Three execution layers

1. **Chronon authoring and compilation:** real definitions in `chronon/` use
   `chronon-ai==0.0.114`.
2. **Chronon Jupyter backfill:** optional §8 runs a tiny Spark-backed `JupyterJoin`
   with a UV-managed Java 17 runtime and a checksum-pinned Chronon assembly JAR.
3. **Transparent semantic reference:** NumPy computes the full UCI point-in-time
   training table so every boundary is easy to inspect. It is not described as the
   Chronon production engine.

Enable the optional real Chronon backfill in `.env`:

```dotenv
RUN_CHRONON_SPARK=1
```

The first enabled run downloads a 25 MB assembly JAR from Maven Central and checks
its SHA-256 digest. Version 0.0.114 needs a documented, narrowly scoped
`TableUtils` namespace compatibility shim. The notebook asserts expected cutoff
values after the job completes.

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

Compiled files land in ignored `chronon/production/`. Production execution still
requires warehouse, stream, scheduler, online key-value store, Chronon online API,
and service integrations.

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

To validate the optional real Chronon path without changing `.env`:

```bash
RUN_CHRONON_SPARK=1 uv run jupyter nbconvert \
  --execute --to notebook feature_store_tutorial.ipynb \
  --output /tmp/feature_store_chronon_executed.ipynb \
  --ExecutePreprocessor.timeout=900
```

## Core references

- [Chronon documentation](https://chronon.ai/contents.html)
- [Chronon GitHub repository](https://github.com/airbnb/chronon)
- [UCI Online Retail](https://doi.org/10.24432/C5BW33)

© mui-group
