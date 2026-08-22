"""Build the feature store tutorial notebook from readable source cells.

Run with:

    uv run python _build_notebook.py

The generated notebook is the learner-facing artifact. Keeping its source here
makes large prose and code edits reviewable, while nbformat guarantees valid
notebook JSON and stable cell identifiers.
"""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "feature_store_tutorial.ipynb"
nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (feature-store-tutorial)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.10"},
}


def add_markdown(cell_id: str, source: str) -> None:
    cell = nbf.v4.new_markdown_cell(dedent(source).strip())
    cell["id"] = cell_id
    nb.cells.append(cell)


def add_code(cell_id: str, source: str) -> None:
    cell = nbf.v4.new_code_cell(dedent(source).strip())
    cell["id"] = cell_id
    nb.cells.append(cell)


add_markdown(
    "title",
    r'''
    # Feature Stores: Giving Models the Right Past with Chronon

    ### A hands-on lesson about features, time, and storage

    ![One clear feature rule supports model training and live decisions](assets/01-feature-store-loop.svg)

    A **feature** is one piece of information you feed a model, like how many
    orders a customer has placed recently. A **feature store** is the shared
    system that calculates those values and hands them to models. Its job sounds
    easy: give a model the **right value**, about the **right thing**, at the
    **right time**.

    That last word, time, is where teams get into trouble. Every prediction happens
    at one specific moment, and the model can only use facts that were already known
    at that moment. But when you train a model on old data, it is surprisingly easy
    to slip in a fact that actually showed up later. The model then looks brilliant
    in testing and falls flat in the real world. This notebook shows you exactly how
    that mistake happens and gives you simple tests that catch it.

    We use [UCI Online Retail](https://doi.org/10.24432/C5BW33), a real log of
    541,909 order lines from a UK online retailer. Each time a customer finishes an
    order, we ask one question:

    > Will this customer place another order within the next 30 days?

    By the end you will be able to:

    1. turn raw order lines into clean purchase and cancellation records,
    2. pick the moment you predict at and the future window the answer looks across,
    3. build features that use only the past, following one exact cutoff rule,
    4. prove that cutoff works using small tests you can check by hand,
    5. write real Chronon `GroupBy` and `Join` definitions,
    6. run a small Chronon Spark job on your own laptop when your setup allows it,
    7. compare an honest model against one that cheats by peeking at the future,
    8. use a simple cache to see how live feature lookups behave,
    9. check that the offline and online values agree, and watch freshness and quality,
    10. describe what a real production Chronon deployment adds on top of this lab.

    Most of the feature math here runs in plain NumPy. Think of that local version
    as a small, readable stand-in: it is short enough that you can inspect every
    cutoff yourself. It is not meant to replace Chronon. It is a way to understand
    what Chronon does for you before you trust a much bigger system to do it.

    **What you need to know already.** Python basics, and enough comfort with tables
    of data to follow a few pandas commands. You do not need to have used Chronon,
    Spark, or a feature store before. Every specialized term gets defined the first
    time it appears, and machine learning ideas are explained as we reach them.

    **How to work through it.** Run the cells in order, top to bottom, because each
    one builds on the tables the earlier ones created. Each section tells you what it
    is about to do, explains the idea behind it, runs the code, and then describes
    what the output should look like. Several sections end in checks written with
    Python's `assert`. An `assert` is a line that stops the notebook with an error
    the instant a claim it was given turns out to be false. So a cell that runs all
    the way to the end is quietly telling you that every claim we made about it held
    up.

    The companion [`FEATURE_STORE_THEORY.md`](FEATURE_STORE_THEORY.md) covers the
    same ideas in more depth, with extra examples, common mistakes, and the same
    diagrams.
    ''',
)

add_markdown(
    "contents",
    r'''
    ## Learning path

    - **§0** How to think about a feature store, plus the vocabulary
    - **§1** Setting up the project with `uv` and `.env`
    - **§2** The decision, the entity, the clocks, and the feature contract
    - **§3** Loading and auditing the real UCI data
    - **§4** Turning order lines into clean event tables
    - **§5** Building past-only features and future labels
    - **§6** Testing the time cutoff, late data, and basic invariants
    - **§7** Writing the same design as Chronon `GroupBy` and `Join` definitions
    - **§8** Running a small real Chronon backfill
    - **§9** Training on a time-ordered split and exposing leakage
    - **§10** Looking up current values and checking that both paths agree
    - **§11** Adding contracts, quality checks, and monitoring
    - **§12** Mapping the lab to production, and knowing when to stop

    You do not need to memorize Chronon commands. Keep coming back to one simple
    question: **could the model really have known this value at the moment it made
    the decision?** The code and tests in each section let you answer that for sure
    instead of guessing.
    ''',
)

add_markdown(
    "mental-model",
    r'''
    ---
    ## §0. How to think about a feature store

    Here is a definition worth keeping:

    > A feature store is a set of agreed-upon feature definitions, plus the
    > machinery that computes their past values, serves their current values, keeps
    > a record of where they came from, and checks that models really receive what
    > those definitions promised.

    People often call a feature store the "single source of truth." What that should
    mean is: one agreed meaning per feature, and one clear record of how it was
    built. It does not mean one database. Here is why. Training a model reads a huge
    pile of history all at once, while a live model needs one customer's values in a
    few thousandths of a second. Those two jobs pull in opposite directions, so most
    systems end up storing the same feature in two different shapes.

    ```mermaid
    flowchart LR
        A[Raw tables and streams] --> B[Feature definition]
        B --> C[Historical backfill]
        B --> D[Current materialization]
        C --> E[Training dataset]
        D --> F[Online model request]
        E --> G[Parity checks]
        F --> G
    ```

    Read the diagram from left to right. One feature definition splits into two
    paths. On the top path, a **backfill** goes back and recomputes what the feature
    would have said at many moments in the past, which is what training needs. On the
    bottom path, a **materialization** is a saved copy of the current value, kept
    ready so a live model can grab it quickly. Both paths end at the same box, because
    a **parity check** compares the two answers and confirms they agree. Those three
    words (backfill, materialization, parity check) come up constantly once you start
    reading about feature stores, so it is worth pinning them down now.

    A feature store also does not replace the other systems around it. It sits
    alongside them and plugs into them:

    - the **warehouse or lake**, the large store that holds your historical tables;
    - the **stream processor**, which handles events one at a time as they arrive;
    - the **job scheduler**, which runs work on a timetable;
    - the **model registry**, which keeps track of trained models and versions;
    - the **vector database**, used for finding items similar to one another;
    - the **data catalog**, the searchable list of what data your company has.

    And it is perfectly fine not to have one at all. If you have a single model that
    runs on a schedule, five cheap features, one owner, no feature definitions shared
    across teams, and a warehouse pipeline that already works, then careful SQL plus
    versioned data jobs can do the whole job.
    ''',
)

add_markdown(
    "glossary",
    r'''
    ### Working vocabulary

    These terms show up again and again in the sections ahead, so read them once now
    and refer back whenever one trips you up.

    | Term | Plain meaning |
    |---|---|
    | **Entity** | The thing whose history you summarize, such as `customer_id`. |
    | **Event time** | When something actually happened. |
    | **Availability time** | When your pipeline could first see it and use it. |
    | **Prediction time** | The moment the model made, or would have made, a decision. |
    | **Request context** | The facts that arrive with the request itself, already known at prediction time. |
    | **Point-in-time join** | A lookup that returns only what was known at each past prediction time. |
    | **Spine** | The list of past moments you want features for, one row per prediction. |
    | **Backfill** | Recomputing historical feature values from the source history. |
    | **Offline store** | Full history, arranged for big scans and model training. |
    | **Online store** | Current values only, arranged for fast lookups by key. |
    | **Freshness** | How up to date a value is, which is not the same as how fast the lookup is. |
    | **Training-serving skew** | Any difference between the inputs used in training and the inputs used in production. |
    | **Parity check** | A test that recomputes a served value offline and confirms the two match. |
    | **Feature contract** | The written rules for a feature: meaning, keys, type, time rule, owner, version, freshness target, what null means, and where it came from. |

    One clarification about the word feature. A plain number copied straight out of a
    table still counts as a feature. The features that gain the most from a feature
    store are the ones computed once and reused by several models, but reuse is a
    bonus, not part of what makes something a feature.
    ''',
)

add_markdown(
    "setup-explanation",
    r'''
    ---
    ## §1. Set up so the results repeat

    Every dependency is listed in `pyproject.toml`. From this directory, run:

    ```bash
    cp .env.example .env
    uv sync
    uv run python _build_notebook.py
    uv run jupyter lab
    ```

    `uv` is a package manager for Python. The `uv sync` command reads two files, a
    manifest that lists what the project wants and a lock file that records the exact
    package versions last known to work together, and then builds an environment that
    matches. This is the trick that makes results repeatable: everyone who runs this
    notebook gets the same versions, instead of whatever happened to be newest that
    day. One version note matters here. Chronon 0.0.114 needs PySpark 3.3.1, and that
    version of PySpark does not work with Python 3.11 or newer, so this project sticks
    with Python 3.10 on purpose. The optional Spark lab in §8 also needs Java, which
    comes in automatically through the `jdk4py` package, so you never have to install
    Java yourself.

    This lesson needs no API key. Settings still load through
    `load_dotenv(find_dotenv())`, a line that reads a file named `.env` in the project
    and turns its contents into environment variables. If you ever do add a password
    or a token, `.env` is where it goes, and never typed directly into a notebook cell
    that you might later share. The `.env.example` file in the repository holds only
    harmless defaults.

    The next cell imports the libraries, locates the project folder, and prints the
    versions it found. Run it and check that the printed path ends in
    `feature_engineering`. If instead you get an error telling you to start Jupyter from
    that folder, stop and restart Jupyter there, because every later cell reads and
    writes files using paths measured from it.
    ''',
)

add_code(
    "setup-code",
    r'''
    import copy
    import importlib.metadata as metadata
    import json
    import os
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import requests
    from dotenv import find_dotenv, load_dotenv
    from IPython.display import display

    load_dotenv(find_dotenv())

    ROOT = Path.cwd().resolve()
    if not (ROOT / "pyproject.toml").exists():
        raise RuntimeError("Start Jupyter from feature_engineering.")

    CACHE_PATH = ROOT / os.getenv(
        "UCI_RETAIL_CACHE", "data/cache/online_retail.csv"
    )
    CUSTOMER_LIMIT = os.getenv("TUTORIAL_CUSTOMER_LIMIT", "").strip()
    CUSTOMER_LIMIT = int(CUSTOMER_LIMIT) if CUSTOMER_LIMIT else None

    versions = {
        name: metadata.version(name)
        for name in ["chronon-ai", "pandas", "numpy", "scikit-learn"]
    }
    print("Project root:", ROOT)
    print("Versions:", versions)
    print("Customer limit:", CUSTOMER_LIMIT or "all identified customers")
    ''',
)

add_markdown(
    "decision",
    r'''
    ---
    ## §2. Start with the decision, not the table

    It is tempting to open the data first and ask "what could I compute from this?"
    Resist that. Start from the other end, with the decision the model has to make.
    Our model runs the moment a completed order is recorded, and it estimates whether
    that same customer will place another order within 30 days.

    Picture each training row as a frozen question: "At this exact moment, what could
    we honestly say about this customer?" We ask that question once for every
    completed order.

    | Design choice | This tutorial |
    |---|---|
    | One prediction covers | one completed order |
    | Entity key | `customer_id` |
    | Prediction time | the `InvoiceDate` of the current order |
    | Known right now | current order value, item count, product count, country |
    | History we summarize | earlier purchases and cancellations |
    | Feature window | `[prediction_time - window, prediction_time)` |
    | Label window | `(prediction_time, prediction_time + 30 days]` |
    | Positive label | at least one later completed order inside that window |

    Two rows in that table use bracket notation, and it is worth decoding now,
    because we lean on it for the rest of the notebook. The rule is simple: a square
    bracket `[` includes the endpoint, and a round bracket `)` leaves it out. So the
    feature window `[prediction_time - window, prediction_time)` reads as "start 30
    days back, include that starting instant, and stop just short of the decision
    itself." The label window `(prediction_time, prediction_time + 30 days]` is the
    mirror image: skip the decision instant, then include everything right up to and
    including 30 days later. §5 shows why those exact edges matter so much.

    The current order is fair to use, because of course we know about it at the moment
    the decision happens. Teams call facts like these the **request context**,
    meaning everything that arrives along with the request itself. The one thing the
    request context must never do is sneak into a feature named something like "spend
    in the prior 30 days," because that name promises history but would now secretly
    contain the present. To keep the two kinds of information from bleeding together,
    we keep them in separate groups of columns: facts known at the decision in one
    group, summaries of earlier history in the other. In §7 you will watch Chronon
    express that exact same split as the left and right sides of a join.

    **One honest caution:** this public dataset records completed invoices, not real
    checkout requests, so our "moment of decision" is really an approximation. A real
    deployment would have to pin down exactly when the model gets called and which
    fields of the current order can actually be trusted at that instant.
    ''',
)

add_markdown(
    "three-clocks",
    r'''
    ![Event time, prediction time, and availability time](assets/02-three-clocks.svg)

    ### Three clocks you have to keep separate

    This is where confusion about feature stores usually begins, so let's go slowly.
    Building training data means recreating past moments after the fact, and to do
    that honestly you have to keep track of three different times. They are easy to
    confuse because in everyday life we only ever think about one of them.

    1. **Event time** is when something happened in the real world. A customer placed
       an order at 09:30.
    2. **Availability time** is when your systems could first *see* that event. The
       row for that order did not land in the company's database until 10:10, forty
       minutes after the customer clicked buy.
    3. **Prediction time** is the moment the model has to decide something. Say the
       model runs at 10:00.

    Now line those three numbers up. The order happened at 09:30, which is before the
    10:00 decision, so at first glance it looks like fair game. But at 10:00 that row
    was still nowhere to be found: it did not arrive until 10:10. A model running live
    at 10:00 simply could not have used it, no matter how badly we might want it to.

    So when you rebuild the past later, you have two reasonable choices:

    - **rebuild by event time:** include everything that *happened* before the
      decision. Our 09:30 order counts.
    - **rebuild by what production could see:** include only what had actually
      *arrived* before the decision. Our 09:30 order does not count, because it showed
      up ten minutes too late.

    The second choice matches what a live model really experienced, which makes it the
    more faithful one. It is also the harder one, because it needs your data to record
    when each row arrived, and plenty of datasets never bother to write that down.

    Ours is one of those datasets. UCI Online Retail tells us when each order happened
    but says nothing about when the retailer's systems received it. So the main part of
    this notebook rebuilds history by event time. That is the honest limit of this
    data, not a shortcut we took to save effort. In §6 we run a tiny example with both
    clocks side by side, so the clock we are missing stays fresh in your mind.

    One last point, about what fixing the clocks does and does not buy you. Getting
    these three times right prevents one specific family of mistakes: information from
    later leaking backward into an earlier decision. It does *not* protect you from
    accidentally feeding the model the answer itself, from letting your test data
    influence how you prepare the training data, or from asking a question whose answer
    has not happened yet. Those are separate traps, and §9 walks through each one.
    ''',
)

add_markdown(
    "contract-preview",
    r'''
    ![A feature needs clear rules, a source, and an owner](assets/07-feature-contract.svg)

    Before anyone writes a line of code for a feature named
    `customer_avg_order_value_30d`, the team should be able to answer a short list of
    questions about it. Do cancellations pull the total down? Is the current order
    left out? Which currency? What does a missing value mean? How old is too old? Who
    gets paged when it breaks? Which models use it? What changed between versions?

    The answers to those questions are the feature contract. To see why it matters,
    picture what happens without one. One team builds `customer_avg_order_value_30d`
    counting cancellations as negative amounts. Another team builds a column with the
    exact same name but ignores cancellations entirely. A third team reuses whichever
    one they stumble on first, because the name sounds right. All three are convinced
    they are talking about the same thing, and none of them are. Reuse only saves you
    work when the *meaning* travels with the name, and a contract is how the meaning
    travels.
    ''',
)

add_markdown(
    "data-provenance",
    r'''
    ---
    ## §3. Load the real data and look at it honestly

    The [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
    holds real transactions from 1 December 2010 through 9 December 2011. The company
    sold mostly gift items, and many of its customers were wholesalers buying in bulk.
    UCI publishes the data under CC BY 4.0 with DOI `10.24432/C5BW33`.

    The first time you run it, the code downloads the CSV into `data/cache/`, a folder
    git ignores, and every later run just reuses that local copy. The download is
    about 45 MB, so give it a moment. It writes to a temporary file ending in `.part`
    and only renames it once the last byte has arrived, so an interrupted download can
    never leave behind a file that looks finished but is not.

    The second cell reads the file and prints a summary. You should see 541,909 rows
    across 8 columns, spanning 1 December 2010 at 08:26 to 9 December 2011 at 12:50,
    with 4,372 identified customers across 38 countries. Below that, the first three
    rows reveal the shape of the raw data: there is one line per product on an invoice,
    so a single order containing five different items shows up as five separate rows.
    That one detail drives everything we do in §4.
    ''',
)

add_code(
    "download-data",
    r'''
    DATA_URL = "https://archive.ics.uci.edu/static/public/352/data.csv"


    def download_if_missing(url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 1_000_000:
            return destination

        partial = destination.with_suffix(destination.suffix + ".part")
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
        partial.replace(destination)
        return destination


    csv_path = download_if_missing(DATA_URL, CACHE_PATH)
    print(f"Dataset cache: {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")
    ''',
)

add_code(
    "read-data",
    r'''
    RAW_DTYPES = {
        "InvoiceNo": "string",
        "StockCode": "string",
        "Description": "string",
        "Quantity": "int32",
        "UnitPrice": "float64",
        "CustomerID": "string",
        "Country": "string",
    }

    raw = pd.read_csv(
        csv_path,
        dtype=RAW_DTYPES,
        parse_dates=["InvoiceDate"],
    )

    raw_summary = pd.Series(
        {
            "rows": len(raw),
            "columns": raw.shape[1],
            "first_event": raw["InvoiceDate"].min(),
            "last_event": raw["InvoiceDate"].max(),
            "identified_customers": raw["CustomerID"].nunique(dropna=True),
            "countries": raw["Country"].nunique(dropna=True),
        },
        name="value",
    )
    display(raw_summary.to_frame())
    display(raw.head(3))
    ''',
)

add_markdown(
    "quality-explanation",
    r'''
    ### Data quality is part of what a feature means

    We will not quietly throw rows away. Instead we first count how many rows have
    problems for this particular task, and spell out why each one is a problem:

    - a missing `CustomerID` gives us no stable customer to attach history to;
    - an invoice number starting with `C` marks a cancellation;
    - a negative quantity is another sign of a returned or reversed line;
    - a zero or negative price does not fit our idea of what an order is worth;
    - unusually large quantities are left in and simply watched, because silently
      trimming them would quietly change what the feature means.

    The next cell counts each of those conditions and reports how many rows it affects
    and what share of the file that is. The headline number is missing customer IDs:
    135,080 rows, or 24.9 percent of the whole file. Those are most likely guest
    checkouts, and since we cannot tie them to anyone's history, a full quarter of the
    dataset simply cannot take part in a lesson about per-customer features.
    Cancellation invoices account for 9,288 rows, negative quantities for 10,624, and
    zero-or-negative prices for 2,517.

    That first number deserves a comment. The UCI page claims the dataset has no
    missing values, and yet a quarter of the rows we just downloaded have no customer
    ID. When the documentation and the data disagree, believe the data. A real pipeline
    would then follow a policy that someone reviewed and wrote down: stop the run, set
    the bad rows aside for a human to look at, or report them and keep going. The point
    is not which choice you make. The point is that a person chose, rather than a
    filter silently deleting a quarter of the input.
    ''',
)

add_code(
    "quality-audit",
    r'''
    quality_flags = pd.DataFrame(
        {
            "missing_customer_id": raw["CustomerID"].isna(),
            "missing_description": raw["Description"].isna(),
            "invoice_starts_c": raw["InvoiceNo"].str.upper().str.startswith(
                "C", na=False
            ),
            "negative_quantity": raw["Quantity"] < 0,
            "nonpositive_price": raw["UnitPrice"] <= 0,
        }
    )

    quality_report = (
        quality_flags.agg(["sum", "mean"])
        .T.rename(columns={"sum": "rows", "mean": "rate"})
        .sort_values("rows", ascending=False)
    )
    quality_report["rows"] = quality_report["rows"].astype(int)
    display(quality_report)

    raw["is_cancellation"] = (
        raw["InvoiceNo"].str.upper().str.startswith("C", na=False)
        | (raw["Quantity"] < 0)
    )
    raw["line_value"] = raw["Quantity"] * raw["UnitPrice"]
    ''',
)

add_markdown(
    "normalize-events-explanation",
    r'''
    ---
    ## §4. Turn order lines into event tables

    Right now the data is stored one product line at a time, but our prediction is
    about a whole order. A customer who bought five different items left behind five
    rows, and we want a single row that says what that order was worth. So we reshape
    the file into two tables:

    - `orders`: one row per completed purchase invoice with a known customer, carrying
      its total value, how many items it contained, and how many distinct products;
    - `cancellations`: one row per cancellation invoice with a known customer, carrying
      the refunded value.

    We group the lines by customer, invoice, timestamp, and country. Grouping on that
    exact combination makes the result deterministic, which just means you get an
    identical table every single time you run it. That matters, because a feature you
    cannot reproduce is a feature you cannot debug. The original line items stay
    untouched in `raw`, so you always have the source to audit against.

    You should watch 541,909 raw lines collapse into 18,562 orders from 4,338
    customers, worth about £8.9 million in total, plus 3,655 cancellations from 1,589
    customers worth about £611,000. That collapse from half a million rows to eighteen
    thousand is worth pausing on: the unit of the data is now the unit of the decision,
    one row per order, which is exactly what we are predicting about.

    In a real production system, these two tables would be maintained further upstream
    as proper data products, complete with event IDs, arrival timestamps, partitions,
    and rules for removing duplicates. Here they are just one step in a notebook.

    If you want a faster classroom run, set `TUTORIAL_CUSTOMER_LIMIT` to keep only the
    N most active customers. The default keeps every identified customer.
    ''',
)

add_code(
    "normalize-events-code",
    r'''
    eligible = raw.loc[
        raw["CustomerID"].notna()
        & (raw["UnitPrice"] > 0)
        & (raw["Quantity"] != 0)
    ].copy()

    purchase_lines = eligible.loc[
        ~eligible["is_cancellation"] & (eligible["Quantity"] > 0)
    ].copy()
    purchase_lines["line_value"] = purchase_lines["line_value"].clip(lower=0)

    orders = (
        purchase_lines.groupby(
            ["CustomerID", "InvoiceNo", "InvoiceDate", "Country"],
            observed=True,
            as_index=False,
        )
        .agg(
            order_value=("line_value", "sum"),
            item_count=("Quantity", "sum"),
            unique_products=("StockCode", "nunique"),
        )
        .rename(
            columns={
                "CustomerID": "customer_id",
                "InvoiceNo": "order_id",
                "InvoiceDate": "prediction_ts",
                "Country": "country",
            }
        )
    )

    cancellation_lines = eligible.loc[eligible["is_cancellation"]].copy()
    cancellation_lines["refund_value"] = cancellation_lines["line_value"].abs()
    cancellations = (
        cancellation_lines.groupby(
            ["CustomerID", "InvoiceNo", "InvoiceDate"],
            observed=True,
            as_index=False,
        )
        .agg(
            refund_value=("refund_value", "sum"),
            item_count=("Quantity", lambda values: int(values.abs().sum())),
        )
        .rename(
            columns={
                "CustomerID": "customer_id",
                "InvoiceNo": "cancellation_id",
                "InvoiceDate": "event_ts",
            }
        )
    )

    if CUSTOMER_LIMIT:
        keep = orders["customer_id"].value_counts().head(CUSTOMER_LIMIT).index
        orders = orders.loc[orders["customer_id"].isin(keep)]
        cancellations = cancellations.loc[cancellations["customer_id"].isin(keep)]

    orders = orders.sort_values(
        ["customer_id", "prediction_ts", "order_id"]
    ).reset_index(drop=True)
    cancellations = cancellations.sort_values(
        ["customer_id", "event_ts", "cancellation_id"]
    ).reset_index(drop=True)

    event_summary = pd.DataFrame(
        {
            "events": [len(orders), len(cancellations)],
            "customers": [
                orders["customer_id"].nunique(),
                cancellations["customer_id"].nunique(),
            ],
            "value_total_gbp": [
                orders["order_value"].sum(),
                cancellations["refund_value"].sum(),
            ],
        },
        index=["completed_orders", "cancellations"],
    )
    display(event_summary)
    display(orders.head(3))
    ''',
)

add_markdown(
    "timeline",
    r'''
    ---
    ## §5. Build past-only features and future labels

    ![Historical feature windows and the future label horizon](assets/03-point-in-time-window.svg)

    Now that we have clean event tables, we can build the actual training data. Every
    training row starts life as a **spine** row. The spine is the list of moments you
    want features for, one row per prediction, and it earns that name because it is the
    backbone that everything else attaches to. Each spine row holds an example ID, the
    entity key, the prediction time, whatever was known at that moment, and eventually
    a label. In our case the spine writes itself: we already decided to predict at
    every completed order, so our 18,562 orders simply become 18,562 spine rows.

    Features then hang off each spine row, and every one of them obeys a single rule: a
    feature may only use events whose timestamps fall strictly *before* that row's
    prediction time.

    We write the feature window as `[t - 30d, t)`, where `t` is the prediction time.
    Because the round bracket excludes the end, `t` itself is left out. Two things fall
    outside the window because of that: the current order, and anything else stamped at
    the exact same instant. Drawing the boundary this strictly is a deliberate choice.
    It is easy to state, easy to test, and it matches a prediction made in the instant
    just before the current order joins the customer's history.

    The label looks the other direction, over `(t, t + 30d]`. This time the excluded
    end has flipped sides: an order at exactly `t` is the order we are predicting
    *from*, so it cannot count as a repeat purchase, while an order landing exactly 30
    days later does count. A concrete example makes both windows easier to hold in your
    head. Say a customer orders on 1 January, again on 10 January, and again on 11
    January, and we make our prediction at the 10 January order. Then the 1 January
    order is history (it goes into the features), the 10 January order is the request
    itself (excluded from both), and the 11 January order is the future that decides
    the label. In §6 we turn this exact example into a test.

    The next cell defines a small helper that both windows use, and it rests on two
    ideas worth understanding:

    - `np.searchsorted` takes a sorted list of timestamps and finds the position where
      a new time would slot in. Ask it where the *start* and the *end* of a window
      would go, and the gap between those two positions tells you which events fall
      inside the window;
    - a **prefix sum** is a running total computed once, up front. Once you have those
      running totals, the sum of any stretch is just "total at the end minus total at
      the start." So adding up a window becomes two quick lookups instead of one
      addition per event.

    Together these two tricks keep the work proportional to the number of examples and
    windows, instead of forcing us to compare every row against every other row, which
    on 18,562 orders would be roughly 344 million comparisons. Chronon reaches the same
    answers with its own machinery, built for data far too big to fit on a laptop.
    ''',
)

add_code(
    "window-helper",
    r'''
    DAY_NS = np.int64(86_400_000_000_000)


    def window_stats(
        history_times: np.ndarray,
        history_values: np.ndarray,
        anchor_times: np.ndarray,
        days: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return count and sum over [anchor - days, anchor)."""
        right = np.searchsorted(history_times, anchor_times, side="left")
        left = np.searchsorted(
            history_times, anchor_times - days * DAY_NS, side="left"
        )
        prefix = np.concatenate(([0.0], np.cumsum(history_values, dtype=float)))
        return right - left, prefix[right] - prefix[left]
    ''',
)

add_markdown(
    "pit-builder-explanation",
    r'''
    ### What the reference builder does, step by step

    For each customer, the function below walks through these steps:

    1. it uses that customer's sorted order timestamps as both the history to look
       back over and the list of moments to predict at;
    2. it counts orders and adds up spending over the prior 7, 30, and 90 days;
    3. it computes prior averages only when at least one earlier order actually exists;
    4. it measures recency as the time since the most recent order strictly before the
       cutoff;
    5. it counts *later* orders, purely to build the label;
    6. it repeats the same window lookups against the cancellation history;
    7. it flags rows that sit too close to the end of the dataset, because their labels
       have not finished forming yet.

    The column `leak_future_order_count_30d` is here only for the demonstration in §9.
    It peeks into the future on purpose, and it must never appear in a real feature
    list. We keep it around so we can show you exactly what cheating looks like.

    Zero and null mean two different things here, and the difference genuinely matters.
    A count of zero means we looked inside the window and found nothing. A null average
    or a null recency means there was no earlier order at all, so there was nothing to
    compute from in the first place. If we collapsed those two into one, we would be
    telling the model that a brand-new customer and a customer who happened to pause
    for a month are the same thing, and they are not.

    When the cell finishes you should see 18,562 spine rows, of which 15,822 have
    labels we can trust, with a positive rate of 45.6 percent among those. That last
    number means almost half of these orders were followed by another order within 30
    days, which makes sense for a wholesale business, and it also tells us our two
    outcomes (repeat or not) are close to evenly balanced. The five sample rows printed
    below are worth reading across: notice how a customer's very first order has zeros
    in the window columns and nulls in the average and recency columns, exactly as
    described above.
    ''',
)

add_code(
    "pit-builder-code",
    r'''
    def build_point_in_time_features(
        order_events: pd.DataFrame,
        cancellation_events: pd.DataFrame,
    ) -> pd.DataFrame:
        out = order_events.rename(
            columns={"order_value": "current_order_value"}
        ).copy()

        for days in (7, 30, 90):
            out[f"purchase_order_value_sum_{days}d"] = 0.0
            out[f"purchase_order_id_count_{days}d"] = 0
        for days in (30, 90):
            out[f"purchase_order_value_average_{days}d"] = np.nan

        out["purchase_item_count_sum_30d"] = 0.0
        out["days_since_last_order"] = np.nan
        out["repeat_purchase_30d"] = 0
        out["leak_future_order_count_30d"] = 0

        for _, row_index in out.groupby("customer_id", sort=False).groups.items():
            row_index = np.asarray(list(row_index), dtype=int)
            times = out.loc[row_index, "prediction_ts"].astype("int64").to_numpy()
            values = out.loc[row_index, "current_order_value"].to_numpy(float)
            items = out.loc[row_index, "item_count"].to_numpy(float)

            for days in (7, 30, 90):
                counts, sums = window_stats(times, values, times, days)
                out.loc[row_index, f"purchase_order_value_sum_{days}d"] = sums
                out.loc[row_index, f"purchase_order_id_count_{days}d"] = counts
                if days in (30, 90):
                    averages = np.divide(
                        sums,
                        counts,
                        out=np.full(len(row_index), np.nan),
                        where=counts > 0,
                    )
                    out.loc[
                        row_index, f"purchase_order_value_average_{days}d"
                    ] = averages

            _, item_sums = window_stats(times, items, times, 30)
            out.loc[row_index, "purchase_item_count_sum_30d"] = item_sums

            right = np.searchsorted(times, times, side="left")
            previous = right - 1
            has_previous = previous >= 0
            recency = np.full(len(row_index), np.nan)
            recency[has_previous] = (
                times[has_previous] - times[previous[has_previous]]
            ) / DAY_NS
            out.loc[row_index, "days_since_last_order"] = recency

            future_start = np.searchsorted(times, times, side="right")
            future_end = np.searchsorted(times, times + 30 * DAY_NS, side="right")
            future_count = future_end - future_start
            out.loc[row_index, "leak_future_order_count_30d"] = future_count
            out.loc[row_index, "repeat_purchase_30d"] = (future_count > 0).astype(int)

        cancellation_groups = {
            customer: group
            for customer, group in cancellation_events.groupby(
                "customer_id", sort=False
            )
        }
        for days in (30, 90):
            out[f"cancel_refund_value_sum_{days}d"] = 0.0
            out[f"cancel_cancellation_id_count_{days}d"] = 0

        for customer, row_index in out.groupby(
            "customer_id", sort=False
        ).groups.items():
            history = cancellation_groups.get(customer)
            if history is None:
                continue
            row_index = np.asarray(list(row_index), dtype=int)
            anchors = out.loc[row_index, "prediction_ts"].astype("int64").to_numpy()
            times = history["event_ts"].astype("int64").to_numpy()
            values = history["refund_value"].to_numpy(float)
            for days in (30, 90):
                counts, sums = window_stats(times, values, anchors, days)
                out.loc[row_index, f"cancel_refund_value_sum_{days}d"] = sums
                out.loc[
                    row_index, f"cancel_cancellation_id_count_{days}d"
                ] = counts

        out["cancel_value_ratio_90d"] = (
            out["cancel_refund_value_sum_90d"]
            / (out["purchase_order_value_sum_90d"] + 1.0)
        )

        complete_label_cutoff = out["prediction_ts"].max() - pd.Timedelta(days=30)
        out["label_is_observed"] = out["prediction_ts"] <= complete_label_cutoff
        return out


    features = build_point_in_time_features(orders, cancellations)
    print(f"Training spine rows: {len(features):,}")
    print(f"Observed label rows: {features['label_is_observed'].sum():,}")
    print(f"Observed positive rate: "
          f"{features.loc[features.label_is_observed, 'repeat_purchase_30d'].mean():.1%}")
    display(features.head(5))
    ''',
)

add_markdown(
    "label-censoring",
    r'''
    ### Why an unfinished label is dangerous

    That count of 15,822 out of 18,562 deserves an explanation, because we just set
    aside 2,740 rows on purpose. Here is why. Our data stops on 9 December 2011. An
    order placed on 8 December cannot get a trustworthy 30-day repeat-purchase label,
    because only one day of its 30-day window actually exists inside the file. "We saw
    no later order" and "there was no later order" are very different statements, and
    labeling those rows as negatives would teach the model something that is simply not
    true. So the builder flags every row from the final 30 days, meaning everything
    after 9 November, as having a label that has not finished forming yet. There is a
    name for this problem: **censoring**. The outcome is not missing at random, it is
    cut off by where our data happens to end.

    We hold on to those rows rather than delete them, because the feature-serving
    exercises in §10 still need current values for real customers. We just exclude them
    from training, using the `label_is_observed` flag. This trap is worth taking
    seriously, because no amount of careful point-in-time feature work can rescue a
    label that never finished forming. In production, the label pipeline needs rules of
    its own: how long to wait before labeling anything, what to do with outcomes that
    trickle in after that deadline, and how to recompute old labels under a new version
    without quietly changing the meaning of a training set that already exists.
    ''',
)

add_markdown(
    "proof-intro",
    r'''
    ---
    ## §6. Test the cutoff on rows you can check by hand

    We just built 18,562 rows of features and the numbers looked reasonable. But
    "looks reasonable" is not the same as "is correct," and nobody can verify a time
    boundary by scrolling through eighteen thousand rows. So we do what engineers do:
    we build a tiny input where we already know every right answer by hand, run the
    real code on it, and check that the code agrees with us.

    A small handmade input like that is called a **test fixture**. Ours has three
    orders for one customer, `C1`, who spends £10 on 1 January, £20 on 10 January, and
    £999 on 11 January. We ask for a prediction at the 10 January order. Try working
    out the answers yourself before running the cell, then compare:

    - prior 30-day spend should be exactly £10, because only the 1 January order sits
      inside the window;
    - the current £20 order should stay out of that total, since the window stops just
      short of the prediction time;
    - the future £999 order should stay out of every feature, since it has not happened
      yet;
    - the label should be positive, because a later order really does arrive within 30
      days.

    The next cell states each of those expectations as an `assert`. Remember, an
    `assert` is a claim that has to be true for the notebook to keep going, so if the
    boundary ever slips, this cell stops with an error instead of letting a wrong number
    slide quietly into a model. Notice too that we run the very same
    `build_point_in_time_features` function we used on the real data, not a simplified
    copy of it. A test that runs different code from production proves nothing about
    production.
    ''',
)

add_code(
    "boundary-test",
    r'''
    fixture_orders = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C1"],
            "order_id": ["O1", "O2", "O3"],
            "prediction_ts": pd.to_datetime(
                ["2025-01-01", "2025-01-10", "2025-01-11"]
            ),
            "country": ["GB", "GB", "GB"],
            "order_value": [10.0, 20.0, 999.0],
            "item_count": [1, 1, 1],
            "unique_products": [1, 1, 1],
        }
    )
    fixture_cancellations = pd.DataFrame(
        columns=[
            "customer_id", "cancellation_id", "event_ts",
            "refund_value", "item_count"
        ]
    )
    fixture = build_point_in_time_features(
        fixture_orders, fixture_cancellations
    )
    at_t = fixture.loc[fixture["order_id"].eq("O2")].iloc[0]

    assert at_t["purchase_order_value_sum_30d"] == 10.0
    assert at_t["purchase_order_id_count_30d"] == 1
    assert at_t["repeat_purchase_30d"] == 1
    assert at_t["leak_future_order_count_30d"] == 1
    display(
        at_t[
            [
                "prediction_ts",
                "current_order_value",
                "purchase_order_value_sum_30d",
                "repeat_purchase_30d",
            ]
        ].to_frame("value")
    )
    print("Boundary test passed.")
    ''',
)

add_markdown(
    "availability-example",
    r'''
    ### The same fixture idea, now for arrival time

    Remember from §2 that our real data records when each order happened but never
    records when it arrived. Here we can have both, because we are writing the rows
    ourselves. The next cell holds a single £42 event that happened at 09:30 and
    arrived at 10:10, and it asks what a prediction made at 10:00 should be allowed to
    see.

    The cell works out both answers side by side. Rebuilding by event time gives £42,
    because 09:30 came before 10:00. Rebuilding by what production could actually see
    gives £0, because at 10:00 that row had not landed yet. Same event, same prediction
    moment, two perfectly valid answers, and the entire difference comes down to which
    clock you filter on. Look closely and you will see the two filters differ by exactly
    one extra condition: the second one also insists that `available_ts <=
    prediction_ts`.

    These two choices come up so often that they have names, which you will meet in the
    code below and in the companion theory guide. Rebuilding by event time is called
    **event truth**, and rebuilding from what had actually arrived is called
    **production-available truth**. The names matter less than the habit of clearly
    saying which one you built, because otherwise two people reading the same training
    set will quietly assume different things about it.

    Here is the part worth remembering. Chronon computes point-in-time values correctly
    along whichever timeline you hand it, so the choice of clock is yours, not the
    tool's. But if your source has been corrected since it was first written, or holds
    rows that arrived late, then replaying exactly what production saw depends on
    someone having recorded arrival times in the first place. A feature store cannot
    rebuild a clock that nobody ever wrote down.
    ''',
)

add_code(
    "availability-code",
    r'''
    delayed = pd.DataFrame(
        {
            "event_ts": pd.to_datetime(["2025-01-01 09:30"]),
            "available_ts": pd.to_datetime(["2025-01-01 10:10"]),
            "amount": [42.0],
        }
    )
    prediction_ts = pd.Timestamp("2025-01-01 10:00")

    event_truth = delayed.loc[delayed["event_ts"] < prediction_ts, "amount"].sum()
    available_truth = delayed.loc[
        (delayed["event_ts"] < prediction_ts)
        & (delayed["available_ts"] <= prediction_ts),
        "amount",
    ].sum()

    display(
        pd.Series(
            {
                "event_time_reconstruction": event_truth,
                "production_available_reconstruction": available_truth,
            },
            name="sum_before_prediction",
        ).to_frame()
    )
    assert event_truth == 42.0 and available_truth == 0.0
    ''',
)

add_markdown(
    "invariant-explanation",
    r'''
    ### Checks that hold for every row

    The fixture tests check specific numbers on rows we wrote by hand. This other kind
    of test checks a statement that must hold for all 15,822 real rows at once,
    whatever their values happen to be. A statement like that is called an
    **invariant**, because it does not vary: if the code is correct it is true
    everywhere, and even a single violation means something is broken.

    The next cell runs five such checks. Here is the reasoning behind each one:

    - a 7-day count can never be larger than a 30-day count, and a 30-day count can
      never be larger than a 90-day count, because a shorter window can only ever look
      at a subset of the events a longer window sees. If this check fails, a window
      boundary is being computed wrong somewhere;
    - no count may be negative, which catches arithmetic slips in the prefix-sum trick,
      since a negative number of orders is nonsense;
    - the label may only be 0 or 1, since it answers a yes-or-no question;
    - every row must have a customer ID, because a feature about a customer with no
      customer to attach it to means nothing.

    Invariants like these are cheap to write and they keep paying off as the code
    changes, which makes them a good habit far beyond feature stores. The cell prints a
    small pass-or-fail table, then asserts that all five passed.
    ''',
)

add_code(
    "invariant-tests",
    r'''
    observed = features.loc[features["label_is_observed"]]

    invariant_results = {
        "7d_count_le_30d": bool(
            (observed["purchase_order_id_count_7d"]
             <= observed["purchase_order_id_count_30d"]).all()
        ),
        "30d_count_le_90d": bool(
            (observed["purchase_order_id_count_30d"]
             <= observed["purchase_order_id_count_90d"]).all()
        ),
        "counts_nonnegative": bool(
            (observed.filter(like="_count_") >= 0).all().all()
        ),
        "target_binary": bool(
            observed["repeat_purchase_30d"].isin([0, 1]).all()
        ),
        "keys_present": bool(observed["customer_id"].notna().all()),
    }
    display(pd.Series(invariant_results, name="passed").to_frame())
    assert all(invariant_results.values())
    ''',
)

add_markdown(
    "chronon-intro",
    r'''
    ---
    ## §7. Write the same design in Chronon

    ![Chronon Source, GroupBy, and Join object model](assets/04-chronon-object-model.svg)

    Our NumPy builder works, and it is tested. So why drag in a whole platform at all?
    Because that little function has three real limits. It only runs on one machine, so
    it could never chew through a whole company's history. It only produces training
    data, so nothing computes these same numbers for a *live* model. And even if
    someone wrote a second version for live serving, nothing would keep the two in step
    as the definition changed over time. A feature platform exists to solve all three
    of those problems from a single definition.

    Chronon expresses the design we already built using a handful of objects, each with
    one clear job:

    - `Source` says where the data lives, in the warehouse and optionally in a stream
      of new events, and which column holds event time;
    - `GroupBy` says which entity to key on, what to **aggregate** (which just means
      which column to summarize and how), over which windows, and how exact the timing
      has to be;
    - `Join` supplies the spine, our list of prediction moments, and attaches the
      `GroupBy` results as of each one. This is the same left-and-right split we set up
      back in §2;
    - the **compiler** turns those Python definitions into JSON files, which is the form
      the engine actually reads when it runs a job.

    The definitions in this project are ordinary Python files:

    ```text
    chronon/
    ├── teams.json
    ├── group_bys/retail/purchases.py
    ├── group_bys/retail/cancellations.py
    └── joins/retail/repeat_purchase_training.py
    ```

    The table and topic names inside them are promises about your data platform, not
    something the notebook conjures up. The pandas frames we built earlier do not turn
    into Chronon sources on their own. §8 writes small temporary Spark tables so the
    definitions have something real to read.
    ''',
)

add_code(
    "import-chronon-configs",
    r'''
    CHRONON_ROOT = ROOT / "chronon"
    if str(CHRONON_ROOT) not in sys.path:
        sys.path.insert(0, str(CHRONON_ROOT))

    from ai.chronon.repo.serializer import thrift_simple_json
    from group_bys.retail.cancellations import v1 as cancellations_v1
    from group_bys.retail.purchases import v1 as purchases_v1
    from joins.retail.repeat_purchase_training import v1 as training_join_v1

    chronon_summary = pd.DataFrame(
        [
            {
                "object": purchases_v1.metaData.name,
                "kind": "GroupBy",
                "keys": purchases_v1.keyColumns,
                "aggregations": len(purchases_v1.aggregations),
                "online": purchases_v1.metaData.online,
            },
            {
                "object": cancellations_v1.metaData.name,
                "kind": "GroupBy",
                "keys": cancellations_v1.keyColumns,
                "aggregations": len(cancellations_v1.aggregations),
                "online": cancellations_v1.metaData.online,
            },
            {
                "object": "retail/repeat_purchase_training.v1",
                "kind": "Join",
                "keys": ["customer_id"],
                "aggregations": sum(
                    len(part.groupBy.aggregations)
                    for part in training_join_v1.joinParts
                ),
                "online": training_join_v1.metaData.online,
            },
        ]
    )
    display(chronon_summary)
    ''',
)

add_markdown(
    "groupby-code-explained",
    r'''
    ### Reading the purchase `GroupBy`

    Here are the source and one aggregation, trimmed down to the essentials:

    ```python
    source = Source(events=EventSource(
        table="retail.orders",
        topic="retail.orders.v1",
        query=Query(
            selects=select("customer_id", "order_id", "order_value", "item_count"),
            time_column="ts",
        ),
    ))

    v1 = GroupBy(
        sources=[source],
        keys=["customer_id"],
        aggregations=[Aggregation(
            input_column="order_value",
            operation=Operation.SUM,
            windows=[Window(7, DAYS), Window(30, DAYS), Window(90, DAYS)],
        )],
        accuracy=Accuracy.TEMPORAL,
        online=True,
    )
    ```

    Read it one line at a time, and notice that every line answers a question we
    already worked out by hand in §2. `table` names the history used for training, and
    `topic` names the stream of new events that keeps the online values current, and
    that pairing is how one definition can cover both paths at once. `time_column`
    picks the clock that everything is measured against, which is exactly the §2 choice
    about which of the three clocks you filter on. `keys` names the entity, so every
    result belongs to one customer. Each `Aggregation` pairs a column with an operation
    and one or more windows, and that is how a single `order_value` line blossoms into
    7, 30, and 90-day sums without anyone writing three separate queries.

    Two settings near the bottom are easy to skim past and genuinely important.
    `accuracy=Accuracy.TEMPORAL` asks for values computed as of the *exact* prediction
    moment. The alternative is a snapshot frozen at the previous midnight, which is
    cheaper and sometimes fine, but for an order placed at 15:00 it would ignore
    everything that customer did that same morning. `online=True` marks the definition
    as one that also needs serving pipelines, not just a one-time training backfill.

    One practical detail worth flagging: Chronon's windowed math expects timestamps as
    milliseconds since 1970 (a standard way computers count time). In the table
    contract, `ts` is that raw number, even though pandas politely shows you a friendly
    date and time inside this notebook.
    ''',
)

add_markdown(
    "thrift-explanation",
    r'''
    A definition is only useful if you can check what it actually says. The next cell
    runs the same conversion the compiler does, turning the Python object into JSON,
    and then prints the handful of fields that decide whether the feature is correct:
    its name, whether it is marked for serving, the key it groups by, the table it
    reads from, the column it treats as time, and how many aggregations it defines.

    Get into the habit of reading this JSON rather than trusting the Python source you
    just typed. The JSON is what the engine will actually run, so if a name or a time
    column is wrong here, it is wrong everywhere downstream too.
    ''',
)

add_code(
    "inspect-thrift",
    r'''
    compiled_preview = json.loads(thrift_simple_json(purchases_v1))
    preview = {
        "name": compiled_preview["metaData"]["name"],
        "online": bool(compiled_preview["metaData"]["online"]),
        "keyColumns": compiled_preview["keyColumns"],
        "source": compiled_preview["sources"][0]["events"]["table"],
        "timeColumn": compiled_preview["sources"][0]["events"]["query"][
            "timeColumn"
        ],
        "aggregation_count": len(compiled_preview["aggregations"]),
    }
    print(json.dumps(preview, indent=2))
    ''',
)

add_markdown(
    "join-explained",
    r'''
    ### The `Join` sets the training timeline

    ```python
    v1 = Join(
        left=prediction_events,
        right_parts=[
            JoinPart(group_by=purchases_v1, prefix="purchase"),
            JoinPart(group_by=cancellations_v1, prefix="cancel"),
        ],
        online=True,
        check_consistency=True,
        sample_percent=1.0,
    )
    ```

    The word `left` is where the spine goes. It is the list of past examples, each one
    carrying a `customer_id` to look up and a `ts` saying when to look. Each entry in
    `right_parts` is then evaluated as of that row's own timestamp, one row at a time,
    and that is the whole reason the resulting training data stays honest. Chronon does
    the same walk through history that our NumPy loop did, only on data that would never
    fit inside a notebook.

    The `prefix` on each part is a small naming device with a very practical purpose.
    Both of our `GroupBy` definitions produce a 30-day sum, so without prefixes the
    output would have two columns fighting over one name. With prefixes you get
    `purchase_order_value_sum_30d` and `cancel_refund_value_sum_30d`, and nobody has to
    guess which is which six months later. Finally, `check_consistency=True` tells
    Chronon to keep the records it needs to compare live lookups against recomputed
    offline values. That is exactly the parity check from §0, and the subject of §10.

    Our local training table already carries a finished label on the left side. In a
    bigger system, Chronon's `LabelPart` can manage labels that only become known well
    after the features were computed. Either way, your model code still has to keep the
    label column out of the feature list.

    **A word about versions:** Chronon deliberately makes it hard to edit a compiled
    online `GroupBy` in place. The safe path is to create a `v2`, move consumers over to
    it, compare the two, and only then retire `v1`. Renaming a column while quietly
    changing what it means is one of the classic ways models break without anyone
    noticing.
    ''',
)

add_markdown(
    "compile-commands",
    r'''
    ### Compile it like a normal Chronon project

    The notebook imports these definitions so you can inspect them. Real projects
    also compile them, from the Chronon root:

    ```bash
    cd chronon
    PYTHONPATH=. uv run --project .. compile.py \
      --conf group_bys/retail/purchases.py --force-overwrite -y
    PYTHONPATH=. uv run --project .. compile.py \
      --conf group_bys/retail/cancellations.py --force-overwrite -y
    PYTHONPATH=. uv run --project .. compile.py \
      --conf joins/retail/repeat_purchase_training.py --force-overwrite -y
    ```

    Compilation checks the definitions and writes JSON under `chronon/production/`.
    Actually running them then happens through `run.py`, wired up to the configured
    Spark, warehouse, job scheduler, stream, and online store. The full platform is
    deliberately not tucked away inside a notebook kernel.
    ''',
)

add_markdown(
    "real-chronon-intro",
    r'''
    ---
    ## §8. Run a small real Chronon backfill

    ![The local notebook version and production Chronon have different roles](assets/08-local-vs-production.svg)

    Everything up to now has been our own code. Now we hand the very same definitions
    to Chronon's real engine and see whether it reaches the same answers we did.

    That engine runs on **Spark**, a system for crunching through large tables by
    spreading the work across many machines. Spark itself is written in Java, and
    `PySpark` is simply the Python doorway into it, which is why this one section needs
    a Java runtime while the rest of the notebook does not. Chronon 0.0.114 ships a
    small set of Jupyter helpers on top of PySpark. If your machine passes the checks in
    the next cell, this section really does run a point-in-time join through Chronon's
    Spark engine, which is completely different code from the NumPy version we have been
    leaning on.

    We keep the input deliberately tiny, for three reasons:

    - a handful of rows makes the cutoff easy to verify by hand and the job quick to
      run;
    - the full UCI data already gives the readable reference builder a real workout;
    - a genuine production backfill would depend on your own warehouse, catalog,
      scheduler, stream, and online store anyway.

    Java and Chronon are both pinned to specific versions, and Chronon also needs its
    matching Spark assembly JAR, which is the bundle of compiled Java code that holds
    the engine. The setup cell downloads that file into `data/cache/` and checks its
    fingerprint (its hash) so you can be sure you got the exact build expected. If
    anything is missing, the section quietly skips itself and tells you why, instead of
    printing something that merely looks like a real Chronon result.
    ''',
)

# The exact Spark-backed cells are inserted below after the local semantic cells.
# They are kept as separate cells so a learner can skip the optional runtime while
# continuing the rest of the notebook.

add_markdown(
    "spark-placeholder",
    r'''
    > **Runtime note:** this section holds the full, version-pinned Chronon code. If
    > your environment blocks the download or cannot start a local Spark process, just
    > skip ahead to §9. Everything about writing and compiling Chronon definitions
    > still works fine without it.
    ''',
)

add_code(
    "chronon-runtime-preflight",
    r'''
    import hashlib
    import tempfile

    RUN_CHRONON_SPARK = os.getenv("RUN_CHRONON_SPARK", "0") == "1"
    CHRONON_JAR_URL = (
        "https://repo1.maven.org/maven2/ai/chronon/"
        "spark_uber_2.12/0.0.114/"
        "spark_uber_2.12-0.0.114-assembly.jar"
    )
    CHRONON_JAR_SHA256 = (
        "b135ba4283f4368c075fd9ae839062f64ac3e3669141e6f7221be4d5efe6d705"
    )
    CHRONON_JAR = CACHE_PATH.parent / "spark_uber_2.12-0.0.114-assembly.jar"


    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()


    def prepare_chronon_jar() -> Path:
        if not CHRONON_JAR.exists():
            download_if_missing(CHRONON_JAR_URL, CHRONON_JAR)
        actual = sha256_file(CHRONON_JAR)
        if actual != CHRONON_JAR_SHA256:
            raise RuntimeError(
                f"Chronon JAR checksum mismatch: expected {CHRONON_JAR_SHA256}, "
                f"received {actual}"
            )
        return CHRONON_JAR


    if RUN_CHRONON_SPARK and sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "The Chronon Spark lab requires Python 3.10 because Chronon 0.0.114 "
            "pins PySpark 3.3.1. Run `uv sync`, restart Jupyter with "
            "`uv run jupyter lab`, and select the project kernel. "
            f"The active kernel is Python {sys.version.split()[0]}."
        )
    if RUN_CHRONON_SPARK:
        prepare_chronon_jar()
        print("Chronon Spark lab enabled. Pinned JAR checksum passed.")
    else:
        print("Chronon Spark lab skipped. Set RUN_CHRONON_SPARK=1 in .env to run it.")
    ''',
)

add_markdown(
    "chronon-runtime-details",
    r'''
    ### What the executable cell does

    When the lab is enabled, the cell below:

    1. points PySpark at the Java runtime that `uv` installed, before importing
       PySpark;
    2. starts Spark on your machine with the pinned Chronon JAR attached;
    3. creates tables split into daily partitions, because Chronon reads those
       partitions to decide what to compute;
    4. writes two earlier purchases, one purchase at the prediction time, one
       cancellation, and one prediction row;
    5. imports the same Chronon definitions you read in §7;
    6. patches one class lookup that is broken in this release;
    7. runs the join through Chronon's Java engine;
    8. asserts that the same-time £999 order is left out, that prior 30-day spend is
       £30, and that prior cancellation value is £7.

    Step 6 deserves a word, because workarounds like it are a normal part of using a
    young library. Chronon 0.0.114's Python code looks for a class at
    `ai.chronon.spark.TableUtils`, while the shipped JAR actually keeps it at
    `ai.chronon.spark.catalog.TableUtils`. Our patch fixes that one lookup and nothing
    else, and it should be deleted the moment a release corrects the mismatch. Notice
    where it lives: right next to a pinned version and a test that will fail loudly if
    the assumption ever changes.

    Each run gets its own temporary warehouse and in-memory Hive metastore (the
    bookkeeping database that tracks which tables exist). That isolation matters inside
    a notebook: stopping Spark does not always stop the underlying Java process, so a
    shared metastore could hang on to table and partition information from an earlier
    run even after its temporary files are gone. The cell also drops its tutorial
    databases during cleanup, which is what makes **Run All** and manual reruns behave
    identically.

    The dates use the `ds=YYYYMMDD` format because this release's Jupyter helpers read
    dates with the pattern `%Y%m%d`. Production configs in `teams.json` commonly use
    `yyyy-MM-dd` instead. Pick one format per runtime and stick to it, because mixing
    them makes jobs silently hunt for partitions that do not exist.
    ''',
)

add_code(
    "chronon-real-backfill",
    r'''
    %%capture chronon_verbose
    chronon_result_pd = None

    if RUN_CHRONON_SPARK:
        import datetime as dt
        import jdk4py

        java_home = Path(jdk4py.JAVA_HOME)
        os.environ["JAVA_HOME"] = str(java_home)
        os.environ["PATH"] = (
            f"{java_home / 'bin'}{os.pathsep}{os.environ['PATH']}"
        )
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

        from pyspark.sql import SparkSession
        from pyspark.sql.types import (
            DoubleType,
            LongType,
            StringType,
            StructField,
            StructType,
        )
        from ai.chronon.pyspark.jupyter import JupyterJoin, JupyterPlatform

        runtime = tempfile.TemporaryDirectory(
            prefix="chronon_runtime_", dir=str(CACHE_PATH.parent)
        )
        runtime_dir = Path(runtime.name)
        metastore_name = runtime_dir.name.replace("-", "_")
        spark = None
        original_get_table_utils = JupyterPlatform.get_table_utils

        try:
            active_spark = SparkSession.getActiveSession()
            if active_spark is not None:
                active_jsc = active_spark.sparkContext._jsc
                if active_jsc is not None and not active_jsc.sc().isStopped():
                    raise RuntimeError(
                        "Another live Spark session is attached to this kernel. "
                        "Stop it or restart the kernel before running the isolated "
                        "Chronon lab."
                    )

            spark = (
                SparkSession.builder
                .master("local[2]")
                .appName("chronon-feature-store-tutorial")
                .config("spark.jars", str(CHRONON_JAR))
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.session.timeZone", "UTC")
                .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
                .config("spark.sql.catalogImplementation", "hive")
                .config(
                    "spark.hadoop.javax.jdo.option.ConnectionURL",
                    f"jdbc:derby:memory:{metastore_name};create=true",
                )
                .config(
                    "spark.hadoop.javax.jdo.option.ConnectionDriverName",
                    "org.apache.derby.jdbc.EmbeddedDriver",
                )
                .config("spark.sql.warehouse.dir", str(runtime_dir / "warehouse"))
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.default.parallelism", "2")
                .config("spark.chronon.outputParallelismOverride", "2")
                .config("spark.chronon.group_by.parallelism", "2")
                .config("spark.chronon.partition.column", "ds")
                .config("spark.chronon.partition.format", "yyyyMMdd")
                .enableHiveSupport()
                .getOrCreate()
            )
            spark.sparkContext.setLogLevel("ERROR")

            # Both namespaces are private to the per-run in-memory metastore.
            # Dropping them is defensive if a prior failure initialized only part
            # of the catalog.
            for namespace in ("feature_store_tutorial", "retail"):
                spark.sql(f"DROP DATABASE IF EXISTS `{namespace}` CASCADE")

            def epoch_ms(value: str) -> int:
                parsed = dt.datetime.fromisoformat(value).replace(
                    tzinfo=dt.timezone.utc
                )
                return int(parsed.timestamp() * 1_000)

            orders_schema = StructType([
                StructField("customer_id", StringType(), False),
                StructField("order_id", StringType(), False),
                StructField("order_value", DoubleType(), False),
                StructField("item_count", LongType(), False),
                StructField("ts", LongType(), False),
                StructField("ds", StringType(), False),
            ])
            cancellations_schema = StructType([
                StructField("customer_id", StringType(), False),
                StructField("cancellation_id", StringType(), False),
                StructField("refund_value", DoubleType(), False),
                StructField("ts", LongType(), False),
                StructField("ds", StringType(), False),
            ])
            predictions_schema = StructType([
                StructField("customer_id", StringType(), False),
                StructField("order_id", StringType(), False),
                StructField("current_order_value", DoubleType(), False),
                StructField("repeat_purchase_30d", LongType(), False),
                StructField("ts", LongType(), False),
                StructField("ds", StringType(), False),
            ])

            fixture_tables = [
                (
                    "retail.orders",
                    [
                        ("C1", "O1", 10.0, 1, epoch_ms("2025-01-01 10:00"), "20250101"),
                        ("C1", "O2", 20.0, 1, epoch_ms("2025-01-05 10:00"), "20250105"),
                        ("C1", "O3", 999.0, 1, epoch_ms("2025-01-10 10:00"), "20250110"),
                    ],
                    orders_schema,
                ),
                (
                    "retail.cancellations",
                    [("C1", "X1", 7.0, epoch_ms("2025-01-07 10:00"), "20250107")],
                    cancellations_schema,
                ),
                (
                    "retail.prediction_events",
                    [("C1", "O3", 999.0, 1, epoch_ms("2025-01-10 10:00"), "20250110")],
                    predictions_schema,
                ),
            ]

            spark.sql("CREATE DATABASE IF NOT EXISTS retail")
            spark.sql("CREATE DATABASE IF NOT EXISTS feature_store_tutorial")
            fixture_counts = {}
            for table_name, rows, schema in fixture_tables:
                (
                    spark.createDataFrame(rows, schema)
                    .write.mode("overwrite")
                    .partitionBy("ds")
                    .saveAsTable(table_name)
                )
                fixture_counts[table_name] = spark.table(table_name).count()
                if fixture_counts[table_name] != len(rows):
                    raise RuntimeError(
                        f"Fixture table {table_name} should contain {len(rows)} "
                        f"rows, but Spark read {fixture_counts[table_name]}."
                    )

            training = copy.deepcopy(training_join_v1)
            training.metaData.name = "retail.repeat_purchase_training.v1"
            training.metaData.team = "retail"

            JupyterPlatform.get_table_utils = lambda self: (
                self.jvm.ai.chronon.spark.catalog.TableUtils(
                    self.java_spark_session
                )
            )

            chronon_result = JupyterJoin(
                training,
                spark,
                output_namespace="feature_store_tutorial",
                use_username_prefix=False,
            ).run(
                start_date="20250110",
                end_date="20250110",
                step_days=1,
            )
            chronon_result_pd = chronon_result.toPandas()
            if len(chronon_result_pd) != 1:
                raise RuntimeError(
                    "Chronon should return exactly one row for the single "
                    "prediction fixture, but returned "
                    f"{len(chronon_result_pd)}. Input table counts were "
                    f"{fixture_counts}. Restart the kernel if Spark emitted an "
                    "earlier JVM error."
                )
        finally:
            if spark is not None:
                for namespace in ("feature_store_tutorial", "retail"):
                    try:
                        spark.sql(f"DROP DATABASE IF EXISTS `{namespace}` CASCADE")
                    except Exception:
                        # Preserve the original exception. The unique metastore and
                        # temporary warehouse still isolate the next execution.
                        pass
                spark.stop()
            JupyterPlatform.get_table_utils = original_get_table_utils
            runtime.cleanup()

        purchase_30d = (
            "purchase_retail_purchases_v1_order_value_sum_30d"
        )
        cancel_30d = (
            "cancel_retail_cancellations_v1_refund_value_sum_30d"
        )
        result_row = chronon_result_pd.iloc[0]
        assert result_row["customer_id"] == "C1"
        assert result_row["order_id"] == "O3"
        assert result_row[purchase_30d] == 30.0
        assert result_row[cancel_30d] == 7.0
        assert result_row["current_order_value"] == 999.0
    ''',
)

add_code(
    "chronon-real-result",
    r'''
    if chronon_result_pd is None:
        print("No Chronon Spark result. The optional lab is disabled.")
    else:
        display(
            chronon_result_pd[
                ["customer_id", "current_order_value", purchase_30d, cancel_30d]
            ]
        )
        known_warning = "unexpected error occurred during validation" in (
            chronon_verbose.stdout + chronon_verbose.stderr
        ).lower()
        print("Real Chronon JupyterJoin boundary assertions passed.")
        print("Known 0.0.114 validation warning observed:", known_warning)
        print("Verbose Spark plan and progress output are available in chronon_verbose.")
    ''',
)

add_markdown(
    "chronon-runtime-cautions",
    r'''
    The run above proves one thing, and proves it well: the local Chronon engine
    respects the time cutoff. What it does *not* do is set up production serving.
    Marking a definition `online=True` only *declares* that it is meant to be served. A
    team still has to supply the database that answers lookups by key, the code that
    decodes the event stream, an implementation of Chronon's online API, a client for
    the model to call, a scheduler, access controls, and monitoring.

    In version 0.0.114 you may also spot a harmless Java warning about a null value just
    before the job finishes. That warning is exactly why "the job returned a table" is
    not a test of correctness. A job can finish successfully and still hand back wrong
    numbers. The £30 and £7 assertions are what actually decide whether this fixture
    passed.
    ''',
)

add_markdown(
    "model-intro",
    r'''
    ---
    ## §9. Train in time order, then make leakage obvious

    ![A latest-value join leaks future information while an as-of join does not](assets/05-wrong-vs-right-join.svg)

    We have honest features and tests that prove the boundary holds. Now we get to do
    what the whole notebook has been building toward: show what leakage looks like from
    the outside, so you can recognize it even when nobody has kindly labeled it for you.

    First, the split. We train only on the 15,822 rows whose 30-day label window has
    fully closed. The earliest 75 percent of them become the training set and the most
    recent 25 percent become the test set, which puts the dividing line around 8
    September 2011 and gives roughly 11,866 training rows and 3,956 test rows. Splitting
    by *time* instead of at random is a deliberate choice. A random split would let the
    model peek at September behavior while training and then be quizzed on March
    behavior, and no deployed model has ever enjoyed that luxury. Testing on the most
    recent slice of time is the closest thing we have to a dress rehearsal for the real
    future.

    Then two models, built with exactly the same setup, differing by a single column:

    - the **causal** model uses only what was known at the decision: the current
      order's own fields plus the past-only features. "Causal" here simply means the
      inputs came *before* the outcome, in the ordinary cause-and-effect sense;
    - the **forbidden future** model adds `leak_future_order_count_30d`, the count of
      future orders that we used to build the label in the first place. It is cheating,
      on purpose.

    Both models are scored with two numbers, and it helps to know what they mean before
    you read them. **ROC AUC** is the chance that the model gives a higher score to a
    randomly picked positive case than to a randomly picked negative one. So 0.500 is a
    coin flip and 1.000 is perfect ranking. **Average precision** summarizes how well
    the model does when you look at its most confident predictions first, and it starts
    from the base positive rate, which here is close to 0.456. Both run from 0 to 1, and
    higher is better.

    Expect the causal model to land clearly above a coin flip but nowhere near perfect,
    which is what an honest, real-world problem usually looks like. Expect the second
    model to come out near 1.000. And that is the whole lesson: a near-perfect score is
    a red flag, not a trophy, because the value driving it could not possibly have been
    known when the prediction was made. Leakage this blatant is rare in real life, since
    nobody ships a column literally named `leak_future_order_count_30d`. But seeing the
    pattern in an obvious case is how you train yourself to spot it later, hidden inside
    a feature with a perfectly innocent-sounding name.

    A few notes on the pipeline, in case you are curious. **Logistic regression** fits
    one weight per input column and turns the weighted total into a probability between
    0 and 1. It is a good pick here precisely because it is simple: one overwhelmingly
    predictive column shows up as an obvious, glaring result. Missing history is filled
    in with zero, and the columns are rescaled so that a feature measured in thousands
    of pounds does not automatically dwarf a feature measured in single orders. `C=0.1`
    turns on regularization, which nudges the fit toward smaller weights so it does not
    overreact to any one column. `class_weight="balanced"` stops the larger group from
    drowning out the smaller one, and `liblinear` is a dependable solver at this data
    size. These settings are here to keep the comparison clean and readable, not to
    squeeze out the highest possible score.
    ''',
)

add_code(
    "model-code",
    r'''
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    CAUSAL_FEATURES = [
        "current_order_value",
        "item_count",
        "unique_products",
        "purchase_order_value_sum_7d",
        "purchase_order_value_sum_30d",
        "purchase_order_value_sum_90d",
        "purchase_order_id_count_7d",
        "purchase_order_id_count_30d",
        "purchase_order_id_count_90d",
        "purchase_order_value_average_30d",
        "purchase_order_value_average_90d",
        "purchase_item_count_sum_30d",
        "days_since_last_order",
        "cancel_refund_value_sum_30d",
        "cancel_refund_value_sum_90d",
        "cancel_cancellation_id_count_30d",
        "cancel_cancellation_id_count_90d",
        "cancel_value_ratio_90d",
    ]
    TARGET = "repeat_purchase_30d"

    model_frame = features.loc[features["label_is_observed"]].copy()
    split_time = model_frame["prediction_ts"].quantile(0.75)
    train_mask = model_frame["prediction_ts"] < split_time
    test_mask = ~train_mask


    def evaluate_feature_list(columns: list[str]) -> dict[str, float]:
        pipeline = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0),
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                max_iter=2_000,
                class_weight="balanced",
                random_state=42,
                solver="liblinear",
            ),
        )
        # The optimized matrix libraries can raise floating-point warnings here.
        # Silence them only inside this block; the finite check below is the guard.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            pipeline.fit(
                model_frame.loc[train_mask, columns],
                model_frame.loc[train_mask, TARGET],
            )
            probability = pipeline.predict_proba(
                model_frame.loc[test_mask, columns]
            )[:, 1]
        if not np.isfinite(probability).all():
            raise FloatingPointError("Model produced a non-finite probability")
        truth = model_frame.loc[test_mask, TARGET]
        return {
            "roc_auc": roc_auc_score(truth, probability),
            "average_precision": average_precision_score(truth, probability),
        }


    model_results = pd.DataFrame(
        {
            "causal_point_in_time": evaluate_feature_list(CAUSAL_FEATURES),
            "forbidden_future_feature": evaluate_feature_list(
                CAUSAL_FEATURES + ["leak_future_order_count_30d"]
            ),
        }
    ).T
    display(
        pd.Series(
            {
                "split_time": split_time,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "test_positive_rate": model_frame.loc[test_mask, TARGET].mean(),
            },
            name="value",
        ).to_frame()
    )
    display(model_results.style.format("{:.3f}"))
    ''',
)

add_markdown(
    "leakage-taxonomy",
    r'''
    ### Correct timing is necessary but not sufficient

    Our leaky model broke two rules at once: it used the answer, and it used the
    future. But fixing the join does not fix any of the failures listed below, and
    every one of them can slip right past a perfectly correct point-in-time lookup.

    | Failure | Example | How to guard against it |
    |---|---|---|
    | Window leakage | the current order lands inside "prior spend" | test the boundary explicitly |
    | Availability leakage | a late-arriving row shows up in rebuilt history | keep arrival time in the source |
    | Revision leakage | a corrected row replaces what production really saw | keep versioned snapshots or a change log |
    | Global-statistics leakage | the rescaling step measures its averages using test rows too | compute every preparation step from training rows only |
    | Cross-entity leakage | a merchant-level average includes this order's own outcome | audit label cutoffs inside aggregations |
    | Label censoring | recent rows with unfinished labels are called negatives | wait out the label window |

    Read that table as a division of labor. A feature store automates the timing
    mechanics, and that is real, valuable work. But deciding what the source data
    means, when the decision happens, where the boundary sits, and how the label is
    built stays with your team. No tool will ever take those judgment calls over for
    you.
    ''',
)

add_markdown(
    "online-intro",
    r'''
    ---
    ## §10. Look up current values and check they agree

    ![Offline backfill and online fetches must be compared](assets/06-offline-online-parity.svg)

    Training and serving want opposite things from storage. The offline store keeps
    the full history and is built for large scans, because training reads millions of
    rows at once. The online store keeps only the latest value for each key and is
    built to answer in milliseconds, because a real customer is waiting on the other
    end.

    Writing one definition for both paths is a great start, but two separate pieces of
    machinery do not magically become identical just because somebody wrote the rule
    down once. They can read different data feeds, fill in different default values, or
    run different versions of the code. So you check by comparing the numbers they each
    actually produce, which is exactly what we do below.

    The class below is called `TeachingFeatureCache` on purpose. It is just a Python
    dictionary dressed up so that lookups, timestamps, defaults, and staleness are all
    easy to inspect. It is **not** Chronon's online store. Real Chronon serving needs a
    proper implementation of its online API, backed by an actual database.

    The next three cells walk through the exercise in order. First we define
    `vector_at`, which computes one customer's complete feature vector as of any moment
    you name, reusing the same window helper from §5. Then we pick a serving moment of 1
    November 2011 at noon, compute vectors for the 100 busiest customers, and store them
    in the cache along with a note of when each one was computed. Recording that
    timestamp is the crucial habit: a value with no computed-at time can never be
    checked for staleness later. Finally we fetch one customer back out, 20 seconds
    after filling the cache, using a rule that anything older than a minute counts as
    stale. You should see `Found: True Fresh: True`, since 20 seconds is comfortably
    inside the one-minute limit.
    ''',
)

add_code(
    "vector-at-code",
    r'''
    def vector_at(
        customer_id: str,
        prediction_time: pd.Timestamp,
        order_events: pd.DataFrame = orders,
        cancellation_events: pd.DataFrame = cancellations,
    ) -> dict[str, float]:
        anchor = np.array([prediction_time.value], dtype=np.int64)
        order_history = order_events.loc[
            order_events["customer_id"].eq(customer_id)
        ].sort_values("prediction_ts")
        order_times = order_history["prediction_ts"].astype("int64").to_numpy()
        order_values = order_history["order_value"].to_numpy(float)
        item_values = order_history["item_count"].to_numpy(float)

        result: dict[str, float] = {}
        for days in (7, 30, 90):
            counts, sums = window_stats(
                order_times, order_values, anchor, days
            )
            result[f"purchase_order_value_sum_{days}d"] = float(sums[0])
            result[f"purchase_order_id_count_{days}d"] = int(counts[0])
            if days in (30, 90):
                result[f"purchase_order_value_average_{days}d"] = (
                    float(sums[0] / counts[0]) if counts[0] else np.nan
                )

        _, item_sums = window_stats(order_times, item_values, anchor, 30)
        result["purchase_item_count_sum_30d"] = float(item_sums[0])
        prior_position = np.searchsorted(
            order_times, anchor[0], side="left"
        ) - 1
        result["days_since_last_order"] = (
            float((anchor[0] - order_times[prior_position]) / DAY_NS)
            if prior_position >= 0
            else np.nan
        )

        cancel_history = cancellation_events.loc[
            cancellation_events["customer_id"].eq(customer_id)
        ].sort_values("event_ts")
        cancel_times = cancel_history["event_ts"].astype("int64").to_numpy()
        cancel_values = cancel_history["refund_value"].to_numpy(float)
        for days in (30, 90):
            counts, sums = window_stats(
                cancel_times, cancel_values, anchor, days
            )
            result[f"cancel_refund_value_sum_{days}d"] = float(sums[0])
            result[f"cancel_cancellation_id_count_{days}d"] = int(counts[0])
        result["cancel_value_ratio_90d"] = (
            result["cancel_refund_value_sum_90d"]
            / (result["purchase_order_value_sum_90d"] + 1.0)
        )
        return result
    ''',
)

add_code(
    "teaching-cache",
    r'''
    class TeachingFeatureCache:
        """A transparent cache for teaching, not a production online store."""

        def __init__(self) -> None:
            self.rows: dict[str, dict] = {}

        def upsert(
            self,
            entity_key: str,
            values: dict[str, float],
            computed_at: pd.Timestamp,
        ) -> None:
            self.rows[entity_key] = {
                "values": copy.deepcopy(values),
                "computed_at": computed_at,
            }

        def fetch(
            self,
            entity_key: str,
            request_time: pd.Timestamp,
            max_compute_age: pd.Timedelta,
        ) -> dict:
            row = self.rows.get(entity_key)
            if row is None:
                return {"found": False, "values": None, "fresh": False}
            compute_age = request_time - row["computed_at"]
            return {
                "found": True,
                "values": copy.deepcopy(row["values"]),
                "computed_at": row["computed_at"],
                "compute_age": compute_age,
                "fresh": compute_age <= max_compute_age,
            }


    serve_time = pd.Timestamp("2011-11-01 12:00:00")
    active_customers = (
        orders.loc[orders["prediction_ts"] < serve_time, "customer_id"]
        .value_counts()
        .head(100)
        .index
    )
    cache = TeachingFeatureCache()
    for customer_id in active_customers:
        cache.upsert(
            customer_id,
            vector_at(customer_id, serve_time),
            computed_at=serve_time,
        )

    example_customer = active_customers[0]
    fetched = cache.fetch(
        example_customer,
        request_time=serve_time + pd.Timedelta(seconds=20),
        max_compute_age=pd.Timedelta(minutes=1),
    )
    print("Customer:", example_customer)
    print("Found:", fetched["found"], "Fresh:", fetched["fresh"])
    display(pd.Series(fetched["values"], name="online_value").to_frame().head(8))
    ''',
)

add_markdown(
    "parity-explanation",
    r'''
    ### Compare the same key at the same moment

    A comparison between the two paths only means something when five things line up:

    1. the same version of the feature definition,
    2. the same entity key,
    3. the same time cutoff,
    4. the same rules for defaults and nulls,
    5. an agreed-upon numeric tolerance, since computer arithmetic is not perfectly
       exact.

    Miss even one of these and any difference you find tells you nothing. Comparing
    today's online value against a value backfilled for last week is the classic
    blunder: different cutoffs are *supposed* to give different numbers, so that
    comparison can only ever confuse you.

    Point 5 needs a short explanation. Computers store decimals using a fixed number of
    bits, so adding the same numbers in a different order can leave you off by a tiny
    amount, something like 0.0000000001. A test that demanded exact equality would flag
    those as failures and bury the real problems under noise. So we allow a tolerance of
    `1e-9` and treat any gap smaller than that as agreement.

    Below we recompute one customer's features offline at the exact moment the cache was
    filled, confirm every value matches, and then deliberately add £25 to one cached
    number to prove the checker actually notices. Expect `Unmodified mismatches: 0`,
    followed by a one-row table showing the number we sabotaged. That second step
    matters more than it looks: a test that has never once failed tells you nothing
    about whether it even *can* fail.
    ''',
)

add_code(
    "parity-code",
    r'''
    def compare_vectors(
        expected: dict[str, float],
        actual: dict[str, float],
        tolerance: float = 1e-9,
    ) -> pd.DataFrame:
        rows = []
        for name in sorted(expected):
            left, right = expected[name], actual.get(name, np.nan)
            both_missing = pd.isna(left) and pd.isna(right)
            absolute_error = (
                0.0 if both_missing else abs(float(left) - float(right))
            )
            rows.append(
                {
                    "feature": name,
                    "offline": left,
                    "online": right,
                    "absolute_error": absolute_error,
                    "matches": both_missing or absolute_error <= tolerance,
                }
            )
        return pd.DataFrame(rows)


    offline_vector = vector_at(example_customer, serve_time)
    parity = compare_vectors(offline_vector, fetched["values"])
    assert parity["matches"].all()
    print("Unmodified mismatches:", int((~parity["matches"]).sum()))

    corrupted = copy.deepcopy(fetched["values"])
    corrupted["purchase_order_value_sum_30d"] += 25.0
    detected = compare_vectors(offline_vector, corrupted)
    display(detected.loc[~detected["matches"]])
    assert (~detected["matches"]).sum() == 1
    ''',
)

add_markdown(
    "freshness-note",
    r'''
    ### Fast is not the same as fresh

    A dictionary lookup can return in a few millionths of a second and *still* hand
    back a value that was computed a week ago. Speed and freshness are two different
    things, and confusing them is how teams end up serving stale numbers very quickly.
    It helps to give each piece of the delay its own name:

    ```text
    source lag       = arrival time - event time
    compute lag      = materialized time - arrival time
    end-to-end lag   = readable time - event time
    feature age      = decision time - newest event inside the value
    lookup latency   = response time - request time
    ```

    So a target for a feature has to cover both freshness and serving speed. Teams
    write this down as a service level objective, usually shortened to SLO, which is
    just an agreed-upon promise about how the feature will behave. These promises are
    written about the *slow* cases rather than the typical ones, because the typical
    case is not what breaks things. That is why `p99` below means the 99th percentile:
    the number that all but the slowest one percent of requests stay under. For example:

    ```text
    customer_order_count_7d
      99% readable within 60 seconds
      p99 fetch below 20 milliseconds
      null rate below 0.1%
      parity mismatch below 0.01%
    ```

    Neither extreme works on its own. Precomputing everything costs money and invites
    stale values, because something has to keep refreshing numbers nobody even asked
    for. Computing everything at request time adds delay and leaves you depending on
    every data source being up at that exact second. So most real systems do both: they
    precompute the heavy historical parts ahead of time, then combine those with
    whatever the request itself already knows.
    ''',
)

add_markdown(
    "observability-intro",
    r'''
    ---
    ## §11. Contracts, quality checks, and monitoring

    ![Feature observability covers the path from data quality to model impact](assets/09-observability-stack.svg)

    A dashboard full of green checkmarks tells you exactly one thing: those jobs ran to
    the end without crashing. It does *not* tell you that a feature means what its name
    claims, that it arrived on time, that it exists for the customers who need it, that
    its values stayed in a sensible range, that it was served fast enough, or that it
    helped the model at all. Each of those is a separate question needing a separate
    test, and that is precisely how an all-green dashboard ends up sitting cheerfully on
    top of a quietly broken model.

    A minimum contract for a production feature writes down its name, owner, entity,
    type, unit, definition, source, time boundary, what happens when data arrives late,
    its freshness target, what null means, how sensitive the data is, its version, who
    uses it, and whether it is being retired. The registry in the next cell records
    three of our features exactly that way. Read across one row and notice how much of
    it is not code at all: the `time_rule` column reuses the bracket notation from §5,
    `null_policy` explains in plain words what a blank value means, and `owner` names
    the team to call when something goes wrong. None of that can be guessed from a
    column of numbers, which is exactly why a human has to write it down.
    ''',
)

add_code(
    "registry-code",
    r'''
    registry = pd.DataFrame(
        [
            {
                "name": "purchase_order_value_sum_30d",
                "entity": "customer_id",
                "dtype_unit": "float64 GBP",
                "time_rule": "[t-30d, t)",
                "null_policy": "zero means no eligible orders",
                "freshness_slo": "99% < 60 s",
                "owner": "retail-ml",
                "version": "v1",
                "consumers": "repeat-purchase-v1",
            },
            {
                "name": "days_since_last_order",
                "entity": "customer_id",
                "dtype_unit": "float64 days",
                "time_rule": "latest event < t",
                "null_policy": "null means no prior order",
                "freshness_slo": "99% < 60 s",
                "owner": "retail-ml",
                "version": "v1",
                "consumers": "repeat-purchase-v1",
            },
            {
                "name": "cancel_value_ratio_90d",
                "entity": "customer_id",
                "dtype_unit": "float64 ratio",
                "time_rule": "both inputs [t-90d, t)",
                "null_policy": "smoothed denominator +1 GBP",
                "freshness_slo": "99% < 5 min",
                "owner": "retail-ml",
                "version": "v1",
                "consumers": "repeat-purchase-v1",
            },
        ]
    )
    display(registry)
    ''',
)

add_markdown(
    "monitoring-explanation",
    r'''
    ### Watching a feature change over time

    Contracts say what a feature *should* be. Monitoring asks whether it still *is*. The
    usual way to check is to compare a recent stretch of data against an earlier stretch
    and look for values that drifted.

    That is what the next cell does. It splits the training rows at the 80th percentile
    of prediction time, so the earliest 80 percent become the baseline and the most
    recent 20 percent become the period we are inspecting, then compares three numbers
    across them:

    - the share of rows where `days_since_last_order` is null, which climbs when new
      customers arrive with no history to summarize yet;
    - the 99th percentile of 30-day spend, meaning the value that 99 percent of rows
      fall below. Watching the near-top rather than the average is deliberate, because
      the biggest values are usually the first to shift when the mix of customers
      changes;
    - the positive rate of the label, which tells you whether the very thing you are
      predicting has itself become more or less common.

    The last column reports the *relative* change between the two periods, so `+12%`
    means the recent number sits 12 percent above the baseline. Pay attention to both
    the direction and the size of the move, not just the fact that something moved.
    ''',
)

add_code(
    "monitoring-code",
    r'''
    recent_cutoff = model_frame["prediction_ts"].quantile(0.8)
    baseline = model_frame.loc[model_frame["prediction_ts"] < recent_cutoff]
    recent = model_frame.loc[model_frame["prediction_ts"] >= recent_cutoff]

    monitoring = pd.DataFrame(
        [
            {
                "metric": "null_rate.days_since_last_order",
                "baseline": baseline["days_since_last_order"].isna().mean(),
                "recent": recent["days_since_last_order"].isna().mean(),
                "interpretation": "cold start or missing history",
            },
            {
                "metric": "p99.purchase_order_value_sum_30d",
                "baseline": baseline["purchase_order_value_sum_30d"].quantile(0.99),
                "recent": recent["purchase_order_value_sum_30d"].quantile(0.99),
                "interpretation": "scale or outlier shift",
            },
            {
                "metric": "positive_rate.repeat_purchase_30d",
                "baseline": baseline[TARGET].mean(),
                "recent": recent[TARGET].mean(),
                "interpretation": "population or label shift",
            },
        ]
    )
    monitoring["relative_change"] = (
        (monitoring["recent"] - monitoring["baseline"])
        / monitoring["baseline"].replace(0, np.nan)
    )
    display(monitoring.style.format({
        "baseline": "{:.3f}", "recent": "{:.3f}", "relative_change": "{:+.1%}"
    }))
    ''',
)

add_markdown(
    "monitoring-cautions",
    r'''
    A number moving is not automatically a bug. Holiday shopping, a product launch, or
    a real change in who is buying will all shift a distribution, and this dataset ends
    in December for exactly that reason. What matters is that someone owns the signal
    and knows what to do when it moves. Useful things to watch, roughly in the order
    data flows through the system:

    - **source health:** missing days of data, changed columns, duplicated events;
    - **quality:** nulls, value ranges, units, how many distinct keys appear, and
      keys that appear far more often than the rest;
    - **freshness:** how far behind the events, their arrival, and the computation
      are running;
    - **parity:** sampled offline and online values, with their timestamps and an
      agreed tolerance;
    - **serving:** typical and worst-case response times, error rate, missing keys,
      and how often a fallback value gets used;
    - **lineage**, the record of what feeds what: which models, datasets, owners, and
      versions are affected when one thing breaks;
    - **impact:** prediction quality, the business metric, incidents, and cost.

    Two closing thoughts on how to judge a feature platform. First, a catalog entry
    with no owner is just documentation, not a working data product, because there is
    nobody to call when it breaks. Second, counting how many features exist is a vanity
    metric that looks impressive and means little. Better questions to ask are: how
    often do features cause incidents, how long does a backfill take to deliver, are the
    freshness targets actually being met, how much reuse has been genuinely validated
    rather than just assumed, how often do parity checks fail, and what does each
    training and serving workload cost?
    ''',
)

add_markdown(
    "production-map",
    r'''
    ---
    ## §12. From this notebook to production Chronon

    ```mermaid
    flowchart TB
        A[Warehouse order history] --> B[Chronon batch compute]
        C[Order event stream] --> D[Chronon stream compute]
        B --> E[Historical feature tables]
        B --> F[Online key-value store]
        D --> F
        G[Prediction spine] --> H[Chronon Join backfill]
        E --> H
        H --> I[Versioned training dataset]
        J[Model service] --> K[Chronon fetch client]
        K --> F
        K --> L[Fetch logs]
        L --> M[Offline-online comparison]
        E --> M
    ```

    Read the diagram as two loops that meet in the middle. Batch jobs build the history
    used for training and also seed the online store; the event stream keeps that
    online store current; and the join backfill turns the prediction spine into a
    training dataset. Meanwhile the model service fetches values through a client, those
    fetches get logged, and the logs flow back into the comparison that keeps both paths
    honest.

    Getting all the way there adds work this notebook should not pretend away:

    1. map each logical source to real tables, topics, schemas, and daily partitions;
    2. keep event IDs and event time, plus arrival time wherever you need it;
    3. review and compile the definitions, and give each one an owning team;
    4. check keys, timestamps, schema, volume, and lopsided keys before backfilling;
    5. run the `Join` backfill, verify the invariants, and record exactly what was
       produced;
    6. load the batch values into the online store and start the streaming updates;
    7. publish the join metadata and wire the fetch client into the model service;
    8. log a sample of live fetches, compare them against offline values, and alert
       when they diverge;
    9. release the model to a small share of traffic first, decide what happens when
       a value is missing or stale, then scale up;
    10. version, migrate, and retire features without quietly changing meanings.

    Backfills also have to cope with duplicates, deletions, late arrivals, corrections,
    partial failures, and reruns that must produce the exact same result rather than
    double-counting. One subtlety is worth holding onto: recomputing an old feature from
    today's corrected source can be *more* accurate about history and still differ from
    what the production model actually saw at the time. Both numbers are right, they
    just answer different questions, so you have to decide up front which one your
    training data is meant to contain.
    ''',
)

add_markdown(
    "when-to-use",
    r'''
    ![Use the smallest system that preserves feature truth](assets/10-when-to-use.svg)

    ### When a feature store is worth the cost

    Good candidates for a feature store are features that several models reuse, that
    are expensive to compute, that change as time passes, that are needed in both
    training and live serving, that need a named owner and a record of where they came
    from, that can be recomputed from history, and that are stable enough to be worth
    writing a contract for.

    Poor candidates are one-off notebook columns, raw documents and images, cheap
    transforms nobody else needs, and anything you can only know from the current
    request. Embeddings sit right on the fence. An embedding is a list of numbers that
    stands in for an item, arranged so that similar items end up with similar numbers.
    Storing one embedding per user and looking it up by user ID fits a feature store
    just fine. But *searching* for the users most similar to a given one belongs in a
    vector database, which is a different tool built for that different job.

    Two tempting reasons are not good enough on their own. "Our predictions happen live"
    is not enough, because a plain lookup by key might do the whole job. "We have several
    teams" is not enough either, if those teams do not actually share any feature
    meanings. And going the other way, a system that only ever runs in batch can still
    benefit from point-in-time joins, lineage, contracts, and repeatable backfills.

    The strongest signals that you really do need one are: repeated incidents where
    training and serving disagreed, the same feature defined in several places, bugs in
    point-in-time datasets, stale values reaching models, several models sharing the
    same entity history, strict requirements to prove where a number came from, and
    experiments that take far too long to reach production.
    ''',
)

add_markdown(
    "final-checklist",
    r'''
    ## Final design review

    Before a feature goes to production, work through these questions:

    1. **Decision:** at exactly what moment does the model act?
    2. **Keys:** which entity is this about, and does its key stay stable over time?
    3. **Clocks:** what are the event, arrival, and prediction times?
    4. **Boundary:** does the current event belong in the history or in the request?
    5. **Label:** when is it finished, and could it leak in through another entity?
    6. **Backfill:** can old values be reproduced from a known version of the source
       and the definition?
    7. **Serving:** what are the freshness, response-time, missing-key, and fallback
       rules?
    8. **Parity:** how do you compare the same key at the same moment across both
       paths?
    9. **Governance:** who owns the meaning, the runtime, access, retention, and
       deletion?
    10. **Lifecycle:** how do the feature's users move from v1 to v2?

    If a design cannot answer these, no amount of extra infrastructure will make its
    features trustworthy. The questions are the real work.
    ''',
)

add_markdown(
    "exercises",
    r'''
    ## Extensions for deeper practice

    1. Add `purchase_order_id_count_1d` and test that `1d <= 7d <= 30d <= 90d`
       holds for every row.
    2. Invent an arrival delay for every row in the dataset, then rebuild the features
       both ways, once by event time and once by what production could have seen, and
       count how many rows end up with different values.
    3. Write a `v2` cancellation feature that ignores administrative reversals, then
       plan how you would move users across and compare the two versions.
    4. Add the customer's country as a feature. Decide whether a daily snapshot is
       enough or you need the exact value as of the prediction, and explain what a
       country change should mean for older rows.
    5. Give one customer a very large share of the orders, then estimate how that
       lopsidedness affects batch computation and online traffic.
    6. Store a computed-at timestamp with each online vector, and implement what the
       caller should do when the value is too old to trust.
    7. Run the official Chronon Docker quickstart, swap its made-up sources for your
       normalized UCI tables, and check its results against `vector_at`.
    ''',
)

add_markdown(
    "references",
    r'''
    ## References and scope

    Primary Chronon references:

    - [What is Chronon?](https://chronon.ai/contents.html)
    - [Chronon GroupBy](https://chronon.ai/authoring_features/GroupBy.html)
    - [Chronon Join](https://chronon.ai/authoring_features/Join.html)
    - [Testing GroupBys and Joins](https://chronon.ai/test_deploy_serve/Test.html)
    - [Chronon GitHub repository and quickstart](https://github.com/airbnb/chronon)

    Feature store perspectives reviewed for the companion theory guide:

    - [Databricks complete guide](https://www.databricks.com/blog/what-feature-store-complete-guide-ml-feature-engineering)
    - [Featurestore.org](https://www.featurestore.org/)
    - [IBM overview](https://www.ibm.com/think/topics/feature-store)
    - [Chalk overview](https://chalk.ai/blog/what-is-a-feature-store)
    - [AWS SageMaker Feature Store](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html)

    Dataset:

    - Daqing Chen, [Online Retail](https://doi.org/10.24432/C5BW33), UCI Machine
      Learning Repository, CC BY 4.0.

    A note on those vendor links. They are genuinely useful for implementation
    details and for seeing where the industry agrees, but they are marketing as much
    as documentation. Treat any claim that one product is best, or that one
    architecture suits everyone, as something to verify for yourself.

    © mui-group
    ''',
)


nbf.write(nb, OUTPUT)
print(f"Wrote {OUTPUT} with {len(nb.cells)} cells")
