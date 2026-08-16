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
    "language_info": {"name": "python", "version": "3.12"},
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

    ### A real-data lesson in feature engineering, time, and storage

    ![One clear feature rule supports model training and live decisions](assets/01-feature-store-loop.svg)

    A feature is one input to a model, such as a customer's number of recent
    orders. A feature store is a shared system for preparing and delivering these
    inputs. Its job is simple to state: give a model the **right value**, for the
    **right thing**, at the **right time**.

    The hard part is being honest about what was known at that time. If a model
    learns from information that arrived later, its test score can look better
    than real life. This notebook makes that mistake easy to see and test.

    This notebook builds that mental model with
    [UCI Online Retail](https://doi.org/10.24432/C5BW33), a real transaction log
    containing 541,909 line items from a UK online retailer. At each completed
    order, we ask:

    > Will this customer place another completed order in the next 30 days?

    By the end, you will be able to:

    1. turn raw line items into purchase and cancellation events,
    2. define a prediction timestamp and a future label horizon,
    3. compute historical features with an explicit half-open time boundary,
    4. prove the boundary with tiny tests,
    5. author real Chronon `GroupBy` and `Join` definitions,
    6. run a small local Chronon Spark backfill when the runtime preflight passes,
    7. compare a causal model with an intentionally leaked model,
    8. simulate a transparent teaching cache for online lookup semantics,
    9. test offline/online parity, freshness, contracts, and monitoring, and
    10. map the lab to a production Chronon deployment.

    The local NumPy runner is a clear, small reference implementation. It is not
    a replacement for Chronon. It lets us inspect every time cutoff before using
    a larger system.

    The companion [`FEATURE_STORE_THEORY.md`](FEATURE_STORE_THEORY.md) explains
    each idea with examples, counterexamples, cautions, and the same diagrams.
    ''',
)

add_markdown(
    "contents",
    r'''
    ## Learning path

    - **§0** Mental model and vocabulary
    - **§1** Reproducible setup with `uv` and `.env`
    - **§2** Decision, entity, clocks, and feature contract
    - **§3** Load and audit the real UCI data
    - **§4** Normalize line items into event tables
    - **§5** Build point-in-time features and future labels
    - **§6** Prove boundaries, availability, and leakage behavior
    - **§7** Author Chronon `GroupBy` and `Join` definitions
    - **§8** Execute a small real Chronon backfill
    - **§9** Train with a chronological split and expose leakage
    - **§10** Explore online lookup and offline/online parity
    - **§11** Add contracts, quality checks, and observability
    - **§12** Map the lab to production and decide when to stop

    You do not need to memorize Chronon commands. Keep returning to one question:
    **Could the model really have known this value when it made the decision?**
    The code and tests make the answer checkable.
    ''',
)

add_markdown(
    "mental-model",
    r'''
    ---
    ## §0. Mental model: definition, materialization, delivery

    A useful feature-store definition is:

    > A governed feature definition plus the machinery that computes historical
    > values, delivers current values, records metadata and lineage, and checks
    > whether models receive what the definition promised.

    The phrase **single source of truth** should mean one governed meaning and
    lineage graph. It does not require one physical database. Historical training
    scans and low-latency keyed reads have different access patterns, so many
    systems use different materializations.

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

    **What a feature store usually does not replace:** the warehouse or lake,
    stream processor, orchestrator, model registry, vector database, or general
    data catalog. A modern feature platform integrates with those systems.

    **Counterexample:** a batch model with five cheap features, one owner, no
    shared definitions, and a reliable warehouse pipeline may not need another
    platform. Point-in-time SQL and versioned ETL can be enough.
    ''',
)

add_markdown(
    "glossary",
    r'''
    ### Working vocabulary

    | Term | Plain meaning |
    |---|---|
    | **Entity** | The key whose history is summarized, such as `customer_id`. |
    | **Event time** | When the source event happened. |
    | **Availability time** | When the feature pipeline could actually use it. |
    | **Prediction time** | When the model made, or would have made, a decision. |
    | **Point-in-time join** | A join that only exposes information eligible at each historical prediction time. |
    | **Backfill** | Recomputing historical feature values from source history. |
    | **Offline materialization** | History optimized for scans, joins, training, and batch scoring. |
    | **Online materialization** | Current values optimized for keyed low-latency lookup. |
    | **Freshness** | How recently the value reflects source events, not how fast a lookup returns. |
    | **Training-serving skew** | Any semantic or value difference between training and inference inputs. |
    | **Feature contract** | Meaning, keys, type, time rule, owner, version, SLO, null policy, and lineage. |

    A raw numeric field can be a model feature. “Derived and reusable” describes
    a strong feature-store candidate, not the definition of a feature.
    ''',
)

add_markdown(
    "setup-explanation",
    r'''
    ---
    ## §1. Reproducible setup

    Dependencies live in the external `pyproject.toml`. From this directory:

    ```bash
    cp .env.example .env
    uv sync
    uv run python _build_notebook.py
    uv run jupyter lab
    ```

    `uv sync` creates the project environment from the manifest and lock file.
    This project uses Python 3.12. The
    optional local Spark lab uses a Java 17 runtime supplied by the `jdk4py`
    package. It does not require a machine-wide JDK.

    This lesson needs no API key. Configuration is still loaded through
    `load_dotenv(find_dotenv())`, so any future secrets belong in `.env`, never in
    notebook source. The checked-in `.env.example` contains only safe settings.
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
        raise RuntimeError("Start Jupyter from the feature_engineering directory.")

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

    Our model acts when a completed order is recorded. Its task is to estimate
    whether the same customer will place another completed order within 30 days.

    Think of a prediction row as a frozen snapshot question: “At this exact
    moment, what could we honestly say about this customer?” We answer that
    question once for each completed order.

    | Design choice | This tutorial |
    |---|---|
    | Prediction unit | one completed order |
    | Entity key | `customer_id` |
    | Prediction time | `InvoiceDate` of the current completed order |
    | Request context | current order value, item count, product count, country |
    | Stored history | prior purchases and cancellations |
    | Feature boundary | `[prediction_time - window, prediction_time)` |
    | Label boundary | `(prediction_time, prediction_time + 30 days]` |
    | Positive label | at least one later completed order in that interval |

    The current order is legitimate request context because it is known at this
    decision. It must not silently enter a feature named “prior 30-day spend.”
    We keep request context on the left side of the training join and historical
    aggregates on the right.

    **Caution:** the public data records completed invoices, not an actual
    production checkout request. We are explicit about that approximation. A real
    deployment must define exactly when the decision fires and which current-order
    fields are already trustworthy at that instant.
    ''',
)

add_markdown(
    "three-clocks",
    r'''
    ![Event time, prediction time, and availability time](assets/02-three-clocks.svg)

    ### Three clocks, two historical truths

    An event can occur before a prediction but arrive after it. That creates two
    defensible reconstructions:

    1. **event truth:** all events that happened before the cutoff;
    2. **production-available truth:** only events the production system could use
       before the cutoff.

    Production-available truth gives the strongest training-serving fidelity.
    UCI Online Retail has event timestamps but no ingestion timestamps. Our main
    backfill therefore demonstrates event-time correctness. §6 uses a tiny explicit
    availability-time example so the missing clock is not forgotten.

    A point-in-time API reduces temporal leakage. It does not prevent target
    leakage, global-statistics leakage, late-data leakage, or a bad label cutoff.
    ''',
)

add_markdown(
    "contract-preview",
    r'''
    ![A feature needs clear rules, a source, and an owner](assets/07-feature-contract.svg)

    Before computing `customer_avg_order_value_30d`, a team should be able to
    answer: Does cancellation value count? Is the current order excluded? Which
    currency? What does null mean? How fresh must it be? Who is paged? Which models
    consume it? What version changed the rule?

    Two columns with the same friendly name can have different meanings. Reuse by
    name alone is unsafe.
    ''',
)

add_markdown(
    "data-provenance",
    r'''
    ---
    ## §3. Load and audit the real UCI data

    The [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
    contains transactions from 1 December 2010 through 9 December 2011. The company
    mainly sold gifts, and many customers were wholesalers. UCI licenses the data
    under CC BY 4.0 and gives DOI `10.24432/C5BW33`.

    The first run downloads the official CSV to the ignored `data/cache/` folder.
    Later runs reuse it. The function writes to a temporary `.part` file first, so
    an interrupted download does not look complete.
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
    ### Data quality is part of feature meaning

    We do not silently drop rows. First we measure why rows may be unusable for
    this entity-based task:

    - missing `CustomerID` means no stable customer key;
    - an invoice beginning with `C` indicates cancellation;
    - negative quantity also indicates a reversal-like record;
    - zero or negative prices do not fit the purchase-value definition;
    - extreme quantities are retained and monitored, not quietly winsorized.

    UCI's page reports no missing values, yet the landed file contains many missing
    customer IDs. The actual data wins over the catalog claim. A feature pipeline
    should fail, quarantine, or report according to a reviewed policy.
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
    ## §4. Normalize line items into event tables

    Raw storage granularity and model-event granularity are different. An invoice
    contains many product lines, while our prediction unit is one order. We build:

    - `orders`: one row per identified, positive, non-cancelled invoice;
    - `cancellations`: one row per identified cancellation invoice.

    Grouping by customer, invoice, timestamp, and country makes the aggregation
    deterministic. The original line items remain the audit source. In production,
    these normalized tables would be governed upstream data products with event
    IDs, ingestion times, partitions, and deduplication contracts.

    `TUTORIAL_CUSTOMER_LIMIT` can keep only the most active N customers for a fast
    classroom run. The default uses every identified customer.
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
    ## §5. Build point-in-time features and future labels

    ![Historical feature windows and the future label horizon](assets/03-point-in-time-window.svg)

    Each training row has a **spine**: an example ID, entity key, prediction time,
    request context, and eventually a label. Right-side feature views contribute
    only events whose timestamp is strictly less than the spine timestamp.

    We use half-open windows such as `[t - 30d, t)`. The strict right boundary
    excludes the current order and every other order with exactly the same source
    timestamp. This conservative rule is easy to test and matches a prediction
    made before historical aggregates consume the current event.

    The future label uses `(t, t + 30d]`. Orders at exactly `t` do not count as a
    repeat purchase. The endpoint 30 days later does count.

    The local algorithm below uses two simple tools:

    - `np.searchsorted` locates left and right time boundaries in sorted history;
    - a prefix sum turns each range sum into two array lookups.

    That makes the work proportional to examples and windows rather than a large
    row-by-row range join. Chronon uses its own scalable aggregation architecture.
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
    ### The reference builder, step by step

    For each customer, the function:

    1. takes sorted order timestamps as both history and prediction anchors;
    2. computes 7, 30, and 90-day prior counts and spend;
    3. computes prior averages only when at least one eligible order exists;
    4. derives recency from the most recent timestamp strictly before the anchor;
    5. counts future orders only to construct the label;
    6. looks up cancellation history on the same anchor timestamps;
    7. marks labels near the dataset end as censored, not negative.

    The column `leak_future_order_count_30d` is kept only for §9's failure
    demonstration. It must never enter a legitimate feature list.

    Null and zero have different meanings. A zero count means eligible history was
    checked and no events matched. A null prior average or recency means there was
    no eligible prior order from which to compute the value.
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
    ### Why label maturity matters

    An order on 8 December cannot receive a trustworthy 30-day repeat-purchase
    label from a dataset ending on 9 December. “No observed later order” does not
    mean “no later order.” We retain those rows for feature-serving exercises but
    exclude them from supervised training with `label_is_observed`.

    This is easy to miss because no point-in-time feature join can repair an
    immature label. A production label pipeline needs an explicit maturity delay,
    late-label policy, and versioned backfill.
    ''',
)

add_markdown(
    "proof-intro",
    r'''
    ---
    ## §6. Prove the time boundary before trusting scale

    A tiny synthetic **test fixture** is appropriate even though the tutorial data
    is real. The fixture isolates one invariant that is hard to see among 541,909
    source rows.

    Customer `C1` orders for £10 on 1 January, £20 at the prediction time on
    10 January, and £999 on 11 January. At 10 January:

    - prior 30-day spend must be £10;
    - the current £20 order must be excluded from history;
    - the future £999 order must be excluded from features;
    - the label should be positive because a later order occurs within 30 days.
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
    ### Availability-time correctness is stricter

    The next cell models an event that happened at 09:30, arrived at 10:10, and a
    prediction made at 10:00. Event-time truth sees it. Production-available truth
    does not.

    Chronon guarantees temporal point-in-time computation from the source timeline
    you define. If a historical source has been corrected or contains records that
    arrived late, exact production replay requires source design that preserves the
    relevant availability timeline. A feature store cannot reconstruct a clock the
    source never recorded.
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
    ## §7. Express the design in Chronon

    ![Chronon Source, GroupBy, and Join object model](assets/04-chronon-object-model.svg)

    Chronon's core authoring objects separate concerns:

    - `Source` describes warehouse and optional streaming inputs plus event time;
    - `GroupBy` defines entity keys, aggregations, windows, accuracy, and metadata;
    - `Join` defines the historical prediction spine and combines `GroupBy` parts;
    - the compiler turns Python definitions into Thrift JSON for execution.

    The checked-in definitions are ordinary Python modules:

    ```text
    chronon/
    ├── teams.json
    ├── group_bys/retail/purchases.py
    ├── group_bys/retail/cancellations.py
    └── joins/retail/repeat_purchase_training.py
    ```

    The logical tables and topics are integration contracts. The local pandas
    frames do not automatically become production Chronon sources. §8 creates
    temporary Spark tables only for a small executable lab.
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

    The essential source and one aggregation are:

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

    `table` supports historical computation. `topic` supplies incremental events
    for real-time accuracy. `time_column` defines the event timeline. `keys`
    defines the entity. Each `Aggregation` combines a value, operation, and window.
    `TEMPORAL` requests point-in-time behavior rather than midnight snapshots.
    `online=True` requests pipelines needed for online serving.

    Chronon's windowed operations need a millisecond timestamp. In our logical
    warehouse contract, `ts` is epoch milliseconds even though pandas keeps a
    readable datetime in this notebook.
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
    ### The `Join` defines the training timeline

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

    The left event source supplies `customer_id` and `ts` for every historical
    example. Each right part is evaluated as of that timestamp. Prefixes keep
    feature names unambiguous. Enabling consistency asks Chronon to support sampled
    comparisons between logged online fetches and offline recomputation.

    The local training table carries a mature label on the left. In a larger
    system, Chronon's `LabelPart` can manage labels that become available after the
    feature timestamp. Either way, the model code must separate the label column
    from the allowed feature list.

    **Versioning caution:** Chronon protects compiled online `GroupBy` definitions
    from casual mutation. Create `v2`, migrate consumers, compare, then deprecate
    `v1`. A renamed column with changed semantics is not a safe in-place edit.
    ''',
)

add_markdown(
    "compile-commands",
    r'''
    ### Compile as a normal Chronon project

    The notebook imports definitions for inspection. The production authoring
    workflow also compiles them from the Chronon root:

    ```bash
    cd chronon
    PYTHONPATH=. uv run --project .. compile.py \
      --conf group_bys/retail/purchases.py --force-overwrite -y
    PYTHONPATH=. uv run --project .. compile.py \
      --conf group_bys/retail/cancellations.py --force-overwrite -y
    PYTHONPATH=. uv run --project .. compile.py \
      --conf joins/retail/repeat_purchase_training.py --force-overwrite -y
    ```

    Compilation validates and writes JSON under `chronon/production/`. Execution
    then uses `run.py` with the configured Spark, warehouse, orchestration, stream,
    and online-store integrations. The full platform is intentionally not hidden
    inside a notebook kernel.
    ''',
)

add_markdown(
    "real-chronon-intro",
    r'''
    ---
    ## §8. Execute a small real Chronon backfill

    ![The local semantic runner and production Chronon have different roles](assets/08-local-vs-production.svg)

    Chronon 0.0.114 includes a Jupyter PySpark interface. The next cells run a tiny
    point-in-time join through Chronon's actual Spark engine when the runtime
    preflight succeeds. This is different from the NumPy reference runner.

    Why a tiny fixture here?

    - it isolates the cutoff rule and keeps execution fast;
    - the full UCI data already exercises our transparent reference builder;
    - a real production backfill depends on organization-specific warehouse,
      catalog, scheduler, stream, and online-store integrations.

    The lab pins Java and Chronon but still needs its matching Chronon assembly JAR.
    The setup cell downloads that versioned artifact to `data/cache/` if necessary.
    A preflight failure becomes an explained skip, not a fake “Chronon result.”
    ''',
)

# The exact Spark-backed cells are inserted below after the local semantic cells.
# They are kept as separate cells so a learner can skip the optional runtime while
# continuing the rest of the notebook.

add_markdown(
    "spark-placeholder",
    r'''
    > **Runtime note:** the generated tutorial includes the full, version-pinned
    > Chronon Jupyter execution code in this section. If a restricted environment
    > blocks the Maven download or local Spark process, continue to §9. The
    > Chronon configuration authoring and compilation sections remain fully usable.
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

    When enabled, the cell below:

    1. points PySpark at the UV-managed Java 17 runtime before importing PySpark;
    2. starts Spark locally with the pinned Chronon assembly JAR;
    3. creates partitioned managed tables because Chronon inspects partitions;
    4. writes two prior purchases, one same-time purchase, one cancellation, and
       one prediction row;
    5. imports the physical Chronon modules from §7;
    6. applies a narrowly scoped 0.0.114 compatibility shim for `TableUtils`;
    7. runs `JupyterJoin` through Chronon's JVM engine;
    8. asserts that same-time £999 is absent, prior 30-day spend is £30, and prior
       cancellation value is £7.

    Chronon 0.0.114's released Python bridge references `ai.chronon.spark.TableUtils`,
    while the assembly places the class under `ai.chronon.spark.catalog.TableUtils`.
    The shim changes that lookup only. It should be removed after upgrading to a
    release that resolves the mismatch. Version-specific workarounds belong beside
    a version pin and an executable assertion.

    The runtime uses `ds=YYYYMMDD` because the 0.0.114 Jupyter executables parse
    dates with `%Y%m%d`. This differs from the common `yyyy-MM-dd` production
    convention in `teams.json`; do not mix the two formats within one runtime.
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
        spark = None
        original_get_table_utils = JupyterPlatform.get_table_utils

        try:
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
                    "jdbc:derby:memory:chronon_tutorial;create=true",
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
            for table_name, rows, schema in fixture_tables:
                (
                    spark.createDataFrame(rows, schema)
                    .write.mode("overwrite")
                    .partitionBy("ds")
                    .saveAsTable(table_name)
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
        finally:
            if spark is not None:
                spark.stop()
            JupyterPlatform.get_table_utils = original_get_table_utils
            runtime.cleanup()

        purchase_30d = (
            "purchase_retail_purchases_v1_order_value_sum_30d"
        )
        cancel_30d = (
            "cancel_retail_cancellations_v1_refund_value_sum_30d"
        )
        assert chronon_result_pd.loc[0, purchase_30d] == 30.0
        assert chronon_result_pd.loc[0, cancel_30d] == 7.0
        assert chronon_result_pd.loc[0, "current_order_value"] == 999.0
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
    The executable path proves the local Chronon engine's point-in-time boundary.
    It does not provision production serving. `online=True` marks a definition for
    upload and serving workflows; teams must still integrate a key-value store,
    stream decoder, Chronon online API implementation, fetch client, scheduler,
    access controls, and monitoring.

    You may see a nonfatal JVM validation warning about a null value before the
    small backfill completes in version 0.0.114. Do not use “the job returned a
    frame” as the correctness test. The explicit £30 and £7 assertions are the
    acceptance criteria for this fixture.
    ''',
)

add_markdown(
    "model-intro",
    r'''
    ---
    ## §9. Train chronologically, then make leakage obvious

    ![A latest-value join leaks future information while an as-of join does not](assets/05-wrong-vs-right-join.svg)

    We use only rows whose 30-day label horizon is fully observed. The earliest
    75 percent of those rows train the model, and the latest 25 percent test it.
    A random split would mix later and earlier behavior and would not resemble a
    forward deployment.

    Two models use the same simple logistic-regression pipeline:

    - **causal:** request context plus point-in-time historical features;
    - **forbidden future:** the same columns plus the count of future orders used to
      create the target.

    The second model should look spectacular. That is the failure signal. A better
    offline metric is not evidence of a better feature when the value was unknown
    at prediction time.

    The pipeline fills missing history with zero, standardizes numeric scales, and
    fits a regularized binary logistic model. `class_weight="balanced"` prevents
    the more common class from dominating, while `C=0.1` applies deliberate
    regularization. `liblinear` is a dependable solver for this modest binary data
    set. These choices keep the comparison readable; they are not a tuning claim.
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
        # Keep any BLAS floating-point flags local to the optimized matrix
        # operations. The explicit finite-value check below remains the guardrail.
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
    ### Point-in-time correct is necessary, not sufficient

    The leaked model demonstrates target and temporal leakage together. Other
    failures can survive a correct as-of join:

    | Failure | Example | Guardrail |
    |---|---|---|
    | Window leakage | current order enters “prior spend” | strict boundary test |
    | Availability leakage | late record appears in reconstructed history | preserve arrival or knowledge time |
    | Revision leakage | corrected record replaces what production saw | versioned source snapshots or change log |
    | Global-statistics leakage | scaler fits on future test data | fit preprocessing on training only |
    | Cross-entity leakage | merchant rate uses current order's future label | label cutoff audit inside aggregation |
    | Label censoring | recent examples treated as negatives | maturity horizon |

    A feature store automates some temporal mechanics. Teams still own the meaning
    of the source, decision, boundary, and label.
    ''',
)

add_markdown(
    "online-intro",
    r'''
    ---
    ## §10. Explore current lookup and parity

    ![Offline backfill and online fetches must be compared](assets/06-offline-online-parity.svg)

    An offline store is optimized for complete history and large scans. An online
    store is optimized for current values and keyed low-latency reads. “One logical
    definition” does not make two independent execution paths magically identical.
    Values must be compared.

    The next code is deliberately named `TeachingFeatureCache`. It is a Python
    dictionary used to expose lookup, timestamp, default, and freshness semantics.
    It is **not** the Chronon online store. Chronon production serving requires an
    implementation of its online API backed by a real key-value system.
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
    ### Compare values at the same key and cutoff

    A useful parity check aligns:

    1. the same feature-definition version,
    2. the same entity key,
    3. the same time cutoff,
    4. the same default and null rules,
    5. an explicit numeric tolerance.

    Comparing today's online value with a value backfilled for last week is not a
    parity test. The timestamps differ.

    We recompute the sample offline at the exact materialization time and then
    perturb one cached value to prove the checker can fail.
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
    ### Fast is not fresh

    A dictionary lookup can take microseconds and still return a week-old value.
    Separate these measures:

    ```text
    source lag       = ingest time - event time
    compute lag      = materialized time - ingest time
    end-to-end lag   = readable time - event time
    feature age      = decision time - latest contributing event time
    lookup latency   = response time - request time
    ```

    A feature SLO needs both freshness and serving reliability, for example:

    ```text
    customer_order_count_7d
      99% readable within 60 seconds
      p99 fetch below 20 milliseconds
      null rate below 0.1%
      parity mismatch below 0.01%
    ```

    Precomputing everything can create cost and staleness. Computing everything on
    demand can create latency and source-availability failures. Most real systems
    use a hybrid of precomputed historical state and request-time context.
    ''',
)

add_markdown(
    "observability-intro",
    r'''
    ---
    ## §11. Contracts, quality, and observability

    ![Feature observability covers the path from data quality to model impact](assets/09-observability-stack.svg)

    Pipeline success only means a job finished. It does not mean the feature is
    semantically correct, fresh, present, stable, served within latency, or useful
    to a model.

    A minimum production contract includes name, owner, entity, type, unit,
    definition, source, time boundary, availability policy, refresh SLO, null
    policy, sensitivity, version, consumers, and deprecation state.
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
    A distribution change is not automatically a data bug. Holiday purchasing,
    product launches, or a real customer shift can move the values. Monitoring
    needs an owner and a response playbook:

    - **source health:** missing partitions, schema changes, duplicate event IDs;
    - **quality:** nulls, ranges, units, cardinality, heavy hitters;
    - **freshness:** event watermark, ingestion lag, compute lag, materialization age;
    - **parity:** sampled offline/online values with timestamps and tolerances;
    - **serving:** p50, p95, p99, error rate, missing keys, fallback rate;
    - **lineage:** affected models, datasets, owners, and versions;
    - **impact:** prediction quality, business metric, incident count, and cost.

    A catalog entry with no owner is documentation, not an operable data product.
    Raw feature count is usually a vanity metric. More useful platform measures are
    feature-related incidents, backfill lead time, SLO attainment, validated reuse,
    parity mismatch, and cost per training or serving workload.
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

    A production rollout adds work that a notebook should not pretend away:

    1. map logical sources to warehouse tables, topics, schemas, and partitions;
    2. preserve event IDs, event time, and where needed availability time;
    3. compile reviewed definitions and assign team ownership;
    4. analyze keys, timestamps, schema, volume, and skew before backfill;
    5. backfill the `Join`, validate invariants, and publish a dataset manifest;
    6. upload batch state and start streaming updates for online `GroupBy`s;
    7. publish join metadata and integrate the fetch client with the model service;
    8. log a controlled sample, compare offline and online values, and alert;
    9. canary the model, define missing or stale-value fallbacks, then scale;
    10. version, migrate, and deprecate features without silently changing meaning.

    Backfills must also handle duplicates, deletes, late data, corrections, partial
    failures, and idempotent reruns. Recomputing an old feature with today's
    corrected source may be more accurate historically but different from what the
    production model actually saw.
    ''',
)

add_markdown(
    "when-to-use",
    r'''
    ![Use the smallest system that preserves feature truth](assets/10-when-to-use.svg)

    ### When a feature store earns its cost

    Strong candidates are reused, expensive, temporally changing, needed in both
    training and serving, governed, backfillable, and stable enough to contract.

    Poor candidates include one-off notebook columns, raw documents and images,
    cheap stateless transforms with no reuse, and values known only in the current
    request. A user embedding keyed by user ID can fit a feature store. Nearest-
    neighbor retrieval belongs in a vector database.

    Do not adopt a feature store merely because inference is online. A simple
    key-value lookup may suffice. Do not adopt one merely because several teams
    exist if they share no semantics. Batch-only systems can still benefit from
    point-in-time joins, lineage, contracts, and reproducible backfills.

    The strongest adoption signals are repeated training-serving incidents,
    duplicated definitions, point-in-time dataset bugs, staleness incidents,
    shared entity history, strict lineage requirements, and slow promotion from
    experiments to production.
    ''',
)

add_markdown(
    "final-checklist",
    r'''
    ## Final design review

    Before approving a feature for production, ask:

    1. **Decision:** exactly when does the model act?
    2. **Keys:** what entity and key mapping are stable?
    3. **Clocks:** what are event, availability, and prediction time?
    4. **Boundary:** does the current event belong in history or request context?
    5. **Label:** when is it mature, and can it leak through another entity?
    6. **Backfill:** can old values be reproduced with source and definition versions?
    7. **Serving:** what freshness, p99 latency, missing-key, and fallback contract applies?
    8. **Parity:** how are same-key, same-time values compared?
    9. **Governance:** who owns meaning, runtime, access, retention, and deletion?
    10. **Lifecycle:** how do consumers migrate from v1 to v2?

    If a design cannot answer those questions, more infrastructure will not make
    its features trustworthy.
    ''',
)

add_markdown(
    "exercises",
    r'''
    ## Extensions for deeper practice

    1. Add `purchase_order_id_count_1d` and prove `1d <= 7d <= 30d <= 90d`.
    2. Add an ingestion-delay simulation and compare event truth with
       production-available truth across the full sample.
    3. Create a `v2` cancellation feature that excludes administrative reversals,
       then write a migration and parity plan.
    4. Add a customer-country entity snapshot and decide whether it is snapshot or
       temporal accuracy. Explain what a country change means.
    5. Introduce a hot key and estimate batch skew and online request concentration.
    6. Add a feature-age timestamp to each online vector and implement stale-value
       fallback behavior.
    7. Run the official Chronon Docker quickstart, replace its fabricated sources
       with normalized UCI tables, and compare the output with `vector_at`.
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

    Feature-store perspectives reviewed for the companion theory guide:

    - [Databricks complete guide](https://www.databricks.com/blog/what-feature-store-complete-guide-ml-feature-engineering)
    - [Featurestore.org](https://www.featurestore.org/)
    - [IBM overview](https://www.ibm.com/think/topics/feature-store)
    - [Chalk overview](https://chalk.ai/blog/what-is-a-feature-store)
    - [AWS SageMaker Feature Store](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html)

    Dataset:

    - Daqing Chen, [Online Retail](https://doi.org/10.24432/C5BW33), UCI Machine
      Learning Repository, CC BY 4.0.

    Vendor sources are useful for implementation details and industry consensus,
    but they are not neutral standards. Product superiority and universal
    architecture claims require independent validation.

    © mui-group
    ''',
)


nbf.write(nb, OUTPUT)
print(f"Wrote {OUTPUT} with {len(nb.cells)} cells")
