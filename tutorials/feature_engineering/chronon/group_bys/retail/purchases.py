"""Historical customer purchase features.

The warehouse table and topic are logical integration points. The notebook uses
the same columns in a local reference runner. A production Chronon deployment
maps these names to its warehouse and streaming infrastructure.
"""

from ai.chronon.api.ttypes import Accuracy, EventSource, Source
from ai.chronon.group_by import Aggregation, GroupBy, Operation, TimeUnit, Window
from ai.chronon.query import Query, select


WINDOWS = [Window(length=days, timeUnit=TimeUnit.DAYS) for days in (7, 30, 90)]

source = Source(
    events=EventSource(
        table="retail.orders",
        topic="retail.orders.v1",
        query=Query(
            selects=select(
                "customer_id",
                "order_id",
                "order_value",
                "item_count",
            ),
            time_column="ts",
        ),
    )
)
v1 = GroupBy(
    sources=[source],
    keys=["customer_id"],
    aggregations=[
        Aggregation(
            input_column="order_value",
            operation=Operation.SUM,
            windows=WINDOWS,
        ),
        Aggregation(
            input_column="order_id",
            operation=Operation.COUNT,
            windows=WINDOWS,
        ),
        Aggregation(
            input_column="order_value",
            operation=Operation.AVERAGE,
            windows=[Window(length=30, timeUnit=TimeUnit.DAYS),
                     Window(length=90, timeUnit=TimeUnit.DAYS)],
        ),
        Aggregation(
            input_column="item_count",
            operation=Operation.SUM,
            windows=[Window(length=30, timeUnit=TimeUnit.DAYS)],
        ),
    ],
    accuracy=Accuracy.TEMPORAL,
    online=True,
    tags={
        "owner": "retail-ml",
        "use_case": "repeat-purchase",
        "entity": "customer",
    },
    description="Customer purchase behavior before a prediction timestamp.",
)
