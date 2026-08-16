"""Historical customer cancellation features."""

from ai.chronon.api.ttypes import Accuracy, EventSource, Source
from ai.chronon.group_by import Aggregation, GroupBy, Operation, TimeUnit, Window
from ai.chronon.query import Query, select


WINDOWS = [Window(length=days, timeUnit=TimeUnit.DAYS) for days in (30, 90)]

source = Source(
    events=EventSource(
        table="retail.cancellations",
        topic="retail.cancellations.v1",
        query=Query(
            selects=select("customer_id", "cancellation_id", "refund_value"),
            time_column="ts",
        ),
    )
)
v1 = GroupBy(
    sources=[source],
    keys=["customer_id"],
    aggregations=[
        Aggregation(
            input_column="refund_value",
            operation=Operation.SUM,
            windows=WINDOWS,
        ),
        Aggregation(
            input_column="cancellation_id",
            operation=Operation.COUNT,
            windows=WINDOWS,
        ),
    ],
    accuracy=Accuracy.TEMPORAL,
    online=True,
    tags={
        "owner": "retail-ml",
        "use_case": "repeat-purchase",
        "entity": "customer",
    },
    description="Customer cancellation behavior before a prediction timestamp.",
)
