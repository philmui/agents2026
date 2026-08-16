"""Point-in-time training set for the repeat-purchase model.

The left side defines one prediction event per completed order. The current
order value stays on the left because it is request-time context, not historical
feature state. Both right-side GroupBys only observe events before `ts`.
"""

from ai.chronon.api.ttypes import EventSource, Source
from ai.chronon.join import Join, JoinPart
from ai.chronon.query import Query, select

from group_bys.retail.cancellations import v1 as cancellations_v1
from group_bys.retail.purchases import v1 as purchases_v1


prediction_events = Source(
    events=EventSource(
        table="retail.prediction_events",
        query=Query(
            selects=select(
                "customer_id",
                "order_id",
                "current_order_value",
                "repeat_purchase_30d",
            ),
            time_column="ts",
        ),
    )
)

v1 = Join(
    left=prediction_events,
    right_parts=[
        JoinPart(group_by=purchases_v1, prefix="purchase"),
        JoinPart(group_by=cancellations_v1, prefix="cancel"),
    ],
    online=True,
    check_consistency=True,
    sample_percent=1.0,
    tags={
        "owner": "retail-ml",
        "model": "repeat-purchase-v1",
        "sla": "p99-under-20ms",
    },
    description="Historical features aligned to completed-order decision events.",
)
