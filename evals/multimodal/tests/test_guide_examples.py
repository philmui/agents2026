"""Check the actual historical-feature example printed in the guide."""
from pathlib import Path
import re


def feature_example():
    text = (Path(__file__).resolve().parents[1] / "FEATURE_STORES.md").read_text()
    block = next(b for b in re.findall(r"```python\n(.*?)```", text, flags=re.S)
                 if "def feature_as_of" in b)
    namespace = {}
    exec(compile(block, "FEATURE_STORES.md:point-in-time-example", "exec"), namespace)
    return namespace


def test_late_arrival_only_enters_after_available_time():
    example = feature_example()
    rows, at, query, age = (example[n] for n in ["rows", "at", "feature_as_of", "timedelta"])
    assert query(rows, "panel_1", at(10, 0), age(hours=2))["value"] == 42
    assert query(rows, "panel_1", at(10, 7), age(hours=2))["value"] == 46


def test_stale_unknown_and_future_features_remain_missing():
    e = feature_example()
    assert e["feature_as_of"](e["rows"], "panel_1", e["at"](10,0), e["timedelta"](minutes=30)) is None
    assert e["feature_as_of"](e["rows"], "unknown", e["at"](10,0), e["timedelta"](hours=2)) is None
    future = [{"entity_id":"panel_1", "event_time":e["at"](10,10),
               "available_time":e["at"](9,0), "value":999}]
    assert e["feature_as_of"](future, "panel_1", e["at"](10,0), e["timedelta"](hours=2)) is None


def test_later_correction_does_not_leak_into_earlier_replay():
    e = feature_example()
    rows = e["rows"] + [{"entity_id":"panel_1", "event_time":e["at"](9,0),
                         "available_time":e["at"](10,1), "value":43}]
    assert e["feature_as_of"](rows, "panel_1", e["at"](10,0), e["timedelta"](hours=2))["value"] == 42
    assert e["feature_as_of"](rows, "panel_1", e["at"](10,2), e["timedelta"](hours=2))["value"] == 43
