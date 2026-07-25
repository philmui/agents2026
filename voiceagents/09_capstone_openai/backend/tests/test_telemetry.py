"""
Tests for the always-safe Langfuse telemetry wrapper.

The whole point of telemetry.py is that the backend runs perfectly whether or
not Langfuse is configured. These tests pin that contract: with no keys set,
telemetry is DISABLED, its context managers are no-ops, and .flush() is safe.

NOTE: the shared course .env may contain real LANGFUSE_* keys (loaded by main.py
at import time). To test the DISABLED path deterministically, we clear those env
vars for the duration of each test with mock.patch.dict, regardless of what the
developer's environment happens to hold.
"""

import unittest
from unittest import mock

from src import telemetry as telemetry_module
from src.telemetry import Telemetry

# Env vars that would otherwise switch telemetry ON. We blank them so the
# "disabled" contract is tested independently of the developer's real .env.
_LANGFUSE_ENV = {
    "LANGFUSE_PUBLIC_KEY": "",
    "LANGFUSE_SECRET_KEY": "",
    "LANGFUSE_BASE_URL": "",
    "LANGFUSE_HOST": "",
}


class TelemetryDisabledTests(unittest.TestCase):
    def test_configure_stays_disabled_without_keys(self):
        # With the Langfuse keys blanked, a fresh Telemetry must not enable itself.
        with mock.patch.dict("os.environ", _LANGFUSE_ENV, clear=False):
            t = Telemetry()
            self.assertFalse(t.enabled)
            t.configure()
            self.assertFalse(t.enabled)

    def test_trace_is_a_noop_context_manager_when_disabled(self):
        t = Telemetry()  # never configured -> disabled
        with t.trace("assist-web-search", input="hi", tags=["assist"]) as span:
            # The handle exists and set_output is safe even with telemetry off.
            span.set_output("done")
        # Reaching here without raising is the assertion.

    def test_flush_is_safe_when_disabled(self):
        Telemetry().flush()  # must not raise

    def test_module_singleton_present(self):
        # main.py imports this shared instance; it must exist and be a Telemetry.
        self.assertIsInstance(telemetry_module.telemetry, Telemetry)


if __name__ == "__main__":
    unittest.main()
