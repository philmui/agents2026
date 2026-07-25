"""
Voice Agents - Langfuse telemetry for the OpenAI Agents SDK backend.

WHAT THIS FILE IS
-----------------
One small module that wires up Langfuse tracing for the whole backend, following
the official Langfuse best-practices skill (github.com/langfuse/skills), and does
it SAFELY: if you have not set Langfuse keys yet, the app still runs perfectly.
Telemetry is a nice-to-have here, never a hard dependency.

WHY LANGFUSE?
-------------
When an agent uses tools, decides things, and calls models, a plain log line
("web search ran") tells you almost nothing. A TRACE tells you the whole story:
which agent ran, what it was asked, which tool it called, how long each step
took, how many tokens it cost, and what it finally answered. Langfuse is a
dashboard for exactly those traces. You open https://cloud.langfuse.com and see
every Assist web search and every Translate session as a tidy timeline.

HOW THE PLUMBING WORKS (three layers, top to bottom)
----------------------------------------------------
  1) The OpenAI Agents SDK already EMITS structured events for every agent run
     (agent started, tool called, model responded, run finished).
  2) `OpenAIAgentsInstrumentor().instrument()` (from the `openinference` package,
     the exact integration the Langfuse docs recommend) listens to those events
     and re-emits them as OpenTelemetry "spans" - the industry-standard shape for
     one timed unit of work. This gives us model name, token usage, and the right
     observation types (generation / tool) automatically, which is a Langfuse
     best practice: prefer a framework integration over manual instrumentation.
  3) The Langfuse client (`langfuse.get_client()`) is an OpenTelemetry exporter:
     it collects those spans and ships them to your Langfuse project, using the
     LANGFUSE_* keys in your environment.

BEST-PRACTICE CHOICES BAKED IN HERE (from the Langfuse skill)
-------------------------------------------------------------
  * Import + init Langfuse AFTER load_dotenv() (main.py loads env first, then
    imports this module), so credentials are present when get_client() runs.
  * Descriptive, LOW-CARDINALITY span names ("assist-web-search", verb-first),
    never the query text, so filters and dashboards stay stable.
  * Set the trace's input/output to "what a reviewer needs at a glance" (the user
    query and the final answer), NOT a raw dump of every function argument.
  * Group turns with a session_id and tag by feature, via propagate_attributes.
  * Set the environment (production / development) so test traces are separable.
  * flush() after each short request so nothing is left un-sent.

This module hides all of that behind two names: `configure_telemetry()` (call
once at startup) and `telemetry` (an object with `.trace(...)`, `.flush()`, and
`.enabled`).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

# We import the third-party telemetry packages defensively. If a classmate has
# not installed them yet (or is on an unusual Python), the backend must still
# start and serve requests. So a failed import flips telemetry OFF instead of
# crashing the server.
try:  # pragma: no cover - exercised indirectly by the app at runtime
    from langfuse import get_client as _langfuse_get_client
    from langfuse import propagate_attributes as _propagate_attributes
    from openinference.instrumentation.openai_agents import (
        OpenAIAgentsInstrumentor as _OpenAIAgentsInstrumentor,
    )

    _TELEMETRY_IMPORTS_OK = True
except Exception:  # noqa: BLE001 - any import problem = telemetry disabled, app still runs.
    _langfuse_get_client = None
    _propagate_attributes = None
    _OpenAIAgentsInstrumentor = None
    _TELEMETRY_IMPORTS_OK = False


def _has_langfuse_keys() -> bool:
    """True only when BOTH Langfuse keys are present in the environment.

    Langfuse needs a public key (pk-lf-...) and a secret key (sk-lf-...). Without
    both, `get_client()` builds a DISABLED client, so we treat "no keys" as
    "telemetry off" and skip the wiring entirely.
    """
    return bool(
        (os.environ.get("LANGFUSE_PUBLIC_KEY") or "").strip()
        and (os.environ.get("LANGFUSE_SECRET_KEY") or "").strip()
    )


def _environment() -> str:
    """Which environment label to stamp on traces (best practice: separate test
    traces from production). Defaults to 'development' for the course."""
    return (os.environ.get("LANGFUSE_TRACING_ENVIRONMENT") or "development").strip()


class Telemetry:
    """A tiny, always-safe wrapper around the Langfuse client.

    Every method is a no-op when telemetry is disabled, so route code can call
    `telemetry.trace(...)` and `telemetry.flush()` unconditionally and never has
    to check "is Langfuse on?" itself.
    """

    def __init__(self) -> None:
        self.enabled = False
        self._client = None  # the langfuse client, or None when disabled

    def configure(self) -> None:
        """Turn telemetry ON if the packages are installed AND keys are set.

        Called once at server startup (see main.py's lifespan). Safe to call more
        than once; the second call is a harmless no-op.
        """
        if self.enabled:
            return
        if not _TELEMETRY_IMPORTS_OK:
            return  # langfuse / openinference not installed. Run without tracing.
        if not _has_langfuse_keys():
            return  # Keys not set. Common in class before you make a Langfuse account.

        # Langfuse reads LANGFUSE_BASE_URL, but many people know the variable as
        # LANGFUSE_HOST. Accept either, mapping HOST -> BASE_URL for the client.
        base_url = (
            os.environ.get("LANGFUSE_BASE_URL")
            or os.environ.get("LANGFUSE_HOST")
            or ""
        ).strip()
        if base_url:
            os.environ.setdefault("LANGFUSE_BASE_URL", base_url)

        # 1) Build (or fetch) the Langfuse OpenTelemetry client from the env keys.
        client = _langfuse_get_client()

        # 2) Verify the keys actually work. If they do not (typo, wrong region),
        #    leave telemetry OFF rather than silently dropping every span.
        try:
            if not client.auth_check():
                return
        except Exception:  # noqa: BLE001 - network/credential failure = stay disabled.
            return

        # 3) Start the OpenAI Agents SDK instrumentation. From now on, every
        #    Runner.run(...) automatically produces spans (with model + token
        #    usage + correct observation types) that flow to Langfuse.
        _OpenAIAgentsInstrumentor().instrument()

        self._client = client
        self.enabled = True

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        input: object | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        as_type: str = "span",
        **metadata: object,
    ) -> Iterator["_TraceHandle"]:
        """Wrap a request in ONE Langfuse trace with best-practice attributes.

        Use it like:

            with telemetry.trace(
                "assist-web-search", input=query, session_id=sid,
                tags=["assist", "web-search"],
            ) as span:
                answer = ... do the work ...
                span.set_output(answer)

        Arguments follow the Langfuse best-practices skill:
          name       : verb-first, LOW-CARDINALITY (no query text in the name).
          input      : "what a reviewer needs at a glance" (the user query), not
                       a dump of every function argument.
          session_id : groups related traces (one per conversation/session).
          tags       : immutable business dimensions set at creation time.
          as_type    : the observation type. Our web-search Agent is a subagent
                       dispatched by the browser voice agent, so the skill says to
                       type its execution as "agent" (it shows as its own node in
                       the Agent Graph). Use "span" for a plain request container.

        When telemetry is disabled this yields a no-op handle, so the surrounding
        code reads identically whether or not Langfuse is on. The tool/model spans
        the OpenAI Agents instrumentation emits nest neatly INSIDE this span.
        """
        if not self.enabled or self._client is None:
            yield _TraceHandle(None)
            return

        observation_cm = self._client.start_as_current_observation(
            name=name,
            as_type=as_type,  # "span" (container) or "agent" (subagent execution)
            input=input,      # explicit input: just the user query, not all args
        )
        with observation_cm as observation:
            # propagate_attributes sets TRACE-level fields (session, tags,
            # environment) so they attach to this trace and everything under it.
            attributes_cm = _propagate_attributes(
                session_id=session_id,
                tags=tags,
                environment=_environment(),
                metadata=dict(metadata) or None,
            )
            with attributes_cm:
                handle = _TraceHandle(observation)
                yield handle

    def flush(self) -> None:
        """Push any buffered spans to Langfuse now.

        Spans are batched for efficiency. In a short web request we call this
        right after finishing so the trace appears promptly and nothing is lost
        if the process is idle or restarts. No-op when disabled. (This is the #1
        "common mistake" in the Langfuse skill: forgetting to flush.)
        """
        if self.enabled and self._client is not None:
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001 - never let telemetry break a response.
                pass


class _TraceHandle:
    """A thin handle returned by telemetry.trace(). Lets a route set the trace's
    OUTPUT (best practice: capture the final answer, not just the input) without
    touching the Langfuse client directly. No-op when telemetry is disabled."""

    def __init__(self, observation: object | None) -> None:
        self._observation = observation

    def set_output(self, output: object) -> None:
        if self._observation is None:
            return
        try:
            self._observation.update(output=output)
        except Exception:  # noqa: BLE001 - telemetry must never break a real response.
            pass


# One shared instance the whole app imports. `configure()` is called at startup.
telemetry = Telemetry()


def configure_telemetry() -> None:
    """Module-level convenience so main.py can just call configure_telemetry()."""
    telemetry.configure()
