"""Content-safe structured telemetry for agent invocations.

The telemetry intentionally records input sizes instead of input contents so
production logs remain useful without becoming a second store for prompts or
uploaded files.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable


@dataclass(frozen=True)
class InvocationEvent:
    request_id: str
    route: str
    outcome: str
    latency_ms: float
    input_text_chars: int
    input_image_bytes: int
    error_type: str | None
    timestamp: str
    event: str = "agent_invocation"


class InvocationSpan:
    """Tracks exactly one invocation from receipt to completion."""

    def __init__(
        self,
        telemetry: AgentTelemetry,
        *,
        request_id: str,
        input_text_chars: int,
        input_image_bytes: int,
    ) -> None:
        self._telemetry = telemetry
        self._request_id = request_id
        self._input_text_chars = input_text_chars
        self._input_image_bytes = input_image_bytes
        self._started_at = telemetry.clock()
        self._route = "unclassified"
        self._event: InvocationEvent | None = None

    @property
    def request_id(self) -> str:
        return self._request_id

    def set_route(self, route: str) -> None:
        if self._event is not None:
            raise RuntimeError("cannot change route after a span is finished")
        self._route = route

    def finish(
        self,
        outcome: str = "success",
        *,
        error_type: str | None = None,
    ) -> InvocationEvent:
        """Finish once and return the original event on repeated calls."""
        if self._event is not None:
            return self._event

        latency_ms = max(0.0, (self._telemetry.clock() - self._started_at) * 1000)
        self._event = InvocationEvent(
            request_id=self._request_id,
            route=self._route,
            outcome=outcome,
            latency_ms=round(latency_ms, 3),
            input_text_chars=self._input_text_chars,
            input_image_bytes=self._input_image_bytes,
            error_type=error_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._telemetry.record(self._event)
        return self._event

    def fail(self, error: BaseException) -> InvocationEvent:
        """Record only an exception class, never its potentially sensitive text."""
        return self.finish("error", error_type=type(error).__name__)


class AgentTelemetry:
    """Emits JSON traces and maintains process-local aggregate metrics."""

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.clock = clock
        self.logger = logger or self._default_logger()
        self._lock = threading.Lock()
        self._events: list[InvocationEvent] = []

    @staticmethod
    def _default_logger() -> logging.Logger:
        logger = logging.getLogger("a2a.telemetry")
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def start(
        self,
        *,
        request_id: str | None = None,
        input_text_chars: int = 0,
        input_image_bytes: int = 0,
    ) -> InvocationSpan:
        if input_text_chars < 0 or input_image_bytes < 0:
            raise ValueError("input sizes must be non-negative")
        return InvocationSpan(
            self,
            request_id=request_id or uuid.uuid4().hex,
            input_text_chars=input_text_chars,
            input_image_bytes=input_image_bytes,
        )

    def record(self, event: InvocationEvent) -> None:
        with self._lock:
            self._events.append(event)
        self.logger.info(
            "%s", json.dumps(asdict(event), sort_keys=True, separators=(",", ":"))
        )

    def snapshot(self) -> dict[str, object]:
        """Return aggregate reliability and latency metrics by route."""
        with self._lock:
            events = tuple(self._events)

        grouped: dict[str, list[InvocationEvent]] = defaultdict(list)
        for event in events:
            grouped[event.route].append(event)

        errors = sum(event.outcome != "success" for event in events)
        return {
            "invocations_total": len(events),
            "errors_total": errors,
            "success_rate": round(
                (len(events) - errors) / len(events), 4
            )
            if events
            else 1.0,
            "routes": {
                route: self._route_metrics(route_events)
                for route, route_events in sorted(grouped.items())
            },
        }

    @staticmethod
    def _route_metrics(events: list[InvocationEvent]) -> dict[str, float | int]:
        latencies = sorted(event.latency_ms for event in events)
        errors = sum(event.outcome != "success" for event in events)
        return {
            "invocations": len(events),
            "errors": errors,
            "p50_latency_ms": AgentTelemetry._percentile(latencies, 0.50),
            "p95_latency_ms": AgentTelemetry._percentile(latencies, 0.95),
        }

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        index = max(0, math.ceil(percentile * len(values)) - 1)
        return values[index]
