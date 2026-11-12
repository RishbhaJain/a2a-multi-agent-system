"""Content-safe structured telemetry for agent invocations.

The telemetry intentionally records input sizes instead of input contents so
production logs remain useful without becoming a second store for prompts or
uploaded files.
"""

from __future__ import annotations

import hmac
import json
import logging
import math
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable


_CIRCUIT_STATES = ("closed", "open", "half_open")


def render_resilience_prometheus(
    circuit_breakers: dict[str, dict[str, int | bool | str]],
    bulkheads: dict[str, dict[str, int]],
) -> str:
    """Render content-free circuit and concurrency state for operations."""
    lines = [
        "# HELP a2a_agent_circuit_consecutive_failures Consecutive failures by route.",
        "# TYPE a2a_agent_circuit_consecutive_failures gauge",
        "# HELP a2a_agent_circuit_state Current circuit state as a one-hot gauge.",
        "# TYPE a2a_agent_circuit_state gauge",
        "# HELP a2a_agent_bulkhead_in_flight Active delegated calls by route.",
        "# TYPE a2a_agent_bulkhead_in_flight gauge",
        "# HELP a2a_agent_bulkhead_capacity Maximum concurrent calls by route.",
        "# TYPE a2a_agent_bulkhead_capacity gauge",
        "# HELP a2a_agent_bulkhead_utilization_ratio Route concurrency utilization.",
        "# TYPE a2a_agent_bulkhead_utilization_ratio gauge",
    ]
    for route, metrics in sorted(circuit_breakers.items()):
        route_label = AgentTelemetry._prometheus_label(route)
        labels = f'route="{route_label}"'
        lines.append(
            "a2a_agent_circuit_consecutive_failures"
            f"{{{labels}}} {metrics['consecutive_failures']}"
        )
        current_state = str(metrics["state"])
        for state in _CIRCUIT_STATES:
            lines.append(
                "a2a_agent_circuit_state"
                f'{{{labels},state="{state}"}} {int(state == current_state)}'
            )

    for route, metrics in sorted(bulkheads.items()):
        route_label = AgentTelemetry._prometheus_label(route)
        labels = f'route="{route_label}"'
        in_flight = metrics["in_flight"]
        capacity = metrics["capacity"]
        utilization = round(in_flight / capacity, 4) if capacity else 0.0
        lines.extend(
            [
                f"a2a_agent_bulkhead_in_flight{{{labels}}} {in_flight}",
                f"a2a_agent_bulkhead_capacity{{{labels}}} {capacity}",
                (
                    "a2a_agent_bulkhead_utilization_ratio"
                    f"{{{labels}}} {utilization}"
                ),
            ]
        )
    return "\n".join(lines) + "\n"


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
        max_events: int = 10_000,
        correlation_key: bytes | None = None,
    ) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self.clock = clock
        self.logger = logger or self._default_logger()
        self._correlation_key = correlation_key or secrets.token_bytes(32)
        self._lock = threading.Lock()
        self._events: deque[InvocationEvent] = deque(maxlen=max_events)
        self._invocations_total = 0
        self._errors_total = 0
        self._route_invocations_total: dict[str, int] = defaultdict(int)
        self._route_errors_total: dict[str, int] = defaultdict(int)

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
            request_id=self._request_token(request_id),
            input_text_chars=input_text_chars,
            input_image_bytes=input_image_bytes,
        )

    def _request_token(self, request_id: str | None) -> str:
        """Return a correlatable token without retaining an external identifier."""
        if not request_id:
            return uuid.uuid4().hex
        digest = hmac.digest(
            self._correlation_key,
            request_id.encode("utf-8"),
            "sha256",
        )
        return f"external_{digest.hex()[:32]}"

    def record(self, event: InvocationEvent) -> None:
        with self._lock:
            self._events.append(event)
            self._invocations_total += 1
            self._route_invocations_total[event.route] += 1
            if event.outcome != "success":
                self._errors_total += 1
                self._route_errors_total[event.route] += 1
        self.logger.info(
            "%s", json.dumps(asdict(event), sort_keys=True, separators=(",", ":"))
        )

    def snapshot(self) -> dict[str, object]:
        """Return lifetime totals and latency metrics for the retained window."""
        with self._lock:
            events = tuple(self._events)
            invocations_total = self._invocations_total
            errors_total = self._errors_total
            route_invocations_total = dict(self._route_invocations_total)
            route_errors_total = dict(self._route_errors_total)

        grouped: dict[str, list[InvocationEvent]] = defaultdict(list)
        for event in events:
            grouped[event.route].append(event)

        return {
            "invocations_total": invocations_total,
            "errors_total": errors_total,
            "success_rate": round(
                (invocations_total - errors_total) / invocations_total, 4
            )
            if invocations_total
            else 1.0,
            "events_retained": len(events),
            "events_dropped_total": invocations_total - len(events),
            "routes": {
                route: self._route_metrics(
                    grouped.get(route, []),
                    invocations_total=route_invocations_total[route],
                    errors_total=route_errors_total.get(route, 0),
                )
                for route in sorted(route_invocations_total)
            },
        }

    def prometheus_snapshot(self) -> str:
        """Render content-safe process metrics in Prometheus text format."""
        snapshot = self.snapshot()
        lines = [
            "# HELP a2a_agent_invocations_total Agent invocations since process start.",
            "# TYPE a2a_agent_invocations_total counter",
            f"a2a_agent_invocations_total {snapshot['invocations_total']}",
            "# HELP a2a_agent_errors_total Failed agent invocations since process start.",
            "# TYPE a2a_agent_errors_total counter",
            f"a2a_agent_errors_total {snapshot['errors_total']}",
            "# HELP a2a_agent_success_ratio Successful invocation ratio since process start.",
            "# TYPE a2a_agent_success_ratio gauge",
            f"a2a_agent_success_ratio {snapshot['success_rate']}",
            "# HELP a2a_agent_events_dropped_total Events omitted from the bounded latency window.",
            "# TYPE a2a_agent_events_dropped_total counter",
            f"a2a_agent_events_dropped_total {snapshot['events_dropped_total']}",
            "# HELP a2a_agent_route_invocations_total Invocations by selected route.",
            "# TYPE a2a_agent_route_invocations_total counter",
            "# HELP a2a_agent_route_errors_total Failed invocations by selected route.",
            "# TYPE a2a_agent_route_errors_total counter",
            "# HELP a2a_agent_route_success_ratio Successful invocation ratio by route.",
            "# TYPE a2a_agent_route_success_ratio gauge",
            "# HELP a2a_agent_route_latency_samples Retained latency samples by route.",
            "# TYPE a2a_agent_route_latency_samples gauge",
            "# HELP a2a_agent_route_latency_milliseconds Retained-window latency percentile by route.",
            "# TYPE a2a_agent_route_latency_milliseconds gauge",
        ]
        routes = snapshot["routes"]
        for route, metrics in routes.items():
            route_label = self._prometheus_label(route)
            labels = f'route="{route_label}"'
            lines.extend(
                [
                    f"a2a_agent_route_invocations_total{{{labels}}} {metrics['invocations']}",
                    f"a2a_agent_route_errors_total{{{labels}}} {metrics['errors']}",
                    f"a2a_agent_route_success_ratio{{{labels}}} {metrics['success_rate']}",
                    f"a2a_agent_route_latency_samples{{{labels}}} {metrics['latency_samples']}",
                    (
                        "a2a_agent_route_latency_milliseconds"
                        f'{{{labels},quantile="0.5"}} {metrics["p50_latency_ms"]}'
                    ),
                    (
                        "a2a_agent_route_latency_milliseconds"
                        f'{{{labels},quantile="0.95"}} {metrics["p95_latency_ms"]}'
                    ),
                ]
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _prometheus_label(value: str) -> str:
        """Escape label values according to the Prometheus text format."""
        return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')

    @staticmethod
    def _route_metrics(
        events: list[InvocationEvent],
        *,
        invocations_total: int,
        errors_total: int,
    ) -> dict[str, float | int]:
        """Return exact route totals plus latency metrics from retained samples."""
        latencies = sorted(event.latency_ms for event in events)
        return {
            "invocations": invocations_total,
            "errors": errors_total,
            "success_rate": round(
                (invocations_total - errors_total) / invocations_total, 4
            ),
            "latency_samples": len(latencies),
            "p50_latency_ms": AgentTelemetry._percentile(latencies, 0.50),
            "p95_latency_ms": AgentTelemetry._percentile(latencies, 0.95),
        }

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        index = max(0, math.ceil(percentile * len(values)) - 1)
        return values[index]
