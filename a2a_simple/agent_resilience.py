"""Timeout and circuit-breaker controls for external agent invocations."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


class AgentCallTimeout(TimeoutError):
    """Raised when a delegated agent exceeds its request deadline."""


class AgentCircuitOpen(RuntimeError):
    """Raised when a route is temporarily blocked after repeated failures."""


class AgentRouteSaturated(RuntimeError):
    """Raised when a route has reached its configured concurrency limit."""


@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    opened_at: float | None = None


class CircuitBreaker:
    """Maintain independent failure circuits for each agent route."""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if recovery_timeout_seconds <= 0:
            raise ValueError("recovery_timeout_seconds must be positive")
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self._clock = clock
        self._states: dict[str, _CircuitState] = {}
        self._lock = threading.Lock()

    def before_call(self, route: str) -> None:
        """Allow a call, a recovery probe, or fail fast for an open route."""
        with self._lock:
            state = self._states.setdefault(route, _CircuitState())
            if state.opened_at is None:
                return
            if self._clock() - state.opened_at >= self.recovery_timeout_seconds:
                state.consecutive_failures = 0
                state.opened_at = None
                return
            raise AgentCircuitOpen(f"agent circuit is open for route {route!r}")

    def record_success(self, route: str) -> None:
        with self._lock:
            self._states[route] = _CircuitState()

    def record_failure(self, route: str) -> None:
        with self._lock:
            state = self._states.setdefault(route, _CircuitState())
            state.consecutive_failures += 1
            if state.consecutive_failures >= self.failure_threshold:
                state.opened_at = self._clock()

    def snapshot(self) -> dict[str, dict[str, int | bool]]:
        """Return content-free state suitable for metrics and diagnostics."""
        with self._lock:
            return {
                route: {
                    "consecutive_failures": state.consecutive_failures,
                    "open": state.opened_at is not None,
                }
                for route, state in sorted(self._states.items())
            }


class AgentBulkhead:
    """Limit concurrent work independently for each delegated agent route."""

    def __init__(self, *, max_in_flight_per_route: int = 8) -> None:
        if max_in_flight_per_route <= 0:
            raise ValueError("max_in_flight_per_route must be positive")
        self.max_in_flight_per_route = max_in_flight_per_route
        self._in_flight: dict[str, int] = {}
        self._lock = threading.Lock()

    @contextmanager
    def slot(self, route: str) -> Iterator[None]:
        """Acquire one route-local slot or fail fast without queueing."""
        with self._lock:
            current = self._in_flight.get(route, 0)
            if current >= self.max_in_flight_per_route:
                raise AgentRouteSaturated(
                    f"agent route {route!r} reached its concurrency limit"
                )
            self._in_flight[route] = current + 1
        try:
            yield
        finally:
            with self._lock:
                self._in_flight[route] -= 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return content-free utilization suitable for diagnostics."""
        with self._lock:
            return {
                route: {
                    "in_flight": in_flight,
                    "capacity": self.max_in_flight_per_route,
                }
                for route, in_flight in sorted(self._in_flight.items())
            }


async def invoke_with_resilience(
    route: str,
    operation: Callable[[], Awaitable[T]],
    *,
    timeout_seconds: float,
    circuit_breaker: CircuitBreaker,
    bulkhead: AgentBulkhead | None = None,
) -> T:
    """Invoke one agent with route-local capacity, deadline, and circuit state."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    slot = bulkhead.slot(route) if bulkhead is not None else nullcontext()
    with slot:
        circuit_breaker.before_call(route)
        try:
            result = await asyncio.wait_for(operation(), timeout=timeout_seconds)
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            circuit_breaker.record_failure(route)
            raise AgentCallTimeout(
                f"agent route {route!r} exceeded {timeout_seconds:g} seconds"
            ) from exc
        except Exception:
            circuit_breaker.record_failure(route)
            raise
        else:
            circuit_breaker.record_success(route)
            return result
