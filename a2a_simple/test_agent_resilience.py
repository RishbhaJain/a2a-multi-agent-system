import asyncio
import unittest

from agent_resilience import (
    AgentCallTimeout,
    AgentCircuitOpen,
    CircuitBreaker,
    invoke_with_resilience,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class AgentResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_call_returns_result_and_resets_failures(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2)

        async def fail() -> str:
            raise RuntimeError("provider unavailable")

        with self.assertRaises(RuntimeError):
            await invoke_with_resilience(
                "math", fail, timeout_seconds=1, circuit_breaker=breaker
            )

        async def succeed() -> str:
            return "answer"

        result = await invoke_with_resilience(
            "math", succeed, timeout_seconds=1, circuit_breaker=breaker
        )
        self.assertEqual(result, "answer")
        self.assertEqual(
            breaker.snapshot()["math"],
            {"consecutive_failures": 0, "open": False},
        )

    async def test_timeout_is_typed_and_counted(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2)

        async def slow() -> None:
            await asyncio.sleep(0.05)

        with self.assertRaises(AgentCallTimeout):
            await invoke_with_resilience(
                "web", slow, timeout_seconds=0.001, circuit_breaker=breaker
            )
        self.assertEqual(breaker.snapshot()["web"]["consecutive_failures"], 1)

    async def test_circuit_opens_and_skips_operation_until_cooldown(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout_seconds=10,
            clock=clock,
        )
        calls = 0

        async def fail() -> None:
            nonlocal calls
            calls += 1
            raise ConnectionError("upstream unavailable")

        for _ in range(2):
            with self.assertRaises(ConnectionError):
                await invoke_with_resilience(
                    "image", fail, timeout_seconds=1, circuit_breaker=breaker
                )

        with self.assertRaises(AgentCircuitOpen):
            await invoke_with_resilience(
                "image", fail, timeout_seconds=1, circuit_breaker=breaker
            )
        self.assertEqual(calls, 2)
        self.assertTrue(breaker.snapshot()["image"]["open"])

        clock.now = 11

        async def recover() -> str:
            return "recovered"

        self.assertEqual(
            await invoke_with_resilience(
                "image", recover, timeout_seconds=1, circuit_breaker=breaker
            ),
            "recovered",
        )
        self.assertFalse(breaker.snapshot()["image"]["open"])

    async def test_failures_are_isolated_by_route(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1)

        async def fail() -> None:
            raise RuntimeError("failure")

        with self.assertRaises(RuntimeError):
            await invoke_with_resilience(
                "web", fail, timeout_seconds=1, circuit_breaker=breaker
            )

        async def succeed() -> str:
            return "ok"

        self.assertEqual(
            await invoke_with_resilience(
                "math", succeed, timeout_seconds=1, circuit_breaker=breaker
            ),
            "ok",
        )
        self.assertTrue(breaker.snapshot()["web"]["open"])
        self.assertFalse(breaker.snapshot()["math"]["open"])

    async def test_cancellation_does_not_count_as_provider_failure(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1)

        async def blocked() -> None:
            await asyncio.Event().wait()

        task = asyncio.create_task(
            invoke_with_resilience(
                "code", blocked, timeout_seconds=10, circuit_breaker=breaker
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(breaker.snapshot()["code"]["consecutive_failures"], 0)

    def test_configuration_rejects_non_positive_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "failure_threshold"):
            CircuitBreaker(failure_threshold=0)
        with self.assertRaisesRegex(ValueError, "recovery_timeout_seconds"):
            CircuitBreaker(recovery_timeout_seconds=0)


if __name__ == "__main__":
    unittest.main()
