import io
import logging
import unittest
from unittest.mock import patch

from agent_executor import MathAgent
from agent_resilience import (
    AgentBulkhead,
    AgentOperationError,
    CircuitBreaker,
)
from code_execution_agent import CodeExecutionAgent
from hash_agent import HashAgent
from image_agent import ImageRecognitionAgent
from router_agent import RouterAgent
from telemetry import AgentTelemetry
from web_agent import WebBrowsingAgent


class FailingSyncModel:
    def generate_content(self, *_args, **_kwargs):
        raise RuntimeError("secret provider detail")


class FailingAsyncModel:
    async def generate_content_async(self, *_args, **_kwargs):
        raise RuntimeError("secret provider detail")


class FailingAgent:
    async def invoke(self, *_args, **_kwargs):
        raise AgentOperationError("secret provider detail")


class AgentFailurePropagationTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_backed_agents_raise_sanitized_typed_failures(self) -> None:
        math = MathAgent.model_construct(model=FailingSyncModel())
        image = ImageRecognitionAgent.__new__(ImageRecognitionAgent)
        image.model = FailingSyncModel()
        code = CodeExecutionAgent.__new__(CodeExecutionAgent)
        code.model = FailingAsyncModel()

        for operation in (
            math.invoke("calculate 2 + 2"),
            image.invoke(b"image bytes"),
            code.invoke("write a program"),
        ):
            with self.assertRaises(AgentOperationError) as raised:
                await operation
            self.assertNotIn("secret provider detail", str(raised.exception))

    async def test_local_and_browser_failures_use_same_typed_boundary(self) -> None:
        hash_agent = HashAgent()
        with patch.object(
            hash_agent,
            "_execute_hash_sequence",
            side_effect=RuntimeError("secret local detail"),
        ):
            with self.assertRaises(AgentOperationError):
                await hash_agent.invoke('hash "hello" with md5')

        web = WebBrowsingAgent()
        with patch(
            "web_agent.ChromeDriverManager.install",
            side_effect=RuntimeError("secret driver detail"),
        ):
            with self.assertRaises(AgentOperationError) as raised:
                await web.invoke("visit https://example.com")
        self.assertNotIn("secret driver detail", str(raised.exception))

    async def test_router_counts_agent_failure_and_returns_public_error(self) -> None:
        stream = io.StringIO()
        logger = logging.Logger("test.agent.failure")
        logger.addHandler(logging.StreamHandler(stream))
        telemetry = AgentTelemetry(logger=logger)
        breaker = CircuitBreaker(failure_threshold=1)

        router = RouterAgent.__new__(RouterAgent)
        router.math_agent = FailingAgent()
        router.telemetry = telemetry
        router.agent_timeout_seconds = 1
        router.circuit_breaker = breaker
        router.bulkhead = AgentBulkhead(max_in_flight_per_route=1)

        result = await router.invoke("Calculate 2 + 2", request_id="req-failure")

        self.assertEqual(
            result,
            "The selected agent is temporarily unavailable. "
            "Please try the request again.",
        )
        self.assertTrue(breaker.snapshot()["math"]["open"])
        self.assertEqual(telemetry.snapshot()["errors_total"], 1)
        self.assertIn('"error_type":"AgentOperationError"', stream.getvalue())
        self.assertNotIn("secret provider detail", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
