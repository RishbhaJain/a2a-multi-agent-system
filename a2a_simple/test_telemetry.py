import io
import json
import logging
import unittest

from telemetry import AgentTelemetry


class FakeClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def capturing_logger() -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    logger = logging.Logger("test.telemetry")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(stream))
    return logger, stream


class AgentTelemetryTests(unittest.TestCase):
    def test_emits_structured_content_safe_trace(self) -> None:
        logger, stream = capturing_logger()
        telemetry = AgentTelemetry(logger=logger, clock=FakeClock(1.0, 1.125))

        span = telemetry.start(
            request_id="req-123", input_text_chars=22, input_image_bytes=512
        )
        span.set_route("image")
        event = span.finish()

        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["request_id"], "req-123")
        self.assertEqual(payload["route"], "image")
        self.assertEqual(payload["latency_ms"], 125.0)
        self.assertEqual(payload["input_text_chars"], 22)
        self.assertEqual(payload["input_image_bytes"], 512)
        self.assertNotIn("prompt", payload)
        self.assertEqual(event.outcome, "success")

    def test_failure_records_exception_type_without_message(self) -> None:
        logger, stream = capturing_logger()
        telemetry = AgentTelemetry(logger=logger, clock=FakeClock(2.0, 2.01))
        span = telemetry.start(request_id="failed-request")
        span.set_route("web")

        span.fail(ValueError("secret provider response"))

        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["outcome"], "error")
        self.assertEqual(payload["error_type"], "ValueError")
        self.assertNotIn("secret provider response", stream.getvalue())

    def test_snapshot_reports_route_reliability_and_tail_latency(self) -> None:
        logger, _ = capturing_logger()
        telemetry = AgentTelemetry(
            logger=logger,
            clock=FakeClock(0.0, 0.010, 1.0, 1.030, 2.0, 2.100),
        )

        first = telemetry.start()
        first.set_route("math")
        first.finish()
        second = telemetry.start()
        second.set_route("math")
        second.fail(TimeoutError())
        third = telemetry.start()
        third.set_route("hash")
        third.finish()

        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot["invocations_total"], 3)
        self.assertEqual(snapshot["errors_total"], 1)
        self.assertEqual(snapshot["success_rate"], 0.6667)
        self.assertEqual(snapshot["routes"]["math"]["p50_latency_ms"], 10.0)
        self.assertEqual(snapshot["routes"]["math"]["p95_latency_ms"], 30.0)
        self.assertEqual(snapshot["routes"]["hash"]["invocations"], 1)

    def test_finish_is_idempotent_and_does_not_double_count(self) -> None:
        logger, stream = capturing_logger()
        telemetry = AgentTelemetry(logger=logger, clock=FakeClock(4.0, 4.02))
        span = telemetry.start()

        first = span.finish()
        second = span.finish()

        self.assertIs(first, second)
        self.assertEqual(telemetry.snapshot()["invocations_total"], 1)
        self.assertEqual(len(stream.getvalue().splitlines()), 1)

    def test_rejects_invalid_input_sizes(self) -> None:
        telemetry = AgentTelemetry()
        with self.assertRaisesRegex(ValueError, "non-negative"):
            telemetry.start(input_text_chars=-1)


if __name__ == "__main__":
    unittest.main()
