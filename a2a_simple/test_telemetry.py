import hmac
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
        self.assertTrue(payload["request_id"].startswith("external_"))
        self.assertNotIn("req-123", stream.getvalue())
        self.assertEqual(payload["route"], "image")
        self.assertEqual(payload["latency_ms"], 125.0)
        self.assertEqual(payload["input_text_chars"], 22)
        self.assertEqual(payload["input_image_bytes"], 512)
        self.assertNotIn("prompt", payload)
        self.assertEqual(event.outcome, "success")

    def test_external_request_ids_are_pseudonymized_but_correlatable(self) -> None:
        logger, stream = capturing_logger()
        key = b"test-correlation-key"
        telemetry = AgentTelemetry(
            logger=logger,
            clock=FakeClock(1.0, 1.01, 2.0, 2.01, 3.0, 3.01),
            correlation_key=key,
        )

        first = telemetry.start(request_id="customer@example.com")
        first.finish()
        repeated = telemetry.start(request_id="customer@example.com")
        repeated.finish()
        different = telemetry.start(request_id="another@example.com")
        different.finish()

        expected = hmac.digest(key, b"customer@example.com", "sha256").hex()[:32]
        self.assertEqual(first.request_id, f"external_{expected}")
        self.assertEqual(repeated.request_id, first.request_id)
        self.assertNotEqual(different.request_id, first.request_id)
        self.assertNotIn("customer@example.com", stream.getvalue())

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
        self.assertEqual(snapshot["routes"]["math"]["success_rate"], 0.5)
        self.assertEqual(snapshot["routes"]["math"]["latency_samples"], 2)
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

    def test_bounds_retained_events_without_losing_lifetime_counters(self) -> None:
        logger, _ = capturing_logger()
        telemetry = AgentTelemetry(
            logger=logger,
            clock=FakeClock(0.0, 0.01, 1.0, 1.02, 2.0, 2.03),
            max_events=2,
        )

        first = telemetry.start()
        first.set_route("math")
        first.fail(RuntimeError())
        second = telemetry.start()
        second.set_route("web")
        second.fail(TimeoutError())
        third = telemetry.start()
        third.set_route("hash")
        third.finish()

        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot["invocations_total"], 3)
        self.assertEqual(snapshot["errors_total"], 2)
        self.assertEqual(snapshot["success_rate"], 0.3333)
        self.assertEqual(snapshot["events_retained"], 2)
        self.assertEqual(snapshot["events_dropped_total"], 1)
        self.assertEqual(
            snapshot["routes"]["math"],
            {
                "invocations": 1,
                "errors": 1,
                "success_rate": 0.0,
                "latency_samples": 0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
            },
        )
        self.assertEqual(snapshot["routes"]["web"]["errors"], 1)

    def test_rejects_non_positive_event_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_events must be positive"):
            AgentTelemetry(max_events=0)

    def test_prometheus_snapshot_exports_lifetime_and_route_metrics(self) -> None:
        logger, _ = capturing_logger()
        telemetry = AgentTelemetry(
            logger=logger,
            clock=FakeClock(0.0, 0.01, 1.0, 1.03),
        )
        success = telemetry.start()
        success.set_route("math")
        success.finish()
        failure = telemetry.start()
        failure.set_route("math")
        failure.fail(TimeoutError())

        metrics = telemetry.prometheus_snapshot()

        self.assertIn("# TYPE a2a_agent_invocations_total counter", metrics)
        self.assertIn("a2a_agent_invocations_total 2", metrics)
        self.assertIn("a2a_agent_errors_total 1", metrics)
        self.assertIn('a2a_agent_route_invocations_total{route="math"} 2', metrics)
        self.assertIn(
            'a2a_agent_route_latency_milliseconds{route="math",quantile="0.95"} 30.0',
            metrics,
        )
        self.assertNotIn("request_id", metrics)

    def test_prometheus_snapshot_escapes_route_labels(self) -> None:
        logger, _ = capturing_logger()
        telemetry = AgentTelemetry(logger=logger, clock=FakeClock(0.0, 0.01))
        span = telemetry.start()
        span.set_route('unsafe"route\\name\nnext')
        span.finish()

        metrics = telemetry.prometheus_snapshot()

        self.assertIn('route="unsafe\\"route\\\\name\\nnext"', metrics)
        self.assertNotIn("\nnext\"}", metrics)


if __name__ == "__main__":
    unittest.main()
