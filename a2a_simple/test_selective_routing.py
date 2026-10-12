"""Regression tests for confident routing and abstention behavior."""

from pathlib import Path
from unittest import TestCase

from evals.evaluate_router import load_scenarios
from evals.evaluate_selective_routing import _validate_scenarios, evaluate
from routing_policy import classify_question


DATASET = Path(__file__).parent / "evals" / "selective_routing_scenarios.jsonl"


class SelectiveRoutingTests(TestCase):
    def test_conflicting_strong_routes_abstain(self) -> None:
        decision = classify_question(
            "Implement an algorithm and search the web for examples."
        )

        self.assertEqual(decision.tool, "unknown")
        self.assertEqual(decision.abstention_reason, "ambiguous_routes")
        self.assertEqual(decision.score_margin, 0)

    def test_single_route_remains_covered(self) -> None:
        decision = classify_question("Implement an algorithm for merge sort.")

        self.assertEqual(decision.tool, "code")
        self.assertIsNone(decision.abstention_reason)

    def test_dataset_has_complete_selective_routing_coverage(self) -> None:
        scenarios = load_scenarios(DATASET)
        _validate_scenarios(scenarios)
        result = evaluate(DATASET)

        self.assertEqual(result["metrics"]["in_domain_coverage"], 1.0)
        self.assertEqual(result["metrics"]["in_domain_accuracy"], 1.0)
        self.assertEqual(result["metrics"]["abstention_recall"], 1.0)
        self.assertEqual(result["metrics"]["false_route_rate"], 0.0)
        self.assertFalse(result["failures"])


if __name__ == "__main__":
    import unittest

    unittest.main()
