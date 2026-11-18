"""Regression tests for confident routing and abstention behavior."""

import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from evals.evaluate_router import load_scenarios
from evals.evaluate_selective_routing import (
    _validate_scenarios,
    check_regression,
    evaluate,
)
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

    def test_margin_threshold_exposes_coverage_risk_tradeoff(self) -> None:
        prompt = "Implement an algorithm and search the web for examples."

        permissive = classify_question(prompt, min_confident_margin=0)
        selective = classify_question(prompt, min_confident_margin=2)

        self.assertEqual(permissive.tool, "code")
        self.assertEqual(selective.tool, "unknown")
        self.assertEqual(selective.abstention_reason, "ambiguous_routes")

    def test_dataset_has_complete_selective_routing_coverage(self) -> None:
        scenarios = load_scenarios(DATASET)
        _validate_scenarios(scenarios)
        result = evaluate(DATASET)

        self.assertEqual(result["metrics"]["in_domain_coverage"], 1.0)
        self.assertEqual(result["metrics"]["in_domain_accuracy"], 1.0)
        self.assertEqual(result["metrics"]["abstention_recall"], 1.0)
        self.assertEqual(result["metrics"]["false_route_rate"], 0.0)
        self.assertEqual(result["metrics"]["coverage"], 0.5556)
        self.assertEqual(result["metrics"]["selective_accuracy"], 1.0)
        self.assertEqual(result["metrics"]["selective_risk"], 0.0)
        self.assertFalse(result["failures"])

        curve = result["risk_coverage_curve"]
        self.assertEqual(curve["production_margin"], 3)
        self.assertEqual(curve["minimum_zero_false_route_margin"], 2)
        self.assertEqual(
            [
                (
                    point["min_confident_margin"],
                    point["coverage"],
                    point["selective_accuracy"],
                    point["false_route_rate"],
                )
                for point in curve["operating_points"]
            ],
            [
                (0, 0.7778, 0.7143, 0.5),
                (1, 0.6667, 0.8333, 0.25),
                (2, 0.5556, 1.0, 0.0),
                (3, 0.5556, 1.0, 0.0),
            ],
        )

    def test_thresholds_must_be_non_negative_integers(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            classify_question("Calculate 2 + 2", min_confident_margin=-1)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            classify_question("Calculate 2 + 2", min_confident_margin=True)

    def test_regression_gate_rejects_a_weaker_zero_false_route_margin(self) -> None:
        baseline = evaluate(DATASET)
        regressed = deepcopy(baseline)
        regressed["risk_coverage_curve"]["minimum_zero_false_route_margin"] = 4

        with TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "zero false routes"):
                check_regression(regressed, baseline_path)


if __name__ == "__main__":
    import unittest

    unittest.main()
