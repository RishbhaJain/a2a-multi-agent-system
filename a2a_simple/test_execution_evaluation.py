import json
import tempfile
import unittest
from pathlib import Path

from evals.evaluate_execution import (
    check_regression,
    evaluate_scenarios,
    load_scenarios,
)


class ExecutionEvaluationTests(unittest.TestCase):
    def test_classifies_policy_and_runtime_boundaries(self):
        scenarios = [
            {
                "id": "success",
                "code": "print(6 * 7)",
                "expected_outcome": "success",
                "expected_stdout": "42",
            },
            {
                "id": "denied",
                "code": "import os",
                "expected_outcome": "policy_denied",
            },
            {
                "id": "runtime",
                "code": "raise ValueError('boom')",
                "expected_outcome": "runtime_error",
            },
            {
                "id": "timeout",
                "code": "while True:\n    pass",
                "expected_outcome": "timeout",
                "timeout_seconds": 0.2,
            },
            {
                "id": "truncated",
                "code": "print('x' * 1000)",
                "expected_outcome": "output_truncated",
                "max_output_bytes": 64,
            },
        ]

        metrics = evaluate_scenarios(scenarios)

        self.assertEqual(metrics["scenario_count"], 5)
        self.assertEqual(metrics["outcome_accuracy"], 1.0)
        self.assertEqual(metrics["safe_execution_success_rate"], 1.0)
        self.assertEqual(metrics["policy_rejection_accuracy"], 1.0)
        self.assertEqual(metrics["timeout_detection_rate"], 1.0)
        self.assertEqual(metrics["truncation_detection_rate"], 1.0)
        self.assertEqual(metrics["failure_counts"], {})

    def test_loader_rejects_duplicate_scenario_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "scenarios.jsonl"
            scenario = {"id": "same", "code": "print(1)", "expected_outcome": "success"}
            dataset.write_text(
                json.dumps(scenario) + "\n" + json.dumps(scenario) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate scenario id"):
                load_scenarios(dataset)

    def test_regression_gate_rejects_lower_accuracy(self):
        baseline = {
            "metrics": {
                "outcome_accuracy": 1.0,
                "safe_execution_success_rate": 1.0,
                "policy_rejection_accuracy": 1.0,
                "runtime_failure_detection_rate": 1.0,
                "timeout_detection_rate": 1.0,
                "truncation_detection_rate": 1.0,
            }
        }
        result = {
            "metrics": {**baseline["metrics"], "outcome_accuracy": 0.8},
            "regression_gate": {
                "outcome_accuracy_min": 1.0,
                "safe_execution_success_rate_min": 1.0,
                "policy_rejection_accuracy_min": 1.0,
                "runtime_failure_detection_rate_min": 1.0,
                "timeout_detection_rate_min": 1.0,
                "truncation_detection_rate_min": 1.0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

            with self.assertRaises(SystemExit):
                check_regression(result, baseline_path)


if __name__ == "__main__":
    unittest.main()
