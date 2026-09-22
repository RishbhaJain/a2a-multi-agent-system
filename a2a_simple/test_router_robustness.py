"""Regression tests for meaning-preserving router perturbations."""

from pathlib import Path
from unittest import TestCase

from evals.evaluate_robustness import PERTURBATIONS, perturb_scenarios
from evals.evaluate_router import evaluate_config, load_scenarios
from routing_policy import classify_question


DATASET = Path(__file__).parent / "evals" / "scenarios.jsonl"


class RouterRobustnessTests(TestCase):
    def test_expands_every_scenario_once_per_perturbation(self) -> None:
        source = load_scenarios(DATASET)
        variants = perturb_scenarios(source)

        self.assertEqual(len(variants), len(source) * len(PERTURBATIONS))
        self.assertEqual(len({item["id"] for item in variants}), len(variants))
        self.assertEqual(
            {item["perturbation"] for item in variants}, set(PERTURBATIONS)
        )

    def test_weighted_policy_preserves_routes_arguments_and_safety(self) -> None:
        variants = perturb_scenarios(load_scenarios(DATASET))
        metrics, failures = evaluate_config(
            variants, lambda prompt: classify_question(prompt).tool
        )

        self.assertEqual(metrics["routing_accuracy"], 1.0)
        self.assertEqual(metrics["argument_accuracy"], 1.0)
        self.assertEqual(metrics["task_success_rate"], 1.0)
        self.assertEqual(metrics["unsafe_action_rate"], 0.0)
        self.assertFalse([row for row in failures if not row["success"]])


if __name__ == "__main__":
    import unittest

    unittest.main()

