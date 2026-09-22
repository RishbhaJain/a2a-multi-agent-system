"""Measure router stability under deterministic, meaning-preserving perturbations."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from evals.evaluate_router import evaluate_config, load_scenarios
from routing_policy import classify_question, legacy_classify_question


Perturbation = Callable[[str], str]
PERTURBATIONS: dict[str, Perturbation] = {
    "polite_prefix": lambda prompt: f"Please help with this request: {prompt}",
    "context_wrapper": lambda prompt: f"A user submitted the following request: {prompt}",
    "trailing_constraint": lambda prompt: f"{prompt} Return only the requested result.",
    "extra_whitespace": lambda prompt: re.sub(r" ", "   ", prompt),
}


def perturb_scenarios(scenarios: list[dict[str, object]]) -> list[dict[str, object]]:
    """Expand every scenario using the registered deterministic perturbations."""
    variants = []
    for scenario in scenarios:
        for name, transform in PERTURBATIONS.items():
            variant = deepcopy(scenario)
            variant["id"] = f'{scenario["id"]}::{name}'
            variant["source_id"] = scenario["id"]
            variant["perturbation"] = name
            variant["prompt"] = transform(str(scenario["prompt"]))
            variants.append(variant)
    return variants


def _slice_metrics(
    scenarios: list[dict[str, object]], classifier: Callable[[str], str]
) -> dict[str, object]:
    metrics, _ = evaluate_config(scenarios, classifier)
    return metrics


def evaluate(dataset_path: Path) -> dict[str, object]:
    source = load_scenarios(dataset_path)
    variants = perturb_scenarios(source)
    classifiers = {
        "legacy_substring_v1": legacy_classify_question,
        "weighted_rules_v2": lambda prompt: classify_question(prompt).tool,
    }
    configs = {}
    failures = {}
    for config_name, classifier in classifiers.items():
        overall, rows = evaluate_config(variants, classifier)
        by_perturbation = {
            name: _slice_metrics(
                [item for item in variants if item["perturbation"] == name], classifier
            )
            for name in PERTURBATIONS
        }
        tools = sorted({str(item["expected_tool"]) for item in variants})
        by_expected_tool = {
            tool: _slice_metrics(
                [item for item in variants if item["expected_tool"] == tool], classifier
            )
            for tool in tools
        }
        configs[config_name] = {
            "overall": overall,
            "by_perturbation": by_perturbation,
            "by_expected_tool": by_expected_tool,
        }
        failures[config_name] = [row for row in rows if not row["success"]]

    return {
        "schema_version": 1,
        "source_dataset": dataset_path.name,
        "source_scenarios": len(source),
        "perturbations": list(PERTURBATIONS),
        "variant_count": len(variants),
        "configs": configs,
        "failures": failures,
        "regression_gate": {
            "routing_accuracy_min": 0.98,
            "task_success_rate_min": 0.98,
            "per_perturbation_routing_accuracy_min": 0.95,
            "unsafe_action_rate_max": 0.0,
        },
    }


def check_regression(result: dict[str, object], baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current_config = result["configs"]["weighted_rules_v2"]
    prior_config = baseline["configs"]["weighted_rules_v2"]
    current = current_config["overall"]
    prior = prior_config["overall"]
    gate = result["regression_gate"]
    errors = []

    for metric in ("routing_accuracy", "task_success_rate"):
        minimum = gate[f"{metric}_min"]
        if current[metric] < minimum:
            errors.append(f"{metric} {current[metric]} is below {minimum}")
        if current[metric] < prior[metric]:
            errors.append(f"{metric} regressed from {prior[metric]} to {current[metric]}")
    if current["unsafe_action_rate"] > gate["unsafe_action_rate_max"]:
        errors.append("unsafe_action_rate exceeded the zero-tolerance gate")

    slice_minimum = gate["per_perturbation_routing_accuracy_min"]
    for name, metrics in current_config["by_perturbation"].items():
        if metrics["routing_accuracy"] < slice_minimum:
            errors.append(
                f"{name} routing_accuracy {metrics['routing_accuracy']} is below {slice_minimum}"
            )
        prior_metric = prior_config["by_perturbation"][name]["routing_accuracy"]
        if metrics["routing_accuracy"] < prior_metric:
            errors.append(
                f"{name} routing_accuracy regressed from {prior_metric} "
                f"to {metrics['routing_accuracy']}"
            )
    if errors:
        raise SystemExit("; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()

    result = evaluate(args.dataset)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.check:
        check_regression(result, args.check)


if __name__ == "__main__":
    main()

