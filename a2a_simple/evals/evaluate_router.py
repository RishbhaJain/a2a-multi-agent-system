"""Evaluate routing, argument extraction, safety policy, and failure modes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from execution_sandbox import UnsafeCodeError, validate_code
from routing_policy import (
    classify_question,
    extract_arguments,
    legacy_classify_question,
)


Classifier = Callable[[str], str]


def load_scenarios(path: Path) -> list[dict[str, object]]:
    scenarios = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            scenarios.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}") from exc
    return scenarios


def _policy_outcome(code: str | None) -> str | None:
    if code is None:
        return None
    try:
        validate_code(code)
    except UnsafeCodeError:
        return "deny"
    return "allow"


def evaluate_config(
    scenarios: list[dict[str, object]], classifier: Classifier
) -> tuple[dict[str, object], list[dict[str, object]]]:
    routing_correct = 0
    argument_correct = 0
    argument_total = 0
    task_successes = 0
    unsafe_allowed = 0
    unsafe_total = 0
    latencies_ms = []
    failures: Counter[str] = Counter()
    rows = []

    for scenario in scenarios:
        started = time.perf_counter()
        predicted_tool = classifier(str(scenario["prompt"]))
        latencies_ms.append((time.perf_counter() - started) * 1_000)

        expected_tool = scenario["expected_tool"]
        route_ok = predicted_tool == expected_tool
        routing_correct += int(route_ok)

        expected_arguments = scenario.get("expected_arguments")
        predicted_arguments = extract_arguments(
            str(scenario["prompt"]), predicted_tool
        )
        args_ok = expected_arguments is None or predicted_arguments == expected_arguments
        if expected_arguments is not None:
            argument_total += 1
            argument_correct += int(args_ok)

        expected_policy = scenario.get("expected_policy")
        predicted_policy = _policy_outcome(scenario.get("generated_code"))
        policy_ok = expected_policy is None or predicted_policy == expected_policy
        if expected_policy == "deny":
            unsafe_total += 1
            unsafe_allowed += int(predicted_policy == "allow")

        categories = []
        if not route_ok:
            categories.append("routing")
            failures["routing"] += 1
        if not args_ok:
            categories.append("arguments")
            failures["arguments"] += 1
        if not policy_ok:
            category = (
                "unsafe_policy"
                if expected_policy == "deny"
                else "unexpected_policy_rejection"
            )
            categories.append(category)
            failures[category] += 1

        success = route_ok and args_ok and policy_ok
        task_successes += int(success)
        rows.append(
            {
                "id": scenario["id"],
                "expected_tool": expected_tool,
                "predicted_tool": predicted_tool,
                "success": success,
                "failure_categories": categories,
            }
        )

    count = len(scenarios)
    sorted_latency = sorted(latencies_ms)
    p95_index = max(0, min(count - 1, int(0.95 * count) - 1))
    metrics = {
        "scenario_count": count,
        "routing_accuracy": round(routing_correct / count, 4),
        "tool_choice_accuracy": round(routing_correct / count, 4),
        "argument_accuracy": round(argument_correct / argument_total, 4),
        "task_success_rate": round(task_successes / count, 4),
        "unsafe_action_rate": round(unsafe_allowed / unsafe_total, 4),
        "latency_ms": {
            "p50": round(statistics.median(sorted_latency), 4),
            "p95": round(sorted_latency[p95_index], 4),
        },
        "failure_counts": dict(sorted(failures.items())),
    }
    return metrics, rows


def evaluate(dataset_path: Path) -> dict[str, object]:
    scenarios = load_scenarios(dataset_path)
    configs = {
        "legacy_substring_v1": legacy_classify_question,
        "weighted_rules_v2": lambda prompt: classify_question(prompt).tool,
    }
    metrics = {}
    rows = {}
    for name, classifier in configs.items():
        metrics[name], rows[name] = evaluate_config(scenarios, classifier)

    failure_matrix = []
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        legacy = next(row for row in rows["legacy_substring_v1"] if row["id"] == scenario_id)
        current = next(row for row in rows["weighted_rules_v2"] if row["id"] == scenario_id)
        if not legacy["success"] or not current["success"]:
            failure_matrix.append(
                {
                    "id": scenario_id,
                    "expected_tool": scenario["expected_tool"],
                    "legacy_prediction": legacy["predicted_tool"],
                    "current_prediction": current["predicted_tool"],
                    "legacy_failures": legacy["failure_categories"],
                    "current_failures": current["failure_categories"],
                }
            )

    return {
        "schema_version": 1,
        "dataset": dataset_path.name,
        "configs": metrics,
        "failure_matrix": failure_matrix,
        "regression_gate": {
            "routing_accuracy_min": 0.95,
            "task_success_rate_min": 0.95,
            "unsafe_action_rate_max": 0.0,
        },
    }


def check_regression(result: dict[str, object], baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = result["configs"]["weighted_rules_v2"]
    prior = baseline["configs"]["weighted_rules_v2"]
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
